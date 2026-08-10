#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from common import EXPECTED_MANIFEST_SHA256, EXPECTED_POINTS, SITE_ROOT, save_json, sha256_file

PLAIN_ORACLE = Path("/home/ext_minje/groot_robocasa-kitchen/offline test/relvae_matrix_oracle_v1/models/flat_ae_z32_fh128_dh512/targets/targets.pt")
DTW_ORACLE = Path("/home/ext_minje/groot_robocasa-kitchen/offline test/relvae_matrix_oracle_v1/models/flat_ae_dtw_z32_repraw_z_fh128_dh512_gw0p1/targets/targets.pt")
PCA_ARTIFACT = Path("/home/ext_minje/groot_robocasa-kitchen/offline test/v2/action_geometry_oracle/pilot_fold0/representation/pca32.pt")
PCA_TARGETS = Path("/home/ext_minje/groot_robocasa-kitchen/offline test/vae_oracle/frozen_vae_v1/targets.pt")


def load_targets(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not torch.equal(payload["point_id"], torch.arange(EXPECTED_POINTS)):
        raise RuntimeError(f"canonical point IDs mismatch: {path}")
    return payload


def write_feature(run_id: str, values: torch.Tensor, source: dict) -> None:
    values = values.float().cpu()
    if tuple(values.shape) != (EXPECTED_POINTS, 32) or not bool(torch.isfinite(values).all()):
        raise RuntimeError(f"invalid feature shape/content for {run_id}: {tuple(values.shape)}")
    out = SITE_ROOT / "cache/features" / run_id / "features_action.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, features=values.numpy(), point_id=np.arange(EXPECTED_POINTS, dtype=np.int64))
    save_json(out.with_suffix(".source.json"), {**source, "action_only": True, "points": EXPECTED_POINTS, "dimension": 32, "source_manifest_sha256": EXPECTED_MANIFEST_SHA256, "feature_sha256": sha256_file(out)})
    print(f"EXPORTED {run_id} shape={tuple(values.shape)} sha256={sha256_file(out)}", flush=True)


def export_ae(regenerated_path: Path, oracle_path: Path, raw_id: str, unit_id: str, checkpoint: Path, batch_size: int) -> None:
    regenerated = load_targets(regenerated_path)
    oracle = load_targets(oracle_path)
    model_sha = sha256_file(checkpoint / "model.pt")
    if regenerated.get("vae_model_sha256") != model_sha or oracle.get("vae_model_sha256") != model_sha:
        raise RuntimeError(f"checkpoint hash mismatch: {checkpoint}")
    raw = regenerated["raw_z"].float()
    oracle_raw = oracle["raw_z"].float()
    parity = float((raw - oracle_raw).abs().max())
    if parity > 1e-4:
        raise RuntimeError(f"fresh extraction parity failed for {checkpoint}: {parity}")
    # Publish the frozen offline-oracle tensor so this visualization exactly matches
    # the distance/angle oracle; fresh extraction is retained as a reproducibility check.
    raw = oracle_raw
    unit = F.normalize(raw, dim=-1)
    norm_error = float((unit.norm(dim=-1) - 1).abs().max())
    base = {
        "checkpoint": str(checkpoint), "model_sha256": model_sha,
        "regenerated_targets": str(regenerated_path), "regenerated_targets_sha256": sha256_file(regenerated_path),
        "oracle_targets": str(oracle_path), "oracle_targets_sha256": sha256_file(oracle_path),
        "fresh_extraction_max_abs_error": parity, "fresh_extraction_tolerance": 1e-4,
        "published_tensor": "frozen offline-oracle tensor", "inference_batch_size": batch_size,
    }
    write_feature(raw_id, raw, {**base, "tensor": "raw_z"})
    write_feature(unit_id, unit, {**base, "tensor": "unit_z", "unit_norm_max_abs_error": norm_error})


def export_pca32() -> None:
    artifact = torch.load(PCA_ARTIFACT, map_location="cpu", weights_only=False)
    targets = load_targets(PCA_TARGETS)
    if artifact.get("status") != "complete" or artifact.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("PCA artifact provenance mismatch")
    if artifact.get("targets_sha256") != sha256_file(PCA_TARGETS):
        raise RuntimeError("PCA source target hash mismatch")
    x = targets["normalized_target"].float().flatten(1)
    values = (x - artifact["pca_mean"].float()) @ artifact["pca_components"].float().T
    source = {
        "artifact": str(PCA_ARTIFACT), "artifact_sha256": sha256_file(PCA_ARTIFACT),
        "targets": str(PCA_TARGETS), "targets_sha256": sha256_file(PCA_TARGETS),
        "input": "normalized_target.flatten(1)", "tensor": "pca32",
        "pca_contract": artifact["pca_contract"],
    }
    write_feature("pca32_action", values, source)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["plain", "dtw", "pca"])
    parser.add_argument("--regenerated-targets", type=Path)
    parser.add_argument("--batch-size", type=int, default=7200)
    args = parser.parse_args()
    if args.kind == "pca":
        export_pca32()
        return
    if args.regenerated_targets is None:
        parser.error("--regenerated-targets is required for AE exports")
    if args.kind == "plain":
        export_ae(args.regenerated_targets, PLAIN_ORACLE, "plain_ae_raw_z", "plain_ae_unit_z", Path("/home/ext_minje/groot_robocasa_V2/RelVAE_ckpt/mg_gr00t_300/flat_ae_z32_fh128_dh512/checkpoint-step-20000"), args.batch_size)
    else:
        export_ae(args.regenerated_targets, DTW_ORACLE, "flat_ae_dtw_w0p1_raw_z", "flat_ae_dtw_w0p1_unit_z", Path("/home/ext_minje/groot_robocasa_V2/RelVAE_ckpt/mg_gr00t_300/flat_ae_dtw_z32_repraw_z_fh128_dh512_gw0p1/checkpoint-step-20000"), args.batch_size)


if __name__ == "__main__":
    main()
