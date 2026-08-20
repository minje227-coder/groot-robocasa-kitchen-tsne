#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common import EXPECTED_MANIFEST_SHA256, EXPECTED_POINTS, embed_tsne, load_json, points_payload, save_json, sha256_file

ROOT = Path(__file__).resolve().parents[1]
FAMILY = "TestV1"
WORK_ROOT = Path("/home/work/workspace/test/seonho")
CHECKPOINT_ROOT = WORK_ROOT / "groot_robocasa-kitchen/train_ckpt/CSN"
N17_ROOT = WORK_ROOT / "clvla/Isaac-GR00T-N1.7"
CSN_ROOT = N17_ROOT / "clvla/RA-SD/CSN"
DATASET_ROOT = WORK_ROOT / "robocasa_mg_gr00t_300"
MODALITY_CONFIG = WORK_ROOT / "clvla/benchmarks/Robocasa-Kitchen/robocasa_config.py"
RUNS = {
    "testv1_rscl_paritystate_beta1_060000": {
        "label": "RS-CL ParityState βs=1",
        "checkpoint_name": "CSN_ParityState_AllOne_N1d7_Full_BS64_60K_Tau002",
        "csn_config_name": "CSN_ParityState_AllOne.yaml",
        "mode": "parity_state",
        "state_beta": 1.0,
        "action_beta": 1.0,
        "beta": "βs=1",
    },
    "testv1_rscl_jointshared_beta1_060000": {
        "label": "RS-CL JointShared βs,βa=1",
        "checkpoint_name": "CSN_JointShared_N1d7_Full_BS64_60K_Tau002",
        "csn_config_name": "CSN_JointShared.yaml",
        "mode": "joint_shared",
        "state_beta": 1.0,
        "action_beta": 1.0,
        "beta": "βs=1, βa=1",
    },
    "testv1_rscl_jointshared_betaa03_060000": {
        "label": "RS-CL JointShared βs=1,βa=0.3",
        "checkpoint_name": "CSN_JointShared_N1d7_Full_BS64_60K_Tau002_BetaA03",
        "csn_config_name": "CSN_JointShared_BetaA03.yaml",
        "mode": "joint_shared",
        "state_beta": 1.0,
        "action_beta": 0.3,
        "beta": "βs=1, βa=0.3",
    },
    "testv1_csn_stateonly_beta1_060000": {
        "label": "CSN StateOnly βs=1",
        "checkpoint_name": "CSN_StateOnly_N1d7_Full_BS64_60K_Tau002",
        "csn_config_name": "CSN_StateOnly.yaml",
        "mode": "state_only",
        "state_beta": 1.0,
        "action_beta": 1.0,
        "beta": "βs=1",
    },
    "testv1_csn_jointsubspace_beta1_060000": {
        "label": "CSN JointSubspace βs,βa=1",
        "checkpoint_name": "CSN_JointSubspace_N1d7_Full_BS64_60K_Tau002",
        "csn_config_name": "CSN_JointSubspace.yaml",
        "mode": "joint_csn",
        "state_beta": 1.0,
        "action_beta": 1.0,
        "beta": "βs=1, βa=1",
    },
    "testv1_csn_jointsubspace_betaa03_060000": {
        "label": "CSN JointSubspace βs=1,βa=0.3",
        "checkpoint_name": "CSN_JointSubspace_N1d7_Full_BS64_60K_Tau002_BetaA03",
        "csn_config_name": "CSN_JointSubspace_BetaA03.yaml",
        "mode": "joint_csn",
        "state_beta": 1.0,
        "action_beta": 0.3,
        "beta": "βs=1, βa=0.3",
    },
}


def expected_checkpoint(name: str) -> Path:
    return CHECKPOINT_ROOT / name / name / "checkpoint-60000"


def validate_hashed_path(source: dict, path_key: str, sha_key: str, expected: Path, run_id: str) -> None:
    expected = expected.resolve()
    if source.get(path_key) != str(expected):
        raise RuntimeError(f"{path_key} path mismatch: {run_id}")
    if not expected.is_file() or source.get(sha_key) != sha256_file(expected):
        raise RuntimeError(f"{sha_key} mismatch: {run_id}")


def reusable_points(path: Path, feature_sha: str, official: dict) -> bool:
    if not path.is_file():
        return False
    try:
        payload = load_json(path)
    except (json.JSONDecodeError, OSError):
        return False
    if not (
        payload.get("version") == 1
        and payload.get("feature") == "processed"
        and payload.get("embedding_dim") == 2
        and payload.get("columns") == ["x", "y", "seq_id", "anchor_i", "frame_index", "progress"]
        and payload.get("point_count") == EXPECTED_POINTS
        and payload.get("retained_input_dims") == 2048
        and payload.get("source_feature_sha256") == feature_sha
        and payload.get("pca_dim") == 50
        and payload.get("perplexity") == 30
        and payload.get("max_iter") == 1000
        and payload.get("seed") == 42
    ):
        return False
    points = payload.get("points", [])
    samples = official.get("samples", [])
    if len(points) != EXPECTED_POINTS or len(samples) != EXPECTED_POINTS:
        return False
    for point, sample in zip(points, samples, strict=True):
        if not isinstance(point, list) or len(point) != 6:
            return False
        try:
            numeric = np.asarray([point[0], point[1], point[5]], dtype=np.float64)
            seq_id, anchor_i, frame_index = int(point[2]), int(point[3]), int(point[4])
        except (TypeError, ValueError):
            return False
        if not np.isfinite(numeric).all():
            return False
        if seq_id != int(sample["seq_id"]) or anchor_i != int(sample["frame_index"]) or frame_index != int(sample["frame_index"]):
            return False
        if abs(float(point[5]) - round(float(sample["progress"]), 6)) > 1e-6:
            return False
    return True


def validate_source(source: dict, complete: dict, cfg: dict, official_file_sha: str, run_id: str) -> Path:
    checkpoint = expected_checkpoint(cfg["checkpoint_name"]).resolve()
    if source.get("checkpoint") != str(checkpoint) or not checkpoint.is_dir():
        raise RuntimeError(f"checkpoint mismatch: {run_id}")
    config_path = checkpoint / "config.json"
    index_path = checkpoint / "model.safetensors.index.json"
    if source.get("config_sha256") != sha256_file(config_path) or source.get("index_sha256") != sha256_file(index_path):
        raise RuntimeError(f"checkpoint hash mismatch: {run_id}")
    checkpoint_config = load_json(config_path)
    if checkpoint_config.get("csn_mode") != cfg["mode"]:
        raise RuntimeError(f"checkpoint CSN mode mismatch: {run_id}")
    if float(checkpoint_config.get("csn_state_distance_temperature")) != cfg["state_beta"]:
        raise RuntimeError(f"checkpoint state beta mismatch: {run_id}")
    if float(checkpoint_config.get("csn_action_distance_temperature")) != cfg["action_beta"]:
        raise RuntimeError(f"checkpoint action beta mismatch: {run_id}")

    validate_hashed_path(
        source,
        "dataset_statistics_path",
        "dataset_statistics_sha256",
        checkpoint / "experiment_cfg/dataset_statistics.json",
        run_id,
    )
    dataset_root = DATASET_ROOT.resolve()
    if source.get("dataset_root") != str(dataset_root) or not dataset_root.is_dir():
        raise RuntimeError(f"dataset root mismatch: {run_id}")
    validate_hashed_path(source, "dataset_metadata_path", "dataset_metadata_sha256", dataset_root / "meta/info.json", run_id)
    validate_hashed_path(source, "modality_config", "modality_config_sha256", MODALITY_CONFIG, run_id)
    validate_hashed_path(source, "csn_patch", "csn_patch_sha256", CSN_ROOT / "csn_patch.py", run_id)
    validate_hashed_path(source, "csn_config", "csn_config_sha256", CSN_ROOT / "cfg" / cfg["csn_config_name"], run_id)

    if source.get("manifest_sha256") != official_file_sha:
        raise RuntimeError(f"feature manifest mismatch: {run_id}")
    if source.get("points") != EXPECTED_POINTS or complete.get("points") != EXPECTED_POINTS:
        raise RuntimeError(f"feature point metadata mismatch: {run_id}")
    if source.get("episodes") != 240 or source.get("feature_dim") != 2048:
        raise RuntimeError(f"feature shape metadata mismatch: {run_id}")
    if source.get("batch_size") != 1344 or complete.get("batch_size") != 1344:
        raise RuntimeError(f"unexpected extraction batch: {run_id}")
    parity = source.get("adapter_parity", {})
    if parity.get("processed_max_abs") != 0.0 or parity.get("summary_max_abs") != 0.0:
        raise RuntimeError(f"CSN adapter parity mismatch: {run_id}")
    summary = source.get("summary_token", {})
    if summary.get("position") != "append" or summary.get("checkpoint_key") != "action_head.rscl_summary_token":
        raise RuntimeError(f"SummaryToken contract mismatch: {run_id}")
    if complete.get("summary_token_sha256_float32") != summary.get("sha256_float32"):
        raise RuntimeError(f"SummaryToken hash mismatch: {run_id}")
    return checkpoint


def main() -> None:
    official_path = ROOT / "data/official_manifest.json"
    official = load_json(official_path)
    if official.get("source_manifest_sha256") != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("canonical source manifest SHA drift")
    if len(official.get("samples", [])) != EXPECTED_POINTS:
        raise RuntimeError("official point count drift")
    official_file_sha = sha256_file(official_path)

    catalog_path = ROOT / "data/catalog.json"
    catalog = load_json(catalog_path)
    profiles_path = ROOT / "data/training_profiles.json"
    profiles_payload = load_json(profiles_path)
    profiles = profiles_payload.setdefault("profiles", {})
    catalog_rows = []

    for run_id, cfg in RUNS.items():
        feature_dir = ROOT / "cache/features" / run_id
        complete_path = feature_dir / "COMPLETE"
        feature_path = feature_dir / "features_processed.npz"
        source_path = feature_dir / "features_processed.source.json"
        if not all(path.is_file() for path in (complete_path, feature_path, source_path)):
            raise FileNotFoundError(f"incomplete TestV1 feature cache: {run_id}")
        complete = load_json(complete_path)
        source = load_json(source_path)
        checkpoint = validate_source(source, complete, cfg, official_file_sha, run_id)

        with np.load(feature_path) as payload:
            features = payload["features"]
            point_id = payload["point_id"]
        if features.shape != (EXPECTED_POINTS, 2048) or features.dtype != np.float32 or not np.isfinite(features).all():
            raise RuntimeError(f"invalid processed-H matrix: {run_id} {features.shape} {features.dtype}")
        if not np.array_equal(point_id, np.arange(EXPECTED_POINTS)):
            raise RuntimeError(f"point identity mismatch: {run_id}")

        feature_sha = sha256_file(feature_path)
        run_dir = ROOT / "runs" / run_id
        points_path = run_dir / "points_processed.json"
        if reusable_points(points_path, feature_sha, official):
            print(f"REUSED_TESTV1 {run_id} source_sha256={feature_sha}", flush=True)
        else:
            xy, pca_dim, retained_dims = embed_tsne(features)
            save_json(points_path, points_payload("processed", xy, official, pca_dim, retained_dims, feature_sha), compact=True)
            print(
                f"RENDERED_TESTV1 {run_id} points={len(xy)} pca={pca_dim} "
                f"retained={retained_dims} source_sha256={feature_sha}",
                flush=True,
            )

        run_manifest = {
            "version": 1,
            "id": run_id,
            "label": cfg["label"],
            "family": FAMILY,
            "manifest_id": official["manifest_id"],
            "source_manifest_sha256": official["source_manifest_sha256"],
            "features": ["processed"],
            "points_files": {"processed": "points_processed.json"},
            "sequences_file": "../../data/sequences.json",
            "fps": 20,
            "source": {"processed": source},
        }
        save_json(run_dir / "manifest.json", run_manifest)
        catalog_rows.append(
            {
                "id": run_id,
                "family": FAMILY,
                "label": cfg["label"],
                "path": f"runs/{run_id}",
                "manifest_id": official["manifest_id"],
                "features": ["processed"],
            }
        )
        profiles[run_id] = {
            "facts": [
                ["Checkpoint", "60,000"],
                ["CSN mode", cfg["mode"]],
                ["Relation temperature", cfg["beta"]],
                ["Published feature", "processed H / 2,048D"],
                ["Pooling", "valid-token masked mean; SummaryToken output excluded"],
                ["Extraction batch", "1,344 on H200"],
                ["Adapter parity", "processed=0.0 / summary=0.0"],
                ["Manifest", "24 tasks / 240 episodes / 7,200 points"],
            ],
            "phases": [],
            "source_label": "checkpoint hashes + CSN config + extraction parity",
            "sources": [str(checkpoint), source["csn_config"]],
        }

    target_ids = set(RUNS)
    catalog["runs"] = [row for row in catalog.get("runs", []) if row.get("id") not in target_ids] + catalog_rows
    families = [family for family in catalog.get("families", []) if family != FAMILY]
    first_action = next((index for index, family in enumerate(families) if family.startswith("Action (")), len(families))
    families.insert(first_action, FAMILY)
    catalog["families"] = families
    save_json(catalog_path, catalog)
    save_json(profiles_path, profiles_payload)
    print(
        f"TESTV1_COMPLETE runs={len(catalog_rows)} catalog_runs={len(catalog['runs'])} "
        f"catalog_charts={sum(len(row['features']) for row in catalog['runs'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
