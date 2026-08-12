#!/usr/bin/env python3
"""Export the trained RKD StudentAttentionHead for a canonical Kitchen cache."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--site-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--run-id", default="dtw_sammon_rawz_angle_only_060000")
    return parser.parse_args()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    project = args.project.resolve()
    cache = args.cache.resolve()
    checkpoint = args.checkpoint.resolve()
    site_root = args.site_root.resolve()
    summary_path = cache / "summary.json"
    summary = json.loads(summary_path.read_text())
    if summary.get("status") != "complete" or summary.get("points") != 7200 or summary.get("shards") != 240:
        raise RuntimeError(f"cache contract mismatch: {cache}")
    if summary["checkpoint"]["config_sha256"] != sha256_file(checkpoint / "config.json"):
        raise RuntimeError("cache/checkpoint config hash mismatch")
    if summary["checkpoint"]["index_sha256"] != sha256_file(checkpoint / "model.safetensors.index.json"):
        raise RuntimeError("cache/checkpoint index hash mismatch")

    # Reuse the already validated loader used by the offline diagnosis. It
    # applies exactly the checkpoint's eight StudentAttentionHead tensors.
    diagnosis_dir = project / "clvla/offline test/v2"
    sys.path.insert(0, str(diagnosis_dir))
    import sammon_rawz_angleonly_diagnosis as diagnosis

    cached = diagnosis.load_cache(cache)
    device = torch.device(args.device)
    student = diagnosis.load_student(
        checkpoint, cached["shards"], project, device, args.batch_size
    ).numpy().astype(np.float32)
    if student.shape != (7200, 512) or not np.isfinite(student).all():
        raise RuntimeError(f"unexpected StudentHead output: {student.shape}")

    output_dir = site_root / "cache/features" / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "features_student_head.npz"
    temporary = feature_path.with_name(f".{feature_path.name}.partial.{os.getpid()}.npz")
    np.savez_compressed(temporary, features=student, point_id=np.arange(7200, dtype=np.int64))
    os.replace(temporary, feature_path)
    source = {
        "feature": "student_head",
        "input_feature": "processed_tokens",
        "pooling": "checkpoint StudentAttentionHead (token MLP + masked attention + out projection)",
        "input_shape": [207, 2048],
        "output_shape": [512],
        "checkpoint": str(checkpoint),
        "config_sha256": sha256_file(checkpoint / "config.json"),
        "index_sha256": sha256_file(checkpoint / "model.safetensors.index.json"),
        "dataset_statistics_sha256": sha256_file(checkpoint / "experiment_cfg/dataset_statistics.json"),
        "cache": str(cache),
        "cache_summary_sha256": sha256_file(summary_path),
        "manifest": "/home/ext_minje/groot_robocasa-kitchen/offline test/manifests/kitchen_24task_ep10_desc3_seed42_frame30_v1/manifest.json",
        "manifest_sha256": "8087bf4893ea0e3a6326a0c406ef15ebf5fed0d6fe22aab86fdf344eff3fc02f",
        "points": 7200,
    }
    atomic_json(feature_path.with_suffix(".source.json"), source)
    print(f"STUDENT_FEATURE_COMPLETE path={feature_path} shape={student.shape}", flush=True)


if __name__ == "__main__":
    main()
