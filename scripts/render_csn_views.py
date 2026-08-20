#!/usr/bin/env python3
"""Publish CSN-only normalized projector and static-mask t-SNE views."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from common import EXPECTED_MANIFEST_SHA256, EXPECTED_POINTS, load_json, points_payload, save_json, sha256_file

ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = Path("/home/work/workspace/test/seonho")
CHECKPOINT_ROOT = WORK_ROOT / "groot_robocasa-kitchen/train_ckpt/CSN"
N17_ROOT = WORK_ROOT / "clvla/Isaac-GR00T-N1.7"
CSN_ROOT = N17_ROOT / "clvla/RA-SD/CSN"
DATASET_ROOT = WORK_ROOT / "robocasa_mg_gr00t_300"
MODALITY_CONFIG = WORK_ROOT / "clvla/benchmarks/Robocasa-Kitchen/robocasa_config.py"
EXTRACTOR = ROOT / "scripts/extract_csn_policy_features.py"
FEATURES = ["projected_norm", "state_masked", "action_masked"]
ALL_FEATURES = ["processed", *FEATURES]
PREPROCESSING = "constant-filter -> centered PCA<=50 without per-dimension scaling -> t-SNE"
RUNS = {
    "testv1_csn_jointsubspace_beta1_060000": {
        "checkpoint_name": "CSN_JointSubspace_N1d7_Full_BS64_60K_Tau002",
        "csn_config_name": "CSN_JointSubspace.yaml",
        "state_beta": 1.0,
        "action_beta": 1.0,
    },
    "testv1_csn_jointsubspace_betaa03_060000": {
        "checkpoint_name": "CSN_JointSubspace_N1d7_Full_BS64_60K_Tau002_BetaA03",
        "csn_config_name": "CSN_JointSubspace_BetaA03.yaml",
        "state_beta": 1.0,
        "action_beta": 0.3,
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


def validate_source(source: dict, feature: str, complete: dict, cfg: dict, official_sha: str, run_id: str) -> Path:
    checkpoint = expected_checkpoint(cfg["checkpoint_name"]).resolve()
    config_path = checkpoint / "config.json"
    index_path = checkpoint / "model.safetensors.index.json"
    if source.get("checkpoint") != str(checkpoint) or not checkpoint.is_dir():
        raise RuntimeError(f"checkpoint mismatch: {run_id}/{feature}")
    if source.get("config_sha256") != sha256_file(config_path) or source.get("index_sha256") != sha256_file(index_path):
        raise RuntimeError(f"checkpoint hash mismatch: {run_id}/{feature}")
    config = load_json(config_path)
    if config.get("csn_mode") != "joint_csn" or config.get("csn_mask_type") != "relu":
        raise RuntimeError(f"checkpoint CSN contract mismatch: {run_id}")
    if float(config.get("csn_state_distance_temperature")) != cfg["state_beta"]:
        raise RuntimeError(f"state beta mismatch: {run_id}")
    if float(config.get("csn_action_distance_temperature")) != cfg["action_beta"]:
        raise RuntimeError(f"action beta mismatch: {run_id}")
    validate_hashed_path(source, "dataset_statistics_path", "dataset_statistics_sha256", checkpoint / "experiment_cfg/dataset_statistics.json", run_id)
    validate_hashed_path(source, "dataset_metadata_path", "dataset_metadata_sha256", DATASET_ROOT / "meta/info.json", run_id)
    validate_hashed_path(source, "modality_config", "modality_config_sha256", MODALITY_CONFIG, run_id)
    validate_hashed_path(source, "csn_patch", "csn_patch_sha256", CSN_ROOT / "csn_patch.py", run_id)
    validate_hashed_path(source, "csn_config", "csn_config_sha256", CSN_ROOT / "cfg" / cfg["csn_config_name"], run_id)
    if source.get("dataset_root") != str(DATASET_ROOT.resolve()) or source.get("manifest_sha256") != official_sha:
        raise RuntimeError(f"dataset/manifest mismatch: {run_id}/{feature}")
    if source.get("feature") != feature or source.get("feature_dim") != 128:
        raise RuntimeError(f"feature source mismatch: {run_id}/{feature}")
    if source.get("points") != EXPECTED_POINTS or source.get("episodes") != 240:
        raise RuntimeError(f"feature sample metadata mismatch: {run_id}/{feature}")
    if source.get("batch_size") != 1344 or complete.get("batch_size") != 1344:
        raise RuntimeError(f"extraction batch mismatch: {run_id}/{feature}")
    parity = source.get("adapter_parity", {})
    if parity.get("processed_max_abs") != 0.0 or parity.get("summary_max_abs") != 0.0:
        raise RuntimeError(f"adapter parity mismatch: {run_id}/{feature}")
    if source.get("mask_diagnostics_sha256") != complete.get("mask_diagnostics_sha256"):
        raise RuntimeError(f"mask diagnostic provenance mismatch: {run_id}/{feature}")
    return checkpoint


def embed_csn(features: np.ndarray) -> tuple[np.ndarray, int, int]:
    x = np.asarray(features, dtype=np.float32)
    if x.shape != (EXPECTED_POINTS, 128) or not np.isfinite(x).all():
        raise RuntimeError(f"invalid CSN feature matrix: {x.shape} {x.dtype}")
    keep = x.var(axis=0) > 1e-10
    x = x[:, keep]
    if x.shape[1] == 0:
        raise RuntimeError("all CSN feature dimensions are constant")
    pca_dim = min(50, x.shape[0] - 1, x.shape[1])
    x = PCA(n_components=pca_dim, random_state=42).fit_transform(x)
    xy = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        max_iter=1000,
        random_state=42,
    ).fit_transform(x)
    return xy.astype(np.float32), int(pca_dim), int(keep.sum())


def reusable_points(path: Path, feature: str, feature_sha: str, official: dict) -> bool:
    if not path.is_file():
        return False
    try:
        payload = load_json(path)
    except (json.JSONDecodeError, OSError):
        return False
    if not (
        payload.get("version") == 1
        and payload.get("feature") == feature
        and payload.get("embedding_dim") == 2
        and payload.get("columns") == ["x", "y", "seq_id", "anchor_i", "frame_index", "progress"]
        and payload.get("point_count") == EXPECTED_POINTS
        and payload.get("source_feature_sha256") == feature_sha
        and payload.get("perplexity") == 30
        and payload.get("max_iter") == 1000
        and payload.get("seed") == 42
        and payload.get("preprocessing") == PREPROCESSING
    ):
        return False
    points = payload.get("points", [])
    if len(points) != EXPECTED_POINTS:
        return False
    for point, sample in zip(points, official["samples"], strict=True):
        if not isinstance(point, list) or len(point) != 6:
            return False
        numeric = np.asarray([point[0], point[1], point[5]], dtype=np.float64)
        if not np.isfinite(numeric).all():
            return False
        if int(point[2]) != int(sample["seq_id"]) or int(point[3]) != int(sample["frame_index"]) or int(point[4]) != int(sample["frame_index"]):
            return False
        if abs(float(point[5]) - round(float(sample["progress"]), 6)) > 1e-6:
            return False
    return True


def feature_metrics(features: np.ndarray) -> dict:
    x = np.asarray(features, dtype=np.float64)
    keep = x.var(axis=0) > 1e-10
    centered = x[:, keep] - x[:, keep].mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = np.square(singular)
    probability = energy / energy.sum()
    effective_rank = float(np.exp(-(probability * np.log(probability.clip(min=1e-15))).sum()))
    norms = np.linalg.norm(x, axis=1)
    subset = x[np.linspace(0, len(x) - 1, 512, dtype=np.int64)]
    distances = np.linalg.norm(subset[:, None, :] - subset[None, :, :], axis=-1)
    upper = distances[np.triu_indices(len(subset), 1)]
    return {
        "retained_dimensions": int(keep.sum()),
        "effective_rank": round(effective_rank, 4),
        "sample_norm_mean": round(float(norms.mean()), 6),
        "sample_norm_rms": round(float(np.sqrt(np.mean(np.square(norms)))), 6),
        "pairwise_distance_median_512": round(float(np.median(upper)), 6),
    }


def validate_mask_diagnostics(payload: dict, cfg: dict, run_id: str) -> None:
    if payload.get("version") != 1 or payload.get("dimension") != 128 or payload.get("activation") != "relu":
        raise RuntimeError(f"mask diagnostic schema mismatch: {run_id}")
    state = np.asarray(payload.get("state_effective"), dtype=np.float64)
    action = np.asarray(payload.get("action_effective"), dtype=np.float64)
    if state.shape != (128,) or action.shape != (128,) or not np.isfinite(state).all() or not np.isfinite(action).all():
        raise RuntimeError(f"mask diagnostic vector mismatch: {run_id}")
    if (state < 0).any() or (action < 0).any():
        raise RuntimeError(f"negative effective mask: {run_id}")
    state_active, action_active = state > 0, action > 0
    intersection, union = state_active & action_active, state_active | action_active
    expected = (66, 20, 10, 76) if cfg["action_beta"] == 1.0 else (67, 70, 22, 115)
    actual = (int(state_active.sum()), int(action_active.sum()), int(intersection.sum()), int(union.sum()))
    if actual != expected:
        raise RuntimeError(f"mask support mismatch: {run_id}: {actual} != {expected}")
    checks = {
        "state_active": actual[0],
        "action_active": actual[1],
        "intersection": actual[2],
        "union": actual[3],
    }
    if any(payload.get(key) != value for key, value in checks.items()):
        raise RuntimeError(f"mask diagnostic count mismatch: {run_id}")
    state_rms = float(np.sqrt(np.mean(np.square(state))))
    action_rms = float(np.sqrt(np.mean(np.square(action))))
    jaccard = float(intersection.sum() / union.sum())
    cosine = float(np.dot(state, action) / (np.linalg.norm(state) * np.linalg.norm(action)))
    top16 = len(set(np.argsort(state)[-16:]) & set(np.argsort(action)[-16:]))
    numeric_checks = {
        "state_rms": state_rms,
        "action_rms": action_rms,
        "jaccard": jaccard,
        "cosine": cosine,
    }
    if any(not np.isclose(float(payload.get(key)), value, atol=1e-6) for key, value in numeric_checks.items()):
        raise RuntimeError(f"mask diagnostic numeric mismatch: {run_id}")
    if payload.get("top16_intersection") != top16:
        raise RuntimeError(f"mask diagnostic Top-16 mismatch: {run_id}")


def main() -> None:
    official_path = ROOT / "data/official_manifest.json"
    official = load_json(official_path)
    if official.get("source_manifest_sha256") != EXPECTED_MANIFEST_SHA256 or len(official.get("samples", [])) != EXPECTED_POINTS:
        raise RuntimeError("official manifest contract drift")
    official_sha = sha256_file(official_path)
    catalog_path = ROOT / "data/catalog.json"
    catalog = load_json(catalog_path)
    profiles_path = ROOT / "data/training_profiles.json"
    profiles_payload = load_json(profiles_path)
    profiles = profiles_payload.setdefault("profiles", {})
    catalog_by_id = {row["id"]: row for row in catalog["runs"]}

    for run_id, cfg in RUNS.items():
        cache = ROOT / "cache/features" / f"csn_views_{run_id}"
        complete_path = cache / "COMPLETE"
        diagnostic_path = cache / "mask_diagnostics.json"
        if not complete_path.is_file() or not diagnostic_path.is_file():
            raise FileNotFoundError(f"incomplete CSN view cache: {run_id}")
        complete = load_json(complete_path)
        if complete.get("features") != FEATURES or complete.get("points") != EXPECTED_POINTS:
            raise RuntimeError(f"CSN cache completion mismatch: {run_id}")
        if complete.get("mask_diagnostics_sha256") != sha256_file(diagnostic_path):
            raise RuntimeError(f"CSN cache diagnostic hash mismatch: {run_id}")
        diagnostics = load_json(diagnostic_path)
        validate_mask_diagnostics(diagnostics, cfg, run_id)
        diagnostics["feature_metrics"] = {}
        diagnostics["interpretation"] = "Static global ReLU gates at the normalized 128D projector output; masked views are not renormalized."
        diagnostics["preprocessing"] = PREPROCESSING
        diagnostics["action_temperature"] = cfg["action_beta"]

        run_dir = ROOT / "runs" / run_id
        manifest_path = run_dir / "manifest.json"
        manifest = load_json(manifest_path)
        if manifest.get("source_manifest_sha256") != EXPECTED_MANIFEST_SHA256 or manifest.get("features") not in (["processed"], ALL_FEATURES):
            raise RuntimeError(f"existing run manifest drift: {run_id}")
        source_by_feature = dict(manifest.get("source", {}))
        points_files = dict(manifest.get("points_files", {}))
        checkpoint = None
        for feature in FEATURES:
            feature_path = cache / f"features_{feature}.npz"
            source_path = cache / f"features_{feature}.source.json"
            if not feature_path.is_file() or not source_path.is_file():
                raise FileNotFoundError(f"missing CSN feature artifact: {run_id}/{feature}")
            source = load_json(source_path)
            current_checkpoint = validate_source(source, feature, complete, cfg, official_sha, run_id)
            if checkpoint is not None and checkpoint != current_checkpoint:
                raise RuntimeError(f"checkpoint changed across features: {run_id}")
            checkpoint = current_checkpoint
            with np.load(feature_path) as payload:
                features = payload["features"]
                point_id = payload["point_id"]
            if features.shape != (EXPECTED_POINTS, 128) or features.dtype != np.float32 or not np.isfinite(features).all():
                raise RuntimeError(f"invalid feature artifact: {run_id}/{feature}")
            if not np.array_equal(point_id, np.arange(EXPECTED_POINTS)):
                raise RuntimeError(f"point identity mismatch: {run_id}/{feature}")
            feature_sha = sha256_file(feature_path)
            points_name = f"points_{feature}.json"
            points_path = run_dir / points_name
            if reusable_points(points_path, feature, feature_sha, official):
                print(f"REUSED_CSN_VIEW {run_id}/{feature}", flush=True)
            else:
                xy, pca_dim, retained = embed_csn(features)
                point_payload = points_payload(feature, xy, official, pca_dim, retained, feature_sha)
                point_payload["preprocessing"] = PREPROCESSING
                save_json(points_path, point_payload, compact=True)
                print(f"RENDERED_CSN_VIEW {run_id}/{feature} pca={pca_dim} retained={retained}", flush=True)
            source["extractor"] = str(EXTRACTOR)
            source["extractor_sha256"] = sha256_file(EXTRACTOR)
            source_by_feature[feature] = source
            points_files[feature] = points_name
            diagnostics["feature_metrics"][feature] = feature_metrics(features)

        save_json(run_dir / "mask_diagnostics.json", diagnostics)
        manifest["features"] = ALL_FEATURES
        manifest["points_files"] = points_files
        manifest["source"] = source_by_feature
        manifest["diagnostics_file"] = "mask_diagnostics.json"
        save_json(manifest_path, manifest)

        row = catalog_by_id.get(run_id)
        if row is None:
            raise RuntimeError(f"catalog row missing: {run_id}")
        row["features"] = ALL_FEATURES
        profile = profiles.get(run_id)
        if profile is None:
            raise RuntimeError(f"training profile missing: {run_id}")
        profile["facts"] = [
            ["Checkpoint", "60,000"],
            ["CSN mode", "joint_csn / static ReLU gates"],
            ["Relation temperature", f"βs=1, βa={cfg['action_beta']:g}"],
            ["Published features", "processed H 2,048D + normalized/projector mask views 128D"],
            ["State / action active", f"{diagnostics['state_active']} / {diagnostics['action_active']} of 128"],
            ["Mask overlap", f"{diagnostics['intersection']}/128 · Jaccard {diagnostics['jaccard']:.3f} · cosine {diagnostics['cosine']:.3f}"],
            ["Mask RMS", f"state {diagnostics['state_rms']:.3f} / action {diagnostics['action_rms']:.3f}"],
            ["Top-16 overlap", f"{diagnostics['top16_intersection']}/16"],
            ["Extraction batch", "1,344 on H200"],
            ["Manifest", "24 tasks / 240 episodes / 7,200 points"],
        ]
        profile["mask_diagnostics"] = diagnostics
        profile["sources"] = [str(checkpoint), str(CSN_ROOT / "cfg" / cfg["csn_config_name"]), str(EXTRACTOR)]
        profile["source_label"] = "checkpoint masks + canonical CSN forward path + extraction parity"

    save_json(catalog_path, catalog)
    save_json(profiles_path, profiles_payload)
    print(
        f"CSN_VIEWS_COMPLETE runs={len(RUNS)} charts={sum(len(row['features']) for row in catalog['runs'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
