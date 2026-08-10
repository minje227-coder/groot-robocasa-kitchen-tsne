from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

SITE_ROOT = Path("/home/ext_minje/groot_robocasa-kitchen/t-sne")
ARTIFACT_ROOT = Path("/home/ext_minje/groot_robocasa-kitchen/offline test")
DATASET_ROOT = Path("/home/ext_minje/robocasa_mg_gr00t_300")
SOURCE_MANIFEST = ARTIFACT_ROOT / "manifests/kitchen_24task_ep10_desc3_seed42_frame30_v1/manifest.json"
EXPECTED_MANIFEST_SHA256 = "8087bf4893ea0e3a6326a0c406ef15ebf5fed0d6fe22aab86fdf344eff3fc02f"
EXPECTED_TASKS = 24
EXPECTED_EPISODES = 240
EXPECTED_POINTS = 7200
SEED = 42

CHECKPOINTS = {
    "baseline_060000": Path("/home/ext_minje/groot_robocasa-kitchen/train_ckpt/baseline/baseline_n17_rldxcfg_bs64_60k_sd0_20260804/checkpoint-60000"),
    "rkd_unit_mu_060000": Path("/home/ext_minje/groot_robocasa-kitchen/train_ckpt/RKD_UnitMu/RKD_UnitMu/checkpoint-60000"),
    "rkd_raw_mu_w0p3to0_cosine_060000": Path("/home/ext_minje/groot_robocasa-kitchen/train_ckpt/RKD_RawMu_W0p3to0_Cosine/RKD_RawMu_W0p3to0_Cosine/checkpoint-60000"),
    "rkd_raw_mu_distance_angle_1to1_060000": Path("/home/ext_minje/groot_robocasa-kitchen/train_ckpt/RKD_RawMu_W0p3to0_Cosine_DistanceAngle1to1/RKD_RawMu_W0p3to0_Cosine_DistanceAngle1to1/checkpoint-60000"),
    "rkd_raw_mu_distance_only_060000": Path("/home/ext_minje/groot_robocasa-kitchen/train_ckpt/RKD_RawMu_W0p3to0_Cosine_DistanceOnly/RKD_RawMu_W0p3to0_Cosine_DistanceOnly/checkpoint-60000"),
    "rkd_raw_mu_angle_only_060000": Path("/home/ext_minje/groot_robocasa-kitchen/train_ckpt/RKD_RawMu_W0p3to0_Cosine_AngleOnly/RKD_RawMu_W0p3to0_Cosine_AngleOnly/checkpoint-60000"),
}

RUNS = {
    "baseline_060000": {"label": "Baseline 60K", "family": "RoboCasa-Kitchen Ckpt", "features": ["raw", "processed"]},
    "rkd_unit_mu_060000": {"label": "RKD UnitMu 60K", "family": "RoboCasa-Kitchen Ckpt", "features": ["processed"]},
    "rkd_raw_mu_w0p3to0_cosine_060000": {"label": "RKD RawMu W0.3 to 0 Cosine 60K", "family": "RoboCasa-Kitchen Ckpt", "features": ["processed"]},
    "rkd_raw_mu_distance_angle_1to1_060000": {"label": "RKD RawMu Distance:Angle 1:1 60K", "family": "RoboCasa-Kitchen Ckpt", "features": ["processed"]},
    "rkd_raw_mu_distance_only_060000": {"label": "RKD RawMu Distance Only 60K", "family": "RoboCasa-Kitchen Ckpt", "features": ["processed"]},
    "rkd_raw_mu_angle_only_060000": {"label": "RKD RawMu Angle Only 60K", "family": "RoboCasa-Kitchen Ckpt", "features": ["processed"]},
    "timewarp_vae_raw_mu": {"label": "TimewarpVAE RawMu", "family": "Action (TimewarpVAE)", "features": ["action"]},
    "timewarp_vae_unit_mu": {"label": "TimewarpVAE UnitMu", "family": "Action (TimewarpVAE)", "features": ["action"]},
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any, *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) if compact else json.dumps(payload, ensure_ascii=False, indent=2)
    path.write_text(text + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_source(checkpoint: Path) -> dict[str, Any]:
    required = [checkpoint / "config.json", checkpoint / "model.safetensors.index.json", checkpoint / "experiment_cfg/dataset_statistics.json"]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if list(checkpoint.rglob("*.part")):
        raise RuntimeError(f"partial checkpoint files found: {checkpoint}")
    return {
        "checkpoint": str(checkpoint),
        "config_sha256": sha256_file(required[0]),
        "index_sha256": sha256_file(required[1]),
        "dataset_statistics_sha256": sha256_file(required[2]),
    }


def load_site_manifest() -> dict[str, Any]:
    manifest = load_json(SITE_ROOT / "data/official_manifest.json")
    if manifest["manifest_id"] != "kitchen_24task_ep10_desc3_seed42_frame30_v1":
        raise RuntimeError("unexpected manifest id")
    if len(manifest["sequences"]) != EXPECTED_EPISODES or len(manifest["samples"]) != EXPECTED_POINTS:
        raise RuntimeError("manifest count drift")
    return manifest


def embed_tsne(features: np.ndarray, *, seed: int = SEED, perplexity: int = 30) -> tuple[np.ndarray, int, int]:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] != EXPECTED_POINTS or not np.isfinite(x).all():
        raise ValueError(f"invalid feature matrix {x.shape}")
    keep = x.var(axis=0) > 1e-10
    x = x[:, keep]
    if x.shape[1] == 0:
        raise RuntimeError("all feature dimensions are constant")
    x = StandardScaler().fit_transform(x)
    pca_dim = min(50, x.shape[0] - 1, x.shape[1])
    x = PCA(n_components=pca_dim, random_state=seed).fit_transform(x)
    xy = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", max_iter=1000, random_state=seed).fit_transform(x)
    return xy.astype(np.float32), int(pca_dim), int(keep.sum())


def points_payload(feature_name: str, xy: np.ndarray, manifest: dict[str, Any], pca_dim: int, retained_dims: int, source_feature_sha256: str) -> dict[str, Any]:
    points = []
    for coord, sample in zip(xy, manifest["samples"], strict=True):
        points.append([round(float(coord[0]), 6), round(float(coord[1]), 6), int(sample["seq_id"]), int(sample["frame_index"]), int(sample["frame_index"]), round(float(sample["progress"]), 6)])
    return {
        "version": 1,
        "feature": feature_name,
        "embedding_dim": 2,
        "columns": ["x", "y", "seq_id", "anchor_i", "frame_index", "progress"],
        "point_count": len(points),
        "retained_input_dims": retained_dims,
        "source_feature_sha256": source_feature_sha256,
        "pca_dim": pca_dim,
        "perplexity": 30,
        "max_iter": 1000,
        "seed": SEED,
        "points": points,
    }
