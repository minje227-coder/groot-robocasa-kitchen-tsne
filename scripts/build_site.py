#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from common import (
    ARTIFACT_ROOT, CHECKPOINTS, DATASET_ROOT, EXPECTED_EPISODES, EXPECTED_MANIFEST_SHA256,
    EXPECTED_POINTS, EXPECTED_TASKS, RUNS, SITE_ROOT, SOURCE_MANIFEST, checkpoint_source,
    embed_tsne, load_json, points_payload, save_json, sha256_file,
)

VAE_TARGETS = ARTIFACT_ROOT / "vae_oracle/frozen_vae_v1/targets.pt"
VAE_CHECKPOINT = Path("/home/ext_minje/groot_robocasa_V2/TimewarpVAE_ckpt/focused_v2/mg_gr00t_300/frame_uniform_top5_b4096_20k/b4096/B24_z32_g4_h512_b2e4/checkpoint-step-20000-epoch-44.64")
CACHE_BY_RUN = {
    "baseline_060000": ARTIFACT_ROOT / "caches/baseline60k",
    "rkd_unit_mu_060000": ARTIFACT_ROOT / "caches/unitmu60k",
    "rkd_raw_mu_w0p3to0_cosine_060000": ARTIFACT_ROOT / "caches/rawmu_w0p3to0_cosine60k",
    "rkd_raw_mu_distance_angle_1to1_060000": ARTIFACT_ROOT / "caches/rawmu_distance_angle_1to1_60k",
    "rkd_raw_mu_distance_only_060000": ARTIFACT_ROOT / "caches/rawmu_distance_only60k",
    "rkd_raw_mu_angle_only_060000": ARTIFACT_ROOT / "caches/rawmu_angle_only60k",
}
TASK_GROUPS = {
    "CloseDoubleDoor": "Cabinet", "CloseDrawer": "Cabinet", "CloseSingleDoor": "Cabinet",
    "OpenDoubleDoor": "Cabinet", "OpenDrawer": "Cabinet", "OpenSingleDoor": "Cabinet",
    "CoffeePressButton": "Coffee", "CoffeeServeMug": "Coffee", "CoffeeSetupMug": "Coffee",
    "PnPCabToCounter": "Pick & Place", "PnPCounterToCab": "Pick & Place", "PnPCounterToMicrowave": "Pick & Place",
    "PnPCounterToSink": "Pick & Place", "PnPCounterToStove": "Pick & Place", "PnPMicrowaveToCounter": "Pick & Place",
    "PnPSinkToCounter": "Pick & Place", "PnPStoveToCounter": "Pick & Place",
    "TurnOffMicrowave": "Appliances", "TurnOffStove": "Appliances", "TurnOnMicrowave": "Appliances", "TurnOnStove": "Appliances",
    "TurnOffSinkFaucet": "Sink", "TurnOnSinkFaucet": "Sink", "TurnSinkSpout": "Sink",
}


def build_manifest() -> dict[str, Any]:
    source = load_json(SOURCE_MANIFEST)
    source_sha = sha256_file(SOURCE_MANIFEST)
    if source_sha != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(f"source manifest sha drift: {source_sha}")
    if source["counts"] != {"episodes": EXPECTED_EPISODES, "points": EXPECTED_POINTS, "points_per_task": 300, "tasks": EXPECTED_TASKS}:
        raise RuntimeError(f"source count drift: {source['counts']}")
    by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for sample in source["samples"]:
        by_episode[int(sample["episode_index"])].append(sample)
    samples: list[dict[str, Any]] = []
    sequences: list[dict[str, Any]] = []
    for seq_id, episode in enumerate(source["episodes"]):
        episode_index = int(episode["episode_index"])
        episode_samples = sorted(by_episode[episode_index], key=lambda row: int(row["point_id"]))
        if len(episode_samples) != 30:
            raise RuntimeError(f"episode {episode_index}: expected 30 points")
        task_name = episode["task_name"]
        if task_name not in TASK_GROUPS:
            raise RuntimeError(f"unmapped task {task_name}")
        video_root = f"assets/clips/{source['manifest_id']}/seq_{seq_id:04d}"
        sequences.append({
            "seq_id": seq_id,
            "task_id": int(episode["task_index"]),
            "task_name": task_name,
            "task_label": task_name,
            "task_group": TASK_GROUPS[task_name],
            "description": episode["description"],
            "episode_index": episode_index,
            "episode_length": int(episode["length"]),
            "video_start_frame": 0,
            "dataset_rel_path": episode["parquet_path"],
            "videos": {camera: f"{video_root}/{camera}.mp4" for camera in ("left_view", "right_view", "wrist_view")},
            "source_videos": {key.rsplit(".", 1)[-1]: value for key, value in episode["video_paths"].items()},
        })
        for sample in episode_samples:
            frame = int(sample["frame_index"])
            samples.append({
                **sample,
                "seq_id": seq_id,
                "progress": frame / max(int(episode["length"]) - 1, 1),
                "task_name": task_name,
                "description": episode["description"],
            })
    samples.sort(key=lambda row: int(row["point_id"]))
    if [row["point_id"] for row in samples] != list(range(EXPECTED_POINTS)):
        raise RuntimeError("point ids are not canonical 0..7199")
    metadata_sha = source["dataset"]["metadata_sha256"]
    current_stats = DATASET_ROOT / "meta/stats.json"
    payload = {
        "version": 1,
        "manifest_id": source["manifest_id"],
        "source_manifest": str(SOURCE_MANIFEST),
        "source_manifest_sha256": source_sha,
        "dataset_root": str(DATASET_ROOT),
        "dataset_metadata_sha256": metadata_sha,
        "runtime_stats_sha256": sha256_file(current_stats),
        "stats_contract": "checkpoint-local experiment_cfg/dataset_statistics.json is used for policy normalization",
        "fps": 20,
        "tasks": source["tasks"],
        "sequences": sequences,
        "samples": samples,
        "counts": source["counts"],
        "selection": source["selection"],
        "action_contract": source["action_contract"],
    }
    save_json(SITE_ROOT / "data/official_manifest.json", payload)
    save_json(SITE_ROOT / "data/sequences.json", {"version": 1, "manifest_id": source["manifest_id"], "sequences": sequences})
    return payload


def export_cache_feature(run_id: str, manifest: dict[str, Any]) -> np.ndarray:
    cache = CACHE_BY_RUN[run_id]
    complete = cache / "COMPLETE"
    summary_path = cache / "summary.json"
    if not complete.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"cache incomplete: {cache}")
    summary = load_json(summary_path)
    source = checkpoint_source(CHECKPOINTS[run_id])
    if summary.get("status") != "complete" or int(summary.get("points", -1)) != EXPECTED_POINTS or int(summary.get("shards", -1)) != EXPECTED_EPISODES:
        raise RuntimeError(f"cache summary incomplete: {cache}")
    if summary["manifest"]["sha256"] != EXPECTED_MANIFEST_SHA256 or summary["manifest"]["manifest_id"] != manifest["manifest_id"]:
        raise RuntimeError(f"cache manifest mismatch: {cache}")
    if summary["checkpoint"]["config_sha256"] != source["config_sha256"] or summary["checkpoint"]["index_sha256"] != source["index_sha256"]:
        raise RuntimeError(f"cache checkpoint mismatch: {cache}")
    point_ids, features = [], []
    shards = sorted((cache / "shards").glob("*.pt"))
    if len(shards) != EXPECTED_EPISODES:
        raise RuntimeError(f"cache shard count mismatch: {len(shards)}")
    for shard_path in shards:
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        point_ids.append(shard["point_id"].numpy())
        features.append(shard["pooled_h"].float().numpy())
    ids = np.concatenate(point_ids)
    values = np.concatenate(features)
    order = np.argsort(ids)
    ids, values = ids[order], values[order]
    if not np.array_equal(ids, np.arange(EXPECTED_POINTS)):
        raise RuntimeError(f"cache point identity mismatch: {cache}")
    out = SITE_ROOT / "cache/features" / run_id / "features_processed.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, features=values.astype(np.float32), point_id=ids)
    save_json(out.with_suffix(".source.json"), {**source, "cache": str(cache), "cache_summary_sha256": sha256_file(summary_path), "source_manifest_sha256": EXPECTED_MANIFEST_SHA256, "points": EXPECTED_POINTS})
    return values.astype(np.float32)


def export_vae_features() -> dict[str, np.ndarray]:
    payload = torch.load(VAE_TARGETS, map_location="cpu", weights_only=False)
    ids = payload["point_id"].numpy()
    if not np.array_equal(ids, np.arange(EXPECTED_POINTS)):
        raise RuntimeError("VAE target point identity mismatch")
    raw = payload["mu"].float().numpy()
    unit = payload["unit_mu"].float().numpy()
    calculated = raw / np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12)
    if not np.allclose(unit, calculated, atol=2e-4, rtol=2e-4):
        raise RuntimeError("unit_mu is not normalized raw mu")
    for run_id, values, tensor_name in (("timewarp_vae_raw_mu", raw, "mu"), ("timewarp_vae_unit_mu", unit, "unit_mu")):
        out = SITE_ROOT / "cache/features" / run_id / "features_action.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, features=values.astype(np.float32), point_id=ids)
        save_json(out.with_suffix(".source.json"), {"checkpoint": str(VAE_CHECKPOINT), "targets": str(VAE_TARGETS), "targets_sha256": sha256_file(VAE_TARGETS), "tensor": tensor_name, "action_only": True, "points": EXPECTED_POINTS, "source_manifest_sha256": EXPECTED_MANIFEST_SHA256})
    action_dir = SITE_ROOT / "data/action_chunks"
    save_json(action_dir / "official_manifest_transformed_actions.json", {
        "version": 1,
        "action_keys": ["action.end_effector_position", "action.end_effector_rotation", "action.gripper_close", "action.base_motion", "action.control_mode"],
        "seq_ids": [int(row["seq_id"]) for row in load_json(SITE_ROOT / "data/official_manifest.json")["samples"]],
        "anchor_ids": [int(row["frame_index"]) for row in load_json(SITE_ROOT / "data/official_manifest.json")["samples"]],
        "frame_indices": [int(row["frame_index"]) for row in load_json(SITE_ROOT / "data/official_manifest.json")["samples"]],
        "chunks": payload["raw_target"].float().numpy().round(7).tolist(),
    }, compact=True)
    return {"timewarp_vae_raw_mu": raw, "timewarp_vae_unit_mu": unit}


def load_npz(run_id: str, feature: str) -> np.ndarray:
    path = SITE_ROOT / "cache/features" / run_id / f"features_{feature}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as payload:
        values = np.asarray(payload["features"], dtype=np.float32)
        ids = np.asarray(payload["point_id"]) if "point_id" in payload else np.arange(len(values))
    if values.shape[0] != EXPECTED_POINTS or not np.array_equal(ids, np.arange(EXPECTED_POINTS)):
        raise RuntimeError(f"feature identity mismatch: {path}")
    return values


def reusable_policy_feature(run_id: str) -> bool:
    source_path = SITE_ROOT / "cache/features" / run_id / "features_processed.source.json"
    if not source_path.is_file():
        return False
    source = load_json(source_path)
    expected = checkpoint_source(CHECKPOINTS[run_id])
    if source.get("source_manifest_sha256") != EXPECTED_MANIFEST_SHA256:
        return False
    if source.get("config_sha256") != expected["config_sha256"] or source.get("index_sha256") != expected["index_sha256"]:
        return False
    load_npz(run_id, "processed")
    return True


def reusable_vae_features() -> bool:
    for run_id in ("timewarp_vae_raw_mu", "timewarp_vae_unit_mu"):
        source_path = SITE_ROOT / "cache/features" / run_id / "features_action.source.json"
        if not source_path.is_file() or load_json(source_path).get("source_manifest_sha256") != EXPECTED_MANIFEST_SHA256:
            return False
        load_npz(run_id, "action")
    return True


def render_runs(manifest: dict[str, Any], *, allow_missing: bool = False) -> None:
    catalog = []
    for run_id, cfg in RUNS.items():
        available_features = [feature for feature in cfg["features"] if (SITE_ROOT / "cache/features" / run_id / f"features_{feature}.npz").is_file()]
        if not allow_missing and available_features != cfg["features"]:
            missing = sorted(set(cfg["features"]) - set(available_features))
            raise FileNotFoundError(f"missing features for {run_id}: {missing}")
        if not available_features:
            print(f"SKIPPED_RENDER {run_id} (no features)", flush=True)
            continue
        run_dir = SITE_ROOT / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        point_files = {}
        source_meta = {}
        for feature in available_features:
            feature_path = SITE_ROOT / "cache/features" / run_id / f"features_{feature}.npz"
            feature_sha = sha256_file(feature_path)
            values = load_npz(run_id, feature)
            filename = f"points_{feature}.json"
            point_path = run_dir / filename
            reuse = False
            if point_path.is_file():
                existing = load_json(point_path)
                reuse = existing.get("point_count") == EXPECTED_POINTS and existing.get("source_feature_sha256") == feature_sha
            if reuse:
                print(f"REUSED_RENDER {run_id}/{feature}", flush=True)
            else:
                xy, pca_dim, retained_dims = embed_tsne(values)
                save_json(point_path, points_payload(feature, xy, manifest, pca_dim, retained_dims, feature_sha), compact=True)
            point_files[feature] = filename
            source_path = SITE_ROOT / "cache/features" / run_id / f"features_{feature}.source.json"
            if source_path.is_file():
                source_meta[feature] = load_json(source_path)
        run_manifest = {
            "version": 1, "id": run_id, "label": cfg["label"], "family": cfg["family"],
            "manifest_id": manifest["manifest_id"], "source_manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "features": available_features, "points_files": point_files, "sequences_file": "../../data/sequences.json",
            "fps": 20, "source": source_meta,
        }
        save_json(run_dir / "manifest.json", run_manifest)
        catalog.append({"id": run_id, "family": cfg["family"], "label": cfg["label"], "path": f"runs/{run_id}", "manifest_id": manifest["manifest_id"], "features": available_features})
        print(f"RENDERED {run_id} features={available_features}", flush=True)
    families = list(dict.fromkeys(cfg["family"] for cfg in RUNS.values()))
    save_json(SITE_ROOT / "data/catalog.json", {"version": 2, "families": families, "runs": catalog})


def write_profiles() -> None:
    def policy_profile(teacher: str, schedule: str, distance: float, angle: float, features: str) -> dict[str, Any]:
        return {
            "facts": [["Checkpoint", "60,000"], ["Teacher target", teacher], ["Published features", features], ["Manifest", "24 tasks / 240 episodes / 7,200 points"]],
            "phases": [{"name": "Fine-tuning", "range": "step 0 to 60K", "badge": "RKD ON" if teacher != "none" else "Baseline", "enabled": teacher != "none", "rows": [["RKD schedule", schedule], ["Distance weight", str(distance)], ["Angle weight", str(angle)]]}],
            "source_label": "checkpoint hashes + training YAML",
            "sources": [],
        }
    profiles = {
        "baseline_060000": policy_profile("none", "disabled", 0.0, 0.0, "raw VLM + processed output"),
        "rkd_unit_mu_060000": policy_profile("TimewarpVAE unit_mu", "1.0 to 0 cosine", 1.0, 2.0, "processed output"),
        "rkd_raw_mu_w0p3to0_cosine_060000": policy_profile("TimewarpVAE raw_mu", "0.3 to 0 cosine", 1.0, 2.0, "processed output"),
        "rkd_raw_mu_distance_angle_1to1_060000": policy_profile("TimewarpVAE raw_mu", "0.3 to 0 cosine", 1.0, 1.0, "processed output"),
        "rkd_raw_mu_distance_only_060000": policy_profile("TimewarpVAE raw_mu", "0.3 to 0 cosine", 1.0, 0.0, "processed output"),
        "rkd_raw_mu_angle_only_060000": policy_profile("TimewarpVAE raw_mu", "0.3 to 0 cosine", 0.0, 1.0, "processed output"),
        "timewarp_vae_raw_mu": {"facts": [["Latent", "posterior mean mu"], ["Dimension", "32"], ["Published feature", "action only"]], "phases": [], "source_label": "frozen TimewarpVAE target tensor", "sources": [str(VAE_TARGETS)]},
        "timewarp_vae_unit_mu": {"facts": [["Latent", "L2-normalized posterior mean"], ["Dimension", "32"], ["Published feature", "action only"]], "phases": [], "source_label": "frozen TimewarpVAE target tensor", "sources": [str(VAE_TARGETS)]},
        "plain_ae_raw_z": {"facts": [["Model", "PlainAE (Flat encoder)"], ["Geometry loss", "none"], ["Latent", "raw z / 32D"], ["Published feature", "action only"]], "phases": [], "source_label": "frozen Rel-AE checkpoint", "sources": ["/home/ext_minje/groot_robocasa_V2/RelVAE_ckpt/mg_gr00t_300/flat_ae_z32_fh128_dh512/checkpoint-step-20000"]},
        "plain_ae_unit_z": {"facts": [["Model", "PlainAE (Flat encoder)"], ["Geometry loss", "none"], ["Latent", "L2-normalized z / 32D"], ["Published feature", "action only"]], "phases": [], "source_label": "frozen Rel-AE checkpoint", "sources": ["/home/ext_minje/groot_robocasa_V2/RelVAE_ckpt/mg_gr00t_300/flat_ae_z32_fh128_dh512/checkpoint-step-20000"]},
        "flat_ae_dtw_w0p1_raw_z": {"facts": [["Model", "FlatAE-DTW"], ["DTW-Sammon weight", "0.1"], ["Latent", "raw z / 32D"], ["Published feature", "action only"]], "phases": [], "source_label": "frozen Rel-AE checkpoint", "sources": ["/home/ext_minje/groot_robocasa_V2/RelVAE_ckpt/mg_gr00t_300/flat_ae_dtw_z32_repraw_z_fh128_dh512_gw0p1/checkpoint-step-20000"]},
        "flat_ae_dtw_w0p1_unit_z": {"facts": [["Model", "FlatAE-DTW"], ["DTW-Sammon weight", "0.1"], ["Latent", "L2-normalized z / 32D"], ["Published feature", "action only"]], "phases": [], "source_label": "frozen Rel-AE checkpoint", "sources": ["/home/ext_minje/groot_robocasa_V2/RelVAE_ckpt/mg_gr00t_300/flat_ae_dtw_z32_repraw_z_fh128_dh512_gw0p1/checkpoint-step-20000"]},
        "pca32_action": {"facts": [["Representation", "PCA32"], ["Input", "normalized action [16,12] flattened to 192D"], ["Fit rows", "5,760 (folds 2-9)"], ["Published feature", "action only"]], "phases": [], "source_label": "frozen PCA32 oracle artifact", "sources": ["/home/ext_minje/groot_robocasa-kitchen/offline test/v2/action_geometry_oracle/pilot_fold0/representation/pca32.pt"]},
    }
    for run_id, profile in profiles.items():
        if run_id in CHECKPOINTS:
            profile["sources"] = [str(CHECKPOINTS[run_id])]
    save_json(SITE_ROOT / "data/training_profiles.json", {"version": 1, "profiles": profiles})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["manifest", "features", "render", "all"])
    parser.add_argument("--allow-missing", action="store_true", help="skip policy caches that are still being generated")
    parser.add_argument("--force", action="store_true", help="rebuild valid cached feature matrices")
    args = parser.parse_args()
    manifest = build_manifest() if args.phase in {"manifest", "all"} else load_json(SITE_ROOT / "data/official_manifest.json")
    if args.phase in {"features", "all"}:
        for run_id in CACHE_BY_RUN:
            if args.allow_missing and not (CACHE_BY_RUN[run_id] / "COMPLETE").is_file():
                print(f"SKIPPED {run_id}/processed (cache not complete)", flush=True)
                continue
            if not args.force and reusable_policy_feature(run_id):
                print(f"REUSED {run_id}/processed", flush=True)
                continue
            export_cache_feature(run_id, manifest)
            print(f"EXPORTED {run_id}/processed", flush=True)
        if not args.force and reusable_vae_features():
            print("REUSED TimewarpVAE raw_mu + unit_mu", flush=True)
        else:
            export_vae_features()
            print("EXPORTED TimewarpVAE raw_mu + unit_mu", flush=True)
    if args.phase in {"render", "all"}:
        render_runs(manifest, allow_missing=args.allow_missing)
        write_profiles()


if __name__ == "__main__":
    main()
