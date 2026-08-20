#!/usr/bin/env python3
"""Extract N1.7 CSN processed-H and canonical 128D masked projector views."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expand(value):
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {key: expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n17-project", type=Path, required=True)
    parser.add_argument("--modality-config", type=Path, required=True)
    parser.add_argument("--csn-config", type=Path, required=True)
    parser.add_argument(
        "--feature",
        choices=["processed", "projected_norm", "state_masked", "action_masked"],
        action="append",
        required=True,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit-episodes", type=int, default=None)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    checkpoint, manifest_path, dataset_root, output_dir = (
        path.expanduser().resolve()
        for path in (args.checkpoint, args.manifest, args.dataset_root, args.output_dir)
    )
    project_root = args.n17_project.expanduser().resolve()
    modality_path = args.modality_config.expanduser().resolve()
    csn_dir = project_root / "clvla" / "RA-SD" / "CSN"
    csn_config_path = args.csn_config.expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint missing: {checkpoint}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest missing: {manifest_path}")
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset root missing: {dataset_root}")
    if not (csn_dir / "csn_patch.py").is_file():
        raise FileNotFoundError(f"N1.7 CSN patch missing: {csn_dir}")
    if not modality_path.is_file():
        raise FileNotFoundError(f"modality config missing: {modality_path}")
    if not csn_config_path.is_file():
        raise FileNotFoundError(f"CSN config missing: {csn_config_path}")

    manifest = json.loads(manifest_path.read_text())
    rows = manifest["samples"]
    if len(rows) != 7200 or [int(row["point_id"]) for row in rows] != list(range(7200)):
        raise RuntimeError("manifest point contract mismatch")
    output_dir.mkdir(parents=True, exist_ok=True)
    if any((output_dir / name).exists() for name in ("features_processed.npz", "COMPLETE")):
        raise FileExistsError(f"refusing to overwrite completed output: {output_dir}")

    csn_yaml = yaml.safe_load(csn_config_path.read_text())
    csn_cfg = expand(dict(csn_yaml["csn"]))
    csn_cfg["max_steps"] = int(csn_yaml["max_steps"])
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(csn_dir))
    from gr00t.configs.data.data_config import DataConfig
    csn_cfg["train_seed"] = int(DataConfig().seed)
    os.environ["CLVLA_CSN_CONFIG_JSON"] = json.dumps(csn_cfg)
    import csn_patch  # noqa: F401
    sys.path.insert(0, str(modality_path.parent))
    importlib.import_module(modality_path.stem)

    for feature in args.feature:
        if (output_dir / f"features_{feature}.npz").exists() or (output_dir / f"features_{feature}.source.json").exists():
            raise FileExistsError(f"refusing to overwrite {output_dir}/features_{feature}")

    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.types import MessageType
    from gr00t.policy.gr00t_policy import Gr00tPolicy, _rec_to_dtype

    tag = EmbodimentTag.resolve("new_embodiment")
    print(f"CSN_H_START checkpoint={checkpoint} config={csn_config_path} mode={csn_cfg.get('mode')}", flush=True)
    policy = Gr00tPolicy(embodiment_tag=tag, model_path=str(checkpoint), device=args.device, strict=True)
    model = policy.model
    model.eval()
    if not hasattr(model.action_head, "rscl_summary_token"):
        raise RuntimeError("loaded CSN checkpoint is missing action_head.rscl_summary_token")
    if not hasattr(model, "rscl_projector"):
        raise RuntimeError("loaded CSN checkpoint is missing rscl_projector")
    projector_features = {"projected_norm", "state_masked", "action_masked"}
    if projector_features.intersection(args.feature):
        if int(csn_cfg.get("projector_output_dim", 0)) != 128:
            raise RuntimeError("CSN projector output contract must be 128D")
    if "state_masked" in args.feature and not hasattr(model, "csn_state_mask_beta"):
        raise RuntimeError("state_masked requested but checkpoint has no state mask")
    if "action_masked" in args.feature and not hasattr(model, "csn_action_mask_beta"):
        raise RuntimeError("action_masked requested but checkpoint has no action mask")
    summary_token = model.action_head.rscl_summary_token.detach().float().cpu().numpy()
    summary_hash = hashlib.sha256(summary_token.tobytes()).hexdigest()
    print(
        "CSN_SUMMARY_TOKEN "
        f"shape={tuple(summary_token.shape)} mean={summary_token.mean():.9g} "
        f"std={summary_token.std():.9g} norm={np.linalg.norm(summary_token):.9g} "
        f"sha256_float32={summary_hash}",
        flush=True,
    )

    mask_diagnostics = None
    state_mask_device = None
    action_mask_device = None
    if hasattr(model, "csn_state_mask_beta"):
        state_raw = model.csn_state_mask_beta.detach().float()
        state_mask_device = torch.relu(state_raw)
    if hasattr(model, "csn_action_mask_beta"):
        action_raw = model.csn_action_mask_beta.detach().float()
        action_mask_device = torch.relu(action_raw)
    if state_mask_device is not None and action_mask_device is not None:
        state_cpu = state_mask_device.cpu()
        action_cpu = action_mask_device.cpu()
        state_active = state_cpu > 0
        action_active = action_cpu > 0
        intersection = state_active & action_active
        union = state_active | action_active
        cosine = torch.nn.functional.cosine_similarity(state_cpu, action_cpu, dim=0)
        state_top16 = set(torch.topk(state_cpu, 16).indices.tolist())
        action_top16 = set(torch.topk(action_cpu, 16).indices.tolist())
        mask_diagnostics = {
            "version": 1,
            "dimension": 128,
            "activation": "relu",
            "state_effective": [round(float(value), 8) for value in state_cpu.tolist()],
            "action_effective": [round(float(value), 8) for value in action_cpu.tolist()],
            "state_active": int(state_active.sum()),
            "action_active": int(action_active.sum()),
            "intersection": int(intersection.sum()),
            "union": int(union.sum()),
            "jaccard": float(intersection.sum() / union.sum()),
            "cosine": float(cosine),
            "state_rms": float(state_cpu.square().mean().sqrt()),
            "action_rms": float(action_cpu.square().mean().sqrt()),
            "top16_intersection": len(state_top16 & action_top16),
        }
        (output_dir / "mask_diagnostics.json").write_text(
            json.dumps(mask_diagnostics, indent=2) + "\n"
        )

    loader = LeRobotEpisodeLoader(dataset_path=dataset_root, modality_configs=policy.modality_configs)
    by_episode = defaultdict(list)
    for row in rows:
        by_episode[int(row["episode_index"])].append(row)
    ordered = sorted(by_episode, key=lambda episode: int(by_episode[episode][0]["point_id"]))
    if args.limit_episodes is not None:
        if args.limit_episodes <= 0 or args.limit_episodes > len(ordered):
            raise ValueError("--limit-episodes must be in [1,240]")
        ordered = ordered[: args.limit_episodes]

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    ids, outputs = [], {feature: [] for feature in args.feature}
    parity = None
    summary_seen = None
    pending_rows, pending_ids = [], []

    def flush_pending() -> None:
        nonlocal parity, summary_seen
        if not pending_rows:
            return
        collated = _rec_to_dtype(policy.collate_fn(pending_rows), dtype=torch.bfloat16)
        backbone_inputs, action_inputs = model.prepare_input(collated["inputs"])
        backbone_outputs = model.backbone(backbone_inputs)
        raw = backbone_outputs["backbone_features"]
        mask = backbone_outputs["backbone_attention_mask"].bool()
        manual_h, manual_w = csn_patch.masked_adapter(model.action_head, raw, mask)
        encoded = model.action_head._encode_features(backbone_outputs, action_inputs)
        processed = encoded["backbone_features"]
        summary = backbone_outputs.get("csn_summary")
        if summary is None:
            raise RuntimeError("CSN patch did not expose csn_summary")
        if parity is None:
            parity = {
                "processed_max_abs": float((processed.float() - manual_h.float()).abs().max().cpu()),
                "summary_max_abs": float((summary.float() - manual_w.float()).abs().max().cpu()),
                "summary_shape": list(summary.shape),
                "processed_shape": list(processed.shape),
                "raw_shape": list(raw.shape),
            }
            if parity["processed_max_abs"] > 1e-6 or parity["summary_max_abs"] > 1e-6:
                raise RuntimeError(f"CSN adapter parity failure: {parity}")
            summary_seen = summary.detach().float().cpu().numpy()
            seen_hash = hashlib.sha256(summary_seen.tobytes()).hexdigest()
            print(
                f"CSN_ADAPTER_PARITY {json.dumps(parity, sort_keys=True)} "
                f"summary_output_sha256_float32={seen_hash} raw_token_sha256_float32={summary_hash} "
                f"summary_output_norm={np.linalg.norm(summary_seen):.9g}",
                flush=True,
            )
        if "processed" in outputs:
            pooled = (processed.float() * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
            outputs["processed"].append(pooled.cpu().numpy())
        if projector_features.intersection(outputs):
            projected_norm = torch.nn.functional.normalize(
                model.rscl_projector(summary).float(), dim=-1
            )
            if projected_norm.shape[1] != 128:
                raise RuntimeError(f"unexpected projected shape: {tuple(projected_norm.shape)}")
            if "projected_norm" in outputs:
                outputs["projected_norm"].append(projected_norm.cpu().numpy())
            if "state_masked" in outputs:
                outputs["state_masked"].append(
                    (projected_norm * state_mask_device.to(projected_norm.device)).cpu().numpy()
                )
            if "action_masked" in outputs:
                outputs["action_masked"].append(
                    (projected_norm * action_mask_device.to(projected_norm.device)).cpu().numpy()
                )
        ids.extend(pending_ids)
        pending_rows.clear()
        pending_ids.clear()

    for episode_rank, episode_index in enumerate(ordered, 1):
        episode_data = loader[episode_index]
        for point in by_episode[episode_index]:
            step = extract_step_data(
                episode_data=episode_data,
                step_index=int(point["frame_index"]),
                modality_configs=policy.modality_configs,
                embodiment_tag=tag,
                allow_padding=True,
            )
            pending_rows.append(policy.processor([{"type": MessageType.EPISODE_STEP.value, "content": step}]))
            pending_ids.append(int(point["point_id"]))
            if len(pending_rows) >= args.batch_size:
                flush_pending()
        if episode_rank % args.progress_every == 0:
            print(
                f"FEATURE_PROGRESS episodes={episode_rank}/{len(ordered)} "
                f"points={len(ids) + len(pending_ids)}/{len(ordered)*30} batch_size={args.batch_size}",
                flush=True,
            )
    flush_pending()

    ids_array = np.asarray(ids, dtype=np.int64)
    expected_points = len(ordered) * 30
    if len(ids_array) != expected_points or not np.array_equal(ids_array, np.arange(expected_points)):
        raise RuntimeError("output point identity mismatch")
    stats_candidates = [checkpoint / "experiment_cfg/dataset_statistics.json", checkpoint / "statistics.json"]
    stats_path = next((path for path in stats_candidates if path.is_file()), None)
    if stats_path is None:
        raise FileNotFoundError(f"no checkpoint statistics file under {checkpoint}")
    dataset_metadata_path = dataset_root / "meta/info.json"
    if not dataset_metadata_path.is_file():
        raise FileNotFoundError(f"dataset metadata missing: {dataset_metadata_path}")
    source = {
        "checkpoint": str(checkpoint),
        "config_sha256": sha256_file(checkpoint / "config.json"),
        "index_sha256": sha256_file(checkpoint / "model.safetensors.index.json"),
        "dataset_statistics_path": str(stats_path),
        "dataset_statistics_sha256": sha256_file(stats_path),
        "dataset_root": str(dataset_root),
        "dataset_metadata_path": str(dataset_metadata_path),
        "dataset_metadata_sha256": sha256_file(dataset_metadata_path),
        "modality_config": str(modality_path),
        "modality_config_sha256": sha256_file(modality_path),
        "csn_patch": str(csn_dir / "csn_patch.py"),
        "csn_patch_sha256": sha256_file(csn_dir / "csn_patch.py"),
        "csn_config": str(csn_config_path),
        "csn_config_sha256": sha256_file(csn_config_path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "points": int(expected_points),
        "episodes": int(len(ordered)),
        "batch_size": args.batch_size,
        "pooling": "CSN appended SummaryToken output; processed H view uses valid-token masked mean and excludes SummaryToken",
        "summary_token": {
            "position": "append",
            "checkpoint_key": "action_head.rscl_summary_token",
            "shape": list(summary_token.shape),
            "sha256_float32": summary_hash,
        },
        "adapter_parity": parity,
    }
    dimensions = {
        "processed": 2048,
        "projected_norm": 128,
        "state_masked": 128,
        "action_masked": 128,
    }
    contracts = {
        "processed": "valid-token masked mean of processed H; appended SummaryToken output excluded",
        "projected_norm": "L2-normalized rscl_projector output of the appended SummaryToken",
        "state_masked": "projected_norm multiplied by ReLU(csn_state_mask_beta); no post-mask renormalization",
        "action_masked": "projected_norm multiplied by ReLU(csn_action_mask_beta); no post-mask renormalization",
    }
    for feature, chunks in outputs.items():
        values = np.concatenate(chunks).astype(np.float32)
        if values.shape != (expected_points, dimensions[feature]) or not np.isfinite(values).all():
            raise RuntimeError(f"invalid {feature} output: {values.shape} {values.dtype}")
        partial = output_dir / f"features_{feature}.partial.npz"
        np.savez_compressed(partial, features=values, point_id=ids_array)
        os.replace(partial, output_dir / f"features_{feature}.npz")
        feature_source = {
            **source,
            "feature": feature,
            "feature_dim": dimensions[feature],
            "representation_contract": contracts[feature],
        }
        if mask_diagnostics is not None:
            feature_source["mask_diagnostics_sha256"] = sha256_file(output_dir / "mask_diagnostics.json")
        (output_dir / f"features_{feature}.source.json").write_text(
            json.dumps(feature_source, indent=2) + "\n"
        )
    peak_gib = torch.cuda.max_memory_reserved() / (1024 ** 3) if torch.cuda.is_available() else 0.0
    complete_payload = {
        "features": args.feature,
        "points": int(expected_points),
        "batch_size": args.batch_size,
        "summary_token_sha256_float32": summary_hash,
    }
    if mask_diagnostics is not None:
        complete_payload["mask_diagnostics_sha256"] = sha256_file(output_dir / "mask_diagnostics.json")
    (output_dir / "COMPLETE").write_text(json.dumps(complete_payload) + "\n")
    print(f"FEATURE_COMPLETE checkpoint={checkpoint} features={args.feature} points={expected_points} batch_size={args.batch_size} peak_reserved_gib={peak_gib:.2f} parity={json.dumps(parity, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
