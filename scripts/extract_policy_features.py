#!/usr/bin/env python3
"""Extract attention-mask pooled raw VLM and/or processed policy features."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature", choices=["raw", "processed"], action="append", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    checkpoint, manifest_path, dataset_root, output_dir = (path.expanduser().resolve() for path in (args.checkpoint, args.manifest, args.dataset_root, args.output_dir))
    manifest = json.loads(manifest_path.read_text())
    rows = manifest["samples"]
    if len(rows) != 7200 or [int(row["point_id"]) for row in rows] != list(range(7200)):
        raise RuntimeError("manifest point contract mismatch")
    output_dir.mkdir(parents=True, exist_ok=True)
    for feature in args.feature:
        if (output_dir / f"features_{feature}.npz").exists() or (output_dir / f"features_{feature}.source.json").exists():
            raise FileExistsError(f"refusing to overwrite {output_dir}/features_{feature}")

    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.types import MessageType
    from gr00t.policy.gr00t_policy import Gr00tPolicy, _rec_to_dtype

    tag = EmbodimentTag.resolve("new_embodiment")
    policy = Gr00tPolicy(embodiment_tag=tag, model_path=str(checkpoint), device=args.device, strict=True)
    model = policy.model
    loader = LeRobotEpisodeLoader(dataset_path=dataset_root, modality_configs=policy.modality_configs)
    by_episode = defaultdict(list)
    for row in rows:
        by_episode[int(row["episode_index"])].append(row)
    ordered = sorted(by_episode, key=lambda episode: int(by_episode[episode][0]["point_id"]))
    ids, outputs = [], {feature: [] for feature in args.feature}
    for episode_rank, episode_index in enumerate(ordered, 1):
        episode_data = loader[episode_index]
        processed_rows = []
        for point in by_episode[episode_index]:
            step = extract_step_data(episode_data=episode_data, step_index=int(point["frame_index"]), modality_configs=policy.modality_configs, embodiment_tag=tag, allow_padding=True)
            processed_rows.append(policy.processor([{"type": MessageType.EPISODE_STEP.value, "content": step}]))
        collated = _rec_to_dtype(policy.collate_fn(processed_rows), dtype=torch.bfloat16)
        backbone_inputs, action_inputs = model.prepare_input(collated["inputs"])
        backbone_outputs = model.backbone(backbone_inputs)
        mask = backbone_outputs["backbone_attention_mask"].bool()
        if "raw" in outputs:
            raw = backbone_outputs["backbone_features"]
            pooled = (raw.float() * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
            outputs["raw"].append(pooled.cpu().numpy())
        if "processed" in outputs:
            processed = model.action_head._encode_features(backbone_outputs, action_inputs)["backbone_features"]
            pooled = (processed.float() * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
            outputs["processed"].append(pooled.cpu().numpy())
        ids.extend(int(row["point_id"]) for row in by_episode[episode_index])
        if episode_rank % args.progress_every == 0:
            print(f"FEATURE_PROGRESS episodes={episode_rank}/240 points={len(ids)}/7200", flush=True)
    ids_array = np.asarray(ids, dtype=np.int64)
    if not np.array_equal(ids_array, np.arange(7200)):
        raise RuntimeError("output point identity mismatch")
    source = {
        "checkpoint": str(checkpoint),
        "config_sha256": sha256_file(checkpoint / "config.json"),
        "index_sha256": sha256_file(checkpoint / "model.safetensors.index.json"),
        "dataset_statistics_sha256": sha256_file(checkpoint / "experiment_cfg/dataset_statistics.json"),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "points": 7200,
        "pooling": "backbone_attention_mask mean",
    }
    for feature, chunks in outputs.items():
        values = np.concatenate(chunks).astype(np.float32)
        partial = output_dir / f"features_{feature}.partial.npz"
        np.savez_compressed(partial, features=values, point_id=ids_array)
        os.replace(partial, output_dir / f"features_{feature}.npz")
        (output_dir / f"features_{feature}.source.json").write_text(json.dumps({**source, "feature": feature}, indent=2) + "\n")
    (output_dir / "COMPLETE").write_text(json.dumps({"features": args.feature, "points": 7200}) + "\n")
    print(f"FEATURE_COMPLETE checkpoint={checkpoint} features={args.feature} points=7200", flush=True)


if __name__ == "__main__":
    main()
