#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SITE_ROOT = Path("/home/ext_minje/groot_robocasa-kitchen/t-sne")
DATASET_ROOT = Path("/home/ext_minje/robocasa_mg_gr00t_300")
FFMPEG = "/home/ext_minje/miniconda3/envs/lerobot060_robocasa/bin/ffmpeg"
FFPROBE = "/home/ext_minje/miniconda3/envs/lerobot060_robocasa/bin/ffprobe"


def probe(path: Path) -> dict:
    result = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,avg_frame_rate,nb_frames,duration", "-of", "json", str(path)], check=True, text=True, capture_output=True)
    return json.loads(result.stdout)["streams"][0]


def transcode(job: tuple[Path, Path]) -> tuple[str, int]:
    source, target = job
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size > 0:
        return str(target), target.stat().st_size
    partial = target.with_suffix(".partial.mp4")
    command = [
        FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
        "-map", "0:v:0", "-vf", "scale=256:192:force_original_aspect_ratio=decrease,pad=256:192:(ow-iw)/2:(oh-ih)/2",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "32", "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", str(partial),
    ]
    subprocess.run(command, check=True)
    partial.replace(target)
    return str(target), target.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    manifest = json.loads((SITE_ROOT / "data/official_manifest.json").read_text())
    jobs = []
    for seq in manifest["sequences"]:
        for camera, relative in seq["source_videos"].items():
            jobs.append((DATASET_ROOT / relative, SITE_ROOT / seq["videos"][camera]))
    if len(jobs) != 720 or any(not source.is_file() for source, _ in jobs):
        raise RuntimeError("expected 720 existing source clips")
    total = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(transcode, job) for job in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            _, size = future.result()
            total += size
            if index % 60 == 0 or index == len(futures):
                print(f"VIDEO_PROGRESS {index}/720 cumulative_bytes={total}", flush=True)
    errors = []
    for source, target in jobs:
        src, dst = probe(source), probe(target)
        src_frames = int(src.get("nb_frames") or round(float(src["duration"]) * 20))
        dst_frames = int(dst.get("nb_frames") or round(float(dst["duration"]) * 20))
        if src_frames != dst_frames:
            errors.append({"source": str(source), "target": str(target), "source_frames": src_frames, "target_frames": dst_frames})
    report = {"status": "complete" if not errors else "failed", "clips": len(jobs), "bytes": sum(target.stat().st_size for _, target in jobs), "frame_count_errors": errors}
    (SITE_ROOT / "data/video_export_report.json").write_text(json.dumps(report, indent=2) + "\n")
    if errors:
        raise RuntimeError(f"video frame count mismatches: {len(errors)}")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
