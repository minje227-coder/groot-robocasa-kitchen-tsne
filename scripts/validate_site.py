#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("SITE_ROOT", "/home/ext_minje/groot_robocasa-kitchen/t-sne"))
KNOWN_SOURCE_MANIFEST_SHAS = {
    # Canonical source manifest SHA embedded in the official site manifest.
    "8087bf4893ea0e3a6326a0c406ef15ebf5fed0d6fe22aab86fdf344eff3fc02f",
}
EXPECTED_POINTS = 7200
EXPECTED_EPISODES = 240
EXPECTED_TASKS = 24


def load(path: Path):
    return json.loads(path.read_text())


def main() -> None:
    official = load(ROOT / "data/official_manifest.json")
    catalog = load(ROOT / "data/catalog.json")
    sequences = load(ROOT / "data/sequences.json")["sequences"]
    official_sha = official.get("source_manifest_sha256")
    if official_sha not in KNOWN_SOURCE_MANIFEST_SHAS:
        raise RuntimeError(f"unrecognized official source manifest: {official_sha}")
    if len(official["samples"]) != EXPECTED_POINTS or len(sequences) != EXPECTED_EPISODES or len(official["tasks"]) != EXPECTED_TASKS:
        raise RuntimeError("official manifest count mismatch")
    if [int(s["point_id"]) for s in official["samples"]] != list(range(EXPECTED_POINTS)):
        raise RuntimeError("official manifest point IDs are not canonical")

    expected_features = {
        "baseline_060000": ["raw", "processed"],
        "testv1_csn_jointsubspace_beta1_060000": ["processed", "projected_norm", "state_masked", "action_masked"],
        "testv1_csn_jointsubspace_betaa03_060000": ["processed", "projected_norm", "state_masked", "action_masked"],
        "n16_official_robocasa_bridge": ["raw", "processed"],
        "rldx1_ft_robocasa_bridge": ["pre_vlln", "post_vlln"],
        "xiaomi_r1_robocasa_bridge": ["vlm_cognition", "dit_kv_l36"],
    }
    known_run_ids = set()
    chart_count = 0
    for run in catalog["runs"]:
        if run["id"] in known_run_ids:
            raise RuntimeError(f"duplicate catalog run: {run['id']}")
        known_run_ids.add(run["id"])
        run_manifest = load(ROOT / run["path"] / "manifest.json")
        if run_manifest["id"] != run["id"] or run_manifest["manifest_id"] != official["manifest_id"]:
            raise RuntimeError(f"run identity mismatch: {run['id']}")
        if run_manifest.get("source_manifest_sha256") not in KNOWN_SOURCE_MANIFEST_SHAS:
            raise RuntimeError(f"unknown run source manifest: {run['id']}")
        if run_manifest["features"] != run["features"]:
            raise RuntimeError(f"catalog/run feature mismatch: {run['id']}")
        if run["id"] in expected_features and run_manifest["features"] != expected_features[run["id"]]:
            raise RuntimeError(f"expected feature contract mismatch: {run['id']}")
        if run["id"] not in expected_features and run_manifest["features"] not in (["processed"], ["student_head"], ["processed", "student_head"], ["action"]):
            raise RuntimeError(f"feature contract mismatch: {run['id']}")

        if run["id"].startswith("testv1_csn_jointsubspace"):
            diagnostic = load(ROOT / run["path"] / run_manifest["diagnostics_file"])
            if diagnostic.get("dimension") != 128 or len(diagnostic.get("state_effective", [])) != 128 or len(diagnostic.get("action_effective", [])) != 128:
                raise RuntimeError(f"CSN diagnostic contract mismatch: {run['id']}")

        if set(run_manifest["points_files"]) != set(run_manifest["features"]):
            raise RuntimeError(f"points-file keys mismatch: {run['id']}")
        for feature, filename in run_manifest["points_files"].items():
            payload = load(ROOT / run["path"] / filename)
            points = payload.get("points", [])
            if payload.get("feature") != feature or payload.get("point_count") != EXPECTED_POINTS or len(points) != EXPECTED_POINTS:
                raise RuntimeError(f"point count/feature mismatch: {run['id']}/{feature}")
            seqs = set()
            for point_id, (point, sample) in enumerate(zip(points, official["samples"], strict=True)):
                if len(point) != 6:
                    raise RuntimeError(f"point schema mismatch: {run['id']}/{feature}/{point_id}")
                if int(point[2]) != int(sample["seq_id"]) or int(point[4]) != int(sample["frame_index"]):
                    raise RuntimeError(f"point identity mismatch: {run['id']}/{feature}/{point_id}")
                if abs(float(point[5]) - float(sample["progress"])) > 1e-5:
                    raise RuntimeError(f"point progress mismatch: {run['id']}/{feature}/{point_id}")
                seqs.add(int(point[2]))
            if seqs != set(range(EXPECTED_EPISODES)):
                raise RuntimeError(f"sequence coverage mismatch: {run['id']}/{feature}")
            if feature in {"projected_norm", "state_masked", "action_masked"}:
                if payload.get("retained_input_dims") not in range(1, 129) or payload.get("preprocessing") != "constant-filter -> centered PCA<=50 without per-dimension scaling -> t-SNE":
                    raise RuntimeError(f"CSN embedding contract mismatch: {run['id']}/{feature}")
            chart_count += 1

    clips = list((ROOT / f"assets/clips/{official['manifest_id']}").glob("seq_*/*.mp4"))
    if len(clips) != 720 or any(path.stat().st_size == 0 for path in clips):
        raise RuntimeError(f"video contract mismatch: {len(clips)}")
    report = load(ROOT / "data/video_export_report.json")
    if report["status"] != "complete" or report["clips"] != 720 or report["frame_count_errors"]:
        raise RuntimeError("video export report mismatch")
    result = {
        "status": "complete", "runs": len(catalog["runs"]), "charts": chart_count,
        "points_per_chart": EXPECTED_POINTS, "episodes": EXPECTED_EPISODES,
        "clips": 720, "clip_bytes": sum(path.stat().st_size for path in clips),
        "manifest_sha256": official_sha,
    }
    (ROOT / "data/site_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
