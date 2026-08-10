#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/home/ext_minje/groot_robocasa-kitchen/t-sne")
EXPECTED_SHA = "8087bf4893ea0e3a6326a0c406ef15ebf5fed0d6fe22aab86fdf344eff3fc02f"


def load(path: Path):
    return json.loads(path.read_text())


def main() -> None:
    official = load(ROOT / "data/official_manifest.json")
    catalog = load(ROOT / "data/catalog.json")
    sequences = load(ROOT / "data/sequences.json")["sequences"]
    if official["source_manifest_sha256"] != EXPECTED_SHA or len(official["samples"]) != 7200 or len(sequences) != 240:
        raise RuntimeError("official manifest contract mismatch")
    expected_features = {"baseline_060000": ["raw", "processed"]}
    for run in catalog["runs"]:
        manifest = load(ROOT / run["path"] / "manifest.json")
        if manifest["manifest_id"] != official["manifest_id"] or manifest["source_manifest_sha256"] != EXPECTED_SHA:
            raise RuntimeError(f"manifest provenance mismatch: {run['id']}")
        if run["id"] in expected_features and manifest["features"] != expected_features[run["id"]]:
            raise RuntimeError("baseline feature contract mismatch")
        if run["id"] != "baseline_060000" and manifest["features"] not in (["processed"], ["action"]):
            raise RuntimeError(f"feature contract mismatch: {run['id']}")
        for feature, filename in manifest["points_files"].items():
            payload = load(ROOT / run["path"] / filename)
            if payload["point_count"] != 7200 or len(payload["points"]) != 7200:
                raise RuntimeError(f"point count mismatch: {run['id']}/{feature}")
            if {int(point[2]) for point in payload["points"]} != set(range(240)):
                raise RuntimeError(f"sequence coverage mismatch: {run['id']}/{feature}")
    clips = list((ROOT / f"assets/clips/{official['manifest_id']}").glob("seq_*/*.mp4"))
    if len(clips) != 720 or any(path.stat().st_size == 0 for path in clips):
        raise RuntimeError(f"video contract mismatch: {len(clips)}")
    report = load(ROOT / "data/video_export_report.json")
    if report["status"] != "complete" or report["clips"] != 720 or report["frame_count_errors"]:
        raise RuntimeError("video export report mismatch")
    result = {"status": "complete", "runs": len(catalog["runs"]), "charts": sum(len(run["features"]) for run in catalog["runs"]), "points_per_chart": 7200, "episodes": 240, "clips": 720, "clip_bytes": sum(path.stat().st_size for path in clips), "manifest_sha256": EXPECTED_SHA}
    (ROOT / "data/site_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
