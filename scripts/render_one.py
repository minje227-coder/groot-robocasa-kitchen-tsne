#!/usr/bin/env python3
from __future__ import annotations

import argparse

from common import EXPECTED_POINTS, RUNS, SITE_ROOT, embed_tsne, load_json, points_payload, save_json, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True, choices=sorted(RUNS))
    args = parser.parse_args()
    run_id = args.run_id
    cfg = RUNS[run_id]
    if cfg["features"] != ["action"]:
        raise RuntimeError("render_one is restricted to action-only runs")
    feature_path = SITE_ROOT / "cache/features" / run_id / "features_action.npz"
    if not feature_path.is_file():
        raise FileNotFoundError(feature_path)
    import numpy as np
    payload = np.load(feature_path)
    values, ids = payload["features"], payload["point_id"]
    if values.shape[0] != EXPECTED_POINTS or not np.array_equal(ids, np.arange(EXPECTED_POINTS)):
        raise RuntimeError("feature point contract mismatch")
    manifest = load_json(SITE_ROOT / "data/official_manifest.json")
    feature_sha = sha256_file(feature_path)
    xy, pca_dim, retained_dims = embed_tsne(values)
    out = SITE_ROOT / "runs" / run_id / "points_action.json"
    save_json(out, points_payload("action", xy, manifest, pca_dim, retained_dims, feature_sha), compact=True)
    print(f"RENDERED_ONE {run_id} points={len(xy)} source_sha256={feature_sha}", flush=True)


if __name__ == "__main__":
    main()
