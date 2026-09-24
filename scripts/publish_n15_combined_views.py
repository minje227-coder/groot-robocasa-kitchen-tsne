#!/usr/bin/env python3
"""Publish frozen N1.5 Combined 60K Processed-H caches in canonical site order."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORK = Path('/home/work/workspace/test/seonho')
AUDIT = WORK / 'analysis/n15_combined_60k_rep_audit_20260924'
sys.path.insert(0, str(ROOT / 'scripts'))
from common import embed_tsne, points_payload, save_json, sha256_file  # noqa: E402

RUNS = {
    'run1': ('n15_combined_run1_task_060000', 'Run1 Task 60K (N1.5)', 'Task loss'),
    'run2': ('n15_combined_run2_task_state_060000', 'Run2 Task+State 60K (N1.5)', 'Task + State loss'),
}
FAMILY = 'N1.5 Intent/Motion'
SOURCE_SHA = '8087bf4893ea0e3a6326a0c406ef15ebf5fed0d6fe22aab86fdf344eff3fc02f'


def main() -> None:
    manifest_file = ROOT / 'data/official_manifest.json'
    manifest = json.loads(manifest_file.read_text())
    assert sha256_file(manifest_file) == '1ce3d151989013fe11386b8e6ce3cbeee4fee3e771416dc81a1b230ae2a572fb'
    assert manifest['source_manifest_sha256'] == SOURCE_SHA
    assert len(manifest['samples']) == 7200
    assert [int(row['point_id']) for row in manifest['samples']] == list(range(7200))
    catalog_path = ROOT / 'data/catalog.json'
    profiles_path = ROOT / 'data/training_profiles.json'
    catalog = json.loads(catalog_path.read_text())
    profiles = json.loads(profiles_path.read_text())
    old_ids = {run['id'] for run in catalog['runs']}
    for name, (run_id, _, _) in RUNS.items():
        if run_id in old_ids or (ROOT / 'runs' / run_id).exists():
            raise RuntimeError(f'target run already exists: {run_id}')
        source = json.loads((AUDIT / f'{name}_SOURCE.json').read_text())
        feature_file = AUDIT / f'{name}_H.npy'
        assert source['status'] == 'complete' and source['run'] == name and source['step'] == 60000
        assert source['sample_count'] == 7200 and source['manifest_sha256'] == sha256_file(manifest_file)
        assert source['feature_sha256'] == sha256_file(feature_file)
        ckpt = Path(source['checkpoint'])
        for key, rel in (('checkpoint_config_sha256', 'config.json'),
                         ('checkpoint_contract_sha256', 'experiment_cfg/clvla_contract.yaml'),
                         ('checkpoint_trainer_sha256', 'trainer_state.json')):
            assert source[key] == sha256_file(ckpt / rel), (name, key)
        x = np.load(feature_file, allow_pickle=False)
        assert x.shape == (7200, 2048) and x.dtype == np.float32 and np.isfinite(x).all()
        xy, pca_dim, retained = embed_tsne(x)
        run_dir = ROOT / 'runs' / run_id
        run_dir.mkdir(parents=True)
        payload = points_payload('processed', xy, manifest, pca_dim, retained, source['feature_sha256'])
        save_json(run_dir / 'points_processed.json', payload, compact=True)
        _, label, objective = RUNS[name]
        source = dict(source, source_manifest_sha256=SOURCE_SHA,
                      representation='N1.5 native Processed-H, float32 valid-token mean',
                      shape=[7200, 2048], feature='processed',
                      extraction_manifest_file_sha256=source['manifest_sha256'])
        save_json(run_dir / 'manifest.json', {
            'version': 1, 'id': run_id, 'label': label, 'family': FAMILY,
            'manifest_id': manifest['manifest_id'], 'source_manifest_sha256': SOURCE_SHA,
            'features': ['processed'], 'points_files': {'processed': 'points_processed.json'},
            'sequences_file': '../../data/sequences.json', 'fps': 20,
            'source': {'processed': source},
            'notes': ['Frozen 60K representation t-SNE on canonical 300-demo probe; not unseen-episode generalization.'],
        })
        if FAMILY not in catalog['families']:
            catalog['families'].append(FAMILY)
        catalog['runs'].append({'id': run_id, 'family': FAMILY, 'label': label,
                                'path': f'runs/{run_id}', 'manifest_id': manifest['manifest_id'],
                                'features': ['processed']})
        profiles['profiles'][run_id] = {
            'facts': [['Checkpoint', '60,000'], ['Objective', objective],
                      ['Published feature', 'N1.5 Processed H'],
                      ['Probe', '24 tasks / 240 episodes / 7,200 points from training dataset']],
            'phases': [], 'source_label': 'checkpoint contract and frozen feature cache',
            'sources': [source['checkpoint']],
        }
        print(f'RENDERED {run_id} points={len(payload["points"])} feature_sha={source["feature_sha256"]}', flush=True)
    save_json(catalog_path, catalog)
    save_json(profiles_path, profiles)
    print('PUBLISH_READY', flush=True)


if __name__ == '__main__':
    main()
