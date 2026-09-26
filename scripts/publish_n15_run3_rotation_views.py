#!/usr/bin/env python3
"""Publish completed Run3 and Rotation+State256 whole policy-H views on KT."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from common import embed_tsne, points_payload, save_json, sha256_file  # noqa: E402

FAMILY = 'N1.5 Intent/Motion'
MANIFEST_FILE_SHA = '1ce3d151989013fe11386b8e6ce3cbeee4fee3e771416dc81a1b230ae2a572fb'
SOURCE_SHA = '8087bf4893ea0e3a6326a0c406ef15ebf5fed0d6fe22aab86fdf344eff3fc02f'
RUNS = [
    {
        'id': 'n15_combined_run3_task_action_060000',
        'label': 'Run3 Task+Action 60K (N1.5)',
        'audit': 'n15_combined_run3_60k_rep_audit_20260925',
        'source': 'run3_SOURCE.json', 'cache': 'run3_H.npy',
        'sha': '5d02942ef02c4638f97f42331b1f0c2f6a3722d0bce696d84176352a8f6e93a1',
        'objective': 'FM + 0.4 Task + 0.5 Action RKD; State disabled',
        'representation': 'N1.5 native Processed-H, float32 valid-token mean',
    },
    {
        'id': 'n15_state_rotation_d256_060000',
        'label': 'Rotation+State256 60K (N1.5)',
        'audit': 'n15_state_subspace_geometry_20260926',
        'source': 'rotation_SOURCE.json', 'cache': 'rotation_H_policy.npy',
        'sha': '98f298063fb62042b8285a8f5afd99648ea07780fe2db92c6e0e04fbb2bd24c3',
        'objective': 'FM + 0.4 State RKD on first 256 rotated dims; Task/Action disabled',
        'representation': 'N1.5 whole policy-visible rotated Processed-H (H @ Q), float32 valid-token mean',
    },
]


def read(path: Path):
    return json.loads(path.read_text())


def checked_source(spec: dict) -> tuple[dict, Path]:
    audit = WORK / 'analysis' / spec['audit']
    source = read(audit / spec['source'])
    feature_file = audit / spec['cache']
    assert source['status'] == 'complete' and source['step'] == 60000
    assert source['sample_count'] == 7200 and source['manifest_sha256'] == MANIFEST_FILE_SHA
    declared_sha = source['feature_sha256']
    if isinstance(declared_sha, dict):
        declared_sha = declared_sha['H_policy']
    assert declared_sha == spec['sha'] == sha256_file(feature_file)
    ckpt = Path(source['checkpoint'])
    for key, relative in (
        ('checkpoint_config_sha256', 'config.json'),
        ('checkpoint_contract_sha256', 'experiment_cfg/clvla_contract.yaml'),
        ('checkpoint_trainer_sha256', 'trainer_state.json'),
        ('checkpoint_index_sha256', 'model.safetensors.index.json'),
    ):
        if key in source:
            assert source[key] == sha256_file(ckpt / relative), (spec['id'], key)
    assert read(ckpt / 'trainer_state.json')['global_step'] == 60000
    index = read(ckpt / 'model.safetensors.index.json')
    assert all((ckpt / shard).is_file() and (ckpt / shard).stat().st_size > 0
               for shard in set(index['weight_map'].values()))
    assert not list(ckpt.glob('*.part*'))
    contract = yaml.safe_load((ckpt / 'experiment_cfg/clvla_contract.yaml').read_text())
    assert contract['future_token_mode'] == 'zero_length'
    if 'rotation' in spec['id']:
        assert contract['state_subspace']['mode'] == 'rotation'
        assert contract['state_subspace']['dim'] == 256
        assert contract['state_rkd']['enabled'] and contract['state_rkd']['weight'] == 0.4
        assert not contract['task_loss']['enabled'] and not contract['action_rkd']['enabled']
        validation = source['validation']
        assert validation['rotation_applied_before_pooling']
        assert validation['policy_capture_max_abs_error'] == 0
        assert validation['subspace_capture_max_abs_error'] == 0
        source['extracted_feature_sha256'] = source['feature_sha256']
    else:
        assert contract['task_loss']['enabled'] and contract['task_loss']['weight'] == 0.4
        assert contract['action_rkd']['enabled'] and contract['action_rkd']['weight'] == 0.5
        assert not contract['state_rkd']['enabled']
    source.update(feature_sha256=spec['sha'], feature_cache=str(feature_file),
                  source_manifest_sha256=SOURCE_SHA, representation=spec['representation'],
                  shape=[7200, 2048], feature='processed',
                  extraction_manifest_file_sha256=MANIFEST_FILE_SHA,
                  published_checkpoint_config_sha256=sha256_file(ckpt / 'config.json'),
                  published_checkpoint_index_sha256=sha256_file(ckpt / 'model.safetensors.index.json'))
    return source, feature_file


def main() -> None:
    manifest_file = ROOT / 'data/official_manifest.json'
    manifest = read(manifest_file)
    assert sha256_file(manifest_file) == MANIFEST_FILE_SHA
    assert manifest['source_manifest_sha256'] == SOURCE_SHA
    assert len(manifest['tasks']) == 24 and len(manifest['sequences']) == 240
    assert [int(row['point_id']) for row in manifest['samples']] == list(range(7200))
    catalog_path = ROOT / 'data/catalog.json'
    profiles_path = ROOT / 'data/training_profiles.json'
    catalog, profiles = read(catalog_path), read(profiles_path)
    old_ids = {run['id'] for run in catalog['runs']}
    checked = []
    for spec in RUNS:
        if spec['id'] in old_ids or (ROOT / 'runs' / spec['id']).exists():
            raise RuntimeError(f'target run already exists: {spec["id"]}')
        checked.append(checked_source(spec))
    # Record original content before generating any view; verify it after site validation.
    mutable = {'data/catalog.json', 'data/training_profiles.json', 'data/site_validation.json'}
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    proof = {'catalog': catalog, 'profiles': profiles, 'tracked_sha256': {
        name: sha256_file(ROOT / name) for name in tracked if name and name not in mutable
    }}
    save_json(ROOT / 'logs/run3_rotation_publication_before.json', proof)
    # Reload copies because the proof above retains the original catalog/profile dictionaries.
    catalog, profiles = read(catalog_path), read(profiles_path)
    for spec, (source, feature_file) in zip(RUNS, checked, strict=True):
        x = np.load(feature_file, allow_pickle=False)
        assert x.shape == (7200, 2048) and x.dtype == np.float32 and np.isfinite(x).all()
        print(f'EMBED_START {spec["id"]} shape={x.shape}', flush=True)
        xy, pca_dim, retained = embed_tsne(x)
        assert xy.shape == (7200, 2) and np.isfinite(xy).all()
        payload = points_payload('processed', xy, manifest, pca_dim, retained, spec['sha'])
        run_dir = ROOT / 'runs' / spec['id']
        save_json(run_dir / 'points_processed.json', payload, compact=True)
        save_json(run_dir / 'manifest.json', {
            'version': 1, 'id': spec['id'], 'label': spec['label'], 'family': FAMILY,
            'manifest_id': manifest['manifest_id'], 'source_manifest_sha256': SOURCE_SHA,
            'features': ['processed'], 'points_files': {'processed': 'points_processed.json'},
            'sequences_file': '../../data/sequences.json', 'fps': 20,
            'source': {'processed': source},
            'notes': ['Frozen 60K representation t-SNE on canonical 300-demo probe; not unseen-episode generalization.',
                      'Published view uses the entire 2048D policy conditioning representation.'],
        })
        if FAMILY not in catalog['families']:
            catalog['families'].append(FAMILY)
        catalog['runs'].append({'id': spec['id'], 'family': FAMILY, 'label': spec['label'],
                                'path': f'runs/{spec["id"]}', 'manifest_id': manifest['manifest_id'],
                                'features': ['processed']})
        profiles['profiles'][spec['id']] = {
            'facts': [['Checkpoint', '60,000'], ['Objective', spec['objective']],
                      ['Published feature', spec['representation']],
                      ['Probe', '24 tasks / 240 episodes / 7,200 points from training dataset']],
            'phases': [], 'source_label': 'checkpoint contract and frozen feature cache',
            'sources': [source['checkpoint']],
        }
        print(f'RENDERED {spec["id"]} points={len(payload["points"])} pca={pca_dim} '
              f'feature_sha={spec["sha"]}', flush=True)
    save_json(catalog_path, catalog)
    save_json(profiles_path, profiles)
    print('PUBLISH_READY', flush=True)


if __name__ == '__main__':
    main()
