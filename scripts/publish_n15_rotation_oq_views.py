"""Publish Sheet O/Q whole policy-H caches using the existing Kitchen renderer."""
from pathlib import Path
import json
import subprocess
import sys
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
AUDIT=ROOT.parent/'analysis/n15_cross_task_state_20260927'
sys.path.insert(0,str(ROOT/'scripts'))
from common import embed_tsne,points_payload,save_json,sha256_file

FAMILY='N1.5 Intent/Motion'
FILE_SHA='1ce3d151989013fe11386b8e6ce3cbeee4fee3e771416dc81a1b230ae2a572fb'
SOURCE_SHA='8087bf4893ea0e3a6326a0c406ef15ebf5fed0d6fe22aab86fdf344eff3fc02f'
RUNS=[
 ('action','O','n15_rotation_action_a256_060000','Rotation+Action256 60K (N1.5)',
  'FM + 0.5 Action RKD on rotated dims 256:512; State/Task disabled',
  'b55c69c0454d411fa1b9f222e26b05e688bbaa57d420234046190f9a21311933'),
 ('state_action','Q','n15_rotation_state_action_s256_a256_060000','Rotation+State256+Action256 60K (N1.5)',
  'FM + 0.4 State RKD on rotated dims 0:256 + 0.5 Action RKD on dims 256:512; Task disabled',
  '52a473cc76a42b94d9c10d22d02f46594842dd9f6188cdc8f382ae608b439d51')]

def read(path):return json.loads(path.read_text())

def main():
 manifest_file=ROOT/'data/official_manifest.json';manifest=read(manifest_file)
 assert sha256_file(manifest_file)==FILE_SHA and manifest['source_manifest_sha256']==SOURCE_SHA
 assert len(manifest['tasks'])==24 and len(manifest['sequences'])==240
 assert [r['point_id'] for r in manifest['samples']]==list(range(7200))
 catalog_file=ROOT/'data/catalog.json';profiles_file=ROOT/'data/training_profiles.json'
 catalog=read(catalog_file);profiles=read(profiles_file)
 mutable={'data/catalog.json','data/training_profiles.json','data/site_validation.json'}
 tracked=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
 proof={'catalog':catalog,'profiles':profiles,'tracked_sha256':{
  f:sha256_file(ROOT/f) for f in tracked if f and f not in mutable}}
 checked=[]
 for mode,column,run_id,label,objective,expected_sha in RUNS:
  assert run_id not in {r['id'] for r in catalog['runs']} and not (ROOT/'runs'/run_id).exists()
  source=read(AUDIT/f'{mode}_SOURCE.json');cache=AUDIT/f'{mode}_H_policy.npy'
  assert source['status']=='complete' and source['mode']==mode and source['step']==60000
  assert source['sample_count']==7200 and source['episodes']==240 and source['manifest_sha256']==FILE_SHA
  assert source['feature_sha256']['H_policy']==expected_sha==sha256_file(cache)
  ckpt=Path(source['checkpoint'])
  assert sha256_file(ckpt/'experiment_cfg/clvla_contract.yaml')==source['checkpoint_contract_sha256']
  assert sha256_file(ckpt/'model.safetensors.index.json')==source['checkpoint_index_sha256']
  assert sha256_file(AUDIT/'extract_split.py')==source['extractor_sha256']
  trainer=read(ckpt/'trainer_state.json');assert trainer['global_step']==trainer['max_steps']==60000
  index=read(ckpt/'model.safetensors.index.json')
  assert all((ckpt/f).is_file() and (ckpt/f).stat().st_size>0 for f in set(index['weight_map'].values()))
  assert not list(ckpt.glob('*.part*'))
  contract=yaml.safe_load((ckpt/'experiment_cfg/clvla_contract.yaml').read_text())
  assert contract['variant']=='rotation_split' and contract['future_token_mode']=='zero_length'
  assert contract['rotation_split']['state_slice']==[0,256] and contract['rotation_split']['action_slice']==[256,512]
  assert not contract['task_loss']['enabled'] and contract['action_rkd']['enabled'] and contract['action_rkd']['weight']==.5
  assert contract['state_rkd']['enabled']==(mode=='state_action') and contract['state_rkd']['weight']==.4
  assert source['validation']['all_points_written'] and source['validation']['rotation_capture_error']==0
  x=np.load(cache,allow_pickle=False)
  assert x.shape==(7200,2048) and x.dtype==np.float32 and np.isfinite(x).all()
  source=dict(source,extracted_feature_sha256=source['feature_sha256'],feature_sha256=expected_sha,
    feature_cache=str(cache),source_manifest_sha256=SOURCE_SHA,extraction_manifest_file_sha256=FILE_SHA,
    checkpoint_config_sha256=sha256_file(ckpt/'config.json'),checkpoint_trainer_sha256=sha256_file(ckpt/'trainer_state.json'),
    shape=[7200,2048],feature='processed',sheet_column=column,
    representation='N1.5 whole policy-visible rotated Processed-H (H @ Q), float32 valid-token mean')
  checked.append((run_id,label,objective,source,x))
 save_json(ROOT/'logs/rotation_oq_publication_before.json',proof)
 catalog=read(catalog_file);profiles=read(profiles_file)
 for run_id,label,objective,source,x in checked:
  print('EMBED_START',run_id,flush=True)
  xy,pca_dim,retained=embed_tsne(x);assert xy.shape==(7200,2) and np.isfinite(xy).all()
  dest=ROOT/'runs'/run_id
  payload=points_payload('processed',xy,manifest,pca_dim,retained,source['feature_sha256'])
  save_json(dest/'points_processed.json',payload,compact=True)
  save_json(dest/'manifest.json',{'version':1,'id':run_id,'label':label,'family':FAMILY,
    'manifest_id':manifest['manifest_id'],'source_manifest_sha256':SOURCE_SHA,'features':['processed'],
    'points_files':{'processed':'points_processed.json'},'sequences_file':'../../data/sequences.json','fps':20,
    'source':{'processed':source},'notes':['Whole 2048D policy-visible H; canonical frozen 300-demo probe.']})
  if FAMILY not in catalog['families']:catalog['families'].append(FAMILY)
  catalog['runs'].append({'id':run_id,'family':FAMILY,'label':label,'path':f'runs/{run_id}',
    'manifest_id':manifest['manifest_id'],'features':['processed']})
  profiles['profiles'][run_id]={'facts':[['Checkpoint','60,000'],['Objective',objective],
    ['Published feature',source['representation']],['Sheet column',source['sheet_column']],
    ['Probe','24 tasks / 240 episodes / 7,200 points']], 'phases':[],
    'source_label':'checkpoint contract and frozen feature cache','sources':[source['checkpoint']]}
  print('RENDERED',run_id,len(payload['points']),source['feature_sha256'],flush=True)
 save_json(catalog_file,catalog);save_json(profiles_file,profiles)
 print('PUBLISH_READY',flush=True)

if __name__=='__main__':main()
