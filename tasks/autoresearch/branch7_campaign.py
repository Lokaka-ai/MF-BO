"""Prepare/freeze Branch7 and launch only explicitly authorized Slurm requests."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config import Config
from config.branch7 import BRANCH7_PRESETS, BRANCH7_ARMS
from engine.shared_drafts import atomic_json
from engine.parent_selectors.runtime import Phase1SelectorRuntime
from omegaconf import OmegaConf
from tasks.autoresearch.adapter import AutoresearchTask
from tasks.autoresearch.audit import audit_campaign, flatten


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def safe_yaml(path):
    import yaml
    class Loader(yaml.SafeLoader):
        pass
    def pathlib_string(loader, node):
        parts = loader.construct_sequence(node)
        if not all(isinstance(p, str) for p in parts):
            raise ValueError("Invalid legacy pathlib YAML")
        return str(Path(*parts))
    Loader.add_constructor('tag:yaml.org,2002:python/object/apply:pathlib.PosixPath', pathlib_string)
    return yaml.load(Path(path).read_text(), Loader=Loader)


def resolved_json(config):
    def convert(value):
        if isinstance(value,Path):return str(value)
        raise TypeError(f'Unexpected effective-config value: {type(value).__name__}')
    return json.loads(json.dumps(OmegaConf.to_container(config,resolve=True),default=convert,allow_nan=False))


def source_files():
    for name in ('agents','config','engine','llm','utils','tasks/autoresearch','tools'):
        for path in sorted((ROOT/name).rglob('*')):
            if (path.is_file() and not any(p.startswith('.') or p=='__pycache__' for p in path.relative_to(ROOT).parts)
                    and path.suffix in {'.py','.yaml','.toml','.json','.md','.txt','.sh'}):
                yield path
    for name in ('run.py','pyproject.toml','uv.lock','README.md','run_autoresearch.sh','.python-version'):
        yield ROOT/name


def requests(campaign):
    plan={}
    for group,partition,gpus,cpus,mem,hours in [
        ('main','main',2,8,'48G','08:00:00'),
        ('unkillable','unkillable',1,4,'32G','08:00:00'),
        ('short','short-unkillable',4,16,'256G','03:00:00')]:
        plan[group]={'gpus':gpus,'partition':partition,'cpus':cpus,'memory':mem,'wall_time':hours,
            'command':['sbatch','--parsable',f'--job-name=bo-branch7-{group}',
                f'--comment=branch7-20260919-{group}',f'--partition={partition}','--nodes=1','--ntasks=1',
                f'--gres=gpu:a100l:{gpus}',f'--cpus-per-task={cpus}',f'--mem={mem}',f'--time={hours}',
                '--constraint=nvlink','--no-requeue',f'--chdir={campaign}',
                f'--output={campaign}/{group}-%j.log',str(campaign/'allocation.sh'),str(campaign),group]}
    return plan


def prepare(campaign):
    if (campaign/'campaign.json').exists():
        raise RuntimeError('Cannot overwrite a frozen Branch7 campaign')
    old=ROOT/'research/study7_20260918'
    pool=campaign/'draft_pool'
    if not pool.exists():shutil.copytree(old/'draft_pool',pool)
    inputs={}
    for name in ['manifest.json',*[f'draft_{i}.json' for i in range(1,6)]]:
        if sha(pool/name)!=sha(old/'draft_pool'/name):raise ValueError('Historical prior draft pool changed')
        inputs[name]=sha(pool/name)
    atomic_json(campaign/'draft_pool_manifest.json',{'origin':str(old/'draft_pool'),'files':inputs,
        'protocol':'prior_code_and_cost_only_no_historical_outcomes_or_ratings'})
    paired=OmegaConf.create({'duration_seconds':21600,'max_trials':500,'cpus':4,
        'settings':list(BRANCH7_PRESETS),'overrides':{'agent.search.parallel_search_num':2,
        'agent.seed':42,'shared_draft_pool':str(pool/'manifest.json')}})
    OmegaConf.save(paired,campaign/'paired_config.yaml')
    methods={};diffs={}
    for i,preset in enumerate(BRANCH7_PRESETS,1):
        method=f'BR{i:02d}';dim,ancestor,rating,tau,historical=BRANCH7_ARMS[method]
        checked=audit_campaign(paired,preset)
        values={**checked['autoresearch_config'],
            'data_dir':str(ROOT/'tasks/autoresearch/repo'),'desc_file':str(ROOT/'tasks/autoresearch/TASK.md'),
            'log_dir':str(campaign/method/'runs'),'workspace_dir':str(campaign/method/'runs'),'exp_name':preset}
        config=OmegaConf.merge(OmegaConf.structured(Config),values)
        runtime=Phase1SelectorRuntime(config);task=AutoresearchTask(config)
        if task.protocol_id!=read(pool/'manifest.json')['protocol_id']:raise ValueError('Task protocol changed')
        resolved=resolved_json(config)
        old_config=safe_yaml(old/'effective_configs'/f'{historical}.yaml')
        before,after=flatten(old_config),flatten(resolved)
        changes=[]
        exact={'log_dir','workspace_dir','exp_name','shared_draft_pool','parent_selector.type',
               'bo_parent_selector.model','study7.method_id','study7.schema','study7.rating.enabled'}
        for key in sorted(before.keys()|after.keys()):
            if before.get(key)!=after.get(key):
                if key not in exact and not key.startswith('branch7.'):
                    raise ValueError(f'Unexpected historical configuration change: {method} {key}')
                changes.append({'key':key,'historical':before.get(key),'branch7':after.get(key)})
        diffs[method]={'historical_method':historical,'changes':changes}
        atomic_json(campaign/'resolved_configs'/f'{method}.json',resolved)
        OmegaConf.save(OmegaConf.create(resolved),campaign/'resolved_configs'/f'{method}.yaml')
        methods[method]={'preset':preset,'historical':historical,'base_dimension':dim,
            'ancestor':ancestor,'rating':rating,'branch_prior_std':tau,'ucb_coefficient':1.5,
            'task_protocol_id':task.protocol_id,'contract':runtime.study7_selector.contract(),
            'config_sha256':sha(campaign/'resolved_configs'/f'{method}.json'),
            'directory':str(campaign/method),'status':'implemented_not_gpu_validated'}
        admission='valid/finite effective proposals including drafts (draft h=s=0)' if dim==6 else 'valid ordinary proposal closures only'
        optimizer='archived GP6: 25 iterations, maxls 10, prior-objective acceptance, no GP4 maxfun cap' if dim==6 else 'archived GP4: 25 iterations, 150 objective calls, maxls 10, 30 seconds'
        (campaign/'method_cards'/f'{method}.md').write_text(
            f'# {method}: {preset}\n\nHistorical base: {historical}; frozen source `study7_20260918/source`.\n\n'
            f'GP{dim}; ancestor B={ancestor}; rating mean/rank-1 covariance={rating}; '
            f'branch prior SD={tau} normalized, raw SD=s_run × {tau}; UCB coefficient=1.5.\n\n'
            f'Admission: {admission}. Scale freeze retains this base’s original gate.\n\n'
            f'Optimizer: {optimizer}. Numerical failure blocks this new arm; fallback disabled.\n\n'
            'Rating, when enabled, is obtained in this run’s original completed-node review. '
            'No historical outcome, repaired code, rating or checkpoint is loaded.\n\n'
            'One A100 SXM4 80GB, 4 logical CPUs, 2 internal workers, 21600 cumulative active seconds; '
            'seed 42; original 5-slot pool / first 3 drafts; no ordinary root draft.\n')
    spec=ROOT.parent/'EvolveFold_Branch7_Codex_Spec_zh.md'
    protocol={'version':'branch7_v1','spec_sha256':sha(spec),'methods':methods,
        'task_protocol_id':task.protocol_id,'duration_seconds':21600,'workers':2,'seed':42,
        'data_manifest_sha256':sha(ROOT/'.cache/autoresearch/manifest.json'),
        'initial_pool_manifest_sha256':inputs['manifest.json']}
    protocol['protocol_id']=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    atomic_json(campaign/'protocol.json',protocol)
    audit=read(campaign/'compatibility_diff.json');audit['effective_config_differences']=diffs
    atomic_json(campaign/'compatibility_diff.json',audit)
    atomic_json(campaign/'source_manifest.json',{str(p.relative_to(ROOT)):sha(p) for p in source_files()})
    atomic_json(campaign/'registration.json',{'methods':methods,'source_snapshot':str(campaign/'source_snapshot'),
        'protocol_id':protocol['protocol_id'],'groups':{'main':list(BRANCH7_PRESETS[:2]),
        'unkillable':[BRANCH7_PRESETS[2]],'short':list(BRANCH7_PRESETS[3:])},
        'method_directories':{v['preset']:key for key,v in methods.items()},'resource_cap':7,
        'status':'prepared_not_launched','allocation_job_prefix':'bo-branch7'})
    atomic_json(campaign/'resource_plan.json',{'status':'proposed_requires_current_authorization',
        'unregistered_job_not_used':'10857987','max_gpus':7,'requests':requests(campaign),
        'smoke':'Two isolated BR05/BR06 probes on the main two-card allocation, at most 3600 seconds each.',
        'renewals':'Only if explicitly authorized for this campaign; otherwise checkpoint and resource_blocked.',
        'order':list(methods),'no_cancel_other_jobs':True})


def freeze(campaign):
    if (campaign/'campaign.json').exists():raise FileExistsError('Branch7 already frozen')
    suites=list(ET.parse(campaign/'tests/cpu_final.xml').getroot().iter('testsuite'))
    if not suites or any(int(s.attrib[k]) for s in suites for k in ('failures','errors')):
        raise RuntimeError('CPU/integration gate has not passed')
    files=read(campaign/'source_manifest.json')
    for method in ('BR05','BR06'):
        smoke=read(campaign/'tests'/f'smoke_{method}.json')
        if not smoke.get('passed') or smoke.get('source_sha256')!=files:
            raise RuntimeError(f'{method} real smoke gate/source mismatch')
    destination=campaign/'source_snapshot'
    destination.mkdir(exist_ok=False)
    for name,expected in files.items():
        if sha(ROOT/name)!=expected:raise RuntimeError(f'Source changed after CPU/smoke: {name}')
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,target);target.chmod(0o555 if target.stat().st_mode&0o111 else 0o444)
    (destination/'.venv').symlink_to(ROOT/'.venv',target_is_directory=True)
    (destination/'.cache').mkdir()
    (destination/'.cache/autoresearch').symlink_to(ROOT/'.cache/autoresearch',target_is_directory=True)
    registration=read(campaign/'registration.json')
    manifest={**registration,'status':'frozen','frozen_at':datetime.now(timezone.utc).isoformat(),
        'source_sha256':files,'existing_allocations':{},'runtime_environment':{
        'MLEVOLVE_CODEX_REASONING_EFFORT':'high','MLEVOLVE_CODEX_TIMEOUT_SECONDS':'1800',
        'OPENCODE_API_KEY_FILE':str(ROOT.parent/'codex/opecode'),
        'AUTORESEARCH_CACHE_DIR':str(ROOT/'.cache/autoresearch'),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}}
    paths=[campaign/'paired_config.yaml',campaign/'protocol.json',campaign/'registration.json',
           *sorted((campaign/'resolved_configs').glob('*')),*sorted((campaign/'draft_pool').glob('*.json'))]
    manifest['input_sha256']={str(p.relative_to(campaign)):sha(p) for p in paths}
    for p in paths:p.chmod(0o444)
    atomic_json(campaign/'campaign.json',manifest);(campaign/'campaign.json').chmod(0o444)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','freeze','launch','resume'])
    parser.add_argument('--campaign',type=Path,default=ROOT/'research/branch7_20260919')
    parser.add_argument('--dry-run',action='store_true',help='Print exact proposed requests; allocate nothing')
    parser.add_argument('--submit',action='store_true',help='Requires current campaign authorization and validation gates')
    parser.add_argument('--smoke-only',action='store_true',help='Acquire main two-card allocation for isolated validation')
    args=parser.parse_args();campaign=args.campaign.resolve()
    if args.action=='prepare':prepare(campaign)
    elif args.action=='freeze':freeze(campaign)
    elif args.dry_run or not args.submit:
        for group,item in requests(campaign).items():
            print(group,shlex.join(item['command']))
        print('No submission. Explicit current authorization, validation gates and --submit are required.')
    else:
        from tasks.autoresearch.branch7_resources import submit_initial
        submit_initial(campaign,smoke_only=args.smoke_only,resume=args.action=='resume')


if __name__=='__main__':main()
