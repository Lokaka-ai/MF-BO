"""Prepare and freeze exactly AQ01-AQ07; resource submission is separate."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from config import Config
from config.acq7 import ACQ7_ARMS, ACQ7_PRESETS
from engine.shared_drafts import atomic_json
from engine.parent_selectors.runtime import Phase1SelectorRuntime
from omegaconf import OmegaConf
from tasks.autoresearch.adapter import AutoresearchTask
from tasks.autoresearch.audit import audit_campaign, flatten
from tasks.autoresearch.branch7_campaign import source_files, sha, read, resolved_json

CAPS={'main':2,'unkillable':1,'short':4}
SMOKE_METHODS=('AQ06','AQ07')
SMOKE_SECONDS=3600


def requests(campaign):
    result={}
    for group,partition,gpus,cpus,mem,limit in [
            ('main','main',2,8,'48G','08:00:00'),
            ('short','short-unkillable',4,16,'256G','03:00:00')]:
        result[group]={'partition':partition,'gpus':gpus,'cpus':cpus,'memory':mem,'wall_time':limit,
            'command':['sbatch','--parsable',f'--job-name=bo-acq7-{group}',
                f'--comment=acq7-20260920-{group}',f'--partition={partition}','--nodes=1','--ntasks=1',
                f'--gres=gpu:a100l:{gpus}',f'--cpus-per-task={cpus}',f'--mem={mem}',f'--time={limit}',
                '--constraint=nvlink','--no-requeue',f'--chdir={campaign}',
                f'--output={campaign}/{group}-%j.log',str(campaign/'allocation.sh'),str(campaign),group]}
    return result


def prepare(campaign):
    if (campaign/'campaign.json').exists():
        raise RuntimeError('Cannot alter a frozen Acq7 campaign')
    old=ROOT/'research/branch7_20260919'
    for name in ('resolved_configs','method_cards','commands','tests','smoke','analysis'):
        (campaign/name).mkdir(parents=True,exist_ok=True)
    pool=campaign/'draft_pool'
    if not pool.exists():shutil.copytree(old/'draft_pool',pool)
    inputs={}
    for name in ['manifest.json',*[f'draft_{i}.json' for i in range(1,6)]]:
        if sha(pool/name)!=sha(old/'draft_pool'/name):raise ValueError('The prior five-slot pool differs')
        inputs[name]=sha(pool/name)
    atomic_json(campaign/'initial_pool_manifest.json',{'origin':str(old/'draft_pool'),
        'files':inputs,'policy':'prior input code/request/cost only; no old result, rating, repaired code or weight'})
    paired=OmegaConf.create({'duration_seconds':21600,'max_trials':500,'cpus':4,
        'settings':list(ACQ7_PRESETS),'overrides':{'agent.search.parallel_search_num':2,
        'agent.seed':42,'shared_draft_pool':str(pool/'manifest.json')}})
    OmegaConf.save(paired,campaign/'paired_config.yaml')
    methods,diffs={},{}
    for (method,(kind,tau,rating,acq,slope)),preset in zip(ACQ7_ARMS.items(),ACQ7_PRESETS):
        checked=audit_campaign(paired,preset)
        values={**checked['autoresearch_config'],'data_dir':str(ROOT/'tasks/autoresearch/repo'),
            'desc_file':str(ROOT/'tasks/autoresearch/TASK.md'),'log_dir':str(campaign/method/'runs'),
            'workspace_dir':str(campaign/method/'runs'),'exp_name':preset}
        cfg=OmegaConf.merge(OmegaConf.structured(Config),values)
        runtime=Phase1SelectorRuntime(cfg);task=AutoresearchTask(cfg)
        if task.protocol_id!=read(pool/'manifest.json')['protocol_id']:
            raise ValueError('The host task/evaluation protocol changed')
        resolved=resolved_json(cfg)
        historical='BR04' if rating else 'BR02'
        before=flatten(read(old/'resolved_configs'/f'{historical}.json'));after=flatten(resolved)
        allowed={'log_dir','workspace_dir','exp_name','shared_draft_pool','parent_selector.type',
            'bo_parent_selector.model','study7.method_id','study7.schema','study7.rating.enabled',
            'branch7.enabled','bo_parent_selector.acquisition.type','bo_parent_selector.acquisition.beta'}
        changes=[]
        for key in sorted(before.keys()|after.keys()):
            if before.get(key)!=after.get(key):
                if key not in allowed and not key.startswith('acq7.'):
                    raise ValueError(f'Unapproved inherited-config difference: {method} {key}')
                changes.append({'key':key,'historical':before.get(key),'acq7':after.get(key)})
        diffs[method]={'historical_base':historical,'changes':changes}
        atomic_json(campaign/'resolved_configs'/f'{method}.json',resolved)
        OmegaConf.save(OmegaConf.create(resolved),campaign/'resolved_configs'/f'{method}.yaml')
        methods[method]={'preset':preset,'branch_kind':kind,'branch_prior_std':tau,
            'branch_lengthscale':1.,'slope_prior_std':slope,'rating':rating,'ancestor':False,
            'acquisition':acq,'task_protocol_id':task.protocol_id,'contract':runtime.study7_selector.contract(),
            'config_sha256':sha(campaign/'resolved_configs'/f'{method}.json'),
            'directory':str(campaign/method),'status':'implemented_not_smoke_validated'}
        (campaign/'method_cards'/f'{method}.md').write_text(
            f'# {method}: {preset}\n\nGP4 with {kind} branch covariance; tau={tau}; '
            f'fixed branch lengthscale=1; slope SD={slope}; original rating={rating}; ancestor B disabled.\n\n'
            f'Official acquisition: {acq}. Legacy UCB fields are inactive; UCB 1.5 is a read-only shadow. '
            'No fantasy pending, forced branch visit, root candidate or pre-generated proposal pool.\n\n'
            'Original GP4 valid ordinary closures, feature map, bounded MAP, initialization terminal gate, '
            'frozen reference, DEBUG, preview, generation, review, memory and FIFO are inherited.\n\n'
            'One A100-SXM4-80GB, four logical CPUs, two internal workers; seed 42; '
            '21600 cumulative active seconds; five prior slots, first three drafts. '
            'No historical feedback import, extra seed, performance-dependent stopping or method fallback.\n')
    spec=ROOT.parent/'EvolveFold_Acq7_Codex_Spec_zh.md'
    protocol={'version':'acq7_v1.0','spec_sha256':sha(spec),'methods':methods,
        'task_protocol_id':task.protocol_id,'duration_seconds':21600,'workers':2,'cpus':4,'seed':42,
        'data_manifest_sha256':sha(ROOT/'.cache/autoresearch/manifest.json'),
        'initial_pool_manifest_sha256':inputs['manifest.json'],'smoke_methods':list(SMOKE_METHODS),
        'smoke_seconds_each':SMOKE_SECONDS,'smoke_contract_origin':str(old/'resource_plan.json'),
        'smoke_contract_note':'Two bounded probes inherit the existing 3600-second maximum; stop after required closure.'}
    from engine.parent_selectors.study7_gp import digest
    protocol['protocol_id']=digest(protocol)
    atomic_json(campaign/'protocol.json',protocol)
    atomic_json(campaign/'compatibility_diff.json',{'effective_config_differences':diffs,
        'unchanged_task_protocol':task.protocol_id,'legacy_acquisition_fields':'inactive except read-only UCB shadow'})
    atomic_json(campaign/'source_manifest.json',{str(p.relative_to(ROOT)):sha(p) for p in source_files()})
    atomic_json(campaign/'registration.json',{'version':'acq7_v1','methods':methods,
        'source_snapshot':str(campaign/'source_snapshot'),'protocol_id':protocol['protocol_id'],
        'method_directories':{v['preset']:key for key,v in methods.items()},
        'method_order':list(methods),'pool_gpu_limits':CAPS,'resource_cap':7,
        'existing_allocations':{'unkillable':'10857987'},'allocation_job_prefix':'bo-acq7',
        'scheduling':'Oldest eligible AQ method; each resume keeps its cumulative budget and frozen identity.'})
    atomic_json(campaign/'resource_plan.json',{'campaign':str(campaign),'spec_sha256':sha(spec),
        'status':'existing_user_card_authorized_new_requests_need_current_confirmation',
        'pool_gpu_limits':CAPS,'maximum_allocated_gpus':7,
        'existing_user_allocation':{'job_id':'10857987','node':'cn-g018','gpus':1,
            'instruction':'unkillable任务可以用现在已经申请上的这个节点，我提前申请了占住的，里面没有任务在跑',
            'return_policy':'Retain parent allocation; stop only this campaign steps.'},
        'proposed_requests':requests(campaign),'formal_seconds_each':21600,'internal_workers':2,
        'smoke':{'methods':list(SMOKE_METHODS),'maximum_active_seconds_each':SMOKE_SECONDS,
            'resource':'Sequential probes on the explicitly authorized existing single GPU',
            'feedback_imported':False},'renewal_policy':'Only explicit current-round authority permits new allocation or renewal.',
        'no_cancel_other_jobs':True,'no_additional_arms_or_seeds':True})


def freeze(campaign):
    if (campaign/'campaign.json').exists():raise FileExistsError('Acq7 already frozen')
    files=read(campaign/'source_manifest.json')
    validation=read(campaign/'tests/cpu_validation.json')
    if not validation.get('passed') or validation.get('source_sha256')!=files:
        raise RuntimeError('CPU/integration source gate not passed')
    if not read(campaign/'tests/provider_probe.json').get('passed'):
        raise RuntimeError('Current original transport probe has not passed')
    if 'Verified all 13' not in (campaign/'tests/data_verification.log').read_text():
        raise RuntimeError('Current data/tokenizer verification missing')
    for method in SMOKE_METHODS:
        result=read(campaign/'tests'/f'smoke_{method}.json')
        if not result.get('passed') or result.get('source_sha256')!=files:
            raise RuntimeError(f'{method} bounded real smoke/source gate not passed')
    destination=campaign/'source_snapshot';destination.mkdir(exist_ok=False)
    for name,expected in files.items():
        if sha(ROOT/name)!=expected:raise RuntimeError(f'Source changed after validation: {name}')
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,target);target.chmod(0o555 if target.stat().st_mode&0o111 else 0o444)
    (destination/'.venv').symlink_to(ROOT/'.venv',target_is_directory=True)
    (destination/'.cache').mkdir()
    (destination/'.cache/autoresearch').symlink_to(ROOT/'.cache/autoresearch',target_is_directory=True)
    registration=read(campaign/'registration.json')
    manifest={**registration,'status':'frozen','frozen_at':datetime.now(timezone.utc).isoformat(),
        'source_sha256':files,'runtime_environment':{'MLEVOLVE_CODEX_REASONING_EFFORT':'high',
        'MLEVOLVE_CODEX_TIMEOUT_SECONDS':'1800','OPENCODE_API_KEY_FILE':str(ROOT.parent/'codex/opecode'),
        'AUTORESEARCH_CACHE_DIR':str(ROOT/'.cache/autoresearch'),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
        'ACQ7_SOURCE_MANIFEST_SHA256':sha(campaign/'source_manifest.json')}}
    paths=[campaign/'paired_config.yaml',campaign/'protocol.json',campaign/'registration.json',
           *sorted((campaign/'resolved_configs').glob('*')),*sorted((campaign/'draft_pool').glob('*.json'))]
    manifest['input_sha256']={str(p.relative_to(campaign)):sha(p) for p in paths}
    for path in paths:path.chmod(0o444)
    atomic_json(campaign/'campaign.json',manifest);(campaign/'campaign.json').chmod(0o444)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','freeze','dry-run'])
    parser.add_argument('--campaign',type=Path,default=ROOT/'research/acq7_20260920')
    args=parser.parse_args();campaign=args.campaign.resolve()
    if args.action=='prepare':prepare(campaign)
    elif args.action=='freeze':freeze(campaign)
    else:print(json.dumps(requests(campaign),indent=2))


if __name__=='__main__':main()
