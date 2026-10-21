"""Two bounded real Branch7 probes; isolated from all seven formal runs."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from engine.shared_drafts import atomic_json
from tasks.autoresearch.branch7_campaign import source_files,sha,read


def probe(campaign,method):
    from config import load_cfg,load_task_desc,prep_agent_workspace,save_run
    from config.branch7 import BRANCH7_PRESETS
    from engine.agent_search import AgentSearch
    from engine.search_node import Journal
    from tasks.autoresearch.adapter import AutoresearchInterpreter
    from utils.logging_config import setup_logging
    from utils.seed import set_global_seed
    output=campaign/'smoke'/method
    output.mkdir(parents=True,exist_ok=False)
    preset=BRANCH7_PRESETS[int(method[2:])-1]
    sys.argv=[sys.argv[0],'task_preset=autoresearch',f'experiment_preset={preset}',
        'cpu_number=4','agent.steps=5','agent.initial_drafts=2','agent.time_limit=3600',
        'agent.search.parallel_search_num=2','shared_draft_pool=null',f'exp_name=smoke_{method}',
        f'log_dir={output}/runs',f'workspace_dir={output}/runs']
    cfg=load_cfg();setup_logging(cfg);set_global_seed(cfg.agent.seed);prep_agent_workspace(cfg)
    agent=AgentSearch(load_task_desc(cfg),cfg,Journal())
    interpreter=AutoresearchInterpreter(cfg.workspace_dir,**dict(cfg.exec),cfg=cfg)
    started=time.time()
    report={'passed':False,'method_id':method,'started_epoch':started,
        'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in source_files()},
        'slurm_job_id':os.environ['SLURM_JOB_ID'],'slurm_step_id':os.environ.get('SLURM_STEP_ID'),
        'gpu':os.environ['CUDA_VISIBLE_DEVICES'],'parallel_workers':2,'cpus':sorted(os.sched_getaffinity(0)),
        'task_protocol_id':agent.autoresearch.protocol_id,'smoke_only_overrides':{
            'initial_drafts':2,'steps':5,'workflow_budget':3600,'shared_draft_pool':None},
        'formal_feedback_exported':False}
    atomic_json(output/'run_location.json',{'log_dir':str(cfg.log_dir),'workspace_dir':str(cfg.workspace_dir)})
    def save(stage):
        report.update(stage=stage,elapsed_seconds=time.time()-started)
        atomic_json(campaign/'tests'/f'smoke_{method}.json',report)
        agent.capture_resume_state();save_run(cfg,agent.journal)
    try:
        first=read(campaign/'draft_pool/draft_1.json')['code']
        initial=output/'initial_code.py';initial.write_text(first)
        drafts=[agent.step(None,interpreter.run,execute_immediately=False,init_solution_path=str(initial)),
                agent.step(None,interpreter.run,execute_immediately=False)]
        save('two_initial_candidates_generated')
        with ThreadPoolExecutor(max_workers=2) as workers:
            pending=[workers.submit(agent.execute_deferred_node,n,interpreter.run) for n in drafts]
            for future in pending:future.result()
        if not all(n.is_valid and n.study7_rating_metadata.get('prompt_sha256') for n in drafts):
            raise RuntimeError('A real initial training/review failed; diagnose before another smoke')
        save('two_concurrent_trainings_and_ratings_valid')
        child=agent.step(drafts[0],interpreter.run)
        repair=False
        if not child.is_valid:
            repair=True;save('ordinary_failed_one_debug_allowed')
            child=agent.step(child,interpreter.run)
        if not child.is_valid:raise RuntimeError('Ordinary smoke closure failed after one DEBUG')
        ctx=agent.phase1_runtime.select_continuation_parent(agent,child)
        agent.phase1_runtime.cancel_reservation(agent,ctx)
        selector=agent.phase1_runtime.study7_selector
        normal=[row for row in selector.gp.rows if row['operator']!='draft']
        expected=3 if method=='BR06' else 1
        if len(normal)!=1 or len(selector.gp.rows)!=expected:
            raise RuntimeError('Real closure admission does not match its GP4/GP6 base')
        if any(row['branch_effect_load']!=1 or not row['root_lineage_id'] for row in normal):
            raise RuntimeError('Real ordinary closure lost branch attribution')
        for row in selector.gp.rows:
            if row['operator']=='draft' and (row['branch_effect_load'] or row['rating_s']):
                raise RuntimeError('Real GP6 draft leaked branch or rating')
        report.update(passed=True,normal_labels=1,total_training_rows=expected,
            initial_ratings=[n.agent_potential_level for n in drafts],child_rating=child.agent_potential_level,
            repaired=repair,child_id=child.id,model_version=selector.gp.model_version,
            reference=selector.state.reference,branch_posterior=selector.gp.branch_posterior(
                [v['root_lineage_id'] for v in selector.state.lineages.roots.values()],selector.state.reference['s0']))
        save('passed')
    except BaseException as exc:
        report.update(error_type=type(exc).__name__,error=str(exc));save('failed');raise
    finally:interpreter.terminate_all_subprocesses()


def group(campaign):
    from tasks.autoresearch.study_worker import allocation_info,visible_devices
    from tasks.autoresearch.launch import compile_cache_environment
    from tasks.autoresearch.branch7_resources import authorization
    from tasks.autoresearch.supervise import stop_tree
    authorization(campaign)
    allocation=allocation_info(os.environ['SLURM_JOB_ID'])
    state=read(campaign/'controller_state.json')
    if (allocation['JobName']!='bo-branch7-main'
            or state['current_jobs'].get('main')!=os.environ['SLURM_JOB_ID']):
        raise RuntimeError('Smoke must use this campaign’s currently registered main allocation')
    devices=visible_devices();cpus=sorted(os.sched_getaffinity(0))
    if len(devices)!=2 or len(cpus)<8:raise RuntimeError('Two smoke workflows require two GPUs and eight CPUs')
    procs=[]
    try:
        for i,method in enumerate(['BR05','BR06']):
            cache=Path(tempfile.mkdtemp(prefix=f'branch7-smoke-{method}-',dir=os.environ.get('SLURM_TMPDIR')))
            env={**os.environ,'CUDA_VISIBLE_DEVICES':devices[i]['uuid'],'PYTHONUNBUFFERED':'1',
                'MLEVOLVE_CODEX_BIN':str(ROOT/'tools/codex_opencode.py'),
                'MLEVOLVE_CODEX_REASONING_EFFORT':'high','MLEVOLVE_CODEX_TIMEOUT_SECONDS':'1800',
                'OPENCODE_API_KEY_FILE':str(ROOT.parent/'codex/opecode'),
                'AUTORESEARCH_CACHE_DIR':str(ROOT/'.cache/autoresearch'),'AUTORESEARCH_CPUS':'4',
                'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4',
                'RAYON_NUM_THREADS':'4','TORCHINDUCTOR_COMPILE_THREADS':'4',
                'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',**compile_cache_environment(cache)}
            command=['taskset','-c',','.join(map(str,cpus[4*i:4*i+4])),sys.executable,__file__,
                     'probe','--campaign',str(campaign),'--method',method]
            with (campaign/'tests'/f'smoke_{method}.log').open('x') as log:
                proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            procs.append(proc)
        deadline=time.monotonic()+3600
        while any(p.poll() is None for p in procs) and time.monotonic()<deadline:time.sleep(5)
        for proc in procs:
            if proc.poll() is None:stop_tree(proc)
        if any(p.returncode for p in procs):raise RuntimeError('A bounded Branch7 smoke failed; inspect isolated logs')
    finally:
        for proc in procs:
            if proc.poll() is None:stop_tree(proc)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['group','probe']);parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--method',choices=['BR05','BR06'])
    args=parser.parse_args()
    if args.action=='group':group(args.campaign.resolve())
    else:
        gpu=os.environ['CUDA_VISIBLE_DEVICES']
        if not gpu.startswith('GPU-') or ',' in gpu:raise RuntimeError('Probe requires one allocated GPU UUID')
        lock=ROOT/'.cache/gpu-locks'/f'{gpu}.lock';lock.parent.mkdir(parents=True,exist_ok=True)
        with lock.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);probe(args.campaign.resolve(),args.method)
