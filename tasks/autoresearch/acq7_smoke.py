"""Two isolated bounded real probes on the user's existing single GPU."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from engine.shared_drafts import atomic_json
from tasks.autoresearch.acq7_campaign import SMOKE_METHODS, SMOKE_SECONDS, read, sha, source_files


def probe(campaign,method):
    from config import load_cfg,load_task_desc,prep_agent_workspace,save_run
    from config.acq7 import ACQ7_PRESETS
    from engine.agent_search import AgentSearch
    from engine.search_node import Journal
    from engine.parent_selectors.acq7_selector import Acq7Selector
    from tasks.autoresearch.adapter import AutoresearchInterpreter
    from utils.logging_config import setup_logging
    from utils.seed import set_global_seed
    output=campaign/'smoke'/method
    output.mkdir(parents=True,exist_ok=False)
    preset=ACQ7_PRESETS[int(method[2:])-1]
    sys.argv=[sys.argv[0],'task_preset=autoresearch',f'experiment_preset={preset}',
        'cpu_number=4','agent.steps=5','agent.initial_drafts=2',f'agent.time_limit={SMOKE_SECONDS}',
        'agent.search.parallel_search_num=2','shared_draft_pool=null',f'exp_name=smoke_{method}',
        f'log_dir={output}/runs',f'workspace_dir={output}/runs']
    cfg=load_cfg();setup_logging(cfg);set_global_seed(cfg.agent.seed);prep_agent_workspace(cfg)
    agent=AgentSearch(load_task_desc(cfg),cfg,Journal())
    interpreter=AutoresearchInterpreter(cfg.workspace_dir,**dict(cfg.exec),cfg=cfg)
    started=time.time()
    report={'passed':False,'method_id':method,'started_epoch':started,
        'source_sha256':{str(path.relative_to(ROOT)):sha(path) for path in source_files()},
        'slurm_job_id':os.environ['SLURM_JOB_ID'],'slurm_step_id':os.environ.get('SLURM_STEP_ID'),
        'gpu':os.environ['CUDA_VISIBLE_DEVICES'],'parallel_workers':2,'cpus':sorted(os.sched_getaffinity(0)),
        'task_protocol_id':agent.autoresearch.protocol_id,'smoke_only_overrides':{
            'initial_drafts':2,'steps':5,'workflow_budget':SMOKE_SECONDS,'shared_draft_pool':None},
        'formal_feedback_exported':False}
    atomic_json(output/'run_location.json',{'log_dir':str(cfg.log_dir),'workspace_dir':str(cfg.workspace_dir)})
    def save(stage):
        report.update(stage=stage,elapsed_seconds=time.time()-started)
        atomic_json(campaign/'tests'/f'smoke_{method}.json',report)
        agent.capture_resume_state();save_run(cfg,agent.journal)
    try:
        initial=output/'initial_code.py';initial.write_text(read(campaign/'draft_pool/draft_1.json')['code'])
        drafts=[agent.step(None,interpreter.run,execute_immediately=False,init_solution_path=str(initial)),
                agent.step(None,interpreter.run,execute_immediately=False)]
        save('two_initial_candidates_generated')
        with ThreadPoolExecutor(max_workers=2) as workers:
            pending=[workers.submit(agent.execute_deferred_node,node,interpreter.run) for node in drafts]
            for future in pending:future.result()
        if not all(node.is_valid for node in drafts):
            raise RuntimeError('An isolated initial training/review failed; diagnose before retry')
        rated=bool(cfg.study7.rating.enabled)
        if rated and not all(node.study7_rating_metadata.get('prompt_sha256') for node in drafts):
            raise RuntimeError('Rating-enabled smoke did not execute the inherited review extension')
        if not rated and any(node.agent_potential_level is not None for node in drafts):
            raise RuntimeError('Rating leaked into an unrated smoke')
        save('two_concurrent_trainings_and_reviews_valid')
        child=agent.step(drafts[0],interpreter.run)
        repair=False
        if not child.is_valid:
            repair=True;save('ordinary_failed_one_debug_allowed')
            child=agent.step(child,interpreter.run)
        if not child.is_valid:raise RuntimeError('Ordinary smoke closure failed after one DEBUG')
        selector=agent.phase1_runtime.study7_selector
        selector.observe(agent.phase1_runtime.transition_store.records())
        if len(selector.gp.rows)!=1 or selector.gp.rows[0]['branch_effect_load']!=1:
            raise RuntimeError('Real Acq7 closure did not produce exactly one ordinary GP4 label')
        frozen=copy.deepcopy(selector.state_dict())
        restored=Acq7Selector(config=selector.config,records_provider=selector.records_provider,selection_logger=None)
        restored.load_state_dict(frozen);restored.observe(selector.records_provider())
        before=selector.gp.predict_joint(selector.gp.rows,selector.state.reference['s0'])
        after=restored.gp.predict_joint(restored.gp.rows,restored.state.reference['s0'])
        if before['posterior_hash']!=after['posterior_hash'] or restored.rng_state!=selector.rng_state:
            raise RuntimeError('Real smoke save/load changed posterior or official RNG')
        decisions=selector.committed_draws
        if not decisions:raise RuntimeError('Smoke did not execute the actual Acq7 reservation path')
        report.update(passed=True,normal_labels=1,total_training_rows=1,
            initial_ratings=[node.agent_potential_level for node in drafts],child_rating=child.agent_potential_level,
            repaired=repair,child_id=child.id,model_version=selector.gp.model_version,
            reference=selector.state.reference,acquisition=selector.config.acquisition_type,
            accepted_ts_draws=selector.accepted_draw_index,ordinary_decisions=len(decisions),
            save_load_posterior_and_rng_equal=True,posterior_hash=before['posterior_hash'])
        save('passed')
    except BaseException as exc:
        report.update(error_type=type(exc).__name__,error=str(exc));save('failed');raise
    finally:interpreter.terminate_all_subprocesses()


def lease(campaign):
    from tasks.autoresearch.study_worker import allocation_info,visible_devices
    from tasks.autoresearch.launch import compile_cache_environment
    from tasks.autoresearch.supervise import stop_tree
    if os.environ.get('SLURM_JOB_ID')!='10857987':
        raise RuntimeError('Smoke must use the explicitly authorized existing allocation')
    allocation_info('10857987');devices=visible_devices();cpus=sorted(os.sched_getaffinity(0))
    if len(devices)!=1 or len(cpus)<4:raise RuntimeError('Smoke needs exactly one A100 and four allocated CPUs')
    expected=read(campaign/'tests/cpu_validation.json')
    current={str(path.relative_to(ROOT)):sha(path) for path in source_files()}
    if not expected.get('passed') or expected['source_sha256']!=current:
        raise RuntimeError('Current CPU/integration source gate has not passed')
    if not read(campaign/'tests/provider_probe.json').get('passed'):
        raise RuntimeError('Original provider is not currently verified')
    if 'Verified all 13' not in (campaign/'tests/data_verification.log').read_text():
        raise RuntimeError('Data/tokenizer verification missing')
    for method in SMOKE_METHODS:
        cache=Path(tempfile.mkdtemp(prefix=f'acq7-smoke-{method}-',dir=os.environ.get('SLURM_TMPDIR')))
        env={**os.environ,'CUDA_VISIBLE_DEVICES':devices[0]['uuid'],'PYTHONUNBUFFERED':'1',
            'MLEVOLVE_CODEX_BIN':str(ROOT/'tools/codex_opencode.py'),
            'MLEVOLVE_CODEX_REASONING_EFFORT':'high','MLEVOLVE_CODEX_TIMEOUT_SECONDS':'1800',
            'OPENCODE_API_KEY_FILE':str(ROOT.parent/'codex/opecode'),
            'AUTORESEARCH_CACHE_DIR':str(ROOT/'.cache/autoresearch'),'AUTORESEARCH_CPUS':'4',
            'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','RAYON_NUM_THREADS':'4',
            'TORCHINDUCTOR_COMPILE_THREADS':'4','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
            'ACQ7_SOURCE_MANIFEST_SHA256':sha(campaign/'source_manifest.json'),**compile_cache_environment(cache)}
        command=['taskset','-c',','.join(map(str,cpus[:4])),sys.executable,__file__,
                 'probe','--campaign',str(campaign),'--method',method]
        started=time.time()
        with (campaign/'tests'/f'smoke_{method}.log').open('x') as log:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(campaign/'commands'/f'smoke_{method}_launch.json',{
            'command':command,'pid':process.pid,'started_epoch':started,'maximum_seconds':SMOKE_SECONDS,
            'job_id':os.environ['SLURM_JOB_ID'],'step_id':os.environ.get('SLURM_STEP_ID')})
        try:
            while process.poll() is None and time.time()-started<SMOKE_SECONDS:time.sleep(2)
            if process.poll() is None:stop_tree(process)
            path=campaign/'tests'/f'smoke_{method}.json'
            result=read(path) if path.exists() else {'passed':False,'method_id':method,'stage':'failed_before_probe_report'}
            result.update(charged_wall_seconds=min(time.time()-started,SMOKE_SECONDS),
                cleanup_wall_seconds=max(0.,time.time()-started-SMOKE_SECONDS),exit_code=process.returncode)
            atomic_json(path,result)
            if process.returncode or not result.get('passed'):
                raise RuntimeError(f'{method} bounded smoke failed; inspect its evidence before retry')
        finally:
            if process.poll() is None:stop_tree(process)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['lease','probe']);parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--method',choices=SMOKE_METHODS)
    args=parser.parse_args()
    if args.action=='lease':lease(args.campaign.resolve())
    else:
        if not args.method:parser.error('--method is required for a probe')
        gpu=os.environ['CUDA_VISIBLE_DEVICES']
        if not gpu.startswith('GPU-') or ',' in gpu:raise RuntimeError('Probe must have one allocated GPU UUID')
        lock=ROOT/'.cache/gpu-locks'/f'{gpu}.lock';lock.parent.mkdir(parents=True,exist_ok=True)
        with lock.open('a+') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);probe(args.campaign.resolve(),args.method)
