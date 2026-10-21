"""Authorized Acq7 resource segments; never cancel the user's parent allocation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from config.acq7 import ACQ7_ARMS
from engine.shared_drafts import atomic_json
from tasks.autoresearch.acq7_campaign import CAPS, read, requests, sha
from tasks.autoresearch.branch7_resources import active_jobs, terminal, RENEWABLE
from tasks.autoresearch.supervise import process_table


def authorization(campaign,renew=False):
    auth=read(campaign/'resource_authorization.json')
    if (auth.get('campaign')!=str(campaign.resolve())
            or auth.get('spec_sha256')!=read(campaign/'protocol.json')['spec_sha256']
            or auth.get('pool_gpu_limits')!=CAPS or not auth.get('user_instruction')
            or auth.get('allow_new_allocations') is not True
            or (renew and auth.get('allow_renewals') is not True)):
        raise RuntimeError('Current-round Acq7 allocation/renewal authorization is missing')
    return auth


def event(campaign,kind,**fields):
    with (campaign/'controller_events.jsonl').open('a') as stream:
        stream.write(json.dumps({'at':datetime.now(timezone.utc).isoformat(),'kind':kind,**fields})+'\n')


def validate_pool(state,live):
    jobs=state.get('current_jobs',{})
    names={f'bo-acq7-{group}' for group in ('main','short')}
    if any(item['name'] in names and job not in jobs.values() for job,item in live.items()):
        raise RuntimeError('Unregistered Acq7 allocations exist; no overlapping request is allowed')
    for group,job in jobs.items():
        if group!='unkillable' and job in live and live[job]['name']!=f'bo-acq7-{group}':
            raise RuntimeError('A registered job changed ownership/name')
        if group=='unkillable' and job!='10857987':
            raise RuntimeError('The user preallocation identity changed')
    reserved=sum(CAPS[group] for group,job in jobs.items() if job in live or terminal(job) is None)
    if reserved>7:raise RuntimeError('The Acq7 pool exceeds seven GPUs')
    return reserved


def submit(campaign,state,group,renew=False):
    authorization(campaign,renew)
    if group not in ('main','short'):raise RuntimeError('Do not request another unkillable allocation')
    live=active_jobs();reserved=validate_pool(state,live)
    previous=state.get('current_jobs',{}).get(group)
    if previous and (previous in live or terminal(previous) is None):
        raise RuntimeError('Previous allocation is not authoritatively terminal')
    if reserved+CAPS[group]>7:raise RuntimeError('Request exceeds the authorized seven-card pool')
    command=requests(campaign)[group]['command']
    result=subprocess.run(command,text=True,capture_output=True,timeout=60)
    if result.returncode:
        event(campaign,'submission_failed',group=group,command=command,stderr=result.stderr)
        raise RuntimeError('Submission failed; inspect evidence before retry')
    job=result.stdout.strip().split(';')[0]
    if not job.isdigit():raise RuntimeError('Ambiguous sbatch response; reconciliation required')
    state.setdefault('current_jobs',{})[group]=job
    state.setdefault('job_history',{}).setdefault(group,[]).append(job)
    atomic_json(campaign/'controller_state.json',state)
    event(campaign,'submitted',group=group,job_id=job,renewal=renew,command=command)


def adoption_command(campaign):
    return ['srun','--jobid=10857987','--overlap','--nodes=1','--ntasks=1',
        '--cpus-per-task=4','--gres=gpu:a100l:1','--time=07:00:00',
        'env',f'ACQ7_LEASE_END_EPOCH={time.time()+7*3600}',
        'bash',str(campaign/'allocation.sh'),str(campaign),'unkillable']


def local_handle_live(handle):
    value=process_table().get(handle['pid'])
    return bool(value and value[1]==handle['start_ticks'] and value[2]!='Z')


def step_terminal(job,step):
    if not str(step).isdigit():return False
    target=f'{job}.{step}'
    output=subprocess.check_output(['sacct','-n','-P','-j',target,'-o','JobIDRaw,State,End'],text=True,timeout=30)
    for line in output.splitlines():
        fields=line.split('|')
        if fields[0]==target and fields[1].split()[0].rstrip('+') in {'COMPLETED','FAILED','CANCELLED','TIMEOUT','PREEMPTED','NODE_FAIL','OUT_OF_MEMORY'}:
            return {'evidence':line,'state':fields[1].split()[0]}
    return False


def adopt(campaign,state):
    previous=state.get('adopted_step')
    if previous:
        if local_handle_live(previous):return
        worker=read(campaign/'worker_unkillable.json') if (campaign/'worker_unkillable.json').exists() else None
        if not worker or not step_terminal('10857987',worker.get('step_id')):
            raise RuntimeError('Previous adopted executor release is unproven; automatic restart forbidden')
        if worker.get('failed_methods'):raise RuntimeError('Adopted workflow failed; diagnose before resume')
    if '10857987' not in active_jobs():raise RuntimeError('The authorized user preallocation is not live')
    command=adoption_command(campaign)
    with (campaign/'unkillable-adopted.log').open('a') as log:
        proc=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,
                              stderr=subprocess.STDOUT,start_new_session=True)
    identity=process_table().get(proc.pid)
    if not identity:raise RuntimeError('Adopted step failed to launch; inspect log')
    state['adopted_step']={'pid':proc.pid,'start_ticks':identity[1],'command':command,'started_epoch':time.time()}
    state.setdefault('current_jobs',{})['unkillable']='10857987'
    state.setdefault('job_history',{}).setdefault('unkillable',['10857987'])
    atomic_json(campaign/'controller_state.json',state)
    event(campaign,'adopted_step_started',**state['adopted_step'])


def statuses(campaign):
    return {method:read(path) if (path:=campaign/method/'status.json').exists() else None for method in ACQ7_ARMS}


def completed(values):
    return len(values)==7 and all(value and value['state']=='finished'
        and abs(float(value.get('active_elapsed_seconds',0))-21600)<1e-6 for value in values.values())


def freeze_gate(campaign):
    manifest=read(campaign/'campaign.json')
    if manifest.get('version')!='acq7_v1' or set(manifest['methods'])!=set(ACQ7_ARMS):
        raise RuntimeError('Only the frozen seven AQ arms may run')
    for name,digest in manifest['source_sha256'].items():
        if sha(Path(manifest['source_snapshot'])/name)!=digest:
            raise RuntimeError('Frozen source changed')
    for name,digest in manifest['input_sha256'].items():
        if sha(campaign/name)!=digest:raise RuntimeError('Frozen campaign input changed')
    return manifest


def controller(campaign):
    freeze_gate(campaign)
    from tasks.autoresearch.acq7_worker import eligible_methods
    with (campaign/'controller.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=read(campaign/'controller_state.json') if (campaign/'controller_state.json').exists() else {
            'current_jobs':{'unkillable':'10857987'},'job_history':{},'status':'prepared'}
        state.update(pid=os.getpid(),start_ticks=process_table()[os.getpid()][1],status='monitoring')
        event(campaign,'controller_started',pid=os.getpid())
        atomic_json(campaign/'controller_state.json',state)
        observation_errors=0
        while True:
            try:
                live=active_jobs();validate_pool(state,live);values=statuses(campaign)
                complete=completed(values)
                claims=read(campaign/'assignments.json') if (campaign/'assignments.json').exists() else {}
                if not complete:
                    if (campaign/'resource_authorization.json').exists():
                        for group in ('main','short'):
                            job=state['current_jobs'].get(group)
                            if job is None:
                                authorization(campaign)
                                if eligible_methods(campaign,claims,'new',group,live):submit(campaign,state,group)
                            elif job not in live:
                                end=terminal(job)
                                if end and end['state'] in RENEWABLE:
                                    if eligible_methods(campaign,claims,'new',group,live):
                                        authorization(campaign,renew=True);submit(campaign,state,group,renew=True)
                    # Reuse the existing card; absence of new-allocation permission
                    # still permits sequential progress in the prescribed order.
                    if eligible_methods(campaign,claims,'10857987','unkillable',live):
                        adopt(campaign,state)
                live=active_jobs();reserved=validate_pool(state,live)
                adopted_active=bool(state.get('adopted_step') and local_handle_live(state['adopted_step']))
                allocated=sum(CAPS[group] for group,job in state['current_jobs'].items()
                    if job in live and live[job]['state'] in {'RUNNING','COMPLETING','SUSPENDED'}
                    and (group!='unkillable' or adopted_active))
                values=statuses(campaign)
                running=sum(bool(value and value['state'] in {'running','starting','pausing'}
                    and value.get('slurm_job_id') in live) for value in values.values())
                pending=sum(CAPS[group] for group,job in state['current_jobs'].items()
                            if job in live and live[job]['state']=='PENDING')
                state.update(heartbeat_epoch=time.time(),allocated_gpus=allocated,
                    reserved_gpus=reserved,running_workflows=running,idle_allocated_slots=max(0,allocated-running),
                    pending_gpus=pending,formal_completed=sum(bool(v and v['state']=='finished'
                    and abs(v.get('active_elapsed_seconds',0)-21600)<1e-6) for v in values.values()))
                new_live={group:job for group,job in state['current_jobs'].items() if group!='unkillable' and job in live}
                if complete and not adopted_active and not new_live:
                    state.update(status='workflows_complete_resources_returned',pid=None,
                        allocated_gpus=0,running_workflows=0,idle_allocated_slots=0,
                        user_allocation_retained='10857987',user_allocation_cancelled=False)
                    atomic_json(campaign/'commands/resource_return_receipt.json',{
                        'at':datetime.now(timezone.utc).isoformat(),'user_allocation_retained':'10857987',
                        'allocation_cancelled':False,'current_jobs':state['current_jobs'],
                        'terminal_new_jobs':{g:terminal(j) for g,j in state['current_jobs'].items() if g!='unkillable'},
                        'adopted_step':read(campaign/'worker_unkillable.json') if (campaign/'worker_unkillable.json').exists() else None})
                atomic_json(campaign/'controller_state.json',state)
                research=read(campaign/'RESEARCH_STATE.json')
                research.update(status='workflows_complete_analysis_pending' if complete else 'running',
                    updated_at=datetime.now(timezone.utc).isoformat(),current_workflows=values,
                    current_allocations=state['current_jobs'],allocated_gpus=state['allocated_gpus'],
                    running_workflows=state['running_workflows'],idle_allocated_slots=state['idle_allocated_slots'])
                atomic_json(campaign/'RESEARCH_STATE.json',research)
                if state['status']=='workflows_complete_resources_returned':
                    event(campaign,'all_seven_full_budgets_complete');return
                observation_errors=0
            except (subprocess.SubprocessError,OSError) as exc:
                observation_errors+=1
                event(campaign,'observation_error',error_type=type(exc).__name__,consecutive=observation_errors)
                if observation_errors>=3:raise
            time.sleep(30)


def launch(campaign):
    freeze_gate(campaign)
    launch_path=campaign/'commands/controller_launch.json'
    if launch_path.exists() and local_handle_live(read(launch_path)):
        raise RuntimeError('The Acq7 controller is already live')
    command=[sys.executable,__file__,'controller','--campaign',str(campaign)]
    with (campaign/'controller.log').open('a') as stream:
        process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,
                                 stderr=subprocess.STDOUT,start_new_session=True)
    identity=process_table().get(process.pid)
    if not identity:raise RuntimeError('Controller did not start')
    receipt={'pid':process.pid,'start_ticks':identity[1],'command':command,'at':datetime.now(timezone.utc).isoformat()}
    atomic_json(launch_path,receipt);print(json.dumps(receipt))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['launch','resume','controller'])
    parser.add_argument('--campaign',type=Path,default=ROOT/'research/acq7_20260920')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args();campaign=args.campaign.resolve()
    if args.action=='controller':controller(campaign)
    elif args.dry_run or not args.execute:
        print(json.dumps({'existing_step':adoption_command(campaign),'new_requests':requests(campaign),
            'allocates_nothing':True,'requires_frozen_campaign':True,
            'new_allocations_require_current_resource_authorization':True},indent=2))
    else:launch(campaign)
