"""Finite Branch7 controller: explicit current authorization, no foreign jobs."""
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
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from engine.shared_drafts import atomic_json
from tasks.autoresearch.branch7_campaign import requests, read, sha

CAPS={'main':2,'unkillable':1,'short':4}
TERMINAL={'COMPLETED','TIMEOUT','PREEMPTED','NODE_FAIL','BOOT_FAIL','REVOKED',
          'CANCELLED','FAILED','OUT_OF_MEMORY','DEADLINE'}
RENEWABLE={'COMPLETED','TIMEOUT','PREEMPTED','NODE_FAIL','BOOT_FAIL','REVOKED'}


def authorization(campaign,renew=False):
    path=campaign/'resource_authorization.json'
    if not path.exists():raise RuntimeError('Current Branch7 resource authorization is missing')
    auth=read(path)
    expected=read(campaign/'protocol.json')['spec_sha256']
    if (auth.get('campaign')!=str(campaign.resolve()) or auth.get('spec_sha256')!=expected
            or auth.get('pool_gpu_limits')!=CAPS or not auth.get('user_instruction')
            or auth.get('allow_new_allocations') is not True):
        raise RuntimeError('Authorization does not cover this exact Branch7 campaign and pool')
    if renew and auth.get('allow_renewals') is not True:
        raise RuntimeError('Current Branch7 renewal authorization is missing')
    return auth


def active_jobs():
    result=subprocess.run(['squeue','-h','-u',str(os.getuid()),'-o','%i|%T|%j|%R'],
                          text=True,capture_output=True,check=True,timeout=30)
    return {p[0]:{'state':p[1],'name':p[2],'node_or_reason':p[3]}
            for line in result.stdout.splitlines() if len(p:=line.strip().split('|',3))==4}


def terminal(job):
    result=subprocess.run(['sacct','-n','-X','-P','-j',job,'-o','JobIDRaw,State,End,ExitCode'],
                          text=True,capture_output=True,check=True,timeout=30)
    for line in result.stdout.splitlines():
        p=line.split('|')
        if len(p)>=4 and p[0]==job and (state:=p[1].split()[0].rstrip('+')) in TERMINAL:
            return {'state':state,'end':p[2],'exit_code':p[3],'evidence':line}
    return None


def event(campaign,kind,**fields):
    with (campaign/'controller_events.jsonl').open('a') as stream:
        stream.write(json.dumps({'at':datetime.now(timezone.utc).isoformat(),'kind':kind,**fields})+'\n')


def validate_pool(state,live):
    jobs=state.get('current_jobs',{})
    names={f'bo-branch7-{g}' for g in CAPS}
    unknown=[job for job,entry in live.items() if entry['name'] in names and job not in jobs.values()]
    if unknown:raise RuntimeError(f'Unregistered Branch7 job(s); refuse overlapping requests: {unknown}')
    for group,job in jobs.items():
        if job in live and live[job]['name']!=f'bo-branch7-{group}':
            raise RuntimeError('Registered job belongs to a different allocation name')
    reserved=sum(CAPS[g] for g,j in jobs.items() if j in live)
    if reserved>7:raise RuntimeError('Branch7 resource pool exceeded')
    return reserved


def method_states(campaign,registration,group):
    return [read(p) if (p:=campaign/registration['method_directories'][preset]/'status.json').exists() else None
            for preset in registration['groups'][group]]


def group_decision(statuses,end):
    if all(s and s.get('state')=='finished' and s.get('active_elapsed_seconds',0)>=21600 for s in statuses):
        return 'complete'
    if any(s and (str(s.get('state','')).startswith('blocked/') or s.get('state') in {'failed','early_stopped'}) for s in statuses):
        return 'blocked_method'
    if any(s and s.get('state')=='finished' and s.get('active_elapsed_seconds',0)<21600 for s in statuses):
        return 'blocked_incomplete_budget'
    if end is None:return 'await_accounting'
    if end['state'] not in RENEWABLE:return 'blocked_allocation'
    if any(s is None for s in statuses):return 'blocked_missing_launch'
    return 'renew'


def submit(campaign,state,group,renew=False):
    authorization(campaign,renew=renew)
    live=active_jobs();validate_pool(state,live)
    # Treat scheduler visibility lag as still reserved until terminal evidence.
    outstanding=0
    for g,job in state.get('current_jobs',{}).items():
        end=None if job in live else terminal(job)
        if end is None:outstanding+=CAPS[g]
        if g==group and end is None:raise RuntimeError('Previous allocation is not terminal')
    if outstanding+CAPS[group]>7:raise RuntimeError('Request would exceed seven-card authorization')
    command=requests(campaign)[group]['command']
    result=subprocess.run(command,text=True,capture_output=True,timeout=60)
    if result.returncode:
        event(campaign,'submission_failed',group=group,stderr=result.stderr,command=command)
        raise RuntimeError(f'Branch7 submission failed for {group}: {result.stderr.strip()}')
    job=result.stdout.strip().split(';')[0]
    if not job.isdigit():raise RuntimeError('Ambiguous sbatch result; do not retry without reconciliation')
    state.setdefault('current_jobs',{})[group]=job
    state.setdefault('job_history',{}).setdefault(group,[]).append(job)
    state.setdefault('submission_commands',{})[job]=command
    atomic_json(campaign/'controller_state.json',state)
    event(campaign,'submitted',group=group,job_id=job,renewal=renew,command=command)
    return job


def cpu_gate(campaign):
    suites=list(ET.parse(campaign/'tests/cpu_final.xml').getroot().iter('testsuite'))
    if not suites or any(int(s.attrib[k]) for s in suites for k in ('failures','errors')):
        raise RuntimeError('Branch7 CPU/integration gate not passed')
    if 'Verified all 13' not in (campaign/'tests/data_verification.log').read_text():
        raise RuntimeError('Data verification missing')
    for name,expected in read(campaign/'source_manifest.json').items():
        if sha(ROOT/name)!=expected:raise RuntimeError(f'Unvalidated source change: {name}')


def submit_initial(campaign,smoke_only=False,resume=False):
    campaign=campaign.resolve();authorization(campaign)
    cpu_gate(campaign)
    if not smoke_only and not (campaign/'campaign.json').exists():
        raise RuntimeError('Formal launch requires both real smoke gates and frozen source/config')
    with (campaign/'controller.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=read(campaign/'controller_state.json') if (campaign/'controller_state.json').exists() else {
            'current_jobs':{},'job_history':{},'status':'prepared','pid':None}
        registration=read(campaign/'registration.json')
        launched={}
        for group in (['main'] if smoke_only else CAPS):
            if group in state['current_jobs']:
                job=state['current_jobs'][group]
                if job in active_jobs():continue
                if not resume:raise RuntimeError('Use resume after a previously submitted allocation terminates')
                decision=group_decision(method_states(campaign,registration,group),terminal(job))
                if decision=='complete':continue
                if decision!='renew':raise RuntimeError(f'Cannot resume {group}: {decision}')
                launched[group]=submit(campaign,state,group,renew=True)
            else:launched[group]=submit(campaign,state,group)
        if not smoke_only:
            for group in state['current_jobs']:(campaign/f'launch_{group}.ready').touch()
        state['status']='smoke_resources_submitted' if smoke_only else 'formal_submitted'
        atomic_json(campaign/'controller_state.json',state)
    print(json.dumps({'submitted':launched,'smoke_only':smoke_only}))


def controller(campaign):
    campaign=campaign.resolve();registration=read(campaign/'campaign.json')
    with (campaign/'controller.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=read(campaign/'controller_state.json');state.update(pid=os.getpid(),status='monitoring')
        event(campaign,'controller_started',pid=os.getpid())
        last_analysis=0;errors=0
        while True:
            try:
                live=active_jobs();validate_pool(state,live)
                statuses={}
                for group,job in list(state['current_jobs'].items()):
                    group_status=method_states(campaign,registration,group)
                    for preset,item in zip(registration['groups'][group],group_status):
                        statuses[registration['method_directories'][preset]]=item
                    if job in live:continue
                    end=terminal(job);decision=group_decision(group_status,end)
                    state.setdefault('groups',{})[group]={'job_id':job,'terminal':end,'decision':decision}
                    if decision=='renew':
                        try:authorization(campaign,renew=True)
                        except RuntimeError:
                            state['groups'][group]['decision']='resource_blocked';continue
                        # This is a new resource segment of an existing six-hour arm.
                        # No failed submission retry loop; ambiguous/error exits require review.
                        submit(campaign,state,group,renew=True)
                complete=(len(statuses)==7 and all(s and s.get('state')=='finished'
                          and s.get('active_elapsed_seconds',0)>=21600 for s in statuses.values()))
                active=active_jobs();reserved=validate_pool(state,active)
                allocated=sum(CAPS[g] for g,j in state['current_jobs'].items()
                              if j in active and active[j]['state'] in {'RUNNING','COMPLETING','SUSPENDED'})
                running=sum(bool(s and s.get('state') in {'running','starting','pausing'}
                                 and s.get('slurm_job_id') in active) for s in statuses.values())
                state.update(heartbeat_epoch=time.time(),allocated_gpus=allocated,reserved_gpus=reserved,
                    running_workflows=running,idle_allocated_slots=max(0,allocated-running),
                    status='complete' if complete and not reserved else 'monitoring')
                atomic_json(campaign/'controller_state.json',state)
                research=read(campaign/'RESEARCH_STATE.json')
                research.update(status='completed' if complete else 'running',
                    updated_at=datetime.now(timezone.utc).isoformat(),allocated_gpus=allocated,
                    reserved_gpus=reserved,running_workflows=running,idle_allocated_slots=max(0,allocated-running),
                    current_allocations=state['current_jobs'],current_workflows=statuses)
                atomic_json(campaign/'RESEARCH_STATE.json',research)
                if time.time()-last_analysis>=600 or complete:
                    with (campaign/'analysis_monitor.log').open('a') as stream:
                        result=subprocess.run([str(ROOT/'.venv/bin/python'),str(campaign/'analyze.py')],
                                              stdout=stream,stderr=subprocess.STDOUT,timeout=300)
                    last_analysis=time.time();state['final_analysis_complete']=complete and result.returncode==0
                    atomic_json(campaign/'controller_state.json',state)
                if complete and not reserved:
                    event(campaign,'all_workflows_completed');return
                if not reserved and any(g.get('decision','').startswith(('blocked','resource_blocked'))
                                        for g in state.get('groups',{}).values()):
                    state['status']='blocked';atomic_json(campaign/'controller_state.json',state);return
                errors=0
            except (subprocess.SubprocessError,OSError) as exc:
                errors+=1;event(campaign,'observation_error',error=repr(exc),consecutive=errors)
                # Observation errors never authorize a restart or cancellation.
                if errors>=3:raise
            time.sleep(30)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign',type=Path)
    controller(parser.parse_args().campaign)
