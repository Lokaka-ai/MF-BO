"""Fill only authorized Acq7 GPU slots using the inherited segment supervisor."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from config.acq7 import ACQ7_ARMS, ACQ7_PRESETS
from engine.shared_drafts import atomic_json
from tasks.autoresearch.acq7_campaign import CAPS, read
from tasks.autoresearch.branch7_resources import active_jobs, terminal
from tasks.autoresearch.study_worker import allocation_info, visible_devices


def eligible_methods(campaign, claims, job, group, live, terminal_lookup=terminal):
    result=[]
    for method in ACQ7_ARMS:
        path=campaign/method/'status.json'
        status=read(path) if path.exists() else None
        if status and status['state']=='finished':
            if abs(float(status.get('active_elapsed_seconds',0))-21600)>1e-6:
                raise RuntimeError('A partial Acq7 arm cannot be called finished')
            continue
        if status and (status['state'].startswith('blocked') or status['state'] in {'failed','early_stopped'}):
            continue
        previous=claims.get(method)
        if previous:
            owner=previous['job_id']
            if owner==job:
                if previous['group']!=group or not status or status['state']!='paused':
                    continue
            elif owner in live or terminal_lookup(owner) is None:
                continue
        result.append(method)
    return result


def claim(campaign,job,group,gpu):
    with (campaign/'queue.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=campaign/'assignments.json'
        claims=read(path) if path.exists() else {}
        eligible=eligible_methods(campaign,claims,job,group,active_jobs())
        if not eligible:return None
        method=eligible[0]
        old=claims.get(method)
        claims[method]={'job_id':job,'group':group,'gpu_uuid':gpu,'at':datetime.now(timezone.utc).isoformat()}
        atomic_json(path,claims)
        with (campaign/'queue_events.jsonl').open('a') as stream:
            stream.write(json.dumps({'method':method,'previous':old,'assignment':claims[method]})+'\n')
        return method


def run(campaign,group):
    campaign=campaign.resolve();manifest=read(campaign/'campaign.json')
    if (manifest.get('version')!='acq7_v1' or set(manifest['methods'])!=set(ACQ7_ARMS)
            or set(manifest['method_directories'])!=set(ACQ7_PRESETS)
            or Path(manifest['source_snapshot']).resolve()!=ROOT):
        raise RuntimeError('Worker requires the immutable registered seven-arm Acq7 source')
    job=os.environ['SLURM_JOB_ID'];allocation=allocation_info(job)
    if group=='unkillable':
        if job!=manifest['existing_allocations']['unkillable']:
            raise RuntimeError('Existing allocation is not the one authorized by the user')
    else:
        from tasks.autoresearch.acq7_resources import authorization
        authorization(campaign)
        if allocation['JobName']!=f'bo-acq7-{group}':
            raise RuntimeError('Batch allocation does not belong to this Acq7 group')
    devices=visible_devices();cpus=sorted(os.sched_getaffinity(0))
    if len(devices)!=CAPS[group] or len(cpus)<4*len(devices):
        raise RuntimeError('Every Acq7 workflow needs one allocated A100 and four allocated CPUs')
    end=datetime.fromisoformat(allocation['EndTime']).timestamp()
    end=min(end,float(os.environ.get('ACQ7_LEASE_END_EPOCH',end)))
    if end-time.time()<900:raise RuntimeError('Insufficient safe allocation window')
    with (campaign/f'{group}.lock').open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        children={};stopping=[];failed=[];sequence=0
        env={**os.environ,**manifest['runtime_environment']}
        def stop(signum,frame):
            stopping.append(signum)
            for item in children.values():
                if item['process'].poll() is None:item['process'].send_signal(signal.SIGTERM)
        for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
        try:
            while True:
                for slot,item in list(children.items()):
                    if item['process'].poll() is not None:
                        status_path=campaign/item['method']/'status.json'
                        if status_path.exists():
                            status=read(status_path)
                        else:
                            status={'state':'blocked/startup','method_id':item['method'],
                                'slurm_job_id':job,'duration_seconds':21600,'active_elapsed_seconds':0,
                                'reason':'Supervisor exited without a workflow status',
                                'exit_code':item['process'].returncode}
                            atomic_json(status_path,status)
                        if item['process'].returncode or status['state'].startswith('blocked') or status['state']=='failed':
                            failed.append(item['method'])
                        if status['state']=='paused':stopping.append('allocation_pause')
                        del children[slot]
                if not stopping and time.time()<end-900:
                    for slot,device in enumerate(devices):
                        if slot in children:continue
                        try:
                            method=claim(campaign,job,group,device['uuid'])
                        except (subprocess.SubprocessError,OSError) as exc:
                            with (campaign/'worker_observation_errors.jsonl').open('a') as stream:
                                stream.write(json.dumps({'at':time.time(),'group':group,
                                    'operation':'claim_scheduler_observation','error_type':type(exc).__name__,
                                    'running_children_untouched':True})+'\n')
                            break
                        if method is None:continue
                        preset=manifest['methods'][method]['preset'];sequence+=1
                        segment_id=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')+f'-{job}-{group}-{sequence}'
                        segment=campaign/'segments'/segment_id;segment.mkdir(parents=True)
                        output=campaign/method;output.mkdir(exist_ok=True)
                        resource={'directory':str(output),'allocation':job,'node':socket.gethostname().split('.')[0],
                            'gpu':device['uuid'],'gpu_model':device['model'],
                            'cpu_ids':','.join(map(str,cpus[4*slot:4*slot+4]))}
                        data={'segment_id':segment_id,'allocation':job,'group':group,'resources':{preset:resource},
                            'allocation_end_epoch':end,'config':str(campaign/'paired_config.yaml'),
                            'created_epoch':time.time(),'allocation_info':allocation,'budget_policy':'active_segments',
                            'duration_seconds':21600}
                        atomic_json(segment/'segment.json',data)
                        command=[sys.executable,str(ROOT/'tasks/autoresearch/segment_supervisor.py'),str(campaign),segment_id,preset]
                        with (segment/f'{preset}.supervisor.log').open('x') as log:
                            child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                                stdout=log,stderr=subprocess.STDOUT)
                        children[slot]={'method':method,'process':child,'segment_id':segment_id}
                        atomic_json(segment/f'{preset}.launcher.json',{'pid':child.pid,'command':command})
                running=[{'slot':slot,'method':item['method'],'pid':item['process'].pid,
                          'segment_id':item['segment_id'],'gpu_uuid':devices[slot]['uuid']} for slot,item in children.items()]
                atomic_json(campaign/f'worker_{group}.json',{'state':'running' if children else 'stopped',
                    'job_id':job,'step_id':os.environ.get('SLURM_STEP_ID'),'pid':os.getpid(),
                    'node':socket.gethostname(),'heartbeat_epoch':time.time(),'allocation_end_epoch':end,
                    'running':running,'failed_methods':failed,'retains_parent_allocation':group=='unkillable'})
                if not children:break
                try:
                    sample=subprocess.run(['nvidia-smi','--query-gpu=uuid,utilization.gpu,memory.used,power.draw',
                        '--format=csv,noheader,nounits'],text=True,capture_output=True,timeout=15)
                    sample_data={'devices':[line for line in sample.stdout.splitlines() if any(d['uuid'] in line for d in devices)],
                                 'returncode':sample.returncode}
                except (subprocess.SubprocessError,OSError) as exc:
                    sample_data={'observation_error':type(exc).__name__,'running_children_untouched':True}
                with (campaign/'gpu_samples.jsonl').open('a') as stream:
                    stream.write(json.dumps({'epoch':time.time(),'job_id':job,'group':group,
                        **sample_data})+'\n')
                time.sleep(10)
        finally:
            for item in children.values():
                if item['process'].poll() is None:item['process'].send_signal(signal.SIGTERM)
            for item in children.values():item['process'].wait(timeout=60)
        return int(bool(failed))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign',type=Path);parser.add_argument('group',choices=tuple(CAPS))
    args=parser.parse_args();raise SystemExit(run(args.campaign,args.group))
