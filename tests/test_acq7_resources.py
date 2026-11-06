"""Acq7 queue, ownership and cumulative-budget guards without Slurm mutation."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tasks.autoresearch.acq7_campaign import CAPS,requests
from tasks.autoresearch import acq7_resources as resources
from tasks.autoresearch.acq7_worker import eligible_methods


def status(root,method,state,active=0):
    path=root/method;path.mkdir(exist_ok=True)
    (path/'status.json').write_text(json.dumps({'state':state,'active_elapsed_seconds':active}))


def test_queue_order_never_restarts_historical_or_live_or_blocked_work(tmp_path):
    status(tmp_path,'BR01','paused',100)
    status(tmp_path,'AQ01','finished',21600)
    status(tmp_path,'AQ02','paused',100.125)
    status(tmp_path,'AQ03','blocked/numerical_failure',900)
    status(tmp_path,'AQ04','failed',1000)
    claims={'AQ02':{'job_id':'old','group':'short'}}
    live={'old':{'state':'RUNNING'}}
    assert eligible_methods(tmp_path,claims,'new','main',live,lambda job:None)==['AQ05','AQ06','AQ07']
    assert eligible_methods(tmp_path,claims,'new','main',{},lambda job:{'state':'COMPLETED'})==['AQ02','AQ05','AQ06','AQ07']
    assert json.loads((tmp_path/'AQ02/status.json').read_text())['active_elapsed_seconds']==100.125
    assert 'BR01' not in eligible_methods(tmp_path,{},'new','main',{},lambda job:None)


def test_missing_scheduler_observation_never_proves_release(tmp_path):
    claims={'AQ01':{'job_id':'old','group':'short'}}
    assert 'AQ01' not in eligible_methods(tmp_path,claims,'new','short',{},lambda job:None)
    status(tmp_path,'AQ01','paused',300)
    assert 'AQ01' in eligible_methods(tmp_path,claims,'old','short',{'old':{}},lambda job:None)
    assert 'AQ01' not in eligible_methods(tmp_path,claims,'old','main',{'old':{}},lambda job:None)


def test_partial_run_cannot_be_completed(tmp_path):
    status(tmp_path,'AQ01','finished',21599)
    with pytest.raises(RuntimeError):eligible_methods(tmp_path,{},'x','main',{},lambda job:None)
    values={f'AQ{i:02}':{'state':'finished','active_elapsed_seconds':21600} for i in range(1,8)}
    assert resources.completed(values)
    values['AQ01']['active_elapsed_seconds']=21599
    assert not resources.completed(values)


def test_pool_quota_exact_names_and_user_parent_retained(monkeypatch):
    live={'10857987':{'name':'mila-gpu'},'1':{'name':'bo-acq7-main'},'2':{'name':'bo-acq7-short'}}
    state={'current_jobs':{'unkillable':'10857987','main':'1','short':'2'}}
    assert resources.validate_pool(state,live)==7
    with pytest.raises(RuntimeError):resources.validate_pool(state,{**live,'3':{'name':'bo-acq7-short'}})
    with pytest.raises(RuntimeError):resources.validate_pool(state,{**live,'1':{'name':'foreign'}})
    monkeypatch.setattr(resources,'terminal',lambda job:None)
    assert resources.validate_pool(state,{'10857987':live['10857987']})==7


def test_authorization_is_current_campaign_specific_and_renewal_specific(tmp_path):
    (tmp_path/'protocol.json').write_text(json.dumps({'spec_sha256':'spec'}))
    auth={'campaign':str(tmp_path.resolve()),'spec_sha256':'spec','pool_gpu_limits':CAPS,
          'user_instruction':'Current Acq7 authorization','allow_new_allocations':True,'allow_renewals':False}
    path=tmp_path/'resource_authorization.json';path.write_text(json.dumps(auth))
    resources.authorization(tmp_path)
    with pytest.raises(RuntimeError):resources.authorization(tmp_path,renew=True)
    auth['campaign']='historical';path.write_text(json.dumps(auth))
    with pytest.raises(RuntimeError):resources.authorization(tmp_path)


def test_no_submission_when_previous_job_release_is_uncertain(tmp_path,monkeypatch):
    monkeypatch.setattr(resources,'authorization',lambda *a,**k:None)
    monkeypatch.setattr(resources,'active_jobs',lambda:{})
    monkeypatch.setattr(resources,'terminal',lambda job:None)
    monkeypatch.setattr(resources.subprocess,'run',lambda *a,**k:pytest.fail('Unexpected sbatch'))
    with pytest.raises(RuntimeError):resources.submit(tmp_path,{'current_jobs':{'main':'1','unkillable':'10857987'}},'main',True)


def test_existing_live_step_is_not_restarted(tmp_path,monkeypatch):
    monkeypatch.setattr(resources,'local_handle_live',lambda handle:True)
    monkeypatch.setattr(resources.subprocess,'Popen',lambda *a,**k:pytest.fail('Duplicate srun'))
    resources.adopt(tmp_path,{'adopted_step':{'pid':1,'start_ticks':'1'}})


def test_requests_use_only_six_new_cards_plus_existing_one_and_exact_worker_entrypoint(tmp_path):
    items=requests(tmp_path)
    assert set(items)=={'main','short'}
    assert sum(item['gpus'] for item in items.values())==6
    assert all(item['cpus']==4*item['gpus'] for item in items.values())
    assert '--gres=gpu:a100l:2' in items['main']['command']
    assert '--gres=gpu:a100l:4' in items['short']['command']
    assert items['short']['partition']=='short-unkillable'
