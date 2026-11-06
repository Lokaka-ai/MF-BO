import json

import pytest
import yaml

from tasks.autoresearch.branch7_campaign import requests,safe_yaml
from tasks.autoresearch.branch7_resources import authorization,group_decision,validate_pool,submit_initial


def test_current_authorization_is_required_before_any_submission(tmp_path,monkeypatch):
    def forbidden(*a,**kw):pytest.fail('Unapproved subprocess invoked')
    monkeypatch.setattr('tasks.autoresearch.branch7_resources.subprocess.run',forbidden)
    with pytest.raises(RuntimeError,match='authorization is missing'):
        submit_initial(tmp_path,smoke_only=True)


def test_old_or_different_campaign_authorization_cannot_be_reused(tmp_path):
    (tmp_path/'protocol.json').write_text(json.dumps({'spec_sha256':'current'}))
    auth={'campaign':'/old/study7','spec_sha256':'current','pool_gpu_limits':{'main':2,'unkillable':1,'short':4},
          'allow_new_allocations':True,'allow_renewals':False,'user_instruction':'current test authorization'}
    p=tmp_path/'resource_authorization.json';p.write_text(json.dumps(auth))
    with pytest.raises(RuntimeError,match='exact Branch7'):authorization(tmp_path)
    auth['campaign']=str(tmp_path);p.write_text(json.dumps(auth))
    assert authorization(tmp_path)
    with pytest.raises(RuntimeError,match='renewal authorization'):authorization(tmp_path,renew=True)


@pytest.mark.parametrize('status,end,expected',[
    ({'state':'finished','active_elapsed_seconds':21600},None,'complete'),
    ({'state':'finished','active_elapsed_seconds':300},{'state':'COMPLETED'},'blocked_incomplete_budget'),
    ({'state':'running'},None,'await_accounting'),
    ({'state':'running'},{'state':'PREEMPTED'},'renew'),
    ({'state':'paused'},{'state':'CANCELLED'},'blocked_allocation'),
    ({'state':'blocked/numerical_failure'},{'state':'COMPLETED'},'blocked_method'),
])
def test_resource_terminal_gates_preserve_budget_and_do_not_restart_on_observation_timeout(status,end,expected):
    assert group_decision([status],end)==expected


def test_requests_exact_pool_and_pending_reservations(tmp_path):
    plan=requests(tmp_path)
    assert [plan[g]['gpus'] for g in ('main','unkillable','short')]==[2,1,4]
    state={'current_jobs':{'main':'1','unkillable':'2','short':'3'}}
    live={str(i):{'state':'PENDING','name':f'bo-branch7-{g}'} for i,g in enumerate(plan,1)}
    live['foreign']={'state':'RUNNING','name':'mila-gpu'}
    assert validate_pool(state,live)==7
    live['4']={'state':'RUNNING','name':'bo-branch7-main'}
    with pytest.raises(RuntimeError,match='Unregistered'):validate_pool(state,live)


def test_historical_yaml_safe_path_conversion_rejects_arbitrary_constructors(tmp_path):
    p=tmp_path/'old.yaml'
    p.write_text('path: !!python/object/apply:pathlib.PosixPath [/, known, logs]\n')
    assert safe_yaml(p)=={'path':'/known/logs'}
    p.write_text('x: !!python/object/apply:os.system ["exit 3"]\n')
    with pytest.raises(yaml.constructor.ConstructorError):safe_yaml(p)
