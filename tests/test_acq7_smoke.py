"""Execute the actual smoke driver with only external provider/training mocked."""
import json
import shutil
import sys
import threading

import pytest

from config.acq7 import ACQ7_PRESETS
from tasks.autoresearch import acq7_smoke
from test_study7_runtime import factory


@pytest.mark.parametrize('method',['AQ06','AQ07'])
def test_isolated_smoke_driver_closes_real_managed_model_and_restore(method,factory,tmp_path,monkeypatch):
    import config
    import tasks.autoresearch.adapter as adapter
    build,_=factory;agent,execute=build(ACQ7_PRESETS[int(method[2:])-1])
    cfg=agent.cfg;cfg.agent.initial_drafts=2;cfg.agent.steps=5;cfg.agent.time_limit=3600
    pool=cfg.shared_draft_pool;cfg.shared_draft_pool=None
    campaign=tmp_path/'isolated_smoke';(campaign/'tests').mkdir(parents=True)
    shutil.copytree(__import__('pathlib').Path(pool).parent,campaign/'draft_pool')
    monkeypatch.setattr(config,'load_cfg',lambda:cfg)
    monkeypatch.setattr(config,'load_task_desc',lambda config:agent.task_desc)
    monkeypatch.setattr(config,'prep_agent_workspace',lambda config:None)
    monkeypatch.setenv('SLURM_JOB_ID','10857987')
    monkeypatch.setenv('SLURM_STEP_ID','unit-test-only')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','GPU-unit-test-only')
    monkeypatch.setattr(sys,'argv',list(sys.argv))
    barrier=threading.Barrier(2);calls=[];lock=threading.Lock()
    class FakeInterpreter:
        def __init__(self,*args,**kwargs):pass
        def run(self,*args,**kwargs):
            with lock:
                index=len(calls);calls.append(index)
            if index<2:barrier.wait(timeout=10)
            return execute(*args,**kwargs)
        def terminate_all_subprocesses(self):pass
    monkeypatch.setattr(adapter,'AutoresearchInterpreter',FakeInterpreter)
    acq7_smoke.probe(campaign,method)
    result=json.loads((campaign/'tests'/f'smoke_{method}.json').read_text())
    assert result['passed'] and result['normal_labels']==1 and result['save_load_posterior_and_rng_equal']
    assert result['formal_feedback_exported'] is False
    assert not (campaign/method).exists() and len(calls)==3
    assert result['initial_ratings']==([2,2] if method=='AQ07' else [None,None])
