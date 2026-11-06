from pathlib import Path
from types import SimpleNamespace
import json
import os
import sys
import time

import pytest

from tasks.autoresearch import segment_supervisor as campaign


def test_active_budget_excludes_allocation_wait_but_includes_interrupted_segment(monkeypatch):
    history = [{'segment_id':'one', 'started_epoch':100, 'ended_epoch':400, 'active_seconds':300}]
    second = {'segment_id':'two', 'segment_started_epoch':10000, 'slurm_job_id':'42', 'hostname':'old-node'}
    monkeypatch.setattr(campaign.subprocess, 'check_output', lambda *a,**k:'42|PREEMPTED|1970-01-01T00:48:20|\n')
    # Toronto is UTC-5 at this date: the second segment ended at epoch 20900.
    recovered = campaign.recover_unrecorded_segment(history, second)
    assert recovered[-1]['active_seconds'] == 10900
    assert campaign.budget_used(recovered) == 11200
    assert campaign.recover_unrecorded_segment(recovered, second) == recovered


@pytest.mark.parametrize('duration,source_duration,near_limit,saved_workers,override,expected_workers', [
    (28800,28800,False,1,None,1), (21600,21600,False,None,None,2),
    (21600,28800,False,1,2,2), (21600,28800,True,2,None,2),
])
def test_supervisor_records_pause_without_restarting_budget(tmp_path,monkeypatch,duration,source_duration,near_limit,saved_workers,override,expected_workers):
    now=time.time();setting='mlevolve_original';arm=tmp_path/setting;arm.mkdir()
    segment=tmp_path/'segments/new';segment.mkdir(parents=True)
    logs=arm/'runs/existing/logs';logs.mkdir(parents=True)
    if saved_workers is not None:
        (logs/'config.yaml').write_text(f'agent:\n  search:\n    parallel_search_num: {saved_workers}\n')
    overrides = {} if override is None else {'agent.search.parallel_search_num': override}
    cfg=tmp_path/'paired.yaml';cfg.write_text(f'duration_seconds: {source_duration}\nmax_trials: 500\ncpus: 4\nsettings: [mlevolve_original]\noverrides: {json.dumps(overrides)}\n')
    write=lambda path,data:path.write_text(json.dumps(data))
    write(segment/'segment.json',{'duration_seconds':duration,'resources':{setting:{'directory':str(arm),'allocation':'42','node':'here','gpu':'GPU-test','cpu_ids':'0,1,2,3'}},'config':str(cfg),'allocation_end_epoch':now+1000})
    write(arm/'run_location.json',{'log_dir':str(logs),'workspace_dir':str(logs.parent/'workspace')})
    write(arm/'status.json',{'state':'paused','segment_id':'old','slurm_job_id':'41','hostname':'old-host','supervisor_pid':99999999,'segment_started_epoch':now-10000})
    offset=duration-.5 if near_limit else 200
    write(arm/'segment_history.json',[{'segment_id':'old','started_epoch':now-10000,'active_seconds':offset}])
    checkpoints=arm/'checkpoints';checkpoints.mkdir();write(checkpoints/'latest.json',{})
    monkeypatch.setattr('tasks.autoresearch.segmented.restore_checkpoint',lambda *_:{'test':True})
    monkeypatch.setattr('tasks.autoresearch.migrate_checkpoint.release',lambda *_:None)
    monkeypatch.setattr(campaign.signal,'signal',lambda *a:None)
    child=tmp_path/'child.py'
    child.write_text('from pathlib import Path\nimport os,json\n'+f'Path({str(logs/"segment_outcome.json")!r}).write_text(json.dumps({{"state":"paused"}}))\n')
    if near_limit:
        child.write_text('import time\ntime.sleep(30)\n')
    monkeypatch.setattr(campaign,'build_command',lambda *a:[sys.executable,str(child),'--runs-dir',str(arm/'runs')])
    campaign.arm(tmp_path,'new',setting)
    status=json.loads((arm/'status.json').read_text())
    assert status['duration_seconds']==duration
    assert status['source_duration_seconds']==source_duration
    assert status['parallel_search_num']==expected_workers
    if near_limit:
        assert status['state']=='finished' and status['stop_reason']=='time_budget'
        assert status['active_elapsed_seconds']==duration
    else:
        assert status['state']=='paused' and status['exit_code']==0
        assert 200 < status['active_elapsed_seconds'] < 210
    assert status['remaining_active_seconds']==duration-status['active_elapsed_seconds']
    assert len(json.loads((arm/'segment_history.json').read_text()))==2


def test_interrupted_segment_recovery_obeys_six_hour_remainder(monkeypatch):
    history=[{'segment_id':'one','active_seconds':20000}]
    old={'segment_id':'two','segment_started_epoch':10000,'slurm_job_id':'42','hostname':'old'}
    monkeypatch.setattr(campaign.subprocess,'check_output',lambda *a,**k:'42|PREEMPTED|1970-01-01T00:48:20|\n')
    recovered=campaign.recover_unrecorded_segment(history,old,21600)
    assert campaign.budget_used(recovered)==21600
    assert recovered[-1]['active_seconds']==1600
