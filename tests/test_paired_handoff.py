import json
from pathlib import Path
import signal
import socket
from types import SimpleNamespace

import pytest

from engine.parent_selectors.runtime import Phase1SelectorRuntime
from tasks.autoresearch import paired_campaign, migrate_checkpoint
from tasks.autoresearch.supervise import supervise


@pytest.mark.parametrize("legacy", [False, True])
def test_renamed_presets_keep_historical_campaign_directories(legacy):
    settings = {}
    for name in paired_campaign.SETTINGS:
        key = paired_campaign.LEGACY_SETTINGS.get(name, name) if legacy else name
        settings[key] = {"directory": f"/saved/campaign/{key}", "launcher_pid": 42}
    manifest = {"settings": settings}
    before = json.dumps(manifest, sort_keys=True)
    for name in paired_campaign.SETTINGS:
        entry = paired_campaign.setting_entry(manifest, name)
        key = paired_campaign.LEGACY_SETTINGS.get(name, name) if legacy else name
        assert entry["directory"] == f"/saved/campaign/{key}"
    assert json.dumps(manifest, sort_keys=True) == before


def test_cross_host_resume_requires_matching_verified_receipt(monkeypatch):
    monkeypatch.setattr(socket, "gethostname", lambda:"new-node")
    state = {"owner_host":"old-node", "owner_pid":123}
    with pytest.raises(RuntimeError, match="another host"):
        Phase1SelectorRuntime._owner_alive(state)
    state['executor_release_receipt'] = {"owner_host":"old-node", "owner_pid":124,"verified_by":"slurm_allocation_release","job_id":"42"}
    with pytest.raises(RuntimeError): Phase1SelectorRuntime._owner_alive(state)
    state['executor_release_receipt']['owner_pid'] = 123
    assert Phase1SelectorRuntime._owner_alive(state) is False


def test_scheduler_must_prove_release_before_migration(monkeypatch):
    monkeypatch.setattr(migrate_checkpoint.subprocess, 'run', lambda *a,**k:SimpleNamespace(returncode=0,stdout='42|RUNNING|old-node\n'))
    with pytest.raises(RuntimeError, match="not proved"):
        migrate_checkpoint.verify_release('42','old-node','new-node')
    monkeypatch.setattr(migrate_checkpoint.subprocess, 'run', lambda *a,**k:SimpleNamespace(returncode=0,stdout='42|PREEMPTED|old-node\n'))
    assert migrate_checkpoint.verify_release('42','old-node','new-node')['scheduler_state']=='PREEMPTED'


def test_stop_does_not_signal_reused_or_unrelated_pid(tmp_path, monkeypatch):
    (tmp_path/'status.json').write_text(json.dumps({'state':'running','hostname':socket.gethostname(),
                                                  'supervisor_pid':123,'supervisor_start_ticks':'10'}))
    monkeypatch.setattr(paired_campaign,'process_table',lambda:{123:(1,'20','S')})
    monkeypatch.setattr(paired_campaign.os,'kill',lambda *a:pytest.fail('Must not signal reused PID'))
    with pytest.raises(RuntimeError,match='reused'):
        paired_campaign.stop_arm(tmp_path)


def test_remote_stop_uses_specific_existing_allocation(monkeypatch):
    monkeypatch.setattr(socket,'gethostname',lambda:'controller')
    monkeypatch.setattr(paired_campaign.shutil,'which',lambda _:'/test/slurm/bin/srun')
    resource={'node':'compute1','allocation':'42'}
    command=paired_campaign.node_command(resource,['python','stop.py'])
    assert '--jobid=42' in command and '--nodelist=compute1' in command and '--overlap' in command
    assert command[-2:]==['python','stop.py']
    assert command[:5] == ['/usr/bin/env', '-u', 'SLURM_CPUS_PER_TASK', '-u', 'SLURM_TRES_PER_TASK']
    assert '/test/slurm/bin/srun' in command and 'SLURM_MEM_PER_NODE' in command


def test_same_host_different_allocation_still_uses_slurm(monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID','41')
    monkeypatch.setattr(paired_campaign.shutil,'which',lambda _:'/test/slurm/bin/srun')
    command=paired_campaign.node_command({'node':socket.gethostname(),'allocation':'42'},['/bin/true'])
    assert '--jobid=42' in command


@pytest.mark.parametrize('value,expected', [('08:05:00',29100), ('1-02:03:04',93784), ('05:03',303), ('UNLIMITED',float('inf'))])
def test_remaining_budget_parses_slurm_time(monkeypatch, value, expected):
    monkeypatch.setattr(paired_campaign.subprocess, 'run', lambda *a,**k:
        SimpleNamespace(stdout=value+'\n',check_returncode=lambda:None))
    assert paired_campaign.allocation_remaining('42') == expected


def test_preflight_rejects_different_a100_hardware(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda _:SimpleNamespace(name='NVIDIA A100 80GB PCIe',total_memory=80*1024**3))
    with pytest.raises(RuntimeError,match='matching A100-SXM4'):
        paired_campaign.preflight('GPU-test')


def test_resumed_supervisor_keeps_original_deadline(tmp_path):
    import sys,time
    start=time.time()-10
    assert supervise([sys.executable,'-c','import time;time.sleep(10)'],tmp_path,10.1,
                     grace_seconds=.1,start_epoch=start)==0
    data=json.loads((tmp_path/'status.json').read_text())
    assert data['stop_reason']=='time_budget'
    assert data['elapsed_seconds']<13
    assert data['supervisor_start_ticks']


def test_gpu_lease_runs_later_request_without_reprocessing_completion_files(tmp_path, monkeypatch):
    import sys,threading,time
    from tasks.autoresearch import resource_lease
    from engine.shared_drafts import atomic_json
    lease=tmp_path/'lease'; lease.mkdir()
    monkeypatch.setattr(sys,'argv',['resource_lease.py',str(lease)])
    monkeypatch.setenv('SLURM_JOB_ID','42')
    monkeypatch.setattr(resource_lease.subprocess,'check_output',lambda *a,**k:'GPU-test, NVIDIA A100-SXM4-80GB, 81920\n')
    atomic_json(lease/'request_01.json',{'command':[sys.executable,'-c','pass'],
        'output':str(tmp_path/'one'),'duration_seconds':30})
    def add_later():
        time.sleep(2.2)
        program=f'from pathlib import Path; Path({str(lease/"release")!r}).touch()'
        atomic_json(lease/'request_02.json',{'command':[sys.executable,'-c',program],
            'output':str(tmp_path/'two'),'duration_seconds':30})
    producer=threading.Thread(target=add_later);producer.start()
    resource_lease.main();producer.join()
    assert json.loads((lease/'request_01.done.json').read_text())['returncode']==0
    assert json.loads((lease/'request_02.done.json').read_text())['returncode']==0
