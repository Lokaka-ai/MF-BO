"""Audit budget accounting and release evidence at actual allocation boundaries."""
import json
from pathlib import Path

import pytest

from tasks.mlebench_lite.aq05_campaign import admit_recovery, atomic, allocation_record, read


def fixture_run(tmp_path, state, stopped, elapsed):
    task = 'fixture'
    root = tmp_path / 'run'
    logs = root / 'logs'
    atomic(logs / 'journal.json', {'elapsed_active_seconds': 95})
    atomic(logs / 'competition_heartbeat.json', {'agent_start_epoch': 1000, 'active_elapsed_seconds': 95})
    atomic(logs / 'phase1/root_draft_state.json', {'owner_pid': 123, 'owner_host': 'old-node',
                                               'pending': {'decision': {'status': 'executing'}}})
    atomic(logs / 'candidate_execution/node.json', {'state': 'running', 'started_epoch': 1080,
                                                  'elapsed_seconds': 20})
    atomic(tmp_path / 'tasks' / task / 'run-info.json', {'log_dir': str(logs)})
    atomic(tmp_path / 'tasks' / task / 'status.json', {
        'state': state, 'search_stopped_epoch': stopped, 'active_elapsed_seconds': elapsed})
    return task, logs


def test_queue_wait_does_not_consume_paused_agent_or_candidate_budget(tmp_path):
    task, logs = fixture_run(tmp_path, 'paused', 1105, 103)
    release = {'job_id': '73', 'state': 'COMPLETED', 'end_epoch': 2000}
    admit_recovery(tmp_path, task, release)
    assert read(tmp_path / 'tasks' / task / 'status.json')['active_elapsed_seconds'] == 103
    assert read(logs / 'candidate_execution/node.json')['elapsed_seconds'] == 45
    ledger = read(logs / 'phase1/root_draft_state.json')
    assert ledger['executor_release_receipt'] == {
        'owner_host': 'old-node', 'owner_pid': 123, 'verified_by': 'slurm_allocation_release', **release}
    assert ledger['pending']['decision']['status'] == 'executing'


def test_abrupt_death_counts_tail_after_heartbeat_until_slurm_end(tmp_path):
    task, logs = fixture_run(tmp_path, 'running', None, 90)
    admit_recovery(tmp_path, task, {'job_id': '73', 'state': 'PREEMPTED', 'end_epoch': 1120})
    assert read(tmp_path / 'tasks' / task / 'status.json')['active_elapsed_seconds'] == 120
    assert read(logs / 'candidate_execution/node.json')['elapsed_seconds'] == 60


def test_finished_search_is_not_restarted_after_finalization_preemption(tmp_path):
    task, logs = fixture_run(tmp_path, 'search_complete', 1100, 100)
    admit_recovery(tmp_path, task, {'job_id': '73', 'state': 'PREEMPTED', 'end_epoch': 1200})
    assert read(tmp_path / 'tasks' / task / 'status.json')['state'] == 'search_complete'


def test_release_is_not_inferred_from_stale_accounting_while_job_is_live(monkeypatch):
    calls = []
    def command(args):
        calls.append(args[0])
        return '73|COMPLETING'
    monkeypatch.setattr('tasks.mlebench_lite.aq05_campaign.command', command)
    assert allocation_record('73') is None
    assert calls == ['squeue']


def test_missing_accounting_is_wait_not_permission_to_restart(monkeypatch):
    monkeypatch.setattr('tasks.mlebench_lite.aq05_campaign.command', lambda args: '')
    assert allocation_record('73') is None
