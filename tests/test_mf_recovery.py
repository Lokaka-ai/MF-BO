"""Kill an actual candidate process, restore its frozen decision, and rerun it."""
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from test_multifidelity_integration import aq05_cfg
from test_fixed_evaluation import fixed_cfg, setup_server, model_responses, CANDIDATE
from config import save_run
from engine.agent_search import AgentSearch
from engine.executor import Interpreter
from engine.search_node import Journal
from engine.mf_recovery import execute
from engine.competition_checkpoint import checkpoint_once
from tasks.mlebench_lite import mila_multifidelity as campaign
from utils import serialize

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("mode", ["carry_over", "reset"])
@pytest.mark.parametrize("phase", ["half", "full"])
def test_hard_kill_preserves_frozen_program_budget_and_stage(aq05_cfg, model_responses, monkeypatch, mode, phase, tmp_path):
    monkeypatch.setenv("MLEVOLVE_MF_RECOVERY", "1")
    aq05_cfg.evaluation.multifidelity_mode = mode
    agent = AgentSearch("Evaluation: binary log loss.", aq05_cfg, Journal())
    interpreter = Interpreter(aq05_cfg.workspace_dir, cfg=aq05_cfg, timeout=30)
    drafts = [agent.step(None, interpreter.run, execute_immediately=False) for _ in range(3)]
    for node in drafts:
        agent.execute_deferred_node(node, interpreter.run)
    if phase == "full":
        agent.fixed_evaluation.switch_to_full()
        agent.update_data_preview()
        if mode == "reset":
            agent.phase1_runtime.study7_selector.full_training_only = True
    # The same code pauses only in the doomed allocation, before doing real ML.
    model_responses.code = ("import os, time\nfrom pathlib import Path\n"
        "if os.environ.get('MF_TEST_PAUSE'):\n"
        "    Path(os.environ['MF_TEST_PAUSE']).write_text('training started')\n"
        "    time.sleep(300)\n" + CANDIDATE)
    pending = agent.step(drafts[0], interpreter.run, execute_immediately=False)
    assert pending.pending_execution
    decision_id = pending._phase1_decision_context.decision.decision_id
    original_code = pending.code
    runtime = agent.phase1_runtime
    before = {key: runtime.root_draft_state[key] for key in ("submitted_count", "decision_count", "initial_submitted")}
    agent.capture_resume_state()
    save_run(aq05_cfg, agent.journal)
    cfg_file = tmp_path / "restart.yaml"
    OmegaConf.save(aq05_cfg, cfg_file)
    output = tmp_path / "campaign-output"
    campaign.save(output / "run-info.json", {"log_dir": str(aq05_cfg.log_dir)})
    marker = tmp_path / "started"
    script = '''
import sys, os
from omegaconf import OmegaConf
from config import save_run
from engine.agent_search import AgentSearch
from engine.search_node import Journal
from engine.executor import Interpreter
from engine.parent_selectors.runtime import Phase1SelectorRuntime
from engine.competition_checkpoint import checkpoint_once
from engine.mf_recovery import execute
from utils import serialize
from pathlib import Path
cfg = OmegaConf.load(sys.argv[1])
# The test's previous parent object is quiescent; production uses Slurm receipts.
Phase1SelectorRuntime._owner_alive = staticmethod(lambda state: False)
journal = serialize.load_json(Path(cfg.log_dir) / 'journal.json', Journal)
agent = AgentSearch('ignored', cfg, journal)
node, = agent.phase1_runtime.recovered_deferred_nodes
interpreter = Interpreter(cfg.workspace_dir, cfg=cfg, timeout=300)
def persist():
    agent.capture_resume_state()
    save_run(cfg, agent.journal)
def callback(code, node_id, *args, **kwargs):
    checkpoint_once(agent, persist)
    return execute(interpreter, cfg, code, node_id, *args, **kwargs)
agent.execute_deferred_node(node, callback)
'''
    started = time.time()
    campaign.save(output / "segment.json", {"job_id": "123", "started_epoch": started, "active_offset": 0})
    with (tmp_path / "killed-child.log").open("w") as log:
        child = subprocess.Popen([sys.executable, "-c", script, str(cfg_file)], cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT), "MF_TEST_PAUSE": str(marker)},
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + 90
            while not marker.exists() and child.poll() is None and time.monotonic() < deadline:
                time.sleep(.05)
            assert marker.exists(), (tmp_path / "killed-child.log").read_text()
        finally:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=10)
    release = {"job_id": "123", "state": "PREEMPTED", "ended_epoch": time.time()}
    recovery = campaign.recovery_record(output, release)
    campaign.reconcile_recovery(output, recovery)
    execution_file = aq05_cfg.log_dir / "mf_execution" / f"{pending.id}.json"
    interrupted = campaign.read(execution_file)
    assert interrupted["state"] == "interrupted" and interrupted["elapsed_seconds"] > 0
    aq05_cfg.resume_elapsed_seconds = recovery["elapsed_seconds"]
    restored = AgentSearch("ignored", aq05_cfg, serialize.load_json(aq05_cfg.log_dir / "journal.json", Journal))
    assert time.time() - restored.start_time >= recovery["elapsed_seconds"]
    assert restored.fixed_evaluation.multifidelity_phase == phase
    current = restored.phase1_runtime
    assert {key: current.root_draft_state[key] for key in before} == before
    resumed, = current.recovered_deferred_nodes
    assert resumed.code == original_code and resumed.id == pending.id
    assert resumed._phase1_decision_context.decision.decision_id == decision_id
    assert len(current.transition_store.records()) == 3  # Kill is not BO feedback.
    calls = []
    new_interpreter = Interpreter(aq05_cfg.workspace_dir, cfg=aq05_cfg, timeout=30)
    original_run = new_interpreter.run
    def tracked(*args, **kwargs):
        calls.append(kwargs["execution_timeout"])
        return original_run(*args, **kwargs)
    new_interpreter.run = tracked
    result = restored.execute_deferred_node(resumed,
        lambda code, node_id, *args, **kw: execute(new_interpreter, aq05_cfg, code, node_id, *args, **kw))
    assert result.is_valid, result.analysis
    assert result.training_fraction == (.5 if phase == "half" else 1.)
    assert calls[0] <= aq05_cfg.exec.timeout - interrupted["elapsed_seconds"]
    assert len(current.transition_store.records()) == 4
    assert current.transition_store.by_decision_id(decision_id).child_node_id == pending.id
    assert current.study7_selector.full_training_only == (mode == "reset" and phase == "full")


def test_scheduler_release_rejects_live_job(monkeypatch):
    monkeypatch.setenv("USER", "tester")
    monkeypatch.setattr(campaign, "run", lambda *a, **k: SimpleNamespace(stdout="123\n"))
    with pytest.raises(RuntimeError, match="still queued"):
        campaign.released_job("123")


def test_recovery_excludes_queue_downtime_and_refuses_failure(tmp_path):
    logs = tmp_path / "run/logs"
    campaign.save(tmp_path / "run-info.json", {"log_dir": str(logs)})
    campaign.save(tmp_path / "segment.json", {"job_id": "123", "started_epoch": 1000, "ended_epoch": 1100, "active_offset": 500})
    campaign.save(logs / "journal.json", {"elapsed_active_seconds": 590, "evaluation_state": {}})
    record = campaign.recovery_record(tmp_path, {"job_id": "123", "state": "PREEMPTED", "ended_epoch": 1105})
    assert record["elapsed_seconds"] == 600
    campaign.save(tmp_path / "result.json", {"state": "failed"})
    with pytest.raises(RuntimeError, match="investigation"):
        campaign.recovery_record(tmp_path, {"job_id": "123", "state": "FAILED", "ended_epoch": 1105})


def test_signal_saves_checkpoint_and_supervisor_reports_interruption(tmp_path, monkeypatch):
    import threading
    monkeypatch.setenv("SLURM_JOB_ID", "456")
    script = '''
import sys, time, threading
from pathlib import Path
from types import SimpleNamespace as NS
from engine.mf_recovery import install
root = Path(sys.argv[1])
agent = NS(journal_lock=threading.RLock(), start_time=time.time()-100,
           cfg=NS(log_dir=root), phase1_runtime=NS(
           transition_store=NS(records=lambda: []), root_draft_state={'pending': {}}))
install(agent, lambda: (root / 'saved').write_text('checkpoint'))
(root / 'ready').write_text('ready')
time.sleep(300)
'''
    def interrupt():
        deadline = time.monotonic() + 30
        while not (tmp_path / "ready").exists() and time.monotonic() < deadline:
            time.sleep(.05)
        os.kill(os.getpid(), signal.SIGUSR1)
    sender = threading.Thread(target=interrupt, daemon=True)
    sender.start()
    with pytest.raises(SystemExit) as error:
        campaign.run_search([sys.executable, "-c", script, str(tmp_path)],
                            {**os.environ, "PYTHONPATH": str(ROOT)}, tmp_path)
    sender.join(timeout=5)
    assert error.value.code == 75
    assert campaign.read(tmp_path / "result.json")["state"] == "interrupted"
    heartbeat = campaign.read(tmp_path / "competition_heartbeat.json")
    assert heartbeat["paused"] and heartbeat["active_elapsed_seconds"] >= 100


@pytest.mark.parametrize("submitted", [False, True])
def test_interrupted_generation_reuses_decision_without_double_charge(aq05_cfg, monkeypatch, submitted):
    monkeypatch.setenv("MLEVOLVE_MF_RECOVERY", "1")
    aq05_cfg.evaluation.multifidelity_mode = "reset"
    agent = AgentSearch("Evaluation: binary log loss.", aq05_cfg, Journal())
    runtime = agent.phase1_runtime
    context = runtime.select_global_parent(agent)
    if submitted:
        runtime.submit_decision(agent, context)
    agent.capture_resume_state()
    save_run(aq05_cfg, agent.journal)
    ledger = aq05_cfg.log_dir / "phase1/root_draft_state.json"
    saved = campaign.read(ledger)
    saved["executor_release_receipt"] = {"owner_host": saved["owner_host"], "owner_pid": saved["owner_pid"],
        "verified_by": "slurm_allocation_release", "job_id": "test"}
    campaign.save(ledger, saved)
    restored = AgentSearch("ignored", aq05_cfg, serialize.load_json(aq05_cfg.log_dir / "journal.json", Journal))
    replay, = restored.phase1_runtime.recovered_generation_contexts
    assert replay.decision.decision_id == context.decision.decision_id
    execute_candidate = Interpreter(aq05_cfg.workspace_dir, cfg=aq05_cfg, timeout=30).run
    node = restored.step(None, execute_candidate, execute_immediately=False, recovery_context=replay)
    assert node.pending_execution
    after = restored.phase1_runtime.root_draft_state
    assert after["initial_submitted"] == after["submitted_count"] == after["decision_count"] == 1


def test_resume_command_preserves_budget_and_is_idempotent(tmp_path, monkeypatch):
    settings = {"spec": campaign.read(campaign.SPEC)}
    task, mode = next(iter(settings["spec"]["tasks"])), "reset"
    output = tmp_path / "tasks" / task / mode
    campaign.save(output / "segment.json", {"job_id": "123"})
    monkeypatch.setattr(campaign, "ready", lambda _: settings)
    monkeypatch.setattr(campaign, "released_job", lambda _: {"job_id": "123"})
    record = {"source_job": "123", "run_dir": str(output / "runs/run"), "elapsed_seconds": 1000}
    monkeypatch.setattr(campaign, "recovery_record", lambda *a: record)
    calls = []
    monkeypatch.setattr(campaign, "run", lambda args, **kw: calls.append(args) or SimpleNamespace(stdout="456\n"))
    args = SimpleNamespace(campaign=tmp_path, task=task, mode=mode, partition="long")
    campaign.resume(args)
    campaign.resume(args)
    assert len(calls) == 1
    assert campaign.read(output / "recovery.json")["elapsed_seconds"] == 1000
    assert "--resume-run" in calls[0][-1]
