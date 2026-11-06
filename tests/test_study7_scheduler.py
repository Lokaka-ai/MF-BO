"""Use the actual two-worker FIFO, checkpoint restore and run.py entrypoint."""
import json
import shutil
import time

import pytest

from config.study7 import SEVEN_PRESETS
from engine.agent_search import AgentSearch
from tasks.autoresearch.adapter import AutoresearchInterpreter
from test_study7_runtime import factory
from test_study_segmented import test_interruption_preserves_shared_drafts_queue_and_native_results as interrupted_fifo


@pytest.mark.usefixtures("autoresearch_cache")
@pytest.mark.parametrize("setting", SEVEN_PRESETS)
def test_t37_t40_two_worker_atomic_initial_recovery(setting, tmp_path, monkeypatch):
    interrupted_fifo(setting, 2, tmp_path, monkeypatch)


@pytest.mark.parametrize("setting", [SEVEN_PRESETS[1], SEVEN_PRESETS[4], SEVEN_PRESETS[6]])
def test_t34_actual_run_py_entrypoint(setting, factory, tmp_path, monkeypatch):
    import run as runner
    build, calls = factory
    initial, execute = build(setting)
    cfg = initial.cfg
    shutil.rmtree(cfg.log_dir)
    shutil.rmtree(cfg.workspace_dir)
    cfg.agent.steps = 5
    cfg.log_dir.mkdir(parents=True)
    seen = []
    def create(*args, **kwargs):
        agent = AgentSearch(*args, **kwargs)
        seen.append(agent)
        return agent
    monkeypatch.setattr(runner, "load_cfg", lambda: cfg)
    monkeypatch.setattr(runner, "Agent", create)
    monkeypatch.setattr(AutoresearchInterpreter, "_run_trial", lambda self, code, node_id, slot: execute(code, node_id))
    monkeypatch.setenv("MLEVOLVE_SEGMENTED_RUN", "1")
    monkeypatch.setenv("MLEVOLVE_SEGMENT_CHECKPOINT", str(tmp_path / "checkpoints"))
    monkeypatch.setenv("MLEVOLVE_SEGMENT_PAUSE_FILE", str(tmp_path / "pause"))
    monkeypatch.setenv("MLEVOLVE_SEGMENT_STARTED", str(time.time()))
    monkeypatch.setenv("MLEVOLVE_SEGMENT_ACTIVE_OFFSET", "100")
    runner.run()
    assert len(seen) == 1
    agent = seen[0]
    assert len(agent.phase1_runtime.transition_store.records()) == 5
    assert agent.scfg.parallel_search_num == 2
    outcome = json.loads((cfg.log_dir / "segment_outcome.json").read_text())
    assert outcome["state"] == "finished" and outcome["completed"] == 5
    assert agent.journal.elapsed_active_seconds >= 100
    assert agent.phase1_runtime.root_draft_state["pending"] == {}
    if setting != SEVEN_PRESETS[6]:
        assert agent.phase1_runtime.surrogate.fit_diagnostics["N"] == 2
    else:
        assert agent.phase1_runtime.surrogate is None


def test_t39_worker_rejects_foreign_allocation_and_occupied_gpu(monkeypatch):
    import os
    from tasks.autoresearch import study_worker
    monkeypatch.setattr(study_worker.subprocess, "check_output", lambda *a, **kw: "JobState=RUNNING UserId=other(999999)\n")
    with pytest.raises(RuntimeError, match="owned"):
        study_worker.allocation_info("42")
    def query(cmd, **kw):
        if "--query-gpu=index,uuid,name,memory.used" in cmd:
            return "0, GPU-a, NVIDIA A100-SXM4-80GB, 1000\n"
        return "1234\n"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(study_worker.subprocess, "check_output", query)
    with pytest.raises(RuntimeError, match="occupied"):
        study_worker.visible_devices()
