import json
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from config import _load_cfg, save_run
from engine.agent_search import AgentSearch
from engine.executor import ExecutionResult
from engine.search_node import Journal
from tasks.autoresearch.adapter import REPO, assemble_candidate, AutoresearchTask, candidate_body
from test_shared_drafts import make_pool
from tasks.autoresearch.segmented import run_segmented, restore_checkpoint
from test_autoresearch import result_for
from utils import serialize


@pytest.mark.usefixtures("autoresearch_cache")
@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("setting", [
    "mlevolve_original", "gp6", "gp6_draft", "gp7_ancestry_ucb",
    "gp7_no_ancestry_ucb", "laplacian_anchor_logei", "gp7_long_horizon_ucb",
    "gp7_short_logei_pibo", "gp7_long_horizon_ucb_pibo", "gp7_short_ucb_pibo",
])
def test_interruption_preserves_shared_drafts_queue_and_native_results(setting, workers, tmp_path, monkeypatch):
    src = REPO.parents[2]
    cfg = OmegaConf.merge(_load_cfg(use_cli_args=False), OmegaConf.load(src / "tasks/autoresearch/config.yaml"),
                         OmegaConf.load(src / f"config/experiments/{setting}.yaml"))
    run_root = tmp_path / "run"
    cfg.log_dir = run_root / "logs"; cfg.workspace_dir = run_root / "workspace"
    cfg.workspace_dir.mkdir(parents=True); cfg.log_dir.mkdir()
    cfg.agent.search.parallel_search_num = workers; cfg.cpu_number = 4
    cfg.agent.steps = 3
    task = AutoresearchTask(cfg)
    cfg.shared_draft_pool = str(make_pool(tmp_path / "pool", task.protocol_id, candidate_body((REPO / "train.py").read_text())))
    check = tmp_path / "checkpoints"
    monkeypatch.setenv("MLEVOLVE_SEGMENT_CHECKPOINT", str(check))
    monkeypatch.setenv("MLEVOLVE_SEGMENT_PAUSE_FILE", str(tmp_path / "pause"))
    monkeypatch.setenv("MLEVOLVE_SEGMENT_STARTED", str(time.time()))
    monkeypatch.setattr("engine.shared_drafts.time.sleep", lambda _: None)
    monkeypatch.setattr("engine.parent_selectors.runtime.Phase1SelectorRuntime._owner_alive", lambda *a: False)

    class Memory:
        def __init__(self, **kwargs): pass
        def save_node(self, node, parent): pass
    class Interpreter:
        def terminate_all_subprocesses(self): pass
    monkeypatch.setattr("agents.memory.global_memory.GlobalMemoryLayer", Memory)
    monkeypatch.setattr("agents.result_parse_agent.query", lambda **kw: {
        "is_bug": False, "summary": "Official evaluation passed.", "code_summary": "Causal GPT", "potential_level": 2})
    task = (src / "tasks/autoresearch/TASK.md").read_text()
    first = AgentSearch(task, cfg, Journal())
    attempts = []
    interrupted_group = threading.Barrier(workers)

    def interrupted(code, id, *a, **kw):
        attempts.append(id)
        (cfg.workspace_dir / f"partial_attempt_{id}.txt").write_text(id)
        interrupted_group.wait(timeout=10)
        raise KeyboardInterrupt

    def persist(agent):
        agent.capture_resume_state(); save_run(cfg, agent.journal)
    with pytest.raises(KeyboardInterrupt):
        run_segmented(first, Interpreter(), cfg, interrupted, lambda: persist(first))
    usage = json.loads((cfg.log_dir / "shared_draft_usage.json").read_text())
    assert [x["slot"] for x in usage["assignments"]] == [1, 2, 3]
    expected_ids = [x["node_id"] for x in usage["assignments"]]
    assert sorted(attempts) == sorted(expected_ids[:workers])
    assert len(first.phase1_runtime.transition_store.records()) == 0
    restored = restore_checkpoint(check)
    assert Path(restored["interrupted_state_archive"]).is_dir()
    for node_id in attempts:
        assert (cfg.workspace_dir / f"partial_attempt_{node_id}.txt").read_text() == node_id
    cfg.resume_run_dir = run_root
    journal = serialize.load_json(cfg.log_dir / "journal.json", Journal)
    second = AgentSearch(task, cfg, journal)
    calls = []
    resumed_group = threading.Barrier(workers)
    def execute(code, id, *a, **kw):
        calls.append(id)
        if id in expected_ids[:workers]:
            # Both slots must be active; a silently serial runner times out here.
            resumed_group.wait(timeout=10)
        trial = cfg.workspace_dir / "trials" / id; trial.mkdir(parents=True)
        (trial / "result.json").write_text(json.dumps(result_for(assemble_candidate(code))))
        return ExecutionResult(["Official evaluation"], 800, None)
    run_segmented(second, Interpreter(), cfg, execute, lambda: persist(second))
    assert sorted(calls) == sorted(expected_ids)
    assert sorted(n.research_direction for n in second.journal.nodes[1:]) == ["draft:1", "draft:2", "draft:3"]
    assert len(second.phase1_runtime.transition_store.records()) == 3
    assert all(r.valid for r in second.phase1_runtime.transition_store.records())
    assert json.loads((cfg.log_dir / "shared_draft_usage.json").read_text()) == usage
    assert len(second.virtual_root.children) == 3
    assert all(n.is_valid for n in second.virtual_root.children)
    assert json.loads((cfg.log_dir / "segment_outcome.json").read_text())["state"] == "finished"


def test_parallel_replenishment_pauses_at_consistent_boundary_and_obeys_trial_cap(tmp_path, monkeypatch):
    run_root = tmp_path / "run"
    logs = run_root / "logs"
    logs.mkdir(parents=True)
    pause = tmp_path / "pause"
    checkpoints = tmp_path / "checkpoints"
    monkeypatch.setenv("MLEVOLVE_SEGMENT_CHECKPOINT", str(checkpoints))
    monkeypatch.setenv("MLEVOLVE_SEGMENT_PAUSE_FILE", str(pause))
    monkeypatch.setenv("MLEVOLVE_SEGMENT_STARTED", str(time.time()))
    monkeypatch.setenv("MLEVOLVE_SEGMENT_ACTIVE_OFFSET", "200")
    cfg = SimpleNamespace(log_dir=logs, resume_run_dir=None,
        agent=SimpleNamespace(steps=7, initial_drafts=0, time_limit=21600,
                              search=SimpleNamespace(parallel_search_num=2)))

    class Journal(list):
        @property
        def nodes(self): return self
        def get_best_node(self): return None

    journal = Journal([SimpleNamespace(id="root")])
    lock = threading.Lock()
    barrier = threading.Barrier(2)
    active = peak = submitted = 0
    committed = []

    def step(**kwargs):
        nonlocal active, peak, submitted
        with lock:
            submitted += 1
            number = submitted
            active += 1
            peak = max(peak, active)
        if number <= 6:
            barrier.wait(timeout=10)
        node = SimpleNamespace(id=f"node-{number}")
        with lock:
            journal.append(node)
            active -= 1
        if number <= 2:
            pause.touch()
        return node

    agent = SimpleNamespace(journal=journal, step=step,
        phase1_runtime=SimpleNamespace(managed_decisions=False),
        autoresearch=SimpleNamespace(protocol_id="test"))
    interpreter = SimpleNamespace(terminate_all_subprocesses=lambda: None)

    def persist():
        assert active == 0, "Do not checkpoint while a worker can mutate state"
        committed.append(len(journal) - 1)
        (logs / "journal.json").write_text(json.dumps([node.id for node in journal]))

    run_segmented(agent, interpreter, cfg, None, persist)
    assert submitted == 2 and peak == 2
    outcome = json.loads((logs / "segment_outcome.json").read_text())
    assert outcome["state"] == "paused" and outcome["completed"] == 2
    assert json.loads((logs / "serial_queue.json").read_text())["parallel_workers"] == 2
    assert agent.start_time == float(os.environ["MLEVOLVE_SEGMENT_STARTED"]) - 200

    restore_checkpoint(checkpoints)
    pause.unlink()
    cfg.resume_run_dir = run_root
    monkeypatch.setenv("MLEVOLVE_SEGMENT_ACTIVE_OFFSET", "350")
    run_segmented(agent, interpreter, cfg, None, persist)
    assert submitted == 7 and peak == 2 and active == 0
    assert len({node.id for node in journal}) == 8
    assert committed == [0, 2, 2, 4, 6, 7]
    assert json.loads((logs / "segment_outcome.json").read_text())["state"] == "finished"
    assert agent.start_time == float(os.environ["MLEVOLVE_SEGMENT_STARTED"]) - 350
