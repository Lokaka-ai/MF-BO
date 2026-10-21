from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import threading

from omegaconf import OmegaConf
import pytest

from config import _load_cfg
from engine.agent_search import AgentSearch
from engine.executor import ExecutionResult
from engine.search_node import Journal
from tasks.autoresearch.adapter import AutoresearchInterpreter, REPO, assemble_candidate, candidate_body
from test_autoresearch import result_for


@pytest.mark.usefixtures("autoresearch_cache")
@pytest.mark.parametrize("setting", ["gp7_ancestry_ucb", "gp7_long_horizon_ucb", "mlevolve_original", "gp7_long_horizon_ucb_pibo"])
def test_native_stepwise_review_feedback_memory_and_transition(setting, tmp_path, monkeypatch):
    root = REPO.parents[2]
    cfg = OmegaConf.merge(_load_cfg(use_cli_args=False),
        OmegaConf.load(root / "tasks/autoresearch/config.yaml"),
        OmegaConf.load(root / f"config/experiments/{setting}.yaml"))
    cfg.workspace_dir = tmp_path / "workspace"
    cfg.log_dir = tmp_path / "logs"
    cfg.workspace_dir.mkdir()
    cfg.cpu_number = 4
    calls, memories = [], []
    body = candidate_body((REPO / "train.py").read_text())

    class Memory:
        def __init__(self, **kwargs):
            pass

        def save_node(self, node, parent):
            memories.append(node.id)

    def generate(**kwargs):
        calls.append("stepwise")
        return "Use a compatible causal GPT candidate.\n```python\n" + body + "\n```"

    def review(**kwargs):
        calls.append("review")
        assert "module" in str(kwargs["system_message"])
        return {"needs_revision": False, "reasoning": "Host API is preserved.", "revised_code": None}

    def feedback(**kwargs):
        calls.append("feedback")
        return {"is_bug": False, "summary": "Completed official BPB evaluation.", "code_summary": "Causal GPT with Muon", "potential_level": 2}

    monkeypatch.setattr("agents.memory.global_memory.GlobalMemoryLayer", Memory)
    monkeypatch.setattr("agents.coder.stepwise_coder.generate", generate)
    monkeypatch.setattr("agents.code_review_agent.query", review)
    monkeypatch.setattr("agents.result_parse_agent.query", feedback)
    agent = AgentSearch((REPO.parent / "TASK.md").read_text(), cfg, Journal())
    node = agent.step(None, lambda *a, **k: pytest.fail("Must defer"), execute_immediately=False)
    assert node.stage == "draft" and node.pending_execution
    assert calls == ["stepwise"] * 4 + ["review"]
    assert agent.global_memory is not None

    def execute(code, id, reset_session=True, **kwargs):
        trial = cfg.workspace_dir / "trials" / id
        trial.mkdir(parents=True)
        (trial / "result.json").write_text(json.dumps(result_for(assemble_candidate(code))))
        return ExecutionResult(["candidate printed 0.000001; this is not the authoritative metric"], 800, None)

    completed = agent.execute_deferred_node(node, execute)
    assert completed.is_valid and completed.metric.value == 1.7
    if setting == "gp7_long_horizon_ucb_pibo":
        assert completed.agent_potential_level == 2
    assert calls[-1] == "feedback" and memories == [node.id]
    records = agent.phase1_runtime.transition_store.records()
    assert len(records) == 1 and records[0].valid


@pytest.mark.parametrize("workers", [1, 2, 3])
def test_interpreter_respects_parallel_slots_and_releases_failed_slot(workers, tmp_path, monkeypatch):
    cfg = _load_cfg(use_cli_args=False)
    cfg.agent.search.parallel_search_num = workers
    cfg.cpu_number = 4
    interpreter = AutoresearchInterpreter(tmp_path, cfg=cfg)
    barrier = threading.Barrier(workers)
    seen = []

    def trial(code, id, slot):
        seen.append(slot)
        barrier.wait(timeout=5)
        assert interpreter.current_parallel_run == workers
        barrier.wait(timeout=5)
        if slot == workers - 1:
            raise ValueError("failed trial")
        return slot

    monkeypatch.setattr(interpreter, "_run_trial", trial)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(interpreter.run, "pass", str(i)) for i in range(workers)]
        for future in futures:
            try:
                future.result()
            except ValueError:
                pass
    assert sorted(seen) == list(range(workers))
    assert interpreter.current_parallel_run == 0 and interpreter.status_map == [0] * workers
