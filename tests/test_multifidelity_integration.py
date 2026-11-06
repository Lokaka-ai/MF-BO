"""Actual AQ05/fixed-evaluation CPU path; only LLM responses are synthetic."""
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest

from test_fixed_evaluation import fixed_cfg, setup_server, model_responses, CANDIDATE
from engine.agent_search import AgentSearch
from engine.executor import Interpreter
from engine.fixed_evaluation import finalize_submission
from engine.multifidelity import full_node_ids
from engine.search_node import Journal
from utils import serialize


@pytest.fixture
def aq05_cfg(fixed_cfg, model_responses, monkeypatch):
    if not hasattr(os, "sched_getaffinity"):
        # Linux-only diagnostic field; do not alter the executor's OS detection.
        monkeypatch.setattr("engine.parent_selectors.acq7_selector.os",
                            SimpleNamespace(environ=os.environ, sched_getaffinity=lambda _: set(range(os.cpu_count()))))
    preset = Path(__file__).resolve().parents[1] / "config/experiments/acq7_aq05_gp4_level15_rating_kg.yaml"
    cfg = OmegaConf.merge(fixed_cfg, OmegaConf.load(preset))
    cfg.evaluation.enabled = True
    cfg.audited_bo = "redesign"
    cfg.agent.initial_drafts = cfg.agent.search.num_drafts = 3
    cfg.agent.search.parallel_search_num = 3
    cfg.cpu_number = 3
    cfg.agent.time_limit = 10000
    from agents import result_parse_agent
    original = result_parse_agent.query
    def review(**kwargs):
        return {**original(**kwargs), "potential_level": 2}
    monkeypatch.setattr(result_parse_agent, "query", review)
    # Exercise the real ML redesign adapter with a deterministic plan response.
    from agents.redesign_agent import FIELDS
    plan = {key: ([] if key == "evidence_node_ids" else "Fit logistic regression using the current T view.")
            for key in FIELDS}
    monkeypatch.setattr("agents.mle_redesign_agent.generate", lambda **kw: json.dumps(plan))
    return cfg


@pytest.mark.parametrize("mode", ["carry_over", "reset"])
def test_aq05_half_full_refit_and_winner(aq05_cfg, mode, monkeypatch):
    aq05_cfg.evaluation.multifidelity_mode = mode
    agent = AgentSearch("Evaluation: binary log loss.", aq05_cfg, Journal())
    execute = Interpreter(aq05_cfg.workspace_dir, cfg=aq05_cfg, timeout=30).run
    drafts = [agent.step(None, execute, execute_immediately=False) for _ in range(3)]
    for node in drafts:
        agent.execute_deferred_node(node, execute)
        assert node.is_valid and node.training_fraction == .5, node.analysis
    runtime = agent.phase1_runtime
    selector = runtime.study7_selector
    # This tests fidelity filtering/refitting, not which parent AQ05 ranks first.
    # Constrain only this test's candidate support; retain real AQ05 prediction,
    # reservation, execution, observation and refit. Production is unchanged.
    select = selector.select
    def execute_from(parent):
        def select_parent(agent, options):
            selected = [option for option in options if option.parent_node_id == parent.id]
            assert selected, "Required test parent is not in the legal candidate set"
            return select(agent, selected)
        with monkeypatch.context() as patch:
            patch.setattr(selector, "select", select_parent)
            child = agent.step(parent, execute)
        assert child.parent is parent
        return child

    cheap = execute_from(drafts[0])
    assert cheap.is_valid and cheap.training_fraction == .5, cheap.analysis
    before = copy.deepcopy(selector.gp.training_hash)
    assert len(selector.gp.rows) > 0
    validation = (aq05_cfg.workspace_dir / "input/validation.csv").read_bytes()
    agent.fixed_evaluation.switch_to_full()
    if mode == "reset":
        selector.full_training_only = True
        selector.observe(runtime.transition_store.records())
        assert not selector.gp.rows and selector.gp.factor is None
    else:
        assert selector.gp.training_hash == before
    assert (aq05_cfg.workspace_dir / "input/validation.csv").read_bytes() == validation
    agent.update_data_preview()
    full = execute_from(cheap)
    assert full.is_valid and full.training_fraction == 1., full.analysis
    if mode == "reset":
        assert not selector.gp.rows  # A full child of a cheap parent is not a full/full gap.
    # Reproduce the Mila failure lineage: two full siblings of a cheap parent
    # still provide zero admissible full/full residuals in reset mode.
    sibling_full = execute_from(cheap)
    assert sibling_full.is_valid and sibling_full.training_fraction == 1., sibling_full.analysis
    if mode == "reset":
        assert not selector.gp.rows
    next_full = execute_from(full)
    assert next_full.is_valid and next_full.training_fraction == 1., next_full.analysis
    assert len(selector.gp.rows) >= 1
    if mode == "reset":
        records = runtime.transition_store.records()
        ids = full_node_ids(records)
        expected = selector.state.training_data(records, ids)
        assert selector.gp.rows == expected[0]
        assert len(selector.gp.rows) == 1
    selected = finalize_submission(aq05_cfg, agent.journal)
    assert selected.training_fraction == 1.
    agent.capture_resume_state()
    journal = serialize.loads_json(serialize.dumps_json(agent.journal), Journal)
    restored = AgentSearch("ignored", copy.deepcopy(aq05_cfg), journal)
    restored_selector = restored.phase1_runtime.study7_selector
    restored_selector.observe(restored.phase1_runtime.transition_store.records())
    assert restored_selector.full_training_only == (mode == "reset")
    assert restored_selector.gp.training_hash == selector.gp.training_hash


def test_all_campaign_commands_load_actual_aq05_config(tmp_path, monkeypatch):
    import sys
    from config import load_cfg
    from config.acq7 import acq7_config
    from tasks.mlebench_lite.mila_multifidelity import read, SPEC, search_command
    settings = {"spec": read(SPEC), "data_dir": str(tmp_path / "data")}
    for task in settings["spec"]["tasks"]:
        for mode in settings["spec"]["modes"]:
            command = search_command(settings, tmp_path, task, mode)
            monkeypatch.setattr(sys, "argv", command[2:])
            cfg = load_cfg()
            assert acq7_config(cfg).acquisition_type == "finite_pool_kg"
            assert cfg.evaluation.enabled and cfg.evaluation.multifidelity_mode == mode
            assert cfg.audited_bo == "redesign"
            assert cfg.agent.time_limit == 43200 and cfg.exec.timeout == 32400
            assert cfg.agent.initial_drafts == cfg.agent.search.parallel_search_num == 3


@pytest.mark.parametrize("mode", ["carry_over", "reset"])
@pytest.mark.parametrize("trigger", ["time", "steps"])
def test_runner_drains_then_continues_full_search(aq05_cfg, mode, trigger, monkeypatch):
    import run as runner
    from engine.fixed_evaluation import FixedEvaluation
    import threading
    aq05_cfg.evaluation.multifidelity_mode = mode
    aq05_cfg.agent.steps = 12
    aq05_cfg.run_result_file = str(Path(aq05_cfg.log_dir).parent / "run-info.json")
    Path(aq05_cfg.run_result_file).parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(runner, "load_cfg", lambda: aq05_cfg)
    monkeypatch.setattr(runner, "load_task_desc", lambda cfg: "Evaluation: binary log loss.")
    captured, active, counts, lock = [], set(), {"cheap": 0, "switches": 0}, threading.Lock()
    class Agent(AgentSearch):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            captured.append(self)
    monkeypatch.setattr(runner, "Agent", Agent)
    real_sleep = runner.time.sleep
    monkeypatch.setattr(runner.time, "sleep", lambda seconds: real_sleep(min(seconds, .001)))
    begin, score, switch = FixedEvaluation.begin_attempt, FixedEvaluation.score, FixedEvaluation.switch_to_full
    def begin_record(self, node, remaining):
        value = begin(self, node, remaining)
        with lock:
            active.add(node.id)
        return value
    def score_record(self, node):
        try:
            return score(self, node)
        finally:
            with lock:
                active.discard(node.id)
                if node.training_fraction == .5:
                    counts["cheap"] += 1
                    if trigger == "time" and counts["cheap"] == 1:
                        # Move only the budget clock across the 40% threshold.
                        captured[0].start_time -= .41 * aq05_cfg.agent.time_limit
    def switch_record(self):
        assert not active, "A cheap evaluation was still running at the switch"
        assert not captured[0].phase1_runtime.root_draft_state["pending"]
        counts["switches"] += 1
        if trigger == "steps":
            agent = captured[0]
            assert runner.time.time() - agent.start_time < .40 * aq05_cfg.agent.time_limit
            assert agent.phase1_runtime.root_draft_state["submitted_count"] >= 5
        return switch(self)
    monkeypatch.setattr(FixedEvaluation, "begin_attempt", begin_record)
    monkeypatch.setattr(FixedEvaluation, "score", score_record)
    monkeypatch.setattr(FixedEvaluation, "switch_to_full", switch_record)
    workers = []
    real_pool = runner.ThreadPoolExecutor
    def pool(*args, **kwargs):
        workers.append(kwargs["max_workers"])
        return real_pool(*args, **kwargs)
    monkeypatch.setattr(runner, "ThreadPoolExecutor", pool)
    runner.run()
    agent = captured[0]
    assert counts["switches"] == 1 and workers == [3]
    assert agent.phase1_runtime.root_draft_state["submitted_count"] <= aq05_cfg.agent.steps
    valid = [node for node in agent.journal.nodes if node.is_valid]
    assert {node.training_fraction for node in valid} == {.5, 1.}
    full = [node for node in valid if node.training_fraction == 1.]
    assert len(full) >= 3
    assert all(node.evaluation_result["n_scored"] == node.evaluation_result["n_expected"] for node in valid)
    final = json.loads((aq05_cfg.log_dir / "final_submission.json").read_text())
    assert final["node_id"] in {node.id for node in full}
    selector = agent.phase1_runtime.study7_selector
    if mode == "reset":
        records = agent.phase1_runtime.transition_store.records()
        assert selector.gp.rows == selector.state.training_data(records, full_node_ids(records))[0]


def test_runner_exhausted_cheap_search_does_not_idle_until_switch(aq05_cfg, monkeypatch):
    import run as runner
    from engine.parent_selectors.runtime import SearchExhausted
    aq05_cfg.evaluation.multifidelity_mode = "reset"
    aq05_cfg.agent.steps = 12
    aq05_cfg.run_result_file = None
    monkeypatch.setattr(runner, "load_cfg", lambda: aq05_cfg)
    monkeypatch.setattr(runner, "load_task_desc", lambda cfg: "Evaluation: binary log loss.")
    real_step = AgentSearch.step
    def exhausted(self, *args, **kwargs):
        if kwargs.get("execute_immediately") is False:
            return real_step(self, *args, **kwargs)
        raise SearchExhausted("No legal normal, draft, or debug action remains")
    monkeypatch.setattr(AgentSearch, "step", exhausted)
    real_sleep = runner.time.sleep
    sleeps = []
    def sleep(seconds):
        if seconds == 1:
            sleeps.append(seconds)
            raise AssertionError("Runner is idling after all legal work is exhausted")
        real_sleep(.001)
    monkeypatch.setattr(runner.time, "sleep", sleep)
    with pytest.raises(RuntimeError, match="no runnable work"):
        runner.run()
