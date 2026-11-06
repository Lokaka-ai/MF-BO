"""Regression cases from the MF-GP launch audit; no live provider calls."""

import time
import pytest
from test_fixed_evaluation import fixed_cfg, setup_server, model_responses, CANDIDATE
from test_multifidelity_integration import aq05_cfg
from test_mf_gp import make_agent
from engine.executor import Interpreter
from engine.parent_selectors.runtime import SearchExhausted


@pytest.mark.parametrize("terminal", [False, True])
def test_failed_initial_program_keeps_native_debug_path(
    aq05_cfg, model_responses, terminal
):
    agent = make_agent(aq05_cfg)
    execute = Interpreter(aq05_cfg.workspace_dir, cfg=aq05_cfg, timeout=30).run
    model_responses.code = "raise RuntimeError('repairable candidate failure')"
    bad = agent.step(None, execute, execute_immediately=False)
    agent.execute_deferred_node(bad, execute)
    assert bad.is_buggy
    model_responses.code = CANDIDATE
    for _ in range(2):
        n = agent.step(None, execute, execute_immediately=False)
        agent.execute_deferred_node(n, execute)
    assert not bad.lock
    agent.phase1_runtime.study7_selector.terminal = terminal
    result = agent.phase1_runtime.select_global_parent(agent)
    assert result.option.operator == "debug" and result.option.parent_node_id == bad.id
    assert result.option.state_features["mf"]["fidelity"] == (1.0 if terminal else 0.5)
    repaired = agent.step(None, execute, recovery_context=result)
    assert repaired.is_valid
    assert agent.phase1_runtime.transition_store.records()[-1].operator == "debug"


def test_deadline_applies_to_forced_initialization(aq05_cfg):
    agent = make_agent(aq05_cfg)
    agent.start_time = time.time() - aq05_cfg.agent.time_limit - 1
    with pytest.raises(SearchExhausted):
        agent.phase1_runtime.select_global_parent(agent)


def test_recovered_generation_does_not_double_count_step_budget(aq05_cfg):
    agent = make_agent(aq05_cfg)
    runtime = agent.phase1_runtime
    runtime.select_global_parent(agent)
    state = runtime.root_draft_state
    state["submitted_count"] = aq05_cfg.agent.steps - 1
    next(iter(state["pending"].values()))["recovery_already_submitted"] = True
    assert runtime._budget_available(agent)


def test_future_import_program_executes_under_mf_wrapper(aq05_cfg, model_responses):
    model_responses.code = "from __future__ import annotations\n" + CANDIDATE
    agent = make_agent(aq05_cfg)
    execute = Interpreter(aq05_cfg.workspace_dir, cfg=aq05_cfg, timeout=30).run
    node = agent.step(None, execute, execute_immediately=False)
    agent.execute_deferred_node(node, execute)
    assert node.is_valid, node.analysis


def test_runner_resumes_reserved_initialization_without_extra_draft(
    aq05_cfg, monkeypatch
):
    from config import save_run
    from engine.mf_gp.controller import MFRuntime
    from engine.mf_gp.runner import run

    agent = make_agent(aq05_cfg)
    frozen = agent.phase1_runtime.select_global_parent(agent)
    agent.capture_resume_state()
    save_run(aq05_cfg, agent.journal)
    aq05_cfg.resume_run_dir = str(aq05_cfg.log_dir.parent)
    aq05_cfg.agent.steps = 6
    # Quiescent prior test object; production requires Slurm release evidence.
    monkeypatch.setattr(MFRuntime, "_owner_alive", staticmethod(lambda _: False))
    resumed = run(aq05_cfg)
    state = resumed.phase1_runtime.root_draft_state
    assert state["initial_submitted"] == state["initial_terminal"] == 3
    assert sum(n.stage == "draft" for n in resumed.journal.nodes) == 3
    assert resumed.phase1_runtime.transition_store.by_decision_id(
        frozen.decision.decision_id
    )
    assert len(resumed.phase1_runtime.study7_selector.observations) == 6


def test_future_setup_also_handles_linux_affinity_prefix():
    from engine.mf_gp.execution import insert_setup

    code = '"module docs"\nfrom __future__ import annotations\nvalue: list[int] = []\n'
    wrapped = insert_setup(
        insert_setup(code, "import random\nrandom.seed(42)\n"), "import os\n"
    )
    compile(wrapped, "<linux-candidate>", "exec")
    assert wrapped.index("from __future__") < wrapped.index("import os")


def test_shared_recovery_entry_uses_separate_fidelity_caps(aq05_cfg):
    from types import SimpleNamespace
    from engine.mf_recovery import execute
    from engine.executor import ExecutionResult

    aq05_cfg.evaluation.multifidelity_mode = "mf_gp"
    interpreter = SimpleNamespace(
        run=lambda *a, **kw: ExecutionResult(["ok"], 0.1, None)
    )
    for fraction in (0.5, 1.0):
        execute(
            interpreter,
            aq05_cfg,
            "pass",
            "p",
            evaluation_context={"training_fraction": fraction},
        )
    import json

    for fraction in (0.5, 1.0):
        path = aq05_cfg.log_dir / "mf_execution" / f"p__{fraction}.json"
        assert json.loads(path.read_text())["elapsed_seconds"] == 0.1
