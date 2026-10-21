"""Real local dispatch/feedback; only external services use the existing test fixture."""
import copy
from pathlib import Path

import pytest
from omegaconf import OmegaConf

import test_root_draft_runtime as baseline
from test_root_draft_runtime import real_agent
from engine.agent_search import AgentSearch
from engine.parent_selectors.base import OperatorPreview
from engine.parent_selectors.runtime import SelectionPending, SearchExhausted
from engine.search_node import Journal
from utils import serialize


@pytest.fixture
def factory(real_agent, monkeypatch):
    original = baseline._load_cfg

    def build(*, preset="gp6", selector="gp_child_ucb", **kwargs):
        def load(*args, **kw):
            cfg = OmegaConf.merge(original(*args, **kw), OmegaConf.load(
                Path(__file__).resolve().parents[1] / f"config/experiments/{preset}.yaml"))
            # This fixture supplies scores through the mocked legacy evaluator.
            cfg.evaluation.enabled = False
            cfg.agent.use_evolution = False
            cfg.agent.use_fusion = False
            return cfg
        monkeypatch.setattr(baseline, "_load_cfg", load)
        return real_agent(selector=selector, enabled=False, **kwargs)
    return build


def seed(agent, execute):
    node = agent.step(None, execute, execute_immediately=False)
    agent.execute_deferred_node(node, execute)
    return node


def trial(agent, parent, execute):
    runtime = agent.phase1_runtime
    context = runtime.record_direct_parent(agent, parent, preview=OperatorPreview("improve"))
    return agent._run_single_step(parent, context, execute)[1]


@pytest.mark.parametrize("where", ["global", "continuation"])
@pytest.mark.parametrize("preset,selector", [
    ("gp6", "gp_child_ucb"), ("gp7_ancestry_ucb", "gp7_ancestry_logei"),
    ("gp7_short_ucb_pibo", "gp7_ancestry_logei"),
    ("gp7_long_horizon_ucb", "gp7_long_horizon_logei"),
])
def test_normal_selection_waits_for_all_initial_drafts(factory, where, preset, selector):
    agent, execute, _ = factory(initial=3, cap=5, preset=preset, selector=selector)
    runtime = agent.phase1_runtime
    nodes = [agent.step(None, execute, execute_immediately=False) for _ in range(3)]
    select = (lambda: runtime.select_global_parent(agent)) if where == "global" else (
        lambda: runtime.select_continuation_parent(agent, nodes[0]))
    for node in nodes[:2]:
        agent.execute_deferred_node(node, execute)
        nodes[0].lock = False
        before = runtime.root_draft_state["decision_count"]
        with pytest.raises(SelectionPending):
            select()
        assert runtime.root_draft_state["decision_count"] == before
    agent.execute_deferred_node(nodes[2], execute)
    context = select()
    assert context.option.operator == "improve"
    child = agent._run_single_step(context.option.parent_node, context, execute)[1]
    assert child.is_valid
    assert runtime.root_draft_state["initial_terminal"] == 3


def test_agent_selector_keeps_its_existing_async_initialization(factory, monkeypatch):
    agent, execute, _ = factory(initial=3, cap=5, preset="agent_selector", selector="agent_selector")
    runtime = agent.phase1_runtime
    nodes = [agent.step(None, execute, execute_immediately=False) for _ in range(3)]
    agent.execute_deferred_node(nodes[0], execute)
    monkeypatch.setattr(runtime.selectors["agent_selector"], "select", lambda a, options:
        runtime._forced_result(a, options[0].parent_node, reason="test_agent_choice", preview=options[0].preview))
    context = runtime.select_continuation_parent(agent, nodes[0])
    assert context.option.operator == "improve"
    assert runtime.root_draft_state["initial_terminal"] == 1


@pytest.mark.parametrize("where", ["global", "continuation"])
@pytest.mark.parametrize("preset,selector", [
    ("gp6", "gp_child_ucb"), ("gp7_ancestry_ucb", "gp7_ancestry_logei"),
    ("gp7_short_ucb_pibo", "gp7_ancestry_logei"),
])
def test_stagnation_does_not_add_drafts_before_or_after_resume(factory, where, preset, selector):
    agent, execute, _ = factory(cap=5, preset=preset, selector=selector)
    current = seed(agent, execute)
    for resuming in [False, True]:
        if resuming:
            agent.capture_resume_state()
            agent = AgentSearch("ignored", copy.deepcopy(agent.cfg),
                serialize.loads_json(serialize.dumps_json(agent.journal), Journal))
            current = next(n for n in agent.journal.nodes if n.id == current.id)
        runtime = agent.phase1_runtime
        for _ in range(4):
            current = trial(agent, current, execute)
        current.lock = False
        context = (runtime.select_global_parent(agent) if where == "global" else
                   runtime.select_continuation_parent(agent, current))
        assert context.option.operator == "improve"
        current = agent._run_single_step(context.option.parent_node, context, execute)[1]
        assert runtime.root_draft_state["draft_submitted"] == 1
        assert all(r.operator != "draft" for r in runtime.transition_store.records()[1:])


@pytest.mark.parametrize("preset,selector", [
    ("gp6", "gp_child_ucb"), ("gp7_ancestry_ucb", "gp7_ancestry_logei"),
    ("agent_selector", "agent_selector"),
])
def test_exhausted_directions_do_not_use_remaining_draft_budget(factory, preset, selector):
    agent, execute, _ = factory(cap=2, preset=preset, selector=selector)
    current = seed(agent, execute)
    current.is_terminal = True
    with pytest.raises(SearchExhausted):
        agent.phase1_runtime.select_global_parent(agent)
    assert agent.phase1_runtime.root_draft_state["draft_submitted"] == 1


def test_failed_additional_draft_must_finish_repair_before_another_is_added(factory):
    agent, execute, services = factory(cap=5)
    runtime = agent.phase1_runtime
    services.fail = "validation"
    seed(agent, execute)
    recovery = runtime.select_global_parent(agent)
    assert recovery.decision.selection_reason == "feasibility_draft"
    failed = agent._run_single_step(recovery.option.parent_node, recovery, execute)[1]
    assert failed.is_valid is False
    context = runtime.select_global_parent(agent)
    assert context.option.operator == "debug"
    assert runtime.root_draft_state["draft_submitted"] == 2
