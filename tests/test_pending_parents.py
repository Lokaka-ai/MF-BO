"""Pending-parent preference through selection, cancellation and task completion."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import threading

import pytest

from config import ParentSelectorConfig, _load_cfg
from conftest import (
    add_valid_node, make_agent, make_bnvs_runtime_config,
    make_reward_runtime_config, make_runtime_config,
)
from engine.agent_search import AgentSearch
from engine.parent_selectors.base import OperatorPreview
from engine.parent_selectors.runtime import Phase1SelectorRuntime
from engine.parent_selectors.selectors import ParentMetricGreedySelector
from engine.search_node import SearchNode
from utils.metric import MetricValue


UNMANAGED_BO = (
    "bayesian_gain_ucb", "bayesian_child_ucb", "cost_aware_bayesian",
    "cost_aware_bayesian_selector", "reward_only_bayesian_ts",
    "bnvs_mean_greedy", "bnvs_ts", "bnvs_ts_local",
)


def setup_runtime(tmp_path, monkeypatch, selector="bnvs_ts", *, enabled=True, scope="all_normal_steps"):
    if selector.startswith("bnvs_"):
        cfg = make_bnvs_runtime_config(tmp_path, selector)
    elif selector == "reward_only_bayesian_ts":
        cfg = make_reward_runtime_config(tmp_path)
    else:
        cfg = make_runtime_config(tmp_path, selector)
    cfg.parent_selector.prefer_unoccupied_parents = enabled
    cfg.parent_selector.decision_scope = scope
    runtime = Phase1SelectorRuntime(cfg)
    agent = make_agent()
    agent.phase1_runtime = runtime
    agent.acfg.use_aggregation = False
    nodes = [add_valid_node(agent, node_id=name, metric=quality, branch_id=1)
             for name, quality in (("a", .9), ("b", .8), ("c", .7))]
    # Isolate scheduling from score randomness; the real selector is retained
    # for the separate cold-start and acquisition integration tests.
    def rank(agent, options):
        result = ParentMetricGreedySelector().select(agent, options)
        return replace(result, decision=replace(result.decision, selector_name=selector))
    monkeypatch.setattr(runtime.selectors[selector], "select", rank)
    return runtime, agent, nodes


def complete(runtime, agent, context, *, valid=True):
    child = SearchNode(id="completed-" + context.decision.decision_id,
        code="pass", stage="improve", metric=MetricValue(.95, maximize=True),
        is_buggy=not valid, is_valid=valid)
    return runtime.record_transition(parent_node=context.option.parent_node, child_node=child,
        context=context, generation_wall_time_sec=1., execution_wall_time_sec=1.,
        metric_maximize=True, reward=1. if valid else -1.)


def test_preference_is_enabled_by_default():
    assert ParentSelectorConfig().prefer_unoccupied_parents is True
    assert _load_cfg(use_cli_args=False).parent_selector.prefer_unoccupied_parents is True


@pytest.mark.parametrize("selector", UNMANAGED_BO)
def test_every_bo_selector_reserves_distinct_parents_and_falls_back(tmp_path, monkeypatch, selector):
    runtime, agent, nodes = setup_runtime(tmp_path, monkeypatch, selector)
    choices = [runtime.select_global_parent(agent) for _ in range(4)]
    assert [context.option.parent_node_id for context in choices] == ["a", "b", "c", "a"]
    assert len(runtime.transition_store.records()) == 0
    # Cancel only one of two attempts using a: the second still owns a.
    runtime.cancel_reservation(agent, choices[0])
    assert runtime._pending_parent_ids() == {"a", "b", "c"}
    runtime.cancel_reservation(agent, choices[3])
    assert [option.parent_node_id for option in runtime._normal_options(agent)] == ["a"]
    assert nodes[0].branch_id == nodes[1].branch_id == nodes[2].branch_id
    decisions = [json.loads(line) for line in
                 (runtime.artifact_root / "selection_decisions.jsonl").read_text().splitlines()]
    assert [row["pending_parent_reuse"] for row in decisions] == [False, False, False, True]


def test_threaded_selection_and_reservation_are_atomic(tmp_path, monkeypatch):
    runtime, agent, _ = setup_runtime(tmp_path, monkeypatch)
    barrier = threading.Barrier(3)
    def select():
        barrier.wait(timeout=10)
        return runtime.select_global_parent(agent).option.parent_node_id
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(select) for _ in range(3)]
        assert {future.result(timeout=20) for future in futures} == {"a", "b", "c"}


@pytest.mark.parametrize("valid", [True, False])
def test_completed_or_failed_attempt_releases_only_its_parent(tmp_path, monkeypatch, valid):
    runtime, agent, _ = setup_runtime(tmp_path, monkeypatch)
    first = runtime.select_global_parent(agent)
    second = runtime.select_global_parent(agent)
    complete(runtime, agent, first, valid=valid)
    assert runtime._pending_parent_ids() == {second.option.parent_node_id}
    assert runtime.select_global_parent(agent).option.parent_node_id == first.option.parent_node_id


def test_parent_mask_is_independent_of_operator_and_does_not_lock_branch(tmp_path, monkeypatch):
    runtime, agent, nodes = setup_runtime(tmp_path, monkeypatch)
    runtime.select_global_parent(agent)
    options = [runtime._make_option(agent, nodes[0], preview=OperatorPreview("evolution")),
               runtime._make_option(agent, nodes[1])]
    assert [option.parent_node_id for option in runtime._prefer_unoccupied_options(options)] == ["b"]


def test_empty_or_illegal_pool_is_not_repopulated(tmp_path, monkeypatch):
    runtime, agent, nodes = setup_runtime(tmp_path, monkeypatch)
    runtime.select_global_parent(agent)
    for node in nodes:
        node.lock = True
    assert runtime._normal_options(agent) == []
    assert runtime._prefer_unoccupied_options([]) == []


def test_root_drafts_are_exempt(tmp_path, monkeypatch):
    runtime, agent, _ = setup_runtime(tmp_path, monkeypatch)
    root = runtime.record_direct_parent(agent, agent.virtual_root,
        preview=OperatorPreview("draft", bypass_selector=True))
    assert agent.virtual_root.id not in runtime._pending_parent_ids()
    assert runtime._prefer_unoccupied_options([root.option]) == [root.option]


def test_preference_can_be_disabled(tmp_path, monkeypatch):
    runtime, agent, _ = setup_runtime(tmp_path, monkeypatch, enabled=False)
    assert [runtime.select_global_parent(agent).option.parent_node_id for _ in range(3)] == ["a"] * 3
    assert not runtime._pending_parent_ids()


@pytest.mark.parametrize("selector", ["legacy_original", "progressive_mcgs", "random_parent", "parent_metric_greedy"])
def test_non_bo_selectors_keep_their_policy(tmp_path, selector):
    runtime = Phase1SelectorRuntime(make_runtime_config(tmp_path, selector))
    assert not runtime.prefer_unoccupied_parents


@pytest.mark.parametrize("scope", ["root_only", "all_normal_steps"])
def test_continuation_excludes_busy_current_and_other_busy_parents(tmp_path, monkeypatch, scope):
    runtime, agent, nodes = setup_runtime(tmp_path, monkeypatch, scope=scope)
    incoming = runtime.record_direct_parent(agent, agent.virtual_root,
        preview=OperatorPreview("draft", bypass_selector=True))
    runtime.record_transition(parent_node=agent.virtual_root, child_node=nodes[0], context=incoming,
        generation_wall_time_sec=1., execution_wall_time_sec=1., metric_maximize=True, reward=1.)
    runtime.select_global_parent(agent)
    runtime.select_global_parent(agent)
    context = runtime.select_continuation_parent(agent, nodes[0])
    assert context.option.parent_node_id == "c"
    assert context.decision.selection_reason.endswith("switch")
    assert runtime._pending_parent_ids() == {"a", "b", "c"}


def test_cold_start_fallback_cannot_reintroduce_occupied_parent(tmp_path, monkeypatch):
    runtime = Phase1SelectorRuntime(make_runtime_config(tmp_path))
    agent = make_agent()
    agent.acfg.use_aggregation = False
    nodes = [add_valid_node(agent, node_id=name, metric=.7) for name in ("a", "b")]
    monkeypatch.setattr("engine.parent_selectors.runtime.node_selection.select_with_soft_switch",
                        lambda agent: nodes[0])
    first = runtime.select_global_parent(agent)
    second = runtime.select_global_parent(agent)
    assert first.option.parent_node_id == "a"
    assert second.option.parent_node_id == "b"
    assert second.decision.fallback_used


def test_continuation_cold_start_cannot_reintroduce_occupied_current(tmp_path, monkeypatch):
    cfg = make_runtime_config(tmp_path)
    cfg.parent_selector.decision_scope = "all_normal_steps"
    runtime = Phase1SelectorRuntime(cfg)
    agent = make_agent()
    agent.acfg.use_aggregation = False
    current = add_valid_node(agent, node_id="a", metric=.9)
    add_valid_node(agent, node_id="b", metric=.8)
    incoming = runtime.record_direct_parent(agent, agent.virtual_root,
        preview=OperatorPreview("draft", bypass_selector=True))
    runtime.record_transition(parent_node=agent.virtual_root, child_node=current, context=incoming,
        generation_wall_time_sec=1., execution_wall_time_sec=1., metric_maximize=True, reward=1.)
    runtime.record_direct_parent(agent, current)
    monkeypatch.setattr("engine.parent_selectors.runtime.node_selection.select_with_soft_switch",
                        lambda agent: current)
    context = runtime.select_continuation_parent(agent, current)
    assert context.option.parent_node_id == "b"
    assert context.decision.fallback_used
    assert runtime._pending_parent_ids() == {"a", "b"}


def test_recovered_deferred_context_restores_occupancy(tmp_path, monkeypatch):
    runtime, agent, _ = setup_runtime(tmp_path, monkeypatch)
    context = runtime.select_global_parent(agent)
    restored = Phase1SelectorRuntime(runtime.cfg)
    node = SearchNode(code="pass", stage="improve")
    restored.attach_decision(node, context)
    assert restored._pending_parent_ids() == {context.option.parent_node_id}
    restored.cancel_reservation(agent, context)
    assert not restored._pending_parent_ids()


def test_generation_exception_releases_unmanaged_parent(tmp_path, monkeypatch):
    runtime, agent, _ = setup_runtime(tmp_path, monkeypatch)
    context = runtime.select_global_parent(agent)
    def fail(*args, **kwargs):
        raise RuntimeError("generation failed")
    agent._run_single_step = fail
    agent.journal_lock = threading.RLock()
    agent.data_preview = "fixture"
    agent.search_start_time = 1.
    agent._check_evaluator_health = lambda: None
    with pytest.raises(RuntimeError, match="generation failed"):
        if hasattr(AgentSearch, "run_selected_step"):
            AgentSearch.run_selected_step(agent, context, None)
        else:
            AgentSearch.step(agent, None, None, recovery_context=context)
    assert not runtime._pending_parent_ids()


def test_deferred_exception_releases_unmanaged_parent(tmp_path, monkeypatch):
    runtime, agent, _ = setup_runtime(tmp_path, monkeypatch)
    context = runtime.select_global_parent(agent)
    node = SearchNode(code="pass", stage="improve")
    runtime.attach_decision(node, context)
    def fail(*args, **kwargs):
        raise RuntimeError("execution failed")
    agent._execute_deferred_node = fail
    with pytest.raises(RuntimeError, match="execution failed"):
        AgentSearch.execute_deferred_node(agent, node, None)
    assert not runtime._pending_parent_ids()
