from __future__ import annotations

import json

from engine.parent_selectors.base import (
    CandidateScore,
    SelectionDecision,
    SelectorResult,
    utc_timestamp,
)
from engine.parent_selectors.runtime import Phase1SelectorRuntime
from conftest import add_valid_node, make_agent, make_runtime_config


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _record_completed_current(runtime, agent, parent, current, reward=1.25):
    prior_context = runtime.record_direct_parent(agent, parent)
    return runtime.record_transition(
        parent_node=parent,
        child_node=current,
        context=prior_context,
        generation_wall_time_sec=2.0,
        execution_wall_time_sec=3.0,
        metric_maximize=agent.metric_maximize,
        reward=reward,
    )


def test_root_only_keeps_original_direct_continuation(tmp_path):
    runtime = Phase1SelectorRuntime(make_runtime_config(tmp_path, "bayesian_child_ucb"))
    agent = make_agent()
    current = add_valid_node(agent, node_id="current", metric=0.7)
    current.continue_improve = True

    context = runtime.select_continuation_parent(agent, current)

    assert context.option.parent_node_id == current.id
    assert context.decision.selection_reason == "direct_continuation"
    assert not (tmp_path / "phase1" / "continuation_events.jsonl").exists()


def test_bayesian_cold_start_continues_current_even_when_locked(tmp_path):
    cfg = make_runtime_config(tmp_path, "bayesian_child_ucb")
    cfg.parent_selector.decision_scope = "all_normal_steps"
    runtime = Phase1SelectorRuntime(cfg)
    agent = make_agent()
    current = add_valid_node(agent, node_id="current", metric=0.7, branch_id=1)
    add_valid_node(agent, node_id="other", metric=0.9, branch_id=2)
    current.continue_improve = True
    current.lock = True

    context = runtime.select_continuation_parent(agent, current)

    assert context.option.parent_node_id == current.id
    assert context.decision.selection_reason == "continuation_cold_start"
    assert context.decision.fallback_used is True
    assert current.lock is True
    event = _read_jsonl(tmp_path / "phase1" / "continuation_events.jsonl")[-1]
    assert event["outcome"] == "cold_start_continue"
    assert event["backpropagated"] is False


def test_all_normal_scope_switches_and_closes_current_exactly_once(tmp_path):
    cfg = make_runtime_config(tmp_path, "parent_metric_greedy")
    cfg.parent_selector.decision_scope = "all_normal_steps"
    runtime = Phase1SelectorRuntime(cfg)
    agent = make_agent()
    parent = add_valid_node(agent, node_id="parent", metric=0.4, branch_id=1)
    current = add_valid_node(
        agent,
        node_id="current",
        metric=0.5,
        parent=parent,
        branch_id=1,
        stage="improve",
    )
    other = add_valid_node(agent, node_id="other", metric=0.9, branch_id=2)
    current.continue_improve = True
    agent.current_node_list.append(current)
    transition = _record_completed_current(runtime, agent, parent, current)

    context = runtime.select_continuation_parent(agent, current)

    assert context.option.parent_node_id == other.id
    assert context.decision.selection_reason == "continuation_ranked_switch"
    assert current.continue_improve is False
    assert current.lock is False
    assert current.continuation_closed_transition_id == transition.transition_id
    assert current not in agent.current_node_list
    assert current.visits == 1
    assert current.total_reward == 1.25
    assert parent.visits == 1

    _, backpropagated = runtime._close_current_continuation(agent, current)
    assert backpropagated is False
    assert current.visits == 1
    assert parent.visits == 1

    event = _read_jsonl(tmp_path / "phase1" / "continuation_events.jsonl")[-1]
    assert event["outcome"] == "switch"
    assert event["transition_id"] == transition.transition_id
    assert event["reward"] == 1.25
    assert event["backpropagated"] is True


def test_all_normal_scope_can_keep_current_without_backpropagation(tmp_path):
    cfg = make_runtime_config(tmp_path, "parent_metric_greedy")
    cfg.parent_selector.decision_scope = "all_normal_steps"
    runtime = Phase1SelectorRuntime(cfg)
    agent = make_agent()
    parent = add_valid_node(agent, node_id="parent", metric=0.4, branch_id=1)
    current = add_valid_node(
        agent,
        node_id="current",
        metric=0.95,
        parent=parent,
        branch_id=1,
        stage="improve",
    )
    add_valid_node(agent, node_id="other", metric=0.9, branch_id=2)
    current.continue_improve = True
    agent.current_node_list.append(current)
    transition = _record_completed_current(runtime, agent, parent, current)

    context = runtime.select_continuation_parent(agent, current)

    assert context.option.parent_node_id == current.id
    assert context.decision.selection_reason == "continuation_ranked_continue"
    assert current.continue_improve is True
    assert current.lock is False
    assert current.continuation_closed_transition_id is None
    assert current.visits == 0
    assert parent.visits == 0
    event = _read_jsonl(tmp_path / "phase1" / "continuation_events.jsonl")[-1]
    assert event["outcome"] == "continue"
    assert event["transition_id"] == transition.transition_id
    assert event["backpropagated"] is False


def test_model_score_near_tie_prefers_current(tmp_path):
    runtime = Phase1SelectorRuntime(make_runtime_config(tmp_path, "bayesian_child_ucb"))
    agent = make_agent()
    current = add_valid_node(agent, node_id="current", metric=0.7, branch_id=1)
    other = add_valid_node(agent, node_id="other", metric=0.8, branch_id=2)
    options = runtime._continuation_options(
        agent,
        current,
        runtime._make_option(agent, current).preview,
    )
    current_option = next(option for option in options if option.parent_node_id == current.id)
    other_option = next(option for option in options if option.parent_node_id == other.id)
    scores = (
        CandidateScore(current.id, current_option.operator, (), 0.7, 0.1, 0.2, 1.0, 1.0, 1.0, 1.0),
        CandidateScore(other.id, other_option.operator, (), 0.8, 0.1, 0.2, 1.0, 1.0, 1.0, 1.0 + 5e-10),
    )
    decision = SelectionDecision(
        selector_name="bayesian_child_ucb",
        selected_parent_id=other.id,
        selected_operator=other_option.operator,
        selected_score=1.0 + 5e-10,
        model_version="model-1",
        fallback_used=False,
        decision_id="decision",
        timestamp=utc_timestamp(),
    )
    result = SelectorResult(decision, other_option, scores, tuple(options))

    preferred = runtime._prefer_current_on_tie(result, current)

    assert preferred.selected_option.parent_node_id == current.id
    assert preferred.decision.selected_parent_id == current.id


def test_debug_path_remains_forced_outside_continuation_ranking(tmp_path):
    cfg = make_runtime_config(tmp_path, "parent_metric_greedy")
    cfg.parent_selector.decision_scope = "all_normal_steps"
    runtime = Phase1SelectorRuntime(cfg)
    agent = make_agent()
    current = add_valid_node(agent, node_id="current", metric=0.7)
    current.is_buggy = True
    current.is_valid = False

    context = runtime.select_continuation_parent(agent, current)

    assert context.option.parent_node_id == current.id
    assert context.option.operator == "debug"
    assert context.decision.selection_reason == "special_path_bypass"
    assert not (tmp_path / "phase1" / "continuation_events.jsonl").exists()
