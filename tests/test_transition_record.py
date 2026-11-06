import pytest

from engine import evaluation
from engine.operator_preview import preview_next_operator
from engine.parent_selectors.base import (
    CandidateScore,
    ExpansionOption,
    OperatorPreview,
    SelectionDecision,
    SelectorResult,
    utc_timestamp,
)
from engine.parent_selectors.features import FEATURE_NAMES
from engine.parent_selectors.runtime import Phase1SelectorRuntime, StepDecisionContext
from engine.parent_selectors.transition_store import TransitionStore
from engine.search_node import SearchNode
from utils.metric import MetricValue
from conftest import add_valid_node, make_agent


def _zero_features(parent_utility=0.5):
    features = {name: 0.0 for name in FEATURE_NAMES}
    features["parent_utility"] = parent_utility
    features["operator__improve"] = 1.0
    return features


def test_invalid_child_records_cost_but_not_gain_and_keeps_original_reward(tmp_path):
    agent = make_agent(maximize=True)
    parent = add_valid_node(agent, node_id="parent", metric=0.5)
    child = SearchNode(
        id="child",
        parent=parent,
        plan="failed",
        code="",
        metric=MetricValue(0.2, maximize=True),
        stage="improve",
        is_buggy=True,
        is_valid=False,
        branch_id=1,
        local_best_node=parent,
    )
    preview = OperatorPreview(operator="improve")
    option = ExpansionOption(
        parent_node_id=parent.id,
        operator="improve",
        reference_node_ids=(),
        state_features=_zero_features(),
        parent_node=parent,
        preview=preview,
    )
    decision = SelectionDecision(
        selector_name="bayesian_child_ucb",
        selected_parent_id=parent.id,
        selected_operator="improve",
        selected_score=1.0,
        model_version="model-1",
        fallback_used=False,
        decision_id="decision-1",
        timestamp=utc_timestamp(),
    )
    score = CandidateScore(parent.id, "improve", (), 0.5, 0.1, 0.2, 0.9, 3.0, 0.8, 0.9)
    context = StepDecisionContext(SelectorResult(decision, option, (score,), (option,)))

    runtime = Phase1SelectorRuntime.__new__(Phase1SelectorRuntime)
    runtime.transition_store = TransitionStore(tmp_path / "transitions.jsonl")
    reward = evaluation.get_node_reward(agent, child)
    record = runtime.record_transition(
        parent_node=parent,
        child_node=child,
        context=context,
        generation_wall_time_sec=2.0,
        execution_wall_time_sec=3.0,
        metric_maximize=True,
        reward=reward,
    )

    assert record.valid is False
    assert record.improvement is None
    assert record.child_utility is None
    assert record.child_metric_raw == 0.2
    assert record.total_wall_time_sec == 5.0
    assert record.reward == -1.0
    assert runtime.transition_store.valid_gain_records() == []

    debug_preview = preview_next_operator(agent, child)
    assert debug_preview.operator == "debug"
    assert debug_preview.bypass_selector is True

    evaluation.backpropagate(child, reward)
    assert child.visits == 1
    assert parent.visits == 1
    assert parent.total_reward == -1.0


def test_transition_store_recovers_from_jsonl(tmp_path):
    agent = make_agent(maximize=False)
    parent = add_valid_node(agent, node_id="p", metric=0.40)
    child = add_valid_node(
        agent,
        node_id="c",
        metric=0.35,
        parent=parent,
        stage="improve",
    )
    preview = OperatorPreview(operator="improve")
    option = ExpansionOption("p", "improve", (), _zero_features(-0.4), parent, preview)
    decision = SelectionDecision(
        "progressive_mcgs", "p", "improve", 0.0, None, False,
        "d", utc_timestamp(),
    )
    score = CandidateScore("p", "improve", (), -0.4, None, None, None, None, None, 0.0)
    context = StepDecisionContext(SelectorResult(decision, option, (score,), (option,)))
    runtime = Phase1SelectorRuntime.__new__(Phase1SelectorRuntime)
    runtime.transition_store = TransitionStore(tmp_path / "transitions.jsonl")
    runtime.record_transition(
        parent_node=parent,
        child_node=child,
        context=context,
        generation_wall_time_sec=1.0,
        execution_wall_time_sec=2.0,
        metric_maximize=False,
        reward=1.0,
    )

    restored_store = TransitionStore(tmp_path / "transitions.jsonl")
    recovered = restored_store.records()
    assert len(recovered) == 1
    assert recovered[0].improvement == pytest.approx(0.05)

    assert restored_store.admit_child_ids({"different-child"}) == 1
    assert restored_store.records() == []
    assert len(TransitionStore(tmp_path / "transitions.jsonl").records()) == 1
