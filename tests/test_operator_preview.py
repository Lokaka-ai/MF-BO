import random
import time

from engine.operator_preview import (
    assert_preview_matches_node,
    eligible_normal_parents,
    execute_operator_preview,
    preview_next_operator,
    requires_topk_extension,
)
from engine.search_node import SearchNode
from utils.metric import MetricValue
from conftest import add_valid_node, make_agent


def _generated_child(parent, stage):
    return SearchNode(
        parent=parent,
        plan=stage,
        code="pass",
        metric=MetricValue(parent.metric.value, maximize=parent.metric.maximize),
        stage=stage,
        is_buggy=None,
        branch_id=parent.branch_id,
        local_best_node=parent,
    )


def _stagnant_parent(agent):
    first = add_valid_node(agent, node_id="first", metric=0.90)
    second = add_valid_node(
        agent, node_id="second", metric=0.80, parent=first, stage="improve"
    )
    parent = add_valid_node(
        agent, node_id="parent", metric=0.70, parent=second, stage="improve"
    )
    return parent


def test_improve_preview_is_the_operator_executed(monkeypatch):
    agent = make_agent()
    parent = add_valid_node(agent, node_id="parent", metric=0.8)
    preview = preview_next_operator(agent, parent)
    assert preview.operator == "improve"

    monkeypatch.setattr(
        "engine.operator_preview.improve_agent.run",
        lambda current_agent, current_parent: _generated_child(current_parent, "improve"),
    )
    child = execute_operator_preview(agent, parent, preview)
    assert_preview_matches_node(preview, child)


def test_evolution_preview_respects_existing_trajectory_fallback(monkeypatch):
    agent = make_agent()
    parent = _stagnant_parent(agent)
    preview = preview_next_operator(agent, parent)
    assert preview.operator == "evolution"

    monkeypatch.setattr(
        "engine.operator_preview.evolution_agent.run",
        lambda current_agent, current_parent: _generated_child(current_parent, "evolution"),
    )
    child = execute_operator_preview(agent, parent, preview)
    assert_preview_matches_node(preview, child)

    shallow_agent = make_agent()
    shallow = add_valid_node(shallow_agent, node_id="shallow", metric=0.7)
    shallow_agent.branch_successful_nodes[1] = [
        add_valid_node(shallow_agent, node_id="best", metric=0.9, branch_id=1),
        add_valid_node(shallow_agent, node_id="worse", metric=0.8, branch_id=1),
        shallow,
    ]
    shallow_preview = preview_next_operator(shallow_agent, shallow)
    assert shallow_preview.operator == "improve"


def test_fusion_preview_reuses_original_reference_nodes(monkeypatch):
    agent = make_agent()
    agent.acfg.use_evolution = False
    agent.acfg.use_fusion = True
    agent.search_start_time = time.time() - 600
    parent = _stagnant_parent(agent)
    reference = add_valid_node(agent, node_id="reference", metric=0.95, branch_id=2)
    monkeypatch.setattr(
        "engine.operator_preview.fusion_agent.get_fusion_candidates",
        lambda current_agent, current_parent: [reference],
    )
    preview = preview_next_operator(agent, parent)
    assert preview.operator == "fusion"
    assert preview.reference_node_ids == (reference.id,)

    seen = {}

    def fake_run(current_agent, current_parent, reference_nodes=None):
        seen["references"] = [node.id for node in reference_nodes]
        return _generated_child(current_parent, "fusion")

    monkeypatch.setattr("engine.operator_preview.fusion_agent.run", fake_run)
    child = execute_operator_preview(agent, parent, preview)
    assert seen["references"] == [reference.id]
    assert_preview_matches_node(preview, child)


def test_debug_and_aggregation_are_selector_bypasses(monkeypatch):
    agent = make_agent()
    parent = add_valid_node(agent, node_id="failed", metric=0.5)
    parent.is_buggy = True
    parent.is_valid = False
    debug = preview_next_operator(agent, parent)
    assert (debug.operator, debug.bypass_selector) == ("debug", True)
    monkeypatch.setattr(
        "engine.operator_preview.debug_agent.run",
        lambda current_agent, current_parent: _generated_child(current_parent, "debug"),
    )
    debug_child = execute_operator_preview(agent, parent, debug)
    assert_preview_matches_node(debug, debug_child)

    add_valid_node(agent, node_id="draft2", metric=0.6, branch_id=2)
    aggregation = preview_next_operator(agent, agent.virtual_root)
    assert (aggregation.operator, aggregation.bypass_selector) == ("aggregation", True)
    monkeypatch.setattr(
        "engine.operator_preview.aggregation_agent.run",
        lambda current_agent, mode, parent_node: _generated_child(
            parent_node, "fusion_draft"
        ),
    )
    aggregation_child = execute_operator_preview(
        agent, agent.virtual_root, aggregation
    )
    assert_preview_matches_node(aggregation, aggregation_child)


def test_unresolved_parent_preserves_original_no_op_path():
    agent = make_agent()
    parent = add_valid_node(agent, node_id="pending", metric=0.5)
    parent.is_buggy = None
    preview = preview_next_operator(agent, parent)
    assert (preview.operator, preview.bypass_selector) == ("no_op", True)
    assert execute_operator_preview(agent, parent, preview) is None


def test_operator_preview_does_not_advance_random_state(monkeypatch):
    agent = make_agent()
    agent.acfg.use_fusion = True
    agent.search_start_time = time.time() - 600
    parent = _stagnant_parent(agent)
    reference = add_valid_node(agent, node_id="ref", metric=0.95, branch_id=2)
    monkeypatch.setattr(
        "engine.operator_preview.fusion_agent.get_fusion_candidates",
        lambda current_agent, current_parent: [reference],
    )
    random.seed(123)
    state = random.getstate()
    preview_next_operator(agent, parent)
    assert random.getstate() == state


def test_eligible_set_allows_only_legal_topk_overflow_parent():
    agent = make_agent()
    agent.scfg.num_improves = 1
    agent.scfg.topk_late_k = 1
    agent.scfg.topk_late_max_per_branch = 1
    agent.search_start_time = time.time() - 900
    top = add_valid_node(agent, node_id="top", metric=0.95, branch_id=1)
    low = add_valid_node(agent, node_id="low", metric=0.50, branch_id=2)
    add_valid_node(agent, node_id="top-child", metric=0.90, parent=top, branch_id=1, stage="improve")
    add_valid_node(agent, node_id="low-child", metric=0.40, parent=low, branch_id=2, stage="improve")
    top.expected_child_count = len(top.children)
    low.expected_child_count = len(low.children)

    eligible_ids = {node.id for node in eligible_normal_parents(agent)}
    assert top.id in eligible_ids
    assert low.id not in eligible_ids
    assert requires_topk_extension(agent, top) is True
    preview = preview_next_operator(agent, top, topk_triggered=True)
    assert preview.topk_triggered is True
