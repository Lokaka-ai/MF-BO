import random

from engine import node_selection
from engine.parent_selectors.base import ExpansionOption, OperatorPreview
from engine.parent_selectors.selectors import ProgressiveMCGSSelector
from conftest import add_valid_node, make_agent


def _tree():
    agent = make_agent()
    first = add_valid_node(agent, node_id="a", metric=0.6, branch_id=1)
    second = add_valid_node(agent, node_id="b", metric=0.7, branch_id=2)
    agent.virtual_root.visits = 20
    first.visits = 4
    first.total_reward = 1.0
    second.visits = 5
    second.total_reward = 10.0
    return agent, first, second


def _options(first, second):
    return [
        ExpansionOption(first.id, "improve", (), {"parent_utility": 0.6}, first, OperatorPreview("improve")),
        ExpansionOption(second.id, "improve", (), {"parent_utility": 0.7}, second, OperatorPreview("improve")),
    ]


def test_adapter_calls_original_selector_and_returns_same_deterministic_node():
    direct_agent, direct_first, direct_second = _tree()
    adapter_agent, adapter_first, adapter_second = _tree()

    random.seed(7)
    direct = node_selection.select_with_soft_switch(direct_agent)
    random.seed(7)
    adapted = ProgressiveMCGSSelector().select(
        adapter_agent, _options(adapter_first, adapter_second)
    )

    assert direct.id == adapted.selected_option.parent_node_id
    assert direct.id == "b"
    assert adapted.decision.selector_name == "progressive_mcgs"


def test_adapter_expands_available_parent_when_child_is_only_in_flight():
    agent = make_agent()
    agent.scfg.num_bugs = 1
    preferred = add_valid_node(agent, node_id="preferred", metric=0.8, branch_id=1)
    alternative = add_valid_node(agent, node_id="alternative", metric=0.7, branch_id=2)
    in_flight = add_valid_node(
        agent,
        node_id="in-flight",
        metric=0.0,
        parent=preferred,
        branch_id=1,
        stage="improve",
    )
    in_flight.is_buggy = True
    in_flight.is_valid = False
    in_flight.expected_child_count = 1
    preferred.expected_child_count = 1
    preferred.continue_improve = True

    agent.virtual_root.visits = 20
    preferred.visits = 2
    preferred.total_reward = 10.0
    alternative.visits = 2
    alternative.total_reward = 0.0

    result = ProgressiveMCGSSelector().select(
        agent, _options(preferred, alternative)
    )

    assert result.selected_option.parent_node_id == preferred.id
    assert in_flight.children == set()
    assert in_flight.expected_child_count == agent.scfg.num_bugs
