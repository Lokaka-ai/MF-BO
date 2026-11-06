from conftest import make_agent
from engine.evaluation import check_improvement
from engine.search_node import SearchNode
from utils.metric import MetricValue, WorstMetricValue


def test_check_improvement_handles_missing_local_best_after_debug_success():
    agent = make_agent()
    parent = SearchNode(
        id="buggy",
        parent=agent.virtual_root,
        code="",
        stage="debug",
        metric=WorstMetricValue(),
        is_buggy=True,
        branch_id=1,
    )
    child = SearchNode(
        id="valid",
        parent=parent,
        code="",
        stage="debug",
        metric=MetricValue(0.9, maximize=True),
        is_buggy=False,
        branch_id=1,
    )

    should_backpropagate = check_improvement(agent, child, parent)

    assert should_backpropagate is False
    assert child.local_best_node is child
    assert child.continue_improve is True
    assert child in agent.current_node_list
