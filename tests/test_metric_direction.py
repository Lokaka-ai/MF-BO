import pytest

from engine.parent_selectors.features import continuous_improvement, utility_from_metric
from engine.search_node import SearchNode
from utils.metric import MetricValue


def test_maximize_improvement_direction():
    assert utility_from_metric(0.40, True) == pytest.approx(0.40)
    assert continuous_improvement(0.40, 0.45, True) == pytest.approx(0.05)


def test_minimize_improvement_direction():
    assert utility_from_metric(0.40, False) == pytest.approx(-0.40)
    assert continuous_improvement(0.40, 0.35, False) == pytest.approx(0.05)


def test_missing_metric_has_no_continuous_target():
    assert continuous_improvement(None, 0.35, False) is None
    assert continuous_improvement(0.40, None, False) is None


@pytest.mark.parametrize("maximize,best,worst", [(False, 0.1, 0.9), (True, 0.9, 0.1)])
def test_child_memory_respects_metric_direction(maximize, best, worst):
    parent = SearchNode(code="", stage="draft", metric=MetricValue(0.5, maximize=maximize))
    for value in (0.1, 0.9):
        SearchNode(
            parent=parent,
            code="",
            plan=f"candidate {value}",
            stage="improve",
            metric=MetricValue(value, maximize=maximize),
            is_buggy=False,
            is_valid=True,
        )

    memory = parent.fetch_child_memory()

    assert f"best: {best:.4f}" in memory
    assert memory.index(f"Design: candidate {best}") < memory.index(f"Design: candidate {worst}")
