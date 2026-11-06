from engine.parent_selectors.base import (
    CandidateScore,
    ExpansionOption,
    OperatorPreview,
    SelectionDecision,
    SelectorResult,
    utc_timestamp,
)
from engine.parent_selectors.features import FEATURE_NAMES
from engine.parent_selectors.selectors import BayesianUCBParentSelector
from engine.parent_selectors.surrogate import BayesianTransitionSurrogate
from engine.parent_selectors.transition_store import TransitionRecord
from conftest import add_valid_node, make_agent


def _features(value=0.5):
    result = {name: 0.0 for name in FEATURE_NAMES}
    result["parent_utility"] = value
    result["global_best_utility"] = value
    result["operator__improve"] = 1.0
    return result


def _option(features=None, operator="improve"):
    agent = make_agent()
    parent = add_valid_node(agent, node_id=f"p-{operator}", metric=0.5)
    preview = OperatorPreview(operator=operator)
    option_features = dict(features or _features())
    option_features["operator__improve"] = float(operator == "improve")
    option_features[f"operator__{operator}"] = 1.0
    return ExpansionOption(
        parent.id, operator, (), option_features, parent, preview
    )


def _record(index):
    return TransitionRecord(
        transition_id=f"t{index}", decision_id=f"d{index}",
        parent_node_id=f"p{index}", child_node_id=f"c{index}", operator="improve",
        reference_node_ids=(), parent_metric_raw=0.5, child_metric_raw=0.6,
        parent_utility=0.5, child_utility=0.6, improvement=0.1,
        valid=True, generation_wall_time_sec=1.0, execution_wall_time_sec=1.0,
        total_wall_time_sec=2.0, gpu_time_sec=None, reward=1.0,
        selector_name="progressive_mcgs", selection_features=_features(0.5 + index * 0.01),
        timestamp="2026-08-27T00:00:00Z",
    )


class StubFallback:
    def select(self, agent, options):
        del agent
        option = options[0]
        score = CandidateScore(option.parent_node_id, option.operator, (), 0.5, None, None, None, None, None, 0.0)
        decision = SelectionDecision(
            "progressive_mcgs", option.parent_node_id, option.operator, 0.0,
            None, False, "fallback-decision", utc_timestamp(),
        )
        return SelectorResult(decision, option, (score,), tuple(options))


def _selector(records, *, min_total=1, min_valid=1):
    selector = BayesianUCBParentSelector(
        name="bayesian_child_ucb",
        acquisition_type="child_ucb",
        surrogate=BayesianTransitionSurrogate(default_cost=2.0),
        records_provider=lambda: list(records),
        min_valid_transitions=min_valid,
        min_total_transitions=min_total,
        min_operator_observations=1,
        refit_every=1,
        beta=1.5,
        use_validity_probability=False,
        cost_power=1.0,
    )
    selector.fallback = StubFallback()
    return selector


def test_small_sample_uses_progressive_mcgs_fallback():
    selector = _selector([], min_total=4, min_valid=3)
    result = selector.select(make_agent(), [_option()])
    assert result.decision.selector_name == "bayesian_child_ucb"
    assert result.decision.fallback_used is True
    assert "total_transitions" in result.decision.fallback_reason


def test_nan_prediction_feature_falls_back_without_terminating_workflow():
    records = [_record(index) for index in range(4)]
    bad_features = _features()
    bad_features["depth"] = float("nan")
    result = _selector(records).select(make_agent(), [_option(bad_features)])
    assert result.decision.fallback_used is True
    assert result.decision.fallback_reason.startswith("model_error:ValueError")


def test_unobserved_current_operator_keeps_selector_in_cold_start():
    records = [_record(index) for index in range(4)]
    result = _selector(records).select(
        make_agent(), [_option(operator="improve"), _option(operator="fusion")]
    )
    assert result.decision.fallback_used is True
    assert result.decision.fallback_reason == "operator_observations=0<1"
