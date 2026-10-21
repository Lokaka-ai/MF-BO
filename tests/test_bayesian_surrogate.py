import json
import math

import numpy as np
import pytest

from engine.parent_selectors.acquisition import score_predictions
from engine.parent_selectors.base import ExpansionOption, OperatorPreview, PredictiveTransition
from engine.parent_selectors.features import FEATURE_NAMES
from engine.parent_selectors.surrogate import (
    BayesianCostModel,
    BayesianTransitionSurrogate,
    cost_from_log,
    cost_to_log,
)
from engine.parent_selectors.transition_store import TransitionRecord
from engine.selection_logging import SelectionLogger
from conftest import add_valid_node, make_agent


def feature_row(value: float, operator: str = "improve") -> dict[str, float]:
    features = {name: 0.0 for name in FEATURE_NAMES}
    features["parent_utility"] = value
    features["global_best_utility"] = 1.0
    features["parent_global_gap"] = 1.0 - value
    features["depth"] = value + 2.0
    features["stage__improve"] = 1.0
    features[f"operator__{operator}"] = 1.0
    return features


def transition(
    index: int,
    value: float,
    *,
    operator: str = "improve",
    valid: bool = True,
    cost: float | None = None,
) -> TransitionRecord:
    gain = 0.25 * value + (0.15 if operator == "fusion" else 0.0)
    return TransitionRecord(
        transition_id=f"t{index}",
        decision_id=f"d{index}",
        parent_node_id=f"p{index}",
        child_node_id=f"c{index}",
        operator=operator,
        reference_node_ids=(),
        parent_metric_raw=value,
        child_metric_raw=value + gain if valid else None,
        parent_utility=value,
        child_utility=value + gain if valid else None,
        improvement=gain if valid else None,
        valid=valid,
        generation_wall_time_sec=0.5,
        execution_wall_time_sec=(cost if cost is not None else 2.0 + value) - 0.5,
        total_wall_time_sec=cost if cost is not None else 2.0 + value,
        gpu_time_sec=None,
        reward=1.0 if valid else -1.0,
        selector_name="progressive_mcgs",
        selection_features=feature_row(value, operator),
        timestamp="2026-08-27T00:00:00Z",
    )


def option(value: float, operator: str = "improve") -> ExpansionOption:
    agent = make_agent()
    parent = add_valid_node(agent, node_id=f"option-{value}-{operator}", metric=value)
    preview = OperatorPreview(operator=operator)
    return ExpansionOption(
        parent.id,
        operator,
        (),
        feature_row(value, operator),
        parent,
        preview,
    )


def test_bayesian_ridge_returns_finite_calibrated_shape_and_recovers(tmp_path):
    records = [transition(index, value) for index, value in enumerate(np.linspace(-1.0, 1.0, 30))]
    model = BayesianTransitionSurrogate(
        default_cost=4.0,
        selection_logger=SelectionLogger(tmp_path),
    )
    model.fit(records)
    fit_log = json.loads((tmp_path / "model_fits.jsonl").read_text())
    assert fit_log["fit_wall_time_sec"] > 0
    near = option(0.0)
    far = option(8.0)
    unseen_operator = option(0.0, "fusion")
    predictions = model.predict([near, far, unseen_operator])

    assert all(math.isfinite(item.mean_gain) for item in predictions)
    assert all(item.std_gain >= 0 for item in predictions)
    assert predictions[0].std_gain < predictions[1].std_gain
    assert predictions[2].std_gain > 0
    assert all(item.predicted_total_cost > 0 for item in predictions)

    recovered = BayesianTransitionSurrogate(default_cost=4.0)
    recovered.fit(records)
    recovered_predictions = recovered.predict([near, far, unseen_operator])
    assert [item.mean_gain for item in recovered_predictions] == pytest.approx(
        [item.mean_gain for item in predictions]
    )
    assert [item.std_gain for item in recovered_predictions] == pytest.approx(
        [item.std_gain for item in predictions]
    )


def test_invalid_transitions_train_cost_but_not_gain():
    records = [
        transition(0, 0.0, valid=True, cost=2.0),
        transition(1, 0.5, valid=True, cost=3.0),
        transition(2, 1.0, valid=False, cost=8.0),
    ]
    model = BayesianTransitionSurrogate(default_cost=4.0)
    model.fit(records)
    assert model.gain_model.training_transition_ids == ["t0", "t1"]
    assert model.cost_model.training_count == 3


def test_cost_transform_fallback_and_zero_safe_cost_acquisition():
    for cost in (0.0, 0.5, 10.0):
        assert cost_from_log(cost_to_log(cost)) == pytest.approx(cost)

    cost_model = BayesianCostModel(default_cost=7.0)
    cost_model.fit([])
    assert cost_model.predict([option(0.2)])[0] == pytest.approx(7.0)

    cost_model.fit([
        transition(0, 0.0, operator="improve", cost=2.0),
        transition(1, 0.5, operator="improve", cost=4.0),
        transition(2, 1.0, operator="fusion", cost=9.0),
    ])
    assert cost_model.fallback_cost("improve") == pytest.approx(3.0)
    assert cost_model.fallback_cost("evolution") == pytest.approx(4.0)

    candidate = option(0.2)
    prediction = PredictiveTransition(
        parent_node_id=candidate.parent_node_id,
        operator=candidate.operator,
        mean_gain=0.1,
        std_gain=0.2,
        predicted_total_cost=1e-12,
        p_valid=0.5,
    )
    score = score_predictions(
        [candidate],
        [prediction],
        acquisition_type="ei_per_cost",
        beta=1.5,
        use_validity_probability=True,
        cost_power=1.0,
    )[0]
    assert math.isfinite(score.acquisition_score)
