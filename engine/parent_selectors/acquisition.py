"""Acquisition functions for Bayesian parent/operator selection."""

from __future__ import annotations

import math
from typing import Sequence

from engine.parent_selectors.base import (
    CandidateScore,
    ExpansionOption,
    PredictiveTransition,
)


def _normal_pdf(value: float) -> float:
    return math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def expected_improvement(mean: float, std: float, incumbent: float) -> float:
    std = max(float(std), 1e-12)
    advantage = float(mean) - float(incumbent)
    z = advantage / std
    return max(advantage * _normal_cdf(z) + std * _normal_pdf(z), 0.0)


def score_predictions(
    options: Sequence[ExpansionOption],
    predictions: Sequence[PredictiveTransition],
    *,
    acquisition_type: str,
    beta: float,
    use_validity_probability: bool,
    cost_power: float,
) -> tuple[CandidateScore, ...]:
    if len(options) != len(predictions):
        raise ValueError("option/prediction length mismatch")
    incumbent = max(
        (float(option.state_features.get("global_best_utility", option.parent_utility)) for option in options),
        default=0.0,
    )
    scores: list[CandidateScore] = []
    for option, prediction in zip(options, predictions):
        if option.parent_node_id != prediction.parent_node_id or option.operator != prediction.operator:
            raise ValueError("prediction does not correspond to expansion option")
        child_mean = option.parent_utility + prediction.mean_gain
        child_ucb = child_mean + beta * prediction.std_gain

        if acquisition_type == "child_ucb":
            acquisition = child_ucb
            if use_validity_probability:
                acquisition *= prediction.p_valid
        elif acquisition_type == "gain_ucb":
            acquisition = prediction.mean_gain + beta * prediction.std_gain
            if use_validity_probability:
                acquisition *= prediction.p_valid
        elif acquisition_type == "posterior_mean":
            acquisition = child_mean
            if use_validity_probability:
                acquisition *= prediction.p_valid
        elif acquisition_type == "ei_per_cost":
            acquisition = expected_improvement(
                child_mean, prediction.std_gain, incumbent
            )
            if use_validity_probability:
                acquisition *= prediction.p_valid
            acquisition /= max(prediction.predicted_total_cost, 1e-6) ** max(cost_power, 0.0)
        else:
            raise ValueError(f"unsupported acquisition type: {acquisition_type}")

        if not math.isfinite(acquisition):
            raise ValueError("acquisition returned a non-finite score")
        scores.append(CandidateScore(
            parent_node_id=option.parent_node_id,
            operator=option.operator,
            reference_node_ids=option.reference_node_ids,
            parent_utility=option.parent_utility,
            mean_gain=prediction.mean_gain,
            std_gain=prediction.std_gain,
            predicted_child_ucb=child_ucb,
            predicted_cost=prediction.predicted_total_cost,
            p_valid=prediction.p_valid,
            acquisition_score=float(acquisition),
        ))
    return tuple(scores)

