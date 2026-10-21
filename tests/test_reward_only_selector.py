import json
import math

import numpy as np
import pytest

from engine.parent_selectors.base import ExpansionOption, OperatorPreview
from engine.parent_selectors.features import FEATURE_NAMES, REWARD_FEATURE_NAMES
from engine.parent_selectors.runtime import Phase1SelectorRuntime
from engine.parent_selectors.selectors import RewardOnlyThompsonSelector
from engine.parent_selectors.surrogate import RewardOnlyBayesianLinearSurrogate
from engine.parent_selectors.transition_store import TransitionRecord
from conftest import add_valid_node, make_agent, make_reward_runtime_config


def _features(value: float, operator: str = "improve") -> dict[str, float]:
    features = {name: 0.0 for name in FEATURE_NAMES}
    features.update({
        "parent_utility": value,
        "global_best_utility": 1.0,
        "parent_global_gap": 1.0 - value,
        "branch_best_utility": value,
        "depth": 2.0 + value,
        "visit_count": 2.0,
        "mcgs_q": value,
        "last_improvement": 0.1 * value,
        "recent_gain_mean": 0.1 * value,
        "recent_gain_std": 0.02,
        "operator__improve": float(operator == "improve"),
        "operator__evolution": float(operator == "evolution"),
        "operator__fusion": float(operator == "fusion"),
        "remaining_steps_fraction": 0.5,
        "elapsed_fraction": 0.2,
        "parent_rank_percentile": value,
    })
    return features


def _record(index: int, value: float, *, valid: bool = True) -> TransitionRecord:
    improvement = 0.1 + 0.2 * value
    return TransitionRecord(
        transition_id=f"t{index}",
        decision_id=f"d{index}",
        parent_node_id=f"p{index}",
        child_node_id=f"c{index}",
        operator="improve",
        reference_node_ids=(),
        parent_metric_raw=value,
        child_metric_raw=value + improvement if valid else None,
        parent_utility=value,
        child_utility=value + improvement if valid else None,
        improvement=improvement if valid else None,
        valid=valid,
        generation_wall_time_sec=999.0 if not valid else 1.0,
        execution_wall_time_sec=999.0 if not valid else 1.0,
        total_wall_time_sec=1998.0 if not valid else 2.0,
        gpu_time_sec=None,
        reward=1.0 if valid else -1.0,
        selector_name="progressive_mcgs",
        selection_features=_features(value),
        timestamp="2026-09-04T00:00:00Z",
    )


def _option(value: float, operator: str = "improve") -> ExpansionOption:
    return ExpansionOption(
        parent_node_id=f"option-{value}-{operator}",
        operator=operator,
        reference_node_ids=(),
        state_features=_features(value, operator),
        parent_node=None,
        preview=OperatorPreview(operator),
    )


def test_reward_only_surrogate_has_compact_schema_and_no_cost_or_validity_heads():
    assert len(REWARD_FEATURE_NAMES) == 15
    records = [_record(index, value) for index, value in enumerate(np.linspace(0.1, 0.9, 12))]
    model = RewardOnlyBayesianLinearSurrogate()
    model.fit(records)
    predictions = model.predict([_option(0.25), _option(0.75)])

    assert not hasattr(model, "cost_model")
    assert not hasattr(model, "validity_model")
    assert model.training_transition_ids == [f"t{index}" for index in range(12)]
    assert all(math.isfinite(item.mean_gain) for item in predictions)
    assert all(item.std_gain > 0 for item in predictions)
    assert all(item.predicted_total_cost is None for item in predictions)
    assert all(item.p_valid is None for item in predictions)


def test_invalid_transition_does_not_change_reward_only_posterior():
    records = [_record(index, value) for index, value in enumerate(np.linspace(0.1, 0.9, 12))]
    option = _option(0.5)
    clean = RewardOnlyBayesianLinearSurrogate()
    clean.fit(records)
    contaminated = RewardOnlyBayesianLinearSurrogate()
    contaminated.fit([*records, _record(99, 100.0, valid=False)])

    assert contaminated.training_transition_ids == clean.training_transition_ids
    assert contaminated.predict([option])[0].mean_gain == pytest.approx(
        clean.predict([option])[0].mean_gain
    )
    assert contaminated.predict([option])[0].std_gain == pytest.approx(
        clean.predict([option])[0].std_gain
    )


def test_zero_temperature_thompson_sample_equals_posterior_mean():
    records = [_record(index, value) for index, value in enumerate(np.linspace(0.1, 0.9, 12))]
    options = [_option(0.2), _option(0.8)]
    model = RewardOnlyBayesianLinearSurrogate()
    model.fit(records)
    means = np.asarray([item.mean_gain for item in model.predict(options)])
    samples = model.sample_gains(options, seed=123, temperature=0.0)
    assert samples == pytest.approx(means)


def test_latent_gain_std_excludes_predictive_noise_and_calibration_multiplier():
    records = [_record(index, value) for index, value in enumerate(np.linspace(0.1, 0.9, 12))]
    options = [_option(0.2), _option(0.8)]
    model = RewardOnlyBayesianLinearSurrogate()
    model.fit(records)

    latent_stds = model.latent_std_gains(options)
    predictive_stds = np.asarray([
        item.std_gain for item in model.predict(options)
    ])

    assert np.all(np.isfinite(latent_stds))
    assert np.all(latent_stds >= 0.0)
    assert np.all(predictive_stds >= model.uncertainty_multiplier * latent_stds)


def test_gain_trust_and_temperature_follow_small_data_schedule():
    selector = RewardOnlyThompsonSelector(
        surrogate=RewardOnlyBayesianLinearSurrogate(),
        records_provider=list,
        min_valid_transitions=10,
        min_total_transitions=10,
        refit_every=1,
        initial_gain_trust=0.10,
        max_gain_trust=0.25,
        full_gain_trust_valid_transitions=30,
        thompson_temperature=0.25,
        anneal_remaining_fraction=0.10,
        random_seed=0,
    )
    assert not hasattr(selector, "_mcgs_ranks")
    assert selector._gain_trust(10) == pytest.approx(0.10)
    assert selector._gain_trust(20) == pytest.approx(0.175)
    assert selector._gain_trust(30) == pytest.approx(0.25)
    assert selector._gain_trust(100) == pytest.approx(0.25)

    early = _option(0.5)
    near_step_end = _option(0.5)
    near_step_end.state_features["remaining_steps_fraction"] = 0.05
    near_time_end = _option(0.5)
    near_time_end.state_features["elapsed_fraction"] = 0.98
    assert selector._temperature([early]) == pytest.approx(1.0)
    assert selector._temperature([near_step_end]) == pytest.approx(0.5)
    assert selector._temperature([near_time_end]) == pytest.approx(0.2)


def test_runtime_falls_back_until_ten_valid_then_logs_reward_only_fields(tmp_path):
    cfg = make_reward_runtime_config(tmp_path)
    assert not hasattr(cfg.bo_parent_selector.acquisition, "gain_clip_scale")
    runtime = Phase1SelectorRuntime(cfg)
    assert not hasattr(runtime.surrogate, "cost_model")
    assert not hasattr(runtime.surrogate, "validity_model")
    agent = make_agent()
    first = add_valid_node(agent, node_id="first", metric=0.4, branch_id=1)
    second = add_valid_node(agent, node_id="second", metric=0.8, branch_id=2)
    first.visits = 2
    first.total_reward = 0.5
    second.visits = 3
    second.total_reward = 2.0
    agent.virtual_root.visits = 5

    for index in range(9):
        runtime.transition_store.append(_record(index, 0.1 + index * 0.05))
    cold = runtime.select_global_parent(agent)
    assert cold.decision.fallback_used is True
    assert cold.decision.fallback_reason == "total_transitions=9<10"

    runtime.transition_store.append(_record(9, 0.55))
    selected = runtime.select_global_parent(agent)
    assert selected.decision.fallback_used is False
    assert selected.decision.model_version.startswith("reward-only-10-")
    assert all(score.predicted_cost is None for score in selected.result.candidate_scores)
    assert all(score.p_valid is None for score in selected.result.candidate_scores)
    assert all(score.thompson_sample is not None for score in selected.result.candidate_scores)
    assert all(not hasattr(score, "mcgs_rank") for score in selected.result.candidate_scores)
    assert all(not hasattr(score, "bo_weight") for score in selected.result.candidate_scores)
    assert all(
        score.thompson_temperature == pytest.approx(0.25)
        for score in selected.result.candidate_scores
    )
    assert all(score.gain_trust == pytest.approx(0.10) for score in selected.result.candidate_scores)
    assert all(
        score.acquisition_score
        == pytest.approx(score.parent_utility + score.gain_trust * score.sampled_gain)
        for score in selected.result.candidate_scores
    )

    snapshots = [
        json.loads(line)
        for line in (tmp_path / "phase1" / "selection_snapshots.jsonl").read_text().splitlines()
    ]
    learned = [item for item in snapshots if item["decision_id"] == selected.decision.decision_id]
    assert learned
    assert all(item["feature_schema_version"] == "reward-only-local-v1" for item in learned)
    assert all(item["predicted_cost"] is None for item in learned)
    assert all(item["p_valid"] is None for item in learned)
    assert all("mcgs_rank" not in item for item in learned)
    assert all("bo_rank" not in item for item in learned)
    assert all("bo_weight" not in item for item in learned)
    assert all(item["gain_trust"] == pytest.approx(0.10) for item in learned)
    assert all(item["sampled_gain"] is not None for item in learned)

    model_fits = [
        json.loads(line)
        for line in (tmp_path / "phase1" / "model_fits.jsonl").read_text().splitlines()
    ]
    assert model_fits[-1]["model_type"] == "conjugate_bayesian_linear_reward_only"
    assert set(model_fits[-1]["fit_diagnostics"]) == {"reward"}
