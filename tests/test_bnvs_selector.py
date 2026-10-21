from __future__ import annotations

import json
import math

import pytest

from engine.parent_selectors.base import ExpansionOption, OperatorPreview
from engine.parent_selectors.features import (
    BNVS_FEATURE_NAMES,
    BNVS_FEATURE_SCHEMA_VERSION,
    FEATURE_NAMES,
    vectorize_bnvs_features,
)
from engine.parent_selectors.runtime import Phase1SelectorRuntime
from engine.parent_selectors.selectors import BayesianNodeValueSelector
from engine.parent_selectors.surrogate import BayesianNodeValueSurrogate
from engine.parent_selectors.transition_store import TransitionRecord
from conftest import add_valid_node, make_agent, make_bnvs_runtime_config


def _features(parent: float, incumbent: float, operator: str = "improve"):
    features = {name: 0.0 for name in FEATURE_NAMES}
    features.update({
        "parent_utility": parent,
        "global_best_utility": incumbent,
        "parent_global_gap": incumbent - parent,
        "branch_best_utility": parent,
        "depth": 2.0,
        "visit_count": 3.0,
        "children_count": 1.0,
        "branch_stagnation": 2.0,
        "recent_gain_mean": 0.05,
        "recent_valid_rate": 0.75,
        "remaining_steps_fraction": 0.6,
        "elapsed_fraction": 0.3,
        "operator__evolution": float(operator == "evolution"),
        "operator__fusion": float(operator == "fusion"),
    })
    return features


def _record(
    index: int,
    *,
    parent: float,
    incumbent: float,
    child: float | None,
) -> TransitionRecord:
    valid = child is not None
    return TransitionRecord(
        transition_id=f"t{index}",
        decision_id=f"d{index}",
        parent_node_id=f"p{index}",
        child_node_id=f"c{index}",
        operator="improve",
        reference_node_ids=(),
        parent_metric_raw=parent,
        child_metric_raw=child,
        parent_utility=parent,
        child_utility=child,
        improvement=(child - parent) if child is not None else None,
        valid=valid,
        generation_wall_time_sec=1.0,
        execution_wall_time_sec=1.0,
        total_wall_time_sec=2.0,
        gpu_time_sec=None,
        reward=1.0 if valid else -1.0,
        selector_name="bnvs_ts",
        selection_features=_features(parent, incumbent),
        timestamp="2026-09-04T00:00:00Z",
    )


def _option(parent: float, incumbent: float, operator: str = "improve"):
    return ExpansionOption(
        parent_node_id=f"p-{parent}-{operator}",
        operator=operator,
        reference_node_ids=(),
        state_features=_features(parent, incumbent, operator),
        parent_node=None,
        preview=OperatorPreview(operator),
    )


def test_bnvs_schema_is_exact_ten_without_active_state():
    assert len(BNVS_FEATURE_NAMES) == 10
    assert "is_active_leaf" not in BNVS_FEATURE_NAMES
    assert "distance_to_active_squashed" not in BNVS_FEATURE_NAMES
    vector = vectorize_bnvs_features(_features(0.6, 0.8), utility_scale=0.1)
    assert len(vector) == 10
    assert all(math.isfinite(value) for value in vector)


def test_bnvs_h1_uses_pre_action_incumbent_and_invalid_minus_one():
    h1 = BayesianNodeValueSurrogate(target_type="incumbent_h1")
    h1.utility_scale = 0.1
    local = BayesianNodeValueSurrogate(target_type="local_gain")
    local.utility_scale = 0.1
    valid = _record(0, parent=0.2, incumbent=0.8, child=0.5)
    invalid = _record(1, parent=0.7, incumbent=0.8, child=None)

    assert h1._target(valid) == pytest.approx(-1.0)
    assert local._target(valid) == pytest.approx(1.0)
    assert h1._target(invalid) == pytest.approx(-1.0)
    assert local._target(invalid) == pytest.approx(-1.0)


def test_bnvs_prior_is_fitted_and_selectable_with_zero_observations():
    options = [_option(0.6, 0.8), _option(0.75, 0.8, "evolution")]
    model = BayesianNodeValueSurrogate()
    model.fit([])
    means, stds = model.predict_values(options)

    assert model.training_transition_ids == []
    assert model.model_version.startswith("bnvs-h1-0-")
    assert means == pytest.approx([0.0, 0.0])
    assert all(value > 0.0 for value in stds)
    sampled = model.sample_values(options, seed=3)
    assert all(math.isfinite(value) for value in sampled)


def test_mean_greedy_and_ts_are_distinct_selection_rules():
    options = [_option(0.6, 0.8), _option(0.75, 0.8, "evolution")]
    mean_model = BayesianNodeValueSurrogate()
    mean_selector = BayesianNodeValueSelector(
        name="bnvs_mean_greedy",
        surrogate=mean_model,
        records_provider=list,
        selection_rule="posterior_mean",
        refit_every=1,
        thompson_temperature=1.0,
        random_seed=0,
    )
    mean_result = mean_selector.select(None, options)
    assert mean_result.selected_option is options[0]
    assert all(score.sampled_value is None for score in mean_result.candidate_scores)

    ts_model = BayesianNodeValueSurrogate()
    ts_selector = BayesianNodeValueSelector(
        name="bnvs_ts",
        surrogate=ts_model,
        records_provider=list,
        selection_rule="thompson",
        refit_every=1,
        thompson_temperature=1.0,
        random_seed=0,
    )
    ts_result = ts_selector.select(None, options)
    assert ts_result.decision.fallback_used is False
    assert all(score.sampled_value is not None for score in ts_result.candidate_scores)


def test_default_bnvs_runtime_uses_prior_without_cold_start(tmp_path):
    runtime = Phase1SelectorRuntime(make_bnvs_runtime_config(tmp_path))
    agent = make_agent()
    add_valid_node(agent, node_id="first", metric=0.4, branch_id=1)
    add_valid_node(agent, node_id="second", metric=0.8, branch_id=2)

    result = runtime.select_global_parent(agent)

    assert result.decision.selector_name == "bnvs_ts"
    assert result.decision.fallback_used is False
    assert result.decision.model_version.startswith("bnvs-h1-0-")
    snapshots = [
        json.loads(line)
        for line in (
            tmp_path / "phase1" / "selection_snapshots.jsonl"
        ).read_text().splitlines()
    ]
    assert snapshots
    assert all(
        item["feature_schema_version"] == BNVS_FEATURE_SCHEMA_VERSION
        for item in snapshots
    )
    assert all(item["posterior_mean_value"] == pytest.approx(0.0) for item in snapshots)
    assert all(item["sampled_value"] is not None for item in snapshots)


@pytest.mark.parametrize(
    ("selector_type", "target_type", "selection_rule"),
    [
        ("bnvs_mean_greedy", "incumbent_h1", "posterior_mean"),
        ("bnvs_ts", "incumbent_h1", "thompson"),
        ("bnvs_ts_local", "local_gain", "thompson"),
    ],
)
def test_bnvs_runtime_baseline_modes(
    tmp_path,
    selector_type,
    target_type,
    selection_rule,
):
    runtime = Phase1SelectorRuntime(
        make_bnvs_runtime_config(tmp_path, selector_type)
    )
    selector = runtime.selectors[selector_type]

    assert runtime.surrogate.target_type == target_type
    assert selector.selection_rule == selection_rule
