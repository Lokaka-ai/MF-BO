import pytest

from config import BOParentSelectorConfig, _load_cfg
from engine.parent_selectors.runtime import (
    Phase1SelectorRuntime,
    SUPPORTED_DECISION_SCOPES,
    SUPPORTED_SELECTOR_TYPES,
)
from conftest import make_runtime_config


def test_required_selector_modes_and_current_production_defaults():
    required = {
        "legacy_original",
        "progressive_mcgs",
        "parent_metric_greedy",
        "random_parent",
        "bayesian_gain_ucb",
        "bayesian_child_ucb",
        "reward_only_bayesian_ts",
        "bnvs_mean_greedy",
        "bnvs_ts",
        "bnvs_ts_local",
        "cost_aware_bayesian",
    }
    assert required <= SUPPORTED_SELECTOR_TYPES
    assert SUPPORTED_DECISION_SCOPES == {"root_only", "all_normal_steps"}
    cfg = _load_cfg(use_cli_args=False)
    assert cfg.parent_selector.type == "bnvs_ts"
    assert cfg.parent_selector.decision_scope == "all_normal_steps"
    assert cfg.bo_parent_selector.model == "conjugate_bayesian_linear"
    assert cfg.bo_parent_selector.min_valid_transitions == 0
    assert cfg.bo_parent_selector.min_total_transitions == 0
    assert cfg.bo_parent_selector.min_operator_observations == 0
    assert cfg.bo_parent_selector.acquisition.type == "thompson"
    assert cfg.bo_parent_selector.acquisition.thompson_temperature == 1.0
    assert cfg.bo_parent_selector.value_model.target == "incumbent_h1"
    assert cfg.bo_parent_selector.value_model.horizon == 1
    assert cfg.bo_parent_selector.value_model.feature_set == "exact10"
    assert cfg.bo_parent_selector.acquisition.use_validity_probability is False
    assert cfg.bo_parent_selector.acquisition.use_cost is False
    assert BOParentSelectorConfig().min_operator_observations == 0
    assert cfg.bo_parent_selector.fallback.type == "progressive_mcgs"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model", "silent-other-model"),
        ("fallback", "silent-other-fallback"),
        ("beta_schedule", "silent-other-schedule"),
    ],
)
def test_unsupported_model_controls_fail_loudly(tmp_path, field, value):
    cfg = make_runtime_config(tmp_path)
    if field == "model":
        cfg.bo_parent_selector.model = value
    elif field == "fallback":
        cfg.bo_parent_selector.fallback.type = value
    else:
        cfg.bo_parent_selector.acquisition.beta_schedule = value
    with pytest.raises(ValueError):
        Phase1SelectorRuntime(cfg)


def test_unsupported_decision_scope_fails_loudly(tmp_path):
    cfg = make_runtime_config(tmp_path)
    cfg.parent_selector.decision_scope = "silent-other-scope"
    with pytest.raises(ValueError):
        Phase1SelectorRuntime(cfg)
