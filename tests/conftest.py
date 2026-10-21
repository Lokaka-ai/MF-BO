from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from engine.search_node import Journal, SearchNode
from utils.metric import MetricValue, WorstMetricValue


@pytest.fixture(autouse=True)
def no_accidental_live_opencode(monkeypatch):
    monkeypatch.setenv("MLEVOLVE_OPENCODE_BIN", "/__mlevolve_test_requires_a_fake_opencode__")


@pytest.fixture
def autoresearch_cache(tmp_path, monkeypatch):
    """Provide manifest fixtures for tests that mock training and evaluation."""
    cache = tmp_path / "autoresearch-cache"
    files = []
    # These placeholders exercise manifest validation; no model loads them.
    for name, content in {
        "data/shard_00000.parquet": b"synthetic training fixture",
        "tokenizer/tokenizer.pkl": b"synthetic tokenizer fixture",
    }.items():
        path = cache / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        files.append({"path": name, "bytes": len(content),
                      "sha256": hashlib.sha256(content).hexdigest()})
    (cache / "manifest.json").write_text(json.dumps({"files": files}, sort_keys=True))
    monkeypatch.setenv("AUTORESEARCH_CACHE_DIR", str(cache))
    return cache


def ns(**kwargs):
    return SimpleNamespace(**kwargs)


def make_runtime_config(tmp_path: Path, selector_type: str = "bayesian_child_ucb"):
    acquisition = ns(
        type="child_ucb",
        beta=1.5,
        beta_schedule="fixed",
        use_validity_probability=False,
        use_cost=False,
        cost_power=1.0,
    )
    return ns(
        log_dir=tmp_path,
        parent_selector=ns(type=selector_type, decision_scope="root_only"),
        bo_parent_selector=ns(
            model="bayesian_ridge",
            min_valid_transitions=3,
            min_total_transitions=4,
            min_operator_observations=1,
            refit_every=1,
            acquisition=acquisition,
            cost=ns(default_total_wall_time_sec=10.0),
            fallback=ns(type="progressive_mcgs"),
            features=ns(
                include_mcgs_statistics=True,
                include_branch_history=True,
                include_reference_summary=True,
            ),
        ),
    )


def make_reward_runtime_config(tmp_path: Path):
    acquisition = ns(
        type="conservative_thompson",
        beta=1.5,
        beta_schedule="fixed",
        use_validity_probability=False,
        use_cost=False,
        cost_power=1.0,
        initial_gain_trust=0.10,
        max_gain_trust=0.25,
        full_gain_trust_valid_transitions=30,
        thompson_temperature=0.25,
        anneal_remaining_fraction=0.10,
    )
    return ns(
        log_dir=tmp_path,
        agent=ns(seed=0),
        parent_selector=ns(
            type="reward_only_bayesian_ts",
            decision_scope="all_normal_steps",
        ),
        bo_parent_selector=ns(
            model="conjugate_bayesian_linear",
            min_valid_transitions=10,
            min_total_transitions=10,
            min_operator_observations=0,
            refit_every=1,
            acquisition=acquisition,
            reward_model=ns(
                prior_precision=4.0,
                intercept_precision=0.25,
                prior_shape=2.0,
                prior_scale=1.0,
                uncertainty_multiplier=2.0,
                uncertainty_floor=0.10,
            ),
            cost=ns(default_total_wall_time_sec=10.0),
            fallback=ns(type="progressive_mcgs"),
            features=ns(
                include_mcgs_statistics=True,
                include_branch_history=True,
                include_reference_summary=True,
            ),
        ),
    )


def make_bnvs_runtime_config(tmp_path: Path, selector_type: str = "bnvs_ts"):
    acquisition = ns(
        type="thompson",
        beta=0.0,
        beta_schedule="fixed",
        use_validity_probability=False,
        use_cost=False,
        cost_power=1.0,
        initial_gain_trust=0.0,
        max_gain_trust=0.0,
        full_gain_trust_valid_transitions=0,
        thompson_temperature=1.0,
        anneal_remaining_fraction=0.0,
    )
    return ns(
        log_dir=tmp_path,
        agent=ns(seed=0),
        parent_selector=ns(type=selector_type, decision_scope="all_normal_steps"),
        bo_parent_selector=ns(
            model="conjugate_bayesian_linear",
            min_valid_transitions=0,
            min_total_transitions=0,
            min_operator_observations=0,
            refit_every=1,
            acquisition=acquisition,
            value_model=ns(
                target="incumbent_h1",
                horizon=1,
                feature_set="exact10",
                prior_precision=4.0,
                intercept_precision=0.25,
                prior_shape=2.0,
                prior_scale=1.0,
            ),
            reward_model=ns(
                prior_precision=4.0,
                intercept_precision=0.25,
                prior_shape=2.0,
                prior_scale=1.0,
                uncertainty_multiplier=2.0,
                uncertainty_floor=0.10,
            ),
            cost=ns(default_total_wall_time_sec=10.0),
            fallback=ns(type="progressive_mcgs"),
            features=ns(
                include_mcgs_statistics=True,
                include_branch_history=True,
                include_reference_summary=True,
            ),
        ),
    )


def make_agent(maximize: bool = True):
    scfg = ns(
        num_drafts=2,
        num_bugs=2,
        num_improves=3,
        topk_max_improves=10,
        max_debug_depth=5,
        back_debug_depth=2,
        metric_improvement_threshold=1e-4,
        max_improve_failure=3,
        branch_stagnation_threshold=2,
        topk_stagnation_threshold=3,
        stagnation_window=3,
        use_stagnation_detection=True,
        explore_switch_start=0.5,
        explore_switch_end=0.7,
        min_exploration_weight=0.2,
        topk_early_k=5,
        topk_early_max_per_branch=3,
        topk_late_k=3,
        topk_late_max_per_branch=2,
        force_backprop_late_threshold=0.8,
        force_backprop_late_prob=0.5,
        force_backprop_mid_threshold=0.4,
        force_backprop_mid_modulo=3,
        recent_best_window=3,
        fusion_min_time_hours=6,
        fusion_max_time_hours=10,
        fusion_min_successful_nodes=2,
        fusion_min_branches=2,
    )
    decay = ns(
        exploration_constant=1.414,
        lower_bound=0.5,
        alpha=0.01,
        phase_ratios=[0.3, 0.7],
    )
    acfg = ns(
        time_limit=1000,
        steps=50,
        use_diff_mode=True,
        use_stepwise_generation=True,
        use_evolution=True,
        use_fusion=False,
        use_aggregation=True,
        fusion_vs_evolution_prob=0.3,
        branch_fusion_trigger_prob=1.0,
        decay=decay,
    )
    root = SearchNode(
        id="root",
        parent=None,
        plan="root",
        code="",
        metric=WorstMetricValue(),
        stage="root",
    )
    journal = Journal()
    journal.append(root)
    agent = ns(
        cfg=ns(agent=acfg),
        acfg=acfg,
        scfg=scfg,
        virtual_root=root,
        journal=journal,
        metric_maximize=maximize,
        best_node=None,
        current_step=1,
        search_start_time=None,
        branch_all_nodes={},
        branch_successful_nodes={},
        branch_node_count={},
        current_node_list=[],
        top_candidates=[],
        fusion_draft_count=0,
        max_fusion_drafts=2,
        stagnation_threshold=3,
        use_stepwise_generation=True,
    )
    agent.is_root = lambda node: node.id == root.id
    return agent


def add_valid_node(
    agent,
    *,
    node_id: str,
    metric: float,
    parent=None,
    branch_id: int = 1,
    stage: str = "draft",
):
    parent = parent or agent.virtual_root
    node = SearchNode(
        id=node_id,
        parent=parent,
        plan=node_id,
        code=f"# {node_id}",
        metric=MetricValue(metric, maximize=agent.metric_maximize),
        stage=stage,
        is_buggy=False,
        is_valid=True,
        branch_id=branch_id,
        local_best_node=parent,
    )
    node.local_best_node = node
    agent.journal.append(node)
    agent.branch_all_nodes.setdefault(branch_id, []).append(node)
    agent.branch_successful_nodes.setdefault(branch_id, []).append(node)
    if agent.best_node is None or node.metric > agent.best_node.metric:
        agent.best_node = node
    agent.current_step = len(agent.journal)
    return node
