"""Auditable selection-time features for parent/operator expansion options."""

from __future__ import annotations

import math
from statistics import mean, median, pstdev
import time
from typing import Iterable

from engine.parent_selectors.base import OperatorPreview
from engine.parent_selectors.transition_store import TransitionStore


FEATURE_SCHEMA_VERSION = "phase1-parent-operator-v1"
REWARD_ONLY_FEATURE_SCHEMA_VERSION = "reward-only-local-v1"
BNVS_FEATURE_SCHEMA_VERSION = "bnvs-h1-exact10-v1"
BNVS_ROOT_FEATURE_SCHEMA_VERSION = "bnvs-h1-exact12-root-draft-v1"
RECENT_WINDOW = 5

STAGES = ("root", "draft", "improve", "evolution", "fusion", "debug", "fusion_draft")
OPERATORS = ("improve", "evolution", "fusion", "debug", "draft", "aggregation")

FEATURE_NAMES = (
    "parent_utility",
    "parent_metric_missing",
    "global_best_utility",
    "parent_global_gap",
    "branch_best_utility",
    "parent_branch_gap",
    "depth",
    "visit_count",
    "mcgs_q",
    "children_count",
    "branch_age",
    "branch_stagnation",
    "recent_gain_mean",
    "recent_gain_std",
    "recent_gain_max",
    "recent_valid_rate",
    "last_improvement",
    "is_local_best",
    "is_global_top_k",
    "improve_count",
    "failure_depth",
    "code_length_log",
    "parent_last_exec_cost_log",
    "elapsed_fraction",
    "remaining_steps_fraction",
    "use_diff_generation",
    "use_stepwise_generation",
    "reference_count",
    "reference_best_utility",
    "primary_reference_utility_gap",
    "reference_branch_diversity",
    "reference_depth_mean",
    "reference_global_top_k_rate",
    "historical_operator_generation_log",
    "historical_operator_execution_log",
    "historical_operator_cost_count",
) + tuple(f"stage__{stage}" for stage in STAGES) + tuple(
    f"operator__{operator}" for operator in OPERATORS
)

# Compact, selection-time-only inputs used by the local reward model.  The
# ``*_scaled`` values are derived at fit/predict time from the raw audit
# features and the reward scale of the current run.
REWARD_FEATURE_NAMES = (
    "parent_rank_percentile",
    "parent_global_gap_scaled",
    "parent_branch_gap_scaled",
    "last_improvement_scaled",
    "recent_gain_mean_scaled",
    "recent_gain_std_scaled",
    "branch_stagnation_log",
    "depth_squashed",
    "visit_count_log",
    "mcgs_q",
    "primary_reference_utility_gap_scaled",
    "reference_missing",
    "operator__improve",
    "operator__evolution",
    "operator__fusion",
)

# Exact decision-time state used by the default BNVS value model.  Unlike the
# active-aware offline ablation, every field here is available in the runtime
# without reconstructing an active leaf from future log structure.
BNVS_FEATURE_NAMES = (
    "incumbent_gap_scaled",
    "branch_backed_gap_scaled",
    "expansion_count_log",
    "depth_squashed",
    "stagnation_log",
    "recent_gain_scaled",
    "recent_valid_rate",
    "remaining_budget_fraction",
    "operator__evolution",
    "operator__fusion",
)

BNVS_ROOT_FEATURE_NAMES = tuple(
    name for name in BNVS_FEATURE_NAMES if name != "remaining_budget_fraction"
) + (
    "operator__draft",
    "draft_remaining_budget_fraction",
    "draft_global_stagnation_log",
)


def utility_from_metric(value: float | int | None, maximize: bool) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return numeric if maximize else -numeric


def continuous_improvement(
    parent_metric: float | int | None,
    child_metric: float | int | None,
    maximize: bool,
) -> float | None:
    parent_utility = utility_from_metric(parent_metric, maximize)
    child_utility = utility_from_metric(child_metric, maximize)
    if parent_utility is None or child_utility is None:
        return None
    return child_utility - parent_utility


def _metric_value(node) -> float | None:
    metric = getattr(node, "metric", None)
    value = getattr(metric, "value", None)
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _node_depth(node) -> int:
    depth = 0
    current = getattr(node, "parent", None)
    while current is not None:
        depth += 1
        current = getattr(current, "parent", None)
    return depth


def _valid_metric_nodes(nodes: Iterable) -> list:
    return [
        node
        for node in nodes
        if getattr(node, "is_buggy", None) is False
        and getattr(node, "is_valid", None) is not False
        and _metric_value(node) is not None
    ]


def _top_k_ids(agent, k: int = 5) -> set[str]:
    nodes = _valid_metric_nodes(getattr(agent.journal, "nodes", ()))
    nodes.sort(key=lambda node: _metric_value(node), reverse=bool(agent.metric_maximize))
    return {node.id for node in nodes[:k]}


def _utility_percentile(agent, node, parent_utility: float | None) -> float:
    """Return a scale-free [0, 1] rank using only the current journal."""
    if parent_utility is None:
        return 0.0
    utilities = [
        utility_from_metric(_metric_value(item), agent.metric_maximize)
        for item in _valid_metric_nodes(getattr(agent.journal, "nodes", ()))
    ]
    values = sorted(value for value in utilities if value is not None)
    if len(values) <= 1:
        return 1.0
    matching_positions = [
        index for index, value in enumerate(values)
        if math.isclose(value, parent_utility, rel_tol=1e-12, abs_tol=1e-12)
    ]
    if matching_positions:
        position = mean(matching_positions)
    else:
        position = sum(value < parent_utility for value in values)
    return float(min(max(position / (len(values) - 1), 0.0), 1.0))


def _branch_nodes(agent, node) -> list:
    branch_id = getattr(node, "branch_id", None)
    return list(getattr(agent, "branch_all_nodes", {}).get(branch_id, ()))


def _branch_stagnation(agent, node) -> float:
    nodes = _valid_metric_nodes(_branch_nodes(agent, node))
    if not nodes:
        return 0.0
    best_index = max(
        range(len(nodes)),
        key=lambda idx: utility_from_metric(_metric_value(nodes[idx]), agent.metric_maximize),
    )
    return float(len(nodes) - best_index - 1)


def _recent_child_statistics(node, maximize: bool) -> tuple[float, float, float, float]:
    children = sorted(
        list(getattr(node, "children", ())),
        key=lambda child: (getattr(child, "step", -1) or -1, getattr(child, "ctime", 0.0)),
    )[-RECENT_WINDOW:]
    if not children:
        return 0.0, 0.0, 0.0, 0.0

    parent_metric = _metric_value(node)
    gains: list[float] = []
    valid_count = 0
    for child in children:
        valid = (
            getattr(child, "is_buggy", None) is False
            and getattr(child, "is_valid", None) is not False
            and _metric_value(child) is not None
        )
        if valid:
            valid_count += 1
            gain = continuous_improvement(parent_metric, _metric_value(child), maximize)
            if gain is not None:
                gains.append(gain)
    gain_mean = mean(gains) if gains else 0.0
    gain_std = pstdev(gains) if len(gains) > 1 else 0.0
    gain_max = max(gains) if gains else 0.0
    return gain_mean, gain_std, gain_max, valid_count / len(children)


def _historical_cost_features(
    transition_store: TransitionStore,
    operator: str,
) -> tuple[float, float, float]:
    records = [record for record in transition_store.records() if record.operator == operator]
    if not records:
        return 0.0, 0.0, 0.0
    generation = mean(max(record.generation_wall_time_sec, 0.0) for record in records)
    execution = mean(max(record.execution_wall_time_sec, 0.0) for record in records)
    return math.log1p(generation), math.log1p(execution), float(len(records))


def extract_selection_features(
    agent,
    parent_node,
    preview: OperatorPreview,
    transition_store: TransitionStore,
    feature_config=None,
) -> dict[str, float]:
    """Extract only information available before the child is generated."""
    maximize = bool(agent.metric_maximize)
    parent_metric = _metric_value(parent_node)
    parent_utility_value = utility_from_metric(parent_metric, maximize)
    parent_utility = parent_utility_value if parent_utility_value is not None else 0.0

    best_node = getattr(agent, "best_node", None)
    global_best_value = utility_from_metric(_metric_value(best_node), maximize)
    global_best = global_best_value if global_best_value is not None else parent_utility

    branch_valid = _valid_metric_nodes(_branch_nodes(agent, parent_node))
    branch_utilities = [
        utility_from_metric(_metric_value(node), maximize) for node in branch_valid
    ]
    branch_utilities = [value for value in branch_utilities if value is not None]
    branch_best = max(branch_utilities) if branch_utilities else parent_utility

    recent_mean, recent_std, recent_max, recent_valid_rate = _recent_child_statistics(
        parent_node, maximize
    )

    parent_parent = getattr(parent_node, "parent", None)
    last_improvement = continuous_improvement(
        _metric_value(parent_parent), parent_metric, maximize
    )

    branch_nodes = _branch_nodes(agent, parent_node)
    branch_steps = [
        getattr(node, "step", None)
        for node in branch_nodes
        if getattr(node, "step", None) is not None
    ]
    current_step = max(len(getattr(agent.journal, "nodes", ())) - 1, 0)
    branch_age = current_step - min(branch_steps) if branch_steps else 0

    top_k_ids = _top_k_ids(agent)
    reference_by_id = {
        node.id: node for node in getattr(agent.journal, "nodes", ())
    }
    references = [
        reference_by_id[node_id]
        for node_id in preview.reference_node_ids
        if node_id in reference_by_id
    ]
    reference_utilities = [
        utility_from_metric(_metric_value(node), maximize) for node in references
    ]
    reference_utilities = [value for value in reference_utilities if value is not None]
    reference_best = max(reference_utilities) if reference_utilities else 0.0
    reference_depth_mean = mean(_node_depth(node) for node in references) if references else 0.0
    reference_branches = {
        getattr(node, "branch_id", None) for node in references
        if getattr(node, "branch_id", None) is not None
    }
    reference_top_rate = (
        sum(node.id in top_k_ids for node in references) / len(references)
        if references else 0.0
    )

    historical_generation, historical_execution, historical_count = _historical_cost_features(
        transition_store, preview.operator
    )
    parent_record = transition_store.by_child_id(parent_node.id)
    parent_last_cost = parent_record.total_wall_time_sec if parent_record else (
        getattr(parent_node, "exec_time", 0.0) or 0.0
    )

    elapsed = max(0.0, float(time.time() - getattr(agent, "search_start_time", 0.0))) \
        if getattr(agent, "search_start_time", None) else 0.0
    time_limit = max(float(getattr(agent.acfg, "time_limit", 1.0)), 1.0)
    total_steps = max(int(getattr(agent.acfg, "steps", 1)), 1)

    features: dict[str, float] = {
        "parent_utility": parent_utility,
        "parent_metric_missing": float(parent_utility_value is None),
        "global_best_utility": global_best,
        "parent_global_gap": global_best - parent_utility,
        "branch_best_utility": branch_best,
        "parent_branch_gap": branch_best - parent_utility,
        "depth": float(_node_depth(parent_node)),
        "visit_count": float(getattr(parent_node, "visits", 0) or 0),
        "mcgs_q": float(getattr(parent_node, "total_reward", 0.0) or 0.0)
        / max(float(getattr(parent_node, "visits", 0) or 0), 1.0),
        "children_count": float(len(getattr(parent_node, "children", ()))),
        "branch_age": float(max(branch_age, 0)),
        "branch_stagnation": _branch_stagnation(agent, parent_node),
        "recent_gain_mean": float(recent_mean),
        "recent_gain_std": float(recent_std),
        "recent_gain_max": float(recent_max),
        "recent_valid_rate": float(recent_valid_rate),
        "last_improvement": float(last_improvement or 0.0),
        "is_local_best": float(
            getattr(getattr(parent_node, "local_best_node", None), "id", None) == parent_node.id
        ),
        "is_global_top_k": float(parent_node.id in top_k_ids),
        "improve_count": float(
            sum(getattr(node, "stage", None) == "improve" for node in branch_nodes)
        ),
        "failure_depth": float(getattr(parent_node, "improve_failure_depth", 0) or 0),
        "code_length_log": math.log1p(len(getattr(parent_node, "code", "") or "")),
        "parent_last_exec_cost_log": math.log1p(max(float(parent_last_cost), 0.0)),
        "elapsed_fraction": min(elapsed / time_limit, 1.0),
        "remaining_steps_fraction": max(total_steps - current_step, 0) / total_steps,
        "use_diff_generation": float(bool(getattr(agent.acfg, "use_diff_mode", False))),
        "use_stepwise_generation": float(bool(getattr(agent, "use_stepwise_generation", False))),
        "reference_count": float(len(references)),
        "reference_best_utility": float(reference_best),
        "primary_reference_utility_gap": float(parent_utility - reference_best) if references else 0.0,
        "reference_branch_diversity": float(len(reference_branches)),
        "reference_depth_mean": float(reference_depth_mean),
        "reference_global_top_k_rate": float(reference_top_rate),
        "historical_operator_generation_log": float(historical_generation),
        "historical_operator_execution_log": float(historical_execution),
        "historical_operator_cost_count": float(historical_count),
    }
    stage = str(getattr(parent_node, "stage", "root"))
    for stage_name in STAGES:
        features[f"stage__{stage_name}"] = float(stage == stage_name)
    for operator_name in OPERATORS:
        features[f"operator__{operator_name}"] = float(preview.operator == operator_name)

    if feature_config is not None:
        if not bool(getattr(feature_config, "include_mcgs_statistics", True)):
            features["visit_count"] = 0.0
            features["mcgs_q"] = 0.0
        if not bool(getattr(feature_config, "include_branch_history", True)):
            for name in (
                "branch_age",
                "branch_stagnation",
                "recent_gain_mean",
                "recent_gain_std",
                "recent_gain_max",
                "recent_valid_rate",
                "last_improvement",
                "is_local_best",
                "improve_count",
                "failure_depth",
            ):
                features[name] = 0.0
        if not bool(getattr(feature_config, "include_reference_summary", True)):
            for name in (
                "reference_count",
                "reference_best_utility",
                "primary_reference_utility_gap",
                "reference_branch_diversity",
                "reference_depth_mean",
                "reference_global_top_k_rate",
            ):
                features[name] = 0.0

    result = {name: float(features[name]) for name in FEATURE_NAMES}
    result["parent_rank_percentile"] = _utility_percentile(
        agent,
        parent_node,
        parent_utility_value,
    )
    return result


def vectorize_features(features: dict[str, float]) -> list[float]:
    values = [float(features[name]) for name in FEATURE_NAMES]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("selection feature vector contains a non-finite value")
    return values


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = min(max(probability, 0.0), 1.0) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def reward_scale_from_records(records: Iterable) -> float:
    """Robust, run-local scale computed only from already observed rewards."""
    utilities: list[float] = []
    absolute_gains: list[float] = []
    for record in records:
        if not getattr(record, "valid", False):
            continue
        parent_utility = getattr(record, "parent_utility", None)
        child_utility = getattr(record, "child_utility", None)
        improvement = getattr(record, "improvement", None)
        for value in (parent_utility, child_utility):
            if value is not None and math.isfinite(float(value)):
                utilities.append(float(value))
        if improvement is not None and math.isfinite(float(improvement)):
            absolute_gains.append(abs(float(improvement)))

    interquartile_range = (
        _quantile(utilities, 0.75) - _quantile(utilities, 0.25)
        if len(utilities) >= 2 else 0.0
    )
    typical_gain = median(absolute_gains) if absolute_gains else 0.0
    return float(max(interquartile_range, typical_gain, 1e-6))


def vectorize_reward_features(
    features: dict[str, float],
    *,
    utility_scale: float,
) -> list[float]:
    """Convert the audit snapshot into the compact reward-only model vector."""
    scale = max(float(utility_scale), 1e-12)

    def scaled(name: str) -> float:
        value = float(features.get(name, 0.0)) / scale
        return min(max(value, -10.0), 10.0)

    global_gap_scaled = scaled("parent_global_gap")
    percentile = features.get("parent_rank_percentile")
    if percentile is None:
        # Backward-compatible approximation for v1 transition logs.
        percentile = 1.0 / (1.0 + max(global_gap_scaled, 0.0))

    depth = max(float(features.get("depth", 0.0)), 0.0)
    vector = [
        min(max(float(percentile), 0.0), 1.0),
        global_gap_scaled,
        scaled("parent_branch_gap"),
        scaled("last_improvement"),
        scaled("recent_gain_mean"),
        min(max(float(features.get("recent_gain_std", 0.0)) / scale, 0.0), 10.0),
        math.log1p(max(float(features.get("branch_stagnation", 0.0)), 0.0)),
        depth / (depth + 5.0),
        math.log1p(max(float(features.get("visit_count", 0.0)), 0.0)),
        min(max(float(features.get("mcgs_q", 0.0)), -10.0), 10.0),
        scaled("primary_reference_utility_gap"),
        float(float(features.get("reference_count", 0.0)) <= 0.0),
        float(features.get("operator__improve", 0.0)),
        float(features.get("operator__evolution", 0.0)),
        float(features.get("operator__fusion", 0.0)),
    ]
    if len(vector) != len(REWARD_FEATURE_NAMES):
        raise AssertionError("reward feature schema and vector length differ")
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("reward feature vector contains a non-finite value")
    return vector


def vectorize_bnvs_features(
    features: dict[str, float],
    *,
    utility_scale: float,
) -> list[float]:
    """Return the exact 10-dimensional BNVS decision-time context."""
    scale = max(float(utility_scale), 1e-12)

    def scaled(value: float) -> float:
        return min(max(float(value) / scale, -10.0), 10.0)

    global_best = float(features.get("global_best_utility", 0.0) or 0.0)
    branch_best = float(features.get("branch_best_utility", global_best) or 0.0)
    depth = max(float(features.get("depth", 0.0) or 0.0), 0.0)
    visits = max(float(features.get("visit_count", 0.0) or 0.0), 0.0)
    children = max(float(features.get("children_count", 0.0) or 0.0), 0.0)
    remaining_steps = min(max(
        float(features.get("remaining_steps_fraction", 1.0) or 0.0), 0.0
    ), 1.0)
    remaining_time = min(max(
        1.0 - float(features.get("elapsed_fraction", 0.0) or 0.0), 0.0
    ), 1.0)
    vector = [
        scaled(float(features.get("parent_global_gap", 0.0) or 0.0)),
        scaled(global_best - branch_best),
        math.log1p(visits + children),
        depth / (depth + 5.0),
        math.log1p(max(
            float(features.get("branch_stagnation", 0.0) or 0.0), 0.0
        )),
        scaled(float(features.get("recent_gain_mean", 0.0) or 0.0)),
        min(max(
            float(features.get("recent_valid_rate", 0.0) or 0.0), 0.0
        ), 1.0),
        min(remaining_steps, remaining_time),
        float(features.get("operator__evolution", 0.0) or 0.0),
        float(features.get("operator__fusion", 0.0) or 0.0),
    ]
    if len(vector) != len(BNVS_FEATURE_NAMES):
        raise AssertionError("BNVS feature schema and vector length differ")
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("BNVS feature vector contains a non-finite value")
    return vector


def vectorize_bnvs_root_features(features, *, utility_scale: float) -> list[float]:
    base = vectorize_bnvs_features(features, utility_scale=utility_scale)
    draft = float(features.get("operator__draft", 0.0))
    vector = base[:7] + base[8:] + [
        draft,
        draft * base[7],
        draft * math.log1p(max(float(features.get("global_stagnation", 0.0)), 0.0)),
    ]
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("root draft feature vector contains a non-finite value")
    return vector


def extract_root_draft_features(agent, records, incumbent, global_stagnation):
    """Direct terminal draft experience only; root has no artificial metric."""
    direct = [record for record in records
              if record.operator == "draft" and record.parent_node_id == agent.virtual_root.id]
    valid = [record.child_utility for record in direct
             if record.valid and record.child_utility is not None]
    source_quality = median(valid) if valid else None
    elapsed = max(time.time() - agent.search_start_time, 0.0) if agent.search_start_time else 0.0
    features = {name: 0.0 for name in FEATURE_NAMES}
    features.update({
        "parent_utility": None,
        "parent_metric_missing": 1.0,
        "source_quality": source_quality,
        "source_quality_missing": float(source_quality is None),
        "global_best_utility": incumbent,
        "incumbent_at_decision": incumbent,
        "parent_global_gap": (incumbent - source_quality)
            if incumbent is not None and source_quality is not None else 0.0,
        "branch_best_utility": incumbent,
        "visit_count": float(len(direct)),
        "recent_valid_rate": len(valid) / len(direct) if direct else 0.0,
        "elapsed_fraction": min(elapsed / max(float(agent.acfg.time_limit), 1.0), 1.0),
        "remaining_steps_fraction": max(
            1.0 - len(records) / max(int(agent.acfg.steps), 1), 0.0
        ),
        "operator__draft": 1.0,
        "global_stagnation": float(global_stagnation),
    })
    return features
