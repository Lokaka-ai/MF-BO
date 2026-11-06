#!/usr/bin/env python3
"""Strictly prequential offline prototype for Bayesian Node Value Search.

This evaluator compares the proposed incumbent-relative node-value target with
the existing local-gain model on the same historical runs. Candidate replay is
behavioral only: unselected candidates have no counterfactual outcomes.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from reward_only_acquisition_replay import candidate_groups, make_selector
from reward_only_offline_eval import (
    finite_gain,
    json_safe,
    metric,
    normalized_ranks,
    option_from_record,
    read_jsonl,
    safe_spearman,
    snapshot_features_by_decision,
)
from engine.parent_selectors.features import (
    reward_scale_from_records,
    vectorize_reward_features,
)
from engine.parent_selectors.surrogate import RewardOnlyBayesianLinearSurrogate
from engine.parent_selectors.transition_store import TransitionRecord


NORMAL_OPERATORS = {"improve", "evolution", "fusion"}
BNVS_FEATURE_NAMES = (
    "incumbent_gap_scaled",
    "branch_backed_gap_scaled",
    "expansion_count_log",
    "depth_squashed",
    "stagnation_log",
    "recent_gain_scaled",
    "recent_valid_rate",
    "is_active_leaf",
    "distance_to_active_squashed",
    "remaining_budget_fraction",
    "operator__evolution",
    "operator__fusion",
)
BNVS_FEASIBLE_10_INDICES = (0, 1, 2, 3, 4, 5, 6, 9, 10, 11)


class ConjugateLinearModel:
    def __init__(
        self,
        *,
        prior_precision: float = 4.0,
        intercept_precision: float = 0.25,
        prior_shape: float = 2.0,
        prior_scale: float = 1.0,
    ):
        self.prior_precision = prior_precision
        self.intercept_precision = intercept_precision
        self.prior_shape = prior_shape
        self.prior_scale = prior_scale
        self.mean: np.ndarray | None = None
        self.precision_inverse: np.ndarray | None = None
        self.shape = 0.0
        self.scale = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        design = np.column_stack([np.ones(len(x)), x])
        dimension = design.shape[1]
        prior = np.eye(dimension) * max(float(self.prior_precision), 1e-9)
        prior[0, 0] = max(float(self.intercept_precision), 1e-9)
        precision = prior + design.T @ design
        precision_inverse = np.linalg.inv(precision)
        posterior_mean = precision_inverse @ design.T @ y
        shape = max(float(self.prior_shape), 1e-9) + len(y) / 2.0
        residual = float(y @ y - posterior_mean @ precision @ posterior_mean)
        scale = max(float(self.prior_scale) + 0.5 * max(residual, 0.0), 1e-12)
        self.mean = posterior_mean
        self.precision_inverse = (precision_inverse + precision_inverse.T) / 2.0
        self.shape = shape
        self.scale = scale

    def _design(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.precision_inverse is None:
            raise ValueError("model is not fitted")
        return np.column_stack([np.ones(len(x)), x])

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        design = self._design(x)
        mean = design @ self.mean
        noise_variance = self.scale / max(self.shape - 1.0, 1e-9)
        epistemic = np.einsum(
            "ij,jk,ik->i", design, self.precision_inverse, design
        )
        std = np.sqrt(np.maximum((1.0 + epistemic) * noise_variance, 0.0))
        return mean, std

    def sample(self, x: np.ndarray, *, seed: int) -> np.ndarray:
        design = self._design(x)
        rng = np.random.default_rng(int(seed))
        precision_sample = rng.gamma(
            shape=self.shape,
            scale=1.0 / self.scale,
        )
        covariance = self.precision_inverse / max(float(precision_sample), 1e-12)
        covariance = (covariance + covariance.T) / 2.0
        cholesky = np.linalg.cholesky(
            covariance + np.eye(covariance.shape[0]) * 1e-12
        )
        coefficients = self.mean + cholesky @ rng.standard_normal(len(self.mean))
        return design @ coefficients


def snapshot_groups(path: Path) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for snapshot in read_jsonl(path):
        grouped.setdefault(str(snapshot["decision_id"]), []).append(snapshot)
    return grouped


def selected_snapshot(group: list[dict]) -> dict | None:
    selected = [item for item in group if item.get("selected")]
    return selected[0] if len(selected) == 1 else None


def active_id_for_group(group: list[dict], root_id: str | None) -> str | None:
    if not group:
        return root_id
    reason = str(group[0].get("selection_reason") or "")
    if reason.startswith("continuation_") or reason == "direct_continuation":
        # Runtime invariant: the current continuation node is the first option.
        return str(group[0]["parent_node_id"])
    return root_id


def graph_distance(
    first: str,
    second: str | None,
    parent_by_child: dict[str, str],
) -> int | None:
    if second is None:
        return None
    first_path: dict[str, int] = {}
    node = first
    distance = 0
    while node not in first_path:
        first_path[node] = distance
        if node not in parent_by_child:
            break
        node = parent_by_child[node]
        distance += 1

    node = second
    distance = 0
    visited = set()
    while node not in visited:
        if node in first_path:
            return first_path[node] + distance
        visited.add(node)
        if node not in parent_by_child:
            break
        node = parent_by_child[node]
        distance += 1
    return None


def bounded_scaled(value: float, scale: float) -> float:
    return min(max(float(value) / max(float(scale), 1e-12), -10.0), 10.0)


def bnvs_features(
    features: dict,
    *,
    operator: str,
    node_id: str,
    active_id: str | None,
    parent_by_child: dict[str, str],
    scale: float,
) -> np.ndarray:
    global_best = float(features.get("global_best_utility", 0.0) or 0.0)
    branch_best = float(features.get("branch_best_utility", global_best) or 0.0)
    depth = max(float(features.get("depth", 0.0) or 0.0), 0.0)
    visits = max(float(features.get("visit_count", 0.0) or 0.0), 0.0)
    children = max(float(features.get("children_count", 0.0) or 0.0), 0.0)
    distance = graph_distance(node_id, active_id, parent_by_child)
    distance_value = 1.0 if distance is None else distance / (distance + 5.0)
    remaining_steps = min(max(
        float(features.get("remaining_steps_fraction", 1.0) or 0.0), 0.0
    ), 1.0)
    remaining_time = min(max(
        1.0 - float(features.get("elapsed_fraction", 0.0) or 0.0), 0.0
    ), 1.0)
    vector = np.asarray([
        bounded_scaled(features.get("parent_global_gap", 0.0), scale),
        bounded_scaled(global_best - branch_best, scale),
        math.log1p(visits + children),
        depth / (depth + 5.0),
        math.log1p(max(float(features.get("branch_stagnation", 0.0) or 0.0), 0.0)),
        bounded_scaled(features.get("recent_gain_mean", 0.0), scale),
        min(max(float(features.get("recent_valid_rate", 0.0) or 0.0), 0.0), 1.0),
        float(active_id is not None and node_id == active_id),
        distance_value,
        min(remaining_steps, remaining_time),
        float(operator == "evolution"),
        float(operator == "fusion"),
    ])
    if len(vector) != len(BNVS_FEATURE_NAMES) or not np.isfinite(vector).all():
        raise ValueError("invalid BNVS feature vector")
    return vector


def target_value(
    item: dict,
    *,
    scale: float,
    direct_child_best: dict[str, float],
    horizon: int,
) -> float:
    record: TransitionRecord = item["record"]
    if not record.valid or record.child_utility is None:
        return -1.0
    best = float(record.child_utility)
    if horizon >= 2 and record.child_node_id in direct_child_best:
        best = max(best, direct_child_best[record.child_node_id])
    gap = (best - item["pre_action_best"]) / max(scale, 1e-12)
    return float(min(max(gap, -1.0), 1.0))


def current_scale(history: list[TransitionRecord]) -> float:
    if not history:
        return 1.0
    return reward_scale_from_records(history)


def seed_for(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def fit_bnvs_models(
    history: list[dict],
    *,
    scale: float,
    direct_child_best: dict[str, float],
) -> dict[str, ConjugateLinearModel]:
    x12 = np.asarray([
        bnvs_features(
            item["features"],
            operator=item["record"].operator,
            node_id=item["record"].parent_node_id,
            active_id=item["active_id"],
            parent_by_child=item["parent_by_child"],
            scale=scale,
        )
        for item in history
    ])
    x10 = x12[:, BNVS_FEASIBLE_10_INDICES]
    x15 = np.asarray([
        vectorize_reward_features(item["features"], utility_scale=scale)
        for item in history
    ])
    y1 = np.asarray([
        target_value(
            item,
            scale=scale,
            direct_child_best=direct_child_best,
            horizon=1,
        )
        for item in history
    ])
    y2 = np.asarray([
        target_value(
            item,
            scale=scale,
            direct_child_best=direct_child_best,
            horizon=2,
        )
        for item in history
    ])
    specifications = {
        "bnvs_h1_feasible10": (x10, y1),
        "bnvs_h1_active12": (x12, y1),
        "bnvs_h2_active12": (x12, y2),
        "incumbent_target_current15": (x15, y1),
    }
    models = {}
    for name, (x, y) in specifications.items():
        model = ConjugateLinearModel()
        model.fit(x, y)
        models[name] = model
    return models


def annotate_current(
    record: TransitionRecord,
    *,
    features: dict,
    active_id: str | None,
    parent_by_child: dict[str, str],
) -> dict:
    return {
        "record": record,
        "features": features,
        "active_id": active_id,
        "parent_by_child": dict(parent_by_child),
        "pre_action_best": float(features.get("global_best_utility", 0.0) or 0.0),
    }


def evaluate_run(
    transition_path: Path,
    *,
    runs_root: Path,
    min_observations: int,
    eventual_direct_child_best: dict[str, float],
) -> tuple[list[dict], list[dict], dict]:
    relative = transition_path.relative_to(runs_root)
    method, task, run_name = relative.parts[:3]
    records = [
        TransitionRecord.from_dict(item) for item in read_jsonl(transition_path)
    ]
    snapshots_path = transition_path.parent / "selection_snapshots.jsonl"
    groups = snapshot_groups(snapshots_path)
    selected_features = snapshot_features_by_decision(snapshots_path)
    replay_groups = candidate_groups(snapshots_path)
    root_id = next((
        record.parent_node_id for record in records
        if record.operator == "draft" and record.parent_utility is None
    ), None)

    all_history: list[TransitionRecord] = []
    normal_history: list[dict] = []
    direct_child_best: dict[str, float] = {}
    parent_by_child: dict[str, str] = {}
    factual_rows: list[dict] = []
    candidate_rows: list[dict] = []
    errors = 0

    for record in records:
        group = groups.get(record.decision_id, [])
        features = selected_features.get(record.decision_id) or record.selection_features
        enriched = replace(record, selection_features=features)
        active_id = active_id_for_group(group, root_id)
        annotated = annotate_current(
            enriched,
            features=features,
            active_id=active_id,
            parent_by_child=parent_by_child,
        )

        if (
            enriched.operator in NORMAL_OPERATORS
            and len(normal_history) >= min_observations
        ):
            try:
                scale = current_scale(all_history)
                models = fit_bnvs_models(
                    normal_history,
                    scale=scale,
                    direct_child_best=direct_child_best,
                )
                current_x12 = bnvs_features(
                    features,
                    operator=enriched.operator,
                    node_id=enriched.parent_node_id,
                    active_id=active_id,
                    parent_by_child=parent_by_child,
                    scale=scale,
                )
                current_vectors = {
                    "bnvs_h1_feasible10": current_x12[list(BNVS_FEASIBLE_10_INDICES)],
                    "bnvs_h1_active12": current_x12,
                    "bnvs_h2_active12": current_x12,
                    "incumbent_target_current15": np.asarray(
                        vectorize_reward_features(features, utility_scale=scale)
                    ),
                }
                row = {
                    "method": method,
                    "task": task,
                    "run": run_name,
                    "decision_id": enriched.decision_id,
                    "operator": enriched.operator,
                    "history_normal": len(normal_history),
                    "history_valid_gain": sum(
                        finite_gain(item) for item in all_history
                    ),
                    "scale": scale,
                    "invalid": not enriched.valid or enriched.child_utility is None,
                    "actual_h1": target_value(
                        annotated,
                        scale=scale,
                        direct_child_best=eventual_direct_child_best,
                        horizon=1,
                    ),
                    "actual_h2": target_value(
                        annotated,
                        scale=scale,
                        direct_child_best=eventual_direct_child_best,
                        horizon=2,
                    ),
                }
                for name, model in models.items():
                    mean, std = model.predict(current_vectors[name][None, :])
                    row[f"{name}_mean"] = float(mean[0])
                    row[f"{name}_std"] = float(std[0])

                valid_gain_count = row["history_valid_gain"]
                if valid_gain_count >= 3:
                    gain_model = RewardOnlyBayesianLinearSurrogate()
                    gain_model.fit(all_history)
                    prediction = gain_model.predict([option_from_record(enriched)])[0]
                    parent_utility = float(enriched.parent_utility or 0.0)
                    pre_best = annotated["pre_action_best"]
                    raw_gap = (
                        parent_utility + prediction.mean_gain - pre_best
                    ) / scale
                    trust = 0.10 + 0.15 * min(
                        max((valid_gain_count - 10) / 20.0, 0.0), 1.0
                    )
                    acquisition_gap = (
                        parent_utility + trust * prediction.mean_gain - pre_best
                    ) / scale
                    row["current_gain_raw15_mean"] = float(raw_gap)
                    row["current_gain_acquisition15_mean"] = float(acquisition_gap)
                factual_rows.append(row)

                replay_group = replay_groups.get(record.decision_id)
                if replay_group is not None and len(replay_group[0]) >= 2:
                    options, historical_key = replay_group
                    candidate_x12 = np.asarray([
                        bnvs_features(
                            option.state_features,
                            operator=option.operator,
                            node_id=option.parent_node_id,
                            active_id=active_id,
                            parent_by_child=parent_by_child,
                            scale=scale,
                        )
                        for option in options
                    ])
                    candidate_vectors = {
                        "bnvs_h1_feasible10": candidate_x12[:, BNVS_FEASIBLE_10_INDICES],
                        "bnvs_h1_active12": candidate_x12,
                        "bnvs_h2_active12": candidate_x12,
                    }
                    parent_utilities = [option.parent_utility for option in options]
                    utility_ranks = normalized_ranks(parent_utilities)
                    continuation_context = str(
                        group[0].get("selection_reason") if group else ""
                    ).startswith("continuation_ranked")
                    current_ts_key = None
                    if valid_gain_count >= 10:
                        gain_model = RewardOnlyBayesianLinearSurrogate()
                        gain_model.fit(all_history)
                        provider = lambda current=all_history: list(current)
                        ts_result = make_selector(
                            gain_model,
                            provider,
                            temperature=0.25,
                            random_seed=0,
                        ).select(None, options)
                        current_ts_key = (
                            ts_result.selected_option.parent_node_id,
                            ts_result.selected_option.operator,
                        )

                    for name in (
                        "bnvs_h1_feasible10",
                        "bnvs_h1_active12",
                        "bnvs_h2_active12",
                    ):
                        model = models[name]
                        vector = candidate_vectors[name]
                        mean, _ = model.predict(vector)
                        sampled = model.sample(
                            vector,
                            seed=seed_for(method, task, run_name, record.decision_id, name),
                        )
                        selected_index = int(np.argmax(sampled))
                        mean_index = int(np.argmax(mean))
                        selected_option = options[selected_index]
                        selected_key = (
                            selected_option.parent_node_id,
                            selected_option.operator,
                        )
                        historical_match = selected_key == historical_key
                        actual_key = "actual_h2" if "h2" in name else "actual_h1"
                        candidate_rows.append({
                            "method": method,
                            "task": task,
                            "run": run_name,
                            "decision_id": record.decision_id,
                            "model": name,
                            "history_normal": len(normal_history),
                            "history_valid_gain": valid_gain_count,
                            "candidate_count": len(options),
                            "historical_agreement": historical_match,
                            "posterior_mean_agreement": selected_index == mean_index,
                            "current_ts_comparable": current_ts_key is not None,
                            "current_ts_agreement": (
                                selected_key == current_ts_key
                                if current_ts_key is not None else None
                            ),
                            "selected_parent_utility_rank": utility_ranks[selected_index],
                            "max_parent_utility": math.isclose(
                                selected_option.parent_utility,
                                max(parent_utilities),
                                rel_tol=1e-12,
                                abs_tol=1e-12,
                            ),
                            "continuation_context": continuation_context,
                            "selected_continue": (
                                selected_option.parent_node_id == active_id
                                if continuation_context else None
                            ),
                            "selected_score": float(sampled[selected_index]),
                            "max_abs_candidate_score": float(np.max(np.abs(sampled))),
                            "known_factual_outcome": historical_match,
                            "factual_target": row[actual_key] if historical_match else None,
                            "factual_new_incumbent": (
                                row[actual_key] > 0.0 if historical_match else None
                            ),
                            "factual_invalid": row["invalid"] if historical_match else None,
                        })
            except Exception:
                errors += 1

        all_history.append(enriched)
        parent_by_child[enriched.child_node_id] = enriched.parent_node_id
        if enriched.valid and enriched.child_utility is not None:
            previous = direct_child_best.get(enriched.parent_node_id, -math.inf)
            direct_child_best[enriched.parent_node_id] = max(
                previous,
                float(enriched.child_utility),
            )
        if enriched.operator in NORMAL_OPERATORS:
            normal_history.append(annotated)

    return factual_rows, candidate_rows, {
        "method": method,
        "task": task,
        "run": run_name,
        "normal_transitions": len(normal_history),
        "factual_predictions": len(factual_rows),
        "candidate_decisions": len({row["decision_id"] for row in candidate_rows}),
        "errors": errors,
    }


def per_run_spearman(rows: list[dict], prediction_key: str, actual_key: str) -> dict:
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        if row.get(prediction_key) is not None:
            grouped.setdefault((row["method"], row["task"], row["run"]), []).append(row)
    correlations = []
    for group in grouped.values():
        value = safe_spearman(
            [row[actual_key] for row in group],
            [row[prediction_key] for row in group],
        )
        if value is not None:
            correlations.append(value)
    return {
        "runs": len(grouped),
        "runs_with_defined_spearman": len(correlations),
        "median": float(np.median(correlations)) if correlations else None,
        "mean": float(np.mean(correlations)) if correlations else None,
    }


def summarize_predictions(
    rows: list[dict],
    prediction_key: str,
    *,
    actual_key: str,
) -> dict:
    usable = [row for row in rows if row.get(prediction_key) is not None]
    if not usable:
        return {"n": 0}
    actual = np.asarray([row[actual_key] for row in usable])
    predicted = np.asarray([row[prediction_key] for row in usable])
    clipped = np.clip(predicted, -1.0, 1.0)
    absolute = np.abs(predicted - actual)
    clipped_absolute = np.abs(clipped - actual)
    actual_positive = actual > 0.0
    predicted_positive = predicted > 0.0
    true_positive = int(np.sum(actual_positive & predicted_positive))
    predicted_positive_count = int(np.sum(predicted_positive))
    positive_count = int(np.sum(actual_positive))
    invalid = np.asarray([row["invalid"] for row in usable], dtype=bool)
    return {
        "n": len(usable),
        "runs": len({(row["method"], row["task"], row["run"]) for row in usable}),
        "positive_target_rate": float(np.mean(actual_positive)),
        "invalid_rate": float(np.mean(invalid)),
        "mae": float(np.mean(absolute)),
        "clipped_prediction_mae": float(np.mean(clipped_absolute)),
        "median_absolute_error": float(np.median(absolute)),
        "p90_absolute_error": float(np.quantile(absolute, 0.90)),
        "rmse": float(np.sqrt(np.mean((predicted - actual) ** 2))),
        "sign_accuracy": float(np.mean(predicted_positive == actual_positive)),
        "new_incumbent_precision": (
            true_positive / predicted_positive_count if predicted_positive_count else None
        ),
        "new_incumbent_recall": (
            true_positive / positive_count if positive_count else None
        ),
        "pooled_spearman": safe_spearman(actual, predicted),
        "per_run_spearman": per_run_spearman(
            usable, prediction_key, actual_key
        ),
        "prediction_outside_target_range": float(np.mean(np.abs(predicted) > 1.0)),
        "valid_mae": float(np.mean(absolute[~invalid])) if np.any(~invalid) else None,
        "invalid_mae": float(np.mean(absolute[invalid])) if np.any(invalid) else None,
        "mean_prediction_on_invalid": (
            float(np.mean(predicted[invalid])) if np.any(invalid) else None
        ),
    }


def summarize_candidates(rows: list[dict]) -> dict:
    if not rows:
        return {"decisions": 0}
    known = [row for row in rows if row["known_factual_outcome"]]
    comparable = [row for row in rows if row["current_ts_comparable"]]
    continuation = [row for row in rows if row["continuation_context"]]
    return {
        "decisions": len(rows),
        "runs": len({(row["method"], row["task"], row["run"]) for row in rows}),
        "historical_agreement_rate": float(np.mean([
            row["historical_agreement"] for row in rows
        ])),
        "posterior_mean_agreement_rate": float(np.mean([
            row["posterior_mean_agreement"] for row in rows
        ])),
        "current_thompson_comparable_decisions": len(comparable),
        "current_thompson_agreement_rate": (
            float(np.mean([row["current_ts_agreement"] for row in comparable]))
            if comparable else None
        ),
        "max_parent_utility_rate": float(np.mean([
            row["max_parent_utility"] for row in rows
        ])),
        "mean_selected_parent_utility_rank": float(np.mean([
            row["selected_parent_utility_rank"] for row in rows
        ])),
        "continuation_decisions": len(continuation),
        "continue_rate": (
            float(np.mean([row["selected_continue"] for row in continuation]))
            if continuation else None
        ),
        "p99_selected_abs_score": float(np.quantile([
            abs(row["selected_score"]) for row in rows
        ], 0.99)),
        "max_selected_abs_score": float(max(
            abs(row["selected_score"]) for row in rows
        )),
        "p99_max_abs_candidate_score": float(np.quantile([
            row["max_abs_candidate_score"] for row in rows
        ], 0.99)),
        "factual_matches": len(known),
        "factual_mean_target": (
            float(np.mean([row["factual_target"] for row in known]))
            if known else None
        ),
        "factual_new_incumbent_rate": (
            float(np.mean([row["factual_new_incumbent"] for row in known]))
            if known else None
        ),
        "factual_invalid_rate": (
            float(np.mean([row["factual_invalid"] for row in known]))
            if known else None
        ),
    }


def eventual_best_by_parent(records: list[TransitionRecord]) -> dict[str, float]:
    result: dict[str, float] = {}
    for record in records:
        if record.valid and record.child_utility is not None:
            result[record.parent_node_id] = max(
                result.get(record.parent_node_id, -math.inf),
                float(record.child_utility),
            )
    return result


def render_markdown(result: dict) -> str:
    comparison = result["same_target_h1_after_10_valid"]
    lines = [
        "# BNVS design offline comparison",
        "",
        "## Evaluation boundary",
        "",
        f"- Evaluated {result['run_count']} historical runs with strictly prequential fitting.",
        "- H=1 target is clipped `(child_utility - pre_action_global_best) / scale`; invalid outcomes are `-1`.",
        "- H=2 uses the best valid utility among the child and its direct children, while training labels are updated only after descendants have actually appeared.",
        "- The 10-feature model uses only exactly reconstructable state. The 12-feature model adds active-node identity and tree distance reconstructed from continuation snapshot ordering and the observed parent graph.",
        "- Root was not a regular candidate in these logs, so root-restart choice quality cannot be replayed.",
        "",
        "## Same H=1 target after 10 valid gain observations",
        "",
        "| Model | N | MAE | Clipped MAE | Median AE | P90 AE | New-incumbent sign accuracy | Precision | Recall | Pooled Spearman | Median per-run Spearman | Invalid MAE | Out-of-range prediction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "current_gain_raw15_mean": "Current local-gain model, raw child estimate",
        "current_gain_acquisition15_mean": "Current gain-trust mean acquisition",
        "incumbent_target_current15_mean": "Incumbent target + current 15 features",
        "bnvs_h1_feasible10_mean": "BNVS H=1, exact 10 features",
        "bnvs_h1_active12_mean": "BNVS H=1, active-aware 12 features",
    }
    for key, label in labels.items():
        item = comparison[key]
        run = item.get("per_run_spearman", {})
        lines.append(
            f"| {label} | {item.get('n', 0)} | {metric(item.get('mae'))} | "
            f"{metric(item.get('clipped_prediction_mae'))} | "
            f"{metric(item.get('median_absolute_error'))} | "
            f"{metric(item.get('p90_absolute_error'))} | "
            f"{metric(item.get('sign_accuracy'))} | "
            f"{metric(item.get('new_incumbent_precision'))} | "
            f"{metric(item.get('new_incumbent_recall'))} | "
            f"{metric(item.get('pooled_spearman'))} | "
            f"{metric(run.get('median'))} | {metric(item.get('invalid_mae'))} | "
            f"{metric(item.get('prediction_outside_target_range'))} |"
        )

    lines.extend([
        "",
        "## BNVS warm-up sensitivity",
        "",
        "| Minimum prior normal observations | Model | N | MAE | Sign accuracy | Pooled Spearman | Median per-run Spearman | Invalid MAE |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ])
    for threshold, models in result["warmup"].items():
        for key in ("bnvs_h1_feasible10_mean", "bnvs_h1_active12_mean"):
            item = models[key]
            lines.append(
                f"| {threshold} | {labels[key]} | {item.get('n', 0)} | "
                f"{metric(item.get('mae'))} | {metric(item.get('sign_accuracy'))} | "
                f"{metric(item.get('pooled_spearman'))} | "
                f"{metric(item.get('per_run_spearman', {}).get('median'))} | "
                f"{metric(item.get('invalid_mae'))} |"
            )

    lines.extend([
        "",
        "## H=1 versus delayed H=2 credit",
        "",
        "| Predictor evaluated on eventual H=2 target | N | MAE | Sign accuracy | Pooled Spearman | Median per-run Spearman |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for key, item in result["h2_target_after_10_valid"].items():
        label = (
            "H=1 active-aware model" if key == "bnvs_h1_active12_mean"
            else "H=2 active-aware model"
        )
        lines.append(
            f"| {label} | {item.get('n', 0)} | {metric(item.get('mae'))} | "
            f"{metric(item.get('sign_accuracy'))} | {metric(item.get('pooled_spearman'))} | "
            f"{metric(item.get('per_run_spearman', {}).get('median'))} |"
        )

    lines.extend([
        "",
        "## Vanilla Thompson candidate replay after 5 observations",
        "",
        "| Model | Decisions | Historical agreement | Mean-policy agreement | Current TS agreement | Max-parent rate | Mean parent rank | Continue rate | P99 selected abs score | P99 max candidate abs score | Factual matches | New-incumbent rate on matches | Invalid rate on matches |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    candidate_labels = {
        "bnvs_h1_feasible10": "BNVS H=1, exact 10 features",
        "bnvs_h1_active12": "BNVS H=1, active-aware 12 features",
        "bnvs_h2_active12": "BNVS H=2, active-aware 12 features",
    }
    for key, label in candidate_labels.items():
        item = result["candidate_replay_after_5"][key]
        lines.append(
            f"| {label} | {item.get('decisions', 0)} | "
            f"{metric(item.get('historical_agreement_rate'))} | "
            f"{metric(item.get('posterior_mean_agreement_rate'))} | "
            f"{metric(item.get('current_thompson_agreement_rate'))} | "
            f"{metric(item.get('max_parent_utility_rate'))} | "
            f"{metric(item.get('mean_selected_parent_utility_rank'))} | "
            f"{metric(item.get('continue_rate'))} | "
            f"{metric(item.get('p99_selected_abs_score'))} | "
            f"{metric(item.get('p99_max_abs_candidate_score'))} | "
            f"{item.get('factual_matches', 0)} | "
            f"{metric(item.get('factual_new_incumbent_rate'))} | "
            f"{metric(item.get('factual_invalid_rate'))} |"
        )

    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "Factual prediction metrics test the target and surrogate on historically selected actions. Candidate replay tests how the policy would behave, but only matched historical actions expose outcomes. It cannot prove final-score improvement or compare counterfactual continue/switch decisions.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--min-observations", type=int, default=3)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "docs"
        / "BNVS_OFFLINE_COMPARISON.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "docs"
        / "BNVS_OFFLINE_COMPARISON.md",
    )
    args = parser.parse_args()
    runs_root = args.runs_root.resolve()
    paths = sorted(runs_root.glob("*/*/*/logs/phase1/transitions.jsonl"))
    if not paths:
        raise SystemExit(f"no transition logs found under {runs_root}")

    all_factual: list[dict] = []
    all_candidates: list[dict] = []
    run_summaries = []
    for path in paths:
        records = [
            TransitionRecord.from_dict(item) for item in read_jsonl(path)
        ]
        factual, candidates, summary = evaluate_run(
            path,
            runs_root=runs_root,
            min_observations=max(int(args.min_observations), 3),
            eventual_direct_child_best=eventual_best_by_parent(records),
        )
        all_factual.extend(factual)
        all_candidates.extend(candidates)
        run_summaries.append(summary)

    h1_keys = (
        "current_gain_raw15_mean",
        "current_gain_acquisition15_mean",
        "incumbent_target_current15_mean",
        "bnvs_h1_feasible10_mean",
        "bnvs_h1_active12_mean",
    )
    after_ten_valid = [
        row for row in all_factual if row["history_valid_gain"] >= 10
    ]
    result = json_safe({
        "runs_root": str(runs_root),
        "run_count": len(paths),
        "minimum_fitted_observations": max(int(args.min_observations), 3),
        "feature_names": list(BNVS_FEATURE_NAMES),
        "feasible_10_feature_names": [
            BNVS_FEATURE_NAMES[index] for index in BNVS_FEASIBLE_10_INDICES
        ],
        "replay_errors": sum(item["errors"] for item in run_summaries),
        "same_target_h1_after_10_valid": {
            key: summarize_predictions(
                after_ten_valid,
                key,
                actual_key="actual_h1",
            )
            for key in h1_keys
        },
        "warmup": {
            str(threshold): {
                key: summarize_predictions(
                    [
                        row for row in all_factual
                        if row["history_normal"] >= threshold
                    ],
                    key,
                    actual_key="actual_h1",
                )
                for key in (
                    "bnvs_h1_feasible10_mean",
                    "bnvs_h1_active12_mean",
                )
            }
            for threshold in (3, 5, 8, 10, 15, 20, 30)
        },
        "h2_target_after_10_valid": {
            key: summarize_predictions(
                after_ten_valid,
                key,
                actual_key="actual_h2",
            )
            for key in (
                "bnvs_h1_active12_mean",
                "bnvs_h2_active12_mean",
            )
        },
        "candidate_replay_after_5": {
            model: summarize_candidates([
                row for row in all_candidates
                if row["model"] == model and row["history_normal"] >= 5
            ])
            for model in (
                "bnvs_h1_feasible10",
                "bnvs_h1_active12",
                "bnvs_h2_active12",
            )
        },
        "run_summaries": run_summaries,
    })
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output_md.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps({
        "same_target_h1_after_10_valid": result["same_target_h1_after_10_valid"],
        "h2_target_after_10_valid": result["h2_target_after_10_valid"],
        "candidate_replay_after_5": result["candidate_replay_after_5"],
        "replay_errors": result["replay_errors"],
    }, ensure_ascii=False, indent=2))
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
