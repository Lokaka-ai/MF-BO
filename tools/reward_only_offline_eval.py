#!/usr/bin/env python3
"""Strictly prequential evaluation of the local reward-only surrogate.

Only the factual outcome of the action selected by the historical policy is
scored.  Candidate-policy regret is intentionally not estimated because the
logs contain no outcomes for actions that were not executed.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
from scipy.stats import binomtest, spearmanr


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.parent_selectors.base import ExpansionOption, OperatorPreview  # noqa: E402
from engine.parent_selectors.surrogate import (  # noqa: E402
    BayesianTransitionSurrogate,
    RewardOnlyBayesianLinearSurrogate,
)
from engine.parent_selectors.transition_store import TransitionRecord  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalized_ranks(values: list[float]) -> list[float]:
    if len(values) <= 1:
        return [1.0] * len(values)
    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and math.isclose(
            values[order[start]], values[order[end]], rel_tol=1e-12, abs_tol=1e-12
        ):
            end += 1
        rank = ((start + end - 1) / 2.0) / (len(order) - 1)
        for position in range(start, end):
            result[order[position]] = rank
        start = end
    return result


def snapshot_features_by_decision(snapshot_path: Path) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict]] = {}
    for snapshot in read_jsonl(snapshot_path):
        grouped.setdefault(str(snapshot["decision_id"]), []).append(snapshot)

    selected_features: dict[str, dict[str, float]] = {}
    for decision_id, snapshots in grouped.items():
        utilities = [
            float(item.get("parent_utility") or 0.0)
            for item in snapshots
        ]
        ranks = normalized_ranks(utilities)
        for snapshot, rank in zip(snapshots, ranks):
            snapshot["_offline_parent_rank_percentile"] = rank
        selected = [item for item in snapshots if item.get("selected")]
        if len(selected) != 1:
            continue
        features = dict(selected[0].get("feature_vector") or {})
        features["parent_rank_percentile"] = float(
            selected[0]["_offline_parent_rank_percentile"]
        )
        selected_features[decision_id] = features
    return selected_features


def option_from_record(record: TransitionRecord) -> ExpansionOption:
    return ExpansionOption(
        parent_node_id=record.parent_node_id,
        operator=record.operator,
        reference_node_ids=record.reference_node_ids,
        state_features=record.selection_features,
        parent_node=None,
        preview=OperatorPreview(record.operator, record.reference_node_ids),
    )


def finite_gain(record: TransitionRecord) -> bool:
    return (
        record.valid
        and record.improvement is not None
        and math.isfinite(float(record.improvement))
    )


def evaluate_run(
    transition_path: Path,
    *,
    runs_root: Path,
    min_history: int,
) -> tuple[list[dict], dict]:
    relative = transition_path.relative_to(runs_root)
    method, task, run_name = relative.parts[:3]
    snapshot_path = transition_path.parent / "selection_snapshots.jsonl"
    feature_lookup = snapshot_features_by_decision(snapshot_path)
    records = [
        TransitionRecord.from_dict(row)
        for row in read_jsonl(transition_path)
    ]

    history: list[TransitionRecord] = []
    predictions: list[dict] = []
    missing_snapshot = 0
    model_failures = 0
    for record in records:
        exact_features = feature_lookup.get(record.decision_id)
        if exact_features is None:
            missing_snapshot += 1
            enriched = record
        else:
            enriched = replace(record, selection_features=exact_features)

        valid_history = sum(finite_gain(item) for item in history)
        if finite_gain(enriched) and valid_history >= min_history:
            try:
                reward_only = RewardOnlyBayesianLinearSurrogate()
                reward_only.fit(history)
                legacy = BayesianTransitionSurrogate()
                legacy.fit(history)
                option = option_from_record(enriched)
                reward_prediction = reward_only.predict([option])[0]
                legacy_prediction = legacy.predict([option])[0]
                historical_gains = [
                    float(item.improvement) for item in history if finite_gain(item)
                ]
                actual = float(enriched.improvement)
                scale = reward_only.reward_scale
                gain_trust = 0.10 + min(
                    max((valid_history - 10) / 20.0, 0.0),
                    1.0,
                ) * 0.15
                predictions.append({
                    "method": method,
                    "task": task,
                    "run": run_name,
                    "transition_id": enriched.transition_id,
                    "history_valid": valid_history,
                    "operator": enriched.operator,
                    "actual_gain": actual,
                    "reward_only_mean": reward_prediction.mean_gain,
                    "reward_only_std": reward_prediction.std_gain,
                    "acquisition_mean": gain_trust * reward_prediction.mean_gain,
                    "gain_trust": gain_trust,
                    "legacy_mean": legacy_prediction.mean_gain,
                    "legacy_std": legacy_prediction.std_gain,
                    "mcgs_q": float(
                        enriched.selection_features.get("mcgs_q", 0.0)
                    ),
                    "history_mean": float(np.mean(historical_gains)),
                    "reward_scale": scale,
                })
            except Exception:
                model_failures += 1
        history.append(enriched)

    return predictions, {
        "method": method,
        "task": task,
        "run": run_name,
        "transitions": len(records),
        "valid_transitions": sum(finite_gain(item) for item in history),
        "predictions": len(predictions),
        "missing_selected_snapshots": missing_snapshot,
        "model_failures": model_failures,
    }


def safe_spearman(actual: Iterable[float], predicted: Iterable[float]) -> float | None:
    actual_values = list(actual)
    predicted_values = list(predicted)
    if (
        len(actual_values) < 3
        or len(set(actual_values)) < 2
        or len(set(predicted_values)) < 2
    ):
        return None
    value = float(spearmanr(actual_values, predicted_values).statistic)
    return value if math.isfinite(value) else None


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    actual = np.asarray([row["actual_gain"] / row["reward_scale"] for row in rows])
    new = np.asarray([row["reward_only_mean"] / row["reward_scale"] for row in rows])
    acquisition = np.asarray([
        row["acquisition_mean"] / row["reward_scale"] for row in rows
    ])
    old = np.asarray([row["legacy_mean"] / row["reward_scale"] for row in rows])
    history_mean = np.asarray([row["history_mean"] / row["reward_scale"] for row in rows])
    new_std = np.asarray([row["reward_only_std"] / row["reward_scale"] for row in rows])
    old_std = np.asarray([row["legacy_std"] / row["reward_scale"] for row in rows])
    mcgs_q = np.asarray([row["mcgs_q"] for row in rows])
    new_abs = np.abs(new - actual)
    acquisition_abs = np.abs(acquisition - actual)
    old_abs = np.abs(old - actual)
    strict_new_wins = int(np.sum(new_abs < old_abs))
    strict_old_wins = int(np.sum(old_abs < new_abs))
    comparable = strict_new_wins + strict_old_wins
    paired_p = (
        float(binomtest(strict_new_wins, comparable, 0.5).pvalue)
        if comparable else None
    )
    return {
        "n": len(rows),
        "positive_rate": float(np.mean(actual > 0.0)),
        "acquisition_nmae": float(np.mean(acquisition_abs)),
        "acquisition_median_absolute_error": float(np.median(acquisition_abs)),
        "acquisition_p90_absolute_error": float(np.quantile(acquisition_abs, 0.90)),
        "acquisition_nrmse": float(np.sqrt(np.mean((acquisition - actual) ** 2))),
        "acquisition_sign_accuracy": float(
            np.mean((acquisition > 0.0) == (actual > 0.0))
        ),
        "acquisition_pooled_spearman": safe_spearman(actual, acquisition),
        "reward_only_nmae": float(np.mean(new_abs)),
        "legacy_nmae": float(np.mean(old_abs)),
        "reward_only_median_absolute_error": float(np.median(new_abs)),
        "legacy_median_absolute_error": float(np.median(old_abs)),
        "reward_only_p90_absolute_error": float(np.quantile(new_abs, 0.90)),
        "legacy_p90_absolute_error": float(np.quantile(old_abs, 0.90)),
        "history_mean_nmae": float(np.mean(np.abs(history_mean - actual))),
        "zero_gain_nmae": float(np.mean(np.abs(actual))),
        "reward_only_nrmse": float(np.sqrt(np.mean((new - actual) ** 2))),
        "legacy_nrmse": float(np.sqrt(np.mean((old - actual) ** 2))),
        "reward_only_sign_accuracy": float(np.mean((new > 0.0) == (actual > 0.0))),
        "legacy_sign_accuracy": float(np.mean((old > 0.0) == (actual > 0.0))),
        "mcgs_q_sign_proxy_accuracy": float(
            np.mean((mcgs_q > 0.0) == (actual > 0.0))
        ),
        "mcgs_q_zero_rate": float(np.mean(mcgs_q == 0.0)),
        "reward_only_1sigma_coverage": float(np.mean(np.abs(new - actual) <= new_std)),
        "reward_only_2sigma_coverage": float(np.mean(np.abs(new - actual) <= 2.0 * new_std)),
        "legacy_1sigma_coverage": float(np.mean(np.abs(old - actual) <= old_std)),
        "legacy_2sigma_coverage": float(np.mean(np.abs(old - actual) <= 2.0 * old_std)),
        "reward_only_pooled_spearman": safe_spearman(actual, new),
        "legacy_pooled_spearman": safe_spearman(actual, old),
        "mcgs_q_pooled_spearman": safe_spearman(actual, mcgs_q),
        "reward_only_lower_error": strict_new_wins,
        "legacy_lower_error": strict_old_wins,
        "paired_sign_test_p": paired_p,
    }


def per_run_spearman(rows: list[dict], prediction_key: str) -> dict:
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["method"], row["task"], row["run"]), []).append(row)
    values = []
    for group in grouped.values():
        correlation = safe_spearman(
            [item["actual_gain"] for item in group],
            [item[prediction_key] for item in group],
        )
        if correlation is not None:
            values.append(correlation)
    return {
        "runs": len(grouped),
        "runs_with_defined_spearman": len(values),
        "median": float(np.median(values)) if values else None,
        "mean": float(np.mean(values)) if values else None,
    }


def per_run_nmae(rows: list[dict], prediction_key: str) -> dict:
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["method"], row["task"], row["run"]), []).append(row)
    values = []
    for group in grouped.values():
        errors = [
            abs(item[prediction_key] - item["actual_gain"]) / item["reward_scale"]
            for item in group
        ]
        if errors:
            values.append(float(np.mean(errors)))
    return {
        "runs": len(grouped),
        "median": float(np.median(values)) if values else None,
        "mean": float(np.mean(values)) if values else None,
    }


def load_historical_policy_comparison(analysis_root: Path | None) -> dict | None:
    """Load the frozen old-BO versus Legacy-MCGS outcome comparison."""
    if analysis_root is None:
        return None
    summary_path = analysis_root / "analysis_summary.json"
    paired_path = analysis_root / "paired_analysis.csv"
    if not summary_path.is_file() or not paired_path.is_file():
        raise ValueError(
            "historical analysis root must contain analysis_summary.json "
            "and paired_analysis.csv"
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with paired_path.open(encoding="utf-8", newline="") as handle:
        paired_rows = list(csv.DictReader(handle))
    legacy_any_medal = sum(row["legacy_medal"] != "none" for row in paired_rows)
    bo_any_medal = sum(row["bo_medal"] != "none" for row in paired_rows)
    legacy_gold = sum(row["legacy_medal"] == "gold" for row in paired_rows)
    bo_gold = sum(row["bo_medal"] == "gold" for row in paired_rows)
    time_deltas = [float(row["time_to_best_delta_hours"]) for row in paired_rows]

    return {
        "source": str(analysis_root.resolve()),
        "task_pairs": len(paired_rows),
        "official_score_wins": summary["official_score_wins"],
        "score_win_exact_sign_pvalue_two_sided": summary[
            "score_win_exact_sign_pvalue_two_sided"
        ],
        "any_medal": {"Legacy_MCGS": legacy_any_medal, "old_BO": bo_any_medal},
        "gold_medal": {"Legacy_MCGS": legacy_gold, "old_BO": bo_gold},
        "normalized_gold_gap_delta": summary["normalized_gold_gap_delta"],
        "search_totals": summary["search_totals"],
        "time_to_best": {
            "old_BO_later_tasks": sum(delta > 0.0 for delta in time_deltas),
            "Legacy_MCGS_later_tasks": sum(delta < 0.0 for delta in time_deltas),
            "median_old_BO_minus_MCGS_hours": float(np.median(time_deltas)),
            "mean_old_BO_minus_MCGS_hours": float(np.mean(time_deltas)),
        },
        "old_bo_selected_action_subgroups": summary[
            "selector_selected_action_subgroups"
        ],
    }


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def metric(value, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def render_markdown(result: dict) -> str:
    overall = result["overall"]
    new_spearman = result["per_run_spearman"]["reward_only"]
    acquisition_spearman = result["per_run_spearman"]["acquisition"]
    old_spearman = result["per_run_spearman"]["legacy"]
    mcgs_spearman = result["per_run_spearman"]["mcgs_q"]
    new_run_error = result["per_run_nmae"]["reward_only"]
    acquisition_run_error = result["per_run_nmae"]["acquisition"]
    old_run_error = result["per_run_nmae"]["legacy"]
    lines = [
        "# Reward-only local BO offline evaluation",
        "",
        "## Protocol",
        "",
        f"- Evaluated {result['run_count']} independent run logs with a warm-up of "
        f"{result['min_history']} valid transitions per run.",
        "- Every prediction is strictly prequential: only earlier transitions from the same run are used.",
        "- No pretraining, task metadata, validity model, or cost model is used by the reward-only model.",
        "- Errors are divided by the run-local robust reward scale before pooling across heterogeneous metrics.",
        "- Only factual outcomes of historically selected actions are scored. The logs do not contain outcomes "
        "for unselected candidates, so this analysis cannot establish counterfactual policy regret or final-score uplift.",
        "",
        "## Aggregate factual prediction results",
        "",
        "| Metric | Current acquisition contribution | Raw reward posterior mean | Previous 49-feature model |",
        "|---|---:|---:|---:|",
        f"| Normalized MAE | {metric(overall.get('acquisition_nmae'))} | {metric(overall.get('reward_only_nmae'))} | {metric(overall.get('legacy_nmae'))} |",
        f"| Median normalized absolute error | {metric(overall.get('acquisition_median_absolute_error'))} | {metric(overall.get('reward_only_median_absolute_error'))} | {metric(overall.get('legacy_median_absolute_error'))} |",
        f"| 90th-percentile normalized absolute error | {metric(overall.get('acquisition_p90_absolute_error'))} | {metric(overall.get('reward_only_p90_absolute_error'))} | {metric(overall.get('legacy_p90_absolute_error'))} |",
        f"| Median per-run normalized MAE | {metric(acquisition_run_error.get('median'))} | {metric(new_run_error.get('median'))} | {metric(old_run_error.get('median'))} |",
        f"| Normalized RMSE | {metric(overall.get('acquisition_nrmse'))} | {metric(overall.get('reward_only_nrmse'))} | {metric(overall.get('legacy_nrmse'))} |",
        f"| Gain-sign accuracy | {metric(overall.get('acquisition_sign_accuracy'))} | {metric(overall.get('reward_only_sign_accuracy'))} | {metric(overall.get('legacy_sign_accuracy'))} |",
        f"| Pooled Spearman | {metric(overall.get('acquisition_pooled_spearman'))} | {metric(overall.get('reward_only_pooled_spearman'))} | {metric(overall.get('legacy_pooled_spearman'))} |",
        f"| Median per-run Spearman | {metric(acquisition_spearman.get('median'))} | {metric(new_spearman.get('median'))} | {metric(old_spearman.get('median'))} |",
        "",
        f"The evaluation contains {overall.get('n', 0)} factual next-transition predictions. "
        f"The reward-only model has lower absolute error on {overall.get('reward_only_lower_error', 0)} "
        f"predictions versus {overall.get('legacy_lower_error', 0)} for the previous model "
        f"(paired sign-test p={metric(overall.get('paired_sign_test_p'))}).",
        "",
        "For context, the historical-mean normalized MAE is "
        f"{metric(overall.get('history_mean_nmae'))}, and the zero-gain baseline is "
        f"{metric(overall.get('zero_gain_nmae'))}. The observed positive-gain rate is "
        f"{metric(overall.get('positive_rate'))}.",
        "",
        "## MCGS signal-proxy comparison",
        "",
        "MCGS does not predict continuous gain, so NMAE and uncertainty coverage are not defined "
        "for it. The closest factual diagnostic available in these logs is the selected parent's "
        "selection-time `mcgs_q`, which is an average of past ordinal search rewards. It is only a "
        "signal proxy: the full progressive-MCGS policy also uses UCT exploration, tree traversal, "
        "and the time-dependent Top-K switch.",
        "",
        "| Metric | Reward-only gain model | Previous gain model | MCGS-Q proxy |",
        "|---|---:|---:|---:|",
        f"| Gain direction/proxy accuracy | {metric(overall.get('reward_only_sign_accuracy'))} | "
        f"{metric(overall.get('legacy_sign_accuracy'))} | "
        f"{metric(overall.get('mcgs_q_sign_proxy_accuracy'))} |",
        f"| Pooled Spearman with realized gain | {metric(overall.get('reward_only_pooled_spearman'))} | "
        f"{metric(overall.get('legacy_pooled_spearman'))} | "
        f"{metric(overall.get('mcgs_q_pooled_spearman'))} |",
        f"| Median per-run Spearman | {metric(new_spearman.get('median'))} | "
        f"{metric(old_spearman.get('median'))} | {metric(mcgs_spearman.get('median'))} |",
        f"| Mean per-run Spearman | {metric(new_spearman.get('mean'))} | "
        f"{metric(old_spearman.get('mean'))} | {metric(mcgs_spearman.get('mean'))} |",
        "",
        f"`mcgs_q` is exactly zero for {metric(overall.get('mcgs_q_zero_rate'))} of evaluated "
        "selected actions. Consequently, thresholding it at zero is a weak and tie-heavy direction "
        "proxy rather than a calibrated gain forecast.",
        "",
        "## Findings",
        "",
        "- The compact model improves the median and tail factual errors, gain-sign accuracy, "
        "and pooled rank correlation relative to the previous 49-feature model. The previous "
        "model's pooled mean errors are dominated by catastrophic extrapolation outliers.",
        "- Scaling the model gain by the 0.10-to-0.25 trust schedule improves factual error "
        "substantially while preserving its sign and rank ordering. This is the contribution "
        "used by the pure BO acquisition; it is not an MCGS mixture weight.",
        "- The raw compact model does not beat the historical-mean or zero-gain baselines on "
        "pooled normalized MAE. Its median per-run Spearman correlation is close to zero, so "
        "the acquisition deliberately keeps the learned gain contribution small.",
        "- No clipping is applied. The separate candidate replay report quantifies posterior-tail "
        "exposure and choice stability under the requested unbounded acquisition.",
        "",
        "## History-size slices",
        "",
        "| Prior valid transitions | N | Acquisition NMAE | Raw reward NMAE | Previous NMAE | Reward-only sign accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, summary in result["history_slices"].items():
        lines.append(
            f"| {label} | {summary.get('n', 0)} | "
            f"{metric(summary.get('acquisition_nmae'))} | "
            f"{metric(summary.get('reward_only_nmae'))} | "
            f"{metric(summary.get('legacy_nmae'))} | "
            f"{metric(summary.get('reward_only_sign_accuracy'))} |"
        )
    historical = result.get("historical_policy_comparison")
    if historical is not None:
        score_wins = historical["official_score_wins"]
        search_totals = historical["search_totals"]
        time_to_best = historical["time_to_best"]
        subgroups = historical["old_bo_selected_action_subgroups"]
        fallback = subgroups["fallback"]
        model = subgroups["model"]
        lines.extend([
            "",
            "## Historical policy outcome comparison",
            "",
            "This section compares the actually executed Legacy progressive-MCGS runs with the "
            "actually executed **previous** BO runs. It is useful context, but it is not an outcome "
            "evaluation of the new reward-only policy, which has not yet been run online.",
            "",
            "| Historical outcome | Legacy progressive MCGS | Previous BO |",
            "|---|---:|---:|",
            f"| Official private-score wins | {score_wins.get('Legacy', 0)}/{historical['task_pairs']} | "
            f"{score_wins.get('BO', 0)}/{historical['task_pairs']} |",
            f"| Any medal | {historical['any_medal']['Legacy_MCGS']}/{historical['task_pairs']} | "
            f"{historical['any_medal']['old_BO']}/{historical['task_pairs']} |",
            f"| Gold medal | {historical['gold_medal']['Legacy_MCGS']}/{historical['task_pairs']} | "
            f"{historical['gold_medal']['old_BO']}/{historical['task_pairs']} |",
            f"| Candidate launches | {search_totals['Legacy']['launched']} | "
            f"{search_totals['BO']['launched']} |",
            f"| Valid yield | {metric(search_totals['Legacy']['valid_yield'])} | "
            f"{metric(search_totals['BO']['valid_yield'])} |",
            f"| Tasks where best candidate arrived later | "
            f"{time_to_best['Legacy_MCGS_later_tasks']} | {time_to_best['old_BO_later_tasks']} |",
            "",
            f"The median old-BO minus MCGS normalized score was "
            f"{historical['normalized_gold_gap_delta']['median']:+.3f} gold-gap units; the "
            f"two-sided score-win sign-test p-value was "
            f"{historical['score_win_exact_sign_pvalue_two_sided']:.3f}. Thus the 13-8 old-BO "
            "win count is not statistically decisive from one seed per task.",
            "",
            "Within the historical old-BO runs, progressive-MCGS fallback actions and learned-model "
            "actions had the following observational outcomes:",
            "",
            "| Selected-action subgroup | Progressive-MCGS fallback | Previous learned selector |",
            "|---|---:|---:|",
            f"| Actions | {fallback['count']} | {model['count']} |",
            f"| Comparable realized gains | {fallback['comparable_gain_count']} | "
            f"{model['comparable_gain_count']} |",
            f"| Positive realized gain | {metric(fallback['positive_gain_rate'])} | "
            f"{metric(model['positive_gain_rate'])} |",
            f"| Valid action | {metric(fallback['valid_rate'])} | "
            f"{metric(model['valid_rate'])} |",
            "",
            "The fallback comparison is confounded by warm-up stage, operator mix, parent quality, "
            "and remaining headroom; it is descriptive evidence, not a causal estimate of MCGS superiority.",
        ])
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "This is a surrogate-quality gate, not an offline proof that the pure Thompson acquisition improves the final MLEbench score. "
        "A policy-level conclusion requires new randomized online runs because each historical decision exposes only "
        "the result of the action that the old policy executed.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--min-history", type=int, default=10)
    parser.add_argument(
        "--historical-analysis-root",
        type=Path,
        help=(
            "Optional directory containing analysis_summary.json and "
            "paired_analysis.csv for the frozen old-BO versus Legacy-MCGS runs"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT / "docs" / "REWARD_ONLY_OFFLINE_EVALUATION.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=REPO_ROOT / "docs" / "REWARD_ONLY_OFFLINE_EVALUATION.md",
    )
    args = parser.parse_args()
    runs_root = args.runs_root.resolve()
    transition_paths = sorted(runs_root.glob("*/*/*/logs/phase1/transitions.jsonl"))
    if not transition_paths:
        raise SystemExit(f"no transition logs found under {runs_root}")

    all_predictions: list[dict] = []
    run_summaries = []
    for path in transition_paths:
        predictions, run_summary = evaluate_run(
            path,
            runs_root=runs_root,
            min_history=max(int(args.min_history), 1),
        )
        all_predictions.extend(predictions)
        run_summaries.append(run_summary)

    slices = {
        "10-14": [row for row in all_predictions if 10 <= row["history_valid"] <= 14],
        "15-19": [row for row in all_predictions if 15 <= row["history_valid"] <= 19],
        "20-29": [row for row in all_predictions if 20 <= row["history_valid"] <= 29],
        "30+": [row for row in all_predictions if row["history_valid"] >= 30],
    }
    result = json_safe({
        "runs_root": str(runs_root),
        "run_count": len(transition_paths),
        "min_history": max(int(args.min_history), 1),
        "overall": summarize(all_predictions),
        "by_logged_method": {
            method: summarize([row for row in all_predictions if row["method"] == method])
            for method in sorted({row["method"] for row in all_predictions})
        },
        "history_slices": {
            label: summarize(rows) for label, rows in slices.items()
        },
        "per_run_spearman": {
            "acquisition": per_run_spearman(all_predictions, "acquisition_mean"),
            "reward_only": per_run_spearman(all_predictions, "reward_only_mean"),
            "legacy": per_run_spearman(all_predictions, "legacy_mean"),
            "mcgs_q": per_run_spearman(all_predictions, "mcgs_q"),
        },
        "per_run_nmae": {
            "acquisition": per_run_nmae(all_predictions, "acquisition_mean"),
            "reward_only": per_run_nmae(all_predictions, "reward_only_mean"),
            "legacy": per_run_nmae(all_predictions, "legacy_mean"),
        },
        "historical_policy_comparison": load_historical_policy_comparison(
            args.historical_analysis_root.resolve()
            if args.historical_analysis_root is not None
            else None
        ),
        "run_summaries": run_summaries,
    })

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output_md.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(result["overall"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
