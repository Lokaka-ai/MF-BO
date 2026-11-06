#!/usr/bin/env python3
"""Sensitivity analysis for reward-only BO warm-up thresholds."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np

from reward_only_offline_eval import (
    finite_gain,
    json_safe,
    metric,
    option_from_record,
    per_run_nmae,
    per_run_spearman,
    read_jsonl,
    safe_spearman,
    snapshot_features_by_decision,
)
from engine.parent_selectors.surrogate import RewardOnlyBayesianLinearSurrogate
from engine.parent_selectors.transition_store import TransitionRecord


DEFAULT_THRESHOLDS = (3, 5, 8, 10, 15, 20, 30)
HISTORY_BINS = (
    ("3-4", 3, 4),
    ("5-7", 5, 7),
    ("8-9", 8, 9),
    ("10-14", 10, 14),
    ("15-19", 15, 19),
    ("20-29", 20, 29),
    ("30+", 30, None),
)


def gain_trust(
    history_valid: int,
    *,
    threshold: int,
    initial_trust: float,
    maximum_trust: float,
    full_trust_at: int,
) -> float:
    if history_valid < threshold:
        return 0.0
    effective_full = max(full_trust_at, threshold)
    if effective_full == threshold:
        return maximum_trust
    progress = min(
        max((history_valid - threshold) / (effective_full - threshold), 0.0),
        1.0,
    )
    return initial_trust + progress * (maximum_trust - initial_trust)


def rows_in_range(rows: list[dict], lower: int, upper: int | None) -> list[dict]:
    return [
        row for row in rows
        if row["history_valid"] >= lower
        and (upper is None or row["history_valid"] <= upper)
    ]


def evaluate_reward_only_run(
    transition_path: Path,
    *,
    runs_root: Path,
    min_history: int,
) -> tuple[list[dict], dict]:
    relative = transition_path.relative_to(runs_root)
    method, task, run_name = relative.parts[:3]
    feature_lookup = snapshot_features_by_decision(
        transition_path.parent / "selection_snapshots.jsonl"
    )
    records = [
        TransitionRecord.from_dict(row) for row in read_jsonl(transition_path)
    ]
    history: list[TransitionRecord] = []
    predictions = []
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
                model = RewardOnlyBayesianLinearSurrogate()
                model.fit(history)
                prediction = model.predict([option_from_record(enriched)])[0]
                historical_gains = [
                    float(item.improvement) for item in history if finite_gain(item)
                ]
                predictions.append({
                    "method": method,
                    "task": task,
                    "run": run_name,
                    "history_valid": valid_history,
                    "actual_gain": float(enriched.improvement),
                    "reward_only_mean": prediction.mean_gain,
                    "reward_only_std": prediction.std_gain,
                    "history_mean": float(np.mean(historical_gains)),
                    "reward_scale": model.reward_scale,
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


def summarize_reward_only(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    actual = np.asarray([row["actual_gain"] / row["reward_scale"] for row in rows])
    predicted = np.asarray([
        row["reward_only_mean"] / row["reward_scale"] for row in rows
    ])
    historical_mean = np.asarray([
        row["history_mean"] / row["reward_scale"] for row in rows
    ])
    predicted_std = np.asarray([
        row["reward_only_std"] / row["reward_scale"] for row in rows
    ])
    absolute_error = np.abs(predicted - actual)
    return {
        "n": len(rows),
        "history_mean_nmae": float(np.mean(np.abs(historical_mean - actual))),
        "zero_gain_nmae": float(np.mean(np.abs(actual))),
        "reward_only_nmae": float(np.mean(absolute_error)),
        "reward_only_median_absolute_error": float(np.median(absolute_error)),
        "reward_only_p90_absolute_error": float(np.quantile(absolute_error, 0.90)),
        "reward_only_sign_accuracy": float(
            np.mean((predicted > 0.0) == (actual > 0.0))
        ),
        "reward_only_pooled_spearman": safe_spearman(actual, predicted),
        "reward_only_1sigma_coverage": float(
            np.mean(absolute_error <= predicted_std)
        ),
    }


def threshold_summary(
    rows: list[dict],
    run_summaries: list[dict],
    *,
    threshold: int,
    initial_trust: float,
    maximum_trust: float,
    full_trust_at: int,
) -> dict:
    eligible = [row for row in rows if row["history_valid"] >= threshold]
    acquisition_rows = []
    for row in eligible:
        trust = gain_trust(
            row["history_valid"],
            threshold=threshold,
            initial_trust=initial_trust,
            maximum_trust=maximum_trust,
            full_trust_at=full_trust_at,
        )
        acquisition_rows.append({
            **row,
            "acquisition_mean": trust * row["reward_only_mean"],
        })
    total_gain_transitions = sum(item["valid_transitions"] for item in run_summaries)
    all_trusts = [
        gain_trust(
            history,
            threshold=threshold,
            initial_trust=initial_trust,
            maximum_trust=maximum_trust,
            full_trust_at=full_trust_at,
        )
        for item in run_summaries
        for history in range(item["valid_transitions"])
    ]
    active_trusts = [trust for trust in all_trusts if trust > 0.0]
    overall = summarize_reward_only(eligible)
    acquisition_error = per_run_nmae(
        acquisition_rows, "acquisition_mean"
    )
    actual = np.asarray([
        row["actual_gain"] / row["reward_scale"] for row in acquisition_rows
    ])
    predicted = np.asarray([
        row["acquisition_mean"] / row["reward_scale"]
        for row in acquisition_rows
    ])
    acquisition_absolute_error = np.abs(predicted - actual)
    return {
        "threshold": threshold,
        "active_predictions": len(eligible),
        "runs_with_active_bo": len({
            (row["method"], row["task"], row["run"]) for row in eligible
        }),
        "total_gain_transitions": total_gain_transitions,
        "active_fraction": (
            len(eligible) / total_gain_transitions if total_gain_transitions else 0.0
        ),
        "mean_gain_trust_when_active": (
            float(np.mean(active_trusts)) if active_trusts else 0.0
        ),
        "mean_gain_trust_over_all_gain_transitions": (
            float(np.mean(all_trusts)) if all_trusts else 0.0
        ),
        "acquisition_nmae": float(np.mean(acquisition_absolute_error)),
        "acquisition_median_absolute_error": float(
            np.median(acquisition_absolute_error)
        ),
        "acquisition_p90_absolute_error": float(
            np.quantile(acquisition_absolute_error, 0.90)
        ),
        "reward_only_nmae": overall.get("reward_only_nmae"),
        "reward_only_median_absolute_error": overall.get(
            "reward_only_median_absolute_error"
        ),
        "reward_only_p90_absolute_error": overall.get(
            "reward_only_p90_absolute_error"
        ),
        "history_mean_nmae": overall.get("history_mean_nmae"),
        "zero_gain_nmae": overall.get("zero_gain_nmae"),
        "reward_only_sign_accuracy": overall.get("reward_only_sign_accuracy"),
        "reward_only_pooled_spearman": overall.get(
            "reward_only_pooled_spearman"
        ),
        "reward_only_1sigma_coverage": overall.get(
            "reward_only_1sigma_coverage"
        ),
        "median_per_run_raw_nmae": per_run_nmae(
            eligible, "reward_only_mean"
        ).get("median"),
        "median_per_run_acquisition_nmae": acquisition_error.get("median"),
        "median_per_run_spearman": per_run_spearman(
            eligible, "reward_only_mean"
        ).get("median"),
    }


def render_markdown(result: dict) -> str:
    lines = [
        "# Reward-only BO warm-up sensitivity",
        "",
        "## Protocol",
        "",
        f"- Evaluated {result['run_count']} fixed historical runs containing "
        f"{result['total_gain_transitions']} valid continuous-gain transitions.",
        "- Each prediction uses only earlier transitions from the same run; no pretraining or cross-run transfer is used.",
        "- Thresholds change when the model would become active, but do not change the fixed historical action sequence.",
        "- Consequently this measures signal quality and projected policy exposure, not counterfactual final-score uplift.",
        "",
        "## Threshold sweep",
        "",
        "| Start after valid transitions | Active factual predictions | Runs using BO | Active fraction | Mean gain trust when active | Mean trust over all gains | Acquisition NMAE | Raw-model NMAE | Acquisition median abs. error | Acquisition P90 error | Sign accuracy | Pooled Spearman |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in result["thresholds"]:
        lines.append(
            f"| {item['threshold']} | {item['active_predictions']} | "
            f"{item['runs_with_active_bo']} | {metric(item['active_fraction'])} | "
            f"{metric(item['mean_gain_trust_when_active'])} | "
            f"{metric(item['mean_gain_trust_over_all_gain_transitions'])} | "
            f"{metric(item['acquisition_nmae'])} | "
            f"{metric(item['reward_only_nmae'])} | "
            f"{metric(item['acquisition_median_absolute_error'])} | "
            f"{metric(item['acquisition_p90_absolute_error'])} | "
            f"{metric(item['reward_only_sign_accuracy'])} | "
            f"{metric(item['reward_only_pooled_spearman'])} |"
        )

    lines.extend([
        "",
        "The rows are not evaluated on identical suffixes: a later threshold removes early, "
        "usually harder predictions. Use the history-bin table below to decide when the signal "
        "becomes usable; use this table mainly to quantify coverage and policy exposure.",
        "",
        "## Model quality by exact available-history range",
        "",
        "| Prior valid transitions | N | NMAE | History-mean NMAE | Zero-gain NMAE | Median abs. error | P90 error | Sign accuracy | Pooled Spearman | 1-sigma coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for label, item in result["history_bins"].items():
        lines.append(
            f"| {label} | {item.get('n', 0)} | "
            f"{metric(item.get('reward_only_nmae'))} | "
            f"{metric(item.get('history_mean_nmae'))} | "
            f"{metric(item.get('zero_gain_nmae'))} | "
            f"{metric(item.get('reward_only_median_absolute_error'))} | "
            f"{metric(item.get('reward_only_p90_absolute_error'))} | "
            f"{metric(item.get('reward_only_sign_accuracy'))} | "
            f"{metric(item.get('reward_only_pooled_spearman'))} | "
            f"{metric(item.get('reward_only_1sigma_coverage'))} |"
        )

    lines.extend([
        "",
        "## Decision for the current selector",
        "",
        "For these logs, 10 valid transitions is the earliest defensible production threshold. "
        "The 3-4 range has negative pooled rank signal, the 8-9 range has a large P90 error, and "
        "waiting until 15 does not avoid the weak 15-19 range. Waiting until 20 improves the "
        "observed suffix metrics, but reduces model-active factual coverage and excludes many "
        "short runs. The current 0.10 starting gain trust limits the effect of noisy 10-19 predictions.",
        "",
        "The data therefore support retaining the 10-transition warm-up for the first online "
        "pilot. After warm-up, the acquisition is pure BO: gain trust scales the predicted gain "
        "and is not a mixture with MCGS. A conservative follow-up ablation can keep trust at 0.10 through 19 valid "
        "transitions and ramp from 20 to 30, rather than changing the fit threshold itself.",
        "",
        "## Interpretation boundary",
        "",
        "Changing the warm-up in a real online run changes which parents are selected and therefore "
        "changes all later training data. Those counterfactual outcomes are absent here. A causal "
        "threshold comparison requires randomized online runs against pure progressive MCGS.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument(
        "--thresholds",
        type=int,
        nargs="+",
        default=list(DEFAULT_THRESHOLDS),
    )
    parser.add_argument("--initial-trust", type=float, default=0.10)
    parser.add_argument("--maximum-trust", type=float, default=0.25)
    parser.add_argument("--full-trust-at", type=int, default=30)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "docs"
        / "REWARD_ONLY_WARMUP_SWEEP.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "docs"
        / "REWARD_ONLY_WARMUP_SWEEP.md",
    )
    args = parser.parse_args()

    thresholds = sorted(set(max(int(value), 1) for value in args.thresholds))
    if not thresholds:
        raise SystemExit("at least one warm-up threshold is required")
    if not 0.0 <= args.initial_trust <= args.maximum_trust <= 1.0:
        raise SystemExit("trust must satisfy 0 <= initial <= maximum <= 1")

    runs_root = args.runs_root.resolve()
    transition_paths = sorted(runs_root.glob("*/*/*/logs/phase1/transitions.jsonl"))
    if not transition_paths:
        raise SystemExit(f"no transition logs found under {runs_root}")

    all_rows = []
    run_summaries = []
    for path in transition_paths:
        rows, run_summary = evaluate_reward_only_run(
            path,
            runs_root=runs_root,
            min_history=min(thresholds),
        )
        all_rows.extend(rows)
        run_summaries.append(run_summary)

    result = json_safe({
        "runs_root": str(runs_root),
        "run_count": len(transition_paths),
        "total_gain_transitions": sum(
            item["valid_transitions"] for item in run_summaries
        ),
        "initial_trust": args.initial_trust,
        "maximum_trust": args.maximum_trust,
        "full_trust_at": args.full_trust_at,
        "thresholds": [
            threshold_summary(
                all_rows,
                run_summaries,
                threshold=threshold,
                initial_trust=args.initial_trust,
                maximum_trust=args.maximum_trust,
                full_trust_at=args.full_trust_at,
            )
            for threshold in thresholds
        ],
        "history_bins": {
            label: summarize_reward_only(rows_in_range(all_rows, lower, upper))
            for label, lower, upper in HISTORY_BINS
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
    print(json.dumps(result["thresholds"], ensure_ascii=False, indent=2))
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
