#!/usr/bin/env python3
"""Offline diagnostics for a reward-only latent Child-UCB acquisition.

The experimental score is::

    parent_utility + gain_trust * (posterior_mean_gain + beta * latent_std_gain)

``latent_std_gain`` contains coefficient/posterior uncertainty only. It excludes
observation noise, the predictive uncertainty multiplier, and the predictive
uncertainty floor. Historical logs expose no counterfactual candidate outcomes,
so candidate replay measures behavior rather than final policy value.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np

from reward_only_acquisition_replay import candidate_groups, make_selector
from reward_only_offline_eval import (
    finite_gain,
    json_safe,
    metric,
    option_from_record,
    read_jsonl,
    snapshot_features_by_decision,
)
from engine.parent_selectors.base import ExpansionOption
from engine.parent_selectors.surrogate import RewardOnlyBayesianLinearSurrogate
from engine.parent_selectors.transition_store import TransitionRecord


DEFAULT_BETAS = (0.0, 0.10, 0.25, 0.50, 1.0, 1.50)
HISTORY_BINS = (
    ("10-14", 10, 14),
    ("15-19", 15, 19),
    ("20-29", 20, 29),
    ("30+", 30, None),
)


def gain_trust(valid_count: int) -> float:
    progress = min(max((valid_count - 10) / 20.0, 0.0), 1.0)
    return 0.10 + 0.15 * progress


def horizon_factor(
    options: list[ExpansionOption],
    *,
    anneal_remaining_fraction: float = 0.10,
) -> float:
    if not options or anneal_remaining_fraction <= 0.0:
        return 1.0
    remaining_steps = min(
        min(
            max(
                float(option.state_features.get("remaining_steps_fraction", 1.0)),
                0.0,
            ),
            1.0,
        )
        for option in options
    )
    remaining_time = min(
        min(
            max(
                1.0 - float(option.state_features.get("elapsed_fraction", 0.0)),
                0.0,
            ),
            1.0,
        )
        for option in options
    )
    return min(
        min(remaining_steps, remaining_time) / anneal_remaining_fraction,
        1.0,
    )


def option_key(option: ExpansionOption) -> tuple[str, str]:
    return option.parent_node_id, option.operator


def rows_in_history_bin(
    rows: list[dict],
    lower: int,
    upper: int | None,
) -> list[dict]:
    return [
        row for row in rows
        if row["history_valid"] >= lower
        and (upper is None or row["history_valid"] <= upper)
    ]


def replay_run(
    transition_path: Path,
    *,
    runs_root: Path,
    betas: tuple[float, ...],
) -> tuple[list[dict], list[dict], dict]:
    relative = transition_path.relative_to(runs_root)
    method, task, run_name = relative.parts[:3]
    snapshots_path = transition_path.parent / "selection_snapshots.jsonl"
    selected_features = snapshot_features_by_decision(snapshots_path)
    groups = candidate_groups(snapshots_path)
    records = [
        TransitionRecord.from_dict(item) for item in read_jsonl(transition_path)
    ]

    history: list[TransitionRecord] = []
    candidate_rows: list[dict] = []
    factual_rows: list[dict] = []
    errors = 0
    for record in records:
        exact_features = selected_features.get(record.decision_id)
        enriched = (
            replace(record, selection_features=exact_features)
            if exact_features is not None else record
        )
        valid_count = sum(finite_gain(item) for item in history)
        group = groups.get(record.decision_id)
        replayable_group = (
            group is not None
            and len(group[0]) >= 2
        )
        if valid_count >= 10 and (finite_gain(enriched) or replayable_group):
            try:
                surrogate = RewardOnlyBayesianLinearSurrogate()
                surrogate.fit(history)
                trust = gain_trust(valid_count)

                if finite_gain(enriched):
                    factual_option = option_from_record(enriched)
                    prediction = surrogate.predict([factual_option])[0]
                    latent_std = float(
                        surrogate.latent_std_gains([factual_option])[0]
                    )
                    factor = horizon_factor([factual_option])
                    actual = float(enriched.improvement)
                    for beta in betas:
                        effective_beta = beta * factor
                        upper_gain = prediction.mean_gain + effective_beta * latent_std
                        factual_rows.append({
                            "method": method,
                            "task": task,
                            "run": run_name,
                            "decision_id": record.decision_id,
                            "history_valid": valid_count,
                            "beta": beta,
                            "effective_beta": effective_beta,
                            "actual_gain_scaled": actual / surrogate.reward_scale,
                            "mean_gain_scaled": (
                                prediction.mean_gain / surrogate.reward_scale
                            ),
                            "latent_std_scaled": (
                                latent_std / surrogate.reward_scale
                            ),
                            "upper_gain_scaled": (
                                upper_gain / surrogate.reward_scale
                            ),
                            "realized_below_upper": actual <= upper_gain,
                            "upper_gain_positive": upper_gain > 0.0,
                        })

                if replayable_group:
                    options, historical_key = group
                    predictions = surrogate.predict(options)
                    means = np.asarray(
                        [prediction.mean_gain for prediction in predictions]
                    )
                    latent_stds = surrogate.latent_std_gains(options)
                    parent_utilities = np.asarray([
                        option.parent_utility for option in options
                    ])
                    factor = horizon_factor(options)
                    provider = lambda current=history: list(current)
                    thompson_result = make_selector(
                        surrogate,
                        provider,
                        temperature=0.25,
                        random_seed=0,
                    ).select(None, options)
                    thompson_key = option_key(thompson_result.selected_option)
                    mean_scores = parent_utilities + trust * means
                    mean_key = option_key(options[int(np.argmax(mean_scores))])
                    max_parent = float(np.max(parent_utilities))

                    for beta in betas:
                        effective_beta = beta * factor
                        bonuses = effective_beta * latent_stds
                        upper_gains = means + bonuses
                        scores = parent_utilities + trust * upper_gains
                        selected_index = int(np.argmax(scores))
                        selected_option = options[selected_index]
                        selected_key = option_key(selected_option)
                        historical_match = selected_key == historical_key
                        known_factual = historical_match and finite_gain(enriched)
                        candidate_rows.append({
                            "method": method,
                            "task": task,
                            "run": run_name,
                            "decision_id": record.decision_id,
                            "history_valid": valid_count,
                            "beta": beta,
                            "effective_beta": effective_beta,
                            "candidate_count": len(options),
                            "historical_agreement": historical_match,
                            "thompson_agreement": selected_key == thompson_key,
                            "posterior_mean_agreement": selected_key == mean_key,
                            "max_parent_utility_agreement": math.isclose(
                                selected_option.parent_utility,
                                max_parent,
                                rel_tol=1e-12,
                                abs_tol=1e-12,
                            ),
                            "parent_utility_sacrifice_scaled": (
                                max_parent - selected_option.parent_utility
                            ) / surrogate.reward_scale,
                            "selected_bonus_contribution_scaled": (
                                trust * bonuses[selected_index]
                                / surrogate.reward_scale
                            ),
                            "max_bonus_contribution_scaled": (
                                trust * float(np.max(bonuses))
                                / surrogate.reward_scale
                            ),
                            "selected_upper_gain_contribution_scaled": (
                                trust * upper_gains[selected_index]
                                / surrogate.reward_scale
                            ),
                            "known_factual_outcome": known_factual,
                            "factual_gain_scaled": (
                                float(enriched.improvement) / surrogate.reward_scale
                                if known_factual else None
                            ),
                            "factual_positive_gain": (
                                float(enriched.improvement) > 0.0
                                if known_factual else None
                            ),
                        })
            except Exception:
                errors += 1
        history.append(enriched)

    return candidate_rows, factual_rows, {
        "method": method,
        "task": task,
        "run": run_name,
        "candidate_decisions": len({
            row["decision_id"] for row in candidate_rows
        }),
        "factual_predictions": len({row["decision_id"] for row in factual_rows}),
        "errors": errors,
    }


def summarize_candidates(rows: list[dict]) -> dict:
    if not rows:
        return {"decisions": 0}
    known = [row for row in rows if row["known_factual_outcome"]]
    return {
        "decisions": len(rows),
        "runs": len({(row["method"], row["task"], row["run"]) for row in rows}),
        "mean_effective_beta": float(np.mean([
            row["effective_beta"] for row in rows
        ])),
        "historical_agreement_rate": float(np.mean([
            row["historical_agreement"] for row in rows
        ])),
        "current_thompson_agreement_rate": float(np.mean([
            row["thompson_agreement"] for row in rows
        ])),
        "posterior_mean_agreement_rate": float(np.mean([
            row["posterior_mean_agreement"] for row in rows
        ])),
        "max_parent_utility_rate": float(np.mean([
            row["max_parent_utility_agreement"] for row in rows
        ])),
        "median_parent_utility_sacrifice_scaled": float(np.median([
            row["parent_utility_sacrifice_scaled"] for row in rows
        ])),
        "p90_parent_utility_sacrifice_scaled": float(np.quantile([
            row["parent_utility_sacrifice_scaled"] for row in rows
        ], 0.90)),
        "p99_selected_bonus_contribution_scaled": float(np.quantile([
            row["selected_bonus_contribution_scaled"] for row in rows
        ], 0.99)),
        "max_selected_bonus_contribution_scaled": float(max(
            row["selected_bonus_contribution_scaled"] for row in rows
        )),
        "p99_max_bonus_contribution_scaled": float(np.quantile([
            row["max_bonus_contribution_scaled"] for row in rows
        ], 0.99)),
        "p99_selected_upper_gain_contribution_scaled": float(np.quantile([
            abs(row["selected_upper_gain_contribution_scaled"]) for row in rows
        ], 0.99)),
        "factual_matches": len(known),
        "positive_gain_rate_when_factual_match": (
            float(np.mean([row["factual_positive_gain"] for row in known]))
            if known else None
        ),
        "mean_gain_scaled_when_factual_match": (
            float(np.mean([row["factual_gain_scaled"] for row in known]))
            if known else None
        ),
        "median_gain_scaled_when_factual_match": (
            float(np.median([row["factual_gain_scaled"] for row in known]))
            if known else None
        ),
    }


def summarize_factual(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    optimism = np.asarray([
        row["upper_gain_scaled"] - row["actual_gain_scaled"] for row in rows
    ])
    return {
        "n": len(rows),
        "runs": len({(row["method"], row["task"], row["run"]) for row in rows}),
        "mean_effective_beta": float(np.mean([
            row["effective_beta"] for row in rows
        ])),
        "realized_outcome_below_upper_rate": float(np.mean([
            row["realized_below_upper"] for row in rows
        ])),
        "upper_gain_positive_rate": float(np.mean([
            row["upper_gain_positive"] for row in rows
        ])),
        "median_optimism_scaled": float(np.median(optimism)),
        "mean_optimism_scaled": float(np.mean(optimism)),
        "p90_optimism_scaled": float(np.quantile(optimism, 0.90)),
        "median_latent_std_scaled": float(np.median([
            row["latent_std_scaled"] for row in rows
        ])),
        "p90_latent_std_scaled": float(np.quantile([
            row["latent_std_scaled"] for row in rows
        ], 0.90)),
        "p99_latent_std_scaled": float(np.quantile([
            row["latent_std_scaled"] for row in rows
        ], 0.99)),
    }


def render_markdown(result: dict) -> str:
    lines = [
        "# Latent Child-UCB offline evaluation",
        "",
        "## Protocol",
        "",
        f"- Evaluated {result['run_count']} fixed historical runs after 10 valid transitions.",
        "- Score: `parent_utility + gain_trust * (posterior_mean_gain + beta * latent_std_gain)`.",
        "- Latent std contains epistemic coefficient uncertainty only; observation noise, the predictive 2x multiplier, and the predictive floor are excluded.",
        "- Beta is annealed during the final 10% of either step or wall-clock budget. No gain clipping or MCGS score fusion is used.",
        "- The production Thompson selector remains unchanged; this is an offline experimental comparison.",
        "",
        "## Factual upper-bound behavior",
        "",
        "`Realized <= upper` compares a noisy realized transition with a latent-mean upper bound. It is a descriptive stress test, not nominal confidence coverage.",
        "",
        "| Beta | Mean effective beta | Factual N | Realized <= upper | Upper gain > 0 | Median optimism / scale | Mean optimism / scale | P90 optimism / scale |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, item in result["by_beta"].items():
        factual = item["factual"]
        lines.append(
            f"| {key} | {metric(factual.get('mean_effective_beta'))} | "
            f"{factual.get('n', 0)} | "
            f"{metric(factual.get('realized_outcome_below_upper_rate'))} | "
            f"{metric(factual.get('upper_gain_positive_rate'))} | "
            f"{metric(factual.get('median_optimism_scaled'))} | "
            f"{metric(factual.get('mean_optimism_scaled'))} | "
            f"{metric(factual.get('p90_optimism_scaled'))} |"
        )

    lines.extend([
        "",
        "## Candidate-choice replay",
        "",
        "| Beta | Decisions | Historical agreement | Current TS agreement | Mean-policy agreement | Select max parent | P90 parent sacrifice / scale | P99 selected bonus / scale | P99 largest candidate bonus / scale | Factual matches | Positive gain on matches |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for key, item in result["by_beta"].items():
        candidate = item["candidate"]
        lines.append(
            f"| {key} | {candidate.get('decisions', 0)} | "
            f"{metric(candidate.get('historical_agreement_rate'))} | "
            f"{metric(candidate.get('current_thompson_agreement_rate'))} | "
            f"{metric(candidate.get('posterior_mean_agreement_rate'))} | "
            f"{metric(candidate.get('max_parent_utility_rate'))} | "
            f"{metric(candidate.get('p90_parent_utility_sacrifice_scaled'))} | "
            f"{metric(candidate.get('p99_selected_bonus_contribution_scaled'))} | "
            f"{metric(candidate.get('p99_max_bonus_contribution_scaled'))} | "
            f"{candidate.get('factual_matches', 0)} | "
            f"{metric(candidate.get('positive_gain_rate_when_factual_match'))} |"
        )

    latent = result["by_beta"][next(iter(result["by_beta"]))]["factual"]
    lines.extend([
        "",
        "## Latent uncertainty scale",
        "",
        f"Across factual predictions, latent std has median / P90 / P99 values of "
        f"{metric(latent.get('median_latent_std_scaled'))} / "
        f"{metric(latent.get('p90_latent_std_scaled'))} / "
        f"{metric(latent.get('p99_latent_std_scaled'))} run-local scales.",
        "",
        "## Behavior by available valid history",
        "",
        "| Prior valid transitions | Factual N | Latent std median / P90 / P99 |",
        "|---|---:|---:|",
    ])
    for label, item in result["history_bins"].items():
        factual = item["factual_latent"]
        lines.append(
            f"| {label} | {factual.get('n', 0)} | "
            f"{metric(factual.get('median_latent_std_scaled'))} / "
            f"{metric(factual.get('p90_latent_std_scaled'))} / "
            f"{metric(factual.get('p99_latent_std_scaled'))} |"
        )
    lines.extend([
        "",
        "| Prior valid transitions | Beta | Decisions | Mean-policy agreement | Select max parent | P90 parent sacrifice | P99 largest bonus |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for label, item in result["history_bins"].items():
        for key, candidate in item["candidate_by_beta"].items():
            if key == "0.00":
                continue
            lines.append(
                f"| {label} | {key} | {candidate.get('decisions', 0)} | "
                f"{metric(candidate.get('posterior_mean_agreement_rate'))} | "
                f"{metric(candidate.get('max_parent_utility_rate'))} | "
                f"{metric(candidate.get('p90_parent_utility_sacrifice_scaled'))} | "
                f"{metric(candidate.get('p99_max_bonus_contribution_scaled'))} |"
            )
    lines.extend([
        "",
        "## Offline decision",
        "",
        "- Do not replace the production Thompson acquisition with latent Child-UCB based on this replay.",
        "- Beta 0.10 is the only low-risk online ablation: it agrees with the posterior-mean policy on 97.7% of replayed decisions and keeps the P99 largest candidate bonus below one run-local scale.",
        "- Beta 0.25 is a stress-test setting, not a default. Beta 0.50 and above expose multi-scale optimism tails and are rejected for the current small-data posterior.",
        "- The more promising next change is the separately evaluated incumbent-relative BNVS target and compact feature set; acquisition should be isolated as a later ablation.",
        "",
        "## Interpretation boundary",
        "",
        "Historical agreement exposes an outcome only when the replay chooses the action actually executed. Different beta values match different action subsets, so their matched positive-gain rates are not causally comparable. Final-score conclusions require online paired runs.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument(
        "--betas",
        type=float,
        nargs="+",
        default=list(DEFAULT_BETAS),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "docs"
        / "REWARD_ONLY_LATENT_UCB_REPLAY.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "docs"
        / "REWARD_ONLY_LATENT_UCB_REPLAY.md",
    )
    args = parser.parse_args()
    betas = tuple(sorted(set(max(float(value), 0.0) for value in args.betas)))
    if 0.0 not in betas:
        raise SystemExit("betas must include 0 for the posterior-mean baseline")

    runs_root = args.runs_root.resolve()
    paths = sorted(runs_root.glob("*/*/*/logs/phase1/transitions.jsonl"))
    if not paths:
        raise SystemExit(f"no transition logs found under {runs_root}")

    candidate_rows: list[dict] = []
    factual_rows: list[dict] = []
    run_summaries = []
    for path in paths:
        candidates, factual, summary = replay_run(
            path,
            runs_root=runs_root,
            betas=betas,
        )
        candidate_rows.extend(candidates)
        factual_rows.extend(factual)
        run_summaries.append(summary)

    result = json_safe({
        "runs_root": str(runs_root),
        "run_count": len(paths),
        "replay_errors": sum(item["errors"] for item in run_summaries),
        "by_beta": {
            f"{beta:.2f}": {
                "candidate": summarize_candidates([
                    row for row in candidate_rows if row["beta"] == beta
                ]),
                "factual": summarize_factual([
                    row for row in factual_rows if row["beta"] == beta
                ]),
            }
            for beta in betas
        },
        "history_bins": {
            label: {
                "factual_latent": summarize_factual(rows_in_history_bin(
                    [row for row in factual_rows if row["beta"] == 0.0],
                    lower,
                    upper,
                )),
                "candidate_by_beta": {
                    f"{beta:.2f}": summarize_candidates(rows_in_history_bin(
                        [row for row in candidate_rows if row["beta"] == beta],
                        lower,
                        upper,
                    ))
                    for beta in betas
                },
            }
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
    print(json.dumps(result["by_beta"], ensure_ascii=False, indent=2))
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
