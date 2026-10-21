#!/usr/bin/env python3
"""Offline replay diagnostics for the pure reward-only acquisition.

The replay can measure choice stability and agreement with logged policies, but
cannot score counterfactual choices because only the historically selected
candidate has an observed outcome.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np

from reward_only_offline_eval import (
    finite_gain,
    json_safe,
    normalized_ranks,
    read_jsonl,
    snapshot_features_by_decision,
)
from engine.parent_selectors.base import ExpansionOption, NORMAL_OPERATORS, OperatorPreview
from engine.parent_selectors.selectors import RewardOnlyThompsonSelector
from engine.parent_selectors.surrogate import RewardOnlyBayesianLinearSurrogate
from engine.parent_selectors.transition_store import TransitionRecord


def candidate_groups(snapshot_path: Path) -> dict[str, tuple[list[ExpansionOption], tuple[str, str]]]:
    grouped: dict[str, list[dict]] = {}
    for snapshot in read_jsonl(snapshot_path):
        grouped.setdefault(str(snapshot["decision_id"]), []).append(snapshot)

    result = {}
    for decision_id, snapshots in grouped.items():
        selected = [item for item in snapshots if item.get("selected")]
        if len(selected) != 1:
            continue
        utilities = [float(item.get("parent_utility") or 0.0) for item in snapshots]
        ranks = normalized_ranks(utilities)
        options = []
        for snapshot, rank in zip(snapshots, ranks):
            operator = str(snapshot.get("operator", ""))
            features = dict(snapshot.get("feature_vector") or {})
            if operator not in NORMAL_OPERATORS or not features:
                continue
            features["parent_rank_percentile"] = rank
            reference_ids = tuple(snapshot.get("reference_node_ids") or ())
            options.append(ExpansionOption(
                parent_node_id=str(snapshot["parent_node_id"]),
                operator=operator,
                reference_node_ids=reference_ids,
                state_features=features,
                parent_node=None,
                preview=OperatorPreview(operator, reference_ids),
            ))
        selected_key = (
            str(selected[0]["parent_node_id"]),
            str(selected[0]["operator"]),
        )
        if any(
            (option.parent_node_id, option.operator) == selected_key
            for option in options
        ):
            result[decision_id] = (options, selected_key)
    return result


def make_selector(
    surrogate: RewardOnlyBayesianLinearSurrogate,
    records_provider,
    *,
    temperature: float,
    random_seed: int,
) -> RewardOnlyThompsonSelector:
    return RewardOnlyThompsonSelector(
        surrogate=surrogate,
        records_provider=records_provider,
        min_valid_transitions=10,
        min_total_transitions=10,
        refit_every=1,
        initial_gain_trust=0.10,
        max_gain_trust=0.25,
        full_gain_trust_valid_transitions=30,
        thompson_temperature=temperature,
        anneal_remaining_fraction=0.10,
        random_seed=random_seed,
    )


def replay_run(
    transition_path: Path,
    *,
    runs_root: Path,
    temperatures: tuple[float, ...],
    stability_seeds: int,
) -> tuple[list[dict], dict]:
    relative = transition_path.relative_to(runs_root)
    method, task, run_name = relative.parts[:3]
    snapshots_path = transition_path.parent / "selection_snapshots.jsonl"
    selected_features = snapshot_features_by_decision(snapshots_path)
    groups = candidate_groups(snapshots_path)
    records = [
        TransitionRecord.from_dict(item) for item in read_jsonl(transition_path)
    ]

    history: list[TransitionRecord] = []
    rows = []
    errors = 0
    for record in records:
        exact_features = selected_features.get(record.decision_id)
        enriched = (
            replace(record, selection_features=exact_features)
            if exact_features is not None else record
        )
        valid_count = sum(finite_gain(item) for item in history)
        group = groups.get(record.decision_id)
        if valid_count >= 10 and group is not None and len(group[0]) >= 2:
            options, historical_key = group
            try:
                surrogate = RewardOnlyBayesianLinearSurrogate()
                surrogate.fit(history)
                provider = lambda current=history: list(current)
                results = {}
                for temperature in temperatures:
                    selector = make_selector(
                        surrogate,
                        provider,
                        temperature=temperature,
                        random_seed=0,
                    )
                    results[temperature] = selector.select(None, options)

                production_temperature = 0.25
                stability_choices = []
                for seed in range(stability_seeds):
                    selector = make_selector(
                        surrogate,
                        provider,
                        temperature=production_temperature,
                        random_seed=seed,
                    )
                    selected = selector.select(None, options).selected_option
                    stability_choices.append((selected.parent_node_id, selected.operator))

                mean_result = results[0.0]
                mean_key = (
                    mean_result.selected_option.parent_node_id,
                    mean_result.selected_option.operator,
                )
                max_parent = max(option.parent_utility for option in options)
                for temperature, result in results.items():
                    selected_option = result.selected_option
                    selected_key = (
                        selected_option.parent_node_id,
                        selected_option.operator,
                    )
                    selected_score = next(
                        score for score in result.candidate_scores
                        if (score.parent_node_id, score.operator) == selected_key
                    )
                    if any(hasattr(score, "mcgs_rank") for score in result.candidate_scores):
                        raise AssertionError("pure acquisition exposed an MCGS rank")
                    if any(hasattr(score, "bo_weight") for score in result.candidate_scores):
                        raise AssertionError("pure acquisition exposed a blend weight")
                    counts = Counter(stability_choices)
                    modal_share = max(counts.values()) / len(stability_choices)
                    rows.append({
                        "method": method,
                        "task": task,
                        "run": run_name,
                        "decision_id": record.decision_id,
                        "history_valid": valid_count,
                        "temperature": temperature,
                        "candidate_count": len(options),
                        "historical_agreement": selected_key == historical_key,
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
                        "selected_abs_gain_contribution_scaled": abs(
                            selected_score.gain_trust * selected_score.sampled_gain
                        ) / surrogate.reward_scale,
                        "max_candidate_gain_contribution_scaled": max(
                            score.gain_trust * score.sampled_gain
                            / surrogate.reward_scale
                            for score in result.candidate_scores
                        ),
                        "max_candidate_abs_gain_contribution_scaled": max(
                            abs(score.gain_trust * score.sampled_gain)
                            / surrogate.reward_scale
                            for score in result.candidate_scores
                        ),
                        "gain_trust": selected_score.gain_trust,
                        "known_factual_outcome": finite_gain(enriched),
                        "factual_positive_gain": (
                            float(enriched.improvement) > 0.0
                            if finite_gain(enriched) else None
                        ),
                        "stability_modal_share": modal_share,
                        "stability_unique_choices": len(counts),
                    })
            except Exception:
                errors += 1
        history.append(enriched)

    return rows, {
        "method": method,
        "task": task,
        "run": run_name,
        "replayed_decisions": len({row["decision_id"] for row in rows}),
        "errors": errors,
    }


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"decisions": 0}
    agreements = [row for row in rows if row["historical_agreement"]]
    known_agreements = [row for row in agreements if row["known_factual_outcome"]]
    return {
        "decisions": len(rows),
        "runs": len({(row["method"], row["task"], row["run"]) for row in rows}),
        "mean_candidate_count": float(np.mean([row["candidate_count"] for row in rows])),
        "historical_agreement_rate": float(np.mean([
            row["historical_agreement"] for row in rows
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
        "p90_selected_abs_gain_contribution_scaled": float(np.quantile([
            row["selected_abs_gain_contribution_scaled"] for row in rows
        ], 0.90)),
        "p99_selected_abs_gain_contribution_scaled": float(np.quantile([
            row["selected_abs_gain_contribution_scaled"] for row in rows
        ], 0.99)),
        "max_selected_abs_gain_contribution_scaled": float(max(
            row["selected_abs_gain_contribution_scaled"] for row in rows
        )),
        "p99_max_candidate_gain_contribution_scaled": float(np.quantile([
            row["max_candidate_gain_contribution_scaled"] for row in rows
        ], 0.99)),
        "max_candidate_gain_contribution_scaled": float(max(
            row["max_candidate_gain_contribution_scaled"] for row in rows
        )),
        "p99_max_candidate_abs_gain_contribution_scaled": float(np.quantile([
            row["max_candidate_abs_gain_contribution_scaled"] for row in rows
        ], 0.99)),
        "factual_matches": len(known_agreements),
        "positive_gain_rate_when_factual_match": (
            float(np.mean([row["factual_positive_gain"] for row in known_agreements]))
            if known_agreements else None
        ),
        "median_stability_modal_share_at_production_temperature": float(np.median([
            row["stability_modal_share"] for row in rows
        ])),
        "mean_unique_choices_across_stability_seeds": float(np.mean([
            row["stability_unique_choices"] for row in rows
        ])),
    }


def render_markdown(result: dict) -> str:
    lines = [
        "# Pure reward-only acquisition offline replay",
        "",
        "## Protocol",
        "",
        f"- Replayed multi-candidate decisions from {result['run_count']} fixed historical runs after 10 valid transitions.",
        "- The production acquisition is parent utility plus a gain-trust-scaled joint Thompson gain; no MCGS rank enters the score and no clipping is applied.",
        "- Temperature 0 is the conservative posterior-mean acquisition. Other rows use the same posterior draw seed with increasing exploration.",
        "- Historical agreement exposes a factual outcome only when replay selects the same action as the logged policy; it is not a counterfactual policy score.",
        "",
        "## Replay results",
        "",
        "| Thompson temperature | Decisions | Historical agreement | Agreement with posterior mean | Select max-utility parent | Median parent sacrifice / scale | P90 sacrifice / scale | P90 / P99 selected abs gain contribution | P99 largest positive / abs candidate contribution | Factual matches | Positive gain on matches |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, item in result["by_temperature"].items():
        lines.append(
            f"| {key} | {item['decisions']} | {item['historical_agreement_rate']:.3f} | "
            f"{item['posterior_mean_agreement_rate']:.3f} | "
            f"{item['max_parent_utility_rate']:.3f} | "
            f"{item['median_parent_utility_sacrifice_scaled']:.3f} | "
            f"{item['p90_parent_utility_sacrifice_scaled']:.3f} | "
            f"{item['p90_selected_abs_gain_contribution_scaled']:.3f} / "
            f"{item['p99_selected_abs_gain_contribution_scaled']:.3f} | "
            f"{item['p99_max_candidate_gain_contribution_scaled']:.3f} / "
            f"{item['p99_max_candidate_abs_gain_contribution_scaled']:.3f} | "
            f"{item['factual_matches']} | "
            f"{item['positive_gain_rate_when_factual_match']:.3f} |"
        )
    production = result["by_temperature"]["0.25"]
    lines.extend([
        "",
        "## Stability of the production temperature",
        "",
        f"Across {result['stability_seeds']} deterministic posterior seeds per decision, "
        f"the median modal-choice share is {production['median_stability_modal_share_at_production_temperature']:.3f} "
        f"and the mean number of unique choices is {production['mean_unique_choices_across_stability_seeds']:.3f}.",
        "",
        f"With no clipping, the production temperature has a P99 / maximum selected absolute "
        f"gain contribution of {production['p99_selected_abs_gain_contribution_scaled']:.3f} / "
        f"{production['max_selected_abs_gain_contribution_scaled']:.3f} run-local scales. The P99 / maximum "
        f"largest positive contribution present in a candidate set is "
        f"{production['p99_max_candidate_gain_contribution_scaled']:.3f} / "
        f"{production['max_candidate_gain_contribution_scaled']:.3f} scales.",
        "",
        "## Interpretation boundary",
        "",
        "Replay choice changes cannot be assigned outcomes when they differ from the logged action. "
        "The report therefore validates acquisition scale, unbounded-tail exposure, and stochastic stability only. "
        "Final-score comparison against progressive MCGS requires online paired runs.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.50, 1.0],
    )
    parser.add_argument("--stability-seeds", type=int, default=16)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "docs"
        / "REWARD_ONLY_ACQUISITION_REPLAY.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "docs"
        / "REWARD_ONLY_ACQUISITION_REPLAY.md",
    )
    args = parser.parse_args()
    temperatures = tuple(sorted(set(float(value) for value in args.temperatures)))
    if 0.0 not in temperatures or 0.25 not in temperatures:
        raise SystemExit("temperatures must include 0 and the production value 0.25")

    runs_root = args.runs_root.resolve()
    paths = sorted(runs_root.glob("*/*/*/logs/phase1/transitions.jsonl"))
    if not paths:
        raise SystemExit(f"no transition logs found under {runs_root}")
    all_rows = []
    run_summaries = []
    for path in paths:
        rows, summary = replay_run(
            path,
            runs_root=runs_root,
            temperatures=temperatures,
            stability_seeds=max(int(args.stability_seeds), 1),
        )
        all_rows.extend(rows)
        run_summaries.append(summary)

    result = json_safe({
        "runs_root": str(runs_root),
        "run_count": len(paths),
        "stability_seeds": max(int(args.stability_seeds), 1),
        "production_temperature": 0.25,
        "replay_errors": sum(item["errors"] for item in run_summaries),
        "by_temperature": {
            f"{temperature:.2f}": summarize([
                row for row in all_rows if row["temperature"] == temperature
            ])
            for temperature in temperatures
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
    print(json.dumps(result["by_temperature"], ensure_ascii=False, indent=2))
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
