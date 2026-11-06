"""Concrete legacy, random, greedy and Bayesian parent selectors."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import math
import random
from typing import Callable, Sequence
import uuid

from engine import node_selection
from engine.parent_selectors.acquisition import score_predictions
from engine.parent_selectors.base import (
    CandidateScore,
    ExpansionOption,
    SelectionDecision,
    SelectorResult,
    utc_timestamp,
)
from engine.parent_selectors.surrogate import (
    BayesianNodeValueSurrogate,
    BayesianTransitionSurrogate,
    RewardOnlyBayesianLinearSurrogate,
)
from engine.parent_selectors.transition_store import TransitionRecord


class SelectionOutsideOptions(RuntimeError):
    def __init__(self, node, *, fallback_reason: str | None = None, selector_name: str | None = None):
        super().__init__(f"selected node {node.id} is outside normal BO options")
        self.node = node
        self.fallback_reason = fallback_reason
        self.selector_name = selector_name


def _decision(
    *,
    selector_name: str,
    option: ExpansionOption,
    selected_score: float,
    model_version: str | None = None,
    fallback_used: bool = False,
    fallback_reason: str | None = None,
) -> SelectionDecision:
    return SelectionDecision(
        selector_name=selector_name,
        selected_parent_id=option.parent_node_id,
        selected_operator=option.operator,
        selected_score=float(selected_score),
        model_version=model_version,
        fallback_used=fallback_used,
        decision_id=uuid.uuid4().hex,
        timestamp=utc_timestamp(),
        fallback_reason=fallback_reason,
    )


def _simple_score(option: ExpansionOption, score: float | None) -> CandidateScore:
    return CandidateScore(
        parent_node_id=option.parent_node_id,
        operator=option.operator,
        reference_node_ids=option.reference_node_ids,
        parent_utility=option.parent_utility,
        mean_gain=None,
        std_gain=None,
        predicted_child_ucb=None,
        predicted_cost=None,
        p_valid=None,
        acquisition_score=score,
    )


class ProgressiveMCGSSelector:
    """Adapter over the repository's original soft-switch selector."""

    name = "progressive_mcgs"

    def select(self, agent, options: Sequence[ExpansionOption]) -> SelectorResult:
        selected_node = node_selection.select_with_soft_switch(agent)
        option_by_id = {option.parent_node_id: option for option in options}
        if selected_node.id not in option_by_id:
            raise SelectionOutsideOptions(selected_node)
        selected_option = option_by_id[selected_node.id]

        candidate_scores = []
        for option in options:
            node = option.parent_node
            visits = float(getattr(node, "visits", 0) or 0)
            score = None
            if visits > 0:
                score = float(getattr(node, "total_reward", 0.0) or 0.0) / visits
                if not math.isfinite(score):
                    score = None
            candidate_scores.append(_simple_score(option, score))
        selected_candidate = next(
            score for score in candidate_scores
            if score.parent_node_id == selected_option.parent_node_id
        )
        selected_score = selected_candidate.acquisition_score or 0.0
        return SelectorResult(
            decision=_decision(
                selector_name=self.name,
                option=selected_option,
                selected_score=selected_score,
            ),
            selected_option=selected_option,
            candidate_scores=tuple(candidate_scores),
            options=tuple(options),
        )


class RandomParentSelector:
    name = "random_parent"

    def select(self, agent, options: Sequence[ExpansionOption]) -> SelectorResult:
        del agent
        if not options:
            raise ValueError("random selector received no options")
        selected_option = random.choice(list(options))
        scores = tuple(_simple_score(option, 0.0) for option in options)
        return SelectorResult(
            decision=_decision(
                selector_name=self.name,
                option=selected_option,
                selected_score=0.0,
            ),
            selected_option=selected_option,
            candidate_scores=scores,
            options=tuple(options),
        )


class ParentMetricGreedySelector:
    name = "parent_metric_greedy"

    def select(self, agent, options: Sequence[ExpansionOption]) -> SelectorResult:
        del agent
        if not options:
            raise ValueError("metric-greedy selector received no options")
        scores = tuple(_simple_score(option, option.parent_utility) for option in options)
        best_index = max(range(len(options)), key=lambda index: options[index].parent_utility)
        selected_option = options[best_index]
        return SelectorResult(
            decision=_decision(
                selector_name=self.name,
                option=selected_option,
                selected_score=selected_option.parent_utility,
            ),
            selected_option=selected_option,
            candidate_scores=scores,
            options=tuple(options),
        )


class BayesianUCBParentSelector:
    def __init__(
        self,
        *,
        name: str,
        acquisition_type: str,
        surrogate: BayesianTransitionSurrogate,
        records_provider: Callable[[], list[TransitionRecord]],
        min_valid_transitions: int,
        min_total_transitions: int,
        min_operator_observations: int,
        refit_every: int,
        beta: float,
        use_validity_probability: bool,
        cost_power: float,
    ):
        self.name = name
        self.acquisition_type = acquisition_type
        self.surrogate = surrogate
        self.records_provider = records_provider
        self.min_valid_transitions = int(min_valid_transitions)
        self.min_total_transitions = int(min_total_transitions)
        self.min_operator_observations = int(min_operator_observations)
        self.refit_every = max(int(refit_every), 1)
        self.beta = float(beta)
        self.use_validity_probability = bool(use_validity_probability)
        self.cost_power = float(cost_power)
        self.fallback = ProgressiveMCGSSelector()

    def _fallback(
        self,
        agent,
        options: Sequence[ExpansionOption],
        reason: str,
    ) -> SelectorResult:
        try:
            result = self.fallback.select(agent, options)
        except SelectionOutsideOptions as exc:
            exc.fallback_reason = reason
            exc.selector_name = self.name
            raise
        decision = replace(
            result.decision,
            selector_name=self.name,
            fallback_used=True,
            fallback_reason=reason,
        )
        return replace(result, decision=decision)

    def cold_start_reason(
        self,
        records: Sequence[TransitionRecord],
        options: Sequence[ExpansionOption],
    ) -> str | None:
        valid = [record for record in records if record.valid and record.improvement is not None]
        if len(records) < self.min_total_transitions:
            return f"total_transitions={len(records)}<{self.min_total_transitions}"
        if len(valid) < self.min_valid_transitions:
            return f"valid_transitions={len(valid)}<{self.min_valid_transitions}"
        relevant = {option.operator for option in options}
        counts = {
            operator: sum(record.operator == operator for record in valid)
            for operator in relevant
        }
        if counts and min(counts.values()) < self.min_operator_observations:
            return (
                f"operator_observations={min(counts.values())}"
                f"<{self.min_operator_observations}"
            )
        return None

    def select(self, agent, options: Sequence[ExpansionOption]) -> SelectorResult:
        if not options:
            raise ValueError("Bayesian selector received no options")
        records = self.records_provider()
        cold_start_reason = self.cold_start_reason(records, options)
        if cold_start_reason:
            return self._fallback(agent, options, cold_start_reason)

        try:
            if (
                self.surrogate.fitted_transition_count < 0
                or len(records) - self.surrogate.fitted_transition_count >= self.refit_every
            ):
                self.surrogate.fit(records)
            predictions = self.surrogate.predict(options)
            candidate_scores = score_predictions(
                options,
                predictions,
                acquisition_type=self.acquisition_type,
                beta=self.beta,
                use_validity_probability=self.use_validity_probability,
                cost_power=self.cost_power,
            )
            best_index = max(
                range(len(candidate_scores)),
                key=lambda index: candidate_scores[index].acquisition_score,
            )
            selected_option = options[best_index]
            selected_score = candidate_scores[best_index].acquisition_score
            if selected_score is None or not math.isfinite(selected_score):
                raise ValueError("selected Bayesian acquisition is not finite")
            return SelectorResult(
                decision=_decision(
                    selector_name=self.name,
                    option=selected_option,
                    selected_score=selected_score,
                    model_version=self.surrogate.model_version,
                ),
                selected_option=selected_option,
                candidate_scores=candidate_scores,
                options=tuple(options),
            )
        except SelectionOutsideOptions:
            raise
        except Exception as exc:
            return self._fallback(agent, options, f"model_error:{type(exc).__name__}:{exc}")


class RewardOnlyThompsonSelector:
    """Pure reward-only Thompson child-utility acquisition with safety shrinkage."""

    name = "reward_only_bayesian_ts"

    def __init__(
        self,
        *,
        surrogate: RewardOnlyBayesianLinearSurrogate,
        records_provider: Callable[[], list[TransitionRecord]],
        min_valid_transitions: int,
        min_total_transitions: int,
        refit_every: int,
        initial_gain_trust: float,
        max_gain_trust: float,
        full_gain_trust_valid_transitions: int,
        thompson_temperature: float,
        anneal_remaining_fraction: float,
        random_seed: int,
    ):
        self.surrogate = surrogate
        self.records_provider = records_provider
        self.min_valid_transitions = max(int(min_valid_transitions), 1)
        self.min_total_transitions = max(int(min_total_transitions), 1)
        self.refit_every = max(int(refit_every), 1)
        self.initial_gain_trust = min(max(float(initial_gain_trust), 0.0), 1.0)
        self.max_gain_trust = min(max(float(max_gain_trust), 0.0), 1.0)
        if self.initial_gain_trust > self.max_gain_trust:
            raise ValueError("initial gain trust exceeds maximum")
        self.full_gain_trust_valid_transitions = max(
            int(full_gain_trust_valid_transitions),
            self.min_valid_transitions,
        )
        self.thompson_temperature = min(
            max(float(thompson_temperature), 0.0),
            1.0,
        )
        self.anneal_remaining_fraction = max(float(anneal_remaining_fraction), 0.0)
        self.random_seed = int(random_seed)
        self.fallback = ProgressiveMCGSSelector()

    @staticmethod
    def _valid_records(records: Sequence[TransitionRecord]) -> list[TransitionRecord]:
        return [
            record for record in records
            if record.valid and record.improvement is not None
        ]

    def cold_start_reason(self, records: Sequence[TransitionRecord], options=None) -> str | None:
        del options
        valid = self._valid_records(records)
        if len(records) < self.min_total_transitions:
            return f"total_transitions={len(records)}<{self.min_total_transitions}"
        if len(valid) < self.min_valid_transitions:
            return f"valid_transitions={len(valid)}<{self.min_valid_transitions}"
        return None

    def _fallback(
        self,
        agent,
        options: Sequence[ExpansionOption],
        reason: str,
    ) -> SelectorResult:
        try:
            result = self.fallback.select(agent, options)
        except SelectionOutsideOptions as exc:
            exc.fallback_reason = reason
            exc.selector_name = self.name
            raise
        return replace(
            result,
            decision=replace(
                result.decision,
                selector_name=self.name,
                fallback_used=True,
                fallback_reason=reason,
            ),
        )

    def _gain_trust(self, valid_count: int) -> float:
        if self.full_gain_trust_valid_transitions <= self.min_valid_transitions:
            return self.max_gain_trust
        progress = (
            valid_count - self.min_valid_transitions
        ) / (
            self.full_gain_trust_valid_transitions - self.min_valid_transitions
        )
        progress = min(max(progress, 0.0), 1.0)
        return self.initial_gain_trust + progress * (
            self.max_gain_trust - self.initial_gain_trust
        )

    def _temperature(self, options: Sequence[ExpansionOption]) -> float:
        if not options or self.anneal_remaining_fraction <= 0.0:
            return 1.0
        remaining = min(
            min(max(float(option.state_features.get("remaining_steps_fraction", 1.0)), 0.0), 1.0)
            for option in options
        )
        remaining_time = min(
            min(max(1.0 - float(option.state_features.get("elapsed_fraction", 0.0)), 0.0), 1.0)
            for option in options
        )
        remaining_budget = min(remaining, remaining_time)
        return min(remaining_budget / self.anneal_remaining_fraction, 1.0)

    def _stable_seed(
        self,
        records: Sequence[TransitionRecord],
        options: Sequence[ExpansionOption],
    ) -> int:
        parts = [str(self.random_seed), str(len(records))]
        parts.extend(record.transition_id for record in records[-3:])
        parts.extend(
            f"{option.parent_node_id}:{option.operator}"
            for option in options
        )
        digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big", signed=False)

    def select(self, agent, options: Sequence[ExpansionOption]) -> SelectorResult:
        if not options:
            raise ValueError("reward-only selector received no options")
        records = self.records_provider()
        reason = self.cold_start_reason(records, options)
        if reason is not None:
            return self._fallback(agent, options, reason)

        try:
            if (
                self.surrogate.fitted_transition_count < 0
                or len(records) - self.surrogate.fitted_transition_count >= self.refit_every
            ):
                self.surrogate.fit(records)
            predictions = self.surrogate.predict(options)
            temperature = self.thompson_temperature * self._temperature(options)
            sampled_gains = self.surrogate.sample_gains(
                options,
                seed=self._stable_seed(records, options),
                temperature=temperature,
            )
            valid_count = len(self._valid_records(records))
            gain_trust = self._gain_trust(valid_count)
            sampled_children = [
                option.parent_utility + gain_trust * float(sampled_gain)
                for option, sampled_gain in zip(options, sampled_gains)
            ]
            candidate_scores = tuple(
                CandidateScore(
                    parent_node_id=option.parent_node_id,
                    operator=option.operator,
                    reference_node_ids=option.reference_node_ids,
                    parent_utility=option.parent_utility,
                    mean_gain=prediction.mean_gain,
                    std_gain=prediction.std_gain,
                    predicted_child_ucb=None,
                    predicted_cost=None,
                    p_valid=None,
                    acquisition_score=float(sampled_child),
                    thompson_sample=float(sampled_child),
                    thompson_temperature=float(temperature),
                    gain_trust=float(gain_trust),
                    sampled_gain=float(sampled_gain),
                )
                for option, prediction, sampled_gain, sampled_child
                in zip(
                    options,
                    predictions,
                    sampled_gains,
                    sampled_children,
                )
            )
            best_index = max(
                range(len(candidate_scores)),
                key=lambda index: candidate_scores[index].acquisition_score,
            )
            selected_option = options[best_index]
            selected_score = candidate_scores[best_index].acquisition_score
            if selected_score is None or not math.isfinite(selected_score):
                raise ValueError("selected reward-only acquisition is not finite")
            return SelectorResult(
                decision=_decision(
                    selector_name=self.name,
                    option=selected_option,
                    selected_score=selected_score,
                    model_version=self.surrogate.model_version,
                ),
                selected_option=selected_option,
                candidate_scores=candidate_scores,
                options=tuple(options),
            )
        except SelectionOutsideOptions:
            raise
        except Exception as exc:
            return self._fallback(agent, options, f"model_error:{type(exc).__name__}:{exc}")


class BayesianNodeValueSelector:
    """BNVS selector that is prior-backed and active from the first decision."""

    def __init__(
        self,
        *,
        name: str,
        surrogate: BayesianNodeValueSurrogate,
        records_provider: Callable[[], list[TransitionRecord]],
        selection_rule: str,
        refit_every: int,
        thompson_temperature: float,
        random_seed: int,
    ):
        if selection_rule not in {"posterior_mean", "thompson"}:
            raise ValueError(f"unsupported BNVS selection rule: {selection_rule}")
        self.name = name
        self.surrogate = surrogate
        self.records_provider = records_provider
        self.selection_rule = selection_rule
        self.refit_every = max(int(refit_every), 1)
        self.thompson_temperature = min(
            max(float(thompson_temperature), 0.0), 1.0
        )
        self.random_seed = int(random_seed)
        self.fallback = ProgressiveMCGSSelector()
        self.decision_index: int | None = None

    def _stable_seed(
        self,
        records: Sequence[TransitionRecord],
        options: Sequence[ExpansionOption],
    ) -> int:
        parts = [self.name, str(self.random_seed), str(len(records))]
        if self.decision_index is not None:
            parts.append(str(self.decision_index))
        parts.extend(record.transition_id for record in records[-3:])
        parts.extend(
            f"{option.parent_node_id}:{option.operator}"
            for option in options
        )
        digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big", signed=False)

    def _fallback(
        self,
        agent,
        options: Sequence[ExpansionOption],
        reason: str,
    ) -> SelectorResult:
        try:
            result = self.fallback.select(agent, options)
        except SelectionOutsideOptions as exc:
            exc.fallback_reason = reason
            exc.selector_name = self.name
            raise
        return replace(
            result,
            decision=replace(
                result.decision,
                selector_name=self.name,
                fallback_used=True,
                fallback_reason=reason,
            ),
        )

    def select(self, agent, options: Sequence[ExpansionOption]) -> SelectorResult:
        if not options:
            raise ValueError("BNVS selector received no options")
        records = self.records_provider()
        try:
            if (
                self.surrogate.fitted_transition_count < 0
                or len(records) - self.surrogate.fitted_transition_count >= self.refit_every
                or self.surrogate.debug_repairs_changed(records)
            ):
                self.surrogate.fit(records)
            means, stds = self.surrogate.predict_values(options)
            if self.selection_rule == "posterior_mean":
                acquisition = means
                sampled = None
                temperature = 0.0
            else:
                acquisition = self.surrogate.sample_values(
                    options,
                    seed=self._stable_seed(records, options),
                    temperature=self.thompson_temperature,
                )
                sampled = acquisition
                temperature = self.thompson_temperature
            candidate_scores = tuple(
                CandidateScore(
                    parent_node_id=option.parent_node_id,
                    operator=option.operator,
                    reference_node_ids=option.reference_node_ids,
                    parent_utility=option.parent_utility,
                    mean_gain=None,
                    std_gain=None,
                    predicted_child_ucb=None,
                    predicted_cost=None,
                    p_valid=None,
                    acquisition_score=float(acquisition[index]),
                    thompson_sample=(
                        float(sampled[index]) if sampled is not None else None
                    ),
                    thompson_temperature=temperature,
                    posterior_mean_value=float(means[index]),
                    posterior_std_value=float(stds[index]),
                    sampled_value=(
                        float(sampled[index]) if sampled is not None else None
                    ),
                )
                for index, option in enumerate(options)
            )
            best_index = max(
                range(len(options)),
                key=lambda index: acquisition[index],
            )
            selected_option = options[best_index]
            selected_score = float(acquisition[best_index])
            if not math.isfinite(selected_score):
                raise ValueError("selected BNVS acquisition is not finite")
            return SelectorResult(
                decision=_decision(
                    selector_name=self.name,
                    option=selected_option,
                    selected_score=selected_score,
                    model_version=self.surrogate.model_version,
                ),
                selected_option=selected_option,
                candidate_scores=candidate_scores,
                options=tuple(options),
            )
        except SelectionOutsideOptions:
            raise
        except Exception as exc:
            return self._fallback(agent, options, f"model_error:{type(exc).__name__}:{exc}")
