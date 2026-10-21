"""Small shared types for Phase 1 parent selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Protocol, Sequence


NORMAL_OPERATORS = ("improve", "evolution", "fusion")
BYPASS_OPERATORS = ("draft", "debug", "aggregation", "no_op")


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class OperatorPreview:
    operator: str
    reference_node_ids: tuple[str, ...] = ()
    bypass_selector: bool = False
    reason: str = ""
    consumes_random_draw: bool = False
    random_value: float | None = None
    topk_triggered: bool = False


@dataclass(frozen=True)
class ExpansionOption:
    parent_node_id: str
    operator: str
    reference_node_ids: tuple[str, ...]
    state_features: Mapping[str, Any]
    parent_node: Any = field(repr=False, compare=False)
    preview: OperatorPreview = field(repr=False, compare=False)

    @property
    def parent_utility(self) -> float | None:
        value = self.state_features.get("parent_utility", 0.0)
        return float(value) if value is not None else None


@dataclass(frozen=True)
class SelectionDecision:
    selector_name: str
    selected_parent_id: str
    selected_operator: str
    selected_score: float
    model_version: str | None
    fallback_used: bool
    decision_id: str
    timestamp: str
    decision_wall_time_sec: float = 0.0
    selection_reason: str = "ranked_options"
    fallback_reason: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PredictiveTransition:
    parent_node_id: str
    operator: str
    mean_gain: float
    std_gain: float
    predicted_total_cost: float | None
    p_valid: float | None


@dataclass(frozen=True)
class CandidateScore:
    parent_node_id: str
    operator: str
    reference_node_ids: tuple[str, ...]
    parent_utility: float | None
    mean_gain: float | None
    std_gain: float | None
    predicted_child_ucb: float | None
    predicted_cost: float | None
    p_valid: float | None
    acquisition_score: float | None
    thompson_sample: float | None = None
    thompson_temperature: float | None = None
    gain_trust: float | None = None
    sampled_gain: float | None = None
    posterior_mean_value: float | None = None
    posterior_std_value: float | None = None
    sampled_value: float | None = None
    posterior_mean_quality: float | None = None
    posterior_std_latent: float | None = None
    predictive_std: float | None = None

    def to_dict(self) -> dict:
        payload = asdict(self)
        gp_fields = ("posterior_mean_quality", "posterior_std_latent", "predictive_std")
        if all(payload[name] is None for name in gp_fields):
            for name in gp_fields:
                payload.pop(name)
        return payload


@dataclass(frozen=True)
class SelectorResult:
    decision: SelectionDecision
    selected_option: ExpansionOption
    candidate_scores: tuple[CandidateScore, ...]
    options: tuple[ExpansionOption, ...]


class ParentSelector(Protocol):
    name: str

    def select(
        self,
        agent: Any,
        options: Sequence[ExpansionOption],
    ) -> SelectorResult:
        ...
