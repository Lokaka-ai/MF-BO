"""Fixed-budget, observed-policy targets for direct long-horizon GP7 A.

h is a label contract, never an input coordinate. A window starts at an ordinary
proposal, includes the first h attempts from its parent and newly created primary
descendants, and may pause or overlap in wall time. Debug belongs to its original
proposal. Pending/unclosed attempts block a window instead of being skipped.
Only completed, disjoint windows train the GP; raw transitions stay unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Mapping, Sequence

from .base import NORMAL_OPERATORS
from .proposal_feedback import effective_proposals
from .transition_store import TransitionRecord

LONG_HORIZON_SELECTOR = "gp7_long_horizon_logei"
LONG_HORIZON_CONTRACT = "primary_direction_budget_nonoverlap_v1"
INDEX_KEY = "long_horizon_decision_index"


@dataclass(frozen=True)
class LongHorizonConfig:
    horizon: int = 2

    def __post_init__(self):
        if isinstance(self.horizon, bool) or not isinstance(self.horizon, int) or self.horizon < 1:
            raise ValueError("gp.long_horizon.horizon must be a positive integer")


def _index(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("Long A requires the saved decision index; start a new run, do not migrate old logs")
    return value


def long_horizon_targets(
    records: Sequence[TransitionRecord], *, horizon: int,
    pending: Sequence[Mapping] = (),
) -> tuple[list[TransitionRecord], dict]:
    """Build targets from this visible prefix, including in-flight blockers.

    Selection order comes from the managed ledger, NOT outcome/finish order.
    Completion order comes from the append-only feedback log. Rebuilding on
    resume therefore recovers the same labels without another persistent store.
    Synthetic rows are an in-memory adapter to A's existing residual GP fitter.
    """
    LongHorizonConfig(horizon)
    proposals = [p for p in effective_proposals(records) if p.operator in NORMAL_OPERATORS]
    positions = {r.transition_id: i for i, r in enumerate(records)}
    attempts = []
    known = {p.decision_id for p in proposals}
    for p in proposals:
        order = _index((p.metadata or {}).get(INDEX_KEY))
        if p.parent_utility is None or not math.isfinite(float(p.parent_utility)):
            raise ValueError("Long A requires a finite starting parent utility")
        attempts.append((order, p.parent_node_id, p))
    for item in pending:
        if item["operator"] in NORMAL_OPERATORS and item["decision_id"] not in known:
            attempts.append((_index(item["decision_index"]), item["parent_node_id"], None))
    attempts.sort(key=lambda item: item[0])
    if len({item[0] for item in attempts}) != len(attempts):
        raise ValueError("Duplicate long-horizon decision indices")

    complete = []
    blocked = 0
    for start, (order, parent, origin) in enumerate(attempts):
        if origin is None:
            continue
        allowed = {parent}
        members = []
        best = float(origin.parent_utility)
        for _, candidate_parent, proposal in attempts[start:]:
            if candidate_parent not in allowed:
                continue
            if proposal is None or not (proposal.metadata or {}).get("proposal_closed", True):
                blocked += 1
                break
            members.append(proposal)
            allowed.update((proposal.metadata or {}).get("proposal_child_ids", ()))
            if proposal.child_node_id:
                allowed.add(proposal.child_node_id)
            if proposal.valid:
                if proposal.child_utility is None or not math.isfinite(float(proposal.child_utility)):
                    raise ValueError("Invalid quality in a closed successful proposal")
                best = max(best, float(proposal.child_utility))
            if len(members) == horizon:
                terminal = max(members, key=lambda p: positions[(p.metadata or {}).get(
                    "outcome_transition_id", p.transition_id)])
                end_id = (terminal.metadata or {}).get("outcome_transition_id", terminal.transition_id)
                end_pos = positions[end_id]
                member_ids = [p.transition_id for p in members]
                window = {
                    "horizon": horizon, "contract": LONG_HORIZON_CONTRACT,
                    "origin_transition_id": origin.transition_id,
                    "origin_decision_index": order,
                    "member_transition_ids": member_ids,
                    "member_decision_ids": [p.decision_id for p in members],
                    "outcome_transition_id": end_id,
                    "available_at": (terminal.metadata or {}).get("available_at", terminal.timestamp),
                    "best_utility": best,
                    "parent_utility": float(origin.parent_utility),
                    "gain": best - float(origin.parent_utility),
                    "failed_attempts": sum(not p.valid for p in members),
                    "total_wall_time_sec": sum(p.total_wall_time_sec for p in members),
                }
                meta = dict(origin.metadata or {})
                meta.update(long_horizon_window=window, outcome_transition_id=end_id,
                            available_at=window["available_at"], proposal_closed=True)
                target = replace(origin, child_utility=best,
                                 improvement=best - float(origin.parent_utility), valid=True,
                                 metadata=meta)
                complete.append((end_pos, order, target))
                break

    # Greedy by availability (then decision order), matching the offline direct
    # long A protocol. No experiment is counted as independent evidence twice.
    complete.sort(key=lambda item: (item[0], item[1]))
    used: set[str] = set()
    targets = []
    for _, _, row in complete:
        members = set(row.metadata["long_horizon_window"]["member_transition_ids"])
        if members.isdisjoint(used):
            targets.append(row)
            used.update(members)
    diagnostics = {
        "long_horizon_contract": LONG_HORIZON_CONTRACT,
        "horizon": horizon,
        "horizon_is_feature": False,
        "window_policy": "observed_policy; pauses_and_parallelism_allowed; no_forced_continuation",
        "ordinary_proposal_count": len(proposals),
        "pending_ordinary_count": sum(row is None for _, _, row in attempts),
        "complete_window_count": len(complete),
        "blocked_window_count": blocked,
        "incomplete_window_count": len(proposals) - len(complete),
        "overlap_excluded_count": len(complete) - len(targets),
        "training_count": len(targets),
        "training_windows": [row.metadata["long_horizon_window"] for row in targets],
    }
    return targets, diagnostics
