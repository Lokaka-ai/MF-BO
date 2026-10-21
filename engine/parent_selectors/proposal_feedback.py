"""Decision-prefix-only proposal feedback, with normal AND draft debug repair.

The append-only transition log is unchanged. One training observation represents
one proposal; a successful debug inherits the ORIGINAL decision's state/action.
Unresolved repair chains are censored, not silently labelled as final failures.
"""
from __future__ import annotations

from dataclasses import replace
import math
from typing import Sequence

from .transition_store import TransitionRecord

PROPOSAL_OPERATORS = ("improve", "evolution", "fusion", "draft")


def effective_proposals(
    records: Sequence[TransitionRecord], *, include_aggregation: bool = False,
) -> list[TransitionRecord]:
    """Fold a visible log prefix. First valid result wins; never mutate raw rows."""
    # Scientific root reconstruction also needs aggregation repair chains.
    # Existing GP callers keep their original proposal/target admission.
    operators = (*PROPOSAL_OPERATORS, "aggregation") if include_aggregation else PROPOSAL_OPERATORS
    proposals: dict[str, TransitionRecord] = {}
    child_to_proposal: dict[str, str] = {}
    seen: set[str] = set()
    for row in records:
        if row.decision_id in seen:
            continue
        seen.add(row.decision_id)
        if row.operator in operators:
            key = row.decision_id
            metadata = dict(row.metadata or {})
            metadata.update(
                proposal_transition_ids=[row.transition_id],
                proposal_child_ids=[row.child_node_id] if row.child_node_id else [],
                outcome_transition_id=row.transition_id,
                available_at=row.timestamp,
                # Old logs without closure metadata are terminal observations.
                proposal_closed=bool(row.valid or metadata.get("proposal_closed", True)),
            )
            proposals[key] = replace(row, metadata=metadata)
        elif row.operator == "debug" and row.parent_node_id in child_to_proposal:
            key = child_to_proposal[row.parent_node_id]
            origin = proposals[key]
            if origin.valid:  # Direct success or first successful repair already observed.
                if row.child_node_id:
                    child_to_proposal[row.child_node_id] = key
                continue
            valid = bool(row.valid and row.child_utility is not None
                         and math.isfinite(float(row.child_utility)))
            metadata = dict(origin.metadata or {})
            metadata.update(
                proposal_transition_ids=[*metadata["proposal_transition_ids"], row.transition_id],
                proposal_child_ids=[*metadata["proposal_child_ids"],
                                    *([row.child_node_id] if row.child_node_id else [])],
                outcome_transition_id=row.transition_id,
                available_at=row.timestamp,
                proposal_closed=bool(valid or (row.metadata or {}).get("proposal_closed", True)),
            )
            if valid:
                metadata["debug_repair"] = {
                    "proposal_transition_id": origin.transition_id,
                    "debug_transition_id": row.transition_id,
                    "available_at": row.timestamp,
                }
            proposals[key] = replace(
                origin,
                child_node_id=row.child_node_id or origin.child_node_id,
                child_metric_raw=row.child_metric_raw if valid else None,
                child_utility=row.child_utility if valid else None,
                improvement=(row.child_utility - origin.parent_utility
                             if valid and origin.parent_utility is not None else None),
                valid=valid,
                generation_wall_time_sec=origin.generation_wall_time_sec + row.generation_wall_time_sec,
                execution_wall_time_sec=origin.execution_wall_time_sec + row.execution_wall_time_sec,
                total_wall_time_sec=origin.total_wall_time_sec + row.total_wall_time_sec,
                metadata=metadata,
            )
        else:
            continue
        if row.child_node_id:
            child_to_proposal[row.child_node_id] = key
    return list(proposals.values())
