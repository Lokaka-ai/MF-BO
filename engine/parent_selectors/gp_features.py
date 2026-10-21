"""Six numerical, decision-time progress features plus a categorical operation.

No fitted per-column standardizer, MCGS Q, embeddings, or future child scores.
Raw features are frozen at selection; only metric-valued coordinates are later
scaled by the run's frozen utility scale.
"""
from __future__ import annotations

import math
from typing import Sequence
import numpy as np

from .proposal_feedback import effective_proposals
from .transition_store import TransitionRecord

GP6_SCHEMA = "gp6_progress_v1"
GP6_NAMES = ("parent_gap", "last_gain", "recent_gain", "stagnation", "trials", "valid_rate")
GP6_OPERATORS = ("improve", "evolution", "fusion", "draft")


def progress_features(agent, node, records: Sequence[TransitionRecord], *, window: int = 5) -> dict:
    """Summarize closed proposals on the primary ancestry and attempts from it.

A newly produced leaf inherits its trajectory. Alternatives tried from an
ancestor count as investments in that direction; unrelated branches and fusion
reference ancestry do not. Pending repair chains do not count as failed trials.
"""
    ancestors = set()
    cursor = node
    while cursor is not None and getattr(cursor, "stage", None) != "root":
        if cursor.id in ancestors:
            raise ValueError("Cycle in primary parent chain")
        ancestors.add(cursor.id)
        cursor = getattr(cursor, "parent", None)
    proposals = [r for r in effective_proposals(records)
                 if (r.metadata or {}).get("proposal_closed", True)]
    is_root = getattr(node, "stage", None) == "root"
    if is_root:
        history = [r for r in proposals if r.operator == "draft"]
    else:
        history = [r for r in proposals
                   if r.parent_node_id in ancestors
                   or ancestors.intersection((r.metadata or {}).get("proposal_child_ids", []))]
    # Feedback time, not the failed original proposal's earlier submission time.
    order = {r.transition_id: i for i, r in enumerate(records)}
    history.sort(key=lambda r: order.get((r.metadata or {}).get("outcome_transition_id", r.transition_id), -1))
    recent = history[-max(int(window), 1):]
    gains = [float(r.improvement) for r in recent
             if r.valid and r.improvement is not None and math.isfinite(float(r.improvement))]
    all_gains = [float(r.improvement) for r in history
                 if r.valid and r.improvement is not None and math.isfinite(float(r.improvement))]
    threshold = float(getattr(agent.scfg, "metric_improvement_threshold", 0.0))
    stagnant = 0
    for r in reversed(history):
        # The successful seed starts a direction; it has no local-gain label.
        if r.valid and (r.operator == "draft" or
                        (r.improvement is not None and r.improvement > threshold)):
            break
        stagnant += 1
    parent_metric = getattr(getattr(node, "metric", None), "value", None)
    parent = None if is_root or parent_metric is None else float(parent_metric) * (1 if agent.metric_maximize else -1)
    utilities = [r.child_utility for r in records if r.valid and r.child_utility is not None]
    for candidate in agent.journal.nodes:
        metric = getattr(getattr(candidate, "metric", None), "value", None)
        if (getattr(candidate, "stage", None) != "root" and
            getattr(candidate, "is_buggy", None) is False and
            getattr(candidate, "is_valid", None) is not False and metric is not None):
            utilities.append(float(metric) * (1 if agent.metric_maximize else -1))
    incumbent = max((v for v in utilities if math.isfinite(v)), default=None)
    successes = sum(bool(r.valid) for r in history)
    return {
        "gp_parent_gap": max(incumbent - parent, 0.0) if incumbent is not None and parent is not None else 0.0,
        "gp_last_gain": all_gains[-1] if all_gains and not is_root else 0.0,
        "gp_recent_gain": float(np.mean(gains)) if gains and not is_root else 0.0,
        "gp_stagnation": float(stagnant if not is_root else 0),
        "gp_trials": float(len(history)),
        "gp_valid_rate": (successes + 1.0) / (len(history) + 2.0),
        "parent_utility": parent,
        "incumbent_at_decision": incumbent,
    }


def gp_vector(features: dict, *, scale: float, draft_anchor: float, operator: str) -> np.ndarray:
    """Six bounded coordinates; operation is handled separately by the kernel."""
    if operator not in GP6_OPERATORS:
        raise ValueError(f"Not a GP action: {operator}")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("GP utility scale must be finite and positive")
    raw = [float(features["gp_" + key]) for key in GP6_NAMES]
    if operator == "draft":
        incumbent = features.get("incumbent_at_decision")
        raw[0] = max(float(incumbent) - draft_anchor, 0.0) if incumbent is not None else 0.0
    if not np.isfinite(raw).all():
        raise ValueError("Non-finite GP6 input")
    counts = np.log1p(np.maximum(raw[3:5], 0.0))
    return np.asarray([*np.tanh(np.asarray(raw[:3]) / scale),
                       *(counts / (1.0 + counts)), min(max(raw[5], 0.0), 1.0)], dtype=float)
