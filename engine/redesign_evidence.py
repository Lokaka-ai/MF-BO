"""Completed-outcome ordering and plateau state for redesigned evolution."""
from __future__ import annotations

import math

from engine.parent_selectors.study7_state import scientific_proposals, closed


def ordered_closed(records):
    # Outcome commit order is the information available to the controller.
    index = {row.transition_id: i for i, row in enumerate(records)}
    return sorted(
        (row for row in scientific_proposals(records, include_aggregation=True) if closed(row)),
        key=lambda row: index[row.metadata["outcome_transition_id"]],
    )


def plateau_state(agent, branch_id=None):
    nodes = {node.id: node for node in agent.journal.nodes}
    incumbent, stale, outcomes = None, 0, []
    mf = getattr(getattr(getattr(agent, "cfg", None), "evaluation", None), "multifidelity_mode", None) == "mf_gp"
    by_fidelity = {}
    for row in ordered_closed(agent.phase1_runtime.transition_store.records()):
        node = nodes.get(row.parent_node_id if row.operator in {"improve", "evolution", "fusion"}
                         else row.child_node_id)
        if branch_id is not None and (node is None or node.branch_id != branch_id):
            continue
        child = nodes.get(row.child_node_id)
        fidelity = getattr(child, "training_fraction", None)
        if mf:
            incumbent = by_fidelity.get(fidelity)
        value = row.child_utility
        improved = bool(row.valid and value is not None and math.isfinite(value)
                        and (incumbent is None or value > incumbent))
        if improved:
            incumbent, stale = float(value), 0
            if mf:
                by_fidelity[fidelity] = incumbent
        else:
            stale += 1
        outcomes.append(row.metadata["outcome_transition_id"])
    return {"incumbent_utility": incumbent, "stale": stale, "outcome_ids": outcomes}
