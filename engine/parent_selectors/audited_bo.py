"""Audited v2 interventions. Original AQ contracts and v1 releases are untouched.

Plateau state is a deterministic fold of the admitted outcome ledger, not journal
creation order. Frontier labels are self-reported coarse descriptors, NOT verified
semantic equivalence classes. Unknown labels earn no diversity reservation.
"""
from __future__ import annotations

import copy
import math
import io
import tokenize
import re

from .acq7_selector import Acq7Selector
from .study7_state import scientific_proposals, closed

VERSION = "audited_bo_v2_1"
METHODS = {"preserved_evolution": "AQ04", "redesign": "AQ04", "plateau": "AQ04", "representation_frontier": "AQ04",
           "rating_free": "AQ03", "retained_kg": "AQ07", "task_correct": "AQ04"}
REPRESENTATIONS = ("discrete", "continuous", "algebraic", "graph", "hybrid", "other", "unknown")
SEARCHES = ("enumeration", "local_search", "optimization", "sampling", "greedy", "recursive", "hybrid", "other", "unknown")


def ordered_closed(records):
    # Append/commit order is the availability order seen by the controller. Wall
    # clocks can tie or move backwards. Debug closes at its FINAL outcome index.
    index = {row.transition_id: i for i, row in enumerate(records)}
    return sorted((row for row in scientific_proposals(records, include_aggregation=True) if closed(row)),
                  key=lambda row: index[row.metadata["outcome_transition_id"]])


def plateau_state(agent, branch_id=None):
    nodes = {node.id: node for node in agent.journal.nodes}
    incumbent, stale, outcomes = None, 0, []
    for row in ordered_closed(agent.phase1_runtime.transition_store.records()):
        # Ordinary/fusion outcomes belong to the original input branch; repaired
        # drafts belong to their resulting native branch. Debug is counted once.
        node = nodes.get(row.parent_node_id if row.operator in {"improve", "evolution", "fusion"}
                         else row.child_node_id)
        if branch_id is not None and (node is None or node.branch_id != branch_id):
            continue
        value = row.child_utility  # already sign-normalized for min/max tasks
        improved = bool(row.valid and value is not None and math.isfinite(value)
                        and (incumbent is None or value > incumbent))
        if improved:
            incumbent, stale = float(value), 0
        else:
            stale += 1
        outcomes.append(row.metadata["outcome_transition_id"])
    return {"incumbent_utility": incumbent, "stale": stale, "outcome_ids": outcomes}


def descriptor(code):
    try:
        comments = [token.string for token in tokenize.generate_tokens(io.StringIO(code or "").readline)
                    if token.type == tokenize.COMMENT]
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return None
    values = []
    for field, vocabulary in (("REPRESENTATION", REPRESENTATIONS), ("SEARCH", SEARCHES)):
        prefix = re.compile(r"#\s*BO_" + field + r"\s*:")
        declarations = [comment for comment in comments if prefix.match(comment)]
        if len(declarations) != 1:
            return None
        match = re.fullmatch(r"#\s*BO_" + field + r"\s*:\s*([a-z_]+)\s*", declarations[0])
        if match is None or match[1] not in vocabulary or match[1] in {"unknown", "other"}:
            return None
        values.append(match[1])
    return "/".join(values)


def representation_prompt(agent, prompt, *, fresh):
    instructions = [
        "Put exactly two descriptive comments at the top of the completed Python code:",
        "# BO_REPRESENTATION: " + " | ".join(REPRESENTATIONS),
        "# BO_SEARCH: " + " | ".join(SEARCHES),
        "Choose ONE lowercase label per line describing the actual code. Use unknown if uncertain; do not choose labels to obtain a reward.",
        "Describe the representation, generation/search procedure and feasibility checks in your plan.",
    ]
    if fresh:
        # Only labels of completed, host-valid programs cross this boundary.
        nodes = {node.id: node for node in agent.journal.nodes}
        labels = sorted({label for row in ordered_closed(agent.phase1_runtime.transition_store.records())
                         if row.valid and row.child_node_id in nodes
                         for label in [descriptor(nodes[row.child_node_id].code)] if label})
        prompt["Memory"] = "Previously observed coarse descriptors: " + (", ".join(labels) or "unknown")
        instructions += [
            "Build a complete new solution from the task specification. Explore a materially different representation or search procedure from the listed descriptors, when feasible.",
            "This is a fresh full rewrite. No previous candidate code or plan is provided. A renamed implementation is not a new approach. Keep the task's exact feasibility rules and runtime budget.",
        ]
    prompt["Instructions"]["Representation exploration"] = instructions


def draft_cap(steps):
    # Leave at least two proposal slots after the final scheduled rewrite.
    return 3 + max(0, (steps - 5) // 8)


def rewrite_due(state, *, steps):
    reserved = [p for p in state["pending"].values()
                if p["status"] == "reserved" and not p.get("recovery_already_submitted")]
    post = state["submitted_count"] + len(reserved) - state["initial_goal"]
    return (steps - state["submitted_count"] - len(reserved) >= 3
            and post >= 0 and (post + 1) % 8 == 0
            and state["draft_submitted"] + sum(p["option"]["operator"] == "draft" for p in reserved) < state["draft_cap"]
            and not any(p["option"]["operator"] == "draft" for p in state["pending"].values()))


class AuditedBOSelector(Acq7Selector):
    def __init__(self, *, variant, **kwargs):
        self.variant = variant
        self._audit = {}
        super().__init__(**kwargs)

    def contract(self):
        return {**super().contract(), "audited_bo": self.variant, "audited_version": ("preserved_evolution_v1" if self.variant == "preserved_evolution" else "redesign_v1" if self.variant == "redesign" else VERSION),
                "rewrite_period": 8, "protected_descriptors": 3, "followup_attempts": 2,
                "descriptor_schema": "self_reported_representation_search_v1",
                "plateau_order": "admitted_final_outcome_commit_order"}

    def select(self, agent, options):
        self._audit = {"variant": self.variant, "version": VERSION}
        if self.variant in {"plateau", "representation_frontier", "redesign", "preserved_evolution"}:
            self._audit["global_plateau"] = plateau_state(agent)
            self._audit["branch_plateaus"] = {str(o.parent_node.branch_id): plateau_state(agent, o.parent_node.branch_id)
                                                for o in options}
        self._protected_parents = set()
        if self.variant == "representation_frontier":
            nodes = {node.id: node for node in agent.journal.nodes}
            groups = {}
            for row in ordered_closed(self.records_provider()):
                if not row.valid or row.operator != "draft" or row.child_node_id not in nodes:
                    continue
                label = descriptor(nodes[row.child_node_id].code)
                if label is None:
                    continue
                # Descriptor coverage, not ancestry coverage. Combine counts of
                # ALL roots with the same label; duplicates do not renew quota.
                root = self.state.lineages.nodes.get(row.child_node_id)
                if root is None:
                    continue
                group = groups.setdefault(label, {"roots": [], "nodes": []})
                group["roots"].append(root["root_lineage_id"])
                group["nodes"].append((row.child_utility, row.child_node_id))
            # FIFO descriptor archive: last three distinct labels by first admission.
            archive = []
            legal = {o.parent_node_id for o in options}
            for label, group in list(groups.items())[-3:]:
                used = sum(self.dispatched_by_branch.get(root, 0) for root in set(group["roots"]))
                candidates = sorted(group["nodes"], reverse=True)
                parent = next((node_id for _, node_id in candidates if node_id in legal), None)
                if used < 2 and parent is not None:
                    self._protected_parents.add(parent)
                archive.append({"descriptor": label, "followups_dispatched": used,
                                "eligible_parent": parent, "root_ids": sorted(set(group["roots"]))})
            self._audit["archive"] = archive
        return super().select(agent, options)

    def predict_joint(self, rows, scale):
        self._rows = rows
        return super().predict_joint(rows, scale)

    def acquire(self, joint, keys, incumbent, acquisition_started):
        values, details, chosen, pre, post = super().acquire(joint, keys, incumbent, acquisition_started)
        eligible = [i for i, row in enumerate(self._rows) if row["parent_id"] in self._protected_parents]
        # Reserve followups as soon as a distinct feasible draft is eligible.
        # Native pending locks remain authoritative; attempts include failures.
        if eligible:
            chosen = min(eligible, key=lambda i: (-float(values[i]), tuple(keys[i])))
        self._audit["protected_indices"] = eligible
        details["audited_protected_indices"] = eligible
        return values, details, chosen, pre, post

    def decision_metadata(self):
        return {"audited_bo": copy.deepcopy(self._audit)}
