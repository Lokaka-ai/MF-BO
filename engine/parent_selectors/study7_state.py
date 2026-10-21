"""Causal, immutable state4 snapshots and exact scientific ancestry."""
from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import math

import numpy as np

from .base import NORMAL_OPERATORS
from .proposal_feedback import effective_proposals
from .study7_gp import Study7Error, digest, rating_value

STATE4_NAMES = ("parent_quality", "birth_gain", "local_valid_gain_shrunk", "local_closed_attempts")


def admitted_records(records):
    seen, result = set(), []
    for row in records:
        if row.decision_id not in seen:
            seen.add(row.decision_id)
            result.append(row)
    return result


def closed(row):
    metadata = row.metadata or {}
    reason = str(metadata.get("failure_reason") or "")
    return (bool(metadata.get("proposal_closed", True))
            and metadata.get("budget_eligible", True)
            and not metadata.get("censored", False)
            and "interrupted" not in reason and "budget_censored" not in reason)


def scientific_proposals(records, *, include_aggregation=False):
    """Keep the original scientific input and the final outcome's admission clock."""
    records = admitted_records(records)
    raw = {row.transition_id: row for row in records}
    results = []
    for row in effective_proposals(records, include_aggregation=include_aggregation):
        last = raw[row.metadata["outcome_transition_id"]].metadata or {}
        final_fields = {key: last[key] for key in (
            "failure_reason", "budget_eligible", "censored", "completed_active_seconds",
            "completed_epoch", "duration_seconds", "late_cleanup_seconds", "raw_metric", "raw_valid")
            if key in last}
        results.append(replace(row, metadata={**row.metadata, **final_fields}))
    return results


class Study7State:
    def __init__(self, config):
        self.config = config
        self.reference = None
        self.nodes = {}

    def freeze_reference(self, records):
        if self.reference is not None:
            return
        drafts = [row for row in scientific_proposals(records)
                  if row.operator == "draft" and row.valid and closed(row)]
        initial = [row for row in drafts if (row.metadata or {}).get("decision_kind") == "initialization"]
        drafts = initial or drafts
        if not drafts:
            raise Study7Error("Ordinary study7 selection requires a valid released draft")
        values = np.asarray([row.child_utility for row in drafts], dtype=np.float64)
        if not np.isfinite(values).all():
            raise Study7Error("Invalid initial draft utility")
        self.reference = {"u_ref": float(np.median(values)),
            "s0": max(float(np.quantile(values, .75) - np.quantile(values, .25)),
                      .05 * float(np.median(np.abs(values))), 1e-6),
            "reference_node_ids": [row.child_node_id for row in drafts],
            "reference_metric_hashes": [digest({"node": row.child_node_id, "metric": row.child_metric_raw,
                                                "utility": row.child_utility}) for row in drafts],
            "reference_outcome_ids": [row.metadata["outcome_transition_id"] for row in drafts],
            "numpy_version": np.__version__, "quantile_method": "linear"}
        self.reference["hash"] = digest(self.reference)

    def sync(self, records):
        if self.reference is None:
            return
        records = admitted_records(records)
        raw = {row.transition_id: row for row in records}
        commit = {row.transition_id: index for index, row in enumerate(records)}
        # A successful debug of a failed aggregation is its committed root
        # outcome, with the original aggregation's explicit source provenance.
        outcomes = [row for row in scientific_proposals(records, include_aggregation=True)
                    if row.valid and closed(row)]
        outcomes.sort(key=lambda row: commit[(row.metadata or {}).get("outcome_transition_id", row.transition_id)])
        for row in outcomes:
            outcome_id = (row.metadata or {}).get("outcome_transition_id", row.transition_id)
            outcome = raw[outcome_id]
            child = row.child_node_id
            if not child or row.child_utility is None or not math.isfinite(float(row.child_utility)):
                raise Study7Error("Invalid scientific node")
            parent = row.parent_node_id if row.operator in NORMAL_OPERATORS else None
            if parent is not None and parent not in self.nodes:
                raise Study7Error("Scientific parent is missing or violates DAG order")
            if parent == child:
                raise Study7Error("Scientific ancestry cycle")
            sources = [parent] if parent else []
            provenance = "single_parent" if parent else "root"
            if row.operator == "fusion":
                sources = list(dict.fromkeys([parent, *row.reference_node_ids]))
                provenance = "preview_verified_sources" if row.reference_node_ids else "primary_only"
            elif row.operator == "aggregation":
                # Only explicit execution provenance is usable; retrieved top-k is not.
                sources = list(dict.fromkeys((row.metadata or {}).get("fusion_source_ids", [])))
                provenance = "explicit_sources" if sources else "unavailable_new_root"
            if any(source == child or source not in self.nodes for source in sources):
                raise Study7Error("Fusion source is missing, not yet committed, or cyclic")
            if any(self.nodes[source]["birth_commit_index"] >= commit[outcome_id] for source in sources):
                raise Study7Error("Scientific sources must precede the child in committed DAG order")
            utility = float(row.child_utility)
            gain = utility - self.nodes[parent]["utility"] if parent else 0.0
            weights = {}
            if self.config.ancestor.enabled:
                weights = {child: .5 if sources else 1.0}
                for source in sources:
                    for support, weight in self.nodes[source]["ancestor_weights"].items():
                        weights[support] = weights.get(support, 0.0) + .5 * weight / len(sources)
                if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-12) or any(w < 0 for w in weights.values()):
                    raise Study7Error("Invalid scientific ancestor weights")
            snapshot = (outcome.metadata or {}).get("node_snapshot") or {}
            code = snapshot.get("node", {}).get("code")
            item = {"utility": utility, "raw_metric": row.child_metric_raw, "birth_gain": gain,
                "scientific_parent_id": parent, "origin_proposal_id": row.decision_id,
                "outcome_transition_id": outcome_id, "birth_commit_index": commit[outcome_id],
                "xi": [math.tanh((utility - self.reference["u_ref"]) / self.reference["s0"]),
                       math.tanh(gain / self.reference["s0"])],
                "ancestor_weights": weights, "fusion_sources": sources,
                "fusion_provenance_status": provenance,
                "code_sha256": hashlib.sha256(code.encode()).hexdigest() if code is not None else None}
            if child in self.nodes and self.nodes[child] != item:
                raise Study7Error("Historical scientific node/metric/source changed in place")
            self.nodes[child] = item

    def features(self, agent, parent, operator, records):
        self.freeze_reference(records)
        self.sync(records)
        if parent.id not in self.nodes:
            raise Study7Error("Legal study7 parent has no committed scientific outcome")
        current = self.nodes[parent.id]
        local = [row for row in scientific_proposals(records)
                 if row.operator == operator and row.operator in NORMAL_OPERATORS
                 and row.parent_node_id == parent.id and closed(row)]
        valid = [row for row in local if row.valid]
        total_gain = math.fsum(float(row.child_utility) - current["utility"] for row in valid)
        intensity = math.log1p(len(local))
        z = [math.tanh((current["utility"] - self.reference["u_ref"]) / self.reference["s0"]),
             math.tanh(current["birth_gain"] / self.reference["s0"]),
             math.tanh(total_gain / ((len(valid) + 1) * self.reference["s0"])),
             intensity / (1.0 + intensity)]
        raw_rating = getattr(parent, "agent_potential_level", None) if self.config.rating.enabled else None
        level = raw_rating if type(raw_rating) is int and raw_rating in (0, 1, 2) else None
        weights = copy.deepcopy(current["ancestor_weights"])
        result = {"schema": "state4_v1", "z": z, "operator": operator, "parent_id": parent.id,
            "parent_utility": current["utility"], "birth_gain_raw": current["birth_gain"],
            "local_closed_attempts": len(local), "local_valid_count": len(valid),
            "local_sum_gain": total_gain, "history_proposal_ids": [row.decision_id for row in local],
            "valid_history_proposal_ids": [row.decision_id for row in valid],
            "reference_hash": self.reference["hash"], "u_ref": self.reference["u_ref"], "s0": self.reference["s0"],
            "rating_level": level, "rating_s": rating_value(level),
            "rating_missing": level is None, "rating_assessed_at": getattr(parent, "agent_potential_assessed_at", None) if self.config.rating.enabled else None,
            "rating_evidence": copy.deepcopy(getattr(parent, "study7_rating_metadata", {})) if self.config.rating.enabled else {},
            "ancestor_weights": weights, "ancestor_weight_hash": digest(weights) if weights else None,
            "ancestor_content_hash": digest({key: self.nodes[key]["xi"] for key in sorted(weights)}) if weights else None,
            "birth_commit_index": next(index for index, node in enumerate(agent.journal.nodes) if node.id == parent.id)}
        return result

    def training_data(self, records):
        rows, targets, ids, exclusions = [], [], [], []
        if self.reference is None:
            return rows, targets, ids, exclusions
        for row in scientific_proposals(records):
            reason = None
            if row.operator not in NORMAL_OPERATORS:
                reason = "not_ordinary_proposal"
            elif not closed(row):
                reason = "pending_or_censored"
            elif not row.valid:
                reason = "closed_failure_no_quality_label"
            if reason:
                exclusions.append({"proposal_id": row.decision_id, "reason": reason})
                continue
            frozen = copy.deepcopy(row.selection_features.get("study7"))
            if not frozen or frozen["reference_hash"] != self.reference["hash"]:
                raise Study7Error("Missing or incompatible original proposal snapshot")
            if frozen["parent_id"] != row.parent_node_id or frozen["operator"] != row.operator:
                raise Study7Error("DEBUG closure lost original parent/operator attribution")
            target = (float(row.child_utility) - float(row.parent_utility)) / self.reference["s0"]
            if not math.isfinite(target):
                raise Study7Error("Non-finite ordinary proposal residual")
            rows.append(frozen)
            targets.append(target)
            ids.append({"proposal_id": row.decision_id, "outcome_id": row.metadata["outcome_transition_id"]})
        return rows, targets, ids, exclusions

    def state_dict(self):
        return {"reference": copy.deepcopy(self.reference), "nodes": copy.deepcopy(self.nodes)}

    def load_state_dict(self, state):
        reference = copy.deepcopy(state["reference"])
        if reference is not None:
            without_hash = {key: value for key, value in reference.items() if key != "hash"}
            if (digest(without_hash) != reference["hash"] or reference["s0"] <= 0
                    or not math.isfinite(reference["s0"]) or not math.isfinite(reference["u_ref"])):
                raise Study7Error("Invalid frozen reference checkpoint")
        self.reference = reference
        self.nodes.clear()
        self.nodes.update(copy.deepcopy(state["nodes"]))
