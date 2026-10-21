"""Causal root-lineage identities and unchanged GP4/GP6 label admission."""
from __future__ import annotations

import copy
import math

import numpy as np

from .base import NORMAL_OPERATORS
from .gp_features import gp_vector, progress_features
from .gp_surrogate import GaussianProposalSurrogate
from .proposal_feedback import effective_proposals
from .study7_gp import Study7Error, digest
from .study7_state import Study7State, admitted_records


class RootLineages:
    def __init__(self, namespace):
        self.namespace = namespace
        self.roots, self.nodes = {}, {}

    def root(self, proposal, timestamp, commit_index):
        if proposal not in self.roots:
            self.roots[proposal] = {
                "root_lineage_id": f"{self.namespace}:{proposal}",
                "lineage_origin_proposal_id": proposal,
                "lineage_assigned_at": timestamp,
                "lineage_birth_commit_index": commit_index,
                "branch_mapping_version": "root_lineage_primary_v1"}
        return copy.deepcopy(self.roots[proposal])

    def reserve(self, decision, option, commit_index):
        if option.operator in {"draft", "aggregation"}:
            return self.root(decision.decision_id, decision.timestamp, commit_index)
        if option.parent_node_id not in self.nodes:
            raise Study7Error("Branch7 reservation has no committed scientific parent lineage")
        return copy.deepcopy(self.nodes[option.parent_node_id])

    def sync(self, records):
        for index, row in enumerate(admitted_records(records)):
            if row.operator in {"draft", "aggregation"}:
                # Archived root aggregation creates a new fusion_draft under dummy root.
                info = self.root(row.decision_id, row.timestamp, index)
            elif row.operator in {*NORMAL_OPERATORS, "debug"}:
                if row.parent_node_id not in self.nodes:
                    raise Study7Error("Missing/cyclic primary lineage in committed feedback")
                info = copy.deepcopy(self.nodes[row.parent_node_id])
                if row.operator in NORMAL_OPERATORS:
                    frozen = row.selection_features.get("study7", {})
                    if frozen.get("root_lineage_id") != info["root_lineage_id"]:
                        raise Study7Error("Ordinary feedback changed its frozen parent lineage")
            else:
                continue
            if row.child_node_id:
                if row.child_node_id == row.parent_node_id:
                    raise Study7Error("Branch7 scientific lineage cycle")
                if row.child_node_id in self.nodes and self.nodes[row.child_node_id] != info:
                    raise Study7Error("Historical branch identity changed in place")
                self.nodes[row.child_node_id] = info

    def state_dict(self):
        return copy.deepcopy({"namespace": self.namespace, "roots": self.roots, "nodes": self.nodes})

    def load_state_dict(self, state):
        if state["namespace"] != self.namespace:
            raise Study7Error("Cannot restore another arm's branch feedback")
        self.roots, self.nodes = copy.deepcopy(state["roots"]), copy.deepcopy(state["nodes"])
        for row in self.nodes.values():
            if self.roots.get(row["lineage_origin_proposal_id"]) != row:
                raise Study7Error("Invalid persisted branch map")


class Branch7State(Study7State):
    def __init__(self, config):
        super().__init__(config)
        self.lineages = RootLineages(config.method_id)
        self.native = GaussianProposalSurrogate() if config.base_dimension == 6 else None

    def freeze_reference(self, records):
        if not self.native:
            return super().freeze_reference(records)
        if self.reference is not None:
            return
        usable = [r for r in effective_proposals(records) if r.valid
                  and r.child_utility is not None and math.isfinite(float(r.child_utility))]
        self.native._freeze_reference(usable)
        if not self.native.reference_frozen:
            raise Study7Error("GP6 ordinary selection has no usable frozen reference")
        self.reference = {"u_ref": self.native.draft_anchor, "s0": self.native.utility_scale,
            "reference_transition_ids": self.native.reference_transition_ids,
            "reference_node_ids": [r.child_node_id for r in usable
                                   if r.transition_id in self.native.reference_transition_ids],
            "reference_metric_hashes": [digest({"node": r.child_node_id, "utility": r.child_utility})
                                       for r in usable if r.transition_id in self.native.reference_transition_ids],
            "numpy_version": np.__version__, "quantile_method": "linear", "base": "archived_gp6"}
        self.reference["hash"] = digest(self.reference)

    def sync(self, records):
        self.lineages.sync(records)
        super().sync(records)

    def features(self, agent, parent, operator, records):
        row = super().features(agent, parent, operator, records)
        if parent.id not in self.lineages.nodes:
            raise Study7Error("Ordinary Branch7 candidate has no root lineage")
        row.update(self.lineages.nodes[parent.id], branch_effect_load=1,
                   schema=f"branch7_gp{self.config.base_dimension}_v1")
        if self.native:
            raw = progress_features(agent, parent, records, window=5)
            row["z"] = self.native._design(raw, operator).tolist()
            row["gp6_raw_features"] = raw
        return row

    def training_data(self, records):
        if not self.native:
            return super().training_data(records)
        rows, targets, ids, exclusions = [], [], [], []
        if self.reference is None:
            return rows, targets, ids, exclusions
        for proposal in effective_proposals(records):
            if (not proposal.valid or proposal.child_utility is None
                    or not math.isfinite(float(proposal.child_utility))):
                exclusions.append({"proposal_id": proposal.decision_id, "reason": "native_gp6_not_valid_finite"})
                continue
            features = proposal.selection_features
            z = self.native._design(features, proposal.operator).tolist()
            baseline = self.native.baseline(features, proposal.operator)
            if proposal.operator in NORMAL_OPERATORS:
                row = copy.deepcopy(features.get("study7"))
                if (not row or row["parent_id"] != proposal.parent_node_id
                        or row["operator"] != proposal.operator
                        or row["reference_hash"] != self.reference["hash"]
                        or row["z"] != z):
                    raise Study7Error("GP6 closure lost original feature/parent/operator snapshot")
            else:
                row = {"schema": "branch7_gp6_v1", "z": z, "parent_utility": baseline,
                    "operator": proposal.operator, "parent_id": proposal.parent_node_id,
                    "branch_effect_load": 0, "root_lineage_id": None, "rating_s": 0,
                    "rating_level": None, "rating_missing": True,
                    "branch_mapping_version": "root_lineage_primary_v1",
                    "reference_hash": self.reference["hash"], "ancestor_weights": {},
                    "lineage_origin_proposal_id": None, "lineage_assigned_at": None}
            rows.append(row)
            targets.append((proposal.child_utility - baseline) / self.native.utility_scale)
            ids.append({"proposal_id": proposal.decision_id, "transition_id": proposal.transition_id,
                        "outcome_id": proposal.metadata.get("outcome_transition_id", proposal.transition_id)})
        return rows, targets, ids, exclusions

    def state_dict(self):
        return {**super().state_dict(), "lineages": self.lineages.state_dict(),
                "gp6_reference": self.native.state_dict() if self.native else None}

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.lineages.load_state_dict(state["lineages"])
        if self.native:
            self.native.load_state_dict(state["gp6_reference"])
            if self.reference and (self.native.utility_scale != self.reference["s0"]
                                   or self.native.draft_anchor != self.reference["u_ref"]):
                raise Study7Error("Mismatched GP6 frozen reference")
