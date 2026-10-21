"""Branch7 managed selection, with same-pool shadows and joint diagnostics."""
from __future__ import annotations

import copy
from dataclasses import asdict
import time
import uuid

import numpy as np

from .base import CandidateScore, NORMAL_OPERATORS, SelectionDecision, SelectorResult, utc_timestamp
from .branch7_gp import BranchGP
from .branch7_state import Branch7State
from .study7_gp import Study7Error, digest
from .study7_selector import Study7Selector, log_fatal, winner
from .study7_state import closed, scientific_proposals


class Branch7Selector(Study7Selector):
    def __init__(self, *, config, records_provider, selection_logger):
        super().__init__(config=config, records_provider=records_provider,
                         selection_logger=selection_logger)
        self.name = f"branch7_gp{config.base_dimension}_ucb"
        self.state = Branch7State(config)
        self.gp = BranchGP(config, self.state.nodes, selection_logger)
        self.fixed_reference_rows = None
        self.previous_lineage = None
        self._selecting = False

    def reserve(self, decision, option, commit_index):
        info = self.state.lineages.reserve(decision, option, commit_index)
        # This is decision-time metadata, never a new GP label or generator input.
        option.state_features["branch7_lineage"] = {
            **info, "branch_effect_load": int(option.operator in NORMAL_OPERATORS),
            "rating_s": option.state_features.get("study7", {}).get("rating_s", 0)}
        return info

    @log_fatal
    def observe(self, records):
        self.state.sync(records)
        # Native GP6 fits at selection after feedback changed; GP4 retains its
        # historical feedback-time fit. Neither freezes a reference on receipt.
        if self.state.reference is not None and (self.config.base_dimension == 4 or self._selecting):
            rows, targets, ids, exclusions = self.state.training_data(records)
            if self.config.base_dimension == 6:
                self.gp.raw_transition_count = len(records)
            changed = self.gp.fit(rows, targets, training_ids=ids)
            if changed and self.logger:
                self.logger._append("branch7_fits.jsonl", {
                    "method_id": self.config.method_id, "model_version": self.gp.model_version,
                    "rows": rows, "targets": targets, "training_ids": ids,
                    "reference": self.state.reference, "gp_state": self.gp.state_dict(),
                    "ancestor_nodes": copy.deepcopy(self.state.nodes) if self.config.ancestor.enabled else {},
                    "excluded": exclusions, "timestamp": utc_timestamp()})
        if not self.logger:
            return
        for row in scientific_proposals(records):
            meta = row.metadata or {}
            eligible = (row.valid and (self.config.base_dimension == 6 or row.operator in NORMAL_OPERATORS)
                        and closed(row))
            payload = {"method_id": self.config.method_id, "proposal_id": row.decision_id,
                "parent_id": row.parent_node_id, "operator": row.operator,
                "proposal_transition_ids": meta["proposal_transition_ids"],
                "proposal_child_ids": meta["proposal_child_ids"],
                "scientific_child_id": row.child_node_id if row.valid else None,
                "closed": closed(row), "valid": row.valid,
                "state": ("censored" if meta.get("censored") or not meta.get("budget_eligible", True)
                          else "valid" if row.valid else "failed" if closed(row) else "pending_repair"),
                "raw_bpb": row.child_metric_raw if row.valid else None,
                "raw_utility": row.child_utility if row.valid else None, "train_eligible": bool(eligible),
                "generation_seconds": row.generation_wall_time_sec,
                "execution_seconds": row.execution_wall_time_sec, "total_seconds": row.total_wall_time_sec,
                "completed_active_seconds": meta.get("completed_active_seconds"),
                "budget_eligible": meta.get("budget_eligible", True),
                "outcome_transition_id": meta["outcome_transition_id"],
                "decision_snapshot": copy.deepcopy(row.selection_features.get("study7")),
                "lineage": self.state.lineages.nodes.get(row.child_node_id),
                "closed_timestamp": meta["available_at"] if closed(row) else None}
            version = digest(payload)
            if self.outcome_versions.get(row.decision_id) != version:
                self.logger._append("study7_proposals.jsonl", payload)
                self.outcome_versions[row.decision_id] = version

    @log_fatal
    def select(self, agent, options):
        if not options or any(option.operator not in NORMAL_OPERATORS for option in options):
            raise Study7Error("Branch7 ordinary selection received an invalid pool")
        self._selecting = True
        try:
            self.observe(self.records_provider())
        finally:
            self._selecting = False
        rows = [copy.deepcopy(o.state_features["study7"]) for o in options]
        scale = self.state.reference["s0"]
        means, latent, predictive, details = self.gp.predict(rows, scale)
        values = means + 1.5 * latent
        # Preserve each base's exact tie rule and current candidate order.
        choose = (lambda values: int(np.argmax(values))) if self.config.base_dimension == 6 else (lambda values: winner(options, values))
        if not np.isfinite(values).all():
            raise Study7Error("Nonfinite Branch7 UCB; fallback prohibited")
        chosen, mean_best = choose(values), choose(means)
        parent = np.asarray([o.parent_utility for o in options])
        d = float(means[mean_best] - means[chosen])
        b = float(1.5 * (latent[chosen] - latent[mean_best]))
        if d > b + 1e-10 * max(1., abs(float(values[chosen]))):
            raise Study7Error("Branch7 same-pool D <= uncertainty advantage violated")
        scores = tuple(CandidateScore(
            parent_node_id=o.parent_node_id, operator=o.operator, reference_node_ids=o.reference_node_ids,
            parent_utility=float(parent[i]), mean_gain=float(means[i] - parent[i]),
            std_gain=float(latent[i]), predicted_child_ucb=float(values[i]),
            predicted_cost=None, p_valid=None, acquisition_score=float(values[i]),
            posterior_mean_quality=float(means[i]), posterior_std_latent=float(latent[i]),
            predictive_std=float(predictive[i])) for i, o in enumerate(options))
        decision = SelectionDecision(self.name, options[chosen].parent_node_id,
            options[chosen].operator, float(values[chosen]), self.gp.model_version, False,
            uuid.uuid4().hex, utc_timestamp())
        # Fixed-HP conditional shadows have no model fit, RNG call or graph mutation.
        without = self.gp.without_branch()
        nb_mean, nb_latent, _, _ = without.predict(rows, scale)
        zero_mean, zero_latent, _, _ = self.gp.predict(rows, scale, shadow=True)
        shadow_branch = choose(nb_mean + 1.5 * nb_latent)
        shadow_rating = choose(zero_mean + 1.5 * zero_latent)
        if self.fixed_reference_rows is None:
            self.fixed_reference_rows = copy.deepcopy(rows)
        ref_mean, ref_sd, _, ref_details = self.gp.predict(self.fixed_reference_rows, scale)
        names = [row["root_lineage_id"] for row in rows]
        pending = getattr(agent.phase1_runtime, "root_draft_state", {}).get("pending", {})
        pending_counts = {name: sum(
            p.get("option", {}).get("operator") in NORMAL_OPERATORS
            and p.get("lineage", {}).get("root_lineage_id") == name for p in pending.values())
            for name in set(names)}
        elapsed = max(0., time.time() - agent.start_time)
        posterior = self.gp.branch_posterior(list(dict.fromkeys([
            *[item["root_lineage_id"] for item in self.state.lineages.roots.values()], *names])), scale)
        header = {"method_id": self.config.method_id, "decision_id": decision.decision_id,
            "model_version": self.gp.model_version, "timestamp": decision.timestamp,
            "active_time": elapsed, "remaining_budget_seconds": max(0., float(agent.acfg.time_limit) - elapsed),
            "candidate_count": len(rows), "lineage_count": len(set(names)),
            "legal_lineages": sorted(set(names)), "chosen_lineage": names[chosen],
            "previous_chosen_lineage": self.previous_lineage,
            "cross_lineage_switch": self.previous_lineage is not None and names[chosen] != self.previous_lineage,
            "pending_by_lineage": pending_counts, "chosen_index": chosen,
            "posterior_mean_winner_index": mean_best, "parent_score_winner_index": choose(parent),
            "decision_kind": "single_candidate" if len(rows) == 1 else "ordinary",
            "D": d, "B": b, "ucb_coefficient": 1.5, "non_tied_mean_change": d > 0,
            "cross_lineage_mean_change": d > 0 and names[chosen] != names[mean_best],
            "shadow_without_branch_index": shadow_branch, "shadow_zero_mean_prior_index": shadow_rating,
            "shadow_protocol": "fixed_fitted_hyperparameters_reconditioned_same_training_rows",
            "reference": copy.deepcopy(self.state.reference), "theta_posterior": self.gp.theta_posterior(),
            "branch_posterior": posterior,
            "fixed_reference": {"rows": self.fixed_reference_rows, "mean": ref_mean.tolist(),
                                "latent_sd": ref_sd.tolist(), "details": ref_details,
                                "hyperparameters": self.gp.hyperparameters.tolist()},
            "protocol_id": getattr(agent.autoresearch, "protocol_id", None),
            "branch_contract_hash": digest(self.contract())}
        self.last_prediction_details = {}
        candidates = []
        for i, option in enumerate(options):
            extra = {**details[i], "method_id": self.config.method_id,
                "acquisition_kind": "branch7_latent_ucb", "acquisition_score_units": "child_utility",
                "is_exact_gp_posterior": True, "uncertainty_used": "latent", "D": d, "B": b,
                "root_lineage_id": names[i], "branch_effect_load": 1,
                "branch_prior_std_normalized": self.config.branch_prior_std,
                "branch_prior_std_utility": scale * self.config.branch_prior_std,
                "shadow_without_branch_mean": float(nb_mean[i]),
                "shadow_without_branch_latent_sd": float(nb_latent[i]),
                "shadow_zero_prior_mean_quality": float(zero_mean[i])}
            self.last_prediction_details[(option.parent_node_id, option.operator)] = extra
            candidates.append({"parent_id": option.parent_node_id, "operator": option.operator,
                "reference_ids": list(option.reference_node_ids), "preview": asdict(option.preview),
                "state": rows[i], "scores": scores[i].to_dict(), **extra})
        if self.logger:
            self.logger._append("study7_decisions.jsonl", {**header, "candidates": candidates})
        self.previous_lineage = names[chosen]
        return SelectorResult(decision, options[chosen], scores, tuple(options))

    def state_dict(self):
        return {**super().state_dict(), "fixed_reference_rows": copy.deepcopy(self.fixed_reference_rows),
                "previous_lineage": self.previous_lineage}

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.fixed_reference_rows = copy.deepcopy(state.get("fixed_reference_rows"))
        self.previous_lineage = state.get("previous_lineage")
