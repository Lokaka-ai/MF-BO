"""Explicit managed dispatch for study7 GP-UCB and GP-free parent greediness."""
from __future__ import annotations

import copy
from dataclasses import asdict
from functools import wraps
import math
import time
import uuid

import numpy as np

from .base import CandidateScore, NORMAL_OPERATORS, SelectionDecision, SelectorResult, utc_timestamp
from .study7_gp import State4GP, Study7Error, digest
from .study7_state import Study7State, closed, scientific_proposals


def log_fatal(method):
    @wraps(method)
    def guarded(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Study7Error as exc:
            if self.logger and not getattr(exc, "study7_logged", False):
                self.logger._append("study7_failures.jsonl", {
                    "method_id": self.config.method_id, "status": "blocked/numerical_failure",
                    "stage": method.__name__, "reason": str(exc), "timestamp": utc_timestamp()})
                exc.study7_logged = True
            raise
    return guarded


def winner(options, values):
    if len(options) != len(values) or not np.isfinite(values).all():
        raise Study7Error("Invalid study7 acquisition values")
    # Only mathematically exact equality triggers the stable tie-break.
    return min(range(len(options)), key=lambda index: (
        -float(values[index]), options[index].state_features["study7"]["birth_commit_index"],
        NORMAL_OPERATORS.index(options[index].operator), options[index].parent_node_id))


class Study7Selector:
    def __init__(self, *, config, records_provider, selection_logger, greedy=False):
        self.config, self.records_provider, self.logger = config, records_provider, selection_logger
        self.greedy = greedy
        self.name = "study7_parent_metric_greedy" if greedy else "study7_state4_ucb"
        self.state = None if greedy else Study7State(config)
        self.gp = None if greedy else State4GP(config, self.state.nodes, selection_logger)
        self.decision_index = None
        self.last_prediction_details = {}
        self.outcome_versions = {}

    def contract(self):
        return asdict(self.config)

    @log_fatal
    def features(self, agent, parent, operator):
        if self.greedy:
            return {"birth_commit_index": next(i for i, node in enumerate(agent.journal.nodes) if node.id == parent.id)}
        return self.state.features(agent, parent, operator, self.records_provider())

    @log_fatal
    def observe(self, records):
        if self.state is not None and self.state.reference is not None:
            self.state.sync(records)
            rows, targets, ids, exclusions = self.state.training_data(records)
            changed = self.gp.fit(rows, targets, training_ids=ids)
            if self.logger:
                self.logger.append_model_fit({"method_id": self.config.method_id,
                    "model_version": self.gp.model_version, "status": "admission" if changed else "reused_no_new_quality_label",
                    "N": len(rows), "admission_exclusions": exclusions})
        if self.logger:
            for row in scientific_proposals(records):
                metadata = row.metadata or {}
                eligible = row.operator in NORMAL_OPERATORS and closed(row) and row.valid
                proposal = {"method_id": self.config.method_id, "proposal_id": row.decision_id,
                    "parent_id": row.parent_node_id, "operator": row.operator,
                    "proposal_transition_ids": metadata["proposal_transition_ids"],
                    "proposal_child_ids": metadata["proposal_child_ids"],
                    "scientific_child_id": row.child_node_id if row.valid else None,
                    "closed": closed(row), "valid": row.valid,
                    "state": ("censored" if metadata.get("censored") or not metadata.get("budget_eligible", True)
                              else "valid" if row.valid else "failed" if closed(row) else "pending_repair"),
                    "raw_bpb": row.child_metric_raw if row.valid else None,
                    "raw_utility": row.child_utility if row.valid else None,
                    "train_eligible": bool(eligible and not self.greedy),
                    "residual": ((row.child_utility - row.parent_utility) / self.state.reference["s0"]
                                 if eligible and self.state and self.state.reference else None),
                    "generation_seconds": row.generation_wall_time_sec,
                    "execution_seconds": row.execution_wall_time_sec,
                    "total_seconds": row.total_wall_time_sec,
                    "closed_timestamp": metadata["available_at"] if closed(row) else None,
                    "completed_active_seconds": metadata.get("completed_active_seconds"),
                    "budget_eligible": metadata.get("budget_eligible", True),
                    "late_cleanup_seconds": metadata.get("late_cleanup_seconds", 0),
                    "decision_snapshot": copy.deepcopy(row.selection_features.get("study7")),
                    "outcome_transition_id": metadata["outcome_transition_id"]}
                version = digest(proposal)
                if self.outcome_versions.get(row.decision_id) != version:
                    self.logger._append("study7_proposals.jsonl", proposal)
                    self.outcome_versions[row.decision_id] = version

    @log_fatal
    def select(self, agent, options):
        if not options or any(option.operator not in NORMAL_OPERATORS for option in options):
            raise Study7Error("study7 ordinary acquisition received an invalid option pool")
        records = self.records_provider()
        self.observe(records)
        parent = np.asarray([option.parent_utility for option in options], dtype=np.float64)
        means = latent = predictive = None
        if self.greedy:
            values, model_version = parent, None
            details = [{} for _ in options]
        else:
            rows = [dict(option.state_features["study7"]) for option in options]
            means, latent, predictive, details = self.gp.predict(rows, self.state.reference["s0"])
            values = means + 1.5 * latent
            model_version = self.gp.model_version
            if self.config.rating.enabled:
                shadow, _, _, _ = self.gp.predict(rows, self.state.reference["s0"], shadow=True)
                for i, item in enumerate(details):
                    item["shadow_zero_prior_mean_quality"] = float(shadow[i])
        chosen = winner(options, values)
        parent_winner = winner(options, parent)
        mean_winner = None if self.greedy else winner(options, means)
        d = None if self.greedy else float(means[mean_winner] - means[chosen])
        b = None if self.greedy else float(1.5 * (latent[chosen] - latent[mean_winner]))
        if d is not None and d > b + 1e-10 * max(1.0, abs(float(values[chosen]))):
            raise Study7Error("UCB winner violates the same-pool D <= B identity")
        scores = tuple(CandidateScore(
            parent_node_id=option.parent_node_id, operator=option.operator,
            reference_node_ids=option.reference_node_ids, parent_utility=float(parent[i]),
            mean_gain=None if self.greedy else float(means[i] - parent[i]),
            std_gain=None if self.greedy else float(latent[i]),
            predicted_child_ucb=None if self.greedy else float(values[i]),
            predicted_cost=None, p_valid=None, acquisition_score=float(values[i]),
            posterior_mean_quality=None if self.greedy else float(means[i]),
            posterior_std_latent=None if self.greedy else float(latent[i]),
            predictive_std=None if self.greedy else float(predictive[i])) for i, option in enumerate(options))
        decision = SelectionDecision(self.name, options[chosen].parent_node_id,
            options[chosen].operator, float(values[chosen]), model_version, False,
            uuid.uuid4().hex, utc_timestamp())
        elapsed = max(0.0, time.time() - agent.start_time)
        same_pool = {"method_id": self.config.method_id, "decision_id": decision.decision_id,
            "model_version": model_version, "active_time": elapsed,
            "remaining_budget_seconds": max(0.0, float(agent.acfg.time_limit) - elapsed),
            "decision_kind": "single_candidate" if len(options) == 1 else "ordinary",
            "candidate_count": len(options), "lineage_count": len({option.parent_node.branch_id for option in options}),
            "chosen_index": chosen, "posterior_mean_winner_index": mean_winner,
            "parent_score_winner_index": parent_winner, "D": d, "B": b,
            "ucb_coefficient": None if self.greedy else 1.5,
            "non_tied_mean_change": False if self.greedy else bool(d > 0),
            "mean_change": False if self.greedy else chosen != mean_winner,
            "parent_score_change": chosen != parent_winner,
            "seed": int(agent.acfg.seed), "protocol_id": getattr(agent.autoresearch, "protocol_id", None),
            "reference": copy.deepcopy(self.state.reference) if self.state else None,
            "theta_posterior": self.gp.theta_posterior() if self.gp else None}
        self.last_prediction_details = {}
        candidate_payloads = []
        for i, option in enumerate(options):
            item = {**details[i], "method_id": self.config.method_id,
                "acquisition_kind": "parent_utility" if self.greedy else "study7_latent_ucb",
                "acquisition_score_units": "child_utility", "is_exact_gp_posterior": not self.greedy,
                "uncertainty_used": None if self.greedy else "latent", "D": d, "B": b,
                "mean_winner": i == mean_winner, "parent_score_winner": i == parent_winner}
            self.last_prediction_details[(option.parent_node_id, option.operator)] = item
            candidate_payloads.append({"parent_id": option.parent_node_id, "operator": option.operator,
                "reference_ids": list(option.reference_node_ids), "preview": asdict(option.preview),
                "state": dict(option.state_features["study7"]), "scores": scores[i].to_dict(),
                "lineage": getattr(option.parent_node, "research_direction", None), **item})
        if self.logger:
            self.logger._append("study7_decisions.jsonl", {**same_pool, "candidates": candidate_payloads})
        return SelectorResult(decision, options[chosen], scores, tuple(options))

    def state_dict(self):
        return {"contract": self.contract(), "state": self.state.state_dict() if self.state else None,
                "gp": self.gp.state_dict() if self.gp else None, "outcome_versions": self.outcome_versions.copy()}

    def load_state_dict(self, state):
        if state["contract"] != self.contract():
            raise Study7Error("Cannot resume after changing study7 scale/kernel/prior/label contract")
        if self.greedy:
            if state["gp"] is not None or state["state"] is not None:
                raise Study7Error("E07 must not restore a GP")
        else:
            self.state.load_state_dict(state["state"])
            self.gp.load_state_dict(state["gp"])
        self.outcome_versions = dict(state.get("outcome_versions", {}))
