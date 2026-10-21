"""Acq7 selection committed with the existing locked native reservation ledger."""
from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import json
import os
import threading
import time
import uuid

import numpy as np

from .acq7_acquisition import joint_thompson, knowledge_gradient, selected_envelope
from .acq7_gp import AcqGP
from .acq7_state import Acq7State
from .base import CandidateScore, NORMAL_OPERATORS, SelectionDecision, SelectorResult, utc_timestamp
from .branch7_selector import Branch7Selector
from .study7_gp import Study7Error, digest
from .study7_selector import log_fatal
from .study7_state import scientific_proposals, closed


def option_key(option):
    row = option.state_features["study7"]
    fingerprint = digest({"parent": option.parent_node_id, "operator": option.operator,
        "references": list(option.reference_node_ids), "preview": asdict(option.preview)})
    return (row["birth_commit_index"], NORMAL_OPERATORS.index(option.operator),
            option.parent_node_id, fingerprint)


class Acq7Selector(Branch7Selector):
    def __init__(self, *, config, records_provider, selection_logger):
        super().__init__(config=config, records_provider=records_provider, selection_logger=selection_logger)
        self.name = "acq7_gp4"
        self.state = Acq7State(config)
        self.gp = AcqGP(config, self.state.nodes, selection_logger)
        self.rng_state = None
        self.accepted_draw_index = 0
        self.dispatched_by_branch = {}
        self.committed_draws = {}
        self.last_committed = None
        self._prepared = None

    def _local_rng(self):
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(
            [self.config.workflow_seed, self.config.rng_purpose_id])))
        if self.rng_state is not None:
            rng.bit_generator.state = copy.deepcopy(self.rng_state)
        return rng

    @log_fatal
    def select(self, agent, options):
        started = time.perf_counter()
        if not options or any(option.operator not in NORMAL_OPERATORS for option in options):
            raise Study7Error("Invalid Acq7 ordinary legal pool")
        options = sorted(options, key=option_key)
        keys = [list(option_key(option)) for option in options]
        if len({tuple(key) for key in keys}) != len(keys):
            raise Study7Error("Duplicate legal Acq7 option key")
        records = self.records_provider()
        self.observe(records)
        runtime = agent.phase1_runtime
        revision = runtime.root_draft_state["revision"]
        rows = [copy.deepcopy(option.state_features["study7"]) for option in options]
        for row, key in zip(rows, keys):
            row["option_key"] = key
        scale = self.state.reference["s0"]
        acquisition_started = time.perf_counter()
        joint = self.gp.predict_joint(rows, scale)
        mean, covariance = joint["mean"], joint["latent_covariance"]
        latent, predictive = joint["latent_sd"], joint["predictive_sd"]
        parent = np.asarray([row["parent_utility"] for row in rows])
        candidates_incumbent = [row for row in scientific_proposals(records)
                                if row.valid and closed(row) and row.child_utility is not None]
        # Aggregation may create a valid saved root; include its committed quality.
        candidates_incumbent += [row for row in records
                                 if row.operator == "aggregation" and row.valid and closed(row)]
        incumbent_row = max(candidates_incumbent, key=lambda row: row.child_utility, default=None)
        incumbent = float(incumbent_row.child_utility) if incumbent_row is not None else None
        rng, pre, post = None, None, None
        kind = self.config.acquisition_type
        if kind == "joint_ts":
            rng = self._local_rng()
            pre = copy.deepcopy(rng.bit_generator.state)
            values, acq_details = joint_thompson(mean, covariance, rng, joint["latent_square_root"])
            post = copy.deepcopy(rng.bit_generator.state)
            chosen = min(range(len(options)), key=lambda i: (-float(values[i]), tuple(keys[i])))
            acq_details.update(rng_pre=pre, rng_post=post, sample=values.tolist(),
                official_draw_index=self.accepted_draw_index + 1 if acq_details["draw"] else None)
        else:
            if kind == "retained_outcome_kg" and incumbent is None:
                raise Study7Error("RO-KG has no committed valid incumbent")
            values, acq_details = knowledge_gradient(mean, covariance, joint["noise_variance"],
                incumbent=incumbent if kind == "retained_outcome_kg" else None,
                max_seconds=self.config.max_acquisition_seconds - (time.perf_counter() - acquisition_started))
            chosen = min(range(len(options)), key=lambda i: (-float(values[i]), -float(mean[i]), tuple(keys[i])))
            acq_details["selected_envelope"] = selected_envelope(mean, covariance,
                joint["noise_variance"], chosen, incumbent if kind == "retained_outcome_kg" else None)
        acquisition_seconds = time.perf_counter() - acquisition_started
        if acquisition_seconds > self.config.max_acquisition_seconds:
            raise Study7Error("blocked_numerical: acquisition wall-time limit")
        decision = SelectionDecision(self.name, options[chosen].parent_node_id,
            options[chosen].operator, float(values[chosen]), self.gp.model_version, False,
            uuid.uuid4().hex, utc_timestamp())
        scores = tuple(CandidateScore(parent_node_id=o.parent_node_id, operator=o.operator,
            reference_node_ids=o.reference_node_ids, parent_utility=float(parent[i]),
            mean_gain=float(mean[i] - parent[i]), std_gain=float(latent[i]), predicted_child_ucb=None,
            predicted_cost=None, p_valid=None, acquisition_score=float(values[i]),
            posterior_mean_quality=float(mean[i]), posterior_std_latent=float(latent[i]),
            predictive_std=float(predictive[i])) for i, o in enumerate(options))
        names = [row["root_lineage_id"] for row in rows]
        all_names = sorted({item["root_lineage_id"] for item in self.state.lineages.roots.values()} | set(names))
        supplied = {option.parent_node_id for option in options}
        eligibility = []
        for node in agent.journal.nodes:
            if node.id in supplied: reason = "supplied_by_native_dispatch"
            elif agent.is_root(node): reason = "ordinary_root_acquisition_disabled"
            elif node.is_terminal: reason = "native_terminal"
            elif node.lock: reason = "native_locked"
            elif node.is_buggy is not False or node.is_valid is False: reason = "invalid_or_unreviewed"
            elif getattr(getattr(node, "metric", None), "value", None) is None: reason = "no_valid_metric"
            elif node.reached_child_limit(agent.scfg): reason = "native_child_limit_or_topk_gate"
            else: reason = "excluded_by_native_dispatch_context"
            eligibility.append({"node_id": node.id, "supplied": node.id in supplied, "reason": reason})
        pending = copy.deepcopy(runtime.root_draft_state["pending"])
        pending = {key: value for key, value in pending.items() if value["option"]["operator"] in NORMAL_OPERATORS}
        pending_counts = {name: sum(value.get("lineage", {}).get("root_lineage_id") == name
                                   for value in pending.values()) for name in all_names}
        outcomes = [row for row in scientific_proposals(records) if row.operator in NORMAL_OPERATORS and closed(row)]
        branch_counts = {name: {"dispatched": self.dispatched_by_branch.get(name, 0),
            "pending": pending_counts[name],
            "running": sum(value.get("lineage", {}).get("root_lineage_id") == name
                           and value["status"] == "executing" for value in pending.values()),
            "valid_closed": sum(row.valid and row.selection_features.get("study7", {}).get("root_lineage_id") == name for row in outcomes),
            "failed_closed": sum(not row.valid and row.selection_features.get("study7", {}).get("root_lineage_id") == name for row in outcomes)} for name in all_names}
        if self.config.branch_kind == "constant":
            branch_posterior = self.gp.branch_posterior(all_names, scale)
        elif self.config.branch_kind == "affine":
            branch_posterior = self.gp.affine_coefficients(all_names)
        else:
            branch_posterior = {"fixed_probes": [self.gp.branch_function_posterior(name, [-2., 0., 2.], scale)
                for name in all_names], "actual_candidate_probes": [self.gp.branch_function_posterior(name,
                    sorted({row["performance_v"] for row in rows if row["root_lineage_id"] == name}), scale)
                    for name in sorted(set(names))]}
        choose = lambda value: min(range(len(options)), key=lambda i: (-float(value[i]), tuple(keys[i])))
        mean_best, parent_best, ucb_best = choose(mean), choose(parent), choose(mean + 1.5 * latent)
        header = {"method_id": self.config.method_id, "decision_id": decision.decision_id,
            "model_version": self.gp.model_version, "schema": self.config.schema,
            "contract_hash": digest(self.contract()), "timestamp": decision.timestamp,
            "snapshot_revision": revision, "active_time": max(0., time.time() - agent.start_time),
            "protocol_id": getattr(agent.autoresearch, "protocol_id", None),
            "source_manifest_sha256": os.environ.get("ACQ7_SOURCE_MANIFEST_SHA256"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
            "segment_started_epoch": os.environ.get("MLEVOLVE_SEGMENT_STARTED"),
            "segment_active_offset": os.environ.get("MLEVOLVE_SEGMENT_ACTIVE_OFFSET", "0"),
            "gpu_lease": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cpu_affinity": sorted(os.sched_getaffinity(0)), "selection_thread": threading.current_thread().name,
            "acquisition_kind": kind, "acquisition_seconds": acquisition_seconds,
            "selection_compute_seconds": time.perf_counter() - started,
            "candidate_order": keys, "candidate_order_hash": digest(keys), "candidate_count": len(rows),
            "joint_posterior_hash": joint["posterior_hash"], "joint_numerical": joint["diagnostics"],
            "noise_variance_utility": joint["noise_variance"], "acquisition": acq_details,
            "decision_kind": "single_candidate" if len(rows) == 1 else "ordinary",
            "chosen_index": chosen, "chosen_lineage": names[chosen],
            "posterior_mean_winner_index": mean_best, "parent_score_winner_index": parent_best,
            "shadow_old_ucb15_index": ucb_best, "shadow_policy": "read_only_current_legal_pool",
            "mean_sacrifice": float(mean[mean_best] - mean[chosen]),
            "parent_sacrifice": float(parent[parent_best] - parent[chosen]),
            "legal_lineages": sorted(set(names)), "released_lineages": all_names,
            "native_candidate_eligibility": eligibility, "previous_chosen_lineage": self.previous_lineage,
            "branch_switch": self.previous_lineage is not None and names[chosen] != self.previous_lineage,
            "first_ordinary_decision": not self.dispatched_by_branch,
            "never_dispatched_branch": self.dispatched_by_branch.get(names[chosen], 0) == 0,
            "no_feedback_branch": self.dispatched_by_branch.get(names[chosen], 0) > 0 and branch_counts[names[chosen]]["valid_closed"] == 0,
            "untried_and_not_pending": self.dispatched_by_branch.get(names[chosen], 0) == 0 and not pending_counts[names[chosen]],
            "branch_counts_before_dispatch": branch_counts, "pending_by_lineage": pending_counts,
            "pending_snapshot": {key: {"status": value["status"], "lineage": value.get("lineage"),
                "option": value["option"]} for key, value in pending.items()},
            "reference": copy.deepcopy(self.state.reference), "freeze_diagnostics": self.state.freeze_diagnostics,
            "training_ids": self.gp.fit_diagnostics.get("training_snapshot_ids", []),
            "fit_diagnostics": copy.deepcopy(self.gp.fit_diagnostics),
            "theta_posterior": self.gp.theta_posterior(), "branch_posterior": branch_posterior,
            "incumbent": incumbent, "incumbent_outcome_id": incumbent_row.transition_id if incumbent_row else None,
            "tie_count": int(np.sum(values == values[chosen])),
            "tie_rule": "stable_option_key" if kind == "joint_ts" else "posterior_mean_then_stable_option_key"}
        payloads, self.last_prediction_details = [], {}
        for i, option in enumerate(options):
            details = {**joint["details"][i], "method_id": self.config.method_id,
                "acquisition_kind": kind, "acquisition_score_units": "child_utility" if kind == "joint_ts" else "utility_value_of_information",
                "is_exact_gp_posterior": True, "uncertainty_used": "latent_full",
                "root_lineage_id": names[i], "branch_kind": self.config.branch_kind,
                "joint_covariance_hash": joint["diagnostics"]["covariance_hash"]}
            self.last_prediction_details[(option.parent_node_id, option.operator)] = details
            payloads.append({"parent_id": option.parent_node_id, "operator": option.operator,
                "reference_ids": list(option.reference_node_ids), "preview": asdict(option.preview),
                "state": rows[i], "scores": scores[i].to_dict(), **details})
        self._prepared = {"decision_id": decision.decision_id, "revision": revision,
            "payload": {**header, "candidates": payloads}, "joint": joint,
            "rng_pre": pre, "rng_post": post, "draw": bool(acq_details.get("draw", False))}
        return SelectorResult(decision, options[chosen], scores, tuple(options))

    def validate_reservation(self, agent, result):
        if result.decision.decision_id in self.committed_draws:
            return False
        if result.decision.selector_name != self.name or result.selected_option.operator not in NORMAL_OPERATORS:
            return True
        transaction = self._prepared
        if not transaction or transaction["decision_id"] != result.decision.decision_id:
            raise Study7Error("Acq7 reservation has no matching frozen computation")
        if transaction["revision"] != agent.phase1_runtime.root_draft_state["revision"]:
            from .runtime import SelectionPending
            revision = transaction["revision"]
            self._prepared = None
            if self.logger:
                self.logger._append("acq7_conflicts.jsonl", {"decision_id": result.decision.decision_id,
                    "snapshot_revision": revision, "current_revision": agent.phase1_runtime.root_draft_state["revision"],
                    "rng_advanced": False})
            raise SelectionPending(revision)
        return True

    def reserve(self, decision, option, commit_index):
        info = super().reserve(decision, option, commit_index)
        if (decision.selector_name != self.name or option.operator not in NORMAL_OPERATORS
                or decision.decision_id in self.committed_draws):
            return info
        transaction = self._prepared
        if not transaction or transaction["decision_id"] != decision.decision_id:
            raise Study7Error("Missing Acq7 reservation transaction")
        payload = transaction["payload"]
        joint = transaction["joint"]
        started = time.perf_counter()
        if self.logger:
            directory = self.logger.root / "acq7_joint"
            directory.mkdir(exist_ok=True)
            path = directory / (decision.decision_id + ".npz")
            temporary = path.with_suffix(".npz.tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, mean=joint["mean"], latent_covariance=joint["latent_covariance"],
                    noise_variance=np.asarray(joint["noise_variance"]),
                    candidate_order=np.asarray([json.dumps(key) for key in payload["candidate_order"]]))
            temporary.replace(path)
            payload["joint_artifact"] = {"path": str(path.relative_to(self.logger.root)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        payload["logger_artifact_seconds"] = time.perf_counter() - started
        if transaction["rng_post"] is not None:
            self.rng_state = copy.deepcopy(transaction["rng_post"])
        self.accepted_draw_index += int(transaction["draw"])
        self.previous_lineage = payload["chosen_lineage"]
        self.dispatched_by_branch[self.previous_lineage] = self.dispatched_by_branch.get(self.previous_lineage, 0) + 1
        self.committed_draws[decision.decision_id] = {"draw_index": payload["acquisition"].get("official_draw_index"),
            "sample": payload["acquisition"].get("sample"), "candidate_order_hash": payload["candidate_order_hash"],
            "posterior_hash": payload["joint_posterior_hash"]}
        self.last_committed = copy.deepcopy(payload)
        self._prepared = None
        return info

    def publish_committed(self, decision_id=None):
        if not self.logger or not self.last_committed:
            return
        payload = self.last_committed
        if decision_id is not None and payload["decision_id"] != decision_id:
            return
        path = self.logger.root / "study7_decisions.jsonl"
        if path.exists():
            with path.open() as stream:
                if any(json.loads(line)["decision_id"] == payload["decision_id"] for line in stream if line.strip()):
                    return
        self.logger._append("study7_decisions.jsonl", payload)

    def state_dict(self):
        return {**super().state_dict(), "rng_state": copy.deepcopy(self.rng_state),
            "accepted_draw_index": self.accepted_draw_index, "dispatched_by_branch": self.dispatched_by_branch.copy(),
            "committed_draws": copy.deepcopy(self.committed_draws), "last_committed": copy.deepcopy(self.last_committed)}

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.rng_state = copy.deepcopy(state["rng_state"])
        self.accepted_draw_index = int(state["accepted_draw_index"])
        self.dispatched_by_branch = dict(state["dispatched_by_branch"])
        self.committed_draws = copy.deepcopy(state["committed_draws"])
        self.last_committed = copy.deepcopy(state["last_committed"])
        if self.rng_state is not None:
            self._local_rng()  # Validate the persisted PCG64 state without drawing.
        self._prepared = None
        self.publish_committed()
