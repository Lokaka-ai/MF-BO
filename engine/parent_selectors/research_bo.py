"""Development variants; never replace the frozen AQ04 baseline implicitly.

All adaptation uses committed host feedback. No historical witness, score target,
extra LLM, or task-specific construction is supplied to the generator.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import math
from functools import lru_cache

import numpy as np
from scipy.special import ndtr

from .acq7_acquisition import joint_thompson
from .acq7_gp import AcqGP, psd_covariance
from .acq7_selector import Acq7Selector
from .acq7_state import Acq7State
from .base import NORMAL_OPERATORS
from .study7_gp import Study7Error, digest
from .study7_state import scientific_proposals, closed

VARIANTS = ("frontier", "tail_ei", "code_kernel", "portfolio")
VERSION = "parallel_bo_v1"


def restart_due(state):
    """Every eighth post-initial submission, within a separately reserved cap."""
    reserved = [p for p in state["pending"].values()
                if p["status"] == "reserved" and not p.get("recovery_already_submitted")]
    post = state["submitted_count"] + len(reserved) - state["initial_goal"]
    draft_reserved = sum(p["option"]["operator"] == "draft" for p in reserved)
    return (post >= 0 and (post + 1) % 8 == 0
            and state["draft_submitted"] + draft_reserved < state["draft_cap"]
            and not any(p["option"]["operator"] == "draft" for p in state["pending"].values()))


@lru_cache(maxsize=512)
def code_features(code):
    """Stable normalized AST edge/call histogram; no identifiers or constants."""
    vector = np.zeros(128, dtype=float)
    try:
        tree = ast.parse(code or "")
    except (SyntaxError, ValueError):
        raise Study7Error("Cannot featurize a valid parent program as Python AST")
    tokens = []
    for node in ast.walk(tree):
        tokens.append(type(node).__name__)
        tokens.extend(type(node).__name__ + ">" + type(child).__name__
                      for child in ast.iter_child_nodes(node))
        if isinstance(node, ast.Call):
            # Attribute name preserves operations such as eigh/round without
            # embedding variable names, scores, or numeric literal values.
            if isinstance(node.func, ast.Attribute):
                tokens.append("call_attribute:" + node.func.attr)
    for token in tokens:
        bucket = int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "big") % len(vector)
        vector[bucket] += 1
    vector = np.log1p(vector)
    norm = np.linalg.norm(vector)
    if norm:
        vector /= norm
    return tuple(vector.tolist())


class CodeState(Acq7State):
    def features(self, agent, parent, operator, records):
        row = super().features(agent, parent, operator, records)
        row["code_features_v1"] = list(code_features(parent.code))
        return row


class CodeGP(AcqGP):
    """AQ04 plus a fixed PSD code-content x operator covariance component."""
    def covariance(self, left, right, hp=None, *, derivatives=False):
        result = super().covariance(left, right, hp, derivatives=derivatives)
        x = np.asarray([r["code_features_v1"] for r in left], dtype=float).reshape(len(left), 128)
        y = np.asarray([r["code_features_v1"] for r in right], dtype=float).reshape(len(right), 128)
        op = .25 + .75 * (np.asarray([r["operator"] for r in left])[:, None]
                          == np.asarray([r["operator"] for r in right])[None, :])
        code = (x @ y.T) * op  # fixed unit amplitude in normalized gain units
        if derivatives:
            return result[0] + code, result[1], result[2]
        return result[0] + code, {**result[1], "code": code}


def outcome_ei(mean, variance, incumbent):
    """Expected positive improvement of ONE generated outcome, including noise."""
    mean, variance = np.asarray(mean), np.asarray(variance)
    if incumbent is None or not np.isfinite(incumbent) or np.any(variance < 0):
        raise Study7Error("Outcome EI requires a finite incumbent and nonnegative variance")
    sd = np.sqrt(variance)
    delta = mean - incumbent
    z = np.divide(delta, sd, out=np.zeros_like(delta, dtype=float), where=sd > 0)
    result = delta * ndtr(z) + sd * np.exp(-.5 * z*z) / math.sqrt(2 * math.pi)
    result = np.where(sd > 0, result, np.maximum(delta, 0))
    return np.maximum(result, 0)


def mix_joint(joints, weights):
    mean = sum(w * j["mean"] for w, j in zip(weights, joints))
    covariance = sum(w * (j["latent_covariance"] + np.outer(j["mean"] - mean, j["mean"] - mean))
                     for w, j in zip(weights, joints))
    covariance, root, numerical = psd_covariance(covariance)
    noise = float(sum(w * j["noise_variance"] for w, j in zip(weights, joints)))
    result = copy.deepcopy(joints[0])
    result.update(mean=mean, latent_covariance=covariance, latent_square_root=root,
                  latent_sd=np.sqrt(np.diag(covariance).clip(0)), noise_variance=noise,
                  predictive_sd=np.sqrt(np.diag(covariance).clip(0) + noise),
                  diagnostics=numerical, exact_gp=False,
                  posterior_hash=digest({"experts": [j["posterior_hash"] for j in joints],
                                         "weights": list(weights)}))
    # Component-specific diagnostics must not masquerade as mixture diagnostics.
    result["details"] = [{"posterior_family": "moment_matched_two_gp_mixture"} for _ in mean]
    return result


class ResearchBOSelector(Acq7Selector):
    def __init__(self, *, variant, **kwargs):
        if variant not in VARIANTS:
            raise ValueError(f"Unknown research BO variant: {variant}")
        self.variant = variant
        self.forecasts, self.scored = {}, {}
        self.expert_loss = np.zeros(2)
        self._research_decision = {}
        super().__init__(**kwargs)
        if variant in {"code_kernel", "portfolio"}:
            self.state = CodeState(self.config)
            self.gp = (CodeGP if variant == "code_kernel" else AcqGP)(self.config, self.state.nodes, self.logger)
        self.code_gp = CodeGP(self.config, self.state.nodes, self.logger) if variant == "portfolio" else None

    def contract(self):
        return {**super().contract(), "research_bo": self.variant, "research_version": VERSION,
                "restart_period": 8, "coverage_period": 4, "code_kernel_amplitude": 1.,
                "router_discount": .95, "router_rate": .2, "router_min_weight": .1,
                "stagnation_window": 8}

    def observe(self, records):
        # Forecasts are frozen at dispatch, scored once after closed host feedback.
        ordered = sorted(scientific_proposals(records), key=lambda r: (
            str(r.metadata.get("available_at") or ""), r.transition_id))
        for row in ordered:
            if row.decision_id not in self.forecasts or row.decision_id in self.scored or not closed(row):
                continue
            forecast = self.forecasts[row.decision_id]
            if row.valid and row.child_utility is not None:
                scale = forecast["scale"]
                loss = []
                for mean, variance in forecast["experts"]:
                    variance = max(variance / scale**2, 1e-8)
                    error = (float(row.child_utility) - mean) / scale
                    loss.append(.5 * (math.log(variance) + error**2 / variance))
                # Bounded *relative* evidence avoids a single catastrophic residual
                # eliminating an expert permanently. This is a heuristic router.
                relative = np.clip(np.asarray(loss) - min(loss), 0., 5.)
                self.expert_loss = .95 * self.expert_loss + relative
                self.scored[row.decision_id] = {"loss": loss, "outcome": row.transition_id}
            else:
                self.scored[row.decision_id] = {"excluded": "no_valid_quality_label"}
        super().observe(records)

    def predict_joint(self, rows, scale):
        self._rows = rows
        joint = self.gp.predict_joint(rows, scale)
        self._research_decision = {"variant": self.variant, "version": VERSION,
                                   "parent_code_is_only_extra_input": True}
        if self.code_gp is not None:
            self.code_gp.fit(self.gp.rows, self.gp.targets,
                             training_ids=self.gp.fit_diagnostics.get("training_snapshot_ids", []))
            code_joint = self.code_gp.predict_joint(rows, scale)
            weights = np.exp(-.2 * (self.expert_loss - min(self.expert_loss)))
            weights = .1 + .8 * weights / weights.sum()
            joint = mix_joint([joint, code_joint], weights)
            self._research_decision.update(expert_weights=weights.tolist(),
                expert_discounted_loss=self.expert_loss.tolist(), scored_count=len(self.scored),
                forecast_scale=scale, experts=[{
                    "mean": j["mean"].tolist(), "variance": (j["predictive_sd"]**2).tolist()
                } for j in [self.gp.predict_joint(rows, scale), code_joint]])
        return joint

    def acquire(self, joint, keys, incumbent, acquisition_started):
        records = self.records_provider()
        outcomes = sorted([r for r in scientific_proposals(records) if closed(r)],
                          key=lambda r: (str(r.metadata.get("available_at") or ""), r.transition_id))
        best, stale = None, 0
        for row in outcomes:
            improved = row.valid and row.child_utility is not None and (best is None or row.child_utility > best)
            if improved:
                best, stale = row.child_utility, 0
            elif row.operator in NORMAL_OPERATORS:
                stale += 1
        use_ei = self.variant == "tail_ei" or (self.variant == "portfolio" and stale >= 8)
        if use_ei:
            values = outcome_ei(joint["mean"], joint["predictive_sd"]**2, incumbent)
            details = {"draw": False, "policy": "outcome_ei", "incumbent": incumbent,
                       "uses_observation_noise": True}
            pre = post = None
            chosen = min(range(len(keys)), key=lambda i: (-float(values[i]), tuple(keys[i])))
        else:
            values, details, chosen, pre, post = super().acquire(joint, keys, incumbent, acquisition_started)
            details["policy"] = "joint_ts"
        eligible = list(range(len(keys)))
        if self.variant in {"frontier", "portfolio"} and (sum(self.dispatched_by_branch.values()) + 1) % 4 == 0:
            counts = [self.dispatched_by_branch.get(r["root_lineage_id"], 0) for r in self._rows]
            eligible = [i for i, count in enumerate(counts) if count == min(counts)]
            chosen = min(eligible, key=lambda i: (-float(values[i]), tuple(keys[i])))
            details["coverage_reservation"] = True
        else:
            details["coverage_reservation"] = False
        details.update(eligible_indices=eligible, stagnation_closed_proposals=stale,
                       variant=self.variant, acquisition_score_units="expected_positive_utility_gain" if use_ei else "child_utility")
        self._research_decision.update(acquisition_policy=details["policy"],
                                      exact_gp_posterior=self.variant != "portfolio")
        return values, details, chosen, pre, post

    def decision_metadata(self):
        return {"research_bo": copy.deepcopy(self._research_decision)}

    def reserve(self, decision, option, commit_index):
        fresh = decision.decision_id not in self.committed_draws
        prepared = self._prepared
        info = super().reserve(decision, option, commit_index)
        if fresh and prepared and self.variant == "portfolio" and option.operator in NORMAL_OPERATORS:
            payload = prepared["payload"]
            i = payload["chosen_index"]
            meta = payload["research_bo"]
            self.forecasts[decision.decision_id] = {
                "scale": meta["forecast_scale"],
                "experts": [[e["mean"][i], e["variance"][i]] for e in meta["experts"]]}
        return info

    def state_dict(self):
        return {**super().state_dict(), "research_state": {
            "forecasts": copy.deepcopy(self.forecasts), "scored": copy.deepcopy(self.scored),
            "expert_loss": self.expert_loss.tolist(),
            "code_gp": self.code_gp.state_dict() if self.code_gp else None}}

    def load_state_dict(self, state):
        super().load_state_dict(state)
        extra = state["research_state"]
        self.forecasts, self.scored = copy.deepcopy(extra["forecasts"]), copy.deepcopy(extra["scored"])
        self.expert_loss = np.asarray(extra["expert_loss"], dtype=float)
        if self.code_gp:
            self.code_gp.load_state_dict(extra["code_gp"])
