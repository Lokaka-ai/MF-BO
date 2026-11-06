"""A: exact residual GP. B: Laplacian anchor + bounded kernel uncertainty.

The two arms share GP7 representation, ancestry, valid ordinary-proposal data,
reference scale and UCB/LogEI. B explicitly does NOT use the exact GP response mean.
The opt-in Long A arm reuses A's posterior and acquisition, replacing only
its training rows with closed fixed-budget direction-window targets.
Existing GP6, HalfShrink and other selectors are not changed by this module.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import math
import time
import uuid

import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.optimize import minimize

from .base import NORMAL_OPERATORS
from .rating_prior import RatingPriorConfig, potential_level, prior_terms
from .gp_features import GP6_OPERATORS
from .gp_graph_features import (GP7_SCHEMA, GP7_NAMES, GRAPH_SELECTOR_TYPES,
                               GraphConfig, ancestry_matrix, graph_matern32, gp7_vector)
from .gp_surrogate import GaussianProposalSurrogate
from .gp_variants import GPVariantConfig, GPVariantSelector
from .proposal_feedback import effective_proposals
from .gp_long_horizon import (LONG_HORIZON_SELECTOR, LONG_HORIZON_CONTRACT,
                              LongHorizonConfig, long_horizon_targets)


class GraphProposalSurrogate(GaussianProposalSurrogate):
    feature_schema_version = GP7_SCHEMA
    feature_names = GP7_NAMES

    def __init__(self, *, mode: str, graph_config=None, config=None, task_id="", beta=1.5, selection_logger=None,
                 long_horizon=None, pending_provider=None, acquisition="logei"):
        if mode not in GRAPH_SELECTOR_TYPES:
            raise ValueError(f"Unknown graph selector {mode!r}")
        super().__init__(config=config, selection_logger=selection_logger)
        self.mode, self.task_id = mode, str(task_id)
        self.graph_config = GraphConfig(**dict(graph_config or {}))
        self.variant_config = GPVariantConfig(shrink=1., acquisition=acquisition, beta=float(beta))
        self.is_anchor = mode == "laplacian_anchor_logei"
        if self.is_anchor and self.config.max_optimizer_iterations != 0:
            raise ValueError("B requires gp.max_optimizer_iterations=0; it does not train GP hyperparameters")
        if self.is_anchor and not math.isclose(self.config.noise_std**2, self.graph_config.covariance_ridge,
                                               rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("B uses gp.graph.covariance_ridge; keep gp.noise_std**2 consistent for diagnostics")
        self.is_long_horizon = mode == LONG_HORIZON_SELECTOR
        if long_horizon is not None and not self.is_long_horizon:
            raise ValueError("gp.long_horizon requires gp7_long_horizon_logei")
        self.long_horizon_config = LongHorizonConfig(**dict(long_horizon or {})) if self.is_long_horizon else None
        self.pending_provider = pending_provider
        self._pending_signature = None
        self.absolute_loss_scale = 1.0
        self.x = np.empty((0, 7))
        self.chains = []
        self.tau = self.utility_scale
        self.last_prediction_details = {}

    def state_dict(self):
        state = {**super().state_dict(), "mode": self.mode, "task_id": self.task_id,
                 "gp_config": asdict(self.config), "graph_config": asdict(self.graph_config),
                 "absolute_loss_scale": self.absolute_loss_scale}
        if self.is_long_horizon:
            state.update(long_horizon=asdict(self.long_horizon_config),
                         long_horizon_contract=LONG_HORIZON_CONTRACT)
        return state

    def load_state_dict(self, state):
        # Old checkpoints predate the switch: absent means enabled, as before.
        state = {**state, "graph_config": {"use_ancestry_kernel": True, **state.get("graph_config", {})}}
        if self.is_long_horizon and (
            state.get("long_horizon") != asdict(self.long_horizon_config)
            or state.get("long_horizon_contract") != LONG_HORIZON_CONTRACT
        ):
            raise ValueError("Cannot resume Long A with a changed horizon or label contract")
        for key, value in {"mode": self.mode, "task_id": self.task_id,
                           "gp_config": asdict(self.config), "graph_config": asdict(self.graph_config)}.items():
            if state.get(key) != value:
                raise ValueError(f"Cannot resume graph selector with a different {key}")
        loss_scale = float(state["absolute_loss_scale"])
        if not math.isfinite(loss_scale) or loss_scale <= 0:
            raise ValueError("Invalid absolute-loss scale")
        super().load_state_dict(state)
        self.absolute_loss_scale = loss_scale

    def _update_reference(self, usable):
        super()._freeze_reference(usable)
        if usable:
            self.absolute_loss_scale = max(float(np.median(
                [abs(float(r.child_utility)) for r in usable])), 1e-6)

    def _design(self, features, operator):
        return gp7_vector(features, scale=self.utility_scale, draft_anchor=self.draft_anchor,
                          loss_scale=self.absolute_loss_scale, task_id=self.task_id,
                          absolute_mode=self.graph_config.absolute_mode, operator=operator)

    def _kernel_graph(self, x, ops, y, qops, cross_graph, hp=None):
        ell, amplitude, _ = self.hyperparameters if hp is None else hp
        return graph_matern32(x, ops, y, qops, cross_graph, length_scale=ell,
                              amplitude=amplitude, rho=self.config.operation_correlation,
                              graph_weight=self.graph_config.effective_graph_weight)

    def _exact_fit(self, x, ops, y, graph):
        """Same three-parameter bounded MAP and analytic gradient as GP6."""
        prior_log = np.log([self.config.length_scale, self.config.amplitude, self.config.noise_std])
        hp, status, n = np.exp(prior_log), "fixed_prior", len(y)
        if not n:
            return hp, None, np.empty(0), status, self.config.jitter
        distance = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
        operation = self.config.operation_correlation + (1-self.config.operation_correlation) * (ops[:, None] == ops[None, :])
        identity = np.eye(n)
        w = self.graph_config.effective_graph_weight

        def objective(log_hp):
            ell, amplitude, noise = np.exp(log_hp)
            q = math.sqrt(3.) * distance / ell
            exponential = np.exp(-q)
            kernel = amplitude**2 * operation * ((1-w)*(1+q)*exponential + w*graph)
            covariance = kernel + (noise**2+self.config.jitter)*identity
            factor = cholesky(covariance, lower=True, check_finite=False)
            weights = cho_solve((factor, True), y, check_finite=False)
            loss = (0.5*y.dot(weights) + np.log(np.diag(factor)).sum() + 0.5*n*math.log(2*math.pi)
                    + 0.5*np.sum((log_hp-prior_log)**2))
            trace_weight = cho_solve((factor, True), identity, check_finite=False) - np.outer(weights, weights)
            derivatives = (amplitude**2*operation*(1-w)*exponential*q**2,
                           2*kernel, 2*noise**2*identity)
            gradient = np.asarray([0.5*np.sum(trace_weight*d) for d in derivatives]) + log_hp-prior_log
            return float(loss), gradient

        if n >= self.config.optimize_after and self.config.max_optimizer_iterations:
            result = minimize(objective, prior_log, jac=True, method="L-BFGS-B",
                              bounds=np.log([(0.25,4.),(.1,5.),(.05,2.)]),
                              options={"maxiter": self.config.max_optimizer_iterations, "maxls": 10})
            if np.isfinite(result.fun) and result.fun <= objective(prior_log)[0]:
                hp = np.exp(result.x)
                status = "map_converged" if result.success else "map_bounded_iterate"
            else:
                status = "map_rejected_used_prior"
        jitter = self.config.jitter
        covariance = self._kernel_graph(x, ops, x, ops, graph, hp) + hp[2]**2*identity
        try:
            factor = cholesky(covariance+jitter*identity, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            jitter = max(100*jitter, 1e-6)
            factor = cholesky(covariance+jitter*identity, lower=True, check_finite=False)
        return hp, factor, cho_solve((factor, True), y, check_finite=False), status, jitter

    def fit(self, records):
        started = time.perf_counter()
        proposals = effective_proposals(records)
        valid = [r for r in proposals if r.valid and r.child_utility is not None
                 and math.isfinite(float(r.child_utility))]
        self._update_reference(valid)  # Drafts calibrate scale, NOT training labels.
        long_diagnostics = {}
        if self.is_long_horizon:
            pending = self.pending_provider() if self.pending_provider else ()
            usable, long_diagnostics = long_horizon_targets(
                records, horizon=self.long_horizon_config.horizon, pending=pending)
        else:
            usable = [r for r in valid if r.operator in NORMAL_OPERATORS
                      and (r.metadata or {}).get("proposal_closed", True)]
        for row in usable:
            if not row.selection_features.get("gp_primary_ancestry"):
                raise ValueError("Missing frozen GP7 ancestry; start a new run, do not migrate old GP6 logs")
            if row.parent_utility is None or not math.isclose(float(row.parent_utility),
                    float(row.selection_features["parent_utility"]), rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("Proposal parent baseline differs from its decision-time snapshot")
        n = len(usable)
        x = np.asarray([self._design(r.selection_features, r.operator) for r in usable], float).reshape(-1, 7)
        ops = np.asarray([GP6_OPERATORS.index(r.operator) for r in usable], int)
        chains = [list(r.selection_features["gp_primary_ancestry"]) for r in usable]
        p = ancestry_matrix(chains, self.graph_config.ancestry_decay)
        graph = p @ p.T
        y = np.asarray([(float(r.child_utility)-float(r.parent_utility))/self.utility_scale for r in usable])
        if not np.isfinite(y).all():
            raise ValueError("Non-finite GP7 residual target")
        extras = {}
        if self.is_anchor:
            hp = np.array([self.config.length_scale, 1., math.sqrt(self.graph_config.covariance_ridge)])
            kernel = self._kernel_graph(x, ops, x, ops, graph, hp)
            factor = cholesky(kernel+self.graph_config.covariance_ridge*np.eye(n), lower=True) if n else None
            weights = np.empty(0)  # No response-mean weights, deliberately.
            if n:
                errors = np.asarray([float(r.child_utility)-float(r.selection_features["gp_graph_center"]) for r in usable])
                if not np.isfinite(errors).all():
                    raise ValueError("Non-finite frozen-center errors")
                # Stable RMS without overflowing the square of a finite outlier.
                largest = float(np.max(np.abs(errors)))
                rms = largest*math.sqrt(float(np.mean((errors/largest)**2))) if largest else 0.
                rms = max(rms, 1e-6*self.utility_scale)
                tau = max(1e-6*self.utility_scale, min(rms, self.graph_config.amplitude_cap*self.utility_scale))
                status = "fixed_kernel_bounded_rms"
            else:
                # No extra warmup policy: keep native initial drafts. Until the
                # first ordinary feedback, use the shared normalized prior SD.
                rms = self.config.amplitude*self.utility_scale
                tau = max(1e-6*self.utility_scale, min(rms, self.graph_config.amplitude_cap*self.utility_scale))
                status = "cold_start_prior_no_ordinary_feedback"
            jitter = 0.
            extras = {"rms_scale": rms, "scale_used": tau,
                      "scale_ceiling_active": tau < rms, "covariance_ridge": self.graph_config.covariance_ridge,
                      "center_target": "outcome_minus_frozen_graph_center",
                      "mean_fitted": False, "is_exact_gp_posterior": False}
        else:
            hp, factor, weights, status, jitter = self._exact_fit(x, ops, y, graph)
            tau = float(hp[1])*self.utility_scale
            extras = {"mean_fitted": True, "is_exact_gp_posterior": True}
        self.x, self.ops, self.chains = x, ops, chains
        self.hyperparameters, self.factor, self.weights, self.tau = hp, factor, weights, tau
        self.training_transition_ids = [r.transition_id for r in usable]
        self.fitted_transition_count = len(records)
        self.model_version = f"{self.mode}-{n}-{uuid.uuid4().hex[:10]}"
        self.last_fit_wall_time_sec = time.perf_counter()-started
        self.fit_diagnostics = {
            "training_count": n, "raw_transition_count": len(records), "proposal_count": len(proposals),
            "drafts_used_for_quality_fit": False, "valid_ordinary_only": not self.is_long_horizon,
            "training_outcome_transition_ids": [(r.metadata or {}).get("outcome_transition_id", r.transition_id) for r in usable],
            "training_available_at": [(r.metadata or {}).get("available_at", r.timestamp) for r in usable],
            "debug_repairs": [(r.metadata or {})["debug_repair"] for r in usable if "debug_repair" in (r.metadata or {})],
            "utility_scale": self.utility_scale, "draft_anchor": self.draft_anchor,
            "absolute_loss_scale": self.absolute_loss_scale, "reference_transition_ids": self.reference_transition_ids,
            "target": "continuous_child_utility_residual_given_valid_result" if not self.is_anchor else "fixed_anchor_error_scale",
            "length_scale": float(hp[0]), "amplitude": float(hp[1]), "noise_std": float(hp[2]),
            "operation_correlation": self.config.operation_correlation, "graph_config": asdict(self.graph_config),
            "optimizer_status": status, "jitter": jitter, **extras}
        if self.is_long_horizon:
            self.fit_diagnostics.update(long_diagnostics,
                target="fixed_h_best_utility_parent_residual", failed_attempts_consume_budget=True,
                acquisition_sigma="latent")
        if self.selection_logger:
            self.selection_logger.append_model_fit({"model_version": self.model_version,
                "training_transition_ids": self.training_transition_ids,
                "feature_schema_version": GP7_SCHEMA, "feature_names": list(GP7_NAMES),
                "operator_categories": list(NORMAL_OPERATORS), "fit_wall_time_sec": self.last_fit_wall_time_sec,
                "fit_diagnostics": self.fit_diagnostics})

    def predict_quality(self, options):
        if not options:
            return np.empty(0), np.empty(0), np.empty(0)
        x = np.asarray([self._design(dict(o.state_features), o.operator) for o in options])
        ops = np.asarray([GP6_OPERATORS.index(o.operator) for o in options])
        chains = [list(o.state_features["gp_primary_ancestry"]) for o in options]
        joint = ancestry_matrix([*self.chains, *chains], self.graph_config.ancestry_decay)
        n = len(self.chains)
        cross = self._kernel_graph(x, ops, self.x, self.ops, joint[n:] @ joint[:n].T)
        variance = np.full(len(options), self.hyperparameters[1]**2)
        if self.factor is not None:
            projected = solve_triangular(self.factor, cross.T, lower=True, check_finite=False)
            variance -= np.sum(projected**2, axis=0)
        variance = np.maximum(variance, 1e-12 if self.is_anchor else 0.)
        if self.is_anchor:
            mean = np.asarray([float(o.state_features["gp_graph_center"]) for o in options])
            latent = self.tau*np.sqrt(variance)
            predictive = self.tau*np.sqrt(variance+self.graph_config.covariance_ridge)
        else:
            base = np.asarray([o.parent_utility for o in options], float)
            mean = base + self.utility_scale*(cross @ self.weights if n else np.zeros(len(options)))
            latent = self.utility_scale*np.sqrt(variance)
            predictive = self.utility_scale*np.sqrt(variance+self.hyperparameters[2]**2)
        if not all(np.isfinite(v).all() for v in (mean, latent, predictive)):
            raise ValueError("Non-finite graph prediction")
        self.last_prediction_details = {(o.parent_node_id, o.operator): {
            "normalized_kernel_variance": float(variance[i]),
            "predictive_center": float(mean[i]), "rms_scale": self.fit_diagnostics.get("rms_scale"),
            "scale_used": self.tau, "scale_ceiling_active": self.fit_diagnostics.get("scale_ceiling_active", False),
            "is_exact_gp_posterior": not self.is_anchor} for i, o in enumerate(options)}
        return mean, latent, predictive


class GraphProposalSelector(GPVariantSelector):
    def __init__(self, *, surrogate, records_provider, rating_config=None):
        self.rating_config = rating_config or RatingPriorConfig()
        if surrogate.graph_config.xi != 0:
            raise ValueError("The two comparison arms require global LogEI xi=0")
        super().__init__(surrogate=surrogate, records_provider=records_provider)
        self.name = surrogate.mode

    def select(self, agent, options):
        # Pending cancellation/completion can unlock a window without changing
        # the raw row count. Only Long A needs this additional cache key.
        if self.surrogate.is_long_horizon and self.surrogate.pending_provider:
            pending = self.surrogate.pending_provider()
            signature = tuple(sorted((p["decision_id"], p["decision_index"],
                                      p["parent_node_id"], p["operator"]) for p in pending))
            if signature != self.surrogate._pending_signature:
                self.surrogate.fitted_transition_count = -1
                self.surrogate._pending_signature = signature
        result = super().select(agent, options)
        if not self.rating_config.enabled or result.decision.fallback_used:
            return result
        # All options were frozen at the same ordinary BO decision; do not use
        # the all-stage decision_index or the delayed GP training sample count.
        iterations = [o.state_features.get("pibo_iteration") for o in options]
        if not iterations or any(n != iterations[0] for n in iterations):
            raise ValueError("Missing/inconsistent piBO selection iteration")
        levels = [potential_level(o.state_features.get("agent_potential_level")) for o in options]
        log_weights, bonuses = prior_terms(levels, beta_prior=self.rating_config.beta_prior,
                                          iteration=iterations[0])
        base = np.asarray([s.acquisition_score for s in result.candidate_scores], float)
        use_ucb = self.surrogate.variant_config.acquisition == "ucb"
        if use_ucb:
            # log(exp(UCB / scale) * prior**(beta_prior / t)), in utility units.
            bonuses = self.surrogate.utility_scale * bonuses
        with np.errstate(over="ignore"):
            # Preserve the existing JSON-safe representation of exact zero EI.
            values = np.maximum(base + bonuses, -np.finfo(float).max)
        scores = tuple(replace(s, acquisition_score=float(values[i]))
                       for i, s in enumerate(result.candidate_scores))
        best = int(np.argmax(values))
        for i, o in enumerate(options):
            self.surrogate.last_prediction_details.setdefault((o.parent_node_id, o.operator), {}).update(
                {"base_ucb" if use_ucb else "base_logei": float(base[i])}, agent_potential_level=levels[i],
                rating_prior_log_weight=float(log_weights[i]), rating_prior_bonus=float(bonuses[i]),
                rating_prior_beta=float(self.rating_config.beta_prior), pibo_iteration=int(iterations[0]))
        return replace(result, decision=replace(result.decision,
            selected_parent_id=options[best].parent_node_id, selected_operator=options[best].operator,
            selected_score=float(values[best])), selected_option=options[best], candidate_scores=scores)
