"""Small exact GP for proposal quality; no Q targets or cost model.

k((z,o),(z',o')) = amplitude^2 * Matern32(z,z'; length_scale)
                  * (rho + (1-rho) * 1[o=o']).
Only three scalar hyperparameters are fit (bounded MAP); the six-dimensional
feature transform and cross-operation sharing are fixed. Cholesky inference
uses NumPy/SciPy already required by the project, not BayesianRidge.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
import uuid
from typing import Sequence

import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.optimize import minimize

from .base import ExpansionOption
from .gp_features import GP6_NAMES, GP6_OPERATORS, GP6_SCHEMA, gp_vector
from .proposal_feedback import effective_proposals
from .transition_store import TransitionRecord


@dataclass(frozen=True)
class GP6Config:
    length_scale: float = 1.0
    amplitude: float = 1.0
    noise_std: float = 0.25
    operation_correlation: float = 0.25
    optimize_after: int = 8
    max_optimizer_iterations: int = 25
    history_window: int = 5
    jitter: float = 1e-8

    def __post_init__(self):
        for key in ("length_scale", "amplitude", "noise_std", "jitter"):
            value = float(getattr(self, key))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"gp.{key} must be finite and positive")
        if not math.isfinite(self.operation_correlation) or not 0 <= self.operation_correlation <= 1:
            raise ValueError("gp.operation_correlation must be in [0,1]")
        if self.optimize_after < 2 or self.max_optimizer_iterations < 0 or self.history_window < 1:
            raise ValueError("Invalid GP fit/history limits")
        for value, bounds in zip((self.length_scale, self.amplitude, self.noise_std),
                                 ((0.25, 4.0), (0.1, 5.0), (0.05, 2.0))):
            if not bounds[0] <= value <= bounds[1]:
                raise ValueError(f"GP prior {value} outside supported bounds {bounds}")


def mixed_matern32(x: np.ndarray, ox: np.ndarray, y: np.ndarray, oy: np.ndarray,
                    *, length_scale: float, amplitude: float, rho: float) -> np.ndarray:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    distances = np.sqrt(np.maximum(np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1), 0.0))
    q = math.sqrt(3.0) * distances / length_scale
    operation = rho + (1.0 - rho) * (np.asarray(ox)[:, None] == np.asarray(oy)[None, :])
    return amplitude ** 2 * (1.0 + q) * np.exp(-q) * operation


class GaussianProposalSurrogate:
    feature_schema_version = GP6_SCHEMA
    feature_names = GP6_NAMES

    def __init__(self, *, config: GP6Config | None = None, selection_logger=None):
        self.config = config or GP6Config()
        self.selection_logger = selection_logger
        self.utility_scale = 1.0
        self.draft_anchor = 0.0  # Only used before any feasible outcome exists.
        self.reference_frozen = False
        self.reference_transition_ids: list[str] = []
        self.model_version = None
        self.fitted_transition_count = -1
        self.training_transition_ids: list[str] = []
        self.last_fit_wall_time_sec = 0.0
        self.fit_diagnostics: dict = {}
        self.x = np.empty((0, 6))
        self.ops = np.empty(0, dtype=int)
        self.factor = None
        self.weights = np.empty(0)
        self.hyperparameters = np.array([self.config.length_scale, self.config.amplitude, self.config.noise_std])

    def state_dict(self) -> dict:
        # Exact refit from admitted observations avoids a second GP matrix cache
        # in the journal. Fixed reference values must survive resume.
        return {"reference_frozen": self.reference_frozen,
                "utility_scale": self.utility_scale, "draft_anchor": self.draft_anchor,
                "reference_transition_ids": self.reference_transition_ids}

    def load_state_dict(self, state: dict) -> None:
        scale, anchor = float(state["utility_scale"]), float(state["draft_anchor"])
        if not math.isfinite(scale) or scale <= 0 or not math.isfinite(anchor):
            raise ValueError("Invalid GP reference state")
        self.reference_frozen = bool(state["reference_frozen"])
        self.utility_scale, self.draft_anchor = scale, anchor
        self.reference_transition_ids = list(state["reference_transition_ids"])
        self.fitted_transition_count = -1

    def _freeze_reference(self, usable: Sequence[TransitionRecord]) -> None:
        if self.reference_frozen or not usable:
            return
        drafts = [r for r in usable if r.operator == "draft"]
        seed_records = drafts or list(usable)
        values = np.asarray([r.child_utility for r in seed_records], dtype=float)
        anchor = float(np.median(values))
        # A scale floor tied to observed magnitude avoids exploding normalized
        # gains when initialization scores happen to be nearly equal.
        spread = float(np.quantile(values, .75) - np.quantile(values, .25))
        magnitude = float(np.median(np.abs(values)))
        self.utility_scale = max(spread, 0.05 * magnitude, 1e-6)
        self.draft_anchor = anchor
        self.reference_transition_ids = [r.transition_id for r in seed_records]
        self.reference_frozen = True

    def _design(self, features: dict, operator: str) -> np.ndarray:
        return gp_vector(features, scale=self.utility_scale, draft_anchor=self.draft_anchor, operator=operator)

    def baseline(self, features: dict, operator: str) -> float:
        if operator == "draft":
            return self.draft_anchor
        value = features.get("parent_utility")
        if value is None or not math.isfinite(float(value)):
            raise ValueError("Normal GP candidate requires a finite parent utility")
        return float(value)

    def _kernel(self, x, ox, y, oy, hp=None):
        ell, amplitude, _ = self.hyperparameters if hp is None else hp
        return mixed_matern32(x, ox, y, oy, length_scale=ell,
                              amplitude=amplitude, rho=self.config.operation_correlation)

    def fit(self, records: Sequence[TransitionRecord]) -> None:
        started = time.perf_counter()
        proposals = effective_proposals(records)
        usable = [r for r in proposals if r.valid and r.child_utility is not None
                  and math.isfinite(float(r.child_utility))]
        self._freeze_reference(usable)
        x = np.asarray([self._design(r.selection_features, r.operator) for r in usable], dtype=float).reshape(-1, 6)
        ops = np.asarray([GP6_OPERATORS.index(r.operator) for r in usable], dtype=int)
        y = np.asarray([(r.child_utility - self.baseline(r.selection_features, r.operator)) / self.utility_scale
                        for r in usable], dtype=float)
        if not np.isfinite(y).all():
            raise ValueError("Non-finite GP training targets")
        prior_log = np.log([self.config.length_scale, self.config.amplitude, self.config.noise_std])
        hp = np.exp(prior_log)
        fit_status = "fixed_prior"
        used_jitter = self.config.jitter
        n = len(y)
        if n:
            distances = np.sqrt(np.maximum(np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1), 0.0))
            mixture = self.config.operation_correlation + (1-self.config.operation_correlation) * (ops[:, None] == ops[None, :])

            def objective(log_hp):
                ell, amplitude, noise = np.exp(log_hp)
                q = math.sqrt(3.0) * distances / ell
                base = amplitude ** 2 * np.exp(-q) * mixture
                kernel = base * (1.0 + q)
                covariance = kernel + (noise ** 2 + self.config.jitter) * np.eye(n)
                chol = cholesky(covariance, lower=True, check_finite=False)
                weights = cho_solve((chol, True), y, check_finite=False)
                loss = (0.5 * y.dot(weights) + np.log(np.diag(chol)).sum()
                        + 0.5 * n * math.log(2 * math.pi) + 0.5 * np.sum((log_hp - prior_log) ** 2))
                trace_weight = cho_solve((chol, True), np.eye(n), check_finite=False) - np.outer(weights, weights)
                derivatives = (base * q ** 2, 2 * kernel, 2 * noise ** 2 * np.eye(n))
                gradient = np.asarray([0.5 * np.sum(trace_weight * d) for d in derivatives]) + log_hp - prior_log
                return float(loss), gradient

            if n >= self.config.optimize_after and self.config.max_optimizer_iterations:
                # One deterministic bounded fit, no restarts or optimizer retry loop.
                result = minimize(objective, prior_log, jac=True, method="L-BFGS-B",
                                  bounds=np.log([(0.25, 4.0), (0.1, 5.0), (0.05, 2.0)]),
                                  options={"maxiter": self.config.max_optimizer_iterations, "maxls": 10})
                if np.isfinite(result.fun) and result.fun <= objective(prior_log)[0]:
                    hp = np.exp(result.x)
                    fit_status = "map_converged" if result.success else "map_bounded_iterate"
                else:
                    fit_status = "map_rejected_used_prior"
            covariance = self._kernel(x, ops, x, ops, hp) + hp[2] ** 2 * np.eye(n)
            # At most one numerical stabilization retry. It does not alter the
            # observation noise used for prediction.
            try:
                factor = cholesky(covariance + used_jitter * np.eye(n), lower=True, check_finite=False)
            except np.linalg.LinAlgError:
                used_jitter = max(100 * used_jitter, 1e-6)
                factor = cholesky(covariance + used_jitter * np.eye(n), lower=True, check_finite=False)
            weights = cho_solve((factor, True), y, check_finite=False)
        else:
            factor, weights = None, np.empty(0)
        # Publish a complete fitted state, not partially updated matrices.
        self.x, self.ops, self.factor, self.weights = x, ops, factor, weights
        self.hyperparameters = hp
        self.fitted_transition_count = len(records)
        self.training_transition_ids = [r.transition_id for r in usable]
        self.model_version = f"gp6-{n}-{uuid.uuid4().hex[:10]}"
        self.last_fit_wall_time_sec = time.perf_counter() - started
        self.fit_diagnostics = {
            "training_count": n, "raw_transition_count": len(records),
            "proposal_count": len(proposals),
            "unresolved_proposals": sum(not (r.metadata or {}).get("proposal_closed", True) for r in proposals),
            "closed_invalid_proposals": sum(not r.valid and (r.metadata or {}).get("proposal_closed", True) for r in proposals),
            "debug_repairs": [(r.metadata or {})["debug_repair"] for r in usable if "debug_repair" in (r.metadata or {})],
            "training_outcome_transition_ids": [(r.metadata or {}).get("outcome_transition_id", r.transition_id) for r in usable],
            "training_proposal_costs": [r.total_wall_time_sec for r in usable],
            "utility_scale": self.utility_scale, "draft_anchor": self.draft_anchor,
            "reference_transition_ids": self.reference_transition_ids,
            "target": "continuous_child_utility_residual_given_valid_result",
            "length_scale": float(hp[0]), "amplitude": float(hp[1]), "noise_std": float(hp[2]),
            "operation_correlation": self.config.operation_correlation,
            "optimizer_status": fit_status, "jitter": used_jitter,
        }
        if self.selection_logger:
            self.selection_logger.append_model_fit({
                "model_version": self.model_version, "training_transition_ids": self.training_transition_ids,
                "feature_schema_version": GP6_SCHEMA, "feature_names": list(GP6_NAMES),
                "operator_categories": list(GP6_OPERATORS), "fit_wall_time_sec": self.last_fit_wall_time_sec,
                "fit_diagnostics": self.fit_diagnostics,
            })

    def predict_quality(self, options: Sequence[ExpansionOption]):
        x = np.asarray([self._design(dict(o.state_features), o.operator) for o in options])
        ops = np.asarray([GP6_OPERATORS.index(o.operator) for o in options])
        mean = np.asarray([self.baseline(dict(o.state_features), o.operator) for o in options])
        variance = np.full(len(options), self.hyperparameters[1] ** 2)
        if self.factor is not None:
            cross = self._kernel(x, ops, self.x, self.ops)
            mean = mean + self.utility_scale * cross.dot(self.weights)
            solved = solve_triangular(self.factor, cross.T, lower=True, check_finite=False)
            variance -= np.sum(solved ** 2, axis=0)
        latent = self.utility_scale * np.sqrt(np.maximum(variance, 0.0))
        predictive = np.sqrt(latent ** 2 + (self.utility_scale * self.hyperparameters[2]) ** 2)
        if not np.isfinite(mean).all() or not np.isfinite(predictive).all():
            raise ValueError("Non-finite GP predictions")
        return mean, latent, predictive
