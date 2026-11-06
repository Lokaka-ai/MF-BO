"""Exact state4/ancestor-content/rating GP specified by study7 v1.

No acquisition fallback, target clipping, ID kernel, cost model or piBO prior.
All covariance calculations use float64 and bounded Cholesky/MAP work.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import time

import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.optimize import minimize

from engine.fixed_evaluation import EvaluationError


class Study7Error(EvaluationError):
    """A fatal study7 contract/numerical error, never an acquisition fallback."""


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def rating_value(value):
    return value - 1 if type(value) is int and value in (0, 1, 2) else 0


def matern32(x, y, ell, *, derivative=False):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if (x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or not math.isfinite(ell) or ell <= 0):
        raise Study7Error("Invalid Matérn inputs")
    distance = np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1)
    q = math.sqrt(3.0) * distance / ell
    return np.exp(-q) * (q * q if derivative else 1.0 + q)


def ancestor_kernel(left, right, nodes, ell, *, derivative=False):
    supports = sorted({key for row in [*left, *right] for key in row["ancestor_weights"]})
    if not supports or any(key not in nodes for key in supports):
        raise Study7Error("Missing ancestor content support")
    xi = np.asarray([nodes[key]["xi"] for key in supports], dtype=np.float64)
    if xi.shape != (len(supports), 2):
        raise Study7Error("Ancestor content must have exactly two frozen dimensions")
    matrices = []
    for rows in (left, right):
        matrix = np.asarray([[row["ancestor_weights"].get(key, 0.0) for key in supports]
                             for row in rows], dtype=np.float64)
        if not np.isfinite(matrix).all() or np.any(matrix < 0) or not np.allclose(matrix.sum(1), 1.0, rtol=0, atol=1e-12):
            raise Study7Error("Ancestor rows must be exact nonnegative convex combinations")
        matrices.append(matrix)
    return matrices[0] @ matern32(xi, xi, ell, derivative=derivative) @ matrices[1].T


class State4GP:
    def __init__(self, config, nodes, logger=None):
        self.config, self.nodes, self.logger = config, nodes, logger
        self.hyperparameters = np.array([config.length_scale, config.amplitude, config.noise_std])
        self.rows, self.targets = [], np.empty(0)
        self.factor, self.alpha = None, np.empty(0)
        self.training_hash = None
        self.model_version = None
        self.fit_diagnostics = {}
        self.jitter = 1e-8
        self._restore_training_hash = None

    def covariance(self, left, right, hp=None, *, derivatives=False):
        hp = self.hyperparameters if hp is None else np.asarray(hp, dtype=np.float64)
        if hp.shape != (3,) or not np.isfinite(hp).all() or np.any(hp <= 0):
            raise Study7Error("Invalid study7 hyperparameters")
        ell, amplitude, _ = hp
        zleft = np.asarray([row["z"] for row in left], dtype=np.float64) if left else np.empty((0, 4))
        zright = np.asarray([row["z"] for row in right], dtype=np.float64) if right else np.empty((0, 4))
        if zleft.shape != (len(left), 4) or zright.shape != (len(right), 4):
            raise Study7Error("State4 inputs must have exactly four dimensions")
        op = .25 + .75 * (np.asarray([row["operator"] for row in left])[:, None]
                          == np.asarray([row["operator"] for row in right])[None, :])
        state = amplitude ** 2 * op * matern32(zleft, zright, ell)
        ancestor = np.zeros_like(state)
        if self.config.ancestor.enabled and left and right:
            ancestor = amplitude ** 2 * op * .5 * ancestor_kernel(left, right, self.nodes, ell)
        rating = np.zeros_like(state)
        if self.config.rating.enabled:
            rating = .0625 * np.outer([row["rating_s"] for row in left], [row["rating_s"] for row in right])
        total = state + ancestor + rating
        if not np.isfinite(total).all():
            raise Study7Error("Non-finite study7 covariance")
        if derivatives:
            dell = amplitude ** 2 * op * matern32(zleft, zright, ell, derivative=True)
            if self.config.ancestor.enabled and left and right:
                dell += amplitude ** 2 * op * .5 * ancestor_kernel(left, right, self.nodes, ell, derivative=True)
            return total, dell, 2.0 * (state + ancestor)
        return total, {"state": state, "ancestor": ancestor, "rating": rating}

    def mean(self, rows, *, shadow=False):
        return np.asarray([.25 * row["rating_s"] if self.config.rating.enabled and not shadow else 0.0
                           for row in rows], dtype=np.float64)

    @staticmethod
    def _factor(covariance, attempts):
        if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1] or not np.isfinite(covariance).all():
            raise Study7Error("Invalid training covariance shape/value")
        if not np.allclose(covariance, covariance.T, rtol=1e-12, atol=1e-12):
            raise Study7Error("Asymmetric training covariance")
        for jitter in (1e-8, 1e-7, 1e-6):
            try:
                factor = cholesky(covariance + jitter * np.eye(len(covariance)), lower=True, check_finite=True)
                attempts.append({"jitter": jitter, "success": True})
                return factor, jitter
            except np.linalg.LinAlgError:
                attempts.append({"jitter": jitter, "success": False})
        raise Study7Error("study7/numerical_failure: Cholesky failed at all three registered jitter levels")

    def fit(self, rows, targets, *, training_ids):
        started = time.perf_counter()
        targets = np.asarray(targets, dtype=np.float64)
        if targets.shape != (len(rows),) or not np.isfinite(targets).all():
            raise Study7Error("Invalid study7 training targets")
        key = digest({"rows": rows, "targets": targets.tolist(), "ids": training_ids,
                      "contract": asdict(self.config)})
        if key == self.training_hash:
            return False
        n, attempts, calls = len(rows), [], 0
        eta0 = np.log([1.0, 1.0, .25])
        hp = np.exp(eta0)
        status = "prior_only" if not n else "fixed_prior_conditioned"
        centered = targets - self.mean(rows)
        last_finite_hp = hp.copy()

        class FitLimit(Exception):
            pass

        def objective(eta):
            nonlocal calls, last_finite_hp
            if time.perf_counter() - started > self.config.max_fit_seconds:
                raise Study7Error("study7/numerical_failure: bounded MAP wall-time limit")
            if calls >= self.config.max_objective_calls:
                raise FitLimit
            calls += 1
            parameters = np.exp(eta)
            kernel, dell, damp = self.covariance(rows, rows, parameters, derivatives=True)
            factor, _ = self._factor(kernel + parameters[2] ** 2 * np.eye(n), attempts)
            alpha = cho_solve((factor, True), centered, check_finite=True)
            value = (.5 * centered @ alpha + np.log(np.diag(factor)).sum()
                     + .5 * n * math.log(2 * math.pi) + .5 * np.sum((eta - eta0) ** 2))
            trace_weight = cho_solve((factor, True), np.eye(n), check_finite=True) - np.outer(alpha, alpha)
            gradient = np.asarray([.5 * np.sum(trace_weight * derivative)
                for derivative in (dell, damp, 2 * parameters[2] ** 2 * np.eye(n))]) + eta - eta0
            if not math.isfinite(value) or not np.isfinite(gradient).all():
                raise Study7Error("study7/numerical_failure: non-finite MAP objective")
            last_finite_hp = parameters
            return float(value), gradient

        if self._restore_training_hash == key:
            hp = self.hyperparameters.copy()
            status = "restored_frozen_fit"
        elif n >= self.config.optimize_after:
            try:
                result = minimize(objective, eta0, jac=True, method="L-BFGS-B",
                    bounds=np.log([(0.25, 4.0), (.1, 5.0), (.05, 2.0)]),
                    options={"maxiter": 25, "maxfun": 150, "maxls": 10})
                if not np.isfinite(result.x).all() or not math.isfinite(result.fun):
                    raise Study7Error("study7/numerical_failure: non-finite MAP endpoint")
                if not result.success and not (result.nit >= 25 or result.nfev >= 150 or "LIMIT" in str(result.message).upper()):
                    raise Study7Error(f"study7/numerical_failure: optimizer {result.message}")
                hp = np.exp(result.x)
                status = "map_converged" if result.success else "bounded_optimizer_limit"
            except FitLimit:
                hp, status = last_finite_hp, "bounded_optimizer_limit"
        factor, alpha, jitter = None, np.empty(0), 1e-8
        if n:
            kernel, _ = self.covariance(rows, rows, hp)
            factor, jitter = self._factor(kernel + hp[2] ** 2 * np.eye(n), attempts)
            alpha = cho_solve((factor, True), centered, check_finite=True)
        self.rows, self.targets = list(rows), targets
        self.hyperparameters, self.factor, self.alpha = hp, factor, alpha
        self.training_hash, self.jitter = key, jitter
        self.model_version = "study7-" + digest({"training": key, "hp": hp.tolist()})[:20]
        theta, theta_var = self.theta_posterior()
        self.fit_diagnostics = {"N": n, "status": status, "training_snapshot_ids": training_ids,
            "training_target_hash": digest(targets.tolist()), "training_hash": key,
            "hyperparameters": dict(zip(("ell", "sigma_f", "nu"), hp.tolist())),
            "objective_calls": calls, "cholesky_attempts": attempts, "jitter": jitter,
            "wall_seconds": time.perf_counter() - started,
            "theta_posterior_mean": theta, "theta_posterior_variance": theta_var}
        if self.logger:
            self.logger.append_model_fit({"model_version": self.model_version,
                "method_id": self.config.method_id, "fit_diagnostics": self.fit_diagnostics})
        self._restore_training_hash = None
        return True

    @staticmethod
    def nonnegative_variance(value, diagonal):
        value, diagonal = np.asarray(value), np.asarray(diagonal)
        if not np.isfinite(value).all() or np.any(value < -1e-10 * np.maximum(1.0, diagonal)):
            raise Study7Error("study7/numerical_failure: materially negative posterior variance")
        return np.maximum(value, 0.0)

    def predict(self, rows, scale, *, shadow=False):
        kernel, components = self.covariance(rows, rows)
        diagonal = np.diag(kernel).copy()
        mean, variance = self.mean(rows, shadow=shadow), diagonal.copy()
        if self.factor is not None:
            cross, _ = self.covariance(rows, self.rows)
            alpha = (cho_solve((self.factor, True), self.targets - self.mean(self.rows, shadow=True))
                     if shadow else self.alpha)
            mean += cross @ alpha
            solved = solve_triangular(self.factor, cross.T, lower=True, check_finite=True)
            variance -= np.sum(solved * solved, axis=0)
        variance = self.nonnegative_variance(variance, diagonal)
        parent = np.asarray([row["parent_utility"] for row in rows])
        quality = parent + scale * mean
        latent = scale * np.sqrt(variance)
        predictive = scale * np.sqrt(variance + self.hyperparameters[2] ** 2)
        if not np.isfinite(quality).all():
            raise Study7Error("Non-finite study7 posterior mean")
        details = [{"prior_mean_residual": float(self.mean([row], shadow=shadow)[0]),
                    "posterior_mean_residual": float(mean[i]),
                    "noise_std": float(self.hyperparameters[2]),
                    **{f"diag_{name}": float(matrix[i, i]) for name, matrix in components.items()},
                    "diag_total": float(diagonal[i]), "posterior_variance_residual": float(variance[i]),
                    "prior_only": not self.rows, "jitter": self.jitter}
                   for i, row in enumerate(rows)]
        return quality, latent, predictive, details

    def theta_posterior(self):
        if not self.config.rating.enabled:
            return None, None
        mean, variance = .25, .0625
        if self.factor is not None:
            scores = np.asarray([row["rating_s"] for row in self.rows])
            mean += .0625 * scores @ self.alpha
            variance -= .0625 ** 2 * scores @ cho_solve((self.factor, True), scores)
        variance = float(self.nonnegative_variance(np.asarray(variance), np.asarray(.0625)))
        return float(mean), variance

    def state_dict(self):
        return {"hyperparameters": self.hyperparameters.tolist(), "training_hash": self.training_hash,
                "model_version": self.model_version, "fit_diagnostics": self.fit_diagnostics,
                "jitter": self.jitter}

    def load_state_dict(self, state):
        hp = np.asarray(state["hyperparameters"], dtype=np.float64)
        bounds = np.asarray([(0.25, 4.0), (.1, 5.0), (.05, 2.0)])
        if hp.shape != (3,) or not np.isfinite(hp).all() or np.any(hp < bounds[:, 0]) or np.any(hp > bounds[:, 1]):
            raise Study7Error("Invalid saved study7 hyperparameters")
        self.hyperparameters = hp
        self._restore_training_hash = state["training_hash"]
        self.model_version, self.fit_diagnostics = state["model_version"], state["fit_diagnostics"]
        self.jitter = float(state["jitter"])
