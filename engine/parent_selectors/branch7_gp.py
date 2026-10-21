"""Integrated branch random effects with the two historical fitting policies."""
from __future__ import annotations

import copy
from dataclasses import asdict, replace
import math
import time

import numpy as np
from scipy.linalg import cholesky, cho_solve
from scipy.optimize import minimize

from .gp_surrogate import mixed_matern32
from .study7_gp import State4GP, Study7Error, digest


def branch_loading(rows):
    loads, names = [], []
    for row in rows:
        h, name = row.get("branch_effect_load"), row.get("root_lineage_id")
        if type(h) is not int or h not in (0, 1):
            raise Study7Error("Branch7 requires an explicit binary training/candidate loading")
        if h and (not isinstance(name, str) or not name):
            raise Study7Error("Ordinary Branch7 input has no root lineage")
        if not h and row.get("rating_s", 0) != 0:
            raise Study7Error("Nonordinary GP6 rows must have zero rating loading")
        loads.append(h)
        names.append(name if h else None)
    return np.asarray(loads, dtype=float), names


def branch_kernel(left, right, tau):
    if not math.isfinite(tau) or tau < 0:
        raise Study7Error("Invalid fixed branch prior standard deviation")
    hl, bl = branch_loading(left)
    hr, br = branch_loading(right)
    equal = np.asarray([[a == b for b in br] for a in bl], dtype=float).reshape(len(left), len(right))
    return tau ** 2 * hl[:, None] * hr[None, :] * equal


class BranchGP(State4GP):
    """GP4 keeps State4GP.fit; GP6 keeps its archived MAP acceptance policy.

    Both use a single full covariance for training, prediction and diagnostics.
    Fixed branch/rating components never enter the amplitude derivative.
    """
    def covariance(self, left, right, hp=None, *, derivatives=False):
        hp = self.hyperparameters if hp is None else np.asarray(hp, dtype=float)
        if hp.shape != (3,) or not np.isfinite(hp).all() or np.any(hp <= 0):
            raise Study7Error("Invalid Branch7 hyperparameters")
        branch = branch_kernel(left, right, self.config.branch_prior_std)
        if self.config.base_dimension == 4:
            values = super().covariance(left, right, hp, derivatives=derivatives)
            if derivatives:
                return values[0] + branch, values[1], values[2]
            return values[0] + branch, {**values[1], "branch": branch}
        if self.config.base_dimension != 6 or self.config.ancestor.enabled:
            raise Study7Error("GP6 Branch7 cannot change dimension or enable ancestor B")
        arrays = []
        for rows in (left, right):
            x = np.asarray([row["z"] for row in rows], dtype=float).reshape(len(rows), 6)
            if not np.isfinite(x).all():
                raise Study7Error("Nonfinite GP6 Branch7 features")
            arrays.append(x)
        x, y = arrays
        ox = np.asarray([row["operator"] for row in left])
        oy = np.asarray([row["operator"] for row in right])
        ell, amplitude, _ = hp
        state = mixed_matern32(x, ox, y, oy, length_scale=ell,
                              amplitude=amplitude, rho=.25)
        rating = (.0625 * np.outer([row["rating_s"] for row in left],
                                  [row["rating_s"] for row in right])
                  if self.config.rating.enabled else np.zeros_like(state))
        total = state + branch + rating
        if not np.isfinite(total).all():
            raise Study7Error("Nonfinite Branch7 covariance")
        if derivatives:
            q = math.sqrt(3) * np.sqrt(np.maximum(np.sum((x[:, None] - y[None, :]) ** 2, axis=-1), 0)) / ell
            mixture = .25 + .75 * (ox[:, None] == oy[None, :])
            return total, amplitude ** 2 * np.exp(-q) * mixture * q ** 2, 2 * state
        return total, {"state": state, "ancestor": np.zeros_like(state), "rating": rating, "branch": branch}

    def objective(self, rows, targets, eta):
        """GP6's exact registered objective and analytic log-hyperparameter gradient."""
        hp = np.exp(eta)
        kernel, dell, damp = self.covariance(rows, rows, hp, derivatives=True)
        n = len(rows)
        centered = np.asarray(targets) - self.mean(rows)
        factor = cholesky(kernel + (hp[2] ** 2 + 1e-8) * np.eye(n), lower=True, check_finite=True)
        alpha = cho_solve((factor, True), centered)
        eta0 = np.log([1., 1., .25])
        value = (.5 * centered @ alpha + np.log(np.diag(factor)).sum()
                 + .5 * n * math.log(2 * math.pi) + .5 * np.sum((eta - eta0) ** 2))
        trace = cho_solve((factor, True), np.eye(n)) - np.outer(alpha, alpha)
        gradient = np.asarray([.5 * np.sum(trace * d)
            for d in (dell, damp, 2 * hp[2] ** 2 * np.eye(n))]) + eta - eta0
        if not math.isfinite(value) or not np.isfinite(gradient).all():
            raise Study7Error("Nonfinite Branch7 MAP objective/gradient")
        return float(value), gradient

    def _factor_gp6(self, kernel, noise):
        attempts = []
        for jitter in (1e-8, 1e-6):
            try:
                factor = cholesky(kernel + (noise ** 2 + jitter) * np.eye(len(kernel)),
                                  lower=True, check_finite=True)
                return factor, jitter, [*attempts, {"jitter": jitter, "success": True}]
            except np.linalg.LinAlgError:
                attempts.append({"jitter": jitter, "success": False})
        raise Study7Error("Branch7 GP6 Cholesky failed; no fallback is permitted")

    def fit(self, rows, targets, *, training_ids):
        # Validate even on an otherwise cache-equivalent call.
        branch_loading(rows)
        if self.config.base_dimension == 4:
            return super().fit(rows, targets, training_ids=training_ids)
        started = time.perf_counter()
        targets = np.asarray(targets, dtype=float)
        if targets.shape != (len(rows),) or not np.isfinite(targets).all():
            raise Study7Error("Invalid GP6 Branch7 targets")
        key = digest({"rows": rows, "targets": targets.tolist(), "ids": training_ids,
                      "raw_transition_count": getattr(self, "raw_transition_count", None),
                      "contract": asdict(self.config)})
        if key == self.training_hash:
            return False
        eta0 = np.log([1., 1., .25])
        hp, status, calls = np.exp(eta0), "fixed_prior", 0
        def objective(eta):
            nonlocal calls
            calls += 1
            return self.objective(rows, targets, eta)
        try:
            if self._restore_training_hash == key:
                hp, status = self.hyperparameters.copy(), "restored_frozen_fit"
            elif len(rows) >= self.config.optimize_after:
                result = minimize(objective, eta0, jac=True, method="L-BFGS-B",
                    bounds=np.log([(.25, 4.), (.1, 5.), (.05, 2.)]),
                    options={"maxiter": 25, "maxls": 10})
                if not np.isfinite(result.x).all() or not math.isfinite(result.fun):
                    raise Study7Error("Nonfinite GP6 Branch7 optimizer endpoint")
                if result.fun <= objective(eta0)[0]:
                    hp = np.exp(result.x)
                    status = "map_converged" if result.success else "map_bounded_iterate"
                else:
                    status = "map_rejected_used_prior"
            factor, alpha, jitter, attempts = None, np.empty(0), 1e-8, []
            if len(rows):
                kernel, _ = self.covariance(rows, rows, hp)
                factor, jitter, attempts = self._factor_gp6(kernel, hp[2])
                alpha = cho_solve((factor, True), targets - self.mean(rows))
        except (np.linalg.LinAlgError, ValueError, FloatingPointError) as exc:
            raise Study7Error(f"Branch7 GP6 numerical failure: {exc}") from exc
        self.rows, self.targets = copy.deepcopy(rows), targets.copy()
        self.factor, self.alpha, self.hyperparameters = factor, alpha, hp
        self.training_hash, self.jitter = key, jitter
        self.model_version = "branch7-gp6-" + digest({"training": key, "hp": hp.tolist()})[:20]
        self.fit_diagnostics = {"N": len(rows), "status": status, "training_snapshot_ids": training_ids,
            "training_target_hash": digest(targets.tolist()), "training_hash": key,
            "hyperparameters": dict(zip(("ell", "sigma_f", "nu"), hp.tolist())),
            "objective_calls": calls, "cholesky_attempts": attempts, "jitter": jitter,
            "wall_seconds": time.perf_counter() - started,
            "optimizer_policy": "archived_gp6_single_start_25_maxls10_accept_if_no_worse_than_prior"}
        if self.logger:
            self.logger.append_model_fit({"model_version": self.model_version,
                "method_id": self.config.method_id, "fit_diagnostics": self.fit_diagnostics})
        self._restore_training_hash = None
        return True

    def branch_posterior(self, names, scale):
        names = list(dict.fromkeys(names))
        if any(not isinstance(name, str) or not name for name in names):
            raise Study7Error("Invalid lineage requested for branch posterior")
        loads, ids = branch_loading(self.rows)
        z = np.asarray([[h * (name == branch) for branch in names]
                        for h, name in zip(loads, ids)], dtype=float).reshape(len(self.rows), len(names))
        tau2 = self.config.branch_prior_std ** 2
        mean, covariance = np.zeros(len(names)), tau2 * np.eye(len(names))
        if self.factor is not None:
            mean = tau2 * z.T @ self.alpha
            covariance -= tau2 ** 2 * z.T @ cho_solve((self.factor, True), z)
        diag = self.nonnegative_variance(np.diag(covariance), np.full(len(names), tau2))
        return {"lineages": names, "mean": mean.tolist(), "covariance": covariance.tolist(),
                "sd": np.sqrt(diag).tolist(), "mean_utility": (scale * mean).tolist(),
                "sd_utility": (scale * np.sqrt(diag)).tolist(),
                "observed_expansions": z.sum(0).astype(int).tolist(),
                "branch_prior_std_normalized": self.config.branch_prior_std,
                "branch_prior_std_utility": scale * self.config.branch_prior_std}

    def without_branch(self):
        """Same observed inputs and fitted HP, reconditioned for a diagnostic shadow."""
        other = copy.copy(self)
        other.config = replace(self.config, branch_prior_std=0.)
        if self.rows:
            kernel, _ = other.covariance(self.rows, self.rows)
            if self.config.base_dimension == 6:
                other.factor, other.jitter, _ = other._factor_gp6(kernel, self.hyperparameters[2])
            else:
                other.factor, other.jitter = other._factor(kernel + self.hyperparameters[2] ** 2 * np.eye(len(self.rows)), [])
            other.alpha = cho_solve((other.factor, True), self.targets - other.mean(self.rows))
        return other
