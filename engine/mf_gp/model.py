"""Noisy autoregressive co-kriging over frozen contexts and program identities.

L_p = g(x_p) + u_p; H_p = a + rho L_p + d(x_p) + v_p.
Independent u/v effects persist across evaluations of one executable. They are
not execution noise and are not shared by two children of the same parent.
All scores have a single frozen task scale and higher-is-better orientation.
"""

from __future__ import annotations
import hashlib
import json
import math
import time
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import ndtr
from engine.parent_selectors.acq7_acquisition import gaussian_envelope

# Deliberately small, versioned contract. Context lengthscale and noise stay
# fixed: one evaluation per fidelity cannot identify all variance components.
CONTRACT = dict(
    version=1,
    lengthscale=1.0,
    identity_variance=0.75**2,
    noise_std_low=0.10,
    noise_std_high=0.10,
    fit_min_high=4,
    fit_min_total=8,
    rho_prior=0.8,
    rho_prior_std=0.5,
    discrepancy_prior=0.7,
    pending="posterior_mean_conditioning",
    target="absolute_signed_quality",
    cost_margin=1.5,
)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


class MultiFidelityGP:
    def __init__(self):
        self.rows, self.values = [], np.empty(0)
        self.center, self.scale = 0.0, 1.0
        self.reference_frozen = False
        self.theta = np.array([0.8, 0.0, 0.0, math.log(0.7)])
        self.factor = None
        self.alpha = np.empty(0)
        self.training_hash = None
        self.model_version = "mf_gp_prior"
        self.fit_diagnostics = {}
        self.restored_hash = None

    @staticmethod
    def _base(left, right):
        if not left or not right:
            return np.zeros((len(left), len(right)))
        x, y = np.array([r["x"] for r in left]), np.array([r["x"] for r in right])
        distance = np.sqrt(np.maximum(0.0, ((x[:, None] - y[None, :]) ** 2).sum(-1)))
        z = math.sqrt(3.0) * distance / CONTRACT["lengthscale"]
        operator = np.array(
            [
                [1.0 if a["operator"] == b["operator"] else 0.25 for b in right]
                for a in left
            ]
        )
        same = np.array(
            [[a["program_id"] == b["program_id"] for b in right] for a in left]
        )
        return (1.0 + z) * np.exp(-z) * operator + CONTRACT["identity_variance"] * same

    def covariance(self, left, right, theta=None):
        rho, _, log_low, log_delta = self.theta if theta is None else theta
        base = self._base(left, right)
        l = np.array([r["fidelity"] == 1.0 for r in left], dtype=float)
        r = np.array([r["fidelity"] == 1.0 for r in right], dtype=float)
        load_l, load_r = 1.0 + (rho - 1.0) * l, 1.0 + (rho - 1.0) * r
        return base * (
            np.exp(2 * log_low) * np.outer(load_l, load_r)
            + np.exp(2 * log_delta) * np.outer(l, r)
        )

    def mean(self, rows, theta=None):
        t = self.theta if theta is None else theta
        return np.array([t[1] if r["fidelity"] == 1.0 else 0.0 for r in rows])

    def noise(self, rows):
        return np.array(
            [
                (
                    CONTRACT["noise_std_high"]
                    if r["fidelity"] == 1.0
                    else CONTRACT["noise_std_low"]
                )
                ** 2
                for r in rows
            ]
        )

    def fit(self, observations, reference=None):
        # Invalid/censored observations never become numeric quality labels.
        admitted = [
            r for r in observations if r.get("valid") and r.get("budget_eligible", True)
        ]
        if any(
            not math.isfinite(r["utility"]) or r["fidelity"] not in (0.5, 1.0)
            for r in admitted
        ):
            raise ValueError("Invalid MF observation")
        if reference is not None and not self.reference_frozen:
            y = np.array(reference, dtype=float)
            if len(y):
                self.center = float(np.median(y))
                self.scale = max(float(np.ptp(y)), 0.05 * abs(self.center), 1e-6)
                self.reference_frozen = True
        signature = digest([admitted, self.center, self.scale, CONTRACT])
        if signature == self.training_hash:
            return False
        self.rows = [
            {k: r[k] for k in ("program_id", "x", "operator", "fidelity")}
            for r in admitted
        ]
        self.values = np.array(
            [(r["utility"] - self.center) / self.scale for r in admitted]
        )
        prior = np.array([0.8, 0.0, 0.0, math.log(0.7)])

        def factor(t):
            k = self.covariance(self.rows, self.rows, t)
            k += np.diag(self.noise(self.rows) + 1e-8)
            return cho_factor(k, lower=True, check_finite=True)

        started = time.monotonic()
        best = [float("inf"), prior.copy()]

        class FitBudget(Exception):
            pass

        def objective(t):
            if time.monotonic() - started > 20.0:
                raise FitBudget()
            f = factor(t)
            residual = self.values - self.mean(self.rows, t)
            value = (
                float(0.5 * residual @ cho_solve(f, residual))
                + float(np.log(np.diag(f[0])).sum())
                + 0.5
                * float(np.square((t - prior) / np.array([0.5, 1.0, 0.7, 0.7])).sum())
            )
            if value < best[0]:
                best[:] = [value, t.copy()]
            return value

        n_high = sum(r["fidelity"] == 1.0 for r in self.rows)
        status = "regularized_prior_conditioned"
        if signature == self.restored_hash:
            status = self.fit_diagnostics.get("status", "restored")
        elif (
            self.reference_frozen
            and len(self.rows) >= CONTRACT["fit_min_total"]
            and n_high >= CONTRACT["fit_min_high"]
        ):
            try:
                result = minimize(
                    objective,
                    prior,
                    method="L-BFGS-B",
                    bounds=[(-2.0, 2.0), (-3.0, 3.0), (-2.3, 1.6), (-2.3, 1.6)],
                    options={"maxiter": 25, "maxfun": 180},
                )
                if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
                    raise ValueError("Nonfinite MF fit; no selector fallback")
                self.theta = result.x
                status = "bounded_map"
            except FitBudget:
                if not math.isfinite(best[0]):
                    raise ValueError("MF fit timed out before a finite objective")
                self.theta = best[1]
                status = "map_walltime_limit"
        else:
            self.theta = prior
        self.factor = factor(self.theta) if self.rows else None
        self.alpha = (
            cho_solve(self.factor, self.values - self.mean(self.rows))
            if self.rows
            else np.empty(0)
        )
        self.training_hash = signature
        self.model_version = "mf_gp_" + signature[:16]
        self.fit_diagnostics = dict(
            n=len(self.rows),
            n_high=n_high,
            status=status,
            theta=self.theta.tolist(),
            reference=[self.center, self.scale],
            uncertainty="conditional_on_regularized_hyperparameters",
        )
        return True

    def posterior(self, rows, pending=()):
        combined = list(rows) + list(pending)
        mu = self.mean(combined)
        cov = self.covariance(combined, combined)
        if self.factor is not None:
            cross = self.covariance(combined, self.rows)
            mu += cross @ self.alpha
            cov -= cross @ cho_solve(self.factor, cross.T)
        n = len(rows)
        if pending:
            # Kriging believer: account for information already in flight,
            # without admitting fantasy scores as observations or fitting them.
            cross = cov[:n, n:]
            f = cho_factor(
                cov[n:, n:] + np.diag(self.noise(pending) + 1e-8), lower=True
            )
            cov = cov[:n, :n] - cross @ cho_solve(f, cross.T)
        else:
            cov = cov[:n, :n]
        cov = (cov + cov.T) / 2
        if not np.isfinite(cov).all() or (np.diag(cov) < -1e-7).any():
            raise ValueError("Invalid MF posterior covariance")
        cov[np.diag_indices(n)] = np.maximum(0.0, np.diag(cov))
        return self.center + self.scale * mu[:n], self.scale**2 * cov

    def state_dict(self):
        return dict(
            reference_frozen=self.reference_frozen,
            center=self.center,
            scale=self.scale,
            contract=CONTRACT,
            theta=self.theta.tolist(),
            training_hash=self.training_hash,
            diagnostics=self.fit_diagnostics,
        )

    def load_state_dict(self, state):
        if state["contract"] != CONTRACT:
            raise ValueError("MF model contract changed on resume")
        self.reference_frozen = state["reference_frozen"]
        self.center, self.scale = state["center"], state["scale"]
        self.training_hash = None
        self.restored_hash = state["training_hash"]
        self.theta = np.array(state["theta"], dtype=float)
        self.fit_diagnostics = state["diagnostics"]


def acquisition(gp, targets, actions, incumbent, pending=(), terminal=False):
    """One-step contextual target-H KG; retained H artifacts; EI at deadline.

    The target set is frozen across all alternatives within one decision. It
    contains saved cheap programs and one prospective fresh child per context.
    This is not an exact finite-horizon optimizer of future evolutionary trees.
    """
    started = time.monotonic()
    rows = list(targets) + [a["row"] for a in actions]
    mean, cov = gp.posterior(rows, pending)
    n = len(targets)
    baseline = max(
        [float(x) for x in mean[:n]] + ([incumbent] if incumbent is not None else [])
    )
    noise = gp.noise([a["row"] for a in actions]) * gp.scale**2
    values = []
    for j, action in enumerate(actions):
        if time.monotonic() - started > 30.0:
            raise ValueError("MF acquisition exceeded its compute budget")
        i = n + j
        sd = math.sqrt(max(0.0, cov[i, i] + noise[j]))
        if terminal:
            if action["row"]["fidelity"] != 1.0:
                raise ValueError("Terminal action must be full fidelity")
            gain = mean[i] - (incumbent if incumbent is not None else gp.center)
            value = (
                max(0.0, gain)
                if sd == 0
                else gain * ndtr(gain / sd)
                + sd * math.exp(-0.5 * (gain / sd) ** 2) / math.sqrt(2 * math.pi)
            )
        else:
            intercepts = list(mean[:n])
            slopes = list(cov[:n, i] / sd) if sd else [0.0] * n
            if incumbent is not None:
                intercepts.append(incumbent)
                slopes.append(0.0)
            if action["row"]["fidelity"] == 1.0:
                intercepts.append(mean[i])
                slopes.append(sd)
            value, _ = gaussian_envelope(intercepts, slopes, baseline)
            if value < -1e-7:
                raise ValueError("Negative MF knowledge gradient")
        values.append(max(0.0, float(value)) / max(1.0, action["cost"]))
    return values, dict(
        mean=mean.tolist(),
        baseline=baseline,
        target_count=n,
        terminal=terminal,
        pending_count=len(pending),
    )
