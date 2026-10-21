"""Full GP4 posterior with the fixed Acq7 branch covariance families."""
from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular

from .branch7_gp import BranchGP, branch_loading
from .study7_gp import State4GP, Study7Error, digest, matern32


def performance_values(rows):
    values = np.asarray([row["performance_v"] for row in rows], dtype=np.float64)
    if not np.isfinite(values).all():
        raise Study7Error("Acq7 has nonfinite frozen performance coordinates")
    return values


def branch_covariance(left, right, config):
    hl, bl = branch_loading(left)
    hr, br = branch_loading(right)
    h = hl[:, None] * hr[None, :] * np.asarray(
        [[a == b for b in br] for a in bl], dtype=np.float64).reshape(len(left), len(right))
    vl, vr = performance_values(left), performance_values(right)
    tau, slope = config.branch_prior_std, config.slope_prior_std
    if not np.isfinite([tau, slope, config.branch_lengthscale]).all() or min(tau, slope) < 0:
        raise Study7Error("Invalid fixed Acq7 branch parameters")
    if config.branch_kind == "constant":
        result = tau ** 2 * h
    elif config.branch_kind == "level_matern32":
        result = tau ** 2 * h * matern32(vl[:, None], vr[:, None], config.branch_lengthscale)
    elif config.branch_kind == "affine":
        result = h * (tau ** 2 + slope ** 2 * np.outer(vl, vr))
    else:
        raise Study7Error("Unregistered Acq7 branch kind")
    if not np.isfinite(result).all():
        raise Study7Error("Nonfinite Acq7 branch covariance")
    return result


def psd_covariance(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]
            or not len(matrix) or not np.isfinite(matrix).all()):
        raise Study7Error("Invalid full latent covariance")
    matrix = (matrix + matrix.T) * .5
    values, vectors = np.linalg.eigh(matrix)
    magnitude = max(1., float(np.linalg.norm(matrix, ord=np.inf)))
    if values[0] < -1e-10 * magnitude:
        raise Study7Error("blocked_numerical: materially indefinite latent covariance")
    clipped = np.maximum(values, 0.)
    if np.any(values < 0):
        matrix = (vectors * clipped) @ vectors.T
        matrix = (matrix + matrix.T) * .5
    # The symmetric root preserves the exact singular joint law; no diagonal noise.
    root = (vectors * np.sqrt(clipped)) @ vectors.T
    return matrix, root, {"min_eigenvalue": float(values[0]),
        "negative_eigenvalues_clipped": int(np.sum(values < 0)),
        "eigenvalue_clip_sum": float(np.sum(clipped - values)),
        "tolerance_scale": magnitude, "covariance_hash": digest(matrix.tolist()),
        "decomposition_policy": "symmetric_eigh_roundoff_clip_no_added_noise_v1"}


class AcqGP(BranchGP):
    def covariance(self, left, right, hp=None, *, derivatives=False):
        branch = branch_covariance(left, right, self.config)
        result = State4GP.covariance(self, left, right, hp, derivatives=derivatives)
        if derivatives:
            return result[0] + branch, result[1], result[2]
        return result[0] + branch, {**result[1], "branch": branch}

    def predict_joint(self, rows, scale=None):
        if not rows:
            raise Study7Error("Joint prediction requires a nonempty legal pool")
        scale = float(rows[0]["s0"] if scale is None else scale)
        if not np.isfinite(scale) or scale <= 0:
            raise Study7Error("Invalid frozen utility scale")
        prior, components = self.covariance(rows, rows)
        mean = self.mean(rows)
        covariance = prior.copy()
        cross = None
        if self.factor is not None:
            cross, _ = self.covariance(rows, self.rows)
            mean += cross @ self.alpha
            solved = solve_triangular(self.factor, cross.T, lower=True, check_finite=True)
            covariance -= solved.T @ solved
        quality = np.asarray([row["parent_utility"] for row in rows], dtype=float) + scale * mean
        if not np.isfinite(quality).all():
            raise Study7Error("Nonfinite Acq7 posterior quality")
        recomputed = False
        try:
            covariance, root, diagnostics = psd_covariance(scale ** 2 * covariance)
        except Study7Error:
            if cross is None:
                raise
            # Same training inputs, HP and jitter; retry a fresh stable solve.
            train, _ = self.covariance(self.rows, self.rows)
            factor = cholesky(train + (self.hyperparameters[2] ** 2 + self.jitter)
                              * np.eye(len(train)), lower=True, check_finite=True)
            covariance = prior - cross @ cho_solve((factor, True), cross.T, check_finite=True)
            covariance, root, diagnostics = psd_covariance(scale ** 2 * covariance)
            recomputed = True
        diagnostics["stable_solve_recomputed"] = recomputed
        noise = float((scale * self.hyperparameters[2]) ** 2)
        details = [{"prior_mean_residual": float(self.mean([row])[0]),
            "posterior_mean_residual": float(mean[i]), "noise_std": float(self.hyperparameters[2]),
            **{f"diag_{name}": float(value[i, i]) for name, value in components.items()},
            "diag_total": float(prior[i, i]),
            "posterior_variance_residual": float(covariance[i, i] / scale ** 2),
            "prior_only": not self.rows, "jitter": self.jitter} for i, row in enumerate(rows)]
        return {"mean": quality, "latent_covariance": covariance,
            "latent_square_root": root, "noise_variance": noise,
            "latent_sd": np.sqrt(np.maximum(np.diag(covariance), 0.)),
            "predictive_sd": np.sqrt(np.maximum(np.diag(covariance), 0.) + noise),
            "row_keys": [row.get("option_key", [row["parent_id"], row["operator"]]) for row in rows],
            "details": details, "diagnostics": diagnostics,
            "posterior_hash": digest({"mean": quality.tolist(), "covariance": covariance.tolist(), "noise": noise})}

    def branch_function_posterior(self, branch, probe_v, scale=1.):
        probes = [{"branch_effect_load": 1, "root_lineage_id": branch,
                   "performance_v": float(v)} for v in probe_v]
        prior = branch_covariance(probes, probes, self.config)
        mean, covariance = np.zeros(len(probes)), prior.copy()
        if self.factor is not None:
            cross = branch_covariance(probes, self.rows, self.config)
            mean = cross @ self.alpha
            covariance -= cross @ cho_solve((self.factor, True), cross.T)
        covariance, _, numerical = psd_covariance(covariance)
        return {"branch": branch, "probe_v": list(probe_v), "mean": mean.tolist(),
            "covariance": covariance.tolist(), "mean_utility": (scale * mean).tolist(),
            "covariance_utility": (scale ** 2 * covariance).tolist(), "numerical": numerical}

    def affine_coefficients(self, names):
        names = list(dict.fromkeys(names))
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise Study7Error("Invalid coefficient lineage names")
        loads, branches = branch_loading(self.rows)
        values = performance_values(self.rows)
        design = np.zeros((len(self.rows), 2 * len(names)))
        for i, (h, branch, value) in enumerate(zip(loads, branches, values)):
            if branch in names:
                j = 2 * names.index(branch)
                design[i, j:j+2] = h, h * value
        diagonal = np.tile([self.config.branch_prior_std ** 2, self.config.slope_prior_std ** 2], len(names))
        mean, covariance = np.zeros(len(diagonal)), np.diag(diagonal)
        if self.factor is not None:
            cross = diagonal[:, None] * design.T
            mean = cross @ self.alpha
            covariance -= cross @ cho_solve((self.factor, True), cross.T)
        covariance, _, numerical = psd_covariance(covariance)
        return {"lineages": names, "coefficient_order": "interleaved_intercept_slope",
                "mean": mean.tolist(), "covariance": covariance.tolist(), "numerical": numerical}

    def without_branch(self):
        other = copy.copy(self)
        other.config = replace(self.config, branch_prior_std=0., slope_prior_std=0.)
        if self.rows:
            kernel, _ = other.covariance(self.rows, self.rows)
            other.factor, other.jitter = other._factor(
                kernel + self.hyperparameters[2] ** 2 * np.eye(len(self.rows)), [])
            other.alpha = cho_solve((other.factor, True), self.targets - other.mean(self.rows))
        return other
