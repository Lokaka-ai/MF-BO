"""One joint latent TS draw and exact Gaussian-envelope KG/RO-KG."""
from __future__ import annotations

import math
import time

import numpy as np
from scipy.special import ndtr

from .acq7_gp import psd_covariance
from .study7_gp import Study7Error

SQRT_2PI = math.sqrt(2. * math.pi)


def _mass(lower, upper):
    if lower >= 0:
        return float(ndtr(-lower) - ndtr(-upper))
    return float(ndtr(upper) - ndtr(lower))


def _density_difference(lower, upper):
    if not math.isfinite(lower):
        return 0. if not math.isfinite(upper) else -math.exp(-.5 * upper * upper) / SQRT_2PI
    if not math.isfinite(upper):
        return math.exp(-.5 * lower * lower) / SQRT_2PI
    a, b = -.5 * lower * lower, -.5 * upper * upper
    if a >= b:
        return math.exp(a) * (-math.expm1(b - a)) / SQRT_2PI
    return math.exp(b) * math.expm1(a - b) / SQRT_2PI


def gaussian_envelope(intercepts, slopes, baseline=0., *, include_segments=False):
    """Return E max(a_i+b_i Z)-baseline using the exact upper envelope."""
    a, b = np.asarray(intercepts, dtype=np.float64), np.asarray(slopes, dtype=np.float64)
    if (a.ndim != 1 or a.shape != b.shape or not len(a)
            or not np.isfinite(a).all() or not np.isfinite(b).all() or not math.isfinite(baseline)):
        raise Study7Error("Invalid Gaussian-envelope lines")
    a = a - baseline
    if not np.isfinite(a).all():
        raise Study7Error("Gaussian-envelope centering overflow")
    order = sorted(range(len(a)), key=lambda i: (float(b[i]), -float(a[i]), i))
    unique = []
    for index in order:
        if not unique or b[index] != b[unique[-1]]:
            unique.append(index)
    hull, starts = [], []
    for index in unique:
        start = -math.inf
        while hull:
            previous = hull[-1]
            with np.errstate(over="ignore", divide="ignore", invalid="raise"):
                start = float((a[previous] - a[index]) / (b[index] - b[previous]))
            if math.isnan(start):
                raise Study7Error("Undefined Gaussian-envelope intersection")
            if start > starts[-1]:
                break
            hull.pop(); starts.pop()
        if not hull:
            start = -math.inf
        if start == math.inf:
            continue
        hull.append(index); starts.append(start)
    terms, segments = [], []
    for k, index in enumerate(hull):
        lower, upper = starts[k], starts[k+1] if k+1 < len(starts) else math.inf
        mass = _mass(lower, upper)
        term = float(a[index]) * mass + float(b[index]) * _density_difference(lower, upper)
        if not math.isfinite(term) or mass < 0:
            raise Study7Error("Nonfinite/inaccurate Gaussian-envelope integral")
        terms.append(term)
        if include_segments:
            segments.append({"line_index": index,
                "lower": None if lower == -math.inf else lower,
                "upper": None if upper == math.inf else upper,
                "centered_intercept": float(a[index]), "slope": float(b[index]),
                "gaussian_mass": mass, "integral": term})
    value = math.fsum(terms)
    return value, {"segment_count": len(hull), "baseline": float(baseline),
        "numerical_status": "exact_piecewise_normal_integral_float64",
        "segments": segments if include_segments else None,
        "summand_absolute_sum": math.fsum(abs(x) for x in terms)}


def _nonnegative(value, scale):
    tolerance = 1e-10 * max(1., scale)
    if not math.isfinite(value) or value < -tolerance:
        raise Study7Error("blocked_numerical: invalid negative KG value")
    return max(value, 0.), max(-value, 0.)


def knowledge_gradient(mean, covariance, noise_variance, *, incumbent=None, max_seconds=30.):
    """All finite-pool scores; incumbent enables the registered RO-KG value."""
    started = time.perf_counter()
    mu = np.asarray(mean, dtype=np.float64)
    sigma = np.asarray(covariance, dtype=np.float64)
    if (mu.ndim != 1 or not len(mu) or sigma.shape != (len(mu), len(mu))
            or not np.isfinite(mu).all() or not np.isfinite(sigma).all()
            or not math.isfinite(noise_variance) or noise_variance < 0
            or (incumbent is not None and not math.isfinite(incumbent))):
        raise Study7Error("Invalid KG posterior/incumbent")
    if not np.allclose(sigma, sigma.T, rtol=1e-12, atol=1e-12) or np.any(np.diag(sigma) < 0):
        raise Study7Error("Invalid KG latent covariance")
    baseline = max(float(np.max(mu)), float(incumbent)) if incumbent is not None else float(np.max(mu))
    scores, diagnostics = [], []
    for j in range(len(mu)):
        if time.perf_counter() - started > max_seconds:
            raise Study7Error("blocked_numerical: acquisition wall-time limit")
        d = math.sqrt(float(sigma[j, j]) + noise_variance)
        slopes = sigma[:, j] / d if d else np.zeros(len(mu))
        if not d and np.any(sigma[:, j] != 0):
            raise Study7Error("Zero predictive variance with nonzero cross covariance")
        intercepts = mu
        if incumbent is not None:
            intercepts = np.r_[mu, incumbent, mu[j]]
            slopes = np.r_[slopes, 0., d]
        value, info = gaussian_envelope(intercepts, slopes, baseline)
        value, clipped = _nonnegative(value, max(float(np.max(np.abs(slopes))),
                                               float(np.max(np.abs(intercepts - baseline)))))
        scores.append(value)
        diagnostics.append({**info, "candidate_index": j, "predictive_sd": d,
            "roundoff_negative_clipped": clipped, "retained_outcome": incumbent is not None})
    elapsed = time.perf_counter() - started
    if elapsed > max_seconds:
        raise Study7Error("blocked_numerical: acquisition wall-time limit")
    return np.asarray(scores), {"candidates": diagnostics, "wall_seconds": elapsed,
                                "baseline": baseline, "incumbent": incumbent}


def selected_envelope(mean, covariance, noise_variance, index, incumbent=None):
    mu, sigma = np.asarray(mean), np.asarray(covariance)
    d = math.sqrt(float(sigma[index, index]) + noise_variance)
    slopes = sigma[:, index] / d if d else np.zeros(len(mu))
    baseline = float(np.max(mu))
    if incumbent is not None:
        mu, slopes = np.r_[mu, incumbent, mu[index]], np.r_[slopes, 0., d]
        baseline = max(baseline, incumbent)
    return gaussian_envelope(mu, slopes, baseline, include_segments=True)[1]


def joint_thompson(mean, covariance, rng, square_root=None):
    mu = np.asarray(mean, dtype=np.float64)
    if mu.ndim != 1 or not len(mu) or not np.isfinite(mu).all():
        raise Study7Error("Invalid TS mean")
    if len(mu) == 1:
        return mu.copy(), {"draw": False, "reason": "single_candidate_no_draw", "standard_normal": None}
    if square_root is None:
        _, square_root, _ = psd_covariance(covariance)
    if square_root.shape != (len(mu), len(mu)) or not np.isfinite(square_root).all():
        raise Study7Error("Invalid joint latent square root")
    z = rng.standard_normal(len(mu))
    sample = mu + square_root @ z
    if not np.isfinite(sample).all():
        raise Study7Error("Nonfinite joint TS sample")
    return sample, {"draw": True, "reason": "one_joint_latent_sample", "standard_normal": z.tolist()}
