"""Binary calibration using T-only held-out/OOF logits (never outer V labels)."""

import numpy as np


def binary_probabilities(logits, temperature=1.0, clip_eps=1e-6):
    """Convert 1-D positive-class log odds, with the same recipe for V and test.

    Two-class model outputs must explicitly be converted to logit[:, 1] -
    logit[:, 0] by the caller. Multi-class matrices are not binary log odds.
    """
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 1 or not np.isfinite(logits).all():
        raise ValueError("Expected finite one-dimensional binary log odds")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not np.isfinite(clip_eps) or not np.finfo(np.float64).eps <= clip_eps < 0.5:
        raise ValueError("clip_eps must be representable in float64: eps <= clip_eps < 0.5")
    with np.errstate(over="ignore"):
        x = logits / float(temperature)
        exp_neg_abs = np.exp(-np.abs(x))
        probabilities = np.where(x >= 0, 1 / (1 + exp_neg_abs), exp_neg_abs / (1 + exp_neg_abs))
    return np.clip(probabilities, clip_eps, 1 - clip_eps)


def fit_binary_temperature(logits, labels, temperatures, clip_eps=1e-6):
    """Return only grid[argmin(losses)], with first-grid-entry tie breaking.

    Calibration labels/logits must come from an internal holdout or OOF of T.
    This function does not return a validation score for search feedback.
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    grid = np.asarray(temperatures, dtype=np.float64)
    if labels.ndim != 1 or labels.shape != logits.shape or not labels.size:
        raise ValueError("Nonempty calibration labels and logits must have identical 1-D shape")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Binary calibration labels must be 0 or 1")
    if grid.ndim != 1 or not grid.size or not np.isfinite(grid).all() or (grid <= 0).any():
        raise ValueError("temperatures must be a nonempty finite positive 1-D grid")
    losses = []
    for temperature in grid:
        p = binary_probabilities(logits, temperature, clip_eps)
        losses.append(-np.mean(labels * np.log(p) + (1 - labels) * np.log1p(-p)))
    return float(grid[np.argmin(losses)])
