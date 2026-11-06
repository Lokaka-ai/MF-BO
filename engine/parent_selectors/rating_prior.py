"""Opt-in, positive piBO rating prior. This module never calls an LLM.

The one generic node rating is a preference, not a response/validity label.
Unrated/malformed ratings are neutral. Iteration is a reserved ordinary BO
selection counter, NOT the number of returned short/long training labels.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from collections.abc import Mapping

import numpy as np


@dataclass(frozen=True)
class RatingPriorConfig:
    enabled: bool = False
    beta_prior: float = 25.0

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("agent_rating.enabled must be a boolean")
        if (isinstance(self.beta_prior, bool) or not isinstance(self.beta_prior, Real)
                or not math.isfinite(self.beta_prior) or self.beta_prior < 0):
            raise ValueError("agent_rating.beta_prior must be finite and nonnegative")

    @classmethod
    def from_config(cls, cfg):
        bo = cfg.get("bo_parent_selector", {}) if isinstance(cfg, Mapping) else getattr(cfg, "bo_parent_selector", None)
        raw = bo.get("agent_rating", {}) if isinstance(bo, Mapping) else getattr(bo, "agent_rating", {})
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raw = vars(raw)
        return cls(**dict(raw))

    def contract(self):
        if not self.enabled:
            return {"enabled": False}
        return {"enabled": True, "beta_prior": float(self.beta_prior),
                "weights": [0.5, 1.0, 2.0], "decay": "beta_over_bo_selection",
                "rating_contract": "generic_completed_node_potential_v1"}


def potential_level(value):
    """Do not treat bools, strings, floats or missing fields as a low rating."""
    if isinstance(value, Integral) and not isinstance(value, bool) and value in (0, 1, 2):
        return int(value)
    return None


def prior_terms(levels, *, beta_prior, iteration):
    """log(weight) and beta/n * log(weight), with weights .5, 1, 2.

    No counter mutation: evaluating candidates or retrying the same frozen
    selection must not advance the annealing schedule.
    """
    RatingPriorConfig(enabled=True, beta_prior=beta_prior)
    if isinstance(iteration, bool) or not isinstance(iteration, Integral) or iteration < 1:
        raise ValueError("piBO iteration must be a positive ordinary-selection integer")
    log_weights = np.asarray([
        0.0 if potential_level(g) is None else (int(g)-1)*math.log(2.0)
        for g in levels
    ])
    return log_weights, float(beta_prior)/int(iteration)*log_weights
