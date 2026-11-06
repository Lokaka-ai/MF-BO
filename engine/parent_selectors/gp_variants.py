"""Two native online BO variants; the native GP6 default remains untouched.

A: shrink the GP6 residual mean and latent SD by .5, retain observation noise.
B: original GP6 posterior, stable log EI relative to current global incumbent.
No SQLite, offline labels, LLM calls, extra model, candidate-generation change,
or continuation/budget change is introduced.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import numpy as np
from scipy.special import ndtr, log_ndtr
from .gp_surrogate import GaussianProposalSurrogate
from .gp_features import GP6_NAMES, GP6_OPERATORS
from .gp_selector import _decision, _simple_score
from .base import CandidateScore, SelectorResult


@dataclass(frozen=True)
class GPVariantConfig:
    shrink: float = 0.5
    acquisition: str = "ucb"
    beta: float = 1.5

    def __post_init__(self):
        if not math.isfinite(self.shrink) or not 0 <= self.shrink <= 1:
            raise ValueError("gp.variant.shrink must be finite in [0,1]")
        if self.acquisition not in ("ucb", "logei"):
            raise ValueError("gp.variant.acquisition must be ucb/logei")
        if not math.isfinite(self.beta) or self.beta < 0:
            raise ValueError("gp.variant.beta must be finite and nonnegative")


def stable_log_ei(mean, sigma, incumbent):
    """Analytic Gaussian log EI. Extreme negative tails never exponentiate EI.

    Negative-tail expansion matches the bounded research implementation.
    A deterministic non-improving candidate returns -inf, not a fake improvement.
    """
    mean, sigma = np.broadcast_arrays(np.asarray(mean, float), np.asarray(sigma, float))
    if not np.isfinite(mean).all() or not np.isfinite(sigma).all() or (sigma < 0).any():
        raise ValueError("Invalid log-EI inputs")
    if not math.isfinite(incumbent):
        raise ValueError("Invalid incumbent")
    gain = mean - incumbent
    answer = np.full(mean.shape, -np.inf)
    positive_sigma = sigma > 0
    z = gain[positive_sigma] / sigma[positive_sigma]
    lp = -.5*z*z - .5*np.log(2*np.pi)
    term = np.empty(z.shape)
    pos = z >= 0
    term[pos] = np.logaddexp(lp[pos], np.log(np.maximum(z[pos], np.finfo(float).tiny)) + log_ndtr(z[pos]))
    mid = (z < 0) & (z >= -10)
    term[mid] = lp[mid] + np.log1p(-np.exp(np.log(-z[mid])+log_ndtr(z[mid])-lp[mid]))
    tail = z < -10
    t = 1. / z[tail]**2
    term[tail] = lp[tail] + np.log(t) + np.log1p(-3*t+15*t*t-105*t**3+945*t**4)
    answer[positive_sigma] = np.log(sigma[positive_sigma]) + term
    deterministic_gain = (~positive_sigma) & (gain > 0)
    answer[deterministic_gain] = np.log(gain[deterministic_gain])
    return answer


class GPVariantSurrogate(GaussianProposalSurrogate):
    def __init__(self, *, variant_config, config=None, selection_logger=None):
        super().__init__(config=config, selection_logger=selection_logger)
        self.variant_config = GPVariantConfig(**dict(variant_config))

    def state_dict(self):
        return {**super().state_dict(), "variant_config": asdict(self.variant_config)}

    def load_state_dict(self, state):
        if state.get("variant_config") != asdict(self.variant_config):
            raise ValueError("Cannot resume under a different GP variant setting")
        super().load_state_dict(state)

    def fit(self, records):
        # Preserve the upstream training algorithm; emit exactly one enriched fit.
        logger = self.selection_logger
        self.selection_logger = None
        try:
            super().fit(records)
        finally:
            self.selection_logger = logger
        self.fit_diagnostics.update(variant_config=asdict(self.variant_config),
            acquisition_sigma="latent", observation_noise_shrunk=False)
        if logger:
            logger.append_model_fit({"model_version": self.model_version,
                "training_transition_ids": self.training_transition_ids,
                "feature_schema_version": self.feature_schema_version,
                "feature_names": list(GP6_NAMES), "operator_categories": list(GP6_OPERATORS),
                "fit_wall_time_sec": self.last_fit_wall_time_sec,
                "fit_diagnostics": self.fit_diagnostics})

    def predict_quality(self, options):
        mean, latent, predictive = super().predict_quality(options)
        strength = self.variant_config.shrink
        if strength == 1.:
            return mean, latent, predictive
        center = np.asarray([self.baseline(dict(o.state_features), o.operator) for o in options])
        observation_noise2 = np.maximum(predictive**2-latent**2, 0.)
        mean = center + strength*(mean-center)
        latent = strength*latent
        return mean, latent, np.sqrt(latent**2+observation_noise2)


class GPVariantSelector:
    def __init__(self, *, surrogate, records_provider):
        self.surrogate, self.records_provider = surrogate, records_provider
        c = surrogate.variant_config
        self.name = "halfshrink_ucb" if c.acquisition == "ucb" else "gp6_global_logei"
        self.decision_index = None

    def select(self, agent, options):
        if not options:
            raise ValueError("No legal options provided")
        try:
            records = self.records_provider()
            if self.surrogate.fitted_transition_count != len(records):
                self.surrogate.fit(records)
            mu, latent, predictive = self.surrogate.predict_quality(options)
            c = self.surrogate.variant_config
            ucb = mu+c.beta*latent
            if c.acquisition == "ucb":
                values = ucb
            else:
                incs = [o.state_features.get("incumbent_at_decision") for o in options]
                incs = [float(v) for v in incs if v is not None and math.isfinite(float(v))]
                if not incs or len(incs) != len(options) or not np.allclose(incs, incs[0], rtol=0., atol=1e-12):
                    raise ValueError("Missing/inconsistent decision-time incumbent")
                incumbent = incs[0]
                # Dimensionless scores preserve EI ranking under changes of units.
                scale = self.surrogate.utility_scale
                values = stable_log_ei((mu-incumbent)/scale, latent/scale, 0.)
                if np.isnan(values).any() or np.isposinf(values).any():
                    raise ValueError("Non-finite positive log-EI")
                # Strict JSON-safe representation for exact mathematical zero EI.
                # The original legal-pool order resolves exact ties, as native GP6.
                values = np.where(np.isneginf(values), -np.finfo(float).max, values)
            if not np.isfinite(values).all():
                raise ValueError("Non-finite acquisition")
            scores = tuple(CandidateScore(
                parent_node_id=o.parent_node_id, operator=o.operator,
                reference_node_ids=o.reference_node_ids, parent_utility=o.parent_utility,
                mean_gain=float(mu[i]-o.parent_utility) if o.parent_utility is not None else None,
                std_gain=float(latent[i]), predicted_child_ucb=float(ucb[i]),
                predicted_cost=None, p_valid=None, acquisition_score=float(values[i]),
                posterior_mean_quality=float(mu[i]), posterior_std_latent=float(latent[i]),
                predictive_std=float(predictive[i])) for i,o in enumerate(options))
            best = int(np.argmax(values))
            return SelectorResult(_decision(selector_name=self.name, option=options[best],
                selected_score=values[best], model_version=self.surrogate.model_version),
                options[best], scores, tuple(options))
        except Exception as exc:
            # Identical bounded parent-quality fallback, no MCGS/legal-pool bypass.
            values = [self.surrogate.draft_anchor if o.operator == "draft" else float(o.parent_utility) for o in options]
            best = max(range(len(options)), key=lambda i: values[i])
            return SelectorResult(_decision(selector_name=self.name, option=options[best],
                selected_score=values[best], fallback_used=True,
                fallback_reason=f"gp_variant_error:{type(exc).__name__}:{exc}"), options[best],
                tuple(_simple_score(o,v) for o,v in zip(options,values)), tuple(options))
