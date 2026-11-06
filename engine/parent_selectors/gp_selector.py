"""GP quality-UCB selector, independent of legacy agent imports."""
from __future__ import annotations

import uuid
from .base import CandidateScore, SelectionDecision, SelectorResult, utc_timestamp


def _decision(*, selector_name, option, selected_score, model_version=None,
              fallback_used=False, fallback_reason=None):
    return SelectionDecision(selector_name, option.parent_node_id, option.operator,
        float(selected_score), model_version, fallback_used, uuid.uuid4().hex,
        utc_timestamp(), fallback_reason=fallback_reason)


def _simple_score(option, score):
    return CandidateScore(option.parent_node_id, option.operator, option.reference_node_ids,
        option.parent_utility, None, None, None, None, None, float(score))


class GaussianProposalSelector:
    """Child-quality UCB using a single shared GP and latent uncertainty."""
    name = "gp_child_ucb"

    def __init__(self, *, surrogate, records_provider, beta=1.5):
        self.surrogate = surrogate
        self.records_provider = records_provider
        self.beta = float(beta)
        self.decision_index = None

    def select(self, agent, options):
        if not options:
            raise ValueError("GP selector received no options")
        try:
            records = self.records_provider()
            if self.surrogate.fitted_transition_count != len(records):
                self.surrogate.fit(records)
            means, latent, predictive = self.surrogate.predict_quality(options)
            acquisition = means + self.beta * latent
            scores = tuple(CandidateScore(
                parent_node_id=o.parent_node_id, operator=o.operator,
                reference_node_ids=o.reference_node_ids, parent_utility=o.parent_utility,
                mean_gain=float(means[i] - o.parent_utility) if o.parent_utility is not None else None,
                std_gain=float(latent[i]), predicted_child_ucb=float(acquisition[i]),
                predicted_cost=None, p_valid=None, acquisition_score=float(acquisition[i]),
                posterior_mean_quality=float(means[i]), posterior_std_latent=float(latent[i]),
                predictive_std=float(predictive[i]),
            ) for i, o in enumerate(options))
            best = max(range(len(options)), key=lambda i: acquisition[i])
            return SelectorResult(
                _decision(selector_name=self.name, option=options[best], selected_score=float(acquisition[best]),
                          model_version=self.surrogate.model_version), options[best], scores, tuple(options))
        except Exception as exc:
            # A bounded, logged fallback must remain inside the already frozen
            # legal pool. Never let MCGS bypass a reserved root cap or pending slot.
            values = [self.surrogate.draft_anchor if o.operator == "draft"
                      else float(o.parent_utility) for o in options]
            best = max(range(len(options)), key=lambda i: values[i])
            return SelectorResult(
                _decision(selector_name=self.name, option=options[best], selected_score=values[best],
                          fallback_used=True, fallback_reason=f"gp_error:{type(exc).__name__}:{exc}"),
                options[best], tuple(_simple_score(o, v) for o, v in zip(options, values)), tuple(options))
