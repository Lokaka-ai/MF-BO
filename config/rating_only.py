"""Explicit user-requested rating-only control; no fitted model or acquisition."""
from dataclasses import asdict
from omegaconf import OmegaConf
from .study7 import Study7Config

RATING_ONLY_PRESETS=('acq7_aq09_rating_only_greedy',)
RATING_ONLY_SELECTORS={'study7_rating_only_greedy'}
RATING_ONLY_SCHEMA='study7_rating_only_v1'

def rating_only_config(cfg):
    actual=OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(Study7Config),cfg.study7))
    expected=Study7Config(method_id='AQ09',schema=RATING_ONLY_SCHEMA)
    expected.rating.enabled=True
    if asdict(actual)!=asdict(expected):raise ValueError('AQ09 rating-only review contract changed')
    if (cfg.parent_selector.type!='study7_rating_only_greedy'
            or cfg.bo_parent_selector.model!='study7_no_surrogate'
            or cfg.parent_selector.include_root_draft
            or cfg.parent_selector.decision_scope!='all_normal_steps'
            or cfg.acq7.enabled or cfg.branch7.enabled
            or cfg.bo_parent_selector.agent_rating.enabled):
        raise ValueError('AQ09 requires the managed rating-only path without a surrogate')
    acq=cfg.bo_parent_selector.acquisition
    if (acq.type!='rating_greedy' or float(acq.beta)!=0 or acq.beta_schedule!='fixed'
            or acq.use_cost or acq.use_validity_probability):
        raise ValueError('AQ09 forbids GP/UCB/TS/KG, cost and validity weighting')
    if int(cfg.agent.seed)!=42:raise ValueError('AQ09 preserves seed 42')
    return actual
