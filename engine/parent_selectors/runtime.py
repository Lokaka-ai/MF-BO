"""Controller-facing Phase 1 selection and transition-recording runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import copy
import json
import math
import os
from pathlib import Path
import random
import socket
import threading
import time
import uuid

import numpy as np

from engine import evaluation, node_selection
from config import validate_root_draft_config
from config.study7 import STUDY7_SELECTORS, STATE4_SCHEMA, GREEDY_SCHEMA, study7_config
from config.branch7 import BRANCH7_SELECTORS, BRANCH7_SCHEMAS, branch7_config
from config.acq7 import ACQ7_SELECTORS, ACQ7_SCHEMA, acq7_config
from config.rating_only import RATING_ONLY_SELECTORS, RATING_ONLY_SCHEMA, rating_only_config
from engine.conditions import should_trigger_branch_fusion
from engine.operator_preview import (
    commit_preview_random,
    eligible_normal_parents,
    is_normal_operator,
    preview_next_operator,
    requires_topk_extension,
)
from engine.parent_selectors.base import (
    CandidateScore,
    ExpansionOption,
    OperatorPreview,
    SelectionDecision,
    SelectorResult,
    utc_timestamp,
)
from engine.parent_selectors.features import (
    BNVS_FEATURE_SCHEMA_VERSION,
    BNVS_ROOT_FEATURE_SCHEMA_VERSION,
    FEATURE_SCHEMA_VERSION,
    REWARD_ONLY_FEATURE_SCHEMA_VERSION,
    continuous_improvement,
    extract_selection_features,
    extract_root_draft_features,
    utility_from_metric,
)
from engine.parent_selectors.selectors import (
    BayesianNodeValueSelector,
    BayesianUCBParentSelector,
    ParentMetricGreedySelector,
    ProgressiveMCGSSelector,
    RandomParentSelector,
    RewardOnlyThompsonSelector,
    SelectionOutsideOptions,
)
from engine.parent_selectors.surrogate import (
    BayesianNodeValueSurrogate,
    BayesianTransitionSurrogate,
    RewardOnlyBayesianLinearSurrogate,
)
from engine.parent_selectors.gp_selector import GaussianProposalSelector
from engine.parent_selectors.gp_variants import GPVariantSurrogate, GPVariantSelector
from engine.parent_selectors.agent_selector import AgentParentSelector, AgentSelectionError, AGENT_SCHEMA
from engine.parent_selectors.gp_surrogate import GaussianProposalSurrogate, GP6Config
from engine.parent_selectors.gp_features import GP6_SCHEMA, GP6_NAMES, progress_features
from engine.parent_selectors.gp_graph_features import GP7_SCHEMA, GRAPH_SELECTOR_TYPES, DecisionGraph
from engine.parent_selectors.gp_graph_surrogate import GraphProposalSurrogate, GraphProposalSelector
from engine.parent_selectors.transition_store import TransitionRecord, TransitionStore
from engine.parent_selectors.proposal_feedback import effective_proposals
from engine.selection_logging import SelectionLogger
from engine.parent_selectors.rating_prior import RatingPriorConfig, potential_level


GP_SELECTOR_TYPES = {"gp_child_ucb", "halfshrink_ucb", "gp6_global_logei", *GRAPH_SELECTOR_TYPES}

SUPPORTED_SELECTOR_TYPES = {
    *STUDY7_SELECTORS,
    *BRANCH7_SELECTORS,
    *ACQ7_SELECTORS,
    *RATING_ONLY_SELECTORS,
    *GRAPH_SELECTOR_TYPES,
    "halfshrink_ucb", "gp6_global_logei", "agent_selector",
    "gp_child_ucb",
    "legacy_original",
    "progressive_mcgs",
    "parent_metric_greedy",
    "random_parent",
    "bayesian_gain_ucb",
    "bayesian_child_ucb",
    "reward_only_bayesian_ts",
    "bnvs_mean_greedy",
    "bnvs_ts",
    "bnvs_ts_local",
    "cost_aware_bayesian",
    "cost_aware_bayesian_selector",
}

SUPPORTED_DECISION_SCOPES = {"root_only", "all_normal_steps"}
BNVS_SELECTOR_TYPES = {"bnvs_mean_greedy", "bnvs_ts", "bnvs_ts_local"}
BO_SELECTOR_TYPES = SUPPORTED_SELECTOR_TYPES - {
    "legacy_original", "progressive_mcgs", "parent_metric_greedy", "random_parent",
    "agent_selector", "study7_parent_metric_greedy", "study7_rating_only_greedy",
}


class SelectionPending(RuntimeError):
    def __init__(self, revision):
        super().__init__("No eligible work until an in-flight decision completes")
        self.revision = revision


class SearchExhausted(RuntimeError):
    """No legal action remains within the existing budget."""


class SearchStatePersistenceError(RuntimeError):
    """A decision checkpoint could not be written; search must stop."""


@dataclass(frozen=True)
class StepDecisionContext:
    result: SelectorResult

    @property
    def decision(self) -> SelectionDecision:
        return self.result.decision

    @property
    def option(self) -> ExpansionOption:
        return self.result.selected_option


class Phase1SelectorRuntime:
    def __init__(self, cfg):
        validate_root_draft_config(cfg)
        self.cfg = cfg
        self.rating_config = RatingPriorConfig.from_config(cfg)
        self.rating_enabled = self.rating_config.enabled
        if self.rating_enabled and str(cfg.parent_selector.type) not in {
            "gp7_ancestry_logei", "gp7_long_horizon_logei"
        }:
            raise ValueError("agent_rating requires the short or long GP7 LogEI selector")
        self.include_root_draft = getattr(cfg.parent_selector, "include_root_draft", False)
        self.gp_enabled = str(cfg.bo_parent_selector.model) == "gp_matern32"
        self.agent_enabled = str(cfg.parent_selector.type) == "agent_selector"
        self.branch7_enabled = str(cfg.parent_selector.type) in BRANCH7_SELECTORS
        self.acq7_enabled = str(cfg.parent_selector.type) in ACQ7_SELECTORS
        self.rating_only_enabled = str(cfg.parent_selector.type) in RATING_ONLY_SELECTORS
        self.branch7_gp6_enabled = str(cfg.parent_selector.type) == "branch7_gp6_ucb"
        if getattr(getattr(cfg, "branch7", None), "enabled", False) and not self.branch7_enabled:
            raise ValueError("Branch7 components require an explicitly registered Branch7 selector")
        if getattr(getattr(cfg, "acq7", None), "enabled", False) and not self.acq7_enabled:
            raise ValueError("Acq7 components require the registered Acq7 selector")
        self.study7_enabled = str(cfg.parent_selector.type) in STUDY7_SELECTORS or self.branch7_enabled or self.acq7_enabled or self.rating_only_enabled
        self.study7_settings = (rating_only_config(cfg) if self.rating_only_enabled
                               else acq7_config(cfg) if self.acq7_enabled
                               else branch7_config(cfg) if self.branch7_enabled
                               else study7_config(cfg) if self.study7_enabled else None)
        study = getattr(cfg, "study7", None)
        if not self.study7_enabled and (getattr(getattr(study, "rating", None), "enabled", False)
                or getattr(getattr(study, "ancestor", None), "enabled", False)):
            raise ValueError("study7 components require an explicitly registered study7 selector")
        from engine.competition_checkpoint import legacy_enabled
        self.legacy_checkpoint = legacy_enabled(cfg)
        self.managed_decisions = (self.include_root_draft or self.gp_enabled or self.agent_enabled
                                  or self.study7_enabled or self.legacy_checkpoint)
        self.wait_initial_results = self.gp_enabled or self.study7_enabled
        self.root_draft_state = None
        self.recovered_deferred_nodes = []
        self.recovered_generation_contexts = []
        self.selector_type = str(cfg.parent_selector.type)
        prefer_unoccupied = getattr(cfg.parent_selector, "prefer_unoccupied_parents", True)
        if not isinstance(prefer_unoccupied, bool):
            raise ValueError("parent_selector.prefer_unoccupied_parents must be a boolean")
        self.prefer_unoccupied_parents = prefer_unoccupied and self.selector_type in BO_SELECTOR_TYPES
        self._unmanaged_pending_parents: dict[str, str] = {}
        self.graph_enabled = self.selector_type in GRAPH_SELECTOR_TYPES
        self.decision_scope = str(
            getattr(cfg.parent_selector, "decision_scope", "root_only")
        )
        if self.selector_type not in SUPPORTED_SELECTOR_TYPES:
            raise ValueError(
                f"Unsupported parent_selector.type={self.selector_type!r}; "
                f"choose one of {sorted(SUPPORTED_SELECTOR_TYPES)}"
            )
        if self.decision_scope not in SUPPORTED_DECISION_SCOPES:
            raise ValueError(
                f"Unsupported parent_selector.decision_scope={self.decision_scope!r}; "
                f"choose one of {sorted(SUPPORTED_DECISION_SCOPES)}"
            )
        model_type = str(cfg.bo_parent_selector.model)
        supported_models = {"bayesian_ridge", "conjugate_bayesian_linear", "gp_matern32", "none", "study7_state4_gp", "study7_no_surrogate", "branch7_gp4", "branch7_gp6", "acq7_gp4"}
        if model_type not in supported_models:
            raise ValueError(
                f"Unsupported bo_parent_selector.model={model_type!r}; "
                f"choose one of {sorted(supported_models)}"
            )
        if self.gp_enabled != (self.selector_type in GP_SELECTOR_TYPES):
            raise ValueError("A GP selector and model=gp_matern32 must be selected together")
        if self.study7_enabled != model_type.startswith(("study7_", "branch7_", "acq7_")):
            raise ValueError("study7 model requires an explicit study7 managed selector")
        if self.agent_enabled != (model_type == "none"):
            raise ValueError("agent_selector and model=none must be selected together")
        if self.agent_enabled and (self.include_root_draft or self.decision_scope != "all_normal_steps"):
            raise ValueError("agent_selector requires all_normal_steps and root draft disabled")
        if self.agent_enabled and not 0 <= int(cfg.agent.initial_drafts) <= int(cfg.agent.search.num_drafts):
            raise ValueError("agent_selector requires 0 <= initial_drafts <= num_drafts")
        if self.gp_enabled:
            acquisition = cfg.bo_parent_selector.acquisition
            supported_acquisitions = ("child_ucb", "logei") if self.graph_enabled else ("child_ucb",)
            if (self.decision_scope != "all_normal_steps" or acquisition.type not in supported_acquisitions
                    or acquisition.beta_schedule != "fixed" or acquisition.use_cost
                    or acquisition.use_validity_probability):
                raise ValueError(f"GP requires all_normal_steps, fixed {'/'.join(supported_acquisitions)}, no cost/validity weighting")
            if int(cfg.bo_parent_selector.refit_every) != 1:
                raise ValueError("GP6 refits each completed feedback change (refit_every=1)")
            if not 0 <= int(cfg.agent.initial_drafts) <= int(cfg.agent.search.num_drafts):
                raise ValueError("GP6 requires 0 <= initial_drafts <= num_drafts")
        legacy_bayesian_selectors = {
            "bayesian_gain_ucb",
            "bayesian_child_ucb",
            "cost_aware_bayesian",
            "cost_aware_bayesian_selector",
        }
        if self.selector_type in legacy_bayesian_selectors and model_type != "bayesian_ridge":
            raise ValueError(
                f"{self.selector_type} requires bo_parent_selector.model=bayesian_ridge"
            )
        if (
            self.selector_type == "reward_only_bayesian_ts"
            and model_type != "conjugate_bayesian_linear"
        ):
            raise ValueError(
                "reward_only_bayesian_ts requires "
                "bo_parent_selector.model=conjugate_bayesian_linear"
            )
        if (
            self.selector_type in BNVS_SELECTOR_TYPES
            and model_type != "conjugate_bayesian_linear"
        ):
            raise ValueError(
                f"{self.selector_type} requires "
                "bo_parent_selector.model=conjugate_bayesian_linear"
            )
        if not self.agent_enabled and str(cfg.bo_parent_selector.fallback.type) != "progressive_mcgs":
            raise ValueError("Phase 1 supports only fallback.type=progressive_mcgs")
        if (
            model_type == "bayesian_ridge"
            and str(cfg.bo_parent_selector.acquisition.beta_schedule) != "fixed"
        ):
            raise ValueError("Phase 1 currently supports only acquisition.beta_schedule=fixed")
        beta = float(cfg.bo_parent_selector.acquisition.beta)
        if not math.isfinite(beta) or beta < 0:
            raise ValueError("acquisition.beta must be finite and non-negative")
        self.artifact_root = Path(cfg.log_dir) / "phase1"
        self.selection_logger = SelectionLogger(self.artifact_root)
        self.transition_store = TransitionStore(self.artifact_root / "transitions.jsonl")
        if self.study7_enabled:
            from .study7_selector import Study7Selector
            greedy = self.selector_type == "study7_parent_metric_greedy"
            if self.rating_only_enabled:
                from .rating_only_selector import RatingOnlySelector
                self.study7_selector = RatingOnlySelector(config=self.study7_settings,
                    records_provider=self.transition_store.records, selection_logger=self.selection_logger)
            elif self.acq7_enabled:
                from .acq7_selector import Acq7Selector
                self.study7_selector = Acq7Selector(config=self.study7_settings,
                    records_provider=self.transition_store.records, selection_logger=self.selection_logger)
            elif self.branch7_enabled:
                from .branch7_selector import Branch7Selector
                self.study7_selector = Branch7Selector(config=self.study7_settings,
                    records_provider=self.transition_store.records, selection_logger=self.selection_logger)
            else:
                self.study7_selector = Study7Selector(config=self.study7_settings,
                    records_provider=self.transition_store.records, selection_logger=self.selection_logger, greedy=greedy)
            self.surrogate = self.study7_selector.gp
            self.feature_schema_version = (self.study7_settings.schema if self.branch7_enabled or self.acq7_enabled or self.rating_only_enabled
                                           else GREEDY_SCHEMA if greedy else STATE4_SCHEMA)
        elif self.agent_enabled:
            self.surrogate = None
            self.feature_schema_version = AGENT_SCHEMA
        elif self.gp_enabled:
            gp_config = dict(getattr(cfg.bo_parent_selector, "gp", {}) or {})
            variant = gp_config.pop("variant", None)
            if self.graph_enabled:
                if variant is not None:
                    raise ValueError("Graph A/B arms cannot also enable gp.variant")
                graph_config = gp_config.pop("graph", {})
                long_horizon = gp_config.pop("long_horizon", None)
                self.surrogate = GraphProposalSurrogate(
                    mode=self.selector_type, graph_config=graph_config, config=GP6Config(**gp_config),
                    task_id=str(getattr(cfg, "exp_id", "") or ""), beta=beta, selection_logger=self.selection_logger,
                    acquisition="ucb" if cfg.bo_parent_selector.acquisition.type == "child_ucb" else "logei",
                    long_horizon=long_horizon, pending_provider=self._long_horizon_pending)
                self.graph_context = DecisionGraph(
                    self.surrogate.graph_config.laplacian_smoothing if self.surrogate.is_anchor else 0.)
            elif self.selector_type == "gp_child_ucb":
                if variant is not None:
                    raise ValueError("gp.variant requires an explicit GP variant selector")
                self.surrogate = GaussianProposalSurrogate(config=GP6Config(**gp_config),
                                                           selection_logger=self.selection_logger)
            else:
                expected = {"shrink": 0.5, "acquisition": "ucb"} if self.selector_type == "halfshrink_ucb" else {
                    "shrink": 1.0, "acquisition": "logei"}
                if dict(variant or {}) != expected:
                    raise ValueError(f"{self.selector_type} requires gp.variant={expected}")
                self.surrogate = GPVariantSurrogate(
                    variant_config={**expected, "beta": beta}, config=GP6Config(**gp_config),
                    selection_logger=self.selection_logger)
            self.feature_schema_version = self.surrogate.feature_schema_version
        elif model_type == "conjugate_bayesian_linear":
            if (
                self.selector_type == "reward_only_bayesian_ts"
                and str(cfg.bo_parent_selector.acquisition.type)
                != "conservative_thompson"
            ):
                raise ValueError(
                    "reward_only_bayesian_ts requires "
                    "acquisition.type=conservative_thompson"
                )
            if bool(cfg.bo_parent_selector.acquisition.use_validity_probability):
                raise ValueError(
                    "conjugate Bayesian selectors do not use validity probability"
                )
            if bool(cfg.bo_parent_selector.acquisition.use_cost):
                raise ValueError("conjugate Bayesian selectors do not use cost")
            if self.selector_type in BNVS_SELECTOR_TYPES:
                value_cfg = cfg.bo_parent_selector.value_model
                if str(value_cfg.feature_set) != "exact10":
                    raise ValueError("BNVS runtime currently supports only feature_set=exact10")
                if self.selector_type != "bnvs_ts_local" and (
                    str(value_cfg.target) != "incumbent_h1"
                    or int(value_cfg.horizon) != 1
                ):
                    raise ValueError(
                        "BNVS runtime default requires target=incumbent_h1 and horizon=1"
                    )
                self.surrogate = BayesianNodeValueSurrogate(
                    include_root_draft=self.include_root_draft,
                    target_type=(
                        "local_gain"
                        if self.selector_type == "bnvs_ts_local"
                        else "incumbent_h1"
                    ),
                    selection_logger=self.selection_logger,
                    prior_precision=value_cfg.prior_precision,
                    intercept_precision=value_cfg.intercept_precision,
                    prior_shape=value_cfg.prior_shape,
                    prior_scale=value_cfg.prior_scale,
                )
                self.feature_schema_version = self.surrogate.feature_schema_version
            else:
                reward_cfg = cfg.bo_parent_selector.reward_model
                self.surrogate = RewardOnlyBayesianLinearSurrogate(
                    selection_logger=self.selection_logger,
                    prior_precision=reward_cfg.prior_precision,
                    intercept_precision=reward_cfg.intercept_precision,
                    prior_shape=reward_cfg.prior_shape,
                    prior_scale=reward_cfg.prior_scale,
                    uncertainty_multiplier=reward_cfg.uncertainty_multiplier,
                    uncertainty_floor=reward_cfg.uncertainty_floor,
                )
                self.feature_schema_version = REWARD_ONLY_FEATURE_SCHEMA_VERSION
        else:
            default_cost = float(cfg.bo_parent_selector.cost.default_total_wall_time_sec)
            self.surrogate = BayesianTransitionSurrogate(
                default_cost=default_cost,
                selection_logger=self.selection_logger,
            )
            self.feature_schema_version = FEATURE_SCHEMA_VERSION
        self._selection_lock = threading.RLock()
        self._state_changed = threading.Condition(self._selection_lock)
        self._selector_failure = None

        acquisition_cfg = cfg.bo_parent_selector.acquisition
        self.selectors = {} if self.agent_enabled else {
            "progressive_mcgs": ProgressiveMCGSSelector(),
            "random_parent": RandomParentSelector(),
            "parent_metric_greedy": ParentMetricGreedySelector(),
        }
        if self.study7_enabled:
            self.selectors[self.selector_type] = self.study7_selector
        elif self.agent_enabled:
            self.selectors[self.selector_type] = AgentParentSelector(
                cfg=cfg, records_provider=self.transition_store.records,
                selection_logger=self.selection_logger)
        elif model_type == "bayesian_ridge":
            selector_kwargs = {
                "surrogate": self.surrogate,
                "records_provider": self.transition_store.records,
                "min_valid_transitions": cfg.bo_parent_selector.min_valid_transitions,
                "min_total_transitions": cfg.bo_parent_selector.min_total_transitions,
                "min_operator_observations": cfg.bo_parent_selector.min_operator_observations,
                "refit_every": cfg.bo_parent_selector.refit_every,
                "beta": acquisition_cfg.beta,
                "cost_power": acquisition_cfg.cost_power,
            }
            self.selectors.update({
                "bayesian_gain_ucb": BayesianUCBParentSelector(
                    name="bayesian_gain_ucb",
                    acquisition_type="gain_ucb",
                    use_validity_probability=acquisition_cfg.use_validity_probability,
                    **selector_kwargs,
                ),
                "bayesian_child_ucb": BayesianUCBParentSelector(
                    name="bayesian_child_ucb",
                    acquisition_type="child_ucb",
                    use_validity_probability=acquisition_cfg.use_validity_probability,
                    **selector_kwargs,
                ),
                "cost_aware_bayesian": BayesianUCBParentSelector(
                    name="cost_aware_bayesian",
                    acquisition_type="ei_per_cost",
                    use_validity_probability=True,
                    **selector_kwargs,
                ),
                "cost_aware_bayesian_selector": BayesianUCBParentSelector(
                    name="cost_aware_bayesian_selector",
                    acquisition_type="ei_per_cost",
                    use_validity_probability=True,
                    **selector_kwargs,
                ),
            })
        elif self.gp_enabled:
            if self.graph_enabled:
                self.selectors[self.selector_type] = GraphProposalSelector(
                    surrogate=self.surrogate, records_provider=self.transition_store.records,
                    rating_config=self.rating_config)
            elif self.selector_type == "gp_child_ucb":
                self.selectors[self.selector_type] = GaussianProposalSelector(
                    surrogate=self.surrogate, records_provider=self.transition_store.records,
                    beta=acquisition_cfg.beta)
            else:
                self.selectors[self.selector_type] = GPVariantSelector(
                    surrogate=self.surrogate, records_provider=self.transition_store.records)
        elif self.selector_type == "reward_only_bayesian_ts":
            seed = int(getattr(getattr(cfg, "agent", None), "seed", 0) or 0)
            self.selectors["reward_only_bayesian_ts"] = RewardOnlyThompsonSelector(
                surrogate=self.surrogate,
                records_provider=self.transition_store.records,
                min_valid_transitions=cfg.bo_parent_selector.min_valid_transitions,
                min_total_transitions=cfg.bo_parent_selector.min_total_transitions,
                refit_every=cfg.bo_parent_selector.refit_every,
                initial_gain_trust=getattr(
                    acquisition_cfg, "initial_gain_trust", 0.10
                ),
                max_gain_trust=getattr(acquisition_cfg, "max_gain_trust", 0.25),
                full_gain_trust_valid_transitions=getattr(
                    acquisition_cfg, "full_gain_trust_valid_transitions", 30
                ),
                thompson_temperature=getattr(
                    acquisition_cfg, "thompson_temperature", 0.25
                ),
                anneal_remaining_fraction=getattr(
                    acquisition_cfg, "anneal_remaining_fraction", 0.10
                ),
                random_seed=seed,
            )
        elif self.selector_type in BNVS_SELECTOR_TYPES:
            seed = int(getattr(getattr(cfg, "agent", None), "seed", 0) or 0)
            self.selectors[self.selector_type] = BayesianNodeValueSelector(
                name=self.selector_type,
                surrogate=self.surrogate,
                records_provider=self.transition_store.records,
                selection_rule=(
                    "posterior_mean"
                    if self.selector_type == "bnvs_mean_greedy"
                    else "thompson"
                ),
                refit_every=cfg.bo_parent_selector.refit_every,
                thompson_temperature=getattr(
                    acquisition_cfg, "thompson_temperature", 0.35
                ),
                random_seed=seed,
            )

    @staticmethod
    def validate_checkpoint(cfg, journal) -> None:
        saved = getattr(journal, "parent_selector_state", None)
        enabled = getattr(cfg.parent_selector, "include_root_draft", False)
        is_gp = str(cfg.bo_parent_selector.model) == "gp_matern32"
        saved_gp = bool(saved and saved.get("feature_schema_version") in {GP6_SCHEMA, GP7_SCHEMA})
        is_agent = str(cfg.parent_selector.type) == "agent_selector"
        saved_agent = bool(saved and saved.get("feature_schema_version") == AGENT_SCHEMA)
        is_study7 = str(cfg.parent_selector.type) in {*STUDY7_SELECTORS, *BRANCH7_SELECTORS, *ACQ7_SELECTORS, *RATING_ONLY_SELECTORS}
        saved_study7 = bool(saved and saved.get("feature_schema_version") in {STATE4_SCHEMA, GREEDY_SCHEMA, *BRANCH7_SCHEMAS, ACQ7_SCHEMA, RATING_ONLY_SCHEMA, "mf_gp_v1"})
        if journal.nodes and ((saved or {}).get("feature_schema_version") == "mf_gp_v1") != (getattr(getattr(cfg, "evaluation", None), "multifidelity_mode", None) == "mf_gp"):
            raise ValueError("Cannot resume across MF-GP and other modes")
        if journal.nodes and is_study7 != saved_study7:
            raise ValueError("Cannot resume across study7 and native selector contracts")
        if journal.nodes and is_agent != saved_agent:
            raise ValueError("Cannot resume across agent and non-agent selector contracts")
        if journal.nodes and is_gp and saved and saved.get("selector_type") != str(cfg.parent_selector.type):
            raise ValueError("Cannot resume across GP variants; start a new run")
        if journal.nodes and is_gp != saved_gp:
            raise ValueError("Cannot resume across GP6 and legacy selector contracts; start a new run")
        if journal.nodes and bool(saved and saved.get("include_root_draft")) != enabled:
            raise ValueError("Cannot resume across include_root_draft modes; automatic migration is not supported")

    def bind_agent(self, agent, *, resuming=False) -> None:
        """Restore the decision ledger before rebuilding the agent's indexes."""
        self._agent = agent
        if not self.managed_decisions:
            return
        from engine.search_node import Journal
        from utils import serialize
        from utils.seed import restore_random_state

        contract = {
            "include_root_draft": self.include_root_draft,
            "feature_schema_version": BNVS_ROOT_FEATURE_SCHEMA_VERSION,
            "feature_dimension": 12,
            "target": "incumbent_h1",
            "initial_goal": min(int(agent.acfg.initial_drafts), int(agent.acfg.steps)),
            "draft_cap": int(agent.scfg.num_drafts),
            "rng_seed": int(getattr(agent.acfg, "seed", 0) or 0),
            "prior": {} if self.gp_enabled or self.agent_enabled or self.study7_enabled else {key: getattr(self.surrogate, key) for key in (
                "prior_precision", "intercept_precision", "prior_shape", "prior_scale")},
        }
        if self.gp_enabled:
            contract.update(feature_schema_version=self.surrogate.feature_schema_version,
                            feature_dimension=len(self.surrogate.feature_names),
                            target="continuous_child_quality", selector_type=self.selector_type,
                            prior=asdict(self.surrogate.config), beta=float(self.cfg.bo_parent_selector.acquisition.beta))
            if getattr(self, "graph_enabled", False):
                contract.update(graph_config=asdict(self.surrogate.graph_config),
                                task_id=self.surrogate.task_id, valid_ordinary_only=True,
                                target="fixed_anchor_error_scale" if self.surrogate.is_anchor else "parent_residual",
                                acquisition="child_ucb" if self.surrogate.variant_config.acquisition == "ucb" else "global_logei")
                if self.surrogate.is_long_horizon:
                    contract.update(long_horizon=asdict(self.surrogate.long_horizon_config),
                        long_horizon_contract=self.surrogate.state_dict()["long_horizon_contract"],
                        target="fixed_h_best_utility_parent_residual", valid_ordinary_only=False)
        if self.agent_enabled:
            contract.update(feature_schema_version=AGENT_SCHEMA, feature_dimension=0,
                            target="agent_selection_no_surrogate", selector_type=self.selector_type,
                            agent_config=self.selectors[self.selector_type].contract())
        if self.study7_enabled:
            contract.update(feature_schema_version=self.feature_schema_version,
                feature_dimension=(8 if self.feature_schema_version == "mf_gp_v1" else 6 if self.branch7_gp6_enabled else 0 if self.study7_selector.greedy else 4),
                target=("continuous_child_quality" if self.branch7_gp6_enabled
                        else "agent_rating_only" if self.rating_only_enabled
                        else "parent_metric" if self.study7_selector.greedy else
                        "absolute_signed_quality" if self.feature_schema_version == "mf_gp_v1" else "valid_ordinary_proposal_residual"),
                selector_type=self.selector_type, study7=self.study7_selector.contract(),
                task_protocol_id=getattr(getattr(agent, "autoresearch", None), "protocol_id", None))
        if getattr(agent, "fixed_evaluation", None):
            contract["evaluation_protocol_id"] = agent.fixed_evaluation.protocol_id
        contract["agent_rating"] = self.rating_config.contract()
        if self.legacy_checkpoint:
            contract["checkpoint_selector"] = "legacy_original_v1"
        saved = getattr(agent.journal, "parent_selector_state", None)

        def compatible_defaults(payload):
            if not payload:
                return payload
            payload = copy.deepcopy(payload)
            payload.setdefault("agent_rating", {"enabled": False})
            if "graph_config" in payload:
                payload["graph_config"].setdefault("use_ancestry_kernel", True)
            return payload

        saved = compatible_defaults(saved)
        if resuming:
            if not saved or any(saved.get(key) != value for key, value in contract.items()):
                raise ValueError("Incompatible managed selector checkpoint contract")
            path = self.artifact_root / "root_draft_state.json"
            latest = compatible_defaults(json.loads(path.read_text())) if path.is_file() else saved
            if any(latest.get(key) != value for key, value in contract.items()):
                raise ValueError("Incompatible managed selector decision ledger")
            saved = latest if latest["revision"] >= saved["revision"] else saved
            if saved["pending"] and self._owner_alive(saved):
                raise RuntimeError("Cannot resume pending decisions while their original process is alive")
            self.root_draft_state = copy.deepcopy(saved)
            if self.legacy_checkpoint:
                agent.journal.study7_search_state = copy.deepcopy(saved.get("legacy_search_state"))
            if self.rating_enabled:
                count = saved.get("rating_bo_decision_count")
                if type(count) is not int or count < 0:
                    raise ValueError("Missing/invalid piBO counter in checkpoint")
            recovered = serialize.loads_json(saved["journal_snapshot"], Journal)
            agent.journal.nodes = recovered.nodes
            agent.virtual_root = agent.journal.nodes[0]
            agent.journal.python_random_state = saved["python_random_state"]
            agent.journal.numpy_random_state = saved["numpy_random_state"]
            restore_random_state(saved["python_random_state"], saved["numpy_random_state"])
            if self.gp_enabled:
                self.surrogate.load_state_dict(saved["gp_reference"])
            if self.study7_enabled:
                self.study7_selector.load_state_dict(saved["study7_state"])
            for key, value in saved.get("posterior", {}).items():
                if key in {"posterior_mean", "posterior_precision_inverse"} and value is not None:
                    value = np.asarray(value, dtype=float)
                    shape = (13,) if key == "posterior_mean" else (13, 13)
                    if value.shape != shape or not np.isfinite(value).all():
                        raise ValueError("Incompatible BNVS checkpoint posterior dimensions")
                setattr(self.surrogate, key, value)
            # A transition append may precede the atomic ledger replacement.
            by_id = {r.decision_id: r for r in self.transition_store.records()}
            for payload in saved["observations"]:
                record = TransitionRecord.from_dict(payload)
                if record.decision_id not in by_id:
                    self.transition_store.append(record)
                    by_id[record.decision_id] = record
        else:
            self.root_draft_state = {
                **contract, "initial_submitted": 0, "initial_terminal": 0,
                "draft_submitted": 0, "submitted_count": 0,
                "decision_count": 0, "revision": 0, "pending": {},
            }
        if self.rating_enabled and not resuming:
            self.root_draft_state["rating_bo_decision_count"] = 0
        self.root_draft_state.update(owner_pid=os.getpid(), owner_host=socket.gethostname())
        if resuming:
            self._reconcile_pending(agent)
        self._persist_root_state(agent)

    @staticmethod
    def _owner_alive(state) -> bool:
        receipt = state.get("executor_release_receipt", {})
        if (receipt.get("owner_host") == state.get("owner_host")
                and receipt.get("owner_pid") == state.get("owner_pid")
                and receipt.get("verified_by") == "slurm_allocation_release"
                and receipt.get("job_id")):
            return False
        if state.get("owner_host") != socket.gethostname():
            raise RuntimeError("Pending decisions belong to another host; reconcile their executor before resuming")
        try:
            os.kill(state["owner_pid"], 0)
        except ProcessLookupError:
            return False
        return True

    def _persist_root_state(self, agent) -> None:
        try:
            self._persist_root_state_unchecked(agent)
        except SearchStatePersistenceError:
            raise
        except Exception as exc:
            raise SearchStatePersistenceError(
                "Parent-selection checkpoint failed; search cannot safely continue"
            ) from exc

    def _persist_root_state_unchecked(self, agent) -> None:
        from engine.search_node import Journal
        from utils import serialize
        from utils.seed import capture_random_state

        state = self.root_draft_state
        if self.legacy_checkpoint:
            if getattr(agent, "_checkpoint_indexes_ready", False):
                from tasks.autoresearch.study7_checkpoint import capture
                from engine.competition_checkpoint import capture_data_preview
                capture(agent)
                capture_data_preview(agent)
            state["legacy_search_state"] = copy.deepcopy(agent.journal.study7_search_state)
        state["revision"] += 1
        state["observations"] = [r.to_dict() for r in self.transition_store.records()]
        state["metric_maximize"] = agent.metric_maximize
        state["elapsed_active_seconds"] = max(
            state.get("elapsed_active_seconds", 0.0), time.time() - getattr(agent, "start_time", time.time()))
        state["python_random_state"], state["numpy_random_state"] = capture_random_state()
        state["posterior"] = {}
        for key in ("posterior_mean", "posterior_precision_inverse", "posterior_shape", "posterior_scale",
                    "utility_scale", "model_version", "fitted_transition_count", "training_transition_ids",
                    "last_fit_wall_time_sec", "fit_diagnostics"):
            if self.gp_enabled or self.agent_enabled or self.study7_enabled or self.legacy_checkpoint:
                continue
            value = getattr(self.surrogate, key)
            state["posterior"][key] = value.tolist() if isinstance(value, np.ndarray) else value
        if self.gp_enabled:
            state["gp_reference"] = self.surrogate.state_dict()
        if self.study7_enabled:
            state["study7_state"] = self.study7_selector.state_dict()
        # A separate, atomic ledger covers generation failures and deferred
        # children which the legacy completed-only journal cannot represent.
        state["journal_snapshot"] = serialize.dumps_json(Journal(nodes=agent.journal.nodes))
        path = self.artifact_root / "root_draft_state.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, allow_nan=False), encoding="utf-8")
        temporary.replace(path)
        with self._state_changed:
            self._state_changed.notify_all()

    def checkpoint_state(self, agent) -> dict:
        if not self.managed_decisions:
            return {"include_root_draft": False}
        with self._selection_lock:
            self._persist_root_state(agent)
            return copy.deepcopy(self.root_draft_state)

    @staticmethod
    def _node_snapshot(node):
        if node is None:
            return None
        shallow = copy.copy(node)
        shallow.parent, shallow.local_best_node = None, None
        shallow.children, shallow.child_count_lock = set(), None
        return {"node": shallow.to_dict(), "parent_id": node.parent.id if node.parent else None,
                "local_best_id": node.local_best_node.id if node.local_best_node else None}

    def _restore_pending_node(self, agent, pending):
        from engine.search_node import SearchNode
        snapshot = pending.get("node_snapshot")
        if not snapshot:
            return None
        by_id = {n.id: n for n in agent.journal.nodes}
        data = snapshot["node"]
        if data["id"] in by_id:
            return by_id[data["id"]]
        node = SearchNode.from_dict(data)
        node.child_count_lock = threading.Lock()
        node.parent = by_id[snapshot["parent_id"]]
        node.local_best_node = by_id.get(snapshot["local_best_id"], node.parent)
        node.__post_init__()
        return node

    def _thaw_context(self, agent, pending):
        frozen = pending["option"]
        parent = next(n for n in agent.journal.nodes if n.id == frozen["parent_node_id"])
        preview = dict(frozen["preview"])
        preview["reference_node_ids"] = tuple(preview["reference_node_ids"])
        option = ExpansionOption(parent.id, frozen["operator"], tuple(frozen["reference_node_ids"]),
                                 frozen["state_features"], parent, OperatorPreview(**preview))
        return StepDecisionContext(SelectorResult(SelectionDecision(**pending["decision"]), option, (), (option,)))

    def _reconcile_pending(self, agent):
        from engine.competition_checkpoint import enabled as competition_checkpoint_enabled
        from engine.mf_recovery import enabled as mf_recovery_enabled
        recover_interrupted = competition_checkpoint_enabled(agent.cfg)
        state = self.root_draft_state
        for decision_id, pending in list(state["pending"].items()):
            record = self.transition_store.by_decision_id(decision_id)
            if record is not None and (record.metadata or {}).get("node_snapshot"):
                pending["node_snapshot"] = record.metadata["node_snapshot"]
            node = self._restore_pending_node(agent, pending)
            if record is not None:
                if node is not None and node not in agent.journal.nodes:
                    from utils.metric import MetricValue, WorstMetricValue
                    node.metric = MetricValue(record.child_metric_raw, maximize=agent.metric_maximize) if record.valid else WorstMetricValue()
                    node.is_buggy, node.is_valid = not record.valid, record.valid
                    node.phase1_transition_id = record.transition_id
                    agent.journal.append(node)
                if pending["initial"]:
                    state["initial_terminal"] += 1
                if self.legacy_checkpoint:
                    state.setdefault("ready_continuations", {})[decision_id] = (
                        (record.metadata or {}).get("continuation_node_id") or record.child_node_id)
                del state["pending"][decision_id]
            elif recover_interrupted and pending["status"] in {"submitted", "reserved"} and (
                    mf_recovery_enabled(agent.cfg) or pending["status"] == "submitted" or pending.get("recovery_already_submitted")):
                # Replay the frozen decision, without another selection, draft
                # charge, acquisition draw, or synthetic failed observation.
                already_submitted = pending["status"] == "submitted" or pending.get("recovery_already_submitted", False)
                pending.update(status="reserved", recovery_already_submitted=already_submitted)
                self.recovered_generation_contexts.append(self._thaw_context(agent, pending))
            elif pending["status"] == "reserved":
                del state["pending"][decision_id]
            elif (pending["status"] == "generated" or
                  (recover_interrupted and pending["status"] == "executing")) and node is not None:
                pending["status"] = "generated"
                context = self._thaw_context(agent, pending)
                node._phase1_decision_context = context
                node.pending_execution = True
                self.recovered_deferred_nodes.append(node)
            else:
                context = self._thaw_context(agent, pending)
                self.record_transition(parent_node=context.option.parent_node, child_node=node,
                    context=context, generation_wall_time_sec=0, execution_wall_time_sec=0,
                    metric_maximize=bool(agent.metric_maximize), reward=-1,
                    failure_reason="interrupted_before_terminal_observation")

    def _rank_options(self, agent, options):
        if self._selector_failure:
            raise AgentSelectionError(self._selector_failure)
        try:
            return self.selectors[self.selector_type].select(agent, options)
        except Exception as exc:
            if not (self.agent_enabled or self.study7_enabled):
                raise
            # Context/serialization errors must stop just like provider failures,
            # rather than being retried forever by the generic experiment loop.
            self._selector_failure = str(exc)
            with self._state_changed:
                self._state_changed.notify_all()
            if isinstance(exc, AgentSelectionError):
                raise
            raise AgentSelectionError(f"Agent selection context failed: {type(exc).__name__}: {exc}") from exc

    def wait_for_change(self, revision, timeout=None):
        with self._state_changed:
            self._state_changed.wait_for(lambda: self._selector_failure or self.root_draft_state["revision"] != revision, timeout=timeout)
            if self._selector_failure:
                raise AgentSelectionError(self._selector_failure)

    def _incumbent(self, agent):
        values = [utility_from_metric(self._raw_metric(n), agent.metric_maximize)
                  for n in agent.journal.nodes if n.stage != "root"
                  and n.is_buggy is False and n.is_valid is not False]
        values.extend(r.child_utility for r in self.transition_store.records() if r.valid)
        return max((v for v in values if v is not None and math.isfinite(v)), default=None)

    def _global_stagnation(self, agent):
        best, stagnant = None, 0
        for record in self.transition_store.records():
            value = record.child_utility if record.valid else None
            if value is not None and (best is None or value - best > agent.scfg.metric_improvement_threshold):
                stagnant = 0
            else:
                stagnant += 1
            if value is not None:
                best = max(best, value) if best is not None else value
        return stagnant

    def _draft_cap_reached(self, agent):
        if self.wait_initial_results:
            return agent.virtual_root.reached_child_limit(agent.scfg)
        state = self.root_draft_state
        reserved = sum(p["status"] == "reserved" and p["option"]["operator"] == "draft"
                       for p in state["pending"].values())
        return state["draft_submitted"] + reserved >= state["draft_cap"]

    def _root_legality(self, agent):
        if not self.include_root_draft:
            return "disabled"
        state = self.root_draft_state
        pending = list(state["pending"].values())
        if state["initial_terminal"] < state["initial_goal"]:
            return "initialization_pending"
        if self._incumbent(agent) is None:
            return "incumbent_unavailable"
        if not self._budget_available(agent):
            return "budget_exhausted"
        if self._draft_cap_reached(agent):
            return "draft_cap_exhausted"
        if any(p["option"]["operator"] == "draft" for p in pending):
            return "draft_pending"
        return "legal"

    def _budget_available(self, agent):
        state = self.root_draft_state
        if self.legacy_checkpoint:
            return len(agent.journal) - 1 < agent.acfg.steps and time.time() - agent.start_time < agent.acfg.time_limit
        reserved = sum(p["status"] == "reserved" and not p.get("recovery_already_submitted")
                       for p in state["pending"].values())
        elapsed = time.time() - agent.search_start_time if agent.search_start_time else 0.0
        return state["submitted_count"] + reserved < agent.acfg.steps and elapsed < agent.acfg.time_limit

    def _add_root_option(self, agent, options):
        if self._root_legality(agent) == "legal":
            options.append(self._make_option(agent, agent.virtual_root,
                preview=OperatorPreview("draft", bypass_selector=False, reason="adaptive_root_draft")))
        return options

    def _reserve_decision(self, agent, result, current_node=None):
        if not self.managed_decisions:
            if self.prefer_unoccupied_parents and is_normal_operator(result.selected_option.preview):
                self._unmanaged_pending_parents[result.decision.decision_id] = result.selected_option.parent_node_id
            return
        state = self.root_draft_state
        decision, option = result.decision, result.selected_option
        if getattr(self, "acq7_enabled", False):
            if decision.decision_id in state["pending"]:
                return
            if not self.study7_selector.validate_reservation(agent, result):
                return
        if getattr(self, "rating_enabled", False) and decision.decision_id in state["pending"]:
            return  # A frozen reservation must not advance piBO twice.
        initial = decision.selection_reason == "initial_draft_bypass"
        if self.legacy_checkpoint:
            initial = option.operator == "draft" and state["initial_submitted"] < state["initial_goal"]
            ready = state.setdefault("ready_continuations", {})
            source = (current_node or agent.virtual_root).id
            consumed = next((key for key, value in ready.items() if value == source), None)
            if consumed is not None:
                del ready[consumed]
        kind = "initialization" if initial else ("fallback" if decision.fallback_used else
                "model" if decision.model_version is not None else "forced")
        if self.study7_enabled and decision.selector_name == self.selector_type:
            kind = "single_candidate" if len(result.options) == 1 else "model"
        movement = "new_direction" if option.operator == "draft" else (
            "continue" if current_node is not None and option.parent_node_id == current_node.id else "switch")
        state["pending"][decision.decision_id] = {
            "status": "reserved", "initial": initial, "kind": kind, "movement": movement,
            "decision_index": state["decision_count"], "decision": decision.to_dict(),
            "root_legality": self._root_legality(agent),
            "option": {"parent_node_id": option.parent_node_id, "operator": option.operator,
                       "reference_node_ids": list(option.reference_node_ids),
                       "state_features": dict(option.state_features), "preview": asdict(option.preview)},
        }
        if self.branch7_enabled or getattr(self, "acq7_enabled", False):
            pending = state["pending"][decision.decision_id]
            pending["lineage"] = self.study7_selector.reserve(
                decision, option, len(self.transition_store.records()))
            pending["option"]["state_features"] = dict(option.state_features)
        if getattr(self, "rating_enabled", False) and option.operator in {"improve", "evolution", "fusion"}:
            # Reserve once under the existing selection lock. Failed attempts
            # still count; debug/draft/aggregation and re-evaluation do not.
            state["rating_bo_decision_count"] += 1
        state["decision_count"] += 1
        self._persist_root_state(agent)
        if getattr(self, "acq7_enabled", False):
            self.study7_selector.publish_committed(decision.decision_id)

    def submit_decision(self, agent, context) -> bool:
        with self._selection_lock:
            pending = self.root_draft_state["pending"].get(context.decision.decision_id)
            if pending is None or pending["status"] != "reserved":
                return False
            pending["status"] = "submitted"
            already_submitted = pending.pop("recovery_already_submitted", False)
            from engine.competition_checkpoint import enabled as competition_checkpoint_enabled
            if competition_checkpoint_enabled(agent.cfg) and not already_submitted:
                pending['child_count_charged'] = False
            self.root_draft_state["submitted_count"] += int(not already_submitted)
            if context.option.operator == "draft" and not already_submitted:
                self.root_draft_state["draft_submitted"] += 1
                self.root_draft_state["initial_submitted"] += int(pending["initial"])
            self._persist_root_state(agent)
            return True

    def cancel_reservation(self, agent, context) -> None:
        with self._selection_lock:
            if not self.managed_decisions:
                self.release_parent(context)
                return
            decision_id = context.decision.decision_id
            pending = self.root_draft_state["pending"].get(decision_id)
            if pending and pending["status"] == "reserved":
                del self.root_draft_state["pending"][decision_id]
                self._persist_root_state(agent)

    def release_parent(self, context: StepDecisionContext) -> None:
        """Release an unmanaged attempt; managed attempts close through their ledger."""
        if not getattr(self, "_unmanaged_pending_parents", None):
            return
        with self._selection_lock:
            self._unmanaged_pending_parents.pop(context.decision.decision_id, None)

    def _pending_parent_ids(self, *, exclude_decision_id=None) -> set[str]:
        with self._selection_lock:
            pending = (self.root_draft_state or {}).get("pending", {})
            return {
                value["option"]["parent_node_id"]
                for decision_id, value in pending.items()
                if decision_id != exclude_decision_id
            } | {
                parent_id for decision_id, parent_id in self._unmanaged_pending_parents.items()
                if decision_id != exclude_decision_id
            }

    def _prefer_unoccupied_options(self, options):
        if not self.prefer_unoccupied_parents:
            return options
        busy = self._pending_parent_ids()
        available = [option for option in options
                     if option.operator in {"draft", "aggregation"}
                     or option.parent_node_id not in busy]
        return available or options

    def _constrain_pending_fallback(self, agent, result, options):
        """Keep legacy cold-start/error fallbacks inside the unoccupied pool."""
        if not self.prefer_unoccupied_parents or not is_normal_operator(result.selected_option.preview):
            return result
        busy = self._pending_parent_ids()
        available = [option for option in options if option.parent_node_id not in busy]
        if result.selected_option.parent_node_id not in busy or not available:
            return result
        option = available[0]
        return self._forced_result(agent, option.parent_node, preview=option.preview,
            reason="pending_parent_fallback", fallback_used=True,
            fallback_reason=result.decision.fallback_reason or "fallback_selected_occupied_parent")

    def _continuation_fallback(self, agent, current_node, current_preview, exc):
        context = self.record_direct_parent(
            agent, current_node, preview=current_preview,
            reason="continuation_selector_outside_options", fallback_used=True,
            fallback_reason=exc.fallback_reason or "selected_special_path",
        )
        record = self.transition_store.by_child_id(current_node.id)
        self._log_continuation_event(
            result=context.result, current_node=current_node, outcome="fallback_continue",
            transition_id=record.transition_id if record else None,
            reward=record.reward if record else None, backpropagated=False,
        )
        return context

    def mark_executing(self, agent, context):
        with self._selection_lock:
            pending = self.root_draft_state["pending"].get(context.decision.decision_id)
            if pending is None or pending["status"] == "executing":
                return False
            pending["status"] = "executing"
            self._persist_root_state(agent)
            return True

    def _long_horizon_pending(self):
        """In-flight attempts are blockers, never fabricated observations."""
        state = self.root_draft_state or {}
        return [{"decision_id": key, "decision_index": value["decision_index"],
                 "parent_node_id": value["option"]["parent_node_id"],
                 "operator": value["option"]["operator"]}
                for key, value in state.get("pending", {}).items()
                if value["option"]["operator"] in {"improve", "evolution", "fusion"}]

    def _make_option(self, agent, parent_node, preview=None) -> ExpansionOption:
        preview = preview or preview_next_operator(
            agent,
            parent_node,
            topk_triggered=(
                bool(getattr(parent_node, "_topk_triggered", False))
                or requires_topk_extension(agent, parent_node)
            ),
        )
        features = {} if self.managed_decisions and not self.legacy_checkpoint and agent.is_root(parent_node) else extract_selection_features(
            agent, parent_node, preview, self.transition_store,
            feature_config=self.cfg.bo_parent_selector.features,
        )
        if self.managed_decisions and not self.legacy_checkpoint:
            incumbent = self._incumbent(agent)
            stagnant = self._global_stagnation(agent)
            if agent.is_root(parent_node):
                features = extract_root_draft_features(agent, self.transition_store.records(), incumbent, stagnant)
                features["operator__draft"] = float(preview.operator == "draft")
                features[f"operator__{preview.operator}"] = 1.0
            else:
                features.update(incumbent_at_decision=incumbent, global_best_utility=incumbent,
                                global_stagnation=float(stagnant))
                if incumbent is not None:
                    features["parent_global_gap"] = incumbent - features["parent_utility"]
        if self.gp_enabled or self.agent_enabled or self.branch7_gp6_enabled:
            features.update(progress_features(agent, parent_node, self.transition_store.records(),
                                              window=self.surrogate.config.history_window if self.gp_enabled else 5))
        if getattr(self, "graph_enabled", False) and preview.operator in {"improve", "evolution", "fusion"}:
            self.graph_context.update(self.transition_store.records())
            features.update(self.graph_context.features(parent_node.id, agent.metric_maximize))
        if getattr(self, "rating_enabled", False) and preview.operator in {"improve", "evolution", "fusion"}:
            level = potential_level(getattr(parent_node, "agent_potential_level", None))
            if parent_node.is_buggy is not False or getattr(parent_node, "is_valid", None) is False:
                level = None
            features.update(agent_potential_level=level,
                agent_potential_assessed_at=getattr(parent_node, "agent_potential_assessed_at", None),
                pibo_iteration=(self.root_draft_state or {}).get("rating_bo_decision_count", 0)+1)
        if self.study7_enabled and preview.operator in {"improve", "evolution", "fusion"}:
            features["study7"] = self.study7_selector.features(agent, parent_node, preview.operator)
        method_id = getattr(getattr(self.cfg, "study7", None), "method_id", "")
        if method_id:
            features["study7_decision_active_seconds"] = max(0.0, time.time() - agent.start_time)
        return ExpansionOption(
            parent_node_id=parent_node.id,
            operator=preview.operator,
            reference_node_ids=preview.reference_node_ids,
            state_features=features,
            parent_node=parent_node,
            preview=preview,
        )

    def _normal_options(self, agent) -> list[ExpansionOption]:
        return self._prefer_unoccupied_options(
            [self._make_option(agent, node) for node in eligible_normal_parents(agent)])

    def _continuation_options(
        self,
        agent,
        current_node,
        current_preview,
    ) -> list[ExpansionOption]:
        current_option = self._make_option(
            agent,
            current_node,
            preview=current_preview,
        )
        other_options = [
            self._make_option(agent, node)
            for node in eligible_normal_parents(agent)
            if node.id != current_node.id
        ]
        # Current is first so exact acquisition ties deterministically continue.
        return self._prefer_unoccupied_options([current_option, *other_options])

    def _candidate_score(self, option: ExpansionOption, score: float | None = None) -> CandidateScore:
        return CandidateScore(
            parent_node_id=option.parent_node_id,
            operator=option.operator,
            reference_node_ids=option.reference_node_ids,
            parent_utility=option.parent_utility,
            mean_gain=None,
            std_gain=None,
            predicted_child_ucb=None,
            predicted_cost=None,
            p_valid=None,
            acquisition_score=score,
        )

    def _forced_result(
        self,
        agent,
        parent_node,
        *,
        reason: str,
        fallback_used: bool = False,
        fallback_reason: str | None = None,
        selector_name: str | None = None,
        preview=None,
    ) -> SelectorResult:
        option = self._make_option(agent, parent_node, preview=preview)
        decision = SelectionDecision(
            selector_name=selector_name or self.selector_type,
            selected_parent_id=option.parent_node_id,
            selected_operator=option.operator,
            selected_score=0.0,
            model_version=None,
            fallback_used=fallback_used,
            decision_id=uuid.uuid4().hex,
            timestamp=utc_timestamp(),
            selection_reason=reason,
            fallback_reason=fallback_reason,
        )
        return SelectorResult(
            decision=decision,
            selected_option=option,
            candidate_scores=(self._candidate_score(option, 0.0),),
            options=(option,),
        )

    def _legacy_result(self, agent, options: list[ExpansionOption]) -> SelectorResult:
        selected_node = node_selection.select_with_soft_switch(agent)
        selected_option = self._make_option(agent, selected_node)
        candidates_by_id = {option.parent_node_id: option for option in options}
        candidates_by_id[selected_option.parent_node_id] = selected_option
        candidates = list(candidates_by_id.values())
        scores = tuple(self._candidate_score(option) for option in candidates)
        return SelectorResult(
            decision=SelectionDecision(
                selector_name="legacy_original",
                selected_parent_id=selected_option.parent_node_id,
                selected_operator=selected_option.operator,
                selected_score=0.0,
                model_version=None,
                fallback_used=False,
                decision_id=uuid.uuid4().hex,
                timestamp=utc_timestamp(),
                selection_reason="original_soft_switch",
            ),
            selected_option=selected_option,
            candidate_scores=scores,
            options=tuple(candidates),
        )

    def _root_bypass_if_required(self, agent) -> SelectorResult | None:
        root = agent.virtual_root
        if not root.reached_child_limit(agent.scfg):
            return self._forced_result(agent, root, reason="initial_draft_bypass")
        if (
            getattr(agent.acfg, "use_aggregation", True)
            and should_trigger_branch_fusion(agent)
            and random.random() < agent.acfg.branch_fusion_trigger_prob
        ):
            return self._forced_result(agent, root, reason="aggregation_bypass")
        return None

    def _select_global_with_root(self, agent):
        if self._selector_failure:
            raise AgentSelectionError(self._selector_failure)
        if self.root_draft_state is None:
            self.bind_agent(agent)
        state = self.root_draft_state
        if not self._budget_available(agent):
            raise SearchExhausted("Search step/time budget exhausted")
        initial_reserved = sum(p["initial"] and p["status"] == "reserved" for p in state["pending"].values())
        if state["initial_submitted"] + initial_reserved < state["initial_goal"]:
            return self._forced_result(agent, agent.virtual_root, reason="initial_draft_bypass",
                preview=OperatorPreview("draft", bypass_selector=True, reason="initial_draft"))
        if self.wait_initial_results and state["initial_terminal"] < state["initial_goal"]:
            raise SelectionPending(state["revision"])
        options = self._prefer_unoccupied_options(self._add_root_option(agent, self._normal_options(agent)))
        if state["initial_terminal"] >= state["initial_goal"] and (
            getattr(agent.acfg, "use_aggregation", True) and should_trigger_branch_fusion(agent)
            and random.random() < agent.acfg.branch_fusion_trigger_prob
        ):
            return self._forced_result(agent, agent.virtual_root, reason="aggregation_bypass",
                preview=OperatorPreview("aggregation", bypass_selector=True, reason="aggregation_bypass"))
        if options:
            self.selectors[self.selector_type].decision_index = state["decision_count"]
            try:
                return self._rank_options(agent, options)
            except SelectionOutsideOptions as exc:
                # Keep the frozen legal option when legacy exception recovery
                # proposes an extra root attempt outside this decision's cap.
                option = next((o for o in options if o.parent_node_id == exc.node.id), options[0])
                return self._forced_result(agent, option.parent_node, preview=option.preview,
                    reason="model_error_fallback", fallback_used=True, fallback_reason=exc.fallback_reason)
        if state["initial_terminal"] >= state["initial_goal"] and self._incumbent(agent) is None:
            drafts_pending = any(p["option"]["operator"] == "draft" for p in state["pending"].values())
            if not self.include_root_draft:
                drafts_pending |= any(r.operator == "draft" and not r.metadata["proposal_closed"]
                    and r.metadata.get("decision_kind") != "initialization"
                    for r in effective_proposals(self.transition_store.records()))
            if not drafts_pending and not self._draft_cap_reached(agent):
                return self._forced_result(agent, agent.virtual_root, reason="feasibility_draft",
                    preview=OperatorPreview("draft", bypass_selector=True, reason="incumbent_unavailable"))
        # Debug remains a forced operation, outside the H1 candidate ranking.
        for node in agent.journal.nodes:
            if node.stage != "root" and not node.is_terminal and not node.lock and not node.reached_child_limit(agent.scfg):
                preview = preview_next_operator(agent, node)
                if preview.operator == "debug":
                    return self._forced_result(agent, node, preview=preview, reason="special_path_bypass")
        if state["pending"]:
            raise SelectionPending(state["revision"])
        raise SearchExhausted("No legal normal, draft, or debug action remains")

    def select_global_parent(self, agent) -> StepDecisionContext:
        """Choose a parent at the exact point where baseline restarts from root."""
        with self._selection_lock:
            decision_started = time.perf_counter()
            options = self._normal_options(agent) if not self.managed_decisions or self.legacy_checkpoint else []
            if self.legacy_checkpoint:
                if not self._budget_available(agent):
                    raise SearchExhausted("Search step/time budget exhausted")
                result = self._legacy_result(agent, options)
            elif self.managed_decisions:
                result = self._select_global_with_root(agent)
            elif self.selector_type == "legacy_original":
                result = self._legacy_result(agent, options)
            elif self.selector_type == "progressive_mcgs":
                try:
                    result = self.selectors[self.selector_type].select(agent, options)
                except SelectionOutsideOptions as exc:
                    result = self._forced_result(
                        agent,
                        exc.node,
                        reason="progressive_special_path",
                        selector_name=self.selector_type,
                    )
            else:
                root_bypass = self._root_bypass_if_required(agent)
                if root_bypass is not None:
                    result = root_bypass
                elif not options:
                    selected = node_selection.select_with_soft_switch(agent)
                    result = self._forced_result(
                        agent,
                        selected,
                        reason="no_normal_options",
                        fallback_used=True,
                        fallback_reason="empty_eligible_parent_set",
                    )
                else:
                    selector = self.selectors[self.selector_type]
                    try:
                        result = selector.select(agent, options)
                    except SelectionOutsideOptions as exc:
                        result = self._forced_result(
                            agent,
                            exc.node,
                            reason="progressive_special_path",
                            fallback_used=bool(exc.fallback_reason),
                            fallback_reason=exc.fallback_reason,
                            selector_name=exc.selector_name or self.selector_type,
                        )
            result = self._constrain_pending_fallback(agent, result, options)
            if (result.decision.selector_name in ("legacy_original", "progressive_mcgs") or result.decision.fallback_used) and (not self.managed_decisions or self.legacy_checkpoint):
                result = self._refresh_selected_option(agent, result)
            commit_preview_random(result.selected_option.preview)
            result = replace(
                result,
                decision=replace(
                    result.decision,
                    decision_wall_time_sec=max(
                        time.perf_counter() - decision_started,
                        0.0,
                    ),
                ),
            )
            self._reserve_decision(agent, result)
            self._log_selection(agent, result)
            return StepDecisionContext(result=result)

    def record_direct_parent(
        self,
        agent,
        parent_node,
        *,
        preview=None,
        reason: str | None = None,
        fallback_used: bool = False,
        fallback_reason: str | None = None,
    ) -> StepDecisionContext:
        with self._selection_lock:
            if self.managed_decisions:
                if self.root_draft_state is None:
                    self.bind_agent(agent)
                if not self._budget_available(agent):
                    raise SearchExhausted("Search step/time budget exhausted")
            decision_started = time.perf_counter()
            preview = preview or preview_next_operator(agent, parent_node)
            reason = reason or (
                "special_path_bypass" if preview.bypass_selector else "direct_continuation"
            )
            result = self._forced_result(
                agent,
                parent_node,
                reason=reason,
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
                preview=preview,
            )
            commit_preview_random(result.selected_option.preview)
            result = replace(
                result,
                decision=replace(
                    result.decision,
                    decision_wall_time_sec=max(
                        time.perf_counter() - decision_started,
                        0.0,
                    ),
                ),
            )
            self._reserve_decision(agent, result, current_node=parent_node)
            self._log_selection(agent, result)
            return StepDecisionContext(result=result)

    def _prefer_current_on_tie(
        self,
        result: SelectorResult,
        current_node,
        *,
        tolerance: float = 1e-9,
    ) -> SelectorResult:
        if result.selected_option.parent_node_id == current_node.id:
            return result
        current_score = next(
            (
                score.acquisition_score
                for score in result.candidate_scores
                if score.parent_node_id == current_node.id
            ),
            None,
        )
        selected_score = result.decision.selected_score
        if (
            current_score is None
            or not math.isfinite(current_score)
            or not math.isfinite(selected_score)
            or abs(selected_score - current_score) > tolerance
        ):
            return result
        current_option = next(
            option for option in result.options
            if option.parent_node_id == current_node.id
        )
        return replace(
            result,
            decision=replace(
                result.decision,
                selected_parent_id=current_option.parent_node_id,
                selected_operator=current_option.operator,
                selected_score=float(current_score),
            ),
            selected_option=current_option,
        )

    def _close_current_continuation(self, agent, current_node):
        record = self.transition_store.by_child_id(current_node.id)
        if record is None or record.reward is None:
            raise RuntimeError(
                f"Cannot switch away from continuation node {current_node.id}: "
                "its completed transition reward is unavailable"
            )
        closed_transition_id = getattr(
            current_node,
            "continuation_closed_transition_id",
            None,
        )
        if closed_transition_id is not None:
            if closed_transition_id != record.transition_id:
                raise RuntimeError(
                    f"Continuation node {current_node.id} was closed with a different transition"
                )
            return record, False

        current_node.continue_improve = False
        agent.current_node_list[:] = [
            node for node in agent.current_node_list if node.id != current_node.id
        ]
        evaluation.backpropagate(current_node, float(record.reward))
        current_node.continuation_closed_transition_id = record.transition_id
        return record, True

    def _log_continuation_event(
        self,
        *,
        result: SelectorResult,
        current_node,
        outcome: str,
        transition_id: str | None,
        reward: float | None,
        backpropagated: bool,
    ) -> None:
        self.selection_logger.append_continuation({
            "decision_id": result.decision.decision_id,
            "selector_name": result.decision.selector_name,
            "selection_reason": result.decision.selection_reason,
            "current_parent_id": current_node.id,
            "selected_parent_id": result.selected_option.parent_node_id,
            "selected_operator": result.selected_option.operator,
            "outcome": outcome,
            "model_backed": (
                result.decision.model_version is not None
                and not result.decision.fallback_used
            ),
            "transition_id": transition_id,
            "reward": reward,
            "backpropagated": backpropagated,
            "timestamp": utc_timestamp(),
        })

    def select_continuation_parent(self, agent, current_node) -> StepDecisionContext:
        with self._selection_lock:
            return self._select_continuation_parent(agent, current_node)

    def _select_continuation_parent(self, agent, current_node) -> StepDecisionContext:
        """Rank current continuation against all other legal normal parents."""
        current_preview = preview_next_operator(agent, current_node)
        occupied_current = self.prefer_unoccupied_parents and current_node.id in self._pending_parent_ids()
        if (
            (self.decision_scope != "all_normal_steps" and not occupied_current)
            or current_node.is_terminal
            or not is_normal_operator(current_preview)
            or self.selector_type == "legacy_original"
        ):
            return self.record_direct_parent(
                agent,
                current_node,
                preview=current_preview,
            )

        with self._selection_lock:
            decision_started = time.perf_counter()
            if self.managed_decisions:
                if self.root_draft_state is None:
                    self.bind_agent(agent)
                if not self._budget_available(agent):
                    raise SearchExhausted("Search step/time budget exhausted")
            if self.wait_initial_results and self.root_draft_state["initial_terminal"] < self.root_draft_state["initial_goal"]:
                raise SelectionPending(self.root_draft_state["revision"])
            options = self._continuation_options(
                agent,
                current_node,
                current_preview,
            )
            selector = self.selectors[self.selector_type]
            if self.managed_decisions:
                self._add_root_option(agent, options)
                selector.decision_index = self.root_draft_state["decision_count"]
            options = self._prefer_unoccupied_options(options)
            current_available = any(option.parent_node_id == current_node.id for option in options)

            if isinstance(
                selector,
                (BayesianUCBParentSelector, RewardOnlyThompsonSelector),
            ):
                cold_start_reason = selector.cold_start_reason(
                    self.transition_store.records(),
                    options,
                )
                if cold_start_reason is not None and current_available:
                    context = self.record_direct_parent(
                        agent,
                        current_node,
                        preview=current_preview,
                        reason="continuation_cold_start",
                        fallback_used=True,
                        fallback_reason=cold_start_reason,
                    )
                    record = self.transition_store.by_child_id(current_node.id)
                    self._log_continuation_event(
                        result=context.result,
                        current_node=current_node,
                        outcome="cold_start_continue",
                        transition_id=record.transition_id if record else None,
                        reward=record.reward if record else None,
                        backpropagated=False,
                    )
                    return context

            original_lock = bool(current_node.lock)
            current_node.lock = False
            try:
                result = self._rank_options(agent, options)
            except SelectionOutsideOptions as exc:
                current_node.lock = original_lock
                if not current_available:
                    option = options[0]
                    result = self._forced_result(agent, option.parent_node, preview=option.preview,
                        reason="pending_parent_fallback", fallback_used=True,
                        fallback_reason=exc.fallback_reason or "fallback_selected_occupied_parent")
                else:
                    return self._continuation_fallback(agent, current_node, current_preview, exc)
            finally:
                current_node.lock = original_lock

            result = self._constrain_pending_fallback(agent, result, options)

            if not self.managed_decisions and (
                result.decision.selector_name in ("legacy_original", "progressive_mcgs")
                or result.decision.fallback_used
            ):
                selected_preview = (
                    preview_next_operator(agent, current_node)
                    if result.selected_option.parent_node_id == current_node.id
                    else None
                )
                result = self._refresh_selected_option(
                    agent,
                    result,
                    preview=selected_preview,
                )
            if not (self.agent_enabled or self.study7_enabled) and result.decision.model_version is not None and not result.decision.fallback_used:
                result = self._prefer_current_on_tie(result, current_node)

            continuing = result.selected_option.parent_node_id == current_node.id
            reason_prefix = (
                "continuation_fallback" if result.decision.fallback_used
                else "continuation_ranked"
            )
            result = replace(
                result,
                decision=replace(
                    result.decision,
                    selection_reason=(
                        f"{reason_prefix}_{'continue' if continuing else 'switch'}"
                    ),
                    decision_wall_time_sec=max(
                        time.perf_counter() - decision_started,
                        0.0,
                    ),
                ),
            )
            commit_preview_random(result.selected_option.preview)
            self._reserve_decision(agent, result, current_node=current_node)
            self._log_selection(agent, result)

            record = self.transition_store.by_child_id(current_node.id)
            backpropagated = False
            if not continuing:
                record, backpropagated = self._close_current_continuation(
                    agent,
                    current_node,
                )
            self._log_continuation_event(
                result=result,
                current_node=current_node,
                outcome=("new_direction" if self.managed_decisions and result.selected_option.operator == "draft"
                         else "continue" if continuing else "switch"),
                transition_id=record.transition_id if record else None,
                reward=record.reward if record else None,
                backpropagated=backpropagated,
            )
            return StepDecisionContext(result=result)

    def _refresh_selected_option(
        self,
        agent,
        result: SelectorResult,
        *,
        preview=None,
    ) -> SelectorResult:
        fresh = self._make_option(
            agent,
            result.selected_option.parent_node,
            preview=preview,
        )
        options = tuple(
            fresh if option.parent_node_id == fresh.parent_node_id else option
            for option in result.options
        )
        scores = tuple(
            replace(
                score,
                operator=fresh.operator,
                reference_node_ids=fresh.reference_node_ids,
                parent_utility=fresh.parent_utility,
            ) if score.parent_node_id == fresh.parent_node_id else score
            for score in result.candidate_scores
        )
        decision = replace(result.decision, selected_operator=fresh.operator)
        return replace(
            result,
            decision=decision,
            selected_option=fresh,
            candidate_scores=scores,
            options=options,
        )

    def _raw_metric(self, node) -> float | None:
        metric = getattr(node, "metric", None)
        value = getattr(metric, "value", None)
        return float(value) if value is not None else None

    def _branch_best_metric(self, agent, node) -> float | None:
        branch_nodes = agent.branch_successful_nodes.get(getattr(node, "branch_id", None), ())
        values = [self._raw_metric(item) for item in branch_nodes]
        values = [value for value in values if value is not None]
        if not values:
            return self._raw_metric(node)
        return (max if agent.metric_maximize else min)(values)

    def _log_selection(self, agent, result: SelectorResult) -> None:
        decision_payload = result.decision.to_dict()
        busy = self._pending_parent_ids(exclude_decision_id=result.decision.decision_id)
        decision_payload.update(
            prefer_unoccupied_parents=self.prefer_unoccupied_parents,
            pending_parent_ids=sorted(busy),
            pending_parent_reuse=(self.prefer_unoccupied_parents
                and is_normal_operator(result.selected_option.preview)
                and result.selected_option.parent_node_id in busy),
        )
        extra = {}
        if self.managed_decisions:
            pending = self.root_draft_state["pending"][result.decision.decision_id]
            extra = {"decision_kind": pending["kind"], "movement": pending["movement"],
                     "root_legality": pending["root_legality"],
                     "draft_submitted": self.root_draft_state["draft_submitted"],
                     "draft_cap": self.root_draft_state["draft_cap"],
                     "decision_index": pending["decision_index"]}
            decision_payload.update(extra)
        method_id = getattr(getattr(self.cfg, "study7", None), "method_id", "")
        if method_id:
            active = max(0.0, time.time() - agent.start_time)
            offered = {option.parent_node_id for option in result.options}
            exclusions = []
            for node in agent.journal.nodes:
                if node.id in offered:
                    continue
                reason = ("root_not_in_normal_pool" if agent.is_root(node) else
                          "terminal" if node.is_terminal else "locked" if node.lock else
                          "invalid" if node.is_buggy is not False or node.is_valid is False else
                          "missing_metric" if self._raw_metric(node) is None else "child_limit_or_native_branch_rule")
                exclusions.append({"node_id": node.id, "reason": reason})
            decision_payload.update(method_id=method_id, run_id=str(agent.cfg.exp_name),
                seed=int(agent.acfg.seed), protocol_id=getattr(getattr(agent, "autoresearch", None), "protocol_id", None),
                active_time=active, remaining_budget_seconds=max(0.0, agent.acfg.time_limit-active),
                candidate_count=len(result.options), excluded_parents=exclusions)
            self.selection_logger._append("study7_dispatch.jsonl", {
                **decision_payload, "candidates": [{"parent_id": option.parent_node_id,
                "operator": option.operator, "reference_ids": list(option.reference_node_ids),
                "preview": asdict(option.preview)} for option in result.options]})
        self.selection_logger.append_decision(decision_payload)
        score_by_key = {
            (score.parent_node_id, score.operator): score
            for score in result.candidate_scores
        }
        for option in result.options:
            node = option.parent_node
            score = score_by_key.get((option.parent_node_id, option.operator))
            features = dict(option.state_features)
            visits = float(getattr(node, "visits", 0) or 0)
            snapshot = {
                "decision_id": result.decision.decision_id,
                "selector_name": result.decision.selector_name,
                "selection_reason": result.decision.selection_reason,
                "fallback_used": result.decision.fallback_used,
                "fallback_reason": result.decision.fallback_reason,
                "parent_node_id": option.parent_node_id,
                "operator": option.operator,
                "reference_node_ids": list(option.reference_node_ids),
                "parent_metric": self._raw_metric(node),
                "global_best_metric": self._raw_metric(agent.best_node),
                "branch_best_metric": self._branch_best_metric(agent, node),
                "depth": features["depth"],
                "visit_count": visits,
                "Q": float(getattr(node, "total_reward", 0.0) or 0.0) / max(visits, 1.0),
                "children_count": len(getattr(node, "children", ())),
                "branch_stagnation": features["branch_stagnation"],
                "recent_child_gains": {
                    "mean": features["recent_gain_mean"],
                    "std": features["recent_gain_std"],
                    "max": features["recent_gain_max"],
                },
                "recent_success_rate": features["recent_valid_rate"],
                "elapsed_time": features["elapsed_fraction"] * max(float(agent.acfg.time_limit), 1.0),
                "remaining_steps": features["remaining_steps_fraction"] * max(int(agent.acfg.steps), 1),
                "feature_schema_version": self.feature_schema_version,
                "feature_vector": features,
                "posterior_mean_gain": score.mean_gain if score else None,
                "posterior_std_gain": score.std_gain if score else None,
                "parent_utility": score.parent_utility if score else option.parent_utility,
                "predicted_child_ucb": score.predicted_child_ucb if score else None,
                "predicted_cost": score.predicted_cost if score else None,
                "p_valid": score.p_valid if score else None,
                "acquisition_score": score.acquisition_score if score else None,
                "thompson_sample": score.thompson_sample if score else None,
                "thompson_temperature": (
                    score.thompson_temperature if score else None
                ),
                "gain_trust": score.gain_trust if score else None,
                "sampled_gain": score.sampled_gain if score else None,
                "posterior_mean_value": (
                    score.posterior_mean_value if score else None
                ),
                "posterior_std_value": (
                    score.posterior_std_value if score else None
                ),
                "sampled_value": score.sampled_value if score else None,
                "posterior_mean_quality": score.posterior_mean_quality if score else None,
                "posterior_std_latent": score.posterior_std_latent if score else None,
                "predictive_std": score.predictive_std if score else None,
                "acquisition_beta": float(self.cfg.bo_parent_selector.acquisition.beta),
                "acquisition_beta_schedule": str(
                    self.cfg.bo_parent_selector.acquisition.beta_schedule
                ),
                "selector_decision_wall_time_sec": (
                    result.decision.decision_wall_time_sec
                ),
                "selected": option.parent_node_id == result.decision.selected_parent_id
                and option.operator == result.decision.selected_operator,
                "timestamp": result.decision.timestamp,
            }
            if self.managed_decisions:
                snapshot.update(extra)
                snapshot["target_available"] = features.get("incumbent_at_decision") is not None
                snapshot["incumbent_at_decision"] = features.get("incumbent_at_decision")
                if agent.is_root(node) and not self.legacy_checkpoint:
                    snapshot.update(parent_metric=None, parent_utility=None, Q=None,
                        visit_count=features["visit_count"], children_count=features["visit_count"],
                        branch_best_metric=None, source_quality=features["source_quality"],
                        source_quality_missing=features["source_quality_missing"])
            if self.gp_enabled:
                snapshot.update(gp_feature_names=list(self.surrogate.feature_names),
                    gp_feature_vector=(self.surrogate._design(features, option.operator).tolist()
                                       if option.operator in ({"improve", "evolution", "fusion"} if getattr(self, "graph_enabled", False)
                                                              else {"improve", "evolution", "fusion", "draft"}) else None),
                    utility_scale=self.surrogate.utility_scale, draft_anchor=self.surrogate.draft_anchor,
                    uncertainty_used="latent", include_root_draft=self.include_root_draft)
            if getattr(self, "graph_enabled", False):
                use_ucb = self.surrogate.variant_config.acquisition == "ucb"
                snapshot.update(graph_config=asdict(self.surrogate.graph_config),
                                acquisition_kind="child_ucb" if use_ucb else "global_logei",
                                acquisition_xi=None if use_ucb else 0.,
                                acquisition_score_units="child_utility" if use_ucb else "log_ei_divided_by_current_scale",
                                absolute_loss_scale=self.surrogate.absolute_loss_scale,
                                uncertainty_used="bounded_kernel_allocation" if self.surrogate.is_anchor else "latent",
                                is_exact_gp_posterior=not self.surrogate.is_anchor)
                if result.decision.model_version == self.surrogate.model_version and not result.decision.fallback_used:
                    snapshot.update(self.surrogate.last_prediction_details.get((option.parent_node_id, option.operator), {}))
                if self.rating_enabled:
                    applied = "rating_prior_bonus" in snapshot and not result.decision.fallback_used
                    snapshot.update(agent_rating_enabled=True, rating_prior_applied=applied)
                    if applied:
                        snapshot.update(acquisition_kind="pibo_child_ucb" if use_ucb else "pibo_global_logei",
                                        acquisition_score_units="child_utility_plus_scaled_log_prior" if use_ucb
                                        else "log_ei_divided_by_current_scale_plus_log_prior")
                if self.surrogate.is_long_horizon:
                    snapshot.update(long_horizon=asdict(self.surrogate.long_horizon_config),
                        prediction_target="fixed_h_best_utility", horizon_is_feature=False)
            elif self.gp_enabled and self.selector_type != "gp_child_ucb":
                snapshot.update(variant_config=asdict(self.surrogate.variant_config),
                                acquisition_kind=self.surrogate.variant_config.acquisition,
                                observation_noise_shrunk=False)
            if self.agent_enabled:
                snapshot.update(acquisition_kind="agent_choice_indicator", uncertainty_used=None,
                                agent_selector_model=self.selectors[self.selector_type].settings.model,
                                agent_selector_provider=self.selectors[self.selector_type].settings.base_url)
            if self.study7_enabled:
                snapshot.update(method_id=self.study7_settings.method_id, study7_contract=self.study7_selector.contract())
                if "study7" in features and result.decision.selector_name == self.selector_type:
                    snapshot.update(self.study7_selector.last_prediction_details.get((option.parent_node_id, option.operator), {}))
            self.selection_logger.append_snapshot(snapshot)
        if result.decision.fallback_used:
            self.selection_logger.append_fallback(result.decision.to_dict())

    def attach_decision(self, node, context: StepDecisionContext) -> None:
        decision = context.decision
        option = context.option
        node.phase1_decision_id = decision.decision_id
        node.phase1_selector_name = decision.selector_name
        node.phase1_selected_operator = option.operator
        node.phase1_reference_node_ids = tuple(option.reference_node_ids)
        node.phase1_selection_features = dict(option.state_features)
        node.phase1_fallback_used = decision.fallback_used
        node._phase1_decision_context = context
        if not self.managed_decisions:
            with self._selection_lock:
                self._reserve_decision(None, context.result)
        if self.managed_decisions:
            with self._selection_lock:
                pending = self.root_draft_state["pending"][decision.decision_id]
                pending.update(status="generated", node_snapshot=self._node_snapshot(node))
                self._persist_root_state(self._agent)

    def _record_root_terminal(self, *, parent_node, child_node, context,
                              generation_wall_time_sec, execution_wall_time_sec,
                              metric_maximize, reward, failure_reason):
        with self._selection_lock:
            decision_id = context.decision.decision_id
            previous = self.transition_store.by_decision_id(decision_id)
            if previous is not None:
                return previous
            pending = self.root_draft_state["pending"].get(decision_id)
            if pending is None or pending["status"] == "reserved":
                raise ValueError("A terminal observation requires a submitted decision")
            child_metric = self._raw_metric(child_node)
            if child_metric is not None and not math.isfinite(child_metric):
                child_metric = None
            valid = (failure_reason is None and child_node is not None
                     and child_node.is_buggy is False and child_node.is_valid is not False
                     and child_metric is not None)
            if getattr(self, "rating_enabled", False) and child_node is not None and not valid:
                # Final host validation can invalidate a previously parsed node.
                child_node.agent_potential_level = None
                child_node.agent_potential_assessed_at = None
            parent_metric = None if parent_node.stage == "root" else self._raw_metric(parent_node)
            features = dict(context.option.state_features)
            incumbent = features.get("incumbent_at_decision")
            target_available = incumbent is not None and math.isfinite(incumbent)
            if self.gp_enabled or self.branch7_gp6_enabled:
                # Unlike BNVS H1, a valid initialization draft already has a
                # continuous quality target even before an incumbent exists.
                target_available = bool(valid and (context.option.operator == "draft" or parent_metric is not None))
            if failure_reason is None and not valid:
                failure_reason = "invalid_outcome"
            record = TransitionRecord(
                transition_id=uuid.uuid4().hex, decision_id=decision_id,
                parent_node_id=parent_node.id, child_node_id=child_node.id if child_node else None,
                operator=context.option.operator, reference_node_ids=context.option.reference_node_ids,
                parent_metric_raw=parent_metric, child_metric_raw=child_metric,
                parent_utility=utility_from_metric(parent_metric, metric_maximize),
                child_utility=utility_from_metric(child_metric, metric_maximize) if valid else None,
                improvement=continuous_improvement(parent_metric, child_metric, metric_maximize) if valid else None,
                valid=valid, generation_wall_time_sec=max(float(generation_wall_time_sec), 0.0),
                execution_wall_time_sec=max(float(execution_wall_time_sec), 0.0),
                total_wall_time_sec=max(float(generation_wall_time_sec), 0.0) + max(float(execution_wall_time_sec), 0.0),
                gpu_time_sec=None, reward=float(reward), selector_name=context.decision.selector_name,
                selection_features=features, timestamp=utc_timestamp(),
                metadata={"feature_schema_version": self.feature_schema_version,
                          "target_available": target_available,
                          "target_status": "available" if target_available else "target_unavailable",
                          "incumbent_at_decision": incumbent, "failure_reason": failure_reason,
                          "decision_kind": pending["kind"], "movement": pending["movement"],
                          "proposal_closed": bool(valid or child_node is None or child_node.is_terminal
                              or child_node.debug_depth >= self._agent.scfg.max_debug_depth),
                          "node_snapshot": self._node_snapshot(child_node)},
            )
            if getattr(self, "graph_enabled", False) and self.surrogate.is_long_horizon:
                record = replace(record, metadata={**record.metadata,
                    "long_horizon_decision_index": pending["decision_index"],
                    "long_horizon": asdict(self.surrogate.long_horizon_config),
                    "decision_timestamp": context.decision.timestamp,
                    **({"target_available": False, "target_status": "deferred_long_horizon_target"}
                       if context.option.operator in {"improve", "evolution", "fusion", "debug"} else {})})
            if self.feature_schema_version == "mf_gp_v1" and child_node is not None:
                if getattr(parent_node, "training_fraction", None) != child_node.training_fraction:
                    record = replace(record, improvement=None)
            from tasks.autoresearch.study7_budget import outcome_metadata
            budget = outcome_metadata(self._agent, child_node)
            if budget:
                record = replace(record, metadata={**record.metadata, **budget})
            if child_node is not None:
                child_node.phase1_transition_id = record.transition_id
                child_node.pending_execution = False
                if failure_reason:
                    from utils.metric import WorstMetricValue
                    child_node.is_buggy, child_node.is_valid = True, False
                    child_node.metric = WorstMetricValue()
                    child_node.lock = False
                if child_node not in self._agent.journal.nodes:
                    self._agent.journal.append(child_node)
            self.transition_store.append(record)
            if self.study7_enabled:
                self.study7_selector.observe(self.transition_store.records())
            self.root_draft_state["initial_terminal"] += int(pending["initial"])
            del self.root_draft_state["pending"][decision_id]
            if child_node is not None and hasattr(child_node, "_phase1_decision_context"):
                # The terminal record has the frozen decision. Keeping this
                # live context on the node retains references to candidate
                # options and their graph after the decision is closed.
                del child_node._phase1_decision_context
            self._persist_root_state(self._agent)
            return record

    def record_transition(
        self,
        *,
        parent_node,
        child_node,
        context: StepDecisionContext,
        generation_wall_time_sec: float,
        execution_wall_time_sec: float,
        metric_maximize: bool,
        reward: float,
        failure_reason: str | None = None,
        continuation_node_id: str | None = None,
    ) -> TransitionRecord:
        if self.legacy_checkpoint and (child_node is None or failure_reason is not None):
            # Original MCGS does not admit generator exceptions as scientific
            # nodes. Close only the bookkeeping reservation; retain the log.
            self._close_legacy_decision(context, self._agent.virtual_root.id)
            return None
        if getattr(self, "managed_decisions", False) and not self.legacy_checkpoint:
            return self._record_root_terminal(parent_node=parent_node, child_node=child_node, context=context,
                generation_wall_time_sec=generation_wall_time_sec, execution_wall_time_sec=execution_wall_time_sec,
                metric_maximize=metric_maximize, reward=reward, failure_reason=failure_reason)
        parent_metric = self._raw_metric(parent_node)
        child_metric = self._raw_metric(child_node)
        maximize = bool(metric_maximize)
        valid = (
            child_node.is_buggy is False
            and child_node.is_valid is not False
            and child_metric is not None
        )
        parent_utility = utility_from_metric(parent_metric, maximize)
        child_utility = utility_from_metric(child_metric, maximize) if valid else None
        improvement = (
            continuous_improvement(parent_metric, child_metric, maximize)
            if valid else None
        )
        generation_cost = max(float(generation_wall_time_sec), 0.0)
        execution_cost = max(float(execution_wall_time_sec), 0.0)
        record = TransitionRecord(
            transition_id=uuid.uuid4().hex,
            decision_id=context.decision.decision_id,
            parent_node_id=parent_node.id,
            child_node_id=child_node.id,
            operator=context.option.operator,
            reference_node_ids=context.option.reference_node_ids,
            parent_metric_raw=parent_metric,
            child_metric_raw=child_metric,
            parent_utility=parent_utility,
            child_utility=child_utility,
            improvement=improvement,
            valid=valid,
            generation_wall_time_sec=generation_cost,
            execution_wall_time_sec=execution_cost,
            total_wall_time_sec=generation_cost + execution_cost,
            gpu_time_sec=None,
            reward=float(reward),
            selector_name=context.decision.selector_name,
            selection_features=dict(context.option.state_features),
            timestamp=utc_timestamp(),
        )
        if getattr(self, "_agent", None) is not None:
            if self.feature_schema_version == "mf_gp_v1" and child_node is not None:
                if getattr(parent_node, "training_fraction", None) != child_node.training_fraction:
                    record = replace(record, improvement=None)
            from tasks.autoresearch.study7_budget import outcome_metadata
            record = replace(record, metadata=outcome_metadata(self._agent, child_node))
        if self.legacy_checkpoint:
            record = replace(record, metadata={**(record.metadata or {}),
                "node_snapshot": self._node_snapshot(child_node),
                "continuation_node_id": continuation_node_id or child_node.id})
        self.transition_store.append(record)
        child_node.phase1_transition_id = record.transition_id
        self.release_parent(context)
        if self.legacy_checkpoint:
            self._close_legacy_decision(context, continuation_node_id or child_node.id)
        return record

    def _close_legacy_decision(self, context, continuation_node_id):
        with self._selection_lock:
            state = self.root_draft_state
            pending = state["pending"].pop(context.decision.decision_id)
            state["initial_terminal"] += int(pending["initial"])
            state.setdefault("ready_continuations", {})[context.decision.decision_id] = continuation_node_id
            self._persist_root_state(self._agent)
