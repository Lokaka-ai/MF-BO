"""Fixed, versioned contract for the seven explicitly requested experiments."""
from dataclasses import asdict, dataclass, field

from omegaconf import OmegaConf


SEVEN_PRESETS = (
    "seven_e01_gp6_original", "seven_e02_state4_ucb",
    "seven_e03_state4_ancestorB_ucb", "seven_e04_state4_ratingmean_ucb",
    "seven_e05_state4_ancestorB_ratingmean_ucb", "seven_e06_mcgs_original",
    "seven_e07_parent_greedy_managed",
)
STUDY7_SELECTORS = {"study7_state4_ucb", "study7_parent_metric_greedy"}
STATE4_SCHEMA = "study7_state4_v1"
GREEDY_SCHEMA = "study7_parent_metric_v1"


@dataclass
class Component:
    enabled: bool = False


@dataclass
class Study7Config:
    method_id: str = ""
    schema: str = STATE4_SCHEMA
    label_protocol: str = "valid_ordinary_proposal_closure_v1"
    ancestor: Component = field(default_factory=Component)
    rating: Component = field(default_factory=Component)
    ucb_coefficient: float = 1.5
    operation_correlation: float = 0.25
    rho_history: float = 0.5
    ancestor_variance_ratio: float = 0.5
    theta_prior_mean: float = 0.25
    theta_prior_std: float = 0.25
    length_scale: float = 1.0
    amplitude: float = 1.0
    noise_std: float = 0.25
    optimize_after: int = 8
    max_optimizer_iterations: int = 25
    max_objective_calls: int = 150
    max_fit_seconds: float = 30.0


def study7_config(cfg):
    raw = getattr(cfg, "study7", {}) or {}
    config = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(Study7Config), raw))
    baseline = asdict(Study7Config())
    actual = asdict(config)
    for key in baseline.keys() - {"method_id", "schema", "ancestor", "rating"}:
        if actual[key] != baseline[key]:
            raise ValueError(f"study7.{key} differs from the registered v1 contract")
    for component in (config.ancestor, config.rating):
        if type(component.enabled) is not bool:
            raise ValueError("study7 component switches must be booleans")
    expected = {"E02": (False, False), "E03": (True, False),
                "E04": (False, True), "E05": (True, True), "E07": (False, False)}
    if config.method_id not in expected:
        raise ValueError("New study7 selectors require E02/E03/E04/E05/E07")
    if config.schema != (GREEDY_SCHEMA if config.method_id == "E07" else STATE4_SCHEMA):
        raise ValueError("study7 method/schema mismatch")
    if (config.ancestor.enabled, config.rating.enabled) != expected[config.method_id]:
        raise ValueError("study7 method identity and component switches disagree")
    greedy = str(cfg.parent_selector.type) == "study7_parent_metric_greedy"
    if greedy != (config.method_id == "E07"):
        raise ValueError("E07 must use its explicit managed non-GP selector")
    expected_model = "study7_no_surrogate" if greedy else "study7_state4_gp"
    if cfg.bo_parent_selector.model != expected_model:
        raise ValueError("study7 selector and model do not match")
    if cfg.parent_selector.include_root_draft or cfg.parent_selector.decision_scope != "all_normal_steps":
        raise ValueError("study7 requires managed all-normal decisions without root acquisition")
    acq = cfg.bo_parent_selector.acquisition
    if (acq.type != "child_ucb" or float(acq.beta) != 1.5 or acq.beta_schedule != "fixed"
            or acq.use_cost or acq.use_validity_probability or int(cfg.bo_parent_selector.refit_every) != 1):
        raise ValueError("study7 requires the fixed 1.5 latent-UCB configuration")
    if getattr(cfg.bo_parent_selector.agent_rating, "enabled", False):
        raise ValueError("study7 and the old piBO rating mode are mutually exclusive")
    return config
