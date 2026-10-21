"""Fixed GP4 model/acquisition contracts for the seven Acq7 arms."""
from dataclasses import asdict, dataclass

from omegaconf import OmegaConf

from .branch7 import Branch7ModelConfig
from .study7 import Study7Config

ACQ7_SCHEMA = "acq7_gp4_v1"
ACQ7_SELECTORS = {"acq7_gp4"}
ACQ7_ARMS = {
    "AQ01": ("constant", 1.5, False, "joint_ts", 0.0),
    "AQ02": ("constant", 2.0, False, "joint_ts", 0.0),
    "AQ03": ("level_matern32", 1.5, False, "joint_ts", 0.0),
    "AQ04": ("level_matern32", 1.5, True, "joint_ts", 0.0),
    "AQ05": ("level_matern32", 1.5, True, "finite_pool_kg", 0.0),
    "AQ06": ("affine", 1.5, False, "joint_ts", 1.0),
    "AQ07": ("level_matern32", 1.5, True, "retained_outcome_kg", 0.0),
}
ACQ7_PRESETS = (
    "acq7_aq01_gp4_const15_jts", "acq7_aq02_gp4_const20_jts",
    "acq7_aq03_gp4_level15_jts", "acq7_aq04_gp4_level15_rating_jts",
    "acq7_aq05_gp4_level15_rating_kg", "acq7_aq06_gp4_affine15_jts",
    "acq7_aq07_gp4_level15_rating_rokg",
)

# Explicit user amendment on 2026-09-20; the original seven-arm registry stays
# intact so its frozen manifests and historical analysis remain reproducible.
ACQ7_EXTENSION_ARMS = {"AQ08": ("level_matern32", 2.0, True, "joint_ts", 0.0)}
ACQ7_EXTENSION_PRESETS = ("acq7_aq08_gp4_level20_rating_jts",)


@dataclass
class Acq7Options:
    enabled: bool = False
    branch_kind: str = "constant"
    branch_prior_std: float = 1.5
    branch_lengthscale: float = 1.0
    slope_prior_std: float = 0.0
    acquisition_type: str = "joint_ts"
    pending_policy: str = "native_no_fantasy"
    performance_coordinate: str = "raw_parent_centered_over_frozen_s0"
    rng_algorithm: str = "PCG64"
    rng_purpose_id: int = 7301
    official_draws_per_decision: int = 1
    max_acquisition_seconds: float = 30.0
    legacy_ucb_fields_inactive: bool = True


@dataclass
class Acq7ModelConfig(Branch7ModelConfig):
    branch_kind: str = "constant"
    branch_lengthscale: float = 1.0
    slope_prior_std: float = 0.0
    acquisition_type: str = "joint_ts"
    pending_policy: str = "native_no_fantasy"
    performance_coordinate: str = "raw_parent_centered_over_frozen_s0"
    rng_algorithm: str = "PCG64"
    rng_purpose_id: int = 7301
    official_draws_per_decision: int = 1
    max_acquisition_seconds: float = 30.0
    legacy_ucb_fields_inactive: bool = True
    workflow_seed: int = 42


def acq7_config(cfg):
    study = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(Study7Config), cfg.study7))
    options = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(Acq7Options), cfg.acq7))
    registry = {**ACQ7_ARMS, **ACQ7_EXTENSION_ARMS}
    if study.method_id not in registry:
        raise ValueError("Acq7 requires an explicitly registered AQ method")
    kind, tau, rated, acquisition, slope = registry[study.method_id]
    expected = Acq7Options(enabled=True, branch_kind=kind, branch_prior_std=tau,
        slope_prior_std=slope, acquisition_type=acquisition)
    if asdict(options) != asdict(expected):
        raise ValueError("Acq7 branch/acquisition/RNG/pending contract mismatch")
    for key, value in asdict(Study7Config()).items():
        if key not in {"method_id", "schema", "ancestor", "rating"} and getattr(study, key) != value:
            raise ValueError(f"Acq7 changed inherited study7.{key}")
    if (type(study.rating.enabled) is not bool or study.rating.enabled != rated
            or type(study.ancestor.enabled) is not bool or study.ancestor.enabled
            or study.schema != ACQ7_SCHEMA or cfg.branch7.enabled):
        raise ValueError("Acq7 component/schema identity mismatch")
    if (cfg.parent_selector.type != "acq7_gp4" or cfg.bo_parent_selector.model != "acq7_gp4"
            or cfg.parent_selector.include_root_draft
            or cfg.parent_selector.decision_scope != "all_normal_steps"):
        raise ValueError("Acq7 requires the inherited GP4 managed candidate path")
    acq = cfg.bo_parent_selector.acquisition
    if (acq.type != acquisition or float(acq.beta) != 0.0 or acq.beta_schedule != "fixed"
            or acq.use_cost or acq.use_validity_probability
            or int(cfg.bo_parent_selector.refit_every) != 1
            or cfg.bo_parent_selector.agent_rating.enabled):
        raise ValueError("Acq7 forbids active legacy UCB, cost/validity or piBO acquisition")
    if int(cfg.agent.seed) != 42:
        raise ValueError("Acq7 has exactly one registered workflow seed: 42")
    extra = asdict(options)
    del extra["enabled"]
    return Acq7ModelConfig(**vars(study), **extra, workflow_seed=42)
