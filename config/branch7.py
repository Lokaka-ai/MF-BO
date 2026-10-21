"""The seven fixed branch-random-effect arms, separate from historical study7."""
from dataclasses import asdict, dataclass

from omegaconf import OmegaConf

from .study7 import Study7Config

BRANCH7_PRESETS = (
    "branch7_br01_gp6_tau15", "branch7_br02_gp4_tau15",
    "branch7_br03_gp4_ancestorB_tau15", "branch7_br04_gp4_rating_tau15",
    "branch7_br05_gp4_ancestorB_rating_tau15", "branch7_br06_gp6_rating_tau15",
    "branch7_br07_gp4_rating_tau20",
)
BRANCH7_SELECTORS = {"branch7_gp4_ucb", "branch7_gp6_ucb"}
BRANCH7_SCHEMAS = {"branch7_gp4_v1", "branch7_gp6_v1"}
BRANCH7_ARMS = {
    "BR01": (6, False, False, 1.5, "E01"),
    "BR02": (4, False, False, 1.5, "E02"),
    "BR03": (4, True, False, 1.5, "E03"),
    "BR04": (4, False, True, 1.5, "E04"),
    "BR05": (4, True, True, 1.5, "E05"),
    "BR06": (6, False, True, 1.5, "E01"),
    "BR07": (4, False, True, 2.0, "E04"),
}


@dataclass
class Branch7Options:
    enabled: bool = False
    branch_prior_std: float = 1.5
    mapping_version: str = "root_lineage_primary_v1"


@dataclass
class Branch7ModelConfig(Study7Config):
    base_dimension: int = 4
    branch_prior_std: float = 1.5
    mapping_version: str = "root_lineage_primary_v1"
    history_window: int = 5


def branch7_config(cfg):
    study = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(Study7Config), cfg.study7))
    branch = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(Branch7Options), cfg.branch7))
    if study.method_id not in BRANCH7_ARMS:
        raise ValueError("Branch7 requires a registered BR01-BR07 identity")
    dimension, ancestor, rating, tau, _ = BRANCH7_ARMS[study.method_id]
    baseline = asdict(Study7Config())
    actual = asdict(study)
    for key in baseline.keys() - {"method_id", "schema", "ancestor", "rating"}:
        if actual[key] != baseline[key]:
            raise ValueError(f"Branch7 changed inherited study7.{key}")
    if (type(branch.enabled) is not bool or not branch.enabled
            or branch.branch_prior_std != tau or branch.mapping_version != "root_lineage_primary_v1"):
        raise ValueError("Branch7 identity, tau, enabled flag or mapping version mismatch")
    if (study.ancestor.enabled, study.rating.enabled) != (ancestor, rating):
        raise ValueError("Branch7 component identity mismatch")
    if study.schema != f"branch7_gp{dimension}_v1":
        raise ValueError("Branch7 schema mismatch")
    if (cfg.parent_selector.type != f"branch7_gp{dimension}_ucb"
            or cfg.bo_parent_selector.model != f"branch7_gp{dimension}"):
        raise ValueError("Branch7 selector/model mismatch")
    acq = cfg.bo_parent_selector.acquisition
    if (cfg.parent_selector.include_root_draft or cfg.parent_selector.decision_scope != "all_normal_steps"
            or acq.type != "child_ucb" or float(acq.beta) != 1.5 or acq.beta_schedule != "fixed"
            or acq.use_cost or acq.use_validity_probability or int(cfg.bo_parent_selector.refit_every) != 1
            or cfg.bo_parent_selector.agent_rating.enabled):
        raise ValueError("Branch7 requires inherited no-forced-draft latent UCB 1.5 without piBO")
    return Branch7ModelConfig(**{**vars(study), "base_dimension": dimension,
        "branch_prior_std": tau, "mapping_version": branch.mapping_version})
