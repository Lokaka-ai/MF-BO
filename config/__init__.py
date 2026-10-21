"""configuration and setup utils"""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Hashable, cast
import datetime
import coolname
import rich
from omegaconf import OmegaConf
from rich.syntax import Syntax
from .study7 import SEVEN_PRESETS, Study7Config
from .branch7 import BRANCH7_PRESETS, Branch7Options
from .acq7 import ACQ7_PRESETS, ACQ7_EXTENSION_PRESETS, Acq7Options
from .rating_only import RATING_ONLY_PRESETS
import shutup
from rich.logging import RichHandler
import logging

# Lazy import to avoid circular dependency with engine.search_node
# Journal and filter_journal are imported where needed via _get_journal_classes()
def _get_journal_classes():
    from engine.search_node import Journal, filter_journal
    return Journal, filter_journal

from utils import copytree, preproc_data, serialize

shutup.mute_warnings()
logger = logging.getLogger("MLEvolve")


""" these dataclasses are just for type hinting, the actual config is in config.yaml """


@dataclass
class StageConfig:
    model: str
    temp: float
    base_url: str
    api_key: str

@dataclass
class DecayConfig:
    exploration_constant: float
    lower_bound: float
    alpha: float
    phase_ratios: list
    

@dataclass
class SearchConfig:
    max_debug_depth: int
    debug_prob: float
    num_drafts: int
    metric_improvement_threshold: float
    back_debug_depth: int
    num_bugs: int
    num_improves: int
    topk_max_improves: int
    max_improve_failure: int
    parallel_search_num: int
    branch_stagnation_threshold: int
    topk_stagnation_threshold: int
    top_candidates_size: int
    stagnation_window: int
    num_gpus: int
    explore_switch_start: float
    explore_switch_end: float
    min_exploration_weight: float
    topk_early_k: int
    topk_early_max_per_branch: int
    topk_late_k: int
    topk_late_max_per_branch: int
    force_backprop_late_threshold: float
    force_backprop_late_prob: float
    force_backprop_mid_threshold: float
    force_backprop_mid_modulo: int
    recent_best_window: int
    fusion_min_time_hours: float
    fusion_max_time_hours: float
    fusion_min_successful_nodes: int
    fusion_min_branches: int
    use_stagnation_detection: bool = True

@dataclass
class AgentConfig:
    steps: int
    time_limit: int
    initial_drafts: int
    seed: int
    data_preview: bool
    code: StageConfig
    feedback: StageConfig
    check_data_leakage: bool
    fusion_vs_evolution_prob: float
    branch_fusion_trigger_prob: float
    max_fusion_drafts: int
    use_global_memory: bool
    memory_similarity_threshold: float
    memory_embedding_device: str
    memory_embedding_model_path: str
    search: SearchConfig
    decay: DecayConfig
    use_diff_mode: bool = True
    use_stepwise_generation: bool = True
    use_evolution: bool = True
    use_fusion: bool = True
    use_aggregation: bool = True
@dataclass
class ExecConfig:
    timeout: int
    agent_file_name: str


@dataclass
class ColdstartConfig:
    use_coldstart: bool
    task_json_path: str
    model_json_path: str
    description: str


@dataclass
class InitSolutionConfig:
    use: bool = False


@dataclass
class ParentSelectorConfig:
    type: str = "bnvs_ts"
    decision_scope: str = "all_normal_steps"
    include_root_draft: bool = False
    prefer_unoccupied_parents: bool = True


def validate_root_draft_config(cfg) -> None:
    enabled = getattr(cfg.parent_selector, "include_root_draft", False)
    if not isinstance(enabled, bool):
        raise ValueError("parent_selector.include_root_draft must be a boolean")
    if enabled and cfg.parent_selector.type in {"halfshrink_ucb", "gp6_global_logei", "agent_selector", "gp7_ancestry_logei", "laplacian_anchor_logei", "gp7_long_horizon_logei"}:
        raise ValueError("These comparison settings require include_root_draft=false")
    if not enabled:
        return
    if (cfg.parent_selector.type == "gp_child_ucb"
            and cfg.bo_parent_selector.model == "gp_matern32"):
        initial = int(cfg.agent.initial_drafts)
        cap = int(cfg.agent.search.num_drafts)
        if initial < 0 or cap < initial:
            raise ValueError("include_root_draft requires 0 <= initial_drafts <= search.num_drafts")
        return
    value = cfg.bo_parent_selector.value_model
    if (
        cfg.parent_selector.type not in {"bnvs_ts", "bnvs_mean_greedy"}
        or cfg.bo_parent_selector.model != "conjugate_bayesian_linear"
        or value.target != "incumbent_h1"
        or int(value.horizon) != 1
        or value.feature_set != "exact10"
    ):
        raise ValueError("include_root_draft requires BNVS TS/mean, incumbent_h1, H=1 and the base exact10 configuration")
    initial = int(cfg.agent.initial_drafts)
    cap = int(cfg.agent.search.num_drafts)
    if initial < 0 or cap < initial:
        raise ValueError("include_root_draft requires 0 <= initial_drafts <= search.num_drafts")


@dataclass
class AgentParentSelectorConfig:
    # Separate from agent.code/feedback: changing this only changes the selector.
    model: str = "opencode/muse-spark-1.3-contributor-free"
    base_url: str = "opencode://opencode"
    reasoning_effort: str = "medium"
    timeout_seconds: int = 120  # TOTAL wall-clock budget across all attempts
    max_attempts: int = 2
    retry_delay_seconds: float = 3.0
    prompt_file: str = "agents/prompts/parent_selector.md"
    max_prompt_chars: int = 60000
    code_chars_per_node: int = 1800
    text_chars_per_node: int = 900
    recent_outcomes: int = 10


@dataclass
class BOFeatureConfig:
    include_mcgs_statistics: bool = True
    include_branch_history: bool = True
    include_reference_summary: bool = True


@dataclass
class BOAcquisitionConfig:
    type: str = "thompson"
    beta: float = 1.5
    beta_schedule: str = "fixed"
    use_validity_probability: bool = False
    use_cost: bool = False
    cost_power: float = 1.0
    initial_gain_trust: float = 0.10
    max_gain_trust: float = 0.25
    full_gain_trust_valid_transitions: int = 30
    thompson_temperature: float = 0.35
    anneal_remaining_fraction: float = 0.10


@dataclass
class BORewardModelConfig:
    prior_precision: float = 4.0
    intercept_precision: float = 0.25
    prior_shape: float = 2.0
    prior_scale: float = 1.0
    uncertainty_multiplier: float = 2.0
    uncertainty_floor: float = 0.10


@dataclass
class BOValueModelConfig:
    target: str = "incumbent_h1"
    horizon: int = 1
    feature_set: str = "exact10"
    prior_precision: float = 4.0
    intercept_precision: float = 0.25
    prior_shape: float = 2.0
    prior_scale: float = 1.0


@dataclass
class BOFallbackConfig:
    type: str = "progressive_mcgs"


@dataclass
class BOCostConfig:
    default_total_wall_time_sec: float = 600.0


@dataclass
class BOParentSelectorConfig:
    model: str = "conjugate_bayesian_linear"
    min_valid_transitions: int = 0
    min_total_transitions: int = 0
    min_operator_observations: int = 0
    refit_every: int = 1
    features: BOFeatureConfig = field(default_factory=BOFeatureConfig)
    acquisition: BOAcquisitionConfig = field(default_factory=BOAcquisitionConfig)
    value_model: BOValueModelConfig = field(default_factory=BOValueModelConfig)
    reward_model: BORewardModelConfig = field(default_factory=BORewardModelConfig)
    fallback: BOFallbackConfig = field(default_factory=BOFallbackConfig)
    cost: BOCostConfig = field(default_factory=BOCostConfig)
    gp: dict = field(default_factory=dict)
    agent_rating: dict = field(default_factory=dict)


@dataclass
class EvaluationConfig:
    enabled: bool = False
    protocol_dir: str | None = None
    setup_timeout_seconds: float = 900
    setup_max_attempts: int = 0  # 0: repair until success or the setup deadline.
    score_timeout_seconds: float = 120


@dataclass
class Config(Hashable):
    data_dir: Path
    dataset_dir: Path
    desc_file: Path | None

    goal: str | None
    eval: str | None

    log_dir: Path
    log_level: str
    workspace_dir: Path

    preprocess_data: bool
    copy_data: bool
    resume_run_dir: Path | None
    resume_elapsed_seconds: int

    exp_name: str
    exp_id: str

    torch_hub_dir: str
    pretrain_model_dir: str

    exec: ExecConfig
    agent: AgentConfig
    start_cpu_id: str
    cpu_number: str

    coldstart: ColdstartConfig

    parent_selector: ParentSelectorConfig
    bo_parent_selector: BOParentSelectorConfig

    agent_parent_selector: AgentParentSelectorConfig = field(default_factory=AgentParentSelectorConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    run_result_file: str | None = None

    use_grading_server: bool = True
    init_solution: InitSolutionConfig = field(default_factory=InitSolutionConfig)
    task_type: str = "competition"
    shared_draft_pool: str | None = None
    study7: Study7Config = field(default_factory=Study7Config)
    branch7: Branch7Options = field(default_factory=Branch7Options)
    acq7: Acq7Options = field(default_factory=Acq7Options)
    audited_bo: str = "baseline"


def _get_next_logindex(dir: Path) -> int:
    """Get the next available index for a log directory."""
    max_index = -1
    for p in dir.iterdir():
        try:
            current_index = int(p.name.split("-")[0])
            if current_index > max_index:
                max_index = current_index
        except ValueError:
            pass
    return max_index + 1


def _load_cfg(
    path: Path = Path(__file__).parent / "config.yaml", use_cli_args=True
) -> Config:
    cfg = OmegaConf.load(path)
    if use_cli_args:
        cli_cfg = OmegaConf.from_cli()
        task_preset = cli_cfg.pop("task_preset", None)
        if task_preset is not None:
            if task_preset not in {"autoresearch", "alphaevolve"}:
                raise ValueError("Unknown task_preset; use autoresearch or alphaevolve")
            task_root = Path(__file__).resolve().parents[1] / "tasks" / task_preset
            cfg = OmegaConf.merge(cfg, OmegaConf.load(task_root / "config.yaml"))
            if task_preset == "alphaevolve":
                from tasks.alphaevolve.registry import task_spec
                task_id = cli_cfg.get("exp_id", cfg.exp_id)
                task_spec(task_id)
                cfg.dataset_dir = str(cli_cfg.get("dataset_dir") or task_root.parents[1] / "data/alphaevolve")
                cfg.data_dir = str(Path(cfg.dataset_dir) / task_id / "prepared/public")
                cfg.desc_file = str(Path(cfg.data_dir) / "description.md")
            else:
                cfg.data_dir = str(task_root / "repo")
                cfg.desc_file = str(task_root / "TASK.md")
        # Current AlphaEvolve research default: the same AQ04 BO used in E06.
        preset = cli_cfg.pop("experiment_preset", "acq7_aq04_gp4_level15_rating_jts" if task_preset == "alphaevolve" else None)
        if preset is not None:
            if preset not in {"mlevolve_original", "gp6", "gp6_draft", "halfshrink_ucb", "gp6_global_logei", "agent_selector", "gp7_ancestry_ucb", "gp7_no_ancestry_ucb", "laplacian_anchor_logei", "gp7_long_horizon_ucb", "gp7_short_logei_pibo", "gp7_long_horizon_ucb_pibo", "gp7_short_ucb_pibo", *SEVEN_PRESETS, *BRANCH7_PRESETS, *ACQ7_PRESETS, *ACQ7_EXTENSION_PRESETS, *RATING_ONLY_PRESETS}:
                raise ValueError("Unknown experiment_preset; use mlevolve_original, gp6, gp6_draft, halfshrink_ucb, gp6_global_logei agent_selector, gp7_ancestry_ucb, gp7_no_ancestry_ucb, laplacian_anchor_logei, gp7_long_horizon_ucb, gp7_short_logei_pibo, gp7_long_horizon_ucb_pibo or gp7_short_ucb_pibo")
            cfg = OmegaConf.merge(cfg, OmegaConf.load(Path(__file__).parent / "experiments" / f"{preset}.yaml"))
        if (task_preset == "alphaevolve" and preset == "acq7_aq04_gp4_level15_rating_jts"
                and "audited_bo" not in cli_cfg
                and not (cli_cfg.get("resume_run_dir") or cfg.get("resume_run_dir"))):
            cfg.audited_bo = "redesign"
        resume_run_dir = cli_cfg.get("resume_run_dir") or cfg.get("resume_run_dir")
        if resume_run_dir:
            saved_cfg_path = Path(resume_run_dir).resolve() / "logs" / "config.yaml"
            if saved_cfg_path.is_file():
                saved_cfg = OmegaConf.load(saved_cfg_path)
                if task_preset is not None and saved_cfg.get("task_type", "competition") != task_preset:
                    raise ValueError("Cannot resume a different task type with this task_preset")
                if preset is not None:
                    expected = OmegaConf.load(Path(__file__).parent / "experiments" / f"{preset}.yaml")
                    keys = ("parent_selector.type", "bo_parent_selector.model", "parent_selector.include_root_draft")
                    if any(OmegaConf.select(saved_cfg, key) != OmegaConf.select(expected, key) for key in keys):
                        raise ValueError("Cannot resume a different selector/draft arm with this experiment_preset")
                    if preset in {*SEVEN_PRESETS, *BRANCH7_PRESETS, *ACQ7_PRESETS, *ACQ7_EXTENSION_PRESETS, *RATING_ONLY_PRESETS}:
                        expected_study = OmegaConf.merge(OmegaConf.structured(Study7Config), expected.get("study7", {}))
                        saved_study = OmegaConf.merge(OmegaConf.structured(Study7Config), saved_cfg.get("study7", {}))
                        if expected_study != saved_study:
                            raise ValueError("Cannot resume a different study7 method contract")
                    if preset in BRANCH7_PRESETS:
                        if expected.get("branch7") != saved_cfg.get("branch7"):
                            raise ValueError("Cannot resume a different Branch7 tau/mapping contract")
                    if preset in {*ACQ7_PRESETS, *ACQ7_EXTENSION_PRESETS}:
                        if expected.get("acq7") != saved_cfg.get("acq7"):
                            raise ValueError("Cannot resume a different Acq7 model/acquisition/RNG contract")
                        if expected.bo_parent_selector.acquisition != saved_cfg.bo_parent_selector.acquisition:
                            full_expected = OmegaConf.merge(OmegaConf.structured(BOAcquisitionConfig), expected.bo_parent_selector.acquisition)
                            if full_expected != saved_cfg.bo_parent_selector.acquisition:
                                raise ValueError("Cannot resume a different Acq7 acquisition")
                    if preset in {"halfshrink_ucb", "gp6_global_logei"} and (
                        OmegaConf.select(saved_cfg, "bo_parent_selector.gp.variant") !=
                        OmegaConf.select(expected, "bo_parent_selector.gp.variant")):
                        raise ValueError("Cannot resume a different GP variant")
                    if preset in {"gp7_ancestry_ucb", "gp7_no_ancestry_ucb", "laplacian_anchor_logei", "gp7_long_horizon_ucb", "gp7_short_logei_pibo", "gp7_long_horizon_ucb_pibo", "gp7_short_ucb_pibo"}:
                        full_expected = OmegaConf.merge(OmegaConf.load(Path(__file__).parent / "config.yaml"), expected)
                        if preset in {"gp7_short_logei_pibo", "gp7_long_horizon_ucb_pibo", "gp7_short_ucb_pibo"}:
                            full_expected = OmegaConf.merge(full_expected, cli_cfg)
                            rating_saved = OmegaConf.select(saved_cfg, "bo_parent_selector.agent_rating")
                            if rating_saved != OmegaConf.select(full_expected, "bo_parent_selector.agent_rating"):
                                raise ValueError("Cannot resume piBO arm with changed agent_rating")
                        for key in ("bo_parent_selector.gp", "bo_parent_selector.acquisition", "parent_selector.decision_scope"):
                            if OmegaConf.select(saved_cfg, key) != OmegaConf.select(full_expected, key):
                                raise ValueError(f"Cannot resume graph arm with changed {key}")
                cfg = OmegaConf.merge(cfg, saved_cfg)
        cfg = OmegaConf.merge(cfg, cli_cfg)
    return cfg

def load_cfg(path: Path = Path(__file__).parent / "config.yaml") -> Config:
    """Load config from .yaml file and CLI args, and set up logging directory."""
    return prep_cfg(_load_cfg(path))


def prep_cfg(cfg: Config):
    if cfg.data_dir is None:
        raise ValueError("`data_dir` must be provided.")

    if cfg.desc_file is None and cfg.goal is None:
        raise ValueError(
            "You must provide either a description of the task goal (`goal=...`) or a path to a plaintext file containing the description (`desc_file=...`)."
        )

    if str(cfg.data_dir).startswith("example_tasks/"):
        cfg.data_dir = Path(__file__).parent.parent / cfg.data_dir
    cfg.data_dir = Path(cfg.data_dir).resolve()

    if cfg.desc_file is not None:
        cfg.desc_file = Path(cfg.desc_file).resolve()

    if cfg.resume_run_dir:
        per_run_root = Path(cfg.resume_run_dir).resolve()
        journal_path = per_run_root / "logs" / "journal.json"
        workspace_path = per_run_root / "workspace"
        if not journal_path.is_file() or not workspace_path.is_dir():
            raise ValueError(
                f"resume_run_dir must contain logs/journal.json and workspace/: {per_run_root}"
            )
        cfg.resume_run_dir = per_run_root
        cfg.exp_name = per_run_root.name
        cfg.log_dir = (per_run_root / "logs").resolve()
        cfg.workspace_dir = workspace_path.resolve()
    else:
        top_log_dir = Path(cfg.log_dir).resolve()
        top_workspace_dir = Path(cfg.workspace_dir).resolve()
        # generate experiment name and prefix with consecutive index
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg.exp_name = f"{timestamp}_{cfg.exp_name or coolname.generate_slug(3)}"

        # If log_dir and workspace_dir point to the same path, treat it as a unified
        # "runs" root and place logs/workspace under the per-run directory
        if top_log_dir == top_workspace_dir:
            runs_root = top_log_dir
            runs_root.mkdir(parents=True, exist_ok=True)
            per_run_root = (runs_root / cfg.exp_name).resolve()
            cfg.log_dir = (per_run_root / "logs").resolve()
            cfg.workspace_dir = (per_run_root / "workspace").resolve()
        else:
            top_log_dir.mkdir(parents=True, exist_ok=True)
            top_workspace_dir.mkdir(parents=True, exist_ok=True)
            cfg.log_dir = (top_log_dir / cfg.exp_name).resolve()
            cfg.workspace_dir = (top_workspace_dir / cfg.exp_name).resolve()

    # validate the config
    cfg_schema: Config = OmegaConf.structured(Config)
    cfg = OmegaConf.merge(cfg_schema, cfg)
    if cfg.audited_bo not in {"baseline", "redesign"}:
        raise ValueError("Only baseline or redesigned evolution is supported")
    if cfg.audited_bo == "redesign":
        if cfg.task_type not in {"alphaevolve", "competition"}:
            raise ValueError("Redesigned evolution requires AlphaEvolve or competition evaluation")
        if cfg.task_type == "competition" and not cfg.evaluation.enabled:
            raise ValueError("Competition redesign requires fixed evaluation")

    validate_root_draft_config(cfg)
    return cast(Config, cfg)


def print_cfg(cfg: Config) -> None:
    rich.print(Syntax(OmegaConf.to_yaml(cfg), "yaml", theme="paraiso-dark"))


def load_task_desc(cfg: Config):
    """Load task description from markdown file or config str."""

    # either load the task description from a file
    if cfg.desc_file is not None:
        if not (cfg.goal is None and cfg.eval is None):
            logger.warning(
                "Ignoring goal and eval args because task description file is provided."
            )

        with open(cfg.desc_file) as f:
            return f.read()

    # or generate it from the goal and eval args
    if cfg.goal is None:
        raise ValueError(
            "`goal` (and optionally `eval`) must be provided if a task description file is not provided."
        )

    task_desc = {"Task goal": cfg.goal}
    if cfg.eval is not None:
        task_desc["Task evaluation"] = cfg.eval

    return task_desc


def prep_agent_workspace(cfg: Config):
    """Setup the agent's workspace and preprocess data if necessary."""
    (cfg.workspace_dir / "input").mkdir(parents=True, exist_ok=True)
    (cfg.workspace_dir / "working").mkdir(parents=True, exist_ok=True)
    (cfg.workspace_dir / "submission").mkdir(parents=True, exist_ok=True)

    if getattr(getattr(cfg, "evaluation", None), "enabled", False):
        # AgentSearch installs only the admitted public T/V/test view.
        return
    copytree(cfg.data_dir, cfg.workspace_dir / "input", use_symlinks=not cfg.copy_data)
    if cfg.preprocess_data:
        preproc_data(cfg.workspace_dir / "input")


def save_run(cfg: Config, journal):
    Journal, filter_journal = _get_journal_classes()
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    filtered_journal = filter_journal(journal)
    # save journal
    serialize.dump_json(journal, cfg.log_dir / "journal.json")
    serialize.dump_json(filtered_journal, cfg.log_dir / "filtered_journal.json")
    # save config
    OmegaConf.save(config=cfg, f=cfg.log_dir / "config.yaml")
    
    # save the best found solution
    best_node = journal.get_best_node()
    if best_node is not None:
        with open(cfg.log_dir / "best_solution.py", "w") as f:
            f.write(best_node.code)
