from dataclasses import dataclass

from llm import compile_prompt_to_md

from engine.agent_search import AgentSearch as Agent
from engine.executor import Interpreter
from engine.search_node import Journal
from engine.search_node import SearchNode
from omegaconf import OmegaConf
from rich.status import Status
from config import load_task_desc, prep_agent_workspace, save_run, _load_cfg, prep_cfg
from pathlib import Path

@dataclass
class Solution:
    code: str
    valid_metric: float

class Experiment:

    def __init__(self, data_dir: str, goal: str, eval: str | None = None, evaluation: dict | None = None):
        """Initialize a new experiment run.

        Args:
            data_dir (str): Path to the directory containing the data files.
            goal (str): Description of the goal of the task.
            eval (str | None, optional): Optional description of the preferred way for the agent to evaluate its solutions.
        """

        _cfg = _load_cfg(use_cli_args=False)
        _cfg.data_dir = data_dir
        _cfg.goal = goal
        _cfg.eval = eval
        if evaluation is not None:
            _cfg.evaluation = OmegaConf.merge(_cfg.evaluation, evaluation)
        self.cfg = prep_cfg(_cfg)

        self.task_desc = load_task_desc(self.cfg)

        with Status("Preparing agent workspace (copying and extracting files) ..."):
            prep_agent_workspace(self.cfg)

        self.journal = Journal()
        self.agent = Agent(
            task_desc=self.task_desc,
            cfg=self.cfg,
            journal=self.journal,
        )
        self.interpreter = Interpreter(
            self.cfg.workspace_dir, cfg=self.cfg, **OmegaConf.to_container(self.cfg.exec)  # type: ignore
        )

    def run(self, steps: int) -> Solution:
        for _i in range(steps):
            self.agent.step(node=None, exec_callback=self.interpreter.run)
            self.agent.capture_resume_state()
            save_run(self.cfg, self.journal)
        self.interpreter.cleanup_session()

        if self.agent.fixed_evaluation:
            from engine.fixed_evaluation import finalize_submission
            finalize_submission(self.cfg, self.journal)
        best_node = self.journal.get_best_node()
        return Solution(code=best_node.code, valid_metric=best_node.metric.value)

