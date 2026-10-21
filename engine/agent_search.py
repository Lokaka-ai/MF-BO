"""AgentSearch: tree search coordinator; delegates to node_selection, evaluation, execution, solution_manager."""

import logging
import time
from typing import Callable, List, Dict, Optional

from engine.executor import ExecutionResult
from engine.fixed_evaluation import FixedEvaluation, EvaluationError, enabled as fixed_enabled
from engine.search_node import SearchNode, Journal
import utils.data_preview as data_preview
from config import Config, validate_root_draft_config
from utils.metric import WorstMetricValue
from utils.seed import capture_random_state, restore_random_state
import threading
import json

from agents import (
    code_review_agent,
    result_parse_agent,
)
from engine import evaluation, execution, solution_manager
from engine.operator_preview import (
    assert_preview_matches_node,
    execute_operator_preview,
)
from engine.parent_selectors.runtime import Phase1SelectorRuntime, StepDecisionContext, SelectionPending
from utils.data_preview import clean_task_desc

logger = logging.getLogger("MLEvolve")


ExecCallbackType = Callable[[str, bool], ExecutionResult]

class AgentSearch:
    def __init__(
            self,
            task_desc: str,
            cfg: Config,
            journal: Journal,
    ):
        validate_root_draft_config(cfg)
        Phase1SelectorRuntime.validate_checkpoint(cfg, journal)
        self.cfg = cfg
        self.acfg = cfg.agent
        self.scfg = cfg.agent.search
        self.journal = journal
        self.autoresearch = None
        self.alphaevolve = None
        if getattr(cfg, "task_type", "competition") == "alphaevolve":
            from tasks.alphaevolve.adapter import AlphaEvolveTask
            self.alphaevolve = AlphaEvolveTask(cfg)
        if getattr(cfg, "task_type", "competition") == "autoresearch":
            from tasks.autoresearch.adapter import AutoresearchTask
            self.autoresearch = AutoresearchTask(cfg)
        is_resume = bool(self.journal.nodes)
        if getattr(journal, "evaluation_state", None) and not fixed_enabled(cfg):
            raise EvaluationError("A fixed-evaluation checkpoint cannot resume in legacy mode")
        self.fixed_evaluation = None
        if fixed_enabled(cfg):
            self.fixed_evaluation = FixedEvaluation.prepare_or_load(cfg, task_desc, journal)
        if is_resume and self.journal.task_desc is not None:
            self.task_desc = self.journal.task_desc
        else:
            self.task_desc = (task_desc if isinstance(task_desc, str) else json.dumps(task_desc)) if (self.fixed_evaluation or self.autoresearch or self.alphaevolve) else clean_task_desc(task_desc, cfg)
            self.journal.task_desc = self.task_desc
        self.data_preview: str | None = None
        self.current_step = 0
        self.current_node: SearchNode | None = None
        self.all_root = True
        if is_resume:
            self.virtual_root = self.journal.nodes[0]
            if self.virtual_root.stage != "root" or self.virtual_root.parent is not None:
                raise ValueError("Resumed journal does not start with a virtual root node")
        else:
            self.virtual_root = SearchNode(
                parent=None,
                plan="(root)",
                code="",
                metric=WorstMetricValue(),
                stage="root",
            )
            self.journal.append(self.virtual_root)
        self.current_node_list = []
        self.best_metric: float = None
        self.best_node: SearchNode = None
        self.search_start_time = None
        self.journal_lock = threading.RLock()
        self.shared_drafts = None
        if getattr(cfg, "shared_draft_pool", None):
            from engine.shared_drafts import SharedDraftPool
            self.shared_drafts = SharedDraftPool(cfg.shared_draft_pool, self)
        self.save_node_lock = threading.Lock()
        self.start_time = time.time()
        if self.fixed_evaluation:
            self.start_time -= self.fixed_evaluation.budget_charge
            self.search_start_time = self.start_time
        self.use_stepwise_generation = getattr(self.acfg, "use_stepwise_generation", True)

        self.next_branch_id = 1
        self.branch_all_nodes: Dict[int, List[SearchNode]] = {}
        self.branch_successful_nodes: Dict[int, List[SearchNode]] = {}
        self.branch_node_count: Dict[int, int] = {}
        self.use_coldstart = cfg.coldstart.use_coldstart
        self.coldstart_description = cfg.coldstart.description

        # Top-N candidates
        self.top_k = self.scfg.top_candidates_size
        self.top_candidates: List[SearchNode] = []

        # Performance stagnation detection
        self.best_metric_history = []
        self.stagnation_threshold = self.scfg.stagnation_window
        self.post_process_triggered = False
        self.post_process_attempts = 0
        self.max_post_process_attempts = 4
        self.improve_attempts_count = 0
        self.last_successful_improve_step = 0

        self.fusion_draft_count = 0
        self.max_fusion_drafts = cfg.agent.max_fusion_drafts

        self.metric_maximize: bool | None = None
        self.metric_maximize_reasoning: str | None = None
        restored_metrics = [
            node.metric
            for node in self.journal.nodes
            if node.stage != "root"
            and node.metric is not None
            and node.metric.value is not None
            and not node.metric.is_worst
        ]
        if self.alphaevolve:
            self.metric_maximize = self.alphaevolve.spec["metric"]["maximize"]
            self.metric_maximize_reasoning = "Pinned AlphaEvolve mathematical objective."
        elif self.autoresearch:
            self.metric_maximize = False
            self.metric_maximize_reasoning = "Fixed autoresearch validation bits per byte; lower is better."
        elif self.fixed_evaluation:
            self.metric_maximize = self.fixed_evaluation.spec["metric"]["maximize"]
            self.metric_maximize_reasoning = self.fixed_evaluation.spec["metric_source"]
        elif is_resume and restored_metrics:
            self.metric_maximize = bool(restored_metrics[0].maximize)
            self.metric_maximize_reasoning = "Restored from the admitted journal."
        elif is_resume and (getattr(cfg.parent_selector, "include_root_draft", False) or (cfg.bo_parent_selector.model == "gp_matern32" or cfg.parent_selector.type == "agent_selector")
                            or (self.journal.parent_selector_state or {}).get("checkpoint_selector") == "legacy_original_v1"):
            self.metric_maximize = self.journal.parent_selector_state["metric_maximize"]
            self.metric_maximize_reasoning = "Restored from the root-draft checkpoint."
        else:
            result_parse_agent.determine_metric_direction(self)
        self.phase1_runtime = Phase1SelectorRuntime(cfg)
        self.phase1_runtime.bind_agent(self, resuming=is_resume)
        if (self.fixed_evaluation and self.fixed_evaluation.multifidelity_mode == "reset"
                and self.fixed_evaluation.multifidelity_phase == "full"):
            self.phase1_runtime.study7_selector.full_training_only = True
        if is_resume and not self.phase1_runtime.managed_decisions:
            dropped = self.phase1_runtime.transition_store.admit_child_ids(
                {node.id for node in self.journal.nodes}
            )
            if dropped:
                logger.warning(
                    "[resume] ignored %d transition records outside the admitted journal",
                    dropped,
                )

        # Global memory
        self.global_memory = None
        if self.acfg.use_global_memory:
            try:
                from agents.memory.global_memory import GlobalMemoryLayer
                memory_dir = str(self.cfg.workspace_dir / "global_memory")
                self.global_memory = GlobalMemoryLayer(
                    memory_dir=memory_dir,
                    embedding_model_path=self.acfg.memory_embedding_model_path,
                    embedding_device=self.acfg.memory_embedding_device,
                    similarity_threshold=self.acfg.memory_similarity_threshold,
                    admitted_node_ids=(
                        {node.id for node in self.journal.nodes}
                        if is_resume
                        else None
                    ),
                )
                logger.info(f"[AgentSearch] Global memory enabled and initialized at {memory_dir}")
            except Exception as e:
                import traceback
                logger.warning(f"[AgentSearch] Failed to initialize global memory: {e}")
                logger.debug(f"[AgentSearch] Global memory initialization traceback: {traceback.format_exc()}")
                self.global_memory = None
                if self.autoresearch or self.alphaevolve:
                    raise RuntimeError("This task requires configured global memory; initialization failed") from e
        else:
            logger.info("[AgentSearch] Global memory is disabled by config")

        if is_resume:
            elapsed_seconds = max(
                float(getattr(self.journal, "elapsed_active_seconds", 0.0)),
                float(getattr(cfg, "resume_elapsed_seconds", 0)),
                0.0,
            )
            if self.phase1_runtime.managed_decisions:
                elapsed_seconds = max(elapsed_seconds, self.phase1_runtime.root_draft_state["elapsed_active_seconds"])
            self._restore_from_journal(
                elapsed_seconds + (self.fixed_evaluation.budget_charge if self.fixed_evaluation else 0)
            )
            for node in self.phase1_runtime.recovered_deferred_nodes:
                self.branch_all_nodes.setdefault(node.branch_id, []).append(node)
                self.branch_successful_nodes.setdefault(node.branch_id, [])
                self.branch_node_count.setdefault(node.branch_id, 0)
                self.next_branch_id = max(self.next_branch_id, node.branch_id + 1)

        self._checkpoint_indexes_ready = True

    def _restore_from_journal(self, elapsed_seconds: float) -> None:
        """Rebuild transient search indexes around an admitted serialized journal."""
        now = time.time()
        self.start_time = now - elapsed_seconds
        self.search_start_time = now - elapsed_seconds
        self.current_step = len(self.journal)

        for node in self.journal.nodes:
            node.child_count_lock = threading.Lock()
            node.expected_child_count = len(node.children)
            node.lock = False
            if hasattr(node, "pending_execution"):
                node.pending_execution = False

        branch_ids = sorted(
            {
                node.branch_id
                for node in self.journal.nodes
                if node.branch_id is not None
            }
        )
        self.branch_all_nodes = {
            branch_id: [
                node
                for node in self.journal.nodes
                if node.branch_id == branch_id
            ]
            for branch_id in branch_ids
        }
        self.branch_successful_nodes = {
            branch_id: [
                node
                for node in nodes
                if node.is_buggy is False
                and node.metric is not None
                and node.metric.value is not None
            ]
            for branch_id, nodes in self.branch_all_nodes.items()
        }
        self.branch_node_count = {
            branch_id: len(nodes)
            for branch_id, nodes in self.branch_all_nodes.items()
        }
        self.next_branch_id = max(branch_ids, default=0) + 1
        self.fusion_draft_count = sum(
            node.stage == "fusion_draft" for node in self.journal.nodes
        )
        self.current_node_list = [
            node
            for node in self.journal.nodes
            if node.continue_improve
            and not node.is_terminal
            and node.continuation_closed_transition_id is None
        ]

        candidates = [
            node
            for node in self.journal.nodes
            if node.is_buggy is False
            and node.is_valid is not False
            and node.metric is not None
            and node.metric.value is not None
        ]
        if candidates:
            self.best_node = max(candidates, key=lambda node: node.metric)

        self.top_candidates = []
        for node in self.journal.nodes:
            solution_manager.update_top_candidates(self, node)
        self.best_metric_history = [
            node.metric.value
            for node in self.journal.nodes
            if node.is_buggy is False
            and node.metric is not None
            and node.metric.value is not None
        ]
        self.last_successful_improve_step = max(
            (
                node.step
                for node in self.journal.nodes
                if node.stage == "improve" and node.is_buggy is False
            ),
            default=0,
        )
        self.improve_attempts_count = sum(
            node.stage == "improve" for node in self.journal.nodes
        )
        restore_random_state(
            getattr(self.journal, "python_random_state", None),
            getattr(self.journal, "numpy_random_state", None),
        )
        from tasks.autoresearch.study7_checkpoint import restore
        restore(self)
        from engine.competition_checkpoint import restore_generation_charges, restore_data_preview
        restore_generation_charges(self)
        restore_data_preview(self)
        logger.info(
            "[resume] restored %d completed steps, %d branches, %.0f elapsed seconds",
            max(len(self.journal) - 1, 0),
            len(self.branch_all_nodes),
            elapsed_seconds,
        )

    def capture_resume_state(self) -> None:
        """Store restart metadata in the next atomic journal snapshot."""
        from tasks.autoresearch.study7_checkpoint import capture
        capture(self)
        from engine.competition_checkpoint import capture_data_preview
        capture_data_preview(self)
        self.journal.elapsed_active_seconds = max(time.time() - self.start_time, 0.0)
        (
            self.journal.python_random_state,
            self.journal.numpy_random_state,
        ) = capture_random_state()
        if hasattr(self, "phase1_runtime"):
            self.journal.parent_selector_state = self.phase1_runtime.checkpoint_state(self)

    def _serialize_prompt(self, prompt_complete) -> str | None:
        """Serialize prompt (str or dict) to string for saving in node."""
        if prompt_complete is None:
            return None
        if isinstance(prompt_complete, str):
            return prompt_complete
        elif isinstance(prompt_complete, dict):
            return json.dumps(prompt_complete, ensure_ascii=False, indent=2)
        else:
            return str(prompt_complete)

    def update_data_preview(self):
        if getattr(self, "alphaevolve", None):
            self.data_preview = self.alphaevolve.preview()
            return
        if self.autoresearch:
            self.data_preview = self.autoresearch.preview()
            return
        if self.fixed_evaluation:
            self.data_preview = self.fixed_evaluation.preview()
            return
        base_preview = data_preview.generate(self.cfg.workspace_dir)
        submission_format_warning = """

        ⚠️  CRITICAL SUBMISSION FORMAT NOTE:
        - If you see sample_submission.csv or similar files, those contain the CORRECT submission format
        - The column names in these files are the FINAL AUTHORITY for submission format
        - Always use the column names from the actual sample submission files
        """
        self.data_preview = base_preview + submission_format_warning

    def is_root(self, node: SearchNode):
        return node.id is self.virtual_root.id

    def _execute_candidate(self, node, exec_callback):
        if getattr(self, "alphaevolve", None):
            remaining = self.acfg.time_limit - (time.time() - self.start_time)
            return exec_callback(node.code, node.id, True, remaining_seconds=remaining)
        if not self.fixed_evaluation:
            return exec_callback(node.code, node.id, True)
        remaining = self.acfg.time_limit - (time.time() - self.start_time)
        context = self.fixed_evaluation.begin_attempt(node, remaining)
        return exec_callback(node.code, node.id, True, evaluation_context=context)

    def _check_evaluator_health(self):
        if getattr(self, "fixed_evaluation", None) and self.fixed_evaluation.failed.is_set():
            raise EvaluationError("Shared evaluator failed; stopping score admission")

    def _run_single_step(
        self,
        parent_node: SearchNode,
        decision_context: StepDecisionContext,
        exec_callback: ExecCallbackType,
        execute_immediately: bool = True,
        init_solution_path: Optional[str] = None,
    ):
        """Run one search step: select action (draft/debug/improve), execute, parse, validate."""
        result_node = None
        _root = False
        root_enabled = getattr(self.phase1_runtime, "managed_decisions", False)
        if root_enabled:
            with self.journal_lock:
                if parent_node.is_terminal:
                    self.phase1_runtime.cancel_reservation(self, decision_context)
                    return True, None
                if not self.phase1_runtime.submit_decision(self, decision_context):
                    return True, None

        # Enabled mode already admitted this frozen attempt under the journal
        # lock; a concurrent closure must not drop its terminal observation.
        if root_enabled or not parent_node.is_terminal:
            try:
                generation_started = time.perf_counter()
                preview = decision_context.option.preview
                if preview.operator == "draft" and self.shared_drafts:
                    if init_solution_path:
                        raise ValueError("Shared drafts and init_solution cannot be combined")
                    result_node = self.shared_drafts.take(self, decision_context.decision.decision_id)
                else:
                    from engine.competition_checkpoint import track_generation
                    with track_generation(self, decision_context):
                        result_node = execute_operator_preview(
                            self,
                            parent_node,
                            preview,
                            init_solution_path=init_solution_path,
                        )

                if result_node:
                    evaluator = getattr(self, "alphaevolve", None) or self.autoresearch or self.fixed_evaluation
                    if evaluator:
                        result_node.evaluation_protocol_id = evaluator.protocol_id
                    assert_preview_matches_node(preview, result_node)
                    if result_node.stage in ("draft", "fusion_draft"):
                        result_node.lock = True
                        logger.info(
                            f"[_run_single_step] {result_node.stage} node {result_node.id} is locked."
                        )
                    if result_node.shared_draft_pool_id:
                        logger.info("Shared draft %s already passed native code review; preserving its source hash",
                                    result_node.shared_draft_slot)
                    elif init_solution_path:
                        logger.info(f"Node {result_node.id} from init_solution, skipping code review")
                    else:
                        reviewed_code = code_review_agent.run(self, result_node)
                        if reviewed_code.strip() != result_node.code.strip():
                            logger.info(f"Node {result_node.id} code has been reviewed and modified")
                            result_node.code = reviewed_code
                        else:
                            logger.info(f"Node {result_node.id} passed code review without changes")

                    if (self.fixed_evaluation and getattr(self.cfg, "audited_bo", "baseline") == "redesign"
                            and result_node.stage == "evolution"):
                        from agents.mle_redesign_agent import record_reviewed_source
                        record_reviewed_source(self, result_node)

                    result_node.generation_wall_time_sec = max(
                        time.perf_counter() - generation_started, 0.0
                    )
                    if root_enabled:
                        with self.journal_lock:
                            self.phase1_runtime.attach_decision(result_node, decision_context)
                    else:
                        self.phase1_runtime.attach_decision(result_node, decision_context)

                    if not execute_immediately:
                        logger.info(f"Node {result_node.id} code generated and reviewed, execution deferred")
                        result_node.pending_execution = True
                        return _root, result_node
                    execution_started = time.perf_counter()
                    if root_enabled:
                        with self.journal_lock:
                            self.phase1_runtime.mark_executing(self, decision_context)
                    exe_res = self._execute_candidate(result_node, exec_callback)
                    result_node = result_parse_agent.run(self,
                        node=result_node,
                        exec_result=exe_res
                    )
                    execution.validate_executed_node(self, result_node)
                    from tasks.autoresearch.study7_budget import admit_result
                    admit_result(self, result_node)
                    logger.info(f"The metric value of node {result_node.id} is {result_node.metric.value}.")
                    result_node.finish_time = time.strftime("%Y-%m-%dT%H:%M:%S")

                    with self.journal_lock:
                        self._check_evaluator_health()
                        if parent_node.is_buggy and result_node.is_buggy is False:
                            parent_node.is_debug_success = True

                        _root = evaluation.check_improvement(self, result_node, parent_node)
                        if self.best_node and result_node.metric.maximize and self.best_node.metric.maximize != result_node.metric.maximize:
                            logger.warning(
                                "New node's metric is inconsistent with metrics in the journal. Returning to the parent node to regenerate.")
                            raise ValueError(
                                "New node's metric is inconsistent with metrics in the journal. Returning to the parent node to regenerate.")
                        self.journal.append(result_node)
                        result_node.execution_wall_time_sec = max(
                            time.perf_counter() - execution_started, 0.0
                        )
                        result_node.total_wall_time_sec = (
                            result_node.generation_wall_time_sec
                            + result_node.execution_wall_time_sec
                        )
                        reward = evaluation.get_node_reward(self, result_node)
                        self.phase1_runtime.record_transition(
                            parent_node=parent_node,
                            child_node=result_node,
                            context=decision_context,
                            generation_wall_time_sec=result_node.generation_wall_time_sec,
                            execution_wall_time_sec=result_node.execution_wall_time_sec,
                            metric_maximize=bool(self.metric_maximize),
                            reward=reward,
                            **({"continuation_node_id": self.virtual_root.id if _root else result_node.id}
                               if self.phase1_runtime.legacy_checkpoint else {}),
                        )
                elif preview.operator == "aggregation":
                    logger.info("Aggregation failed or limit reached; returning to root selection.")
                if root_enabled and result_node is None:
                    with self.journal_lock:
                        self.phase1_runtime.record_transition(parent_node=parent_node, child_node=None,
                            context=decision_context, generation_wall_time_sec=time.perf_counter() - generation_started,
                            execution_wall_time_sec=0, metric_maximize=bool(self.metric_maximize), reward=-1,
                            failure_reason="empty_generation_result")

            except EvaluationError:
                # Infrastructure failure is not a failed proposal/GP observation.
                raise
            except Exception as e:
                logger.warning(f"Step failed for parent {parent_node.id}, rolling back expected child count and propagating zero reward.")
                with self.journal_lock:
                    if root_enabled:
                        self.phase1_runtime.record_transition(parent_node=parent_node, child_node=result_node,
                            context=decision_context, generation_wall_time_sec=time.perf_counter() - generation_started,
                            execution_wall_time_sec=0, metric_maximize=bool(self.metric_maximize), reward=-1,
                            failure_reason=type(e).__name__)
                    evaluation.backpropagate(node=parent_node, value=0, add_to_tree=False)
                    parent_node.sub_expected_child_count()
                raise e

        else:
            with self.journal_lock:
                evaluation.backpropagate(node=parent_node, value=0)
            _root = True
        return _root, result_node

    def step(
        self,
        node: SearchNode,
        exec_callback: ExecCallbackType,
        execute_immediately: bool = True,
        init_solution_path: Optional[str] = None,
        recovery_context: StepDecisionContext | None = None,
    ) -> SearchNode:
        self._check_evaluator_health()
        if self.data_preview is None:
            self.update_data_preview()
        if self.search_start_time is None:
            self.search_start_time = time.time()

        decision_context = recovery_context
        while decision_context is None:
            self._check_evaluator_health()
            try:
                with self.journal_lock:
                    if not node or node.stage == "root":
                        decision_context = self.phase1_runtime.select_global_parent(self)
                    else:
                        decision_context = self.phase1_runtime.select_continuation_parent(self, node)
                break
            except SelectionPending as pending:
                if self.fixed_evaluation:
                    self.phase1_runtime.wait_for_change(pending.revision, timeout=0.5)
                else:
                    self.phase1_runtime.wait_for_change(pending.revision)
        node = decision_context.option.parent_node

        result_node = None
        try:
            _root, result_node = self._run_single_step(
                node,
                decision_context=decision_context,
                exec_callback=exec_callback,
                execute_immediately=execute_immediately,
                init_solution_path=init_solution_path,
            )
        finally:
            if result_node is None or not getattr(result_node, "pending_execution", False):
                self.phase1_runtime.release_parent(decision_context)

        if result_node:
            metric_value = result_node.metric.value if result_node.metric else None
            best_metric = self.best_node.metric.value if (self.best_node and self.best_node.metric) else None
            logger.info(f"[step] {node.id} → {result_node.id}: metric={metric_value}, best={best_metric}")

        if result_node and result_node.metric and result_node.metric.value is not None:
            with self.journal_lock:
                solution_manager.update_best_solution(self, result_node)

        self.current_step = len(self.journal)

        # Cumulative stats
        total_nodes = len(self.journal)
        n_branches = len(self.branch_all_nodes)
        best_val = self.best_node.metric.value if (self.best_node and self.best_node.metric) else None
        logger.info(f"[stats] step={self.current_step}, nodes={total_nodes}, branches={n_branches}, best={best_val}")

        if _root or result_node is None:
            return self.virtual_root
        else:
            return result_node

    def execute_deferred_node(self, node: SearchNode, exec_callback: ExecCallbackType) -> SearchNode:
        context = getattr(node, "_phase1_decision_context", None)
        try:
            return self._execute_deferred_node(node, exec_callback)
        finally:
            if context is not None:
                self.phase1_runtime.release_parent(context)

    def _execute_deferred_node(self, node: SearchNode, exec_callback: ExecCallbackType) -> SearchNode:
        """Execute a node that was generated and reviewed but not yet run (pending_execution=True)."""
        if not hasattr(node, 'pending_execution') or not node.pending_execution:
            logger.warning(f"Node {node.id} is not marked for deferred execution")
            return node

        logger.info(f"Executing deferred node {node.id}")
        parent_node = node.parent
        decision_context = getattr(node, "_phase1_decision_context", None)
        if decision_context is None:
            raise RuntimeError(f"Deferred node {node.id} is missing its pre-generation decision snapshot")
        root_enabled = getattr(self.phase1_runtime, "managed_decisions", False)
        if root_enabled:
            with self.journal_lock:
                if not self.phase1_runtime.mark_executing(self, decision_context):
                    return node

        try:
            execution_started = time.perf_counter()
            exe_res = self._execute_candidate(node, exec_callback)
            node = result_parse_agent.run(self,
                node=node,
                exec_result=exe_res
            )

            execution.validate_executed_node(self, node)
            from tasks.autoresearch.study7_budget import admit_result
            admit_result(self, node)

            logger.info(f"Node {node.id} execution completed: metric={node.metric.value}, is_buggy={node.is_buggy}")

            node.finish_time = time.strftime("%Y-%m-%dT%H:%M:%S")

            with self.journal_lock:
                self._check_evaluator_health()
                if parent_node and parent_node.is_buggy and node.is_buggy is False:
                    parent_node.is_debug_success = True

                _root = evaluation.check_improvement(self, node, parent_node)
                if self.best_node and node.metric.maximize and self.best_node.metric.maximize != node.metric.maximize:
                    logger.warning("New node's metric is inconsistent with metrics in the journal")
                    raise ValueError("New node's metric is inconsistent with metrics in the journal")
                self.journal.append(node)
                logger.info(f"Node {node.id} added to journal")

                node.execution_wall_time_sec = max(
                    time.perf_counter() - execution_started, 0.0
                )
                node.total_wall_time_sec = (
                    node.generation_wall_time_sec + node.execution_wall_time_sec
                )
                reward = evaluation.get_node_reward(self, node)
                self.phase1_runtime.record_transition(
                    parent_node=parent_node,
                    child_node=node,
                    context=decision_context,
                    generation_wall_time_sec=node.generation_wall_time_sec,
                    execution_wall_time_sec=node.execution_wall_time_sec,
                    metric_maximize=bool(self.metric_maximize),
                    reward=reward,
                    **({"continuation_node_id": node.id} if self.phase1_runtime.legacy_checkpoint else {}),
                )

            node.pending_execution = False
            with self.journal_lock:
                solution_manager.update_best_solution(self, node)

            return node

        except EvaluationError:
            raise
        except Exception as e:
            logger.exception(f"Exception during deferred node execution: {e}")
            with self.journal_lock:
                if root_enabled:
                    self.phase1_runtime.record_transition(parent_node=parent_node, child_node=node,
                        context=decision_context, generation_wall_time_sec=node.generation_wall_time_sec,
                        execution_wall_time_sec=time.perf_counter() - execution_started,
                        metric_maximize=bool(self.metric_maximize), reward=-1, failure_reason=type(e).__name__)
                evaluation.backpropagate(node=parent_node, value=0, add_to_tree=False)
                parent_node.sub_expected_child_count()
            raise e
