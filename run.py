import atexit
import os
import logging
import sys
import shutil
import time
import threading
import json
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from engine.agent_search import AgentSearch as Agent
from engine.executor import Interpreter
from engine.search_node import Journal
from engine.parent_selectors.runtime import SearchExhausted
from engine.parent_selectors.agent_selector import AgentSelectionError
from engine.fixed_evaluation import EvaluationError, enabled as fixed_enabled, finalize_submission
from omegaconf import OmegaConf
from rich.status import Status
from config import load_task_desc, prep_agent_workspace, save_run, load_cfg
from utils.visualization import journal_to_string_tree
from utils.seed import set_global_seed
from engine.coldstart import build_guidance_description
from utils.logging_config import setup_logging
from utils import serialize
from engine.competition_checkpoint import enabled as competition_checkpoint_enabled, install as install_competition_checkpoint
import torch



def run():
    cfg = load_cfg()
    if getattr(cfg.evaluation, "multifidelity_mode", None) == "mf_gp":
        from engine.mf_gp.runner import run as run_mf_gp
        return run_mf_gp(cfg)
    run_info = {"log_dir": str(cfg.log_dir), "workspace_dir": str(cfg.workspace_dir), "fixed_evaluation": fixed_enabled(cfg)}
    if cfg.run_result_file:
        result_path = Path(cfg.run_result_file)
        result_temp = result_path.with_suffix(".tmp")
        result_temp.write_text(json.dumps(run_info))
        result_temp.replace(result_path)
    is_resume = bool(cfg.resume_run_dir)
    if cfg.torch_hub_dir:
        torch.hub.set_dir(cfg.torch_hub_dir)
    set_global_seed(cfg.agent.seed)
    logger = setup_logging(cfg)
    logger.info(f'Starting run "{cfg.exp_name}"')

    if is_resume:
        journal = serialize.load_json(cfg.log_dir / "journal.json", Journal)
        if len(journal) < 2 and not os.environ.get("MLEVOLVE_SEGMENTED_RUN") and not competition_checkpoint_enabled(cfg) and not (getattr(cfg.parent_selector, "include_root_draft", False) or (cfg.bo_parent_selector.model == "gp_matern32" or cfg.parent_selector.type == "agent_selector")):
            raise ValueError("Resumed journal must contain at least one completed step")
        task_desc = journal.task_desc or load_task_desc(cfg)
    else:
        journal = Journal()
        task_desc = load_task_desc(cfg)

    if cfg.coldstart.use_coldstart and not is_resume:
        logger.info("Loading guidance from knowledge base")
        cfg.coldstart.description = build_guidance_description(cfg)
        logger.info(f"Guidance description: {cfg.coldstart.description}")

    if is_resume:
        logger.info("Resuming existing workspace at %s", cfg.resume_run_dir)
    else:
        with Status("Preparing agent workspace (copying and extracting files) ..."):
            prep_agent_workspace(cfg)

    global_step = 0

    def cleanup():
        if not is_resume and global_step == 0 and not fixed_enabled(cfg):
            shutil.rmtree(cfg.workspace_dir)

    atexit.register(cleanup)

    agent = Agent(
        task_desc=task_desc,
        cfg=cfg,
        journal=journal,
    )
    multifidelity = agent.fixed_evaluation if (agent.fixed_evaluation and agent.fixed_evaluation.multifidelity_mode) else None

    def switch_due():
        return (multifidelity is not None and multifidelity.multifidelity_phase == "half"
                and time.time() - agent.start_time >= .40 * float(cfg.agent.time_limit))

    def switch_if_drained(futures):
        if not switch_due() or futures:
            return False
        if agent.phase1_runtime.root_draft_state["pending"]:
            raise EvaluationError("Cannot switch training view with pending search decisions")
        multifidelity.switch_to_full()
        if multifidelity.multifidelity_mode == "reset":
            selector = agent.phase1_runtime.study7_selector
            selector.full_training_only = True
            selector.observe(agent.phase1_runtime.transition_store.records())
        agent.update_data_preview()
        persist_run_state()
        logger.info("Training view switched to full T after draining cheap work; mode=%s", multifidelity.multifidelity_mode)
        return True

    def persist_run_state():
        with agent.journal_lock:
            agent.capture_resume_state()
            save_run(cfg, journal)
            if agent.phase1_runtime.managed_decisions and not agent.phase1_runtime.legacy_checkpoint:
                return len(agent.phase1_runtime.transition_store.records())
            return len(journal) - 1

    if is_resume or agent.phase1_runtime.managed_decisions or fixed_enabled(cfg):
        persist_run_state()
        logger.info("Persisted admitted resume state before continuing search")
    competition_checkpoint = install_competition_checkpoint(agent, persist_run_state)
    if fixed_enabled(cfg) and time.time() - agent.start_time >= float(cfg.agent.time_limit):
        if competition_checkpoint is not None:
            competition_checkpoint.set()
        finalize_submission(cfg, journal)
        return

    interpreter_class = Interpreter
    if getattr(agent, "alphaevolve", None):
        from tasks.alphaevolve.adapter import AlphaEvolveInterpreter
        interpreter_class = AlphaEvolveInterpreter
    elif agent.autoresearch:
        from tasks.autoresearch.adapter import AutoresearchInterpreter
        interpreter_class = AutoresearchInterpreter
    interpreter = interpreter_class(
        cfg.workspace_dir, **OmegaConf.to_container(cfg.exec), cfg=cfg  # type: ignore
    )

    global_step = len(journal)
    status = Status("[green]Generating code...")

    def exec_callback(*args, **kwargs):
        status.update("[magenta]Executing code...")
        if competition_checkpoint is not None:
            from engine.competition_checkpoint import execute
            res = execute(interpreter, cfg, *args, **kwargs)
        else:
            res = interpreter.run(*args, **kwargs)
        status.update("[green]Generating code...")
        return res

    def step_task(node=None):
        if node:
            logger.info(f"[step_task] Processing node: {node.id}")
        else:
            logger.info(f"[step_task] Processing virtual root node.")
        return agent.step(exec_callback=exec_callback, node=node)

    max_workers = interpreter.max_parallel_run
    total_steps = cfg.agent.steps
    initial_draft_count = cfg.agent.initial_drafts
    logger.info(f"🚀 ThreadPool max_workers set to: {max_workers} (matching interpreter capacity)")
    logger.info(f"🎯 Initial draft count: {initial_draft_count} (will be executed sequentially for diversity)")

    if os.environ.get("MLEVOLVE_SEGMENTED_RUN"):
        from tasks.autoresearch.segmented import run_segmented
        if not agent.autoresearch:
            raise ValueError("Segmented orchestration requires autoresearch")
        run_segmented(agent, interpreter, cfg, exec_callback, persist_run_state)
        return

    lock = threading.Lock()
    completed = max(len(journal) - 1, 0)

    root_enabled = agent.phase1_runtime.managed_decisions
    pending_draft_nodes = list(agent.phase1_runtime.recovered_deferred_nodes)
    recovered_generation = list(agent.phase1_runtime.recovered_generation_contexts)
    # An interrupted initial generation retains its original decision identity.
    for context in list(recovered_generation):
        pending = agent.phase1_runtime.root_draft_state["pending"][context.decision.decision_id]
        if pending["initial"]:
            node = agent.step(None, exec_callback, execute_immediately=False, recovery_context=context)
            if getattr(node, "pending_execution", False):
                pending_draft_nodes.append(node)
            recovered_generation.remove(context)
    if root_enabled:
        initial_draft_count = (agent.phase1_runtime.root_draft_state["initial_goal"]
                               - agent.phase1_runtime.root_draft_state["initial_submitted"])
        completed = len(agent.phase1_runtime.transition_store.records())
        if agent.phase1_runtime.legacy_checkpoint:
            completed = len(journal) - 1
    if (not is_resume or root_enabled) and initial_draft_count > 0 and total_steps > 0:
        logger.info(f"📝 Phase 1: Sequential draft generation (code only, {initial_draft_count} drafts)")

        def step_task_generate_only():
            logger.info(f"[step_task_generate_only] Generating draft from virtual root")
            return agent.step(exec_callback=exec_callback, node=None, execute_immediately=False)

        for draft_idx in range(min(initial_draft_count, total_steps)):
            try:
                if switch_due():
                    logger.info("Cheap-stage deadline reached during draft generation; draining generated drafts")
                    break
                logger.info(f"🔨 Generating draft {draft_idx + 1}/{min(initial_draft_count, total_steps)} (code only)")
                cur_node = step_task_generate_only()
                if not root_enabled or getattr(cur_node, "pending_execution", False):
                    pending_draft_nodes.append(cur_node)
                logger.info(f"✅ Draft {draft_idx + 1} code generated: node.id={cur_node.id}, added to virtual_root.children")

            except EvaluationError:
                persist_run_state()
                raise
            except Exception as e:
                logger.exception(f"❌ Exception during draft {draft_idx + 1} generation: {e}")

        logger.info(f"✅ Phase 1 complete: {len(pending_draft_nodes)} draft codes generated")

    if root_enabled:
        completed = persist_run_state()

    if pending_draft_nodes or recovered_generation or completed < total_steps:
        logger.info(f"🚀 Phase 2: Pipelined parallel execution")
        logger.info(f"   - Pending draft executions: {len(pending_draft_nodes)}")
        logger.info(f"   - Remaining steps: {total_steps - completed}")

        def execute_draft_node(node):
            try:
                executed_node = agent.execute_deferred_node(node, exec_callback)
                logger.info(f"✅ Draft node {executed_node.id} executed: metric={executed_node.metric.value}")
                return executed_node
            except EvaluationError:
                raise
            except Exception as e:
                logger.exception(f"❌ Exception during draft node {node.id} execution: {e}")
                return None

        executor = ThreadPoolExecutor(max_workers=max_workers)
        interrupted = False
        try:
            futures = set()
            ready_nodes = []
            if agent.phase1_runtime.legacy_checkpoint:
                by_id = {node.id: node for node in journal.nodes}
                ready_nodes = [by_id[node_id] for node_id in
                    agent.phase1_runtime.root_draft_state.get("ready_continuations", {}).values()]
            for i, node in enumerate(pending_draft_nodes):
                futures.add(executor.submit(execute_draft_node, node))
                logger.info(f"📤 Submitted draft execution: {node.id}")
                if i < len(pending_draft_nodes) - 1:
                    time.sleep(10)
                    logger.info(f"⏱️  Waiting 10s before next draft to stagger initialization...")

            for context in recovered_generation:
                futures.add(executor.submit(agent.step, None, exec_callback, recovery_context=context))
            initial_step_tasks = (0 if switch_due() else min(max_workers, total_steps - completed) - len(pending_draft_nodes) - len(recovered_generation))
            if initial_step_tasks > 0:
                for _ in range(initial_step_tasks):
                    next_node = ready_nodes.pop() if ready_nodes else None
                    futures.add(executor.submit(step_task, next_node))
                    logger.info(f"📤 Submitted initial step_task to fill thread pool")

            while (completed < total_steps or (multifidelity and futures)) and (futures or not root_enabled or (multifidelity and multifidelity.multifidelity_phase == "half")):
                if switch_if_drained(futures):
                    if completed < total_steps:
                        remaining_drafts = (agent.phase1_runtime.root_draft_state["initial_goal"]
                                            - agent.phase1_runtime.root_draft_state["initial_submitted"])
                        if remaining_drafts:
                            for _ in range(min(remaining_drafts, total_steps - completed)):
                                node = agent.step(exec_callback=exec_callback, node=None, execute_immediately=False)
                                if getattr(node, "pending_execution", False):
                                    futures.add(executor.submit(execute_draft_node, node))
                        else:
                            for _ in range(min(max_workers, total_steps - completed)):
                                next_node = ready_nodes.pop() if ready_nodes else None
                                futures.add(executor.submit(step_task, next_node))
                    continue
                if not futures:
                    if root_enabled and multifidelity:
                        raise RuntimeError("Multi-fidelity search has no runnable work before the training switch; stopping instead of idling until the deadline")
                    time.sleep(1)
                    continue
                done, _ = wait(futures, return_when=FIRST_COMPLETED, timeout=1.0)

                if not done:
                    continue  # timeout, no completed futures, retry (allows SIGINT handling)

                for fut in done:
                    futures.remove(fut)
                    exhausted = False
                    try:
                        cur_node = fut.result()
                        if cur_node:
                            logger.info(f"✅ Task completed: node_id={cur_node.id}, step={cur_node.step}, is_buggy={cur_node.is_buggy}, metric={cur_node.metric.value if cur_node.metric else 'N/A'}")
                        else:
                            logger.warning(f"⚠️  Task returned None (execution failed)")
                    except SearchExhausted as e:
                        logger.info("Search worker finished: %s", e)
                        cur_node = None
                        exhausted = True
                    except (AgentSelectionError, EvaluationError):
                        logger.exception("Agent selector failed; stopping this run without an algorithm fallback")
                        with lock:
                            persist_run_state()
                        for pending in futures:
                            pending.cancel()
                        interpreter.terminate_all_subprocesses()
                        raise
                    except Exception as e:
                        logger.exception(f"❌ Exception during task execution: {e}")
                        cur_node = None

                    with lock:
                        completed = persist_run_state()
                        if completed == total_steps:
                            logger.info(journal_to_string_tree(journal))

                    if cur_node:
                        ready_nodes.append(cur_node)
                    # A concurrently exhausted global selector must not discard
                    # the continuation returned by a completed deferred draft.
                    if completed + len(futures) < total_steps and not switch_due() and (ready_nodes or not exhausted):
                        next_node = ready_nodes.pop() if ready_nodes else cur_node
                        futures.add(executor.submit(step_task, next_node))
                        logger.info(f"📤 Submitted next task based on node {next_node.id if next_node else 'None'}")
                    logger.info(f"📊 Progress: {completed}/{total_steps} steps completed, {len(futures)} tasks running")
        except KeyboardInterrupt:
            interrupted = True
            logger.info("KeyboardInterrupt received, terminating subprocesses and shutting down...")
            interpreter.terminate_all_subprocesses()
            with lock:
                persist_run_state()
                if getattr(agent, "alphaevolve", None):
                    best = journal.get_best_node()
                    if best is not None and best.is_valid:
                        agent.alphaevolve.finalize(journal)
                elif agent.autoresearch:
                    best = journal.get_best_node()
                    if best is not None and best.is_valid:
                        agent.autoresearch.finalize(journal)
            logger.info("Persisted the latest completed journal before shutdown")
            executor.shutdown(wait=False, cancel_futures=True) if sys.version_info >= (3, 9) else executor.shutdown(wait=False)
            raise
        finally:
            if sys.exc_info()[0] is not None and competition_checkpoint is not None:
                competition_checkpoint.set()
            if not interrupted:
                executor.shutdown(wait=True)
    else:
        if is_resume:
            logger.info(
                "✅ Resume target already satisfied "
                f"(completed={completed}, total_steps={total_steps})"
            )
        else:
            logger.info(f"✅ All steps completed in Phase 1 (total_steps={total_steps} <= initial_draft_count={initial_draft_count})")

    interpreter.cleanup_session(-1)
    persist_run_state()
    if competition_checkpoint is not None:
        competition_checkpoint.set()
        # Allocation boundaries never invoke fusion/grading. The supervisor
        # finalizes once, after the cumulative budget or step cap is reached.
        if not fixed_enabled(cfg):
            return
    if getattr(agent, "alphaevolve", None):
        agent.alphaevolve.finalize(journal)
    elif agent.autoresearch:
        agent.autoresearch.finalize(journal)
    elif fixed_enabled(cfg):
        finalize_submission(cfg, journal)
    else:
        from utils.submission_fusion_utils import ensemble
        ensemble(SimpleNamespace(
            task_id=cfg.exp_id,
            tag_path=str(Path(__file__).resolve().parent / "engine/coldstart/competition_tag_classified.json"),
            use_llm_selection=False,
        ), run_info=run_info)


if __name__ == "__main__":
    run()
