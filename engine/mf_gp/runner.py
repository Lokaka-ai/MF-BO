"""Dedicated adaptive-MF runner; existing staged runner remains unchanged."""

from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
import json
import time
from omegaconf import OmegaConf
from config import load_task_desc, prep_agent_workspace, save_run
from engine.agent_search import AgentSearch
from engine.executor import Interpreter
from engine.fixed_evaluation import finalize_submission
from engine.search_node import Journal
from engine.parent_selectors.runtime import SearchExhausted
from engine.mf_gp.controller import PromotionRequested
from engine.mf_gp.execution import promote
from engine.mf_gp.recovery import execute
from utils import serialize
from utils.seed import set_global_seed
from utils.logging_config import setup_logging


def run(cfg):
    if not cfg.evaluation.enabled or cfg.audited_bo != "redesign":
        raise ValueError("MF-GP requires the fixed evaluator and redesigned evolution")
    set_global_seed(cfg.agent.seed)
    logger = setup_logging(cfg)
    if getattr(cfg, "run_result_file", None):
        path = Path(cfg.run_result_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                dict(
                    log_dir=str(cfg.log_dir),
                    workspace_dir=str(cfg.workspace_dir),
                    fixed_evaluation=True,
                )
            )
        )
    if getattr(cfg.evaluation, "mf_gp_policy_version", 1) == 3:
        from engine.mf_gp.v3.runner import run_search
        return run_search(cfg, logger)
    resume = bool(cfg.resume_run_dir)
    journal = (
        serialize.load_json(cfg.log_dir / "journal.json", Journal)
        if resume
        else Journal()
    )
    if not resume:
        prep_agent_workspace(cfg)
    agent = AgentSearch(journal.task_desc or load_task_desc(cfg), cfg, journal)
    from engine.mf_gp.evidence import update_best

    update_best(agent)
    agent.update_data_preview()
    interpreter = Interpreter(
        cfg.workspace_dir, **OmegaConf.to_container(cfg.exec), cfg=cfg
    )

    def persist():
        with agent.journal_lock:
            agent.capture_resume_state()
            save_run(cfg, journal)

    persist()
    from engine.mf_recovery import install

    stop = install(agent, persist)

    def callback(code, node_id, *args, **kwargs):
        return execute(interpreter, cfg, code, node_id, *args, **kwargs)

    def step():
        try:
            return agent.step(None, callback)
        except PromotionRequested as request:
            return promote(agent, request.ticket, callback)

    pool = ThreadPoolExecutor(max_workers=interpreter.max_parallel_run)
    futures = set()
    try:
        # Native recovered generation keeps its decision, code and fidelity.
        for context in agent.phase1_runtime.recovered_generation_contexts:
            futures.add(
                pool.submit(agent.step, None, callback, recovery_context=context)
            )
        for node in agent.phase1_runtime.recovered_deferred_nodes:
            futures.add(pool.submit(agent.execute_deferred_node, node, callback))
        for ticket in list(agent.phase1_runtime.study7_selector.pending_promotions):
            futures.add(pool.submit(promote, agent, ticket, callback))
        # Same three-draft initialization: code generation sequentially, then
        # execution in the existing worker budget. Never start extra drafts.
        state = agent.phase1_runtime.root_draft_state
        with agent.journal_lock:
            reserved_initial = sum(
                p["initial"]
                and p["status"] == "reserved"
                and not p.get("recovery_already_submitted")
                for p in state["pending"].values()
            )
            missing_initial = max(
                0, state["initial_goal"] - state["initial_submitted"] - reserved_initial
            )
        for _ in range(missing_initial):
            node = agent.step(None, callback, execute_immediately=False)
            if getattr(node, "pending_execution", False):
                futures.add(pool.submit(agent.execute_deferred_node, node, callback))
        while len(futures) < interpreter.max_parallel_run:
            futures.add(pool.submit(step))
        while futures:
            done, _ = wait(futures, timeout=1.0, return_when=FIRST_COMPLETED)
            for future in done:
                futures.remove(future)
                exhausted = False
                try:
                    future.result()
                except SearchExhausted as exc:
                    logger.info("MF worker finished: %s", exc)
                    exhausted = True
                persist()
                if (
                    not exhausted
                    and time.time() - agent.start_time < cfg.agent.time_limit
                ):
                    futures.add(pool.submit(step))
    except BaseException:
        # Completed data remain admitted; interrupted work remains pending.
        stop.set()
        agent.fixed_evaluation.failed.set()
        interpreter.terminate_all_subprocesses()
        for future in futures:
            future.cancel()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        stop.set()
        persist()
    finalize_submission(cfg, journal)
    return agent
