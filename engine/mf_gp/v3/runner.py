"""V3 scheduler: generation/probe/execution share the same bounded worker pool."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import copy
import json
from pathlib import Path
import sys
import time
from omegaconf import OmegaConf
from config import load_task_desc, prep_agent_workspace, save_run
from engine.agent_search import AgentSearch
from engine.executor import Interpreter
from engine.fixed_evaluation import finalize_submission, EvaluationError
from engine.search_node import Journal
from engine.parent_selectors.runtime import SearchExhausted
from engine.mf_gp.controller import PromotionRequested
from engine.mf_gp.execution import promote
from engine.mf_gp.recovery import execute
from engine.mf_gp.v3.controller import PreparedRequested
from engine.mf_gp.v3.probe import run_probe
from utils import serialize


def training_view(agent):
    """Bounded T-only table/assets, created once before worker dispatch."""
    import pandas as pd
    destination = Path(agent.cfg.log_dir) / "v3_probe_input"
    if (destination / "complete.json").exists():
        receipt = json.loads((destination / "complete.json").read_text())
        if receipt["protocol_id"] != agent.fixed_evaluation.protocol_id:
            raise EvaluationError("Probe input protocol changed")
        return destination
    destination.mkdir(exist_ok=True, parents=True)
    public = agent.fixed_evaluation.directory / "public"
    table = pd.read_csv(public / "train.csv", dtype={"sample_id": str})
    sample = table.sample(n=min(64, len(table)), random_state=int(agent.acfg.seed))
    sample.to_csv(destination / "train.csv", index=False)
    assets = agent.fixed_evaluation.directory / "assets.csv"
    if assets.exists():
        rows = pd.read_csv(assets, dtype=str)
        for row in rows.loc[rows.sample_id.isin(set(sample.sample_id))].to_dict("records"):
            target = destination / row["public_path"]
            if not target.resolve().is_relative_to(destination.resolve()):
                raise EvaluationError("Unsafe probe asset path")
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.is_symlink():
                target.symlink_to((public / row["public_path"]).resolve(strict=True))
    (destination / "complete.json").write_text(json.dumps(dict(
        protocol_id=agent.fixed_evaluation.protocol_id, rows=len(sample), full_training_rows=len(table),
        low_training_rows=len(pd.read_csv(agent.fixed_evaluation.half_view / "train.csv")))))
    return destination


def prepare_node(agent, node, view):
    runtime = agent.phase1_runtime
    s = runtime.study7_selector
    ticket = node.phase1_decision_id
    work = Path(agent.cfg.log_dir) / "v3_probes" / ticket
    with agent.journal_lock:
        e = s.prepared.entries[ticket]
        if e["state"] not in ("ready", "deferred"):
            return node
        previous = s.probes.get(ticket)
        receipt = None
        if previous and previous["state"] != "running":
            receipt = previous
        elif work.exists():
            receipt_path = work / "receipt.json"
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                if receipt["code_sha256"] != e["code_sha256"]:
                    raise EvaluationError("Probe source identity changed")
            else:
                receipt = dict(state="interrupted", metadata=None,
                    elapsed_seconds=min(120., max(0., time.time()-agent.start_time-(previous or {}).get("started_active", 0.))))
        remaining = agent.acfg.time_limit-(time.time()-agent.start_time)
        if receipt is None and remaining <= 0:
            receipt = dict(state="not_started", metadata=None, elapsed_seconds=0., reason="deadline")
        if receipt is None:
            s.probes[ticket] = dict(state="running", started_active=time.time()-agent.start_time)
            runtime._persist_root_state(agent)
    if receipt is None:
        receipt = run_probe(node.code, python=sys.executable, work=work, training_view=view,
                            timeout=min(120., remaining))
    with agent.journal_lock:
        s.probes[ticket] = receipt
        runtime._persist_root_state(agent)
        if receipt["state"] in ("syntax_error", "runtime_error"):
            node.is_buggy, node.is_valid = True, False
            node.analysis = "V3 bounded probe failed: " + json.dumps(receipt)
            # Preserve stderr for native debug prompts.
            stderr = work / "stderr.log"
            if stderr.exists():
                node.analysis += "\n" + stderr.read_text(errors="replace")[-16000:]
            context = getattr(node, "_phase1_decision_context", None)
            runtime.record_transition(parent_node=node.parent, child_node=node, context=context,
                generation_wall_time_sec=node.generation_wall_time_sec,
                execution_wall_time_sec=0., metric_maximize=bool(agent.metric_maximize), reward=-1,
                failure_reason="probe_"+receipt["state"])
    return node


def execute_prepared(agent, ticket, callback):
    runtime = agent.phase1_runtime
    s = runtime.study7_selector
    with agent.journal_lock:
        entry = s.prepared.entries[ticket]
        if entry["state"] != "executing":
            raise EvaluationError("Execution lacks committed v3 reservation")
        pending = runtime.root_draft_state["pending"][ticket]
        node = runtime._restore_pending_node(agent, pending)
        if node is None or node.code != entry["code"]:
            raise EvaluationError("Prepared executable differs from native checkpoint")
        context = runtime._thaw_context(agent, pending)
        mf = context.option.state_features["mf"]
        mf.update(fidelity=entry["fidelity"], estimated_action_seconds=entry.get("cost"),
                  cost_estimate_scope="execution_only")
        node.training_fraction = entry["fidelity"]
        node.phase1_selection_features["mf"] = copy.deepcopy(mf)
        node._phase1_decision_context = context
        node.pending_execution = True
        pending["option"]["state_features"]["mf"] = copy.deepcopy(mf)
        pending["node_snapshot"] = runtime._node_snapshot(node)
        runtime._persist_root_state(agent)
    return agent.execute_deferred_node(node, callback)


def worker_step(agent, callback, view, recovery_context=None):
    try:
        node = agent.step(None, callback, execute_immediately=False, recovery_context=recovery_context)
    except PreparedRequested as request:
        return execute_prepared(agent, request.ticket, callback)
    except PromotionRequested as request:
        return promote(agent, request.ticket, callback)
    if getattr(node, "pending_execution", False):
        return prepare_node(agent, node, view)
    return node


def run_search(cfg, logger):
    resume = bool(cfg.resume_run_dir)
    journal = serialize.load_json(cfg.log_dir / "journal.json", Journal) if resume else Journal()
    if not resume:
        prep_agent_workspace(cfg)
    agent = AgentSearch(journal.task_desc or load_task_desc(cfg), cfg, journal)
    from engine.mf_gp.evidence import update_best
    update_best(agent)
    agent.update_data_preview()
    view = training_view(agent)
    interpreter = Interpreter(cfg.workspace_dir, **OmegaConf.to_container(cfg.exec), cfg=cfg)
    def persist():
        with agent.journal_lock:
            agent.capture_resume_state()
            save_run(cfg, journal)
    persist()
    from engine.mf_recovery import install
    stop = install(agent, persist)
    def callback(code, node_id, *args, **kwargs):
        return execute(interpreter, cfg, code, node_id, *args, **kwargs)
    runtime = agent.phase1_runtime
    selector = runtime.study7_selector
    recovered = []
    for context in runtime.recovered_generation_contexts:
        recovered.append((worker_step, (agent, callback, view, context)))
    for node in runtime.recovered_deferred_nodes:
        ticket = node.phase1_decision_id
        if selector.prepared.entries[ticket]["state"] == "executing":
            recovered.append((execute_prepared, (agent, ticket, callback)))
        else:
            recovered.append((prepare_node, (agent, node, view)))
    for ticket in selector.pending_promotions:
        recovered.append((promote, (agent, ticket, callback)))
    workers = min(3, interpreter.max_parallel_run)
    pool = ThreadPoolExecutor(max_workers=workers)
    futures = set()
    def submit_next():
        if recovered:
            fn, args = recovered.pop(0)
            futures.add(pool.submit(fn, *args))
        else:
            futures.add(pool.submit(worker_step, agent, callback, view))
    normal_exit = False
    try:
        for _ in range(workers):
            submit_next()
        while futures:
            done, _ = wait(futures, timeout=1., return_when=FIRST_COMPLETED)
            for future in done:
                futures.remove(future)
                exhausted = False
                try:
                    future.result()
                except SearchExhausted as exc:
                    logger.info("V3 worker finished: %s", exc)
                    exhausted = True
                persist()
                if not exhausted and time.time()-agent.start_time < cfg.agent.time_limit:
                    submit_next()
        normal_exit = True
    except BaseException:
        stop.set()
        agent.fixed_evaluation.failed.set()
        interpreter.terminate_all_subprocesses()
        for future in futures:
            future.cancel()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        stop.set()
        if normal_exit:
            with agent.journal_lock:
                selector.prepared.expire_unexecuted()
                for ticket, e in selector.prepared.entries.items():
                    if e["state"] == "expired" and ticket in runtime.root_draft_state["pending"]:
                        e["native_reservation"] = runtime.root_draft_state["pending"].pop(ticket)
        persist()
    finalize_submission(cfg, journal)
    return agent
