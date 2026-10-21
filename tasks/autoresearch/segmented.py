"""Checkpoint bounded parallel work without changing agent/selector calls.

A committed boundary contains the search journal, decision ledger, memory and
remaining FIFO. Boundaries are committed after the current group drains, so no
worker can mutate a snapshot while it is copied. Interrupted work is retried
from that boundary; its raw
artifacts remain in the run directory and the interrupted segment archive.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import json
import logging
import os
from pathlib import Path
import shutil
import time

from engine.shared_drafts import atomic_json

logger = logging.getLogger("MLEvolve")


def state_items(run_root):
    run_root = Path(run_root)
    logs = run_root / "logs"
    for path in sorted(logs.iterdir()) if logs.exists() else []:
        if path.name == "phase1" or (path.is_file() and path.suffix in {".json", ".yaml", ".py"}):
            yield path.relative_to(run_root)
    for name in ("global_memory", "best_solution", "top_candidates"):
        path = run_root / "workspace" / name
        if path.exists():
            yield path.relative_to(run_root)


def copy_item(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, target)
    else:
        shutil.copy2(source, target)


def commit_checkpoint(run_root, checkpoint_root):
    """Publish only after all files have been copied; retain previous boundaries."""
    run_root, checkpoint_root = Path(run_root), Path(checkpoint_root)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    directory = checkpoint_root / f"boundary-{time.time_ns()}"
    directory.mkdir()
    items = list(state_items(run_root))
    for item in items:
        copy_item(run_root / item, directory / item)
    record = {"run_root": str(run_root), "directory": str(directory),
              "items": [str(x) for x in items], "committed_at": time.time()}
    atomic_json(checkpoint_root / "latest.json", record)
    return record


def restore_checkpoint(checkpoint_root):
    """Caller must verify that the previous executor has stopped first."""
    checkpoint_root = Path(checkpoint_root)
    record = json.loads((checkpoint_root / "latest.json").read_text())
    root, source = Path(record["run_root"]), Path(record["directory"])
    if not source.is_relative_to(checkpoint_root) or not (source / "logs/serial_queue.json").is_file():
        raise RuntimeError("Incomplete or invalid segment checkpoint")
    for name in record["items"]:
        item = Path(name)
        if item.is_absolute() or ".." in item.parts or not (source / item).exists():
            raise RuntimeError("Invalid or incomplete checkpoint path")
    archive = checkpoint_root / f"interrupted-state-{time.time_ns()}"
    archive.mkdir()
    for item in list(state_items(root)):
        target = archive / item
        target.parent.mkdir(parents=True, exist_ok=True)
        (root / item).rename(target)
    for name in record["items"]:
        item = Path(name)
        if item.is_absolute() or ".." in item.parts:
            raise RuntimeError("Invalid checkpoint path")
        copy_item(source / item, root / item)
    return {**record, "interrupted_state_archive": str(archive)}


def freeze_work(kind, node=None):
    from engine.parent_selectors.runtime import Phase1SelectorRuntime
    work = {"kind": kind, "node_id": node.id if node else None}
    if kind == "deferred":
        context = node._phase1_decision_context
        option = context.option
        work["node_snapshot"] = Phase1SelectorRuntime._node_snapshot(node)
        work["context"] = {"decision": asdict(context.decision), "option": {
            "parent_node_id": option.parent_node_id, "operator": option.operator,
            "reference_node_ids": list(option.reference_node_ids),
            "state_features": option.state_features, "preview": asdict(option.preview)}}
    return work


def thaw_work(agent, work):
    if work["kind"] == "deferred":
        runtime = agent.phase1_runtime
        if not hasattr(agent, "_segment_deferred_nodes"):
            agent._segment_deferred_nodes = {}
        node = agent._segment_deferred_nodes.get(work["node_id"])
        if node is None:
            node = next((n for n in runtime.recovered_deferred_nodes if n.id == work["node_id"]), None)
        if node is None:
            node = runtime._restore_pending_node(agent, work)
        agent._segment_deferred_nodes[node.id] = node
        node._phase1_decision_context = runtime._thaw_context(agent, work["context"])
        node.pending_execution = True
        branch = agent.branch_all_nodes.setdefault(node.branch_id, [])
        if all(n.id != node.id for n in branch):
            branch.append(node)
        agent.branch_successful_nodes.setdefault(node.branch_id, [])
        agent.branch_node_count.setdefault(node.branch_id, 0)
        agent.next_branch_id = max(agent.next_branch_id, node.branch_id + 1)
        node.parent.expected_child_count = max(node.parent.expected_child_count, len(node.parent.children))
        return node
    if work.get("node_id") is not None:
        return next(n for n in agent.journal.nodes if n.id == work["node_id"])
    return None


def advance_fifo(queue, kind, node, exhausted, completed, total):
    """Same FIFO/replenishment order as run.py with ThreadPoolExecutor(1)."""
    remaining = list(queue[1:])
    if kind == "generate":
        if node is not None and getattr(node, "pending_execution", False):
            remaining.append(freeze_work("deferred", node))
        if not remaining and completed < total:
            remaining.append(freeze_work("step"))
    elif completed + len(remaining) < total and (node is not None or not exhausted):
        remaining.append(freeze_work("step", node))
    return remaining


def run_segmented(agent, interpreter, cfg, exec_callback, persist):
    from engine.fixed_evaluation import EvaluationError
    from engine.parent_selectors.agent_selector import AgentSelectionError
    from engine.parent_selectors.runtime import SearchExhausted

    queue_path = Path(cfg.log_dir) / "serial_queue.json"
    checkpoint_root = Path(os.environ["MLEVOLVE_SEGMENT_CHECKPOINT"])
    pause_file = Path(os.environ["MLEVOLVE_SEGMENT_PAUSE_FILE"])
    total = int(cfg.agent.steps)
    workers = int(cfg.agent.search.parallel_search_num)
    if workers < 1:
        raise ValueError("parallel_search_num must be positive")
    if cfg.resume_run_dir:
        saved = json.loads(queue_path.read_text())
        queue = saved["queue"]
        if saved["protocol_id"] != agent.autoresearch.protocol_id:
            raise RuntimeError("Segment checkpoint task protocol differs")
        # Restore all deferred siblings before any selection can inspect them.
        for work in queue:
            if work["kind"] == "deferred":
                thaw_work(agent, work)
    else:
        queue = [freeze_work("generate") for _ in range(min(int(cfg.agent.initial_drafts), total))]
        if not queue and total:
            queue = [freeze_work("step") for _ in range(min(workers, total))]
    offset = float(os.environ.get("MLEVOLVE_SEGMENT_ACTIVE_OFFSET", "0"))
    segment_start = float(os.environ["MLEVOLVE_SEGMENT_STARTED"])
    agent.start_time = agent.search_start_time = segment_start - offset

    def completed_count():
        if agent.phase1_runtime.managed_decisions:
            return len(agent.phase1_runtime.transition_store.records())
        return max(len(agent.journal) - 1, 0)

    def checkpoint():
        persist()
        atomic_json(queue_path, {"version": 1, "protocol_id": agent.autoresearch.protocol_id,
                                "queue": queue, "completed": completed_count(),
                                "parallel_workers": workers})
        commit_checkpoint(Path(cfg.log_dir).parent, checkpoint_root)

    def execute(work, node):
        logger.info("[segment] Starting %s node=%s", work["kind"], work.get("node_id"))
        try:
            if work["kind"] == "generate":
                result = agent.step(exec_callback=exec_callback, node=None, execute_immediately=False)
            elif work["kind"] == "deferred":
                result = agent.execute_deferred_node(node, exec_callback)
            else:
                result = agent.step(exec_callback=exec_callback, node=node)
            return result, False
        except (AgentSelectionError, EvaluationError):
            raise
        except SearchExhausted:
            return None, True
        except Exception:
            logger.exception("[segment] Native action failed")
            return None, False

    # Protect the empty original-search journal and every partially generated
    # initial pool. Subsequent pauses never need to regenerate completed drafts.
    checkpoint()
    logger.info("[segment] Native FIFO active; %d experiment workers; %s queued actions", workers, len(queue))
    executor = ThreadPoolExecutor(max_workers=workers)
    try:
        while queue and completed_count() < total:
            if pause_file.exists() or time.time() - agent.start_time >= float(cfg.agent.time_limit):
                logger.info("[segment] Pausing at committed boundary with %d queued actions", len(queue))
                break
            # Keep initial generation/replay sequential and ordered, then execute
            # up to the configured worker count concurrently on the setting's GPU.
            if queue[0]["kind"] == "generate" or workers == 1:
                work = queue[0]
                result, exhausted = execute(work, thaw_work(agent, work))
                queue = advance_fifo(queue, work["kind"], result, exhausted, completed_count(), total)
            else:
                batch = queue[:workers]
                # Restore node/branch bookkeeping in the controller before threads run.
                pending = [(work, thaw_work(agent, work)) for work in batch]
                futures = [executor.submit(execute, work, node) for work, node in pending]
                outcomes = [future.result() for future in as_completed(futures)]
                queue = queue[len(batch):]
                for result, exhausted in outcomes:
                    if completed_count() + len(queue) < total and (result is not None or not exhausted):
                        queue.append(freeze_work("step", result))
            checkpoint()
            logger.info("[segment] Committed %d completed trials; %d queued actions", completed_count(), len(queue))
    except BaseException:
        logger.info("[segment] Interrupted group was not committed; resume the last complete boundary")
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    finally:
        interpreter.terminate_all_subprocesses()
    best = agent.journal.get_best_node()
    if best is not None and best.is_valid:
        agent.autoresearch.finalize(agent.journal)
    atomic_json(Path(cfg.log_dir) / "segment_outcome.json", {
        "state": "paused" if queue and completed_count() < total and time.time() - agent.start_time < float(cfg.agent.time_limit) else "finished",
        "completed": completed_count(), "queued_actions": len(queue), "at": time.time()})


# Compatibility for callers of the historical single-worker entry point.
run_serial = run_segmented
