"""Opt-in checkpoints for preemptible Acq7 competition allocations.

The native asynchronous search loop and selector stay in charge. A checkpoint
records completed observations and in-flight decisions under the journal lock;
the process then exits without parsing killed training as a failed candidate.
The external Slurm supervisor releases all children before permitting recovery.
"""
from __future__ import annotations

import json
import hashlib
import logging
import os
from pathlib import Path
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar


_generation = ContextVar('competition_generation', default=None)
logger = logging.getLogger(__name__)


def capture_data_preview(agent):
    """Keep the once-generated input preview across competition leases."""
    if enabled(agent.cfg):
        agent.journal.study7_search_state['competition_data_preview'] = agent.data_preview


def restore_data_preview(agent):
    """Recover old checkpoints from the exact preview recorded in a draft."""
    if not enabled(agent.cfg):
        return
    if agent.cfg.evaluation.enabled:
        return  # Fixed evaluation derives its preview from the current fidelity view.
    saved = agent.journal.study7_search_state or {}
    preview = saved.get('competition_data_preview')
    if isinstance(preview, str):
        agent.data_preview = preview
        return
    prefix = "Let me approach this systematically.\nFirst, I'll examine the dataset:\n"
    for node in agent.journal.nodes:
        if node.stage != 'draft' or not node.prompt_input:
            continue
        try:
            prompt = json.loads(node.prompt_input)
        except (TypeError, json.JSONDecodeError):
            continue
        assistant = prompt.get('assistant') if isinstance(prompt, dict) else None
        if isinstance(assistant, str) and assistant.startswith(prefix):
            agent.data_preview = assistant[len(prefix):]
            logger.info("[resume] restored the original data preview from draft %s", node.id)
            return


@contextmanager
def track_generation(agent, context):
    """Associate the native counter charge with its frozen decision, per thread."""
    if not enabled(agent.cfg):
        yield
        return
    token = _generation.set((agent, context.decision.decision_id))
    try:
        yield
    finally:
        _generation.reset(token)


def charge_child_count(node):
    """Return True when this opt-in ledger handles the native counter call."""
    active = _generation.get()
    if active is None:
        return False
    agent, decision_id = active
    with agent.journal_lock:
        pending = agent.phase1_runtime.root_draft_state['pending'][decision_id]
        assert pending['option']['parent_node_id'] == node.id
        if not pending.get('child_count_charged', False):
            with node.child_count_lock:
                node.expected_child_count += 1
            pending['child_count_charged'] = True
    return True


def restore_generation_charges(agent):
    """Migrate unambiguous v1 pending counters; new ledgers persist exact charges."""
    if not enabled(agent.cfg):
        return
    runtime = agent.phase1_runtime
    # A decision can be persisted after the periodic full journal. The runtime
    # admits that newer ledger, so its parent counters must also win over the
    # older study7_search_state restored from journal.json. This is especially
    # important for generated children, which resume execution without passing
    # through track_generation a second time.
    snapshot = json.loads(runtime.root_draft_state['journal_snapshot'])
    counts = {node['id']: node['expected_child_count'] for node in snapshot['nodes']}
    for node in agent.journal.nodes:
        if node.id in counts:
            node.expected_child_count = counts[node.id]
    groups = {}
    for context in runtime.recovered_generation_contexts:
        groups.setdefault(context.option.parent_node_id, []).append(context)
    for parent_id, contexts in groups.items():
        parent = contexts[0].option.parent_node
        pending = [runtime.root_draft_state['pending'][c.decision.decision_id] for c in contexts]
        unknown = [p for p in pending if 'child_count_charged' not in p]
        if not unknown:
            continue
        known_charges = sum(p.get('child_count_charged', False) for p in pending)
        residual = parent.expected_child_count - len(parent.children) - known_charges
        if residual not in {0, len(unknown)}:
            raise RuntimeError(f'Ambiguous legacy generation counter for parent {parent_id}')
        for p in unknown:
            p['child_count_charged'] = bool(residual)
        runtime.root_draft_state.setdefault('legacy_generation_counter_migrations', []).append({
            'parent_id': parent_id, 'expected_child_count': parent.expected_child_count,
            'materialized_children': len(parent.children), 'pending_generations': len(unknown),
            'charged_generations': residual})


def legacy_enabled(cfg):
    return (os.environ.get("MLEVOLVE_COMPETITION_CHECKPOINT") == "1"
            and getattr(cfg, "task_type", None) == "competition"
            and getattr(getattr(cfg, "parent_selector", None), "type", None) == "legacy_original")


def enabled(cfg):
    from engine.mf_recovery import enabled as mf_enabled
    if mf_enabled(cfg):
        return True
    if os.environ.get("MLEVOLVE_COMPETITION_CHECKPOINT") != "1":
        return False
    aq = cfg.parent_selector.type == "acq7_gp4" and cfg.study7.method_id in {"AQ04", "AQ05"}
    if cfg.task_type != "competition" or not (aq or legacy_enabled(cfg)) or cfg.evaluation.enabled:
        raise ValueError("Competition checkpoint mode requires native AQ04/AQ05 or original MCGS")
    return True


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False))
    temporary.replace(path)


def execute(interpreter, cfg, code, node_id, *args, **kwargs):
    """Retain completed training output and the 9h execution cap across leases."""
    from engine.mf_recovery import enabled as mf_enabled, execute as mf_execute
    if mf_enabled(cfg):
        return mf_execute(interpreter, cfg, code, node_id, *args, **kwargs)
    from engine.executor import ExecutionResult
    path = Path(cfg.log_dir) / 'candidate_execution' / f'{node_id}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(code.encode()).hexdigest()
    previous = json.loads(path.read_text()) if path.exists() else {}
    if previous:
        if previous['code_sha256'] != digest or previous['state'] == 'running':
            raise RuntimeError('Candidate identity/release has not been reconciled')
        if previous['state'] == 'completed':
            return ExecutionResult.from_dict(previous['result'])
    used = float(previous.get('elapsed_seconds', 0.))
    remaining = max(0., float(cfg.exec.timeout) - used)
    record = {**previous, 'node_id': str(node_id), 'code_sha256': digest,
              'elapsed_seconds': used, 'owner_pid': os.getpid(),
              'started_epoch': time.time(), 'state': 'running',
              'attempts': int(previous.get('attempts', 0)) + 1}
    atomic_json(path, record)
    if remaining == 0:
        result = ExecutionResult(['Execution time: TimeoutError: cumulative execution limit exhausted'],
                                 0., 'TimeoutError', {}, [])
    else:
        result = interpreter.run(code, node_id, *args, execution_timeout=remaining, **kwargs)
    record.update(state='completed', finished_epoch=time.time(),
                  elapsed_seconds=used + result.exec_time, result=result.to_dict())
    atomic_json(path, record)
    return result


def checkpoint_once(agent, persist, *, pause=False):
    """Never race a committed transition; return an auditable active clock."""
    with agent.journal_lock:
        persist()
        stamp = time.time()
        record = {"version": 1, "pid": os.getpid(), "updated_epoch": stamp,
                  "agent_start_epoch": agent.start_time,
                  "active_elapsed_seconds": max(0.0, stamp - agent.start_time),
                  "paused": pause,
                  "completed_transitions": len(agent.phase1_runtime.transition_store.records()),
                  "pending_decisions": len(agent.phase1_runtime.root_draft_state["pending"])}
        atomic_json(Path(agent.cfg.log_dir) / "competition_heartbeat.json", record)
    return record


def install(agent, persist):
    """Start only after AgentSearch restored its journal/RNG/active budget."""
    if not enabled(agent.cfg):
        return None
    from engine.mf_recovery import enabled as mf_enabled, install as mf_install
    if mf_enabled(agent.cfg):
        return mf_install(agent, persist)
    # Acq7 time checks must also count initial draft generation.
    agent.search_start_time = agent.start_time
    pause_file = Path(os.environ["MLEVOLVE_COMPETITION_PAUSE_FILE"])
    lease_end = float(os.environ["MLEVOLVE_COMPETITION_LEASE_END"])
    stop = threading.Event()
    checkpoint_once(agent, persist)

    def heartbeat():
        last_save = time.monotonic()
        while not stop.wait(1.0):
            budget_done = time.time() - agent.start_time >= agent.acfg.time_limit
            pause = pause_file.exists() or time.time() >= lease_end or budget_done
            if pause or time.monotonic() - last_save >= 15:
                try:
                    record = checkpoint_once(agent, persist, pause=pause)
                except Exception:
                    # Never publish a heartbeat for an incomplete save, and do
                    # not permanently lose checkpointing after one failed copy.
                    logger.exception("Competition checkpoint failed; retrying")
                    continue
                last_save = time.monotonic()
                if pause:
                    record["reason"] = "budget_complete" if budget_done else "allocation_boundary"
                    atomic_json(Path(agent.cfg.log_dir) / "competition_pause.json", record)
                    # Exit all agent threads first. The supervisor then kills
                    # only this process group and Slurm releases its cgroup.
                    # No exception/parse path may turn interruption into data.
                    os._exit(75)

    thread = threading.Thread(target=heartbeat, name="competition-checkpoint", daemon=True)
    thread.start()
    return stop
