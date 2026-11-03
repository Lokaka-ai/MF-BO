"""Independent per-(program,fidelity) execution budgets across released jobs."""

from pathlib import Path
import json
import time
import os
import uuid
from engine.competition_checkpoint import atomic_json
from engine.executor import ExecutionResult
from engine.mf_gp.controller import code_hash


def execute(interpreter, cfg, code, node_id, *args, **kwargs):
    context = kwargs.get("evaluation_context") or {}
    fraction = context.get("training_fraction")
    from engine.mf_gp.settings import low_fraction
    if fraction not in (low_fraction(cfg), 1.0):
        raise ValueError("Recovery execution requires explicit fidelity")
    path = Path(cfg.log_dir) / "mf_execution" / f"{node_id}__{fraction}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = json.loads(path.read_text()) if path.exists() else {}
    if previous and (
        previous["code_sha256"] != code_hash(code) or previous["state"] == "running"
    ):
        raise RuntimeError(
            "MF executable changed or prior allocation has not been released"
        )
    used = float(previous.get("elapsed_seconds", 0.0))
    remaining = max(0.0, float(cfg.exec.timeout) - used)
    attempts = previous.get("attempts", [])
    attempt = dict(
        segment_id=uuid.uuid4().hex,
        attempt_id=context.get("attempt_id"),
        protocol_id=context.get("protocol_id"),
        job_id=os.environ.get("SLURM_JOB_ID"),
        estimated_action_seconds=context.get("estimated_action_seconds"),
        cost_estimate_scope=context.get("cost_estimate_scope"),
        started_epoch=time.time(), state="running", elapsed_seconds=None,
        measurement="candidate_process_wall_seconds",
    )
    attempts.append(attempt)
    record = dict(
        code_sha256=code_hash(code),
        program_id=node_id,
        fidelity=fraction,
        elapsed_seconds=used,
        started_epoch=time.time(),
        state="running",
        attempts=attempts,
        unsegmented_elapsed_seconds=previous.get("unsegmented_elapsed_seconds", used if len(attempts) == 1 else 0.0),
    )
    atomic_json(path, record)
    if remaining <= 0:
        result = ExecutionResult(
            ["Cumulative execution cap exhausted"], 0.0, "TimeoutError", {}, []
        )
    else:
        kwargs["execution_timeout"] = min(
            remaining, kwargs.get("execution_timeout", remaining)
        )
        result = interpreter.run(code, node_id, *args, **kwargs)
    attempt.update(
        state="timeout" if result.exc_type == "TimeoutError" else ("failed" if result.exc_type else "completed"),
        elapsed_seconds=float(result.exec_time), ended_epoch=time.time(),
        exception_type=result.exc_type,
        completed_runtime_observation=result.exc_type is None,
        runtime_lower_bound=result.exc_type == "TimeoutError",
    )
    record.update(state="completed", elapsed_seconds=used + result.exec_time)
    atomic_json(path, record)
    return result
