"""Independent per-(program,fidelity) execution budgets across released jobs."""

from pathlib import Path
import json
import time
from engine.competition_checkpoint import atomic_json
from engine.executor import ExecutionResult
from engine.mf_gp.controller import code_hash


def execute(interpreter, cfg, code, node_id, *args, **kwargs):
    context = kwargs.get("evaluation_context") or {}
    fraction = context.get("training_fraction")
    if fraction not in (0.5, 1.0):
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
    record = dict(
        code_sha256=code_hash(code),
        program_id=node_id,
        fidelity=fraction,
        elapsed_seconds=used,
        started_epoch=time.time(),
        state="running",
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
    record.update(state="completed", elapsed_seconds=used + result.exec_time)
    atomic_json(path, record)
    return result
