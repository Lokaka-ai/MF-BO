"""Periodic MF snapshots and interrupted-program accounting; no model checkpoints."""
import hashlib
import json
import os
from pathlib import Path
import signal
import threading
import time
import logging


def enabled(cfg):
    return (os.environ.get("MLEVOLVE_MF_RECOVERY") == "1"
            and cfg.evaluation.enabled
            and cfg.evaluation.multifidelity_mode in {"carry_over", "reset", "mf_gp"})


def install(agent, persist):
    from engine.competition_checkpoint import checkpoint_once
    pause = threading.Event()
    prior = {}
    class Stop(threading.Event):
        def set(self):
            super().set()
            if threading.current_thread() is threading.main_thread():
                for sig, handler in prior.items():
                    signal.signal(sig, handler)
    stop = Stop()
    def interrupt(signum, frame):
        pause.set()
    for sig in (signal.SIGUSR1, signal.SIGTERM):
        prior[sig] = signal.signal(sig, interrupt)
    checkpoint_once(agent, persist)
    def heartbeat():
        saved = time.monotonic()
        while not stop.wait(1):
            if pause.is_set() or time.monotonic() - saved >= 15:
                try:
                    checkpoint_once(agent, persist, pause=pause.is_set())
                except Exception:
                    logging.getLogger(__name__).exception("MF checkpoint failed; retrying")
                    continue
                saved = time.monotonic()
                if pause.is_set():
                    # Supervisor/Slurm releases subprocesses; no killed attempt
                    # is parsed into a synthetic failed BO observation.
                    os._exit(75)
    threading.Thread(target=heartbeat, name="mf-checkpoint", daemon=True).start()
    return stop


def execute(interpreter, cfg, code, node_id, *args, **kwargs):
    from engine.competition_checkpoint import atomic_json
    from engine.executor import ExecutionResult
    path = Path(cfg.log_dir) / "mf_execution" / f"{node_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = json.loads(path.read_text()) if path.exists() else {}
    digest = hashlib.sha256(code.encode()).hexdigest()
    if previous and (previous["code_sha256"] != digest or previous["state"] == "running"):
        raise RuntimeError("MF execution identity/release was not reconciled")
    used = float(previous.get("elapsed_seconds", 0))
    remaining = max(0., float(cfg.exec.timeout) - used)
    record = {"code_sha256": digest, "elapsed_seconds": used,
              "started_epoch": time.time(), "state": "running"}
    atomic_json(path, record)
    # Even a completed subprocess may not have committed its host score before
    # interruption. Rerun frozen code with fresh prediction paths, never reuse
    # a stale output directory or add another GP observation.
    if remaining == 0:
        result = ExecutionResult(["Cumulative candidate execution cap exhausted"], 0., "TimeoutError", {}, [])
    else:
        kwargs["execution_timeout"] = min(remaining, kwargs.get("execution_timeout", remaining))
        result = interpreter.run(code, node_id, *args, **kwargs)
    record.update(state="completed", elapsed_seconds=used + result.exec_time)
    atomic_json(path, record)
    return result
