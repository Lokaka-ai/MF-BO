"""nanoGPT task hooks for the existing MLEvolve_BO search and transition runtime."""
from __future__ import annotations

import ast
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from engine.executor import ExecutionResult, Interpreter

TASK_ROOT = Path(__file__).resolve().parent
REPO = TASK_ROOT / "repo"
PROJECT = TASK_ROOT.parents[1]
START = "# GPT Model"
END = "# IMMUTABLE HARNESS"
logger = logging.getLogger("MLEvolve")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cache_directory():
    return Path(os.environ.get("AUTORESEARCH_CACHE_DIR", PROJECT / ".cache" / "autoresearch")).expanduser().resolve()


def validate_result(result, code, elapsed):
    """Reject failed, stale, partial, non-finite or shortened trials."""
    if not result.get("valid") or result.get("code_sha256") != hashlib.sha256(code.encode()).hexdigest():
        raise ValueError("Missing valid result or candidate source hash mismatch")
    for key in ("val_bpb", "training_seconds", "max_step_seconds", "peak_vram_mb"):
        if isinstance(result.get(key), bool) or not isinstance(result.get(key), (int, float)) or not math.isfinite(result[key]):
            raise ValueError(f"Invalid result field: {key}")
    seconds, step = result["training_seconds"], result["max_step_seconds"]
    if not (step > 0 and 300 <= seconds <= 300 + step + 0.1 and elapsed >= seconds):
        raise ValueError("The complete 300-second training window was not respected")
    if result["val_bpb"] <= 0 or result["peak_vram_mb"] <= 0:
        raise ValueError("Invalid BPB or missing CUDA allocation")
    if result.get("eval_tokens") != 20_971_520 or result.get("num_steps", 0) <= 11:
        raise ValueError("Incomplete evaluation or training")
    if result.get("model_contract_check", {}).get("passed") is not True:
        raise ValueError("Missing causality and cross-entropy validation")


class AutoresearchTask:
    def __init__(self, cfg):
        self.cfg = cfg
        if cfg.evaluation.enabled:
            raise ValueError("Use task_preset=autoresearch; Kaggle evaluation must be disabled")
        self.baseline = (REPO / "train.py").read_text()
        self.prefix = self.baseline.split(START, 1)[0]
        self.suffix = END + self.baseline.split(END, 1)[1]
        manifest = cache_directory() / "manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError("Run bash tasks/autoresearch/setup.sh first (data manifest missing)")
        data = json.loads(manifest.read_text())
        for entry in data["files"]:
            path = cache_directory() / entry["path"]
            if not path.is_file() or path.stat().st_size != entry["bytes"]:
                raise ValueError(f"Prepared data changed or is missing: {path}")
            if entry["path"].startswith("tokenizer/") and digest(path) != entry["sha256"]:
                raise ValueError(f"Prepared tokenizer changed: {path}")
        self.spec = {"metric": {"maximize": False}}
        self.protocol = {
            "candidate_interface": "native-agent-module-v1",
            "task": "autoresearch-nanogpt", "training_seconds": 300,
            "eval_tokens": 20_971_520, "metric": "val_bpb", "maximize": False,
            "data_manifest_sha256": digest(manifest),
            "source_sha256": {name: digest(REPO / name) for name in ("prepare.py", "trial_config.py", "train.py")},
            "worker_sha256": digest(TASK_ROOT / "worker.py"),
        }
        self.protocol_id = hashlib.sha256(json.dumps(self.protocol, sort_keys=True).encode()).hexdigest()
        protocol_path = Path(cfg.log_dir) / "autoresearch_protocol.json"
        protocol_path.parent.mkdir(parents=True, exist_ok=True)
        if protocol_path.exists() and json.loads(protocol_path.read_text()) != self.protocol:
            raise ValueError("Cannot resume autoresearch with a changed dataset, baseline or evaluator")
        protocol_path.write_text(json.dumps(self.protocol, indent=2) + "\n")

    def candidate_contract(self):
        search = getattr(getattr(self.cfg, "agent", None), "search", None)
        workers = int(getattr(search, "parallel_search_num", 2))
        name = "TASK_SERIAL.md" if workers == 1 else "TASK.md"
        cpus = int(getattr(self.cfg, "cpu_number", 4))
        return ((TASK_ROOT / name).read_text()
                + f"\nActive resource configuration: at most {workers} concurrent trials share one GPU; "
                + f"{cpus} allocated CPUs, {cpus // workers} CPUs per training worker.\n")

    def preview(self):
        return (self.candidate_contract() + "\n\n## Reference candidate module (editable)\n```python\n"
                + candidate_body(self.baseline) + "\n```\n\n## Host training/evaluation harness (read-only)\n```python\n"
                + self.suffix + "\n```")

    def validate_code(self, code):
        assemble_candidate(code)

    def score(self, node):
        trial = Path(self.cfg.workspace_dir) / "trials" / node.id
        node.evaluation_output_dir = str(trial)
        try:
            result = json.loads((trial / "result.json").read_text())
            validate_result(result, assemble_candidate(node.code), node.exec_time)
        except (OSError, ValueError, SyntaxError, TypeError, KeyError) as exc:
            return {"valid": False, "metric": None, "error": str(exc),
                    "protocol_id": self.protocol_id, "node_id": node.id, "n_scored": 0}
        return dict(result, metric=result["val_bpb"], protocol_id=self.protocol_id,
                    node_id=node.id, n_scored=result["eval_tokens"])

    def finalize(self, journal):
        best = journal.get_best_node()
        if best is None or not best.is_valid:
            raise RuntimeError("Autoresearch finished without a valid 300-second GPU trial")
        trial = Path(self.cfg.workspace_dir) / "trials" / best.id
        summary = {"best_node_id": best.id, "val_bpb": best.metric.value,
                   "model": self.cfg.agent.code.model, "provider": self.cfg.agent.code.base_url,
                   "trials_completed": len(journal.nodes) - 1,
                   "trial_dir": str(trial), "checkpoint": str(trial / "checkpoint.pt"),
                   "result": best.evaluation_result}
        (Path(self.cfg.log_dir) / "autoresearch_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def candidate_body(source):
    """Keep the immutable reference envelope outside LLM-generated modules."""
    return source.split(START, 1)[1].split(END, 1)[0].strip()


def assemble_candidate(code):
    baseline = (REPO / "train.py").read_text()
    prefix = baseline.split(START, 1)[0]
    suffix = END + baseline.split(END, 1)[1]
    # Backwards-compatible input for an explicitly supplied full reference source.
    if START in code or END in code:
        if not code.startswith(prefix) or not code.endswith(suffix):
            raise ValueError("Candidate changed immutable training/evaluation harness")
        code = candidate_body(code)
    tree = ast.parse(code)
    protected = {"TIME_BUDGET", "MAX_SEQ_LEN", "EVAL_TOKENS", "VAL_SHARD", "TRAIN_DTYPE",
                 "Tokenizer", "make_dataloader", "evaluate_bpb", "TRIAL_CONFIG", "cap",
                 "prepare", "time", "os", "runpy", "sys", "builtins", "contextlib", "gc"}
    # Accidental redefinitions of host controls are invalid candidate programs.
    # This is a protocol validator, not an adversarial Python security sandbox.
    for item in ast.walk(tree):
        bound = []
        if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store):
            bound.append(item.id)
        elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.append(item.name)
        elif isinstance(item, (ast.Import, ast.ImportFrom)):
            if isinstance(item, ast.ImportFrom) and item.module in {"prepare", "trial_config", "__future__"}:
                raise ValueError("Candidate cannot import or replace the immutable protocol")
            bound.extend(alias.asname or alias.name.split(".")[0] for alias in item.names)
        if protected.intersection(bound):
            raise ValueError("Candidate cannot redefine immutable protocol controls: " + str(protected.intersection(bound)))
    assembled = prefix + START + "\n" + code.strip() + "\n\n" + suffix
    ast.parse(assembled)
    return assembled


class AutoresearchInterpreter(Interpreter):
    """Native worker count, isolated trial files/cache/CPU slots, one shared GPU."""

    def run(self, code, id, reset_session=True, **kwargs):
        with self.lock:
            slot = next((i for i, busy in enumerate(self.status_map) if not busy), None)
            if slot is None:
                raise RuntimeError("No free autoresearch execution slot")
            self.status_map[slot] = 1
            self.current_parallel_run += 1
        try:
            return self._run_trial(code, id, slot)
        finally:
            with self.lock:
                self.status_map[slot] = 0
                self.current_parallel_run -= 1

    def _run_trial(self, code, id, slot):
        trial = self.working_dir / "trials" / str(id)
        if trial.exists():
            trial.rename(trial.with_name(f"{id}.interrupted.{time.time_ns()}"))
        trial.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        try:
            assembled = assemble_candidate(code)
        except (ValueError, SyntaxError) as exc:
            (trial / "train.log").write_text(str(exc) + "\n")
            return ExecutionResult([str(exc)], time.monotonic() - started, "ProtocolError")
        (trial / "candidate.py").write_text(code)
        (trial / "train.py").write_text(assembled)
        for name in ("prepare.py", "trial_config.py"):
            shutil.copy2(REPO / name, trial / name)
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "AUTORESEARCH_CACHE_DIR": str(cache_directory())}
        env.pop("EVOLVEFOLD_TRIAL_CONFIG", None)
        available = sorted(os.sched_getaffinity(0))[:self.cpu_number]
        count = max(1, self.cpu_number // self.max_parallel_run)
        cpus = available[slot * count:(slot + 1) * count]
        if not cpus:
            raise RuntimeError("Execution slot has no allocated CPU")
        for key in ("AUTORESEARCH_CPUS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "TORCHINDUCTOR_COMPILE_THREADS"):
            env[key] = str(len(cpus))
        for key in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "CUDA_CACHE_PATH"):
            if key in env:
                path = Path(env[key]) / f"worker-{slot}"
                path.mkdir(parents=True, exist_ok=True)
                env[key] = str(path)
        (trial / "resources.json").write_text(json.dumps({"slot": slot, "cpus": cpus,
            "visible_gpu": env.get("CUDA_VISIBLE_DEVICES"),
            "parallel_workers": self.max_parallel_run}, indent=2) + "\n")
        log_path = trial / "train.log"
        logger.info("[autoresearch] Worker %s CPUs %s training log: %s", slot, cpus, log_path)
        with log_path.open("w") as output:
            proc = subprocess.Popen(["taskset", "-c", ",".join(map(str, cpus)), sys.executable,
                                     str(TASK_ROOT / "worker.py"), str(trial)],
                                    cwd=trial, env=env, stdout=output, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            with self._procs_lock:
                self._active_procs[slot] = proc
            try:
                returncode = proc.wait(timeout=self.timeout)
                error = None if returncode == 0 else "RuntimeError"
            except subprocess.TimeoutExpired:
                self._stop_group(proc)
                error = "TimeoutError"
            finally:
                with self._procs_lock:
                    self._active_procs.pop(slot, None)
        return ExecutionResult(log_path.read_text().splitlines(), time.monotonic() - started, error)

    @staticmethod
    def _stop_group(proc):
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            except ProcessLookupError:
                pass

    def terminate_all_subprocesses(self):
        with self._procs_lock:
            processes = list(self._active_procs.values())
        for proc in processes:
            self._stop_group(proc)
