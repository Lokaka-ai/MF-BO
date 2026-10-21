"""Mathematical task adapter for the existing AgentSearch/GP6 runtime."""
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
from tasks.alphaevolve.prepare import digest, save
from tasks.alphaevolve.registry import ROOT, task_spec

TASK_ROOT = Path(__file__).resolve().parent
logger = logging.getLogger("MLEvolve")


class AlphaEvolveTask:
    def __init__(self, cfg):
        self.cfg = cfg
        self.task = task_spec(cfg.exp_id)
        self.spec = {"metric": {"maximize": self.task["maximize"]}}
        if cfg.evaluation.enabled or cfg.use_grading_server:
            raise ValueError("AlphaEvolve requires its mathematical verifier, with Kaggle evaluation/grading disabled")
        data_dir = Path(cfg.dataset_dir).resolve()
        manifest = json.loads((data_dir/"manifest.json").read_text())
        public = data_dir/cfg.exp_id/"prepared/public"
        for name in ("problem.json", "initial.json", "description.md"):
            path = public/name
            if digest(path) != manifest["tasks"][cfg.exp_id][str(path.relative_to(data_dir))]:
                raise ValueError(f"Prepared public asset changed: {name}")
        if Path(cfg.data_dir).resolve() != public:
            raise ValueError("AlphaEvolve data_dir must be the prepared public directory")
        self.problem = json.loads((public/"problem.json").read_text())
        self.protocol = {"version": 1, "task_id": cfg.exp_id, "source_commit": manifest["source_commit"],
                         "public_sha256": {name: digest(public/name) for name in ("problem.json", "initial.json", "description.md")},
                         "host_sha256": {name: digest(TASK_ROOT/name) for name in ("registry.py", "graders.py", "worker.py", "adapter.py")},
                         "maximize": self.task["maximize"], "exec_timeout": cfg.exec.timeout,
                         "cpu_only": True, "candidate_api": "solve(problem, initial, time_budget, seed) -> dict"}
        self.protocol_id = hashlib.sha256(json.dumps(self.protocol, sort_keys=True).encode()).hexdigest()
        path = Path(cfg.log_dir)/"alphaevolve_protocol.json"
        if path.exists() and json.loads(path.read_text()) != self.protocol:
            raise ValueError("Cannot resume with a changed AlphaEvolve protocol")
        save(path, self.protocol)

    def candidate_contract(self):
        return (f"Mathematical optimization task: {self.task['title']}.\n"
                "Implement a complete Python module defining solve(problem, initial, time_budget, seed). "
                "The host calls this function with two JSON dictionaries, a time budget in seconds, and an integer seed. "
                "Return a dictionary of the constructed mathematical objects (lists or NumPy arrays are accepted). "
                "initial is a simple feasible construction you may copy, improve or replace. "
                "Return your best FEASIBLE construction before time_budget expires; use a wall-clock deadline and bounded iterations. "
                "Do not run solve at import time. Use CPU NumPy/SciPy/SymPy only; no GPU is allocated. "
                "The host independently validates every constraint and computes the score in a separate process. "
                "No learned ML model, training split, prediction CSV, fake printed score, or candidate-written result.json is required. "
                "Read only provided inputs. Do not access private reference constructions, prior runs, credentials, or download published answers. "
                f"{self.problem['contract']}\n"
                "Geometric comparisons use absolute tolerance 1e-9; prefer a small feasibility margin. "
                "All outputs must be finite. Public inputs are passed to solve, so no filesystem paths are needed. "
                "Available packages include numpy 2.2.6, scipy 1.16.3, sympy 1.14.0. "
                "Any internal objective must match the host formula exactly. Scores and maximum/minimum directions are immutable.")

    def preview(self):
        initial = json.loads((Path(self.cfg.data_dir)/"initial.json").read_text())
        import numpy as np
        summary = {key: {"shape": list(np.asarray(value).shape), "first_rows": value[:4]} for key, value in initial.items()}
        return self.candidate_contract()+"\nInitial construction structure:\n"+json.dumps(summary)+"\nThe complete initial construction is passed as the initial argument."

    def score(self, node):
        trial = Path(self.cfg.workspace_dir)/"trials"/node.id
        node.evaluation_output_dir = str(trial)
        try:
            result = json.loads((trial/"result.json").read_text())
            if digest(trial/"candidate.py") != hashlib.sha256(node.code.encode()).hexdigest():
                raise ValueError("Candidate source hash mismatch")
            if result.get("construction_sha256") != digest(trial/"construction.json"):
                raise ValueError("Construction changed after host grading")
            if result.get("task_id") != self.task["id"]:
                raise ValueError("Task identity mismatch")
            if result.get("valid"):
                metric = result.get("metric")
                if isinstance(metric, bool) or not isinstance(metric, (int, float)) or not math.isfinite(metric):
                    raise ValueError("Missing finite host metric")
                if result.get("maximize") != self.task["maximize"]:
                    raise ValueError("Wrong metric direction")
        except (OSError, ValueError, TypeError) as exc:
            result = {"valid": False, "metric": None, "error": str(exc)}
        return dict(result, protocol_id=self.protocol_id, node_id=node.id, n_scored=int(result.get("valid", False)))

    def finalize(self, journal):
        best = journal.get_best_node()
        if best is None or not best.is_valid:
            raise RuntimeError("AlphaEvolve finished without a verified mathematical construction")
        trial = Path(self.cfg.workspace_dir)/"trials"/best.id
        exported = Path(self.cfg.workspace_dir)/"best_solution"
        exported.mkdir(parents=True, exist_ok=True)
        shutil.copy2(trial/"construction.json", exported/"construction.json")
        (exported/"solution.py").write_text(best.code)
        # Independent, post-search grading of the exported artifact.
        grade_path = exported/"final_grade.json"
        subprocess.run([sys.executable, str(TASK_ROOT/"worker.py"), "grade", self.task["id"],
                        str(exported/"construction.json"), str(grade_path)], check=True, timeout=900)
        grade = json.loads(grade_path.read_text())
        if not grade["valid"] or not math.isclose(grade["metric"], best.metric.value, rel_tol=1e-12, abs_tol=1e-12):
            raise RuntimeError("Final exported construction failed independent regrading")
        save(Path(self.cfg.log_dir)/"alphaevolve_summary.json", {
            "task_id": self.task["id"], "best_node_id": best.id, "metric": best.metric.value,
            "maximize": self.task["maximize"], "protocol_id": self.protocol_id,
            "construction": str(exported/"construction.json"), "code": str(exported/"solution.py"),
            "final_grade": grade, "completed_nodes": len(journal.nodes)-1})


class AlphaEvolveInterpreter(Interpreter):
    """Isolated per-node files, CPU affinity slots and independent host grading."""
    def run(self, code, id, reset_session=True, **kwargs):
        with self.lock:
            slot = next((i for i, busy in enumerate(self.status_map) if not busy), None)
            if slot is None:
                raise RuntimeError("No free mathematical execution slot")
            self.status_map[slot] = 1
            self.current_parallel_run += 1
        try:
            return self._run_trial(code, id, slot, kwargs.get("remaining_seconds"))
        finally:
            with self.lock:
                self.status_map[slot] = 0
                self.current_parallel_run -= 1

    def _run_trial(self, code, node_id, slot, remaining):
        trial = self.working_dir/"trials"/str(node_id)
        if trial.exists():
            trial.rename(trial.with_name(f"{node_id}.interrupted.{time.time_ns()}"))
        trial.mkdir(parents=True)
        (trial/"candidate.py").write_text(code)
        for name in ("problem.json", "initial.json"):
            shutil.copy2(self.working_dir/"input"/name, trial/name)
        problem = json.loads((trial/"problem.json").read_text())
        available = sorted(os.sched_getaffinity(0))[:self.cpu_number]
        count = max(1, self.cpu_number//self.max_parallel_run)
        cpus = available[slot*count:(slot+1)*count]
        if not cpus:
            raise RuntimeError("No allocated CPU for trial")
        env = {key: value for key, value in os.environ.items()
               if not any(word in key.upper() for word in ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL"))}
        env.update(CUDA_VISIBLE_DEVICES="", USE_TF="0", PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[name] = str(len(cpus))
        save(trial/"resources.json", {"cpus": cpus, "slot": slot, "workers": self.max_parallel_run,
                                      "cuda_visible_devices": "", "slurm_job_id": os.environ.get("SLURM_JOB_ID")})
        limit = min(self.timeout, max(1, remaining-5)) if remaining is not None else self.timeout
        budget = max(1, limit-10)
        seed = int(os.environ.get("ALPHAEVOLVE_SEED", "42"))
        command = [sys.executable, str(TASK_ROOT/"worker.py"), "generate", str(trial), "--seconds", str(budget), "--seed", str(seed)]
        started = time.monotonic()
        error = self._process(command, trial, env, cpus, slot, limit, trial/"execution.log")
        if error is None:
            # Never import candidate code in the scoring process.
            command = [sys.executable, str(TASK_ROOT/"worker.py"), "grade", problem["id"], str(trial/"construction.json"), str(trial/"result.json")]
            error = self._process(command, trial, env, cpus, slot, 900, trial/"grading.log")
        logs = []
        for name in ("execution.log", "grading.log"):
            if (trial/name).exists():
                logs.extend((trial/name).read_text(errors="replace")[-20000:].splitlines())
        return ExecutionResult(logs, time.monotonic()-started, error)

    def _process(self, command, cwd, env, cpus, slot, timeout, log):
        with log.open("w") as handle:
            proc = subprocess.Popen(["taskset", "-c", ",".join(map(str, cpus)), *command], cwd=cwd, env=env,
                                    stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            with self._procs_lock:
                self._active_procs[slot] = proc
            try:
                return None if proc.wait(timeout=timeout) == 0 else "RuntimeError"
            except subprocess.TimeoutExpired:
                return "TimeoutError"
            finally:
                self._stop_group(proc)
                with self._procs_lock:
                    self._active_procs.pop(slot, None)

    @staticmethod
    def _stop_group(proc):
        # Also terminate descendants if the group leader already exited.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()

    def terminate_all_subprocesses(self):
        with self._procs_lock:
            procs = list(self._active_procs.values())
        for proc in procs:
            self._stop_group(proc)
