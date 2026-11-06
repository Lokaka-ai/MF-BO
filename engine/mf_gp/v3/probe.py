"""Bounded T-only execution probe. Does not generate or admit quality labels.

The caller must supply a T-only input view. This is an execution contract, not a
filesystem security sandbox. Candidate-reported throughput remains unverified.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

CONTRACT = """
Support BO_MF_PROBE=1 using the SAME preprocessing/model/loss/training and
prediction functions as normal execution. Use only ./input (a small T-only view).
Run one warm-up and two timed training steps, then one prediction batch. Discard
all weights after the probe. Never read hidden validation/test labels or report
quality scores. Write workload metadata to BO_MF_PROBE_REPORT: schema=1,
steps=3, seconds_per_step, planned_low_steps, planned_full_steps, inference_seconds.
If this estimator cannot support such a probe, report schema=1, supported=false.
Use BO_MF_LOW_T_ROWS and BO_MF_FULL_T_ROWS for declared workload sizes, when
available. Time the two post-warm-up steps with synchronization on CUDA; report
their mean as seconds_per_step. Normal execution must start afresh. Do not use this flag to substitute a different
model or preprocessing path. The host records probe overhead separately.
"""


def run_probe(code, *, python, work, training_view, timeout=120.):
    if not math.isfinite(timeout) or timeout <= 0 or timeout > 120:
        raise ValueError("Probe needs a positive cap of at most 120 seconds")
    work = Path(work)
    training_view = Path(training_view).resolve(strict=True)
    if not training_view.is_dir():
        raise ValueError("Probe training view must be a directory")
    work.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    receipt = dict(schema=1, code_sha256=hashlib.sha256(code.encode()).hexdigest(),
                   state="unknown", elapsed_seconds=0., metadata=None)

    def finish(state, **extra):
        receipt.update(state=state, elapsed_seconds=time.monotonic()-started, **extra)
        tmp = work / "receipt.tmp"
        tmp.write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")
        tmp.replace(work / "receipt.json")
        return receipt

    try:
        compile(code, "candidate.py", "exec")
    except SyntaxError as exc:
        return finish("syntax_error", error=str(exc))
    (work / "candidate.py").write_text(code)
    (work / "input").symlink_to(training_view, target_is_directory=True)
    env = dict(os.environ, BO_MF_PROBE="1", BO_MF_PROBE_STEPS="3",
               BO_MF_PROBE_REPORT=str((work / "workload.json").resolve()))
    sizes = training_view / "complete.json"
    if sizes.exists():
        counts = json.loads(sizes.read_text())
        env["BO_MF_FULL_T_ROWS"] = str(counts["full_training_rows"])
        env["BO_MF_LOW_T_ROWS"] = str(counts["low_training_rows"])
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[3] / "utils") + os.pathsep + env.get("PYTHONPATH", "")
    for name in ("working", "submission"):
        (work / name).mkdir()
    # File-backed stdout/stderr prevents unbounded RAM growth from verbose code.
    with (work / "stdout.log").open("wb") as out, (work / "stderr.log").open("wb") as err:
        proc = subprocess.Popen([str(python), "candidate.py"], cwd=work,
                                env=env, stdout=out, stderr=err, start_new_session=True)
        try:
            proc.wait(timeout=max(.001, timeout-(time.monotonic()-started)))
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            return finish("timeout", returncode=proc.returncode)
        except BaseException:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            raise
    if proc.returncode:
        return finish("runtime_error", returncode=proc.returncode)
    report = work / "workload.json"
    if not report.is_file() or report.stat().st_size > 16384:
        return finish("unknown", reason="missing_or_oversized_workload_report")
    try:
        data = json.loads(report.read_text())
        if not isinstance(data, dict) or data.get("schema") != 1:
            raise ValueError("Unsupported workload schema")
        if data.get("supported") is False:
            return finish("unsupported")
        if data.get("steps") != 3:
            raise ValueError("Probe did not report three steps")
        for key in ("seconds_per_step", "inference_seconds"):
            v = data[key]
            if isinstance(v, bool) or not isinstance(v, (int,float)) or not math.isfinite(v) or v < 0:
                raise ValueError("Invalid timing")
        for key in ("planned_low_steps", "planned_full_steps"):
            v = data[key]
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                raise ValueError("Invalid workload")
        # Whitelist workload fields; do not persist purported validation scores.
        metadata = {k: data[k] for k in ("steps", "seconds_per_step", "inference_seconds",
                                         "planned_low_steps", "planned_full_steps")}
    except (ValueError, KeyError, TypeError) as exc:
        return finish("unknown", reason=str(exc))
    return finish("completed", metadata=metadata, timing_source="candidate_reported",
                  coverage="declared_training_and_prediction_path_not_independently_proven")
