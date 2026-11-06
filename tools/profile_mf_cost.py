"""Measure a fixed Python workload at both fidelities; never grade or submit jobs.
Run inside an appropriate allocation. The workload must use ./input, train on T,
and include its normal inference work. No validation/test labels are provided.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine.multifidelity import prepare_view
from engine.mf_gp.policy import load_cost_prior


def profile(candidate, protocol, work, output, fraction=.2, seed=42, timeout=300):
    if output.exists():
        raise ValueError("Refusing to overwrite a cost profile")
    if timeout <= 0:
        raise ValueError("Timeout must be positive")
    protocol, work = protocol.resolve(), work.resolve()
    work.mkdir(parents=True, exist_ok=False)
    code = candidate.read_bytes()
    from engine.mf_gp.execution import insert_setup
    executable = insert_setup(code.decode(),
        f"import random, numpy, torch\nrandom.seed({seed}); numpy.random.seed({seed}); torch.manual_seed({seed})\n")
    low, _ = prepare_view(protocol, work, seed, fraction)
    timings = []
    for name, view in (("low", low), ("full", protocol / "public")):
        cwd = work / name
        cwd.mkdir()
        (cwd / "input").symlink_to(view, target_is_directory=True)
        for subdir in ("working", "submission", "evaluation_outputs"):
            (cwd / subdir).mkdir()
        (cwd / "candidate.py").write_text(executable)
        env = dict(os.environ, MLEV_EVALUATION_OUTPUT_DIR=str(cwd / "evaluation_outputs"),
                   MLEV_EVALUATION_PROTOCOL_ID=json.loads((protocol / "protocol.json").read_text())["protocol_id"],
                   PYTHONHASHSEED=str(seed))
        started = time.monotonic()
        with (cwd / "stdout.log").open("w") as out, (cwd / "stderr.log").open("w") as err:
            proc = subprocess.Popen([sys.executable, "candidate.py"], cwd=cwd,
                                    env=env, stdout=out, stderr=err, start_new_session=True)
            code_exit = None
            timed_out = False
            try:
                code_exit = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                raise
            finally:
                # Also clean up descendants of a completed/failed workload.
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
                (cwd / "status.json").write_text(json.dumps(dict(
                    returncode=proc.returncode, timed_out=timed_out,
                    elapsed_seconds=time.monotonic()-started,
                    completed_runtime_observation=code_exit == 0 and not timed_out)))
        elapsed = time.monotonic() - started
        if code_exit:
            raise RuntimeError(f"{name} profile failed ({code_exit}); logs preserved at {cwd}")
        timings.append(elapsed)
    receipt = dict(schema=1, protocol_id=env["MLEV_EVALUATION_PROTOCOL_ID"],
        training_fraction=fraction, completed=True, code_sha256=hashlib.sha256(code).hexdigest(),
        executable_sha256=hashlib.sha256(executable.encode()).hexdigest(),
        measurement="candidate_process_wall_seconds", low_seconds=timings[0], full_seconds=timings[1],
        seed=seed, job_id=os.environ.get("SLURM_JOB_ID"),
        cpus=os.environ.get("SLURM_CPUS_PER_TASK"), memory_mb=os.environ.get("SLURM_MEM_PER_NODE"),
        warning="One workload is a prior, not a runtime guarantee for future generated programs.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(receipt, stream, indent=2)
    load_cost_prior(output, protocol, fraction)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fraction", type=float, default=.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timeout", type=float, default=300)
    p.add_argument("--local", action="store_true", help="Explicitly allow local profiling without Slurm")
    a = p.parse_args()
    if not os.environ.get("SLURM_JOB_ID") and not a.local:
        p.error("Run in an allocation; use --local only for intentional local profiling")
    print(json.dumps(profile(a.candidate, a.protocol, a.work, a.output, a.fraction, a.seed, a.timeout), indent=2))

if __name__ == "__main__":
    main()
