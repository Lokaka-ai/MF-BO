"""Resource lease and reproducible environment for the standard run.py entrypoint."""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[2]


def select_cpus(available, count, requested=None):
    available = sorted(available)
    selected = [int(value) for value in requested.split(",")] if requested else available[:count]
    if count < 1 or len(selected) != count or len(set(selected)) != count:
        raise ValueError("Request exactly --cpus distinct CPU IDs within the allocation")
    if not set(selected).issubset(available):
        raise ValueError(f"Requested CPUs are outside the allocated affinity: {available}")
    return selected


def compile_cache_environment(cache_dir):
    cache_dir = Path(cache_dir).resolve()
    paths = {"TRITON_CACHE_DIR": cache_dir / "triton",
             "TORCHINDUCTOR_CACHE_DIR": cache_dir / "torchinductor",
             "CUDA_CACHE_PATH": cache_dir / "cuda"}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return {key: str(path) for key, path in paths.items()}


def main():
    parser = argparse.ArgumentParser(description="Run MLEvolve_BO autoresearch on a single A100.")
    parser.add_argument("--gpu", default="0", help="Physical GPU index or GPU UUID (default: 0)")
    parser.add_argument("--steps", type=int, default=500, help="Native total trial limit, including initial drafts")
    parser.add_argument("--resume", type=Path, help="Existing per-run directory containing logs/ and workspace/")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--name", default="autoresearch-nanogpt")
    parser.add_argument("--cpus", type=int, help="CPU count; defaults to at most eight allocated CPUs")
    parser.add_argument("--cpu-ids", help="Comma-separated allocated CPU IDs for concurrent runs")
    parser.add_argument("--compile-cache", type=Path, help="Exclusive node-local compilation cache; default is a fresh local temporary directory")
    parser.add_argument("overrides", nargs="*", help="Additional MLEvolve OmegaConf key=value overrides")
    args = parser.parse_args()
    if args.steps < 1 or "," in args.gpu:
        parser.error("Use a positive step count and exactly one GPU")
    available = os.sched_getaffinity(0)
    count = args.cpus if args.cpus is not None else min(int(os.environ.get("AUTORESEARCH_CPUS", "8")), len(available))
    try:
        cpus = select_cpus(available, count, args.cpu_ids)
    except ValueError as exc:
        parser.error(str(exc))
    os.sched_setaffinity(0, cpus)
    gpu_uuid = subprocess.check_output(["nvidia-smi", "-i", args.gpu, "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, PYTHONUNBUFFERED="1",
               MLEVOLVE_CODEX_BIN=str(ROOT / "tools" / "codex_opencode.py"))
    env.setdefault("AUTORESEARCH_CACHE_DIR", str(ROOT / ".cache" / "autoresearch"))
    env.update(AUTORESEARCH_CPUS=str(count), RAYON_NUM_THREADS=str(count),
               OMP_NUM_THREADS=str(count), OPENBLAS_NUM_THREADS=str(count),
               MKL_NUM_THREADS=str(count), NUMEXPR_NUM_THREADS=str(count),
               TORCHINDUCTOR_COMPILE_THREADS=str(min(count, 4)))
    # BeeGFS can fail removing Triton temporary directories under concurrent
    # compilation. Share immutable data only; keep compilation local to each run.
    cache_dir = args.compile_cache or Path(tempfile.mkdtemp(
        prefix=f"mlevolve-autoresearch-{os.getuid()}-",
        dir=os.environ.get("SLURM_TMPDIR") or tempfile.gettempdir()))
    env.update(compile_cache_environment(cache_dir))
    env.setdefault("MLEVOLVE_CODEX_REASONING_EFFORT", "medium")
    env.setdefault("MLEVOLVE_CODEX_TIMEOUT_SECONDS", "600")
    env.setdefault("OPENCODE_SESSION_ID", "ses_" + uuid.uuid4().hex)
    # Also inherited by the search process, keeping the lease for its lifetime.
    lock_dir = ROOT / ".cache" / "gpu-locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / f"{gpu_uuid}.lock").open("a+") as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            sys.exit(f"GPU {args.gpu} already has an autoresearch run in this checkout")
        subprocess.run([sys.executable, str(ROOT / "tasks" / "autoresearch" / "manifest.py"), "--verify"],
                       cwd=ROOT, env=env, check=True)
        command = [sys.executable, str(ROOT / "run.py"), "task_preset=autoresearch",
                   f"agent.steps={args.steps}", f"cpu_number={count}", f"exp_name={args.name}",
                   f"log_dir={args.runs_dir.resolve()}", f"workspace_dir={args.runs_dir.resolve()}"]
        if args.resume:
            command.append(f"resume_run_dir={args.resume.resolve()}")
        command.extend(args.overrides)
        print(f"Autoresearch: GPU {args.gpu} ({gpu_uuid}), CPUs {cpus}, {args.steps} total trials, 300s training per trial", flush=True)
        print(f"Exclusive compilation cache: {cache_dir}", flush=True)
        os.chdir(ROOT)
        os.set_inheritable(lease.fileno(), True)
        os.execve(sys.executable, command, env)


if __name__ == "__main__":
    main()
