#!/usr/bin/env python3
"""Run native MLEvolve searches, then grade their exported submissions locally."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import requests

ROOT = Path(__file__).resolve().parents[2]


def save(path, obj):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj, indent=2, default=str) + "\n")
    temp.replace(path)


def stop(process):
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=["leaf-classification", "spooky-author-identification"])
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/mlebench")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu", default="auto", help="auto uses the first CUDA device visible in the current Slurm allocation")
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--time-limit", type=int, default=2400)
    parser.add_argument("--model", help="Optional model override for this run's code and feedback stages")
    parser.add_argument("--provider", help="Provider paired with --model, e.g. opencode or openai")
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE",
                        help="Repeatable native config override for this run only")
    args = parser.parse_args()
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    allowed = set((ROOT / "config/mlebench_lite22.txt").read_text().split())
    if set(args.tasks) - allowed or args.steps < 1 or args.cpus < 3:
        parser.error("Use Lite tasks, at least one step and at least three CPUs for the native three workers")
    if bool(args.model) != bool(args.provider):
        parser.error("Supply --model and --provider together")
    if any("=" not in item or not item.split("=", 1)[0] for item in args.override):
        parser.error("Each --override must have the form KEY=VALUE")
    output = (args.output or ROOT / "runs" / ("mlebench_lite_smoke_" + datetime.now().strftime("%Y%m%d_%H%M%S"))).resolve()
    output.mkdir(parents=True, exist_ok=True)
    args.data_dir = args.data_dir.resolve()
    cpu_ids = sorted(os.sched_getaffinity(0))[:args.cpus]
    if len(cpu_ids) < args.cpus:
        raise RuntimeError("Not enough allocated CPU cores")
    os.sched_setaffinity(0, cpu_ids)
    if args.gpu == "auto":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("The current allocation has no visible CUDA device")
        gpu_uuid = str(torch.cuda.get_device_properties(0).uuid)
        if not gpu_uuid.startswith("GPU-"):
            gpu_uuid = "GPU-" + gpu_uuid
    else:
        gpu_uuid = subprocess.check_output(["nvidia-smi", "-i", args.gpu, "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    lock_dir = ROOT / ".cache/gpu-locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    gpu_lock = (lock_dir / f"{gpu_uuid}.lock").open("a")
    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = os.environ.copy()
    env.setdefault("HF_HOME", str(ROOT / ".cache/mlebench-lite/huggingface"))
    env.setdefault("TORCH_HOME", str(ROOT / ".cache/mlebench-lite/torch"))
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/mlebench-lite/matplotlib"))
    env.update(CUDA_VISIBLE_DEVICES=gpu_uuid, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1", TF_CPP_MIN_LOG_LEVEL="3")
    env.pop("KAGGLE_API_TOKEN", None)
    report = {"started_at": datetime.now(timezone.utc).isoformat(), "hostname": socket.gethostname(),
              "gpu": gpu_uuid, "cpu_ids": cpu_ids, "python": sys.executable,
              "steps_per_task": args.steps, "time_limit_seconds": args.time_limit,
              "model_override": args.model, "provider_override": args.provider,
              "config_overrides": args.override,
              "data_dir": str(args.data_dir), "tasks": {}}
    report_path = output / "report.json"
    save(report_path, report)
    print(f"Smoke output: {output}", flush=True)
    for task in args.tasks:
        directory = output / task
        directory.mkdir(exist_ok=True)
        started = time.monotonic()
        server = runner = None
        try:
            public = args.data_dir / task / "prepared/public"
            if not (public / "description.md").is_file():
                raise FileNotFoundError(f"Task is not prepared: {public}")
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            task_env = dict(env, GRADING_SERVER_PORT=str(port))
            with (directory / "grading-server.log").open("w") as log:
                server = subprocess.Popen([sys.executable, "-u", "-m", "engine.validation.format_server",
                    f"dataset_dir={args.data_dir}", "data_dir=none", "desc_file=none",
                    f"log_dir={directory / 'server'}", f"workspace_dir={directory / 'server'}"],
                    cwd=ROOT, env=task_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            for _ in range(60):
                if server.poll() is not None:
                    raise RuntimeError("Format server exited; see grading-server.log")
                try:
                    response = requests.get(f"http://127.0.0.1:{port}/health", timeout=1)
                    if response.status_code == 200:
                        break
                except requests.RequestException:
                    pass
                time.sleep(1)
            else:
                raise RuntimeError("Format server did not become healthy")
            command = [sys.executable, "-u", "run.py", f"exp_id={task}", f"exp_name={task}",
                       "evaluation.enabled=false",
                       f"dataset_dir={args.data_dir}", f"data_dir={public}", f"desc_file={public / 'description.md'}",
                       f"cpu_number={args.cpus}", f"agent.steps={args.steps}", f"agent.time_limit={args.time_limit}",
                       f"log_dir={directory}", f"workspace_dir={directory}",
                       f"run_result_file={directory / 'run-info.json'}"]
            if args.model:
                from llm.opencode import provider_base_url
                for stage in ("code", "feedback"):
                    command.extend([f"agent.{stage}.model={args.model}",
                                    f"agent.{stage}.base_url={provider_base_url(args.provider)}"])
            command.extend(args.override)
            save(directory / "command.json", command)
            print(f"Running native search: {task}", flush=True)
            with (directory / "console.log").open("w") as log:
                runner = subprocess.Popen(command, cwd=ROOT, env=task_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                code = runner.wait(timeout=args.time_limit + 60)
            if code:
                raise RuntimeError(f"Native search exited with code {code}; see console.log")
            info = json.loads((directory / "run-info.json").read_text())
            final = json.loads((Path(info["log_dir"]) / "final_submission.json").read_text())
            submission = Path(final["submission"])
            with submission.open("rb") as handle:
                validation = requests.post(f"http://127.0.0.1:{port}/validate", headers={"exp-id": task},
                    files={"file": handle}, timeout=120)
            validation.raise_for_status()
            if validation.json().get("is_valid") is not True:
                raise RuntimeError(f"Submission format validation failed: {validation.json()}")
            # Private holdout scoring happens only after the complete search ends.
            from mlebench.grade import grade_csv
            from mlebench.registry import registry
            grade = grade_csv(submission, registry.set_data_dir(args.data_dir).get_competition(task)).to_dict()
            if not grade["valid_submission"] or grade["score"] is None:
                raise RuntimeError("Official grader rejected the exported submission")
            save(directory / "official-grade.json", grade)
            journal = json.loads((Path(info["log_dir"]) / "journal.json").read_text())
            nodes = journal["nodes"]
            valid = [node for node in nodes if node.get("stage") != "root"
                     and node.get("is_valid") is True and node.get("is_buggy") is False]
            if not valid:
                raise RuntimeError("No valid node in the native journal")
            result = {"state": "passed", "run_info": info, "final_submission": final,
                      "valid_nodes": len(valid), "completed_nodes": len(nodes) - 1,
                      "format_validation": validation.json(), "official_grade": grade}
            print(f"PASSED {task}: valid_nodes={len(valid)}, official_score={grade['score']}", flush=True)
        except Exception as error:
            result = {"state": "failed", "error": str(error)}
            print(f"FAILED {task}: {error}", flush=True)
        finally:
            stop(runner)
            stop(server)
        result["elapsed_seconds"] = round(time.monotonic() - started, 1)
        report["tasks"][task] = result
        save(report_path, report)
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["passed"] = all(task["state"] == "passed" for task in report["tasks"].values())
    save(report_path, report)
    print(f"Report: {report_path}", flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
