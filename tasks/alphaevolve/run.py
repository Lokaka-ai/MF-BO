#!/usr/bin/env python3
"""Run native MLEvolve with GP6 on one AlphaEvolve mathematical task."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tasks.alphaevolve.prepare import save
from tasks.alphaevolve.registry import TASKS


def verify_smoke(log_dir):
    journal = json.loads((log_dir/"journal.json").read_text())
    nodes = [n for n in journal["nodes"] if n["stage"] != "root"]
    if len(nodes) != 2:
        raise ValueError(f"Expected exactly two completed nodes, got {len(nodes)}")
    draft, child = sorted(nodes, key=lambda n: n.get("step", 0))
    if draft["stage"] != "draft" or child["stage"] != "improve":
        raise ValueError(f"Expected draft then improve, got {[n['stage'] for n in nodes]}")
    parents = journal.get("node2parent", {})
    if parents.get(child["id"]) != draft["id"]:
        raise ValueError("Improved node is not a child of the generated draft")
    for node in nodes:
        if node.get("is_valid") is not True or node.get("is_buggy") is not False or not node.get("evaluation_result", {}).get("valid"):
            raise ValueError(f"Unverified mathematical node: {node['id']}")
    if draft["code"].strip() == child["code"].strip():
        raise ValueError("Iteration did not change the generated program")
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(log_dir/"config.yaml")
    if cfg.parent_selector.type != "gp_child_ucb" or cfg.bo_parent_selector.model != "gp_matern32":
        raise ValueError("Smoke did not use GP6")
    if not cfg.agent.use_global_memory or cfg.agent.memory_embedding_device != "cpu":
        raise ValueError("Native memory is not configured on CPU")
    return [{"id": n["id"], "parent": parents.get(n["id"]), "stage": n["stage"], "metric": n["metric"],
             "code_changed": n is child} for n in (draft, child)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=list(TASKS), required=True)
    p.add_argument("--data-dir", type=Path, default=ROOT/"data/alphaevolve")
    p.add_argument("--output", type=Path)
    p.add_argument("--cpus", type=int, default=21)
    p.add_argument("--time-limit", type=int, default=43200)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--exec-timeout", type=int)
    p.add_argument("--preset", default="gp6")
    p.add_argument("--model")
    p.add_argument("--provider")
    p.add_argument("--smoke", action="store_true", help="One generated draft, then one real GP6-selected improve; one worker")
    args = p.parse_args()
    if args.time_limit <= 0 or args.steps <= 0 or (args.exec_timeout is not None and args.exec_timeout <= 10):
        p.error("Positive agent/step budgets and an execution timeout above 10 seconds are required")
    if args.smoke and args.preset != "gp6":
        p.error("The draft/improve smoke contract requires the gp6 preset")
    if not os.environ.get("SLURM_JOB_ID"):
        p.error("Run real workloads in a Slurm CPU allocation")
    if bool(args.model) != bool(args.provider):
        p.error("Supply --model and --provider together")
    if args.cpus < (1 if args.smoke else 3):
        p.error("Allocate enough CPUs for native workers")
    available = sorted(os.sched_getaffinity(0))
    if len(available) < args.cpus:
        p.error("Insufficient allocated CPU affinity")
    os.sched_setaffinity(0, available[:args.cpus])
    directory = (args.output or ROOT/"runs"/("alphaevolve_"+datetime.now().strftime("%Y%m%d_%H%M%S"))/args.task).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    report = {"task_id": args.task, "started_at": datetime.now(timezone.utc).isoformat(), "state": "running",
              "slurm_job_id": os.environ["SLURM_JOB_ID"], "host": socket.gethostname(), "smoke": args.smoke}
    save(directory/"report.json", report)
    command = [sys.executable, "-u", "run.py", "task_preset=alphaevolve", f"experiment_preset={args.preset}",
               f"exp_id={args.task}", f"exp_name={args.task}", f"dataset_dir={args.data_dir.resolve()}",
               f"cpu_number={args.cpus}", f"agent.time_limit={args.time_limit}", f"agent.steps={2 if args.smoke else args.steps}",
               f"log_dir={directory}", f"workspace_dir={directory}", f"run_result_file={directory/'run-info.json'}",
               f"exec.timeout={args.exec_timeout or (120 if args.smoke else 32400)}"]
    if args.smoke:
        command += ["agent.initial_drafts=1", "agent.search.parallel_search_num=1"]
    if args.model:
        from llm.opencode import provider_base_url
        for stage in ("code", "feedback"):
            command += [f"agent.{stage}.model={args.model}", f"agent.{stage}.base_url={provider_base_url(args.provider)}"]
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="", USE_TF="0", MPLBACKEND="Agg", PYTHONUNBUFFERED="1")
    env.setdefault("HF_HOME", str(ROOT/".cache/mlebench-lite/huggingface"))
    env.setdefault("MPLCONFIGDIR", str(ROOT/".cache/alphaevolve/matplotlib"))
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = str(args.cpus)
    save(directory/"command.json", command)
    process = None
    started = time.monotonic()
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        print(f"Native AlphaEvolve {args.task}, preset={args.preset}, CPU={args.cpus}, agent budget={args.time_limit}s", flush=True)
        with (directory/"console.log").open("w") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            returncode = process.wait(timeout=args.time_limit+120)
        if returncode:
            raise RuntimeError(f"Native AgentSearch exited {returncode}; see console.log")
        info = json.loads((directory/"run-info.json").read_text())
        logs = Path(info["log_dir"])
        summary = json.loads((logs/"alphaevolve_summary.json").read_text())
        nodes = verify_smoke(logs) if args.smoke else []
        report.update(state="passed", run_info=info, summary=summary, nodes=nodes)
        print(f"PASS {args.task}: {summary['metric']}", flush=True)
    except (Exception, KeyboardInterrupt) as exc:
        report.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        print(f"FAIL {args.task}: {exc}", flush=True)
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        report.update(elapsed_seconds=time.monotonic()-started, finished_at=datetime.now(timezone.utc).isoformat())
        save(directory/"report.json", report)
    if report["state"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
