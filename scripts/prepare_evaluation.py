"""Prepare real task protocols without starting search or candidate training."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import _load_cfg, load_task_desc, prep_agent_workspace
from engine.fixed_evaluation import FixedEvaluation, _write_json


def prepare_task(task, args, api_key):
    started = time.monotonic()
    run_dir = args.output_root / task
    cfg = _load_cfg(use_cli_args=False)
    cfg.exp_id = task
    cfg.data_dir = str(args.dataset_root / task / "prepared/public")
    cfg.desc_file = str(Path(cfg.data_dir) / "description.md")
    cfg.log_dir = str(run_dir / "logs")
    cfg.workspace_dir = run_dir / "workspace"
    cfg.agent.seed = args.seed
    cfg.agent.time_limit = args.timeout
    cfg.agent.code.model = args.model
    cfg.agent.code.base_url = args.base_url
    cfg.agent.code.api_key = api_key
    cfg.evaluation.enabled = True
    cfg.evaluation.protocol_dir = str(args.output_root / "protocols" / task / f"seed-{args.seed}")
    cfg.evaluation.setup_timeout_seconds = args.timeout
    cfg.evaluation.setup_max_attempts = args.attempts
    result = {"task": task, "model": args.model, "seed": args.seed, "status": "running"}
    _write_json(run_dir / "preparation_result.json", result)
    print(f"START {task}", flush=True)
    try:
        prep_agent_workspace(cfg)
        journal = SimpleNamespace(nodes=[], evaluation_state=None)
        evaluation = FixedEvaluation.prepare_or_load(cfg, load_task_desc(cfg), journal)
        result.update(status="passed", protocol_dir=str(evaluation.directory),
                      protocol_id=evaluation.protocol_id, n_scored=evaluation.spec["n_expected"],
                      metric=evaluation.spec["metric"], split=evaluation.spec["split"],
                      setup_cost_seconds=evaluation.spec["setup_cost_seconds"])
    except Exception as exc:
        result.update(status="failed", error=str(exc).replace(api_key, "[REDACTED]"),
                      error_type=type(exc).__name__, log_dir=cfg.log_dir)
    result["wall_seconds"] = round(time.monotonic() - started, 3)
    _write_json(run_dir / "preparation_result.json", result)
    print(f"{result['status'].upper()} {task} {result['wall_seconds']}s", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--api-key-file", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--attempts", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--tasks", nargs="+")
    args = parser.parse_args()
    if args.timeout <= 0 or args.attempts < 0 or args.workers < 1:
        parser.error("Timeout and worker count must be positive; attempts must be nonnegative (0 means deadline only)")
    tasks = [line.strip() for line in (ROOT / "config/mlebench_lite22.txt").read_text().splitlines()
             if line.strip() and not line.startswith("#")]
    selected = args.tasks or tasks
    if len(set(selected)) != len(selected) or not set(selected) <= set(tasks):
        parser.error("Tasks must be unique entries in config/mlebench_lite22.txt")
    args.dataset_root = args.dataset_root.resolve()
    args.output_root = args.output_root.resolve()
    api_key = args.api_key_file.read_text().strip()
    if not api_key:
        parser.error("API key file is empty")
    summary_path = args.output_root / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(prepare_task, task, args, api_key) for task in selected]
        for future in as_completed(futures):
            result = future.result()
            summary[result["task"]] = result
            _write_json(summary_path, summary)
    failed = [task for task in selected if summary[task]["status"] != "passed"]
    print(f"Prepared {len(selected) - len(failed)}/{len(selected)} tasks; results: {summary_path}", flush=True)
    return int(bool(failed))


if __name__ == "__main__":
    sys.exit(main())
