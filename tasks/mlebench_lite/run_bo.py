#!/usr/bin/env python3
"""Launch AQ04 BO with redesigned evolution and fixed full evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
TASKS = ROOT / "config/mlebench_lite22.txt"
sys.path.insert(0, str(ROOT))


def audit_protocol(args):
    """Verify the prepared host protocol before spending a search budget."""
    if args.protocol_dir is None:
        raise ValueError("A full run requires --protocol-dir")
    from engine.fixed_evaluation import _protocol_identity, _source_identity, validate_spec

    directory = args.protocol_dir.resolve()
    path = directory / "protocol.json"
    if not path.is_file():
        raise FileNotFoundError(f"Frozen protocol is missing: {path}")
    spec = json.loads(path.read_text())
    public = (args.data_root.resolve() / args.task / "prepared/public").resolve()
    if (spec.get("task_id") != args.task or spec.get("split", {}).get("seed") != args.seed
            or spec.get("source_dir") != str(public)
            or spec.get("source_identity") != _source_identity(public)):
        raise ValueError("Frozen protocol task, seed or prepared source differs from this run")
    validate_spec(spec, args.task)
    identity = {k: v for k, v in spec.items() if k not in ("protocol_id", "setup_cost_seconds")}
    if _protocol_identity(directory, identity) != spec.get("protocol_id"):
        raise ValueError("Frozen protocol files differ from their recorded identity")
    return {"protocol_id": spec["protocol_id"], "candidate_filesystem_isolation": "not_enforced"}


def build_command(args):
    allowed = {row.strip() for row in TASKS.read_text().splitlines()
               if row.strip() and not row.startswith("#")}
    if args.task not in allowed:
        raise ValueError(f"Unsupported Lite task: {args.task}")
    if args.time_limit <= 0 or args.steps < 1 or args.cpus < 3 or args.workers < 1:
        raise ValueError("Invalid search resource settings")
    public = args.data_root.resolve() / args.task / "prepared/public"
    desc = public / "description.md"
    if not desc.is_file():
        raise FileNotFoundError(f"Prepared Lite task is absent: {desc}")
    output = args.output_root.resolve()
    command = [sys.executable, "-u", "run.py", f"exp_id={args.task}", f"exp_name={args.task}",
               f"dataset_dir={args.data_root.resolve()}", f"data_dir={public}", f"desc_file={desc}",
               f"log_dir={output}", f"workspace_dir={output}",
               "experiment_preset=acq7_aq04_gp4_level15_rating_jts", "audited_bo=redesign",
               "evaluation.enabled=true", f"agent.time_limit={args.time_limit}",
               f"agent.steps={args.steps}", f"agent.seed={args.seed}",
               f"cpu_number={args.cpus}", f"agent.search.parallel_search_num={args.workers}"]
    if args.protocol_dir:
        command.append(f"evaluation.protocol_dir={args.protocol_dir.resolve()}")
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--protocol-dir", type=Path)
    parser.add_argument("--time-limit", type=int, default=43200)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--check-only", action="store_true", help="Print the command without running search")
    args = parser.parse_args(argv)
    try:
        command = build_command(args)
        audit = audit_protocol(args) if args.protocol_dir else None
        if not args.check_only and audit is None:
            raise ValueError("Prepare and supply a frozen --protocol-dir before launching search")
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    if args.check_only:
        print(json.dumps({"project": str(ROOT), "task": args.task, "mode": "check_only",
                          "protocol_verified": audit is not None, "evidence": audit,
                          "command": command}, indent=2))
        return 0
    print(json.dumps({"mode": "verified_launch", "task": args.task, **audit}, indent=2), flush=True)
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
