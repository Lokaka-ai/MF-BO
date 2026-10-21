#!/usr/bin/env python3
"""Candidate execution and scoring run in separate fresh Python processes."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Non-JSON candidate output: {type(value).__name__}")


def generate(trial, seconds, seed):
    spec = importlib.util.spec_from_file_location("alphaevolve_candidate", trial/"candidate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    problem = json.loads((trial/"problem.json").read_text())
    initial = json.loads((trial/"initial.json").read_text())
    result = module.solve(problem, initial, seconds, seed)
    payload = json.dumps(result, allow_nan=False, default=json_default)
    if len(payload) > 64*1024*1024:
        raise ValueError("Construction exceeds 64 MiB")
    (trial/"construction.json").write_text(payload+"\n")


def grade(task_id, construction, output):
    from tasks.alphaevolve.graders import score
    try:
        if construction.stat().st_size > 64*1024*1024:
            raise ValueError("Oversized construction")
        result = score(task_id, json.loads(construction.read_text()))
    except Exception as exc:
        result = {"valid": False, "metric": None, "task_id": task_id,
                  "error": f"{type(exc).__name__}: {exc}"}
    result["construction_sha256"] = hashlib.sha256(construction.read_bytes()).hexdigest() if construction.exists() else None
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    p = commands.add_parser("generate")
    p.add_argument("trial", type=Path)
    p.add_argument("--seconds", type=float, required=True)
    p.add_argument("--seed", type=int, default=42)
    p = commands.add_parser("grade")
    p.add_argument("task_id")
    p.add_argument("construction", type=Path)
    p.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.mode == "generate":
        generate(args.trial.resolve(), args.seconds, args.seed)
    else:
        grade(args.task_id, args.construction.resolve(), args.output.resolve())
