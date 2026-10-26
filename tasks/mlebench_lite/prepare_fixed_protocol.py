#!/usr/bin/env python3
"""Prepare a frozen fixed-evaluation protocol without starting BO search."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import _load_cfg, load_task_desc, prep_cfg
from engine.fixed_evaluation import FixedEvaluation
from engine.search_node import Journal
from tasks.mlebench_lite.run_bo import TASKS


def prepare(args):
    allowed = {row.strip() for row in TASKS.read_text().splitlines()
               if row.strip() and not row.startswith("#")}
    if args.task not in allowed:
        raise ValueError(f"Task is not in the supported Lite set: {args.task}")
    public = args.data_root.resolve() / args.task / "prepared/public"
    desc = public / "description.md"
    if not desc.is_file():
        raise FileNotFoundError(f"Prepared task is absent: {desc}")
    cfg = _load_cfg(use_cli_args=False)
    if getattr(args, "model", None):
        for stage in (cfg.agent.code, cfg.agent.feedback):
            stage.model = args.model
            stage.base_url = args.provider
    cfg.exp_id = args.task
    cfg.exp_name = args.task + "_fixed_protocol_setup"
    cfg.dataset_dir = str(args.data_root.resolve())
    cfg.data_dir = str(public)
    cfg.desc_file = str(desc)
    cfg.log_dir = str(args.output_root.resolve())
    cfg.workspace_dir = str(args.output_root.resolve())
    cfg.agent.seed = args.seed
    cfg.evaluation.enabled = True
    if args.protocol_dir:
        cfg.evaluation.protocol_dir = str(args.protocol_dir.resolve())
    cfg = prep_cfg(cfg)
    evaluator = FixedEvaluation.prepare_or_load(cfg, load_task_desc(cfg), Journal())
    return {"task": args.task, "seed": args.seed,
            "protocol_dir": str(evaluator.directory),
            "protocol_id": evaluator.protocol_id, "metric": evaluator.spec["metric"],
            "split": evaluator.spec["split"], "setup_log_dir": str(cfg.log_dir)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--protocol-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model")
    parser.add_argument("--provider")
    args = parser.parse_args(argv)
    if bool(args.model) != bool(args.provider):
        parser.error("Provide --model and --provider together")
    print(json.dumps(prepare(args), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
