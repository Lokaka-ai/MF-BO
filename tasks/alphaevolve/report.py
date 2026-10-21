#!/usr/bin/env python3
"""Collect all 15 real draft/improve checks; missing/failed tasks fail the suite."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tasks.alphaevolve.registry import TASKS
from tasks.alphaevolve.prepare import save


def collect(directories):
    tasks = {}
    for task in TASKS:
        attempts = []
        for directory in directories:
            path = directory/task/"report.json"
            if path.is_file():
                entry = json.loads(path.read_text())
                if entry.get("state") == "passed":
                    from tasks.alphaevolve.audit import audit_settings, audit_smoke_trace
                    config = Path(entry["run_info"]["log_dir"])/"config.yaml"
                    command = json.loads((path.parent/"command.json").read_text())
                    entry["settings_audit"] = audit_settings(config, command, smoke=entry.get("smoke", False))
                    if not entry["settings_audit"]["passed"]:
                        entry["state"] = "failed"
                        entry["error"] = "Saved native settings failed the GP6/mlebench_lite audit"
                    if entry.get("smoke"):
                        entry["trace_audit"] = audit_smoke_trace(Path(entry["run_info"]["log_dir"]),
                                                               Path(entry["run_info"]["workspace_dir"]), entry["nodes"])
                        if not entry["trace_audit"]["passed"]:
                            entry["state"] = "failed"
                            entry["error"] = "Native GP6/CPU trace audit failed"
                attempts.append(dict(entry, report=str(path.resolve())))
        # Explicit input ordering selects the latest attempted run, not the best score.
        tasks[task] = {"state": attempts[-1]["state"] if attempts else "missing", "attempts": attempts,
                       "selected_attempt": attempts[-1] if attempts else None}
    return {"collected_at": datetime.now(timezone.utc).isoformat(), "expected_tasks": 15,
            "passed_tasks": sum(x["state"] == "passed" for x in tasks.values()),
            "passed": all(x["state"] == "passed" for x in tasks.values()), "tasks": tasks}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = collect(args.directories)
    save(args.output, report)
    for task, entry in report["tasks"].items():
        print(f"{entry['state']:8} {task}")
    print(f"Passed {report['passed_tasks']}/15; {args.output}")
    if not report["passed"]:
        raise SystemExit(1)
