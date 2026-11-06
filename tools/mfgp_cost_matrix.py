#!/usr/bin/env python3
"""Export execution costs without conflating worker time and campaign time.

Usage: python tools/mfgp_cost_matrix.py --log-dir RUN/logs --output costs.csv
Read-only input. A process that has not finished has unknown elapsed time here.
Join to phase1/mf_gp_decisions.jsonl and llm_calls.jsonl for decision/generation
costs; these have different identities and must not be added once per retry.
"""
import argparse
import csv
import json
from pathlib import Path


def rows(log_dir):
    for path in sorted((Path(log_dir) / "mf_execution").glob("*.json")):
        record = json.loads(path.read_text())
        attempts = record.get("attempts", [])
        completed = [a for a in attempts if a.get("elapsed_seconds") is not None]
        accounted = sum(a["elapsed_seconds"] for a in completed)
        unsegmented = record.get("unsegmented_elapsed_seconds", record["elapsed_seconds"] if not attempts else 0.)
        if abs(accounted + unsegmented - record["elapsed_seconds"]) > 1e-5:
            raise ValueError(f"Cost segments do not reconcile: {path}")
        for attempt in attempts or [dict(state=record["state"],
                                        elapsed_seconds=record["elapsed_seconds"],
                                        measurement="legacy_cumulative_execution")]:
            yield dict(
                program_id=record.get("program_id", path.stem),
                fidelity=record.get("fidelity"),
                code_sha256=record["code_sha256"],
                segment_id=attempt.get("segment_id"),
                attempt_id=attempt.get("attempt_id"),
                job_id=attempt.get("job_id"),
                state=attempt["state"],
                execution_seconds=attempt.get("elapsed_seconds"),
                measurement=attempt.get("measurement"),
                completed_runtime_observation=attempt.get("completed_runtime_observation"),
                runtime_lower_bound=attempt.get("runtime_lower_bound"),
                estimated_action_seconds=attempt.get("estimated_action_seconds"),
                estimate_scope=attempt.get("cost_estimate_scope"),
                unsegmented_prior_seconds=unsegmented if attempts and attempt is attempts[0] else 0.0,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = list(rows(args.log_dir))
    if not data:
        raise SystemExit("No execution records found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
    print(f"Exported {len(data)} segments. Worker times may overlap; not GPU busy time.")


if __name__ == "__main__":
    main()
