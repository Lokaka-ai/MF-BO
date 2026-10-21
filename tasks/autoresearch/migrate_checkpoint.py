"""Admit a cross-node resume only after Slurm proves the old executor was released."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from engine.shared_drafts import atomic_json

TERMINAL = {"PREEMPTED", "CANCELLED", "COMPLETED", "FAILED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY"}


def verify_release(job_id, owner_host, current_host):
    if not str(job_id).isdigit():
        raise ValueError("Invalid Slurm job id")
    acct = subprocess.run(["sacct", "-n", "-P", "-X", "-j", str(job_id), "-o", "JobID,State,NodeList"],
                          capture_output=True, text=True, check=True, timeout=30).stdout
    for line in acct.splitlines():
        fields = line.split("|")
        if len(fields) >= 3 and fields[0] == str(job_id) and fields[1].split()[0].rstrip('+') in TERMINAL:
            return {"job_id":str(job_id),"scheduler_state":fields[1],"scheduler_evidence":line}
    live = subprocess.run(["scontrol", "show", "job", "-o", str(job_id)], capture_output=True, text=True, timeout=30)
    def field(name):
        match = re.search(r"(?:^|\s)" + name + r"=(\S+)", live.stdout)
        return match.group(1) if match else ""
    if (live.returncode == 0 and field("JobState") == "RUNNING" and int(field("Restarts") or 0) > 0
            and field("NodeList") == current_host.split(".")[0]
            and field("NodeList") != owner_host.split(".")[0]):
        return {"job_id":str(job_id),"scheduler_state":"REQUEUED_ON_NEW_NODE","scheduler_evidence":live.stdout}
    raise RuntimeError("Slurm has not proved release of the checkpoint's old executor")


def release(log_dir, job_id):
    log_dir = Path(log_dir)
    ledger = log_dir / "phase1/root_draft_state.json"
    if not ledger.exists():
        return
    state = json.loads(ledger.read_text())
    if not state.get("pending") or state.get("owner_host") == socket.gethostname():
        return
    evidence = verify_release(job_id, state["owner_host"], socket.gethostname())
    receipt = dict(evidence, verified_by="slurm_allocation_release", owner_host=state["owner_host"],
                   owner_pid=state["owner_pid"], checked_at=time.time(), checked_on=socket.gethostname())
    backup = ledger.with_name(f"root_draft_state.before-migration-{time.time_ns()}.json")
    backup.write_bytes(ledger.read_bytes())
    state["executor_release_receipt"] = receipt
    atomic_json(ledger, state)
    atomic_json(log_dir / "phase1/executor_release_receipt.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    release(args.log_dir, args.job_id)
