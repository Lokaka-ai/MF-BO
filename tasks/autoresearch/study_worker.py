"""Use one approved Slurm allocation for independent, frozen study workflows.

No training configuration or selector is changed here. Allocation expiry is
handled by the historical arm supervisor and its cumulative segment ledger.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from engine.shared_drafts import atomic_json


def allocation_info(job):
    raw = subprocess.check_output(["scontrol", "show", "job", "-o", str(job)], text=True)
    fields = dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9/:]*)=(\S+)", raw))
    if fields.get("JobState") != "RUNNING" or not fields.get("UserId", "").endswith(f"({os.getuid()})"):
        raise RuntimeError("The registered allocation must be live and owned by this user")
    return fields


def visible_devices():
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used",
        "--format=csv,noheader,nounits"], text=True)
    devices = [dict(zip(("index", "uuid", "model", "memory_mb"), [v.strip() for v in row.split(",")]))
               for row in raw.splitlines()]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        raise RuntimeError("Slurm must provide CUDA_VISIBLE_DEVICES; do not guess GPU ownership")
    ids = visible.split(",")
    devices = [d for d in devices if d["index"] in ids or d["uuid"] in ids]
    if len(devices) != len(ids) or any(d["model"] != "NVIDIA A100-SXM4-80GB" for d in devices):
        raise RuntimeError("This study stratum requires allocated A100 SXM4 80GB devices")
    for d in devices:
        pids = subprocess.check_output(["nvidia-smi", "-i", d["uuid"],
            "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
        if pids:
            raise RuntimeError(f"Allocated device is occupied; refusing to share {d['uuid']}")
    return devices


def run(campaign, group):
    campaign = campaign.resolve()
    manifest = json.loads((campaign / "campaign.json").read_text())
    methods = manifest["groups"][group]
    if Path(manifest["source_snapshot"]).resolve() != ROOT:
        raise RuntimeError("Launch the worker from the registered immutable source")
    with (campaign / f"{group}.lock").open("a+") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        job = os.environ["SLURM_JOB_ID"]
        allocation = allocation_info(job)
        devices = visible_devices()
        cpus = sorted(os.sched_getaffinity(0))
        if len(methods) != len(devices) or len(cpus) < 4 * len(methods):
            raise RuntimeError("Need one allocated GPU and four allocated CPUs per workflow")
        registered_job = manifest.get("existing_allocations", {}).get(group)
        if registered_job and registered_job != job:
            raise RuntimeError("Existing allocation differs from the user's registered resource")
        if group not in manifest.get("existing_allocations", {}):
            # Batch job name binds newly requested allocations to this study.
            if allocation.get("JobName") != f"{manifest.get('allocation_job_prefix', 'bo-study')}-{group}":
                raise RuntimeError("Batch allocation was not submitted for this study")
        end_epoch = datetime.fromisoformat(allocation["EndTime"]).timestamp()
        if end_epoch - time.time() < 900:
            raise RuntimeError("Less than 15 minutes remains; wait for the next allocation")
        segment_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + f"-{job}-{group}"
        segment = campaign / "segments" / segment_id
        segment.mkdir(parents=True)
        resources = {}
        for i, method in enumerate(methods):
            output = campaign / manifest.get("method_directories", {}).get(method, method)
            output.mkdir(exist_ok=True)
            resources[method] = {"directory": str(output), "allocation": job,
                "node": socket.gethostname().split(".")[0], "gpu": devices[i]["uuid"],
                "gpu_model": devices[i]["model"], "cpu_ids": ",".join(map(str, cpus[4*i:4*i+4]))}
        data = {"segment_id": segment_id, "allocation": job, "group": group,
            "resources": resources, "allocation_end_epoch": end_epoch,
            "config": str(campaign / "paired_config.yaml"), "created_epoch": time.time(),
            "allocation_info": allocation, "budget_policy": "active_segments"}
        atomic_json(segment / "segment.json", data)
        atomic_json(campaign / f"resource_{group}.json", data)
        env = {**os.environ, **manifest["runtime_environment"]}
        children = []
        def requested_stop(signum, frame):
            for child in children:
                if child.poll() is None:
                    child.send_signal(signal.SIGTERM)
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, requested_stop)
        for method in methods:
            status = campaign / method / "status.json"
            if status.exists() and json.loads(status.read_text())["state"] in {"finished", "early_stopped"}:
                continue
            cmd = [sys.executable, str(ROOT / "tasks/autoresearch/segment_supervisor.py"),
                   str(campaign), segment_id, method]
            with (segment / f"{method}.supervisor.log").open("x") as log:
                child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT)
            children.append(child)
            atomic_json(segment / f"{method}.launcher.json", {"pid": child.pid, "command": cmd})
        while any(child.poll() is None for child in children):
            sample = subprocess.run(["nvidia-smi", "--query-gpu=uuid,utilization.gpu,memory.used,power.draw",
                "--format=csv,noheader,nounits"], text=True, capture_output=True)
            with (segment / "gpu_samples.jsonl").open("a") as stream:
                stream.write(json.dumps({"epoch": time.time(), "returncode": sample.returncode,
                    "devices": [row for row in sample.stdout.splitlines()
                        if any(d["uuid"] in row for d in devices)]}) + "\n")
            atomic_json(segment / "worker_status.json", {"state": "running", "heartbeat_epoch": time.time(),
                "pid": os.getpid(), "slurm_job_id": job, "children": [p.pid for p in children]})
            time.sleep(20)
        atomic_json(segment / "worker_status.json", {"state": "stopped", "ended_epoch": time.time(),
            "slurm_job_id": job, "exit_codes": [p.returncode for p in children]})
        return int(any(p.returncode for p in children))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("group")
    args = parser.parse_args()
    raise SystemExit(run(args.campaign, args.group))
