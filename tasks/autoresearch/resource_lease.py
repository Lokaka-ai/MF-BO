"""Keep one Slurm GPU allocation for the authorized experiment handoff."""
from __future__ import annotations

import argparse
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
from engine.shared_drafts import atomic_json
from tasks.autoresearch.supervise import supervise, timestamp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,name,memory.total", "--format=csv,noheader,nounits"], text=True).splitlines()
    if len(rows) == 1:
        row = rows[0]
    else:
        device = os.environ.get("CUDA_VISIBLE_DEVICES", os.environ.get("SLURM_JOB_GPUS", "")).split(",")[0]
        if not device:
            raise RuntimeError("Cannot identify the allocated GPU")
        row = subprocess.check_output(["nvidia-smi", "-i", device, "--query-gpu=uuid,name,memory.total", "--format=csv,noheader,nounits"], text=True).strip()
    gpu, model, memory = [s.strip() for s in row.split(",")]
    if "A100-SXM4-80GB" not in model or int(memory) < 75000:
        raise RuntimeError(f"Expected matching A100-SXM4-80GB, got {row}")
    atomic_json(directory / "ready.json", {"hostname": socket.gethostname(), "gpu": gpu,
        "model": model, "memory_mb": int(memory), "cpus": sorted(os.sched_getaffinity(0)),
        "job_id": os.environ["SLURM_JOB_ID"], "pid": os.getpid(), "ready_at": timestamp()})
    deadline = time.monotonic() + 12.5 * 3600
    while time.monotonic() < deadline and not (directory / "release").exists():
        for request_path in sorted(directory.glob("request_*.json")):
            if request_path.name.endswith(".done.json"):
                continue
            response_path = directory / (request_path.stem + ".done.json")
            request = json.loads(request_path.read_text())
            output = Path(request["output"])
            start_epoch = request.get("start_epoch")
            deadline_epoch = request.get("deadline_epoch")
            previous = json.loads((output / "status.json").read_text()) if (output / "status.json").exists() else None
            if previous:
                from datetime import datetime
                start_epoch = datetime.fromisoformat(previous["started_at"]).timestamp()
                deadline_epoch = datetime.fromisoformat(previous["deadline_at"]).timestamp()
            if response_path.exists():
                interrupted = previous and (previous.get("stop_reason") == "signal" or
                    (isinstance(previous.get("exit_code"), int) and previous["exit_code"] < 0))
                if not interrupted or deadline_epoch is None or time.time() >= deadline_epoch:
                    continue
                response_path.rename(response_path.with_name(response_path.name + f".interrupted-{time.time_ns()}"))
            if deadline_epoch is not None and deadline_epoch <= time.time():
                atomic_json(response_path, {"state": "expired", "finished_at": timestamp()})
                continue
            command = [s.replace("{GPU}", gpu) for s in request["command"]]
            if "--cpu-ids" in command:
                command[command.index("--cpu-ids") + 1] = ",".join(map(str, sorted(os.sched_getaffinity(0))[:4]))
            # Slurm can requeue this batch lease after preemption. Recover the
            # admitted journal while preserving the original wall-clock deadline.
            if previous:
                location_file = output / "run_location.json"
                if location_file.exists():
                    location = json.loads(location_file.read_text())
                    run_root = str(Path(location["log_dir"]).parent)
                    if "--resume" not in command:
                        position = command.index("--runs-dir")
                        command[position:position] = ["--resume", run_root]
                    command.append(f"resume_elapsed_seconds={int(time.time()-start_epoch)}")
                if "--resume" in command:
                    from tasks.autoresearch.migrate_checkpoint import release
                    release(Path(command[command.index("--resume") + 1]) / "logs", previous["slurm_job_id"])
                suffix = str(time.time_ns())
                for name in ("status.json", "console.log"):
                    source = output / name
                    if source.exists():
                        source.rename(output / f"{name}.before-resume-{suffix}")
            duration = request["duration_seconds"]
            if deadline_epoch is not None:
                if start_epoch is None:
                    start_epoch = time.time()
                duration = deadline_epoch - start_epoch
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu
            metadata = dict(request.get("metadata", {}), gpu=gpu, cpus=4)
            result = supervise(command, output, duration, metadata, grace_seconds=30, start_epoch=start_epoch)
            final = json.loads((output / "status.json").read_text())
            if final.get("stop_reason") == "signal" and time.time() < (deadline_epoch or float("inf")):
                # Leave the request unfinished so a requeued lease resumes it.
                raise SystemExit(0)
            atomic_json(response_path, {"state": "finished", "returncode": result, "finished_at": timestamp()})
        time.sleep(2)
    print("Releasing experiment GPU allocation", flush=True)


if __name__ == "__main__":
    main()
