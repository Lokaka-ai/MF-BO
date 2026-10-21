"""Preserve the historical active-segment budget for one immutable study run."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from engine.shared_drafts import atomic_json
from tasks.autoresearch.supervise import build_command, process_table, stop_tree, timestamp


def read(path):
    return json.loads(Path(path).read_text())


def budget_used(history):
    return sum(float(s["active_seconds"]) for s in history)


def recover_unrecorded_segment(history, previous, duration_seconds=28800):
    """Account for a hard Slurm kill even if the supervisor could not finalize."""
    if any(s["segment_id"] == previous["segment_id"] for s in history):
        return history
    job = previous["slurm_job_id"]
    result = subprocess.check_output(["sacct", "-n", "-X", "-P", "-j", job, "-o", "JobID,State,End"], text=True)
    terminal = {"PREEMPTED", "CANCELLED", "COMPLETED", "FAILED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY"}
    for line in result.splitlines():
        fields = line.split("|")
        if len(fields) >= 3 and fields[0] == job and fields[1].split()[0].rstrip("+") in terminal:
            end = datetime.fromisoformat(fields[2]).replace(tzinfo=ZoneInfo("America/Toronto")).timestamp()
            start = previous["segment_started_epoch"]
            used = min(max(0, end-start), max(0, duration_seconds-budget_used(history)))
            return [*history, {"segment_id": previous["segment_id"], "started_epoch": start, "ended_epoch": end,
                "active_offset_seconds": budget_used(history), "active_seconds": used, "allocation": job,
                "node": previous["hostname"].split(".")[0], "state": "paused", "stop_reason": "slurm_interruption",
                "scheduler_evidence": line}]
    raise RuntimeError("Cannot account for the interrupted segment before Slurm confirms its end")


def arm(campaign, segment_id, setting):
    from omegaconf import OmegaConf
    from tasks.autoresearch.segmented import restore_checkpoint
    from tasks.autoresearch.migrate_checkpoint import release, verify_release
    campaign = campaign.resolve(); data = read(campaign / "segments" / segment_id / "segment.json")
    r = data["resources"][setting]; output = Path(r["directory"])
    cfg = OmegaConf.load(data["config"]); history_file = output / "segment_history.json"
    duration = int(data.get("duration_seconds", cfg.duration_seconds))
    if not 0 < duration <= int(cfg.duration_seconds):
        raise ValueError("Invalid cumulative workflow budget")
    history = read(history_file) if history_file.exists() else []
    checkpoint = output / "checkpoints"
    if output.joinpath("status.json").exists():
        old = read(output / "status.json")
        if str(old["state"]).startswith("blocked/"):
            raise RuntimeError("A blocked study7 arm requires diagnosis; automatic retry is forbidden")
        if old["state"] not in {"paused", "finished", "failed"}:
            verify_release(old["slurm_job_id"], old["hostname"], socket.gethostname())
        old_pid = old.get("supervisor_pid")
        if old["hostname"] == socket.gethostname() and old_pid in process_table():
            identity = process_table()[old_pid]
            if identity[1] == str(old.get("supervisor_start_ticks")) and identity[2] != "Z":
                raise RuntimeError("Previous supervisor still alive")
        history = recover_unrecorded_segment(history, old, duration)
        atomic_json(history_file, history)
        if old["state"] == "finished" or budget_used(history) >= duration:
            old.update(state="finished", segment_id=segment_id, remaining_active_seconds=0)
            atomic_json(output / "status.json", old)
            return
        if (checkpoint / "latest.json").exists():
            receipt = restore_checkpoint(checkpoint)
            location = read(output / "run_location.json")
            release(Path(location["log_dir"]), old["slurm_job_id"])
            atomic_json(output / f"restore-{segment_id}.json", receipt)
        else:
            raise RuntimeError("Cannot resume without a committed checkpoint")
        (output / "status.json").rename(output / f"status.json.before-resume-{segment_id}")
    offset = budget_used(history)
    segment_output = output / "segments" / segment_id; segment_output.mkdir(parents=True)
    pause_file = segment_output / "pause.request"
    command = build_command(cfg, setting, r["gpu"], output, r["cpu_ids"])
    worker_cfg = OmegaConf.load(ROOT / "tasks/autoresearch/config.yaml")
    if history:
        location = read(output / "run_location.json")
        saved_config = Path(location["log_dir"]) / "config.yaml"
        if saved_config.is_file():
            worker_cfg = OmegaConf.merge(worker_cfg, OmegaConf.load(saved_config))
        index = command.index("--runs-dir")
        command[index:index] = ["--resume", str(Path(location["log_dir"]).parent)]
        command.append(f"resume_elapsed_seconds={int(offset)}")
    workers = int(cfg.overrides.get("agent.search.parallel_search_num", worker_cfg.agent.search.parallel_search_num))
    start = time.time()
    env = {**os.environ, "MLEVOLVE_SEGMENTED_RUN": "1", "MLEVOLVE_SEGMENT_CHECKPOINT": str(checkpoint),
           "MLEVOLVE_SEGMENT_PAUSE_FILE": str(pause_file), "MLEVOLVE_SEGMENT_STARTED": str(start),
           "MLEVOLVE_SEGMENT_ACTIVE_OFFSET": str(offset)}
    table = process_table()
    status = {"setting": setting, "state": "starting", "segment_id": segment_id, "hostname": socket.gethostname(),
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
              "gpu": r["gpu"], "cpus": int(cfg.cpus), "cpu_ids": r["cpu_ids"], "parallel_search_num": workers,
              "supervisor_pid": os.getpid(), "supervisor_start_ticks": table[os.getpid()][1],
              "started_at": timestamp(history[0]["started_epoch"] if history else start), "segment_started_at": timestamp(start),
              "segment_started_epoch": start, "active_offset_seconds": offset, "active_elapsed_seconds": offset,
              "duration_seconds": duration, "source_duration_seconds": int(cfg.duration_seconds), "deadline_at": None, "allocation_end_at": timestamp(data["allocation_end_epoch"]), "command": command}
    atomic_json(output / "status.json", status)
    requested = []
    def stop(signum, frame): requested.append(signum)
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, stop)
    console = segment_output / "console.log"
    link = output / "console.log"
    if link.is_symlink(): link.unlink()
    if not link.exists(): link.symlink_to(console)
    with console.open("x") as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    globals()["_active_child"] = child
    status.update(state="running", child_pid=child.pid)
    reason = "process_exit"
    while child.poll() is None:
        now = time.time(); active = offset + now - start
        status.update(active_elapsed_seconds=active, heartbeat_epoch=now)
        if now >= data["allocation_end_epoch"] - 720 and not pause_file.exists():
            pause_file.touch(); status["state"] = "pausing"
        atomic_json(output / "status.json", status)
        if requested or active >= duration or now >= data["allocation_end_epoch"] - 90:
            reason = "signal" if requested else "time_budget" if active >= duration else "allocation_checkpoint"
            pause_file.touch(); stop_tree(child, grace_seconds=30)
            break
        time.sleep(2)
    ended = time.time(); used = min(duration - offset, ended - start)
    state = "paused"
    location = read(output / "run_location.json") if (output / "run_location.json").exists() else {}
    outcome_file = Path(location.get("log_dir", "/nonexistent")) / "segment_outcome.json"
    if reason == "time_budget" or offset + used >= duration:
        state = "finished"
    elif child.returncode == 0 and outcome_file.exists():
        state = read(outcome_file)["state"]
    elif reason == "process_exit":
        state = "failed"
        failures = Path(location.get("log_dir", "/nonexistent")) / "phase1/study7_failures.jsonl"
        if failures.exists():
            failure = json.loads(failures.read_text().splitlines()[-1])
            state = failure["status"]
            atomic_json(segment_output / "failure.json", failure)
    if child.returncode != 0 and (checkpoint / "latest.json").exists():
        receipt = restore_checkpoint(checkpoint)
        atomic_json(segment_output / "interrupted_restore.json", receipt)
    entry = {"segment_id": segment_id, "started_epoch": start, "ended_epoch": ended, "active_offset_seconds": offset,
             "active_seconds": used, "wall_seconds": ended-start,
             "cleanup_after_budget_seconds": max(0.0, offset+ended-start-duration),
             "allocation": r["allocation"], "node": r["node"], "state": state, "stop_reason": reason}
    history.append(entry); atomic_json(history_file, history)
    status.update(state=state, active_elapsed_seconds=offset+used, remaining_active_seconds=max(0, duration-offset-used),
                  stop_reason=reason, exit_code=child.returncode, segment_finished_at=timestamp(ended))
    if state == "finished": status["finished_at"] = timestamp(ended)
    atomic_json(output / "status.json", status); atomic_json(segment_output / "status.json", status)
    print(json.dumps({"setting": setting, "state": state, "active_seconds": offset+used, "remaining_seconds": duration-offset-used}), flush=True)


if __name__ == "__main__":
    arm(Path(sys.argv[1]), sys.argv[2], sys.argv[3])
