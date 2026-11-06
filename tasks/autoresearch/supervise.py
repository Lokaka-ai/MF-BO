"""Run one comparison arm with a durable, independent wall-clock deadline."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def timestamp(epoch=None):
    return datetime.fromtimestamp(time.time() if epoch is None else epoch, timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def process_table():
    result = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rpartition(")")[2].split()
            result[int(path.parent.name)] = (int(fields[1]), fields[19], fields[0])
        except (OSError, ValueError, IndexError):
            continue
    return result


def descendants(root, table):
    members = {root}
    while True:
        expanded = members | {pid for pid, (parent, _, _) in table.items() if parent in members}
        if members == expanded:
            return {pid: table[pid][1] for pid in members if pid in table}
        members = expanded


def live_members(members, table=None):
    table = process_table() if table is None else table
    return {pid: start for pid, start in members.items()
            if pid in table and table[pid][1] == start and table[pid][2] != "Z"}


def signal_members(members, sig):
    for pid in live_members(members):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


def stop_tree(proc, grace_seconds=30):
    """Let run.py persist its journal, then clean up even detached CUDA children."""
    known = descendants(proc.pid, process_table())
    if proc.poll() is None:
        try:
            proc.send_signal(signal.SIGINT)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        known.update(descendants(proc.pid, process_table()))
        if proc.poll() is not None and not live_members(known):
            return proc.returncode
        time.sleep(0.2)
    signal_members(known, signal.SIGTERM)
    deadline = time.monotonic() + 5
    while live_members(known) and time.monotonic() < deadline:
        known.update(descendants(proc.pid, process_table()))
        time.sleep(0.1)
    signal_members(known, signal.SIGKILL)
    return proc.wait(timeout=10)


def supervise(command, directory, duration_seconds, metadata=None, grace_seconds=30, start_epoch=None):
    directory.mkdir(parents=True, exist_ok=True)
    status_path = directory / "status.json"
    if status_path.exists():
        raise FileExistsError(f"Use a new output directory; existing status at {status_path}")
    requested_stop = []

    def request_stop(signum, _frame):
        requested_stop.append(signum)

    old_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    started = time.time() if start_epoch is None else start_epoch
    started_monotonic = time.monotonic() - max(0, time.time() - started)
    status = dict(metadata or {})
    status.update(state="starting", hostname=socket.gethostname(), supervisor_pid=os.getpid(),
                  started_at=timestamp(started), deadline_at=timestamp(started + duration_seconds),
                  duration_seconds=duration_seconds, command=command,
                  slurm_job_id=os.environ.get("SLURM_JOB_ID"), slurm_step_id=os.environ.get("SLURM_STEP_ID"))
    own_identity = process_table().get(os.getpid())
    if own_identity:
        status["supervisor_start_ticks"] = own_identity[1]
    write_json(status_path, status)
    proc = None
    try:
        with (directory / "console.log").open("x") as output:
            proc = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                    stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            status.update(state="running", child_pid=proc.pid)
            write_json(status_path, status)
            print(f"Started {status.get('setting', 'run')} on {status['hostname']}: "
                  f"pid={proc.pid}, deadline={status['deadline_at']}", flush=True)
            while proc.poll() is None:
                if requested_stop or time.monotonic() - started_monotonic >= duration_seconds:
                    reason = "signal" if requested_stop else "time_budget"
                    status.update(state="stopping", stop_reason=reason)
                    write_json(status_path, status)
                    print(f"Stopping run: {reason}; saving journal and releasing GPU", flush=True)
                    stop_tree(proc, grace_seconds)
                    break
                time.sleep(min(1, max(0.01, duration_seconds - (time.monotonic() - started_monotonic))))
            status.update(state="finished", exit_code=proc.returncode, finished_at=timestamp(),
                          elapsed_seconds=time.monotonic() - started_monotonic)
            status.setdefault("stop_reason", "process_exit")
            write_json(status_path, status)
            print(json.dumps(status, indent=2), flush=True)
            return 0 if status["stop_reason"] == "time_budget" else proc.returncode
    except BaseException as exc:
        if proc is not None and proc.poll() is None:
            stop_tree(proc, grace_seconds)
        status.update(state="failed", error=f"{type(exc).__name__}: {exc}", finished_at=timestamp())
        write_json(status_path, status)
        raise
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


def build_command(cfg, setting, gpu, directory, cpu_ids=None):
    if setting not in cfg.settings:
        raise ValueError(f"Setting is not enabled in this campaign: {setting}")
    seconds, trials, cpus = int(cfg.duration_seconds), int(cfg.max_trials), int(cfg.cpus)
    if min(seconds, trials, cpus) <= 0:
        raise ValueError("Duration, trial count and CPU count must be positive")
    command = [str(ROOT / ".venv" / "bin" / "python"),
               str(ROOT / "tasks" / "autoresearch" / "launch.py"),
               "--gpu", gpu, "--cpus", str(cpus), "--steps", str(trials),
               "--name", setting, "--runs-dir", str(directory / "runs")]
    if cpu_ids:
        command.extend(["--cpu-ids", cpu_ids])
    command.extend([f"experiment_preset={setting}", f"agent.time_limit={seconds}",
                    f"cpu_number={cpus}", f"run_result_file={directory / 'run_location.json'}"])
    command.extend(f"{key}={json.dumps(value)}" for key, value in cfg.overrides.items())
    return command


def main():
    from omegaconf import OmegaConf
    from audit import audit_campaign

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("three_settings_8h.yaml"))
    parser.add_argument("--setting", required=True)
    parser.add_argument("--gpu", required=True, help="Physical GPU UUID or index")
    parser.add_argument("--cpu-ids", help="Allocated CPU IDs; default is the first --cpus allocated CPUs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    output = args.output.resolve()
    command = build_command(cfg, args.setting, args.gpu, output, args.cpu_ids)
    if output.exists() and any(output.iterdir()):
        # The launching shell may already have opened supervisor.log.
        if any(p.name != "supervisor.log" for p in output.iterdir()):
            parser.error("Use a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "configuration_audit.json", audit_campaign(cfg, args.setting))
    (output / "campaign_config.yaml").write_text(OmegaConf.to_yaml(cfg))
    metadata = {"setting": args.setting, "gpu": args.gpu, "cpus": int(cfg.cpus),
                "cpu_ids_requested": args.cpu_ids, "config": str(args.config.resolve()),
                "python": sys.executable}
    raise SystemExit(supervise(command, output, int(cfg.duration_seconds), metadata))


if __name__ == "__main__":
    main()
