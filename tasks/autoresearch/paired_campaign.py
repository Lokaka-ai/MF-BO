"""Durable three-hour handoff into four paired, eight-hour native workflows."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import shutil
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from engine.shared_drafts import atomic_json, sha256
from tasks.autoresearch.supervise import process_table, timestamp

SETTINGS = ("gp7_ancestry_ucb", "gp7_long_horizon_ucb", "mlevolve_original", "gp7_long_horizon_ucb_pibo")
LEGACY_SETTINGS = {
    "gp7_ancestry_ucb": "gp7_ancestry_logei",
    "gp7_long_horizon_ucb": "gp7_long_horizon_logei",
    "gp7_long_horizon_ucb_pibo": "gp7_long_horizon_pibo",
}


def setting_entry(manifest, setting):
    """Read historical registrations without renaming their directories or keys."""
    settings = manifest["settings"]
    if setting in settings:
        return settings[setting]
    return settings[LEGACY_SETTINGS.get(setting, setting)]


def read(path):
    return json.loads(Path(path).read_text())


def epoch(value):
    return datetime.fromisoformat(value).timestamp()


def live_job(job_id):
    result = subprocess.run(["squeue", "-h", "-j", str(job_id), "-o", "%T"], capture_output=True, text=True, timeout=20)
    return result.returncode == 0 and result.stdout.strip() == "RUNNING"


def allocation_remaining(job_id):
    result = subprocess.run(["squeue", "-h", "-j", str(job_id), "-o", "%L"], capture_output=True, text=True, timeout=20)
    result.check_returncode()
    value = result.stdout.strip()
    if value == "UNLIMITED":
        return float("inf")
    days, separator, clock = value.partition("-")
    if not separator:
        days, clock = "0", days
    parts = [int(part) for part in clock.split(":")]
    if len(parts) not in (2, 3):
        raise RuntimeError(f"Unrecognized allocation time remaining: {value!r}")
    return int(days) * 86400 + sum(part * 60 ** i for i, part in enumerate(reversed(parts)))


def node_command(resource, command, seconds=180):
    if (resource["node"].split(".")[0] == socket.gethostname().split(".")[0]
            and str(resource["allocation"]) == os.environ.get("SLURM_JOB_ID")):
        return command
    # A CPU-only controller exports its own task size. Do not pass those
    # resource settings into a step in a different existing allocation.
    # Absolute executable paths also avoid Slurm resolving an unrelated PATH
    # directory named "env" as the executable (the original handoff failure).
    srun = shutil.which("srun")
    if not srun:
        raise RuntimeError("Slurm srun executable is unavailable")
    return ["/usr/bin/env", "-u", "SLURM_CPUS_PER_TASK", "-u", "SLURM_TRES_PER_TASK",
            "-u", "SLURM_MEM_PER_NODE", "-u", "SLURM_MEM_PER_CPU", "-u", "SLURM_MEM_PER_GPU",
            srun, f"--jobid={resource['allocation']}", "--overlap", "--nodes=1", "--ntasks=1",
            f"--cpus-per-task={resource.get('step_cpus', 4)}", f"--time={seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}",
            f"--nodelist={resource['node']}", f"--chdir={ROOT}", *command]


def stop_arm(directory):
    directory = Path(directory).resolve()
    state = read(directory / "status.json")
    if state["hostname"].split(".")[0] != socket.gethostname().split(".")[0]:
        raise RuntimeError("Stop must execute on the owning compute node")
    if state["state"] in {"finished", "failed"}:
        return state
    pid = state["supervisor_pid"]
    table = process_table()
    if pid not in table or table[pid][2] == "Z":
        raise RuntimeError("Supervisor is gone; refusing to signal an unrelated PID")
    if state.get("supervisor_start_ticks"):
        if str(state["supervisor_start_ticks"]) != table[pid][1]:
            raise RuntimeError("Supervisor PID was reused")
    else:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode()
        if str(directory) not in command or "supervise.py" not in command:
            raise RuntimeError("Supervisor command does not match this experiment")
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 80
    while time.monotonic() < deadline:
        current = read(directory / "status.json")
        if current["state"] in {"finished", "failed"}:
            return current
        time.sleep(0.5)
    raise RuntimeError("Supervisor did not confirm cleanup")


def preflight(gpu):
    import torch
    info = torch.cuda.get_device_properties(0)
    if "A100-SXM4-80GB" not in info.name or info.total_memory < 75_000_000_000:
        raise RuntimeError(f"Need matching A100-SXM4-80GB, got {info}")
    # The launcher uses this same nonblocking lease before spawning the workflow.
    lockdir = ROOT / ".cache/gpu-locks"
    lockdir.mkdir(parents=True, exist_ok=True)
    with (lockdir / f"{gpu}.lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        x = torch.ones((128, 128), device="cuda")
        assert (x @ x).sum().item() == 128 ** 3
        torch.cuda.synchronize()
    print(json.dumps({"gpu":gpu,"node":socket.gethostname(),"model":info.name,"cpus":sorted(os.sched_getaffinity(0))}))


def monitor(campaign, state, lease):
    from tasks.autoresearch.compare import report
    manifest = read(campaign / "campaign.json")
    directories = [Path(setting_entry(manifest, name)["directory"]) for name in SETTINGS]
    while True:
        summaries = report(campaign)
        statuses = [read(d / "status.json") if (d / "status.json").exists() else {} for d in directories]
        state.update(heartbeat_at=timestamp(), summaries=[{k:s[k] for k in ("setting","state","completed","valid","best_bpb")} for s in summaries])
        atomic_json(campaign / "controller_status.json", state)
        if all(s.get("state") in {"finished","failed"} for s in statuses):
            state.update(state="finished",finished_at=timestamp())
            atomic_json(campaign / "controller_status.json", state)
            (lease / "release").touch()
            return
        time.sleep(30)


def main(campaign):
    from omegaconf import OmegaConf
    from tasks.autoresearch.audit import audit_campaign
    from tasks.autoresearch.supervise import build_command
    from tasks.autoresearch.compare import report

    campaign = campaign.resolve()
    with (campaign / "controller.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        schedule = read(campaign / "schedule.json")
        switch_at = epoch(schedule["switch_at"])
        old = Path(schedule["old_campaign"])
        old_manifest = read(old / "campaign.json")
        lease = campaign / "replacement_gpu"
        resources = {}
        for name in SETTINGS[:3]:
            value = setting_entry(old_manifest, name)
            status = read(Path(value["directory"]) / "status.json")
            resources[name] = dict(node=value["node"], gpu=value["gpu"], cpu_ids=value.get("cpu_ids"),
                                   allocation=status["slurm_job_id"], step_cpus=8 if name in SETTINGS[:2] else 4)
        overrides = campaign / "resource_overrides.json"
        if overrides.exists():
            for name, resource in read(overrides).items():
                name = next((current for current, legacy in LEGACY_SETTINGS.items() if legacy == name), name)
                if name not in SETTINGS[:3]:
                    raise ValueError(f"Unexpected resource override: {name}")
                resources[name] = resource
        state = {"state":"waiting_for_switch", "pid":os.getpid(), "hostname":socket.gethostname(),
                 "switch_at":schedule["switch_at"], "started_at":timestamp()}
        if (campaign / "campaign.json").exists():
            existing = read(campaign / "campaign.json")
            existing_settings = existing.get("settings", {})
            if all(existing_settings.get(name, existing_settings.get(LEGACY_SETTINGS.get(name), {})).get("launcher_pid") for name in SETTINGS):
                state.update(state="running_paired_comparison", resumed_controller_at=timestamp())
                return monitor(campaign, state, lease)
            raise RuntimeError("Partial launch found; refusing to duplicate already launched arms")
        # Restore the preempted exploratory piBO arm if the replacement is ready
        # before the requested stop time. It must still stop at the same deadline.
        old_pibo = Path(setting_entry(old_manifest, SETTINGS[3])["directory"])
        if not (lease / "request_01_old_pibo.json").exists():
            loc = read(old_pibo / "run_location.json")
            command = [str(ROOT / ".venv/bin/python"), str(ROOT / "tasks/autoresearch/launch.py"),
                       "--gpu", "{GPU}", "--cpus", "4", "--steps", "500", "--name", SETTINGS[3],
                       "--resume", str(Path(loc["log_dir"]).parent), "--runs-dir", str(old_pibo / "runs"),
                       "experiment_preset=gp7_long_horizon_ucb_pibo", f"run_result_file={old_pibo/'run_location.json'}"]
            recovery = campaign / "old_pibo_recovery"
            # Keep the original interrupted files untouched. This recovery has
            # its own supervisor log while using the original admitted journal.
            atomic_json(lease / "request_01_old_pibo.json", {"command":command, "output":str(recovery),
                "duration_seconds":max(1,switch_at-time.time()), "deadline_epoch":switch_at,
                "metadata":{"setting":SETTINGS[3],"old_arm_directory":str(old_pibo),"recovery":True}})
        while time.time() < switch_at:
            state["heartbeat_at"] = timestamp()
            state["seconds_until_switch"] = max(0, int(switch_at-time.time()))
            pool_status = campaign / "draft_pool/status.json"
            state["draft_pool"] = read(pool_status) if pool_status.exists() else {"state":"starting"}
            ready = lease / "ready.json"
            last_ready = read(ready) if ready.exists() else {}
            if (last_ready.get("model") == "NVIDIA A100-SXM4-80GB" and live_job(read(lease/'submission.json')["job_id"])):
                state["replacement_gpu"] = dict(last_ready, state="ready")
            else:
                state["replacement_gpu"] = {"state":"pending", **read(lease/'submission.json')}
            atomic_json(campaign / "controller_status.json", state)
            time.sleep(min(20, max(0.1,switch_at-time.time())))

        state.update(state="stopping_previous", actual_switch_at=timestamp())
        atomic_json(campaign / "controller_status.json", state)
        def stop_one(name):
            value = setting_entry(old_manifest, name)
            directory = Path(value["directory"])
            status = read(directory / "status.json")
            if status["state"] in {"finished", "failed"}:
                return {"setting":name,"returncode":0,"already_stopped":True,"finished_at":status.get("finished_at")}
            resource = dict(node=value["node"], allocation=status["slurm_job_id"],
                            step_cpus=8 if name in SETTINGS[:2] else 4)
            command = node_command(resource, [str(ROOT / ".venv/bin/python"), __file__, "stop", str(directory)])
            result = subprocess.run(command, capture_output=True, text=True, timeout=120)
            return {"setting":name,"returncode":result.returncode,"stdout":result.stdout[-5000:],"stderr":result.stderr[-2000:]}
        with ThreadPoolExecutor(3) as executor:
            stopped = list(executor.map(stop_one, SETTINGS[:3]))
        atomic_json(campaign / "previous_stop_results.json", stopped)
        if any(row["returncode"] for row in stopped):
            raise RuntimeError("Could not confirm all previous workflows stopped; refusing overlapping GPU use")
        state.update(state="waiting_for_pool_and_four_gpus")
        atomic_json(campaign / "controller_status.json", state)
        while True:
            ready = lease / "ready.json"
            pibo_recovered = campaign / "old_pibo_recovery/status.json"
            recovery_done = not pibo_recovered.exists() or read(pibo_recovered)["state"] in {"finished","failed"}
            if recovery_done and (campaign / "draft_pool/manifest.json").exists() and ready.exists() and live_job(read(ready)["job_id"]):
                break
            state["heartbeat_at"] = timestamp()
            atomic_json(campaign / "controller_status.json", state)
            time.sleep(20)
        replacement = read(lease / "ready.json")
        resources[SETTINGS[3]] = dict(node=replacement["hostname"].split(".")[0], gpu=replacement["gpu"],
                                    allocation=replacement["job_id"], cpu_ids=",".join(map(str,replacement["cpus"][:4])))
        pool_path = Path(schedule.get("draft_pool", campaign / "draft_pool/manifest.json"))
        cfg = OmegaConf.load(ROOT / "tasks/autoresearch/four_settings_8h.yaml")
        cfg.overrides["shared_draft_pool"] = str(pool_path)
        cfg_path = campaign / "paired_config.yaml"
        OmegaConf.save(cfg, cfg_path)
        for name in SETTINGS:
            arm = campaign / name; arm.mkdir(exist_ok=True)
            atomic_json(arm / "configuration_audit.json", audit_campaign(cfg, name))
            (arm / "campaign_config.yaml").write_text(OmegaConf.to_yaml(cfg))
        manifest = {"campaign":str(campaign),"created_at":timestamp(),"workflow":"full_native_with_shared_reviewed_drafts",
                    "duration_seconds":28800,"shared_draft_pool":str(pool_path),"shared_draft_pool_id":sha256(pool_path.read_text()),
                    "initial_drafts":3,"draft_cap":5,"replicates":1,"settings":{}}
        # Only start after every allocation is present and every GPU lease is free.
        checks = []
        for name, resource in resources.items():
            if not live_job(resource["allocation"]):
                raise RuntimeError(f"Allocation vanished before launch: {resource}")
            if allocation_remaining(resource["allocation"]) < int(cfg.duration_seconds) + 300:
                raise RuntimeError(f"Allocation {resource['allocation']} cannot cover the full eight-hour run plus cleanup")
            command = node_command(resource, ["/usr/bin/env", f"CUDA_VISIBLE_DEVICES={resource['gpu']}",
                str(ROOT / ".venv/bin/python"), __file__, "preflight", resource["gpu"]])
            result = subprocess.run(command, capture_output=True, text=True, timeout=120)
            checks.append({"setting":name,"returncode":result.returncode,"stdout":result.stdout,"stderr":result.stderr})
        atomic_json(campaign / "gpu_preflight.json", checks)
        if any(c["returncode"] for c in checks):
            raise RuntimeError("At least one GPU failed preflight")
        for name in SETTINGS:
            resource = resources[name]; arm = campaign / name
            manifest["settings"][name] = dict(resource, directory=str(arm))
            if name == SETTINGS[3]:
                command = build_command(cfg, name, "{GPU}", arm, resource["cpu_ids"])
                atomic_json(lease / "request_02_paired_pibo.json", {"command":command,"output":str(arm),
                    "duration_seconds":28800,"metadata":{"setting":name,"gpu":resource["gpu"],"cpus":4,"config":str(cfg_path)}})
                manifest["settings"][name]["launcher_pid"] = replacement["pid"]
            else:
                # Audits above must not make supervise.main mistake this for an
                # existing run: call the dedicated entrypoint below instead.
                command = node_command(resource, [str(ROOT / ".venv/bin/python"), __file__, "arm", str(campaign), name], seconds=29100)
                with (arm / "supervisor.log").open("a") as log:
                    proc = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                manifest["settings"][name]["launcher_pid"] = proc.pid
            atomic_json(campaign / "campaign.json", manifest)
        atomic_json(ROOT / "runs/latest_autoresearch_campaign.json", manifest)
        state.update(state="running_paired_comparison", launched_at=timestamp())
        atomic_json(campaign / "controller_status.json", state)
        error_file = campaign / "controller_error.json"
        if error_file.exists():
            error_file.rename(campaign / f"controller_error.recovered-{time.time_ns()}.json")
        monitor(campaign, state, lease)


def run_arm(campaign, setting):
    from omegaconf import OmegaConf
    from tasks.autoresearch.supervise import supervise, build_command
    campaign = Path(campaign)
    # The manifest is written atomically immediately after each spawn.
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            resource = read(campaign / "campaign.json")["settings"][setting]
            break
        except (OSError, KeyError): time.sleep(0.1)
    else: raise RuntimeError("Arm missing from campaign manifest")
    cfg = OmegaConf.load(campaign / "paired_config.yaml")
    output = campaign / setting
    command = build_command(cfg, setting, resource["gpu"], output, resource["cpu_ids"])
    return supervise(command, output, int(cfg.duration_seconds), {"setting":setting,"gpu":resource["gpu"],"cpus":4})


if __name__ == "__main__":
    action = sys.argv[1]
    if action == "stop":
        print(json.dumps(stop_arm(sys.argv[2])))
    elif action == "preflight":
        preflight(sys.argv[2])
    elif action == "arm":
        raise SystemExit(run_arm(sys.argv[2], sys.argv[3]))
    elif action == "run":
        directory = Path(sys.argv[2])
        try:
            main(directory)
        except BaseException as exc:
            atomic_json(directory / "controller_error.json", {"type":type(exc).__name__,"error":str(exc),"at":timestamp()})
            status_path = directory / "controller_status.json"
            state = read(status_path) if status_path.exists() else {}
            state.update(state="failed", failed_at=timestamp(), error=str(exc))
            atomic_json(status_path, state)
            raise
    else:
        raise ValueError("Unknown command")
