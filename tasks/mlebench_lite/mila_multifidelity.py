#!/usr/bin/env python3
"""Prepare, submit, run and collect the approved 14-task, two-mode campaign.

The login-node commands use only Python's standard library. Preparation and
all model calls/training/grading run in Slurm allocations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import signal
import time
import math
from datetime import datetime, timezone
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "config/mlebench_mf14.json"
PYTHON = ROOT / ".venv-mlebench/bin/python"
SCRIPT = Path(__file__).resolve()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(argv, **kwargs):
    return subprocess.run([str(x) for x in argv], cwd=ROOT, check=True, **kwargs)


def source_hashes():
    packaged = ROOT / "SOURCE_SHA256.json"
    if packaged.exists():
        expected = read(packaged)
        actual = {name: sha(ROOT / name) for name in expected}
        if actual != expected:
            raise RuntimeError("Packaged source changed; rebuild the package before a new campaign")
        return actual
    names = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("Commit source changes before preparing the campaign")
    return {name: sha(ROOT / name) for name in names if (ROOT / name).is_file()}


def environment(settings):
    env = dict(os.environ)
    env.update(OPENCODE_API_KEY_FILE=settings["key_file"], MLEVOLVE_OPENCODE_REASONING_EFFORT="high",
               MLEVOLVE_MF_RECOVERY="1",
               MLEVOLVE_CODEX_REASONING_EFFORT="high", MLEVOLVE_OPENCODE_TIMEOUT_SECONDS="1800",
               PYTHONUNBUFFERED="1", PYTHONPATH=str(ROOT), USE_TF="0",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               HF_HOME=str(ROOT / ".cache/mlebench-lite/huggingface"),
               TORCH_HOME=str(ROOT / ".cache/mlebench-lite/torch"))
    if settings["spec"]["provider"] == "codex://opencode-go":
        env.pop("OPENCODE_API_KEY", None)  # Use the explicit campaign credential file.
        env["MLEVOLVE_CODEX_TIMEOUT_SECONDS"] = "1800"
        env["MLEVOLVE_CODEX_BIN"] = str(ROOT / "tools/codex_opencode.py")
        env["MLEVOLVE_CODEX_REASONING_EFFORT"] = settings["spec"]["reasoning_effort"]
    # The legacy lease supervisor explicitly forbids fixed evaluation.
    for key in ("MLEVOLVE_COMPETITION_CHECKPOINT", "MLEVOLVE_COMPETITION_PAUSE_FILE",
                "MLEVOLVE_COMPETITION_LEASE_END", "MLEVOLVE_SEGMENTED_RUN"):
        env.pop(key, None)
    return env


@contextmanager
def submission_lock(campaign):
    import fcntl
    campaign.mkdir(parents=True, exist_ok=True)
    with (campaign / "submit.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def sbatch(campaign, key, argv):
    ledger = campaign / "jobs.json"
    jobs = read(ledger) if ledger.exists() else {}
    if key in jobs:
        if jobs[key].get("job_id"):
            print(f"Already submitted {key}: {jobs[key]['job_id']}")
            return
        raise RuntimeError(f"Unresolved submission for {key}; inspect jobs.json and Slurm before retrying")
    jobs[key] = {"state": "submitting", "command": argv}
    save(ledger, jobs)
    result = run(argv, capture_output=True, text=True)
    job_id = result.stdout.strip().split(";")[0]
    if not job_id.isdigit():
        raise RuntimeError("Slurm returned no unambiguous job ID; inspect jobs.json")
    jobs[key].update(state="submitted", job_id=job_id)
    save(ledger, jobs)
    print(f"Submitted {key}: {job_id}", flush=True)


def batch_command(campaign, partition, cpus, memory, walltime, arguments, gpu=False):
    logs = campaign / "slurm"
    logs.mkdir(parents=True, exist_ok=True)
    spec = read(SPEC)
    command = ["sbatch", "--parsable", "--nodes=1", "--ntasks=1", "--no-requeue",
               f"--partition={partition}", f"--cpus-per-task={cpus}", f"--mem={memory}G",
               f"--time={walltime}", f"--chdir={ROOT}", f"--output={logs}/%x-%j.log",
               "--export=ALL", f"--job-name=mf-{'-'.join(arguments[:1])}"]
    if gpu:
        command.append(f"--gres={spec['gpu_gres']}")
    if arguments[0] == "worker":
        command.append("--signal=B:USR1@60")
    command.append("--wrap=exec " + shlex.join([sys.executable, str(SCRIPT), *arguments,
                                          "--campaign", str(campaign)]))
    return command


def submit_preparation(args):
    campaign = args.campaign
    for path in (args.key_file, args.token_file):
        if not path.is_file():
            raise ValueError(f"Credential file does not exist: {path}")
    spec = read(SPEC)
    selected = getattr(args, "tasks", None)
    if selected is not None:
        if not selected or len(set(selected)) != len(selected) or set(selected) - set(spec["tasks"]):
            raise ValueError("Provide unique tasks from the supported manifest")
        # Freeze the selected scope in settings; all later stages use this map.
        spec["tasks"] = {task: resources for task, resources in spec["tasks"].items() if task in selected}
    settings = {"key_file": str(args.key_file.resolve()), "token_file": str(args.token_file.resolve()),
                "data_dir": str(args.data_dir.resolve()), "spec": spec, "source": source_hashes()}
    if getattr(args, "protocol_source", None) is not None:
        from tasks.mlebench_lite.mfgp_protocols import register_source
        settings["protocol_source"] = register_source(args.protocol_source, settings["spec"]["tasks"])
    with submission_lock(campaign):
        path = campaign / "settings.json"
        if path.exists() and read(path) != settings:
            raise ValueError("Campaign settings already differ; use a new campaign directory")
        save(path, settings)
        sbatch(campaign, "prepare", batch_command(campaign, args.partition, 4, 64,
                                                 "24:00:00", ["prepare"]))


def verify_codex(settings, output):
    binary = os.environ.get("MLEVOLVE_REAL_CODEX_BIN") or shutil.which("codex")
    if not binary:
        raise RuntimeError("Codex CLI is missing")
    result = run([binary, "--version"], capture_output=True, text=True)
    version = result.stdout.strip()
    expected = "codex-cli " + settings["spec"]["codex_version"]
    if version != expected:
        raise RuntimeError(f"Codex version differs: expected {expected}, got {version}")
    save(output, {"binary": str(Path(binary).resolve()), "version": version,
                  "sha256": sha(Path(binary).resolve())})


def provider_check_command(settings, output):
    spec = settings["spec"]
    return [PYTHON, "tools/check_go_provider.py", "--model", spec["model"],
            "--provider", spec["provider"], "--output", output]


def require_allocation():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("This command must execute inside a Slurm allocation")


def prepare(args):
    require_allocation()
    settings = read(args.campaign / "settings.json")
    if settings["source"] != source_hashes():
        raise RuntimeError("Campaign source changed")
    missing = [name for name in ("uv", "git", "unzip", "7z", "ffmpeg") if not shutil.which(name)]
    codex = os.environ.get("MLEVOLVE_REAL_CODEX_BIN") or shutil.which("codex")
    if not codex or not Path(codex).is_file():
        missing.append("codex (or MLEVOLVE_REAL_CODEX_BIN)")
    if missing:
        raise RuntimeError("Load/install prerequisites before preparation: " + ", ".join(missing))
    verify_codex(settings, args.campaign / "codex-runtime.json")
    run(["git", "lfs", "version"])
    env = environment(settings)
    run(["bash", "tasks/mlebench_lite/setup.sh"], env=env)
    run([PYTHON, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_multifidelity.py",
         "tests/test_multifidelity_campaign.py", "tests/test_study7_math.py", "tests/test_fixed_evaluation.py",
         "tests/test_acq7_math.py", "tests/test_multifidelity_integration.py", "tests/test_mf_recovery.py",
         "tests/test_mila_failure_regressions.py",
         "tests/test_go_campaign.py", "tests/test_codex_retry.py", "tests/test_kaggle_download.py",
         *(["tests/test_mf_gp.py", "tests/test_mfgp_audit.py", "tests/test_mfgp_campaign.py"] if "mf_gp" in settings["spec"]["modes"] else [])], env=env)
    run([PYTHON, "tasks/mlebench_lite/check_environment.py", "--embedding",
         "--output", args.campaign / "cpu-environment.json"], env=env)
    run(provider_check_command(settings, args.campaign / "provider-check.json"), env=env)
    spec = settings["spec"]
    run([PYTHON, "tasks/mlebench_lite/prepare.py", "--data-dir", settings["data_dir"],
         "--token-file", settings["token_file"], "--workers", "3", "--tasks", *spec["tasks"]], env=env)
    protocols = {}
    for task in spec["tasks"]:
        if "mf_gp" in spec["modes"]:
            from tasks.mlebench_lite.mfgp_protocols import attach_source
            attach_source(settings, args.campaign, task)
        directory = args.campaign / "protocols" / task
        run([PYTHON, "tasks/mlebench_lite/prepare_fixed_protocol.py", "--task", task,
             "--data-root", settings["data_dir"], "--seed", str(spec["seed"]),
             "--output-root", args.campaign / "protocol_setup" / task,
             "--protocol-dir", directory, "--model", spec["model"],
             "--provider", spec["provider"]], env=env)
        protocol = read(directory / "protocol.json")
        protocols[task] = {key: protocol[key] for key in ("protocol_id", "metric", "split")}
    save(args.campaign / "ready.json", {"settings_sha256": sha(args.campaign / "settings.json"),
                                       "protocols": protocols})
    print("Preparation passed. Submit the short GPU preflight before any full search.")


def ready(campaign):
    record = read(campaign / "ready.json")
    if record["settings_sha256"] != sha(campaign / "settings.json"):
        raise RuntimeError("Prepared campaign settings changed")
    settings = read(campaign / "settings.json")
    if settings["source"] != source_hashes():
        raise RuntimeError("Prepared campaign source changed")
    return settings


def validate_gpu_partition(partition, resources):
    # The documented Mila per-user main/unkillable memory limits are below
    # the larger pilot request. Refuse a partial multi-job submission.
    if resources["memory_gb"] > 48 and partition not in {"long", "long-grace"}:
        raise ValueError(
            f"{resources['memory_gb']} GB host RAM requires a long partition; "
            f"got {partition!r}"
        )


def submit(args):
    settings = ready(args.campaign)
    gpu_ready = read(args.campaign / "gpu-ready.json")
    if gpu_ready.get("settings_sha256") != sha(args.campaign / "settings.json"):
        raise RuntimeError("Short GPU preflight has not passed for these campaign settings")
    spec = settings["spec"]
    tasks = args.tasks or list(spec["tasks"])
    if set(tasks) - set(spec["tasks"]):
        raise ValueError("Task is not in this prepared campaign")
    for task in tasks:
        validate_gpu_partition(args.partition, spec["tasks"][task])
    with submission_lock(args.campaign):
        for task in tasks:
            resources = spec["tasks"][task]
            for mode in spec["modes"]:
                command = batch_command(args.campaign, args.partition, resources["cpus"],
                    resources["memory_gb"], spec["slurm_time"], ["worker", "--task", task, "--mode", mode], gpu=True)
                sbatch(args.campaign, f"{task}/{mode}", command)


def submit_preflight(args):
    ready(args.campaign)
    with submission_lock(args.campaign):
        sbatch(args.campaign, "gpu-preflight", batch_command(
            args.campaign, args.partition, 4, 24, "00:30:00", ["preflight"], gpu=True))


def preflight(args):
    require_allocation()
    if Path(sys.prefix).resolve() != PYTHON.parent.parent.resolve():
        os.execv(str(PYTHON), [str(PYTHON), str(SCRIPT), *sys.argv[1:]])
    settings = ready(args.campaign)
    env = environment(settings)
    import torch
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Preflight requires one Slurm-visible GPU")
    props = torch.cuda.get_device_properties(0)
    if "A100" not in props.name or props.total_memory < 70 * 1024**3:
        raise RuntimeError("Preflight requires A100 80GB")
    model = torch.nn.Linear(16, 1).cuda()
    optimizer = torch.optim.Adam(model.parameters())
    features = torch.randn(64, 16, device="cuda")
    targets = features.sum(dim=1, keepdim=True)
    for _ in range(10):
        optimizer.zero_grad()
        loss = (model(features) - targets).square().mean()
        loss.backward()
        optimizer.step()
    if not torch.isfinite(loss).item():
        raise RuntimeError("CUDA training preflight produced nonfinite loss")
    run([PYTHON, "tasks/mlebench_lite/check_environment.py", "--require-cuda", "--embedding",
         "--output", args.campaign / "gpu-environment.json"], env=env)
    verify_codex(settings, args.campaign / "gpu-codex-runtime.json")
    run(provider_check_command(settings, args.campaign / "gpu-provider-check.json"), env=env)
    run([PYTHON, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         "tests/test_multifidelity_integration.py", "tests/test_mf_recovery.py",
         *(["tests/test_mf_gp.py", "tests/test_mfgp_audit.py", "tests/test_mfgp_campaign.py"] if "mf_gp" in settings["spec"]["modes"] else [])], env=env)
    save(args.campaign / "gpu-ready.json", {
        "settings_sha256": sha(args.campaign / "settings.json"),
        "job_id": os.environ["SLURM_JOB_ID"], "gpu": props.name,
        "cuda_training_loss": loss.item(),
        "checks": "CUDA training/embedding, live provider, both mode integration/runner and process-kill recovery"})


def search_command(settings, campaign, task, mode, python=PYTHON):
    spec = settings["spec"]
    resources = spec["tasks"][task]
    output = campaign / "tasks" / task / mode
    return [str(python), "-u", "run.py", f"exp_id={task}", f"exp_name={task}_{mode}",
        f"dataset_dir={settings['data_dir']}", f"data_dir={settings['data_dir']}/{task}/prepared/public",
        f"desc_file={settings['data_dir']}/{task}/prepared/public/description.md",
        f"log_dir={output / 'runs'}", f"workspace_dir={output / 'runs'}",
        f"run_result_file={output / 'run-info.json'}", "experiment_preset=acq7_aq05_gp4_level15_rating_kg",
        "audited_bo=redesign", "evaluation.enabled=true", f"evaluation.multifidelity_mode={mode}",
        f"evaluation.protocol_dir={campaign / 'protocols' / task}",
        f"agent.time_limit={spec['time_limit_seconds']}", f"agent.steps={spec['steps']}",
        f"agent.initial_drafts={spec['initial_drafts']}", f"agent.seed={spec['seed']}",
        f"agent.search.parallel_search_num={spec['workers']}", "agent.search.num_gpus=1",
        f"exec.timeout={spec['candidate_timeout_seconds']}", f"cpu_number={resources['cpus']}",
        *[f"agent.{stage}.{key}={spec[value]}" for stage in ("code", "feedback")
          for key, value in (("model", "model"), ("base_url", "provider"))]]


def released_job(job_id):
    """Require scheduler evidence, never trust a stale local PID or elapsed time."""
    if not str(job_id).isdigit():
        raise RuntimeError("Invalid previous Slurm job ID")
    active = run(["squeue", "--noheader", "--format=%i", "--user", os.environ["USER"]], capture_output=True, text=True)
    if str(job_id) in active.stdout.split():
        raise RuntimeError("Previous allocation is still queued/running/completing")
    result = run(["sacct", "-X", "-n", "-P", "-j", job_id, "--format=JobIDRaw,State,End"],
                 capture_output=True, text=True, env={**os.environ, "TZ": "UTC"})
    rows = [line.split("|") for line in result.stdout.splitlines() if line.split("|")[0] == str(job_id)]
    if len(rows) != 1:
        raise RuntimeError("Slurm accounting has not published an unambiguous final record; retry later")
    _, state, end, *_ = rows[0]
    state = state.split()[0].rstrip("+")
    if state not in {"PREEMPTED", "TIMEOUT", "CANCELLED", "NODE_FAIL", "FAILED"}:
        raise RuntimeError(f"Job is not a recoverable interruption: {state}")
    ended = datetime.fromisoformat(end).replace(tzinfo=timezone.utc).timestamp()
    return {"job_id": str(job_id), "state": state, "ended_epoch": ended}


def recovery_record(output, release):
    segment = read(output / "segment.json")
    if segment["job_id"] != release["job_id"]:
        raise RuntimeError("Release evidence does not match the last search segment")
    result = read(output / "result.json") if (output / "result.json").exists() else {}
    if result.get("state") == "complete":
        raise RuntimeError("Completed runs cannot be resumed")
    if release["state"] == "FAILED" and result.get("state") != "interrupted":
        raise RuntimeError("A program/infrastructure failure needs investigation, not automatic replay")
    logs = Path(read(output / "run-info.json")["log_dir"])
    journal = read(logs / "journal.json")
    if journal.get("evaluation_state", {}).get("failure"):
        raise RuntimeError("Shared evaluator failure cannot be resumed")
    # Stop charging at the process exit if the supervisor recorded it; otherwise
    # Slurm's final End is a conservative upper bound. Queue downtime is excluded.
    ended = min(release["ended_epoch"], segment.get("ended_epoch", release["ended_epoch"]))
    heartbeat = logs / "competition_heartbeat.json"
    clock_start = segment["started_epoch"] - segment.get("active_offset", 0.)
    if heartbeat.exists():
        beat = read(heartbeat)
        if beat["updated_epoch"] >= segment["started_epoch"]:
            clock_start = beat["agent_start_epoch"]
    elapsed = max(journal.get("elapsed_active_seconds", 0.), ended - clock_start)
    return {**release, "ended_epoch": ended, "elapsed_seconds": math.ceil(elapsed),
            "run_dir": str(logs.parent), "source_job": segment["job_id"]}


def reconcile_recovery(output, recovery):
    logs = Path(recovery["run_dir"]) / "logs"
    ledger = logs / "phase1/root_draft_state.json"
    state = read(ledger)
    state["executor_release_receipt"] = {"owner_host": state["owner_host"],
        "owner_pid": state["owner_pid"], "job_id": recovery["source_job"],
        "verified_by": "slurm_allocation_release"}
    save(ledger, state)
    for path in (logs / "mf_execution").glob("*.json"):
        record = read(path)
        if record["state"] == "running":
            record["elapsed_seconds"] += max(0., recovery["ended_epoch"] - record["started_epoch"])
            record["state"] = "interrupted"
            save(path, record)


def resume(args):
    settings = ready(args.campaign)
    if args.task not in settings["spec"]["tasks"] or args.mode not in settings["spec"]["modes"]:
        raise ValueError("Provide the registered --task and --mode to recover")
    output = args.campaign / "tasks" / args.task / args.mode
    with submission_lock(args.campaign):
        segment = read(output / "segment.json")
        recovery = recovery_record(output, released_job(segment["job_id"]))
        resources = settings["spec"]["tasks"][args.task]
        validate_gpu_partition(args.partition, resources)
        save(output / "recovery.json", recovery)
        command = batch_command(args.campaign, args.partition, resources["cpus"], resources["memory_gb"],
            settings["spec"]["slurm_time"], ["worker", "--task", args.task, "--mode", args.mode, "--resume-run"], gpu=True)
        sbatch(args.campaign, f"{args.task}/{args.mode}/resume-after-{segment['job_id']}", command)


def run_search(command, env, output, active_offset=0):
    """Keep the supervisor alive through TERM long enough to save and release."""
    interrupted = []
    proc = None
    def pause(signum, frame):
        interrupted.append(time.monotonic())
        if proc is not None and proc.poll() is None:
            try:
                proc.send_signal(signal.SIGUSR1)
            except ProcessLookupError:
                pass
    prior = {s: signal.signal(s, pause) for s in (signal.SIGTERM, signal.SIGUSR1)}
    record = {"job_id": os.environ["SLURM_JOB_ID"], "started_epoch": time.time(), "active_offset": active_offset}
    save(output / "segment.json", record)
    try:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True)
        while proc.poll() is None:
            if interrupted and time.monotonic() - interrupted[0] > 20:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                break
            time.sleep(.2)
        code = proc.wait()
        record.update(ended_epoch=time.time(), returncode=code)
        save(output / "segment.json", record)
        save(output / "segments" / f"{record['job_id']}.json", record)
        if interrupted or code in (75, -signal.SIGTERM, -signal.SIGKILL):
            save(output / "result.json", {"state": "interrupted", "job_id": record["job_id"]})
            raise SystemExit(75)
        if code:
            raise subprocess.CalledProcessError(code, command)
    finally:
        if proc is not None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
        for sig, handler in prior.items():
            signal.signal(sig, handler)


def official_grade(logs, data_dir, env, output):
    grade = subprocess.run(
        [str(PYTHON), "-m", "utils.grade_submission", "--log-dir", str(logs),
         "--data-dir", str(data_dir)],
        cwd=ROOT, env=env, capture_output=True, text=True,
    )
    # Capture the actual grader exception before raising. The previous worker
    # discarded stderr, leaving a failed grade indistinguishable from OOM.
    diagnostic = output / "official-grade-process.json"
    save(diagnostic, {"returncode": grade.returncode,
                      "stdout": grade.stdout, "stderr": grade.stderr})
    if grade.returncode:
        raise RuntimeError(f"Official grading exited {grade.returncode}; see {diagnostic}")
    return json.loads(grade.stdout)


def validate_terminal_ledger(journal, mode):
    state = journal.get("parent_selector_state") or {}
    pending = state.get("pending") or {}
    if pending:
        raise RuntimeError(
            f"Search returned with {len(pending)} unresolved decisions; "
            "do not grade or report this run as complete"
        )
    if mode == "mf_gp":
        selector = state.get("study7_state") or {}
        promotions = selector.get("pending_promotions") or {}
        if promotions:
            raise RuntimeError(
                f"MF-GP returned with {len(promotions)} unresolved promotions"
            )


def worker(args):
    require_allocation()
    # sbatch may invoke system Python; re-exec into the prepared environment.
    if Path(sys.prefix).resolve() != PYTHON.parent.parent.resolve():
        os.execv(str(PYTHON), [str(PYTHON), str(SCRIPT), *sys.argv[1:]])
    settings = ready(args.campaign)
    spec = settings["spec"]
    if args.task not in spec["tasks"] or args.mode not in spec["modes"]:
        raise ValueError("Unregistered task/mode")
    env = environment(settings)
    output = args.campaign / "tasks" / args.task / args.mode
    output.mkdir(parents=True, exist_ok=True)
    if (output / "run-info.json").exists() and not args.resume_run:
        raise RuntimeError("Existing run found; automatic restarting/re-budgeting is prohibited")
    try:
        verify_codex(settings, output / "segments" / f"{os.environ['SLURM_JOB_ID']}-codex-runtime.json")
        import torch
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Each run requires exactly one Slurm-visible GPU")
        props = torch.cuda.get_device_properties(0)
        if "A100" not in props.name or props.total_memory < 70 * 1024**3:
            raise RuntimeError("Historical owner hardware was A100 80GB; allocation differs")
        if len(os.sched_getaffinity(0)) < spec["tasks"][args.task]["cpus"]:
            raise RuntimeError("Insufficient allocated CPU cores")
        save(output / "hardware.json", {"gpu": props.name, "gpu_bytes": props.total_memory,
            "cpu_affinity": sorted(os.sched_getaffinity(0)), "job_id": os.environ["SLURM_JOB_ID"]})
        run([PYTHON, "tasks/mlebench_lite/check_environment.py", "--require-cuda", "--embedding",
             "--output", output / "environment.json"], env=env)
        for name in ("hardware", "environment"):
            save(output / "segments" / f"{os.environ['SLURM_JOB_ID']}-{name}.json",
                 read(output / f"{name}.json"))
        from types import SimpleNamespace
        sys.path.insert(0, str(ROOT))
        from tasks.mlebench_lite.run_bo import audit_protocol
        audit_protocol(SimpleNamespace(protocol_dir=args.campaign / "protocols" / args.task,
                                       data_root=Path(settings["data_dir"]), task=args.task, seed=spec["seed"]))
        prepared = read(args.campaign / "ready.json")["protocols"][args.task]
        if read(args.campaign / "protocols" / args.task / "protocol.json")["protocol_id"] != prepared["protocol_id"]:
            raise RuntimeError("Protocol differs from the prepared campaign")
        command = search_command(settings, args.campaign, args.task, args.mode)
        active_offset = 0
        if args.resume_run:
            recovery = read(output / "recovery.json")
            # Verify again when the replacement allocation actually starts.
            released_job(recovery["source_job"])
            reconcile_recovery(output, recovery)
            command += [f"resume_run_dir={recovery['run_dir']}", f"resume_elapsed_seconds={recovery['elapsed_seconds']}"]
            active_offset = recovery["elapsed_seconds"]
        save(output / "command.json", command)
        save(output / "segments" / f"{os.environ['SLURM_JOB_ID']}-command.json", command)
        run_search(command, env, output, active_offset)
        logs = Path(read(output / "run-info.json")["log_dir"])
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(logs / "config.yaml")
        expected = {arg.split("=", 1)[0]: arg.split("=", 1)[1] for arg in command[3:]
                    if "=" in arg and not arg.startswith(("experiment_preset=", "exp_name=", "log_dir=", "workspace_dir="))}
        for key, value in expected.items():
            actual = OmegaConf.select(cfg, key)
            if str(actual).lower() != value.lower():
                raise RuntimeError(f"Saved configuration differs at {key}")
        journal = read(logs / "journal.json")
        validate_terminal_ledger(journal, args.mode)
        records = (logs / "phase1/transitions.jsonl").read_text().splitlines()
        ended_before_limit = (journal["elapsed_active_seconds"] < spec["time_limit_seconds"]
                              and len(records) < spec["steps"])
        final = read(logs / "final_submission.json")
        node = next(n for n in journal["nodes"] if n["id"] == final["node_id"])
        if args.mode == "mf_gp":
            observations = journal["parent_selector_state"]["study7_state"]["observations"]
            winner = next(o for o in observations if o["transition_id"] == final["evaluation_id"])
            if (winner["program_id"] != node["id"] or winner["fidelity"] != 1.0
                    or not winner["valid"] or not winner.get("budget_eligible", True)
                    or winner["result"]["metric"] != final["metric"]):
                raise RuntimeError("Winner is not a completed full-data MF evaluation")
            completed_actions = len(records) + sum(o["kind"] == "promotion" for o in observations)
            ended_before_limit = (journal["elapsed_active_seconds"] < spec["time_limit_seconds"]
                                  and completed_actions < spec["steps"])
        elif node.get("training_fraction") != 1.0:
            raise RuntimeError("Winner was not trained on full T")
        report = official_grade(logs, settings["data_dir"], env, output)
        if report.get("report") is None:
            raise RuntimeError("Official grading returned no report")
        save(output / "official-grade.json", report)
        save(output / "result.json", {"state": "complete", "log_dir": str(logs),
             "active_seconds": journal["elapsed_active_seconds"], "transitions": len(records),
             "ended_before_limit": ended_before_limit,
             "submission_sha256": sha(final["submission"]), "official_grade": report})
    except Exception as exc:
        save(output / "result.json", {"state": "failed", "error": str(exc), "type": type(exc).__name__})
        raise


def collect(args):
    args.campaign.mkdir(parents=True, exist_ok=True)
    settings_path = args.campaign / "settings.json"
    spec = read(settings_path)["spec"] if settings_path.exists() else read(SPEC)
    rows = []
    jobs = read(args.campaign / "jobs.json") if (args.campaign / "jobs.json").exists() else {}
    for task in spec["tasks"]:
        for mode in spec["modes"]:
            path = args.campaign / "tasks" / task / mode / "result.json"
            result = read(path) if path.exists() else {"state": "no_result"}
            rows.append({"task": task, "mode": mode, "job_id": jobs.get(f"{task}/{mode}", {}).get("job_id", ""),
                         "state": result["state"], "active_seconds": result.get("active_seconds", ""),
                         "ended_before_limit": result.get("ended_before_limit", ""),
                         "error": result.get("error", ""), "result_file": str(path)})
    with (args.campaign / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


def main(*, modes=("carry_over", "reset"), default_campaign=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("submit-prepare", "prepare", "submit-preflight", "preflight", "submit", "resume", "worker", "collect"))
    parser.add_argument("--campaign", type=Path, default=default_campaign or ROOT / "runs/mf14_aq05")
    if "mf_gp" in modes:
        parser.add_argument("--protocol-source", type=Path,
            help="Existing baseline campaign's protocols directory; reuse its exact T/V/scorer")
    parser.add_argument("--partition", help="Your available Slurm partition; required for submissions")
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/mlebench")
    parser.add_argument("--tasks", nargs="+", help="Freeze preparation scope, or select already prepared tasks for submission")
    parser.add_argument("--task")
    parser.add_argument("--mode", choices=modes)
    parser.add_argument("--resume-run", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.campaign = args.campaign.resolve()
    if (args.command.startswith("submit") or args.command == "resume") and not args.partition:
        parser.error("Provide --partition for your Mila account")
    if args.command == "submit-prepare" and (args.key_file is None or args.token_file is None):
        parser.error("Provide --key-file and --token-file paths (never raw credentials)")
    {"submit-prepare": submit_preparation, "prepare": prepare, "submit": submit,
     "submit-preflight": submit_preflight, "preflight": preflight,
     "resume": resume,
     "worker": worker, "collect": collect}[args.command](args)


if __name__ == "__main__":
    main()
