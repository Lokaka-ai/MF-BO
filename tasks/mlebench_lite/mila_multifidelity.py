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
               MLEVOLVE_CODEX_REASONING_EFFORT="high", MLEVOLVE_OPENCODE_TIMEOUT_SECONDS="1800",
               PYTHONUNBUFFERED="1", PYTHONPATH=str(ROOT), USE_TF="0",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               HF_HOME=str(ROOT / ".cache/mlebench-lite/huggingface"),
               TORCH_HOME=str(ROOT / ".cache/mlebench-lite/torch"))
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
    command.append("--wrap=" + shlex.join([sys.executable, str(SCRIPT), *arguments,
                                          "--campaign", str(campaign)]))
    return command


def submit_preparation(args):
    campaign = args.campaign
    for path in (args.key_file, args.token_file):
        if not path.is_file():
            raise ValueError(f"Credential file does not exist: {path}")
    settings = {"key_file": str(args.key_file.resolve()), "token_file": str(args.token_file.resolve()),
                "data_dir": str(args.data_dir.resolve()), "spec": read(SPEC), "source": source_hashes()}
    with submission_lock(campaign):
        path = campaign / "settings.json"
        if path.exists() and read(path) != settings:
            raise ValueError("Campaign settings already differ; use a new campaign directory")
        save(path, settings)
        sbatch(campaign, "prepare", batch_command(campaign, args.partition, 4, 64,
                                                 "24:00:00", ["prepare"]))


def require_allocation():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("This command must execute inside a Slurm allocation")


def prepare(args):
    require_allocation()
    settings = read(args.campaign / "settings.json")
    if settings["source"] != source_hashes():
        raise RuntimeError("Campaign source changed")
    missing = [name for name in ("uv", "git", "unzip", "7z", "ffmpeg") if not shutil.which(name)]
    opencode = os.environ.get("MLEVOLVE_OPENCODE_BIN") or shutil.which("opencode") or str(Path.home() / ".opencode/bin/opencode")
    if not Path(opencode).is_file():
        missing.append("opencode (or MLEVOLVE_OPENCODE_BIN)")
    if missing:
        raise RuntimeError("Load/install prerequisites before preparation: " + ", ".join(missing))
    run(["git", "lfs", "version"])
    env = environment(settings)
    run(["bash", "tasks/mlebench_lite/setup.sh"], env=env)
    run([PYTHON, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_multifidelity.py",
         "tests/test_multifidelity_campaign.py", "tests/test_study7_math.py", "tests/test_fixed_evaluation.py"], env=env)
    run([PYTHON, "tools/check_opencode.py", "--output", args.campaign / "provider-check.json"], env=env)
    spec = settings["spec"]
    run([PYTHON, "tasks/mlebench_lite/prepare.py", "--data-dir", settings["data_dir"],
         "--token-file", settings["token_file"], "--workers", "3", "--tasks", *spec["tasks"]], env=env)
    protocols = {}
    for task in spec["tasks"]:
        directory = args.campaign / "protocols" / task
        run([PYTHON, "tasks/mlebench_lite/prepare_fixed_protocol.py", "--task", task,
             "--data-root", settings["data_dir"], "--seed", str(spec["seed"]),
             "--output-root", args.campaign / "protocol_setup" / task,
             "--protocol-dir", directory], env=env)
        protocol = read(directory / "protocol.json")
        protocols[task] = {key: protocol[key] for key in ("protocol_id", "metric", "split")}
    save(args.campaign / "ready.json", {"settings_sha256": sha(args.campaign / "settings.json"),
                                       "protocols": protocols})
    print("Preparation passed. Submit the pilot pair, then the remaining tasks.")


def ready(campaign):
    record = read(campaign / "ready.json")
    if record["settings_sha256"] != sha(campaign / "settings.json"):
        raise RuntimeError("Prepared campaign settings changed")
    settings = read(campaign / "settings.json")
    if settings["source"] != source_hashes():
        raise RuntimeError("Prepared campaign source changed")
    return settings


def submit(args):
    settings = ready(args.campaign)
    spec = settings["spec"]
    tasks = args.tasks or list(spec["tasks"])
    if set(tasks) - set(spec["tasks"]):
        raise ValueError("Task is not in the approved 14-task manifest")
    with submission_lock(args.campaign):
        for task in tasks:
            resources = spec["tasks"][task]
            for mode in spec["modes"]:
                command = batch_command(args.campaign, args.partition, resources["cpus"],
                    resources["memory_gb"], spec["slurm_time"], ["worker", "--task", task, "--mode", mode], gpu=True)
                sbatch(args.campaign, f"{task}/{mode}", command)


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
    if (output / "run-info.json").exists():
        raise RuntimeError("Existing run found; automatic restarting/re-budgeting is prohibited")
    try:
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
        from types import SimpleNamespace
        sys.path.insert(0, str(ROOT))
        from tasks.mlebench_lite.run_bo import audit_protocol
        audit_protocol(SimpleNamespace(protocol_dir=args.campaign / "protocols" / args.task,
                                       data_root=Path(settings["data_dir"]), task=args.task, seed=spec["seed"]))
        prepared = read(args.campaign / "ready.json")["protocols"][args.task]
        if read(args.campaign / "protocols" / args.task / "protocol.json")["protocol_id"] != prepared["protocol_id"]:
            raise RuntimeError("Protocol differs from the prepared campaign")
        command = search_command(settings, args.campaign, args.task, args.mode)
        save(output / "command.json", command)
        run(command, env=env)
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
        records = (logs / "phase1/transitions.jsonl").read_text().splitlines()
        ended_before_limit = (journal["elapsed_active_seconds"] < spec["time_limit_seconds"]
                              and len(records) < spec["steps"])
        final = read(logs / "final_submission.json")
        node = next(n for n in journal["nodes"] if n["id"] == final["node_id"])
        if node.get("training_fraction") != 1.0:
            raise RuntimeError("Winner was not trained on full T")
        grade = run([PYTHON, "-m", "utils.grade_submission", "--log-dir", logs,
                     "--data-dir", settings["data_dir"]], env=env, capture_output=True, text=True)
        report = json.loads(grade.stdout)
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
    spec = read(SPEC)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("submit-prepare", "prepare", "submit", "worker", "collect"))
    parser.add_argument("--campaign", type=Path, default=ROOT / "runs/mf14_aq05")
    parser.add_argument("--partition", help="Your available Slurm partition; required for submissions")
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/mlebench")
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--task")
    parser.add_argument("--mode", choices=("carry_over", "reset"))
    args = parser.parse_args()
    args.campaign = args.campaign.resolve()
    if args.command.startswith("submit") and not args.partition:
        parser.error("Provide --partition for your Mila account")
    if args.command == "submit-prepare" and (args.key_file is None or args.token_file is None):
        parser.error("Provide --key-file and --token-file paths (never raw credentials)")
    {"submit-prepare": submit_preparation, "prepare": prepare, "submit": submit,
     "worker": worker, "collect": collect}[args.command](args)


if __name__ == "__main__":
    main()
