#!/usr/bin/env python3
"""Run the owner's seven formal AQ05 competitions on Mila's 1+2+4 GPUs.

The controller only polls Slurm and launches job steps. All agent work, candidate
execution, fusion and grading runs inside GPU allocations. A short allocation
pauses a single continuing search; it never starts a new 12-hour experiment.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[2]
PRESET = 'acq7_aq05_gp4_level15_rating_kg'
METHOD_LABEL = 'aq05'
GROUPS = {
    'unkillable': ['aptos2019-blindness-detection'],
    'main': ['ranzcr-clip-catheter-line-classification', 'aerial-cactus-identification'],
    'short-unkillable': ['new-york-city-taxi-fare-prediction', 'jigsaw-toxic-comment-classification-challenge',
                         'mlsp-2013-birds', 'text-normalization-challenge-english-language'],
}
EXPECTED = {
    'task_type': 'competition', 'agent.time_limit': 43200, 'agent.steps': 500,
    'agent.initial_drafts': 3, 'agent.seed': 42, 'agent.search.parallel_search_num': 3,
    'agent.search.num_gpus': 1, 'exec.timeout': 32400, 'evaluation.enabled': False,
    'agent.code.model': 'opencode/muse-spark-1.3-contributor-free', 'agent.code.base_url': 'opencode://opencode',
    'agent.feedback.model': 'opencode/muse-spark-1.3-contributor-free', 'agent.feedback.base_url': 'opencode://opencode',
    'parent_selector.type': 'acq7_gp4', 'parent_selector.decision_scope': 'all_normal_steps',
    'study7.method_id': 'AQ05', 'study7.rating.enabled': True, 'study7.ancestor.enabled': False,
    'acq7.enabled': True, 'acq7.branch_kind': 'level_matern32', 'acq7.branch_prior_std': 1.5,
    'acq7.branch_lengthscale': 1.0, 'acq7.slope_prior_std': 0.0,
    'acq7.acquisition_type': 'finite_pool_kg',
}
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'PREEMPTED', 'NODE_FAIL',
            'OUT_OF_MEMORY', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}
RECOVERABLE = {'TIMEOUT', 'PREEMPTED', 'NODE_FAIL', 'BOOT_FAIL'}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def command(argv, **kw):
    kw.setdefault('timeout', 45)
    return subprocess.check_output([str(a) for a in argv], text=True, **kw).strip()


@contextmanager
def locked(path):
    with Path(path).open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def task_dir(campaign, task):
    return campaign / 'tasks' / task


def task_status(campaign, task):
    return read(task_dir(campaign, task) / 'status.json', {'state': 'new'})


def write_status(campaign, task, **fields):
    path = task_dir(campaign, task) / 'status.json'
    atomic(path, {**read(path, {}), **fields, 'updated_at': stamp()})


def verify_settings(path):
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(path)
    errors = {key: {'expected': value, 'actual': OmegaConf.select(cfg, key)}
              for key, value in EXPECTED.items() if OmegaConf.select(cfg, key) != value}
    if errors:
        raise RuntimeError(f'Formal {METHOD_LABEL} configuration mismatch: {errors}')
    return cfg


def init(campaign, allocation):
    if campaign.exists():
        raise RuntimeError(f'Campaign already exists: {campaign}')
    campaign.mkdir(parents=True)
    code = campaign / 'code'
    code.mkdir()
    repository = Path(command(['git', 'rev-parse', '--show-toplevel'], cwd=PROJECT))
    files = command(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '--',
                     str(PROJECT)], cwd=repository).splitlines()
    hashes = {}
    for name in files:
        source = repository / name
        if not source.is_file():
            continue
        relative = source.relative_to(PROJECT)
        target = code / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[str(relative)] = hashlib.sha256(target.read_bytes()).hexdigest()
    for name in ['.venv-mlebench', '.cache', 'data', 'mle-bench']:
        (code / name).symlink_to(PROJECT / name, target_is_directory=True)
    (campaign / 'source.patch').write_text(command(['git', 'diff', '--', str(PROJECT)], cwd=repository))
    (campaign / 'source-status.txt').write_text(command(['git', 'status', '--short'], cwd=repository))
    atomic(campaign / 'source-sha256.json', hashes)
    tasks = {}
    for group, ids in GROUPS.items():
        for task in ids:
            path = task_dir(campaign, task)
            path.mkdir(parents=True)
            tasks[task] = {'partition': group, 'server_id': 310 + len(tasks),
                           'cpus': 6 if group == 'unkillable' else 4,
                           'memory_gb': {'unkillable': 32, 'main': 24, 'short-unkillable': 64}[group]}
    manifest = {'created_at': stamp(), 'owner': os.environ['USER'], 'project': str(PROJECT),
                'code': str(code), 'campaign': str(campaign), 'tasks': tasks,
                'existing_allocation': str(allocation), 'expected_settings': EXPECTED,
                'commit': command(['git', 'rev-parse', 'HEAD'], cwd=repository),
                'mlebench_commit': command(['git', 'rev-parse', 'HEAD'], cwd=PROJECT / 'mle-bench'),
                'credential_file': os.environ.get('OPENCODE_API_KEY_FILE', str(PROJECT.parent / 'codex/opecode')),
                'codex_bin': os.environ.get('MLEVOLVE_REAL_CODEX_BIN') or shutil.which('codex'),
                'interruption_policy': 'Preserve decisions, RNG, journal, completed execution outputs and active time. '
                    'Interrupted candidate code is replayed with its cumulative 32400-second cap; arbitrary '
                    'candidate model/optimizer state cannot be restored by the supervisor. Queue time is excluded.'}
    atomic(campaign / 'manifest.json', manifest)
    print(campaign, flush=True)


def child_environment(manifest, task):
    env = dict(os.environ)
    env.update(OPENCODE_API_KEY_FILE=manifest['credential_file'],
               MLEVOLVE_CODEX_REASONING_EFFORT='high', MLEVOLVE_CODEX_TIMEOUT_SECONDS='1800',
               TIME_LIMIT='12hrs 0mins 0secs', STEP_LIMIT='500', MEMORY_INDEX='0',
               PYTHONUNBUFFERED='1', USE_TF='0', TF_CPP_MIN_LOG_LEVEL='3',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    if manifest.get('codex_bin'):
        env['MLEVOLVE_REAL_CODEX_BIN'] = manifest['codex_bin']
    # The runtime uses the one-device mask assigned by Slurm. Never reset it to
    # a physical index, which would make different tasks collide on one GPU.
    if not env.get('SLURM_JOB_ID'):
        raise RuntimeError('Agent work requires a Slurm allocation')
    return env


def stop_group(process):
    """Only this runner's child session; do not signal unrelated user processes."""
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    time.sleep(2)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=20)


def reconcile_executions(logs, stopped_epoch, proof):
    for path in (logs / 'candidate_execution').glob('*.json'):
        record = read(path)
        if record['state'] == 'running':
            record.update(state='interrupted',
                          elapsed_seconds=record['elapsed_seconds'] + max(0., stopped_epoch - record['started_epoch']),
                          interrupted_epoch=stopped_epoch, release_proof=proof)
            atomic(path, record)


def allocation_record(job):
    # sacct is authoritative only once squeue no longer contains the job.
    # Query the user's queue: squeue -j exits nonzero once an old job has been
    # purged, even though sacct still contains its authoritative terminal row.
    live = command(['squeue', '-h', '-u', os.environ['USER'], '-o', '%i|%T'])
    if any(line.split('|')[0] == str(job) for line in live.splitlines()):
        return None
    output = command(['sacct', '-X', '-n', '-P', '-j', str(job), '-o', 'JobIDRaw,State,End,ExitCode'])
    for line in output.splitlines():
        fields = line.split('|')
        if fields[0] == str(job):
            state = fields[1].split()[0].rstrip('+')
            if state in TERMINAL and fields[2] not in {'Unknown', 'None', ''}:
                return {'job_id': str(job), 'state': state, 'end': fields[2], 'exit_code': fields[3],
                        'end_epoch': datetime.fromisoformat(fields[2]).timestamp(), 'verified_at': stamp()}
    return None


def admit_recovery(campaign, task, released):
    info = read(task_dir(campaign, task) / 'run-info.json')
    if not info:
        raise RuntimeError(f'No recoverable run-info for {task}')
    logs = Path(info['log_dir'])
    if not (logs / 'journal.json').is_file():
        raise RuntimeError(f'No persisted journal for {task}')
    heartbeat = read(logs / 'competition_heartbeat.json')
    status = task_status(campaign, task)
    stopped = status.get('search_stopped_epoch') or released['end_epoch']
    # A graceful checkpoint has an exact active clock. An abrupt Slurm death
    # includes the tail after the last heartbeat, up to allocation termination.
    elapsed = (status.get('active_elapsed_seconds', 0.) if status['state'] in {'paused', 'search_complete'}
               else max(0., stopped - heartbeat['agent_start_epoch']))
    reconcile_executions(logs, stopped, released)
    ledger_path = logs / 'parent_selector' / 'root_draft_state.json'
    if not ledger_path.exists():
        matches = list(logs.glob('**/root_draft_state.json'))
        if len(matches) != 1:
            raise RuntimeError('Cannot locate unique selector checkpoint')
        ledger_path = matches[0]
    ledger = read(ledger_path)
    ledger['executor_release_receipt'] = {
        'owner_host': ledger['owner_host'], 'owner_pid': ledger['owner_pid'],
        'verified_by': 'slurm_allocation_release', **released}
    atomic(ledger_path, ledger)
    write_status(campaign, task, state='search_complete' if status['state'] == 'search_complete' else 'paused',
                 active_elapsed_seconds=elapsed, release_receipt=released)


def finalize(campaign, task, env, attempt):
    directory = task_dir(campaign, task)
    info = read(directory / 'run-info.json')
    logs = Path(info['log_dir'])
    verify_settings(logs / 'config.yaml')
    marker = logs / 'fusion-completed.json'
    if not marker.exists():
        with (attempt / 'fusion.log').open('a') as stream:
            result = subprocess.run([sys.executable, 'utils/submission_fusion_utils.py', '--task_id', task,
                                     '--run_info', str(directory / 'run-info.json')],
                                    env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=2400)
        if result.returncode:
            raise RuntimeError(f'Final fusion failed: {result.returncode}')
        atomic(marker, {'finished_at': stamp()})
    final = read(logs / 'final_submission.json')
    if not final or not final.get('submission') or not Path(final['submission']).is_file():
        raise RuntimeError('Search produced no final submission')
    grade_path = logs / 'official-grade.json'
    if not grade_path.exists():
        with (attempt / 'grade.stderr.log').open('a') as errors:
            result = subprocess.run([sys.executable, '-m', 'utils.grade_submission', '--log-dir', str(logs),
                                     '--data-dir', str(PROJECT / 'data/mlebench')], env=env, text=True,
                                    stdout=subprocess.PIPE, stderr=errors, timeout=1800)
        if result.returncode:
            raise RuntimeError(f'Official grading failed: {result.returncode}')
        atomic(grade_path, json.loads(result.stdout))
    grade = read(grade_path)
    if grade.get('report') is None:
        raise RuntimeError('Official grading has no report')
    write_status(campaign, task, state='complete', finished_at=stamp(),
                 submission=final['submission'], official_grade=str(grade_path),
                 final_submission_sha256=hashlib.sha256(Path(final['submission']).read_bytes()).hexdigest())


def lease(campaign, task):
    manifest = read(campaign / 'manifest.json')
    resources = manifest['tasks'][task]
    directory = task_dir(campaign, task)
    with locked(directory / 'lease.lock'):
        state = task_status(campaign, task)
        if state['state'] == 'complete':
            return 0
        if state['state'] not in {'new', 'paused', 'search_complete'}:
            raise RuntimeError(f'Lease not admitted: {state}')
        env = child_environment(manifest, task)
        os.chdir(manifest['code'])
        job = env['SLURM_JOB_ID']
        number = len(list(directory.glob('attempt-*'))) + 1
        attempt = directory / f'attempt-{number:02d}-{job}'
        attempt.mkdir()
        import torch
        if torch.cuda.device_count() != 1 or 'A100' not in torch.cuda.get_device_name(0):
            raise RuntimeError('Each task must see exactly one A100')
        gpu = torch.cuda.get_device_name(0)
        allocation = command(['scontrol', 'show', 'job', '-o', job])
        end = next(part.split('=', 1)[1] for part in allocation.split() if part.startswith('EndTime='))
        end_epoch = datetime.fromisoformat(end).timestamp()
        lease_end = end_epoch - 600
        atomic(attempt / 'resources.json', {'job_id': job, 'step_id': env.get('SLURM_STEP_ID'),
               'host': socket.gethostname(), 'gpu': gpu, 'cuda_visible_devices': env.get('CUDA_VISIBLE_DEVICES'),
               'cpu_affinity': sorted(os.sched_getaffinity(0)), 'allocation': allocation,
               'started_at': stamp(), 'lease_end_epoch': lease_end})
        write_status(campaign, task, state='starting', job_id=job, attempt=str(attempt), gpu=gpu,
                     host=socket.gethostname(), started_at=state.get('started_at', stamp()))
        if state['state'] == 'search_complete' or state.get('active_elapsed_seconds', 0.) >= 43200:
            finalize(campaign, task, env, attempt)
            return 0
        if lease_end - time.time() < 300:
            write_status(campaign, task, state='paused')
            return 75
        server = process = None
        try:
            port = 5005 + resources['server_id']
            env['GRADING_SERVER_PORT'] = str(port)
            from urllib.request import urlopen
            with (attempt / 'format-server.log').open('a') as output:
                server = subprocess.Popen([sys.executable, '-u', '-m', 'engine.validation.format_server',
                    f'dataset_dir={PROJECT / "data/mlebench"}', 'data_dir=none', 'desc_file=none'],
                    env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            for _ in range(120):
                if server.poll() is not None:
                    raise RuntimeError('Format server exited during startup')
                try:
                    with urlopen(f'http://127.0.0.1:{port}/health', timeout=2) as response:
                        if response.status == 200:
                            break
                except OSError:
                    time.sleep(1)
            else:
                raise RuntimeError('Format server health timeout')
            pause = attempt / 'pause-request'
            env.update(MLEVOLVE_COMPETITION_CHECKPOINT='1', MLEVOLVE_COMPETITION_PAUSE_FILE=str(pause),
                       MLEVOLVE_COMPETITION_LEASE_END=str(lease_end))
            def request_pause(signum, frame):
                pause.touch()
            signal.signal(signal.SIGTERM, request_pause)
            signal.signal(signal.SIGUSR1, request_pause)
            args = [sys.executable, '-u', 'run.py', f'experiment_preset={PRESET}', f'exp_id={task}',
                    f'dataset_dir={PROJECT / "data/mlebench"}',
                    f'data_dir={PROJECT / "data/mlebench" / task / "prepared/public"}',
                    f'desc_file={PROJECT / "data/mlebench" / task / "prepared/public/description.md"}',
                    f'exp_name={task}', f'cpu_number={resources["cpus"]}', 'start_cpu_id=0',
                    f'log_dir={campaign / "runs"}', f'workspace_dir={campaign / "runs"}',
                    f'run_result_file={directory / "run-info.json"}', 'preprocess_data=true', 'copy_data=false',
                    'use_grading_server=true']
            for key in ['agent.time_limit', 'agent.steps', 'agent.initial_drafts', 'agent.seed',
                        'agent.search.parallel_search_num', 'agent.search.num_gpus', 'exec.timeout',
                        'evaluation.enabled', 'agent.code.model', 'agent.code.base_url',
                        'agent.feedback.model', 'agent.feedback.base_url']:
                value = EXPECTED[key]
                args.append(f'{key}={str(value).lower() if isinstance(value, bool) else value}')
            previous = read(directory / 'run-info.json')
            if previous:
                logs = Path(previous['log_dir'])
                verify_settings(logs / 'config.yaml')
                args += [f'resume_run_dir={logs.parent}',
                         f'resume_elapsed_seconds={math.ceil(state.get("active_elapsed_seconds", 0.))}']
                (logs / 'competition_pause.json').unlink(missing_ok=True)
            atomic(attempt / 'command.json', args)
            with (attempt / 'search.log').open('a') as output:
                process = subprocess.Popen(args, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            write_status(campaign, task, state='running', search_pid=process.pid, attempt_started_epoch=time.time(),
                         search_stopped_epoch=None)
            checked = False
            while process.poll() is None:
                info = read(directory / 'run-info.json')
                if info and not checked and (Path(info['log_dir']) / 'config.yaml').is_file():
                    verify_settings(Path(info['log_dir']) / 'config.yaml')
                    atomic(attempt / 'setting-check.json', {'verified_at': stamp(), 'expected': EXPECTED})
                    checked = True
                if time.time() > lease_end + 90:
                    raise RuntimeError('Agent did not checkpoint before its allocation boundary')
                time.sleep(2)
            exit_code = process.returncode
            stopped = time.time()
            stop_group(process)
            process = None
            info = read(directory / 'run-info.json')
            if not info:
                raise RuntimeError(f'Agent exited {exit_code} without run-info')
            logs = Path(info['log_dir'])
            verify_settings(logs / 'config.yaml')
            heartbeat = read(logs / 'competition_pause.json') or read(logs / 'competition_heartbeat.json')
            journal = read(logs / 'journal.json')
            elapsed = max(heartbeat['active_elapsed_seconds'], journal.get('elapsed_active_seconds', 0.))
            reconcile_executions(logs, stopped, {'verified_by': 'owned_process_group_killed', 'job_id': job})
            budget_complete = elapsed >= 43200
            records = journal.get('parent_selector_state', {}).get('observations', [])
            steps_complete = len(records) >= 500
            atomic(attempt / 'exit.json', {'exit_code': exit_code, 'stopped_epoch': stopped,
                   'active_elapsed_seconds': elapsed, 'budget_complete': budget_complete,
                   'completed_transitions': len(records), 'checkpoint': heartbeat})
            write_status(campaign, task, search_exit_code=exit_code, search_stopped_epoch=stopped,
                         active_elapsed_seconds=elapsed, budget_complete=budget_complete,
                         completed_transitions=len(records), run_root=str(logs.parent))
            if exit_code not in {0, 75}:
                raise RuntimeError(f'Agent failed with exit code {exit_code}')
            if budget_complete or steps_complete:
                write_status(campaign, task, state='search_complete', search_finished_at=stamp())
                finalize(campaign, task, env, attempt)
                return 0
            if exit_code == 0:
                raise RuntimeError('Search ended before the 43200-second budget or 500-step cap')
            write_status(campaign, task, state='paused')
            return 75
        except Exception as exc:
            write_status(campaign, task, state='failed', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            stop_group(process)
            stop_group(server)


def entry_command(manifest, *args):
    code = Path(manifest['code'])
    script = f'cd {shlex.quote(str(code))}\nsource tasks/mlebench_lite/env.sh\nexec python -u tasks/mlebench_lite/aq05_campaign.py ' + shlex.join(list(args))
    return ['bash', '-c', script]


def group(campaign, partition):
    manifest = read(campaign / 'manifest.json')
    children = []
    for task in GROUPS[partition]:
        if task_status(campaign, task)['state'] in {'complete', 'failed'}:
            continue
        resources = manifest['tasks'][task]
        argv = ['srun', '--exclusive', '--exact', '--nodes=1', '--ntasks=1', '--gres=gpu:1',
                f'--cpus-per-task={resources["cpus"]}', f'--mem={resources["memory_gb"]}G', '--cpu-bind=cores',
                *entry_command(manifest, 'lease', '--campaign', str(campaign), '--task', task)]
        stream = (task_dir(campaign, task) / f'job-{os.environ["SLURM_JOB_ID"]}.log').open('a')
        children.append((subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT), stream))
    codes = []
    for child, stream in children:
        codes.append(child.wait())
        stream.close()
    return 1 if any(code not in {0, 75} for code in codes) else 0


def submit(campaign, partition, manifest):
    name = f'{METHOD_LABEL}-{campaign.name[-10:]}-{partition.split("-")[0]}'
    count = len(GROUPS[partition])
    argv = ['sbatch', '--parsable', f'--job-name={name}', f'--partition={partition}', '--nodes=1', '--ntasks=1',
            f'--gres=gpu:a100l:{count}', f'--cpus-per-task={4 * count}',
            f'--mem={sum(manifest["tasks"][task]["memory_gb"] for task in GROUPS[partition])}G',
            '--time=03:00:00' if partition == 'short-unkillable' else '--time=13:00:00',
            '--constraint=nvlink', '--no-requeue', '--export=ALL', f'--comment={campaign}:{partition}',
            f'--output={campaign}/slurm-{partition}-%j.log',
            '--wrap=' + shlex.join(entry_command(manifest, 'group', '--campaign', str(campaign), '--partition', partition))]
    intent = campaign / f'submit-intent-{partition}.json'
    if intent.exists():
        raise RuntimeError(f'Unresolved prior sbatch submission; inspect Slurm before retrying: {intent}')
    atomic(intent, {'command': argv, 'requested_at': stamp()})
    job = command(argv).split(';')[0]
    if not job.isdigit():
        raise RuntimeError(f'Unexpected sbatch response: {job}')
    return job


def controller(campaign):
    manifest = read(campaign / 'manifest.json')
    with locked(campaign / 'controller.lock'):
        atomic(campaign / 'controller-process.json', {'pid': os.getpid(), 'host': socket.gethostname(), 'started_at': stamp()})
        queue_path = campaign / 'queue.json'
        queue = read(queue_path, {partition: {'history': []} for partition in GROUPS})
        direct = None
        while True:
            for partition, tasks in GROUPS.items():
                q = queue[partition]
                job = q.get('active_job')
                if job:
                    if partition == 'unkillable':
                        # This allocation was provided by the user. Watch only
                        # our srun step, and leave their interactive shell alone.
                        if direct is not None:
                            rc = direct.poll()
                            if rc is None:
                                continue
                            q['direct_exit_code'] = rc
                            direct = None
                        else:
                            import psutil
                            pid = q.get('direct_pid')
                            if (pid and psutil.pid_exists(pid) and psutil.Process(pid).create_time() == q.get('direct_created')
                                    and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE):
                                continue
                        current = task_status(campaign, tasks[0])
                        if current['state'] not in {'complete', 'failed', 'paused', 'search_complete'}:
                            released = allocation_record(job)
                            if released is None:
                                write_status(campaign, tasks[0], state='failed', error='srun ended unexpectedly; inspect step log before recovery')
                            else:
                                admit_recovery(campaign, tasks[0], released)
                        q.pop('active_job')
                    else:
                        released = allocation_record(job)
                        if released is None:
                            continue
                        q['history'][-1]['release'] = released
                        for task in tasks:
                            current = task_status(campaign, task)
                            if current['state'] in {'paused', 'search_complete'} or (
                                    released['state'] in RECOVERABLE and current['state'] in {'running', 'starting'}):
                                try:
                                    admit_recovery(campaign, task, released)
                                except Exception as exc:
                                    write_status(campaign, task, state='failed', error=f'Recovery admission: {exc}')
                            elif current['state'] in {'running', 'starting', 'new'}:
                                write_status(campaign, task, state='failed', error=f'Allocation terminated: {released}')
                        q.pop('active_job')
                    atomic(queue_path, queue)
                remaining = [task for task in tasks if task_status(campaign, task)['state'] not in {'complete', 'failed'}]
                if not remaining:
                    continue
                if partition == 'unkillable':
                    job = manifest['existing_allocation']
                    live = command(['squeue', '-h', '-j', job, '-o', '%T'])
                    if live != 'RUNNING':
                        # A replacement is needed only after the existing lease
                        # has really ended, never on a failed status observation.
                        write_status(campaign, tasks[0], state='failed', error='Existing unkillable allocation unavailable; needs replacement')
                        continue
                    resources = manifest['tasks'][tasks[0]]
                    argv = ['srun', f'--jobid={job}', '--overlap', '--nodes=1', '--ntasks=1', '--gres=gpu:1',
                            '--cpus-per-task=6', '--mem=32G', '--cpu-bind=cores',
                            *entry_command(manifest, 'lease', '--campaign', str(campaign), '--task', tasks[0])]
                    with (campaign / 'existing-unkillable-step.log').open('a') as output:
                        direct = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=output,
                                                  stderr=subprocess.STDOUT, start_new_session=True)
                    import psutil
                    q.update(direct_pid=direct.pid, direct_created=psutil.Process(direct.pid).create_time())
                else:
                    job = submit(campaign, partition, manifest)
                q['active_job'] = job
                q['history'].append({'job_id': job, 'submitted_at': stamp()})
                atomic(queue_path, queue)
                (campaign / f'submit-intent-{partition}.json').unlink(missing_ok=True)
                print(f'{stamp()} {partition}: {job} ({", ".join(remaining)})', flush=True)
            statuses = {task: task_status(campaign, task) for task in manifest['tasks']}
            atomic(campaign / 'summary.json', {'updated_at': stamp(), 'tasks': statuses, 'queue': queue})
            if all(value['state'] in {'complete', 'failed'} for value in statuses.values()):
                print(f'{stamp()} Campaign stopped; inspect summary.json', flush=True)
                return int(any(value['state'] != 'complete' for value in statuses.values()))
            time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['init', 'controller', 'group', 'lease'])
    parser.add_argument('--campaign', required=True, type=Path)
    parser.add_argument('--allocation')
    parser.add_argument('--task')
    parser.add_argument('--partition')
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    if args.action == 'init':
        init(campaign, args.allocation)
        return 0
    if args.action == 'controller':
        while True:
            try:
                return controller(campaign)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                # Observation failure is never evidence that a job stopped.
                print(f'{stamp()} Slurm command failed; revalidate saved handles in 30s: {exc}', flush=True)
                time.sleep(30)
    if args.action == 'group':
        return group(campaign, args.partition)
    return lease(campaign, args.task)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
