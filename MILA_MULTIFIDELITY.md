# Mila: AQ05 carry-over versus reset

## What is frozen

14 Lite tasks, excluding the eight agreed tasks; two modes; seed 42. The exact
task list and CPU/RAM allocations are in `config/mlebench_mf14.json`.
Each search requests one A100 **80GB**, 12 hours cumulative active search,
500 steps, three drafts, three workers, and a nine-hour full-data candidate
execution cap. Half-data executions stop at the 40% active-time boundary.
Slurm allocation is 13 hours to allow startup and final grading. A candidate's
effective cap can be shorter when the search budget is nearly exhausted.

Both modes use AQ05 GP4 knowledge gradient and `muse-spark-1.3-contributor`
through `codex://opencode-go`, high reasoning. Codex CLI 0.164.0 is pinned. Both start on the
same uniformly random 50% of frozen training rows and use the same full fixed
validation set. At 40% of the active budget, scheduling cheap work stops;
existing work drains before the full-data switch. Carry-over keeps GP history;
reset fits only on full-parent/full-child transitions. Population and normal
evolution continue. Only valid full-training programs can win.

Cheap executions are bounded by the remaining time to the 40% switch, so
training is stopped there if it has not completed. A run with no valid
full-data winner is a failure, not a cheap-data result. There is no automatic
restart or replacement of failed runs.
If the cheap search exhausts legal work before the cutoff, it stops immediately
with an explicit failure instead of holding an idle GPU until the switch.

**This is not an exact reproduction of the historical upstream campaign.**
The approved differences are fixed validation, redesigned evolution, the
unverified historical Codex CLI version (ours is 0.164.0), and selection
of the best full-data fixed-validation node instead of legacy final fusion.
We request continuous allocations; historical runs included interruptions and
repairs. The Aerial Cactus pilot requests 64 GB host RAM after both 24 GB
pilot workers reached the memory limit; this differs from the recorded owner
allocation. Other resources come from the owner run or collaborator runbook;
the latter are requested resources, not verified completed runs. See the
manifest for per-task provenance.

## 1. Transfer and unpack once

On your Mac, replace the placeholders with the delivered archive and your host:

```bash
scp /path/to/mila-mf14-COMMIT.tar.gz USER@MILA_HOST:~/
```

On Mila, unpack in a persistent project/scratch directory with enough room for
the 14 datasets, environment, generated programs and outputs. Avoid node-local
temporary storage: subsequent jobs must see the same paths.

```bash
mkdir -p /YOUR/PERSISTENT/PATH/mila-mf14
cd /YOUR/PERSISTENT/PATH/mila-mf14
tar -xzf ~/mila-mf14-COMMIT.tar.gz
```

The archive contains the committed source and per-file SHA256 checksums, without
credentials, datasets or Python environments. You do **not** need to clone
MLEBench locally. The preparation job clones the pinned MLEBench revision on
Mila and downloads/prepares only the 14 tasks. Existing prepared data can be
reused through `--data-dir` below.

## 2. Supply your account-specific prerequisites

Load/install these commands using your Mila environment: `python3`, `uv`, `git`
with Git LFS, `unzip`, `7z`, `ffmpeg`, and Codex CLI 0.164.0. The preparation job
checks them before installing the locked Python environment. If Codex is outside
PATH, export `MLEVOLVE_REAL_CODEX_BIN=/absolute/path/to/codex`.

Create two private text files outside this checkout:

- OpenCode Go API key: a file containing only your key, from a workspace with an active Go subscription.
- Kaggle access token: a file containing your `KGAT_...` token.

Restrict their permissions (`chmod 600 /path/to/KEY_FILE /path/to/TOKEN_FILE`).
Accept the competition rules on Kaggle for the selected tasks. Your account must
have access to the model and those datasets. Jobs need outbound network access
for installation/downloads and OpenCode requests. Check your quota and available
partitions with your Mila account; partition names are deliberately required
arguments rather than assumed. Current cluster availability was not verified
from this local machine.

## 3. Submit preparation

From the unpacked directory:

```bash
python3 tasks/mlebench_lite/mila_multifidelity.py submit-prepare \
  --partition YOUR_CPU_PARTITION \
  --key-file /absolute/path/to/KEY_FILE \
  --token-file /absolute/path/to/TOKEN_FILE \
  --data-dir /YOUR/PERSISTENT/PATH/mlebench-data
```

Preparation requests 4 CPUs, 64GB RAM and up to 24 hours (operational setup,
outside the search allocation). It installs locked dependencies, runs regression
tests, checks the live provider, prepares datasets, and creates one shared fixed
protocol per task. It makes model calls for provider checks and protocol setup.
Logs are in `runs/mf14_aq05/slurm/`. Proceed only after
`runs/mf14_aq05/ready.json` exists and the preparation job completes successfully.
The protocol's recorded setup cost is charged by the existing evaluator to each
search; dataset download and environment installation are outside search time.
Protocol creation uses the existing setup agent: automated structural audits do
not establish task-specific metric/split semantics. Review the metric/split
summaries in `ready.json` before spending the full campaign budget.

If preparation fails, inspect its log. The submission ledger blocks duplicate
submissions; after confirming the job has stopped, remove only the `prepare`
entry from `runs/mf14_aq05/jobs.json` before retrying the same preparation.
Do not delete settings, protocols or search outputs to conceal a failed run.

## 4. Run the short GPU preflight

```bash
python3 tasks/mlebench_lite/mila_multifidelity.py submit-preflight \
  --partition main
```

This requests one A100 80GB, 4 CPUs, 24GB RAM, with a **30-minute maximum**.
It checks actual CUDA training/backpropagation, the embedding model, live
OpenCode calls, and both modes through real CPU candidate execution and the
three-worker search loop. LLM candidate replies in the integration tests are
synthetic; the separate provider check makes real calls. No benchmark results
are produced. Only a successful job writes `runs/mf14_aq05/gpu-ready.json`.
Full search submission refuses to proceed without that record matching the
prepared settings. On failure, inspect logs before retrying; the submission
ledger uses key `gpu-preflight`.

## 5. Launch a pilot pair, then the remaining jobs

```bash
python3 tasks/mlebench_lite/mila_multifidelity.py submit \
  --partition YOUR_GPU_PARTITION --tasks aerial-cactus-identification
```

This launches both modes using the full approved budget. After both finish,
check their `result.json` files under `runs/mf14_aq05/tasks/`; both should say
`complete`. If either fails, investigate before launching the remaining tasks.

```bash
python3 tasks/mlebench_lite/mila_multifidelity.py submit \
  --partition YOUR_GPU_PARTITION
```

The second command skips jobs already submitted, including the pilot pair.
Each job checks actual GPU hardware, environment, protocol and source identity,
then runs search and official private grading of exactly its selected submission.
All jobs request `gpu:a100l:1`; workers reject non-A100 or non-80GB allocations.
If your cluster exposes another GRES name or cannot accommodate the historical
per-task resources, resolve that before changing the frozen resource protocol.
The nominal total is 28 × 12 = **336 GPU-hours** of search, plus startup/grading;
queue time and preparation are additional. All jobs may queue concurrently;
Slurm applies your account limits.

Mila's partition documentation checked on 2026-10-22 lists `unkillable` at
6 CPUs/32GB, `main` at 8 CPUs/48GB, and `long` without these resource limits.
The 4CPU/24GB RANZCR job and the 6CPU/32GB APTOS job fit
`unkillable`. Aerial Cactus now requests 64GB and must use `long`; the other
larger task allocations also require `long` under these documented
limits. **`long` is preemptible; interrupted searches support explicit recovery.**
Use the recovery command below without changing resources or granting a fresh budget. CPU preparation
(4CPU/64GB/24h) fits `main-cpu` or `long-cpu`, not `unkillable-cpu`.
Source: https://docs.mila.quebec/technical_reference/clusters/mila/partitions/

## 6. Recover an interrupted search

After Slurm has released the interrupted allocation:

```bash
python3 tasks/mlebench_lite/mila_multifidelity.py resume \
  --partition long --task TASK_ID --mode carry_over
```

Use the interrupted run's task ID, mode (`carry_over` or `reset`) and original
partition. The command checks `squeue` and `sacct`, refuses a still-active job,
and submits at most one replacement per interrupted allocation. There is no
automatic requeue or unlimited retry loop. Ordinary program failures require
investigation; they are not automatically treated as interruptions.

Recovery preserves completed results, the search population, saved BO state,
fidelity stage and original 12-hour cumulative search budget. Queue downtime is
excluded. Checkpoints are saved every 15 seconds and on interruption signals.
Hard termination uses saved state and Slurm's recorded ending time; time spent
on interrupted work remains charged. Incomplete training reruns the same saved
program, with its earlier execution time deducted from the nine-hour candidate
cap and the remaining active time to the cheap-stage boundary, whichever is
smaller. This does not restore an in-training model/optimizer checkpoint. An
unfinished generation replays its saved decision and may require another LLM
call; recovery does not guarantee an identical stochastic trajectory.

Keep the entire campaign directory. Recovery requires an initialized search
checkpoint; a setup failure before that point needs investigation. The short
GPU preflight includes local-process interruption tests for both modes; actual
Slurm interruption behavior still needs verification on Mila.

## 7. Collect results

```bash
python3 tasks/mlebench_lite/mila_multifidelity.py collect
```

`runs/mf14_aq05/summary.csv` always contains all 28 task/mode rows, including
failed or missing results. Per-run `official-grade.json`, `result.json`, command,
hardware and environment records are beside the search logs. For interrupted
jobs without a result, inspect Slurm state rather than treating them as losses
or silently rerunning them. Preserve the entire campaign directory.
Successful searches ending before the time/step cap are flagged
`ended_before_limit`; inspect their stopping reason in the logs and retain their
outcomes. The original workflow can exhaust legal actions before its budget.

## Interpretation

The comparison is carry-over versus reset under the same wrapper, shared data
protocol, AQ05 and resources. Unit of analysis is the task-level pair; report
both official outcomes and all failures/missing outcomes across the 14 tasks.
Seed 42 does not guarantee end-to-end deterministic LLM sampling. One run per
task/mode cannot establish low-variance performance. Without a matched full-data
baseline, this campaign cannot establish that multi-fidelity improves over
ordinary BO. Historical comparisons have the differences listed above.
The existing candidate execution path does not enforce filesystem isolation;
fixed validation is a host-scoring protocol, not an OS-level data-access sandbox.

## Local verification boundary

The standard-library launch tests can run with:
`python3 tests/test_multifidelity_campaign.py`.
They check all 28 commands, task exclusions, settings parity, path quoting,
duplicate prevention, source verification, the GPU preflight gate and missing-result reporting.
Full ML regression tests and CUDA/provider checks must pass on Mila before
search. No local test can certify your account access, Slurm availability,
dataset permissions, or successful model training.
