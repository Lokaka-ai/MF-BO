# Mila: AQ05 carry-over versus reset

## What is frozen

14 Lite tasks, excluding the eight agreed tasks; two modes; seed 42. The exact
task list and CPU/RAM allocations are in `config/mlebench_mf14.json`.
Each search requests one A100 **80GB**, 12 hours cumulative active search,
500 steps, three drafts, three workers, and a nine-hour candidate execution cap.
Slurm allocation is 13 hours to allow startup and final grading. A candidate's
effective cap can be shorter when the search budget is nearly exhausted.

Both modes use AQ05 GP4 knowledge gradient and the native OpenCode route
`opencode/muse-spark-1.3-contributor-free`, high reasoning. Both start on the
same uniformly random 50% of frozen training rows and use the same full fixed
validation set. At 40% of the active budget, scheduling cheap work stops;
existing work drains before the full-data switch. Carry-over keeps GP history;
reset fits only on full-parent/full-child transitions. Population and normal
evolution continue. Only valid full-training programs can win.

Long cheap evaluations can delay the switch substantially. A run with no valid
full-data winner is a failure, not a cheap-data result. There is no automatic
restart or replacement of failed runs.

**This is not an exact reproduction of the historical upstream campaign.**
The approved differences are fixed validation, redesigned evolution, the native
OpenCode provider/model alias (backend equivalence is unverified), and selection
of the best full-data fixed-validation node instead of legacy final fusion.
We request continuous allocations; historical runs included interruptions and
repairs. Seven tasks' resources come from completed owner run records; the other
seven come from the collaborator runbook's requested resources, not verified
completed runs. See the manifest for per-task provenance.

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
with Git LFS, `unzip`, `7z`, `ffmpeg`, and `opencode`. The preparation job checks
them before installing the locked Python environment. If OpenCode is outside
PATH, export `MLEVOLVE_OPENCODE_BIN=/absolute/path/to/opencode`.

Create two private text files outside this checkout:

- OpenCode API key: a file containing only your key.
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

## 4. Launch a pilot pair, then the remaining jobs

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

## 5. Collect results

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
duplicate prevention, source verification and missing-result reporting.
Full ML regression tests and CUDA/provider checks must pass on Mila before
search. No local test can certify your account access, Slurm availability,
dataset permissions, or successful model training.
