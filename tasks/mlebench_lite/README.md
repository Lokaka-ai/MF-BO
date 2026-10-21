# MLE-bench Lite preparation and smoke runs

These tools prepare the 22 tasks in `config/mlebench_lite22.txt` with official
MLE-bench commit `507f92e1138bb6e40dac5c6ee7a6758e6424bf97`, then exercise the
native MLEvolve workflow. Datasets, run results and local setup records are not
included in Git.

## Environment and data

From `MLEvolve_BO`:

```bash
bash tasks/mlebench_lite/setup.sh
source tasks/mlebench_lite/env.sh
python tasks/mlebench_lite/check_environment.py --require-cuda --embedding
python tasks/mlebench_lite/prepare.py
```

Setup creates `.venv-mlebench` with Python 3.12 and the dependencies pinned in
`requirements.lock`, including CUDA 12.8 PyTorch, the official benchmark,
format grader and common image, text and audio libraries. It leaves an existing
`.venv` unchanged. Package and model caches use scratch-local project paths.
Prerequisites include `uv`, Git, Git LFS, `unzip`, `7z` and `ffmpeg`.

The downloader reads a Kaggle token from `../kaggle_token`, or from the file
specified by `--token-file`. It supports resumable parallel downloads and
requires the official archive MD5 before invoking upstream preparation. Official
public/private splits and prepared-file checksums remain enforced. Credentials
are not copied into configs or reports.

Prepared data lives under `data/mlebench/<task>/prepared/{public,private}`.
Original ZIPs are retained, and repeating preparation verifies existing data.
Download and preparation can be run in separate processes:

```bash
python tasks/mlebench_lite/prepare.py --download-only --workers 4
python tasks/mlebench_lite/prepare.py --prepare-only --wait-for-downloads --workers 1
```

Atomic directory locks prevent concurrent preparation of the same task across
nodes. `patch_upstream.py` changes the benchmark's disposable SQLite cache path
to `/tmp/mlebench-<uid>` (override with `MLEBENCH_CACHE_DIR`). This avoids sharing
SQLite WAL files across nodes; it does not change splits, metrics or checksums.

## Native smoke test

On an allocated GPU, with the prepared environment active:

```bash
python tasks/mlebench_lite/smoke.py --gpu auto
```

The default tasks are `leaf-classification` and `spooky-author-identification`,
run sequentially with four CPUs, four steps and 2,400 seconds per task. Smoke
runs explicitly disable fixed outer evaluation and retain the native three
initial drafts, three workers, search selection, review, debugging and memory
settings. These short runs check functionality rather than full benchmark
performance. `--tasks`, `--steps`, `--time-limit`, `--cpus` and `--output` provide
explicit overrides.

Without backend overrides, the command inherits the models and providers in
`config/config.yaml`. To use an existing Codex OpenAI login, supply both model
and provider, for example:

```bash
python tasks/mlebench_lite/smoke.py --gpu auto \
  --model gpt-6-astra --provider openai
```

For a native OpenCode Zen model, supply its model name and `--provider opencode`, with
`OPENCODE_API_KEY` or `OPENCODE_API_KEY_FILE` configured. The wrapper also supports
the local `../codex/opecode` credential file. Backend overrides affect only the
requested run and are saved with its command; they do not edit default config.

For one initial candidate followed by one search iteration on the small prepared
leaf dataset, use the repeatable native config overrides:

```bash
python tasks/mlebench_lite/smoke.py --gpu auto --tasks leaf-classification \
  --steps 2 --time-limit 1200 \
  --model opencode/muse-spark-1.3-contributor-free --provider opencode \
  --override agent.initial_drafts=1 --override agent.search.num_drafts=1 \
  --override agent.search.parallel_search_num=1 --override exec.timeout=180
```

After search, the tool validates the exported submission through the local
Flask service and scores it with the official private holdout. Private scores
are never fed back into search. The command exits nonzero unless all requested
tasks pass and stops its temporary grading servers. Outputs under
`runs/mlebench_lite_smoke_<timestamp>/` include commands, journals, logs,
submissions, `report.json` and per-task `official-grade.json`.

## Normal runs and Slurm

The existing launcher retains its original 500-step / 12-hour defaults and
uses whichever evaluation mode is selected by its configuration:

```bash
source tasks/mlebench_lite/env.sh
CPUS_PER_TASK=4 MEMORY_INDEX=0 bash run_single_task.sh \
  leaf-classification "$MLEBENCH_DATA" 111
```

Append the existing experiment preset and model overrides as needed. Match CPU
and GPU settings to the allocation and use a distinct grading-server ID for
concurrent runs. This launcher does not activate the nanoGPT task or study7.

The included `smoke.sbatch` and `prepare_large.sbatch` are Mila Slurm templates;
review partition, resource and time requests for the target cluster before use.
For example, `sbatch tasks/mlebench_lite/smoke.sbatch` runs the short GPU check.

To regenerate the dependency lock after intentional version changes:

```bash
UV_CACHE_DIR="$PWD/.cache/uv-mlebench" uv pip compile \
  pyproject.toml tasks/mlebench_lite/requirements.in --python 3.12 \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match \
  --output-file tasks/mlebench_lite/requirements.lock
```

Sources: [official MLE-bench](https://github.com/openai/mle-bench),
[official Kaggle CLI](https://github.com/Kaggle/kaggle-cli).
