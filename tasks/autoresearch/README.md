# MLEvolve nanoGPT task

This task adapts the native MLEvolve workflow to from-scratch language-model
training on an A100. Draft specialists, merger, review, improvement, DEBUG,
evolution, fusion and aggregation use the existing search machinery. The task
overlay supplies a candidate-module API and a fixed host training/evaluation
harness. Ordinary MLE-bench runs keep their original configuration and prompts.

## Setup and execution

From `MLEvolve_BO`, on an allocated CUDA GPU:

```bash
bash tasks/autoresearch/setup.sh
uv run --frozen python tools/check_opencode.py
bash run_autoresearch.sh --gpu 0 --cpus 4 --steps 500 \
  experiment_preset=seven_e02_state4_ucb agent.time_limit=21600
```

The locked environment uses Python 3.12, PyTorch 2.9.1+cu128,
sentence-transformers 2.6.1, transformers 4.57.1 and faiss-cpu 1.11.0.
Setup downloads ten ClimbMix training shards, a held-out shard and the tokenizer
to `.cache/autoresearch`, creates a SHA256 manifest and checks the configured
CUDA memory embedding model. Data, caches and model weights are local artifacts.

The task overlay selects `opencode/muse-spark-1.3-contributor-free` through
the native `opencode://opencode` backend. Use `opencode auth login`,
`OPENCODE_API_KEY` or `OPENCODE_API_KEY_FILE`; the backend also accepts the existing local `../codex/opecode` credential file.
Credentials are read at execution time and are not part of this repository.
The backend requires the official OpenCode CLI. Thinking overrides remain unchanged.
See [OpenCode setup and verification](../../docs/OPENCODE_zh.md).

Each candidate trains from scratch for 300 measured seconds, excluding the
11 warmup steps. Compilation, validation and checkpoint saving take additional
wall time. Official validation uses 20,971,520 tokens and returns BPB, with lower
values better. The host checks causal next-token predictions and unreduced
cross entropy before admitting a score. The candidate may change the model and
optimizer while preserving the host API and immutable protocol; see
[TASK.md](TASK.md) and [source provenance](SOURCE.md).

Defaults are three initial drafts, up to five ordinary draft branches, two
concurrent workers sharing one GPU and 500 total steps. Other search, memory,
operator and timeout settings inherit `config/config.yaml`; the command above
explicitly sets a six-hour workflow budget. With four CPUs and two workers,
each training worker receives two CPUs. Set
`agent.search.parallel_search_num=1` for serial execution. Global-memory
initialization failure stops this task instead of silently disabling memory.

Resume with the same task, data, method and shared-pool configuration:

```bash
bash run_autoresearch.sh --gpu 0 --cpus 4 --resume runs/<run-directory>
```

The launcher verifies the prepared data manifest and uses separate node-local
compilation caches. Saved configurations retain their original worker count
unless explicitly overridden. Final outputs include the best candidate,
`result.json`, `checkpoint.pt` and `logs/autoresearch_summary.json`.

## Seven registered search methods

The original GP6/GP7/MCGS implementations remain available. These additional
presets register independent study7 methods:

| Preset | Search method |
| --- | --- |
| `seven_e01_gp6_original` | Existing GP6, including its frozen utility scale |
| `seven_e02_state4_ucb` | Four-dimensional state GP with latent UCB |
| `seven_e03_state4_ancestorB_ucb` | State4 plus ancestor-content kernel B |
| `seven_e04_state4_ratingmean_ucb` | State4 plus completed-node rating mean/covariance |
| `seven_e05_state4_ancestorB_ratingmean_ucb` | State4 plus both components |
| `seven_e06_mcgs_original` | Existing original MCGS/Top-K path |
| `seven_e07_parent_greedy_managed` | Managed normal decisions by parent score, without a GP |

`mlevolve_original` is an alias configuration for the existing `legacy_original`
selector. It is a task-adapted control, not a reproduction of a full MLE-bench
paper experiment. Ordinary legacy presets do not enable study7.

For E02–E05, utility is `-BPB`. Four bounded coordinates describe parent quality,
birth gain, shrunk valid direct gain for the parent/operator, and closed local
attempts. Initial valid drafts freeze the reference median and scale. Training
uses `(child_utility - original_parent_utility) / scale` for valid closed
ordinary proposals. DEBUG repairs retain the original parent/operator and
selection-time snapshot; failures do not receive artificial zero labels, and
pending or budget-censored proposals do not enter the quality model.

The covariance uses Matérn 3/2 state similarity with fixed cross-operation
sharing. Ancestor B combines immutable two-dimensional ancestor content using
recursive source weights. The optional rating is collected once during the
completed-node execution review: levels 0/1/2 map to -1/0/+1, and unavailable
ratings are neutral. A shared coefficient with prior `N(0.25, 0.25^2)` contributes
both a prior mean and rank-one covariance. The old piBO rating mode is mutually
exclusive with this study7 mode.

Acquisition is predicted child utility plus `1.5 * latent_std`, without cost or
validity weighting. Below eight valid labels, hyperparameters stay at their
registered initial values; later fits use bounded MAP optimization. Numerical
or method-contract errors stop the affected run and are recorded explicitly.
E01/E06 retain their original search implementations; all seven method IDs
activate common budget admission and checkpoint bookkeeping.

## Shared drafts and segmented execution

A shared pool replays reviewed code while each workflow independently trains,
scores, reviews results and updates memory:

```bash
uv run --frozen python tasks/autoresearch/prepare_draft_pool.py \
  --output runs/shared-drafts
bash run_autoresearch.sh --gpu 0 --cpus 4 \
  experiment_preset=seven_e02_state4_ucb agent.time_limit=21600 \
  shared_draft_pool="$PWD/runs/shared-drafts/manifest.json"
```

Pool generation uses the native specialists, merger and code review, without
training or selecting drafts by score. Five ordered programs are sealed by
hash; three are consumed initially. Remaining slots are released only by the
selected method's native draft policy. Generation/review latency is charged
when a draft is consumed. Unused drafts never enter the search tree or memory.

`segmented.py`, `segment_supervisor.py` and `study_worker.py` support registered
Slurm campaigns. Campaign manifests and allocation assignments are external
run inputs. The worker validates ownership and GPU availability before launch.
Checkpoints commit only after a complete execution group drains and preserve
search queues, GP state, draft consumption, global memory and search bookkeeping.
Cross-allocation recovery requires evidence that the previous executor ended.

Active budget is cumulative across segments. Queue waiting is excluded;
generation, compilation, evaluation, review, failures and interrupted work are
charged. Results completed after a study7 budget expires retain their raw trial
artifacts but cannot enter best-so-far scores or GP labels. Ordinary MLE-bench
execution does not enable segmented orchestration.

The `three_settings_8h.yaml` and `four_settings_8h.yaml` files are reusable
GP7/MCGS comparison templates for `supervise.py`. `paired_campaign.py` additionally
supports scheduled handoffs using externally supplied resource manifests.
Their runtime reporting dependency, `compare.py`, generates local outputs;
no campaign records or generated reports are versioned here.

## Logs and verification

Native logs include selection decisions/snapshots, transitions, continuation
events and GP fits. Study7 additionally records decision-time state, ancestor
provenance, rating evidence, budget admission and latent-UCB diagnostics.
`llm_calls.jsonl` records available Codex usage independently of checkpoint
rollback; unavailable token counts remain unknown.

```bash
uv run --frozen python -m pytest -q tests_gp6 \
  tests/test_study7_math.py tests/test_study7_runtime.py \
  tests/test_study7_scheduler.py tests/test_study_budget.py \
  tests/test_study_segmented.py tests/test_autoresearch.py \
  tests/test_autoresearch_campaign.py tests/test_autoresearch_native.py \
  tests/test_autoresearch_causality.py tests/test_shared_drafts.py \
  tests/test_paired_handoff.py
```

These tests replace external LLM and training services where applicable; they
are not full GPU benchmark runs. Experiment records, frozen campaign copies,
analysis outputs, datasets, credentials and caches remain outside Git history.
