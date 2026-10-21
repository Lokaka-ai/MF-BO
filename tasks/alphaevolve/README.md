# AlphaEvolve: the 15 mathematical tasks in MLEvolve Table 2

This task package uses the existing MLEvolve_BO `AgentSearch`, GP6 parent
selection, global memory, stepwise/diff coding, code review and result review.
It adds mathematical inputs, a CPU interpreter and independent constraint
checking. See [SOURCE.md](SOURCE.md) for the pinned source, exact task mapping,
licensing and the limits of reproducing unpublished author settings.

The [2026-09-20 CPU validation report](../../docs/ALPHAEVOLVE_IMPLEMENTATION_AND_VALIDATION_zh.md)
records all 15 successful draft/improve checks, the retained failed attempt,
configuration audits and runnable commands.

## Current local BO research default

For new AlphaEvolve runs, `task_preset=alphaevolve` now selects the AQ04 joint-TS BO preset and `audited_bo=redesign`. When evolution is selected, the live engine uses the evidence-packet, structured-plan, full-module route copied from the frozen E06 comparison. Use `audited_bo=preserved_evolution` with AQ04 for an authentic-MLEvolve evolution control; use `audited_bo=baseline experiment_preset=gp6` to request the older GP6 setup explicitly. Existing saved runs retain their saved configuration when resumed. The redesign is currently integrated for the AlphaEvolve host evaluator; other task types keep their existing evolution implementation pending a separate adaptation.

The historical validation table below describes the earlier 2026-09-20 GP6 setup.

## Defaults and comparison with mlebench_lite

| Setting | Normal AlphaEvolve run | Short validation |
|---|---|---|
| Agent budget | 43,200 s (12 h) | 43,200 s (stops after two nodes) |
| Search preset | existing `gp6.yaml` | same GP6 |
| Steps / initial drafts / workers | 500 / 3 / 3, inherited | 2 / 1 / 1 |
| CPU allocation | 21 cores | 3 cores per array element |
| GPU | none; CUDA hidden | none |
| Execution timeout | 32,400 s, inherited | 120 s per candidate |
| Memory | enabled, existing BGE model on CPU | same |
| Code/feedback model, temperatures | inherit `config/config.yaml` | inherit unless explicitly overridden |
| Stepwise, diff, review, memory planning | unchanged native mechanisms | same |
| ML pretrained-model cold start | disabled: no applicable math entries | same |
| Evaluation | fixed mathematical objective and feasibility | same |

Slurm's short-test allocation is 2 hours; the configured agent budget remains
12 hours. The short-test overrides only reduce the requested work and candidate
execution cap. They do not constitute a 12-hour benchmark-performance run.
`gp6` remains Matérn-3/2 GP, child UCB, fixed beta=1.5, refit every observation,
and root drafting disabled in BO selection, exactly as the existing preset.

## Preparation

From `MLEvolve_BO`:

```bash
bash tasks/alphaevolve/setup.sh
source tasks/alphaevolve/env.sh
sbatch tasks/alphaevolve/verify.sbatch
```

The environment is the same `.venv-mlebench` locked stack, including its exact
NumPy/SciPy/SymPy/Torch versions; the CUDA-capable Torch build works on CPU and
does not require a GPU. Existing installations are checked, not resynchronized.
The setup downloads the pinned official notebook, verifies its SHA256, creates
all 15 public/private task directories under `data/alphaevolve`, and records
per-file hashes. No Kaggle download or token is needed for these math tasks.
The CPU verification checks actual memory embeddings, every initial construction
and all 15 official reference constructions. It writes
`.cache/alphaevolve/environment-ready.json` and per-task grading evidence.

## Real draft + one improvement on parallel CPU allocations

```bash
sbatch tasks/alphaevolve/smoke.sbatch \
  --model muse-spark-1.3-contributor --provider opencode-go
```

The Slurm array submits all 15 tasks with concurrency up to 15, subject to the
CPU queue. A task passes only if its native journal contains exactly one valid
LLM-generated draft and one valid `improve` child, the child code changed, the
GP6 configuration and CPU memory are preserved, and the exported best
construction passes an independent post-search grade. A debug attempt does not
count as the requested improvement. A lower child score may still pass this
workflow check; no performance improvement is promised in one iteration.
No seed program is injected as a substitute for a generated draft.

```bash
python tasks/alphaevolve/report.py runs/alphaevolve_smoke_<job_id> \
  --output reports/alphaevolve_smoke.json
```

Missing tasks fail collection. If a failed task is rerun, supply campaign
directories in chronological order; all attempts remain in the report and
the latest explicit attempt is used, not the best score.
Collection also checks 120 saved native configuration fields against the base
configuration and GP6 preset, the actual GP6 model decision (without fallback),
and both candidates' recorded CPU allocations and disabled CUDA visibility.
Per-task folders contain `command.json`, `report.json`, `console.log`, native
journals, selection logs, candidate modules, CPU resource records, generated
constructions, host scores, protocol hashes and `alphaevolve_summary.json`.

## Normal 12-hour runs

```bash
sbatch tasks/alphaevolve/run.sbatch \
  --model muse-spark-1.3-contributor --provider opencode-go
```

To select one task use `sbatch --array=1 tasks/alphaevolve/run.sbatch ...`;
indices follow `config/alphaevolve15.txt`. Within an existing CPU allocation:

```bash
python tasks/alphaevolve/run.py --task circle_packing_square --cpus 21 \
  --model muse-spark-1.3-contributor --provider opencode-go
```

Without model/provider overrides, the code and feedback stages use the base
agent configuration, just like mlebench_lite. Credentials stay in the existing
backend configuration or credential file; never place them in task arguments.
