# Frozen Muse Contributor Go campaign — 2026-10-26

This supersedes the native OpenCode free-model route for the newly approved
three-arm experiment. Earlier packages and failed preparation logs remain
historical artifacts; do not edit their source manifests or reuse their ready
receipts with this source version.

## Fixed reference

- Carry-over/reset selector: AQ05 `acq7_aq05_gp4_level15_rating_kg`.
- MF-GP: approved contextual AR GP and adaptive cheap/full actions.
- Code, planning, feedback/rating and fixed-protocol preparation model:
  `muse-spark-1.3-contributor`, transport `codex://opencode-go`, high reasoning.
- Responses base URL: `https://opencode.ai/inference/go/openai/v1`.
- Codex CLI: **0.164.0**, latest stable verified on 2026-10-26, pinned across all
  arms. Historical Xixian CLI version is unverified. The launcher rejects a
  different version and records the executable's SHA256 for each run segment.
- Budget: 43,200 active seconds; 500 steps/actions (promotions count in MF-GP);
  3 initial drafts; 3 workers; seed 42; one A100 80GB; 32,400-second candidate cap.
- Task-specific CPU/RAM and approved fidelity/evaluation/winner rules unchanged.

Go requires a credential from a workspace with a Go subscription. The configured
key file is authoritative; inherited shell API keys do not override it. Do not
print credentials. An installed CLI, a listed model, or a local mocked test does
not prove entitlement, endpoint availability, or that the backend honors high
reasoning. Live text/structured-output checks remain mandatory on Mila.

The wrapper explicitly enables reasoning metadata for the external model, with
`model_reasoning_effort="high"` and reasoning summaries disabled. The existing
Codex backend retains control of prompts/output parsing; BO retains control of
program execution. No selector, acquisition, GP, fidelity or evaluation engine
code was changed in this routing revision.

## Fixed exploratory task set and launch sequence

1. aerial-cactus-identification (first task, all three modes)
2. aptos2019-blindness-detection
3. jigsaw-toxic-comment-classification-challenge
4. mlsp-2013-birds
5. new-york-city-taxi-fare-prediction
6. nomad2018-predict-transparent-conductors
7. tabular-playground-series-may-2022

The manifests retain 14 supported tasks. For the approved pilot, pass
`--tasks aerial-cactus-identification` to `submit-prepare` for BOTH launchers.
This freezes a one-task campaign in settings before downloads or protocol
registration. Preparation, default submission and collection then use only
that task. Later tasks require a separately scoped campaign; an existing
campaign cannot silently change its task set. Keep at most three search jobs running by submitting
one task's three modes at a time. Inspect the first task for engineering
correctness/utilization, not to change the task set based on who wins.

Use both launchers from this new bundle, with separate default campaign dirs:
`mila_multifidelity.py` -> `runs/mf14_aq05` (carry-over/reset),
`mila_mfgp.py` -> `runs/mf14_gp` (MF-GP). MF-GP preparation must use
`--protocol-source` pointing to this bundle's baseline `runs/mf14_aq05/protocols`.
Keep those protocols available. Each campaign requires its own successful
preparation and GPU preflight receipts.

After both gates pass, submit only the first task:

```
python3 tasks/mlebench_lite/mila_multifidelity.py submit --partition long --tasks aerial-cactus-identification
python3 tasks/mlebench_lite/mila_mfgp.py submit --partition long --tasks aerial-cactus-identification
```

Do not run these search commands before the gates. This is 21 planned search
runs, up to 252 active GPU-hours plus overhead/restarts, not 42 runs. Preserve
failures and interruptions in the seven-task denominator; the generic collector
also lists the seven unselected tasks, which must be labelled outside scope.

## Evidence and reporting

The route change passed 26 focused local tests, including 6 real CPU integration
cases and mocked Go transport, strict-output handling, CLI-version drift,
protocol preparation, campaign isolation and package integrity. A final
20-test targeted rerun after making the key file authoritative also passed.
These overlap. The temporary local environment uses CPU PyTorch 2.9.1 and the
campaign-pinned NumPy/SciPy/pandas/sklearn versions; it is not the Mila CUDA
installation. No live Go request or GPU experiment was executed locally.

The original 143-test preparation suite passed on Mila for source 216cd88;
that run then failed its obsolete native OpenCode provider check. That receipt
is not proof of this new transport's access or readiness.

Comparison with Xixian remains qualified by fixed validation, redesigned
evolution, full-data winner selection, CLI-version uncertainty and historical
interruptions. Without a matched ordinary-BO arm, three MF arms do not isolate
improvement over ordinary BO. One run per task/arm is exploratory, not a reliable
estimate of stochastic run-to-run variance. Never use official private grading
to select programs, change this task set, or tune methods during this study.

Sources:
- `docs/MLEBENCH_LITE_AQ05_OWNER_RUN_20260921_zh.md`
- `docs/MLEBENCH_LITE_AQ05_COLLABORATOR_RUNBOOK_20260920_zh.md`
- https://dev.opencode.ai/v2/docs/console/harnesses/codex/
- https://developers.openai.com/codex/config-reference/
- https://github.com/openai/codex/releases/tag/rust-v0.164.0

## Kaggle transport correction — 2026-10-26

Mila job 11401722 reached and passed the live Go provider check, then the legacy
Kaggle download endpoint returned 404. The isolated official client downloaded
Aerial Cactus successfully in job 11402230 after the user accepted its rules.
The old endpoint still returned 404 afterward.

Download transport now uses Kaggle 2.2.5 / kagglesdk 0.1.38 in `.venv-kaggle`,
with a separate fully pinned dependency lock. MLE-Bench's Kaggle 1.6.17 and ML
runtime lock remain unchanged. The official client handles transfer/resumption;
the former custom parallel range downloader is removed. Temporary downloads
stay under each task's `.kaggle-download` directory and are promoted to the
canonical archive only after the original upstream ZIP checksum matches.
All upstream preparation, public/private checksums, fixed protocols and search
logic remain unchanged. Competition rules must be accepted by the credential's
account for every task being downloaded. No automatic acceptance is performed.

Old preparation failures remain evidence, not ready receipts. This updated
package still needs preparation and GPU preflight on Mila. Local mocked
transport tests cannot establish live download permission or data validity.

Validation for this correction: 19 targeted local tests passed (6 download
integrity/helper tests plus campaign and provider configuration checks). The
34-package download lock installed and passed dependency compatibility checks.
The engine, agents, LLM adapters, main runtime lock and fixed-protocol preparer
have no diff from 6f3aad2. No live download was performed locally.

## One-task pilot preparation

For the baseline launcher, use `submit-prepare --partition main-cpu --tasks
 aerial-cactus-identification` with the existing key-file, token-file and
 data-dir arguments. Once it succeeds, prepare the MF-GP campaign with the
 same task argument and `--protocol-source` pointing to the baseline campaign's
 protocols directory. The original manifests and task-specific resources remain
 unchanged. No unselected task is downloaded or required for protocol reuse.

Five new scope checks cover both launchers, selected protocol registration,
unchanged resources/settings, exactly two baseline jobs plus one MF-GP job,
rejection of unprepared tasks and campaign expansion, scoped result collection,
and invalid selections before any submission. They are offline checks, not
Mila readiness evidence.
