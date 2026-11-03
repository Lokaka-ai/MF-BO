# MF-GP implementation and launch audit — 2026-10-25

> **Superseded as launch-readiness evidence on 2026-10-28.** Mila pilot jobs
> 11417260–11417262 exposed two host-memory failures, a failed MF-GP draft path,
> delayed fidelity switching, checkpoint recursion, and an unreported official
> grading error. The verification and source-isolation claims below describe
> the earlier source snapshot only. Use the repaired source and its fresh
> preflight before any new full experiment.

## Current campaign update

The 2026-10-26 approved provider revision is documented in `MILA_GO_ROUTE.md`.
It supersedes native OpenCode routing described in the historical audit below.

## Outcome and boundary

The local engineering checks below passed after correcting reproducible defects
in implementation commit `cb400eb`. The source-only bundle is built from the
clean follow-up commit and records that commit in `BUNDLE_INFO.json`.

No full MLE-Bench experiment, live provider call, real Kaggle download, or Mila
allocation/CUDA test was performed in this audit. Those remain explicit launch
gates, not assumed successes. No performance improvement is claimed.

## Corrections

1. **Failed-program repair:** completed failed drafts now return to the legal
   pool and receive the native forced debug operation. Pending repairs prevent
   duplicate reservations. A repaired initial program remains eligible for the
   initial calibration pair. Debug/aggregation bypasses obey the same remaining
   time and fidelity guards as ordinary actions.
2. **Budget accounting:** forced initialization respects total active time;
   recovered generation already counted as submitted is not charged twice.
   Resuming a reserved initial draft does not create extra initial drafts.
3. **Recovery entry paths:** MF-GP recovery bookkeeping is enabled even without
   the campaign environment variable. The shared recovery API dispatches MF-GP
   to separate `(program, fidelity)` execution caps. Optional result-file
   reporting no longer prevents a direct run from starting.
4. **Valid Python programs:** host seed setup and Linux CPU-affinity setup retain
   legal module docstrings and `from __future__` imports at the top of generated
   programs. The saved original code and promotion identity are preserved.
5. **Matched experimental evaluation:** the new `--protocol-source` option pins
   and reuses the baseline campaign's exact T/V/scorer. A separate LLM-generated
   protocol with the same seed is not assumed to be equivalent. Existing source,
   seed, membership and scorer audits still run before search.
6. **Packaging:** a deterministic clean-commit bundle includes per-file SHA256
   hashes and an archive checksum. Untracked credentials, data, environments and
   runs are excluded; dirty source and conflicting existing bundles are refused.

These corrections preserve the approved model and action space: contextual
AR co-kriging, cheap/full expansions, unchanged-code full promotions, cost-aware
acquisition, deadline full-only actions, and measured-full-only final selection.

## Verification

| Check | Observed result |
| --- | --- |
| Campaign CPU preparation test command | **165 passed**, including the old carry-over/reset cases, real CPU candidate execution, fixed evaluation and process-kill recovery |
| Final MF-GP, audit and campaign tests | **24 passed** across final targeted invocations, including the subsequent forced-operation deadline and shared recovery routing checks |
| Native AQ05 math/runtime, redesign and reward tests | **100 passed** |
| Linux dependency dry resolution | **240 locked packages resolved**, Linux x86_64/manylinux 2.28 and Python 3.12, including pinned CUDA PyTorch wheels and MLE-Bench commit `507f92e1138bb6e40dac5c6ee7a6758e6424bf97`; nothing installed into the target |
| Controller scale check | 500 observations, 500 targets, 1,000 actions; fit **2.73 s**, acquisition **1.41 s**, finite nonnegative scores; local CPU, BLAS threads=1 |
| Offline submission | Exactly 14 MF-GP jobs; correct dedicated entry point, GPU request and mode; duplicate submissions suppressed |
| Package tests | Repeatable archive checksum, ignored credential excluded, per-file corruption detected, dirty source rejected |
| Static checks | Shell syntax, launcher help and `git diff --check` passed |

Counts overlap; they are not a sum of independent experiments. The preparation
suite ran before the final two targeted additions; those additions and their
affected MF paths passed separately. Tests use synthetic LLM replies and real
local training/scoring processes. Kill/recovery tests use local processes and a
synthetic scheduler-release receipt, not a live Slurm preemption event.

The previous 46 native test failures were caused by stale shared-draft model
names in test fixtures. Fixtures now use the actual configured model; the
macOS-only diagnostic shim supplies CPU affinity to tests. No production AQ05
selector/acquisition changes were made. All 100 tests in that audit set now pass.
Existing dataclass deserialization warnings remain visible; assertions were not
removed or suppressed to obtain these results. The local test environment is a
CPU preflight environment, not the complete Linux/CUDA lock installation.

## Isolation and experimental settings

SHA256/content comparisons against `a5986ac` confirm byte-identical:

- `engine/multifidelity.py` (carry-over/reset staging and subset construction)
- `config/mlebench_mf14.json` (original two-mode campaign)
- `engine/parent_selectors/acq7_selector.py`
- `engine/parent_selectors/acq7_acquisition.py`

Shared execution/recovery changes are guarded by MF-GP mode or its per-attempt
context flag. The old campaign's mode defaults are unchanged. The previous Mila
archive is not replaced. MF-GP uses its own manifest, launcher and run directory.

Checked settings: the same retained 14/22 Lite tasks, 12-hour active budget,
500 total actions including promotions, 3 workers/drafts, seed 42, one A100 80GB,
9-hour per-program/fidelity execution cap, fixed validation, redesigned evolution
and native OpenCode/high reasoning. Existing task-specific CPU/RAM allocations
are retained. The 13-hour Slurm wall request is distinct from the search budget.

The documented `long` resource limits permit all those allocations; several
exceed `main` or `unkillable` CPU/RAM limits. Resource availability, account
permissions and actual hardware still require live cluster verification.
Sources: [Mila partition limits](https://docs.mila.quebec/technical_reference/clusters/mila/partitions/),
[Mila node profiles](https://docs.mila.quebec/technical_reference/clusters/mila/nodes/).

## Handoff

Follow `MILA_MFGP.md`. For a matched comparison, provide the existing baseline
protocols directory with `--protocol-source` and preserve that directory.
Preparation, actual data/protocol admission and short GPU/provider preflight
must pass before the launcher admits a full search. Keep every failed, missing
or interrupted task in the reported denominator.

## Mila follow-up: reset integration test lineage (2026-10-25)

Preparation job 11394186 on source a5986ac stopped with 142 passed and one
failed reset integration test. Targeted job 11394457 reproduced the failure.
The recorded full nodes 3b2710d2a51a40efac96fab7ce3b1dde and
e2fb5d945f384d009dad1c920129dc30 both had cheap parent
497a0a8a177042dfac02d4eb2f83dd45. There was no full/full transition, so zero
reset GP rows was correct. The test incorrectly treated agent.step(parent)
as forcing that parent, although AQ05 ranks other legal parents as well.

The test now constrains its candidate support to the specified legal parent,
while retaining real AQ05 prediction, reservation, execution, observation and
refit. It explicitly checks that two full siblings of a cheap parent add no
reset rows, then that a full child of a full parent adds exactly one. Existing
carry-over, full-only winner and restored-state checks remain. All six tests
in test_multifidelity_integration.py passed locally (68.34 s, 16 existing
serialization warnings). Production code and campaign parameters are unchanged
by this correction. This is a test of filtering/refitting under prescribed
lineages, not evidence about unconstrained parent selection or performance.
The corrected source must still pass preparation and GPU preflight on Mila.
