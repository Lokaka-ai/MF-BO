# Implemented MF-GP scheduling policy v2

Status updated 2026-11-05: the three-task v2 campaign completed in frozen source
14605cb. Its reports and trajectories are under the evidence path in PROJECT.md.
The behavior below remains the reproduction contract. The [v3 proposal](MFGP_V3_PROPOSAL.md)
documents observed limitations and requested changes; those changes are not implemented.

Policy 2 is explicit in `config/mlebench_mfgp20.json`. Default config and legacy
50% manifests retain policy 1. The GP covariance, noise, features, target-full
acquisition, Harness operations, 20% subset, fixed validation and winner rule
are unchanged. Frozen cluster bundles must not be patched in place.

## Runtime evidence

All execution time remains in the accounting ledger. Only executions with a
completed process become successful-runtime samples. A completed process with
an invalid validation submission still has a measured execution duration; it
is not admitted as a quality label. Early failures and timeouts do not lower
the estimated successful full runtime. Comparable full timeouts provide lower
bounds to the feasibility guard. Generation cost still includes unsuccessful
code generation. Promotions use process wall time for cost estimation and
retain total promotion time, including host scoring, separately.

The first ratio and durations come from a measured same-code low/full profile.
Completed search pairs subsequently replace that ratio using their median.
This is a simple heuristic, not an architecture-aware predictor or guaranteed
finish-time bound. The full guard keeps a 1.5 margin and upper completed costs.
Profiling does not eliminate contention or model-family differences.

## Calibration

The first available valid initial candidate is selected by cheap quality. The
second slot uses a seeded random eligible initial candidate. If only one exists,
normal affordable work can continue and a later successful initial repair may
fill the second slot. Existing slots are never replaced; attempted full failures
remain attempted slots, not silently retried. Existing pending calibration is
allowed to finish before ordinary acquisitions, as before. All promotions remain
subject to the existing remaining-time guard; unaffordable slots do not force
work or indefinite waiting. Decisions and slot additions are logged.

## Exact ceiling

New decisions stop when a valid, budget-eligible full-data observation exactly
reaches a known metric bound: 1 for maximizing ROC AUC, accuracy or quadratic
kappa; 0 for minimizing RMSE, RMSLE or log loss. There is no epsilon comparison,
plateau rule or claim about unseen performance. Already committed work drains
normally and the existing finalizer selects the full-data winner. Unknown metrics
never trigger this rule. The surrogate is not clipped or replaced.

## Measured profile before a new full run

Run inside an appropriately provisioned compute allocation, using the prepared
Python environment. `tools/profile_mf_cost.py` does not submit a job. Supply a
fixed representative training pipeline that uses `./input` and includes normal
validation/test inference. It must not use hidden labels. It is engineering work,
not an experimental candidate; its scores must not be used to select the pipeline.
Keep its resource usage and receipt with the campaign and report this setup cost.

```sh
.venv-mlebench/bin/python tools/profile_mf_cost.py \
  --candidate /absolute/path/to/fixed_workload.py \
  --protocol runs/mf20_gp/protocols/TASK \
  --work runs/mf20_gp/profile_work/TASK \
  --output runs/mf20_gp/cost_profiles/TASK.json \
  --fraction 0.2 --seed 42 --timeout 300
```

The same bytes run in separate low/full working directories. Outputs and logs are
preserved; failures/timeouts produce no admissible cost prior. No official grading
is performed. The timeout is an engineering cap, not a license to treat a partial
run as a completed measurement. Use `--local` only for intentional local work.
Existing profile paths/workspaces cannot be overwritten.

The launcher validates all selected task profiles before submitting any full
worker and freezes their parsed values and receipt hashes in
`cost_profiles/submitted.json`. Worker/resume paths recheck them; the selector
checkpoint contract also includes the prior and policy version. Changed profiles
or switching policy versions cannot silently resume an existing run.

The initial implementation note said the campaign had not yet been profiled or
certified. That readiness statement is superseded: the 14605cb campaign subsequently
passed preparation, GPU and task gates, measured its priors and completed all
three tasks (Jigsaw completed through budget-preserving recovery). See
`analysis/mf20_v2_2026-11-04/REPORT.txt` in the evidence hub. This verifies that
campaign, not arbitrary future workloads or a proposed v3 implementation.

Audit caveat: expansion runtime samples currently include host scoring/review,
whereas promotion/profile samples use candidate-process wall time. The estimator
pools those scopes. Historical receipts are unchanged; harmonizing the scopes is
an explicit v3 correction, not something policy v2 already guarantees.

## Verification

`tests/test_mfgp_policy_v2.py` covers early failures, censored lower bounds,
asynchronous calibration, metric-bound eligibility, profile provenance, actual
profiling subprocesses, resume contract rejection and the real synthetic
20% execution/promotion path. Existing MF-GP/fraction/recovery tests cover the
unchanged legacy and shared paths. No live model or benchmark call is needed.
