# Contextual multi-fidelity GP mode

This mode implements the approved third experiment. It is opt-in via
`evaluation.multifidelity_mode=mf_gp` and requires the AQ05 preset, redesigned
ML evolution and host fixed evaluation. `carry_over` and `reset` retain their
existing staged runner, selector and campaign manifest.

## Search and data

- Same frozen random half of T, full T and unchanged complete outer V/test.
  Full T excludes V. Public data/outputs are bound separately for every attempt.
- Three native initial cheap drafts. Best valid cheap draft and one seeded-random
  remaining draft receive full evaluations, when affordable. These pairs are
  initialization evidence, not sufficient correlation estimation.
- One GP selects cheap expansion, full expansion or promotion. Saved executable
  code is retrained unchanged for promotion; no new child, review or LLM call.
  The same host seed initializes Python, NumPy and Torch for each execution;
  this does not guarantee deterministic GPU kernels or externally seeded libraries.
- Native parent eligibility, operation previews, redesign, debugging and
  aggregation remain in charge of generating programs. Promotion eligibility
  is independent of evolutionary parent eligibility and child limits.
- Every real observation is retained with its fidelity. No predicted quality
  is admitted as a label. Failed/censored outcomes remain recorded without a
  fabricated numeric score. Mixed-fidelity gains are not evolutionary gains.
- Population ranking uses measured full utility when available and predicted
  full utility otherwise. This never overwrites the original node metric.
  Redesign evidence includes both evaluations with explicit fidelity labels.

## Surrogate and acquisition

`L_p = g(x_p) + u_p`; `H_p = a + rho L_p + d(x_p) + v_p`.
Context x is frozen before generation: parent quality, same-fidelity birth and
local gains (with missingness), attempt count, parent fidelity and rating.
Matern-3/2 context covariance with operator similarity plus persistent identity
variance distinguishes exact-code pairs from stochastic siblings. Independent
execution noise is added on the observation diagonal. All targets are absolute
signed host scores on one scale frozen from valid initial cheap drafts.

The model contract is versioned in `engine/mf_gp/model.py`: fixed lengthscale1,
identity variance0.5625, standardized execution noise SD0.1 per fidelity,
initial rho0.8, offset0, low amplitude1, discrepancy amplitude0.7. After at least
8 valid observations including4 full observations, bounded MAP fits rho, offset
and the two amplitudes with Gaussian/log-scale priors. Lengthscale, identity
fraction and noise remain fixed to avoid fitting unidentifiable components from
few pairs. Uncertainty is conditional on these regularized parameters; it is
not a claim that two initial pairs establish reliable transfer.

The cost-aware retained-outcome target-H KG is a **one-step contextual
approximation**, not stock fixed-domain MF-KG or an exact budget-optimal policy.
Each decision freezes the target pool across alternatives. L/H alternatives of
one prospective child share identity; subsequent expansions get fresh identities.
Completed full results contribute their actual incumbent. Pending outcomes use
posterior-mean conditioning (kriging believer), without inserting fantasy labels.

Costs use observed generation/execution times, paired full/cheap ratios and a
1.5x conservative runtime guard, including the complete full rerun, validation
and inference. Before any measured pair the ratio starts at2, an assumption
that subsequent timings replace. These guards are heuristics, not deadlines
proved safe for every newly generated program. Near the deadline, stop cheap
work when a cheap run plus a subsequent full run no longer fits; continue
feasible full-data actions by expected improvement per additional time.
Only completed valid full-data submissions can win.

All elapsed active search time, initial pairs, promotions and search overhead
count toward12h. Promotions count toward the500 action cap. Three workers share
one GPU; runtime savings are not assumed to equal the training-data fraction.

## Isolation and launch

Use the separate entry point:

```
python3 tasks/mlebench_lite/mila_mfgp.py --help
```

It has the same prepare / preflight / submit / resume / collect commands as the
existing launcher, but defaults to `runs/mf14_gp`, uses
`config/mlebench_mfgp14.json`, and submits only `mf_gp`. Do not reuse an old
campaign directory or change its source package. The original 14x2 campaign
manifest and default modes remain unchanged.

A direct single-task search uses the existing launcher command configuration
with `evaluation.multifidelity_mode=mf_gp`; `run.py` routes that mode to the
separate adaptive runner. Preserve the same native provider, seed, resources,
protocol and operation settings as the comparison modes.

## Recovery and evidence

The native decision ledger preserves in-flight generation and frozen code.
Promotions have separate immutable evaluation records and reservations in the
MF selector checkpoint. Execution caps are tracked by(program,fidelity), so a
cheap evaluation does not consume the full-run execution cap. An interrupted
execution is retried from scratch only after the old allocation is verified
released; lost execution time remains charged. There is no training checkpoint
reuse across fidelities. Protocol/model contracts prevent cross-mode resumes.

`mf_gp_decisions.jsonl` records candidate actions, costs, acquisition scores,
incumbent, pending count and selected action. `mf_gp_fits.jsonl` records fit
admission and model diagnostics. `final_submission.json` identifies both the
program and winning evaluation. Official private grading is performed only
after the search has selected its completed full-data artifact.

## Experiment interpretation

The question is whether adaptive MF improves the final official MLE-Bench result
within the same 12-hour active budget. Compare each task against the frozen
carry-over/reset campaign with the same model route, evaluator, random data
subset, worker/GPU resources and search limits. The intervention includes both
the surrogate and fidelity controller; this comparison does not isolate the GP
kernel alone. Legal actions are native expansions at either fidelity and saved
program promotions; private grading never enters selection. The experimental
unit is one complete task/run, not individual children or correlated snapshots.
Report all 14 tasks, failures, missing/full-data outcomes, interrupted runs,
actual active time and evaluation counts alongside official grading. A run with
no valid completed full evaluation is a failure, never a cheap-score substitute.
One stochastic run per task supports exploratory comparison, not a general claim
of superiority. Keep the original campaign artifacts and source revision frozen.

## Validation boundary

CPU integration tests use synthetic LLM replies but real execution, fixed CSV
scoring, promotions and checkpoint restoration. They are not evidence of GPU
speedups or improved MLE-Bench results. Run the campaign's CPU preparation and
short GPU/provider preflight on Mila before starting full searches. The new
campaign explicitly includes `tests/test_mf_gp.py` in both gates.

### Local verification (2026-10-24)

- Final `tests/test_mf_gp.py`: **13 passed**, including actual candidate processes,
  two-fidelity views, search completion, promotions, learned-GP restoration and
  killing/resuming a promotion without admitting an interrupted score.
- **79 existing regression tests passed** across `test_multifidelity.py`,
  `test_multifidelity_campaign.py`, `test_multifidelity_integration.py`,
  `test_mf_recovery.py` and `test_fixed_evaluation.py`.
- The combined run initially reported 91 passed and one MF test failure: the
  test finalized a stale journal without capturing the recovered state. After
  aligning it with the production runner's checkpoint-before-finalization order,
  the final 13-test MF suite passed. No failed production evaluation was excluded.
- Additional AQ05 math/runtime, redesign and reward tests: 54 passed, 46 failed.
  The identical 46 failure IDs and 54 passes were reproduced from untouched
  baseline commit `a5986ac`; failures report shared-draft generator configuration
  mismatches. Thus this is not a claim that the entire repository suite passes.
- Baseline SHA256 comparisons confirm unchanged `engine/multifidelity.py`,
  `config/mlebench_mf14.json`, AQ05 selector and AQ05 acquisition. Shared entry
  points contain explicit MF-only routing; old campaign defaults remain intact.
- No live model-provider, Mila CUDA or complete MLE-Bench run was performed.

Tests used the local CPU preflight Python environment with synthetic model
responses. Existing journal deserialization warnings remain visible in pytest
output. The campaign preparation and GPU/provider checks remain mandatory.
