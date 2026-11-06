# Current research contract

Updated 2026-11-05. Current stage: v3 integrated and locally validated; see docs/MFGP_V3_IMPLEMENTATION.md.
Policy v2 at 14605cb remains the historical reproduction baseline. Mila readiness is unverified.

## Objective and decision

Improve final MLE-bench performance under a fixed active-time and resource budget
using Bayesian optimization and multi-fidelity evidence. BO remains the search
engine. Performance and scientific validity take priority over novelty, program
counts, or making the controller select a prescribed amount of cheap work.

The approved v3 implementation is complete locally; the next gate is a separately
authorized Mila preflight, not another full performance campaign. Codex leads the evidence review and recommends
changes; substantive design choices require user approval before implementation.
See [the v3 proposal](docs/MFGP_V3_PROPOSAL.md). It is implemented locally; it is not a launch
instruction. No performance improvement is guaranteed.

## Completed study and reproduction settings

The 20% policy-v2 study completed Jigsaw Toxic Comment Classification, SIIM-ISIC
Melanoma Classification, and Dog Breed Identification. Frozen source: 14605cb.
The local 2026-11-04 evidence snapshot contains their completed trajectories and
official reports; this is artifact-derived status, not a live cluster query.

- Mode: contextual MF-GP, 20%/100% training; complete fixed validation unchanged.
- Per task: 43,200 active seconds, 500 actions, seed 42, three initial drafts and
  workers, one A100 80 GB, candidate cap 32,400 seconds.
- Muse Spark 1.3 Contributor/high through codex://opencode-go; CLI 0.164.0.
- Jigsaw: 4 CPUs/64 GB, matching the historical owner allocation. Melanoma and
  Dog Breed: 21 CPUs/128 GB, from the collaborator runbook; that runbook does not
  establish completed original-BO results for those tasks.
- config/mlebench_mfgp20.json remains the machine-readable v2 specification.
  It is preserved for reproduction, not authorization to rerun.

Policy v2 uses completed-runtime evidence, a measured same-code cost prior, two
calibration slots, and exact-known-metric-ceiling stopping. It still has a pending
calibration barrier and heterogeneous cost scopes; see the proposal's code audit.
A measured profile is evidence for one workload, not a guarantee for future code.

## Evidence and interpretation

Evidence hub: /Users/mac/Documents/ChatGPT/AI4AI/deliverables/mila-mfgp14/.

- Latest results: analysis/mf20_v2_2026-11-04/REPORT.txt, summary.json and
  evidence.json.gz; analyze.py reproduces the main tables.
- Earlier 50% pilots: analysis/lab_meeting_2026-11-02/.
- New design diagnostics and preserved pre-edit guidance: next_version_2026-11-05/.
- Operational history: mf20_policy_v2/OPERATIONS.md. Earlier
  mf20_three_tasks/OPERATIONS.md and ACTIVE_PILOT_OPERATIONS.md are historical.
  Their queued/running status and SSH socket names may be stale.

The unit is a complete stochastic run. Nodes within one trajectory are dependent;
promotions are not new unique programs. Observed paired runtime savings do not
establish a final-score benefit from multi-fidelity. All three v2 winners were
new full-data programs, and ordinary cheap exploration was rare. Historical
Xixian comparisons differ in evaluation, generation or resources; they are not
matched superiority tests. Same seed does not imply paired LLM sampling.

Use official test scores only after validation-based winner selection. They must
not tune the controller or reselect candidates. Separate verified observations,
mechanistic hypotheses and claims not supported by the sample. Completed tasks
now inform development; fresh tasks and matched controls are needed for broader
claims. Validate candidate access boundaries before claiming leakage freedom:
filesystem isolation is not currently enforced.

## Preservation and next steps

Keep 50% pilots and 20% v1/v2 campaigns reproducible. Preserve frozen source,
protocols, settings, failures and consumed budgets. Never patch a prepared or
running bundle during local maintenance. A documentation commit changes source
bundle identity even when executable behavior is unchanged.

The proposed v3 keeps the two fidelities, GP and Harness, and changes cost
accounting, checks before training, fidelity allocation after generation, GP
scaling, calibration and budget feasibility. MF-MES is an offline challenger,
not a selected replacement. Runtime implementation is locally validated. New cluster experiments
remain a separate approval step. Do not launch or schedule work from this document.
