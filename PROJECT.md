# Current research contract

## Objective

Improve the existing BO framework's final MLE-bench performance under a fixed
active-time budget using multi-fidelity evidence and cost-aware resource
allocation. BO remains the search engine. Novelty is not a substitute for valid
and useful benchmark outcomes. No performance improvement is guaranteed.

## Selected experiment

- Tasks: Jigsaw Toxic Comment Classification, SIIM-ISIC Melanoma Classification,
  Dog Breed Identification.
- Mode: contextual MF-GP, 20%/100% training; unchanged complete fixed validation.
- Each task: 43,200 active seconds, 500 actions, seed 42, three initial drafts
  and workers, one A100 80 GB, candidate cap 32,400 seconds.
- Muse Spark 1.3 Contributor/high through codex://opencode-go; CLI 0.164.0.
- Jigsaw: 4 CPUs/64 GB, from the owner record. Other tasks: 21 CPUs/128 GB, from
  the collaborator runbook; these are not verified completed owner allocations.
- Treat config/mlebench_mfgp20.json as the machine-readable specification.

## Interpretation boundary

The statistical unit is a complete stochastic task run. Nodes in a shared
trajectory are dependent, and promoted evaluations are not new unique programs.
The current three-task campaign is exploratory, not a matched superiority test.
Report official full-data winner scores, failures, missing outcomes, total costs,
and validation trajectories. Test scores are obtained only after winner selection.
Historical Xixian comparisons have evaluation/generator/resource differences.

The approved 20% campaign uses scheduling policy v2: completed-runtime evidence,
a measured same-code task cost prior, two asynchronous calibration slots, and
exact-known-metric-ceiling stopping. A profile is a prior for one workload, not a
runtime guarantee for future agent programs. Legacy manifests retain policy v1.
See docs/MFGP_POLICY_V2.md for profiling and launch requirements.

## Preservation and operations

Keep legacy 50% campaigns reproducible and separate. Preserve frozen source,
protocols, settings and spent budget. Do not change running cluster code when
editing this checkout. Existing prepared cluster bundle cb09bec is unchanged by
this documentation cleanup; a new local commit requires a new bundle for future
campaigns, not an in-place replacement of the existing one.

Local operational receipts and artifacts are maintained outside the source repo:
/Users/mac/Documents/ChatGPT/AI4AI/deliverables/mila-mfgp14/.
For current job IDs and remote paths consult mf20_three_tasks/OPERATIONS.md and
then verify live Slurm/artifact state. Do not treat stale notes as live status.
