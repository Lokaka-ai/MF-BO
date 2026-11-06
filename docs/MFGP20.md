# Implemented 20% MF-GP contract

Status updated 2026-11-05: this fraction/cost-record implementation was followed by
[policy v2](MFGP_POLICY_V2.md) and the completed three-task campaign in source
14605cb. The historical implementation/test notes below are retained with their
scope. Current research is the [v3 proposal](MFGP_V3_PROPOSAL.md), not another v2 launch.

This checkout branches from frozen pilot commit 315687813e501695ac249046779016965ede0d6b.
The prior source repository and campaign manifests are preserved. The initial
implementation was verified locally before the later cluster preparations.
Use external operational receipts and live Slurm evidence for current status.

## Task and fidelity contract

- Completed policy-v2 campaign: Jigsaw, SIIM-ISIC Melanoma, Dog Breed Identification.
- Melanoma and Dog Breed: 21 CPU / 128 GiB host memory / one A100 80 GB,
  from the collaborator runbook (not verified completed owner allocations).
- RANZCR remains prior development evidence; excluded from this new campaign.
- Jigsaw Toxic Comments: historical comparison available, 4 CPU / 64 GiB / one A100 80 GB.
  Jigsaw uses the historical owner allocation; it was not 24 GiB.
- New launcher: `python tasks/mlebench_lite/mila_mfgp20.py --help`.
- New manifest: `config/mlebench_mfgp20.json`; default campaign `runs/mf20_gp`.
- Low training fraction 0.20; target 1.00. The existing MF-GP manifest/launcher
  and carryover/reset retain 0.50. Global config defaults remain legacy-compatible.
- `evaluation.mf_gp_low_fraction` supports 0.10, 0.20, 0.50. Ten percent is for
  separately approved diagnostic campaigns; changing a live campaign is rejected.
- Full validation and test stay fixed. Training subsets are uniform seeded
  sample-ID subsets; smaller prefixes are nested inside the legacy half subset.
  Existing 50% IDs and manifest bytes remain unchanged. The sampling unit is a
  sample ID, not necessarily a patient. Empty subsets are rejected.
- GP covariance remains two-level autoregressive co-kriging. Actual low fraction
  enters the frozen contract; observations from a different low fraction cannot
  be loaded into the model. Promotions reuse code and retrain at 100%.
- Full-data-only winner eligibility and initial cheap-work deadline are retained.

## Measured cost records

Execution ledgers `logs/mf_execution/*.json` retain cumulative elapsed time and
now also contain individual attempt segments, code/fidelity identity, Slurm job
ID, pre-dispatch action estimate where available, exception status, and whether
an observed time is a completed duration or a censored lower bound.

At a confirmed scheduler release, a running segment becomes interrupted exactly
once. Interrupted work remains charged. A pre-existing ledger with no segments
keeps its earlier cost as `unsegmented_elapsed_seconds`. Queue/preemption downtime
is not treated as candidate training. Release-bounded durations are identified
as such, because the exact process kill time may be unavailable.

`host_cost.json` beside evaluated outputs records successful host scoring duration
and scorer validity. It does not measure training/validation inference separately.
LLM generation and model-fit/acquisition evidence retain their existing logs.
Failed generation has no execution cost and remains in transition/LLM evidence.
Missing measurements remain unknown; no synthetic GPU utilization, RAM peaks,
token dollar costs or training-only durations are inferred.

Export read-only accounting:

```bash
python tools/mfgp_cost_matrix.py --log-dir /path/to/run/logs --output /path/to/new-costs.csv
```

The exporter checks cumulative/segment consistency and refuses output overwrite.
Worker seconds overlap; they are not elapsed search seconds or allocated GPU-hours.
An expansion estimate includes generation, while a promotion estimate excludes
generation. Do not compare expansion prediction to process runtime alone.

## Historical fraction-only scope and later policy-v2 changes

The original fraction-only change added cost measurement without a new fitted
runtime model; it retained the historical median/paired-ratio estimator and a
default ratio of 2. That uncalibrated ratio is still relevant to legacy defaults,
but is not the selected policy-v2 campaign prior. Policy v2 requires a measured
same-code profile and uses completed-runtime evidence. It changed calibration
slot handling and exact-ceiling stopping; see its own contract for details.

The two-level GP and regularization remain. Limited paired evidence was observed
in RANZCR and again in the completed Melanoma study. Fraction support and a measured
cost prior do not establish cross-fidelity calibration or final-performance gains.

Use identical-code low/full diagnostics to measure runtime saving, rank changes,
minority-label coverage, failures and memory feasibility. Report diagnostic costs
separately. No benchmark test labels/scores may select the low fraction. A new
matched full-only baseline is required to attribute a gain to this intervention;
historical scores alone cannot establish that comparison.

Local regression checks use synthetic LLM replies and real CPU candidate/scorer
execution. They cannot establish GPU runtime, real-task memory feasibility or
20% ranking reliability. Local test dependencies may differ from the frozen
Linux lock; the existing preparation/provider/GPU gates remain required.

## Local verification, 2026-11-02

45 existing MF-GP/staged-mode/campaign/recovery tests passed in 265.52 seconds.
14 new fraction/accounting tests passed in 15.62 seconds. The new integration
executes real CPU candidates at 20%, promotes identical code, verifies full-data
winner eligibility, restores the posterior and rejects a changed fraction on
resume. Other checks cover 10%/20%/50% covariance and nested samples, legacy sample
identity, censored cost retention and idempotent interrupted-cost reconciliation.
No external model calls or Slurm jobs were used. Compile checks, CLI help and
Git whitespace checks passed. Full logs and local package versions are saved in
AI4AI/deliverables/mila-mfgp14/next_stage_2026-11-02/.
