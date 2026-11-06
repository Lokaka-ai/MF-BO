# Maintainer instructions — Multi Fidelity BO

## Scope and sources of truth

- Read PROJECT.md, then docs/DEVELOPMENT.md. Policy v2 is implemented; its
  three-task study is completed in the 2026-11-04 evidence snapshot. The approved v3 is integrated and locally validated; see
  docs/MFGP_V3_IMPLEMENTATION.md. Real-task Mila validation remains outstanding.
  config/mlebench_mfgp20.json and tasks/mlebench_lite/mila_mfgp20.py reproduce v2;
  they are not a new launch plan.
- Current canonical checkout: /Users/mac/Coding Project/Multi Fidelity BO.
  The old AI4AI/projects/bo_engine/mfgp20 path is a compatibility symlink.
- docs/archive and dated owner/collaborator reports are historical evidence.
  They do not override current user decisions or current campaign settings.
- Preserve old 50% carryover/reset/MF-GP workflows. Ten percent is supported for
  diagnostics, but is not the selected campaign.

## Evidence and project state

- Start with the completed trajectories under the evidence hub named in PROJECT.md,
  verify the actual implementation, then use literature to assess interventions.
- Preserve the distinction between implemented behavior, observed outcomes, and a
  proposed policy. V3 is locally implemented; do not describe it as cluster-validated.
- Optimize the final eligible full-data outcome under budget. Extra cheap actions
  or generated nodes are not evidence of better performance by themselves.
- Keep project guidance concise; put mechanism, evidence and implementation plans
  in the linked proposal. Operational receipts are history, not live job status.

## Scientific and implementation invariants

- Harness prepares legal parent/operation options; BO ranks prepared options.
  MF-GP additionally chooses fidelity or an immutable-code promotion.
- Low data is a fixed seeded uniform subset of training T; V/test stay unchanged.
  Never admit predicted full scores as observed labels or select a cheap winner.
- Preserve failures, source/config/protocol hashes, actual cost and consumed
  budgets across interruption. Never silently restart or erase failed artifacts.
- Before a substantive design change, discuss and obtain user approval. Routine
  fixes preserving approved semantics may proceed with focused verification.
- Stop generation on a confirmed scientific-validity defect. Do not use official
  test outcomes to tune the controller or select candidates.
- Review actual execution paths; do not infer correctness from names or comments.
  Distinguish descriptive observations from causal claims and synthetic checks
  from real-task validation. Same seed does not guarantee paired LLM sampling.

## Work efficiently

- Use targeted searches and tests for the affected path. Avoid broad refactors,
  speculative features, or repeated full test suites for documentation changes.
- Do not move/delete prompts, config files, modules, task descriptors, fixtures,
  or pinned dependencies as 'unused' without checking their consumers.
- For docs-only maintenance, verify executable/config/prompt hashes unchanged;
  archive superseded prose rather than discard provenance.
- Keep secrets out of output/Git. Never read or print credential file contents.
- Do not launch or resume experiments outside existing user authorization.
  Check native campaign jobs.json and Slurm first to prevent duplicate runs.
- Do not create scheduling automations. Ask for SSH renewal when necessary;
  never request the OTP in chat. Do not update First Mila Cluster Experiment Logs
  unless explicitly asked.
- CLI help is safe; submit/prepare commands may allocate resources, call models,
  or download data. Follow the campaign's actual readiness and recovery gates.
