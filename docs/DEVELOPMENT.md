# Code map and focused debugging

Policy v3 is integrated and locally validated; see [implementation evidence](MFGP_V3_IMPLEMENTATION.md).
Its runner/controller live in `engine/mf_gp/v3`; the separate v3 manifest selects them.
The saved three-task study and call path below describe preserved policy v2.
Real-task Mila validation remains outstanding.

## Execution path

`mila_mfgp20.py` selects the manifest and delegates to `mila_multifidelity.py`.
The worker constructs `run.py` arguments; `run.py` dispatches MF-GP to
`engine/mf_gp/runner.py`. `AgentSearch` and the native harness retain generation,
operator legality, debugging and redesigned evolution.

| Investigate | Primary implementation | Relevant checks |
|---|---|---|
| Settings/task resources/launch | config/mlebench_mfgp20.json; tasks/mlebench_lite/mila_mfgp20.py; mila_multifidelity.py | test_mfgp_campaign.py; test_go_campaign.py |
| GP covariance/posterior/acquisition | engine/mf_gp/model.py | test_mf_gp.py |
| Action feasibility, cost estimates, promotions | engine/mf_gp/controller.py | test_mf_gp.py; test_mfgp_audit.py |
| Search budget, initialization, terminal winner | engine/mf_gp/runner.py | test_mfgp_audit.py; test_mf_gp.py |
| Data fraction and execution cost segments | engine/multifidelity.py; engine/mf_recovery.py | test_mfgp_fraction.py; test_mf_recovery.py |
| Frozen outer validation/metrics | engine/fixed_evaluation.py; tasks/mlebench_lite/prepare_fixed_protocol.py | test_fixed_evaluation.py |
| Generation and prepared legal options | engine/agent_search.py; engine/operator_preview.py; agents/mle_redesign_agent.py | test_multifidelity_integration.py; test_mfgp_audit.py |
| Program execution/timeout | engine/executor.py; engine/execution.py | test_mfgp_audit.py; test_mf_recovery.py |
| Provider transport | llm/codex.py; tools/codex_opencode.py; tools/check_go_provider.py | test_codex_retry.py; test_go_campaign.py |
| Data download failures | tasks/mlebench_lite/prepare.py; kaggle_download.py | test_kaggle_download.py |
| Resume/official grading | tasks/mlebench_lite/mila_multifidelity.py | test_mf_recovery.py; test_mila_failure_regressions.py |

Test filenames above are under tests/. Run in the prepared interpreter, selecting
only the affected tests. Some tests bind localhost or launch subprocesses; a
sandbox permission failure is not a failing model or algorithm assertion.

## Failure triage

1. Establish exact bundle commit, settings hash, protocol ID, task/mode and job ID.
2. Read sacct plus the native jobs.json ledger; do not infer success from squeue
   disappearance or create duplicate jobs to investigate an unknown submission.
3. Inspect stage receipts, traceback, execution ledger and source identity before
   changing resources or algorithms. Preserve the original failed files.
4. For a preflight with damaged stdout, separate command stdout/stderr, return
   codes and test XML. A passed rerun does not establish the first failure's cause.
5. For data/score concerns, check T/V membership and labels, actual 20% view,
   immutable-code promotion, full-data winner eligibility, and official grading.
6. Test the smallest relevant reproduction, then the required real-task gate.

## Known boundaries

- No enforced candidate filesystem sandbox; public-view instructions alone are
  not a security guarantee against accessing hidden labels or omitted rows.
- Policy v2 cost estimation uses completed runtimes and a measured task prior;
  it remains a median/paired-ratio heuristic, not architecture-aware.
- The Gaussian surrogate remains unbounded. Policy v2 stops new work at an exact
  known full-data metric ceiling; it does not clip predictions or stop on plateaus.
- Historical source archives and experiments remain useful but do not define
  current launch defaults. Keep test/demo prompts and task descriptors in place.

## Findings from the completed study

The v3 proposal maps each finding to its evidence, mechanism and proposed change.
Prioritize these paths over broad refactoring:

- Expansion timing spans candidate execution plus host review; promotion timing
  used by the estimator is candidate-process-only. Align scopes before fitting a
  new cost model. Keep total spent cost separately.
- Output scaling freezes from the first available initial reference. Context
  features depend on that scale too; changing normalization requires consistent
  raw-context storage and a new checkpoint contract, not just a new divisor.
- Pending calibration blocks ordinary allocation; the global budget guard can
  irreversibly enter terminal mode. These are current v2 behavior, not bugs fixed
  by the v3 documentation.
- New-program fidelity/cost is chosen before child generation. Any deferred-choice
  implementation must separate generation reservations from pending observations,
  count sunk costs, bound prepared code and preserve exact recovery state.
- Syntax and partial-patch errors reached real executions. A new pre-execution
  contract needs coverage tests; passing a tiny probe cannot guarantee a long run.

Do not attribute the rare cheap decisions exclusively to runtime estimates: the
logged acquisition value of low alternatives was often orders of magnitude
smaller even when their estimated costs were lower. An offline alternative-policy
replay cannot provide unknown rewards for actions the original run did not take.

## Safe documentation maintenance

Only human-facing documentation was reorganized in the 2026-11-03 cleanup.
The protected-file manifest outside this repo covers all tracked non-doc files
and runtime Markdown (prompts/task descriptors). Verify it before committing.
Do not rerun expensive GPU tests merely because documentation moved. Full source
bundle hashes will still change with a documentation commit; preserve old bundles.

The 2026-11-05 research review is also documentation-only in this repository.
Pre-edit copies and all tracked-file hashes are under the evidence hub's
next_version_2026-11-05/. Verify that only the explicitly listed guidance files
changed. Do not treat this documentation update as a runtime correction or a
reason to rerun cluster gates.
