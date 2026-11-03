# Local audit — 2026-10-22

The earlier dependency blockage below was resolved using an isolated temporary
environment with the campaign's numerical versions: Python 3.12, PyTorch 2.9.1
(Mac build), NumPy 2.2.6, SciPy 1.16.3, pandas 2.3.3, scikit-learn 1.6.1.
This is not the Linux CUDA lockfile environment.

- 126 tests passed across `test_multifidelity.py`,
  `test_multifidelity_campaign.py`, `test_study7_math.py`,
  `test_fixed_evaluation.py`, and `test_acq7_math.py` (176.56 seconds).
- Two new tests passed for AQ05 carry-over/reset: real CPU candidate training,
  scoring, GP fit/admission, full-data winner eligibility, and journal restore.
- Three additional tests passed: all 28 commands through the actual config
  loader, and the real three-worker runner through drained half-to-full switches
  in both modes (25.92 seconds). The runner's budget clock was advanced for the
  test; no 12-hour wall-clock run was performed.
- Six offline campaign tests passed after adding the required GPU preflight gate.
- Fixed an idle-GPU bug: when cheap-stage legal work was exhausted before 40%,
  the runner previously waited until the switch time with no work running. It
  now exits with an explicit failure. Its real-runner regression passed (6.95
  seconds); both successful mode-switch tests also passed after this change.
- Model replies are synthetic in integration tests; CPU candidate execution,
  scoring and GP computations are real. On macOS only, the Linux affinity
  diagnostic is supplied by a fixture; the production Linux path is unchanged.
- Source mode settings remain 50% uniform T, full V, 40% drain/switch, AQ05,
  12-hour budget, 500 steps, three drafts/workers, seed 42 and nine-hour cap.
- Added a mandatory 20-minute maximum Mila GPU preflight before full submission:
  CUDA training, embedding, live provider and both-mode integration checks.
- The six new integration cases are all included in CPU preparation and the
  short GPU preflight. All 133 distinct CPU/offline checks passed across the
  audit runs; this does not claim a full benchmark or Linux CUDA run.
- Remaining verification requires Mila: the locked Linux environment, actual
  A100/driver compatibility, provider credentials, task data/protocol semantics
  and cluster scheduling. High-memory jobs require preemptible `long` under the
  documented limits; explicit recovery is implemented as described below.

## Interruption recovery — 2026-10-22

- Final combined run: 32 tests passed in 79.52s across legacy checkpoint,
  MF recovery and campaign launcher suites. The six launcher tests passed again
  after enforcing shell `exec` so Slurm signals reach the Python supervisor.
- Ten recovery tests passed: actual process-group SIGKILL during candidate
  execution in each mode and fidelity stage (four cases), signal-driven
  checkpoint/supervisor exit, reserved/submitted generation replay (two cases),
  live-job refusal, queue-time/failure accounting, and duplicate resume submission.
- The six existing integration cases also passed with recovery enabled (65.95s).
- Recovery retains frozen generated code, decision identity, saved search state
  and stage; it charges interrupted search/candidate time and avoids duplicate
  BO observations. Incomplete training restarts; model weights are not saved.
- These tests use synthetic model replies and simulated Slurm accounting;
  CPU training and process interruption are real. They do not certify live
  scheduler behavior, CUDA execution or provider access. Both CPU preparation
  and the short GPU preflight include the recovery suite.

## Earlier package checks — 2026-10-21 (dependency limitation superseded)

- Five offline campaign tests passed via `python3 tests/test_multifidelity_campaign.py`.
- Two existing subset/provenance tests passed by direct invocation with temporary
  directories: frozen uniform half-view/full validation and full-node filtering.
  No dependency stubs were used.
- Changed Python files passed AST syntax checks; `git diff --check` passed.
- The full ML pytest suite could not collect in the available local environment:
  `shutup` was missing. A collection attempt without conftest also found missing
  `backoff`. Dependency installation failed because PyPI DNS was unavailable.
- Full regression tests, including the AQ05 reset forwarding regression, are
  mandatory in the cluster preparation job, before provider calls or searches.
- No real LLM search, CUDA training, dataset download, Slurm submission or
  official grading was run locally. Actual Mila execution remains unverified.

Packaging fixes include forwarding reset's full/full filter through AQ05's
inherited state class, preserving three workers after the fidelity switch,
initializing direct fixed-evaluator instances safely, and accepting long CSV
text cells in the uniform subsampler. These implement the approved behavior.
