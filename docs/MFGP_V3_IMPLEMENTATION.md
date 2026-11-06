# V3 implementation and validation

2026-11-05: the approved v3 is integrated into the native search path and locally
validated. This is implementation evidence, not Mila readiness or a performance
result. No Slurm jobs, provider calls to the production model, or experiments
were launched during this implementation.

## Implemented path

`mila_mfgp20_v3.py` selects `config/mlebench_mfgp20_v3.json`. The shared launcher
passes policy 3 and its measured task cost profile; `engine/mf_gp/runner.py`
dispatches to `v3/runner.py`. The v1/v2 manifests and launch defaults are preserved.

- Native Harness legality and frozen operator previews still govern generation.
  Generated/reviewed source is saved before probing. Generation fidelity is
  advisory; a separate allocation commits an actual 20% or 100% execution.
- Compile and bounded subprocess probes use at most 64 T rows and their T assets,
  with no V/test tables. The contract requests one warm-up, two timed training
  steps, and one prediction batch on the intended path. The 120-second cap
  includes loading. Unsupported/time-out probes use the conservative prior;
  syntax/runtime failures enter the native failed-node/debug path without scores.
- Generation, probes and executions share at most three worker slots. Only
  committed executions condition the GP as pending measurements. Prepared code
  has immutable IDs/hashes, a capacity of three, durable deferred states and
  terminal expiration with its native reservation retained for audit.
- Ready executions exclude sunk generation/probe cost from acquisition
  denominators. Process timing and host work are separate for expansions and
  promotions. Completed runtimes inform cost estimates; crashes do not become
  fast successful samples. Same-code full timeouts bound completion guards.
  Existing execution segments preserve interruption expenses and cumulative caps.
- All admitted raw utilities use a common mean/std transformation. Observed,
  proposed and pending feature vectors are rebuilt from immutable birth context.
  The model/controller contract rejects legacy checkpoint migration.
- One affordable full anchor is prioritized as soon as an initial cheap success
  exists. Its pending execution does not block other workers. Further promotions
  compete normally; failed full attempts are not silently repeated.
- Feasibility is recomputed per action, reserving a same-program full path for
  cheap work and outstanding execution work on the shared GPU. The initial
  cheap deadline remains 40% of active time. There is no permanent terminal latch.
- Live acquisition is KG, with terminal full-artifact improvement. Decisions save
  raw acquisition, incremental costs, feasibility, pending support and the exact
  GP/observation snapshot. `tools/replay_mfgp_v3.py` computes the marginal-truncation
  MF-MES challenger offline on that same pool. It never invents unobserved rewards.
- Recovery preserves reserved generation, ready/deferred code, committed fidelity,
  and failed probe receipts. It requires the native executor-release checks.
  Infrastructure failure leaves unexecuted state recoverable rather than expiring
  it as a normal end of search. Only completed valid full artifacts can win.

## Local acceptance evidence

The v3 tests cover real subprocess success/error/timeout; common normalization,
posterior restoration and checkpoint rejection; source/state corruption; native
fixed-evaluator execution and immutable promotion; ready/deferred/executing
recovery; generation recovery without double charging; a persisted failed probe;
nonblocking anchor behavior; the three-worker runner and action cap; offline
MF-MES reconstruction; and an advisory cheap choice changed to full after actual
generation with unchanged bytes/identity.

Run with `/private/tmp/mf-policy-venv/bin/python -m pytest -q` and the six
`tests/test_mfgp_v3_*.py` files (31 tests). Legacy focused regression suite:
`test_mf_gp.py`, `test_mfgp_policy_v2.py`, `test_mfgp_audit.py`,
`test_mfgp_fraction.py`, `test_mfgp_campaign.py`, `test_mf_recovery.py` (61 tests).
The native integration tests use synthetic LLM responses, a local setup transport,
real CPU training processes, and the actual fixed scorer. They are not real-task
GPU validation. Campaign preparation/preflight now includes the v3 checks when
policy 3 is selected. CLI help and `git diff --check` are also checked.

## Remaining external readiness gate

Before any performance experiment, use a separately frozen bundle and campaign
with the existing measured same-code cost profiles and exact protocol identity.
Verify data, provider, CUDA and Slurm on Mila, then run a small separately authorized
real-task lifecycle: production generation → bounded probe → cheap evaluation →
unchanged full promotion → fixed validation → released-allocation resume → final
submission. The existing generic GPU preflight plus synthetic tests does not by
itself establish that complete real-task gate. Budget/task choice is a separate
compute decision under the approved proposal; no full campaign is authorized here.

Candidate-declared workload/step timing is explicitly labeled unverified. It can
increase estimates but cannot optimistically replace comparable measured/prior
costs. The 1.5 multiplier and outstanding-work reservation are heuristics, not
probabilistic guarantees. A T-only view is a cooperative contract, not a filesystem
sandbox. Probe success cannot rule out later OOM or prove identical execution
paths. No calibration, causal performance, or generalization claim is made.

## Superseded progress snapshots

The following records describe earlier incomplete stages. Their statements that
native integration is missing are superseded by the implementation above.

### Earlier implementation status

2026-11-05: user approved the v3 design. No further design approval is required
for that scope. Implementation is in progress, not cluster-ready.

Completed foundations (not yet wired to the search runner):
- engine/mf_gp/v3/cost.py: separate process/host/generation/probe costs,
  failure/censoring semantics, workload estimates, marginal action costs and
  individual-action completion guards.
- engine/mf_gp/v3/context.py: common normalization and immutable raw birth
  context, with consistent feature reconstruction after JSON restoration.
- tests/test_mfgp_v3_foundation.py: five lightweight stdlib regression tests pass.
  Command: python3 -m unittest discover -s tests -p test_mfgp_v3_foundation.py -v.
- git diff --check passed. Existing runner, models, manifests and prompts unchanged.

Remaining authorized implementation:
1. Integrate common process/host timing on every execution path and update GP
   normalization with the new raw-context schema and versioned checkpoints.
2. Compile/probe contract and bounded runtime checks on actual generated code.
3. Durable generation/ready/execution reservations and post-generation fidelity
   selection, bounded queue, deferred/expired states and exact recovery.
4. Nonblocking full anchor and individual budget feasibility in actual scheduler.
5. Offline MF-MES challenger, numerical checks and trajectory diagnostics.
6. End-to-end synthetic execution/recovery verification and documented limits.

Do not expose a v3 launcher or claim v3 scientific behavior until the components
are connected and tested. No new experiments are authorized by implementation.

## Component implementation checkpoint

The isolated v3 package now also contains:
- model.py: actual discrepancy GP subclass using all-admitted normalization,
  immutable raw-context transformation and a distinct checkpoint contract.
- prepared.py: generation/ready/execution/deferred/expired state transitions,
  bounded prepared capacity, code hashes and JSON recovery validation. Caller
  must still integrate this with the native journal lock/persistence boundary.
- probe.py: compile gate and real bounded subprocess probe with T-only input
  contract, process-group termination, file-backed logs and preserved receipts.
  Candidate throughput is labeled candidate-reported, not independently verified.
- mes.py: offline finite-pool MF-MES challenger using a documented marginal
  truncation approximation, noisy observations and seeded maximum samples.
  It is not the live acquisition and has no demonstrated performance advantage.

Validation: all 21 tests in tests/test_mfgp_v3_{foundation,model,prepared,probe,mes}.py
passed together in 1.58 seconds using /private/tmp/mf-policy-venv/bin/python.
These include real subprocess success/error/timeout, GP posterior recovery,
corrupted queue checkpoint rejection, sunk-cost semantics and acquisition
independence/noise/sign/cost checks. git diff --check passed.

IMPORTANT: native runner/controller integration is NOT complete. None of these
components is exposed as a runnable campaign. Remaining work includes actual
cost-field plumbing, generation/probe contract injection, selection after code
creation, pending-state wiring, nonblocking calibration, native recovery and
end-to-end synthetic validation. Do not mistake component tests for a complete
implementation. Existing v1/v2 runtime paths and frozen campaigns remain unchanged.
