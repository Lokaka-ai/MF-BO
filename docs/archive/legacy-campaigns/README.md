# BO Evolve — ML-E Bench Lite, redesigned evolution (2026-10-21)

This independent source repository packages Xixian's MLEvolve-based Bayesian
optimization engine from `ZeroKnighting/EvolveFold` `xixian/dev` at `ced7bb5a54e91e3ea6236b871f9fcd27027fb493`,
including his AQ04 selector and fixed ML evaluation. The sole added search
operation is our evidence-guided redesigned evolution. An optional two-stage
training view uses a fixed random half of T before switching to full T.

The retained workflow includes draft, improve, debug, review, fusion,
aggregation, memory and submission. MLEVHarness determines the legal operation
for each parent before AQ04 BO selects among the prepared parent options. The
selected `audited_bo=redesign` route uses the plateau trigger and produces a
structural replacement plan from completed host-scored attempts. It generates
a complete new program rather than patching the parent. ML-E Bench Lite uses
`agents/mle_redesign_agent.py`; shared planning helpers remain in
`agents/redesign_agent.py`. The original upstream README is preserved in
`docs/UPSTREAM_README.md`.

The AQ04 selection rule remains the upstream rule. The optional reset mode
filters its GP training rows to full/full transitions after the switch. The
redesign flag changes evolution and its stagnation trigger; it does not replace
AQ04's parent-selection policy. The inherited AlphaEvolve
package is not the target of this repository's testing; its upstream default
is unchanged.

## ML-E Bench Lite use

Prepare the pinned environment and task data using
`tasks/mlebench_lite/README.md`. Create and inspect a fixed evaluation
protocol separately before a long search:

```bash
bash tasks/mlebench_lite/setup.sh
source tasks/mlebench_lite/env.sh
python tasks/mlebench_lite/prepare_fixed_protocol.py \
  --task leaf-classification --data-root "$MLEBENCH_DATA" \
  --output-root "$PWD/runs/protocol_setup" --seed 42
```

The protocol preparation may call the configured LLM, but it does not launch
BO search or candidate training. Review its T/V split, metric, targets and
asset paths for the task. Then inspect the intended AQ04 + redesign command
without running it:

```bash
python tasks/mlebench_lite/run_bo.py \
  --task leaf-classification --data-root "$MLEBENCH_DATA" \
  --output-root "$PWD/runs/bo_redesign" \
  --protocol-dir "$PROTOCOL_DIR" --check-only
```

When a run is authorized, remove `--check-only`. The launcher verifies that
the protocol matches the prepared task and seed. It defaults to 12 hours and
500 steps; record any overrides alongside the task, model, worker count and
seed. The configured model/provider is Muse Spark 1.3 through OpenCode.
Only full host-scored V results enter AQ04 BO and choose the final submission.
The private test score is obtained after search and is never BO feedback.

For a two-stage run, add `--multifidelity-mode carry_over` or
`--multifidelity-mode reset` to the same launcher command. Both modes use one
seeded, uniform sample of floor(50% of T), with unchanged full V/test. At 40%
of the total time budget, the scheduler drains submitted cheap work and then
switches the training view to full T. Carry over keeps the AQ04 fit. Reset
refits AQ04 only on full/full parent-child transitions; cheap nodes remain in
the search population. Only full-trained nodes can win. If no valid full node
finishes within the budget, finalization fails rather than selecting a cheap
node. Use separate output directories for the two modes and baseline.

## Evidence and execution boundary

This package has not established a performance advantage on ML-E Bench Lite.
No benchmark run was launched when creating it. A valid comparison requires
matched task, model, protocol, time budget, resources and independent runs.
An interrupted or infrastructure-failed run is incomplete evidence.

The fixed evaluator's candidate programs currently execute as subprocesses
under the same Unix user. The input view is a cooperative contract, not a
filesystem security boundary. Before claiming a leakage-free benchmark result,
use cluster-verified filesystem isolation or independently audit each executed
program and its file access. Inspect the prepared protocol itself; structural
validation cannot establish that every task's target and metric are correct.
