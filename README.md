# BO Evolve — redesigned evolution only (2026-10-21)

This independent source repository packages Xixian's MLEvolve-based AQ04 Bayesian
optimization engine and its existing agent workflow, with our evidence-guided
redesigned evolution operation selected. Candidate evaluation uses the full
training set provided by the prepared task protocol. The fidelity scheduler,
its configuration, calibration tools and prompts are absent.

The retained workflow includes draft, improve, debug, review, fusion,
aggregation, memory and submission. MLEVHarness determines the legal operation
for each parent before AQ04 BO selects among the prepared parent options. The
selected `audited_bo=redesign` route uses the plateau trigger and produces a
structural replacement plan from completed host-scored attempts. It generates
a complete new program rather than patching the parent. AlphaEvolve uses
`agents/redesign_agent.py`; fixed-evaluation ML tasks use
`agents/mle_redesign_agent.py`. The original upstream README is preserved in
`docs/UPSTREAM_README.md`.

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
