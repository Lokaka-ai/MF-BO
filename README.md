# Multi Fidelity BO

Active research: budget-aware Bayesian optimization for MLE-bench Lite, using
20% training data for low fidelity and 100% for target fidelity. The fixed outer
validation set is unchanged. This is the maintained successor to the historical
50% pilots, not a claim of equivalence to Xixian's latest upstream branch.

The policy-v2 three-task study is complete in the saved 2026-11-04 evidence.
Current work is the [v3 proposal](docs/MFGP_V3_PROPOSAL.md), grounded in those
trajectories. V3 is integrated and locally validated, with Mila validation still outstanding. See
[implementation evidence](docs/MFGP_V3_IMPLEMENTATION.md) and
[PROJECT.md](PROJECT.md) for evidence locations and interpretation limits.

## Start here

1. [Project scope and scientific contract](PROJECT.md)
2. [Maintainer instructions](AGENTS.md)
3. [Next design proposal](docs/MFGP_V3_PROPOSAL.md) and [v2 reproduction specification](docs/MFGP20.md)
4. [Code map and debugging guide](docs/DEVELOPMENT.md)
5. [Documentation index and historical evidence](docs/README.md)

## Authoritative entry points

| Purpose | Entry point |
|---|---|
| Preserved v2 20% MF-GP launcher | `tasks/mlebench_lite/mila_mfgp20.py` |
| Isolated v3 launcher/settings | `tasks/mlebench_lite/mila_mfgp20_v3.py`, `config/mlebench_mfgp20_v3.json` |
| Preserved v2 task/resources/model settings | `config/mlebench_mfgp20.json` |
| Shared prepare/preflight/submit/resume/grade logic | `tasks/mlebench_lite/mila_multifidelity.py` |
| MF-GP search loop | `engine/mf_gp/runner.py` |
| Surrogate and acquisition | `engine/mf_gp/model.py`, `engine/mf_gp/controller.py` |
| Frozen evaluator and data views | `engine/fixed_evaluation.py`, `engine/multifidelity.py` |
| Measured cost export | `tools/mfgp_cost_matrix.py` |

Inspect CLI options without launching anything:

```bash
python3 tasks/mlebench_lite/mila_mfgp20.py --help
```

The completed 20% campaign used MF-GP only. Legacy `mila_mfgp.py` and
`mila_multifidelity.py` launchers retain their 50% configurations. Do not infer
the active campaign from global defaults or invoke a legacy launcher by accident.
Preparation, GPU checks, task-boundary checks, and explicit run authorization
precede full searches. A successful synthetic test suite is not an experiment.

## Repository boundaries

Keep runtime code, prompts, fixtures, pinned dependencies, and compatibility
launchers together: inherited modules may remain on the active call path.
Historical runbooks are references, not instructions to launch old campaigns.
Credentials, environments, datasets, logs, and experiment outputs are excluded
from source bundles. Frozen cluster bundles must never be patched in place by
local housekeeping changes.

Fixed evaluation uses a cooperative data contract; candidate filesystem isolation
is not enforced. Review executed programs before making leakage-free claims.
A matched full-only baseline and stochastic replicates are needed for causal
performance claims; historical scores alone do not establish superiority.
