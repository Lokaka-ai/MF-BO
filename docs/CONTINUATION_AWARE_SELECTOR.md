# Continuation-aware parent selection

`parent_selector.decision_scope` controls where the configured selector runs:

- `root_only` keeps the Phase 1 behavior and is the default.
- `all_normal_steps` also ranks a valid direct-continuation node against every
  other legal normal parent.

Draft creation, Debug, Aggregation, terminal nodes, unknown execution states,
and Bayesian cold start keep their original forced paths. When another parent
wins, the runtime closes the current chain once with the reward already stored
in `transitions.jsonl`; `continuation_closed_transition_id` prevents duplicate
backpropagation. Every continuation opportunity is written to
`phase1/continuation_events.jsonl`.

The four-arm Leaf experiment is:

| Arm | Scope | Selector |
| --- | --- | --- |
| A `root_progressive` | root only | Progressive MCGS |
| B `all_steps_progressive` | all normal steps | Progressive MCGS |
| C `all_steps_bayesian_child_ucb` | all normal steps | Bayesian Child-UCB |
| D `all_steps_cost_aware_bayesian` | all normal steps | cost-aware Bayesian EI |

All arms use 30 steps, three initial drafts, seed 42, one GPU, and the same
online cold-start thresholds (`3/3/1`). Use
`experiments/continuation_scope_ablation/launch.sh` on a four-GPU node and
`summarize.py` to report model-backed decisions, continue/switch behavior,
backpropagation integrity, gain, diversity, best metric, anytime AUC, and cost.
