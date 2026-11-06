# Optional BNVS root draft

The only new setting is `parent_selector.include_root_draft`, defaulting to
`false` (including configurations where the field is absent). Disabled runs
keep exact10, the original root bypass, sampling, observation/scale filters,
initialization and continuation behavior. This does not switch BNVS to MCGS.

When enabled, `bnvs_ts` and `bnvs_mean_greedy` with `incumbent_h1`/H=1 rank
existing legal expansions and at most one `draft(root)` in the same posterior.
The existing `value_model.feature_set: exact10` configuration is automatically
interpreted as the versioned `bnvs-h1-exact12-root-draft-v1` schema. Other selector,
target or base-schema combinations are rejected at startup. `root_only` adds
root only to global ranking; `all_normal_steps` also adds it to rankable
continuations. Existing operator previews and draft generation are reused.

Initialization still generates `min(initial_drafts, steps)` drafts sequentially
and executes them through the existing pipeline. All initial attempts must be
terminal before an adaptive draft is eligible; final failures count too.
`agent.search.num_drafts` limits **all submitted direct draft attempts**, including
initialization. For `initial_drafts=3, num_drafts=5`, at most two subsequent drafts
can be chosen, and neither is mandatory. Other chains may run while initialization
is pending. One pending adaptive draft is the fixed concurrency limit.

Exact12 drops the common remaining-budget column from exact10, keeps the other
nine columns in order, and appends `draft`, `draft * remaining_budget_fraction`,
and `draft * log1p(global_stagnation)`. Normal options have zero in these three
columns. Root uses completed direct draft attempts: median valid utility as
source quality, completed attempt count, and valid rate. It uses no descendant
rewards or artificial root metric. Missing source quality remains `null`, with
`source_quality_missing=1` and a neutral gap of zero. Global stagnation counts
completed experiments since an outcome exceeded the previous best utility by
the existing `metric_improvement_threshold`; pending work does not increment it.

All trainable actions use `clip((child - incumbent_at_decision) / scale, -1, 1)`.
Missing incumbents are explicitly `target_unavailable`, excluded from H1 fitting.
Valid initial/draft/normal results still contribute to the unchanged robust
scale formula. Submitted final failures contribute -1 when their target is
available, including failures without a child ID. Normalization remains at fit
time, using saved raw decision features. **H1 measures one-step value relative to
the incumbent; it does not estimate the long-term value of a new branch.**

The selection lock also protects reservation. Submission charges the attempt
once; cancellation before submission releases only the reservation. Generation,
execution, parsing and validation share one terminal record per decision.
Selecting root from a continuation closes/backpropagates that chain once, then
uses the usual draft lifecycle and existing step/time budget.

`phase1/root_draft_state.json` is an atomic decision ledger, also embedded in
`journal.parent_selector_state` at checkpoint. It stores the mode/schema,
dimensions/target/prior, initialization and attempt counters, pending frozen
decisions and generated code, posterior, RNG state/decision counter, completed
observations and journal snapshot. Resume reconciles terminal JSONL records
before scheduling. Generated but unexecuted children resume with their original
decision; interrupted submitted work without a terminal outcome becomes one
failure, while unsubmitted reservations are released. A live original process
or pending work owned by another host must be reconciled before resume; it is
never silently launched a second time. Cross-mode and incompatible checkpoints
are rejected; no automatic migration is provided. Old disabled checkpoints
remain supported.

Decision/snapshot logs identify `initialization`, `model`, `fallback`, `forced`
and `continue`, `switch`, `new_direction`, plus root eligibility, draft cap,
decision index and label availability. Forced feasibility drafts (when no
incumbent exists), aggregation and debug are recorded as forced operations.

From `MLEvolve_BO`, these two example launches differ only in the switch:

```bash
python run.py data_dir="$DATA_DIR" desc_file="$DESC_FILE" agent.initial_drafts=3 agent.search.num_drafts=5 parent_selector.include_root_draft=false
python run.py data_dir="$DATA_DIR" desc_file="$DESC_FILE" agent.initial_drafts=3 agent.search.num_drafts=5 parent_selector.include_root_draft=true
```

`tests/fixtures/bnvs_d753bc7.json` was captured before implementation with fixed
history, clock and RNG. `test_bnvs_legacy_contract.py` compares disabled and absent
switches against that oracle. `test_root_draft_runtime.py` exercises production
selection, agents, dispatch, parsing, validation, transitions and fitting.
LLM transport, grading HTTP and the external interpreter are substituted;
clock/ID controls and filesystem/process fault injection exercise recovery.

The relevant regression suite passes 108 tests. The unrelated memory suite
cannot collect without `faiss`; the historical external-repository admission
test is excluded because its recorded path/checkout does not match this
workspace. No full MLEBench run was performed.
