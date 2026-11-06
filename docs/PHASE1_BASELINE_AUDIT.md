# Phase 1 Baseline Audit

Audit date: 2026-08-27

## Repository boundary

- Read-only original: `/gpfs/projects/AI4D/core-132/xixian/EvolveFold_markdown_protocol/MLEvolve`
- Independent working copy: `/gpfs/projects/AI4D/core-132/xixian/EvolveFold_markdown_protocol/MLEvolve_BO_Phase1`
- Original branch at admission: `main`
- Original commit at admission: `7d8403c899c40f01941c0429f1c4ef51e82ae41c`
- Working-copy branch: `phase1-bo-parent-selector`
- Original status at admission:

  ```text
   M llm/__init__.py
  ?? llm/codex.py
  ```

The pre-existing LLM changes above were copied intact. They are not Phase 1
changes. `ORIGINAL_REPO_BASELINE.json` is the machine-readable protection
record used by the final read-only status check.

## Files inspected

The audit followed the current code rather than the paper description. It
covered `run.py`, all modules in `engine/` named by the Phase 1 goal,
`agents/triggers.py`, the Draft, Improve, Evolution, Fusion, Aggregation and
Debug agents, result parsing, code review, metric handling, configuration, and
both launch scripts.

## Current control flow

1. **Real parent-selection call site.** `AgentSearch.step()` in
   `engine/agent_search.py` calls
   `engine.node_selection.select_with_soft_switch(self)` only when its incoming
   node is absent or is the virtual root. `select_with_soft_switch()` samples
   between recursive UCT traversal and inverse-rank Top-K exploitation.

2. **When selection restarts from root.** `run.py` passes each completed node
   back into the next `step()` call. `AgentSearch.step()` returns the completed
   node while the branch should continue and returns the virtual root after a
   backpropagation/terminal decision. A new global parent selection therefore
   happens after a root return, on an empty worker refill, or when a caller
   explicitly passes root.

3. **Direct continuation after a completed node.** Yes. A successful child is
   normally expanded directly on the next worker step. This continuation does
   not invoke the global parent selector. Phase 1 must retain this behavior and
   record it as a singleton, forced selection-time observation rather than
   pretending that BO chose among the whole graph.

4. **Operator decision.** `_run_single_step()` chooses the operator after the
   parent is known. Root produces Draft or Aggregation. Buggy/invalid parents
   produce Debug. A valid parent produces Improve unless branch-stagnation,
   elapsed-time and feature flags activate Evolution or Fusion. The
   Fusion-versus-Evolution choice is randomized. Evolution can fall back to
   Improve when its trajectory is too short; Fusion can fall back to Improve
   when it has no references.

5. **Selectable nodes.** Recursive UCT follows main-tree children and returns
   a non-terminal node that still has regular expansion capacity. Root Draft
   children are temporarily locked while in flight. Top-K exploitation first
   ranks successful nodes with valid metrics and branch-diversity limits, uses
   the larger Top-K child limit, and otherwise descends from a sampled Top-K
   node with the same UCT routine. Reference edges are not part of selection or
   backpropagation.

6. **Debug and Aggregation bypass.** Debug is triggered from a selected or
   directly continued failed parent and is not an alternative operator chosen
   by the selector. Aggregation is triggered only at root after the regular
   Draft limit and the existing time, branch-success, global-stagnation,
   attempt and probability conditions. Initial Draft also bypasses normal
   parent selection. Phase 1 must log these paths but must not rank them with
   normal `(parent, operator)` options.

7. **Reward and backpropagation.** `engine/evaluation.py` computes the existing
   ordinal signal. A failed/metric-less node receives `-1`; a valid non-root
   child normally receives `1`; a successful Debug child receives `1.5`.
   There is code for an additional `1.5` global-best improvement bonus, but the
   current repository never updates `agent.best_metric`, so that branch is
   inactive. Backpropagation updates visits and total reward only along parent
   links and also maintains continuation, lock and failure-depth state. Phase 1
   will not change this behavior.

8. **Best metric update.** `solution_manager.update_best_solution()` runs after
   a completed node has been parsed, validated, evaluated and journaled. It
   maintains Top-K files and replaces `agent.best_node` using `MetricValue`
   direction-aware comparison. The BO feature extractor must read
   `best_node.metric`; it must not depend on the unused `best_metric` scalar.

9. **Cost measurement points.** Generation begins immediately before the
   original Agent call and ends after Code Review. Execution/scoring begins
   before `Interpreter.run()` and ends after result parsing, submission
   validation and `check_improvement()`. `ExecutionResult.exec_time` is the
   inner child-process runtime. Phase 1 can therefore record generation,
   execution/scoring and total wall time without changing any Agent prompt or
   executor behavior. Reliable GPU time is not exposed and must remain null.

10. **Minimum Phase 1 insertion points.** Add a local selector package and a
    side-effect-free operator preview; route only the existing root-selection
    call through a selector adapter; create singleton decisions for direct and
    special-path continuations; attach immutable decision metadata to the new
    node; append transition/cost observations after scoring; and add selector
    configuration. No controller rewrite is needed.

11. **Behavior kept unchanged.** Agent and planner prompts, generation modes,
    Code Review, executor, result parsing, submission validation, Debug,
    Memory, branch/reference graph semantics, stagnation rules, reward,
    backpropagation, parallelism, timeouts, step budgets and output selection
    remain on their existing paths. Fusion reference selection remains the
    original Fusion policy; Phase 1 may only preview and reuse its result.

## Baseline-specific constraints for implementation

- `legacy_original` must keep calling the original soft-switch selector.
- `progressive_mcgs` must be an adapter over that same function, not a new
  approximation.
- Normal BO modes may rank only valid, metric-bearing, non-terminal parents
  with regular expansion capacity, plus current original Top-K parents that
  have Top-K overflow capacity. Non-Top-K nodes cannot borrow that larger
  limit, and an overflow winner preserves the original `_topk_triggered`
  child/operator semantics.
- Draft, Debug, Aggregation and emergency/no-option fallback stay outside the
  BO acquisition.
- An unresolved `is_buggy=None` parent keeps the original warning/no-op path;
  Phase 1 must not silently turn it into Debug.
- A previewed Evolution or Fusion fallback must be represented by the operator
  that will actually execute.
- Continuous gain is separate from the existing ordinal reward.
- Selection features are written before child generation and never rebuilt
  from the final journal.
