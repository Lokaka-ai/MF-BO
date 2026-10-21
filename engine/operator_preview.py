"""Preview and execute the repository's existing next-operator policy."""

from __future__ import annotations

import logging
import math
import random
import threading
import time

from agents import (
    aggregation_agent,
    debug_agent,
    draft_agent,
    evolution_agent,
    fusion_agent,
    improve_agent,
)
from engine import node_selection
from engine.conditions import is_branch_stagnant
from engine.parent_selectors.base import NORMAL_OPERATORS, OperatorPreview


logger = logging.getLogger("MLEvolve")
_RANDOM_PREVIEW_LOCK = threading.RLock()


def _peek_random() -> float:
    """Observe the next module-level RNG value without advancing its state."""
    with _RANDOM_PREVIEW_LOCK:
        state = random.getstate()
        try:
            return random.random()
        finally:
            random.setstate(state)


def _normal_preview(agent, parent_node, operator: str, **kwargs) -> OperatorPreview:
    if operator == "evolution" and not evolution_agent.can_run_evolution(parent_node):
        return OperatorPreview(operator="improve", reason="evolution_fallback_no_trajectory", **kwargs)

    if operator == "fusion":
        references = fusion_agent.get_fusion_candidates(agent, parent_node)
        if not references:
            return OperatorPreview(operator="improve", reason="fusion_fallback_no_references", **kwargs)
        references = references[:5]
        return OperatorPreview(
            operator="fusion",
            reference_node_ids=tuple(node.id for node in references),
            reason="branch_stagnation",
            **kwargs,
        )

    return OperatorPreview(operator=operator, reason="branch_stagnation" if operator == "evolution" else "normal", **kwargs)


def preview_next_operator(
    agent,
    parent_node,
    *,
    random_value: float | None = None,
    topk_triggered: bool | None = None,
) -> OperatorPreview:
    """Return the operator the current MLEvolve rules would execute.

    The function does not mutate the node, counters, journal, or RNG. The
    selected preview is later dispatched directly, so prediction and execution
    share one decision.
    """
    if agent.is_root(parent_node):
        if parent_node.reached_child_limit(scfg=agent.scfg):
            return OperatorPreview(
                operator="aggregation",
                bypass_selector=True,
                reason="root_draft_limit",
            )
        return OperatorPreview(operator="draft", bypass_selector=True, reason="initial_draft")

    if parent_node.is_buggy or parent_node.is_valid is False:
        return OperatorPreview(operator="debug", bypass_selector=True, reason="failed_parent")

    if parent_node.is_buggy is not False:
        return OperatorPreview(operator="no_op", bypass_selector=True, reason="unknown_parent_state")

    can_use_fusion = False
    if agent.search_start_time:
        elapsed_time = time.time() - agent.search_start_time
        can_use_fusion = elapsed_time >= agent.acfg.time_limit / 2

    is_from_topk = (
        bool(getattr(parent_node, "_topk_triggered", False))
        if topk_triggered is None
        else bool(topk_triggered)
    )
    threshold = (
        agent.scfg.topk_stagnation_threshold
        if is_from_topk
        else agent.scfg.branch_stagnation_threshold
    )
    if not is_branch_stagnant(agent, parent_node.branch_id, threshold=threshold):
        return OperatorPreview(
            operator="improve", reason="normal", topk_triggered=is_from_topk
        )

    evolution_enabled = bool(getattr(agent.acfg, "use_evolution", True))
    fusion_enabled = bool(getattr(agent.acfg, "use_fusion", True)) and can_use_fusion

    if evolution_enabled and fusion_enabled:
        value = _peek_random() if random_value is None else float(random_value)
        operator = "fusion" if value < agent.acfg.fusion_vs_evolution_prob else "evolution"
        return _normal_preview(
            agent,
            parent_node,
            operator,
            consumes_random_draw=True,
            random_value=value,
            topk_triggered=is_from_topk,
        )
    if fusion_enabled:
        return _normal_preview(
            agent, parent_node, "fusion", topk_triggered=is_from_topk
        )
    if evolution_enabled:
        return _normal_preview(
            agent, parent_node, "evolution", topk_triggered=is_from_topk
        )
    return OperatorPreview(
        operator="improve",
        reason="stagnation_fallback",
        topk_triggered=is_from_topk,
    )


def commit_preview_random(preview: OperatorPreview) -> None:
    """Consume the one RNG draw the original operator branch would consume."""
    if not preview.consumes_random_draw:
        return
    with _RANDOM_PREVIEW_LOCK:
        consumed = random.random()
    if preview.random_value is not None and not math.isclose(consumed, preview.random_value):
        logger.debug(
            "Operator RNG advanced between preview and commit (preview=%s, commit=%s); "
            "executing the already selected operator.",
            preview.random_value,
            consumed,
        )


def execute_operator_preview(agent, parent_node, preview: OperatorPreview, init_solution_path=None):
    """Dispatch the preview through the unchanged Agent implementation."""
    operator = preview.operator
    if preview.topk_triggered:
        parent_node._topk_triggered = True
    if operator == "draft":
        return draft_agent.run(agent, init_solution_path=init_solution_path)
    if operator == "aggregation":
        return aggregation_agent.run(agent, mode="node", parent_node=parent_node)
    if operator == "debug":
        return debug_agent.run(agent, parent_node)
    if operator == "no_op":
        logger.warning("Node %s has unresolved execution state; preserving the original no-op path.", parent_node.id)
        return None
    if operator == "improve":
        return improve_agent.run(agent, parent_node)
    if operator == "evolution":
        return evolution_agent.run(agent, parent_node)
    if operator == "fusion":
        node_by_id = {node.id: node for node in agent.journal.nodes}
        references = [
            node_by_id[node_id]
            for node_id in preview.reference_node_ids
            if node_id in node_by_id
        ]
        if len(references) != len(preview.reference_node_ids):
            raise RuntimeError("previewed Fusion reference disappeared before execution")
        return fusion_agent.run(agent, parent_node, reference_nodes=references)
    raise ValueError(f"Unsupported operator preview: {operator}")


def assert_preview_matches_node(preview: OperatorPreview, result_node) -> None:
    actual = "aggregation" if result_node.stage == "fusion_draft" else result_node.stage
    if preview.operator != actual:
        raise AssertionError(
            f"operator preview/execution mismatch: preview={preview.operator}, actual={actual}"
        )


def _topk_expansion_ids(agent) -> set[str]:
    if not getattr(agent, "search_start_time", None):
        return set()
    elapsed = max(time.time() - agent.search_start_time, 0.0)
    total = max(float(agent.acfg.time_limit), 1.0)
    progress = elapsed / total
    if progress < agent.scfg.explore_switch_start:
        return set()
    if progress < agent.scfg.explore_switch_end:
        k = agent.scfg.topk_early_k
        per_branch = agent.scfg.topk_early_max_per_branch
    else:
        k = agent.scfg.topk_late_k
        per_branch = agent.scfg.topk_late_max_per_branch
    return {
        item["node"].id
        for item in node_selection.get_top_k_nodes_global(agent, k, per_branch)
    }


def eligible_normal_parents(agent) -> list:
    """Return the common legal parent set offered to all normal selectors."""
    topk_ids = _topk_expansion_ids(agent)
    parents = []
    for node in agent.journal.nodes:
        if agent.is_root(node):
            continue
        if node.is_terminal or node.lock:
            continue
        if node.is_buggy is not False or node.is_valid is False:
            continue
        metric = getattr(node, "metric", None)
        if metric is None or metric.value is None:
            continue
        regular_limit = node.reached_child_limit(agent.scfg)
        topk_extension = (
            node.id in topk_ids
            and not node.reached_child_limit(agent.scfg, for_topk=True)
        )
        if regular_limit and not topk_extension:
            continue
        parents.append(node)
    return parents


def requires_topk_extension(agent, parent_node) -> bool:
    return parent_node.reached_child_limit(agent.scfg)


def is_normal_operator(preview: OperatorPreview) -> bool:
    return preview.operator in NORMAL_OPERATORS and not preview.bypass_selector
