"""Preserve native search bookkeeping at study7's atomic group boundaries."""
import copy

from .study7_budget import registered

SCALARS = ("current_step", "best_metric", "next_branch_id", "fusion_draft_count",
           "post_process_triggered", "post_process_attempts", "improve_attempts_count",
           "last_successful_improve_step", "best_metric_history")
NODE_LISTS = ("current_node_list", "top_candidates")


def capture(agent):
    from engine.competition_checkpoint import legacy_enabled
    if not registered(agent) and not legacy_enabled(getattr(agent, "cfg", None)):
        return
    admitted = {node.id for node in agent.journal.nodes}
    agent.journal.study7_search_state = {
        "version": 1, "method_id": agent.cfg.study7.method_id,
        "values": {key: copy.deepcopy(getattr(agent, key)) for key in SCALARS},
        "node_lists": {key: [node.id for node in getattr(agent, key) if node.id in admitted] for key in NODE_LISTS},
        "best_node_id": agent.best_node.id if agent.best_node else None,
        "branch_node_count": dict(agent.branch_node_count),
        "node_flags": {node.id: {"lock": node.lock, "expected_child_count": node.expected_child_count,
                                  "_topk_triggered": getattr(node, "_topk_triggered", False)}
                       for node in agent.journal.nodes}}


def restore(agent):
    from engine.competition_checkpoint import legacy_enabled
    saved = getattr(agent.journal, "study7_search_state", None)
    if (not registered(agent) and not legacy_enabled(getattr(agent, "cfg", None))) or saved is None:
        return
    if saved["version"] != 1 or saved["method_id"] != agent.cfg.study7.method_id:
        raise ValueError("Incompatible study7 search checkpoint")
    nodes = {node.id: node for node in agent.journal.nodes}
    for key in SCALARS:
        setattr(agent, key, copy.deepcopy(saved["values"][key]))
    for key in NODE_LISTS:
        setattr(agent, key, [nodes[node_id] for node_id in saved["node_lists"][key]])
    agent.best_node = nodes.get(saved["best_node_id"])
    agent.branch_node_count = {int(key): value for key, value in saved["branch_node_count"].items()}
    for node_id, flags in saved["node_flags"].items():
        for key, value in flags.items():
            setattr(nodes[node_id], key, value)
