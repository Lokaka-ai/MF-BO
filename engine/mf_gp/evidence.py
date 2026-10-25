"""Comparable selection utilities; measured scores never change in the journal."""

import copy
from utils.metric import MetricValue


def quality(agent, node):
    selector = agent.phase1_runtime.study7_selector
    own = [
        o
        for o in selector.observations
        if o["program_id"] == node.id and o["valid"] and o.get("budget_eligible", True)
    ]
    full = next((o for o in own if o["fidelity"] == 1.0), None)
    if full:
        return full["utility"]
    if own:
        row = {k: own[0][k] for k in ("program_id", "x", "operator")}
        return float(selector.gp.posterior([{**row, "fidelity": 1.0}])[0][0])
    return float("-inf")


def update_best(agent):
    selector = agent.phase1_runtime.study7_selector
    valid = [
        o
        for o in selector.observations
        if o["valid"] and o["fidelity"] == 1.0 and o.get("budget_eligible", True)
    ]
    if not valid:
        agent.best_node = None
        agent.best_metric = None
        return
    best = max(valid, key=lambda o: o["utility"])
    original = next(n for n in agent.journal.nodes if n.id == best["program_id"])
    node = copy.copy(original)
    node.metric = MetricValue(best["result"]["metric"], maximize=agent.metric_maximize)
    node.evaluation_result = best["result"]
    node.evaluation_attempt_id = best["result"]["attempt_id"]
    node.training_fraction = 1.0
    agent.best_node = node
    agent.best_metric = node.metric.value
