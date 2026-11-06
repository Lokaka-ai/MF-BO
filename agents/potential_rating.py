"""Extend the EXISTING execution review only when explicitly enabled.

One generic scalar for both short- and long-horizon GP. No extra call, no plan,
no separate g1/gh, and no speculation written into the factual summary.
"""
from __future__ import annotations

import copy
import json
import logging
import math
import time
from datetime import datetime, timezone

from engine.parent_selectors.rating_prior import RatingPriorConfig, potential_level
from agents import study7_rating

logger = logging.getLogger("MLEvolve")

RATING_INSTRUCTIONS = """
Separately from the factual execution summary, return ONE generic forward-looking
potential_level for the completed node's CURRENT code, not for its parent or for
the experiment that just produced it. Use only the supplied code, observed public
training/validation diagnostics, task rules and available resources. This is a
coarse remaining research-potential assessment, NOT a metric, probability,
validity verdict or a prediction tied to one step or a fixed number of steps.
0 = no clear feasible remaining improvement path under the given constraints;
1 = plausible but mainly limited/local incremental improvement;
2 = concrete evidence of substantial, feasible remaining improvement potential.
Distinguish remaining potential from current score, code complexity and pride in
your own implementation. Consider competitiveness with the known best result:
an easy repair of a low-quality or invalid score is not by itself evidence of a
substantial scientific improvement. Do not assume extra data, hardware or runtime
outside the stated rules. Use null when execution is invalid or evidence is
insufficient. Never use private/test scores or unseen future experiments.
Keep summary limited to observed facts/errors and code_summary descriptive only.
Put the subjective judgment ONLY in potential_level; do not add a potential
rationale, additional ratings, recommendations or a future experiment plan.
""".strip()


def enabled(agent):
    return study7_rating.enabled(agent) or RatingPriorConfig.from_config(agent.cfg).enabled


def _metric(node):
    value = getattr(getattr(node, "metric", None), "value", None)
    try:
        return float(value) if value is not None and math.isfinite(float(value)) else None
    except (TypeError, ValueError):
        return None


def _context(agent, node):
    runtime = getattr(agent, "phase1_runtime", None)
    state = getattr(runtime, "root_draft_state", None) or {}
    acfg = agent.acfg
    step_limit = getattr(acfg, "steps", None)
    submitted = state.get("submitted_count")
    seconds = getattr(acfg, "time_limit", None)
    start = getattr(agent, "search_start_time", None)
    remaining = max(float(seconds) - max(time.time()-start, 0.0), 0.0) if seconds is not None and start else seconds
    maximize = getattr(agent, "metric_maximize", None)
    observed = [_metric(n) for n in getattr(getattr(agent, "journal", None), "nodes", [])
                if n.id != node.id and n.stage != "root" and n.is_buggy is False
                and getattr(n, "is_valid", None) is not False]
    observed = [v for v in observed if v is not None]
    incumbent = (max(observed) if maximize else min(observed)) if observed and maximize is not None else None
    return {
        "completed_node_id": node.id,
        "completed_node_stage": node.stage,
        "parent_metric_for_context_only": _metric(getattr(node, "parent", None)),
        "metric_maximize": maximize,
        "known_best_metric_before_this_result": incumbent,
        "remaining_run_attempts": max(int(step_limit)-int(submitted), 0) if step_limit is not None and submitted is not None else None,
        "remaining_run_seconds": remaining,
        "per_execution_timeout_seconds": getattr(getattr(agent.cfg, "exec", None), "timeout", None),
        "configured_cpu_count": getattr(agent.cfg, "cpu_number", None),
        "allowed_operations": ["improve", "debug"] +
            (["evolution"] if getattr(acfg, "use_evolution", True) else []) +
            (["fusion"] if getattr(acfg, "use_fusion", True) else []),
        "max_debug_depth": getattr(getattr(agent, "scfg", None), "max_debug_depth", None),
        "resource_note": "Only the stated resources are known; do not assume additional GPU capacity.",
    }


def extend_review(agent, node, prompt, spec):
    """Disabled path returns the original prompt/spec unchanged, not even copied."""
    if study7_rating.enabled(agent):
        return study7_rating.extend_review(agent, node, prompt, spec)
    if not enabled(agent):
        return prompt, spec
    prompt, spec = copy.deepcopy(prompt), copy.deepcopy(spec)
    prompt["Introduction"] += (
        '\nAdditional required field: "potential_level" (integer 0, 1, 2, or null).'
        ' This is separate from all factual result fields.'
    )
    prompt["Task and constraints for potential assessment"] = getattr(agent, "task_desc", "")
    prompt["Completed-node assessment context"] = json.dumps(_context(agent, node), ensure_ascii=False)
    prompt["Generic potential assessment"] = RATING_INSTRUCTIONS
    spec.json_schema["properties"]["potential_level"] = {
        "anyOf": [{"type": "integer", "enum": [0, 1, 2]}, {"type": "null"}],
        "description": "Generic remaining potential of this completed node only: 0 low, 1 limited, 2 substantial; null if unavailable. Not a metric or validity judgment.",
    }
    spec.json_schema["required"].append("potential_level")
    return prompt, spec


def reset_rating(agent, node):
    if enabled(agent):
        node.agent_potential_level = None
        node.agent_potential_assessed_at = None
        if study7_rating.enabled(agent):
            node.study7_rating_metadata = {"missing_reason": "not_reviewed", "usable": False}


def save_rating(agent, node, response):
    """Call after native result checks; malformed ratings never trigger retries."""
    if study7_rating.enabled(agent):
        return study7_rating.save_rating(agent, node, response)
    if not enabled(agent):
        return
    reset_rating(agent, node)
    if node.is_buggy is not False or getattr(node, "is_valid", None) is False:
        return
    raw = response.get("potential_level")
    value = potential_level(raw)
    if raw is not None and value is None:
        logger.warning("Node %s has malformed potential_level; using neutral missing prior", node.id)
    if value is not None:
        node.agent_potential_level = value
        node.agent_potential_assessed_at = datetime.now(timezone.utc).isoformat()
