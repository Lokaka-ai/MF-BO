"""Common study7 wall-clock admission; raw late results remain auditable."""
from __future__ import annotations

import time


def registered(agent):
    return getattr(getattr(getattr(agent, "cfg", None), "study7", None), "method_id", "") in {
        "E01", "E02", "E03", "E04", "E05", "E06", "E07",
        "BR01", "BR02", "BR03", "BR04", "BR05", "BR06", "BR07",
        "AQ01", "AQ02", "AQ03", "AQ04", "AQ05", "AQ06", "AQ07", "AQ08", "AQ09"}


def admit_result(agent, node):
    if not registered(agent):
        return
    completed = time.time()
    active = max(0.0, completed - agent.start_time)
    limit = float(agent.acfg.time_limit)
    metric = getattr(getattr(node, "metric", None), "value", None)
    node.study7_budget_metadata = {
        "method_id": agent.cfg.study7.method_id,
        "completed_epoch": completed, "completed_active_seconds": active,
        "duration_seconds": limit, "budget_eligible": active <= limit,
        "censored": active > limit, "raw_metric": metric,
        "raw_valid": node.is_valid is not False and node.is_buggy is False,
        "late_cleanup_seconds": max(0.0, active - limit)}
    if active > limit:
        # Do not let a result released during shutdown enter best-so-far,
        # top candidates, or a quality label. Keep the actual trial artifact.
        from utils.metric import WorstMetricValue
        node.metric = WorstMetricValue()
        node.is_valid, node.is_buggy = False, True
        node.agent_potential_level = None
        node.agent_potential_assessed_at = None


def outcome_metadata(agent, node):
    if not registered(agent):
        return {}
    data = dict(getattr(node, "study7_budget_metadata", {}) or {})
    if not data:
        active = max(0.0, time.time() - agent.start_time)
        data = {"completed_active_seconds": active, "duration_seconds": float(agent.acfg.time_limit),
                "budget_eligible": active <= agent.acfg.time_limit, "censored": active > agent.acfg.time_limit}
    if data["censored"]:
        data.update(proposal_closed=False, failure_reason="budget_censored", target_available=False,
                    target_status="budget_censored")
    return data
