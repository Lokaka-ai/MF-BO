"""Native v3 dispatch and checkpoint reconciliation, under the journal lock."""
import time
from engine.parent_selectors.runtime import SearchExhausted, SelectionPending
from engine.operator_preview import OperatorPreview, preview_next_operator
from engine.mf_gp.policy import exact_ceiling_reached


def select_global(runtime, agent):
    s = runtime.study7_selector
    state = runtime.root_draft_state
    s.observe(runtime.transition_store.records())
    remaining = s.remaining_budget(agent)
    if remaining <= 0 or exact_ceiling_reached(s.observations, agent.fixed_evaluation.spec["metric"]):
        raise SearchExhausted("V3 deadline or exact full metric ceiling")
    # Prepared actions already consumed a generation slot; they may execute even
    # after the generation step cap, but no additional promotion is free.
    ready = any(e["state"] in ("ready", "deferred") for e in s.prepared.entries.values())
    if ready:
        options = runtime._prefer_unoccupied_options(runtime._add_root_option(agent, runtime._normal_options(agent))) if runtime._budget_available(agent) and state["initial_submitted"] >= state["initial_goal"] else []
        return s.select(agent, options)
    if not runtime._budget_available(agent):
        if state["pending"] or s.pending_promotions:
            raise SelectionPending(state["revision"])
        raise SearchExhausted("V3 action budget exhausted")
    occupied = sum(e["state"] in ("generating", "ready") for e in s.prepared.entries.values())
    if occupied >= 3:
        return s.select(agent, [])
    # Anchor promotions take priority, including while other initial work runs.
    if s.calibration is None and any(o["valid"] and s.programs[o["program_id"]]["initial"] for o in s.observations if o.get("program_id")):
        try:
            return s.select(agent, [])
        except SearchExhausted:
            pass
        except SelectionPending:
            pass
    reserved = sum(p["initial"] and p["status"] == "reserved" for p in state["pending"].values())
    if state["initial_submitted"] + reserved < state["initial_goal"]:
        cheap, _, gen, full_guard = s.costs()
        initial_left = .4*agent.acfg.time_limit-(time.time()-agent.start_time)
        if 1.5*(gen+cheap) <= initial_left and 1.5*(gen+cheap)+full_guard <= remaining:
            return runtime._forced_result(agent, agent.virtual_root, reason="initial_draft_bypass",
                preview=OperatorPreview("draft", bypass_selector=True, reason="initial_draft"))
        return s.select(agent, [])
    pending_parents = runtime._pending_parent_ids()
    for node in agent.journal.nodes:
        if node.stage != "root" and node.id not in pending_parents and not node.is_terminal and not node.lock and not node.reached_child_limit(agent.scfg):
            preview = preview_next_operator(agent, node)
            if preview.operator == "debug":
                return runtime._forced_result(agent, node, preview=preview, reason="special_path_bypass")
    options = runtime._prefer_unoccupied_options(runtime._add_root_option(agent, runtime._normal_options(agent)))
    return s.select(agent, options)


def reconcile(runtime, agent):
    s = runtime.study7_selector
    pending = runtime.root_draft_state["pending"]
    # Native reconciliation restores exact frozen generation and node snapshots.
    # Unsubmitted reservations removed by the native ledger release their slots.
    for ticket, e in s.prepared.entries.items():
        if ticket not in pending and e["state"] == "generating":
            s.prepared.finish(ticket, success=False, reason="unsubmitted_reservation_released")
    runtime.recovered_deferred_nodes = [n for n in runtime.recovered_deferred_nodes
        if s.prepared.entries[n.phase1_decision_id]["state"] not in ("expired", "failed", "completed")]
