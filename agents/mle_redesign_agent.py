"""Evidence-guided, complete-program evolution for fixed-evaluation ML tasks.

This preserves the AlphaEvolve redesign mechanism (closed measured attempts ->
structured replacement plan -> full program) while using the ML prediction
contract and host-scored results. It does not expose private validation labels
to the LLM.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

from agents.coder import plan_and_code_query
from agents.planner import build_chat_prompt_for_model
from agents.redesign_agent import FIELDS, parse_plan, prose, sha, structure
from agents.triggers import register_node
from engine.parent_selectors.audited_bo import ordered_closed
from engine.search_node import SearchNode
from llm import generate


def host_evidence(agent, node):
    """Expose only a completed host grade under this frozen outer protocol."""
    result = getattr(node, "evaluation_result", None)
    if not isinstance(result, dict):
        return {"available": False}
    protocol = agent.fixed_evaluation.protocol_id
    if result.get("protocol_id") != protocol or node.evaluation_protocol_id != protocol:
        raise RuntimeError("Redesign evidence protocol mismatch")
    if result.get("node_id") != node.id or result.get("attempt_id") != node.evaluation_attempt_id:
        raise RuntimeError("Redesign evidence attempt mismatch")
    if type(result.get("valid")) is not bool:
        raise RuntimeError("Malformed redesign host evidence")
    return {"available": True, "valid": result["valid"],
            "metric": result.get("metric") if result["valid"] else None,
            "n_scored": result.get("n_scored"), "n_expected": result.get("n_expected"),
            "error": prose(result.get("error"), 400)}


def evidence_packet(agent, parent):
    """Use ledger-closed, host-scored attempts."""
    rows = [r for r in ordered_closed(agent.phase1_runtime.transition_store.records())
            if r.metadata.get("budget_eligible", True)
            and r.metadata.get("target_status") != "budget_censored"]
    nodes = {n.id: n for n in list(agent.journal.nodes)}
    same = [r for r in rows if r.child_node_id in nodes
            and nodes[r.child_node_id].branch_id == parent.branch_id]
    other = [r for r in rows if r.child_node_id in nodes
             and nodes[r.child_node_id].branch_id != parent.branch_id]
    ids = [r.child_node_id for r in same[-6:] + other[-2:]]
    if parent.id not in ids:
        ids.insert(0, parent.id)
    packet = []
    for node_id in dict.fromkeys(ids):
        node = nodes[node_id]
        observation = host_evidence(agent, node)
        if not observation["available"]:
            continue
        prior = Path(agent.cfg.log_dir) / "redesign" / f"{node.id}.json"
        entry = {"node_id": node.id, "parent_id": node.parent.id if node.parent else None,
                 "stage": node.stage, "code_sha256": sha(node.code),
                 "llm_plan_claim": prose(node.plan), "code_structure": structure(node.code),
                 "host_observation": observation, "execution_seconds": getattr(node, "exec_time", None)}
        if prior.is_file():
            entry["previous_redesign_claim"] = json.loads(prior.read_text()).get("plan_claim")
        packet.append(entry)
    if not any(row["node_id"] == parent.id for row in packet):
        raise RuntimeError("Evolution parent lacks a completed host observation")
    return {"schema": "mle_redesign_evidence_v1", "task_id": agent.cfg.exp_id,
            "protocol_id": agent.fixed_evaluation.protocol_id, "parent_id": parent.id,
            "observations": packet,
            "interpretation": "Plans and static code structure are hypotheses; host grades are measured outcomes."}


def run(agent, parent):
    if agent.cfg.task_type != "competition" or agent.fixed_evaluation is None:
        raise RuntimeError("ML redesign requires competition fixed evaluation")
    packet = evidence_packet(agent, parent)
    evidence = json.dumps(packet, ensure_ascii=False, allow_nan=False)
    # The fixed evaluator's public preview includes table headers/rows and the
    # candidate contract. Other ML operators receive this same task evidence.
    if not agent.data_preview:
        raise RuntimeError("ML redesign requires the prepared public data preview")
    task = agent.task_desc + "\nPublic data preview and fixed evaluation contract:\n" + agent.data_preview
    intro = "You design complete machine-learning pipelines for a Kaggle-style task using measured host evaluation."
    request = (
        task + "\nCompleted host-scored evidence:\n" + evidence +
        "\nPropose ONE feasible structural redesign of the ML solution. Identify an inherited limitation suggested "
        "by the evidence and replace an algorithmic component, such as representation, model, training or ensemble "
        "architecture; do not merely rename the same method or tune parameters. Do not treat an LLM plan as a "
        "measured fact. Preserve the frozen outer validation, metric and prediction interfaces. The archive already "
        "protects the best full-scored candidate, so the redesign may score lower. Do not request or infer private V "
        "labels. Return only a JSON object with these fields: " + ", ".join(FIELDS) +
        ". evidence_node_ids is a list of supplied IDs; all other fields are short strings. "
        "expected_observable_change must name a checkable code or runtime difference, not an expected score.")
    planner_prompt = build_chat_prompt_for_model(agent.acfg.code.model, intro, request, "")
    responses = []
    plan = None
    for _ in range(3):
        response = generate(prompt=planner_prompt, temperature=agent.acfg.code.temp, cfg=agent.cfg)
        responses.append(response)
        try:
            plan = parse_plan(response, {r["node_id"] for r in packet["observations"]})
            break
        except (ValueError, TypeError):
            pass
    log_dir = Path(agent.cfg.log_dir) / "redesign"
    log_dir.mkdir(parents=True, exist_ok=True)
    record = {"version": 1, "task_type": "competition", "parent_id": parent.id,
              "evidence": packet, "planner_prompt": planner_prompt,
              "planner_responses": responses, "started_at": time.time(), "plan_claim": plan}
    if plan is None:
        code = "raise RuntimeError('Redesign plan format failed')\n"
        plan_text = "Invalid redesign planning output; no structural claim was accepted."
        coder_prompt = None
    else:
        plan_text = json.dumps(plan, ensure_ascii=False)
        instruction = (
            task + "\nCompleted host-scored evidence:\n" + evidence +
            "\nSelected redesign plan:\n" + plan_text +
            "\nImplement this plan as a COMPLETE standalone Python program. The previous program is intentionally "
            "not supplied; build the pipeline from the task and fixed evaluation contract. Do not emit a diff or "
            "SEARCH/REPLACE patch. Read the host-provided ./input/ view, fit preprocessing and models using T only, "
            "predict every fixed V record, write exactly the required validation.csv, and write "
            "the complete test submission. Keep V and test prediction logic identical. Stay within the execution budget. "
            "Return a short explanation followed by one Python code block; state if any plan element is omitted.")
        coder_prompt = build_chat_prompt_for_model(agent.acfg.code.model, intro, instruction, "")
        _, code = plan_and_code_query(agent, coder_prompt)
        code = code or ""
    record.update(coder_prompt=coder_prompt, code_sha256=sha(code),
                  implementation_structure=structure(code),
                  parent_code_identical=(code == parent.code), semantic_change_verified=None)
    parent.add_expected_child_count()
    node = SearchNode(plan=plan_text, code=code, parent=parent, stage="evolution",
                      local_best_node=parent.local_best_node,
                      from_topk=getattr(parent, "_topk_triggered", False))
    register_node(agent, node, {"planner": planner_prompt, "coder": coder_prompt}, parent_node=parent)
    if hasattr(parent, "_topk_triggered"):
        del parent._topk_triggered
    (log_dir / f"{node.id}.json").write_text(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return node


def record_outcome(agent, node, result):
    path = Path(agent.cfg.log_dir) / "redesign" / f"{node.id}.json"
    if not path.is_file():
        return
    data = json.loads(path.read_text())
    if result.get("protocol_id") != data["evidence"]["protocol_id"] or result.get("node_id") != node.id:
        raise RuntimeError("Redesign outcome protocol/node mismatch")
    if result.get("attempt_id") != node.evaluation_attempt_id:
        raise RuntimeError("Redesign outcome attempt mismatch")
    reviewed = data.get("reviewed_code_sha256")
    if reviewed != sha(node.code):
        raise RuntimeError("Redesign execution source differs from reviewed source")
    data["outcome"] = {"valid": result.get("valid"), "metric": result.get("metric"),
                       "n_scored": result.get("n_scored"), "n_expected": result.get("n_expected"),
                       "error": prose(result.get("error"), 500),
                       "attempt_id": result.get("attempt_id")}
    data["completed_at"] = time.time()
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def record_reviewed_source(agent, node):
    """Code review can change generated code; bind the record to what executes."""
    path = Path(agent.cfg.log_dir) / "redesign" / f"{node.id}.json"
    if not path.is_file():
        raise RuntimeError("Missing redesign generation record before execution")
    data = json.loads(path.read_text())
    data["reviewed_code_sha256"] = sha(node.code)
    data["reviewed_structure"] = structure(node.code)
    data["review_changed_code"] = data["reviewed_code_sha256"] != data["code_sha256"]
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
