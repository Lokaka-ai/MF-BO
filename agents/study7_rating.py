"""The study7 review extension: one completed-node rating, no piBO coupling."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import time

INSTRUCTIONS = """Separately from the factual execution summary, return ONE potential_level
for the completed node's CURRENT code, not for its parent. Use only the
supplied code, observed public training/validation diagnostics, task rules,
and the fixed per-experiment resource protocol. Assess remaining feasible
research potential; do not equate current score or code complexity with potential.
0 = no clear evidence of a feasible remaining improvement;
1 = plausible mainly local/incremental improvement;
2 = concrete evidence of substantial feasible remaining improvement potential.
Return null for invalid execution or insufficient evidence.
Do not use private/test scores, unreleased draft slots or future experiments.
Keep summary factual and code_summary descriptive. Put the subjective judgment
only in potential_level. Do not add another rating, rationale or future plan."""


def enabled(agent):
    study = getattr(agent.cfg, "study7", None)
    return bool(getattr(getattr(study, "rating", None), "enabled", False))


def extend_review(agent, node, prompt, spec):
    prompt, spec = copy.deepcopy(prompt), copy.deepcopy(spec)
    context = {"node_id": node.id, "node_stage": node.stage,
        "protocol_id": getattr(getattr(agent, "autoresearch", None), "protocol_id", None),
        "training_seconds_per_trial": 300, "initialization": "from_scratch_seed_42",
        "gpu_count_per_workflow": 1, "parallel_workers": int(agent.scfg.parallel_search_num),
        "allocated_cpus": int(agent.cfg.cpu_number),
        "max_debug_depth": int(agent.scfg.max_debug_depth),
        "execution_timeout_seconds": int(agent.cfg.exec.timeout)}
    if getattr(agent.cfg, "task_type", "competition") == "competition":
        # Competition candidates choose their own training/initialization under
        # the task rules. The 300-second from-scratch protocol is nanoGPT-only.
        context.pop("training_seconds_per_trial")
        context.pop("initialization")
        context.update(task_type="competition",
                       workflow_budget_seconds=int(agent.acfg.time_limit),
                       workflow_seed=int(agent.acfg.seed),
                       gpu_count_per_workflow=int(agent.scfg.num_gpus),
                       cpus_per_worker=max(1, int(agent.cfg.cpu_number) // int(agent.scfg.parallel_search_num)))
    prompt["Introduction"] += '\nReturn one additional nullable integer field: "potential_level".'
    prompt["Fixed task and resource protocol for potential assessment"] = {
        "task": getattr(agent, "task_desc", ""),
        # The shared Markdown compiler accepts string/list/dict leaves only.
        # Keep typed context in the audit metadata, and render it for transport.
        "resources": json.dumps(context, ensure_ascii=False, sort_keys=True)}
    prompt["Completed-node potential assessment"] = INSTRUCTIONS
    spec.json_schema["properties"]["potential_level"] = {
        "anyOf": [{"type": "integer", "enum": [0, 1, 2]}, {"type": "null"}],
        "description": "Remaining potential of the completed current code; null if unavailable."}
    spec.json_schema["required"].append("potential_level")
    encode = lambda value: json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    node.study7_rating_metadata = {"mode": "study7_positive_mean_rank1_v1",
        "prompt_sha256": hashlib.sha256(encode(prompt)).hexdigest(),
        "schema_sha256": hashlib.sha256(encode(spec.json_schema)).hexdigest(),
        "input_code_sha256": hashlib.sha256(node.code.encode()).hexdigest(),
        "model": agent.acfg.feedback.model, "provider": agent.acfg.feedback.base_url,
        "context": context, "started_epoch": time.time(), "missing_reason": "review_pending"}
    node._study7_rating_prompt = prompt
    return prompt, spec


def save_rating(agent, node, response):
    node.agent_potential_level = None
    node.agent_potential_assessed_at = None
    metadata = dict(node.study7_rating_metadata or {})
    if node.is_buggy is not False or node.is_valid is False:
        metadata.update(missing_reason="invalid_execution", usable=False)
    else:
        raw = response.get("potential_level")
        legal = type(raw) is int and raw in (0, 1, 2)
        node.agent_potential_level = raw if legal else None
        node.agent_potential_assessed_at = datetime.now(timezone.utc).isoformat()
        reason = (None if legal else "missing_field" if "potential_level" not in response
                  else "insufficient_evidence" if raw is None else "malformed_field")
        metadata.update(missing_reason=reason, usable=legal, assessed_at=node.agent_potential_assessed_at,
                        raw_type=type(raw).__name__)
    metadata["review_wall_seconds"] = max(0.0, time.time() - metadata.get("started_epoch", time.time()))
    node.study7_rating_metadata = metadata
    runtime = getattr(agent, "phase1_runtime", None)
    logger = getattr(runtime, "selection_logger", None)
    if logger:
        logger._append("study7_ratings.jsonl", {"node_id": node.id,
            "potential_level": node.agent_potential_level, "metadata": metadata,
            "prompt": getattr(node, "_study7_rating_prompt", None)})
