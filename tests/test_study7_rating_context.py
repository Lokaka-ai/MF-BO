"""Prevent nanoGPT resource assumptions leaking into competition AQ04 ratings."""
import copy
import json
from types import SimpleNamespace as NS

import pytest

from agents.study7_rating import extend_review
from agents.result_parse_agent import get_review_func_spec
from llm.gemini import compile_prompt_to_md


def agent(task_type):
    search = NS(parallel_search_num=3, num_gpus=1, max_debug_depth=20)
    acfg = NS(search=search, time_limit=43200, seed=42,
              feedback=NS(model="muse-spark-1.3-contributor", base_url="codex://opencode-go"))
    return NS(cfg=NS(task_type=task_type, cpu_number="21", exec=NS(timeout=32400)),
              acfg=acfg, scfg=search, task_desc="Competition allows pretrained models.",
              autoresearch=NS(protocol_id="pinned-autoresearch-fixture") if task_type == "autoresearch" else None)


def review(a):
    node = NS(id="completed", stage="improve", code="# completed candidate")
    prompt = {"Introduction": "Review completed training", "Execution output": "validation score 0.8"}
    spec = get_review_func_spec(False)
    original_prompt, original_spec = copy.deepcopy(prompt), copy.deepcopy(spec)
    enriched, schema = extend_review(a, node, prompt, spec)
    context = json.loads(enriched["Fixed task and resource protocol for potential assessment"]["resources"])
    assert context == node.study7_rating_metadata["context"]
    assert prompt == original_prompt and spec.json_schema == original_spec.json_schema
    assert schema.json_schema["required"].count("potential_level") == 1
    compile_prompt_to_md(enriched)
    return context, enriched


def test_formal_competition_rating_uses_actual_resources_without_nanogpt_constraints():
    context, prompt = review(agent("competition"))
    assert "training_seconds_per_trial" not in context
    assert "initialization" not in context
    assert "from_scratch_seed_42" not in json.dumps(prompt)
    assert context["task_type"] == "competition"
    assert context["workflow_budget_seconds"] == 43200
    assert context["execution_timeout_seconds"] == 32400
    assert context["workflow_seed"] == 42
    assert context["parallel_workers"] == 3
    assert context["allocated_cpus"] == 21 and context["cpus_per_worker"] == 7
    assert context["gpu_count_per_workflow"] == 1
    assert "remaining" not in json.dumps(context)


def test_competition_resources_are_read_from_the_actual_run_config():
    a = agent("competition")
    a.acfg.time_limit, a.cfg.exec.timeout, a.acfg.seed = 12345, 900, 19
    a.cfg.cpu_number, a.scfg.parallel_search_num, a.scfg.num_gpus = "8", 2, 2
    context, _ = review(a)
    assert (context["workflow_budget_seconds"], context["execution_timeout_seconds"], context["workflow_seed"]) == (12345, 900, 19)
    assert (context["allocated_cpus"], context["parallel_workers"], context["cpus_per_worker"], context["gpu_count_per_workflow"]) == (8, 2, 4, 2)


def test_autoresearch_rating_context_preserves_original_contract_exactly():
    context, _ = review(agent("autoresearch"))
    assert context == {"node_id": "completed", "node_stage": "improve",
        "protocol_id": "pinned-autoresearch-fixture", "training_seconds_per_trial": 300,
        "initialization": "from_scratch_seed_42", "gpu_count_per_workflow": 1,
        "parallel_workers": 3, "allocated_cpus": 21, "max_debug_depth": 20,
        "execution_timeout_seconds": 32400}
