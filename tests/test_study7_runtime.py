"""Study7 production dispatch with only external LLM/execution/memory mocked."""
import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose
from omegaconf import OmegaConf
import pytest

from config import Config, _load_cfg
from config.study7 import SEVEN_PRESETS, STUDY7_SELECTORS, study7_config
from engine.agent_search import AgentSearch
from engine.executor import ExecutionResult
from engine.search_node import Journal, SearchNode
from engine.parent_selectors.runtime import Phase1SelectorRuntime, SelectionPending
from engine.parent_selectors.study7_gp import Study7Error
from tasks.autoresearch.adapter import AutoresearchTask, REPO, assemble_candidate, candidate_body
from tasks.autoresearch.study7_budget import admit_result
from tests_gp6.test_core import record, option
from test_autoresearch import result_for
from test_shared_drafts import make_pool
from utils import serialize
from utils.metric import MetricValue

ROOT = REPO.parents[2]


def load_setting(setting):
    with patch.object(OmegaConf, "from_cli", return_value=OmegaConf.create({
            "task_preset": "autoresearch", "experiment_preset": setting})):
        return OmegaConf.merge(OmegaConf.structured(Config), _load_cfg())


@pytest.fixture
def factory(tmp_path, monkeypatch, autoresearch_cache):
    clock, charges, feedbacks, generations, memories = [1_000_000.], [], [], [], []
    state = NS(fail_execution=False, fail_review=None, rating=2)
    monkeypatch.setattr("time.time", lambda: clock[0])
    def replay(seconds):
        charges.append(seconds)
        clock[0] += seconds
    monkeypatch.setattr("engine.shared_drafts.time.sleep", replay)
    monkeypatch.setattr(Phase1SelectorRuntime, "_owner_alive", lambda *a: False)

    class Memory:
        def __init__(self, **kwargs):
            self.records = []
        def save_node(self, node, parent):
            memories.append((node.id, node.analysis, node.code_summary))

    body = candidate_body((REPO / "train.py").read_text())
    def generate(**kw):
        generations.append(kw)
        if kw.get("json_schema"):
            return json.dumps({"reason": "Preserve the causal API.", "module": ["model_design"],
                               "plan": {"model_design": "Add a harmless implementation comment."}})
        return "Preserve causal GPT.\n```python\n" + body + "\n```"
    def diff(**kw):
        generations.append(kw)
        return "<<<<<<< SEARCH\nclass GPTConfig:\n=======\nclass GPTConfig:\n    # Dry-run interface unchanged.\n>>>>>>> REPLACE"
    def feedback(**kw):
        from llm.gemini import compile_prompt_to_md
        compile_prompt_to_md(kw["system_message"])
        feedbacks.append(kw)
        clock[0] += 2
        if state.fail_review:
            raise state.fail_review
        return {"is_bug": False, "summary": "The host evaluation passed.",
                "code_summary": "Causal GPT", "potential_level": state.rating}
    monkeypatch.setattr("agents.memory.global_memory.GlobalMemoryLayer", Memory)
    monkeypatch.setattr("agents.coder.stepwise_coder.generate", generate)
    monkeypatch.setattr("agents.coder.base_coder.generate", generate)
    monkeypatch.setattr("agents.planner.base_planner.generate", generate)
    monkeypatch.setattr(importlib.import_module("agents.coder.diff_coder.diff_generate"), "generate", diff)
    monkeypatch.setattr("agents.debug_agent.generate", diff)
    monkeypatch.setattr("agents.code_review_agent.query", lambda **kw: {
        "needs_revision": False, "reasoning": "Host API preserved", "revised_code": None})
    monkeypatch.setattr("agents.result_parse_agent.query", feedback)
    monkeypatch.setattr("engine.parent_selectors.rating_prior.prior_terms", lambda *a, **kw: pytest.fail("Old piBO invoked"))
    def build(setting, shared=True):
        cfg = load_setting(setting)
        directory = tmp_path / setting
        cfg.log_dir, cfg.workspace_dir = directory / "logs", directory / "workspace"
        cfg.workspace_dir.mkdir(parents=True, exist_ok=True)
        cfg.agent.time_limit = 21600
        cfg.cpu_number = 4
        if shared:
            protocol = AutoresearchTask(cfg).protocol_id
            cfg.shared_draft_pool = str(make_pool(directory / "pool", protocol, body))
        agent = AgentSearch((ROOT / "tasks/autoresearch/TASK.md").read_text(), cfg, Journal())
        def execute(code, node_id, *a, **kw):
            if state.fail_execution:
                return ExecutionResult(["Injected external failure"], 1, "RuntimeError")
            trial = cfg.workspace_dir / "trials" / node_id
            trial.mkdir(parents=True, exist_ok=True)
            (trial / "result.json").write_text(json.dumps(result_for(assemble_candidate(code))))
            return ExecutionResult(["Authoritative host result"], 800, None)
        return agent, execute
    return build, NS(clock=clock, charges=charges, feedbacks=feedbacks, generations=generations,
                     memories=memories, state=state)


def initialize(agent, execute):
    drafts = [agent.step(None, execute, execute_immediately=False) for _ in range(3)]
    for node in drafts:
        agent.execute_deferred_node(node, execute)
    return drafts


@pytest.mark.parametrize("setting", SEVEN_PRESETS)
def test_t28_t29_t34_t38_shared_initial_review_and_ordinary_dispatch(setting, factory):
    build, calls = factory
    agent, execute = build(setting)
    start = agent.start_time
    drafts = initialize(agent, execute)
    assert len(calls.feedbacks) == 3 and not calls.generations
    assert calls.charges == [11, 12, 13]
    assert calls.clock[0] - start == 42  # replay36 + three ordinary reviews of2sec
    assert all(node.metric.value == 1.7 and node.is_valid for node in drafts)
    rated = setting in {SEVEN_PRESETS[3], SEVEN_PRESETS[4]}
    for node, review in zip(drafts, calls.feedbacks):
        assert ("potential_level" in review["func_spec"].json_schema["properties"]) == rated
        assert node.agent_potential_level == (2 if rated else None)
        if rated:
            assert node.study7_rating_metadata["review_wall_seconds"] == 2
            assert "remaining" not in str(node.study7_rating_metadata["context"]).lower()
    assert len(agent.virtual_root.children) == 3 and agent.scfg.num_drafts == 5
    assert agent.scfg.parallel_search_num == 2
    child = agent.step(drafts[0], execute)
    assert child.is_valid and child.stage in {"improve", "evolution", "fusion"}
    runtime = agent.phase1_runtime
    assert runtime.transition_store.records()[-1].child_node_id == child.id
    if runtime.study7_enabled:
        assert runtime.root_draft_state["pending"] == {}
        assert not runtime.include_root_draft and runtime.managed_decisions
        if setting == SEVEN_PRESETS[6]:
            assert runtime.surrogate is None and runtime.study7_selector.gp is None
            assert not (runtime.artifact_root / "model_fits.jsonl").exists()
        else:
            assert runtime.surrogate.fit_diagnostics["N"] == 1
            assert runtime.study7_selector.state.reference["reference_node_ids"] == [n.id for n in drafts]
            assert runtime.surrogate.rows[0]["rating_s"] == (1 if rated else 0)
    for _, summary, description in calls.memories:
        assert "potential_level" not in summary + description


@pytest.mark.parametrize("setting", [SEVEN_PRESETS[1], SEVEN_PRESETS[4], SEVEN_PRESETS[6]])
def test_t21_t34_debug_closure_uses_production_repair_path(setting, factory):
    build, calls = factory
    agent, execute = build(setting)
    drafts = initialize(agent, execute)
    calls.state.fail_execution = True
    bad = agent.step(drafts[0], execute)
    assert bad.is_buggy
    runtime = agent.phase1_runtime
    if runtime.surrogate:
        assert runtime.surrogate.fit_diagnostics["N"] == 0
    original = runtime.transition_store.records()[-1]
    calls.state.fail_execution = False
    calls.state.rating = 0
    good = agent.step(bad, execute)
    assert good.stage == "debug" and good.is_valid
    if runtime.surrogate:
        assert runtime.surrogate.fit_diagnostics["N"] == 1
        assert runtime.surrogate.rows[0] == original.selection_features["study7"]
        scientific = runtime.study7_selector.state.nodes[good.id]
        assert scientific["scientific_parent_id"] == original.parent_node_id
        assert scientific["origin_proposal_id"] == original.decision_id
    assert good.agent_potential_level == (0 if setting == SEVEN_PRESETS[4] else None)


@pytest.mark.parametrize("setting", SEVEN_PRESETS[1:5] + SEVEN_PRESETS[6:])
def test_t26_t27_t33_resume_same_options_scores_and_choice(setting, factory):
    build, _ = factory
    agent, execute = build(setting)
    drafts = initialize(agent, execute)
    agent.step(drafts[0], execute)
    agent.capture_resume_state()
    saved = serialize.loads_json(serialize.dumps_json(agent.journal), Journal)
    cfg = copy.deepcopy(agent.cfg)
    cfg.resume_run_dir = cfg.log_dir.parent
    resumed = AgentSearch("ignored", cfg, saved)
    contexts = []
    for instance in (agent, resumed):
        ctx = instance.phase1_runtime.select_global_parent(instance)
        contexts.append(ctx)
        assert all(o.operator != "draft" for o in ctx.result.options)
    a, b = contexts
    assert a.option.parent_node_id == b.option.parent_node_id
    assert a.option.operator == b.option.operator
    assert [(o.parent_node_id, o.operator, o.state_features.get("study7")) for o in a.result.options] == [
        (o.parent_node_id, o.operator, o.state_features.get("study7")) for o in b.result.options]
    assert_allclose([s.acquisition_score for s in a.result.candidate_scores], [s.acquisition_score for s in b.result.candidate_scores])
    for instance, ctx in zip((agent, resumed), contexts):
        instance.phase1_runtime.cancel_reservation(instance, ctx)
    if setting != SEVEN_PRESETS[6]:
        assert resumed.phase1_runtime.study7_selector.state.reference == agent.phase1_runtime.study7_selector.state.reference


def test_t33_initial_barrier_and_e07_never_constructs_gp(factory, monkeypatch):
    monkeypatch.setattr("engine.parent_selectors.study7_selector.State4GP", lambda *a, **kw: pytest.fail("E07 created a GP"))
    build, _ = factory
    agent, execute = build(SEVEN_PRESETS[6])
    nodes = [agent.step(None, execute, execute_immediately=False) for _ in range(3)]
    with pytest.raises(SelectionPending):
        agent.phase1_runtime.select_global_parent(agent)
    for node in nodes:
        agent.execute_deferred_node(node, execute)
    context = agent.phase1_runtime.select_continuation_parent(agent, nodes[0])
    assert context.option.parent_node_id == nodes[0].id
    assert context.decision.model_version is None
    assert all(s.posterior_std_latent is None for s in context.result.candidate_scores)


@pytest.mark.parametrize("value,reason", [(True, "malformed_field"), ("2", "malformed_field"),
                                         (2.0, "malformed_field"), (None, "insufficient_evidence")])
def test_t14_t28_malformed_rating_does_not_rerun_or_invalidate_trial(value, reason, factory):
    build, calls = factory
    calls.state.rating = value
    agent, execute = build(SEVEN_PRESETS[3])
    drafts = initialize(agent, execute)
    assert len(calls.feedbacks) == 3
    assert all(n.is_valid and n.agent_potential_level is None for n in drafts)
    assert all(n.study7_rating_metadata["missing_reason"] == reason for n in drafts)


@pytest.mark.parametrize("setting", [SEVEN_PRESETS[3], SEVEN_PRESETS[4]])
def test_rating_resource_prompt_uses_real_transport_compiler(setting, factory):
    from llm.gemini import compile_prompt_to_md
    build, calls = factory
    agent, execute = build(setting)
    drafts = initialize(agent, execute)
    rendered = compile_prompt_to_md(calls.feedbacks[0]["system_message"])
    assert '"parallel_workers": 2' in rendered
    assert '"training_seconds_per_trial": 300' in rendered
    assert "Completed-node potential assessment" in rendered
    assert drafts[0].study7_rating_metadata["context"]["parallel_workers"] == 2


def test_provider_failure_is_blocked_not_silent_all_null(factory):
    build, calls = factory
    agent, execute = build(SEVEN_PRESETS[3])
    calls.state.fail_review = RuntimeError("provider unavailable")
    node = agent.step(None, execute, execute_immediately=False)
    with pytest.raises(Study7Error, match="provider_unavailable"):
        agent.execute_deferred_node(node, execute)
    assert not agent.phase1_runtime.transition_store.records()


@pytest.mark.parametrize("setting", SEVEN_PRESETS)
def test_t37_late_result_is_audited_and_excluded(setting, factory):
    build, calls = factory
    agent, execute = build(setting)
    drafts = initialize(agent, execute)
    calls.clock[0] = agent.start_time + 21599
    agent.step(drafts[0], execute)  # feedback completes2s later; native may return root
    result = agent.journal.nodes[-1]
    assert result.study7_budget_metadata["censored"]
    assert result.study7_budget_metadata["raw_metric"] == 1.7
    assert result.metric.value is None and result.is_valid is False
    assert agent.journal.get_best_node().id in {n.id for n in drafts}
    last = agent.phase1_runtime.transition_store.records()[-1]
    assert not last.valid and last.metadata["budget_eligible"] is False
    assert last.metadata["proposal_closed"] is False
    assert last.metadata["completed_active_seconds"] == 21601


def test_t35_all_presets_and_only_component_differences(tmp_path):
    configs = []
    for index, name in enumerate(SEVEN_PRESETS):
        cfg = load_setting(name)
        cfg.log_dir = tmp_path / name
        runtime = Phase1SelectorRuntime(cfg)
        assert cfg.study7.method_id == f"E{index+1:02d}"
        assert cfg.agent.search.parallel_search_num == 2
        if index in {1, 2, 3, 4}:
            simple = OmegaConf.to_container(cfg, resolve=True)
            for key in ("method_id", "ancestor", "rating"):
                simple["study7"].pop(key)
            simple.pop("log_dir")
            configs.append(simple)
    assert configs == [configs[0]]*4


def test_t31_t32_native_aliases_and_gp6_fixed_replay_unchanged(tmp_path):
    for native, alias in [("gp6", SEVEN_PRESETS[0]), ("mlevolve_original", SEVEN_PRESETS[5])]:
        configs, runtimes = [], []
        for name in (native, alias):
            cfg = load_setting(name)
            cfg.log_dir = tmp_path / name
            runtimes.append(Phase1SelectorRuntime(cfg))
            payload = OmegaConf.to_container(cfg, resolve=True)
            for key in ("study7", "log_dir"):
                payload.pop(key)
            configs.append(payload)
        assert configs[0] == configs[1]
        assert runtimes[0].selector_type == runtimes[1].selector_type
        if native == "gp6":
            for runtime in runtimes:
                runtime.surrogate.fit([record("d", value=-1.7)])
            assert_allclose(runtimes[0].surrogate.predict_quality([option("a")]), runtimes[1].surrogate.predict_quality([option("a")]))
        else:
            assert not runtimes[0].managed_decisions and not runtimes[1].managed_decisions
            assert runtimes[0]._legacy_result.__func__ is runtimes[1]._legacy_result.__func__


@pytest.mark.parametrize("override", [{"bo_parent_selector.model": "unknown"},
    {"study7.rating.enabled": True}, {"study7.rho_history": .4},
    {"study7.label_protocol": "other"}, {"bo_parent_selector.agent_rating.enabled": True}])
def test_t26_t36_invalid_config_rejected(override, tmp_path):
    cfg = load_setting(SEVEN_PRESETS[1])
    cfg.log_dir = tmp_path
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([f"{key}={str(value).lower() if isinstance(value,bool) else value}" for key, value in override.items()]))
    with pytest.raises(ValueError):
        Phase1SelectorRuntime(cfg)


def test_native_cannot_enable_study7_rating(tmp_path):
    cfg = load_setting(SEVEN_PRESETS[0])
    cfg.log_dir = tmp_path
    cfg.study7.rating.enabled = True
    with pytest.raises(ValueError, match="study7 components"):
        Phase1SelectorRuntime(cfg)
