"""Root-draft tests run production agents/selection/dispatch/parsing/validation.

Only LLM transport, grading HTTP, and the external interpreter are replaced.
"""

import itertools
import json
import re
import uuid
import copy
import shutil
import sys
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from config import _load_cfg, validate_root_draft_config
from engine.agent_search import AgentSearch
from engine.executor import ExecutionResult
from engine.search_node import Journal
from engine.parent_selectors.base import OperatorPreview
from engine.parent_selectors.features import BNVS_ROOT_FEATURE_NAMES, vectorize_bnvs_root_features
from engine.parent_selectors.features import reward_scale_from_records
from engine.parent_selectors.runtime import Phase1SelectorRuntime, SelectionPending, SearchExhausted
from utils import serialize


class ExternalServices:
    metric = 0.5
    fail = None

    def generate(self, **kwargs):
        if self.fail == "generation":
            raise RuntimeError("external generation failure")
        return "A candidate plan.\n```python\nprint('candidate')\n```"

    def query(self, **kwargs):
        spec = kwargs.get("func_spec")
        if spec is None:
            return "Predict labels; maximize validation accuracy."
        if spec.name == "determine_metric_direction":
            return {"lower_is_better": False, "reasoning": "accuracy"}
        if spec.name == "submit_code_review":
            return {"needs_revision": False, "reasoning": "OK", "revised_code": None}
        if self.fail == "parse":
            raise ValueError("malformed external parser response")
        metric = float(re.search(r"metric:([-0-9.]+)", str(kwargs["system_message"])).group(1))
        return {"is_bug": False, "summary": "Completed", "metric": metric, "lower_is_better": False}

    def execute(self, cfg, code, node_id, *args):
        if self.fail == "execution":
            raise TimeoutError("external execution timed out")
        path = cfg.workspace_dir / "submission" / f"submission_{node_id}.csv"
        path.parent.mkdir(exist_ok=True)
        path.write_text("id,prediction\n" + "".join(f"{i},{i / 33}\n" for i in range(32)))
        return ExecutionResult([f"metric:{self.metric}"], 1.0, None)

    def http(self, *args, **kwargs):
        return SimpleNamespace(status_code=200, json=lambda: {"is_valid": self.fail != "validation", "result": "test"})


@pytest.fixture
def real_agent(tmp_path, monkeypatch):
    services = ExternalServices()
    ids = itertools.count(1)
    monkeypatch.setattr(uuid, "uuid4", lambda: uuid.UUID(int=next(ids)))
    monkeypatch.setattr("time.time", lambda: 1000.0)
    monkeypatch.setattr("llm.query", services.query)
    monkeypatch.setattr("agents.result_parse_agent.query", services.query)
    monkeypatch.setattr("agents.code_review_agent.query", services.query)
    monkeypatch.setattr("agents.coder.base_coder.generate", services.generate)
    monkeypatch.setattr("engine.validation.format_client.requests.get", services.http)
    monkeypatch.setattr("engine.validation.format_client.requests.post", services.http)

    def build(*, initial=1, cap=3, seed=0, scope="all_normal_steps", selector="bnvs_ts", enabled=True):
        cfg = _load_cfg(use_cli_args=False)
        cfg.evaluation.enabled = False  # ExternalServices supplies legacy parsed scores.
        cfg.log_dir = tmp_path / f"logs-{seed}-{initial}-{cap}-{selector}"
        cfg.workspace_dir = tmp_path / f"workspace-{seed}-{initial}-{cap}-{selector}"
        (cfg.workspace_dir / "input").mkdir(parents=True, exist_ok=True)
        cfg.exp_name = "test_root_draft"
        cfg.agent.use_global_memory = False
        cfg.coldstart.use_coldstart = False
        cfg.agent.use_stepwise_generation = False
        cfg.agent.use_diff_mode = False
        cfg.agent.use_aggregation = False
        cfg.agent.initial_drafts = initial
        cfg.agent.search.num_drafts = cap
        cfg.agent.seed = seed
        cfg.parent_selector.include_root_draft = enabled
        cfg.parent_selector.type = selector
        cfg.parent_selector.decision_scope = scope
        agent = AgentSearch("Predict labels", cfg, Journal())
        callback = lambda *args: services.execute(cfg, *args)
        return agent, callback, services

    return build


def test_root_draft_real_vertical(real_agent):
    agent, execute, services = real_agent(seed=0)
    runtime = agent.phase1_runtime
    first = agent.step(agent.virtual_root, execute, execute_immediately=False)
    assert first.stage == "draft" and first.parent is agent.virtual_root
    agent.execute_deferred_node(first, execute)
    initial = runtime.transition_store.records()[0]
    assert initial.metadata["target_status"] == "target_unavailable"
    assert initial.selection_features["incumbent_at_decision"] is None
    child = agent.step(first, execute)
    records = runtime.transition_store.records()
    assert records[-1].operator == "draft"
    assert child.parent is agent.virtual_root and child.stage == "draft"
    assert child.branch_id != first.branch_id
    assert records[-1].metadata["decision_kind"] == "model"
    assert records[-1].metadata["target_available"] is True
    assert records[-1].parent_utility is None
    assert records[-1].improvement is None
    model = runtime.surrogate
    model.fit(records)
    assert model.training_transition_ids == [records[-1].transition_id]
    assert model.posterior_mean.shape == (13,)
    assert np.isfinite(model.posterior_mean).all()
    assert runtime.root_draft_state["draft_submitted"] == 2
    assert runtime.root_draft_state["pending"] == {}
    assert first.continuation_closed_transition_id == initial.transition_id
    assert first.visits == 1
    _, repeated = runtime._close_current_continuation(agent, first)
    assert repeated is False and first.visits == 1
    event = json.loads((runtime.artifact_root / "continuation_events.jsonl").read_text().splitlines()[-1])
    assert event["outcome"] == "new_direction" and event["backpropagated"] is True


def complete_initial(agent, execute, count=1):
    nodes = [agent.step(agent.virtual_root, execute, execute_immediately=False) for _ in range(count)]
    for node in nodes:
        agent.execute_deferred_node(node, execute)
    return nodes


@pytest.mark.parametrize("failure", ["generation", "execution", "parse", "validation"])
def test_final_failures_and_frozen_retries_count_once(real_agent, failure):
    agent, execute, services = real_agent()
    first, = complete_initial(agent, execute)
    runtime = agent.phase1_runtime
    context = runtime.select_continuation_parent(agent, first)
    assert context.option.operator == "draft"
    assert context.option.preview.reason == "adaptive_root_draft"
    services.fail = failure
    if failure in {"generation", "execution"}:
        with pytest.raises((RuntimeError, TimeoutError)):
            agent._run_single_step(context.option.parent_node, context, execute)
    else:
        agent._run_single_step(context.option.parent_node, context, execute)
    records = runtime.transition_store.records()
    assert len(records) == 2 and records[-1].valid is False
    if failure == "generation":
        assert records[-1].child_node_id is None
    assert records[-1].metadata["target_available"] is True
    model = runtime.surrogate
    model.fit(records)
    assert model._target(records[-1]) == -1
    assert model.training_transition_ids == [records[-1].transition_id]
    index = runtime.root_draft_state["decision_count"]
    # Retrying the same submitted/final decision does not dispatch or draw again.
    agent._run_single_step(context.option.parent_node, context, execute)
    assert len(runtime.transition_store.records()) == 2
    assert runtime.root_draft_state["draft_submitted"] == 2
    assert runtime.root_draft_state["decision_count"] == index


@pytest.mark.parametrize("scope,where,count", [
    ("all_normal_steps", "global", 1), ("all_normal_steps", "continuation", 1),
    ("root_only", "global", 1), ("root_only", "continuation", 0),
])
def test_scope_and_single_frozen_root(real_agent, scope, where, count):
    agent, execute, _ = real_agent(scope=scope, selector="bnvs_mean_greedy")
    first, = complete_initial(agent, execute)
    runtime = agent.phase1_runtime
    context = (runtime.select_global_parent(agent) if where == "global"
               else runtime.select_continuation_parent(agent, first))
    roots = [o for o in context.result.options if o.parent_node_id == agent.virtual_root.id]
    assert len(roots) == count
    if roots:
        assert roots[0].preview == OperatorPreview("draft", bypass_selector=False, reason="adaptive_root_draft")
        assert context.result.options[-1] is roots[0]
    assert runtime.decision_scope == scope


def test_mean_tie_continues_without_filling_draft_cap(real_agent):
    agent, execute, _ = real_agent(selector="bnvs_mean_greedy")
    first, = complete_initial(agent, execute)
    child = agent.step(first, execute)
    assert child.stage == "improve" and child.parent is first
    assert agent.phase1_runtime.root_draft_state["draft_submitted"] == 1
    event = json.loads((agent.phase1_runtime.artifact_root / "continuation_events.jsonl").read_text().splitlines()[-1])
    assert event["outcome"] == "continue" and event["backpropagated"] is False


def test_learned_value_switches_to_another_node(real_agent):
    agent, execute, services = real_agent(initial=2, cap=2, selector="bnvs_mean_greedy")
    nodes = [agent.step(agent.virtual_root, execute, execute_immediately=False) for _ in range(2)]
    for node, metric in zip(nodes, [0.2, 0.8]):
        services.metric = metric
        agent.execute_deferred_node(node, execute)
    low, high = nodes
    runtime = agent.phase1_runtime
    runtime._close_current_continuation(agent, high)
    services.metric = 0.25
    child = agent.step(low, execute)
    assert child.parent is low
    services.metric = 0.9
    switched = agent.step(child, execute)
    assert switched.parent is high and switched.stage == "improve"
    assert child.continuation_closed_transition_id is not None


def test_initial_terminal_failures_and_cap(real_agent):
    agent, execute, services = real_agent(initial=3, cap=5)
    runtime = agent.phase1_runtime
    generated = [agent.step(agent.virtual_root, execute, execute_immediately=False) for _ in range(2)]
    services.fail = "generation"
    with pytest.raises(RuntimeError):
        agent.step(agent.virtual_root, execute, execute_immediately=False)
    assert runtime.root_draft_state["initial_submitted"] == 3
    assert runtime.root_draft_state["initial_terminal"] == 1
    services.fail = None
    agent.execute_deferred_node(generated[0], execute)
    assert runtime._root_legality(agent) == "initialization_pending"
    # Completed normal chains may already run while another initial is pending.
    context = runtime.select_continuation_parent(agent, generated[0])
    assert all(o.operator != "draft" for o in context.result.options)
    runtime.cancel_reservation(agent, context)
    agent.execute_deferred_node(generated[1], execute)
    assert runtime.root_draft_state["initial_terminal"] == 3
    assert runtime._root_legality(agent) == "legal"
    for _ in range(2):
        context = runtime.select_global_parent(agent)
        assert context.option.operator == "draft"
        services.fail = "generation"
        with pytest.raises(RuntimeError):
            agent._run_single_step(context.option.parent_node, context, execute)
    assert runtime.root_draft_state["draft_submitted"] == 5
    assert runtime._root_legality(agent) == "draft_cap_exhausted"
    assert len(runtime.transition_store.records()) == 5


def test_exact12_direct_draft_statistics_and_interactions(real_agent):
    agent, execute, services = real_agent(initial=2, cap=4, selector="bnvs_mean_greedy")
    nodes = [agent.step(agent.virtual_root, execute, execute_immediately=False) for _ in range(2)]
    for node, metric in zip(nodes, [0.2, 0.6]):
        services.metric = metric
        agent.execute_deferred_node(node, execute)
    services.metric = 0.9
    descendant = agent.step(nodes[0], execute)
    assert descendant.stage == "improve"
    runtime = agent.phase1_runtime
    root = runtime._make_option(agent, agent.virtual_root,
        preview=OperatorPreview("draft", reason="adaptive_root_draft"))
    f = dict(root.state_features)
    assert f["source_quality"] == pytest.approx(0.4)
    assert f["parent_global_gap"] == pytest.approx(0.5)
    assert f["visit_count"] == 2 and f["recent_valid_rate"] == 1
    assert f["parent_utility"] is None and agent.virtual_root.metric.value is None
    vector = vectorize_bnvs_root_features(f, utility_scale=0.1)
    assert len(vector) == len(BNVS_ROOT_FEATURE_NAMES) == 12
    assert vector[:3] == pytest.approx([5, 0, np.log1p(2)])
    assert vector[3:6] == [0, 0, 0]
    normal = runtime._make_option(agent, descendant)
    assert vectorize_bnvs_root_features(normal.state_features, utility_scale=0.1)[-3:] == [0, 0, 0]
    f.update(remaining_steps_fraction=0.25, elapsed_fraction=0.5, global_stagnation=7)
    assert vectorize_bnvs_root_features(f, utility_scale=0.1)[-3:] == pytest.approx([1, 0.25, np.log1p(7)])


def test_concurrent_reservation_and_pre_submit_cancellation(real_agent):
    agent, execute, _ = real_agent(cap=2)
    complete_initial(agent, execute)
    runtime = agent.phase1_runtime
    def select():
        try:
            return runtime.select_global_parent(agent)
        except SelectionPending:
            return None
    with ThreadPoolExecutor(4) as pool:
        decisions = list(pool.map(lambda _: select(), range(4)))
    selected = [context for context in decisions if context is not None]
    assert len(selected) == 1 and selected[0].option.operator == "draft"
    assert runtime.root_draft_state["draft_submitted"] == 1
    runtime.cancel_reservation(agent, selected[0])
    assert runtime.root_draft_state["pending"] == {}
    replacement = runtime.select_global_parent(agent)
    assert runtime.submit_decision(agent, replacement) is True
    assert runtime.submit_decision(agent, replacement) is False
    runtime.cancel_reservation(agent, replacement)
    assert runtime.root_draft_state["draft_submitted"] == 2
    assert len(runtime.root_draft_state["pending"]) == 1


def test_empty_candidates_wait_then_stop_without_extra_submission(real_agent):
    agent, execute, _ = real_agent(cap=2)
    complete_initial(agent, execute)
    started, release, waiting = threading.Event(), threading.Event(), threading.Event()
    def blocked_execute(*args):
        started.set()
        assert release.wait(5)
        return execute(*args)
    def waiting_worker():
        waiting.set()
        return agent.step(agent.virtual_root, execute)
    with ThreadPoolExecutor(2) as pool:
        running = pool.submit(agent.step, agent.virtual_root, blocked_execute)
        assert started.wait(5)
        idle = pool.submit(waiting_worker)
        assert waiting.wait(5)
        assert not idle.done()
        release.set()
        assert running.result(timeout=5).stage == "draft"
        with pytest.raises(SearchExhausted):
            idle.result(timeout=5)
    assert agent.phase1_runtime.root_draft_state["draft_submitted"] == 2


def clone_checkpoint(agent, directory):
    cfg = copy.deepcopy(agent.cfg)
    cfg.log_dir = directory / "logs"
    shutil.copytree(agent.phase1_runtime.artifact_root, cfg.log_dir / "phase1")
    journal = serialize.loads_json(serialize.dumps_json(agent.journal), Journal)
    return cfg, journal


def dead_owner(*args):
    raise ProcessLookupError()


def test_resume_preserves_posterior_counter_and_next_joint_draw(real_agent, tmp_path):
    agent, execute, _ = real_agent()
    first, = complete_initial(agent, execute)
    second = agent.step(first, execute)
    runtime = agent.phase1_runtime
    runtime._close_current_continuation(agent, second)
    context = runtime.select_global_parent(agent)  # Fit both observations.
    runtime.cancel_reservation(agent, context)
    agent.capture_resume_state()
    cfg, journal = clone_checkpoint(agent, tmp_path / "restart")
    fit_count = len((runtime.artifact_root / "model_fits.jsonl").read_text().splitlines())
    resumed = AgentSearch("ignored on resume", cfg, journal)
    restored = resumed.phase1_runtime
    assert restored.surrogate.model_version == runtime.surrogate.model_version
    assert restored.surrogate.fitted_transition_count == 2
    np.testing.assert_array_equal(restored.surrogate.posterior_mean, runtime.surrogate.posterior_mean)
    np.testing.assert_array_equal(restored.surrogate.posterior_precision_inverse, runtime.surrogate.posterior_precision_inverse)
    actual = restored.select_global_parent(resumed)
    expected = runtime.select_global_parent(agent)
    assert [o.parent_node_id for o in actual.result.options] == [o.parent_node_id for o in expected.result.options]
    assert [s.sampled_value for s in actual.result.candidate_scores] == pytest.approx(
        [s.sampled_value for s in expected.result.candidate_scores])
    assert actual.option.parent_node_id == expected.option.parent_node_id
    assert len((restored.artifact_root / "model_fits.jsonl").read_text().splitlines()) == fit_count
    assert restored.root_draft_state["initial_submitted"] == 1
    assert restored.root_draft_state["draft_submitted"] == 2


def test_resume_deferred_initial_without_regeneration_or_double_charge(real_agent, tmp_path, monkeypatch):
    agent, execute, _ = real_agent()
    node = agent.step(agent.virtual_root, execute, execute_immediately=False)
    agent.capture_resume_state()
    cfg, journal = clone_checkpoint(agent, tmp_path / "restart")
    with pytest.raises(RuntimeError, match="original process is alive"):
        AgentSearch("ignored", cfg, journal)
    monkeypatch.setattr("engine.parent_selectors.runtime.os.kill", dead_owner)
    resumed = AgentSearch("ignored", cfg, journal)
    runtime = resumed.phase1_runtime
    assert len(runtime.recovered_deferred_nodes) == 1
    recovered = runtime.recovered_deferred_nodes[0]
    assert recovered.id == node.id and recovered.code == node.code
    assert recovered.parent is resumed.virtual_root
    assert recovered._phase1_decision_context.decision.decision_id == node.phase1_decision_id
    resumed.execute_deferred_node(recovered, execute)
    resumed.execute_deferred_node(recovered, execute)
    assert runtime.root_draft_state["draft_submitted"] == 1
    assert runtime.root_draft_state["initial_terminal"] == 1
    assert len(runtime.transition_store.records()) == 1
    assert runtime.transition_store.records()[0].metadata["target_available"] is False


def test_resume_interrupted_submission_records_one_terminal_without_child(real_agent, tmp_path, monkeypatch):
    agent, _, _ = real_agent()
    runtime = agent.phase1_runtime
    context = runtime.select_global_parent(agent)
    runtime.submit_decision(agent, context)
    agent.capture_resume_state()
    cfg, journal = clone_checkpoint(agent, tmp_path / "restart")
    monkeypatch.setattr("engine.parent_selectors.runtime.os.kill", dead_owner)
    resumed = AgentSearch("ignored", cfg, journal)
    restored = resumed.phase1_runtime
    records = restored.transition_store.records()
    assert len(records) == 1 and records[0].child_node_id is None
    assert records[0].decision_id == context.decision.decision_id
    assert records[0].metadata["target_status"] == "target_unavailable"
    assert records[0].metadata["failure_reason"] == "interrupted_before_terminal_observation"
    assert restored.root_draft_state["draft_submitted"] == restored.root_draft_state["initial_terminal"] == 1
    resumed.capture_resume_state()
    # A second recovery observes the same terminal, without learning it twice.
    again = AgentSearch("ignored", cfg, serialize.loads_json(serialize.dumps_json(journal), Journal))
    assert len(again.phase1_runtime.transition_store.records()) == 1


@pytest.mark.parametrize("change", ["mode", "dimension", "schema", "target", "prior"])
def test_incompatible_checkpoint_rejected(real_agent, tmp_path, change):
    agent, execute, _ = real_agent()
    complete_initial(agent, execute)
    agent.capture_resume_state()
    cfg, journal = clone_checkpoint(agent, tmp_path / "restart")
    if change == "mode":
        cfg.parent_selector.include_root_draft = False
    elif change == "prior":
        cfg.bo_parent_selector.value_model.prior_precision += 1
    else:
        key = {"dimension": "feature_dimension", "schema": "feature_schema_version", "target": "target"}[change]
        journal.parent_selector_state[key] = "incompatible"
    with pytest.raises(ValueError, match="[Ii]ncompatible|across"):
        AgentSearch("ignored", cfg, journal)


@pytest.mark.parametrize("field,value", [
    ("parent_selector.type", "bnvs_ts_local"), ("parent_selector.type", "bayesian_child_ucb"),
    ("bo_parent_selector.value_model.target", "local_gain"), ("bo_parent_selector.value_model.horizon", 2),
    ("agent.search.num_drafts", 0), ("parent_selector.include_root_draft", "yes"),
])
def test_invalid_enabled_combinations_fail_at_startup(field, value):
    cfg = _load_cfg(use_cli_args=False)
    cfg.parent_selector.include_root_draft = True
    cfg.agent.initial_drafts = 1
    target = cfg
    parts = field.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value
    with pytest.raises(ValueError):
        validate_root_draft_config(cfg)


def test_missing_draft_quality_with_real_incumbent_and_no_synthetic_target(real_agent):
    agent, execute, services = real_agent()
    services.fail = "validation"
    first, = complete_initial(agent, execute)
    assert first.is_buggy is True
    services.fail = None
    debugged = agent.step(first, execute)
    assert debugged.stage == "debug" and debugged.metric.value == 0.5
    runtime = agent.phase1_runtime
    option = runtime._make_option(agent, agent.virtual_root,
        preview=OperatorPreview("draft", reason="adaptive_root_draft"))
    assert option.state_features["source_quality"] is None
    assert option.state_features["source_quality_missing"] == 1
    assert option.state_features["parent_global_gap"] == 0
    assert option.state_features["incumbent_at_decision"] == 0.5
    assert option.state_features["visit_count"] == 1
    assert option.state_features["recent_valid_rate"] == 0
    runtime.surrogate.fit(runtime.transition_store.records())
    assert runtime.surrogate.training_transition_ids == []
    assert np.isfinite(runtime.surrogate._design(option.state_features)).all()


def test_scale_uses_valid_initials_even_without_h1_labels(real_agent):
    agent, execute, services = real_agent(initial=2, cap=3, selector="bnvs_mean_greedy")
    initial = [agent.step(agent.virtual_root, execute, execute_immediately=False) for _ in range(2)]
    for node, metric in zip(initial, [0.1, 0.9]):
        services.metric = metric
        agent.execute_deferred_node(node, execute)
    services.metric = 0.101
    agent.step(initial[0], execute)
    runtime = agent.phase1_runtime
    records = runtime.transition_store.records()
    runtime.surrogate.fit(records)
    assert runtime.surrogate.utility_scale == reward_scale_from_records(records)
    assert runtime.surrogate.utility_scale > 10 * reward_scale_from_records(records[-1:])
    assert runtime.surrogate.training_transition_ids == [records[-1].transition_id]
    assert records[-1].selection_features["incumbent_at_decision"] == 0.9
    assert all(r.metadata["target_available"] is False for r in records[:2])


def test_global_stagnation_counts_only_terminal_experiments(real_agent):
    agent, execute, services = real_agent(selector="bnvs_mean_greedy")
    agent.scfg.metric_improvement_threshold = 0.01
    node, = complete_initial(agent, execute)
    runtime = agent.phase1_runtime
    assert runtime._global_stagnation(agent) == 0
    for metric, expected in [(0.505, 1), (0.511, 2), (0.53, 0)]:
        services.metric = metric
        node = agent.step(node, execute)
        assert runtime._global_stagnation(agent) == expected
    context = runtime.select_continuation_parent(agent, node)
    assert runtime._global_stagnation(agent) == 0
    runtime.cancel_reservation(agent, context)


def test_joint_draw_has_no_independent_candidate_noise(real_agent):
    agent, execute, _ = real_agent()
    first, = complete_initial(agent, execute)
    runtime = agent.phase1_runtime
    context = runtime.select_continuation_parent(agent, first)
    options = list(context.result.options)
    assert {o.operator for o in options} == {"improve", "draft"}
    values = runtime.surrogate.sample_values(options + options, seed=31)
    np.testing.assert_array_equal(values[:len(options)], values[len(options):])
    old_index = runtime.root_draft_state["decision_count"]
    runtime.cancel_reservation(agent, context)
    next_context = runtime.select_global_parent(agent)
    assert runtime.root_draft_state["decision_count"] == old_index + 1
    assert next_context.decision.decision_id != context.decision.decision_id


def test_no_incumbent_and_no_budget_stops_cleanly(real_agent):
    agent, execute, _ = real_agent(initial=0, cap=0)
    with pytest.raises(SearchExhausted):
        agent.step(agent.virtual_root, execute)
    assert agent.phase1_runtime.transition_store.records() == []
    assert agent.phase1_runtime.root_draft_state["submitted_count"] == 0


def test_disabled_checkpoint_without_new_field_remains_compatible(real_agent, tmp_path):
    agent, execute, _ = real_agent(enabled=False)
    complete_initial(agent, execute)
    agent.capture_resume_state()
    cfg = copy.deepcopy(agent.cfg)
    del cfg.parent_selector.include_root_draft
    payload = json.loads(serialize.dumps_json(agent.journal))
    payload.pop("parent_selector_state")
    resumed = AgentSearch("ignored", cfg, serialize.loads_json(json.dumps(payload), Journal))
    runtime = resumed.phase1_runtime
    assert runtime.include_root_draft is False
    assert len(runtime.surrogate.feature_names) == 10
    context = runtime.select_global_parent(resumed)
    assert context.decision.selection_reason == "initial_draft_bypass"
    assert context.option.preview.bypass_selector is True


def test_global_mean_tie_has_no_root_priority(real_agent):
    agent, execute, _ = real_agent(selector="bnvs_mean_greedy")
    first, = complete_initial(agent, execute)
    runtime = agent.phase1_runtime
    runtime._close_current_continuation(agent, first)
    context = runtime.select_global_parent(agent)
    assert [o.parent_node_id for o in context.result.options] == [first.id, agent.virtual_root.id]
    assert context.option.parent_node_id == first.id


def test_parent_closed_after_submission_does_not_drop_terminal_record(real_agent, monkeypatch):
    agent, execute, _ = real_agent(selector="bnvs_mean_greedy")
    first, = complete_initial(agent, execute)
    runtime = agent.phase1_runtime
    original_replace = Path.replace
    def replace(path, target):
        result = original_replace(path, target)
        if path.name == "root_draft_state.tmp":
            state = json.loads(Path(target).read_text())
            if state["submitted_count"] == 2:
                first.is_terminal = True
        return result
    monkeypatch.setattr(Path, "replace", replace)
    agent.step(first, execute)
    records = runtime.transition_store.records()
    assert len(records) == runtime.root_draft_state["submitted_count"] == 2
    assert records[-1].parent_node_id == first.id
    assert runtime.root_draft_state["pending"] == {}


def test_aggregation_remains_a_separate_forced_preview(real_agent):
    agent, execute, _ = real_agent(initial=2, cap=2)
    complete_initial(agent, execute, 2)
    agent.acfg.use_aggregation = True
    agent.acfg.branch_fusion_trigger_prob = 1.0
    agent.scfg.fusion_min_time_hours = 0
    agent.scfg.fusion_min_branches = 2
    agent.scfg.fusion_min_successful_nodes = 1
    agent.stagnation_threshold = 2
    runtime = agent.phase1_runtime
    context = runtime.select_global_parent(agent)
    assert context.option.operator == "aggregation"
    assert context.option.preview.bypass_selector is True
    assert context.option.state_features["operator__draft"] == 0
    decision = json.loads((runtime.artifact_root / "selection_decisions.jsonl").read_text().splitlines()[-1])
    assert decision["decision_kind"] == "forced"
    assert runtime.root_draft_state["draft_submitted"] == 2


def test_terminal_jsonl_is_reconciled_if_ledger_replace_was_interrupted(real_agent, tmp_path, monkeypatch):
    agent, execute, _ = real_agent()
    first, = complete_initial(agent, execute)
    agent.capture_resume_state()
    checkpoint = serialize.dumps_json(agent.journal)
    runtime = agent.phase1_runtime
    original_replace = Path.replace
    interrupted = []
    def replace(path, target):
        if path.name == "root_draft_state.tmp" and not interrupted:
            payload = json.loads(path.read_text())
            if len(payload["observations"]) == 2:
                interrupted.append(True)
                raise OSError("interrupted atomic replacement")
        return original_replace(path, target)
    monkeypatch.setattr(Path, "replace", replace)
    with pytest.raises(OSError):
        agent.step(first, execute)
    assert len(runtime.transition_store.records()) == 2
    cfg = copy.deepcopy(agent.cfg)
    cfg.log_dir = tmp_path / "restart-logs"
    shutil.copytree(runtime.artifact_root, cfg.log_dir / "phase1")
    monkeypatch.setattr("engine.parent_selectors.runtime.os.kill", dead_owner)
    resumed = AgentSearch("ignored", cfg, serialize.loads_json(checkpoint, Journal))
    restored = resumed.phase1_runtime
    assert len(restored.transition_store.records()) == 2
    assert restored.root_draft_state["pending"] == {}
    assert restored.root_draft_state["draft_submitted"] == 2
    recovered = resumed.journal.nodes[-1]
    assert recovered.id == restored.transition_store.records()[-1].child_node_id
    assert recovered.parent is resumed.virtual_root and recovered.metric.value == 0.5
    assert recovered.phase1_transition_id == restored.transition_store.records()[-1].transition_id


@pytest.mark.parametrize("generation_fails", [False, True])
def test_actual_runner_counts_attempts_and_resumes_without_extra_steps(
    real_agent, tmp_path, monkeypatch, generation_fails,
):
    import run as runner
    _, _, services = real_agent()
    if generation_fails:
        services.fail = "generation"
    interpreters = []
    class Interpreter:
        max_parallel_run = 1
        def __init__(self, workspace, *, cfg, **kwargs):
            self.cfg, self.executions = cfg, 0
            interpreters.append(self)
        def run(self, *args):
            self.executions += 1
            return services.execute(self.cfg, *args)
        def cleanup_session(self, session):
            pass
    monkeypatch.setattr(runner, "Interpreter", Interpreter)
    data = tmp_path / "dataset"
    data.mkdir()
    (data / "train.csv").write_text("x,y\n1,0\n2,1\n")
    desc = tmp_path / "task.md"
    desc.write_text("Predict labels; maximize validation accuracy.")
    runs = tmp_path / "runner-runs"
    monkeypatch.setattr(sys, "argv", ["run.py", f"data_dir={data}", f"desc_file={desc}",
        f"log_dir={runs}", f"workspace_dir={runs}", "exp_name=root_draft", "agent.steps=3",
        "agent.initial_drafts=1", "agent.search.num_drafts=3", "agent.seed=0",
        "agent.use_global_memory=false", "coldstart.use_coldstart=false",
        "agent.use_stepwise_generation=false", "agent.use_diff_mode=false",
        "evaluation.enabled=false",
        "agent.use_aggregation=false", "parent_selector.include_root_draft=true"])
    runner.run()
    cfg = interpreters[-1].cfg
    journal = serialize.load_json(cfg.log_dir / "journal.json", Journal)
    state = journal.parent_selector_state
    assert state["submitted_count"] == len(state["observations"]) == 3
    assert state["initial_submitted"] == state["initial_terminal"] == 1
    assert state["draft_submitted"] <= 3
    assert interpreters[-1].executions == (0 if generation_fails else 3)
    monkeypatch.setattr(sys, "argv", ["run.py", f"resume_run_dir={cfg.workspace_dir.parent}"])
    runner.run()
    assert interpreters[-1].executions == 0
    again = serialize.load_json(interpreters[-1].cfg.log_dir / "journal.json", Journal)
    assert again.parent_selector_state["submitted_count"] == 3
