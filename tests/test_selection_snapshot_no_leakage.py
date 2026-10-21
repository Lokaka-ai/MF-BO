import json

from engine.parent_selectors.runtime import Phase1SelectorRuntime
from engine.search_node import SearchNode
from utils.metric import MetricValue
from conftest import add_valid_node, make_agent, make_runtime_config


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_snapshot_is_written_before_future_child_outcome_and_does_not_change(tmp_path):
    runtime = Phase1SelectorRuntime(make_runtime_config(tmp_path, "random_parent"))
    agent = make_agent()
    parent = add_valid_node(agent, node_id="parent", metric=0.6)

    context = runtime.record_direct_parent(agent, parent)
    snapshot_path = tmp_path / "phase1" / "selection_snapshots.jsonl"
    before = _read_jsonl(snapshot_path)
    assert len(before) == 1
    assert before[0]["decision_id"] == context.decision.decision_id
    assert before[0]["parent_metric"] == 0.6

    child = SearchNode(
        id="future-child",
        parent=parent,
        plan="future",
        code="future",
        metric=MetricValue(0.99, maximize=True),
        stage="improve",
        is_buggy=False,
        is_valid=True,
        branch_id=1,
        local_best_node=parent,
    )
    child.metric = MetricValue(0.1, maximize=True)

    after = _read_jsonl(snapshot_path)
    assert after == before
    assert "future-child" not in json.dumps(after)


def test_ranked_decision_logs_every_current_option(tmp_path):
    runtime = Phase1SelectorRuntime(make_runtime_config(tmp_path, "random_parent"))
    agent = make_agent()
    add_valid_node(agent, node_id="draft-a", metric=0.7, branch_id=1)
    add_valid_node(agent, node_id="draft-b", metric=0.8, branch_id=2)

    context = runtime.select_global_parent(agent)
    snapshots = _read_jsonl(tmp_path / "phase1" / "selection_snapshots.jsonl")
    assert {item["parent_node_id"] for item in snapshots} == {"draft-a", "draft-b"}
    assert sum(item["selected"] for item in snapshots) == 1
    assert all(item["decision_id"] == context.decision.decision_id for item in snapshots)
    assert all("feature_vector" in item for item in snapshots)
    assert all(item["acquisition_beta"] == 1.5 for item in snapshots)
    assert all(item["acquisition_beta_schedule"] == "fixed" for item in snapshots)
    assert all(item["selector_decision_wall_time_sec"] >= 0 for item in snapshots)


def test_configured_feature_ablations_zero_only_the_requested_groups(tmp_path):
    cfg = make_runtime_config(tmp_path, "random_parent")
    cfg.bo_parent_selector.features.include_mcgs_statistics = False
    cfg.bo_parent_selector.features.include_branch_history = False
    cfg.bo_parent_selector.features.include_reference_summary = False
    runtime = Phase1SelectorRuntime(cfg)
    agent = make_agent()
    parent = add_valid_node(agent, node_id="parent", metric=0.6)
    parent.visits = 4
    parent.total_reward = 3.0

    context = runtime.record_direct_parent(agent, parent)
    features = context.option.state_features
    assert features["parent_utility"] == 0.6
    assert features["visit_count"] == 0.0
    assert features["mcgs_q"] == 0.0
    assert features["branch_age"] == 0.0
    assert features["recent_valid_rate"] == 0.0
    assert features["reference_count"] == 0.0


def test_remaining_step_feature_uses_completed_journal_expansions(tmp_path):
    runtime = Phase1SelectorRuntime(make_runtime_config(tmp_path, "random_parent"))
    agent = make_agent()
    parent = add_valid_node(agent, node_id="parent", metric=0.6)
    agent.current_step = 999

    context = runtime.record_direct_parent(agent, parent)

    assert context.option.state_features["remaining_steps_fraction"] == 49 / 50
