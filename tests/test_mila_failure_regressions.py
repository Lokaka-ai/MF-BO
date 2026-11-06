"""Small reproductions of failures observed in Mila jobs 11417260-11417262."""

from types import SimpleNamespace
import time
import pytest

from engine.search_node import Journal, SearchNode
from engine.parent_selectors.base import ExpansionOption, OperatorPreview
from engine.mf_gp.controller import MFSelector
from utils import serialize
from utils.metric import MetricValue
from test_multifidelity_integration import aq05_cfg
from test_fixed_evaluation import fixed_cfg, model_responses, setup_server


def test_initial_draft_memory_accepts_completed_execution_awaiting_score():
    root = SearchNode(code="", stage="root")
    child = SearchNode(code="pass", stage="draft", parent=root, is_buggy=False)
    memory = root.fetch_child_memory()
    assert "pending" in memory.lower()
    assert "0 successful" not in memory
    child.metric = MetricValue(0.8, maximize=True)
    assert "1 successful" in root.fetch_child_memory()


@pytest.mark.parametrize("child_id", [None, "generated-but-unscored"])
def test_failed_generation_with_null_snapshot_is_an_invalid_mf_observation(child_id):
    selector = MFSelector(SimpleNamespace(), lambda: [], None)
    record = SimpleNamespace(
        transition_id="failed", metadata={"node_snapshot": None},
        child_node_id=child_id, selection_features={"mf": {"fidelity": 0.5}},
        generation_wall_time_sec=2.0, execution_wall_time_sec=0.0,
    )
    selector.observe([record])
    assert selector.observations[0]["kind"] == "generation_failure"
    assert not selector.gp.rows


def test_journal_snapshot_excludes_live_decision_graph():
    root = SearchNode(code="", stage="root")
    child = SearchNode(code="pass", stage="draft", parent=root,
                       metric=MetricValue(0.8, maximize=True), is_buggy=False)
    root.local_best_node = child
    # This is a live, non-persisted context. The previous whole-graph deepcopy
    # traversed it before dropping parent/child links and hit recursion limits.
    child._phase1_decision_context = [None]
    cursor = child._phase1_decision_context
    for _ in range(1200):
        next_item = [None]
        cursor[0] = next_item
        cursor = next_item
    journal = Journal(nodes=[root, child])
    restored = serialize.loads_json(serialize.dumps_json(journal), Journal)
    assert restored.nodes[1].parent is restored.nodes[0]
    assert restored.nodes[0].local_best_node is restored.nodes[1]
    assert restored.nodes[1] in restored.nodes[0].children
    assert restored.nodes[1].metric.value == 0.8
    from engine.search_node import filter_on_path
    filtered = filter_on_path(journal, [root.id, child.id])
    assert len(filtered.nodes) == 2


def test_mf_knowledge_gradient_targets_full_quality(monkeypatch):
    from engine.mf_gp import controller

    captured = {}

    def acquire(gp, targets, actions, incumbent, pending, terminal):
        captured["targets"] = targets
        captured["actions"] = actions
        return [1.0] * len(actions), {}

    monkeypatch.setattr(controller, "acquisition", acquire)
    logger = SimpleNamespace(_append=lambda *_: None)
    selector = MFSelector(SimpleNamespace(), lambda: [], logger)
    selector.calibration = []
    runtime = SimpleNamespace(root_draft_state={"pending": {}, "revision": 1})
    agent = SimpleNamespace(phase1_runtime=runtime,
                            acfg=SimpleNamespace(time_limit=10000, seed=42),
                            start_time=time.time())
    option = ExpansionOption(
        "parent", "improve", (), {"study7": {"x": [0.0] * 8}},
        SimpleNamespace(), OperatorPreview("improve")
    )
    selector.select(agent, [option])
    assert {a["row"]["fidelity"] for a in captured["actions"]} == {0.5, 1.0}
    assert len(captured["targets"]) == 1
    assert captured["targets"][0]["fidelity"] == 1.0


def test_topk_extension_ranks_common_full_quality(monkeypatch):
    from engine import node_selection
    from engine.mf_gp import evidence

    cheap = SearchNode(code="a", stage="draft", branch_id=1,
                       metric=MetricValue(0.99, maximize=True), is_buggy=False)
    full = SearchNode(code="b", stage="draft", branch_id=2,
                      metric=MetricValue(0.80, maximize=True), is_buggy=False)
    monkeypatch.setattr(evidence, "quality", lambda agent, n: {cheap.id: 0.7,
                                                               full.id: 0.8}[n.id])
    agent = SimpleNamespace(
        cfg=SimpleNamespace(evaluation=SimpleNamespace(multifidelity_mode="mf_gp")),
        branch_all_nodes={1: [cheap], 2: [full]}, metric_maximize=True,
    )
    ranked = node_selection.get_top_k_nodes_global(agent, 2, 1)
    assert [r["node"] for r in ranked] == [full, cheap]


def test_checkpoint_failure_stops_before_candidate_generation(aq05_cfg, monkeypatch):
    from engine.agent_search import AgentSearch
    from engine.parent_selectors.runtime import SearchStatePersistenceError

    aq05_cfg.evaluation.multifidelity_mode = "reset"
    agent = AgentSearch("Evaluation: binary log loss.", aq05_cfg, Journal())

    def fail_snapshot(_):
        raise RecursionError("simulated checkpoint graph recursion")

    monkeypatch.setattr(serialize, "dumps_json", fail_snapshot)
    with pytest.raises(SearchStatePersistenceError, match="checkpoint failed"):
        agent.step(None, None, execute_immediately=False)
    assert len(agent.journal.nodes) == 1
    assert agent.phase1_runtime.root_draft_state["decision_count"] == 1


def test_mf_archival_best_uses_full_observation_not_cheap_raw_score():
    from engine.multifidelity import completed_full_winner
    from engine.search_node import filter_journal

    root = SearchNode(code="", stage="root")
    cheap = SearchNode(code="cheap", stage="draft", parent=root,
                       metric=MetricValue(0.99, maximize=True), is_buggy=False)
    full = SearchNode(code="full", stage="draft", parent=root,
                      metric=MetricValue(0.75, maximize=True), is_buggy=False)
    journal = Journal(
        nodes=[root, cheap, full],
        evaluation_state={"multifidelity_mode": "mf_gp"},
        parent_selector_state={"study7_state": {"observations": [
            {"program_id": cheap.id, "fidelity": 0.5, "valid": True,
             "utility": 0.99},
            {"program_id": full.id, "fidelity": 1.0, "valid": True,
             "utility": 0.75},
        ]}},
    )
    assert completed_full_winner(journal) is full
    assert filter_journal(journal).nodes[-1].id == full.id


def test_staged_archival_best_uses_only_full_valid_node():
    from engine.multifidelity import completed_full_winner
    from engine.search_node import filter_journal

    root = SearchNode(code="", stage="root")
    cheap = SearchNode(code="cheap", stage="draft", parent=root,
                       metric=MetricValue(0.99, maximize=True),
                       is_buggy=False, is_valid=True, training_fraction=0.5)
    full = SearchNode(code="full", stage="draft", parent=root,
                      metric=MetricValue(0.75, maximize=True),
                      is_buggy=False, is_valid=True, training_fraction=1.0,
                      evaluation_protocol_id="protocol",
                      evaluation_result={"valid": True, "metric": 0.75})
    journal = Journal(nodes=[root, cheap, full], evaluation_state={
        "multifidelity_mode": "reset", "protocol_id": "protocol"
    })
    assert completed_full_winner(journal) is full
    assert filter_journal(journal).nodes[-1].id == full.id


def test_official_grader_failure_keeps_actual_error(tmp_path, monkeypatch):
    from tasks.mlebench_lite import mila_multifidelity as campaign

    monkeypatch.setattr(campaign.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=137, stdout="",
                                        stderr="killed by cgroup memory limit"))
    with pytest.raises(RuntimeError, match="Official grading exited 137"):
        campaign.official_grade(tmp_path / "logs", tmp_path / "data", {}, tmp_path)
    diagnostic = campaign.read(tmp_path / "official-grade-process.json")
    assert diagnostic["returncode"] == 137
    assert "cgroup memory" in diagnostic["stderr"]


def test_unresolved_decisions_cannot_be_reported_as_complete():
    from tasks.mlebench_lite.mila_multifidelity import validate_terminal_ledger

    journal = {"parent_selector_state": {"pending": {"ticket": {}}}}
    with pytest.raises(RuntimeError, match="unresolved decisions"):
        validate_terminal_ledger(journal, "reset")
    journal["parent_selector_state"]["pending"] = {}
    journal["parent_selector_state"]["study7_state"] = {
        "pending_promotions": {"promotion": {}}
    }
    with pytest.raises(RuntimeError, match="unresolved promotions"):
        validate_terminal_ledger(journal, "mf_gp")
    journal["parent_selector_state"]["study7_state"]["pending_promotions"] = {}
    validate_terminal_ledger(journal, "mf_gp")


@pytest.mark.parametrize("mode", ["carry_over", "reset"])
def test_staged_half_execution_stops_at_active_time_switch(mode):
    from engine.multifidelity import cheap_execution_seconds_left

    node = SimpleNamespace(training_fraction=0.5)
    assert cheap_execution_seconds_left(mode, "half", node, 100, 43200, 200) == 17180
    assert cheap_execution_seconds_left(mode, "half", node, 100, 43200, 20000) == 0
    assert cheap_execution_seconds_left(mode, "full", node, 100, 43200, 200) is None
    node.training_fraction = 1.0
    assert cheap_execution_seconds_left(mode, "full", node, 100, 43200, 200) is None


def test_mfgp_only_initial_cheap_executions_have_switch_deadline():
    from engine.multifidelity import cheap_execution_seconds_left

    node = SimpleNamespace(training_fraction=0.5,
                           phase1_selection_features={"mf": {"initial": True}})
    assert cheap_execution_seconds_left("mf_gp", "adaptive", node, 100, 43200, 200) == 17180
    node.phase1_selection_features["mf"]["initial"] = False
    assert cheap_execution_seconds_left("mf_gp", "adaptive", node, 100, 43200, 200) is None
    node.training_fraction = 1.0
    assert cheap_execution_seconds_left("mf_gp", "adaptive", node, 100, 43200, 200) is None


@pytest.mark.parametrize("mode,phase,features", [
    ("reset", "half", {}),
    ("mf_gp", "adaptive", {"mf": {"initial": True}}),
])
def test_execution_callback_receives_capped_cheap_context(monkeypatch, mode, phase, features):
    from engine.agent_search import AgentSearch

    monkeypatch.setattr("engine.agent_search.time.time", lambda: 200)
    evaluator = SimpleNamespace(multifidelity_mode=mode, multifidelity_phase=phase,
                                begin_attempt=lambda node, remaining: {"timeout": 32400,
                                    "output_dir": "unused", "protocol_id": "fixed"},
                                executable=lambda code: code)
    agent = AgentSearch.__new__(AgentSearch)
    agent.fixed_evaluation = evaluator
    agent.start_time = 100
    agent.acfg = SimpleNamespace(time_limit=43200)
    node = SearchNode(code="pass", stage="draft", training_fraction=0.5,
                      phase1_selection_features=features)
    received = {}

    def callback(*args, **kwargs):
        received.update(kwargs)
        return "executed"

    assert agent._execute_candidate(node, callback) == "executed"
    assert received["evaluation_context"]["timeout"] == 17180
    evaluator.multifidelity_phase = "full"
    node.training_fraction = 1.0
    received.clear()
    agent._execute_candidate(node, callback)
    assert received["evaluation_context"]["timeout"] == 32400


def test_mila_memory_guard_rejects_partial_submission_before_sbatch():
    from tasks.mlebench_lite.mila_multifidelity import validate_gpu_partition

    for partition in ("main", "main-grace", "unkillable", "short-unkillable", "long-cpu"):
        with pytest.raises(ValueError, match="requires a long partition"):
            validate_gpu_partition(partition, {"memory_gb": 64})
    validate_gpu_partition("long", {"memory_gb": 64})
    validate_gpu_partition("main", {"memory_gb": 24})
