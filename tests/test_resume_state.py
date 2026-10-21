import json
import random
import threading
import time
from types import SimpleNamespace

import numpy as np

import engine.agent_search as agent_search_module
from config import _load_cfg
from engine.agent_search import AgentSearch
from engine.search_node import Journal, SearchNode
from utils import serialize
from utils.metric import MetricValue, WorstMetricValue
from utils.seed import restore_random_state


def test_restore_from_journal_rebuilds_transient_indexes(tmp_path):
    root = SearchNode(
        id="root",
        code="",
        plan="root",
        stage="root",
        metric=WorstMetricValue(),
    )
    draft = SearchNode(
        id="draft",
        code="solution",
        plan="draft",
        stage="draft",
        parent=root,
        metric=MetricValue(value=0.4, maximize=True),
        is_buggy=False,
        is_valid=True,
        branch_id=3,
        continue_improve=True,
        lock=True,
        expected_child_count=4,
    )
    child = SearchNode(
        id="child",
        code="better",
        plan="improve",
        stage="improve",
        parent=draft,
        metric=MetricValue(value=0.6, maximize=True),
        is_buggy=False,
        is_valid=True,
        branch_id=3,
        local_best_node=draft,
        lock=True,
        expected_child_count=2,
    )
    child.local_best_node = child
    journal = Journal()
    for node in (root, draft, child):
        journal.append(node)

    best_dir = tmp_path / "best_solution"
    best_dir.mkdir()
    (best_dir / "node_id.txt").write_text("draft")

    agent = AgentSearch.__new__(AgentSearch)
    agent.journal = journal
    agent.cfg = SimpleNamespace(workspace_dir=tmp_path)
    agent.top_k = 10
    agent.metric_maximize = True
    agent.max_fusion_drafts = 2
    agent.best_node = None

    before = time.time()
    agent._restore_from_journal(1234)

    assert agent.current_step == 3
    assert before - agent.search_start_time >= 1233
    assert agent.next_branch_id == 4
    assert agent.branch_all_nodes == {3: [draft, child]}
    assert agent.branch_successful_nodes == {3: [draft, child]}
    assert agent.branch_node_count == {3: 2}
    assert agent.current_node_list == [draft]
    assert agent.best_node is child
    assert {node.id for node in agent.top_candidates} == {"draft", "child"}
    assert draft.expected_child_count == 1
    assert child.expected_child_count == 0
    assert draft.lock is False
    assert child.lock is False
    assert isinstance(draft.child_count_lock, type(threading.Lock()))


def test_resume_metadata_round_trip_restores_elapsed_time_and_rng():
    random.seed(17)
    np.random.seed(23)
    agent = AgentSearch.__new__(AgentSearch)
    agent.journal = Journal()
    agent.start_time = time.time() - 120
    agent.cfg = SimpleNamespace()

    agent.capture_resume_state()
    restored = serialize.loads_json(serialize.dumps_json(agent.journal), Journal)
    expected_python = random.random()
    expected_numpy = np.random.random()

    random.seed(99)
    np.random.seed(99)
    restore_random_state(
        restored.python_random_state,
        restored.numpy_random_state,
    )

    assert restored.elapsed_active_seconds >= 119
    assert random.random() == expected_python
    assert np.random.random() == expected_numpy


def test_legacy_journal_without_resume_metadata_still_loads():
    root = SearchNode(
        id="root",
        code="",
        plan="root",
        stage="root",
        metric=WorstMetricValue(),
    )
    journal = Journal()
    journal.append(root)
    payload = json.loads(serialize.dumps_json(journal))
    payload.pop("elapsed_active_seconds")
    payload.pop("python_random_state")
    payload.pop("numpy_random_state")
    payload.pop("task_desc")

    restored = serialize.loads_json(json.dumps(payload), Journal)

    assert restored.elapsed_active_seconds == 0.0
    assert restored.python_random_state is None
    assert restored.numpy_random_state is None
    assert restored.task_desc is None


def test_first_resumed_step_refreshes_preview_without_resetting_clock():
    root = SearchNode(
        id="root",
        code="",
        plan="root",
        stage="root",
        metric=WorstMetricValue(),
    )
    journal = Journal(nodes=[root])
    agent = AgentSearch.__new__(AgentSearch)
    agent.journal = journal
    agent.virtual_root = root
    agent.data_preview = None
    agent.search_start_time = time.time() - 600
    agent.journal_lock = threading.RLock()
    agent.branch_all_nodes = {}
    agent.best_node = None
    agent.update_data_preview = lambda: setattr(agent, "data_preview", "restored")
    context = SimpleNamespace(option=SimpleNamespace(parent_node=root))
    agent.phase1_runtime = SimpleNamespace(
        select_global_parent=lambda unused_agent: context,
        release_parent=lambda unused_context: None,
    )
    agent._run_single_step = lambda *args, **kwargs: (True, None)
    restored_start = agent.search_start_time

    result = agent.step(node=root, exec_callback=lambda *args: None)

    assert result is root
    assert agent.data_preview == "restored"
    assert agent.search_start_time == restored_start


def test_agent_resume_reuses_saved_clean_task_description(tmp_path, monkeypatch):
    root = SearchNode(
        id="root",
        code="",
        plan="root",
        stage="root",
        metric=WorstMetricValue(),
    )
    draft = SearchNode(
        id="draft",
        code="solution",
        plan="draft",
        stage="draft",
        parent=root,
        metric=MetricValue(value=0.4, maximize=True),
        is_buggy=False,
        is_valid=True,
        branch_id=1,
    )
    journal = Journal(task_desc="saved cleaned task")
    journal.append(root)
    journal.append(draft)

    cfg = _load_cfg(use_cli_args=False)
    cfg.log_dir = tmp_path / "logs"
    cfg.workspace_dir = tmp_path / "workspace"
    cfg.workspace_dir.mkdir()
    cfg.agent.use_global_memory = False
    cfg.coldstart.use_coldstart = False

    def fail_if_called(*args, **kwargs):
        raise AssertionError("resume must not clean the task description again")

    monkeypatch.setattr(
        agent_search_module,
        "clean_task_desc",
        fail_if_called,
    )

    agent = AgentSearch(task_desc="new raw task", cfg=cfg, journal=journal)

    assert agent.task_desc == "saved cleaned task"
    assert agent.current_step == 2


def test_new_agent_initialization_keeps_original_root_flow(tmp_path, monkeypatch):
    cfg = _load_cfg(use_cli_args=False)
    cfg.log_dir = tmp_path / "logs"
    cfg.workspace_dir = tmp_path / "workspace"
    cfg.workspace_dir.mkdir()
    cfg.agent.use_global_memory = False
    cfg.coldstart.use_coldstart = False
    monkeypatch.setattr(
        agent_search_module,
        "clean_task_desc",
        lambda task_desc, unused_cfg: f"cleaned: {task_desc}",
    )

    def set_metric_direction(agent):
        agent.metric_maximize = True
        agent.metric_maximize_reasoning = "test"

    monkeypatch.setattr(
        agent_search_module.result_parse_agent,
        "determine_metric_direction",
        set_metric_direction,
    )

    journal = Journal()
    agent = AgentSearch(task_desc="new task", cfg=cfg, journal=journal)

    assert len(journal) == 1
    assert agent.virtual_root is journal[0]
    assert agent.virtual_root.stage == "root"
    assert journal.task_desc == "cleaned: new task"
