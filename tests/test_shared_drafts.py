from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest

from config import _load_cfg
from engine.agent_search import AgentSearch
from engine.fixed_evaluation import EvaluationError
from engine.executor import ExecutionResult
from engine.search_node import Journal, SearchNode
from engine.shared_drafts import SharedDraftPool, atomic_json, sha256
from tasks.autoresearch.adapter import AutoresearchTask, REPO, candidate_body, assemble_candidate
from utils import serialize
from test_autoresearch import result_for


def make_pool(path, protocol="test", code=None):
    path.mkdir(parents=True, exist_ok=True)
    drafts = []
    for slot in range(1, 6):
        artifact = {"code": code or f"x = {slot}", "plan": f"Plan {slot}", "prompt_input": f"Only previous {slot-1} drafts"}
        target = path / f"draft_{slot}.json"
        atomic_json(target, artifact)
        drafts.append({"slot": slot, "artifact": target.name, "artifact_sha256": sha256(target.read_text()),
                       "code_sha256": sha256(artifact["code"]), "generation_seconds": 10 + slot, "reviewed": True})
    atomic_json(path / "manifest.json", {"version": 1, "state": "ready", "model": "muse-spark-1.3-contributor",
                                        "protocol_id": protocol, "drafts": drafts})
    return path / "manifest.json"


def minimal_agent(path):
    agent = SimpleNamespace(scfg=SimpleNamespace(num_drafts=5), acfg=SimpleNamespace(code=SimpleNamespace(model="muse-spark-1.3-contributor")),
        autoresearch=SimpleNamespace(protocol_id="test"), cfg=SimpleNamespace(log_dir=path), journal_lock=threading.RLock(),
        virtual_root=SearchNode(code="", stage="root"), next_branch_id=1,
        branch_all_nodes={}, branch_successful_nodes={}, _serialize_prompt=lambda x:x)
    return agent


def test_same_order_later_drafts_hidden_and_separate_arm_cursors(tmp_path, monkeypatch):
    path = make_pool(tmp_path / "pool")
    charged = []
    monkeypatch.setattr("engine.shared_drafts.time.sleep", charged.append)
    arms = [minimal_agent(tmp_path / f"arm_{i}") for i in range(4)]
    pools = [SharedDraftPool(path, a) for a in arms]
    for a, p in zip(arms, pools):
        first = [p.take(a, f"d{i}") for i in range(3)]
        assert [n.shared_draft_slot for n in first] == [1, 2, 3]
        assert {n.code for n in a.virtual_root.children} == {"x = 1", "x = 2", "x = 3"}
    fourth = pools[0].take(arms[0], "later4")
    assert fourth.shared_draft_slot == 4 and len(arms[1].virtual_root.children) == 3
    resumed = SharedDraftPool(path, arms[0])
    fifth = resumed.take(arms[0], "later5")
    assert fifth.code == "x = 5" and charged == [11,12,13]*4 + [14,15]
    with pytest.raises(EvaluationError, match="exhausted"):
        resumed.take(arms[0], "illegal6")


def test_concurrent_requests_reserve_distinct_slots_and_tampering_stops(tmp_path, monkeypatch):
    path = make_pool(tmp_path / "pool"); a = minimal_agent(tmp_path / "run")
    p = SharedDraftPool(path, a)
    monkeypatch.setattr("engine.shared_drafts.time.sleep", lambda _:None)
    with ThreadPoolExecutor(3) as ex:
        nodes = list(ex.map(lambda i:p.take(a, str(i)), range(3)))
    assert sorted(n.shared_draft_slot for n in nodes) == [1,2,3]
    assert sorted(a.branch_all_nodes) == [1,2,3]
    (path.parent / "draft_4.json").write_text('{"code":"changed"}')
    with pytest.raises(EvaluationError, match="modified"):
        p.take(a, "four")


@pytest.mark.usefixtures("autoresearch_cache")
@pytest.mark.parametrize("setting", ["gp7_ancestry_ucb", "gp7_long_horizon_ucb", "gp7_long_horizon_ucb_pibo", "mlevolve_original"])
def test_replay_keeps_native_execution_feedback_rating_memory_and_transitions(setting, tmp_path, monkeypatch):
    root = REPO.parents[2]
    cfg = OmegaConf.merge(_load_cfg(use_cli_args=False), OmegaConf.load(root / "tasks/autoresearch/config.yaml"),
                         OmegaConf.load(root / f"config/experiments/{setting}.yaml"))
    cfg.workspace_dir = tmp_path / "workspace"; cfg.log_dir = tmp_path / "logs"
    cfg.workspace_dir.mkdir(); cfg.cpu_number = 4
    saved = []
    class Memory:
        def __init__(self, **kwargs): pass
        def save_node(self, node, parent): saved.append(node.id)
    monkeypatch.setattr("agents.memory.global_memory.GlobalMemoryLayer", Memory)
    task = AutoresearchTask(cfg)
    path = make_pool(tmp_path / "pool", task.protocol_id, candidate_body((REPO / "train.py").read_text()))
    cfg.shared_draft_pool = str(path)
    a = AgentSearch((REPO.parent / "TASK.md").read_text(), cfg, Journal())
    monkeypatch.setattr("engine.shared_drafts.time.sleep", lambda _:None)
    monkeypatch.setattr("agents.draft_agent.run", lambda *a,**k:pytest.fail("Do not regenerate shared code"))
    monkeypatch.setattr("agents.code_review_agent.run", lambda *a,**k:pytest.fail("Do not re-review and change shared code"))
    feedback = []
    def summarize(**kwargs):
        feedback.append(True)
        return {"is_bug":False,"summary":"Official trial completed","code_summary":"Causal GPT","potential_level":2}
    monkeypatch.setattr("agents.result_parse_agent.query", summarize)
    node = a.step(None, lambda *a,**k:pytest.fail("Deferred"), execute_immediately=False)
    assert node.shared_draft_slot == 1 and len(a.virtual_root.children) == 1
    executed = []
    def execute(code, id, *args, **kwargs):
        executed.append(id)
        trial = cfg.workspace_dir / "trials" / id; trial.mkdir(parents=True)
        (trial / "result.json").write_text(json.dumps(result_for(assemble_candidate(code))))
        return ExecutionResult(["Real host result"], 800, None)
    a.execute_deferred_node(node, execute)
    assert node.metric.value == 1.7 and node.is_valid
    assert executed == [node.id] and saved == [node.id] and feedback == [True]
    assert len(a.phase1_runtime.transition_store.records()) == 1
    assert node.agent_potential_level == (2 if setting.endswith("pibo") else None)
    restored = serialize.loads_json(serialize.dumps_json(a.journal), Journal)
    assert restored.nodes[-1].shared_draft_slot == 1
    assert restored.nodes[-1].shared_draft_pool_id == a.shared_drafts.pool_id
