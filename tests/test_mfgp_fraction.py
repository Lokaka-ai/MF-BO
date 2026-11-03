"""Fraction identity and cost accounting; real CPU execution with synthetic LLM."""
import copy
import csv
import json
import random
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from engine.multifidelity import prepare_view, cheap_execution_seconds_left
from engine.mf_gp.settings import validate_low_fraction
from engine.mf_gp.model import MultiFidelityGP, CONTRACT
from engine.mf_gp.controller import PromotionRequested
from engine.mf_gp.execution import promote, finalize
from engine.mf_gp.recovery import execute
from engine.executor import Interpreter, ExecutionResult
from test_fixed_evaluation import fixed_cfg, setup_server, model_responses
from test_multifidelity_integration import aq05_cfg
from test_mf_gp import make_agent


def test_nested_views_and_legacy_identity(tmp_path):
    protocol = tmp_path / "protocol"
    public = protocol / "public"
    public.mkdir(parents=True)
    ids = [str(i) for i in range(101)]
    (public / "train.csv").write_text("sample_id,label\n" + "".join(f"{i},{int(i)%2}\n" for i in ids))
    for name in ("validation.csv", "test.csv", "sample_submission.csv"):
        (public / name).write_text("sample_id\nv\n")
    selected = {}
    for fraction in (0.1, 0.2, 0.5):
        work = tmp_path / str(fraction)
        view, manifest = prepare_view(protocol, work, 42, fraction)
        with (view / "train.csv").open() as stream:
            selected[fraction] = {r["sample_id"] for r in csv.DictReader(stream)}
        assert len(selected[fraction]) == int(len(ids) * fraction)
        assert (view / "validation.csv").resolve() == public / "validation.csv"
        assert prepare_view(protocol, work, 42, fraction)[1] == manifest
        with pytest.raises(ValueError, match="fraction|sample"):
            prepare_view(protocol, work, 42, 0.2 if fraction != 0.2 else 0.5)
    assert selected[0.1] < selected[0.2] < selected[0.5]
    assert selected[0.5] == set(random.Random(42).sample(ids, len(ids)//2))


@pytest.mark.parametrize("fraction", [0.1, 0.2, 0.5])
def test_gp_conditions_and_rejects_cross_fraction_resume(fraction):
    gp = MultiFidelityGP(fraction)
    low = dict(program_id="p", fidelity=fraction, x=[0.] * 8, operator="draft")
    high = {**low, "fidelity": 1.0}
    _, prior = gp.posterior([high])
    gp.fit([{**low, "valid": True, "utility": 0.8}])
    mean, cov = gp.posterior([high])
    assert mean[0] > 0 and cov[0, 0] < prior[0, 0]
    assert np.linalg.eigvalsh(gp.covariance([low, high], [low, high])).min() > 0
    other = MultiFidelityGP(fraction)
    other.load_state_dict(gp.state_dict())
    other.fit([{**low, "valid": True, "utility": 0.8}])
    np.testing.assert_array_equal(other.posterior([high])[0], mean)
    wrong = MultiFidelityGP(0.2 if fraction != 0.2 else 0.5)
    with pytest.raises(ValueError, match="contract"):
        wrong.load_state_dict(gp.state_dict())
    with pytest.raises(ValueError, match="observation"):
        wrong.fit([{**low, "valid": True, "utility": 0.8}])
    if fraction == 0.5:
        assert gp.state_dict()["contract"] == CONTRACT


@pytest.mark.parametrize("value", [0., 1., 0.3, True, float("nan")])
def test_reject_unsupported_fraction(value):
    with pytest.raises(ValueError):
        validate_low_fraction(value)


def test_initial_deadline_applies_at_twenty_percent():
    n = NS(training_fraction=0.2, phase1_selection_features={"mf": {"initial": True}})
    assert cheap_execution_seconds_left("mf_gp", "adaptive", n, 100, 1000, 450) == 50
    n.phase1_selection_features["mf"]["initial"] = False
    assert cheap_execution_seconds_left("mf_gp", "adaptive", n, 100, 1000, 450) is None


def test_twenty_percent_execution_promotion_and_resume(aq05_cfg, model_responses):
    aq05_cfg.evaluation.mf_gp_low_fraction = 0.2
    agent = make_agent(aq05_cfg)
    interpreter = Interpreter(aq05_cfg.workspace_dir, cfg=aq05_cfg, timeout=30)
    callback = lambda code, node_id, *a, **kw: execute(interpreter, aq05_cfg, code, node_id, *a, **kw)
    nodes = [agent.step(None, callback, execute_immediately=False) for _ in range(3)]
    for n in nodes:
        agent.execute_deferred_node(n, callback)
        assert n.training_fraction == 0.2 and n.is_valid, n.analysis
    selector = agent.phase1_runtime.study7_selector
    assert {r["fidelity"] for r in selector.gp.rows} == {0.2}
    before = len(agent.journal.nodes)
    with pytest.raises(PromotionRequested) as request:
        agent.step(None, callback)
    promote(agent, request.value.ticket, callback)
    assert len(agent.journal.nodes) == before
    agent.capture_resume_state()
    assert finalize(aq05_cfg, agent.journal)["fidelity"] == 1.0
    for path in (aq05_cfg.log_dir / "mf_execution").glob("*.json"):
        record = json.loads(path.read_text())
        assert record["fidelity"] in (0.2, 1.0)
        assert record["attempts"][0]["attempt_id"]
        assert sum(x["elapsed_seconds"] for x in record["attempts"]) == record["elapsed_seconds"]
    from engine.agent_search import AgentSearch
    from engine.search_node import Journal
    from utils import serialize
    saved = serialize.loads_json(serialize.dumps_json(agent.journal), Journal)
    restored = AgentSearch("ignored", copy.deepcopy(aq05_cfg), saved)
    assert restored.phase1_runtime.study7_selector.gp.training_hash == selector.gp.training_hash
    wrong = copy.deepcopy(aq05_cfg)
    wrong.evaluation.mf_gp_low_fraction = 0.5
    with pytest.raises((ValueError, RuntimeError), match="fidelity|fraction|sample|contract"):
        AgentSearch("ignored", wrong, serialize.loads_json(serialize.dumps_json(agent.journal), Journal))


def test_timeout_cost_is_retained_but_not_completed_runtime(tmp_path):
    cfg = NS(log_dir=tmp_path, exec=NS(timeout=50),
             evaluation=NS(multifidelity_mode="mf_gp", mf_gp_low_fraction=0.2))
    interpreter = NS(run=lambda *a, **kw: ExecutionResult([], 12., "TimeoutError", {}, []))
    execute(interpreter, cfg, "pass", "p", evaluation_context={"training_fraction": 0.2})
    record = json.loads((tmp_path / "mf_execution/p__0.2.json").read_text())
    assert record["elapsed_seconds"] == 12
    assert record["attempts"][0]["runtime_lower_bound"]
    assert not record["attempts"][0]["completed_runtime_observation"]


def test_new_launcher_preserves_legacy_specs(tmp_path):
    from tasks.mlebench_lite.mila_multifidelity import search_command, ROOT
    old = json.loads((ROOT / "config/mlebench_mfgp14.json").read_text())
    new = json.loads((ROOT / "config/mlebench_mfgp20.json").read_text())
    assert old["training_fraction"] == 0.5 and new["training_fraction"] == 0.2
    assert len(old["tasks"]) == 14
    assert set(new["tasks"]) == {"jigsaw-toxic-comment-classification-challenge", "siim-isic-melanoma-classification", "dog-breed-identification"}
    task = "jigsaw-toxic-comment-classification-challenge"
    command = search_command({"spec": new, "data_dir": "/data"}, tmp_path, task, "mf_gp")
    assert "evaluation.mf_gp_low_fraction=0.2" in command


def test_cost_export_and_interrupted_reconciliation(tmp_path):
    from tasks.mlebench_lite.mila_multifidelity import reconcile_recovery, save
    from tools.mfgp_cost_matrix import rows
    logs = tmp_path / "logs"
    save(logs / "phase1/root_draft_state.json", {"owner_host": "test", "owner_pid": 10})
    save(logs / "mf_execution/p__0.2.json", dict(
        program_id="p", fidelity=0.2, code_sha256="abc", state="running",
        started_epoch=100., elapsed_seconds=5., unsegmented_elapsed_seconds=5.,
        attempts=[dict(segment_id="s", state="running", elapsed_seconds=None)]))
    recovery = dict(run_dir=str(tmp_path), source_job="123", ended_epoch=120.)
    reconcile_recovery(tmp_path, recovery)
    reconcile_recovery(tmp_path, recovery)  # Release reconciliation is idempotent.
    exported = list(rows(logs))
    assert len(exported) == 1
    assert exported[0]["state"] == "interrupted"
    assert exported[0]["execution_seconds"] == 20.
    assert exported[0]["unsegmented_prior_seconds"] == 5.
    assert exported[0]["runtime_lower_bound"] is True
    assert exported[0]["completed_runtime_observation"] is False
    save(logs / "mf_execution/legacy.json", dict(code_sha256="old", state="completed", elapsed_seconds=3.))
    assert len(list(rows(logs))) == 2
