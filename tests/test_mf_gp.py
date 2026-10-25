"""MF model math and actual fixed-evaluator/selector integration (synthetic LLM)."""

import copy
from pathlib import Path
import numpy as np
import pytest
from engine.mf_gp.model import MultiFidelityGP, acquisition


def row(p="p", f=0.5, x=0.0):
    return dict(program_id=p, fidelity=f, x=[x] * 8, operator="improve")


def observation(p="p", f=0.5, y=0.0):
    return dict(**row(p, f), valid=True, utility=y)


def test_ar_covariance_identity_and_psd():
    gp = MultiFidelityGP()
    rows = [row("a", 0.5), row("a", 1.0), row("b", 0.5), row("b", 1.0)]
    k = gp.covariance(rows, rows)
    assert np.linalg.eigvalsh(k).min() > 0
    assert k[0, 1] == pytest.approx(0.8 * k[0, 0])
    assert k[1, 1] == pytest.approx((0.8**2 + 0.7**2) * k[0, 0])
    assert k[0, 1] > k[0, 3]  # exact-code pair != two stochastic siblings


def test_cheap_conditions_high_without_pseudo_observations():
    gp = MultiFidelityGP()
    prior, var = gp.posterior([row(f=1.0)])
    gp.fit([observation(y=1.0)])
    mean, cov = gp.posterior([row(f=1.0)])
    assert mean[0] > prior[0] and cov[0, 0] < var[0, 0]
    assert cov[0, 0] > 0.3  # irreducible discrepancy remains
    assert len(gp.rows) == 1
    assert gp.rows[0]["fidelity"] == 0.5


def test_bad_and_censored_scores_never_train():
    gp = MultiFidelityGP()
    gp.fit(
        [
            observation(),
            {**observation("b", 1.0, 100), "budget_eligible": False},
            {**observation("c", 1.0, 100), "valid": False},
        ]
    )
    assert len(gp.rows) == 1


def test_pending_reduces_information_without_admitting_scores():
    gp = MultiFidelityGP()
    _, v = gp.posterior([row(f=1.0)])
    _, pending_v = gp.posterior([row(f=1.0)], [row()])
    assert pending_v[0, 0] < v[0, 0]
    assert not gp.rows


def test_kg_cost_and_terminal_ei():
    gp = MultiFidelityGP()
    actions = [dict(row=row(), cost=1.0), dict(row=row(), cost=2.0)]
    scores, _ = acquisition(gp, [row(f=1.0)], actions, 0.0)
    assert scores[0] > 0 and scores[0] == pytest.approx(2 * scores[1])
    with pytest.raises(ValueError, match="full fidelity"):
        acquisition(gp, [row(f=1.0)], actions, 0.0, terminal=True)
    scores, _ = acquisition(
        gp, [row(f=1.0)], [dict(row=row(f=1.0), cost=1.0)], 0.0, terminal=True
    )
    assert scores[0] > 0


def test_model_restore_same_posterior():
    observations = [
        observation(str(i), f, 0.1 * i + (0.2 if f == 1.0 else 0.0))
        for i in range(4)
        for f in (0.5, 1.0)
    ]
    gp = MultiFidelityGP()
    gp.fit(observations, [0.5])
    other = MultiFidelityGP()
    other.load_state_dict(gp.state_dict())
    other.fit(observations)
    assert gp.fit_diagnostics["status"] == "bounded_map"
    np.testing.assert_array_equal(gp.theta, other.theta)
    for a, b in zip(gp.posterior([row("b", 1.0)]), other.posterior([row("b", 1.0)])):
        np.testing.assert_array_equal(a, b)


# Real execution, CSV scoring, serialization and expansion path. Only external
# model replies are substituted by the existing integration fixture.
from test_fixed_evaluation import fixed_cfg, setup_server, model_responses
from test_multifidelity_integration import aq05_cfg
from engine.agent_search import AgentSearch
from engine.executor import Interpreter
from engine.search_node import Journal
from engine.mf_gp.controller import PromotionRequested
from engine.mf_gp.execution import promote, finalize
from utils import serialize


def make_agent(cfg):
    cfg.evaluation.multifidelity_mode = "mf_gp"
    return AgentSearch("Evaluation: binary log loss.", cfg, Journal())


def drafts(agent, execute):
    nodes = [agent.step(None, execute, execute_immediately=False) for _ in range(3)]
    for node in nodes:
        agent.execute_deferred_node(node, execute)
        assert node.training_fraction == 0.5 and node.is_valid, node.analysis
    return nodes


def test_real_promotions_keep_program_and_old_observation(aq05_cfg):
    agent = make_agent(aq05_cfg)
    execute = Interpreter(aq05_cfg.workspace_dir, cfg=aq05_cfg, timeout=30).run
    initial = drafts(agent, execute)
    selector = agent.phase1_runtime.study7_selector
    assert len(selector.gp.rows) == 3
    before = len(agent.journal.nodes)
    counters = [n.expected_child_count for n in initial]
    for _ in range(2):
        with pytest.raises(PromotionRequested) as exc:
            agent.step(None, execute)
        ticket = exc.value.ticket
        p = selector.pending_promotions[ticket]["program_id"]
        original = next(n for n in initial if n.id == p)
        saved = (original.metric.value, original.code, original.evaluation_attempt_id)
        promote(agent, ticket, execute)
        assert saved == (
            original.metric.value,
            original.code,
            original.evaluation_attempt_id,
        )
        assert original.training_fraction == 0.5
    assert len(agent.journal.nodes) == before
    assert counters == [n.expected_child_count for n in initial]
    assert len(selector.gp.rows) == 5
    agent.capture_resume_state()
    winner = finalize(aq05_cfg, agent.journal)
    assert winner["fidelity"] == 1.0 and winner["kind"] == "promotion"
    journal = serialize.loads_json(serialize.dumps_json(agent.journal), Journal)
    restored = AgentSearch("ignored", copy.deepcopy(aq05_cfg), journal)
    assert (
        restored.phase1_runtime.study7_selector.gp.training_hash
        == selector.gp.training_hash
    )
    # The next action is an ordinary search or promotion, without any stage switch.
    try:
        child = restored.step(None, execute)
        assert child.is_valid, child.analysis
    except PromotionRequested as exc:
        promote(restored, exc.ticket, execute)


def test_concurrent_views_are_per_attempt(aq05_cfg):
    agent = make_agent(aq05_cfg)
    from engine.search_node import SearchNode

    low = SearchNode(parent=None, plan="a", code="pass", stage="draft")
    high = SearchNode(parent=None, plan="b", code="pass", stage="draft")
    low.training_fraction = 0.5
    high.training_fraction = 1.0
    a = agent.fixed_evaluation.begin_attempt(low, 30)
    b = agent.fixed_evaluation.begin_attempt(high, 30)
    assert a["working_dir"] != b["working_dir"]
    la = Path(a["working_dir"]) / "input/train.csv"
    hb = Path(b["working_dir"]) / "input/train.csv"
    assert len(la.read_text().splitlines()) < len(hb.read_text().splitlines())
    assert (la.parent / "validation.csv").read_bytes() == (
        hb.parent / "validation.csv"
    ).read_bytes()


def test_real_runner_searches_and_finalizes(aq05_cfg, monkeypatch):
    import engine.mf_gp.runner as runner

    aq05_cfg.evaluation.multifidelity_mode = "mf_gp"
    aq05_cfg.agent.steps = 9
    aq05_cfg.run_result_file = str(Path(aq05_cfg.log_dir).parent / "run-info.json")
    monkeypatch.setattr(
        runner, "load_task_desc", lambda cfg: "Evaluation: binary log loss."
    )
    agent = runner.run(aq05_cfg)
    selector = agent.phase1_runtime.study7_selector
    assert len(selector.observations) == 9
    assert sum(o["kind"] == "promotion" for o in selector.observations) >= 2
    assert len(agent.journal.nodes) >= 4
    assert any(
        o["kind"] == "expansion" and not selector.programs[o["program_id"]]["initial"]
        for o in selector.observations
        if o["program_id"]
    )
    assert (Path(aq05_cfg.log_dir) / "final_submission.json").is_file()
    assert not selector.pending_promotions
    assert not agent.phase1_runtime.root_draft_state["pending"]


def test_terminal_controller_excludes_cheap_and_does_not_fake_labels(aq05_cfg):
    agent = make_agent(aq05_cfg)
    execute = Interpreter(aq05_cfg.workspace_dir, cfg=aq05_cfg, timeout=30).run
    drafts(agent, execute)
    selector = agent.phase1_runtime.study7_selector
    selector.calibration = []
    selector.terminal = True
    try:
        result = agent.step(None, execute)
        assert result.training_fraction == 1.0
    except PromotionRequested as exc:
        assert selector.pending_promotions[exc.ticket]["row"]["fidelity"] == 1.0
        promote(agent, exc.ticket, execute)
    assert len(selector.gp.rows) == 4


def test_mode_checkpoint_isolation(aq05_cfg):
    agent = make_agent(aq05_cfg)
    agent.capture_resume_state()
    from engine.parent_selectors.runtime import Phase1SelectorRuntime

    for mode in ("reset", "carry_over", None):
        cfg = copy.deepcopy(aq05_cfg)
        cfg.evaluation.multifidelity_mode = mode
        with pytest.raises(ValueError, match="MF-GP"):
            Phase1SelectorRuntime.validate_checkpoint(cfg, agent.journal)


def test_campaign_configuration_is_separate(tmp_path, monkeypatch):
    import json, sys
    from config import load_cfg
    from tasks.mlebench_lite.mila_multifidelity import search_command

    root = Path(__file__).resolve().parents[1]
    old = json.loads((root / "config/mlebench_mf14.json").read_text())
    new = json.loads((root / "config/mlebench_mfgp14.json").read_text())
    assert old["modes"] == ["carry_over", "reset"] and old["switch_fraction"] == 0.4
    assert new["modes"] == ["mf_gp"] and "switch_fraction" not in new
    assert old["tasks"] == new["tasks"] and len(new["tasks"]) == 14
    settings = dict(spec=new, data_dir=str(tmp_path / "data"))
    command = search_command(settings, tmp_path, "leaf-classification", "mf_gp")
    monkeypatch.setattr(sys, "argv", command[2:])
    cfg = load_cfg()
    assert cfg.evaluation.multifidelity_mode == "mf_gp"
    assert cfg.agent.time_limit == 43200 and cfg.agent.search.parallel_search_num == 3
    assert cfg.exec.timeout == 32400 and cfg.agent.steps == 500


def test_killed_promotion_resumes_same_program_without_fake_score(
    aq05_cfg, model_responses, monkeypatch, tmp_path
):
    import os, signal, subprocess, sys, time
    from omegaconf import OmegaConf
    from config import save_run
    from engine.mf_gp.recovery import execute
    from tasks.mlebench_lite import mila_multifidelity as campaign
    from test_fixed_evaluation import CANDIDATE

    monkeypatch.setenv("MLEVOLVE_MF_RECOVERY", "1")
    model_responses.code = (
        "import os, time\nfrom pathlib import Path\n"
        "if os.environ.get('MF_TEST_PAUSE'):\n"
        "    Path(os.environ['MF_TEST_PAUSE']).write_text('started')\n"
        "    time.sleep(300)\n" + CANDIDATE
    )
    agent = make_agent(aq05_cfg)
    drafts(agent, Interpreter(aq05_cfg.workspace_dir, cfg=aq05_cfg, timeout=30).run)
    with pytest.raises(PromotionRequested) as request:
        agent.step(None, None)
    ticket = request.value.ticket
    selector = agent.phase1_runtime.study7_selector
    pid = selector.pending_promotions[ticket]["program_id"]
    frozen = next(n.code for n in agent.journal.nodes if n.id == pid)
    agent.capture_resume_state()
    save_run(aq05_cfg, agent.journal)
    config_file = tmp_path / "resume.yaml"
    OmegaConf.save(aq05_cfg, config_file)
    output = tmp_path / "campaign"
    campaign.save(output / "run-info.json", {"log_dir": str(aq05_cfg.log_dir)})
    campaign.save(
        output / "segment.json",
        {"job_id": "123", "started_epoch": time.time(), "active_offset": 0},
    )
    marker = tmp_path / "started"
    script = """
import sys
from pathlib import Path
from omegaconf import OmegaConf
from config import save_run
from engine.agent_search import AgentSearch
from engine.search_node import Journal
from engine.executor import Interpreter
from engine.parent_selectors.runtime import Phase1SelectorRuntime
from engine.competition_checkpoint import checkpoint_once
from engine.mf_gp.execution import promote
from engine.mf_gp.recovery import execute
from utils import serialize
cfg = OmegaConf.load(sys.argv[1])
Phase1SelectorRuntime._owner_alive = staticmethod(lambda state: False)
agent = AgentSearch('ignored', cfg, serialize.load_json(Path(cfg.log_dir)/'journal.json', Journal))
interpreter = Interpreter(cfg.workspace_dir, cfg=cfg, timeout=300)
def persist():
    agent.capture_resume_state()
    save_run(cfg, agent.journal)
def callback(code, node_id, *args, **kwargs):
    checkpoint_once(agent, persist)
    return execute(interpreter, cfg, code, node_id, *args, **kwargs)
promote(agent, sys.argv[2], callback)
"""
    root = Path(__file__).resolve().parents[1]
    with (tmp_path / "child.log").open("w") as log:
        child = subprocess.Popen(
            [sys.executable, "-c", script, str(config_file), ticket],
            cwd=root,
            env={**os.environ, "PYTHONPATH": str(root), "MF_TEST_PAUSE": str(marker)},
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 90
            while (
                not marker.exists()
                and child.poll() is None
                and time.monotonic() < deadline
            ):
                time.sleep(0.05)
            assert marker.exists(), (tmp_path / "child.log").read_text()
        finally:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=10)
    receipt = campaign.recovery_record(
        output, {"job_id": "123", "state": "PREEMPTED", "ended_epoch": time.time()}
    )
    campaign.reconcile_recovery(output, receipt)
    budget = campaign.read(Path(aq05_cfg.log_dir) / "mf_execution" / f"{pid}__1.0.json")
    assert budget["state"] == "interrupted" and budget["elapsed_seconds"] > 0
    aq05_cfg.resume_elapsed_seconds = receipt["elapsed_seconds"]
    restored = AgentSearch(
        "ignored",
        aq05_cfg,
        serialize.load_json(Path(aq05_cfg.log_dir) / "journal.json", Journal),
    )
    resumed = restored.phase1_runtime.study7_selector
    assert len(resumed.observations) == 3 and ticket in resumed.pending_promotions
    assert next(n.code for n in restored.journal.nodes if n.id == pid) == frozen
    interpreter = Interpreter(aq05_cfg.workspace_dir, cfg=aq05_cfg, timeout=30)
    promote(
        restored,
        ticket,
        lambda code, node_id, *args, **kw: execute(
            interpreter, aq05_cfg, code, node_id, *args, **kw
        ),
    )
    assert {n.id for n in restored.journal.nodes} == {n.id for n in agent.journal.nodes}
    assert len(resumed.observations) == 4 and not resumed.pending_promotions
    assert (
        resumed.observations[-1]["valid"]
        and resumed.observations[-1]["fidelity"] == 1.0
    )
    restored.capture_resume_state()
    assert finalize(aq05_cfg, restored.journal)["program_id"] == pid
