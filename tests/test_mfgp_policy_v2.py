"""Approved scheduling corrections; no live provider or benchmark calls."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from engine.mf_gp.policy import (estimate_costs, fill_calibration, exact_ceiling_reached,
                                validate_policy, load_cost_prior)

PRIOR = dict(training_fraction=.2, low_seconds=100., full_seconds=500.)

def observation(p, f, seconds, complete=True, lower=False):
    return dict(program_id=p, fidelity=f, execution_seconds=seconds, generation_seconds=10.,
                completed_runtime_observation=complete, runtime_lower_bound=lower,
                operator="improve", valid=complete, utility=.9)


def test_failed_and_censored_costs_never_become_completed_pairs():
    rows = [observation("a", .2, 100), observation("a", 1., 500), observation("b", .2, 100)]
    baseline = estimate_costs(rows, .2, PRIOR, "b")
    rows += [observation("b", 1., 1, False), observation("c", .2, 9000, False, True)]
    assert estimate_costs(rows, .2, PRIOR, "b") == baseline
    assert sum(o["execution_seconds"] for o in rows) == 9701  # accounting retained
    rows += [observation("b", 1., 1000, False, True)]
    cheap, full, _, guard = estimate_costs(rows, .2, PRIOR, "b")
    assert cheap == 100 and full == 500 and guard >= 1500


def test_profile_prior_and_operator_generation():
    assert estimate_costs([], .2, PRIOR) == (100., 500., 60., 750.)
    rows = [observation("a", .2, 200)]
    assert estimate_costs(rows, .2, PRIOR, "a")[1] == 1000
    with pytest.raises(ValueError, match="measured"):
        validate_policy(2, None, .2)
    assert validate_policy(1, None, .5) == (1, None)


def test_async_calibration_refills_once_and_keeps_first():
    programs = {p: {"initial": True} for p in "abc"}
    rows = [observation("a", .2, 100)]
    selected = fill_calibration([], ["a"], programs, rows, 42)
    assert selected == ["a"]
    # Completed first promotion is no longer a candidate; repaired drafts arrive later.
    rows += [observation("b", .2, 100), observation("c", .2, 100)]
    updated = fill_calibration(selected, ["c", "b"], programs, rows, 42)
    assert updated == fill_calibration(selected, ["b", "c"], programs, rows, 42)
    assert len(updated) == 2 and updated[0] == "a"
    assert fill_calibration(updated, list(programs), programs, rows, 42) == updated
    assert fill_calibration([], [], programs, rows, 42) == []


@pytest.mark.parametrize("utility,fidelity,valid,eligible,expected", [
    (1.,1.,True,True,True), (.999999999,1.,True,True,False),
    (1.,.2,True,True,False), (1.,1.,False,True,False), (1.,1.,True,False,False)])
def test_exact_ceiling_eligibility(utility,fidelity,valid,eligible,expected):
    rows = [dict(utility=utility,fidelity=fidelity,valid=valid,budget_eligible=eligible)]
    assert exact_ceiling_reached(rows, dict(name="roc_auc",maximize=True)) is expected
    assert not exact_ceiling_reached(rows, dict(name="unknown",maximize=True))


def test_cost_profile_identity_and_fraction(tmp_path):
    (tmp_path/"protocol.json").write_text(json.dumps(dict(protocol_id="frozen")))
    receipt = dict(schema=1,protocol_id="frozen",training_fraction=.2,completed=True,
                   code_sha256="a"*64,measurement="candidate_process_wall_seconds",
                   low_seconds=100.,full_seconds=500.)
    path=tmp_path/"profile.json"; path.write_text(json.dumps(receipt))
    first=load_cost_prior(path,tmp_path,.2)
    assert first["full_seconds"] == 500
    receipt["full_seconds"]=600; path.write_text(json.dumps(receipt))
    assert first["receipt_sha256"] != load_cost_prior(path,tmp_path,.2)["receipt_sha256"]
    with pytest.raises(ValueError):load_cost_prior(path,tmp_path,.5)
    receipt["completed"]=False; path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):load_cost_prior(path,tmp_path,.2)


def test_profiler_real_subprocess_and_fail_closed(tmp_path):
    from tools.profile_mf_cost import profile
    protocol=tmp_path/"protocol"; public=protocol/"public"; public.mkdir(parents=True)
    (protocol/"protocol.json").write_text(json.dumps(dict(protocol_id="test")))
    (public/"train.csv").write_text("sample_id,label\n"+"".join(f"{i},{i%2}\n" for i in range(20)))
    for name in ("validation.csv","test.csv","sample_submission.csv"):
        (public/name).write_text("sample_id\nv\n")
    candidate=tmp_path/"code.py"
    candidate.write_text("from pathlib import Path\nassert Path('input/train.csv').is_file()\n")
    output=tmp_path/"prior.json"
    result=profile(candidate,protocol,tmp_path/"work",output,timeout=30)
    assert result["completed"] and result["low_seconds"]>0 and result["full_seconds"]>0
    assert load_cost_prior(output,protocol,.2)
    candidate.write_text("raise RuntimeError('expected')")
    with pytest.raises(RuntimeError):profile(candidate,protocol,tmp_path/"failed",tmp_path/"bad.json",timeout=30)
    assert not (tmp_path/"bad.json").exists()
    assert "expected" in (tmp_path/"failed/low/stderr.log").read_text()


def test_legacy_contract_and_v2_resume_reject_changed_prior():
    from engine.mf_gp.controller import MFSelector
    from config.acq7 import ACQ7_PRESETS
    # Native selector contract uses a dataclass; no agent needed for this check.
    from dataclasses import dataclass
    @dataclass
    class Config: marker: int = 1
    legacy=MFSelector(Config(),lambda:[],None)
    assert "policy_version" not in legacy.contract()
    revised=MFSelector(Config(),lambda:[],None,.2,2,PRIOR)
    saved=revised.state_dict()
    same=MFSelector(Config(),lambda:[],None,.2,2,PRIOR)
    same.load_state_dict(saved)
    different=MFSelector(Config(),lambda:[],None,.2,2,{**PRIOR,"full_seconds":600.})
    with pytest.raises(ValueError, match="contract"):different.load_state_dict(saved)


from test_fixed_evaluation import fixed_cfg, setup_server, model_responses
from test_multifidelity_integration import aq05_cfg


def test_v2_real_execution_promotion_resume_and_ceiling(aq05_cfg):
    from engine.mf_gp.execution import MFEvaluation, promote, finalize
    from engine.mf_gp.controller import PromotionRequested
    from engine.mf_gp.recovery import execute
    from engine.agent_search import AgentSearch
    from engine.executor import Interpreter
    from engine.search_node import Journal
    from engine.parent_selectors.runtime import SearchExhausted
    from utils import serialize
    cfg=aq05_cfg
    cfg.evaluation.multifidelity_mode=None
    cfg.evaluation.mf_gp_low_fraction=.5
    # Prepare the fixture protocol before producing its synthetic profiling receipt.
    from engine.fixed_evaluation import FixedEvaluation
    FixedEvaluation.prepare_or_load(cfg,"Evaluation: binary log loss.",Journal())
    cfg.evaluation.multifidelity_mode="mf_gp"
    cfg.evaluation.mf_gp_low_fraction=.2
    spec=json.loads((Path(cfg.evaluation.protocol_dir)/"protocol.json").read_text())
    path=Path(cfg.log_dir)/"cost-prior.json"
    path.write_text(json.dumps(dict(schema=1,protocol_id=spec["protocol_id"],
        training_fraction=.2,completed=True,code_sha256="a"*64,
        measurement="candidate_process_wall_seconds",low_seconds=1.,full_seconds=2.)))
    cfg.evaluation.mf_gp_policy_version=2
    cfg.evaluation.mf_gp_cost_profile=str(path)
    agent=AgentSearch("Evaluation: binary log loss.",cfg,Journal())
    interpreter=Interpreter(cfg.workspace_dir,cfg=cfg,timeout=30)
    callback=lambda code,node_id,*a,**kw:execute(interpreter,cfg,code,node_id,*a,**kw)
    nodes=[agent.step(None,callback,execute_immediately=False) for _ in range(3)]
    for node in nodes:agent.execute_deferred_node(node,callback)
    selector=agent.phase1_runtime.study7_selector
    assert all(o["completed_runtime_observation"] for o in selector.observations)
    assert all(n.training_fraction==.2 and n.is_valid for n in nodes)
    with pytest.raises(PromotionRequested) as request:agent.step(None,callback)
    promote(agent,request.value.ticket,callback)
    agent.capture_resume_state()
    assert finalize(cfg,agent.journal)["fidelity"]==1.
    saved=serialize.dumps_json(agent.journal)
    restored=AgentSearch("ignored",copy.deepcopy(cfg),serialize.loads_json(saved,Journal))
    assert restored.phase1_runtime.study7_selector.contract()==selector.contract()
    # Real selector entry stops new work, never a cheap-score-only ceiling test.
    restored.fixed_evaluation.spec["metric"]={"name":"roc_auc","maximize":True}
    restored.phase1_runtime.study7_selector.observations.append(dict(
        program_id=nodes[0].id,valid=True,budget_eligible=True,fidelity=1.,utility=1.,
        x=[0.]*8,operator="draft",execution_seconds=1.,generation_seconds=0.))
    with pytest.raises(SearchExhausted,match="exact metric ceiling"):
        restored.phase1_runtime._select_global_with_root(restored)


def test_submission_cost_profile_freeze(tmp_path):
    from tasks.mlebench_lite.mila_multifidelity import validate_cost_profiles
    task="example"
    protocol=tmp_path/"protocols"/task;protocol.mkdir(parents=True)
    (protocol/"protocol.json").write_text(json.dumps(dict(protocol_id="frozen")))
    profiles=tmp_path/"cost_profiles";profiles.mkdir()
    receipt=dict(schema=1,protocol_id="frozen",training_fraction=.2,completed=True,
        code_sha256="b"*64,measurement="candidate_process_wall_seconds",low_seconds=2.,full_seconds=9.)
    path=profiles/(task+".json");path.write_text(json.dumps(receipt))
    settings=dict(spec=dict(mf_gp_policy_version=2,training_fraction=.2))
    frozen=validate_cost_profiles(tmp_path,settings,[task])
    (profiles/"submitted.json").write_text(json.dumps(frozen))
    receipt["full_seconds"]=10.;path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match="prior changed"):
        validate_cost_profiles(tmp_path,settings,[task])
    assert validate_cost_profiles(tmp_path,dict(spec={}),[task])=={}
