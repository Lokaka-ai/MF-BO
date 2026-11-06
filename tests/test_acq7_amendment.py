"""User-authorized AQ08 tau change and later rating-only control."""
import copy
import math
import numpy as np
from numpy.testing import assert_allclose
from omegaconf import OmegaConf
import pytest

from config.acq7 import acq7_config
from engine.parent_selectors.acq7_gp import AcqGP, branch_covariance
from engine.parent_selectors.runtime import Phase1SelectorRuntime
from test_study7_runtime import factory, initialize, load_setting
from test_acq7_runtime import (
    check_initialization, check_debug, check_resume, check_late,
    check_two, check_interruption,
)
from test_acq7_math import rows, oracle_kernel

TAU_PRESET = 'acq7_aq08_gp4_level20_rating_jts'


def test_aq08_effective_diff_and_independent_posterior(tmp_path):
    left, right = load_setting('acq7_aq04_gp4_level15_rating_jts'), load_setting(TAU_PRESET)
    expected = copy.deepcopy(left)
    expected.study7.method_id = 'AQ08'
    expected.acq7.branch_prior_std = 2.0
    assert OmegaConf.to_container(expected) == OmegaConf.to_container(right)
    right.log_dir = tmp_path
    runtime = Phase1SelectorRuntime(right)
    cfg = runtime.study7_settings
    assert cfg.branch_prior_std == 2.0 and cfg.rating.enabled
    data = rows()
    branch = branch_covariance(data, data, cfg)
    for i,a in enumerate(data):
        for j,b in enumerate(data):
            z=math.sqrt(3)*abs(a['performance_v']-b['performance_v'])
            expected=4*(a['root_lineage_id']==b['root_lineage_id'])*(1+z)*math.exp(-z)
            assert_allclose(branch[i,j], expected, atol=1e-14)
    gp = runtime.surrogate
    target=np.array([.2,-.1,.3]);gp.fit(data[:3],target,training_ids=['a','b','c'])
    hp=gp.hyperparameters
    k=oracle_kernel(data[:3],data[:3],cfg,hp)+(hp[2]**2+gp.jitter)*np.eye(3)
    cross=oracle_kernel(data,data[:3],cfg,hp)
    prior=np.array([.25*x['rating_s'] for x in data])
    mean=np.array([x['parent_utility'] for x in data])+.2*(prior+cross@np.linalg.solve(k,target-prior[:3]))
    covariance=.04*(oracle_kernel(data,data,cfg,hp)-cross@np.linalg.solve(k,cross.T))
    actual=gp.predict_joint(data,.2)
    assert_allclose(actual['mean'],mean,atol=1e-13)
    assert_allclose(actual['latent_covariance'],covariance,atol=1e-13)
    saved=gp.state_dict();restored=AcqGP(cfg,{})
    restored.load_state_dict(saved)
    # Production resume rebuilds the factor from the same committed observations.
    restored.fit(data[:3],target,training_ids=['a','b','c'])
    assert restored.fit_diagnostics.get('objective_calls', 0) == 0
    assert_allclose(restored.predict_joint(data,.2)['latent_covariance'],covariance,atol=1e-13)
    right.acq7.branch_prior_std=1.5
    with pytest.raises(ValueError):acq7_config(right)


def test_aq08_original_review_and_initialization(factory):
    check_initialization(TAU_PRESET, factory)


def test_aq08_debug_admission(factory):
    check_debug(TAU_PRESET, factory)


def test_aq08_resume_same_draw_and_reference(factory):
    check_resume(TAU_PRESET, factory)


def test_aq08_deadline_censors_late_result(factory):
    check_late(TAU_PRESET, factory)


def test_aq08_two_reservations_do_not_change_training_or_global_rng(factory):
    check_two(TAU_PRESET, factory)


def test_aq08_two_worker_checkpoint(tmp_path,monkeypatch,autoresearch_cache):
    check_interruption(TAU_PRESET,2,tmp_path,monkeypatch)


RATING_PRESET='acq7_aq09_rating_only_greedy'


def test_aq09_original_rating_no_gp_and_rating_over_metric(factory,monkeypatch):
    import random
    from engine.parent_selectors.study7_gp import State4GP
    from engine.parent_selectors.branch7_gp import BranchGP
    from utils.metric import MetricValue
    for cls in (State4GP,BranchGP,AcqGP):
        monkeypatch.setattr(cls,'__init__',lambda *a,**k:pytest.fail('Baseline constructed a GP'))
    build,calls=factory;agent,execute=build(RATING_PRESET)
    drafts=initialize(agent,execute);runtime=agent.phase1_runtime
    assert runtime.surrogate is None and runtime.study7_selector.gp is None
    assert runtime.wait_initial_results and not runtime.include_root_draft
    assert agent.scfg.parallel_search_num==2 and agent.scfg.num_drafts==5
    assert len(calls.feedbacks)==3 and calls.charges==[11,12,13]
    assert all(n.agent_potential_level==2 for n in drafts)
    assert all('potential_level' in x['func_spec'].json_schema['properties'] for x in calls.feedbacks)
    for n,level,bpb in zip(drafts,[0,2,None],[1.,9.,2.]):
        n.lock=False;n.agent_potential_level=level;n.metric=MetricValue(bpb,maximize=False)
    options=runtime._normal_options(agent)
    assert len(options)==3
    before_py=random.getstate();before_np=np.random.get_state()
    result=runtime.study7_selector.select(agent,options)
    assert result.selected_option.parent_node_id==drafts[1].id
    assert result.decision.model_version is None and not result.decision.fallback_used
    assert random.getstate()==before_py
    after_np=np.random.get_state()
    assert before_np[0]==after_np[0] and np.array_equal(before_np[1],after_np[1]) and before_np[2:]==after_np[2:]
    assert [x.state_features['study7']['rating_score'] for x in options]==[-1,1,0]
    assert all(x.mean_gain is None and x.std_gain is None for x in result.candidate_scores)


def test_aq09_stable_tie_resume_and_contract_rejection(factory):
    from engine.agent_search import AgentSearch
    from engine.search_node import Journal
    from engine.parent_selectors.study7_gp import Study7Error
    from utils import serialize
    build,_=factory;agent,execute=build(RATING_PRESET);drafts=initialize(agent,execute)
    for n in drafts:n.lock=False
    runtime=agent.phase1_runtime;options=runtime._normal_options(agent)
    one=runtime.study7_selector.select(agent,options)
    two=runtime.study7_selector.select(agent,list(reversed(options)))
    assert one.selected_option.parent_node_id==two.selected_option.parent_node_id==drafts[0].id
    agent.capture_resume_state();path=agent.cfg.log_dir/'rating_resume.json'
    serialize.dump_json(agent.journal,path)
    agent.cfg.resume_run_dir=agent.cfg.log_dir.parent
    resumed=AgentSearch(agent.task_desc,agent.cfg,serialize.load_json(path,Journal))
    assert resumed.phase1_runtime.surrogate is None
    assert [n.agent_potential_level for n in resumed.journal.nodes]==[n.agent_potential_level for n in agent.journal.nodes]
    assert resumed.phase1_runtime.study7_selector.state_dict()==runtime.study7_selector.state_dict()
    bad=copy.deepcopy(runtime.study7_selector.state_dict());bad['contract']['ranking']='parent_metric'
    with pytest.raises(Study7Error):runtime.study7_selector.load_state_dict(bad)


def test_aq09_admission_and_debug_review(factory):
    from tasks.autoresearch.study7_budget import admit_result
    build,calls=factory;agent,execute=build(RATING_PRESET);drafts=initialize(agent,execute)
    calls.state.fail_execution=True;bad=agent.step(drafts[0],execute)
    calls.state.fail_execution=False;calls.state.rating=0;good=agent.step(bad,execute)
    assert good.stage=='debug' and good.agent_potential_level==0
    assert agent.phase1_runtime.surrogate is None
    calls.clock[0]=agent.start_time+21601
    admit_result(agent,good)
    assert not good.is_valid and good.study7_budget_metadata['censored']
    assert good.agent_potential_level is None


def test_aq09_two_worker_checkpoint(tmp_path,monkeypatch,autoresearch_cache):
    check_interruption(RATING_PRESET,2,tmp_path,monkeypatch)
