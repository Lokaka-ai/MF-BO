"""Acq7 production-managed dispatch; external provider/training only are mocked."""
import copy
import json
import random

import numpy as np
from numpy.testing import assert_allclose
from omegaconf import OmegaConf
import pytest

from config.acq7 import ACQ7_PRESETS
from engine.parent_selectors.runtime import Phase1SelectorRuntime, SelectionPending
from engine.parent_selectors.study7_gp import Study7Error
from test_study7_runtime import factory, initialize, load_setting
from test_branch7_runtime import (
    test_r09_d10_actual_initialization_review_and_admission as check_initialization,
    test_d04_d05_d06_k16_debug_closes_original_parent_lineage_rating as check_debug,
    test_r07_resume_preserves_scale_branches_scores_and_choice as check_resume,
    test_r04_r06_late_review_result_excluded_without_losing_raw_metric as check_late,
    test_r02_r10_two_reservations_do_not_become_training_feedback_or_change_rng as check_two,
)
from test_study_segmented import test_interruption_preserves_shared_drafts_queue_and_native_results as check_interruption


@pytest.mark.parametrize("preset", [ACQ7_PRESETS[3], ACQ7_PRESETS[4]])
def test_pending_parent_pool_acquisition_and_frozen_resume(factory, preset):
    from engine.agent_search import AgentSearch
    from engine.search_node import Journal
    from utils import serialize
    build, _ = factory
    agent, execute = build(preset)
    drafts = initialize(agent, execute)
    runtime = agent.phase1_runtime
    for node in drafts:
        node.lock = False
    contexts = [runtime.select_global_parent(agent) for _ in range(len(drafts))]
    assert len({ctx.option.parent_node_id for ctx in contexts}) == len(drafts)
    selector = runtime.study7_selector
    evidence = selector.last_committed
    assert evidence["candidate_count"] == 1
    with np.load(runtime.artifact_root / evidence["joint_artifact"]["path"]) as arrays:
        assert arrays["latent_covariance"].shape == (1, 1)
    if selector.config.acquisition_type == "joint_ts":
        assert selector.accepted_draw_index == len(drafts) - 1
        assert evidence["acquisition"]["reason"] == "single_candidate_no_draw"
    extra = runtime.select_global_parent(agent)
    assert selector.last_committed["candidate_count"] == len(drafts)
    assert len(selector.gp.targets) == 0
    runtime.cancel_reservation(agent, extra)
    for context in contexts[:-1]:
        runtime.cancel_reservation(agent, context)
    retained = contexts[-1]
    pending_node = agent.step(None, execute, execute_immediately=False, recovery_context=retained)
    assert pending_node.pending_execution
    assert retained.option.parent_node_id not in {o.parent_node_id for o in runtime._normal_options(agent)}
    agent.capture_resume_state()
    saved = serialize.loads_json(serialize.dumps_json(agent.journal), Journal)
    resumed = AgentSearch("ignored", agent.cfg, saved)
    restored = resumed.phase1_runtime
    assert restored._pending_parent_ids() == {retained.option.parent_node_id}
    recovered, = restored.recovered_deferred_nodes
    resumed.execute_deferred_node(recovered, execute)
    assert not restored._pending_parent_ids()
    assert len(restored.transition_store.records()) == len(drafts) + 1


@pytest.mark.parametrize("preset", [ACQ7_PRESETS[3], ACQ7_PRESETS[4]])
def test_pending_parent_execution_failure_releases_slot(factory, preset):
    build, calls = factory
    agent, execute = build(preset)
    drafts = initialize(agent, execute)
    for node in drafts:
        node.lock = False
    runtime = agent.phase1_runtime
    context = runtime.select_global_parent(agent)
    calls.state.fail_execution = True
    agent.step(None, execute, recovery_context=context)
    assert not runtime._pending_parent_ids()
    assert not runtime.transition_store.records()[-1].valid


@pytest.mark.parametrize("preset",ACQ7_PRESETS)
def test_original_initialization_review_gp4_admission(preset,factory):
    check_initialization(preset,factory)


@pytest.mark.parametrize("preset",[ACQ7_PRESETS[i] for i in (3,4,6)])
def test_debug_keeps_original_proposal_rating_and_lineage(preset,factory):
    check_debug(preset,factory)


@pytest.mark.parametrize("preset",ACQ7_PRESETS)
def test_resume_same_event_sequence_same_choice_and_sample(preset,factory):
    check_resume(preset,factory)


@pytest.mark.parametrize("preset",ACQ7_PRESETS)
def test_late_result_not_quality_feedback(preset,factory):
    check_late(preset,factory)


@pytest.mark.parametrize("preset",ACQ7_PRESETS)
def test_native_pending_and_global_rng_unchanged(preset,factory):
    check_two(preset,factory)


@pytest.mark.parametrize("preset",ACQ7_PRESETS)
def test_two_worker_interruption_keeps_shared_pool_and_fifo(preset,tmp_path,monkeypatch,autoresearch_cache):
    check_interruption(preset,2,tmp_path,monkeypatch)


def test_ts_atomic_revision_retry_duplicate_and_committed_evidence(factory):
    build,_=factory;agent,execute=build(ACQ7_PRESETS[0])
    drafts=initialize(agent,execute)
    for node in drafts:node.lock=False
    runtime=agent.phase1_runtime;selector=runtime.study7_selector
    options=runtime._normal_options(agent)
    before_py=random.getstate();before_np=np.random.get_state()
    result=selector.select(agent,options)
    first=copy.deepcopy(selector._prepared)
    assert selector.rng_state is None and selector.accepted_draw_index==0
    assert first['payload']['first_ordinary_decision'] and first['payload']['never_dispatched_branch']
    assert not first['payload']['no_feedback_branch']
    runtime.root_draft_state["revision"]+=1
    with pytest.raises(SelectionPending):runtime._reserve_decision(agent,result)
    assert selector.rng_state is None and selector.accepted_draw_index==0
    second=selector.select(agent,list(reversed(options)))
    assert selector._prepared["payload"]["candidate_order"]==first["payload"]["candidate_order"]
    assert selector._prepared["payload"]["acquisition"]["sample"]==first["payload"]["acquisition"]["sample"]
    runtime._reserve_decision(agent,second)
    saved=copy.deepcopy(selector.state_dict())
    assert selector.accepted_draw_index==1
    runtime._reserve_decision(agent,second)
    assert selector.state_dict()==saved
    assert random.getstate()==before_py
    after_np=np.random.get_state()
    assert before_np[0]==after_np[0] and np.array_equal(before_np[1],after_np[1]) and before_np[2:]==after_np[2:]
    evidence=json.loads((runtime.artifact_root/'study7_decisions.jsonl').read_text().splitlines()[-1])
    assert evidence['decision_id']==second.decision.decision_id
    with np.load(runtime.artifact_root/evidence['joint_artifact']['path']) as arrays:
        assert arrays['latent_covariance'].shape==(len(options),len(options))
        reconstructed=arrays['mean']+selector.gp.predict_joint([o.state_features['study7'] for o in second.options],selector.state.reference['s0'])['latent_square_root']@np.array(evidence['acquisition']['standard_normal'])
        assert_allclose(reconstructed,evidence['acquisition']['sample'])
    # Durable reservation can restore a missing diagnostic append, without a draw.
    path=runtime.artifact_root/'study7_decisions.jsonl';path.unlink()
    selector.load_state_dict(saved)
    assert selector.accepted_draw_index==1 and len(path.read_text().splitlines())==1
    selector.publish_committed();assert len(path.read_text().splitlines())==1


@pytest.mark.parametrize("key,value",[("acq7.branch_kind","constant"),
    ("acq7.branch_lengthscale",2.),("acq7.slope_prior_std",1.),
    ("acq7.acquisition_type","finite_pool_kg"),("acq7.rng_purpose_id",7302),
    ("acq7.pending_policy","fantasy"),("study7.ancestor.enabled",True),
    ("study7.rating.enabled",False),("bo_parent_selector.acquisition.beta",1.5),
    ("agent.seed",43),("parent_selector.include_root_draft",True)])
def test_registered_identity_rejects_mismatch(key,value,tmp_path):
    cfg=load_setting(ACQ7_PRESETS[3]);cfg.log_dir=tmp_path
    OmegaConf.update(cfg,key,value)
    with pytest.raises(ValueError):Phase1SelectorRuntime(cfg)


def test_resume_cannot_change_acquisition_or_model_contract(factory):
    build,_=factory;agent,execute=build(ACQ7_PRESETS[3]);initialize(agent,execute)
    selector=agent.phase1_runtime.study7_selector
    saved=selector.state_dict();saved['contract']['acquisition_type']='finite_pool_kg'
    with pytest.raises(Study7Error):selector.load_state_dict(saved)


@pytest.mark.parametrize("preset",[ACQ7_PRESETS[5],ACQ7_PRESETS[6]])
def test_ordinary_double_worker_pause_resume_keeps_committed_draws_and_fractional_budget(preset,factory,tmp_path,monkeypatch):
    import threading
    import time
    from types import SimpleNamespace
    from config import save_run
    from engine.agent_search import AgentSearch
    from engine.search_node import Journal
    from tasks.autoresearch.segmented import run_segmented,restore_checkpoint
    from utils import serialize
    build,calls=factory;agent,execute=build(preset)
    cfg=agent.cfg;cfg.agent.steps=8
    pause=tmp_path/'ordinary_pause';checkpoint=tmp_path/'ordinary_checkpoints'
    monkeypatch.setenv('MLEVOLVE_SEGMENT_CHECKPOINT',str(checkpoint))
    monkeypatch.setenv('MLEVOLVE_SEGMENT_PAUSE_FILE',str(pause))
    monkeypatch.setenv('MLEVOLVE_SEGMENT_STARTED',str(time.time()))
    barrier=threading.Barrier(2);lock=threading.Lock();seen=[];finished=[]
    def wrapper(*args,**kwargs):
        with lock:
            number=len(seen);seen.append(number)
        if number<2:
            barrier.wait(timeout=10)
            if number==0:threading.Event().wait(.05)
        result=execute(*args,**kwargs)
        with lock:
            finished.append(number)
            if len(finished)>=5:pause.touch()
        return result
    interpreter=SimpleNamespace(terminate_all_subprocesses=lambda:None)
    def persist(instance):instance.capture_resume_state();save_run(cfg,instance.journal)
    run_segmented(agent,interpreter,cfg,wrapper,lambda:persist(agent))
    assert finished[:2]==[1,0]
    prior=agent.phase1_runtime.study7_selector.state_dict()
    assert len(prior['committed_draws'])>=2
    assert 5<=len(agent.phase1_runtime.transition_store.records())<8
    assert json.loads((cfg.log_dir/'segment_outcome.json').read_text())['state']=='paused'
    restore_checkpoint(checkpoint);pause.unlink()
    cfg.resume_run_dir=cfg.log_dir.parent
    journal=serialize.load_json(cfg.log_dir/'journal.json',Journal)
    resumed=AgentSearch(agent.task_desc,cfg,journal)
    monkeypatch.setenv('MLEVOLVE_SEGMENT_STARTED',str(time.time()))
    monkeypatch.setenv('MLEVOLVE_SEGMENT_ACTIVE_OFFSET','123.75')
    expected_start=time.time()-123.75
    run_segmented(resumed,interpreter,cfg,execute,lambda:persist(resumed))
    assert resumed.start_time==expected_start
    assert len(resumed.phase1_runtime.transition_store.records())==8
    after=resumed.phase1_runtime.study7_selector.state_dict()
    assert after['state']['reference']==prior['state']['reference']
    assert after['accepted_draw_index']>=prior['accepted_draw_index']
    for decision,value in prior['committed_draws'].items():assert after['committed_draws'][decision]==value
