"""Production Branch7 paths; only external training/provider/memory are mocked."""
import copy
import json
import random
import shutil
import time
from dataclasses import replace

import numpy as np
from numpy.testing import assert_allclose
from omegaconf import OmegaConf
import pytest

from config.branch7 import BRANCH7_PRESETS, BRANCH7_ARMS
from engine.agent_search import AgentSearch
from engine.parent_selectors.branch7_state import RootLineages
from engine.parent_selectors.runtime import Phase1SelectorRuntime
from engine.parent_selectors.study7_gp import Study7Error
from engine.search_node import Journal
from tasks.autoresearch.study7_budget import registered
from test_study7_runtime import factory, initialize, load_setting
from test_study_segmented import test_interruption_preserves_shared_drafts_queue_and_native_results as check_interruption
from tests_gp6.test_core import record
from utils import serialize


@pytest.mark.parametrize('setting',BRANCH7_PRESETS)
def test_r09_d10_actual_initialization_review_and_admission(setting,factory):
    build,calls=factory
    agent,execute=build(setting)
    runtime=agent.phase1_runtime
    assert registered(agent)
    assert not runtime.study7_selector.state.lineages.nodes
    drafts=initialize(agent,execute)
    config=runtime.study7_settings
    assert len(calls.feedbacks)==3 and not calls.generations
    assert calls.charges==[11,12,13]
    assert agent.scfg.parallel_search_num==2 and agent.scfg.num_drafts==5
    assert len({runtime.study7_selector.state.lineages.nodes[n.id]['root_lineage_id'] for n in drafts})==3
    for n,review in zip(drafts,calls.feedbacks):
        assert ('potential_level' in review['func_spec'].json_schema['properties'])==config.rating.enabled
        assert n.agent_potential_level==(2 if config.rating.enabled else None)
    child=agent.step(drafts[0],execute)
    assert child.is_valid
    ctx=runtime.select_global_parent(agent)
    runtime.cancel_reservation(agent,ctx)
    gp=runtime.surrogate
    assert len(gp.rows)==(4 if config.base_dimension==6 else 1)
    for row in gp.rows:
        if row['operator']=='draft':
            assert row['branch_effect_load']==0 and row['root_lineage_id'] is None and row['rating_s']==0
        else:
            assert row['branch_effect_load']==1 and row['root_lineage_id']
            assert row['rating_s']==int(config.rating.enabled)
    assert not runtime.include_root_draft
    for _,summary,description in calls.memories:
        assert 'potential_level' not in summary+description


@pytest.mark.parametrize('setting',[BRANCH7_PRESETS[4],BRANCH7_PRESETS[5]])
def test_d04_d05_d06_k16_debug_closes_original_parent_lineage_rating(setting,factory):
    build,calls=factory;agent,execute=build(setting)
    drafts=initialize(agent,execute)
    calls.state.fail_execution=True
    bad=agent.step(drafts[0],execute)
    runtime=agent.phase1_runtime
    original=runtime.transition_store.records()[-1]
    calls.state.fail_execution=False;calls.state.rating=0
    good=agent.step(bad,execute)
    assert good.stage=='debug' and good.is_valid
    ctx=runtime.select_global_parent(agent);runtime.cancel_reservation(agent,ctx)
    normal=[r for r in runtime.surrogate.rows if r['operator']!='draft']
    assert len(normal)==1 and normal[0]==original.selection_features['study7']
    assert normal[0]['rating_s']==1 and good.agent_potential_level==0
    mapping=runtime.study7_selector.state.lineages.nodes
    assert mapping[bad.id]==mapping[good.id]==mapping[original.parent_node_id]
    assert runtime.study7_selector.state.nodes[good.id]['scientific_parent_id']==original.parent_node_id


@pytest.mark.parametrize('setting',BRANCH7_PRESETS)
def test_r07_resume_preserves_scale_branches_scores_and_choice(setting,factory):
    build,_=factory;agent,execute=build(setting)
    drafts=initialize(agent,execute);agent.step(drafts[0],execute)
    agent.capture_resume_state()
    saved=serialize.loads_json(serialize.dumps_json(agent.journal),Journal)
    cfg=copy.deepcopy(agent.cfg);cfg.resume_run_dir=cfg.log_dir.parent
    resumed=AgentSearch('ignored',cfg,saved)
    contexts=[a.phase1_runtime.select_global_parent(a) for a in (agent,resumed)]
    a,b=contexts
    assert (a.option.parent_node_id,a.option.operator)==(b.option.parent_node_id,b.option.operator)
    assert_allclose([s.acquisition_score for s in a.result.candidate_scores],
                    [s.acquisition_score for s in b.result.candidate_scores])
    for instance,ctx in zip((agent,resumed),contexts):instance.phase1_runtime.cancel_reservation(instance,ctx)
    s1=agent.phase1_runtime.study7_selector.state
    s2=resumed.phase1_runtime.study7_selector.state
    assert s1.reference==s2.reference and s1.lineages.state_dict()==s2.lineages.state_dict()


@pytest.mark.parametrize('setting',BRANCH7_PRESETS)
def test_r04_r06_late_review_result_excluded_without_losing_raw_metric(setting,factory):
    build,calls=factory;agent,execute=build(setting)
    drafts=initialize(agent,execute)
    calls.clock[0]=agent.start_time+21599
    agent.step(drafts[0],execute)
    node=agent.journal.nodes[-1]
    assert node.study7_budget_metadata['completed_active_seconds']==21601
    assert node.study7_budget_metadata['raw_metric']==1.7
    assert node.metric.value is None and not node.is_valid
    assert agent.journal.get_best_node().id in {n.id for n in drafts}
    rows=agent.phase1_runtime.transition_store.records()
    assert not rows[-1].valid and not rows[-1].metadata['budget_eligible']


@pytest.mark.parametrize('setting',[BRANCH7_PRESETS[3],BRANCH7_PRESETS[5]])
def test_r02_r10_two_reservations_do_not_become_training_feedback_or_change_rng(setting,factory):
    build,_=factory;agent,execute=build(setting)
    drafts=initialize(agent,execute)
    # Supply three legally idle parents for the reservation test. The native
    # FIFO initially retains draft locks for their own continuation work items.
    for draft in drafts:draft.lock=False
    runtime=agent.phase1_runtime
    before_py=random.getstate();before_np=np.random.get_state()
    contexts=[runtime.select_global_parent(agent) for _ in range(2)]
    # Pending-parent preference changes the legal pool without adding outcomes.
    assert contexts[0].option.parent_node_id != contexts[1].option.parent_node_id
    assert contexts[0].decision.decision_id!=contexts[1].decision.decision_id
    assert len(runtime.surrogate.rows)==(3 if runtime.branch7_gp6_enabled else 0)
    assert len(runtime.root_draft_state['pending'])==2
    assert random.getstate()==before_py
    after_np=np.random.get_state()
    assert before_np[0]==after_np[0] and np.array_equal(before_np[1],after_np[1]) and before_np[2:]==after_np[2:]
    for ctx in contexts:runtime.cancel_reservation(agent,ctx)


@pytest.mark.parametrize('setting',BRANCH7_PRESETS)
def test_r01_r03_r05_r08_dual_worker_atomic_interruption_resume(setting,tmp_path,monkeypatch,autoresearch_cache):
    check_interruption(setting,2,tmp_path,monkeypatch)


@pytest.mark.parametrize('field,value',[('branch7.branch_prior_std',2.),
    ('study7.rating.enabled',False),('study7.ucb_coefficient',2.),
    ('parent_selector.include_root_draft',True),('bo_parent_selector.agent_rating.enabled',True)])
def test_illegal_registered_contract_rejected(field,value,tmp_path):
    cfg=load_setting(BRANCH7_PRESETS[3]);cfg.log_dir=tmp_path
    OmegaConf.update(cfg,field,value)
    with pytest.raises(ValueError):Phase1SelectorRuntime(cfg)


def test_br04_cannot_resume_as_br07(factory):
    build,_=factory;agent,execute=build(BRANCH7_PRESETS[3])
    drafts=initialize(agent,execute);agent.step(drafts[0],execute);agent.capture_resume_state()
    cfg=copy.deepcopy(agent.cfg);cfg.resume_run_dir=cfg.log_dir.parent
    cfg.study7.method_id='BR07';cfg.branch7.branch_prior_std=2.
    saved=serialize.loads_json(serialize.dumps_json(agent.journal),Journal)
    with pytest.raises(ValueError,match='Incompatible'):AgentSearch('ignored',cfg,saved)


def test_d06_d07_primary_fusion_aggregation_and_draft_debug_lineage():
    mapping=RootLineages('BR05')
    d1=record('d1',value=None,closed=False)
    repaired=record('repair',parent=d1.child_node_id,operator='debug',value=-1.5)
    d2=record('d2',value=-1.7)
    mapping.sync([d1,repaired,d2])
    first=mapping.nodes[repaired.child_node_id].copy()
    assert first==mapping.nodes[d1.child_node_id]
    fusion=record('fusion',parent=repaired.child_node_id,operator='fusion',parent_value=-1.5,value=-1.4,
                  state={'study7':{'root_lineage_id':first['root_lineage_id']}})
    fusion=replace(fusion,reference_node_ids=(d2.child_node_id,))
    aggregation=record('aggregation',operator='aggregation',value=-1.6)
    mapping.sync([d1,repaired,d2,fusion,aggregation])
    assert mapping.nodes[fusion.child_node_id]==first
    assert mapping.nodes[aggregation.child_node_id]!=first
    before=copy.deepcopy(mapping.nodes)
    mapping.sync([d1,repaired,d2,fusion,aggregation,record('later',value=-1.2)])
    assert all(mapping.nodes[key]==value for key,value in before.items())
    with pytest.raises(Study7Error):mapping.sync([record('cycle',parent='cyclechild',operator='debug')])


def test_br06_numerical_failure_stops_without_gp6_fallback(factory,monkeypatch):
    from engine.parent_selectors.agent_selector import AgentSelectionError
    build,_=factory;agent,execute=build(BRANCH7_PRESETS[5]);drafts=initialize(agent,execute)
    def fail(*args,**kwargs):raise Study7Error('injected illegal matrix')
    monkeypatch.setattr(agent.phase1_runtime.surrogate,'covariance',fail)
    with pytest.raises(AgentSelectionError,match='illegal matrix'):
        agent.phase1_runtime.select_continuation_parent(agent,drafts[0])
    failures=agent.phase1_runtime.artifact_root/'study7_failures.jsonl'
    assert 'blocked/numerical_failure' in failures.read_text()
    assert not (agent.phase1_runtime.artifact_root/'fallbacks.jsonl').exists()


@pytest.mark.parametrize('setting',[BRANCH7_PRESETS[4],BRANCH7_PRESETS[5]])
def test_r09_real_run_py_double_worker_entrypoint(setting,factory,tmp_path,monkeypatch):
    import run as runner
    from tasks.autoresearch.adapter import AutoresearchInterpreter
    build,_=factory;initial,execute=build(setting);cfg=initial.cfg
    shutil.rmtree(cfg.log_dir);shutil.rmtree(cfg.workspace_dir)
    cfg.agent.steps=5;cfg.log_dir.mkdir(parents=True)
    seen=[]
    def create(*args,**kwargs):
        agent=AgentSearch(*args,**kwargs);seen.append(agent);return agent
    monkeypatch.setattr(runner,'load_cfg',lambda:cfg)
    monkeypatch.setattr(runner,'Agent',create)
    monkeypatch.setattr(AutoresearchInterpreter,'_run_trial',lambda self,code,node_id,slot:execute(code,node_id))
    monkeypatch.setenv('MLEVOLVE_SEGMENTED_RUN','1')
    monkeypatch.setenv('MLEVOLVE_SEGMENT_CHECKPOINT',str(tmp_path/'checkpoints'))
    monkeypatch.setenv('MLEVOLVE_SEGMENT_PAUSE_FILE',str(tmp_path/'pause'))
    monkeypatch.setenv('MLEVOLVE_SEGMENT_STARTED',str(time.time()))
    monkeypatch.setenv('MLEVOLVE_SEGMENT_ACTIVE_OFFSET','100')
    runner.run()
    agent=seen[0];runtime=agent.phase1_runtime
    assert len(runtime.transition_store.records())==5
    assert agent.scfg.parallel_search_num==2 and not runtime.root_draft_state['pending']
    assert agent.journal.elapsed_active_seconds>=100
    outcome=json.loads((cfg.log_dir/'segment_outcome.json').read_text())
    assert outcome['state']=='finished' and outcome['completed']==5
    assert len(runtime.surrogate.rows)>= (4 if runtime.branch7_gp6_enabled else 2)
