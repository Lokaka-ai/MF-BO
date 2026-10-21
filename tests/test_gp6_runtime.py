"""Native production-agent integration tests; needs the normal agent dependencies.

Reuses the repository's existing fixture: only LLM transport, grading HTTP and
external code execution are mocked. Runtime, generation dispatch, parser,
validation, journal serialization and resume run their production code.
"""
import copy
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

import test_root_draft_runtime as baseline
from test_root_draft_runtime import real_agent
from engine.agent_search import AgentSearch
from engine.search_node import Journal, SearchNode
from engine.parent_selectors.proposal_feedback import effective_proposals
from engine import evaluation
from utils import serialize
from utils.metric import MetricValue


@pytest.fixture
def gp_factory(real_agent, monkeypatch):
    original = baseline._load_cfg
    preset_path = Path(__file__).resolve().parents[1] / 'config/experiments/gp6.yaml'
    def loader(*args, **kwargs):
        return OmegaConf.merge(original(*args, **kwargs), OmegaConf.load(preset_path))
    monkeypatch.setattr(baseline, '_load_cfg', loader)
    def build(enabled=False, **kwargs):
        return real_agent(selector='gp_child_ucb', enabled=enabled, **kwargs)
    return build


def initialize(agent, execute):
    node = agent.step(None, execute, execute_immediately=False)
    assert node.stage == 'draft'
    agent.execute_deferred_node(node, execute)
    return node


@pytest.mark.parametrize('enabled', [False, True])
def test_gp_real_dispatch_and_six_feature_training(gp_factory, enabled):
    agent, execute, _ = gp_factory(enabled=enabled)
    first = initialize(agent, execute)
    child = agent.step(first, execute)
    r = agent.phase1_runtime
    assert r.gp_enabled and r.managed_decisions
    assert not r.selectors['gp_child_ucb'].surrogate.fit_diagnostics.get('debug_repairs')
    assert r.surrogate.x.shape[1] == 6
    assert child.stage == 'improve'
    assert r.surrogate.training_transition_ids == [r.transition_store.records()[0].transition_id]
    assert not child.phase1_fallback_used
    assert r.root_draft_state['pending'] == {}
    assert r.transition_store.records()[0].metadata['target_available']


def test_gp_root_candidate_can_be_selected_and_reservation_released(gp_factory, monkeypatch):
    agent, execute, _ = gp_factory(enabled=True, cap=3)
    first = initialize(agent, execute)
    r=agent.phase1_runtime
    def prefer_draft(options):
        mean=np.array([.9 if o.operator=='draft' else .1 for o in options])
        return mean,np.zeros(len(options)),np.full(len(options),.1)
    monkeypatch.setattr(r.surrogate,'predict_quality',prefer_draft)
    context=r.select_continuation_parent(agent,first)
    assert context.option.operator=='draft'
    assert r._root_legality(agent)=='draft_pending'
    assert not any(o.operator=='draft' for o in r._add_root_option(agent,[]))
    r.cancel_reservation(agent,context)
    assert r._root_legality(agent)=='legal'
    second=agent.step(first,execute)
    assert second.stage=='draft' and second.parent is agent.virtual_root
    assert second.branch_id!=first.branch_id
    assert r.root_draft_state['draft_submitted']==2


@pytest.mark.parametrize('enabled', [False, True])
def test_initial_draft_debug_repair_returns_to_original_proposal(gp_factory, enabled):
    agent,execute,services=gp_factory(enabled=enabled)
    services.fail='validation'
    first=initialize(agent,execute)
    assert first.is_buggy or first.is_valid is False
    services.fail=None
    repaired=agent.step(first,execute)
    assert repaired.stage=='debug'
    rows=agent.phase1_runtime.transition_store.records()
    proposals=effective_proposals(rows)
    assert len(proposals)==1 and proposals[0].operator=='draft' and proposals[0].valid
    assert proposals[0].selection_features==rows[0].selection_features
    assert proposals[0].total_wall_time_sec==sum(r.total_wall_time_sec for r in rows)


@pytest.mark.parametrize('enabled', [False, True])
def test_gp_generation_failure_is_counted_once(gp_factory, enabled):
    agent,execute,services=gp_factory(enabled=enabled)
    services.fail='generation'
    with pytest.raises(RuntimeError):agent.step(None,execute)
    r=agent.phase1_runtime
    assert r.root_draft_state['initial_terminal']==1
    assert r.root_draft_state['submitted_count']==1
    assert r.root_draft_state['pending']=={}
    assert len(r.transition_store.records())==1
    assert not r.transition_store.records()[0].valid
    assert r.transition_store.records()[0].metadata['proposal_closed']


@pytest.mark.parametrize('enabled', [False, True])
def test_failed_gp_draft_leaves_room_for_a_generated_node(gp_factory, enabled):
    agent, execute, services = gp_factory(enabled=enabled, cap=1)
    runtime = agent.phase1_runtime
    services.fail = 'generation'
    with pytest.raises(RuntimeError, match='external generation failure'):
        agent.step(None, execute)
    assert not agent.virtual_root.children
    assert not runtime._draft_cap_reached(agent)

    services.fail = None
    recovered = initialize(agent, execute)
    assert recovered.stage == 'draft' and recovered.is_valid
    assert runtime._draft_cap_reached(agent)
    assert runtime.root_draft_state['draft_submitted'] == 2
    assert runtime.root_draft_state['submitted_count'] == 2
    assert len(agent.virtual_root.children) == 1

    agent.capture_resume_state()
    resumed = AgentSearch('ignored', copy.deepcopy(agent.cfg),
        serialize.loads_json(serialize.dumps_json(agent.journal), Journal))
    assert resumed.phase1_runtime._draft_cap_reached(resumed)


def test_gp_draft_cap_counts_failed_nodes_and_inflight_but_not_fusion(gp_factory):
    agent, execute, services = gp_factory(cap=2)
    services.fail = 'validation'
    first = initialize(agent, execute)
    assert first.is_valid is False
    runtime, root = agent.phase1_runtime, agent.virtual_root
    assert not runtime._draft_cap_reached(agent)

    root.add_expected_child_count()
    SearchNode(code='pass', parent=root, stage='fusion_draft')
    assert not runtime._draft_cap_reached(agent)
    root.add_expected_child_count()
    assert runtime._draft_cap_reached(agent)
    root.sub_expected_child_count()
    assert not runtime._draft_cap_reached(agent)


@pytest.mark.parametrize('enabled', [False, True])
def test_gp_reference_and_predictions_survive_journal_resume(gp_factory, enabled):
    agent,execute,_=gp_factory(enabled=enabled)
    first=initialize(agent,execute)
    child=agent.step(first,execute)
    r=agent.phase1_runtime
    r.surrogate.fit(r.transition_store.records())
    agent.capture_resume_state()
    original=r.surrogate.predict_quality([r._make_option(agent,child)])
    resumed=AgentSearch('ignored',copy.deepcopy(agent.cfg),serialize.loads_json(serialize.dumps_json(agent.journal),Journal))
    rr=resumed.phase1_runtime
    rr.surrogate.fit(rr.transition_store.records())
    restored=next(n for n in resumed.journal.nodes if n.id==child.id)
    np.testing.assert_allclose(original,rr.surrogate.predict_quality([rr._make_option(resumed,restored)]))
    assert rr.surrogate.state_dict()==r.surrogate.state_dict()
    assert rr.root_draft_state['pending']=={}


def test_valid_gp_node_not_terminated_by_old_soft_failure_streak(gp_factory):
    agent,execute,_=gp_factory()
    parent=initialize(agent,execute)
    parent.improve_failure_depth=100
    node=SearchNode(code='pass',parent=parent,stage='improve',metric=MetricValue(.5,maximize=True),
                    is_buggy=False,is_valid=True,local_best_node=parent)
    assert evaluation.check_improvement(agent,node,parent) is False
    assert node.continue_improve and not node.is_terminal
