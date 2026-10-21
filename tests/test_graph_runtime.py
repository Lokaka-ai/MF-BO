"""Native AgentSearch A/B tests (require the repository's normal dependencies).

Only external LLM transport, HTTP grading and code execution are mocked by the
existing real_agent fixture. Run with pytest tests/test_graph_runtime.py.
"""
import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

import test_root_draft_runtime as baseline
from test_root_draft_runtime import real_agent
from engine.agent_search import AgentSearch
from engine.search_node import Journal
from engine.parent_selectors.proposal_feedback import effective_proposals
from utils import serialize

A,B='gp7_ancestry_logei','laplacian_anchor_logei'
A_PRESET='gp7_ancestry_ucb'

@pytest.fixture(params=[A,B])
def graph_factory(request,real_agent,monkeypatch):
    mode=request.param
    original=baseline._load_cfg
    preset_name=A_PRESET if mode==A else B
    preset=Path(__file__).resolve().parents[1]/f'config/experiments/{preset_name}.yaml'
    monkeypatch.setattr(baseline,'_load_cfg',lambda *a,**kw:OmegaConf.merge(original(*a,**kw),OmegaConf.load(preset)))
    def build():return real_agent(selector=mode,enabled=False)
    return build


def initialize(agent,execute):
    node=agent.step(None,execute,execute_immediately=False)
    agent.execute_deferred_node(node,execute)
    return node


@pytest.mark.parametrize('preset', [A_PRESET, 'gp7_long_horizon_ucb', 'gp7_long_horizon_ucb_pibo', 'gp7_short_ucb_pibo'])
def test_configured_gp7_ucb_and_prior_reach_native_selection(real_agent, monkeypatch, preset):
    original = baseline._load_cfg
    path = Path(__file__).resolve().parents[1] / f'config/experiments/{preset}.yaml'
    overrides = OmegaConf.load(path)
    monkeypatch.setattr(baseline, '_load_cfg', lambda *a, **kw:
        OmegaConf.merge(original(*a, **kw), overrides))
    agent, execute, _ = real_agent(selector=overrides.parent_selector.type, enabled=False)
    parent = initialize(agent, execute)
    parent.agent_potential_level = 2
    child = agent.step(parent, execute)
    runtime = agent.phase1_runtime
    assert not child.phase1_fallback_used
    assert agent.cfg.evaluation.enabled is False
    assert runtime.root_draft_state['acquisition'] == 'child_ucb'
    snapshot = json.loads((runtime.artifact_root / 'selection_snapshots.jsonl').read_text().splitlines()[-1])
    ucb = snapshot['posterior_mean_quality'] + 1.5 * snapshot['posterior_std_latent']
    assert snapshot['predicted_child_ucb'] == pytest.approx(ucb)
    expected = ucb
    if preset.endswith('_pibo'):
        expected += snapshot['utility_scale'] * overrides.bo_parent_selector.agent_rating.beta_prior / snapshot['pibo_iteration'] * math.log(2)
        assert snapshot['acquisition_kind'] == 'pibo_child_ucb'
    else:
        assert snapshot['acquisition_kind'] == 'child_ucb'
    assert snapshot['acquisition_score'] == pytest.approx(expected)


def test_native_normal_dispatch_and_seven_inputs(graph_factory):
    agent,execute,services=graph_factory();parent=initialize(agent,execute)
    for i in range(6):
        services.metric=.5+.002*(i+1)
        parent=agent.step(parent,execute)
        assert not parent.phase1_fallback_used
        assert parent.stage in ('improve','evolution','fusion')
    r=agent.phase1_runtime;r.surrogate.fit(r.transition_store.records())
    assert r.surrogate.x.shape[1]==7
    assert len(r.surrogate.training_transition_ids)==6
    assert len(effective_proposals(r.transition_store.records()))==7
    assert r.root_draft_state['pending']=={}


def test_native_debug_preserves_original_graph_anchor(graph_factory):
    agent,execute,services=graph_factory();seed=initialize(agent,execute)
    services.fail='validation';bad=agent.step(seed,execute)
    original=dict(bad.phase1_selection_features)
    assert not bad.is_valid
    services.fail=None;services.metric=.55
    repaired=agent.step(bad,execute)
    assert repaired.stage=='debug'
    r=agent.phase1_runtime;r.surrogate.fit(r.transition_store.records())
    ordinary=[p for p in effective_proposals(r.transition_store.records()) if p.operator!='draft']
    assert len(ordinary)==1 and ordinary[0].valid
    assert ordinary[0].selection_features==original
    assert r.surrogate.training_transition_ids==[ordinary[0].transition_id]
    assert r.graph_context is not None
    option=r._make_option(agent,repaired)
    assert option.state_features['gp_primary_ancestry']==[repaired.id,seed.id]


def test_native_journal_resume_keeps_nested_ancestry_and_predictions(graph_factory):
    agent,execute,services=graph_factory();parent=initialize(agent,execute)
    for i in range(3):
        services.metric=.52+.01*i;parent=agent.step(parent,execute)
    r=agent.phase1_runtime;r.surrogate.fit(r.transition_store.records())
    expected=r.surrogate.predict_quality([r._make_option(agent,parent)])
    agent.capture_resume_state()
    journal=serialize.loads_json(serialize.dumps_json(agent.journal),Journal)
    resumed=AgentSearch('ignored',copy.deepcopy(agent.cfg),journal)
    rr=resumed.phase1_runtime;rr.surrogate.fit(rr.transition_store.records())
    restored=next(n for n in resumed.journal.nodes if n.id==parent.id)
    assert restored.phase1_selection_features==parent.phase1_selection_features
    np.testing.assert_allclose(rr.surrogate.predict_quality([rr._make_option(resumed,restored)]),expected)
    assert rr.surrogate.state_dict()==r.surrogate.state_dict()
