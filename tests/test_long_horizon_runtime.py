"""Native AgentSearch integration (requires the project's normal dependencies).

Only external generation, HTTP grading and code execution use the existing
real_agent fixture's mocks. No LLM API or GPU experiment is launched.
"""
import copy
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

import test_root_draft_runtime as baseline
from test_root_draft_runtime import real_agent
from engine.agent_search import AgentSearch
from engine.search_node import Journal
from engine.parent_selectors.gp_long_horizon import LONG_HORIZON_SELECTOR as LONG
from utils import serialize


@pytest.fixture
def long_factory(real_agent, monkeypatch):
    original = baseline._load_cfg
    preset = Path(__file__).resolve().parents[1] / 'config/experiments/gp7_long_horizon_ucb.yaml'
    monkeypatch.setattr(baseline, '_load_cfg', lambda *a, **kw:
                        OmegaConf.merge(original(*a, **kw), OmegaConf.load(preset)))
    return lambda: real_agent(selector=LONG, enabled=False)


def initialize(agent, execute):
    node = agent.step(None, execute, execute_immediately=False)
    agent.execute_deferred_node(node, execute)
    return node


def test_native_delay_and_long_gp_selection(long_factory):
    agent, execute, services = long_factory()
    parent = initialize(agent, execute)
    services.metric = .49
    parent = agent.step(parent, execute)
    r = agent.phase1_runtime
    r.surrogate.fit(r.transition_store.records())
    assert len(r.surrogate.training_transition_ids) == 0
    for i in range(5):
        services.metric = .52 + .01*i
        parent = agent.step(parent, execute)
        assert not parent.phase1_fallback_used
    r.surrogate.fit(r.transition_store.records())
    assert r.surrogate.x.shape[1] == 7
    windows = r.surrogate.fit_diagnostics['training_windows']
    assert windows and all(len(w['member_transition_ids']) == 2 for w in windows)
    members = [m for w in windows for m in w['member_transition_ids']]
    assert len(members) == len(set(members))
    assert r.root_draft_state['pending'] == {}
    assert r.root_draft_state['long_horizon'] == {'horizon': 2}


def test_native_debug_is_one_budget_unit(long_factory):
    agent, execute, services = long_factory()
    seed = initialize(agent, execute)
    services.fail = 'validation'
    bad = agent.step(seed, execute)
    state = dict(bad.phase1_selection_features)
    services.fail = None
    services.metric = .51
    repaired = agent.step(bad, execute)
    assert repaired.stage == 'debug'
    r = agent.phase1_runtime
    r.surrogate.fit(r.transition_store.records())
    assert len(r.surrogate.training_transition_ids) == 0
    services.metric = .55
    agent.step(repaired, execute)
    r.surrogate.fit(r.transition_store.records())
    assert len(r.surrogate.training_transition_ids) == 1
    w = r.surrogate.fit_diagnostics['training_windows'][0]
    assert w['origin_transition_id'] == bad.phase1_transition_id
    origin = r.transition_store.by_child_id(bad.id)
    assert origin.selection_features == state
    assert len(w['member_transition_ids']) == 2


def test_native_resume_rebuilds_labels_and_rejects_changed_h(long_factory):
    agent, execute, services = long_factory()
    parent = initialize(agent, execute)
    for i in range(4):
        services.metric = .52 + .01*i
        parent = agent.step(parent, execute)
    r = agent.phase1_runtime
    r.surrogate.fit(r.transition_store.records())
    expected = r.surrogate.predict_quality([r._make_option(agent, parent)])
    windows = copy.deepcopy(r.surrogate.fit_diagnostics['training_windows'])
    agent.capture_resume_state()
    serialized = serialize.dumps_json(agent.journal)
    resumed = AgentSearch('ignored', copy.deepcopy(agent.cfg), serialize.loads_json(serialized, Journal))
    rr = resumed.phase1_runtime
    rr.surrogate.fit(rr.transition_store.records())
    restored = next(n for n in resumed.journal.nodes if n.id == parent.id)
    np.testing.assert_allclose(rr.surrogate.predict_quality([rr._make_option(resumed, restored)]), expected)
    assert rr.surrogate.fit_diagnostics['training_windows'] == windows
    altered = copy.deepcopy(agent.cfg)
    altered.bo_parent_selector.gp.long_horizon.horizon = 3
    with pytest.raises(ValueError, match='checkpoint|contract|horizon'):
        AgentSearch('ignored', altered, serialize.loads_json(serialized, Journal))
