"""Research policy mathematics and production dispatch without paid model calls."""
import copy
import json
import time

import numpy as np
import pytest
from numpy.testing import assert_allclose

from engine.parent_selectors.research_bo import (CodeGP, VARIANTS, code_features,
    mix_joint, outcome_ei, restart_due)
from engine.parent_selectors.acq7_gp import AcqGP
from engine.parent_selectors.study7_gp import Study7Error
from config.acq7 import ACQ7_PRESETS
from test_study7_runtime import factory, initialize
import test_study7_runtime as fixtures


def build_variant(factory, monkeypatch, variant):
    original = fixtures.load_setting
    def setting(name):
        cfg = original(name)
        cfg.research_bo = variant
        cfg.agent.steps = 27
        cfg.agent.initial_drafts = 3
        cfg.agent.search.num_drafts = 6 if variant in {"frontier", "portfolio"} else 3
        return cfg
    monkeypatch.setattr(fixtures, "load_setting", setting)
    build, calls = factory
    return (*build(ACQ7_PRESETS[3], shared=False), calls)


def test_ei_matches_monte_carlo_and_zero_variance():
    means = np.array([-1., 0., 1.])
    variance = np.array([.5, 2., 0.])
    exact = outcome_ei(means, variance, .3)
    rng = np.random.default_rng(7)
    samples = rng.normal(means, np.sqrt(variance), size=(500000, 3))
    assert_allclose(exact, np.maximum(samples - .3, 0).mean(0), atol=.003)
    assert exact[2] == .7
    # A high-variance, lower-mean generator can have greater expected record gain.
    assert outcome_ei(np.array([10., 9.]), np.array([.01, 16.]), 10.)[1] > .04


def test_restart_budget_and_pending_boundaries():
    state = dict(pending={}, submitted_count=10, initial_goal=3, draft_submitted=3, draft_cap=6)
    assert restart_due(state)
    assert not restart_due({**state, "submitted_count": 9})
    assert not restart_due({**state, "draft_cap": 3})
    assert not restart_due({**state, "pending": {"x": {"status": "executing", "option": {"operator": "draft"}}}})
    # A reserved ordinary action consumes the scheduled slot before submission.
    assert restart_due({**state, "submitted_count": 9, "pending": {"x": {
        "status": "reserved", "option": {"operator": "improve"}}}})


def test_code_features_ignore_renaming_literals_and_preserve_structure():
    a = code_features('def f(x):\n return x + 3')
    b = code_features('def g(y):\n return y + 99')
    c = code_features('def g(y):\n for i in y:\n  yield i')
    assert a == b and a != c
    assert_allclose(np.linalg.norm(a), 1.)
    with pytest.raises(Study7Error): code_features('this is ! invalid')


@pytest.mark.parametrize('variant', VARIANTS)
def test_production_dispatch_feedback_and_resume(factory, monkeypatch, variant):
    from engine.agent_search import AgentSearch
    from engine.search_node import Journal
    from utils import serialize
    agent, execute, calls = build_variant(factory, monkeypatch, variant)
    drafts = initialize(agent, execute)
    for n in drafts: n.lock = False
    runtime = agent.phase1_runtime
    selector = runtime.study7_selector
    assert selector.contract()['research_bo'] == variant
    context = runtime.select_global_parent(agent)
    agent.step(None, execute, recovery_context=context)
    assert len(selector.gp.targets) == 1
    if variant == 'portfolio':
        assert len(selector.scored) == 1
    # Identical legal support, predictions and policy after restoring a real ledger.
    for n in agent.journal.nodes: n.lock = False
    agent.capture_resume_state()
    saved = serialize.loads_json(serialize.dumps_json(agent.journal), Journal)
    options = runtime._normal_options(agent)
    before = selector.select(agent, options)
    prepared = copy.deepcopy(selector._prepared)
    resumed = AgentSearch('ignored', agent.cfg, saved)
    restored = resumed.phase1_runtime
    after = restored.study7_selector.select(resumed, restored._normal_options(resumed))
    assert before.selected_option.parent_node_id == after.selected_option.parent_node_id
    assert prepared['payload']['candidate_order'] == restored.study7_selector._prepared['payload']['candidate_order']
    assert_allclose(prepared['joint']['mean'], restored.study7_selector._prepared['joint']['mean'])
    assert_allclose(prepared['joint']['latent_covariance'], restored.study7_selector._prepared['joint']['latent_covariance'])
    assert prepared['payload']['acquisition'] == restored.study7_selector._prepared['payload']['acquisition']
    assert selector.forecasts == restored.study7_selector.forecasts
    assert_allclose(selector.expert_loss, restored.study7_selector.expert_loss)
    evidence = json.loads((runtime.artifact_root/'study7_decisions.jsonl').read_text().splitlines()[-1])
    assert evidence['research_bo']['variant'] == variant
    if variant == 'code_kernel':
        rows = selector.gp.rows
        extra = selector.gp.covariance(rows, rows)[0] - AcqGP.covariance(selector.gp, rows, rows)[0]
        assert np.linalg.eigvalsh(extra).min() >= -1e-9
        assert extra.diagonal().min() > 0


@pytest.mark.parametrize('variant', ['frontier', 'portfolio'])
def test_frontier_fresh_draft_uses_native_dispatch_and_quota(factory, monkeypatch, variant):
    agent, execute, _ = build_variant(factory, monkeypatch, variant)
    drafts = initialize(agent, execute)
    runtime = agent.phase1_runtime
    selector = runtime.study7_selector
    for n in drafts: n.lock = False
    options = runtime._normal_options(agent)
    rows = [o.state_features['study7'] for o in options]
    # Counterfactual *test-only* imbalance: fourth reservation must cover untouched roots.
    selector.dispatched_by_branch = {rows[0]['root_lineage_id']: 3}
    result = selector.select(agent, options)
    assert result.selected_option.state_features['study7']['root_lineage_id'] != rows[0]['root_lineage_id']
    assert selector._prepared['payload']['acquisition']['coverage_reservation']
    # Exercise real scheduler branch and generation/admission of an extra root.
    runtime.root_draft_state['submitted_count'] = 10
    ctx = runtime.select_global_parent(agent)
    assert ctx.option.operator == 'draft'
    assert ctx.decision.selection_reason == 'research_frontier_restart'
    child = agent.step(None, execute, recovery_context=ctx)
    assert child.is_valid
    assert runtime.root_draft_state['draft_submitted'] == 4
    assert len(selector.state.lineages.roots) == 4


def test_mixture_keeps_disagreement_variance():
    from engine.parent_selectors.acq7_gp import psd_covariance
    def joint(m):
        cov, root, diag = psd_covariance(np.eye(2))
        return dict(mean=np.array(m), latent_covariance=cov, latent_square_root=root,
                    noise_variance=2., posterior_hash=str(m), details=[{},{}], diagnostics=diag)
    result = mix_joint([joint([0.,0.]), joint([2.,2.])], [.5,.5])
    assert_allclose(result['mean'], [1.,1.])
    assert_allclose(result['latent_covariance'], [[2.,1.],[1.,2.]])
    assert_allclose(result['predictive_sd']**2, [4.,4.])
    assert not result['exact_gp']

@pytest.mark.parametrize('variant', VARIANTS)
def test_fifteen_real_steps_no_network(factory, monkeypatch, variant):
    agent, execute, _ = build_variant(factory, monkeypatch, variant)
    initialize(agent, execute)
    for _ in range(12):
        for node in agent.journal.nodes: node.lock = False
        agent.step(None, execute)
    runtime = agent.phase1_runtime
    selector = runtime.study7_selector
    assert runtime.root_draft_state['submitted_count'] == 15
    assert runtime.root_draft_state['draft_submitted'] == (4 if variant in {'frontier','portfolio'} else 3)
    decisions = [json.loads(line) for line in (runtime.artifact_root/'study7_decisions.jsonl').read_text().splitlines()]
    assert len(decisions) == (11 if variant in {'frontier','portfolio'} else 12)
    assert all(d['research_bo']['variant'] == variant for d in decisions)
    if variant == 'portfolio':
        assert any(d['acquisition_kind'] == 'outcome_ei' for d in decisions)
        assert all(min(d['research_bo']['expert_weights']) >= .1 for d in decisions)
        loss = selector.expert_loss.copy()
        selector.observe(selector.records_provider())
        assert_allclose(loss, selector.expert_loss)  # feedback cannot be counted twice
        assert len(selector.scored) == 11
    snapshot = selector.state_dict()
    snapshot['contract']['research_bo'] = 'another_variant'
    with pytest.raises(Study7Error): selector.load_state_dict(snapshot)
