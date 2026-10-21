"""V2: production scheduling/resume plus causal feedback and prompt boundaries."""
import copy
import json
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest
from numpy.testing import assert_allclose

from engine.parent_selectors.audited_bo import (METHODS, descriptor, draft_cap, ordered_closed,
    plateau_state, representation_prompt, rewrite_due)
from engine.conditions import is_branch_stagnant, is_globally_stagnant
from engine.research_termination import TerminationReceipt
from config.acq7 import ACQ7_PRESETS
from test_study7_runtime import factory, initialize
import test_study7_runtime as fixtures
from tests_gp6.test_core import record


def build_variant(factory, monkeypatch, variant):
    original = fixtures.load_setting
    def setting(name):
        cfg = original(name)
        cfg.audited_bo = variant
        cfg.agent.steps = 15
        cfg.agent.initial_drafts = 3
        cfg.agent.search.num_drafts = draft_cap(15) if variant == 'representation_frontier' else 3
        return cfg
    monkeypatch.setattr(fixtures, 'load_setting', setting)
    build, calls = factory
    return (*build(ACQ7_PRESETS[int(METHODS[variant][-2:])-1], shared=False), calls)


@pytest.mark.parametrize('variant', [v for v in METHODS if v not in {'preserved_evolution', 'redesign'}])
def test_production_multistep_and_resume(factory, monkeypatch, variant):
    from engine.agent_search import AgentSearch
    from engine.search_node import Journal
    from utils import serialize
    agent, execute, calls = build_variant(factory, monkeypatch, variant)
    initialize(agent, execute)
    # Enough real native steps to exercise scheduled rewrite, ties, evolution,
    # AQ03 review schema, AQ07 acquisition and durable reservation/recovery.
    for _ in range(12):
        for node in agent.journal.nodes: node.lock = False
        agent.step(None, execute)
    runtime = agent.phase1_runtime
    selector = runtime.study7_selector
    assert runtime.root_draft_state['submitted_count'] == 15
    assert runtime.root_draft_state['draft_submitted'] == (4 if variant == 'representation_frontier' else 3)
    assert selector.contract()['audited_bo'] == variant
    assert selector.config.rating.enabled == (variant != 'rating_free')
    for review in calls.feedbacks:
        assert ('potential_level' in review['func_spec'].json_schema['properties']) == (variant != 'rating_free')
    assert selector.config.acquisition_type == ('retained_outcome_kg' if variant == 'retained_kg' else 'joint_ts')
    if variant in {'plateau', 'representation_frontier'}:
        assert is_globally_stagnant(agent)
        assert any(r.operator == 'evolution' for r in runtime.transition_store.records())
    for node in agent.journal.nodes: node.lock = False
    agent.capture_resume_state()
    saved = serialize.loads_json(serialize.dumps_json(agent.journal), Journal)
    resumed = AgentSearch('ignored', agent.cfg, saved)
    assert plateau_state(agent) == plateau_state(resumed)
    # Use eligible ordinary support directly (full workflow budget is spent).
    before = selector.select(agent, runtime._normal_options(agent))
    prepared = copy.deepcopy(selector._prepared)
    other = resumed.phase1_runtime.study7_selector
    after = other.select(resumed, resumed.phase1_runtime._normal_options(resumed))
    assert before.selected_option.parent_node_id == after.selected_option.parent_node_id
    assert prepared['payload']['candidate_order'] == other._prepared['payload']['candidate_order']
    assert_allclose(prepared['joint']['mean'], other._prepared['joint']['mean'])
    left, right = copy.deepcopy(prepared['payload']['acquisition']), copy.deepcopy(other._prepared['payload']['acquisition'])
    left.pop('wall_seconds', None); right.pop('wall_seconds', None)
    assert left == right
    assert prepared['payload']['audited_bo'] == other._prepared['payload']['audited_bo']
    bad = selector.state_dict()
    bad['contract']['audited_bo'] = 'wrong'
    with pytest.raises(Exception): other.load_state_dict(bad)


def feedback_rows(values, maximize=True):
    rows = []
    for i, value in enumerate(values):
        row = record(str(i), parent='parent', operator='improve', value=0)
        rows.append(replace(row, child_node_id='child'+str(i), valid=value is not None,
                            child_metric_raw=value, child_utility=(value if maximize else -value) if value is not None else None,
                            timestamp=str(10-i), metadata={'proposal_closed': True}))
    return rows


def fake_agent(rows):
    return NS(cfg=NS(audited_bo='plateau'), scfg=NS(use_stagnation_detection=True),
              stagnation_threshold=3, journal=NS(nodes=[NS(id='parent', branch_id=1)]),
              phase1_runtime=NS(transition_store=NS(records=lambda: rows)))


@pytest.mark.parametrize('maximize,values', [(True,[348,348,348,348]), (False,[3,3,3,3])])
def test_plateau_ties_records_invalid_pending(maximize, values):
    rows = feedback_rows(values, maximize)
    agent = fake_agent(rows)
    assert is_branch_stagnant(agent, 1, 3)
    assert is_globally_stagnant(agent)
    rows[-1] = replace(rows[-1], child_utility=(349 if maximize else -2))
    assert not is_branch_stagnant(agent, 1, 3)
    assert plateau_state(agent)['stale'] == 0
    rows.extend(feedback_rows([None])[0:1])  # replace identity to avoid duplicate admission
    rows[-1] = replace(rows[-1], transition_id='invalid', decision_id='invalid')
    assert plateau_state(agent)['stale'] == 1
    rows[-1] = replace(rows[-1], metadata={'proposal_closed':False})
    assert plateau_state(agent)['stale'] == 0
    rows[-1] = replace(rows[-1], metadata={'proposal_closed':True,'censored':True})
    assert plateau_state(agent)['stale'] == 0


def test_debug_is_one_proposal_at_final_admission():
    a,b,c = feedback_rows([10,None,11])
    b = replace(b, metadata={'proposal_closed':False})
    repair = replace(b, transition_id='repair', decision_id='repair', operator='debug',
                     parent_node_id=b.child_node_id, child_node_id='fixed', child_utility=10,
                     valid=True, metadata={'proposal_closed':True})
    rows = [a,b,c,repair]
    folded = ordered_closed(rows)
    assert [r.metadata['outcome_transition_id'] for r in folded] == [a.transition_id,c.transition_id,'repair']
    assert plateau_state(fake_agent(rows))['stale'] == 1


def test_descriptor_and_fresh_context_boundary():
    code = '# BO_REPRESENTATION: discrete\n# BO_SEARCH: local_search\nprint(1)'
    assert descriptor(code) == 'discrete/local_search'
    assert descriptor(code+'\n# BO_SEARCH: greedy') is None
    assert descriptor(code.replace('discrete','unknown')) is None
    prompt = {'Introduction':'task', 'Memory':'SECRET_PARENT_CODE_AND_PLAN', 'Instructions':{}}
    representation_prompt(fake_agent([]), prompt, fresh=True)
    assert 'SECRET_PARENT' not in str(prompt)
    assert 'full rewrite' in str(prompt)
    assert draft_cap(12) == 3 and draft_cap(13) == 4
    state = dict(pending={},submitted_count=10,initial_goal=3,draft_submitted=3,draft_cap=4)
    assert rewrite_due(state, steps=15)
    state['pending']['x'] = {'status':'reserved','option':{'operator':'draft'}}
    assert not rewrite_due(state, steps=15)


def test_task_correct_rating_context():
    from agents.study7_rating import extend_review
    agent = NS(cfg=NS(audited_bo='task_correct', task_type='alphaevolve', cpu_number=6,exec=NS(timeout=120)),
               scfg=NS(parallel_search_num=3,max_debug_depth=3),
               acfg=NS(time_limit=7200,seed=42,feedback=NS(model='muse',base_url='test')),task_desc='math')
    node = NS(id='x',stage='improve',code='pass')
    spec = NS(json_schema={'properties':{},'required':[]})
    prompt,_ = extend_review(agent,node,{'Introduction':'review'},spec)
    context = node.study7_rating_metadata['context']
    assert context['gpu_count_per_workflow'] == 0 and context['cpus_per_worker'] == 2
    assert 'training_seconds_per_trial' not in context and 'initialization' not in context
    assert json.loads(prompt['Fixed task and resource protocol for potential assessment']['resources']) == context


def test_termination_all_invalid_is_zero_but_exceptions_are_not(tmp_path,monkeypatch):
    monkeypatch.setenv('BO_TERMINATION_FILE',str(tmp_path/'termination.json'))
    receipt = TerminationReceipt(NS(agent=NS(steps=15)))
    receipt.exhausted('No legal normal, draft, or debug action remains')
    assert receipt.search_finished(3)
    receipt.complete(no_valid=True)
    assert receipt.data['status'] == 'completed' and receipt.data['algorithmic_no_valid']
    receipt.error(RuntimeError())
    assert not receipt.search_finished(15)
    receipt.complete()
    assert receipt.data['status'] == 'blocked'
    receipt.failed(KeyboardInterrupt())
    assert receipt.data['status'] == 'interrupted'


def test_frontier_protected_dispatch_pending_and_rewrite_context(factory, monkeypatch):
    import agents.draft_agent as draft
    agent, execute, calls = build_variant(factory, monkeypatch, 'representation_frontier')
    counter = [0]
    def decorate(original):
        def generate(*args, **kwargs):
            plan, code = original(*args, **kwargs)
            label = ('discrete','continuous','algebraic')[counter[0] % 3]
            counter[0] += 1
            return plan, f'# BO_REPRESENTATION: {label}\n# BO_SEARCH: sampling\n# PARENT_CODE_SENTINEL\n'+code
        return generate
    monkeypatch.setattr(draft, 'stepwise_plan_and_code_query', decorate(draft.stepwise_plan_and_code_query))
    monkeypatch.setattr(draft, 'plan_and_code_query', decorate(draft.plan_and_code_query))
    initial = initialize(agent, execute)
    runtime = agent.phase1_runtime
    selector = runtime.study7_selector
    # In-flight committed reservations consume quota immediately, not only on feedback.
    for node in initial: node.lock = False
    contexts = [runtime.select_global_parent(agent) for _ in range(3)]
    assert len({c.option.parent_node_id for c in contexts}) == 3
    for context in contexts: agent.step(None, execute, recovery_context=context)
    for _ in range(3):
        for node in agent.journal.nodes: node.lock = False
        agent.step(None, execute)
    assert sorted(selector.dispatched_by_branch.values()) == [2,2,2]
    # One ordinary proposal then scheduled fresh draft (post-initial slot8).
    for node in agent.journal.nodes: node.lock = False
    agent.step(None, execute)
    before = len(calls.generations)
    context = runtime.select_global_parent(agent)
    assert context.decision.selection_reason == 'audited_representation_rewrite'
    child = agent.step(None, execute, recovery_context=context)
    assert child.is_valid and child.stage == 'draft'
    prompts = str(calls.generations[before:])
    assert 'PARENT_CODE_SENTINEL' not in prompts
    assert 'fresh full rewrite' in prompts
    assert 'discrete/sampling' in prompts


@pytest.mark.parametrize("mode", ["frontier", "all_invalid", "frontier_entry"])
def test_alphaevolve_frontier_real_worker_and_verifier(factory, monkeypatch, tmp_path, mode):
    """No paid LLM: actual AlphaEvolve interpreter, hashes, grader and final export."""
    import hashlib
    from pathlib import Path
    import shutil
    from unittest.mock import patch
    from omegaconf import OmegaConf
    from config import Config, _load_cfg
    from engine.agent_search import AgentSearch
    from engine.search_node import Journal
    from tasks.alphaevolve.adapter import AlphaEvolveInterpreter
    from tasks.alphaevolve.prepare import initial, contract
    from tasks.alphaevolve.registry import TASKS
    import agents.draft_agent as draft
    _, calls = factory  # external memory/reviewer stubs only
    with patch.object(OmegaConf, 'from_cli', return_value=OmegaConf.create({
            'task_preset':'alphaevolve', 'experiment_preset':ACQ7_PRESETS[3], 'exp_id':'kissing_number_d11', 'exp_name':'alpha_integration'})):
        cfg = OmegaConf.merge(OmegaConf.structured(Config), _load_cfg())
    cfg.audited_bo = 'representation_frontier'
    cfg.agent.steps, cfg.agent.initial_drafts = 15, 3
    cfg.agent.search.num_drafts = draft_cap(15)
    cfg.agent.search.parallel_search_num = 1
    cfg.agent.time_limit = 1000000000  # fixture virtual clock is unrelated to process wall time
    cfg.cpu_number, cfg.exec.timeout = 2, 15
    cfg.log_dir, cfg.workspace_dir = tmp_path/'logs', tmp_path/'work'
    cfg.dataset_dir = tmp_path/'data'
    public = cfg.dataset_dir/'kissing_number_d11/prepared/public'
    public.mkdir(parents=True)
    task = TASKS['kissing_number_d11']
    (public/'problem.json').write_text(json.dumps({'id':task['id'],'contract':contract(task)}))
    (public/'initial.json').write_text(json.dumps(initial(task, {})))
    (public/'description.md').write_text(contract(task))
    manifest = {'source_commit':'test_public_fixture','tasks':{task['id']:{
        str(p.relative_to(cfg.dataset_dir)):hashlib.sha256(p.read_bytes()).hexdigest() for p in public.iterdir()}}}
    (cfg.dataset_dir/'manifest.json').write_text(json.dumps(manifest))
    cfg.data_dir = public
    cfg.log_dir.mkdir(); cfg.workspace_dir.mkdir()
    shutil.copytree(public,cfg.workspace_dir/'input')
    generated = []
    def generate(*args, **kwargs):
        generated.append((args,kwargs))
        if mode == 'all_invalid':
            return 'Invalid test construction.', 'def solve(problem, initial, time_budget, seed):\n    return {\"points\": []}\n'
        return 'Public initial construction for integration testing.', '# BO_REPRESENTATION: discrete\n# BO_SEARCH: enumeration\ndef solve(problem, initial, time_budget, seed):\n    return initial\n'
    monkeypatch.setattr(draft,'stepwise_plan_and_code_query',generate)
    monkeypatch.setattr(draft,'plan_and_code_query',generate)
    if mode in {'all_invalid', 'frontier_entry'}:
        import logging
        import run as entry
        if mode == 'all_invalid':
            cfg.agent.steps, cfg.agent.search.num_drafts = 3, 3
        else:
            import importlib
            cfg.agent.search.parallel_search_num, cfg.cpu_number = 3, 6
            def model(**kwargs):
                if kwargs.get('json_schema'):
                    return json.dumps({'reason':'Keep host API.', 'module':['model_design'],
                                       'plan':{'model_design':'Return the public construction.'}})
                return 'Public construction.\n```python\n'+generate()[1]+'\n```'
            monkeypatch.setattr('agents.coder.base_coder.generate',model)
            monkeypatch.setattr('agents.coder.stepwise_coder.generate',model)
            monkeypatch.setattr('agents.planner.base_planner.generate',model)
            monkeypatch.setattr(importlib.import_module('agents.coder.diff_coder.diff_generate'),'generate',
                lambda **kwargs: '<<<<<<< SEARCH\n    return initial\n=======\n    return initial  # integration\n>>>>>>> REPLACE')
        cfg.run_result_file = str(tmp_path/'run-info.json')
        monkeypatch.setenv('BO_TERMINATION_FILE',str(tmp_path/'termination.json'))
        monkeypatch.setattr(entry,'load_cfg',lambda:cfg)
        monkeypatch.setattr(entry,'load_task_desc',lambda _:contract(task))
        monkeypatch.setattr(entry,'prep_agent_workspace',lambda _:None)
        monkeypatch.setattr(entry,'setup_logging',lambda _:logging.getLogger('MLEvolve'))
        entry.run()
        receipt = json.loads((tmp_path/'termination.json').read_text())
        assert receipt['status'] == 'completed'
        assert receipt['algorithmic_no_valid'] == (mode == 'all_invalid')
        assert receipt['completed_transitions'] == cfg.agent.steps
        if mode == 'frontier_entry':
            records = [json.loads(x) for x in (cfg.log_dir/'phase1/transitions.jsonl').read_text().splitlines()]
            assert sum(row['operator'] == 'draft' for row in records) == 4
            assert all(row['valid'] for row in records)
            assert json.loads((cfg.workspace_dir/'best_solution/final_grade.json').read_text())['metric'] == 22
        return
    agent = AgentSearch(contract(task), cfg, Journal())
    interpreter = AlphaEvolveInterpreter(cfg.workspace_dir, **OmegaConf.to_container(cfg.exec), cfg=cfg)
    try:
        drafts = initialize(agent,interpreter.run)
        assert all(n.is_valid and n.metric.value == 22 for n in drafts)
        # Force the scheduling boundary for a bounded integration check, not a performance run.
        runtime = agent.phase1_runtime
        runtime.root_draft_state['submitted_count'] = 10
        context = runtime.select_global_parent(agent)
        assert context.decision.selection_reason == 'audited_representation_rewrite'
        child = agent.step(None, interpreter.run, recovery_context=context)
        assert child.is_valid and child.metric.value == 22
        agent.alphaevolve.finalize(agent.journal)
        grade = json.loads((cfg.workspace_dir/'best_solution/final_grade.json').read_text())
        assert grade['valid'] and grade['metric'] == 22
        assert all(n.evaluation_result['protocol_id'] == agent.alphaevolve.protocol_id for n in [*drafts,child])
    finally:
        interpreter.terminate_all_subprocesses()


def test_rewrite_on_native_continuation_path(factory, monkeypatch):
    agent, execute, _ = build_variant(factory, monkeypatch, 'representation_frontier')
    drafts = initialize(agent, execute)
    current = drafts[0]
    # Exercise the continuation caller used by run.py. Do NOT unlock nodes or
    # replace current with None to force the global-only selector path.
    for _ in range(12):
        current = agent.step(current, execute)
    runtime = agent.phase1_runtime
    assert runtime.root_draft_state['submitted_count'] == 15
    assert runtime.root_draft_state['draft_submitted'] == 4
    decisions = [json.loads(x) for x in (runtime.artifact_root/'selection_decisions.jsonl').read_text().splitlines()]
    assert any(x.get('selection_reason') == 'audited_representation_rewrite' for x in decisions)


def test_rewrite_keeps_two_slots_even_after_skipped_slot():
    state = dict(pending={}, submitted_count=58,initial_goal=3,draft_submitted=8,draft_cap=draft_cap(60))
    assert not rewrite_due(state, steps=60)
    state['submitted_count'] = 50
    assert rewrite_due(state, steps=60)


def test_descriptor_requires_real_python_comments():
    fake = '\"\"\"\n# BO_REPRESENTATION: discrete\n# BO_SEARCH: greedy\n\"\"\"\npass\n'
    assert descriptor(fake) is None
    real = '# BO_REPRESENTATION: discrete\n# BO_SEARCH: greedy\npass\n'
    assert descriptor(real) == 'discrete/greedy'
    assert descriptor(real+'# BO_SEARCH: not a valid label\n') is None


def test_rating_free_review_transport_failure_is_reported(factory, monkeypatch, tmp_path):
    agent, execute, calls = build_variant(factory, monkeypatch, 'rating_free')
    monkeypatch.setenv('BO_TERMINATION_FILE',str(tmp_path/'receipt.json'))
    calls.state.fail_review = RuntimeError('injected provider failure')
    child = agent.step(None,execute,execute_immediately=False)
    agent.execute_deferred_node(child,execute)
    events = [json.loads(x) for x in (agent.phase1_runtime.artifact_root/'study7_failures.jsonl').read_text().splitlines()]
    assert events[0]['status'] == 'blocked/review_transport_failure'
    assert events[0]['method_id'] == 'AQ03'
