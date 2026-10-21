"""Live checkout contract for the downstream AlphaEvolve evolution route."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

from agents import evolution_agent, redesign_agent
from config import _load_cfg


def test_alphaevolve_keeps_upstream_default(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['run.py', 'task_preset=alphaevolve', 'exp_id=kissing_number_d11'])
    cfg = _load_cfg()
    assert cfg.get('audited_bo', 'baseline') == 'baseline'
    assert cfg.parent_selector.type == 'gp_child_ucb'


def test_live_evolution_dispatches_to_frozen_redesign_route(tmp_path, monkeypatch):
    parent_code = 'def solve(problem, initial, time_budget, seed):\n    return "UNIQUE_PARENT_BODY"\n'
    parent = NS(id='parent', code=parent_code, plan='old plan', stage='improve', parent=None,
                branch_id=1, local_best_node=None, exec_time=1.0,
                add_expected_child_count=lambda: None,
                get_root_to_current_trajectory=lambda max_steps=10: 'Step 1: draft\nStep 2: improve')
    captured = []
    plan = {field: ([] if field == 'evidence_node_ids' else 'Replace the construction algorithm.')
            for field in redesign_agent.FIELDS}
    monkeypatch.setattr(redesign_agent, 'generate', lambda **kwargs: captured.append(kwargs['prompt']) or json.dumps(plan))
    monkeypatch.setattr(redesign_agent, 'plan_and_code_query',
                        lambda agent, prompt: (captured.append(prompt) or 'plan',
                                                'def solve(problem, initial, time_budget, seed):\n    return initial\n'))
    monkeypatch.setattr(redesign_agent, 'SearchNode', lambda **kwargs: NS(id='child', **kwargs))
    monkeypatch.setattr(redesign_agent, 'register_node', lambda *args, **kwargs: None)
    monkeypatch.setattr(evolution_agent, 'can_run_evolution', lambda node: True)
    agent = NS(cfg=NS(audited_bo='redesign', exp_id='kissing_number_d11', log_dir=tmp_path,
                       workspace_dir=tmp_path),
               acfg=NS(code=NS(model='muse-spark-1.3-contributor', temp=1.0)),
               task_desc='Find a feasible mathematical construction.',
               alphaevolve=NS(candidate_contract=lambda: 'Define solve(problem, initial, time_budget, seed).'),
               phase1_runtime=NS(transition_store=NS(records=lambda: [])),
               journal=NS(nodes=[parent]))
    child = evolution_agent.run(agent, parent)
    assert child.stage == 'evolution' and child.parent is parent
    assert len(captured) == 2
    assert 'COMPLETE Python module' in str(captured[1])
    assert 'UNIQUE_PARENT_BODY' not in str(captured[0]) + str(captured[1])
    record = json.loads((Path(tmp_path) / 'redesign' / 'child.json').read_text())
    assert record['plan_claim']['replacement_algorithm'] == 'Replace the construction algorithm.'
    assert record['semantic_change_verified'] is None
