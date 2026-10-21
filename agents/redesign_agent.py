"""E04 full-module redesign. Claims never replace measured outcomes or BO targets."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import time

from llm import generate
from agents.coder import plan_and_code_query
from agents.planner import build_chat_prompt_for_model
from agents.triggers import register_node
from engine.search_node import SearchNode
from engine.redesign_evidence import ordered_closed

FIELDS = ('inherited_limitation', 'evidence_node_ids', 'replacement_algorithm',
          'retained_constraints', 'expected_observable_change', 'runtime_plan')


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def prose(value, limit=1200):
    # Plans may contain code. Do not accidentally supply the old implementation.
    return re.sub(r'```.*?```', '[code omitted]', str(value or ''), flags=re.S)[:limit]


def structure(code):
    try:
        tree = ast.parse(code)
    except (SyntaxError, TypeError):
        return {'parseable': False}
    return {'parseable': True,
            'functions': [n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))][:24],
            'imports': sorted({n.module or '' for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} |
                              {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names})[:24],
            'loop_count': sum(isinstance(n, (ast.For, ast.While)) for n in ast.walk(tree))}


def host_evidence(agent, node):
    trial = Path(agent.cfg.workspace_dir)/'trials'/node.id
    result_path = trial/'result.json'
    if not result_path.exists():
        return {'available': False}
    result = json.loads(result_path.read_text())
    if result.get('task_id') != agent.cfg.exp_id:
        raise RuntimeError('Redesign evidence task mismatch')
    candidate = trial/'candidate.py'
    if not candidate.exists() or candidate.read_bytes() != node.code.encode():
        raise RuntimeError('Redesign evidence source differs from evaluated code')
    construction = trial/'construction.json'
    h = hashlib.sha256(construction.read_bytes()).hexdigest() if construction.exists() else None
    if result.get('construction_sha256') != h:
        raise RuntimeError('Redesign evidence construction hash mismatch')
    return {'available': True, 'valid': result.get('valid'), 'metric': result.get('metric'),
            'construction_sha256': h, 'error': prose(result.get('error'), 400)}


def evidence_packet(agent, parent):
    # Closed decision outcomes only. No in-flight program/claim can leak in.
    rows = [r for r in ordered_closed(agent.phase1_runtime.transition_store.records())
            if r.metadata.get('budget_eligible', True) and r.metadata.get('target_status') != 'budget_censored']
    nodes = {n.id: n for n in list(agent.journal.nodes)}
    same = [r for r in rows if r.child_node_id in nodes and
            nodes[r.child_node_id].branch_id == parent.branch_id]
    other = [r for r in rows if r.child_node_id in nodes and
             nodes[r.child_node_id].branch_id != parent.branch_id]
    selected = same[-6:] + other[-2:]
    ids = [r.child_node_id for r in selected]
    if parent.id not in ids:
        ids.insert(0, parent.id)
    packet = []
    for node_id in dict.fromkeys(ids):
        n = nodes[node_id]
        record = {'node_id': n.id, 'parent_id': n.parent.id if n.parent else None,
                  'stage': n.stage, 'code_sha256': sha(n.code),
                  'llm_plan_claim': prose(n.plan), 'code_structure': structure(n.code),
                  'host_observation': host_evidence(agent, n),
                  'execution_seconds': getattr(n, 'exec_time', None)}
        earlier = Path(agent.cfg.log_dir)/'redesign'/f'{n.id}.json'
        if earlier.exists():
            record['previous_redesign_claim'] = json.loads(earlier.read_text())['plan_claim']
        packet.append(record)
    return {'schema': 'redesign_evidence_v1', 'task_id': agent.cfg.exp_id,
            'parent_id': parent.id, 'observations': packet,
            'interpretation': 'LLM plans are hypotheses. Equal scores or output hashes do not prove equivalent algorithms. '
                              'Code structure is static evidence, not proof of executed behavior.'}


def parse_plan(text, allowed_ids):
    raw = text.strip()
    if raw.startswith('```'):
        raw = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw)
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != set(FIELDS):
        raise ValueError('Expected the six redesign fields exactly')
    if not isinstance(value['evidence_node_ids'], list) or not all(
            isinstance(x, str) and x in allowed_ids for x in value['evidence_node_ids']):
        raise ValueError('Unknown evidence node referenced')
    for field in set(FIELDS)-{'evidence_node_ids'}:
        if not isinstance(value[field], str) or not value[field].strip() or len(value[field]) > 2200:
            raise ValueError('Missing, empty or oversized plan field: '+field)
    return value


def run(agent, parent):
    packet = evidence_packet(agent, parent)
    evidence = json.dumps(packet, ensure_ascii=False, allow_nan=False)
    intro = 'You design mathematical construction algorithms from measured experiments.'
    task = agent.task_desc + '\n' + agent.alphaevolve.candidate_contract()
    request = (
        task + '\nCompleted evidence:\n' + evidence +
        '\nPropose ONE feasible structural redesign. Identify an inherited design restriction suggested by '
        'the evidence and replace its algorithmic component, not merely its parameters or name. '
        'Do not claim a mathematical impossibility from a failed trial. Retain useful lessons and interfaces, '
        'but do not preserve the old architecture by default. No published answer lookup. '
        'The search archive already preserves the best verified candidate. A new construction may score lower. '
        'Respect the exact host constraints and CPU deadline. Return only a JSON object with these fields: '
        + ', '.join(FIELDS) + '. evidence_node_ids is a list of supplied IDs; all other fields are short strings. '
        'expected_observable_change must describe something that code or execution can check, not just an expected score.')
    planner_prompt = build_chat_prompt_for_model(agent.acfg.code.model, intro, request, '')
    attempts = []
    plan = None
    for _ in range(3):
        response = generate(prompt=planner_prompt, temperature=agent.acfg.code.temp, cfg=agent.cfg)
        attempts.append(response)
        try:
            plan = parse_plan(response, {r['node_id'] for r in packet['observations']})
            break
        except (ValueError, TypeError):
            continue
    agent_dir = Path(agent.cfg.log_dir)/'redesign'
    agent_dir.mkdir(parents=True, exist_ok=True)
    record = {'version': 1, 'parent_id': parent.id, 'evidence': packet,
              'planner_prompt': planner_prompt, 'planner_responses': attempts,
              'started_at': time.time(), 'plan_claim': plan}
    if plan is None:
        # A model-format failure consumes a proposal and is observable as invalid.
        code = "def solve(problem, initial, time_budget, seed):\n    raise RuntimeError('Redesign plan format failed')\n"
        plan_text = 'Invalid redesign planning output; no structural claim was accepted.'
        coder_prompt = None
    else:
        plan_text = json.dumps(plan, ensure_ascii=False)
        instruction = (task + '\nMeasured evidence:\n' + evidence + '\nSelected redesign plan:\n' + plan_text +
            '\nImplement this plan as a COMPLETE Python module. Full architectural replacement is allowed. '
            'No previous full candidate implementation is supplied. Do not emit SEARCH/REPLACE patches. '
            'Return a short explanation followed by one Python code block. Define solve exactly as required. '
            'Bound work using time_budget and return a feasible result before the deadline. '
            'Explain any part of the plan that could not be implemented; do not claim novelty merely from wording.')
        coder_prompt = build_chat_prompt_for_model(agent.acfg.code.model, intro, instruction, '')
        _, code = plan_and_code_query(agent, coder_prompt)
        code = code or ''
    record['coder_prompt'] = coder_prompt
    record['code_sha256'] = sha(code)
    record['implementation_structure'] = structure(code)
    record['parent_code_identical'] = code == parent.code
    record['semantic_change_verified'] = None
    parent.add_expected_child_count()
    node = SearchNode(plan=plan_text, code=code, parent=parent, stage='evolution',
                      local_best_node=parent.local_best_node,
                      from_topk=getattr(parent, '_topk_triggered', False))
    register_node(agent, node, {'planner': planner_prompt, 'coder': coder_prompt}, parent_node=parent)
    if hasattr(parent, '_topk_triggered'):
        del parent._topk_triggered
    (agent_dir/f'{node.id}.json').write_text(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    return node


def record_outcome(agent, node, result):
    path = Path(agent.cfg.log_dir)/'redesign'/f'{node.id}.json'
    if not path.exists():
        return
    data = json.loads(path.read_text())
    observation = host_evidence(agent, node) if result.get('valid') else {
        'available': True, 'valid': False, 'metric': None, 'error': prose(result.get('error'), 500)}
    parent = next((r for r in data['evidence']['observations'] if r['node_id'] == data['parent_id']), {})
    parent_h = parent.get('host_observation', {}).get('construction_sha256')
    h = observation.get('construction_sha256')
    data['outcome'] = observation
    data['returned_parent_output'] = h == parent_h if h and parent_h else None
    data['fallback_executed'] = None  # A matching output alone does not prove fallback use.
    data['completed_at'] = time.time()
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
