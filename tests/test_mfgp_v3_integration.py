"""Native v3 lifecycle with real CPU processes and synthetic model responses."""
import copy
import json
from pathlib import Path
import pytest
from test_fixed_evaluation import fixed_cfg, setup_server, model_responses, CANDIDATE
from test_multifidelity_integration import aq05_cfg
from engine.agent_search import AgentSearch
from engine.search_node import Journal
from engine.executor import Interpreter
from engine.fixed_evaluation import FixedEvaluation
from engine.mf_gp.v3.runner import training_view, prepare_node, execute_prepared, worker_step
from engine.mf_gp.v3.controller import PreparedRequested
from engine.mf_gp.controller import PromotionRequested
from engine.mf_gp.execution import promote, finalize
from engine.mf_gp.recovery import execute
from utils import serialize

PROBE = """import os, json
if os.environ.get('BO_MF_PROBE') == '1':
    from pathlib import Path
    assert not Path('input/validation.csv').exists()
    assert not Path('input/test.csv').exists()
    json.dump({'schema':1, 'supported':False}, open(os.environ['BO_MF_PROBE_REPORT'], 'w'))
    raise SystemExit(0)
"""

@pytest.fixture
def v3(aq05_cfg, model_responses):
    cfg=aq05_cfg
    FixedEvaluation.prepare_or_load(cfg, "Evaluation: binary log loss.", Journal())
    cfg.evaluation.multifidelity_mode="mf_gp"
    cfg.evaluation.mf_gp_low_fraction=.2
    cfg.evaluation.mf_gp_policy_version=3
    protocol=json.loads((Path(cfg.evaluation.protocol_dir)/"protocol.json").read_text())
    path=Path(cfg.log_dir)/"prior.json"
    path.write_text(json.dumps(dict(schema=1, protocol_id=protocol['protocol_id'], training_fraction=.2,
        completed=True,code_sha256='a'*64,measurement='candidate_process_wall_seconds',low_seconds=1.,full_seconds=2.)))
    cfg.evaluation.mf_gp_cost_profile=str(path)
    model_responses.code=PROBE+CANDIDATE
    agent=AgentSearch("Evaluation: binary log loss.", cfg, Journal())
    interpreter=Interpreter(cfg.workspace_dir,cfg=cfg,timeout=30)
    callback=lambda code,node_id,*args,**kw:execute(interpreter,cfg,code,node_id,*args,**kw)
    return agent, callback, training_view(agent)


def generate(agent, callback, view):
    node=agent.step(None,callback,execute_immediately=False)
    assert node.training_fraction is None
    assert not agent.phase1_runtime.study7_selector.pending_rows(agent)
    prepare_node(agent,node,view)
    return node


def commit(agent, ticket, fidelity):
    s=agent.phase1_runtime.study7_selector
    s.prepared.start(ticket,fidelity)
    agent.phase1_runtime._persist_root_state(agent)


def test_native_v3_prepare_allocate_promote_resume(v3):
    agent,callback,view=v3
    runtime=agent.phase1_runtime;s=runtime.study7_selector
    node=generate(agent,callback,view)
    ticket=node.phase1_decision_id
    assert s.probes[ticket]['state']=='unsupported'
    with pytest.raises(PreparedRequested) as request:
        agent.step(None,callback)
    assert request.value.ticket==ticket and request.value.fidelity==.2
    result=execute_prepared(agent,ticket,callback)
    assert result.is_valid and result.training_fraction==.2
    assert s.observations[0]['execution_seconds']==result.exec_time
    assert s.observations[0]['probe_seconds']>0
    # First cheap success immediately launches the full anchor.
    with pytest.raises(PromotionRequested) as request:
        agent.step(None,callback)
    reservation=copy.deepcopy(s.pending_promotions[request.value.ticket])
    # A pending anchor does not stop the second initial generation/probe.
    following=worker_step(agent,callback,view)
    assert following.pending_execution
    promote(agent,request.value.ticket,callback)
    assert s.observations[-1]['kind']=='promotion'
    assert reservation['row']['raw_context']==s.raw_contexts[reservation['program_id']]
    commit(agent,following.phase1_decision_id,.2)
    execute_prepared(agent,following.phase1_decision_id,callback)
    last=generate(agent,callback,view)
    commit(agent,last.phase1_decision_id,.2)
    execute_prepared(agent,last.phase1_decision_id,callback)
    agent.capture_resume_state()
    restored=AgentSearch('ignored',copy.deepcopy(agent.cfg),serialize.loads_json(serialize.dumps_json(agent.journal),Journal))
    assert restored.phase1_runtime.study7_selector.contract()==s.contract()
    assert restored.phase1_runtime.study7_selector.raw_contexts==s.raw_contexts
    assert finalize(agent.cfg,agent.journal)['fidelity']==1.


@pytest.mark.parametrize('broken', ['def broken(:\n    pass',
    "open('input/validation.csv').read()"])
def test_syntax_probe_returns_native_debug_not_quality(v3, model_responses, broken):
    agent,callback,view=v3
    model_responses.code=broken
    node=generate(agent,callback,view)
    s=agent.phase1_runtime.study7_selector
    assert node.is_buggy
    assert s.prepared.entries[node.phase1_decision_id]['state']=='failed'
    assert s.observations[-1]['valid'] is False
    assert not s.gp.rows
    assert s.observations[-1]['execution_seconds']==0
    assert 'V3 bounded probe failed' in node.term_out
    agent.capture_resume_state()
    restored=AgentSearch('ignored',copy.deepcopy(agent.cfg),serialize.loads_json(serialize.dumps_json(agent.journal),Journal))
    parent=next(n for n in restored.journal.nodes if n.id==node.id)
    assert parent.term_out==node.term_out
    # Exercise the real native debug prompt and code generator after recovery.
    from engine.operator_preview import execute_operator_preview, OperatorPreview
    model_responses.code=PROBE+CANDIDATE
    repaired=execute_operator_preview(restored,parent,OperatorPreview(operator='debug',reason='probe_failure'))
    import ast
    assert ast.dump(ast.parse(repaired.code))==ast.dump(ast.parse(model_responses.code))


@pytest.mark.parametrize('state', ['ready','executing','deferred'])
def test_pending_recovery_preserves_source_and_fidelity(v3, monkeypatch, state):
    agent,callback,view=v3
    node=generate(agent,callback,view)
    s=agent.phase1_runtime.study7_selector;ticket=node.phase1_decision_id
    if state=='executing':commit(agent,ticket,1.)
    if state=='deferred':s.prepared.defer(ticket,'test_budget')
    agent.capture_resume_state()
    from engine.parent_selectors.runtime import Phase1SelectorRuntime
    monkeypatch.setattr(Phase1SelectorRuntime,'_owner_alive',staticmethod(lambda _:False))
    restored=AgentSearch('ignored',copy.deepcopy(agent.cfg),serialize.loads_json(serialize.dumps_json(agent.journal),Journal))
    entry=restored.phase1_runtime.study7_selector.prepared.entries[ticket]
    assert entry['state']==state and entry['code']==node.code
    assert entry['fidelity']==(1. if state=='executing' else None)
    assert restored.phase1_runtime.root_draft_state['initial_submitted']==1
    assert len(restored.phase1_runtime.recovered_deferred_nodes)==1
    if state=='executing':
        result=execute_prepared(restored,ticket,callback)
        assert result.training_fraction==1. and result.is_valid


def test_real_threaded_runner_finishes_at_action_cap(v3, monkeypatch):
    import logging
    from engine.mf_gp.v3.runner import run_search
    agent, _, _ = v3
    cfg=agent.cfg
    cfg.agent.steps=4
    cfg.log_dir=cfg.log_dir.parent / "threaded_logs"
    cfg.workspace_dir=cfg.workspace_dir.parent / "threaded_workspace"
    monkeypatch.setattr('engine.mf_gp.v3.runner.load_task_desc',lambda _: 'Evaluation: binary log loss.')
    result=run_search(cfg,logging.getLogger('test'))
    s=result.phase1_runtime.study7_selector
    assert sum(o['kind']=='expansion' for o in s.observations)==3
    assert sum(o['kind']=='promotion' for o in s.observations)==1
    assert not s.pending_promotions
    assert not result.phase1_runtime.root_draft_state['pending']
    assert finalize(cfg,result.journal)['fidelity']==1.


def test_saved_failed_probe_cannot_become_executable_after_resume(v3):
    agent,callback,view=v3
    node=agent.step(None,callback,execute_immediately=False)
    s=agent.phase1_runtime.study7_selector
    ticket=node.phase1_decision_id
    # Crash boundary: probe receipt persisted, native failure not yet recorded.
    s.probes[ticket]=dict(state='runtime_error',elapsed_seconds=.25,metadata=None)
    prepare_node(agent,node,view)
    assert s.prepared.entries[ticket]['state']=='failed'
    assert not s.observations[-1]['valid']
    assert not s.observations[-1]['completed_runtime_observation']


def test_recovered_generation_is_not_reserved_or_charged_twice(v3,monkeypatch):
    agent,callback,view=v3
    runtime=agent.phase1_runtime
    with agent.journal_lock:
        context=runtime.select_global_parent(agent)
        runtime.submit_decision(agent,context)
    ticket=context.decision.decision_id
    agent.capture_resume_state()
    from engine.parent_selectors.runtime import Phase1SelectorRuntime
    monkeypatch.setattr(Phase1SelectorRuntime,'_owner_alive',staticmethod(lambda _:False))
    restored=AgentSearch('ignored',copy.deepcopy(agent.cfg),serialize.loads_json(serialize.dumps_json(agent.journal),Journal))
    recovered=restored.phase1_runtime.recovered_generation_contexts
    assert len(recovered)==1 and recovered[0].decision.decision_id==ticket
    node=worker_step(restored,callback,view,recovered[0])
    assert node.phase1_decision_id==ticket
    assert restored.phase1_runtime.root_draft_state['initial_submitted']==1
    assert len(restored.phase1_runtime.study7_selector.prepared.entries)==1


def test_offline_mes_reconstructs_logged_pool(v3,tmp_path):
    agent,callback,view=v3
    generate(agent,callback,view)
    with pytest.raises(PreparedRequested):agent.step(None,callback)
    from tools.replay_mfgp_v3 import replay
    path=agent.phase1_runtime.artifact_root/'mf_gp_decisions.jsonl'
    output=tmp_path/'offline.jsonl'
    replay(path,output)
    row=json.loads(output.read_text().splitlines()[0])
    assert row['unknown_counterfactual_rewards']
    assert row['challenger']['offline_only']


def test_actual_post_generation_fidelity_overrides_advisory(v3, monkeypatch):
    agent,callback,view=v3
    runtime=agent.phase1_runtime;s=runtime.study7_selector
    for _ in range(12):
        if runtime.root_draft_state['initial_terminal']==3 and any(o['valid'] and o['fidelity']==1. for o in s.observations):
            break
        worker_step(agent,callback,view)
    else:pytest.fail('Native initialization did not finish')
    # Only ranking is controlled: native legal options, generation, allocation,
    # process execution, host evaluation and transition admission remain real.
    def choose(gp, targets, actions, incumbent, pending, terminal):
        return [100. if a['kind']=='expand' and a['row']['fidelity']==.2 else
                200. if a['kind']=='prepared' and a['row']['fidelity']==1. else 0.
                for a in actions], dict(terminal=terminal)
    monkeypatch.setattr('engine.mf_gp.v3.controller.acquisition',choose)
    node=worker_step(agent,callback,view)
    assert node.pending_execution and node.training_fraction is None
    assert node.phase1_selection_features['mf']['fidelity']==.2
    source=node.code;ticket=node.phase1_decision_id
    result=worker_step(agent,callback,view)
    assert result.id==node.id and result.code==source
    assert result.training_fraction==1. and result.is_valid
    assert s.prepared.entries[ticket]['state']=='completed'
    assert s.observations[-1]['fidelity']==1.
