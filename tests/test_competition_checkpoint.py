"""Production AQ05 recovery; replace only external model/training/HTTP services."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest

from config import Config, _load_cfg
from engine.agent_search import AgentSearch
from engine.executor import ExecutionResult
from engine.parent_selectors.runtime import Phase1SelectorRuntime
from engine.search_node import Journal, SearchNode
from utils import serialize


@pytest.fixture
def competition(tmp_path, monkeypatch, request):
    clock = [1000.]
    monkeypatch.setattr('time.time', lambda: clock[0])
    monkeypatch.setattr(Phase1SelectorRuntime, '_owner_alive', lambda *a: False)
    monkeypatch.setenv('MLEVOLVE_COMPETITION_CHECKPOINT', '1')
    preset = getattr(request, 'param', 'acq7_aq05_gp4_level15_rating_kg')
    monkeypatch.setattr(sys, 'argv', ['test', f'experiment_preset={preset}', f'data_dir={tmp_path}', f'desc_file={tmp_path}/task.md', 'exp_name=checkpoint_test'])
    cfg = OmegaConf.merge(OmegaConf.structured(Config), _load_cfg())
    cfg.log_dir = tmp_path / 'logs'
    cfg.workspace_dir = tmp_path / 'workspace'
    (cfg.workspace_dir / 'input').mkdir(parents=True)
    cfg.exp_name = 'test_competition_recovery'
    cfg.cpu_number = '4'
    cfg.agent.use_global_memory = False
    cfg.agent.use_stepwise_generation = False
    cfg.agent.use_diff_mode = False
    cfg.agent.use_aggregation = False
    cfg.agent.initial_drafts = 3
    cfg.agent.search.parallel_search_num = 3
    cfg.agent.time_limit = 43200
    cfg.exec.timeout = 32400
    cfg.agent.seed = 42
    cfg.evaluation.enabled = False

    def query(**kw):
        spec = kw.get('func_spec')
        if spec is None:
            return 'Predict labels; maximize validation accuracy.'
        if spec.name == 'determine_metric_direction':
            return {'lower_is_better': False, 'reasoning': 'accuracy'}
        if spec.name == 'submit_code_review':
            return {'needs_revision': False, 'reasoning': 'OK', 'revised_code': None}
        return {'is_bug': False, 'summary': 'Completed', 'metric': .6,
                'lower_is_better': False, 'code_summary': 'classifier', 'potential_level': 2}

    for target in ['llm.query', 'agents.result_parse_agent.query', 'agents.code_review_agent.query']:
        monkeypatch.setattr(target, query)
    monkeypatch.setattr('agents.coder.base_coder.generate', lambda **kw: "Plan.\n```python\nprint('candidate')\n```")
    http = lambda *a, **kw: SimpleNamespace(status_code=200, json=lambda: {'is_valid': True, 'result': 'valid'})
    monkeypatch.setattr('engine.validation.format_client.requests.get', http)
    monkeypatch.setattr('engine.validation.format_client.requests.post', http)

    def execute(code, node_id, *a, **kw):
        path = cfg.workspace_dir / 'submission' / f'submission_{node_id}.csv'
        path.parent.mkdir(exist_ok=True)
        path.write_text('id,prediction\n' + ''.join(f'{i},{i/33}\n' for i in range(32)))
        return ExecutionResult(['validation metric: 0.6'], 1., None)

    agent = AgentSearch('Predict labels using public training data.', cfg, Journal())
    return agent, execute, clock


def resume(agent, elapsed):
    agent.capture_resume_state()
    cfg = copy.deepcopy(agent.cfg)
    cfg.resume_run_dir = cfg.log_dir.parent
    cfg.resume_elapsed_seconds = elapsed
    journal = serialize.loads_json(serialize.dumps_json(agent.journal), Journal)
    return AgentSearch('ignored', cfg, journal)


def test_node_copy_tolerates_attributes_added_while_snapshotting():
    node = SearchNode(code="pass", stage="draft")

    class ConcurrentAnnotation:
        def __deepcopy__(self, memo):
            node.late_annotation = "added by result parsing"
            return "snapshot value"

    node.annotation = ConcurrentAnnotation()
    restored = copy.deepcopy(node)
    assert restored.id == node.id
    assert restored.annotation == "snapshot value"
    assert node.late_annotation == "added by result parsing"


@pytest.mark.parametrize('legacy_checkpoint', [False, True])
def test_resume_preserves_original_data_preview(competition, monkeypatch, legacy_checkpoint):
    agent, execute, _ = competition
    agent.acfg.code.model = 'muse-spark-1.3-contributor'
    agent.data_preview = "Original input-only preview, before training artifacts existed."
    drafts = [agent.step(None, execute, execute_immediately=False) for _ in range(3)]
    for node in drafts:
        agent.execute_deferred_node(node, execute)
    agent.capture_resume_state()
    if legacy_checkpoint:
        agent.journal.study7_search_state.pop('competition_data_preview', None)
    cfg = copy.deepcopy(agent.cfg)
    cfg.resume_run_dir = cfg.log_dir.parent
    cfg.resume_elapsed_seconds = 10500
    journal = serialize.loads_json(serialize.dumps_json(agent.journal), Journal)

    def workspace_changed(*args, **kwargs):
        raise FileNotFoundError("A temporary runfile disappeared during the rescan")

    monkeypatch.setattr('engine.agent_search.data_preview.generate', workspace_changed)
    restored = AgentSearch('ignored', cfg, journal)
    assert restored.data_preview == agent.data_preview


def test_checkpoint_thread_retries_a_transient_snapshot_failure(tmp_path):
    root = Path(__file__).resolve().parents[1]
    code = '''
import os, threading, time
from types import SimpleNamespace as NS
from pathlib import Path
from engine.competition_checkpoint import install
p=Path(os.environ['CHECKPOINT_TEST_ROOT'])
cfg=NS(task_type='competition',parent_selector=NS(type='acq7_gp4'),study7=NS(method_id='AQ05'),evaluation=NS(enabled=False),log_dir=p)
runtime=NS(transition_store=NS(records=lambda: []),root_draft_state={'pending':{}})
agent=NS(cfg=cfg,journal_lock=threading.RLock(),phase1_runtime=runtime,start_time=time.time()-1234,acfg=NS(time_limit=43200))
attempts = 0
def persist():
    global attempts
    attempts += 1
    if attempts == 2:
        raise RuntimeError('dictionary changed size during iteration')
    (p/'persisted').write_text(str(attempts))
install(agent, persist)
time.sleep(5)
raise RuntimeError('checkpoint thread did not recover')
'''
    env = dict(os.environ, PYTHONPATH=str(root), CHECKPOINT_TEST_ROOT=str(tmp_path),
               MLEVOLVE_COMPETITION_CHECKPOINT='1', MLEVOLVE_COMPETITION_PAUSE_FILE=str(tmp_path/'pause'),
               MLEVOLVE_COMPETITION_LEASE_END='1')
    proc = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True, timeout=15)
    assert proc.returncode == 75, proc.stderr
    assert (tmp_path/'persisted').read_text() == '3'
    record = json.loads((tmp_path/'competition_pause.json').read_text())
    assert record['reason'] == 'allocation_boundary' and record['paused']


@pytest.mark.parametrize('counter_charged', [False, True])
@pytest.mark.parametrize('legacy_checkpoint', [False, True])
def test_generation_replay_charges_native_parent_once(competition, monkeypatch,
                                                     counter_charged, legacy_checkpoint):
    import engine.agent_search as search
    agent, execute, _ = competition
    drafts = [agent.step(None, execute, execute_immediately=False) for _ in range(3)]
    for node in drafts:
        agent.execute_deferred_node(node, execute)
    parent = drafts[0]
    original = search.execute_operator_preview

    class CheckpointBoundary(BaseException):
        pass

    def interrupted(agent, node, preview, **kwargs):
        if counter_charged:
            node.add_expected_child_count()
        raise CheckpointBoundary()

    monkeypatch.setattr(search, 'execute_operator_preview', interrupted)
    with pytest.raises(CheckpointBoundary):
        agent.step(parent, execute, execute_immediately=False)
    runtime = agent.phase1_runtime
    assert len(runtime.root_draft_state['pending']) == 1
    if legacy_checkpoint:
        for pending in runtime.root_draft_state['pending'].values():
            pending.pop('child_count_charged', None)
    submitted = runtime.root_draft_state['submitted_count']
    resumed = resume(agent, 10500)
    context = resumed.phase1_runtime.recovered_generation_contexts[0]
    monkeypatch.setattr(search, 'execute_operator_preview', original)

    def generate(**kw):
        if kw.get('json_schema'):
            return json.dumps({'reason': 'test', 'module': ['model_design'],
                               'plan': {'model_design': 'train classifier'}})
        return "Plan.\n```python\nprint('candidate')\n```"

    monkeypatch.setattr('agents.planner.base_planner.generate', generate)
    candidate = resumed.step(None, execute, execute_immediately=False, recovery_context=context)
    assert candidate.parent.id == parent.id
    assert candidate.parent.expected_child_count == 1
    assert len(candidate.parent.children) == 1
    assert resumed.phase1_runtime.root_draft_state['submitted_count'] == submitted
    assert len(resumed.phase1_runtime.transition_store.records()) == 3


def test_three_interrupted_training_slots_replay_exact_nodes_without_false_gp_rows(competition):
    agent, execute, clock = competition
    nodes = [agent.step(None, execute, execute_immediately=False) for _ in range(3)]
    runtime = agent.phase1_runtime
    for node in nodes:
        runtime.mark_executing(agent, node._phase1_decision_context)
    before_rng = copy.deepcopy(runtime.study7_selector.state_dict())
    resumed = resume(agent, 10500)
    restored = resumed.phase1_runtime
    assert len(restored.transition_store.records()) == 0
    assert restored.root_draft_state['submitted_count'] == 3
    assert restored.root_draft_state['initial_terminal'] == 0
    assert [n.id for n in restored.recovered_deferred_nodes] == [n.id for n in nodes]
    assert [n.code for n in restored.recovered_deferred_nodes] == [n.code for n in nodes]
    assert restored.study7_selector.state_dict() == before_rng
    assert clock[0] - resumed.start_time == 10500
    for node in restored.recovered_deferred_nodes:
        resumed.execute_deferred_node(node, execute)
    assert len(restored.transition_store.records()) == 3
    assert restored.root_draft_state['initial_terminal'] == 3
    assert restored.root_draft_state['submitted_count'] == 3
    assert all(n.study7_rating_metadata['context']['allocated_cpus'] == 4
               for n in resumed.journal.nodes[1:])


def test_interrupted_generation_replays_frozen_decision_once_and_counts_elapsed(competition):
    agent, execute, clock = competition
    runtime = agent.phase1_runtime
    context = runtime.select_global_parent(agent)
    assert runtime.submit_decision(agent, context)
    resumed = resume(agent, 32000)
    restored = resumed.phase1_runtime
    assert len(restored.transition_store.records()) == 0
    assert len(restored.recovered_generation_contexts) == 1
    recovered = restored.recovered_generation_contexts[0]
    assert recovered.decision.decision_id == context.decision.decision_id
    # A second interruption, before generation even resumes, retains the work.
    resumed = resume(resumed, 32100)
    restored = resumed.phase1_runtime
    recovered = restored.recovered_generation_contexts[0]
    node = resumed.step(None, execute, execute_immediately=False, recovery_context=recovered)
    assert node.phase1_decision_id == context.decision.decision_id
    assert restored.root_draft_state['submitted_count'] == 1
    assert restored.root_draft_state['initial_submitted'] == 1
    assert restored.root_draft_state['decision_count'] == 1
    assert clock[0] - resumed.start_time == 32100
    assert resumed.acfg.time_limit == 43200
    resumed.execute_deferred_node(node, execute)
    assert len(restored.transition_store.records()) == 1
    assert restored.root_draft_state['initial_terminal'] == 1


@pytest.mark.parametrize('selector', ['acq7_gp4', 'legacy_original'])
@pytest.mark.parametrize('reason', ['allocation_boundary', 'budget_complete'])
def test_checkpoint_exit_saves_clock_before_stopping_process(tmp_path, selector, reason):
    root = Path(__file__).resolve().parents[1]
    code = '''
import os, threading, time
from types import SimpleNamespace as NS
from pathlib import Path
from engine.competition_checkpoint import install
p=Path(os.environ['CHECKPOINT_TEST_ROOT'])
cfg=NS(task_type='competition',parent_selector=NS(type=os.environ['CHECKPOINT_SELECTOR']),study7=NS(method_id='AQ05'),evaluation=NS(enabled=False),log_dir=p)
runtime=NS(transition_store=NS(records=lambda: []),root_draft_state={'pending':{}})
agent=NS(cfg=cfg,journal_lock=threading.RLock(),phase1_runtime=runtime,start_time=time.time()-1234,acfg=NS(time_limit=int(os.environ['CHECKPOINT_BUDGET'])))
install(agent, lambda: (p/'persisted').write_text('saved'))
time.sleep(10)
raise RuntimeError('checkpoint did not pause')
'''
    env = dict(os.environ, PYTHONPATH=str(root), CHECKPOINT_TEST_ROOT=str(tmp_path),
               CHECKPOINT_SELECTOR=selector, CHECKPOINT_BUDGET='1000' if reason == 'budget_complete' else '43200',
               MLEVOLVE_COMPETITION_CHECKPOINT='1', MLEVOLVE_COMPETITION_PAUSE_FILE=str(tmp_path/'pause'),
               MLEVOLVE_COMPETITION_LEASE_END='1')
    proc = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True, timeout=20)
    assert proc.returncode == 75, proc.stderr
    record = json.loads((tmp_path/'competition_pause.json').read_text())
    assert 1234 <= record['active_elapsed_seconds'] < 1250
    assert record['reason'] == reason and record['paused'] is True
    assert (tmp_path/'persisted').read_text() == 'saved'
    assert not (tmp_path/'final_submission.json').exists()


def test_finished_training_output_reused_and_execution_budget_not_reset(tmp_path):
    from engine.competition_checkpoint import execute, atomic_json
    from engine.executor import Interpreter
    cfg = SimpleNamespace(log_dir=tmp_path/'logs', exec=SimpleNamespace(timeout=1),
                          start_cpu_id='0', cpu_number='3', agent=SimpleNamespace(
                              search=SimpleNamespace(parallel_search_num=3)))
    interpreter = Interpreter(tmp_path, timeout=1, cfg=cfg)
    code = "from pathlib import Path\np=Path('count')\np.write_text(p.read_text()+'x' if p.exists() else 'x')\nprint('finished')"
    first = execute(interpreter, cfg, code, 'node1', True)
    second = execute(interpreter, cfg, code, 'node1', True)
    assert first.to_dict() == second.to_dict()
    assert (tmp_path/'count').read_text() == 'x'
    path = cfg.log_dir/'candidate_execution/node1.json'
    record = json.loads(path.read_text())
    record.update(state='interrupted', elapsed_seconds=1.)
    atomic_json(path, record)
    exhausted = execute(interpreter, cfg, code, 'node1', True)
    assert exhausted.exc_type == 'TimeoutError'
    assert (tmp_path/'count').read_text() == 'x'


def test_newer_decision_ledger_preserves_parent_charge_after_abrupt_death(competition, monkeypatch):
    """The 15-second journal may precede a newly generated pending child."""
    agent, execute, _ = competition
    drafts = [agent.step(None, execute, execute_immediately=False) for _ in range(3)]
    for node in drafts:
        agent.execute_deferred_node(node, execute)
    agent.capture_resume_state()
    stale_journal = serialize.loads_json(serialize.dumps_json(agent.journal), Journal)

    def generate(**kw):
        if kw.get('json_schema'):
            return json.dumps({'reason': 'test', 'module': ['model_design'],
                               'plan': {'model_design': 'train classifier'}})
        return "Plan.\n```python\nprint('candidate')\n```"
    monkeypatch.setattr('agents.planner.base_planner.generate', generate)
    child = agent.step(drafts[0], execute, execute_immediately=False)
    original_count = child.parent.expected_child_count
    assert original_count == 1
    cfg = copy.deepcopy(agent.cfg)
    cfg.resume_run_dir = cfg.log_dir.parent
    cfg.resume_elapsed_seconds = 1000
    resumed = AgentSearch('ignored', cfg, stale_journal)
    recovered = resumed.phase1_runtime.recovered_deferred_nodes
    assert len(recovered) == 1 and recovered[0].id == child.id
    assert recovered[0].parent.expected_child_count == original_count
