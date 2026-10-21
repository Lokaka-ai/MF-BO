"""Original MCGS stays the selector while lease recovery journals its work."""
import copy
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest

from engine.agent_search import AgentSearch
from engine.parent_selectors.runtime import Phase1SelectorRuntime
from engine.search_node import Journal
from test_competition_checkpoint import competition, resume
from utils import serialize

pytestmark = pytest.mark.parametrize('competition', ['mlevolve_original'], indirect=True)


def test_three_interrupted_slots_preserve_code_rng_counters_and_continuations(competition, monkeypatch):
    agent, execute, clock = competition
    agent.data_preview = 'Original public input preview'
    nodes = [agent.step(None, execute, execute_immediately=False) for _ in range(3)]
    runtime = agent.phase1_runtime
    assert runtime.legacy_checkpoint and not runtime.prefer_unoccupied_parents
    for node in nodes:
        runtime.mark_executing(agent, node._phase1_decision_context)
    agent.fusion_draft_count = 1
    py_state, np_state = random.getstate(), np.random.get_state()
    restored = resume(agent, 10500)
    runtime = restored.phase1_runtime
    assert runtime.root_draft_state['initial_submitted'] == 3
    assert runtime.root_draft_state['submitted_count'] == 3
    assert not runtime.transition_store.records()
    assert restored.data_preview == agent.data_preview
    assert restored.fusion_draft_count == 1
    assert restored.virtual_root.expected_child_count == 3
    assert clock[0] - restored.start_time == 10500
    assert random.getstate() == py_state
    assert np.array_equal(np.random.get_state()[1], np_state[1])
    assert [(n.id, n.code) for n in runtime.recovered_deferred_nodes] == [(n.id, n.code) for n in nodes]
    for node in runtime.recovered_deferred_nodes:
        restored.execute_deferred_node(node, execute)
    ready = dict(runtime.root_draft_state['ready_continuations'])
    assert set(ready.values()) == {node.id for node in nodes}
    restored = resume(restored, 10600)
    runtime = restored.phase1_runtime
    assert runtime.root_draft_state['ready_continuations'] == ready
    assert len(runtime.transition_store.records()) == 3
    parent = restored.journal.nodes[1]
    context = runtime.select_continuation_parent(restored, parent)
    assert context.option.parent_node_id == parent.id
    assert context.decision.selection_reason == 'direct_continuation'
    assert len(runtime.root_draft_state['ready_continuations']) == 2


@pytest.mark.parametrize('charged', [False, True])
def test_generation_replay_keeps_decision_and_charges_only_once(competition, monkeypatch, charged):
    import engine.agent_search as search
    agent, execute, _ = competition
    original = search.execute_operator_preview
    class Interrupted(BaseException):
        pass
    def stop(agent, parent, preview, **kwargs):
        if charged:
            parent.add_expected_child_count()
        raise Interrupted()
    monkeypatch.setattr(search, 'execute_operator_preview', stop)
    with pytest.raises(Interrupted):
        agent.step(None, execute, execute_immediately=False)
    pending = copy.deepcopy(agent.phase1_runtime.root_draft_state['pending'])
    restored = resume(agent, 10400)
    restored = resume(restored, 10500)
    context, = restored.phase1_runtime.recovered_generation_contexts
    assert context.decision.decision_id in pending
    monkeypatch.setattr(search, 'execute_operator_preview', original)
    node = restored.step(None, execute, execute_immediately=False, recovery_context=context)
    assert node.phase1_decision_id == context.decision.decision_id
    assert restored.virtual_root.expected_child_count == 1
    assert restored.phase1_runtime.root_draft_state['submitted_count'] == 1
    assert not restored.phase1_runtime.transition_store.records()


def test_global_selection_uses_native_soft_switch_and_no_bo(competition, monkeypatch):
    agent, execute, _ = competition
    for _ in range(3):
        node = agent.step(None, execute, execute_immediately=False)
        agent.execute_deferred_node(node, execute)
    runtime = agent.phase1_runtime
    def forbidden(*args, **kwargs):
        raise AssertionError('MCGS entered a managed BO/root policy')
    monkeypatch.setattr(runtime, '_select_global_with_root', forbidden)
    monkeypatch.setattr(runtime, '_rank_options', forbidden)
    import engine.node_selection as selection
    original = selection.select_with_soft_switch
    calls = []
    def native(agent):
        result = original(agent)
        calls.append(result.id)
        return result
    monkeypatch.setattr(selection, 'select_with_soft_switch', native)
    context = runtime.select_global_parent(agent)
    assert calls == [context.option.parent_node_id]
    assert context.decision.selector_name == 'legacy_original'
    assert context.decision.model_version is None
    assert not agent.cfg.acq7.enabled and not agent.cfg.study7.rating.enabled


def test_generator_failure_does_not_create_a_node_or_consume_scientific_steps(competition, monkeypatch):
    import engine.agent_search as search
    agent, execute, _ = competition
    def fail(*args, **kwargs):
        raise RuntimeError('injected generation failure')
    monkeypatch.setattr(search, 'execute_operator_preview', fail)
    with pytest.raises(RuntimeError, match='injected'):
        agent.step(None, execute)
    assert len(agent.journal) == 1
    assert not agent.phase1_runtime.transition_store.records()
    assert not agent.phase1_runtime.root_draft_state['pending']
    assert agent.phase1_runtime._budget_available(agent)


@pytest.mark.parametrize('clock_step', [0, 3600])
def test_checkpoint_bookkeeping_matches_uninterrupted_native_search(competition, monkeypatch, tmp_path, clock_step):
    import itertools
    import uuid
    template, _, clock = competition
    from engine.executor import ExecutionResult

    def trial(enabled):
        clock[0] = 1000.
        with monkeypatch.context() as patch:
            if enabled:
                patch.setenv('MLEVOLVE_COMPETITION_CHECKPOINT', '1')
            else:
                patch.delenv('MLEVOLVE_COMPETITION_CHECKPOINT', raising=False)
            ids = itertools.count(1)
            patch.setattr(uuid, 'uuid4', lambda: uuid.UUID(int=next(ids)))
            random.seed(42)
            np.random.seed(42)
            cfg = copy.deepcopy(template.cfg)
            cfg.log_dir = tmp_path / str(enabled) / 'logs'
            cfg.workspace_dir = tmp_path / str(enabled) / 'workspace'
            cfg.workspace_dir.mkdir(parents=True)
            cfg.agent.use_evolution = cfg.agent.use_fusion = False
            agent = AgentSearch('Predict labels using public training data.', cfg, Journal())
            agent.data_preview = 'same public input preview'
            def execute(code, node_id, *args, **kwargs):
                path = cfg.workspace_dir / 'submission' / f'submission_{node_id}.csv'
                path.parent.mkdir(exist_ok=True)
                path.write_text('id,prediction\n' + ''.join(f'{i},{i/33}\n' for i in range(32)))
                return ExecutionResult(['validation metric: 0.6'], 1., None)
            nodes = [agent.step(None, execute, execute_immediately=False) for _ in range(3)]
            for node in nodes:
                agent.execute_deferred_node(node, execute)
            current = nodes[0]
            for i in range(8):
                clock[0] = 1000. + i * clock_step
                current = agent.step(current, execute)
            trace = [(n.id, n.parent.id if n.parent else None, n.stage, n.branch_id,
                      n.code, n.metric.value, n.is_buggy, n.visits, n.total_reward,
                      n.expected_child_count, n.lock) for n in agent.journal.nodes]
            return trace, random.getstate(), np.random.get_state()[1].tolist()
    assert trial(False) == trial(True)
