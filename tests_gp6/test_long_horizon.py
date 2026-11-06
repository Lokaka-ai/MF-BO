"""Direct Long A: labels -> original exact GP7 -> native LogEI/runtime.

Uses the existing isolated production-runtime harness. No LLM, HTTP, GPU or
AgentSearch import is required. See tests/test_long_horizon_runtime.py for the
optional native AgentSearch integration test in the project's full environment.
"""
from dataclasses import replace
import copy
import json
import math
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
from types import SimpleNamespace as NS, ModuleType
import sys
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose
from omegaconf import OmegaConf

from engine.parent_selectors.base import SelectionDecision, SelectorResult, utc_timestamp
from engine.parent_selectors.gp_graph_features import DecisionGraph, ancestry_matrix
from engine.parent_selectors.gp_graph_surrogate import GraphProposalSurrogate, GraphProposalSelector
from engine.parent_selectors.gp_surrogate import GP6Config
from engine.parent_selectors.gp_features import GP6_OPERATORS
from engine.parent_selectors.gp_long_horizon import (
    INDEX_KEY, LONG_HORIZON_SELECTOR as LONG, LongHorizonConfig, long_horizon_targets,
)
from engine.parent_selectors.gp_variants import stable_log_ei
from engine.parent_selectors.transition_store import TransitionStore
from test_core import features, record, option
from test_graph_selectors import ROOT, cfg, PRESETS
from test_runtime_contract import Runtime
import test_runtime_contract as runtime_tests

LONG_PRESET = PRESETS[LONG]


def attempt(key, index, *, parent='s', child=None, value=.82, parent_value=.8,
            closed=True, operator='improve', chain=None, **kwargs):
    state = features(parent_value, max(.8, parent_value), gp_trials=float(index))
    state.update(gp_primary_ancestry=chain or [parent], gp_metric_maximize=True,
                 gp_graph_center=parent_value)
    row = record(key, parent=parent, child=child or f'{key}-child', operator=operator,
                 value=value, parent_value=parent_value, closed=closed, state=state, **kwargs)
    return replace(row, metadata={**row.metadata, INDEX_KEY: index,
                                'long_horizon': {'horizon': 2}})


def seeded(*rows):
    return [record('seed', child='s', value=.8), *rows]


def model(h=2, pending_provider=None, optimize=False):
    return GraphProposalSurrogate(mode=LONG, long_horizon={'horizon': h},
        pending_provider=pending_provider,
        config=GP6Config(max_optimizer_iterations=25 if optimize else 0))


def window(row):
    return row.metadata['long_horizon_window']


class TestLongTargets(unittest.TestCase):
    def test_horizon_is_positive_integer_not_feature(self):
        self.assertEqual(LongHorizonConfig().horizon, 2)
        for h in (0, -1, True, 2.5, '2'):
            with self.assertRaises(ValueError): LongHorizonConfig(h)

    def test_delayed_gain_target_keeps_original_state(self):
        first = attempt('a', 1, value=.76)
        second = attempt('b', 2, parent=first.child_node_id, parent_value=.76,
                         value=.91, chain=[first.child_node_id, 's'])
        rows = seeded(first, second)
        saved = copy.deepcopy(rows)
        self.assertEqual(long_horizon_targets(rows[:-1], horizon=2)[0], [])
        targets, d = long_horizon_targets(rows, horizon=2)
        self.assertEqual(len(targets), 1)
        t = targets[0]
        self.assertEqual(t.transition_id, 'a')
        self.assertEqual(t.selection_features, first.selection_features)
        self.assertAlmostEqual(t.improvement, .11)
        self.assertEqual(t.child_utility, .91)
        self.assertEqual(window(t)['member_transition_ids'], ['a', 'b'])
        self.assertEqual(window(t)['outcome_transition_id'], 'b')
        self.assertFalse(d['horizon_is_feature'])
        self.assertEqual(rows, saved)

    def test_closed_failures_consume_budget_and_give_zero_not_minus_one(self):
        rows = seeded(attempt('a', 1, value=None), attempt('b', 2, value=None))
        targets, _ = long_horizon_targets(rows, horizon=2)
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].child_utility, .8)
        self.assertEqual(targets[0].improvement, 0)
        self.assertEqual(window(targets[0])['failed_attempts'], 2)
        m = model(); m.fit(rows)
        self.assertEqual(m.fit_diagnostics['training_count'], 1)
        self.assertFalse(m.fit_diagnostics['valid_ordinary_only'])
        assert_allclose(m.weights, [0.])

    def test_failure_then_success_is_two_attempts(self):
        targets, _ = long_horizon_targets(seeded(attempt('a', 1, value=None),
                                                attempt('b', 2, value=.83)), horizon=2)
        self.assertAlmostEqual(targets[0].improvement, .03)
        self.assertEqual(window(targets[0])['failed_attempts'], 1)

    def test_h1_is_best_of_one_not_signed_old_A(self):
        rows = seeded(attempt('a', 1, value=.7))
        targets, _ = long_horizon_targets(rows, horizon=1)
        self.assertEqual(targets[0].improvement, 0.)
        old = GraphProposalSurrogate(mode='gp7_ancestry_logei', config=GP6Config(max_optimizer_iterations=0))
        old.fit(rows)
        self.assertLess(old.weights[0], 0)

    def test_debug_waits_then_counts_as_one_proposal(self):
        a = attempt('a', 1, value=None, closed=False)
        b = attempt('b', 2, value=.84)
        rows = seeded(a, b)
        self.assertEqual(long_horizon_targets(rows, horizon=2)[0], [])
        repair = record('repair', operator='debug', parent=a.child_node_id, child='fixed',
                        value=.88, parent_value=None)
        targets, _ = long_horizon_targets([*rows, repair], horizon=2)
        self.assertEqual(len(targets), 1)
        self.assertEqual(window(targets[0])['member_transition_ids'], ['a', 'b'])
        self.assertEqual(window(targets[0])['outcome_transition_id'], 'repair')
        self.assertAlmostEqual(targets[0].improvement, .08)
        self.assertEqual(targets[0].selection_features, a.selection_features)

    def test_repaired_child_can_continue_primary_direction(self):
        a = attempt('a', 1, value=None, closed=False)
        repair = record('repair', operator='debug', parent=a.child_node_id, child='fixed', value=.79)
        b = attempt('b', 3, parent='fixed', parent_value=.79, value=.9, chain=['fixed', 's'])
        targets, _ = long_horizon_targets(seeded(a, repair, b), horizon=2)
        self.assertEqual(window(targets[0])['member_transition_ids'], ['a', 'b'])
        self.assertEqual(targets[0].child_utility, .9)

    def test_preexisting_child_and_fusion_reference_do_not_get_free_credit(self):
        old = attempt('old', 0, child='old-child', value=.81)
        a = attempt('a', 1, value=.82, operator='fusion')
        a = replace(a, reference_node_ids=('other',))
        unrelated = attempt('other-result', 2, parent='other', value=.99)
        old_branch = attempt('old-result', 3, parent='old-child', value=.98)
        b = attempt('b', 4, parent=a.child_node_id, parent_value=.82, value=.85)
        # Add an unclosed earlier start so its window cannot consume a's evidence.
        old = replace(old, metadata={**old.metadata, 'proposal_closed': False}, valid=False, child_utility=None)
        targets, _ = long_horizon_targets(seeded(old, a, unrelated, old_branch, b), horizon=2)
        t = next(t for t in targets if t.transition_id == 'a')
        self.assertEqual(window(t)['member_transition_ids'], ['a', 'b'])
        self.assertEqual(t.child_utility, .85)

    def test_pending_earlier_attempt_cannot_be_skipped(self):
        a, c = attempt('a', 1), attempt('c', 3, value=.9)
        pending = [{'decision_id':'b', 'decision_index':2, 'parent_node_id':'s', 'operator':'improve'}]
        self.assertEqual(long_horizon_targets(seeded(a, c), horizon=2, pending=pending)[0], [])
        b = attempt('b', 2, value=None)
        targets, _ = long_horizon_targets(seeded(a, c, b), horizon=2)
        self.assertEqual(window(targets[0])['member_transition_ids'], ['a', 'b'])
        self.assertEqual(targets[0].child_utility, a.child_utility)

    def test_canceled_reservation_is_not_a_failed_trial(self):
        a, c = attempt('a', 1), attempt('c', 3, value=.9)
        targets, _ = long_horizon_targets(seeded(a, c), horizon=2)
        self.assertEqual(window(targets[0])['member_transition_ids'], ['a', 'c'])
        self.assertEqual(window(targets[0])['failed_attempts'], 0)

    def test_completion_order_not_used_as_budget_order(self):
        rows = seeded(attempt('c', 3, value=.95), attempt('b', 2, value=.83), attempt('a', 1, value=.82))
        targets, _ = long_horizon_targets(rows, horizon=3)
        self.assertEqual(window(targets[0])['member_transition_ids'], ['a', 'b', 'c'])
        self.assertEqual(window(targets[0])['outcome_transition_id'], 'a')

    def test_training_windows_have_disjoint_attempt_members(self):
        rows = seeded(*[attempt(str(i), i, value=.8 + i*.01) for i in range(1, 6)])
        targets, d = long_horizon_targets(rows, horizon=2)
        self.assertEqual([window(t)['member_transition_ids'] for t in targets], [['1','2'], ['3','4']])
        self.assertEqual(d['complete_window_count'], 4)
        self.assertEqual(d['overlap_excluded_count'], 2)
        self.assertEqual(d['incomplete_window_count'], 1)

    def test_missing_order_and_duplicate_order_rejected(self):
        a = attempt('a', 1)
        with self.assertRaises(ValueError):
            long_horizon_targets(seeded(replace(a, metadata={})), horizon=2)
        with self.assertRaises(ValueError):
            long_horizon_targets(seeded(a, attempt('b', 1)), horizon=2)

    def test_minimization_uses_existing_negative_utility(self):
        a = attempt('a', 1, value=-3.2, parent_value=-3.)
        b = attempt('b', 2, value=-2.8, parent_value=-3.)
        targets, _ = long_horizon_targets([a, b], horizon=2)
        self.assertEqual(targets[0].child_utility, -2.8)
        self.assertAlmostEqual(targets[0].improvement, .2)


class TestLongGP(unittest.TestCase):
    def test_seven_dimensional_exact_posterior_no_shrink_or_log_transform(self):
        rows = seeded(*[attempt(str(i), i, value=.8 + .01*i) for i in range(1, 7)])
        m = model(); m.fit(rows)
        self.assertEqual(m.x.shape, (3, 7))
        opt = option('s', attempt('query', 7).selection_features)
        mu, latent, predictive = m.predict_quality([opt])
        targets, _ = long_horizon_targets(rows, horizon=2)
        y = np.asarray([t.improvement / m.utility_scale for t in targets])
        joint = ancestry_matrix([*m.chains, opt.state_features['gp_primary_ancestry']])
        cov = m._kernel_graph(m.x, m.ops, m.x, m.ops, joint[:-1] @ joint[:-1].T)
        cov += (m.hyperparameters[2]**2 + m.config.jitter)*np.eye(3)
        x = np.array([m._design(opt.state_features, opt.operator)])
        cross = m._kernel_graph(x, np.array([GP6_OPERATORS.index(opt.operator)]),
                                m.x, m.ops, joint[-1:] @ joint[:-1].T)
        expected = .8 + m.utility_scale * cross @ np.linalg.solve(cov, y)
        variance = m.hyperparameters[1]**2 - cross @ np.linalg.solve(cov, cross.T)
        assert_allclose(mu, expected)
        assert_allclose(latent**2, m.utility_scale**2 * variance.ravel())
        assert_allclose(predictive**2-latent**2, (m.utility_scale*m.hyperparameters[2])**2)
        result = GraphProposalSelector(surrogate=m, records_provider=lambda: rows).select(NS(), [opt])
        self.assertFalse(result.decision.fallback_used, result.decision.fallback_reason)
        expected_ei = stable_log_ei((mu-.8)/m.utility_scale, latent/m.utility_scale, 0.)[0]
        self.assertAlmostEqual(result.candidate_scores[0].acquisition_score, expected_ei)

    def test_prior_until_h_complete_and_map_after_eight_windows(self):
        rows = seeded(attempt('1', 1))
        m = model(optimize=True); m.fit(rows)
        self.assertEqual(m.x.shape, (0, 7))
        mu, latent, _ = m.predict_quality([option('s', rows[-1].selection_features)])
        assert_allclose(mu, [.8]); self.assertGreater(latent[0], 0)
        rows += [attempt(str(i), i, value=.8+.01*math.sin(i)) for i in range(2, 17)]
        m.fit(rows)
        self.assertEqual(len(m.training_transition_ids), 8)
        self.assertTrue(m.fit_diagnostics['optimizer_status'].startswith('map'))

    def test_same_row_count_pending_cancellation_invalidates_fit(self):
        rows = seeded(attempt('a', 1), attempt('c', 3, value=.9))
        pending = [{'decision_id':'b', 'decision_index':2, 'parent_node_id':'s', 'operator':'improve'}]
        m = model(pending_provider=lambda: pending)
        selector = GraphProposalSelector(surrogate=m, records_provider=lambda: rows)
        opt = option('s', rows[1].selection_features)
        selector.select(NS(), [opt]); self.assertEqual(len(m.training_transition_ids), 0)
        pending.clear()
        result = selector.select(NS(), [opt])
        self.assertFalse(result.decision.fallback_used)
        self.assertEqual(len(m.training_transition_ids), 1)

    def test_model_resume_rebuilds_same_targets_and_predictions(self):
        rows = seeded(*[attempt(str(i), i) for i in range(1, 5)])
        m = model(); m.fit(rows)
        saved = json.loads(json.dumps(m.state_dict()))
        new = model(); new.load_state_dict(saved); new.fit(rows)
        opts = [option('s', rows[-1].selection_features)]
        assert_allclose(new.predict_quality(opts), m.predict_quality(opts))
        self.assertEqual(new.fit_diagnostics['training_windows'], m.fit_diagnostics['training_windows'])
        with self.assertRaises(ValueError): model(h=3).load_state_dict(saved)
        with self.assertRaises(ValueError):
            GraphProposalSurrogate(mode='gp7_ancestry_logei').load_state_dict(saved)

    def test_old_A_rejects_long_config_instead_of_silent_ignored_horizon(self):
        with self.assertRaises(ValueError):
            GraphProposalSurrogate(mode='gp7_ancestry_logei', long_horizon={'horizon':2})


class TestLongRuntime(unittest.TestCase):
    def test_preset_changes_only_selector_and_fixed_label_contract(self):
        a = OmegaConf.to_container(OmegaConf.load(ROOT/'config/experiments/gp7_ancestry_ucb.yaml'))
        new = OmegaConf.to_container(OmegaConf.load(ROOT/f'config/experiments/{LONG_PRESET}.yaml'))
        self.assertEqual(new['bo_parent_selector']['gp'].pop('long_horizon'), {'horizon':2})
        new['parent_selector']['type'] = a['parent_selector']['type']
        self.assertEqual(a, new)
        loaded = runtime_tests.TestPresetLoading().loader([f'experiment_preset={LONG_PRESET}'])
        with tempfile.TemporaryDirectory() as tmp:
            loaded.log_dir=tmp; r=Runtime(loaded)
            self.assertTrue(r.gp_enabled and r.graph_enabled)
            self.assertTrue(r.surrogate.is_long_horizon)
            self.assertFalse(r.include_root_draft)
            self.assertEqual(r.surrogate.feature_names, ('parent_gap','last_gain','recent_gain',
                'stagnation','trials','valid_rate','absolute_parent'))

    def test_preset_and_checkpoint_reject_horizon_change(self):
        loader = runtime_tests.TestPresetLoading().loader
        with tempfile.TemporaryDirectory() as tmp:
            logs=Path(tmp)/'logs'; logs.mkdir()
            config = loader([f'experiment_preset={LONG_PRESET}'])
            OmegaConf.save(config, logs/'config.yaml')
            self.assertEqual(loader([f'experiment_preset={LONG_PRESET}',f'resume_run_dir={tmp}']).parent_selector.type, LONG)
            with self.assertRaises(ValueError):
                loader(['experiment_preset=gp7_ancestry_ucb',f'resume_run_dir={tmp}'])
            config.bo_parent_selector.gp.long_horizon.horizon = 3
            OmegaConf.save(config, logs/'config.yaml')
            with self.assertRaises(ValueError):
                loader([f'experiment_preset={LONG_PRESET}',f'resume_run_dir={tmp}'])

    def test_native_runtime_record_fit_select_and_jsonl_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = cfg(LONG, tmp); r = Runtime(c)
            # Exercise production terminal recorder and ledger order, replacing
            # only journal I/O here (native AgentSearch has a separate test).
            parent=NS(id='s',stage='draft',metric=NS(value=.8))
            journal=NS(nodes=[parent]); journal.append=journal.nodes.append
            r._agent=NS(journal=journal,scfg=NS(max_debug_depth=3))
            r.root_draft_state={'pending':{}, 'initial_terminal':1}
            r._persist_root_state=lambda agent: None
            r._node_snapshot=lambda node: None
            r.transition_store.append(record('seed',child='s',value=.8))
            state=attempt('origin',1).selection_features
            for i in (1,2):
                opt=option('s',state)
                dec=SelectionDecision(LONG,'s','improve',0.,None,False,f'd{i}',utc_timestamp())
                context=NS(decision=dec,option=opt)
                r.root_draft_state['pending'][dec.decision_id]={
                    'status':'executing','initial':False,'kind':'model','movement':'continue',
                    'decision_index':i,'option':{'parent_node_id':'s','operator':'improve'}}
                child=NS(id=f'c{i}',metric=NS(value=.78 if i==1 else .9),is_buggy=False,
                         is_valid=True,is_terminal=False,debug_depth=0,pending_execution=True)
                row=r.record_transition(parent_node=parent,child_node=child,context=context,
                    generation_wall_time_sec=1.,execution_wall_time_sec=1.,metric_maximize=True,
                    reward=0.,failure_reason=None)
                self.assertEqual(row.metadata[INDEX_KEY],i)
                r.surrogate.fit(r.transition_store.records())
                self.assertEqual(len(r.surrogate.training_transition_ids),i//2)
            result=r.selectors[LONG].select(NS(),[option('s',state)])
            self.assertFalse(result.decision.fallback_used,result.decision.fallback_reason)
            self.assertEqual(len(r.surrogate.fit_diagnostics['training_windows']),1)
            saved=r.surrogate.state_dict()
            restored=TransitionStore(r.transition_store.path).records()
            m=GraphProposalSurrogate(mode=LONG, task_id='synthetic')
            m.load_state_dict(saved); m.fit(restored)
            assert_allclose(m.predict_quality([option('s',state)]),r.surrogate.predict_quality([option('s',state)]))

    def test_long_snapshot_has_horizon_but_seven_gp_coordinates(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Runtime(cfg(LONG,tmp)); rows=seeded(attempt('a',1),attempt('b',2,value=.9))
            for row in rows:r.transition_store.append(row)
            parent=NS(id='s',stage='draft',metric=NS(value=.8),visits=0.,children=[])
            state=features(.8,.9,depth=1.,branch_stagnation=0.,recent_gain_mean=0.,
                recent_gain_std=0.,recent_gain_max=0.,recent_valid_rate=1.,elapsed_fraction=0.,remaining_steps_fraction=1.)
            state.update(gp_primary_ancestry=['s'],gp_metric_maximize=True,gp_graph_center=.8)
            opt=replace(option('s',state),parent_node=parent)
            result=r.selectors[LONG].select(NS(),[opt])
            r.managed_decisions=False
            agent=NS(best_node=parent,branch_successful_nodes={},acfg=NS(time_limit=100,steps=20))
            r._log_selection(agent,result)
            path=next(Path(tmp).rglob('*snapshot*.jsonl'))
            snap=json.loads(path.read_text().splitlines()[-1])
            self.assertEqual(snap['long_horizon'],{'horizon':2})
            self.assertEqual(len(snap['gp_feature_vector']),7)
            self.assertEqual(snap['prediction_target'],'fixed_h_best_utility')
            self.assertEqual(snap['uncertainty_used'],'latent')

    def test_real_suite_wrapper_routes_new_preset_for_all_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'scripts').mkdir();(root/'config').mkdir()
            for name in ['run_selector_task.sh','run_selector_suite.sh']:
                shutil.copy2(ROOT/'scripts'/name,root/'scripts'/name)
            shutil.copy2(ROOT/'config/mlebench_lite22.txt',root/'config/mlebench_lite22.txt')
            (root/'run_single_task.sh').write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$CAPTURE"\n')
            env={**os.environ,'TIME_LIMIT_SECS':'1','CPUS_PER_TASK':'1','CAPTURE':str(root/'calls.txt')}
            subprocess.run(['bash',str(root/'scripts/run_selector_suite.sh'),LONG_PRESET,'/fake-data','112','agent.seed=42'],
                           env=env,check=True,capture_output=True,text=True)
            calls=(root/'calls.txt').read_text().splitlines()
            self.assertEqual(len(calls),22)
            self.assertTrue(all(f'experiment_preset={LONG_PRESET}' in line for line in calls))


if __name__ == '__main__':
    unittest.main()
