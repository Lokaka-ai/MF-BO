"""A/B numerical, causal-prefix, configuration and isolated-runtime tests.

Runs without LLM clients or GPUs. Production numeric modules and runtime class
body are exercised; this does not replace native AgentSearch integration tests.
"""
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import tempfile
import subprocess
import shutil
import os
from types import SimpleNamespace as NS, ModuleType
import sys
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose
from omegaconf import OmegaConf

from engine.parent_selectors.base import ExpansionOption, OperatorPreview, SelectionDecision, SelectorResult
from engine.parent_selectors.gp_graph_features import (GP7_NAMES, GP7_SCHEMA, GraphConfig,
    DecisionGraph, ancestry_matrix, graph_matern32, gp7_vector)
from engine.parent_selectors.gp_graph_surrogate import GraphProposalSurrogate, GraphProposalSelector
from engine.parent_selectors.gp_surrogate import GP6Config, mixed_matern32
from engine.parent_selectors.gp_selector import GaussianProposalSelector
from engine.parent_selectors.gp_variants import stable_log_ei
from engine.parent_selectors.transition_store import TransitionRecord, TransitionStore
from engine.parent_selectors.proposal_feedback import effective_proposals
from engine.selection_logging import SelectionLogger
from test_core import features, record, option
from test_runtime_contract import Runtime, namespace, SOURCE
import test_runtime_contract as runtime_tests

ROOT = Path(__file__).resolve().parents[1]
A, B = "gp7_ancestry_logei", "laplacian_anchor_logei"
PRESETS = {A: "gp7_ancestry_ucb", B: B,
           "gp7_long_horizon_logei": "gp7_long_horizon_ucb"}


def cfg(mode, log_dir):
    c = OmegaConf.merge(OmegaConf.load(ROOT/'config/config.yaml'),
                        OmegaConf.load(ROOT/f'config/experiments/{PRESETS[mode]}.yaml'))
    c.log_dir = str(log_dir)
    c.exp_id = 'synthetic'
    return c


def model(mode=A, **kwargs):
    return GraphProposalSurrogate(mode=mode,
        config=GP6Config(max_optimizer_iterations=0 if mode == B else 25), **kwargs)


def trace(count=12, negative=False):
    """Completed rows with features frozen before each synthetic outcome."""
    graph = DecisionGraph(.3)
    seed = -.7 if negative else .7
    rows = [record('seed', child='s', value=seed)]
    for i in range(count):
        graph.update(rows)
        parent = 's' if i % 3 else (rows[-1].child_node_id or 's')
        pu = graph.utilities[parent]
        state = features(pu, max(graph.utilities.values()), gp_trials=float(i),
                         gp_last_gain=.01*math.sin(i), gp_recent_gain=.01*math.cos(i),
                         gp_stagnation=float(i%4), gp_valid_rate=.8)
        state.update(graph.features(parent, not negative))
        rows.append(record(f't{i}', parent=parent, child=f'n{i}', operator=('improve','evolution','fusion')[i%3],
                           value=pu+.018*math.sin(i+.4), parent_value=pu, state=state))
    graph.update(rows)
    opts=[]
    for j, parent in enumerate(['s', rows[-1].child_node_id, rows[-2].child_node_id]):
        state=features(graph.utilities[parent],max(graph.utilities.values()), gp_trials=float(count), gp_valid_rate=.8)
        state.update(graph.features(parent,not negative))
        opts.append(option(parent,state,('improve','evolution','fusion')[j]))
    return rows,opts


class TestGraphInputs(unittest.TestCase):
    def test_kernel_psd_diagonal_and_ancestry_extension(self):
        chains=[['p'],['a','p'],['b','p'],['c']]
        p=ancestry_matrix(chains);g=p@p.T
        assert_allclose(np.diag(g),1)
        self.assertEqual(g[0,3],0)
        self.assertGreater(g[1,2],0)
        extended=ancestry_matrix(chains+[['d','a','p']])
        assert_allclose(extended[:4]@extended[:4].T,g)
        x=np.random.default_rng(32).normal(size=(4,7));op=np.arange(4)%3
        for weight in [0.,.3,1.]:
            k=graph_matern32(x,op,x,op,g,graph_weight=weight)
            assert_allclose(np.diag(k),1)
            self.assertGreater(np.linalg.eigvalsh(k).min(),-1e-10)
        assert_allclose(graph_matern32(x,op,x,op,g,graph_weight=0),
                        mixed_matern32(x,op,x,op,length_scale=1.,amplitude=1.,rho=.25))

    def test_feature_modes_and_missing_direction(self):
        state=features(.8);state['gp_metric_maximize']=True
        def design(s,task='',mode='auto',loss_scale=.5):
            return gp7_vector(s,scale=.1,draft_anchor=.5,loss_scale=loss_scale,
                              task_id=task,absolute_mode=mode,operator='improve')
        self.assertEqual(len(design(state)),7)
        self.assertAlmostEqual(design(state)[-1],.8)
        self.assertAlmostEqual(design(state,'aptos2019-blindness-detection')[-1],.9)
        state.update(parent_utility=-.5,gp_metric_maximize=False)
        self.assertAlmostEqual(design(state)[-1],-math.log(2))
        with self.assertRaises(ValueError):design(features(.8))
        with self.assertRaises(ValueError):design(features(.8),'','log_loss')

    def test_two_node_laplacian_and_excluded_virtual_root(self):
        rows=[record('seed',child='p',value=.7),
              record('next',parent='p',child='v',operator='improve',value=.9,parent_value=.7)]
        graph=DecisionGraph(.3);graph.update(rows)
        self.assertEqual(graph.ancestry('v'),['v','p'])
        self.assertNotIn('root',graph.utilities)
        self.assertAlmostEqual(graph.centers['v'],.8625)
        self.assertEqual(graph.utilities['v'],.9)
        self.assertEqual(rows[-1].child_utility,.9)

    def test_prefix_debug_is_one_research_edge(self):
        rows=[record('seed',child='p',value=.7),
              record('bad',parent='p',child='failed',operator='improve',value=None,parent_value=.7,closed=False),
              record('repair1',parent='failed',child='failed2',operator='debug',value=None,closed=False),
              record('repair2',parent='failed2',child='valid',operator='debug',value=.8)]
        graph=DecisionGraph(.3);graph.update(rows[:2])
        self.assertEqual(set(graph.utilities),{'p'})
        with self.assertRaises(ValueError):graph.features('valid',True)
        graph.update(rows)
        self.assertEqual(graph.ancestry('valid'),['valid','p'])
        self.assertEqual(graph.edge_count,1)
        self.assertEqual(set(graph.utilities),{'p','valid'})
        self.assertEqual(len(effective_proposals(rows)),2)

    def test_separate_drafts_are_not_connected(self):
        rows=[record('seed1',child='a',value=.5),record('seed2',child='b',value=.9)]
        graph=DecisionGraph(.3);graph.update(rows)
        self.assertEqual(graph.edge_count,0)
        self.assertEqual(graph.centers,{'a':.5,'b':.9})
        p=ancestry_matrix([graph.ancestry('a'),graph.ancestry('b')])
        self.assertEqual(float((p@p.T)[0,1]),0)

    def test_validation_and_cycles(self):
        for kw in [{'ancestry_decay':1.},{'graph_weight':-1.},{'absolute_mode':'anything'},
                   {'covariance_ridge':0.},{'amplitude_cap':float('nan')}, {'laplacian_smoothing':-1.}]:
            with self.assertRaises(ValueError):GraphConfig(**kw)
        with self.assertRaises(ValueError):ancestry_matrix([['a','a']])
        with self.assertRaises(ValueError):ancestry_matrix([[]])
        graph=DecisionGraph(.3)
        graph.parents={'a':'b','b':'a'};graph.utilities={'a':.5,'b':.6}
        with self.assertRaises(ValueError):graph.ancestry('a')


class TestGraphModels(unittest.TestCase):
    def test_a_exact_posterior_independent_matrix_formula(self):
        rows,opts=trace(5);m=model(A);m.fit(rows)
        self.assertEqual(m.x.shape,(5,7))
        self.assertNotIn('seed',m.training_transition_ids)
        mu,sd,pred=m.predict_quality(opts)
        xq=np.asarray([m._design(dict(o.state_features),o.operator) for o in opts])
        oq=np.arange(3)
        p=ancestry_matrix(m.chains+[o.state_features['gp_primary_ancestry'] for o in opts]);n=len(m.x)
        k=graph_matern32(m.x,m.ops,m.x,m.ops,p[:n]@p[:n].T)
        cross=graph_matern32(xq,oq,m.x,m.ops,p[n:]@p[:n].T)
        cov=k+(.25**2+1e-8)*np.eye(n)
        y=np.asarray([(r.child_utility-r.parent_utility)/m.utility_scale for r in rows[1:]])
        expected=np.asarray([o.parent_utility for o in opts])+m.utility_scale*cross@np.linalg.solve(cov,y)
        v=1-np.diag(cross@np.linalg.solve(cov,cross.T))
        assert_allclose(mu,expected,atol=1e-12)
        assert_allclose(sd,m.utility_scale*np.sqrt(v),atol=1e-12)
        assert_allclose(pred**2-sd**2,(.25*m.utility_scale)**2,atol=1e-12)

    def test_b_matches_frozen_center_covariance_formula_and_never_optimizes(self):
        rows,opts=trace(12);m=model(B)
        with patch('engine.parent_selectors.gp_graph_surrogate.minimize',side_effect=AssertionError('B optimized')):
            m.fit(rows)
        self.assertEqual(m.weights.size,0)
        mu,sd,pred=m.predict_quality(opts)
        assert_allclose(mu,[o.state_features['gp_graph_center'] for o in opts])
        errors=np.asarray([r.child_utility-r.selection_features['gp_graph_center'] for r in rows[1:]])
        tau=min(np.sqrt(np.mean(errors**2)),5*m.utility_scale)
        self.assertAlmostEqual(m.tau,tau)
        xq=np.asarray([m._design(dict(o.state_features),o.operator) for o in opts])
        p=ancestry_matrix(m.chains+[o.state_features['gp_primary_ancestry'] for o in opts]);n=len(m.x)
        k=graph_matern32(m.x,m.ops,m.x,m.ops,p[:n]@p[:n].T)
        cross=graph_matern32(xq,np.arange(3),m.x,m.ops,p[n:]@p[:n].T)
        q=1-np.diag(cross@np.linalg.solve(k+.0625*np.eye(n),cross.T))
        assert_allclose(sd,tau*np.sqrt(q),atol=1e-12)
        assert_allclose(pred,tau*np.sqrt(q+.0625),atol=1e-12)
        self.assertFalse(m.fit_diagnostics['mean_fitted'])

    def test_frozen_centers_not_recomputed_and_debug_attributed(self):
        rows,opts=trace(3)
        original=rows[-1];frozen=original.selection_features['gp_graph_center']
        failed=replace(original,valid=False,child_utility=None,child_metric_raw=None,improvement=None,
                       metadata={'proposal_closed':False})
        m=model(B);m.fit([*rows[:-1],failed])
        self.assertEqual(len(m.training_transition_ids),2)
        debug=record('fixed',parent=original.child_node_id,child='fixednode',operator='debug',
                     value=original.parent_utility+.03,state=features(999))
        repaired=[*rows[:-1],failed,debug]
        m.fit(repaired)
        self.assertEqual(len(m.training_transition_ids),3)
        self.assertEqual(m.training_transition_ids[-1],original.transition_id)
        self.assertEqual(m.fit_diagnostics['training_outcome_transition_ids'][-1],'fixed')
        effective=effective_proposals(repaired)[-1]
        self.assertEqual(effective.selection_features['gp_graph_center'],frozen)
        self.assertEqual(m.chains[-1],original.selection_features['gp_primary_ancestry'])

    def test_bound_controls_scale_not_labels(self):
        rows,opts=trace(8)
        last=replace(rows[-1],child_utility=-1e12)
        m=model(B);m.fit([*rows[:-1],last]);m.predict_quality(opts)
        self.assertAlmostEqual(m.tau,5*m.utility_scale)
        self.assertTrue(m.fit_diagnostics['scale_ceiling_active'])
        self.assertEqual(last.child_utility,-1e12)
        self.assertEqual(len(m.training_transition_ids),8)
        self.assertTrue(np.isfinite(m.predict_quality(opts)).all())

    def test_cold_start_uses_prior_but_no_draft_training(self):
        rows,opts=trace(1)
        for mode in [A,B]:
            m=model(mode);m.fit(rows[:1]);mu,sd,pred=m.predict_quality(opts)
            self.assertEqual(m.x.shape,(0,7))
            assert_allclose(sd,m.utility_scale)
            self.assertTrue(np.isfinite([mu,sd,pred]).all())

    def test_minimize_and_dynamic_loss_scale(self):
        rows,opts=trace(10,negative=True)
        m=model(A);m.fit(rows[:2]);scale=m.absolute_loss_scale
        self.assertAlmostEqual(scale,np.median([abs(r.child_utility) for r in rows[:2]]))
        m.fit(rows)
        self.assertAlmostEqual(m.absolute_loss_scale,np.median([abs(r.child_utility) for r in rows]))
        self.assertNotEqual(m.absolute_loss_scale,scale)
        self.assertTrue(np.isfinite(m.predict_quality(opts)).all())
        self.assertTrue((m.x[:,-1]<0).all())

    def test_additional_draft_preserves_scale_without_adding_training_label(self):
        rows,opts=trace(3)
        for mode in [A,B]:
            m=model(mode);m.fit(rows)
            ids=list(m.training_transition_ids);scale=m.utility_scale;raw=[dict(r.selection_features) for r in rows]
            anchor=m.draft_anchor
            m.fit([*rows,record('extra',value=10.)])
            self.assertEqual(m.training_transition_ids,ids)
            self.assertEqual(m.utility_scale,scale)
            self.assertEqual(m.draft_anchor,anchor)
            self.assertEqual([r.selection_features for r in rows],raw)
            self.assertTrue(np.isfinite(m.predict_quality(opts)).all())

    def test_resume_predictions_and_reject_changed_contract(self):
        rows,opts=trace(12)
        for mode in [A,B]:
            m=model(mode,task_id='test');m.fit(rows)
            expected=m.predict_quality(opts)
            state=json.loads(json.dumps(m.state_dict()))
            restored=model(mode,task_id='test');restored.load_state_dict(state);restored.fit(rows)
            assert_allclose(restored.predict_quality(opts),expected,rtol=1e-10,atol=1e-12)
            self.assertEqual(restored.state_dict(),m.state_dict())
            with self.assertRaises(ValueError):model(mode,task_id='different').load_state_dict(state)
            with self.assertRaises(ValueError):model(mode,graph_config={'graph_weight':.1},task_id='test').load_state_dict(state)

    def test_no_legacy_state_silent_migration(self):
        rows,opts=trace(2);rows[-1]=replace(rows[-1],selection_features=features(.7))
        with self.assertRaisesRegex(ValueError,'frozen GP7 ancestry'):model(A).fit(rows)
        with self.assertRaises(ValueError):GraphProposalSurrogate(mode=B,config=GP6Config())

    def test_selector_global_logei_and_fit_logging(self):
        rows,opts=trace(12)
        for mode in [A,B]:
            with tempfile.TemporaryDirectory() as tmp:
                m=model(mode,selection_logger=SelectionLogger(tmp))
                selector=GraphProposalSelector(surrogate=m,records_provider=lambda:rows)
                result=selector.select(NS(),opts)
                self.assertFalse(result.decision.fallback_used,result.decision.fallback_reason)
                self.assertEqual(result.decision.selector_name,mode)
                mu,sd,_=m.predict_quality(opts);inc=opts[0].state_features['incumbent_at_decision']
                expected=stable_log_ei((mu-inc)/m.utility_scale,sd/m.utility_scale,0.)
                self.assertEqual(result.selected_option,opts[int(np.argmax(expected))])
                assert_allclose([s.acquisition_score for s in result.candidate_scores],expected)
                logs=list(Path(tmp).glob('*'));self.assertTrue(logs)
                self.assertIn(GP7_SCHEMA,'\n'.join(p.read_text() for p in logs))

    def test_fallback_stays_inside_legal_pool(self):
        rows,opts=trace(3);m=model(B)
        selector=GraphProposalSelector(surrogate=m,records_provider=lambda:rows)
        with patch.object(m,'predict_quality',side_effect=ArithmeticError('synthetic')):
            result=selector.select(NS(),opts)
        self.assertTrue(result.decision.fallback_used)
        self.assertIn(result.selected_option,opts)
        self.assertEqual(result.selected_option,max(opts,key=lambda o:o.parent_utility))

    def test_ucb_matches_gp6_acquisition_for_positive_and_negative_utility(self):
        for negative in [False, True]:
            rows, opts = trace(10, negative=negative)
            gp = model(acquisition='ucb')
            gp6 = GaussianProposalSelector(surrogate=gp, records_provider=lambda: rows, beta=1.5)
            graph = GraphProposalSelector(surrogate=gp, records_provider=lambda: rows)
            expected, actual = gp6.select(None, opts), graph.select(None, opts)
            self.assertFalse(actual.decision.fallback_used, actual.decision.fallback_reason)
            self.assertEqual(actual.candidate_scores, expected.candidate_scores)
            self.assertEqual(actual.selected_option, expected.selected_option)
            mu, latent, _ = gp.predict_quality(opts)
            assert_allclose([s.acquisition_score for s in actual.candidate_scores], mu + 1.5 * latent)

    def test_frozen_metadata_jsonl_roundtrip(self):
        rows,_=trace(4)
        with tempfile.TemporaryDirectory() as tmp:
            store=TransitionStore(Path(tmp)/'rows.jsonl')
            for row in rows:store.append(row)
            restored=TransitionStore(store.path).records()
            self.assertEqual([r.to_dict() for r in restored],[r.to_dict() for r in rows])
            for mode in [A,B]:
                m=model(mode);m.fit(restored);self.assertEqual(len(m.training_transition_ids),4)


class TestGraphRouting(unittest.TestCase):
    def test_runtime_and_config_loader_both_arms(self):
        for mode in [A,B]:
            c=runtime_tests.TestPresetLoading().loader([f'experiment_preset={PRESETS[mode]}'])
            with tempfile.TemporaryDirectory() as tmp:
                c.log_dir=tmp;r=Runtime(c)
                self.assertTrue(r.graph_enabled and r.gp_enabled and r.managed_decisions)
                self.assertEqual(r.feature_schema_version,GP7_SCHEMA)
                self.assertEqual(r.selectors[mode].name,mode)
                self.assertFalse(r.include_root_draft)

    def test_presets_differ_in_selector_optimizer_and_requested_acquisition(self):
        a=OmegaConf.to_container(OmegaConf.load(ROOT/f'config/experiments/{PRESETS[A]}.yaml'))
        b=OmegaConf.to_container(OmegaConf.load(ROOT/f'config/experiments/{B}.yaml'))
        b['parent_selector']['type']=a['parent_selector']['type']
        b['bo_parent_selector']['gp']['max_optimizer_iterations']=25
        self.assertEqual(a['bo_parent_selector']['acquisition']['type'],'child_ucb')
        self.assertEqual(b['bo_parent_selector']['acquisition']['type'],'logei')
        b['bo_parent_selector']['acquisition']['type']='child_ucb'
        self.assertEqual(a,b)
        self.assertEqual(len([x for x in (ROOT/'config/mlebench_lite22.txt').read_text().splitlines() if x.strip()]),22)

    def test_forbidden_config_rejected(self):
        for mode in [A,B]:
            with tempfile.TemporaryDirectory() as tmp:
                for key,value in [('parent_selector.include_root_draft',True),
                                  ('parent_selector.decision_scope','root_only'),
                                  ('bo_parent_selector.acquisition.type','unknown'),
                                  ('bo_parent_selector.acquisition.use_cost',True),
                                  ('bo_parent_selector.acquisition.use_validity_probability',True),
                                  ('bo_parent_selector.gp.graph.xi',.01)]:
                    c=cfg(mode,tmp);OmegaConf.update(c,key,value)
                    with self.assertRaises(ValueError,msg=key):Runtime(c)

    def test_loader_cross_arm_and_changed_parameters_resume(self):
        loader=runtime_tests.TestPresetLoading().loader
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'logs';path.mkdir()
            c=loader([f'experiment_preset={PRESETS[A]}']);OmegaConf.save(c,path/'config.yaml')
            self.assertEqual(loader([f'experiment_preset={PRESETS[A]}',f'resume_run_dir={tmp}']).parent_selector.type,A)
            with self.assertRaises(ValueError):loader([f'experiment_preset={B}',f'resume_run_dir={tmp}'])
            c.bo_parent_selector.gp.graph.graph_weight=.2;OmegaConf.save(c,path/'config.yaml')
            with self.assertRaises(ValueError):loader([f'experiment_preset={PRESETS[A]}',f'resume_run_dir={tmp}'])

    def test_make_option_freezes_graph_and_snapshot_seven_coordinates(self):
        rows,old_opts=trace(6)
        for mode in [A,B]:
            with tempfile.TemporaryDirectory() as tmp:
                c=cfg(mode,tmp);r=Runtime(c)
                for row in rows:r.transition_store.append(row)
                root=NS(id='root',stage='root',parent=None,metric=NS(value=None))
                p=NS(id='s',stage='draft',parent=root,metric=NS(value=.7),visits=0,total_reward=0.,children=[],branch_id=1)
                agent=NS(journal=NS(nodes=[root,p]),metric_maximize=True,scfg=c.agent.search,acfg=c.agent,
                         best_node=p,branch_successful_nodes={},is_root=lambda n:n.id=='root')
                r._incumbent=lambda a:max(float(v.child_utility) for v in rows if v.valid)
                r._global_stagnation=lambda a:0.
                raw=features(.7,r._incumbent(agent),depth=1.,branch_stagnation=0.,recent_gain_mean=0.,
                             recent_gain_std=0.,recent_gain_max=0.,recent_valid_rate=1.,elapsed_fraction=0.,remaining_steps_fraction=1.)
                with patch.dict(namespace,{'extract_selection_features':lambda *a,**k:dict(raw)}):
                    opt=r._make_option(agent,p,OperatorPreview('improve'))
                saved=opt.state_features['gp_graph_center']
                self.assertEqual(opt.state_features['gp_primary_ancestry'],['s'])
                result=r.selectors[mode].select(agent,[opt])
                self.assertFalse(result.decision.fallback_used,result.decision.fallback_reason)
                r.managed_decisions=False  # Isolate snapshot formatting, not scheduling.
                r._log_selection(agent,result)
                snaps=[json.loads(line) for f in Path(tmp).rglob('*snapshot*.jsonl') for line in f.read_text().splitlines()]
                self.assertTrue(snaps)
                self.assertEqual(len(snaps[-1]['gp_feature_vector']),7)
                self.assertEqual(snaps[-1]['acquisition_kind'],'child_ucb' if mode == A else 'global_logei')
                self.assertEqual(snaps[-1]['feature_vector']['gp_graph_center'],saved)
                later=record('later',parent='s',child='later',operator='improve',value=.99,parent_value=.7)
                r.transition_store.append(later);r.graph_context.update(r.transition_store.records())
                self.assertEqual(opt.state_features['gp_graph_center'],saved)
                if mode==B:self.assertNotEqual(r.graph_context.centers['s'],saved)

    def test_managed_checkpoint_records_graph_contract_and_frozen_scale(self):
        for mode in [A,B]:
            with tempfile.TemporaryDirectory() as tmp:
                c=cfg(mode,tmp);r=Runtime(c);rows,_=trace(5)
                for row in rows:r.transition_store.append(row)
                r.surrogate.fit(rows)
                root=NS(id='root',stage='root',parent=None)
                agent=NS(acfg=c.agent,scfg=c.agent.search,metric_maximize=True,journal=NS(nodes=[root]))
                module=ModuleType('engine.search_node');module.Journal=lambda **kw:NS(**kw)
                util=ModuleType('utils');util.serialize=NS(dumps_json=lambda v:json.dumps({'ids':[n.id for n in v.nodes]}))
                seed=ModuleType('utils.seed');seed.capture_random_state=lambda:('python','numpy');seed.restore_random_state=lambda *args:None
                with patch.dict(sys.modules,{'engine.search_node':module,'utils':util,'utils.seed':seed}):
                    r.bind_agent(agent)
                state=json.loads((r.artifact_root/'root_draft_state.json').read_text())
                self.assertEqual(state['feature_dimension'],7)
                self.assertEqual(state['feature_schema_version'],GP7_SCHEMA)
                self.assertEqual(state['graph_config']['ancestry_decay'],.5)
                self.assertEqual(state['gp_reference'],r.surrogate.state_dict())
                self.assertEqual(len(state['observations']),len(rows))
                journal=NS(nodes=[root],parent_selector_state=state)
                Runtime.validate_checkpoint(c,journal)
                wrong=cfg(B if mode==A else A,tmp)
                with self.assertRaises(ValueError):Runtime.validate_checkpoint(wrong,journal)

    def test_real_shell_wrappers_route_both_arms_and_all_22_tasks(self):
        # Execute actual wrapper scripts, replace only their external task runner.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'scripts').mkdir();(root/'config').mkdir()
            for name in ['run_selector_task.sh','run_selector_suite.sh']:
                shutil.copy2(ROOT/'scripts'/name,root/'scripts'/name)
            shutil.copy2(ROOT/'config/mlebench_lite22.txt',root/'config/mlebench_lite22.txt')
            runner=root/'run_single_task.sh'
            runner.write_text('#!/usr/bin/env bash\nprintf "%s\n" "$*" >> "$CAPTURE"\n')
            env={**os.environ,'TIME_LIMIT_SECS':'1','CPUS_PER_TASK':'1','CAPTURE':str(root/'calls.txt')}
            for mode in [A,B]:
                subprocess.run(['bash',str(root/'scripts/run_selector_suite.sh'),PRESETS[mode],'/fake-data','112','agent.seed=42'],
                               env=env,check=True,capture_output=True,text=True)
            calls=(root/'calls.txt').read_text().splitlines()
            self.assertEqual(len(calls),44)
            self.assertTrue(all(f'experiment_preset={PRESETS[A]}' in x for x in calls[:22]))
            self.assertTrue(all(f'experiment_preset={B}' in x for x in calls[22:]))
            self.assertTrue(all('agent.seed=42' in x and '/fake-data' in x for x in calls))

    def test_initial_draft_log_has_no_fabricated_gp7_design(self):
        for mode in [A,B]:
            with tempfile.TemporaryDirectory() as tmp:
                r=Runtime(cfg(mode,tmp));r.managed_decisions=False
                root=NS(id='root',stage='root',metric=NS(value=None),visits=0.,children=[])
                f=features(None,None,depth=0.,branch_stagnation=0.,recent_gain_mean=0.,recent_gain_std=0.,
                    recent_gain_max=0.,recent_valid_rate=0.,elapsed_fraction=0.,remaining_steps_fraction=1.)
                opt=ExpansionOption('root','draft',(),f,root,OperatorPreview('draft'))
                dec=SelectionDecision(mode,'root','draft',0.,None,False,'d','time')
                agent=NS(best_node=None,branch_successful_nodes={},acfg=NS(time_limit=100,steps=20))
                r._log_selection(agent,SelectorResult(dec,opt,(),(opt,)))
                snaps=[json.loads(line) for path in Path(tmp).rglob('*snapshot*.jsonl') for line in path.read_text().splitlines()]
                self.assertIsNone(snaps[-1]['gp_feature_vector'])


if __name__ == '__main__':
    unittest.main()
