"""Run without agent/LLM dependencies: python -m unittest discover -s tests_gp6 -v."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose
from omegaconf import OmegaConf
from sklearn.gaussian_process.kernels import Matern

from engine.parent_selectors.base import ExpansionOption, OperatorPreview
from engine.parent_selectors.gp_surrogate import GaussianProposalSurrogate, GP6Config, mixed_matern32
from engine.parent_selectors.gp_features import GP6_NAMES, GP6_OPERATORS, gp_vector, progress_features
from engine.parent_selectors.gp_selector import GaussianProposalSelector
from engine.parent_selectors.proposal_feedback import effective_proposals
from engine.parent_selectors.surrogate import BayesianGainModel
from engine.parent_selectors.features import FEATURE_NAMES
from engine.parent_selectors.transition_store import TransitionRecord
from engine.selection_logging import SelectionLogger


def features(parent=.5, incumbent=.6, **extra):
    out = dict(gp_parent_gap=max(incumbent-parent, 0) if parent is not None else 0,
               gp_last_gain=0., gp_recent_gain=0., gp_stagnation=0., gp_trials=0.,
               gp_valid_rate=.5, parent_utility=parent, incumbent_at_decision=incumbent)
    out.update(extra)
    return out


def record(key, parent='root', child=None, operator='draft', value=.5, parent_value=None,
           state=None, closed=True, cost=2.):
    valid = value is not None
    return TransitionRecord(key, key, parent, child or key+'child', operator, (), parent_value, value,
        parent_value, value, value-parent_value if valid and parent_value is not None else None,
        valid, cost/2, cost/2, cost, None, None, 'gp_child_ucb',
        state or features(parent_value), key,
        {'proposal_closed':closed})


def option(key, state=None, operator='improve'):
    return ExpansionOption(key, operator, (), state or features(), NS(id=key), OperatorPreview(operator))


class TestKernelAndPosterior(unittest.TestCase):
    def test_matern_matches_reference_and_category_formula(self):
        rng=np.random.default_rng(4); x=rng.normal(size=(10,6));ops=np.arange(10)%4
        k=mixed_matern32(x,ops,x,ops,length_scale=.8,amplitude=1.2,rho=.25)
        expected=1.2**2 * Matern(length_scale=.8, nu=1.5)(x) * (.25+.75*(ops[:,None]==ops[None,:]))
        assert_allclose(k,expected,atol=1e-12)
        self.assertGreater(np.linalg.eigvalsh(k).min(), -1e-10)

    def test_prior_retains_uncertainty_with_zero_samples(self):
        model=GaussianProposalSurrogate();model.fit([])
        mean,latent,predictive=model.predict_quality([option('a')])
        assert_allclose(mean,[.5]);assert_allclose(latent,[1.])
        self.assertGreater(predictive[0],latent[0])

    def test_exact_posterior_matches_full_matrix_formula(self):
        rows=[record('s',value=.5),record('a',parent='s',operator='improve',value=.6,parent_value=.5),
              record('b',parent='s',operator='evolution',value=.4,parent_value=.5)]
        model=GaussianProposalSurrogate();model.fit(rows)
        options=[option('a'),option('b',features(.4),operator='fusion')]
        mean,latent,predictive=model.predict_quality(options)
        x=np.array([model._design(dict(o.state_features),o.operator) for o in options])
        ops=np.array([GP6_OPERATORS.index(o.operator) for o in options])
        cross=model._kernel(x,ops,model.x,model.ops)
        cov=model._kernel(model.x,model.ops,model.x,model.ops)+(.25**2+1e-8)*np.eye(3)
        y=np.array([(r.child_utility-model.baseline(r.selection_features,r.operator))/model.utility_scale for r in rows])
        expected=np.array([.5,.4])+model.utility_scale*cross@np.linalg.solve(cov,y)
        var=1.-np.einsum('ij,ji->i',cross,np.linalg.solve(cov,cross.T))
        assert_allclose(mean,expected);assert_allclose(latent**2,model.utility_scale**2*var)
        assert_allclose(predictive**2-latent**2,(model.utility_scale*.25)**2)

    def test_negative_metric_root_uses_anchor_not_zero(self):
        model=GaussianProposalSurrogate();model.fit([record('s',value=-2.)])
        self.assertEqual(model.draft_anchor,-2.)
        mean,_,_=model.predict_quality([option('root',features(None,incumbent=-1.),'draft')])
        self.assertAlmostEqual(mean[0],-2.)

    def test_frozen_reference_resume_and_optimized_fit(self):
        rows=[record('s',value=.5)]
        model=GaussianProposalSurrogate();model.fit(rows);state=json.loads(json.dumps(model.state_dict()))
        for i in range(12):
            rows.append(record(str(i),parent='s',operator='improve',parent_value=.5,
                value=.5+.04*np.sin(i),state=features(.5,gp_trials=i,gp_last_gain=.02*np.cos(i))))
        model.fit(rows)
        restored=GaussianProposalSurrogate();restored.load_state_dict(state);restored.fit(rows)
        assert_allclose(model.hyperparameters,restored.hyperparameters,atol=1e-10)
        assert_allclose(model.predict_quality([option('x')]),restored.predict_quality([option('x')]))
        self.assertEqual(model.utility_scale,.025)
        self.assertTrue(model.fit_diagnostics['optimizer_status'].startswith('map_'))

    def test_all_identical_states_are_finite(self):
        model=GaussianProposalSurrogate()
        model.fit([record(str(i),value=0.) for i in range(15)])
        for a in model.predict_quality([option('r',features(None,0.),'draft')]):
            self.assertTrue(np.isfinite(a).all())

    def test_fixed_covariance_restores_unobserved_ridge_directions(self):
        rng=np.random.default_rng(42); rows=[]
        for i in range(3):
            f=dict(zip(FEATURE_NAMES,rng.normal(size=len(FEATURE_NAMES))))
            rows.append(record(str(i),operator='improve',parent_value=.5,value=.5+.1*i,state=f))
        m=BayesianGainModel();m.fit(rows)
        x=m.scaler.transform(np.array([list(r.selection_features.values()) for r in rows]));x-=x.mean(0)
        precision=m.model.lambda_*np.eye(49)+m.model.alpha_*x.T@x
        assert_allclose(precision@m.model.sigma_,np.eye(49),atol=1e-6)
        self.assertGreater(np.linalg.eigvalsh(m.model.sigma_).min(),0)

    def test_bad_parameters_fail_fast(self):
        for kwargs in ({'noise_std':0.}, {'operation_correlation':1.1}, {'history_window':0}):
            with self.assertRaises(ValueError):GP6Config(**kwargs)


class TestFeedbackAndFeatures(unittest.TestCase):
    def test_normal_and_draft_repair_prefix_and_full_cost(self):
        for operator,pv in [('improve',.5),('draft',None)]:
            rows=[record('p',parent='x',child='broken',operator=operator,value=None,parent_value=pv,closed=False),
                  record('d1',parent='broken',child='broken2',operator='debug',value=None,closed=False,cost=3),
                  record('d2',parent='broken2',child='good',operator='debug',value=.7,cost=4)]
            first=effective_proposals(rows[:1])[0]
            self.assertFalse(first.valid);self.assertFalse(first.metadata['proposal_closed'])
            out=effective_proposals(rows)[0]
            self.assertEqual(out.operator,operator);self.assertTrue(out.valid)
            self.assertEqual(out.decision_id,'p');self.assertEqual(out.total_wall_time_sec,9)
            self.assertEqual(out.selection_features,rows[0].selection_features)
            self.assertEqual(len(effective_proposals(rows)),1)
            self.assertFalse(rows[0].valid)
            model=GaussianProposalSurrogate();model.fit(rows)
            self.assertEqual(model.training_transition_ids,['p'])

    def test_first_success_wins_and_duplicates_do_not_double_count(self):
        rows=[record('p',child='broken',value=None,closed=False),
              record('d',parent='broken',operator='debug',value=.6),
              record('late',parent='broken',operator='debug',value=.9)]
        out=effective_proposals([*rows,rows[1]])
        self.assertEqual(out[0].child_utility,.6);self.assertEqual(out[0].total_wall_time_sec,4)

    def test_leaf_inherits_progress_and_excludes_other_branch(self):
        root=NS(id='root',stage='root',parent=None,metric=NS(value=None))
        seed=NS(id='seed',stage='draft',parent=root,metric=NS(value=.5),is_buggy=False,is_valid=True)
        leaf=NS(id='leaf',stage='improve',parent=seed,metric=NS(value=.6),is_buggy=False,is_valid=True)
        agent=NS(journal=NS(nodes=[root,seed,leaf]),scfg=NS(metric_improvement_threshold=1e-4),metric_maximize=True)
        rows=[record('s',child='seed',value=.5),record('i',parent='seed',child='leaf',operator='improve',value=.6,parent_value=.5),
              record('other',parent='elsewhere',operator='improve',value=.99,parent_value=.4)]
        f=progress_features(agent,leaf,rows)
        self.assertAlmostEqual(f['gp_last_gain'],.1);self.assertEqual(f['gp_trials'],2)
        self.assertEqual(f['gp_stagnation'],0)
        self.assertEqual(len(gp_vector(f,scale=.05,draft_anchor=.5,operator='improve')),6)

    def test_minimize_direction_and_zero_metric(self):
        node=NS(id='n',stage='draft',parent=None,metric=NS(value=0.),is_buggy=False,is_valid=True)
        agent=NS(journal=NS(nodes=[node]),scfg=NS(metric_improvement_threshold=0),metric_maximize=False)
        f=progress_features(agent,node,[])
        self.assertEqual(f['parent_utility'],0.);self.assertEqual(f['gp_valid_rate'],.5)
        self.assertEqual(f['gp_trials'],0.)

    def test_counts_do_not_explode_and_pending_is_not_failure(self):
        f=features(gp_trials=1e20,gp_stagnation=1e20,gp_last_gain=100000.)
        vector=gp_vector(f,scale=.01,draft_anchor=.5,operator='improve')
        self.assertTrue((np.abs(vector)<=1).all())
        root=NS(id='root',stage='root',parent=None,metric=NS(value=None))
        agent=NS(journal=NS(nodes=[root]),scfg=NS(metric_improvement_threshold=0),metric_maximize=True)
        row=record('pending',value=None,closed=False)
        f=progress_features(agent,root,[row]);self.assertEqual(f['gp_trials'],0.)


class TestSelectionAndPresets(unittest.TestCase):
    def test_ucb_uses_latent_not_predictive_and_beta_zero(self):
        class Fake:
            fitted_transition_count=0
            model_version='test'
            def predict_quality(self,options):return np.array([.8,.7]),np.array([.01,.2]),np.array([10.,.3])
        options=[option('known'),option('unknown')]
        s=GaussianProposalSelector(surrogate=Fake(),records_provider=lambda:[],beta=1.5)
        out=s.select(None,options);self.assertEqual(out.selected_option.parent_node_id,'unknown')
        s.beta=0.;out=s.select(None,options);self.assertEqual(out.selected_option.parent_node_id,'known')
        self.assertFalse(out.decision.fallback_used)

    def test_model_error_fallback_stays_in_legal_options(self):
        class Fake:
            fitted_transition_count=0
            draft_anchor=-2.
            def predict_quality(self,options):raise ValueError('test')
        options=[option('normal',features(-1.,-1.)),option('root',features(None,-1.),'draft')]
        out=GaussianProposalSelector(surrogate=Fake(),records_provider=lambda:[]).select(None,options)
        self.assertEqual(out.selected_option.parent_node_id,'normal');self.assertTrue(out.decision.fallback_used)

    def test_presets_only_differ_in_root_flag_and_tasklist_has_22(self):
        root=Path(__file__).resolve().parents[1]
        a=OmegaConf.to_container(OmegaConf.load(root/'config/experiments/gp6.yaml'))
        b=OmegaConf.to_container(OmegaConf.load(root/'config/experiments/gp6_draft.yaml'))
        self.assertFalse(a['parent_selector']['include_root_draft'])
        self.assertTrue(b['parent_selector']['include_root_draft'])
        b['parent_selector']['include_root_draft']=False;self.assertEqual(a,b)
        tasks=(root/'config/mlebench_lite22.txt').read_text().splitlines()
        self.assertEqual(len(tasks),22);self.assertEqual(len(set(tasks)),22)

    def test_model_fit_log_contains_training_outcomes_and_kernel(self):
        with tempfile.TemporaryDirectory() as path:
            model=GaussianProposalSurrogate(selection_logger=SelectionLogger(path))
            model.fit([record('s')])
            payload=json.loads((Path(path)/'model_fits.jsonl').read_text())
            self.assertEqual(len(payload['feature_names']),6)
            self.assertEqual(payload['fit_diagnostics']['training_outcome_transition_ids'],['s'])


if __name__=='__main__':unittest.main()
