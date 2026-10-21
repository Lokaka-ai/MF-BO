"""Targeted no-GPU tests; Codex/provider is a fake response or executable.

The runtime tests use the existing isolated production-class harness. They do
not claim a live provider or full AgentSearch benchmark has run.
"""
from dataclasses import replace
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS, ModuleType
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose
from omegaconf import OmegaConf
from scipy.special import ndtr

from engine.parent_selectors.agent_selector import AgentParentSelector, AgentSelectionError, AgentSettings, _text
from engine.parent_selectors.gp_variants import GPVariantSurrogate, GPVariantSelector, stable_log_ei
from engine.parent_selectors.gp_surrogate import GaussianProposalSurrogate
from engine.parent_selectors.base import ExpansionOption, OperatorPreview
from engine.selection_logging import SelectionLogger
from test_core import record, option, features
from test_runtime_contract import Runtime, namespace, SOURCE
import test_runtime_contract as runtime_fixture

ROOT = Path(__file__).resolve().parents[1]


def cfg_for(mode, log_dir):
    cfg = OmegaConf.merge(OmegaConf.load(ROOT/'config/config.yaml'),
                          OmegaConf.load(ROOT/f'config/experiments/{mode}.yaml'))
    cfg.log_dir = str(log_dir)
    return cfg


def toy_pool(n=3):
    nodes = [NS(id=(f'{i+1:032x}'), code='from sklearn.ensemble import RandomForestClassifier\nlr=0.001',
                plan='Improve representation', analysis='Validation accuracy .7',
                stage='improve', parent=None, metric=NS(value=.7), is_buggy=False,
                is_valid=True, lock=False, is_terminal=False, children=[], branch_id=i,
                visits=5, total_reward=1000., step=i, ctime=0)
             for i in range(n)]
    options = [ExpansionOption(node.id, 'improve', (), features(.7, .8, mcgs_q=1000., elapsed_fraction=.25,
               remaining_steps_fraction=.5), node, OperatorPreview('improve')) for node in nodes]
    agent = NS(journal=NS(nodes=nodes), task_desc='Classify examples using validation accuracy.', metric_maximize=True)
    return agent, options


class TestBOComparison(unittest.TestCase):
    def test_shrink_preserves_fit_and_only_changes_predictions(self):
        rows = [record('s', value=.5), record('a', parent='s', operator='improve', value=.65, parent_value=.5)]
        plain = GaussianProposalSurrogate()
        half = GPVariantSurrogate(variant_config=dict(shrink=.5, acquisition='ucb', beta=1.5))
        plain.fit(rows); half.fit(rows)
        opts = [option('x'), option('y', features(.4))]
        mu, sd, pred = plain.predict_quality(opts)
        hm, hs, hp = half.predict_quality(opts)
        centers = np.array([.5, .4])
        assert_allclose(hm, centers+.5*(mu-centers))
        assert_allclose(hs, .5*sd)
        assert_allclose(hp**2-hs**2, pred**2-sd**2)
        assert_allclose(plain.hyperparameters, half.hyperparameters)
        assert_allclose(plain.weights, half.weights)

    def test_logei_matches_analytic_ei_and_extreme_tail(self):
        z = np.array([-8., -3., -1., 0., 1., 8.])
        expected = z*ndtr(z)+np.exp(-.5*z*z)/np.sqrt(2*np.pi)
        assert_allclose(np.exp(stable_log_ei(z, np.ones(6), 0)), expected, rtol=1e-10, atol=1e-18)
        vals = stable_log_ei(np.array([-50., -49.]), np.ones(2), 0.)
        self.assertTrue(np.isfinite(vals).all()); self.assertGreater(vals[1], vals[0])

    def test_zero_sigma_and_invalid_sigma(self):
        x=stable_log_ei(np.array([1., 0., -1.]), np.zeros(3), 0)
        self.assertEqual(x[0], 0); self.assertTrue(np.isneginf(x[1:]).all())
        with self.assertRaises(ValueError): stable_log_ei([0.], [-1.], 0.)

    def test_choice_depends_on_actual_method(self):
        means=np.array([.81,.82,.76]); sd=np.array([.008,.04,.10]); baselines=np.array([.80,.70,.68])
        opts=[option(str(i), features(float(b),.8), 'improve') for i,b in enumerate(baselines)]
        for kind,alpha,expected in [('ucb',.5,0),('logei',1.,1)]:
            surrogate=GPVariantSurrogate(variant_config=dict(shrink=alpha,acquisition=kind,beta=1.5))
            surrogate.fitted_transition_count=0; surrogate.model_version='test'; surrogate.utility_scale=.05
            surrogate.predict_quality=lambda options,a=alpha:(baselines+a*(means-baselines),a*sd,np.sqrt((a*sd)**2+.01**2))
            out=GPVariantSelector(surrogate=surrogate,records_provider=list).select(None,opts)
            self.assertEqual(out.selected_option.parent_node_id,str(expected))
            self.assertFalse(out.decision.fallback_used)

    def test_logei_unit_invariance_and_global_threshold(self):
        mean=np.array([-.2,-.3]);sigma=np.array([.05,.15]); incumbent=-.1
        a=stable_log_ei(mean,sigma,incumbent)
        b=stable_log_ei(mean*100+7,sigma*100,incumbent*100+7)
        assert_allclose(b-a,math.log(100),rtol=1e-10)
        model=GPVariantSurrogate(variant_config=dict(shrink=1.,acquisition='logei',beta=1.5))
        model.fitted_transition_count=0
        # Missing incumbent is a logged fallback, never use candidate max as global best.
        opt=option('a'); opt=replace(opt,state_features={**opt.state_features,'incumbent_at_decision':None})
        out=GPVariantSelector(surrogate=model,records_provider=list).select(None,[opt])
        self.assertTrue(out.decision.fallback_used)
        self.assertIn('incumbent',out.decision.fallback_reason)

    def test_state_rejects_cross_variant_resume(self):
        half=GPVariantSurrogate(variant_config=dict(shrink=.5,acquisition='ucb',beta=1.5))
        ei=GPVariantSurrogate(variant_config=dict(shrink=1.,acquisition='logei',beta=1.5))
        half.fit([record('s')]); saved=half.state_dict()
        with self.assertRaises(ValueError): ei.load_state_dict(saved)
        other=GPVariantSurrogate(variant_config=dict(shrink=.5,acquisition='ucb',beta=1.5))
        other.load_state_dict(saved);other.fit([record('s')])
        assert_allclose(other.predict_quality([option('x')]),half.predict_quality([option('x')]))


class TestAgentSelector(unittest.TestCase):
    def make(self, callback, **overrides):
        settings=dict(retry_delay_seconds=0, **overrides)
        cfg=NS(agent_parent_selector=NS(**settings))
        return AgentParentSelector(cfg=cfg, records_provider=list, request_fn=callback)

    def test_selects_response_not_gp_q_or_first_candidate(self):
        agent, opts=toy_pool(); calls=[]
        def reply(**kwargs):
            calls.append(kwargs)
            return {'candidate':'c2','reason':'Try a distinct representation.'}
        model=self.make(reply);out=model.select(agent,opts)
        self.assertEqual(out.selected_option,opts[2]);self.assertEqual(len(calls),1)
        self.assertEqual(out.decision.selector_name,'agent_selector');self.assertFalse(out.decision.fallback_used)
        self.assertEqual([s.acquisition_score for s in out.candidate_scores],[0.,0.,1.])
        self.assertNotIn('mcgs_q', calls[0]['prompt'])
        self.assertNotIn(opts[0].parent_node_id, calls[0]['prompt'])
        self.assertIn('lr=0.001',calls[0]['prompt'])

    def test_one_invalid_response_can_be_repaired(self):
        agent,opts=toy_pool();responses=iter(['{"candidate":"root","reason":"draft"}', '{"candidate":"c1","reason":"valid choice"}'])
        calls=[]
        def reply(**kwargs):calls.append(kwargs);return next(responses)
        out=self.make(reply).select(agent,opts)
        self.assertEqual(out.selected_option,opts[1]);self.assertEqual(len(calls),2)
        self.assertLessEqual(calls[1]['timeout_seconds'],calls[0]['timeout_seconds'])

    def test_exhausted_invalid_response_raises_without_fallback(self):
        agent,opts=toy_pool();calls=[]
        def reply(**kwargs):calls.append(1);return 'not json'
        with self.assertRaises(AgentSelectionError): self.make(reply).select(agent,opts)
        self.assertEqual(len(calls),2)

    def test_single_candidate_does_not_invoke_provider(self):
        agent,opts=toy_pool(1)
        def bad(**kwargs):raise AssertionError('not called')
        out=self.make(bad).select(agent,opts)
        self.assertEqual(out.selected_option,opts[0]);self.assertEqual(out.decision.selection_reason,'only_legal_option')

    def test_root_not_a_model_option(self):
        agent,opts=toy_pool();opts[0]=replace(opts[0],operator='draft')
        with self.assertRaises(AgentSelectionError):self.make(None).select(agent,opts)

    def test_bound_preserves_all_candidates(self):
        agent,opts=toy_pool(12)
        for o in opts:o.parent_node.code='x=0.001\n'*10000;o.parent_node.analysis='observed text '*5000
        model=self.make(None,max_prompt_chars=12000)
        prompt,payload,_=model.build_prompt(agent,opts)
        self.assertLessEqual(len(prompt),12000);self.assertEqual(len(payload['candidates']),12)
        self.assertEqual(payload['remaining_time_fraction'],.75)
        self.assertNotIn('total_reward',prompt)

    def test_invalid_provider_and_retry_policy_rejected(self):
        for kw in [{'max_attempts':20},{'base_url':'https://provider'},{'reasoning_effort':'maximum'}]:
            with self.assertRaises(ValueError):AgentSettings(**kw)
        self.assertIn('0.001',_text('lr=0.001 /mnt/data/private/file.py '+ 'f'*40, 500))
        self.assertNotIn('/mnt/data',_text('/mnt/data/private/file.py',500))

    def test_decision_id_links_prompt_and_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent,opts=toy_pool()
            model=self.make(lambda **kw:'{"candidate":"c1","reason":"progress evidence"}')
            model.selection_logger=SelectionLogger(tmp)
            result=model.select(agent,opts)
            row=json.loads((Path(tmp)/'agent_selection_calls.jsonl').read_text())
            self.assertEqual(row['decision_id'],result.decision.decision_id)
            self.assertEqual(row['selected_candidate'],'c1')
            self.assertNotIn('api_key',row)


class TestNativePresetRouting(unittest.TestCase):
    def test_three_real_runtime_constructors(self):
        with tempfile.TemporaryDirectory() as tmp:
            for mode in ['halfshrink_ucb','gp6_global_logei','agent_selector']:
                cfg=cfg_for(mode,Path(tmp)/mode)
                runtime=Runtime(cfg)
                self.assertFalse(runtime.include_root_draft)
                self.assertTrue(runtime.managed_decisions)
                self.assertEqual(runtime.selector_type,mode)
                if mode=='agent_selector':
                    self.assertIsNone(runtime.surrogate)
                    self.assertEqual(set(runtime.selectors),{'agent_selector'})
                else:
                    self.assertIsInstance(runtime.surrogate,GPVariantSurrogate)
                    self.assertEqual(runtime.selectors[mode].name,mode)

    def test_no_root_override_or_mismatched_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            for mode in ['halfshrink_ucb','gp6_global_logei','agent_selector']:
                cfg=cfg_for(mode,tmp);cfg.parent_selector.include_root_draft=True
                with self.assertRaises(ValueError):Runtime(cfg)
            cfg=cfg_for('agent_selector',tmp);cfg.bo_parent_selector.model='gp_matern32'
            with self.assertRaises(ValueError):Runtime(cfg)

    def test_loader_recognizes_all_presets_and_rejects_cross_resume(self):
        loader=runtime_fixture.TestPresetLoading().loader
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp,'logs').mkdir()
            for mode in ['halfshrink_ucb','gp6_global_logei','agent_selector']:
                cfg=loader([f'experiment_preset={mode}'])
                self.assertEqual(cfg.parent_selector.type,mode)
                self.assertFalse(cfg.parent_selector.include_root_draft)
            saved=loader(['experiment_preset=halfshrink_ucb'])
            OmegaConf.save(saved,Path(tmp,'logs/config.yaml'))
            with self.assertRaises(ValueError):loader(['experiment_preset=gp6_global_logei',f'resume_run_dir={tmp}'])

    def test_failure_latch_blocks_further_provider_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime=Runtime(cfg_for('agent_selector',tmp));agent,opts=toy_pool()
            calls=[]
            def bad(**kw):calls.append(1);return 'bad'
            runtime.selectors['agent_selector'].request_fn=bad
            runtime.selectors['agent_selector'].settings=replace(runtime.selectors['agent_selector'].settings,retry_delay_seconds=0)
            for _ in range(2):
                with self.assertRaises(AgentSelectionError):runtime._rank_options(agent,opts)
            self.assertEqual(len(calls),2)


    def test_unexpected_context_failure_is_fatal_not_scheduler_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime=Runtime(cfg_for('agent_selector',tmp));agent,opts=toy_pool()
            def broken(*args):raise TypeError('bad context')
            runtime.selectors['agent_selector'].select=broken
            with self.assertRaises(AgentSelectionError):runtime._rank_options(agent,opts)
            self.assertIn('bad context',runtime._selector_failure)

    def test_agent_bind_and_persist_has_no_surrogate_state(self):
        # Production bind/atomic-ledger code; only journal serialization and RNG
        # modules are doubles (full agent dependencies are absent in this test).
        with tempfile.TemporaryDirectory() as tmp:
            runtime=Runtime(cfg_for('agent_selector',tmp));agent,opts=toy_pool()
            agent.acfg=NS(initial_drafts=3,steps=10,seed=42)
            agent.scfg=NS(num_drafts=5)
            engine_node=ModuleType('engine.search_node');engine_node.Journal=lambda **kw:NS(**kw)
            util=ModuleType('utils');util.serialize=NS(dumps_json=lambda value:json.dumps({'node_ids':[n.id for n in value.nodes]}))
            seed=ModuleType('utils.seed');seed.restore_random_state=lambda *args:None;seed.capture_random_state=lambda:('py','numpy')
            with patch.dict(sys.modules,{'engine.search_node':engine_node,'utils':util,'utils.seed':seed}):
                runtime.bind_agent(agent)
            state=json.loads((Path(tmp)/'phase1/root_draft_state.json').read_text())
            self.assertFalse(state['include_root_draft'])
            self.assertEqual(state['posterior'],{})
            self.assertNotIn('gp_reference',state)
            self.assertEqual(state['target'],'agent_selection_no_surrogate')
            self.assertEqual(state['agent_config']['base_url'],'opencode://opencode')
            self.assertEqual(state['initial_goal'],3)


class TestCodexTransport(unittest.TestCase):
    def test_actual_subprocess_command_with_fake_codex_binary(self):
        # Reuse the PRODUCTION transport; fake only imports it does not use here
        # and the external CLI executable. No live provider request is made.
        package=ModuleType('_transport_test');package.__path__=[]
        config=ModuleType('config');config.Config=object
        gemini=ModuleType('_transport_test.gemini');gemini.FunctionSpec=object;gemini.compile_prompt_to_md=str
        spec=importlib.util.spec_from_file_location('_transport_test.codex',ROOT/'llm/codex.py')
        module=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,{'_transport_test':package,'_transport_test.gemini':gemini,'config':config}):
            spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            binary=Path(tmp)/'fake_codex';capture=Path(tmp)/'args.json'
            binary.write_text('#!'+sys.executable+'\nimport sys,json,pathlib,os\n'
                'args=sys.argv[1:]\n'
                'pathlib.Path(os.environ["CAPTURE_ARGS"]).write_text(json.dumps({"args":args,"prompt":sys.stdin.read()}))\n'
                'pathlib.Path(args[args.index("--output-last-message")+1]).write_text(\'{"candidate":"c1","reason":"provider response"}\')\n')
            binary.chmod(0o755)
            with patch.dict(os.environ,{'MLEVOLVE_CODEX_BIN':str(binary),'CAPTURE_ARGS':str(capture)}):
                output, elapsed=module._run_codex(prompt='Select one option',cfg=NS(workspace_dir='/must-not-read'),
                    model='muse-spark-1.2-contributor',stage=NS(base_url='codex://opencode-go'),
                    json_schema={'type':'object','properties':{'candidate':{'type':'string'},'reason':{'type':'string'}},'required':['candidate','reason']},
                    timeout_seconds=5,reasoning_effort='medium',prompt_only=True)
            self.assertEqual(json.loads(output)['candidate'],'c1')
            args=json.loads(capture.read_text())['args']
            self.assertIn('model_provider="opencode-go"',args)
            self.assertIn('model_reasoning_effort="medium"',args)
            self.assertIn('features.shell_tool=false',args)
            self.assertIn('web_search="disabled"',args)
            self.assertNotEqual(args[args.index('--cd')+1],'/must-not-read')
            self.assertIn('--output-schema',args)
            self.assertGreaterEqual(elapsed,0.)


if __name__ == '__main__':unittest.main()
