"""Focused numeric + isolated production-parser/runtime checks; no LLM or GPU.

The parser function bodies are compiled unchanged; transport, node/metric types
and unrelated HTTP validity services are test doubles. The existing runtime
harness also isolates journal serialization. This is NOT a native AgentSearch
or real-provider test. Run: python -m unittest discover -s tests_gp6 -q
"""
import ast
import copy
from dataclasses import dataclass, replace
import json
import logging
import math
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import cast
import unittest
from unittest.mock import patch

import jsonschema
import numpy as np
from numpy.testing import assert_allclose
from omegaconf import OmegaConf

from agents import potential_rating
from engine.parent_selectors.rating_prior import RatingPriorConfig, potential_level, prior_terms
from engine.parent_selectors.base import OperatorPreview, ExpansionOption, SelectionDecision, SelectorResult, utc_timestamp
from engine.parent_selectors.gp_graph_features import GraphConfig
from engine.parent_selectors.gp_graph_surrogate import GraphProposalSurrogate, GraphProposalSelector
from engine.parent_selectors.gp_surrogate import GP6Config
from engine.parent_selectors.transition_store import TransitionRecord
from test_core import features, option
from test_graph_selectors import trace, model, A, ROOT
from test_long_horizon import LONG, attempt, seeded
from test_runtime_contract import Runtime, namespace
import test_runtime_contract as runtime_tests


@dataclass
class Spec:
    name: str
    json_schema: dict
    description: str


class ParsedNode(NS):
    def absorb_exec_result(self, result):
        self.term_out = ''.join(result.term_out)
        self.exc_type = result.exc_type


def parser_namespace():
    source = ROOT/'agents/result_parse_agent.py'
    tree = ast.parse(source.read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    ns = dict(FunctionSpec=Spec, SearchNode=object, ExecutionResult=object,
              cast=cast, time=time, logger=logging.getLogger('rating_tests'),
              potential_rating=potential_rating,
              MetricValue=lambda v, **kw: NS(value=v, **kw),
              WorstMetricValue=lambda: NS(value=None),
              wrap_code=lambda s, lang='python': f'```{lang}\n{s}\n```')
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), ns)
    ns.update(_check_submission_file=lambda *a: True,
              _validate_format_with_retry=lambda a,n: setattr(n, 'is_valid', True),
              _validate_metric_direction=lambda a,n,r: setattr(n, 'metric', NS(value=r['metric'], maximize=True)),
              _check_data_leakage=lambda *a: None,
              _save_to_global_memory=lambda *a: None)
    return ns


def agent_and_node(enabled=True, long=False, memory=False):
    cfg = runtime_tests.TestPresetLoading().loader([
        'experiment_preset=' + ('gp7_long_horizon_ucb_pibo' if long else 'gp7_short_logei_pibo'),
        f'bo_parent_selector.agent_rating.enabled={str(enabled).lower()}',
        f'agent.use_global_memory={str(memory).lower()}',
    ])
    parent = NS(id='parent', stage='draft', metric=NS(value=.70),
                agent_potential_level=1, is_buggy=False, is_valid=True)
    node = ParsedNode(id='new', parent=parent, stage='improve', code='print("train")',
                      is_buggy=None, is_valid=None, metric=None, analysis=None,
                      phase1_selection_features={'agent_potential_level':1},
                      agent_potential_level=None, agent_potential_assessed_at=None,
                      exc_type=None)
    agent = NS(cfg=cfg, acfg=cfg.agent, scfg=cfg.agent.search,
               task_desc='Public validation task; no external data. Use the supplied budget.',
               metric_maximize=True, journal=NS(nodes=[parent]), search_start_time=None,
               phase1_runtime=NS(root_draft_state={'submitted_count':2}))
    result = NS(term_out=['validation metric: 0.75'], exc_type=None)
    return agent, node, result


class TestRatingMath(unittest.TestCase):
    def test_formula_and_missing_are_neutral(self):
        logs, terms = prior_terms([0,1,2,None,True,'2',2.0,9], beta_prior=25., iteration=10)
        assert_allclose(logs[:3], np.log([.5,1,2]))
        assert_allclose(terms, 2.5*logs)
        assert_allclose(terms[3:], 0)
        assert_allclose(prior_terms([0,1,2], beta_prior=0., iteration=1)[1], 0)

    def test_bad_settings_and_iteration(self):
        for beta in [-1, float('nan'), float('inf'), True, '25']:
            with self.assertRaises(ValueError): RatingPriorConfig(True, beta)
        for n in [0,-1,1.5,True,None]:
            with self.assertRaises(ValueError): prior_terms([1], beta_prior=25., iteration=n)
        with self.assertRaises(ValueError): RatingPriorConfig(enabled='false')
        for v in [None,True,False,2.,'1',-1,3]: self.assertIsNone(potential_level(v))

    def test_multiplication_is_on_ei_not_logei(self):
        base = np.array([-3.593793,-3.942243])
        _, bonus = prior_terms([0,1],beta_prior=25.,iteration=30)
        assert_allclose(base+bonus, np.log(np.exp(base)*np.array([.5,1])**(25/30)))
        self.assertEqual(int(np.argmax(base)),0)
        self.assertEqual(int(np.argmax(base+bonus)),1)

    def test_numerical_gp_stays_unchanged_and_candidate_order(self):
        rows, opts = trace(10)
        opts=[replace(o,state_features={**o.state_features,'agent_potential_level':i,'pibo_iteration':12})
              for i,o in enumerate(opts)]
        gp=model();base=GraphProposalSelector(surrogate=gp, records_provider=lambda: rows)
        original=base.select(None,opts)
        prior=GraphProposalSelector(surrogate=gp, records_provider=lambda: rows,
                                   rating_config=RatingPriorConfig(True,25.))
        result=prior.select(None,opts)
        _,bonus=prior_terms([0,1,2],beta_prior=25.,iteration=12)
        for i,(a,b) in enumerate(zip(original.candidate_scores,result.candidate_scores)):
            self.assertEqual(a.posterior_mean_quality,b.posterior_mean_quality)
            self.assertEqual(a.posterior_std_latent,b.posterior_std_latent)
            self.assertEqual(a.predictive_std,b.predictive_std)
            self.assertAlmostEqual(b.acquisition_score,a.acquisition_score+bonus[i])
        self.assertEqual(result.selected_option,opts[int(np.argmax([s.acquisition_score for s in result.candidate_scores]))])
        self.assertEqual(len(gp.feature_names),7)
        prior.rating_config=RatingPriorConfig(True,0.)
        zero=prior.select(None,opts)
        self.assertEqual(zero.candidate_scores,original.candidate_scores)
        self.assertEqual(zero.selected_option,original.selected_option)

    def test_ucb_prior_is_scaled_and_preserves_negative_quality_order(self):
        for negative in [False, True]:
            rows, opts = trace(10, negative=negative)
            opts = [replace(o, state_features={**o.state_features,
                    'agent_potential_level': i, 'pibo_iteration': 12}) for i, o in enumerate(opts)]
            gp = model(acquisition='ucb')
            selector = GraphProposalSelector(surrogate=gp, records_provider=lambda: rows)
            plain = selector.select(None, opts)
            selector.rating_config = RatingPriorConfig(True, 25.)
            rated = selector.select(None, opts)
            self.assertFalse(rated.decision.fallback_used, rated.decision.fallback_reason)
            base = np.asarray([s.acquisition_score for s in plain.candidate_scores])
            _, bonus = prior_terms([0, 1, 2], beta_prior=25., iteration=12)
            expected = base + gp.utility_scale * bonus
            assert_allclose([s.acquisition_score for s in rated.candidate_scores], expected)
            self.assertLess(expected[0], base[0])
            self.assertEqual(expected[1], base[1])
            self.assertGreater(expected[2], base[2])
            self.assertEqual(rated.selected_option, opts[int(np.argmax(expected))])
            selector.rating_config = RatingPriorConfig(True, 0.)
            self.assertEqual(selector.select(None, opts).candidate_scores, plain.candidate_scores)

    def test_ancestry_false_equals_zero_weight_for_fit_and_predict(self):
        for long in [False,True]:
            if long:
                r1=attempt('a',1,value=.76)
                r2=attempt('b',2,parent=r1.child_node_id,parent_value=.76,value=.91,
                           chain=[r1.child_node_id,'s'])
                rows=seeded(r1,r2)
                opts=[option('s',r1.selection_features,'improve')]
            else: rows,opts=trace(10)
            mode=LONG if long else A
            kwargs={'long_horizon':{'horizon':2}} if long else {}
            one=GraphProposalSurrogate(mode=mode,graph_config={'use_ancestry_kernel':False},**kwargs)
            two=GraphProposalSurrogate(mode=mode,graph_config={'graph_weight':0.},**kwargs)
            one.fit(rows);two.fit(rows)
            assert_allclose(one.hyperparameters,two.hyperparameters,atol=1e-12)
            assert_allclose(one.predict_quality(opts),two.predict_quality(opts),atol=1e-12)
            self.assertEqual(len(one.feature_names),7)
        with self.assertRaises(ValueError): GraphConfig(use_ancestry_kernel='false')

    def test_old_gp_checkpoint_default_and_changed_ancestry_rejected(self):
        gp=model();rows,_=trace(5);gp.fit(rows)
        old=gp.state_dict();old['graph_config'].pop('use_ancestry_kernel')
        model().load_state_dict(old)
        with self.assertRaises(ValueError):
            model(graph_config={'use_ancestry_kernel':False}).load_state_dict(old)


class TestRatingReview(unittest.TestCase):
    def test_disabled_legacy_prompt_schema_and_behavior_unchanged(self):
        for long in [False,True]:
            a,n,e=agent_and_node(False,long,True);ns=parser_namespace();calls=[]
            expected={'Introduction':ns['_build_introduction'](a),
                      'Implementation':ns['wrap_code'](n.code),
                      'Execution output':ns['wrap_code'](''.join(e.term_out),lang='')}
            response={'is_bug':False,'summary':'Observed 0.75.','metric':.75,
                      'lower_is_better':False,'code_summary':'Train classifier.',
                      'potential_level':2}
            def query(**kw): calls.append(kw);return dict(response)
            ns['query']=query
            ns['run'](a,n,e)
            self.assertEqual(len(calls),1)
            self.assertEqual(calls[0]['system_message'],expected)
            self.assertEqual(calls[0]['func_spec'].json_schema,ns['get_review_func_spec'](True).json_schema)
            self.assertIsNone(n.agent_potential_level)
            self.assertEqual(n.analysis,'Observed 0.75.')
            self.assertEqual(n.metric.value,.75)

    def test_enabled_one_generic_score_both_targets_and_memory_modes(self):
        prompts=[]
        for long in [False,True]:
            for memory in [False,True]:
                a,n,e=agent_and_node(True,long,memory);ns=parser_namespace();calls=[]
                def query(**kw):
                    calls.append(kw)
                    return dict(is_bug=False,summary='Observed 0.75.',metric=.75,
                                lower_is_better=False,code_summary='Train classifier.',potential_level=2)
                ns['query']=query;ns['run'](a,n,e)
                self.assertEqual(len(calls),1)
                schema=calls[0]['func_spec'].json_schema
                jsonschema.Draft7Validator.check_schema(schema)
                self.assertEqual(set(schema['properties'])-set(ns['get_review_func_spec'](memory).json_schema['properties']),{'potential_level'})
                self.assertEqual(n.agent_potential_level,2)
                self.assertIsNotNone(n.agent_potential_assessed_at)
                self.assertEqual(n.parent.agent_potential_level,1)
                self.assertEqual(n.phase1_selection_features,{'agent_potential_level':1})
                prompt=calls[0]['system_message'];prompts.append(prompt['Generic potential assessment'])
                context=json.loads(prompt['Completed-node assessment context'])
                self.assertEqual(context['completed_node_id'],'new')
                self.assertEqual(context['known_best_metric_before_this_result'],.70)
                self.assertEqual(prompt['Task and constraints for potential assessment'],a.task_desc)
                self.assertNotIn('potential_evidence',schema['properties'])
                self.assertEqual(n.analysis,'Observed 0.75.')
        self.assertTrue(all(p==prompts[0] for p in prompts))

    def test_extended_schema_survives_real_codex_normalizer(self):
        a,n,_=agent_and_node();ns=parser_namespace()
        prompt,spec=potential_rating.extend_review(a,n,{'Introduction':'Original facts'},ns['get_review_func_spec'](False))
        source=ROOT/'llm/codex.py'
        fn=next(x for x in ast.parse(source.read_text()).body
                if isinstance(x,ast.FunctionDef) and x.name=='_strict_output_schema')
        env={'deepcopy':copy.deepcopy}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),env)
        normalized=env['_strict_output_schema'](spec.json_schema)
        self.assertIsNotNone(normalized)
        self.assertFalse(normalized['additionalProperties'])
        self.assertIn('potential_level',normalized['required'])
        for level in [None,0,1,2]:
            jsonschema.validate(dict(is_bug=False,summary='Facts',metric=.75,lower_is_better=False,
                                     potential_level=level), normalized)
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(dict(is_bug=False,summary='Facts',metric=.75,lower_is_better=False,
                                     potential_level=True), normalized)

    def test_bad_missing_rating_does_not_retry_or_invalidate(self):
        for raw in [None,True,'2',2.,-1,3,{'score':2}]:
            a,n,e=agent_and_node();ns=parser_namespace();calls=[]
            def query(**kw):
                calls.append(kw)
                return dict(is_bug=False,summary='Observed',metric=.75,lower_is_better=False,potential_level=raw)
            ns['query']=query;ns['run'](a,n,e)
            self.assertEqual(len(calls),1)
            self.assertFalse(n.is_buggy)
            self.assertEqual(n.metric.value,.75)
            self.assertIsNone(n.agent_potential_level)

    def test_failed_or_host_invalidated_node_is_not_high_potential(self):
        for failure in ['llm','host']:
            a,n,e=agent_and_node();ns=parser_namespace()
            ns['query']=lambda **kw: dict(is_bug=failure=='llm',summary='Observed',metric=.75,
                                         lower_is_better=False,potential_level=2)
            if failure=='host':
                def bad(a,n): n.is_buggy=True;n.is_valid=False
                ns['_validate_format_with_retry']=bad
            ns['run'](a,n,e)
            self.assertTrue(n.is_buggy)
            self.assertIsNone(n.agent_potential_level)
            self.assertEqual(n.parent.agent_potential_level,1)

    def test_fixed_evaluation_summary_enabled_disabled_and_authoritative_metric(self):
        for on in [False,True]:
            a,n,e=agent_and_node(on);ns=parser_namespace();calls=[]
            a.fixed_evaluation=NS(protocol_id='fixed',candidate_contract=lambda:'No test access.',
                score=lambda n:dict(valid=True,metric=.812,error=None),spec={'metric':{'maximize':True}})
            def query(**kw):
                calls.append(kw);return dict(is_bug=False,summary='Runtime completed.',code_summary='Train',potential_level=2,metric=.99)
            ns['query']=query;ns['run'](a,n,e)
            self.assertEqual(len(calls),1)
            self.assertEqual(n.metric.value,.812)
            self.assertEqual(n.agent_potential_level,2 if on else None)
            self.assertEqual('potential_level' in calls[0]['func_spec'].json_schema['properties'],on)
            self.assertEqual(n.analysis,'Runtime completed.')
            if not on:
                self.assertEqual(set(calls[0]['system_message']),{'Introduction','Fixed evaluation','Authoritative program result','Implementation','Execution output'})

    def test_fixed_invalid_early_exit_and_summary_failure_still_bounded(self):
        for valid in [False,True]:
            a,n,e=agent_and_node();ns=parser_namespace();calls=[]
            a.fixed_evaluation=NS(protocol_id='fixed',candidate_contract=lambda:'contract',
                score=lambda n:dict(valid=valid,metric=.812 if valid else None,error=None),spec={'metric':{'maximize':True}})
            def query(**kw): calls.append(kw);raise RuntimeError('transport unavailable')
            ns['query']=query;ns['run'](a,n,e)
            self.assertEqual(len(calls),int(valid))
            self.assertIsNone(n.agent_potential_level)
            self.assertEqual(n.metric.value,.812 if valid else None)


class TestRatingRuntime(unittest.TestCase):
    def make(self,long=False,enabled=True):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        c=runtime_tests.TestPresetLoading().loader(['experiment_preset='+('gp7_long_horizon_ucb_pibo' if long else 'gp7_short_logei_pibo'),
                                    f'bo_parent_selector.agent_rating.enabled={str(enabled).lower()}'])
        c.log_dir=tmp.name;c.exp_id='synthetic'
        return Runtime(c),c

    def test_two_presets_and_cli_switches(self):
        for long in [False,True]:
            r,c=self.make(long)
            self.assertEqual(r.selector_type,LONG if long else A)
            self.assertTrue(r.rating_enabled)
            self.assertEqual(r.rating_config.beta_prior,25)
            self.assertTrue(r.surrogate.graph_config.use_ancestry_kernel)
            self.assertFalse(r.include_root_draft)
            self.assertEqual(r.surrogate.is_long_horizon,long)
            args=['experiment_preset='+('gp7_long_horizon_ucb_pibo' if long else 'gp7_short_logei_pibo'),
                  'bo_parent_selector.gp.graph.use_ancestry_kernel=false',
                  'bo_parent_selector.agent_rating.enabled=false']
            c=runtime_tests.TestPresetLoading().loader(args);c.log_dir=r.cfg.log_dir
            off=Runtime(c);self.assertFalse(off.rating_enabled)
            self.assertEqual(off.surrogate.graph_config.effective_graph_weight,0)

    def test_counter_reservation_not_training_count_or_repeated_scoring(self):
        r,c=self.make(True)
        r.root_draft_state=dict(pending={},decision_count=99,rating_bo_decision_count=0)
        r._root_legality=lambda a:'disabled';r._persist_root_state=lambda a:None
        for i,op in enumerate(['draft','improve','debug','evolution','aggregation','fusion']):
            opt=option('s',features(.7),op)
            decision=SelectionDecision(r.selector_type,'s',op,0.,'gp',False,str(i),utc_timestamp())
            result=SelectorResult(decision,opt,(),(opt,))
            r._reserve_decision(None,result)
            r._reserve_decision(None,result)  # a duplicate reservation is not a new choice
        self.assertEqual(r.root_draft_state['rating_bo_decision_count'],3)
        self.assertEqual(r.root_draft_state['decision_count'],105)
        self.assertEqual(len(r.transition_store.records()),0)

    def test_completed_child_rating_enters_only_its_future_option_and_logs(self):
        for long in [False,True]:
            r,c=self.make(long);rows,_=trace(6)
            if long:
                rows=[replace(row,metadata={**row.metadata,'long_horizon_decision_index':i})
                      for i,row in enumerate(rows)]
            for row in rows:r.transition_store.append(row)
            root=NS(id='root',stage='root',parent=None,metric=NS(value=None))
            parent=NS(id='s',stage='draft',parent=root,metric=NS(value=.7),visits=0,total_reward=0.,
                      children=[],branch_id=1,is_buggy=False,is_valid=True,
                      agent_potential_level=2,agent_potential_assessed_at='2026-09-15T10:00:00+00:00')
            agent=NS(journal=NS(nodes=[root,parent]),metric_maximize=True,scfg=c.agent.search,acfg=c.agent,
                     best_node=parent,branch_successful_nodes={},is_root=lambda n:n.id=='root')
            r.root_draft_state={'rating_bo_decision_count':5}
            r._incumbent=lambda a:.8;r._global_stagnation=lambda a:0.
            raw=features(.7,.8,depth=1.,branch_stagnation=0.,recent_gain_mean=0.,recent_gain_std=0.,
                         recent_gain_max=0.,recent_valid_rate=1.,elapsed_fraction=0.,remaining_steps_fraction=1.)
            with patch.dict(namespace,{'extract_selection_features':lambda *a,**k:dict(raw)}):
                opt=r._make_option(agent,parent,OperatorPreview('improve'))
            self.assertEqual(opt.state_features['agent_potential_level'],2)
            self.assertEqual(opt.state_features['pibo_iteration'],6)
            parent.agent_potential_level=0
            self.assertEqual(opt.state_features['agent_potential_level'],2)
            result=r.selectors[r.selector_type].select(agent,[opt])
            self.assertFalse(result.decision.fallback_used,result.decision.fallback_reason)
            self.assertEqual(r.root_draft_state['rating_bo_decision_count'],5)
            r.managed_decisions=False
            r._log_selection(agent,result)
            snap=json.loads((r.artifact_root/'selection_snapshots.jsonl').read_text().splitlines()[-1])
            self.assertEqual(len(snap['gp_feature_vector']),7)
            self.assertEqual(snap['pibo_iteration'],6)
            scale = r.surrogate.utility_scale if long else 1.
            self.assertAlmostEqual(snap['rating_prior_bonus'],scale*25/6*math.log(2))
            self.assertEqual(snap['acquisition_kind'],'pibo_child_ucb' if long else 'pibo_global_logei')
            self.assertAlmostEqual(snap['acquisition_score'],
                                   snap['base_ucb' if long else 'base_logei']+snap['rating_prior_bonus'])

    def test_config_resume_rejects_changed_beta_and_ancestry(self):
        loader=runtime_tests.TestPresetLoading().loader
        with tempfile.TemporaryDirectory() as tmp:
            logs=Path(tmp)/'logs';logs.mkdir()
            for preset in ['gp7_short_logei_pibo','gp7_long_horizon_ucb_pibo','gp7_short_ucb_pibo']:
                c=loader([f'experiment_preset={preset}'])
                OmegaConf.save(c,logs/'config.yaml')
                self.assertTrue(loader([f'experiment_preset={preset}',f'resume_run_dir={tmp}']).bo_parent_selector.agent_rating.enabled)
                for extra in ['bo_parent_selector.agent_rating.beta_prior=30',
                              'bo_parent_selector.agent_rating.enabled=false',
                              'bo_parent_selector.gp.graph.use_ancestry_kernel=false']:
                    with self.assertRaises(ValueError):
                        loader([f'experiment_preset={preset}',f'resume_run_dir={tmp}',extra])
                c.bo_parent_selector.gp.graph.use_ancestry_kernel=False
                OmegaConf.save(c,logs/'config.yaml')
                resumed=loader([f'experiment_preset={preset}',f'resume_run_dir={tmp}',
                                'bo_parent_selector.gp.graph.use_ancestry_kernel=false'])
                self.assertFalse(resumed.bo_parent_selector.gp.graph.use_ancestry_kernel)

    def test_counter_and_rating_survive_actual_runtime_ledger_resume(self):
        for long in [False,True]:
            r,c=self.make(long)
            root=NS(id='root',stage='root',agent_potential_level=None,agent_potential_assessed_at=None)
            child=NS(id='s',stage='draft',agent_potential_level=2,agent_potential_assessed_at='2026-09-15T10:00:00Z')
            a=NS(acfg=c.agent,scfg=c.agent.search,metric_maximize=True,journal=NS(nodes=[root,child]))
            # Runtime persistence/contract logic is production code. Only the
            # external Journal/serializer and RNG services are isolated doubles.
            module=ModuleType('engine.search_node');module.Journal=lambda **kw:NS(**kw)
            def dump(v): return json.dumps({'nodes':[vars(n) for n in v.nodes]})
            def load(s,cls): return NS(nodes=[NS(**n) for n in json.loads(s)['nodes']])
            util=ModuleType('utils');util.serialize=NS(dumps_json=dump,loads_json=load)
            seed=ModuleType('utils.seed');seed.capture_random_state=lambda:('python','numpy')
            seed.restore_random_state=lambda *args:None
            with patch.dict(sys.modules,{'engine.search_node':module,'utils':util,'utils.seed':seed}):
                r.bind_agent(a)
                r.root_draft_state['rating_bo_decision_count']=7
                r._persist_root_state(a)
                saved=copy.deepcopy(r.root_draft_state)
                restored=NS(cfg=c,acfg=c.agent,scfg=c.agent.search,metric_maximize=True,
                            journal=NS(nodes=[root,child],parent_selector_state=saved))
                rr=Runtime(c);rr.bind_agent(restored,resuming=True)
            self.assertEqual(rr.root_draft_state['rating_bo_decision_count'],7)
            self.assertEqual(restored.journal.nodes[1].agent_potential_level,2)
            self.assertEqual(rr.root_draft_state['agent_rating']['beta_prior'],25.)

    def test_actual_shell_wrappers_accept_pibo_presets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'scripts').mkdir();(root/'config').mkdir()
            for name in ['run_selector_task.sh','run_selector_suite.sh']:
                shutil.copy2(ROOT/'scripts'/name,root/'scripts'/name)
            (root/'config/mlebench_lite22.txt').write_text('synthetic-task\n')
            (root/'run_single_task.sh').write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$CAPTURE"\n')
            env={**os.environ,'TIME_LIMIT_SECS':'1','CPUS_PER_TASK':'1','CAPTURE':str(root/'calls')}
            for preset in ['gp7_short_logei_pibo','gp7_long_horizon_ucb_pibo','gp7_short_ucb_pibo']:
                subprocess.run(['bash',str(root/'scripts/run_selector_suite.sh'),preset,'/fake-data','112'],
                               env=env,check=True,capture_output=True,text=True)
            calls=(root/'calls').read_text().splitlines()
            self.assertEqual(len(calls),3)
            self.assertIn('experiment_preset=gp7_short_logei_pibo',calls[0])
            self.assertIn('experiment_preset=gp7_long_horizon_ucb_pibo',calls[1])
            self.assertIn('experiment_preset=gp7_short_ucb_pibo',calls[2])


if __name__ == '__main__':
    unittest.main()
