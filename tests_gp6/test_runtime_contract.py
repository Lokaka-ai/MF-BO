"""Isolated tests of the production runtime's scheduling/feedback methods.

The class body is compiled unchanged from runtime.py so these tests can run in a
numerical-only environment. Agent, node and persistence are test doubles; this
is NOT an end-to-end AgentSearch or journal serialization test. Native integration
coverage is in tests/test_gp6_runtime.py when the agent dependencies are installed.
"""
import ast
import copy
from dataclasses import asdict, replace
import json
import math
import os
import runpy
import socket
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import threading
import time
import unittest
import uuid

from engine.parent_selectors.base import OperatorPreview, ExpansionOption, SelectionDecision, SelectorResult, utc_timestamp
from engine.parent_selectors.transition_store import TransitionStore, TransitionRecord
from engine.parent_selectors.rating_prior import RatingPriorConfig, potential_level

SOURCE=Path(__file__).resolve().parents[1]/'engine/parent_selectors/runtime.py'
study_config=runpy.run_path(str(SOURCE.parents[2]/'config/study7.py'))
branch_source=SOURCE.parents[2]/'config/branch7.py'
branch_tree=ast.parse(branch_source.read_text())
# Keep this harness usable without the agent dependency stack imported by
# config/__init__.py; the relative Study7Config dependency is already loaded.
branch_config={'Study7Config':study_config['Study7Config']}
exec(compile(ast.Module(body=[node for node in branch_tree.body
    if not (isinstance(node,ast.ImportFrom) and node.level)],type_ignores=[]),
    str(branch_source),'exec'),branch_config)
acq_source=SOURCE.parents[2]/'config/acq7.py'
acq_tree=ast.parse(acq_source.read_text())
acq_config={'Study7Config':study_config['Study7Config'],
            'Branch7ModelConfig':branch_config['Branch7ModelConfig']}
exec(compile(ast.Module(body=[node for node in acq_tree.body
    if not (isinstance(node,ast.ImportFrom) and node.level)],type_ignores=[]),
    str(acq_source),'exec'),acq_config)
rating_source=SOURCE.parents[2]/'config/rating_only.py'
rating_tree=ast.parse(rating_source.read_text())
rating_config={'Study7Config':study_config['Study7Config']}
exec(compile(ast.Module(body=[node for node in rating_tree.body
    if not (isinstance(node,ast.ImportFrom) and node.level)],type_ignores=[]),
    str(rating_source),'exec'),rating_config)
tree=ast.parse(SOURCE.read_text())
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Phase1SelectorRuntime')
namespace=dict(globals())
namespace.update({name: study_config[name] for name in (
    'STUDY7_SELECTORS', 'STATE4_SCHEMA', 'GREEDY_SCHEMA', 'study7_config')})
namespace.update({name: branch_config[name] for name in (
    'BRANCH7_SELECTORS', 'BRANCH7_SCHEMAS', 'branch7_config')})
namespace.update({name: acq_config[name] for name in (
    'ACQ7_SELECTORS', 'ACQ7_SCHEMA', 'acq7_config')})
namespace.update({name: rating_config[name] for name in (
    'RATING_ONLY_SELECTORS', 'RATING_ONLY_SCHEMA', 'rating_only_config')})
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),cls],type_ignores=[])),str(SOURCE),'exec'),namespace)
Runtime=namespace['Phase1SelectorRuntime']
namespace['utility_from_metric']=lambda m,maximize: None if m is None else (m if maximize else -m)
namespace['continuous_improvement']=lambda a,b,maximize: None if a is None or b is None else ((b-a) if maximize else (a-b))


class TestRuntimeContract(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.runtime=object.__new__(Runtime)
        r=self.runtime;r.include_root_draft=True;r.gp_enabled=True;r.agent_enabled=False;r.managed_decisions=True
        r.study7_enabled=False;r.wait_initial_results=True
        r.branch7_enabled=False;r.branch7_gp6_enabled=False;r.acq7_enabled=False
        r.cfg=NS()
        r._selection_lock=threading.RLock();r.feature_schema_version='gp6_progress_v1'
        r.transition_store=TransitionStore(Path(self.tmp.name)/'transitions.jsonl')
        root=NS(id='root',stage='root',metric=NS(value=None),
                children=[NS(stage='draft')],expected_child_count=1)
        root.reached_child_limit=lambda scfg: (
            sum(n.stage=='draft' for n in root.children)
            + max(0,root.expected_child_count-len(root.children)) >= scfg.num_drafts)
        journal=NS(nodes=[]);journal.append=journal.nodes.append
        self.agent=NS(virtual_root=root,journal=journal,metric_maximize=True,
                      scfg=NS(max_debug_depth=3,metric_improvement_threshold=0.,num_drafts=3),
                      acfg=NS(steps=5,time_limit=1000),search_start_time=None)
        r._agent=self.agent
        r.root_draft_state=dict(initial_terminal=1,initial_goal=1,initial_submitted=1,
            draft_submitted=1,draft_cap=3,submitted_count=1,decision_count=1,revision=0,pending={})
        r._incumbent=lambda a:.5
        r._persist_root_state=lambda a:r.root_draft_state.update(revision=r.root_draft_state['revision']+1)
        r._node_snapshot=lambda n:None
        r._make_option=lambda a,n,preview: ExpansionOption(n.id,preview.operator,(),{},n,preview)

    def context(self,key='x',operator='draft'):
        r=self.runtime
        parent=self.agent.virtual_root if operator=='draft' else NS(id='parent',stage='improve',metric=NS(value=.5))
        option=ExpansionOption(parent.id,operator,(),{'parent_utility':None if operator=='draft' else .5,
                               'incumbent_at_decision':.5},parent,OperatorPreview(operator))
        decision=SelectionDecision('gp_child_ucb',parent.id,operator,1.,'gp6-test',False,key,utc_timestamp())
        result=SelectorResult(decision,option,(),(option,))
        return NS(decision=decision,option=option,result=result)

    def test_switch_changes_root_candidate_only(self):
        r=self.runtime
        self.assertEqual(len(r._add_root_option(self.agent,[])),1)
        r.include_root_draft=False
        self.assertEqual(r._root_legality(self.agent),'disabled')
        self.assertEqual(r._add_root_option(self.agent,[]),[])
        self.assertTrue(r.managed_decisions)

    def test_pending_draft_prevents_second_root(self):
        r=self.runtime;c=self.context();r._reserve_decision(self.agent,c.result)
        self.assertEqual(r._root_legality(self.agent),'draft_pending')
        self.assertEqual(r._add_root_option(self.agent,[]),[])
        self.assertTrue(r.submit_decision(self.agent,c))
        self.assertFalse(r.submit_decision(self.agent,c))
        self.assertEqual(r.root_draft_state['draft_submitted'],2)

    def test_cancel_only_reserved_and_releases_slot(self):
        r=self.runtime;c=self.context();r._reserve_decision(self.agent,c.result)
        r.cancel_reservation(self.agent,c)
        self.assertEqual(r._root_legality(self.agent),'legal')
        self.assertEqual(r.root_draft_state['draft_submitted'],1)
        r._reserve_decision(self.agent,c.result);r.submit_decision(self.agent,c)
        r.cancel_reservation(self.agent,c)
        self.assertIn('x',r.root_draft_state['pending'])

    def test_pending_and_inflight_drafts_block_another_root(self):
        r=self.runtime;self.agent.scfg.num_drafts=2;c=self.context()
        r._reserve_decision(self.agent,c.result)
        self.assertEqual(r._root_legality(self.agent),'draft_pending')
        self.agent.virtual_root.expected_child_count=2
        self.assertEqual(r._root_legality(self.agent),'draft_cap_exhausted')

    def test_budget_includes_reservations(self):
        r=self.runtime;self.agent.acfg.steps=2
        r._reserve_decision(self.agent,self.context().result)
        self.assertFalse(r._budget_available(self.agent))

    def test_pending_is_not_a_training_observation(self):
        r=self.runtime;c=self.context();r._reserve_decision(self.agent,c.result)
        r.submit_decision(self.agent,c);r.mark_executing(self.agent,c)
        self.assertEqual(r.transition_store.records(),[])

    def test_terminal_record_is_idempotent_and_generation_failure_is_closed(self):
        r=self.runtime;c=self.context();r._reserve_decision(self.agent,c.result);r.submit_decision(self.agent,c)
        kwargs=dict(parent_node=c.option.parent_node,child_node=None,context=c,
            generation_wall_time_sec=2.,execution_wall_time_sec=0.,metric_maximize=True,
            reward=-1.,failure_reason='generation_failed')
        first=r.record_transition(**kwargs);second=r.record_transition(**kwargs)
        self.assertEqual(first.transition_id,second.transition_id)
        self.assertEqual(len(r.transition_store.records()),1)
        self.assertTrue(first.metadata['proposal_closed'])
        self.assertEqual(r.root_draft_state['pending'],{})

    def test_successful_draft_has_continuous_target_without_incumbent(self):
        r=self.runtime;c=self.context();c.option.state_features['incumbent_at_decision']=None
        r._reserve_decision(self.agent,c.result);r.submit_decision(self.agent,c)
        child=NS(id='c',metric=NS(value=.5),is_buggy=False,is_valid=True,is_terminal=False,debug_depth=0)
        row=r.record_transition(parent_node=c.option.parent_node,child_node=child,context=c,
             generation_wall_time_sec=1.,execution_wall_time_sec=1.,metric_maximize=True,reward=1.,failure_reason=None)
        self.assertTrue(row.valid);self.assertIsNone(row.parent_utility)
        self.assertEqual(row.child_utility,.5);self.assertTrue(row.metadata['target_available'])



# Configuration construction and root snapshot formatting also run the actual
# production class; unrelated legacy selectors are inert import-boundary doubles.
import numpy as np
from omegaconf import OmegaConf
from engine.parent_selectors.gp_surrogate import GaussianProposalSurrogate, GP6Config
from engine.parent_selectors.gp_selector import GaussianProposalSelector
from engine.parent_selectors.gp_variants import GPVariantSurrogate, GPVariantSelector
from engine.parent_selectors.agent_selector import AgentParentSelector, AgentSelectionError, AGENT_SCHEMA
from engine.parent_selectors.gp_features import GP6_SCHEMA, GP6_NAMES, progress_features
from engine.parent_selectors.gp_graph_features import GP7_SCHEMA, GRAPH_SELECTOR_TYPES, DecisionGraph
from engine.parent_selectors.gp_graph_surrogate import GraphProposalSurrogate, GraphProposalSelector
namespace.update(GP7_SCHEMA=GP7_SCHEMA, GRAPH_SELECTOR_TYPES=GRAPH_SELECTOR_TYPES,
                 DecisionGraph=DecisionGraph, GraphProposalSurrogate=GraphProposalSurrogate,
                 GraphProposalSelector=GraphProposalSelector)
from engine.parent_selectors import features as feature_module
from engine.selection_logging import SelectionLogger

config_source=SOURCE.parents[2]/'config/__init__.py'
config_tree=ast.parse(config_source.read_text())
validator=next(n for n in config_tree.body if isinstance(n,ast.FunctionDef) and n.name=='validate_root_draft_config')
exec(compile(ast.Module(body=[validator],type_ignores=[]),str(config_source),'exec'),namespace)
for node in tree.body:
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id.startswith(('SUPPORTED_','BNVS_SELECTOR','GP_SELECTOR','BO_SELECTOR')) for t in node.targets):
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE),'exec'),namespace)
namespace.update(AgentParentSelector=AgentParentSelector, AgentSelectionError=AgentSelectionError, AGENT_SCHEMA=AGENT_SCHEMA,
                 GPVariantSurrogate=GPVariantSurrogate, GPVariantSelector=GPVariantSelector,
                 np=np,GaussianProposalSurrogate=GaussianProposalSurrogate,GP6Config=GP6Config,
                 GaussianProposalSelector=GaussianProposalSelector,SelectionLogger=SelectionLogger,
                 GP6_SCHEMA=GP6_SCHEMA,GP6_NAMES=GP6_NAMES,progress_features=progress_features,
                 ProgressiveMCGSSelector=lambda:None,RandomParentSelector=lambda:None,
                 ParentMetricGreedySelector=lambda:None)
for key in ['BNVS_FEATURE_SCHEMA_VERSION','BNVS_ROOT_FEATURE_SCHEMA_VERSION','FEATURE_SCHEMA_VERSION',
            'REWARD_ONLY_FEATURE_SCHEMA_VERSION','extract_selection_features','extract_root_draft_features']:
    namespace[key]=getattr(feature_module,key)


class TestRuntimeConstruction(unittest.TestCase):
    def test_both_presets_construct_the_actual_gp_runtime(self):
        root=SOURCE.parents[2]
        with tempfile.TemporaryDirectory() as path:
            for mode,enabled in [('gp6',False),('gp6_draft',True)]:
                cfg=OmegaConf.merge(OmegaConf.load(root/'config/config.yaml'),OmegaConf.load(root/f'config/experiments/{mode}.yaml'))
                cfg.log_dir=str(Path(path)/mode)
                runtime=Runtime(cfg)
                self.assertEqual(runtime.include_root_draft,enabled)
                self.assertTrue(runtime.gp_enabled and runtime.managed_decisions)
                self.assertIsInstance(runtime.surrogate,GaussianProposalSurrogate)
                self.assertIsInstance(runtime.selectors['gp_child_ucb'],GaussianProposalSelector)

    def test_cost_mode_is_rejected_instead_of_silently_changing_acquisition(self):
        root=SOURCE.parents[2]
        with tempfile.TemporaryDirectory() as path:
            cfg=OmegaConf.merge(OmegaConf.load(root/'config/config.yaml'),OmegaConf.load(root/'config/experiments/gp6.yaml'))
            cfg.log_dir=path;cfg.bo_parent_selector.acquisition.use_cost=True
            with self.assertRaises(ValueError):Runtime(cfg)

    def test_root_option_and_snapshot_have_exactly_six_gp_coordinates(self):
        root=SOURCE.parents[2]
        with tempfile.TemporaryDirectory() as path:
            cfg=OmegaConf.merge(OmegaConf.load(root/'config/config.yaml'),OmegaConf.load(root/'config/experiments/gp6_draft.yaml'))
            cfg.log_dir=path;r=Runtime(cfg)
            virtual=NS(id='root',stage='root',parent=None,metric=NS(value=None),children=[],visits=0)
            journal=NS(nodes=[virtual])
            agent=NS(virtual_root=virtual,journal=journal,metric_maximize=True,scfg=cfg.agent.search,
                acfg=cfg.agent,search_start_time=None,best_node=None,branch_successful_nodes={})
            agent.is_root=lambda n:n.id=='root'
            option=r._make_option(agent,virtual,OperatorPreview('draft'))
            decision=SelectionDecision('gp_child_ucb','root','draft',0.,None,False,'test',utc_timestamp())
            result=SelectorResult(decision,option,(),(option,))
            r.root_draft_state={'draft_submitted':0,'draft_cap':5,'pending':{
                'test':dict(kind='initialization',movement='new_direction',root_legality='initialization_pending',decision_index=0)}}
            r._log_selection(agent,result)
            snapshot=json.loads((Path(path)/'phase1/selection_snapshots.jsonl').read_text())
            self.assertEqual(len(snapshot['gp_feature_vector']),6)
            self.assertIsNone(snapshot['parent_utility'])
            self.assertEqual(snapshot['uncertainty_used'],'latent')


class TestPresetLoading(unittest.TestCase):
    def loader(self, args):
        from unittest.mock import patch
        load_function = next(n for n in config_tree.body
                             if isinstance(n, ast.FunctionDef) and n.name == '_load_cfg')
        ns = dict(Path=Path, Config=object, OmegaConf=OmegaConf, __file__=str(config_source))
        ns.update({name: study_config[name] for name in ('SEVEN_PRESETS', 'Study7Config')})
        ns['BRANCH7_PRESETS']=branch_config['BRANCH7_PRESETS']
        ns['ACQ7_PRESETS']=acq_config['ACQ7_PRESETS']
        ns['ACQ7_EXTENSION_PRESETS']=acq_config['ACQ7_EXTENSION_PRESETS']
        ns['RATING_ONLY_PRESETS']=rating_config['RATING_ONLY_PRESETS']
        exec(compile(ast.Module(body=[load_function], type_ignores=[]), str(config_source), 'exec'), ns)
        with patch.object(OmegaConf, 'from_cli', return_value=OmegaConf.from_dotlist(args)):
            return ns['_load_cfg']()

    def test_preset_loader_and_explicit_overrides(self):
        for mode, enabled in [('gp6', False), ('gp6_draft', True)]:
            cfg = self.loader([f'experiment_preset={mode}', 'bo_parent_selector.acquisition.beta=0.0'])
            self.assertEqual(cfg.parent_selector.type, 'gp_child_ucb')
            self.assertEqual(cfg.parent_selector.include_root_draft, enabled)
            self.assertEqual(cfg.bo_parent_selector.acquisition.beta, 0.0)

    def test_resume_cannot_silently_change_requested_experiment_arm(self):
        with tempfile.TemporaryDirectory() as path:
            logs = Path(path) / 'logs'; logs.mkdir()
            saved = self.loader(['experiment_preset=gp6'])
            OmegaConf.save(saved, logs / 'config.yaml')
            resumed = self.loader(['experiment_preset=gp6', f'resume_run_dir={path}'])
            self.assertFalse(resumed.parent_selector.include_root_draft)
            with self.assertRaisesRegex(ValueError, 'Cannot resume'):
                self.loader(['experiment_preset=gp6_draft', f'resume_run_dir={path}'])


if __name__ == '__main__':
    unittest.main()
