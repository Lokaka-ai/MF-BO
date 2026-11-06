#!/usr/bin/env python3
"""Synthetic A/B selector smoke: no API calls, training jobs, datasets or GPUs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from omegaconf import OmegaConf
from engine.parent_selectors.base import ExpansionOption,OperatorPreview
from engine.parent_selectors.gp_graph_features import DecisionGraph
from engine.parent_selectors.gp_graph_surrogate import GraphProposalSurrogate,GraphProposalSelector
from engine.parent_selectors.gp_surrogate import GP6Config
from engine.parent_selectors.transition_store import TransitionRecord


def features(parent,incumbent,step):
    return dict(parent_utility=parent,incumbent_at_decision=incumbent,
        gp_parent_gap=max(incumbent-parent,0.) if parent is not None else 0.,
        gp_last_gain=.01,gp_recent_gain=.005,gp_stagnation=0.,gp_trials=float(step),gp_valid_rate=.9)


def row(key,parent,child,operator,before,after,state):
    return TransitionRecord(key,key,parent,child,operator,(),before,after,before,after,
        None if before is None else after-before,True,0.,1.,1.,None,None,'synthetic_smoke',state,key,
        {'proposal_closed':True})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,help='Optional JSON report path')
    args=parser.parse_args()
    results=[]
    for mode in ('gp7_ancestry_ucb','laplacian_anchor_logei'):
        cfg=OmegaConf.merge(OmegaConf.load(ROOT/'config/config.yaml'),
            OmegaConf.load(ROOT/f'config/experiments/{mode}.yaml'))
        gp=OmegaConf.to_container(cfg.bo_parent_selector.gp,resolve=True)
        graph_cfg=gp.pop('graph')
        model=GraphProposalSurrogate(mode=cfg.parent_selector.type,graph_config=graph_cfg,
            config=GP6Config(**gp),task_id='synthetic',
            acquisition='ucb' if cfg.bo_parent_selector.acquisition.type=='child_ucb' else 'logei')
        graph=DecisionGraph(.3 if model.is_anchor else 0.)
        records=[row('seed','root','s','draft',None,.7,features(None,.7,0))]
        for i in range(10):
            graph.update(records)
            parent='s' if i%2 else records[-1].child_node_id
            before=graph.utilities[parent]
            state=features(before,max(graph.utilities.values()),i)
            state.update(graph.features(parent,True))
            records.append(row(f't{i}',parent,f'n{i}','improve',before,before+.005*(i%3-1),state))
        graph.update(records)
        options=[]
        for parent in ('s','n9','n8'):
            state=features(graph.utilities[parent],max(graph.utilities.values()),10)
            state.update(graph.features(parent,True))
            options.append(ExpansionOption(parent,'improve',(),state,NS(id=parent),OperatorPreview('improve')))
        selector=GraphProposalSelector(surrogate=model,records_provider=lambda:records)
        result=selector.select(NS(),options)
        if result.decision.fallback_used:
            raise RuntimeError(result.decision.fallback_reason)
        assert model.x.shape==(10,7)
        assert all(np.isfinite(s.acquisition_score) for s in result.candidate_scores)
        results.append({'preset':mode,'selected_parent':result.selected_option.parent_node_id,
            'training_count':10,'numerical_features':7,'fallback_used':False,
            'optimizer_status':model.fit_diagnostics['optimizer_status'],
            'is_exact_gp_posterior':not model.is_anchor,
            'candidate_scores':[s.to_dict() for s in result.candidate_scores]})
    report={'kind':'synthetic_numeric_selector_smoke','api_calls':0,'gpu_jobs':0,
            'native_agent_integration':False,'results':results}
    encoded=json.dumps(report,indent=2,allow_nan=False)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(encoded)
    print(encoded)
    return 0


if __name__=='__main__':raise SystemExit(main())
