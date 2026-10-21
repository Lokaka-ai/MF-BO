#!/usr/bin/env python3
"""Small selector-only smoke; --live-agent explicitly makes one provider request.

No dataset, GPU, candidate code execution, or benchmark is run. Default mode
uses a fake model response and exercises the production numeric selectors.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from omegaconf import OmegaConf
from engine.parent_selectors.agent_selector import AgentParentSelector
from engine.parent_selectors.base import ExpansionOption, OperatorPreview
from engine.parent_selectors.gp_variants import GPVariantSurrogate, GPVariantSelector
from engine.parent_selectors.transition_store import TransitionRecord
from engine.selection_logging import SelectionLogger


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live-agent', action='store_true', help='Make one real Codex/provider request (uses your API quota)')
    parser.add_argument('--model', help='Explicit selector model override')
    parser.add_argument('--provider', help='Provider name from your existing Codex config, e.g. opencode-go')
    parser.add_argument('--output', type=Path, help='Optional directory for smoke prompts/results')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='mlevolve-selector-smoke-') as tmp:
        output = args.output or Path(tmp)
        output.mkdir(parents=True, exist_ok=True)
        nodes, options = [], []
        for i, quality in enumerate((.70, .72)):
            node = NS(id=f'example-node-{i}', stage='improve', parent=None,
                      metric=NS(value=quality), plan='Tune a small tabular classifier.',
                      analysis=f'Observed validation accuracy {quality}.',
                      code='from sklearn.ensemble import RandomForestClassifier\nmodel = RandomForestClassifier(n_estimators=100)')
            nodes.append(node)
            features = dict(parent_utility=quality, incumbent_at_decision=.72,
                gp_parent_gap=.72-quality, gp_last_gain=.01, gp_recent_gain=.01,
                gp_stagnation=0., gp_trials=float(i+1), gp_valid_rate=.75,
                elapsed_fraction=.25, remaining_steps_fraction=.8)
            options.append(ExpansionOption(node.id,'improve',(),features,node,OperatorPreview('improve')))
        agent = NS(journal=NS(nodes=nodes), metric_maximize=True,
                   task_desc='Synthetic smoke only: maximize validation accuracy on a tabular classification task.')
        rows = [TransitionRecord('seed','seed','root','seed-child','draft',(),None,.70,None,.70,None,
                   True,1.,1.,2.,None,None,'smoke',dict(parent_utility=None,incumbent_at_decision=None,gp_parent_gap=0.,gp_last_gain=0.,gp_recent_gain=0.,gp_stagnation=0.,gp_trials=0.,gp_valid_rate=.5),'seed',{'proposal_closed':True})]
        results = []
        for mode in ('halfshrink_ucb', 'gp6_global_logei', 'agent_selector'):
            cfg = OmegaConf.merge(OmegaConf.load(ROOT/'config/config.yaml'),
                OmegaConf.load(ROOT/f'config/experiments/{mode}.yaml'))
            assert cfg.parent_selector.include_root_draft is False
            if mode == 'agent_selector':
                if args.model: cfg.agent_parent_selector.model = args.model
                if args.provider: cfg.agent_parent_selector.base_url = f'codex://{args.provider}'
                cfg.agent_parent_selector.max_attempts = 1
                response = None if args.live_agent else lambda **kw: {'candidate':'c1','reason':'Fake smoke response; no provider request.'}
                selector = AgentParentSelector(cfg=cfg, records_provider=lambda:rows,
                    selection_logger=SelectionLogger(output/'agent'), request_fn=response)
            else:
                gp = dict(cfg.bo_parent_selector.gp)
                variant = dict(gp.pop('variant'))
                variant['beta'] = float(cfg.bo_parent_selector.acquisition.beta)
                from engine.parent_selectors.gp_surrogate import GP6Config
                surrogate = GPVariantSurrogate(variant_config=variant,config=GP6Config(**gp))
                selector = GPVariantSelector(surrogate=surrogate, records_provider=lambda:rows)
            result = selector.select(agent,options)
            if result.decision.fallback_used:
                raise RuntimeError(f'{mode} unexpectedly fell back: {result.decision.fallback_reason}')
            assert result.selected_option in options and result.selected_option.operator != 'draft'
            results.append(dict(mode=mode,selected_parent=result.selected_option.parent_node_id,
                selector=result.decision.selector_name,root_draft=False,fallback=False,
                provider_call=(mode=='agent_selector' and args.live_agent)))
        report = {'kind':'selector_only_synthetic_smoke', 'full_agent_or_gpu_test':False,'results':results}
        if args.output: (output/'smoke_result.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
