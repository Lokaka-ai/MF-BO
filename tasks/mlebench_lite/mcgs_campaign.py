#!/usr/bin/env python3
"""Owner's original-MCGS campaign, with native 1+2+4 allocation handoffs.

Reuse the audited allocation lifecycle, not AQ05's selector or run state.
"""
from pathlib import Path
import shlex
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from tasks.mlebench_lite import aq05_campaign as campaign

campaign.PRESET = 'mlevolve_original'
campaign.METHOD_LABEL = 'mcgs'
COMMON_KEYS = (
    'task_type', 'agent.time_limit', 'agent.steps', 'agent.initial_drafts', 'agent.seed',
    'agent.search.parallel_search_num', 'agent.search.num_gpus', 'exec.timeout',
    'evaluation.enabled', 'agent.code.model', 'agent.code.base_url',
    'agent.feedback.model', 'agent.feedback.base_url',
)
campaign.EXPECTED = {key: campaign.EXPECTED[key] for key in COMMON_KEYS} | {
    'parent_selector.type': 'legacy_original',
    'parent_selector.decision_scope': 'root_only',
    'parent_selector.include_root_draft': False,
    'bo_parent_selector.model': 'conjugate_bayesian_linear',
    'bo_parent_selector.agent_rating.enabled': False,
    'study7.rating.enabled': False, 'study7.ancestor.enabled': False,
    'acq7.enabled': False,
}


def entry_command(manifest, *args):
    code = Path(manifest['code'])
    script = (f'cd {shlex.quote(str(code))}\nsource tasks/mlebench_lite/env.sh\n'
              'exec python -u tasks/mlebench_lite/mcgs_campaign.py ' + shlex.join(list(args)))
    return ['bash', '-c', script]


campaign.entry_command = entry_command

if __name__ == '__main__':
    raise SystemExit(campaign.main())
