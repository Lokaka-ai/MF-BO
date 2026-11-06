"""Prepare one fresh trial with the existing immutable harness; no GPU/LLM calls.

Run using the project's locked Python environment. The resulting directory can
be passed to tasks/autoresearch/worker.py on an allocated GPU.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True, help='MLEvolve_BO source directory')
    parser.add_argument('--trial', type=Path, required=True, help='Fresh output directory; must not exist')
    parser.add_argument('--candidate', type=Path, help='Candidate module; default: extract the reference candidate')
    args = parser.parse_args()
    project = args.project.expanduser().resolve()
    trial = args.trial.expanduser().resolve()
    if trial.exists():
        parser.error('Use a fresh trial directory; existing trials are not overwritten.')
    if not (project / 'tasks/autoresearch/adapter.py').is_file():
        parser.error('--project must point to MLEvolve_BO or an equivalent frozen source tree.')
    sys.path.insert(0, str(project))
    from tasks.autoresearch.adapter import assemble_candidate, candidate_body
    repo = project / 'tasks/autoresearch/repo'
    code = args.candidate.read_text() if args.candidate else candidate_body((repo / 'train.py').read_text())
    assembled = assemble_candidate(code)
    trial.mkdir(parents=True, exist_ok=False)
    (trial / 'candidate.py').write_text(code)
    (trial / 'train.py').write_text(assembled)
    for name in ['prepare.py', 'trial_config.py']:
        shutil.copyfile(repo / name, trial / name)
    manifest = {
        'prepared_only': True,
        'source_project': str(project),
        'candidate_source': str(args.candidate.resolve()) if args.candidate else 'reference candidate',
        'files': {name: hashlib.sha256((trial / name).read_bytes()).hexdigest()
                  for name in ['candidate.py', 'train.py', 'prepare.py', 'trial_config.py']},
        'source_files': {str(p.relative_to(project)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in [repo / 'train.py', repo / 'prepare.py', repo / 'trial_config.py',
                                   project / 'tasks/autoresearch/adapter.py', project / 'tasks/autoresearch/worker.py']},
        'no_training_or_LLM_calls': True,
    }
    (trial / 'handoff_preparation.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'trial': str(trial), 'prepared_only': True, 'candidate_sha256': manifest['files']['candidate.py']}, indent=2))


if __name__ == '__main__':
    main()
