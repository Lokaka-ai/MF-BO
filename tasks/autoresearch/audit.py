"""Fail closed if a campaign silently changes the MLEbench-lite agent settings."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
ALLOWED_DIFFERENCES = {
    "task_type": "Task adapter",
    "exp_id": "Task identifier",
    "exp_name": "Task identifier",
    "preprocess_data": "Fixed, pretokenized language-model data",
    "use_grading_server": "Local official BPB evaluator",
    "evaluation.enabled": "Replace Kaggle V/test scorer with immutable BPB scorer",
    "cpu_number": "Actual per-setting cluster CPU allocation",
    "agent.time_limit": "Requested eight-hour campaign",
    "agent.code.model": "Requested Muse 1.3",
    "agent.feedback.model": "Requested Muse 1.3",
    "agent.code.base_url": "Requested OpenCode Go transport",
    "agent.feedback.base_url": "Requested OpenCode Go transport",
    "agent.search.parallel_search_num": "User-configurable nanoGPT concurrency; two workers by default",
    "shared_draft_pool": "User-requested paired comparison with five shared native drafts",
}


def flatten(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(flatten(item, name))
        else:
            result[name] = item
    return result


def audit_campaign(campaign, setting):
    base = OmegaConf.load(ROOT / "config/config.yaml")
    preset = OmegaConf.load(ROOT / f"config/experiments/{setting}.yaml")
    benchmark = OmegaConf.merge(base, preset)
    actual = OmegaConf.merge(base, OmegaConf.load(ROOT / "tasks/autoresearch/config.yaml"), preset)
    actual.agent.time_limit = int(campaign.duration_seconds)
    actual.agent.steps = int(campaign.max_trials)
    actual.cpu_number = int(campaign.cpus)
    for key, value in campaign.overrides.items():
        OmegaConf.update(actual, key, value)
    expected_flat = flatten(OmegaConf.to_container(benchmark, resolve=True))
    actual_flat = flatten(OmegaConf.to_container(actual, resolve=True))
    differences = []
    for key in sorted(expected_flat.keys() | actual_flat.keys()):
        before, after = expected_flat.get(key), actual_flat.get(key)
        if before != after:
            if key not in ALLOWED_DIFFERENCES:
                raise ValueError(f"Unexpected MLEbench-lite configuration change: {key}: {before!r} -> {after!r}")
            differences.append(dict(key=key, benchmark=before, autoresearch=after,
                                    reason=ALLOWED_DIFFERENCES[key]))
    files = [ROOT / "config/config.yaml", ROOT / "tasks/autoresearch/config.yaml",
             ROOT / f"config/experiments/{setting}.yaml", ROOT / "tasks/autoresearch/TASK.md"]
    files += sorted((ROOT / "agents").rglob("*.py"))
    files += sorted((ROOT / "engine").rglob("*.py"))
    files += sorted((ROOT / "llm").rglob("*.py"))
    files += sorted((ROOT / "tasks/autoresearch").rglob("*.py"))
    return {"setting": setting, "passed": True, "differences": differences,
            "benchmark_config": OmegaConf.to_container(benchmark, resolve=True),
            "autoresearch_config": OmegaConf.to_container(actual, resolve=True),
            "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}


if __name__ == "__main__":
    campaign = OmegaConf.load(Path(__file__).with_name("three_settings_8h.yaml"))
    for setting in campaign.settings:
        result = audit_campaign(campaign, setting)
        print(json.dumps({k: result[k] for k in ("setting", "passed", "differences")}, indent=2))
