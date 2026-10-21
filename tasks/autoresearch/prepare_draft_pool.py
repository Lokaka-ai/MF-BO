"""Generate five native, reviewed drafts without evaluating or selecting by score."""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    from omegaconf import OmegaConf
    from config import _load_cfg, prep_cfg, prep_agent_workspace, load_task_desc
    from agents import draft_agent, code_review_agent
    from engine.agent_search import AgentSearch
    from engine.coldstart import build_guidance_description
    from engine.search_node import Journal, SearchNode
    from engine.shared_drafts import atomic_json, sha256
    from utils.logging_config import setup_logging
    from utils.seed import set_global_seed

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    if (output / "manifest.json").exists():
        raise FileExistsError("Pool is already sealed")
    cfg = OmegaConf.merge(_load_cfg(use_cli_args=False),
        OmegaConf.load(ROOT / "tasks/autoresearch/config.yaml"),
        OmegaConf.load(ROOT / "config/experiments/mlevolve_original.yaml"))
    cfg.data_dir = str(ROOT / "tasks/autoresearch/repo")
    cfg.desc_file = str(ROOT / "tasks/autoresearch/TASK.md")
    cfg.log_dir = str(output / "generator")
    cfg.workspace_dir = cfg.log_dir
    cfg.exp_name = "shared-drafts5-generator"
    cfg.cpu_number = 1
    # Preparation only: no CUDA trials or memory retrieval are performed here.
    cfg.agent.memory_embedding_device = "cpu"
    cfg.agent.time_limit = 28800
    cfg.shared_draft_pool = None
    cfg = prep_cfg(cfg)
    setup_logging(cfg)
    prep_agent_workspace(cfg)
    set_global_seed(cfg.agent.seed)
    cfg.coldstart.description = build_guidance_description(cfg)
    agent = AgentSearch(load_task_desc(cfg), cfg, Journal())
    agent.update_data_preview()
    entries = []
    state = {"state": "generating", "started_at": time.time(), "pid": os.getpid(),
             "generator_log_dir": str(cfg.log_dir), "generator_workspace_dir": str(cfg.workspace_dir)}
    atomic_json(output / "status.json", state)
    for slot in range(1, int(cfg.agent.search.num_drafts) + 1):
        target = output / f"draft_{slot}.json"
        if target.exists():
            artifact = json.loads(target.read_text())
            node = SearchNode(code=artifact["code"], plan=artifact["plan"],
                              parent=agent.virtual_root, stage="draft", local_best_node=agent.virtual_root)
            from agents.triggers import register_node
            agent.virtual_root.add_expected_child_count()
            register_node(agent, node, artifact["prompt_input"], new_branch=True)
        else:
            started = time.monotonic()
            node = draft_agent.run(agent)
            node.code = code_review_agent.run(agent, node)
            # Keep invalid programs too: native workflow debug must handle them.
            artifact = {"slot": slot, "plan": node.plan, "code": node.code,
                        "prompt_input": node.prompt_input, "reviewed": True,
                        "generation_seconds": time.monotonic() - started}
            atomic_json(target, artifact)
        entries.append({"slot": slot, "artifact": target.name, "artifact_sha256": sha256(target.read_text()),
                        "code_sha256": sha256(artifact["code"]), "reviewed": True,
                        "generation_seconds": artifact["generation_seconds"]})
        atomic_json(output / "status.json", dict(state, completed=slot))
        logging.getLogger("MLEvolve").info("Shared draft %d/5 saved, sha256=%s", slot, entries[-1]["code_sha256"])
    manifest = {"version": 1, "state": "ready", "model": cfg.agent.code.model,
                "provider": cfg.agent.code.base_url, "protocol_id": agent.autoresearch.protocol_id,
                "generation": "native stepwise specialists + merger + native code review",
                "cost_policy": "Replay measured generation/review latency on each consumed draft",
                "evaluation_policy": "Each workflow executes and reviews results independently on its A100",
                "initial_drafts": int(cfg.agent.initial_drafts), "drafts": entries}
    atomic_json(output / "manifest.json", manifest)
    atomic_json(output / "status.json", dict(state, state="ready", completed=len(entries),
        finished_at=time.time(), pool_id=sha256((output / "manifest.json").read_text())))
    OmegaConf.save(cfg, output / "generator_config.yaml")


if __name__ == "__main__":
    main()
