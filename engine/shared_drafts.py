"""Replay native draft generation only; keep execution and search method-specific."""
from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path
import threading
import time

from agents.triggers import register_node
from engine.fixed_evaluation import EvaluationError
from engine.search_node import SearchNode

logger = logging.getLogger("MLEvolve")


def sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class SharedDraftPool:
    def __init__(self, manifest_path, agent):
        self.path = Path(manifest_path).resolve()
        raw = self.path.read_text()
        self.pool_id = sha256(raw)
        self.manifest = json.loads(raw)
        self.entries = self.manifest["drafts"]
        if self.manifest.get("version") != 1 or self.manifest.get("state") != "ready":
            raise ValueError("Shared draft pool must be complete and sealed")
        if len(self.entries) != int(agent.scfg.num_drafts):
            raise ValueError("Shared pool size must equal search.num_drafts")
        if self.manifest.get("model") != agent.acfg.code.model:
            raise ValueError("Shared draft generator model differs from the experiment")
        if self.manifest.get("protocol_id") != agent.autoresearch.protocol_id:
            raise ValueError("Shared drafts have a different immutable task protocol")
        self.lock = threading.Lock()
        self.state_path = Path(agent.cfg.log_dir) / "shared_draft_usage.json"
        self.state = {"pool_id": self.pool_id, "manifest": str(self.path), "assignments": []}
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text())
            if self.state["pool_id"] != self.pool_id:
                raise ValueError("Cannot resume with a changed shared draft pool")
        self.verify()

    def verify(self):
        for slot, entry in enumerate(self.entries, 1):
            if entry["slot"] != slot or not entry.get("reviewed"):
                raise ValueError("Shared drafts must be ordered and reviewed")
            path = (self.path.parent / entry["artifact"]).resolve()
            if not path.is_relative_to(self.path.parent):
                raise ValueError("Shared draft artifact escaped the pool directory")
            artifact = json.loads(path.read_text())
            if sha256(path.read_text()) != entry["artifact_sha256"] or sha256(artifact["code"]) != entry["code_sha256"]:
                raise ValueError(f"Shared draft {slot} changed after sealing")
            seconds = entry["generation_seconds"]
            if not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or not math.isfinite(seconds) or seconds < 0:
                raise ValueError("Invalid shared generation time")

    def take(self, agent, decision_id):
        # Reserve/register atomically, but never hold the search lock while charging time.
        with agent.journal_lock, self.lock:
            assignments = self.state["assignments"]
            if any(a["decision_id"] == decision_id for a in assignments):
                raise EvaluationError("Shared draft decision was already consumed; resume its saved node")
            slot = len(assignments) + 1
            if slot > len(self.entries):
                raise EvaluationError("Shared draft pool exhausted; unpaired LLM fallback is forbidden")
            entry = self.entries[slot - 1]
            path = self.path.parent / entry["artifact"]
            raw = path.read_text()
            artifact = json.loads(raw)
            if sha256(raw) != entry["artifact_sha256"] or sha256(artifact["code"]) != entry["code_sha256"]:
                raise EvaluationError("Shared draft was modified after experiment initialization")
            agent.virtual_root.add_expected_child_count()
            node = SearchNode(code=artifact["code"], plan=artifact["plan"], parent=agent.virtual_root,
                              stage="draft", local_best_node=agent.virtual_root,
                              shared_draft_pool_id=self.pool_id, shared_draft_slot=slot,
                              shared_draft_generation_seconds=entry["generation_seconds"])
            register_node(agent, node, artifact["prompt_input"], new_branch=True)
            assignments.append({"slot": slot, "node_id": node.id, "decision_id": decision_id,
                                "code_sha256": entry["code_sha256"], "reserved_at": time.time(),
                                "generation_seconds": entry["generation_seconds"]})
            atomic_json(self.state_path, self.state)
        logger.info("[shared-draft] Taking slot %d/%d (sha256=%s); charging %.1fs of native generation/review time",
                    slot, len(self.entries), entry["code_sha256"], entry["generation_seconds"])
        time.sleep(entry["generation_seconds"])
        with self.lock:
            assignments[slot - 1]["ready_at"] = time.time()
            atomic_json(self.state_path, self.state)
        return node
