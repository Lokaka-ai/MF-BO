"""Append-only transition records used by the Phase 1 surrogates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import threading
from typing import Any


logger = logging.getLogger("MLEvolve")


@dataclass(frozen=True)
class TransitionRecord:
    transition_id: str
    decision_id: str
    parent_node_id: str
    child_node_id: str | None
    operator: str
    reference_node_ids: tuple[str, ...]
    parent_metric_raw: float | None
    child_metric_raw: float | None
    parent_utility: float | None
    child_utility: float | None
    improvement: float | None
    valid: bool
    generation_wall_time_sec: float
    execution_wall_time_sec: float
    total_wall_time_sec: float
    gpu_time_sec: float | None
    reward: float | None
    selector_name: str
    selection_features: dict[str, Any]
    timestamp: str
    metadata: dict | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        if self.metadata is None:
            data.pop("metadata")
        data["reference_node_ids"] = list(self.reference_node_ids)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "TransitionRecord":
        payload = dict(data)
        payload["reference_node_ids"] = tuple(payload.get("reference_node_ids", ()))
        return cls(**payload)


class TransitionStore:
    """Thread-safe JSONL store; restart recovery is a replay of this file."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._records = self._load()

    def _load(self) -> list[TransitionRecord]:
        if not self.path.is_file():
            return []
        records: list[TransitionRecord] = []
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    records.append(TransitionRecord.from_dict(json.loads(line)))
                except Exception as exc:
                    logger.warning(
                        "Skipping malformed transition line %s in %s: %s",
                        line_number,
                        self.path,
                        exc,
                    )
        return records

    def append(self, record: TransitionRecord) -> None:
        encoded = json.dumps(record.to_dict(), ensure_ascii=False, allow_nan=False)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(encoded + "\n")
                handle.flush()
            self._records.append(record)

    def records(self) -> list[TransitionRecord]:
        with self._lock:
            return list(self._records)

    def admit_child_ids(self, child_node_ids: set[str]) -> int:
        """Ignore records newer than the admitted journal snapshot."""
        with self._lock:
            admitted = [
                record
                for record in self._records
                if record.child_node_id in child_node_ids
            ]
            dropped = len(self._records) - len(admitted)
            self._records = admitted
            return dropped

    def valid_gain_records(self) -> list[TransitionRecord]:
        return [
            record
            for record in self.records()
            if record.valid and record.improvement is not None
        ]

    def by_child_id(self, child_node_id: str) -> TransitionRecord | None:
        with self._lock:
            for record in reversed(self._records):
                if record.child_node_id == child_node_id:
                    return record
        return None

    def by_decision_id(self, decision_id: str) -> TransitionRecord | None:
        with self._lock:
            return next((record for record in self._records
                         if record.decision_id == decision_id), None)
