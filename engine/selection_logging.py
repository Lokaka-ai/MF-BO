"""Append-only Phase 1 decision, candidate-snapshot and model-fit logging."""

from __future__ import annotations

import json
import math
from pathlib import Path
import threading
from typing import Any


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return value


class SelectionLogger:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _append(self, filename: str, payload: dict) -> None:
        encoded = json.dumps(
            _json_safe(payload),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        with self._lock:
            with (self.root / filename).open("a", encoding="utf-8") as handle:
                handle.write(encoded + "\n")
                handle.flush()

    def append_decision(self, payload: dict) -> None:
        self._append("selection_decisions.jsonl", payload)

    def append_snapshot(self, payload: dict) -> None:
        self._append("selection_snapshots.jsonl", payload)

    def append_model_fit(self, payload: dict) -> None:
        self._append("model_fits.jsonl", payload)

    def append_fallback(self, payload: dict) -> None:
        self._append("selector_fallbacks.jsonl", payload)

    def append_continuation(self, payload: dict) -> None:
        self._append("continuation_events.jsonl", payload)
