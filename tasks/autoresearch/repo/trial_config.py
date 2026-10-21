"""Typed access to immutable EvolveFold trial configuration snapshots."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping


class TrialConfig:
    def __init__(self, values: Mapping[str, Any], environ: Mapping[str, str]) -> None:
        self._values = dict(values)
        self._environ = dict(environ)
        self._consumed: set[str] = set()

    @classmethod
    def from_environment(
        cls, environ: Mapping[str, str] | None = None
    ) -> "TrialConfig":
        resolved_environ = dict(os.environ if environ is None else environ)
        config_path = resolved_environ.get("EVOLVEFOLD_TRIAL_CONFIG", "").strip()
        if not config_path:
            return cls({}, resolved_environ)
        try:
            payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read EvolveFold trial config {config_path}: {exc}") from exc
        if not isinstance(payload, dict) or not all(
            isinstance(key, str) and key for key in payload
        ):
            raise ValueError("EvolveFold trial config must be a JSON object with string keys")
        return cls(payload, resolved_environ)

    def value(
        self,
        name: str,
        default: Any,
        cast: Callable[[Any], Any] | None = None,
    ) -> Any:
        if name in self._values:
            raw = self._values[name]
            self._consumed.add(name)
        elif name in self._environ:
            raw = self._environ[name]
        else:
            return default
        converter = cast or type(default)
        if converter is bool:
            if isinstance(raw, bool):
                return raw
            if isinstance(raw, str) and raw.lower() in {"true", "false"}:
                return raw.lower() == "true"
            raise ValueError(f"Trial config {name} must be a boolean")
        try:
            return converter(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Trial config {name} has invalid value {raw!r}") from exc

    def reject_unused(self) -> None:
        unused = sorted(set(self._values) - self._consumed)
        if unused:
            raise ValueError(
                "Trial config contains keys not consumed by this frozen train.py: "
                + ", ".join(unused)
            )
