#!/usr/bin/env python3
"""Codex transport using OpenCode Go, without modifying the user's Codex home."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
import tomllib
import uuid

ROOT = Path(__file__).resolve().parents[1]


def toml_value(value):
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{json.dumps(k)} = {toml_value(v)}" for k, v in value.items()) + " }"
    return json.dumps(value)


def main():
    env = os.environ.copy()
    if not env.get("OPENCODE_API_KEY"):
        key_file = Path(env.get("OPENCODE_API_KEY_FILE", ROOT.parent / "codex" / "opecode"))
        try:
            key = key_file.read_text().strip()
        except OSError:
            sys.exit("Set OPENCODE_API_KEY or OPENCODE_API_KEY_FILE to your OpenCode Go credential.")
        if not key or any(c.isspace() for c in key):
            sys.exit("The OpenCode credential file must contain a single raw API key.")
        env["OPENCODE_API_KEY"] = key
    env.setdefault("OPENCODE_SESSION_ID", "ses_" + uuid.uuid4().hex)
    codex = env.get("MLEVOLVE_REAL_CODEX_BIN") or shutil.which("codex")
    if not codex or Path(codex).resolve() == Path(__file__).resolve():
        sys.exit("Codex CLI is missing; set MLEVOLVE_REAL_CODEX_BIN to its executable.")
    config = tomllib.loads((ROOT / "config" / "opencode-go.toml").read_text())
    args = sys.argv[1:]
    # exec has its own flattened config arguments in recent Codex releases.
    command = [codex]
    if args and args[0] in ("exec", "e", "review"):
        command.append(args.pop(0))
    for name, value in config.items():
        if name in ("model_providers", "features"):
            for child, settings in value.items():
                command.extend(["--config", f"{name}.{child}={toml_value(settings)}"])
        else:
            command.extend(["--config", f"{name}={toml_value(value)}"])
    os.execvpe(codex, command + args, env)


if __name__ == "__main__":
    main()
