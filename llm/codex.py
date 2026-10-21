"""Codex CLI backend for MLEvolve's existing LLM interface.

This module only adapts model transport. Search, planning, memory, coding modes,
execution, and evaluation remain owned by MLEvolve.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from copy import deepcopy

from config import Config
from .gemini import FunctionSpec, compile_prompt_to_md

logger = logging.getLogger("MLEvolve")

OutputType = str | dict


def _usage_event(cfg, event):
    """Append cost evidence for every attempt, including interrupted/retried calls.

    This file is deliberately outside the selector checkpoint: rollback must not
    erase API costs. Never store prompts, transport output or credentials here.
    """
    directory = getattr(cfg, "log_dir", None)
    if directory is None:
        return
    path = Path(directory) / "llm_calls.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
    try:
        os.write(fd, (json.dumps(event, allow_nan=False) + "\n").encode())
    finally:
        os.close(fd)


def _strict_output_schema(schema: dict) -> dict | None:
    """Normalize ordinary object schemas for Codex strict output.

    A schema-valued ``additionalProperties`` represents a dynamic mapping and
    cannot be expressed by the provider's strict-output subset. Those schemas
    fall back to prompt-constrained JSON instead of changing their semantics.
    """
    normalized = deepcopy(schema)

    def visit(node):
        if isinstance(node, list):
            for value in node:
                if not visit(value):
                    return False
            return True
        if not isinstance(node, dict):
            return True

        if node.get("type") == "object" or "properties" in node:
            if isinstance(node.get("additionalProperties"), dict):
                return False
            node["additionalProperties"] = False
            properties = node.get("properties", {})
            required = set(node.get("required", []))
            for name, value in properties.items():
                if not visit(value):
                    return False
                if name not in required:
                    properties[name] = {"anyOf": [value, {"type": "null"}]}
            if properties:
                node["required"] = list(properties)

        for key, value in node.items():
            if key == "properties":
                continue
            if isinstance(value, (dict, list)) and not visit(value):
                return False
        return True

    return normalized if visit(normalized) else None


def _stage_config_for_model(cfg: Config, model: str):
    if cfg.agent.code.model == model:
        return cfg.agent.code
    return cfg.agent.feedback


def _provider(stage) -> str:
    base_url = str(stage.base_url or "")
    prefix = "codex://"
    if not base_url.startswith(prefix) or not base_url[len(prefix) :]:
        raise ValueError(
            "Codex backend requires base_url=codex://<model-provider>"
        )
    return base_url[len(prefix) :]


def _working_directory(cfg: Config) -> Path:
    workspace = Path(cfg.workspace_dir)
    working = workspace / "working"
    return working if working.is_dir() else workspace


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            stripped = "\n".join(lines[1:-1]).strip()
    return stripped


def _run_codex(
    *,
    prompt: str,
    cfg: Config,
    model: str,
    stage,
    json_schema: dict | None = None,
    timeout_seconds: float | None = None,
    reasoning_effort: str | None = None,
    prompt_only: bool = False,
) -> tuple[str, float]:
    default_bin = (
        str(Path(__file__).resolve().parents[1] / "tools" / "codex_opencode.py")
        if _provider(stage) == "opencode-go" else shutil.which("codex")
    )
    codex_bin = os.environ.get("MLEVOLVE_CODEX_BIN") or default_bin
    if not codex_bin:
        raise FileNotFoundError("codex executable not found")

    timeout_seconds = timeout_seconds if timeout_seconds is not None else int(os.environ.get("MLEVOLVE_CODEX_TIMEOUT_SECONDS", "1800"))
    reasoning_effort = reasoning_effort or os.environ.get("MLEVOLVE_CODEX_REASONING_EFFORT", "high")

    with tempfile.TemporaryDirectory(prefix="mlevolve-codex-") as temp_dir:
        temp_root = Path(temp_dir)
        output_path = temp_root / "last_message.txt"
        command = [
            codex_bin,
            "exec",
            "--disable",
            "apps",
            "--ephemeral",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--model",
            model,
            "--config",
            f'model_provider={json.dumps(_provider(stage))}',
            "--config",
            f'model_reasoning_effort={json.dumps(reasoning_effort)}',
            "--config",
            'approval_policy="never"',
            "--cd",
            str(temp_root if prompt_only else _working_directory(cfg)),
            "--output-last-message",
            str(output_path),
        ]
        # A Slurm node's system python3 may predate tomllib. Run the project's
        # Python wrapper with the same uv interpreter as the calling workflow.
        if Path(codex_bin).suffix == ".py":
            command.insert(0, sys.executable)
        if prompt_only:
            # No workspace browsing or external tools for a selection-only decision.
            # Keep the user's provider/auth configuration; do not rewrite CODEX_HOME.
            for setting in ('features.shell_tool=false', 'features.unified_exec=false',
                            'features.multi_agent=false', 'web_search="disabled"',
                            'mcp_servers={}', 'project_doc_max_bytes=0'):
                command.extend(["--config", setting])
        strict_schema = (
            _strict_output_schema(json_schema) if json_schema is not None else None
        )
        if json_schema is not None and strict_schema is None:
            prompt = (
                f"{prompt}\n\nReturn only valid JSON matching this schema exactly:\n"
                f"{json.dumps(json_schema, sort_keys=True)}"
            )
        if strict_schema is not None:
            schema_path = temp_root / "output.schema.json"
            schema_path.write_text(
                json.dumps(strict_schema, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            command.extend(["--output-schema", str(schema_path)])
        command.append("-")

        started = time.time()
        call_id = uuid.uuid4().hex
        common = {"call_id": call_id, "model": model, "provider": _provider(stage),
                  "reasoning_effort": reasoning_effort, "started_epoch": started}
        _usage_event(cfg, {**common, "event": "started"})
        try:
            completed = subprocess.run(
                command, input=prompt, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, timeout=timeout_seconds, check=False,
            )
        except BaseException as exc:
            _usage_event(cfg, {**common, "event": "interrupted", "ended_epoch": time.time(),
                "error_type": type(exc).__name__, "reported_total_tokens": None})
            raise
        elapsed = time.time() - started
        usage = re.findall(r"tokens used\s+([\d,]+)", completed.stdout or "")
        _usage_event(cfg, {**common, "event": "finished", "ended_epoch": time.time(),
            "elapsed_seconds": elapsed, "exit_code": completed.returncode,
            "reported_total_tokens": int(usage[-1].replace(",", "")) if usage else None,
            "token_source": "codex_cli_footer", "input_output_breakdown_available": False})
        if completed.returncode != 0:
            tail = completed.stdout[-4000:]
            # Backoff on 429 to avoid hammering opencode-go
            if not prompt_only and ("429" in tail or "Too Many Requests" in tail or "exceeded retry limit" in tail):
                logger.warning("Codex 429 detected, sleeping 45s before surfacing error")
                time.sleep(45)
            raise RuntimeError(
                f"codex exec failed with exit code {completed.returncode}: {tail}"
            )
        if not output_path.is_file():
            raise RuntimeError("codex exec completed without a final-message file")
        return output_path.read_text(encoding="utf-8").strip(), elapsed


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    cfg: Config | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    """Run one MLEvolve query through Codex and its configured model provider."""
    if cfg is None:
        raise ValueError("cfg is required for Codex backend")
    model = str(model_kwargs.get("model") or cfg.agent.code.model)
    stage = _stage_config_for_model(cfg, model)

    sections = [
        "Act only as the language-model backend for the supplied workflow. "
        "Return the requested answer; do not edit files or run shell commands."
    ]
    if system_message:
        sections.append(f"# System instructions\n\n{system_message}")
    if user_message:
        sections.append(f"# User request\n\n{user_message}")
    schema = None
    if func_spec is not None:
        schema = func_spec.json_schema
        sections.append(
            f"Return the argument object for `{func_spec.name}`. "
            f"Function purpose: {func_spec.description}"
        )

    text, elapsed = _run_codex(
        prompt="\n\n".join(sections),
        cfg=cfg,
        model=model,
        stage=stage,
        json_schema=schema,
    )
    output: OutputType
    if schema is None:
        output = text
    else:
        output = json.loads(_strip_json_fence(text))
    return output, elapsed, 0, 0, {"model": model, "created": int(time.time())}


def _compile_generation_prompt(prompt: str | dict | list) -> str:
    if isinstance(prompt, dict) and any(
        key in prompt for key in ("system", "user", "assistant")
    ):
        sections = []
        if prompt.get("system"):
            sections.append(f"# System instructions\n\n{prompt['system']}")
        if prompt.get("user"):
            sections.append(f"# User request\n\n{prompt['user']}")
        if prompt.get("assistant"):
            sections.append(
                "# Response prefix or context\n\n"
                f"{prompt['assistant']}"
            )
        return "\n\n".join(sections)
    return prompt if isinstance(prompt, str) else compile_prompt_to_md(prompt)


def generate(
    prompt: str | dict | list,
    cfg: Config,
    temperature: float | None = None,
    max_tokens: int | None = None,
    stop_tokens: list[str] | None = None,
    json_schema: dict | None = None,
    max_retries: int = 20,
    retry_delay: float = 3,
) -> str:
    """Run MLEvolve text generation through Codex.

    Sampling-only arguments are accepted for interface compatibility; Codex and
    its configured provider own their supported decoding controls.
    """
    del temperature, max_tokens, stop_tokens
    stage = cfg.agent.code
    compiled = _compile_generation_prompt(prompt)
    request = (
        "Act only as the language-model backend for the supplied workflow. "
        "Return the requested content; do not edit files or run shell commands.\n\n"
        f"{compiled}"
    )
    attempts = max(1, min(int(max_retries), 3))
    for attempt in range(attempts):
        try:
            text, _ = _run_codex(prompt=request, cfg=cfg, model=stage.model,
                                  stage=stage, json_schema=json_schema)
            return text
        except (RuntimeError, subprocess.TimeoutExpired):
            if attempt + 1 == attempts:
                raise
            time.sleep(retry_delay)
