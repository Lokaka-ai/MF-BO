"""Native OpenCode CLI transport (Zen free models must run inside OpenCode).

Kept identical in MLEvolve_BO and MLEvolve_BO_agent so either tree can run alone.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile
import time
import uuid

DEFAULT_MODEL = "opencode/muse-spark-1.3-contributor-free"
DEFAULT_BASE_URL = "opencode://opencode"


class OpenCodeError(RuntimeError):
    def __init__(self, message, *, stdout="", stderr="", elapsed=0.0):
        super().__init__(message)
        self.stdout, self.stderr, self.elapsed = stdout, stderr, elapsed


def provider_base_url(provider: str) -> str:
    """Accept explicit transport URLs; retain legacy bare Codex providers."""
    if provider.startswith(("codex://", "opencode://")):
        return provider
    return DEFAULT_BASE_URL if provider == "opencode" else "codex://" + provider


def model_id(model: str, base_url: str = DEFAULT_BASE_URL) -> str:
    prefix = "opencode://"
    if not base_url.startswith(prefix) or not base_url[len(prefix):]:
        raise ValueError("OpenCode requires base_url=opencode://<provider>")
    provider = base_url[len(prefix):]
    if not model or ("/" in model and model.split("/", 1)[0] != provider):
        raise ValueError("OpenCode model and base_url providers must agree")
    return model if "/" in model else provider + "/" + model


def executable(value=None) -> str:
    binary = value or os.environ.get("MLEVOLVE_OPENCODE_BIN") or shutil.which("opencode")
    if not binary:
        candidate = Path.home() / ".opencode/bin/opencode"
        if candidate.is_file():
            binary = str(candidate)
    if not binary:
        raise FileNotFoundError("opencode not found; install OpenCode or set MLEVOLVE_OPENCODE_BIN")
    return str(binary)


def setting(name: str, default: str) -> str:
    # Existing campaign overrides retain their reasoning and timeout semantics.
    return os.environ.get("MLEVOLVE_OPENCODE_" + name,
                          os.environ.get("MLEVOLVE_CODEX_" + name, default))


def edit_permissions(cwd: Path, writable_dirs=()) -> dict:
    # OpenCode edit/write permissions match paths relative to its worktree,
    # not the absolute filePath argument. Outside git, its worktree is '/'.
    worktree = Path("/")
    if shutil.which("git"):
        probe = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=cwd,
                               text=True, capture_output=True, timeout=10, check=False)
        if probe.returncode == 0:
            worktree = Path(probe.stdout.strip()).resolve()
    rules = {"*": "deny"}
    for path in (cwd, *writable_dirs):
        path = Path(path).resolve()
        relative = os.path.relpath(path, worktree)
        if relative == ".":
            rules.update({"*": "allow", "../*": "deny", "/*": "deny"})
        else:
            rules[relative + "/**"] = "allow"
    return rules


def permissions(*, cwd: Path, role: str | None = None, writable_dirs=()) -> dict:
    # Do not set bash/read to deny: that removes native tool declarations and
    # causes Zen's free tier to reject even the official CLI. Noninteractive
    # `run` rejects `ask` requests; never pass --auto for these workflows.
    result = {"*": "ask", "task": "deny", "skill": "deny"}
    if role is None:  # Prompt-only generation / selection: no tool execution.
        return result
    result.update(read="allow", glob="allow", grep="allow", list="allow",
                  external_directory="allow", edit="deny")
    if role not in {"manager", "rating"}:
        result["edit"] = edit_permissions(cwd, writable_dirs)
    if role == "experiment":
        result["bash"] = "allow"
    return result


def environment(home: Path, *, model: str, permission: dict, api_key="") -> dict:
    """Isolate configuration/session DBs, reusing credentials without logging them."""
    env = dict(os.environ)
    provider = model.split("/", 1)[0]
    auth = env.get("OPENCODE_AUTH_CONTENT")
    if not auth:
        source = Path(env.get("XDG_DATA_HOME", str(Path.home() / ".local/share"))) / "opencode/auth.json"
        if source.is_file():
            auth = source.read_text(encoding="utf-8")
    key = api_key or env.get("OPENCODE_API_KEY", "")
    key_file = env.get("OPENCODE_API_KEY_FILE")
    if not key and key_file:
        key = Path(key_file).expanduser().read_text(encoding="utf-8").strip()
    if not key and not auth:
        legacy_key = Path(__file__).resolve().parents[2] / "codex/opecode"
        if legacy_key.is_file():
            key = legacy_key.read_text(encoding="utf-8").strip()
    if key:
        records = json.loads(auth) if auth else {}
        records[provider] = {"type": "api", "key": key}
        auth = json.dumps(records)
        env["OPENCODE_API_KEY"] = key
    if auth:
        env["OPENCODE_AUTH_CONTENT"] = auth
    for var in ("OPENCODE_CONFIG", "OPENCODE_CONFIG_DIR"):
        env.pop(var, None)
    home = home.resolve()
    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    for var, folder in (("XDG_DATA_HOME", "data"), ("XDG_CONFIG_HOME", "config"),
                        ("XDG_STATE_HOME", "state")):
        env[var] = str(home / folder)
    config = {
        "$schema": "https://opencode.ai/config.json", "model": model, "small_model": model,
        "share": "disabled", "autoupdate": False, "snapshot": False,
        "permission": permission, "agent": {"build": {"permission": permission}},
    }
    if key:
        config["provider"] = {provider: {"options": {"apiKey": "{env:OPENCODE_API_KEY}"}}}
    env.update(OPENCODE_CONFIG_CONTENT=json.dumps(config), OPENCODE_PERMISSION=json.dumps(permission),
               OPENCODE_DISABLE_PROJECT_CONFIG="true", OPENCODE_DISABLE_CLAUDE_CODE="true",
               OPENCODE_DISABLE_EXTERNAL_SKILLS="true", OPENCODE_DISABLE_LSP_DOWNLOAD="true")
    return env


def command(*, model: str, effort: str, cwd: Path, binary=None, session_id=None) -> list[str]:
    args = [executable(binary), "run", "--pure", "--format", "json", "--agent", "build",
            "--model", model, "--variant", effort, "--dir", str(cwd)]
    if session_id:
        args.extend(["--session", session_id])
    return args


def events(stdout: str) -> list[dict]:
    parsed = []
    for line in stdout.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            parsed.append(value)
    return parsed


def decode(stdout: str) -> tuple[str, str, dict]:
    parsed = events(stdout)
    failures = [entry for entry in parsed if entry.get("type") == "error"]
    if failures:
        raise OpenCodeError("OpenCode error: " + json.dumps(failures[-1].get("error"), ensure_ascii=False))
    session_id = next((entry["sessionID"] for entry in reversed(parsed) if entry.get("sessionID")), "")
    # Tool loops can emit progress text before their final answer. Only return
    # text from the last assistant message, joining multiple parts in that message.
    texts = [entry.get("part", {}) for entry in parsed if entry.get("type") == "text"]
    finishes = [entry.get("part", {}) for entry in parsed if entry.get("type") == "step_finish"]
    last_message = (finishes[-1].get("messageID") if finishes else None) or (texts[-1].get("messageID") if texts else None)
    message = "\n".join(part.get("text", "") for part in texts
                        if part.get("messageID") == last_message).strip()
    usage = {"input": 0, "output": 0, "reasoning": 0, "cache_read": 0, "cache_write": 0, "cost": 0.0}
    for part in finishes:
        tokens = part.get("tokens", {})
        for name in ("input", "output", "reasoning"):
            usage[name] += tokens.get(name, 0) or 0
        for name in ("read", "write"):
            usage["cache_" + name] += (tokens.get("cache") or {}).get(name, 0) or 0
        usage["cost"] += part.get("cost", 0) or 0
    if not session_id or not message or not finishes or finishes[-1].get("reason") != "stop":
        raise OpenCodeError("OpenCode returned no complete final response (session/text/stop event missing)")
    return message, session_id, usage


def schema_prompt(prompt: str, schema: dict | None) -> str:
    if schema is None:
        return prompt
    # CLI --format=json describes events, not the model's output format.
    return prompt + "\n\nReturn ONLY one JSON value matching this JSON Schema. No Markdown fences or commentary.\n" + json.dumps(schema, ensure_ascii=False)


def validate_json(message: str, schema: dict):
    import jsonschema
    text = re.sub(r"^```(?:json)?\s*\n?(.*?)\n?```$", r"\1", message.strip(), flags=re.DOTALL)
    value = json.loads(text)
    jsonschema.validate(value, schema)
    return value


@dataclass
class CLIResult:
    message: str
    session_id: str
    usage: dict
    stdout: str
    stderr: str
    elapsed: float


def run_cli(args: list[str], prompt: str, *, cwd: Path, env: dict,
            timeout: float, deadline_epoch: float | None = None, process_record: Path | None = None) -> CLIResult:
    started = time.monotonic()
    if deadline_epoch is not None:
        timeout = min(timeout, deadline_epoch - time.time())
    if timeout <= 0:
        raise OpenCodeError("OpenCode provider deadline exhausted")
    # Kill the entire CLI/tool process group on timeout, including shell checks.
    process_kwargs = dict(stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, cwd=cwd, env=env)
    if process_record is not None:
        from research.process_guard import close_guard, spawn_guarded
        proc, guard = spawn_guarded(args, record_path=process_record, backend="local", **process_kwargs)
    else:
        proc = subprocess.Popen(args, start_new_session=True, **process_kwargs)
    try:
        stdout, stderr = proc.communicate(prompt, timeout=timeout)
    except BaseException as exc:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = proc.communicate()
        if isinstance(exc, subprocess.TimeoutExpired):
            raise OpenCodeError(f"OpenCode timed out after {timeout:g}s", stdout=stdout,
                                stderr=stderr, elapsed=time.monotonic() - started) from exc
        raise
    finally:
        if process_record is not None:
            close_guard(process_record, guard)
    elapsed = time.monotonic() - started
    try:
        if proc.returncode:
            raise OpenCodeError(f"OpenCode exited {proc.returncode}: {(stdout + stderr)[-4000:]}")
        message, session_id, usage = decode(stdout)
    except OpenCodeError as exc:
        raise OpenCodeError(str(exc), stdout=stdout, stderr=stderr, elapsed=elapsed) from exc
    return CLIResult(message, session_id, usage, stdout, stderr, elapsed)


def _usage_event(cfg, event):
    if not getattr(cfg, "log_dir", None):
        return
    path = Path(cfg.log_dir) / "llm_calls.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, (json.dumps(event, ensure_ascii=False) + "\n").encode())
    finally:
        os.close(fd)


def _run_opencode(*, prompt, cfg, model, stage, json_schema=None,
                  timeout_seconds=None, reasoning_effort=None, prompt_only=False):
    del prompt_only  # All stateless calls are prompt-only; tools need explicit role sessions.
    model = model_id(model, str(stage.base_url))
    effort = reasoning_effort or setting("REASONING_EFFORT", "high")
    timeout = timeout_seconds if timeout_seconds is not None else float(setting("TIMEOUT_SECONDS", "1800"))
    call = {"call_id": uuid.uuid4().hex, "model": model, "provider": "opencode",
            "reasoning_effort": effort, "started_epoch": time.time()}
    _usage_event(cfg, {**call, "status": "started"})
    started = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix="mlevolve-opencode-") as directory:
            root = Path(directory)
            cwd = root / "workspace"
            cwd.mkdir()
            env = environment(root / "home", model=model, permission=permissions(cwd=cwd),
                              api_key=getattr(stage, "api_key", "") or "")
            result = run_cli(command(model=model, effort=effort, cwd=cwd), schema_prompt(prompt, json_schema),
                             cwd=cwd, env=env, timeout=timeout)
            message = result.message
            if json_schema is not None:
                message = json.dumps(validate_json(message, json_schema), ensure_ascii=False)
    except BaseException as exc:
        _usage_event(cfg, {**call, "status": "failed", "elapsed_seconds": time.monotonic() - started,
                           "error_type": type(exc).__name__})
        raise
    _usage_event(cfg, {**call, "status": "finished", "session_id": result.session_id,
                       "elapsed_seconds": result.elapsed, "usage": result.usage})
    return message, result.elapsed


def query(system_message, user_message, model, cfg, func_spec=None, **kwargs):
    stage = cfg.agent.code if cfg.agent.code.model == model else cfg.agent.feedback
    prompt = "\n\n".join(part for part in (system_message, user_message) if part)
    prompt = "Return the requested response. Do not use tools, read files or run commands.\n\n" + prompt
    schema = func_spec.json_schema if func_spec else None
    message, elapsed = _run_opencode(prompt=prompt, cfg=cfg, model=model, stage=stage,
                                    json_schema=schema, reasoning_effort=kwargs.get("reasoning_effort"))
    output = json.loads(message) if schema is not None else message
    return output, elapsed, 0, 0, {"model": model, "created": time.time()}


def generate(prompt, cfg, temperature=None, max_tokens=None, stop_tokens=None,
             json_schema=None, max_retries=20, retry_delay=3):
    # Decoding is owned by OpenCode/model variants, as with the Codex transport.
    from .codex import _compile_generation_prompt
    from jsonschema import ValidationError
    request = ("Act only as the language-model backend for this workflow. Return the requested content; "
               "do not use tools, read files or run shell commands.\n\n" + _compile_generation_prompt(prompt))
    attempts = max(1, min(int(max_retries), 3))
    for attempt in range(attempts):
        try:
            text, _ = _run_opencode(prompt=request, cfg=cfg, model=cfg.agent.code.model,
                                    stage=cfg.agent.code, json_schema=json_schema)
            return text
        except (OpenCodeError, json.JSONDecodeError, ValidationError):
            if attempt + 1 == attempts:
                raise
            time.sleep(retry_delay)
