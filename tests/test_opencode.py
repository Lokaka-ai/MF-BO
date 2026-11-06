"""Exercise the production native transport through a fake external CLI."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest

from llm import opencode


FAKE_CLI = r'''
import json, os, subprocess, sys, time
from pathlib import Path
args = sys.argv[1:]
prompt = sys.stdin.read()
log = Path(os.environ['FAKE_OPENCODE_LOG'])
with log.open('a') as f:
    f.write(json.dumps({'args': args, 'prompt': prompt, 'cwd': os.getcwd(),
        'config': json.loads(os.environ['OPENCODE_CONFIG_CONTENT']),
        'permission': json.loads(os.environ['OPENCODE_PERMISSION']),
        'data_home': os.environ['XDG_DATA_HOME']}) + '\n')
case = os.environ.get('FAKE_OPENCODE_CASE', '')
if case == 'timeout':
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    log.with_suffix('.pid').write_text(str(child.pid))
    time.sleep(60)
if case == 'error' or (case == 'resume_error' and '--session' in args):
    print(json.dumps({'type':'error', 'sessionID':'ses_test',
                      'error':{'name':'APIError', 'data':{'statusCode':403, 'message':'free tier rejected'}}}))
    sys.exit(0)
sid = args[args.index('--session')+1] if '--session' in args else 'ses_test'
def emit(kind, **part):
    print(json.dumps({'type':kind, 'sessionID':sid, 'part':part}))
emit('text', messageID='progress', text='Working on it...')
emit('reasoning', messageID='final', text='Private reasoning is not a final answer')
if case != 'empty':
    emit('text', messageID='final', text=os.environ.get('FAKE_OPENCODE_MESSAGE', '{"ok":true}'))
emit('step_finish', messageID='final', reason='length' if case == 'truncated' else 'stop',
     tokens={'input':7,'output':3,'reasoning':2,'cache':{'read':4,'write':0}}, cost=0)
'''


@pytest.fixture
def fake_opencode(tmp_path, monkeypatch):
    path = tmp_path / "opencode"
    path.write_text(f"#!{sys.executable}\n" + FAKE_CLI)
    path.chmod(0o755)
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("MLEVOLVE_OPENCODE_BIN", str(path))
    monkeypatch.setenv("FAKE_OPENCODE_LOG", str(log))
    monkeypatch.setenv("OPENCODE_AUTH_CONTENT", "{}")
    for key in ("OPENCODE_API_KEY", "OPENCODE_API_KEY_FILE", "MLEVOLVE_OPENCODE_REASONING_EFFORT",
                "MLEVOLVE_CODEX_REASONING_EFFORT"):
        monkeypatch.delenv(key, raising=False)
    stage = NS(model=opencode.DEFAULT_MODEL, base_url=opencode.DEFAULT_BASE_URL, api_key="")
    cfg = NS(workspace_dir=tmp_path, log_dir=tmp_path / "logs", agent=NS(code=stage, feedback=stage))
    return cfg, log


def request(cfg, **kwargs):
    return opencode._run_opencode(prompt="Return JSON.", cfg=cfg, model=cfg.agent.code.model,
                                  stage=cfg.agent.code, **kwargs)


def test_native_dispatch_schema_usage_and_no_auto_approval(fake_opencode):
    from llm import FunctionSpec, generate, query
    cfg, log = fake_opencode
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
    assert json.loads(generate("Return JSON.", cfg, json_schema=schema)) == {"ok": True}
    spec = FunctionSpec(name="check", description="Check response", json_schema=schema)
    assert query("Return JSON.", None, cfg.agent.code.model, func_spec=spec, cfg=cfg) == {"ok": True}
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    for call, effort in zip(calls, ("high", "high")):
        args = call["args"]
        assert args[0] == "run" and args[args.index("--variant") + 1] == effort
        assert "--auto" not in args and "--output-schema" not in args
        assert "JSON Schema" in call["prompt"]
        assert call["permission"]["*"] == "ask"
        assert call["config"]["small_model"] == opencode.DEFAULT_MODEL
        assert str(cfg.workspace_dir) != call["cwd"]
    assert calls[0]["data_home"] != calls[1]["data_home"]
    usage = [json.loads(line) for line in (cfg.log_dir / "llm_calls.jsonl").read_text().splitlines()]
    assert usage[-1]["usage"] == {"input": 7, "output": 3, "reasoning": 2,
                                  "cache_read": 4, "cache_write": 0, "cost": 0}


def test_reasoning_preserves_legacy_override_and_explicit_role_effort(fake_opencode, monkeypatch):
    cfg, log = fake_opencode
    monkeypatch.setenv("MLEVOLVE_CODEX_REASONING_EFFORT", "medium")
    request(cfg)
    monkeypatch.setenv("MLEVOLVE_OPENCODE_REASONING_EFFORT", "low")
    request(cfg)
    request(cfg, reasoning_effort="high")
    calls = [json.loads(line)["args"] for line in log.read_text().splitlines()]
    assert [args[args.index("--variant") + 1] for args in calls] == ["medium", "low", "high"]


@pytest.mark.parametrize("case", ["error", "empty", "truncated"])
def test_exit_zero_is_not_success_without_complete_model_answer(fake_opencode, monkeypatch, case):
    cfg, _ = fake_opencode
    monkeypatch.setenv("FAKE_OPENCODE_CASE", case)
    with pytest.raises(opencode.OpenCodeError):
        request(cfg)
    assert json.loads((cfg.log_dir / "llm_calls.jsonl").read_text().splitlines()[-1])["status"] == "failed"


def test_dynamic_json_schema_is_preserved_and_validated(fake_opencode, monkeypatch):
    import jsonschema
    cfg, _ = fake_opencode
    schema = {"type": "object", "additionalProperties": {"type": "number"}}
    monkeypatch.setenv("FAKE_OPENCODE_MESSAGE", '```json\n{"candidate-a": 0.75}\n```')
    text, _ = request(cfg, json_schema=schema)
    assert json.loads(text) == {"candidate-a": 0.75}
    monkeypatch.setenv("FAKE_OPENCODE_MESSAGE", '{"candidate-a": "invalid"}')
    with pytest.raises(jsonschema.ValidationError):
        request(cfg, json_schema=schema)


def test_timeout_stops_cli_and_its_shell_children(fake_opencode, monkeypatch):
    cfg, log = fake_opencode
    monkeypatch.setenv("FAKE_OPENCODE_CASE", "timeout")
    with pytest.raises(opencode.OpenCodeError, match="timed out"):
        request(cfg, timeout_seconds=0.5)
    pid = int(log.with_suffix(".pid").read_text())
    state = Path(f"/proc/{pid}/stat")
    assert not state.exists() or state.read_text().split()[2] == "Z"


def test_credentials_are_in_environment_not_argv_or_logs(fake_opencode, monkeypatch, tmp_path):
    cfg, log = fake_opencode
    key = "synthetic-test-credential"
    source = tmp_path / "credential"
    source.write_text(key)
    monkeypatch.setenv("OPENCODE_API_KEY_FILE", str(source))
    env = opencode.environment(tmp_path / "isolated", model=opencode.DEFAULT_MODEL,
                               permission={"*": "ask"})
    assert json.loads(env["OPENCODE_AUTH_CONTENT"])["opencode"]["key"] == key
    request(cfg)
    assert key not in log.read_text()
    assert key not in (cfg.log_dir / "llm_calls.jsonl").read_text()
    assert not list((tmp_path / "isolated").rglob("auth.json"))


def test_provider_routing_is_explicit():
    assert opencode.provider_base_url("opencode") == opencode.DEFAULT_BASE_URL
    assert opencode.provider_base_url("opencode-go") == "codex://opencode-go"
    assert opencode.model_id("muse-spark-1.3-contributor-free") == opencode.DEFAULT_MODEL
    with pytest.raises(ValueError):
        opencode.model_id(opencode.DEFAULT_MODEL, "opencode://different")


def test_generation_retries_are_bounded_inside_one_workflow_call(fake_opencode, monkeypatch):
    cfg, _ = fake_opencode
    calls = []
    def transient(**kwargs):
        calls.append(kwargs["prompt"])
        if len(calls) < 3:
            raise opencode.OpenCodeError("temporary provider failure")
        return "recovered", 0.0
    monkeypatch.setattr(opencode, "_run_opencode", transient)
    assert opencode.generate("request", cfg, retry_delay=0) == "recovered"
    assert len(calls) == 3 and len(set(calls)) == 1
    calls.clear()
    with pytest.raises(opencode.OpenCodeError):
        opencode.generate("request", cfg, max_retries=1, retry_delay=0)
    assert len(calls) == 1


def test_edit_permissions_use_worktree_relative_paths(tmp_path):
    import subprocess
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    candidate = tmp_path / "candidates/one"
    candidate.mkdir(parents=True)
    assert opencode.permissions(cwd=candidate, role="experiment")["edit"] == {
        "*": "deny", "candidates/one/**": "allow"}


def test_parent_selector_dispatches_native_cli_with_medium_effort(fake_opencode, monkeypatch):
    from engine.parent_selectors.agent_selector import AgentParentSelector
    cfg, log = fake_opencode
    monkeypatch.setenv("FAKE_OPENCODE_MESSAGE", '{"candidate":"c0","reason":"test evidence"}')
    selector = AgentParentSelector(cfg=cfg, records_provider=lambda: [])
    response = selector._request("Choose c0.", {"type": "object", "required": ["candidate", "reason"]}, 5)
    assert json.loads(response)["candidate"] == "c0"
    call = json.loads(log.read_text().splitlines()[-1])
    assert call["args"][call["args"].index("--variant") + 1] == "medium"
