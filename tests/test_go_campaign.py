"""Approved Go routing from protocol setup through generation/rating and launch."""
import json
from pathlib import Path
from types import SimpleNamespace as NS
import subprocess
import sys
import tomllib
import pytest
from tasks.mlebench_lite import mila_multifidelity as campaign
from llm import codex, generate, query, FunctionSpec

ROOT = Path(__file__).resolve().parents[1]


def test_manifests_and_all_search_roles_match_go(monkeypatch):
    monkeypatch.setenv("OPENCODE_API_KEY", "unrelated-inherited-key")
    for filename in ("mlebench_mf14.json", "mlebench_mfgp14.json"):
        spec = campaign.read(ROOT / "config" / filename)
        assert (spec["model"], spec["provider"], spec["reasoning_effort"], spec["codex_version"]) == (
            "muse-spark-1.3-contributor", "codex://opencode-go", "high", "0.164.0")
        settings = {"spec": spec, "data_dir": "/data", "key_file": "/key", "token_file": "/token"}
        for mode in spec["modes"]:
            command = campaign.search_command(settings, ROOT / "runs/test", "aerial-cactus-identification", mode)
            for role in ("code", "feedback"):
                assert f"agent.{role}.model={spec['model']}" in command
                assert f"agent.{role}.base_url={spec['provider']}" in command
        env = campaign.environment(settings)
        assert "OPENCODE_API_KEY" not in env
        assert env["OPENCODE_API_KEY_FILE"] == "/key"
        assert env["MLEVOLVE_CODEX_REASONING_EFFORT"] == "high"
        assert env["MLEVOLVE_CODEX_BIN"] == str(ROOT / "tools/codex_opencode.py")
        assert "tools/check_go_provider.py" in campaign.provider_check_command(settings, "/out")
    config = tomllib.loads((ROOT / "config/opencode-go.toml").read_text())
    assert config["model_reasoning_effort"] == "high"
    assert config["model_supports_reasoning_summaries"] is True
    assert config["model_providers"]["opencode-go"]["base_url"] == "https://opencode.ai/inference/go/openai/v1"


def test_cli_version_drift_blocks_run(tmp_path, monkeypatch):
    spec = campaign.read(campaign.SPEC)
    binary = tmp_path / "codex"
    binary.write_text("fixture")
    monkeypatch.setenv("MLEVOLVE_REAL_CODEX_BIN", str(binary))
    monkeypatch.setattr(campaign, "run", lambda *a, **kw: NS(stdout="codex-cli 0.163.0\n"))
    with pytest.raises(RuntimeError, match="version differs"):
        campaign.verify_codex({"spec": spec}, tmp_path / "receipt.json")
    assert not (tmp_path / "receipt.json").exists()
    monkeypatch.setattr(campaign, "run", lambda *a, **kw: NS(stdout="codex-cli 0.164.0\n"))
    campaign.verify_codex({"spec": spec}, tmp_path / "receipt.json")
    assert campaign.read(tmp_path / "receipt.json")["sha256"] == campaign.sha(binary)


def test_generation_and_feedback_dispatch_to_go_high(tmp_path, monkeypatch):
    monkeypatch.delenv("MLEVOLVE_CODEX_BIN", raising=False)
    monkeypatch.setenv("MLEVOLVE_CODEX_REASONING_EFFORT", "high")
    stage = NS(model="muse-spark-1.3-contributor", base_url="codex://opencode-go")
    cfg = NS(workspace_dir=tmp_path, log_dir=tmp_path / "logs", agent=NS(code=stage, feedback=stage))
    commands = []
    def fake_run(command, **kw):
        commands.append(command)
        assert command[:3] == [sys.executable, str(ROOT / "tools/codex_opencode.py"), "exec"]
        assert 'model_reasoning_effort="high"' in command
        assert 'model_provider="opencode-go"' in command
        assert command[command.index("--model")+1] == stage.model
        output = Path(command[command.index("--output-last-message")+1])
        output.write_text('{"ok":true}' if "--output-schema" in command else "plan")
        return subprocess.CompletedProcess(command, 0, stdout="tokens used\n12")
    monkeypatch.setattr(codex.subprocess, "run", fake_run)
    assert generate("Write plan", cfg, max_retries=1) == "plan"
    schema = FunctionSpec(name="rate", description="rate", json_schema={"type":"object", "properties":{"ok":{"type":"boolean"}}, "required":["ok"]})
    assert query("Rate", None, stage.model, cfg=cfg, func_spec=schema) == {"ok":True}
    assert len(commands) == 2


def test_protocol_preparation_uses_campaign_model(tmp_path, monkeypatch):
    from tasks.mlebench_lite import prepare_fixed_protocol as prep
    from omegaconf import OmegaConf
    task = "aerial-cactus-identification"
    public = tmp_path / task / "prepared/public"
    public.mkdir(parents=True)
    (public / "description.md").write_text("fixture")
    cfg = OmegaConf.create({"agent":{"code":{"model":"old","base_url":"old"},"feedback":{"model":"old","base_url":"old"},"seed":0},"evaluation":{"enabled":False}})
    monkeypatch.setattr(prep, "_load_cfg", lambda **kw: cfg)
    monkeypatch.setattr(prep, "prep_cfg", lambda c: c)
    monkeypatch.setattr(prep, "load_task_desc", lambda c: "fixture")
    def check(c, *a):
        assert c.agent.code.model == c.agent.feedback.model == "muse-spark-1.3-contributor"
        assert c.agent.code.base_url == c.agent.feedback.base_url == "codex://opencode-go"
        return NS(directory=tmp_path, protocol_id="fixture", spec={"metric":{},"split":{}})
    monkeypatch.setattr(prep.FixedEvaluation,"prepare_or_load",check)
    prep.prepare(NS(task=task,data_root=tmp_path,output_root=tmp_path,protocol_dir=None,seed=42,
        model="muse-spark-1.3-contributor",provider="codex://opencode-go"))
