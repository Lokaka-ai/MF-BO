from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest

from config import _load_cfg, validate_root_draft_config
from tasks.autoresearch.launch import select_cpus, compile_cache_environment
from tasks.autoresearch.supervise import ROOT, build_command, process_table, supervise


@pytest.mark.parametrize("setting,selector,scope", [
    ("gp7_ancestry_ucb", "gp7_ancestry_logei", "all_normal_steps"),
    ("gp7_long_horizon_ucb", "gp7_long_horizon_logei", "all_normal_steps"),
    ("mlevolve_original", "legacy_original", "root_only"),
])
def test_campaign_keeps_training_and_model_fixed(setting, selector, scope, monkeypatch, tmp_path):
    campaign = OmegaConf.load(ROOT / "tasks/autoresearch/three_settings_8h.yaml")
    command = build_command(campaign, setting, "GPU-test", tmp_path, "1,3,5,7")
    overrides = command[command.index(f"experiment_preset={setting}"):]
    monkeypatch.setattr(sys, "argv", ["run.py", "task_preset=autoresearch", *overrides])
    cfg = _load_cfg()
    validate_root_draft_config(cfg)
    assert cfg.parent_selector.type == selector
    assert cfg.parent_selector.decision_scope == scope
    assert cfg.parent_selector.include_root_draft is False
    assert cfg.agent.time_limit == 28800 and cfg.cpu_number == 4
    assert command[command.index("--steps") + 1] == "500"
    assert cfg.exec.timeout == 32400 and cfg.agent.search.parallel_search_num == 2
    assert cfg.agent.use_evolution and cfg.agent.use_global_memory
    assert cfg.agent.initial_drafts == 3 and cfg.agent.search.num_drafts == 5
    from tasks.autoresearch.audit import audit_campaign
    assert audit_campaign(campaign, setting)["passed"]
    for stage in (cfg.agent.code, cfg.agent.feedback):
        assert stage.model == "opencode/muse-spark-1.3-contributor-free"
        assert stage.base_url == "opencode://opencode"
    if setting.startswith("gp7"):
        assert cfg.bo_parent_selector.model == "gp_matern32"
        assert cfg.bo_parent_selector.acquisition.type == "child_ucb"


def test_cpu_partition_respects_noncontiguous_allocation():
    assert select_cpus({0, 1, 4, 5, 18, 19}, 4, "4,5,18,19") == [4, 5, 18, 19]
    for invalid in ("0,1,2,3", "0,0,4,5", "0,1"):
        with pytest.raises(ValueError):
            select_cpus({0, 1, 4, 5, 18, 19}, 4, invalid)
    with pytest.raises(ValueError):
        select_cpus({0, 1}, 4)


def test_concurrent_runs_use_disjoint_compilation_caches(tmp_path):
    first = compile_cache_environment(tmp_path / "first")
    second = compile_cache_environment(tmp_path / "second")
    assert set(first) == {"TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "CUDA_CACHE_PATH"}
    assert set(first.values()).isdisjoint(second.values())
    assert all(Path(path).is_dir() for path in [*first.values(), *second.values()])


def test_deadline_stops_detached_children_and_records_completion(tmp_path):
    pid_file = tmp_path / "child.json"
    program = (
        "import json,signal,subprocess,sys,time; from pathlib import Path; "
        "signal.signal(signal.SIGINT,signal.SIG_IGN); "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True); "
        f"Path({str(pid_file)!r}).write_text(json.dumps(child.pid)); time.sleep(60)"
    )
    code = supervise([sys.executable, "-c", program], tmp_path / "run", 1, grace_seconds=0.2)
    assert code == 0
    status = json.loads((tmp_path / "run/status.json").read_text())
    assert status["stop_reason"] == "time_budget" and status["state"] == "finished"
    assert status["elapsed_seconds"] < 10
    table = process_table()
    for pid in (status["child_pid"], json.loads(pid_file.read_text())):
        assert pid not in table or table[pid][2] == "Z"


def test_normal_failure_does_not_look_like_eight_hour_success(tmp_path):
    assert supervise([sys.executable, "-c", "raise SystemExit(7)"], tmp_path, 10) == 7
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["exit_code"] == 7 and status["stop_reason"] == "process_exit"


def test_supervisor_preserves_existing_run(tmp_path):
    (tmp_path / "status.json").write_text('{"state":"running"}')
    with pytest.raises(FileExistsError):
        supervise([sys.executable, "-c", "pass"], tmp_path, 1)
    assert json.loads((tmp_path / "status.json").read_text())["state"] == "running"


def test_opencode_python_wrapper_uses_uv_interpreter(monkeypatch, tmp_path):
    from llm import codex

    monkeypatch.delenv("MLEVOLVE_CODEX_BIN", raising=False)
    captured = []

    def complete(command, **kwargs):
        captured.extend(command)
        Path(command[command.index("--output-last-message") + 1]).write_text("ok")
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(codex.subprocess, "run", complete)
    result, _ = codex._run_codex(prompt="test", cfg=SimpleNamespace(workspace_dir=tmp_path),
        model="muse-spark-1.3-contributor", stage=SimpleNamespace(base_url="codex://opencode-go"))
    assert result == "ok"
    assert captured[:2] == [sys.executable, str(ROOT / "tools/codex_opencode.py")]


def test_audit_rejects_agent_simplifications():
    from tasks.autoresearch.audit import audit_campaign
    cfg = OmegaConf.load(ROOT / "tasks/autoresearch/three_settings_8h.yaml")
    cfg.overrides["agent.use_global_memory"] = False
    with pytest.raises(ValueError, match="agent.use_global_memory"):
        audit_campaign(cfg, "gp7_ancestry_ucb")
