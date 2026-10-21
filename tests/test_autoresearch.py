from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from engine.executor import ExecutionResult
from engine.search_node import SearchNode
from tasks.autoresearch.adapter import AutoresearchTask, REPO, START, END, validate_result, assemble_candidate, candidate_body
from agents import result_parse_agent
from omegaconf import OmegaConf
from tools import codex_opencode
from config import _load_cfg


@pytest.fixture
def task(tmp_path):
    task = AutoresearchTask.__new__(AutoresearchTask)
    task.baseline = (REPO / "train.py").read_text()
    task.prefix = task.baseline.split(START, 1)[0]
    task.suffix = END + task.baseline.split(END, 1)[1]
    task.cfg = SimpleNamespace(workspace_dir=tmp_path)
    task.protocol_id = "test-protocol"
    task.spec = {"metric": {"maximize": False}}
    return task


def result_for(code):
    return {"valid": True, "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
            "val_bpb": 1.7, "training_seconds": 301.5, "max_step_seconds": 2.1,
            "peak_vram_mb": 16000.0, "num_steps": 160, "eval_tokens": 20_971_520,
            "model_contract_check": {"passed": True},
            "gpu": "NVIDIA A100-SXM4-80GB"}


def test_model_edit_preserves_measurement(task):
    code = candidate_body(task.baseline).replace("MATRIX_LR = 0.04", "MATRIX_LR = 0.03")
    assembled = assemble_candidate(code)
    assert "MATRIX_LR = 0.03" in assembled
    assert assembled.startswith(task.prefix) and assembled.endswith(task.suffix)


@pytest.mark.parametrize("code", [
    "TIME_BUDGET = 1", "def evaluate_bpb(*args): return 0.1", "import time",
    "from prepare import make_dataloader", "MAX_SEQ_LEN = 128",
])
def test_timing_and_evaluator_edits_rejected(code):
    with pytest.raises(ValueError, match="immutable"):
        assemble_candidate(code)


@pytest.mark.parametrize("change", [
    {"training_seconds": 299.9}, {"training_seconds": 310.0},
    {"val_bpb": float("nan")}, {"val_bpb": float("inf")}, {"val_bpb": True},
    {"eval_tokens": 1024}, {"num_steps": 3}, {"code_sha256": "stale"},
    {"valid": False}, {"peak_vram_mb": 0.0},
    {"model_contract_check": {"passed": False}},
])
def test_incomplete_and_stale_trials_rejected(change):
    result = result_for("train") | change
    with pytest.raises(ValueError):
        validate_result(result, "train", 900)


def test_score_is_captured_result_not_candidate_stdout(task, tmp_path, monkeypatch):
    node = SearchNode(code=task.baseline, stage="draft")
    trial = tmp_path / "trials" / node.id
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(json.dumps(result_for(assemble_candidate(node.code))))
    cfg = _load_cfg(use_cli_args=False)
    cfg.workspace_dir = tmp_path
    saved = []
    agent = SimpleNamespace(autoresearch=task, cfg=cfg, acfg=cfg.agent,
                            global_memory=SimpleNamespace(save_node=lambda *args: saved.append(args)))
    monkeypatch.setattr(result_parse_agent, "query", lambda **kwargs: {
        "is_bug": False, "summary": "Measured BPB", "code_summary": "Causal GPT"})
    result_parse_agent.run(agent, node, ExecutionResult(["val_bpb: 0.00001"], 800, None))
    assert len(saved) == 1 and node.code_summary == "Causal GPT"
    assert node.is_valid and not node.is_buggy
    assert node.metric.value == 1.7 and node.metric.maximize is False


def test_crash_cannot_admit_stale_result(task, tmp_path, monkeypatch):
    node = SearchNode(code=task.baseline, stage="draft")
    trial = tmp_path / "trials" / node.id
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(json.dumps(result_for(assemble_candidate(node.code))))
    cfg = _load_cfg(use_cli_args=False)
    agent = SimpleNamespace(autoresearch=task, cfg=cfg)
    monkeypatch.setattr(result_parse_agent, "query", lambda **kwargs: pytest.fail("Do not summarize a crashed process as valid"))
    result_parse_agent.run(agent, node, ExecutionResult(["CUDA out of memory"], 900, "RuntimeError"))
    assert node.is_buggy and not node.is_valid and node.metric.is_worst


def test_provider_config_applies_to_exec_without_key_in_argv(tmp_path, monkeypatch):
    key_file = tmp_path / "key"
    key_file.write_text("test-private-credential")
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    monkeypatch.setenv("OPENCODE_API_KEY_FILE", str(key_file))
    monkeypatch.setenv("MLEVOLVE_REAL_CODEX_BIN", "/tmp/real-codex")
    monkeypatch.setattr("sys.argv", ["codex_opencode.py", "exec", "--ephemeral", "-"])
    called = {}

    def execute(binary, args, env):
        called.update(binary=binary, args=args, env=env)

    monkeypatch.setattr(codex_opencode.os, "execvpe", execute)
    codex_opencode.main()
    assert called["args"][:2] == ["/tmp/real-codex", "exec"]
    assert 'model_provider="opencode-go"' in called["args"]
    assert not any("test-private-credential" in arg for arg in called["args"])
    assert called["env"]["OPENCODE_API_KEY"] == "test-private-credential"
    assert called["env"]["OPENCODE_SESSION_ID"]
    assert any("x-opencode-session" in arg for arg in called["args"])


def test_preset_cannot_reinterpret_a_competition_checkpoint(tmp_path, monkeypatch):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "config.yaml").write_text("agent:\n  steps: 2\n")
    monkeypatch.setattr("sys.argv", ["run.py", "task_preset=autoresearch", f"resume_run_dir={tmp_path}"])
    with pytest.raises(ValueError, match="different task type"):
        _load_cfg()


def test_autoresearch_preset_keeps_both_models_on_go(monkeypatch):
    monkeypatch.setattr("sys.argv", ["run.py", "task_preset=autoresearch"])
    cfg = _load_cfg()
    assert cfg.task_type == "autoresearch"
    assert not cfg.evaluation.enabled and cfg.coldstart.use_coldstart
    assert cfg.parent_selector.type == "bnvs_ts"
    for stage in (cfg.agent.code, cfg.agent.feedback):
        assert stage.model == "opencode/muse-spark-1.3-contributor-free"
        assert stage.base_url == "opencode://opencode"


@pytest.mark.parametrize("saved_workers,override,expected", [
    (None, None, 2), (None, 1, 1), (None, 3, 3), (1, None, 1), (1, 2, 2),
])
def test_nanogpt_concurrency_default_override_and_resume(saved_workers, override, expected, tmp_path, monkeypatch):
    args = ["run.py", "task_preset=autoresearch"]
    if saved_workers is not None:
        logs = tmp_path / "logs"
        logs.mkdir()
        OmegaConf.save(OmegaConf.create({"task_type": "autoresearch", "agent": {
            "search": {"parallel_search_num": saved_workers}}}), logs / "config.yaml")
        args.append(f"resume_run_dir={tmp_path}")
    if override is not None:
        args.append(f"agent.search.parallel_search_num={override}")
    monkeypatch.setattr("sys.argv", args)
    cfg = _load_cfg()
    assert cfg.agent.search.parallel_search_num == expected
    assert cfg.agent.initial_drafts == 3 and cfg.agent.search.num_drafts == 5
