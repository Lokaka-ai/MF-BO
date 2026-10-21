import sys

from omegaconf import OmegaConf

from config import _load_cfg, load_cfg


def test_resume_loads_saved_config_before_explicit_cli_overrides(tmp_path, monkeypatch):
    run_dir = tmp_path / "existing-run"
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True)
    (run_dir / "workspace").mkdir()
    (log_dir / "journal.json").write_text("{}")

    saved = _load_cfg(use_cli_args=False)
    saved.data_dir = str(tmp_path / "saved-input")
    saved.goal = "saved goal"
    saved.agent.steps = 17
    OmegaConf.save(saved, log_dir / "config.yaml")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            f"resume_run_dir={run_dir}",
            "agent.steps=23",
        ],
    )
    resumed = load_cfg()

    assert resumed.data_dir == (tmp_path / "saved-input").resolve()
    assert resumed.goal == "saved goal"
    assert resumed.agent.steps == 23
    assert resumed.resume_run_dir == run_dir.resolve()
    assert resumed.log_dir == log_dir.resolve()
    assert resumed.workspace_dir == (run_dir / "workspace").resolve()


def test_non_resume_config_path_is_unchanged(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run.py", "agent.steps=19"])

    cfg = _load_cfg()

    assert cfg.resume_run_dir is None
    assert cfg.data_dir is None
    assert cfg.agent.steps == 19
