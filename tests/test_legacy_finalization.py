"""Legacy final ensembles use the complete existing fusion implementation."""

import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from omegaconf import OmegaConf

from test_root_draft_runtime import real_agent
from utils import submission_fusion_utils as fusion


@pytest.mark.parametrize("case", ["numeric", "text", "budget", "skip", "best_only"])
def test_exact_run_paths_preserve_legacy_ensemble_outputs(tmp_path, case, monkeypatch):
    legacy_run = tmp_path / "runs/20260916_120000_fixture"
    workspace = legacy_run / "workspace"
    task = "rsna-miccai-brain-tumor-radiogenomic-classification" if case == "skip" else "fixture"
    for rank in range(1, 7):
        directory = workspace / "top_solution" / f"top{rank}"
        directory.mkdir(parents=True)
        values = ["one", "two", "three"] if case == "text" else [0.1 * rank, 0.2, 0.3]
        frame = pd.DataFrame({"id": ["a", "b", "c"], "prediction": values})
        if rank % 2 == 0:
            frame = frame.iloc[::-1]
        frame.to_csv(directory / "submission.csv", index=False)
        directory.joinpath("metric.txt").write_text(
            f"Metric: {0.9 - rank * 0.05}\nMaximize: True\n"
            f"Execution Time(s): {rank * 3600 if case == 'budget' else 60}\n"
        )
    best = workspace / "best_submission"
    best.mkdir()
    shutil.copyfile(workspace / "top_solution/top1/submission.csv", best / "submission.csv")
    if case == "best_only":
        shutil.rmtree(workspace / "top_solution")
    exact_workspace = tmp_path / "separate workspace"
    shutil.copytree(workspace, exact_workspace)
    exact_logs = tmp_path / "separate logs"
    exact_logs.mkdir()
    OmegaConf.save(OmegaConf.create({"evaluation": {"enabled": False}}), exact_logs / "config.yaml")
    args = SimpleNamespace(
        task_id=task, runs_root=str(legacy_run.parent), exp_name=legacy_run.name,
        use_llm_selection=False, tag_path=str(tmp_path / "tags.json"),
    )
    monkeypatch.setattr(fusion, "_tag_cache", None)
    fusion.ensemble(args)
    expected = {p.name: p.read_bytes() for p in (workspace / "ensembles_csv").glob("*.csv")}
    sizes = {int(name.split("ens", 1)[0][3:]) for name in expected}
    assert sizes == ({1} if case in {"skip", "best_only"} else {1, 2, 3} if case == "budget" else {1, 2, 3, 4, 6})

    def wrong_run(*args):
        raise AssertionError("Explicit run paths must not search for another run")
    monkeypatch.setattr(fusion, "get_closest_run_dir", wrong_run)
    info = {"log_dir": str(exact_logs), "workspace_dir": str(exact_workspace), "fixed_evaluation": False}
    fusion.ensemble(args, run_info=info)
    actual = {p.name: p.read_bytes() for p in (exact_workspace / "ensembles_csv").glob("*.csv")}
    assert actual == expected
    final = json.loads((exact_logs / "final_submission.json").read_text())
    assert final["selection"] == "latest_structurally_valid_ensemble"
    assert Path(final["submission"]).parent == exact_workspace / "ensembles_csv"

    # Exercise the shell fallback entry with paths containing spaces, from another cwd.
    marker = tmp_path / "run info.json"
    marker.write_text(json.dumps(info))
    shutil.rmtree(exact_workspace / "ensembles_csv")
    result = subprocess.run([
        sys.executable, str(Path(fusion.__file__).resolve()), "--task_id", task,
        "--run_info", str(marker), "--tag_path", args.tag_path,
    ], cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert {p.name: p.read_bytes() for p in (exact_workspace / "ensembles_csv").glob("*.csv")} == expected
    assert Path(json.loads((exact_logs / "final_submission.json").read_text())["submission"]).is_file()


@pytest.mark.parametrize("preset,budget_exhausted", [
    (None, False), ("gp7_ancestry_ucb", False), ("gp7_ancestry_ucb", True),
])
def test_run_automatically_ensembles_legacy_candidates_and_resumes(real_agent, monkeypatch, preset, budget_exhausted):
    import run as runner

    agent, _, services = real_agent(initial=1, selector="legacy_original", enabled=False)
    cfg = agent.cfg
    if preset:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(Path(__file__).parents[1] / f"config/experiments/{preset}.yaml"))
    cfg.agent.steps = 2
    if budget_exhausted:
        cfg.agent.time_limit = 2
    cfg.exp_id = "fixture"
    cfg.run_result_file = str(cfg.workspace_dir / "run-info.json")
    monkeypatch.setattr(runner, "load_cfg", lambda: cfg)
    monkeypatch.setattr(runner, "load_task_desc", lambda cfg: "Predict labels")
    monkeypatch.setattr(runner, "prep_agent_workspace", lambda cfg: None)

    class Interpreter:
        max_parallel_run = 1

        def __init__(self, *args, **kwargs):
            pass

        def run(self, *args, **kwargs):
            result = services.execute(cfg, *args)
            if budget_exhausted:
                monkeypatch.setattr("time.time", lambda: 1003.0)
            return result

        def cleanup_session(self, *args):
            pass

    monkeypatch.setattr(runner, "Interpreter", Interpreter)
    runner.run()
    outputs = {p.name: p.read_bytes() for p in (cfg.workspace_dir / "ensembles_csv").glob("*.csv")}
    expected_count = 1 if budget_exhausted else 2
    assert len(outputs) == expected_count
    assert (cfg.workspace_dir / f"top_solution/top{expected_count}/submission.csv").is_file()
    final = json.loads((cfg.log_dir / "final_submission.json").read_text())
    assert final["selection"] == "latest_structurally_valid_ensemble"
    assert Path(final["submission"]).parent == cfg.workspace_dir / "ensembles_csv"
    cfg.resume_run_dir = str(cfg.log_dir.parent)
    runner.run()
    assert {p.name: p.read_bytes() for p in (cfg.workspace_dir / "ensembles_csv").glob("*.csv")} == outputs


def test_partial_fusion_failure_still_records_completed_submission(tmp_path, monkeypatch):
    workspace, logs = tmp_path / "workspace", tmp_path / "logs"
    for rank in (1, 2):
        directory = workspace / "top_solution" / f"top{rank}"
        directory.mkdir(parents=True)
        (directory / "metric.txt").write_text("Metric: 0.8\nMaximize: True\nExecution Time(s): 1\n")
    first = workspace / "top_solution/top1/submission.csv"
    first.write_text("id,prediction\na,0.1\nb,0.2\n")
    best = workspace / "best_submission/submission.csv"
    best.parent.mkdir()
    shutil.copyfile(first, best)
    monkeypatch.setattr(fusion, "_tag_cache", {})
    args = SimpleNamespace(task_id="fixture", use_llm_selection=False, tag_path="unused")
    with pytest.raises(FileNotFoundError, match="top2"):
        fusion.ensemble(args, run_info={"log_dir": str(logs), "workspace_dir": str(workspace)})
    record = json.loads((logs / "final_submission.json").read_text())
    assert record["selection"] == "latest_structurally_valid_ensemble"
    assert Path(record["submission"]).read_bytes() == first.read_bytes()
