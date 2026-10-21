"""The final grader follows the historical Base rule, without private-score selection."""

import csv
import json
import os
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from omegaconf import OmegaConf

from utils import grade_submission
from utils.grade_submission import finalize_legacy_submission, grade_run


def write_csv(path, text, timestamp=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    os.utime(path, ns=(timestamp, timestamp))
    return path


@pytest.fixture
def saved_run(tmp_path):
    logs, workspace = tmp_path / "separate logs", tmp_path / "separate workspace"
    logs.mkdir()
    OmegaConf.save(OmegaConf.create({"evaluation": {"enabled": False},
                                   "workspace_dir": str(workspace), "dataset_dir": "/data",
                                   "exp_id": "fixture"}), logs / "config.yaml")
    best = write_csv(workspace / "best_submission/submission.csv", "id,prediction\na,0.1\nb,0.2\n")
    return logs, workspace, best


@pytest.mark.parametrize("tied", [False, True])
def test_latest_valid_ensemble_uses_timestamp_then_path_not_size(saved_run, tied):
    logs, workspace, best = saved_run
    original = best.read_bytes()
    write_csv(workspace / "ensembles_csv/top6.csv", "id,prediction\na,0.6\nb,0.7\n", 1)
    chosen = write_csv(workspace / "ensembles_csv/top2.csv", "id,prediction\na,0.8\nb,0.9\n", 3)
    write_csv(workspace / "ensembles_csv/top1.csv", "id,prediction\na,0.4\nb,0.5\n", 3 if tied else 2)
    record = finalize_legacy_submission(logs, workspace)
    assert record["submission"] == str(chosen)
    assert record["selection"] == "latest_structurally_valid_ensemble"
    assert record == json.loads((logs / "final_submission.json").read_text())
    assert best.read_bytes() == original


def test_selection_stops_after_first_match(saved_run, monkeypatch):
    logs, workspace, best = saved_run
    write_csv(workspace / "ensembles_csv/older.csv", "id,prediction\na,0.6\nb,0.7\n", 1)
    chosen = write_csv(workspace / "ensembles_csv/newer.csv", "id,prediction\na,0.8\nb,0.9\n", 2)
    checked = []
    signature = grade_submission._csv_signature

    def check(path):
        checked.append(path)
        return signature(path)

    monkeypatch.setattr(grade_submission, "_csv_signature", check)
    assert finalize_legacy_submission(logs, workspace)["submission"] == str(chosen)
    assert checked == [best, chosen]


@pytest.mark.parametrize("bad_csv", [
    b"id,wrong\na,0.9\nb,0.8\n",              # changed header
    b"prediction,id\n0.9,a\n0.8,b\n",        # reordered columns
    b"id,prediction\na,0.9\n",                # missing row
    b"id,prediction\nb,0.8\na,0.9\n",        # reordered IDs
    b"id,prediction\na,0.9,extra\nb,0.8\n",  # wrong row width
    b"id,id\na,0.9\nb,0.8\n",               # duplicate headers
    b"id,prediction\na,\xff\nb,0.8\n",      # decoding failure
    b"",                                      # incomplete output
])
def test_malformed_latest_ensemble_does_not_hide_older_valid_one(saved_run, bad_csv):
    logs, workspace, _ = saved_run
    valid = write_csv(workspace / "ensembles_csv/older.csv", "id,prediction\na,0.6\nb,0.7\n", 1)
    invalid = valid.with_name("newer.csv")
    invalid.write_bytes(bad_csv)
    os.utime(invalid, ns=(2, 2))
    record = finalize_legacy_submission(logs, workspace)
    assert record["submission"] == str(valid)


@pytest.mark.parametrize("case", ["no_ensemble", "invalid_ensemble", "missing_best", "broken_best", "missing_all"])
def test_historical_fallback_including_unparseable_reference(saved_run, case):
    logs, workspace, best = saved_run
    if case not in {"no_ensemble", "missing_all"}:
        write_csv(workspace / "ensembles_csv/top2.csv", "id,prediction\na,0.8\nb,0.9\n")
    if case in {"missing_best", "missing_all"}:
        best.unlink()
    elif case == "broken_best":
        best.write_bytes(b"\xff")
    elif case == "invalid_ensemble":
        (workspace / "ensembles_csv/top2.csv").write_text("id,prediction\nwrong,1\n")
    record = finalize_legacy_submission(logs, workspace)
    assert record["submission"] == (str(best) if best.exists() else None)
    assert record["selection"] == ("best_submission_fallback" if best.exists() else "missing")


def test_bom_multiline_unicode_and_large_fields_preserve_csv_semantics(saved_run):
    logs, workspace, best = saved_run
    candidate = workspace / "ensembles_csv/text.csv"
    candidate.parent.mkdir()
    for path, prediction in [(best, "原文本"), (candidate, "新文本" + "x" * 150_000)]:
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerows([["id", "after"], ["重复,\n标识", prediction], ["重复,\n标识", "."]])
    record = finalize_legacy_submission(logs, workspace)
    assert record["submission"] == str(candidate)


def test_historical_first_column_check_also_applies_without_id_column(saved_run):
    logs, workspace, best = saved_run
    best.write_text("prediction,comment\n0.1,same\n0.2,same\n")
    write_csv(workspace / "ensembles_csv/top2.csv", "prediction,comment\n0.8,same\n0.9,same\n")
    assert finalize_legacy_submission(logs, workspace)["submission"] == str(best)


@pytest.fixture
def official_grader(monkeypatch):
    calls = []
    registry = SimpleNamespace()
    registry.set_data_dir = lambda path: calls.append(("data", path)) or registry
    registry.get_competition = lambda task: calls.append(("task", task)) or "competition"

    def grade_csv(path, competition):
        calls.append(("grade", path, competition))
        path.read_text(encoding="utf-8")
        return SimpleNamespace(to_dict=lambda: {"valid_submission": False, "gold_medal": False})

    for name, attributes in [("mlebench", {}), ("mlebench.grade", {"grade_csv": grade_csv}),
                             ("mlebench.registry", {"registry": registry})]:
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    return calls


def test_grader_uses_recorded_choice_once_even_if_invalid_or_new_file_arrives(saved_run, official_grader):
    logs, workspace, best = saved_run
    chosen = write_csv(workspace / "ensembles_csv/top2.csv", "id,prediction\na,0.8\nb,0.9\n", 1)
    finalize_legacy_submission(logs, workspace)
    write_csv(workspace / "ensembles_csv/later.csv", "id,prediction\na,0.3\nb,0.4\n", 2)
    result = grade_run(logs, data_dir="/grading-data")
    assert result["submission"] == str(chosen)
    assert result["report"]["valid_submission"] is False
    assert official_grader == [("data", Path("/grading-data")), ("task", "fixture"),
                               ("grade", chosen, "competition")]


@pytest.mark.parametrize("missing", [False, True])
def test_grader_reads_recorded_file_without_hash_gate_or_reselection(saved_run, official_grader, missing):
    logs, workspace, best = saved_run
    finalize_legacy_submission(logs, workspace)
    write_csv(workspace / "ensembles_csv/later.csv", "id,prediction\na,0.3\nb,0.4\n", 2)
    if missing:
        best.unlink()
        with pytest.raises(FileNotFoundError):
            grade_run(logs)
    else:
        best.write_text("id,prediction\na,0.8\nb,0.9\n")
        assert grade_run(logs)["submission"] == str(best)
    assert [call for call in official_grader if call[0] == "grade"] == [("grade", best, "competition")]


@pytest.mark.parametrize("fixed", [False, True])
def test_grading_cli_dispatches_from_saved_evaluation_mode(saved_run, official_grader, monkeypatch, capsys, fixed):
    logs, workspace, best = saved_run
    ensemble = write_csv(workspace / "ensembles_csv/top2.csv", "id,prediction\na,0.8\nb,0.9\n")
    if fixed:
        cfg = OmegaConf.load(logs / "config.yaml")
        cfg.evaluation.enabled = True
        OmegaConf.save(cfg, logs / "config.yaml")
        (logs / "final_submission.json").write_text(json.dumps({
            "selection": "best_fixed_evaluated_node", "submission": str(best),
        }))
    monkeypatch.setattr(sys, "argv", ["grade_submission", "--log-dir", str(logs)])
    runpy.run_path(str(Path(__file__).parents[1] / "utils/grade_submission.py"), run_name="__main__")
    result = json.loads(capsys.readouterr().out)
    assert result["submission"] == str(best if fixed else ensemble)
    assert official_grader[-1] == ("grade", best if fixed else ensemble, "competition")


def test_missing_submission_is_ungraded(saved_run, official_grader):
    logs, workspace, best = saved_run
    best.unlink()
    assert grade_run(logs) == {"submission": None, "selection": "missing",
                               "report": None}
    assert official_grader == []
