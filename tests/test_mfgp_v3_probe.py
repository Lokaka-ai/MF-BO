import json
import sys
import pytest
from engine.mf_gp.v3.probe import run_probe


def run(tmp_path, code, timeout=2):
    data = tmp_path / "train"
    data.mkdir()
    (data / "sample").write_text("training_only")
    return run_probe(code, python=sys.executable, work=tmp_path/"probe",
                     training_view=data, timeout=timeout)


def test_syntax_failure_does_not_start_process(tmp_path):
    r = run(tmp_path, "=======")
    assert r["state"] == "syntax_error"
    assert not (tmp_path/"probe/stdout.log").exists()


def test_actual_probe_success_is_not_a_quality_observation(tmp_path):
    code = """import os,json
from pathlib import Path
assert Path('input/sample').read_text() == 'training_only'
assert os.environ['BO_MF_PROBE'] == '1'
Path(os.environ['BO_MF_PROBE_REPORT']).write_text(json.dumps(dict(schema=1,
steps=3,seconds_per_step=.1,planned_low_steps=20,planned_full_steps=100,
inference_seconds=2,validation_score=1.0)))
"""
    r = run(tmp_path, code)
    assert r["state"] == "completed"
    assert "validation_score" not in r["metadata"]
    assert r["elapsed_seconds"] > 0
    assert json.loads((tmp_path/"probe/receipt.json").read_text()) == r
    with pytest.raises(FileExistsError):
        run_probe(code, python=sys.executable, work=tmp_path/"probe", training_view=tmp_path/"train")


def test_timeout_is_unknown_timing_not_fast_completion(tmp_path):
    r = run(tmp_path, "import time; time.sleep(60)", .1)
    assert r["state"] == "timeout" and r["metadata"] is None


def test_runtime_error_retained(tmp_path):
    r = run(tmp_path, "raise ValueError('wrong device')")
    assert r["state"] == "runtime_error"
    assert 'wrong device' in (tmp_path/"probe/stderr.log").read_text()


def test_missing_contract_is_not_fabricated_success(tmp_path):
    assert run(tmp_path, "print('done')")["state"] == "unknown"
