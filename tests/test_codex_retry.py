import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from llm import generate
from llm import codex


@pytest.fixture
def codex_request(tmp_path, monkeypatch):
    cfg = SimpleNamespace(
        workspace_dir=tmp_path,
        agent=SimpleNamespace(code=SimpleNamespace(
            model="muse-spark-1.2-contributor",
            base_url="codex://opencode-go",
        )),
    )
    monkeypatch.setenv("MLEVOLVE_CODEX_BIN", "codex-test")
    sleep = Mock()
    monkeypatch.setattr(codex.time, "sleep", sleep)
    return cfg, sleep


@pytest.mark.parametrize("failures", [0, 1, 2])
def test_codex_generation_recovers_without_extra_attempts(codex_request, monkeypatch, failures):
    cfg, sleep = codex_request
    requests = []

    def run(command, **kwargs):
        requests.append(kwargs["input"])
        if len(requests) <= failures:
            return subprocess.CompletedProcess(command, 1, stdout="HTTP 503 unavailable")
        output = Path(command[command.index("--output-last-message") + 1])
        output.write_text("A plan.\n```python\nprint('ok')\n```", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="")

    monkeypatch.setattr(codex.subprocess, "run", run)

    assert generate("Write a solution", cfg) == "A plan.\n```python\nprint('ok')\n```"
    assert len(requests) == failures + 1
    assert len(set(requests)) == 1
    assert [call.args for call in sleep.call_args_list] == [(3,)] * failures


@pytest.mark.parametrize("failure", ["exit", "timeout", "missing_output"])
def test_codex_generation_stops_after_three_attempts(codex_request, monkeypatch, failure):
    cfg, sleep = codex_request

    def run(command, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return subprocess.CompletedProcess(
            command, 1 if failure == "exit" else 0, stdout="HTTP 503 unavailable"
        )

    transport = Mock(side_effect=run)
    monkeypatch.setattr(codex.subprocess, "run", transport)
    error = subprocess.TimeoutExpired if failure == "timeout" else RuntimeError

    # The public interface defaults to 20; Codex generation must cap this at 3.
    with pytest.raises(error):
        generate("Write a plan", cfg)

    assert transport.call_count == 3
    assert [call.args for call in sleep.call_args_list] == [(3,), (3,)]


def test_codex_generation_respects_smaller_attempt_limit(codex_request, monkeypatch):
    cfg, sleep = codex_request
    transport = Mock(return_value=subprocess.CompletedProcess(
        [], 1, stdout="HTTP 503 unavailable"
    ))
    monkeypatch.setattr(codex.subprocess, "run", transport)

    with pytest.raises(RuntimeError, match="HTTP 503 unavailable"):
        generate("Write a solution", cfg, max_retries=1)

    assert transport.call_count == 1
    sleep.assert_not_called()
