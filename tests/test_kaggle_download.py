import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from tasks.mlebench_lite import prepare, kaggle_download


@pytest.fixture
def fixture(monkeypatch, tmp_path):
    task = "example-competition"
    upstream = tmp_path / "upstream"
    checksums = upstream / "mlebench/competitions" / task / "checksums.yaml"
    checksums.parent.mkdir(parents=True)
    payload = b"official archive bytes"
    checksums.write_text("zip: " + hashlib.md5(payload).hexdigest())
    python = tmp_path / ".venv-kaggle/bin/python"
    python.parent.mkdir(parents=True)
    python.touch()
    token = tmp_path / "credential"
    token.write_text("KGAT_test-secret")
    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    monkeypatch.setattr(prepare, "UPSTREAM", upstream)
    return task, tmp_path / "data", token, payload


def transfer(monkeypatch, fixture, payload=None, failed=False):
    task, data, token, expected = fixture
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        assert token.read_text() not in str(argv)
        assert argv[0].endswith(".venv-kaggle/bin/python")
        if failed:
            return SimpleNamespace(returncode=1, stdout="KAGGLE_DOWNLOAD_ERROR: HTTP 403", stderr="secret URL")
        destination = Path(argv[argv.index("--destination") + 1])
        (destination / (task + ".zip")).write_bytes(expected if payload is None else payload)
        return SimpleNamespace(returncode=0, stdout="KAGGLE_DOWNLOAD_OK", stderr="")
    monkeypatch.setattr(prepare.subprocess, "run", run)
    return calls


def test_verified_archive_and_cache(monkeypatch, fixture):
    calls = transfer(monkeypatch, fixture)
    task, data, token, payload = fixture
    target = prepare.download(task, data, token)
    assert target.read_bytes() == payload
    assert prepare.download(task, data, token) == target
    assert len(calls) == 1
    assert not (target.parent / ".download.guard").exists()


def test_corrupt_download_never_published(monkeypatch, fixture):
    transfer(monkeypatch, fixture, payload=b"wrong archive")
    task, data, token, _ = fixture
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        prepare.download(task, data, token)
    assert not (data / task / (task + ".zip")).exists()
    assert (data / task / ".kaggle-download" / (task + ".zip")).is_file()


def test_existing_corruption_preserved(monkeypatch, fixture):
    calls = transfer(monkeypatch, fixture)
    task, data, token, _ = fixture
    target = data / task / (task + ".zip")
    target.parent.mkdir(parents=True)
    target.write_bytes(b"bad cached data")
    with pytest.raises(RuntimeError, match="Existing archive checksum mismatch"):
        prepare.download(task, data, token)
    assert target.read_bytes() == b"bad cached data"
    assert not calls


def test_permission_failure_preserved_without_secrets(monkeypatch, fixture):
    transfer(monkeypatch, fixture, failed=True)
    task, data, token, _ = fixture
    with pytest.raises(RuntimeError, match="HTTP 403") as error:
        prepare.download(task, data, token)
    assert "secret" not in str(error.value)
    assert not (data / task / ".download.guard").exists()


def test_helper_version_drift_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(kaggle_download, "version", lambda package: "unexpected")
    with pytest.raises(RuntimeError, match="version mismatch"):
        kaggle_download.download("task", tmp_path, tmp_path / "absent-token")


def test_helper_uses_explicit_token_and_preserves_zip(monkeypatch, tmp_path):
    import sys
    import types
    monkeypatch.setattr(kaggle_download, "version", lambda package: {"kaggle": "2.2.5", "kagglesdk": "0.1.38"}[package])
    token = tmp_path / "token"
    token.write_text("KGAT_test-secret")
    monkeypatch.setenv("KAGGLE_API_TOKEN", "wrong-inherited-token")
    calls = []
    class API:
        def authenticate(self):
            assert kaggle_download.os.environ["KAGGLE_API_TOKEN"] == "KGAT_test-secret"
        def competition_download_files(self, *args, **kwargs):
            calls.append((args, kwargs))
    for name in ("kaggle", "kaggle.api", "kaggle.api.kaggle_api_extended"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["kaggle.api.kaggle_api_extended"].KaggleApi = API
    kaggle_download.download("task", tmp_path, token)
    assert calls == [(("task",), {"path": str(tmp_path), "quiet": True, "unzip": False})]
