"""Offline MF-GP campaign safety and baseline-protocol reuse."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from tasks.mlebench_lite import mila_multifidelity as campaign
from tasks.mlebench_lite.mfgp_protocols import register_source, attach_source

ROOT = Path(__file__).resolve().parents[1]


def test_pinned_protocol_reuse_rejects_changed_identity(tmp_path):
    source = tmp_path / "baseline/protocols"
    path = source / "task/protocol.json"
    campaign.save(path, {"task_id": "task", "protocol_id": "frozen"})
    settings = {"protocol_source": register_source(source, ["task"])}
    destination = tmp_path / "mf"
    attach_source(settings, destination, "task")
    assert (destination / "protocols/task").resolve() == path.parent
    attach_source(settings, destination, "task")
    campaign.save(path, {"task_id": "task", "protocol_id": "changed"})
    with pytest.raises(ValueError, match="identity changed"):
        attach_source(settings, destination, "task")


def test_submit_exactly_fourteen_mf_jobs_and_isolate_campaign(tmp_path, monkeypatch):
    spec_path = ROOT / "config/mlebench_mfgp14.json"
    settings = {"spec": campaign.read(spec_path), "data_dir": "/data"}
    monkeypatch.setattr(campaign, "SPEC", spec_path)
    monkeypatch.setattr(campaign, "SCRIPT", ROOT / "tasks/mlebench_lite/mila_mfgp.py")
    monkeypatch.setattr(campaign, "ready", lambda _: settings)
    campaign.save(tmp_path / "settings.json", settings)
    campaign.save(
        tmp_path / "gpu-ready.json",
        {"settings_sha256": campaign.sha(tmp_path / "settings.json")},
    )
    with patch.object(
        campaign, "run", return_value=SimpleNamespace(stdout="12345\n")
    ) as run:
        args = SimpleNamespace(campaign=tmp_path, partition="long", tasks=None)
        campaign.submit(args)
        assert run.call_count == 14
        for call in run.call_args_list:
            args = call.args[0]
            assert "--gres=gpu:a100l:1" in args
            wrap = next(a for a in args if a.startswith("--wrap"))
            assert "mila_mfgp.py" in wrap and "--mode mf_gp" in wrap
            assert "carry_over" not in wrap and "reset" not in wrap
        campaign.submit(
            SimpleNamespace(campaign=tmp_path, partition="long", tasks=None)
        )
        assert run.call_count == 14


def test_package_is_clean_source_only_reproducible_and_hash_checked(
    tmp_path, monkeypatch
):
    import subprocess, tarfile
    from tasks.mlebench_lite.package_mfgp import build

    root = tmp_path / "source"
    root.mkdir()

    def git(*args):
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)

    git("init")
    (root / "program.py").write_text('print("source")\n')
    (root / ".gitignore").write_text("credential\n")
    (root / "credential").write_text("not for packaging")
    git("add", "program.py", ".gitignore")
    git(
        "-c",
        "user.name=Local Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "fixture",
    )
    first = build(root, tmp_path / "out")
    assert build(root, tmp_path / "out") == first
    unpack = tmp_path / "unpacked"
    with tarfile.open(first["archive"]) as archive:
        assert not any(n.endswith("/credential") for n in archive.getnames())
        archive.extractall(unpack, filter="data")
    checkout = next(unpack.iterdir())
    monkeypatch.setattr(campaign, "ROOT", checkout)
    assert len(campaign.source_hashes()) == 2
    (checkout / "program.py").write_text("modified")
    with pytest.raises(RuntimeError, match="source changed"):
        campaign.source_hashes()
    (root / "program.py").write_text("uncommitted")
    with pytest.raises(RuntimeError, match="Commit source"):
        build(root, tmp_path / "out")
