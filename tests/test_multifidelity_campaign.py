"""Offline launch checks; also runnable without ML dependencies via unittest."""
import importlib.util
import json
from pathlib import Path
import shlex
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("mf_campaign", ROOT / "tasks/mlebench_lite/mila_multifidelity.py")
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="mf test ")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.settings = {"spec": campaign.read(campaign.SPEC), "data_dir": "/data with spaces"}

    def test_commands_match_approved_protocol(self):
        manifest = self.settings["spec"]
        self.assertEqual(len(manifest["tasks"]), 14)
        self.assertEqual(manifest["modes"], ["carry_over", "reset"])
        excluded = {"denoising-dirty-documents", "detecting-insults-in-social-commentary",
                    "dogs-vs-cats-redux-kernels-edition", "histopathologic-cancer-detection",
                    "plant-pathology-2020-fgvc7", "spooky-author-identification",
                    "tabular-playground-series-dec-2021", "the-icml-2013-whale-challenge-right-whale-redux"}
        lite = set((ROOT / "config/mlebench_lite22.txt").read_text().split())
        self.assertEqual(set(manifest["tasks"]), lite - excluded)
        for task in manifest["tasks"]:
            commands = [campaign.search_command(self.settings, self.path, task, mode)
                        for mode in manifest["modes"]]
            for command in commands:
                for setting in ("agent.time_limit=43200", "agent.steps=500", "agent.initial_drafts=3",
                                "agent.search.parallel_search_num=3", "agent.seed=42", "exec.timeout=32400",
                                "evaluation.enabled=true", "audited_bo=redesign",
                                "experiment_preset=acq7_aq05_gp4_level15_rating_kg"):
                    self.assertIn(setting, command)
            # Apart from mode and output naming, both arms have identical settings.
            normalized = [[arg.replace("carry_over", "MODE").replace("reset", "MODE") for arg in c]
                          for c in commands]
            self.assertEqual(*normalized)

    def test_submit_all_is_idempotent_and_quotes_paths(self):
        args = SimpleNamespace(campaign=self.path, partition="long", tasks=None)
        campaign.save(self.path / "settings.json", self.settings)
        campaign.save(self.path / "gpu-ready.json", {"settings_sha256": campaign.sha(self.path / "settings.json")})
        with patch.object(campaign, "ready", return_value=self.settings), \
             patch.object(campaign, "run", return_value=SimpleNamespace(stdout="12345\n")) as run:
            campaign.submit(args)
            self.assertEqual(run.call_count, 28)

            for call in run.call_args_list:
                argv = call.args[0]
                self.assertIn("--gres=gpu:a100l:1", argv)
                self.assertIn("--time=13:00:00", argv)
                wrapper = shlex.split(next(x[7:] for x in argv if x.startswith("--wrap=")))
                self.assertEqual(wrapper[0], "exec")
                self.assertEqual(wrapper[-1], str(self.path))
            campaign.submit(args)
            self.assertEqual(run.call_count, 28)

    def test_full_submission_requires_short_gpu_preflight(self):
        args = SimpleNamespace(campaign=self.path, partition="long", tasks=None)
        with patch.object(campaign, "ready", return_value=self.settings), patch.object(campaign, "run") as run:
            with self.assertRaises(FileNotFoundError):
                campaign.submit(args)
        run.assert_not_called()

    def test_unknown_submission_is_not_retried(self):
        campaign.save(self.path / "jobs.json", {"x": {"state": "submitting"}})
        with self.assertRaisesRegex(RuntimeError, "Unresolved"), patch.object(campaign, "run") as run:
            campaign.sbatch(self.path, "x", ["sbatch"])
        run.assert_not_called()

    def test_missing_results_are_preserved(self):
        campaign.collect(SimpleNamespace(campaign=self.path))
        import csv
        with (self.path / "summary.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 28)
        self.assertTrue(all(row["state"] == "no_result" for row in rows))

    def test_changed_packaged_source_fails(self):
        (self.path / "file.py").write_text("original")
        campaign.save(self.path / "SOURCE_SHA256.json", {"file.py": campaign.sha(self.path / "file.py")})
        with patch.object(campaign, "ROOT", self.path):
            campaign.source_hashes()
            (self.path / "file.py").write_text("changed")
            with self.assertRaisesRegex(RuntimeError, "source changed"):
                campaign.source_hashes()

    def test_baseline_preparation_reuses_requested_frozen_protocol(self):
        from tasks.mlebench_lite.mfgp_protocols import register_source

        task = "aerial-cactus-identification"
        source = self.path / "source"
        campaign.save(source / task / "protocol.json", {
            "task_id": task, "protocol_id": "frozen", "metric": {"name": "roc_auc"},
            "split": {"seed": 42},
        })
        destination = self.path / "prepared"
        settings = {
            "source": {"source": "hash"}, "data_dir": "/data", "token_file": "/token",
            "spec": {"tasks": {task: {}}, "modes": ["carry_over", "reset"],
                     "seed": 42, "model": "muse", "provider": "codex://opencode-go"},
            "protocol_source": register_source(source, [task]),
        }
        campaign.save(destination / "settings.json", settings)
        with patch.object(campaign, "require_allocation"), \
             patch.object(campaign, "source_hashes", return_value=settings["source"]), \
             patch.object(campaign, "verify_codex"), \
             patch.object(campaign, "verify_leaderboards", return_value={task: {"sha256": "x", "rows": 2}}), \
             patch.object(campaign, "environment", return_value={}), \
             patch.object(campaign.shutil, "which", return_value="/usr/bin/true"), \
             patch.object(campaign, "run"):
            campaign.prepare(SimpleNamespace(campaign=destination))
        self.assertEqual((destination / "protocols" / task).resolve(), (source / task).resolve())
        self.assertEqual(campaign.read(destination / "ready.json")["protocols"][task]["protocol_id"], "frozen")


if __name__ == "__main__":
    unittest.main()


# Freeze pilot scope before downloads, protocol registration or Slurm submission.
import pytest


@pytest.mark.parametrize("manifest, modes", [
    ("mlebench_mf14.json", ["carry_over", "reset"]),
    ("mlebench_mfgp14.json", ["mf_gp"]),
])
def test_one_task_preparation_scope_and_submission(tmp_path, monkeypatch, manifest, modes):
    task = "aerial-cactus-identification"
    spec_path = ROOT / "config" / manifest
    original = campaign.read(spec_path)
    monkeypatch.setattr(campaign, "SPEC", spec_path)
    monkeypatch.setattr(campaign, "source_hashes", lambda: {"source": "hash"})
    key, token = tmp_path / "key", tmp_path / "token"
    key.touch()
    token.touch()
    source = tmp_path / "baseline/protocols"
    campaign.save(source / task / "protocol.json", {"task_id": task, "protocol_id": "frozen"})
    directory = tmp_path / "campaign"
    args = SimpleNamespace(campaign=directory, partition="main-cpu", key_file=key,
        token_file=token, data_dir=tmp_path / "data", tasks=[task],
        protocol_source=source if modes == ["mf_gp"] else None)
    with patch.object(campaign, "run", return_value=SimpleNamespace(stdout="12345\n")):
        campaign.submit_preparation(args)
    settings = campaign.read(directory / "settings.json")
    assert settings["spec"]["tasks"] == {task: original["tasks"][task]}
    assert {k: v for k, v in settings["spec"].items() if k != "tasks"} == {k: v for k, v in original.items() if k != "tasks"}
    if args.protocol_source:
        assert settings["protocol_source"]["protocol_ids"] == {task: "frozen"}
    monkeypatch.setattr(campaign, "ready", lambda _: settings)
    campaign.save(directory / "gpu-ready.json", {"settings_sha256": campaign.sha(directory / "settings.json")})
    with patch.object(campaign, "run", return_value=SimpleNamespace(stdout="12346\n")) as run:
        campaign.submit(SimpleNamespace(campaign=directory, partition="long", tasks=None))
        assert run.call_count == len(modes)
        for call in run.call_args_list:
            assert f"--task {task}" in call.args[0][-1]
        with pytest.raises(ValueError, match="prepared campaign"):
            campaign.submit(SimpleNamespace(campaign=directory, partition="long", tasks=["leaf-classification"]))
    campaign.collect(SimpleNamespace(campaign=directory))
    import csv
    with (directory / "summary.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(modes)
    assert {row["task"] for row in rows} == {task}
    # A later expansion cannot silently alter the prepared pilot scope.
    args.tasks = [task, "leaf-classification"]
    args.protocol_source = None
    with pytest.raises(ValueError, match="settings already differ"):
        campaign.submit_preparation(args)


@pytest.mark.parametrize("tasks", [["unknown"], ["aerial-cactus-identification"] * 2, []])
def test_invalid_preparation_scope_never_submits(tmp_path, monkeypatch, tasks):
    credential = tmp_path / "key"
    credential.touch()
    args = SimpleNamespace(campaign=tmp_path / "campaign", key_file=credential,
        token_file=credential, tasks=tasks)
    with patch.object(campaign, "sbatch") as submit:
        with pytest.raises(ValueError, match="unique tasks"):
            campaign.submit_preparation(args)
        submit.assert_not_called()
