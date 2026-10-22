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


if __name__ == "__main__":
    unittest.main()
