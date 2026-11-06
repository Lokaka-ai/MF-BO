"""Regression for the unresolved Git LFS leaderboard in Mila job 11417261."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "mf_campaign_leaderboard", ROOT / "tasks/mlebench_lite/mila_multifidelity.py"
)
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


class LeaderboardChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.board = self.root / "mle-bench/mlebench/competitions/task/leaderboard.csv"
        self.board.parent.mkdir(parents=True)

    def verify(self):
        with patch.object(campaign, "ROOT", self.root):
            return campaign.verify_leaderboards(["task"])

    def test_lfs_pointer_fails_before_gpu_submission(self):
        self.board.write_text(
            "version https://git-lfs.github.com/spec/v1\n"
            "oid sha256:cdb280161eca63c9ab1f937dcfb715b710fd222afe4c65f13a763d64e4c48318\n"
            "size 63555\n"
        )
        with self.assertRaisesRegex(RuntimeError, "no score column.*Git LFS"):
            self.verify()

    def test_hydrated_leaderboard_is_recorded_and_must_have_scores(self):
        self.board.write_text("rank,score\n1,0.9\n2,0.8\n")
        self.assertEqual(self.verify(), {"task": {"sha256": campaign.sha(self.board), "rows": 2}})
        self.board.write_text("rank,score\n1,nan\n")
        with self.assertRaisesRegex(RuntimeError, "Nonfinite leaderboard score"):
            self.verify()

    def test_prepared_campaign_rejects_changed_leaderboard(self):
        self.board.write_text("rank,score\n1,0.9\n2,0.8\n")
        settings = {"source": {"source": "hash"}, "spec": {"tasks": ["task"]}}
        campaign.save(self.root / "settings.json", settings)
        campaign.save(self.root / "ready.json", {
            "settings_sha256": campaign.sha(self.root / "settings.json"),
            "leaderboards": self.verify(),
        })
        with patch.object(campaign, "ROOT", self.root), \
             patch.object(campaign, "source_hashes", return_value=settings["source"]):
            self.assertEqual(campaign.ready(self.root), settings)
            self.board.write_text("rank,score\n1,0.8\n2,0.9\n")
            with self.assertRaisesRegex(RuntimeError, "leaderboards changed"):
                campaign.ready(self.root)


if __name__ == "__main__":
    unittest.main()
