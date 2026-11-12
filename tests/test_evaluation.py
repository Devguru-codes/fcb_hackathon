"""The local scorer (evaluation/run_evaluation.py) end to end on synthetic files only."""

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
import run_evaluation  # noqa: E402


def write_labels(path, labels):
    pd.DataFrame({"frame": range(len(labels)), "label": labels}).to_csv(path, index=False)


class RunEvaluationTests(unittest.TestCase):
    def run_scorer(self, folder, *extra):
        gt = Path(folder) / "gt.csv"
        out = io.StringIO()
        argv = ["run_evaluation.py", "--gt", str(gt), "--results", str(Path(folder) / "results"),
                "--out", str(folder), "--no-detail", *extra]
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(out):
            code = run_evaluation.main()
        return code, out.getvalue()

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        folder = Path(self.folder.name)
        labels = ["X"] * 100 + ["H"] * 300 + ["A"] * 300 + ["D"] * 100 + ["H"] * 200
        write_labels(folder / "gt.csv", labels)
        (folder / "results").mkdir()
        write_labels(folder / "results" / "good.csv", labels)
        write_labels(folder / "results" / "bad.csv", ["Y"] * len(labels))   # a skip reason with commas

    def tearDown(self):
        self.folder.cleanup()

    def test_a_skipped_submission_leaves_a_readable_scoreboard(self):
        code, _ = self.run_scorer(self.folder.name)
        self.assertEqual(code, 1)                                   # something was skipped
        board = pd.read_csv(Path(self.folder.name) / "scoreboard.csv")
        self.assertEqual(sorted(board["status"]), ["scored", "skipped"])
        self.assertTrue((Path(self.folder.name) / "reports" / "good.json").exists())

    def test_a_ground_truth_on_another_drive_still_writes_the_scoreboard(self):
        def other_drive(path, start=None):
            raise ValueError("path is on mount 'D:', start on mount 'C:'")
        with patch.object(run_evaluation.os.path, "relpath", side_effect=other_drive):
            code, printed = self.run_scorer(self.folder.name)
        self.assertEqual(code, 1)
        self.assertTrue((Path(self.folder.name) / "scoreboard.csv").exists())
        self.assertIn("gt.csv", printed)

    def test_a_skipped_submission_loses_its_old_reports(self):
        reports = Path(self.folder.name) / "reports"
        reports.mkdir()
        (reports / "bad.json").write_text("{}")                     # from when bad.csv scored
        (reports / "bad_transitions.csv").write_text("x")
        self.run_scorer(self.folder.name)
        self.assertEqual(sorted(p.name for p in reports.iterdir()),
                         ["good.json", "good_transitions.csv"])


if __name__ == "__main__":
    unittest.main()
