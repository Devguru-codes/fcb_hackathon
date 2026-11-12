"""Starter tooling checks, on synthetic files only."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter"))
import load_atd  # noqa: E402
import load_smart_tagging as lst  # noqa: E402


class SubmissionBytesTests(unittest.TestCase):
    def test_starter_writers_use_lf_on_every_os(self):
        import baseline
        import make_dummy_submission
        with tempfile.TemporaryDirectory() as folder:
            a = Path(baseline.write_submission(str(Path(folder) / "a.csv"), ["H", "A", "D"]))
            b = Path(make_dummy_submission.write_constant("b", "D", 3, folder))
            for path in (a, b):
                data = path.read_bytes()
                self.assertNotIn(b"\r", data, path.name)
                self.assertTrue(data.startswith(b"frame,label\n"), path.name)


class KickoffTagTests(unittest.TestCase):
    def test_a_tag_ending_on_the_kickoff_frame_does_not_contain_it(self):
        truth = lst.KICKOFF_VIDEO_FRAMES["1st Half"]

        def tag(start, end):
            return {"code": lst.KICKOFF_CODE, "start_frame": start, "end_frame": end,
                    "tags": {"Type": lst.KICKOFF_TYPE, "Half": "1st Half"}}
        rows = lst.check_time_base([tag(truth - 100, truth)])["halves"]       # end is exclusive
        self.assertFalse(rows[0]["contains_kickoff"])
        rows = lst.check_time_base([tag(truth - 100, truth + 1)])["halves"]
        self.assertTrue(rows[0]["contains_kickoff"])


class AlignmentRawTests(unittest.TestCase):
    def check(self, text):
        with tempfile.TemporaryDirectory() as folder:
            raw = Path(folder) / "atd.txt"
            raw.write_text(text)
            with patch.object(load_atd, "require", return_value=[str(raw)]), \
                    patch.object(load_atd, "source_frames", return_value=4):
                return load_atd.check_alignment_raw()

    def test_a_duplicate_and_a_gap_are_not_contiguous(self):
        report = self.check("1:p:b\n2:p:b\n2:p:b\n4:p:b\n")     # count 4 = range 1..4
        self.assertFalse(report["contiguous"])
        self.assertFalse(report["agrees"])

    def test_a_trailing_empty_line_is_skipped(self):
        report = self.check("1:p:b\n2:p:b\n3:p:b\n4:p:b\n\n")
        self.assertTrue(report["contiguous"])
        self.assertTrue(report["agrees"])
        self.assertEqual(report["n_records"], 4)


if __name__ == "__main__":
    unittest.main()
