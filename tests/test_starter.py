"""Starter tooling checks, on synthetic files only."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter"))
import load_atd  # noqa: E402


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
