import contextlib
import io
import shutil
import tempfile
import unittest
from pathlib import Path

from tempolens import realcheck


class RealCheckTests(unittest.TestCase):
    def test_public_window_scores_the_truth_as_perfect(self):
        with tempfile.TemporaryDirectory() as folder:
            copy = Path(folder) / "truth.csv"
            shutil.copy(realcheck.PUBLIC_GT, copy)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                realcheck.public([copy])
        self.assertIn("spellF1 1.0000 macroF1 1.0000", out.getvalue())
        self.assertIn("D recall 1.000 precision 1.000", out.getvalue())


if __name__ == "__main__":
    unittest.main()
