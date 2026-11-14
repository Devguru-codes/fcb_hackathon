import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Every module tempolens.predict imports, eagerly or inside its functions.
PREDICT_PATH = ("tempolens.predict", "tempolens.temporal", "tempolens.match", "tempolens.features",
                "tempolens.decoder", "tempolens.tagprior", "tempolens.tags")


class PredictImportTests(unittest.TestCase):
    def test_submission_path_does_not_load_the_experiment_harness(self):
        # A fresh interpreter: this test process has already imported the studies.
        code = ("import importlib, sys\n"
                "for m in {!r}: importlib.import_module(m)\n"
                "print(sorted(m for m in sys.modules if m.startswith('tempolens')))\n"
                "print('pandas' in sys.modules)\n").format(PREDICT_PATH)
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout.splitlines()
        self.assertEqual(eval(out[0]), sorted(("tempolens",) + PREDICT_PATH))
        self.assertEqual(out[1], "False")


if __name__ == "__main__":
    unittest.main()
