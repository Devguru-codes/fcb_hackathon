import ast
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

    def test_predict_imports_only_the_listed_modules(self):
        # predict.py imports inside its functions, which the check above never runs.
        tree = ast.parse((ROOT / "tempolens" / "predict.py").read_text(encoding="utf-8"))
        found = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 1:
                found.update(["tempolens." + node.module] if node.module
                             else ["tempolens." + a.name for a in node.names])
        self.assertLessEqual(found, set(PREDICT_PATH))


if __name__ == "__main__":
    unittest.main()
