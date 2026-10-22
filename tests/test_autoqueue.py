import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tempolens import autoqueue


LOG = """trained temporal_fg2 without dfl_J03WMX in 300s

temporal on sim protocol; decoder costs switch/alive>dead/dead>alive/change
mean 0.7612 (dfl 0.7540, metrica 0.7780)  costs (4.0, 8.0, 8.0, 0.5)
mean 0.7600 (dfl 0.7530, metrica 0.7770)  costs (4.0, 4.0, 8.0, 0.5)
"""


class AutoqueueTests(unittest.TestCase):
    def test_parse_takes_the_best_line(self):
        self.assertEqual(autoqueue.parse(LOG), {"all": 0.7612, "dfl": 0.754, "metrica": 0.778,
                                                "costs": "(4.0, 8.0, 8.0, 0.5)"})
        self.assertIsNone(autoqueue.parse("Traceback (most recent call last):"))

    def test_verdict_uses_the_noise_floor_and_baseline_references(self):
        base = {"id": "B", "status": "done", "result": {"all": 0.75, "dfl": 0.74}}
        win = {"id": "W", "status": "done", "baseline_from": "B",
               "result": {"all": 0.758, "dfl": 0.741}}
        tie = {"id": "T", "status": "done", "baseline": {"all": 0.75, "dfl": 0.74},
               "result": {"all": 0.752, "dfl": 0.745}}
        jobs = [base, win, tie]
        self.assertTrue(autoqueue.verdict(win, jobs).startswith("candidate"))
        self.assertTrue(autoqueue.verdict(tie, jobs).startswith("no gain"))
        self.assertEqual(autoqueue.verdict(base, jobs), "measured")

    def test_plan_table_is_replaced_between_the_markers_only(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = Path(folder) / "plan.md"
            plan.write_text("# Plan\n\nkeep me\n", encoding="utf-8")
            jobs = [{"id": "Q1", "title": "first", "args": [], "status": "pending"}]
            with patch.object(autoqueue, "PLAN", plan):
                autoqueue.write_plan(jobs)
                jobs[0].update(status="done", result={"all": 0.7, "dfl": 0.6, "metrica": 0.8,
                                                      "costs": "(4.0,)"})
                autoqueue.write_plan(jobs)
            text = plan.read_text(encoding="utf-8")
        self.assertIn("keep me", text)
        self.assertEqual(text.count(autoqueue.START), 1)
        self.assertIn("| ✅ | Q1 | first | 0.7000 |", text)


if __name__ == "__main__":
    unittest.main()
