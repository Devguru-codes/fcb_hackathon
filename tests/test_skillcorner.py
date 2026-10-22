import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tempolens.match import A, D, H, X
from tempolens.opendata import skillcorner_game


def write_match(folder, n=20):
    """A 2-second, 10 fps toy match: home owns frames 2-7, a loose ball at 8, away from 9, a
    replay (no tracking) at 14-15, dead ball outside the phases."""
    meta = {"pitch_length": 100, "pitch_width": 60,
            "home_team": {"id": 1, "name": "Home"}, "away_team": {"id": 2, "name": "Away"},
            "players": [{"id": 10, "team_id": 1}, {"id": 20, "team_id": 2},
                        {"id": 99, "team_id": 7}],                 # a referee-like extra
            "match_periods": [{"period": 1, "start_frame": 0, "end_frame": n - 1}]}
    (folder / "5_match.json").write_text(json.dumps(meta), encoding="utf-8")
    rows = []
    for f in range(n):
        tracked = f not in (14, 15)
        group = "home team" if 2 <= f <= 7 else "away team" if 9 <= f <= 12 else None
        rows.append({"frame": f, "period": 1,
                     "ball_data": {"x": float(f), "y": 0.0, "z": 0.0,
                                   "is_detected": tracked and f != 5},
                     "player_data": [{"x": 0.0, "y": 1.0, "player_id": 10, "is_detected": True},
                                     {"x": 5.0, "y": -1.0, "player_id": 20,
                                      "is_detected": False}] if tracked else [],
                     "possession": {"player_id": None, "group": group}})
    (folder / "5_tracking_extrapolated.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    (folder / "5_phases_of_play.csv").write_text(
        "frame_start,frame_end,team_in_possession_id\n2,9,1\n9,13,2\n", encoding="utf-8")


class SkillCornerTests(unittest.TestCase):
    def test_labels_coordinates_and_resampling(self):
        with tempfile.TemporaryDirectory() as folder:
            write_match(Path(folder))
            t = skillcorner_game(5, folder, fps=25)
        self.assertEqual(t.n_frames, 50)                    # 20 frames at 10 fps -> 25 fps
        self.assertEqual(t.slot_team.tolist(), [H, A])      # non-team people are dropped
        at = lambda f10: int(round(f10 * 2.5))              # 10 fps frame -> 25 fps frame
        self.assertEqual(t.labels[at(0)], D)                # before the first phase
        self.assertEqual(t.labels[at(4)], H)
        self.assertEqual(t.labels[at(8)], H)                # loose ball: the last owner keeps it
        self.assertEqual(t.labels[at(10)], A)
        self.assertEqual(t.labels[at(13)], D)               # after the last phase
        self.assertEqual(t.labels[at(14)], X)               # replay: no tracking, no label
        self.assertEqual(t.period[at(14)], 0)
        np.testing.assert_allclose(t.players[at(3), 0], [50.0, 31.0])    # centre -> 0..L, 0..W
        self.assertTrue(np.isnan(t.ball[at(5), 0]))         # extrapolated ball is not a ball
        self.assertAlmostEqual(float(t.ball[1, 0]), 50.4, places=4)       # interpolated in time


if __name__ == "__main__":
    unittest.main()
