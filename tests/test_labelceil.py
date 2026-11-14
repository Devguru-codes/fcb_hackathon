import unittest

import numpy as np

from tempolens.studies.labelceil import compare, event_rows
from tempolens.match import A, D, H, X
from tempolens.opendata import labels_from_events


class EventRowTests(unittest.TestCase):
    RAW = [(10, "KickOff", H, {"KickOff", "Play"}), (20, "Play", H, {"Play", "Pass"}),
           (40, "OtherBallAction", A, {"OtherBallAction"}),
           (100, "ThrowIn", A, {"ThrowIn", "Play"}),                  # out after frame 40
           (120, "TacklingGame", H, {"TacklingGame"}),                # changes nothing
           (150, "Foul", -1, {"Foul"}), (170, "FreeKick", H, {"FreeKick", "Play"}),
           (200, "ShotAtGoal", H, {"ShotAtGoal", "ShotWide"}),
           (260, "GoalKick", A, {"GoalKick", "Play"}), (280, "Offside", A, {"Offside"}),
           (300, "FinalWhistle", -1, {"FinalWhistle"})]

    def test_mapping_and_inferred_ball_outs(self):
        rows = event_rows(list(reversed(self.RAW)))          # input order does not matter
        kinds = list(zip(rows.type, rows.subtype, rows.start))
        self.assertEqual(kinds[0], ("SET PIECE", "KICKOFF", 10))   # no ball-out before kickoff
        self.assertIn(("BALL OUT", "INFERRED", 40), kinds)        # before the throw-in
        self.assertIn(("BALL OUT", "INFERRED", 150), kinds)       # before the free kick
        self.assertIn(("BALL OUT", "INFERRED", 200), kinds)       # before the goal kick
        self.assertNotIn(120, list(rows.start))
        self.assertEqual(list(rows.order), list(range(len(rows))))
        self.assertTrue((rows.team[rows.type == "BALL OUT"] == -1).all())

    def test_labels_through_the_metrica_rules(self):
        period = np.zeros(320, dtype=np.int8)
        period[5:310] = 1
        labels = labels_from_events(event_rows(self.RAW), period, 320)
        self.assertTrue((labels[5:10] == D).all())          # before the kickoff
        self.assertTrue((labels[10:40] == H).all())
        self.assertTrue((labels[40:100] == D).all())         # dead from the last event
        self.assertTrue((labels[100:150] == A).all())        # the tackle changes nothing
        self.assertTrue((labels[150:170] == D).all())        # foul
        self.assertTrue((labels[170:200] == H).all())
        self.assertTrue((labels[200:260] == D).all())        # shot wide
        self.assertTrue((labels[260:280] == A).all())
        self.assertTrue((labels[280:310] == D).all())        # offside, then the whistle
        self.assertTrue((labels[310:] == X).all())

    def test_compare_on_graded_frames_only(self):
        period = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 0])
        flag = np.array([X, H, H, H, H, D, D, A, A, X])
        events = np.array([H, H, H, D, D, D, D, A, X, A])     # frame 8 X: not graded
        r = compare(flag, events, period)
        self.assertAlmostEqual(r["agree"], 5 / 7)
        self.assertAlmostEqual(r["alive"], 5 / 7)
        same = compare(flag, flag, period)
        self.assertEqual((same["agree"], same["alive"]), (1.0, 1.0))
        self.assertAlmostEqual(same["spellF1"], 1.0)


if __name__ == "__main__":
    unittest.main()
