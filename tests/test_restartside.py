import unittest

import numpy as np

from tempolens import restartside as rs
from tempolens.match import A, D, H, X, MatchTable

PITCH = (105.0, 68.0)


def _table(n, ball, home_x=20.0, away_x=80.0, period=None):
    players = np.zeros((n, 4, 2), dtype=np.float32)
    players[:, :2, 0], players[:, 2:, 0] = home_x, away_x
    return MatchTable("r", np.asarray(ball, dtype=np.float32), players,
                      np.array([H, H, A, A], dtype=np.int8),
                      np.ones(n, dtype=np.int8) if period is None else period)


class ZoneTests(unittest.TestCase):
    def test_corners_goal_areas_and_boundaries(self):
        self.assertEqual(rs.zone((1.0, 1.0), PITCH), ("corner", 0))
        self.assertEqual(rs.zone((105.0, 68.0 - 2.0), PITCH), ("corner", 1))     # exactly 2 m
        self.assertIsNone(rs.zone((0.0, 2.01), PITCH))                          # just outside
        self.assertEqual(rs.zone((5.5, 34.0 + 9.16), PITCH), ("goal kick", 0))  # area corner
        self.assertIsNone(rs.zone((5.51, 34.0), PITCH))
        self.assertEqual(rs.zone((101.0, 30.0), PITCH), ("goal kick", 1))
        self.assertIsNone(rs.zone((52.5, 34.0), PITCH))
        self.assertIsNone(rs.zone((np.nan, 1.0), PITCH))


class FindTests(unittest.TestCase):
    def ball(self, n=600):
        return np.full((n, 2), [52.5, 34.0])

    def test_still_then_moved_is_found_once(self):
        ball = self.ball()
        ball[100:140] = [1.0, 1.0]                     # still at the left corner
        ball[140:] = [1.0, 1.0] + np.arange(460)[:, None] * [0.2, 0.0]
        # the first frame whose ball is 2 m away a second later: 0.6 s before the kick here
        self.assertEqual(rs.find_restarts(_table(600, ball)), [(125, "corner", 0)])

    def test_moving_or_unseen_before_is_not_a_restart(self):
        ball = self.ball()
        ball[100:200, 1] = 34.0
        ball[100:200, 0] = 1.0 + 0.1 * np.arange(100)       # rolling through the goal area
        self.assertEqual(rs.find_restarts(_table(600, ball)), [])
        ball = self.ball()
        ball[:226] = np.nan                                  # unseen, then seen rolling
        ball[226:] = [2.0, 34.0] + np.arange(374)[:, None] * [0.3, 0.0]
        self.assertEqual(rs.find_restarts(_table(600, ball)), [])
        ball = self.ball()
        ball[:200] = np.nan                                  # seen still for 8 frames: enough
        ball[200:] = [2.0, 34.0]
        ball[226:] = [2.0, 34.0] + np.arange(374)[:, None] * [0.3, 0.0]
        self.assertEqual(rs.find_restarts(_table(600, ball)), [(208, "goal kick", 0)])
        ball[200:204] = np.nan                               # only 4 of 12 seen: not enough
        self.assertEqual(rs.find_restarts(_table(600, ball))[0][0], 210)

    def test_outside_the_halves_is_ignored(self):
        ball = self.ball()
        ball[100:140] = [1.0, 1.0]
        ball[140:] = [1.0, 1.0] + np.arange(460)[:, None] * [0.2, 0.0]
        period = np.ones(600, dtype=np.int8)
        period[:200] = 0
        self.assertEqual(rs.find_restarts(_table(600, ball, period=period)), [])


class SideTests(unittest.TestCase):
    def test_ends_and_implied_sides(self):
        period = np.ones(400, dtype=np.int8)
        period[200:] = 2
        table = _table(400, np.full((400, 2), np.nan), period=period)
        table.players[200:, :2, 0], table.players[200:, 2:, 0] = 80.0, 20.0   # ends swap
        ends = rs.defended_ends(table)
        self.assertEqual(ends, {1: {H: 0, A: 1}, 2: {H: 1, A: 0}})
        self.assertEqual(rs.implied_side("goal kick", 0, ends[1]), H)       # home defends x 0
        self.assertEqual(rs.implied_side("corner", 0, ends[1]), A)          # away attacks it
        self.assertEqual(rs.implied_side("corner", 0, ends[2]), H)
        self.assertIsNone(rs.implied_side("corner", 0, None))
        late = _table(400, np.full((400, 2), np.nan), period=period)
        late.players[200:, :2, 0], late.players[200:, 2:, 0] = 20.0, 80.0  # walking out ...
        late.players[300:, :2, 0], late.players[300:, 2:, 0] = 80.0, 20.0  # ... kickoff at 300
        self.assertEqual(rs.defended_ends(late)[2], {H: 0, A: 1})          # read too early
        self.assertEqual(rs.defended_ends(late, {2: 300})[2], {H: 1, A: 0})
        table.players[:200] = np.nan
        self.assertNotIn(1, rs.defended_ends(table))                       # no players seen


class FillTests(unittest.TestCase):
    def codes(self, *runs):
        return np.concatenate([np.full(n, c, dtype=np.int8) for c, n in runs])

    def test_short_stretches_ending_at_the_side_only(self):
        codes = self.codes((A, 75), (H, 10), (D, 76), (H, 5), (A, 10), (X, 5), (H, 5),
                           (H, 20), (A, 30))
        rows = [(0, "corner", H), (85, "corner", H), (166, "corner", H), (171, "goal kick", H),
                (191, "goal kick", None), (196, "corner", H)]
        out, changed = rs.fill(codes, rows)
        np.testing.assert_array_equal(out[:75], np.full(75, H))           # exactly 75: filled
        np.testing.assert_array_equal(out[85:161], codes[85:161])         # 76: too long
        np.testing.assert_array_equal(out[171:181], codes[171:181])       # ends at X: left
        self.assertEqual(changed, 1)                                     # 166 and 196 are H
        self.assertEqual(int((out != codes).sum()), 75)

    def test_tally_counts_agree_disagree_undecided(self):
        codes = self.codes((H, 60), (A, 60), (D, 60))
        rows = [(0, "corner", H), (60, "goal kick", H), (120, "corner", A), (0, "corner", None)]
        t = rs.tally(codes, rows, 25)
        self.assertEqual(t["corner"], [3, 1, 0, 2])
        self.assertEqual(t["goal kick"], [1, 0, 1, 0])
        self.assertEqual(t["all"], [4, 1, 1, 2])


if __name__ == "__main__":
    unittest.main()
