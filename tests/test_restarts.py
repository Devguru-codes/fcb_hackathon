import unittest

import numpy as np

from tempolens import kickoffs, restarts
from tempolens.match import A, D, H, X, MatchTable


def _codes(*runs):
    return np.concatenate([np.full(n, c, dtype=np.int8) for c, n in runs])


def _table(ball):
    n = len(ball)
    return MatchTable("r", ball.astype(np.float32), np.zeros((n, 2, 2), dtype=np.float32),
                      np.array([H, A], dtype=np.int8), np.ones(n, dtype=np.int8))


class LeadingDeadTests(unittest.TestCase):
    def test_runs_at_and_beyond_the_window(self):
        self.assertEqual(restarts.leading_dead(_codes((D, 75), (H, 5)), 0), 75)
        self.assertIsNone(restarts.leading_dead(_codes((D, 76), (H, 5)), 0))
        self.assertEqual(restarts.leading_dead(_codes((D, 1), (A, 5)), 0), 1)

    def test_not_dead_open_ended_and_x(self):
        self.assertEqual(restarts.leading_dead(_codes((H, 5), (D, 5)), 0), 0)
        self.assertIsNone(restarts.leading_dead(_codes((D, 10)), 0))              # to the end
        self.assertIsNone(restarts.leading_dead(_codes((D, 10), (X, 5), (H, 5)), 0))

    def test_backfill_known_team_only_and_counts_stretches(self):
        codes = _codes((D, 10), (A, 10), (D, 80), (H, 10), (D, 5), (H, 5))
        out, changed = restarts.backfill(codes, [(0, H), (20, A), (110, None)])
        np.testing.assert_array_equal(out[:10], np.full(10, H))      # filled with the team named
        np.testing.assert_array_equal(out[20:100], codes[20:100])    # 80 > 75: left
        np.testing.assert_array_equal(out[110:115], codes[110:115])  # unknown team: left
        self.assertEqual(changed, 1)
        self.assertEqual(int((out != codes).sum()), 10)
        self.assertEqual(int((codes == D).sum()), 95)                 # the input is not changed

    def test_start_delay_and_alive_share(self):
        codes = _codes((D, 30), (H, 10))
        self.assertEqual(restarts.start_delay(codes, 0, 40), 30)
        self.assertIsNone(restarts.start_delay(codes, 0, 30))
        self.assertAlmostEqual(restarts.alive_share(codes), 0.25)
        self.assertTrue(np.isnan(restarts.alive_share(codes[:0])))


class GoalMomentTests(unittest.TestCase):
    def test_beyond_the_line_between_the_posts(self):
        length, width = 105.0, 68.0
        ball = np.full((10, 2), [50.0, 34.0])
        ball[2] = np.nan                                       # no ball: never a goal
        ball[3] = [-0.5, 34.0 + restarts.POST_M + 0.01]        # outside the post
        ball[4] = [length + 0.5, 34.0 - restarts.POST_M]       # at the post: counts
        ball[6] = [-1.0, 34.0]
        table = _table(ball)
        self.assertEqual(table.pitch, (length, width))
        self.assertEqual(restarts.goal_moment(table, 0, 10), 4)
        self.assertEqual(restarts.goal_moment(table, 5, 10), 6)
        self.assertIsNone(restarts.goal_moment(table, 0, 4))


class KickoffRestartTests(unittest.TestCase):
    def test_strict_returns_none_when_no_restart_is_seen(self):
        ball = np.full((400, 2), np.nan)
        table = _table(ball)
        self.assertEqual(kickoffs.restart_frame(table, 10), 10)
        self.assertIsNone(kickoffs.restart_frame(table, 10, strict=True))
        ball[100:200] = [52.5, 34.0]
        ball[125:200, 0] = 52.5 + np.arange(75) * 0.2
        table = _table(ball)
        frame = kickoffs.restart_frame(table, 50, strict=True)
        self.assertTrue(100 <= frame <= 125)

    def test_restarts_lists_kickoffs_in_time_order_with_the_implied_team(self):
        n = 130000
        ball = np.full((n, 2), np.nan)
        for f in (11185, 113742, 60100):
            ball[f:f + 60] = [52.5, 34.0]
            ball[f + 25:f + 60, 0] = 52.5 + np.arange(35) * 0.2
        table = _table(ball)
        codes = np.full(n, D, dtype=np.int8)
        codes[11185:11300] = H
        goals = [(60000, A, 59000, 59900)]                      # goal_restarts' 4-tuples
        rows = kickoffs.restarts(codes, table, goals)
        self.assertEqual([(k, i) for k, _, i in rows], [("half 1", H), ("goal", A), ("half 2", A)])
        self.assertTrue(60100 <= rows[1][1] <= 60125)
        self.assertEqual([r[1:] for r in kickoffs.check(codes, table, goals)],
                         [(H, H), (A, None), (A, None)])


if __name__ == "__main__":
    unittest.main()
