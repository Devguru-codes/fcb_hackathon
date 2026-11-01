import unittest

import numpy as np

from tempolens.match import A, H, MatchTable
from tempolens.touches import snap_to_contact


class SnapToContactTests(unittest.TestCase):
    def test_removes_the_clock_offset_and_drops_events_without_contact(self):
        n = 2000
        players = np.full((n, 2, 2), 30.0, dtype=np.float32)
        players[:, 1] = 70.0
        ball = np.full((n, 2), 50.0, dtype=np.float32)
        touches = [300, 700, 1100, 1500]                  # the true contacts, home then away
        for i, f in enumerate(touches):
            players[f - 3:f + 4, i % 2] = 50.0                # that team's player reaches the ball
        table = MatchTable(name="t", ball=ball, players=players,
                           slot_team=np.array([H, A], dtype=np.int8), period=np.ones(n, np.int8))
        logged = np.array(touches + [1800]) + 30           # event clock 30 frames late, one fake
        teams = np.array([H, A, H, A, H], dtype=np.int8)
        frames, kept = snap_to_contact(table, logged, teams)
        self.assertEqual(len(frames), 4)                   # the event with no contact is dropped
        np.testing.assert_array_equal(kept, teams[:4])
        self.assertTrue(np.all(np.abs(frames - np.array(touches)) <= 3))


if __name__ == "__main__":
    unittest.main()
