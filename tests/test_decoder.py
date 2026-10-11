import unittest

import numpy as np

from tempolens.config import DecoderConfig
from tempolens.decoder import decode, merge_short_runs, viterbi
from tempolens.tracking import TrackingFeatures


class DecoderTests(unittest.TestCase):
    def test_viterbi_prefers_a_single_possession_over_a_one_frame_flicker(self):
        logits = np.tile(np.array([3.0, 0.0, -2.0]), (40, 1))
        logits[20] = np.array([0.0, 5.0, -2.0])
        states = viterbi(logits, DecoderConfig(team_switch_cost=3.2))
        self.assertTrue(np.all(states == 0))

    def test_short_middle_run_is_merged_only_when_neighbours_match(self):
        states = np.array([0] * 20 + [1] * 3 + [0] * 20, dtype=np.int8)
        merged = merge_short_runs(states, DecoderConfig(min_alive_run_frames=12))
        self.assertTrue(np.all(merged == 0))

    def test_short_dead_ball_between_teams_is_preserved(self):
        states = np.array([0] * 20 + [2] * 3 + [1] * 20, dtype=np.int8)
        merged = merge_short_runs(states, DecoderConfig(min_dead_run_frames=8))
        self.assertEqual(merged[21], 2)

    def test_missing_ball_preserves_the_current_spell(self):
        n = 30
        features = TrackingFeatures(
            ball_seen=np.array([True] * 10 + [False] * 10 + [True] * 10),
            ball_x=np.zeros(n), ball_y=np.zeros(n), ball_age=np.arange(n),
            nearest_home_m=np.array([1.0] * 10 + [np.nan] * 10 + [1.0] * 10),
            nearest_away_m=np.array([7.0] * 10 + [np.nan] * 10 + [7.0] * 10),
            close_home=np.ones(n), close_away=np.zeros(n), tracked_players=np.full(n, 22),
        )
        states = decode(features, np.zeros((n, 3)), DecoderConfig())
        self.assertTrue(np.all(states == 0))


if __name__ == "__main__":
    unittest.main()
