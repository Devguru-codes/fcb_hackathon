import unittest

import numpy as np

from tempolens.match import A, H, MatchTable
from tempolens.vision.io import choose_training_frames, project, unproject
from tempolens.vision.track import _viterbi_gap, fill_ball


def _matrix():
    # A perspective homography: unit square -> a trapezoid in a 960x540 image.
    return np.array([[820.0, -60.0, 70.0], [0.0, 380.0, 140.0], [0.0, 0.05, 1.0]])


class ProjectionTests(unittest.TestCase):
    def test_unproject_inverts_project(self):
        rng = np.random.default_rng(0)
        xy = rng.uniform(0, 1, (50, 2))
        matrices = np.repeat(_matrix()[None], 50, axis=0)
        np.testing.assert_allclose(unproject(matrices, project(matrices, xy)), xy, atol=1e-9)

    def test_invalid_matrix_gives_nan(self):
        matrices = np.full((1, 3, 3), np.nan)
        self.assertTrue(np.isnan(unproject(matrices, np.array([[10.0, 10.0]]))).all())


class TrainingSplitTests(unittest.TestCase):
    def test_small_match_keeps_negatives_and_an_audit_reserve(self):
        pixels = np.full((5000, 2), 300.0)
        pixels[::3] = np.nan
        graded = np.ones(5000, dtype=bool)
        positive, negative = choose_training_frames(pixels, (960, 540), graded)
        tracked = np.isfinite(pixels[:, 0]).sum()
        self.assertGreater(len(negative), 0)
        self.assertLessEqual(len(positive) + len(negative), int(0.85 * tracked))
        self.assertFalse(np.intersect1d(positive, negative).size)


class LinkingTests(unittest.TestCase):
    def test_moving_ball_beats_scattered_false_peaks(self):
        rng = np.random.default_rng(1)
        n = 60
        true = np.column_stack([np.linspace(30, 45, n), np.full(n, 34.0)])   # 6 m/s
        cands = []
        for t in range(n):
            frame = [(true[t, 0], true[t, 1], 0.7)]
            frame.append((rng.uniform(0, 105), rng.uniform(0, 68), 0.9))   # brighter decoy
            cands.append(frame)
        path = _viterbi_gap(cands, before=true[0] - [0.25, 0], after=true[-1] + [0.25, 0], fps=25)
        self.assertGreaterEqual(np.mean(np.array(path) == 0), 0.95)

    def test_impossible_jumps_are_not_linked(self):
        cands = [[(10.0, 10.0, 0.6)], [(90.0, 60.0, 0.6)], [(10.0, 10.0, 0.6)]]
        path = _viterbi_gap(cands, before=None, after=None, fps=25)
        chosen = [cands[t][j][:2] for t, j in enumerate(path) if j >= 0]
        for a, b in zip(chosen, chosen[1:]):
            self.assertLess(np.hypot(a[0] - b[0], a[1] - b[1]), 3.0)

    def test_fill_only_touches_gap_frames_and_respects_the_mask(self):
        n = 40
        ball = np.full((n, 2), 50.0, dtype=np.float32)
        ball[10:20] = np.nan
        table = MatchTable(name="t", ball=ball, players=np.zeros((n, 2, 2), np.float32),
                           slot_team=np.array([H, A], np.int8), period=np.ones(n, np.int8))
        candidates = {f: [(50.0 + 0.1 * (f - 9), 50.0, 0.8)] for f in range(10, 20)}
        filled, source = fill_ball(table, candidates, fill_min=0.5, min_run=1)
        self.assertTrue(np.all(source[10:20] == 2) and np.all(source[:10] == 1))
        np.testing.assert_array_equal(filled[:10], ball[:10])
        mask = np.zeros(n, dtype=bool)
        mask[10:15] = True
        _, source = fill_ball(table, candidates, fill_mask=mask, fill_min=0.5, min_run=1)
        self.assertEqual(source[10:20].tolist(), [2] * 5 + [0] * 5)
        # the default threshold (0.95) refuses these 0.8 detections
        _, source = fill_ball(table, candidates)
        self.assertTrue(np.all(source[10:20] == 0))

    def test_short_video_stretches_are_dropped(self):
        n = 40
        ball = np.full((n, 2), 50.0, dtype=np.float32)
        ball[10:30] = np.nan
        table = MatchTable(name="t", ball=ball, players=np.zeros((n, 2, 2), np.float32),
                           slot_team=np.array([H, A], np.int8), period=np.ones(n, np.int8))
        # a 3-frame stretch and a 8-frame stretch of confident detections
        candidates = {f: [(50.0, 50.0, 0.99)] for f in list(range(10, 13)) + list(range(20, 28))}
        _, source = fill_ball(table, candidates, min_run=5)
        self.assertTrue(np.all(source[10:13] == 0))
        self.assertTrue(np.all(source[20:28] == 2))


class TrackerFaultTests(unittest.TestCase):
    def test_faults_add_slots_but_keep_labels_and_frame_count(self):
        from tempolens.simulate import TrackerFaults, add_tracker_faults
        rng = np.random.default_rng(0)
        n = 5000
        players = rng.uniform([0, 0], [105, 68], (n, 22, 2)).astype(np.float32)
        table = MatchTable(name="t", ball=rng.uniform([0, 0], [105, 68], (n, 2)).astype(np.float32),
                           players=players, slot_team=np.array([H, A] * 11, np.int8),
                           period=np.ones(n, np.int8), labels=np.zeros(n, np.int8))
        faults = TrackerFaults(swap_rate=0.01, duplicate_rate=0.002, phantom_rate=0.002)
        out = add_tracker_faults(table, faults, seed=1)
        self.assertEqual(out.n_frames, n)
        self.assertGreater(out.players.shape[1], 22)
        self.assertEqual(len(out.slot_team), out.players.shape[1])
        np.testing.assert_array_equal(out.labels, table.labels)
        self.assertTrue(np.array_equal(table.players, players))   # input untouched


if __name__ == "__main__":
    unittest.main()
