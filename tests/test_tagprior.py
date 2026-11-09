import unittest

import numpy as np

from tempolens.match import A, H
from tempolens.tagprior import apply, calibrate, group_of, self_calibrated, votes


def _logp(sides, confidence=0.95):
    """Model output whose confident side per frame is `sides` (H/A), D small."""
    p = np.full((len(sides), 3), 0.02)
    p[np.arange(len(sides)), sides] = confidence
    p[np.arange(len(sides)), 1 - np.asarray(sides)] = 1 - confidence - 0.02
    return np.column_stack([np.log(p), np.zeros(len(sides))])


class TagPriorTests(unittest.TestCase):
    def test_codes_map_to_families(self):
        self.assertEqual(group_of("BUILD UP"), "attack")
        self.assertEqual(group_of("Defensive Transition"), "defence")
        self.assertIsNone(group_of("SET PIECES"))
        from tempolens.tagprior import FAMILIES, GROUPS
        self.assertEqual(group_of("Ball in the Box", FAMILIES), "final third / box")
        self.assertEqual(group_of("Defensive Transition", FAMILIES), "defensive transition")
        self.assertEqual(sorted(c for codes in FAMILIES.values() for c in codes),
                         sorted(c for codes in GROUPS.values() for c in codes))

    def test_votes_sign_and_overlap(self):
        v = votes([(0, 10, "attack", H), (5, 15, "attack", A), (20, 30, "defence", A)], 40)
        self.assertEqual(v["attack"][:5].tolist(), [1] * 5)
        self.assertEqual(v["attack"][5:10].tolist(), [0] * 5)    # conflicting tags cancel
        self.assertEqual(v["attack"][10:15].tolist(), [-1] * 5)
        self.assertEqual(v["defence"][20:30].tolist(), [-1] * 10)

    def test_reliable_tags_get_weight_and_random_tags_do_not(self):
        rng = np.random.default_rng(0)
        sides = rng.integers(0, 2, 5000)
        logp = _logp(sides)
        inside = np.ones(5000, dtype=bool)
        truth_vote = np.where(sides == H, 1, -1).astype(np.int8)
        good = calibrate(logp, {"attack": truth_vote}, inside)["attack"]
        random = calibrate(logp, {"attack": rng.choice([-1, 1], 5000).astype(np.int8)},
                           inside)["attack"]
        self.assertGreater(good[0], 1.0)
        self.assertLess(random[0], 0.1)

    def test_apply_raises_h_lowers_a_and_leaves_d_unnormalised(self):
        logp = _logp(np.array([H, A]), confidence=0.6)
        out = apply(logp, {"attack": np.array([1, 1], np.int8)}, {"attack": (1.0, 0.9, 999)})
        np.testing.assert_allclose(out[:, 2], logp[:, 2])
        np.testing.assert_allclose(out[:, H] - logp[:, H], 0.5)
        np.testing.assert_allclose(out[:, A] - logp[:, A], -0.5)

    def test_too_few_confident_frames_means_no_weight(self):
        logp = _logp(np.zeros(100, dtype=int))
        _, report = self_calibrated(logp, [(0, 100, "attack", H)], np.ones(100, dtype=bool))
        self.assertEqual(report["attack"][0], 0.0)


if __name__ == "__main__":
    unittest.main()
