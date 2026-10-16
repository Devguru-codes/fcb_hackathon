import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
import possession_metrics as pm  # noqa: E402

from tempolens.decoder import transition_matrix, viterbi_matrix
from tempolens.match import A, D, H, X
from tempolens.mbr import fast_spell_f1, sample_paths, viterbi_min_duration


def _random_codes(rng, n, mean_run):
    codes, t = np.empty(n, dtype=np.int8), 0
    while t < n:
        length = int(rng.geometric(1.0 / mean_run))
        codes[t:t + length] = rng.choice([H, A, D], p=[0.4, 0.4, 0.2])
        t += length
    return codes


class FastSpellF1Tests(unittest.TestCase):
    def test_matches_the_official_metric(self):
        rng = np.random.default_rng(0)
        for trial in range(30):
            n = 4000
            gt = _random_codes(rng, n, 120)
            pred = gt.copy()
            # perturb: shift some boundaries, flip some runs, add flicker
            for _ in range(rng.integers(0, 40)):
                a = int(rng.integers(0, n - 50))
                pred[a:a + int(rng.integers(1, 50))] = rng.choice([H, A, D])
            gt[:100] = X
            gt[2000:2100] = X
            runs = pm.valid_runs(gt)
            official = pm.spell_scores(gt, pred, runs)["mean_f1"]
            self.assertAlmostEqual(fast_spell_f1(gt, pred, runs), official, places=12,
                                   msg="trial {}".format(trial))


class MinDurationTests(unittest.TestCase):
    def test_minimum_one_is_plain_viterbi(self):
        rng = np.random.default_rng(1)
        logits = np.log(rng.dirichlet([0.6, 0.6, 0.6], 2000))
        transition = transition_matrix(3.0, 2.0, 2.0)
        np.testing.assert_array_equal(viterbi_min_duration(logits, transition, (1, 1, 1)),
                                      viterbi_matrix(logits, transition))

    def test_runs_respect_the_minimum(self):
        rng = np.random.default_rng(2)
        logits = np.log(rng.dirichlet([0.3, 0.3, 0.3], 3000))
        out = viterbi_min_duration(logits, transition_matrix(0.5, 0.5, 0.5), (10, 10, 6))
        bounds = np.r_[0, np.flatnonzero(out[1:] != out[:-1]) + 1, len(out)]
        for a, b in zip(bounds[1:-2], bounds[2:-1]):     # interior runs only
            self.assertGreaterEqual(b - a, 6 if out[a] == D else 10)


class SamplingTests(unittest.TestCase):
    def test_samples_are_deterministic_and_follow_confident_evidence(self):
        class Sample:
            n_frames = 600
            period = np.ones(600, dtype=np.int8)

            def runs(self):
                return [(0, 600)]
        logp = np.log(np.tile([0.98, 0.01, 0.01], (600, 1)))
        logp[300:] = np.log([0.01, 0.98, 0.01])
        transition = transition_matrix(2.0, 4.0, 8.0)
        a = sample_paths(Sample(), logp, transition, n_samples=5, seed=3)
        b = sample_paths(Sample(), logp, transition, n_samples=5, seed=3)
        np.testing.assert_array_equal(a, b)
        self.assertGreater(np.mean(a[:, :290] == H), 0.95)
        self.assertGreater(np.mean(a[:, 310:] == A), 0.95)


if __name__ == "__main__":
    unittest.main()
