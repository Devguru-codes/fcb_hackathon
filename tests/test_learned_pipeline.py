import unittest

import numpy as np
import pandas as pd

from tempolens.experiments import decode_table
from tempolens.features import frame_features, impute_ball, swap_labels
from tempolens.match import A, D, H, X, MatchTable
from tempolens.opendata import labels_from_events


def _events(rows):
    frame = pd.DataFrame(rows, columns=["team", "type", "subtype", "start", "end"])
    frame["order"] = np.arange(len(frame))
    return frame


def _table(n=200, seed=0):
    rng = np.random.default_rng(seed)
    players = rng.uniform([0, 0], [105, 68], size=(n, 6, 2)).astype(np.float32)
    ball = rng.uniform([0, 0], [105, 68], size=(n, 2)).astype(np.float32)
    ball[50:60] = np.nan
    period = np.zeros(n, dtype=np.int8)
    period[10:100], period[100:190] = 1, 2
    return MatchTable(name="synthetic", ball=ball, players=players,
                      slot_team=np.array([H, H, H, A, A, A], dtype=np.int8), period=period)


class EventLabelTests(unittest.TestCase):
    def test_possession_and_dead_ball_rules(self):
        period = np.zeros(100, dtype=np.int8)
        period[:] = 1
        events = _events([
            (H, "SET PIECE", "KICK OFF", 10, 10),
            (H, "PASS", "", 10, 20),
            (A, "CHALLENGE", "GROUND-WON", 25, 25),   # never confers possession
            (A, "RECOVERY", "", 30, 30),
            (A, "BALL OUT", "", 50, 55),              # dead from the frame it crosses
            (H, "SET PIECE", "THROW IN", 70, 70),
            (H, "PASS", "", 70, 80),
        ])
        labels = labels_from_events(events, period, 100)
        expected = np.array([D] * 10 + [H] * 20 + [A] * 25 + [D] * 15 + [H] * 11 + [D] * 19)
        np.testing.assert_array_equal(labels, expected)

    def test_frames_outside_periods_are_x(self):
        period = np.zeros(50, dtype=np.int8)
        period[10:40] = 1
        labels = labels_from_events(_events([(H, "SET PIECE", "KICK OFF", 15, 15)]), period, 50)
        self.assertTrue(np.all(labels[:10] == X) and np.all(labels[40:] == X))
        self.assertTrue(np.all(labels[10:15] == D))


class FeatureTests(unittest.TestCase):
    def test_swapping_teams_mirrors_the_features(self):
        table = _table()
        x, names = frame_features(table)
        xs, _ = frame_features(table, swap=True)
        col = {name: i for i, name in enumerate(names)}
        inside = table.period > 0
        np.testing.assert_allclose(xs[inside, col["d_margin"]], -x[inside, col["d_margin"]],
                                   rtol=1e-5, equal_nan=True)
        np.testing.assert_allclose(xs[inside, col["t0_d1"]], x[inside, col["t1_d1"]],
                                   rtol=1e-5, equal_nan=True)
        np.testing.assert_allclose(xs[inside, col["ball_x"]], -x[inside, col["ball_x"]],
                                   rtol=1e-5, atol=1e-4, equal_nan=True)

    def test_features_blank_outside_periods(self):
        table = _table()
        x, _ = frame_features(table)
        self.assertTrue(np.all(np.isnan(x[table.period == 0])))

    def test_short_gaps_are_imputed_within_a_period_only(self):
        ball = np.array([[0, 0], [np.nan, np.nan], [2, 2], [np.nan, np.nan], [4, 4]], float)
        out, imputed = impute_ball(ball, [(0, 3), (3, 5)])
        self.assertEqual(imputed.tolist(), [False, True, False, False, False])
        self.assertEqual(out[1].tolist(), [1.0, 1.0])
        self.assertTrue(np.isnan(out[3, 0]))

    def test_row_quantile_matches_numpy(self):
        from tempolens.features import _row_quantile
        rng = np.random.default_rng(0)
        values = rng.normal(size=(500, 11))
        values[rng.random(values.shape) < 0.3] = np.nan
        values[0] = np.nan
        for q in (0.1, 0.5, 0.9):
            np.testing.assert_allclose(_row_quantile(values, q),
                                       np.nanpercentile(values, 100 * q, axis=1), equal_nan=True)

    def test_carry_across_gaps(self):
        from tempolens.features import _carry
        values = np.array([np.nan, 1.0, np.nan, np.nan, 2.0, np.nan])
        np.testing.assert_array_equal(_carry(values, True), [np.nan, 1, 1, 1, 2, 2])
        np.testing.assert_array_equal(_carry(values, False), [1, 1, 2, 2, 2, np.nan])

    def test_swap_labels(self):
        np.testing.assert_array_equal(swap_labels(np.array([H, A, D, X], dtype=np.int8)),
                                      [A, H, D, X])


class SimulatorTests(unittest.TestCase):
    def _labelled(self, n=20000):
        rng = np.random.default_rng(3)
        players = rng.uniform([0, 0], [105, 68], size=(n, 6, 2)).astype(np.float32)
        ball = rng.uniform([0, 0], [105, 68], size=(n, 2)).astype(np.float32)
        period = np.zeros(n, dtype=np.int8)
        period[100:10000], period[10000:19900] = 1, 2
        labels = np.full(n, X, dtype=np.int8)
        labels[period > 0] = np.tile(np.repeat([H, D, A, D], 250), n // 1000)[:np.sum(period > 0)]
        return MatchTable(name="sim", ball=ball, players=players,
                          slot_team=np.array([H, H, H, A, A, A], dtype=np.int8),
                          period=period, labels=labels)

    def test_hits_the_missing_target_deterministically_and_keeps_labels(self):
        from tempolens.simulate import AtdNoise, simulate_atd
        table = self._labelled()
        first = simulate_atd(table, AtdNoise(ball_missing=0.4), seed=5)
        second = simulate_atd(table, AtdNoise(ball_missing=0.4), seed=5)
        missing = np.isnan(first.ball[table.period > 0, 0]).mean()
        self.assertAlmostEqual(missing, 0.4, delta=0.06)
        np.testing.assert_array_equal(np.isnan(first.ball), np.isnan(second.ball))
        np.testing.assert_array_equal(first.labels, table.labels)
        self.assertFalse(np.isnan(table.ball).any())   # the input is not modified

    def test_stoppage_gaps_make_missing_ball_likelier_when_dead(self):
        from tempolens.simulate import AtdNoise, simulate_atd
        table = self._labelled()
        sim = simulate_atd(table, AtdNoise(dead_cover=0.9), seed=1)
        missing = np.isnan(sim.ball[:, 0])
        self.assertGreater(missing[table.labels == D].mean(),
                           2 * missing[np.isin(table.labels, [H, A])].mean())


class EnsembleTests(unittest.TestCase):
    def test_combine_averages_probabilities_and_change_logits(self):
        from tempolens.temporal import combine
        a = np.column_stack([np.log([[0.8, 0.1, 0.1]]), [[2.0]]])
        b = np.column_stack([np.log([[0.2, 0.7, 0.1]]), [[0.0]]])
        out = combine([a, b])
        np.testing.assert_allclose(np.exp(out[0, :3]), [0.5, 0.4, 0.1], atol=1e-3)
        self.assertAlmostEqual(out[0, 3], 1.0)

    def test_member_noise_seeds_are_disjoint(self):
        from tempolens.experiments import member_noise_seeds
        seeds = [set(member_noise_seeds(k)) for k in range(4)]
        for i in range(4):
            for j in range(i + 1, 4):
                self.assertFalse(seeds[i] & seeds[j])


class TemporalVariantTests(unittest.TestCase):
    def test_every_variant_trains_saves_loads_and_predicts(self):
        import tempfile
        from tempolens import temporal
        n = 3000
        rng = np.random.default_rng(2)
        period = np.ones(n, dtype=np.int8)
        period[1500:] = 2
        table = MatchTable(name="tiny", ball=rng.uniform([0, 0], [105, 68], (n, 2)).astype(np.float32),
                           players=rng.uniform([0, 0], [105, 68], (n, 22, 2)).astype(np.float32),
                           slot_team=np.array([H, A] * 11, dtype=np.int8), period=period,
                           labels=np.tile(np.repeat(np.array([H, D, A], np.int8), 250), n // 750))
        table.ball[100:300] = np.nan
        configs = [temporal.TemporalConfig(aux_ball=1.0),
                   temporal.TemporalConfig(encoder="players", hidden=16),
                   temporal.TemporalConfig(arch="tcn", hidden=16),
                   temporal.TemporalConfig(arch="transformer", hidden=16)]
        with tempfile.TemporaryDirectory() as folder:
            for config in configs:
                model, normaliser = temporal.train([table], epochs=1, device="cpu",
                                                   verbose=False, config=config)
                path = "{}/{}.pt".format(folder, config.tag())
                temporal.save(model, normaliser, path)
                loaded, norm = temporal.load(path)
                self.assertEqual(loaded.config, config)
                out = temporal.predict(loaded, norm, table)
                self.assertEqual(out.shape, (n, 4), config.tag())
                self.assertTrue(np.all(np.isfinite(out)), config.tag())
                np.testing.assert_allclose(np.exp(out[:, :3]).sum(axis=1), 1.0, atol=1e-2)

    def test_committed_model_still_loads_as_default_gru(self):
        from pathlib import Path
        from tempolens import temporal
        path = Path(__file__).resolve().parents[1] / "models" / "tempolens_gru.pt"
        if not path.exists():
            self.skipTest("no committed model")
        model, _ = temporal.load(path)
        self.assertEqual(model.config, temporal.DEFAULT_CONFIG)


class DecodeTests(unittest.TestCase):
    def test_numba_and_numpy_viterbi_agree(self):
        from unittest.mock import patch
        import tempolens.decoder as decoder
        rng = np.random.default_rng(1)
        logits = np.log(rng.dirichlet([0.5, 0.5, 0.5], 3000))
        transition = decoder.transition_matrix(6.0, 3.0, 2.0)
        bonus = rng.normal(0, 2, 3000)
        fast = decoder.viterbi_matrix(logits, transition, bonus)
        with patch.object(decoder, "_kernel", None):
            slow = decoder.viterbi_matrix(logits, transition, bonus)
        np.testing.assert_array_equal(fast, slow)
        self.assertGreater(len(np.unique(fast)), 1)

    def test_decode_writes_x_outside_periods_and_respects_evidence(self):
        table = _table()
        logp = np.log(np.tile([0.8, 0.1, 0.1], (table.n_frames, 1)))
        logp[120:190] = np.log([0.1, 0.8, 0.1])
        out = decode_table(table, logp, np.zeros((3, 3)))
        self.assertTrue(np.all(out[table.period == 0] == X))
        self.assertTrue(np.all(out[10:120] == H) and np.all(out[120:190] == A))


if __name__ == "__main__":
    unittest.main()
