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


class TrackingParserTests(unittest.TestCase):
    def test_read_atd_aligns_frames_and_teams(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from tempolens.match import read_atd
        with TemporaryDirectory() as folder:
            meta, raw = Path(folder) / "atd.xml", Path(folder) / "atd.txt"
            meta.write_text("<root><Team id='h'><Name>Barcelona</Name></Team>"
                            "<Team id='a'><Name>Athletic</Name></Team>"
                            "<Player teamId='h'/><Player teamId='a'/></root>", encoding="utf-8")
            # tracking frames count from 1; a NaN ball and a malformed line are skipped
            raw.write_text("1:0.50,0.50,0;0.70,0.50,0:0.51,0.50,0\n"
                           "2:0.50,0.50,0;0.70,0.50,0:NaN,NaN,NaN\n"
                           "garbage\n"
                           "3:0.50,0.50,0;N:0.69,0.50,0\n", encoding="utf-8")
            table = read_atd(meta, raw, 3, (100.0, 50.0), period_blocks=[(1, 0, 1)])
        self.assertEqual((table.meta["home"], table.meta["away"]), ("Barcelona", "Athletic"))
        self.assertEqual(table.slot_team.tolist(), [H, A])
        np.testing.assert_allclose(table.ball[0], [51.0, 25.0])
        self.assertTrue(np.isnan(table.ball[1, 0]))
        np.testing.assert_allclose(table.players[2, 0], [50.0, 25.0])
        self.assertTrue(np.isnan(table.players[2, 1, 0]))
        self.assertEqual(table.period.tolist(), [1, 1, 0])

    def test_read_atd_refuses_a_file_with_missing_frames(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from tempolens.match import read_atd
        with TemporaryDirectory() as folder:
            meta, raw = Path(folder) / "atd.xml", Path(folder) / "atd.txt"
            meta.write_text("<root><Team id='h'/><Team id='a'/><Player teamId='h'/></root>",
                            encoding="utf-8")
            raw.write_text("1:0.5,0.5,0:N\n3:0.5,0.5,0:N\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "2 of 3 frames"):
                read_atd(meta, raw, 3, (100.0, 50.0))


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
    def test_empty_and_reordered_slots_change_nothing(self):
        # Slots are anonymous tracklets: the real ATD has 46 for ~20 visible players, open data
        # ~15 per team. Features must not see how many slots exist or in which order.
        table = _table()
        n = table.n_frames
        extra = np.full((n, 20, 2), np.nan, dtype=np.float32)
        players = np.concatenate([table.players, extra], axis=1)
        slot_team = np.concatenate([table.slot_team, np.array([H, A] * 10, dtype=np.int8)])
        order = np.random.default_rng(3).permutation(len(slot_team))
        padded = MatchTable(name="padded", ball=table.ball, players=players[:, order],
                            slot_team=slot_team[order], period=table.period)
        x, names = frame_features(table)
        xp, _ = frame_features(padded)
        for i, name in enumerate(names):
            np.testing.assert_allclose(xp[:, i], x[:, i], rtol=1e-5, atol=1e-5, err_msg=name)

    def test_mirroring_the_pitch_twice_is_the_identity(self):
        from tempolens.temporal import mirror_y
        x, names = frame_features(_table())
        once = mirror_y(x, names)
        np.testing.assert_allclose(mirror_y(once, names), x, equal_nan=True)
        col = names.index("t0_y0")
        self.assertFalse(np.allclose(once[:, col], x[:, col], equal_nan=True))

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

    def test_randomised_noise_is_seeded_varied_and_reached(self):
        from tempolens.simulate import random_noise, simulate_atd
        self.assertEqual(random_noise(7), random_noise(7))
        settings = [random_noise(s) for s in range(20)]
        self.assertGreater(np.ptp([n.ball_missing for n in settings]), 0.1)
        self.assertTrue(all(0.30 <= n.ball_missing <= 0.50 and 0.40 <= n.dead_cover <= 0.90
                            for n in settings))
        table = self._labelled()
        sim = simulate_atd(table, settings[0], seed=0)
        self.assertAlmostEqual(np.isnan(sim.ball[table.period > 0, 0]).mean(),
                               settings[0].ball_missing, delta=0.06)


class EnsembleTests(unittest.TestCase):
    def test_adapted_normaliser_moves_to_the_target_and_keeps_the_width(self):
        from tempolens.features import frame_features
        from tempolens.temporal import Normaliser, adapt_normaliser
        table = SimulatorTests()._labelled(4000)
        x = frame_features(table)[0]
        base = Normaliser(np.asarray(x[table.period > 0][::10]) + 5.0)   # a shifted "training" fit
        same = adapt_normaliser(base, table, alpha=0.0)
        np.testing.assert_array_equal(same.median, base.median)
        full = adapt_normaliser(base, table, alpha=1.0)
        own = Normaliser(np.asarray(x[table.period > 0][::10]))
        np.testing.assert_allclose(full.median, own.median)
        np.testing.assert_array_equal(full.flagged, base.flagged)
        self.assertEqual(full(x).shape, base(x).shape)

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
        import torch
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
                   temporal.TemporalConfig(arch="tcn", hidden=16, seg_balance=0.5, focal=2.0),
                   temporal.TemporalConfig(arch="lstm", hidden=16, smooth=0.15),
                   temporal.TemporalConfig(hidden=16, dead_weight=2.0),
                   temporal.TemporalConfig(arch="transformer", hidden=16, mirror=True)]
        with tempfile.TemporaryDirectory() as folder:
            for config in configs:
                model, normaliser = temporal.train([table], epochs=1, device="cpu",
                                                   verbose=False, config=config)
                path = "{}/{}.pt".format(folder, config.tag())
                temporal.save(model, normaliser, path)
                loaded, norm = temporal.load(path)
                self.assertEqual(loaded.config, config)
                # float64 inference is what makes another CPU produce the same file
                self.assertEqual(next(loaded.parameters()).dtype, torch.float64)
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
