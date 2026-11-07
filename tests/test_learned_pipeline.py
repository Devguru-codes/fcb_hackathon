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

    def test_head_adaptation_is_deterministic_and_lowers_entropy(self):
        import copy
        from tempolens import temporal
        table = SimulatorTests()._labelled(3000)
        model, normaliser = temporal.train([table], epochs=1, device="cpu", verbose=False,
                                           config=temporal.TemporalConfig(hidden=16))
        model = model.double()

        def entropy(m):
            p = np.exp(temporal.predict(m, normaliser, table)[table.period > 0, :3])
            return float(-(p * np.log(np.clip(p, 1e-12, 1))).sum(axis=1).mean())
        first = temporal.adapt_head(copy.deepcopy(model), normaliser, table, steps=20, lr=1e-2,
                                    gate=1.0)
        second = temporal.adapt_head(copy.deepcopy(model), normaliser, table, steps=20, lr=1e-2,
                                     gate=1.0)
        np.testing.assert_array_equal(first.head.weight.detach().numpy(),
                                      second.head.weight.detach().numpy())
        self.assertLess(entropy(first), entropy(model))

    def test_dropped_features_reach_the_network_as_zeros(self):
        from tempolens.experiments import drop_names
        from tempolens.features import RESTART_FEATURES
        from tempolens.temporal import Normaliser, TemporalConfig
        x = np.random.default_rng(0).normal(size=(50, 4))
        x[::3, 1] = np.nan                       # column 1 has a missing-value flag
        norm = Normaliser(x, drop=[1, 3])
        z = norm(x)
        self.assertEqual(z.shape, (50, 5))       # width unchanged: 4 columns + 1 flag
        self.assertTrue(np.all(z[:, [1, 3, 4]] == 0))
        self.assertTrue(np.any(z[:, 0] != 0))
        np.testing.assert_array_equal(Normaliser.from_state(norm.state())(x), z)
        self.assertEqual(drop_names(["restart"]), tuple(sorted(RESTART_FEATURES)))
        self.assertEqual(TemporalConfig().tag(), "temporal")   # old caches keep their names
        self.assertNotEqual(TemporalConfig(drop=drop_names(["restart"])).tag(), "temporal")

    def test_adaptation_study_reproduces_predict_from_head_inputs(self):
        from tempolens import adaptcv, temporal
        from tempolens.features import TableSample
        table = SimulatorTests()._labelled(3000)
        model, normaliser = temporal.train([table], epochs=1, device="cpu", verbose=False,
                                           config=temporal.TemporalConfig(hidden=16))
        model = model.double()
        sample = TableSample(table)
        inside = table.period > 0
        h0, h1 = (temporal.head_inputs(model, normaliser, sample, swap) for swap in (False, True))
        np.testing.assert_allclose(adaptcv.member_output(model.head, h0, h1, inside),
                                   temporal.predict(model, normaliser, sample), atol=1e-4)

    def test_study_eata_equals_predict_eata(self):
        import copy
        from tempolens import adaptcv, temporal
        from tempolens.features import TableSample
        table = SimulatorTests()._labelled(3000)
        model, normaliser = temporal.train([table], epochs=1, device="cpu", verbose=False,
                                           config=temporal.TemporalConfig(hidden=16))
        sample = TableSample(table)
        for double in (False, True):
            m = model.double() if double else model.float()
            grid = adaptcv.member_grid([(m, normaliser)], sample, double=double)
            adapted = temporal.adapt_head(copy.deepcopy(m), normaliser, sample, 10)
            np.testing.assert_allclose(grid[(0.0, 10)][0],
                                       temporal.predict(adapted, normaliser, sample), atol=2e-4)
            small = adaptcv.member_grid([(m, normaliser)], sample, double=double,
                                        alphas=(0.0, 0.5), steps=(0, 10))   # hedgestudy's grid
            self.assertEqual(sorted(small), [(0.0, 0), (0.0, 10), (0.5, 0), (0.5, 10)])
            for key, outs in small.items():
                np.testing.assert_array_equal(outs[0], grid[key][0])

    def test_per_member_scoring_keeps_the_ensemble_rows(self):
        from tempolens import adaptcv, temporal
        from tempolens.features import TableSample
        table = SimulatorTests()._labelled(3000)
        model, normaliser = temporal.train([table], epochs=1, device="cpu", verbose=False,
                                           config=temporal.TemporalConfig(hidden=16))
        sample, costs = TableSample(table), ((2.0, 8.0, 8.0, 0.5),)
        members = [(model, normaliser), (model, normaliser)]
        plain = adaptcv.score_match(members, sample, costs=costs)
        rows = adaptcv.score_match(members, sample, per_member=True, costs=costs)
        self.assertEqual([{k: v for k, v in r.items() if k != "seed"}
                          for r in rows if r["seed"] == -1], plain)
        self.assertEqual(sorted({r["seed"] for r in rows}), [-1, 0, 1])

    def _keeper_table(self, n=30000):
        from tempolens.match import MatchTable
        players = np.zeros((n, 4, 2), dtype=np.float32)
        players[:, :, 1] = 34.0
        players[:, :, 0] = [2.0, 40.0, 103.0, 65.0]   # home keeper, home, away keeper, away
        return MatchTable("k", np.zeros((n, 2), dtype=np.float32), players,
                          np.array([H, H, A, A], dtype=np.int8), np.ones(n, dtype=np.int8))

    def test_keeper_swap_fault_is_found_and_repaired(self):
        from tempolens.diagnostics import keeper_window_share
        from tempolens.match import keeper_windows, repair_keepers
        from tempolens.simulate import TrackerFaults, add_tracker_faults
        clean = self._keeper_table()
        self.assertEqual(keeper_windows(clean), [])
        np.testing.assert_array_equal(repair_keepers(clean).players, clean.players)
        faulted = add_tracker_faults(clean, TrackerFaults(
            swap_rate=0, duplicate_rate=0, phantom_rate=0, keeper_swaps=1,
            keeper_frames=(3000, 3001)), seed=1)
        swapped = (faulted.players != clean.players).any(axis=(1, 2))
        self.assertEqual(swapped.sum(), 3000)
        self.assertTrue((faulted.players[swapped][:, [0, 2]] ==
                         clean.players[swapped][:, [2, 0]]).all())
        repaired = repair_keepers(faulted)
        restored = swapped & (repaired.players == clean.players).all(axis=(1, 2))
        self.assertGreaterEqual(restored.sum(), 1500)          # every full window inside
        self.assertFalse(((repaired.players != clean.players).any(axis=(1, 2)) & ~swapped).any())
        share = keeper_window_share(faulted)
        self.assertEqual(share["home"]["slots"], 1)
        self.assertGreater(share["home"]["share_of_graded"], 0.0)

    def test_team_cue_pack_is_symmetric_and_trains(self):
        import tempfile
        from pathlib import Path
        from tempolens import temporal
        from tempolens.features import TEAMCUE_NAMES, TableSample, team_cue_features
        table = SimulatorTests()._labelled(3000)
        x0, x1 = team_cue_features(table), team_cue_features(table, swap=True)
        self.assertEqual(x0.shape, (table.n_frames, len(TEAMCUE_NAMES)))
        inside = table.period > 0
        self.assertTrue(np.isfinite(x0[inside][:, [2, 3, 6, 7]]).mean() > 0.9)   # times to the ball
        np.testing.assert_allclose(x0[:, [0, 2, 3]], x1[:, [4, 6, 7]], equal_nan=True)
        self.assertTrue(np.isnan(x0[~inside]).all())
        config = temporal.TemporalConfig(hidden=16, teamcue=True)
        self.assertTrue(config.tag().endswith("_tc1"))
        model, normaliser = temporal.train([table], epochs=1, device="cpu", verbose=False,
                                           config=config)
        out = temporal.predict(model, normaliser, TableSample(table))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "m.pt"
            temporal.save(model, normaliser, path)
            again = temporal.predict(*temporal.load(path), TableSample(table))
        np.testing.assert_allclose(out, again, atol=1e-4)

    def test_kickoff_rule_check(self):
        from tempolens import kickoffs
        from tempolens.match import MatchTable
        n = 130000
        ball = np.full((n, 2), np.nan, dtype=np.float32)
        for f in (11185, 113742, 60000 + 100):            # kickoffs: ball leaves the centre spot
            ball[f:f + 60] = [52.5, 34.0]
            ball[f + 25:f + 60, 0] = 52.5 + np.arange(35) * 0.2
        table = MatchTable("k", ball, np.zeros((n, 2, 2), dtype=np.float32),
                           np.array([H, A], dtype=np.int8), np.ones(n, dtype=np.int8))
        self.assertTrue(60100 <= kickoffs.restart_frame(table, 60000) <= 60125)
        codes = np.full(n, D, dtype=np.int8)
        codes[11185:11300] = H          # home kicks off the first half
        codes[113742:113900] = A        # away the second: agrees
        codes[60100:60200] = H          # home kicks off after home scored: disagrees
        rows = kickoffs.check(codes, table, [(60000, A)])
        self.assertEqual([(k, i, d) for k, i, d in rows],
                         [("half 1", H, H), ("goal", A, H), ("half 2", A, A)])

    def test_soup_of_one_member_and_of_copies_is_that_member(self):
        from tempolens import soup, temporal
        from tempolens.features import TableSample
        table = SimulatorTests()._labelled(3000)
        model, normaliser = temporal.train([table], epochs=1, device="cpu", verbose=False,
                                           config=temporal.TemporalConfig(hidden=16))
        model, sample = model.double(), TableSample(table)
        want = temporal.predict(model, normaliser, sample)
        for members in ([(model, normaliser)], [(model, normaliser)] * 3):
            np.testing.assert_allclose(temporal.predict(*soup.soup(members), sample), want,
                                       atol=2e-4)
        # the same function under another normaliser: soup must re-express it exactly
        import copy
        import torch
        shifted = temporal.Normaliser.from_state(dict(normaliser.state(),
                                                      median=normaliser.median + 0.1 * normaliser.scale))
        twin = copy.deepcopy(model)
        k = len(normaliser.median)
        with torch.no_grad():
            w = twin.inp[0].weight
            twin.inp[0].bias -= w[:, :k] @ torch.from_numpy(
                (normaliser.median - shifted.median) / normaliser.scale)
        # exact but for values the +/-10 clip catches under one scaling and not the other
        np.testing.assert_allclose(temporal.predict(twin, shifted, sample), want, atol=1e-2)
        np.testing.assert_allclose(temporal.predict(*soup.soup([(model, normaliser),
                                                                 (twin, shifted)]), sample),
                                   want, atol=1e-2)

    def test_seed_rule_needs_the_ensemble_and_every_seed_to_gain(self):
        from tempolens.adaptseeds import decide
        c = (2.0, 8.0, 8.0, 0.5)
        base = {"sim": 0.7, "shift": 0.7, "sc": 0.4}
        bump = {   # setting -> spellF1 added on every condition, per seed (-1 = ensemble)
            (0.0, 0, False, c): (0, 0, 0),
            (0.5, 0, False, c): (0.002, 0.001, 0.003),     # small, but all gain: accepted
            (1.0, 0, False, c): (0.010, 0.020, -0.001),    # seed 1 loses: rejected
            (0.5, 10, True, c): (0.001, 0.001, 0.0),       # a tie is not a gain: rejected
        }
        summary = {(s,) + k: {cond: v + d[i] for cond, v in base.items()}
                   for k, d in bump.items() for i, s in enumerate((-1, 0, 1))}
        rows, chosen = decide(summary)
        self.assertEqual(len(rows), 3)
        self.assertEqual([r["accepted"] for r in rows], [False, True, False])
        self.assertEqual((chosen["alpha"], chosen["steps"], chosen["gate"]), (0.5, 0, False))
        self.assertAlmostEqual(chosen["gain"]["-1"], 0.002)
        del summary[(-1, 0.5, 0, False, c)], summary[(0, 0.5, 0, False, c)], \
            summary[(1, 0.5, 0, False, c)]
        self.assertIsNone(decide(summary)[1])

    def test_dead_ball_temperature_keeps_the_team_ratio(self):
        from tempolens.cvrescore import soften_dead
        p = np.array([[0.6, 0.2, 0.2], [0.05, 0.05, 0.9]])
        logp = np.c_[np.log(p), [0.3, -1.0]]
        np.testing.assert_allclose(soften_dead(logp, 1.0), logp, atol=1e-12)
        out = soften_dead(logp, 2.0)
        q = np.exp(out[:, :3])
        np.testing.assert_allclose(q.sum(axis=1), 1.0)
        np.testing.assert_allclose(q[:, H] / q[:, A], p[:, H] / p[:, A])
        self.assertTrue(q[0, D] > 0.2 and q[1, D] < 0.9)      # pulled towards 0.5
        np.testing.assert_array_equal(out[:, 3], logp[:, 3])

    def test_keeper_zone_counts_long_stays_in_the_other_goal_area(self):
        from tempolens.diagnostics import keeper_zone
        from tempolens.match import MatchTable
        n = 1000
        players = np.zeros((n, 4, 2), dtype=np.float32)
        players[:, :, 1] = 34.0
        players[:, :, 0] = [10.0, 60.0, 95.0, 40.0]   # home keeper, home, away keeper, away
        players[:200, 1, 0] = 100.0                   # 8 s in the away goal area: counted
        players[500:550, 1, 0] = 100.0                # 2 s: too short
        table = MatchTable("t", np.zeros((n, 2), dtype=np.float32), players,
                           np.array([H, H, A, A], dtype=np.int8), np.ones(n, dtype=np.int8))
        out = keeper_zone(table)
        self.assertEqual(out["home"], {"share_of_graded": 0.2, "slots": 1, "slots_of_team": 2,
                                       "minutes": 0.1})
        self.assertEqual(out["away"]["share_of_graded"], 0.0)

    def test_team_gate_damps_only_home_vs_away_where_the_ball_flies_or_is_loose(self):
        from tempolens.features import frame_features
        from tempolens.predict import gate_team_evidence
        table = SimulatorTests()._labelled(2000)
        rng = np.random.default_rng(0)
        p = rng.dirichlet([1, 1, 1], size=2000)
        logp = np.column_stack([np.log(p), rng.normal(size=2000)])
        out = gate_team_evidence(logp, frame_features(table)[0], flight=0.0, loose=0.0)
        same = np.isclose(out[:, :3], logp[:, :3], atol=1e-9).all(axis=1)
        damped = ~same
        self.assertTrue(damped.any() and same.any())
        np.testing.assert_allclose(out[damped, 0], out[damped, 1])          # factor 0: H = A
        np.testing.assert_allclose(np.exp(out[:, :2]).sum(axis=1), p[:, :2].sum(axis=1), rtol=1e-6)
        np.testing.assert_array_equal(out[:, 2:], logp[:, 2:])              # D and change kept

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
                   temporal.TemporalConfig(hidden=16, chunk=1500),
                   temporal.TemporalConfig(hidden=16, touch=0.5),
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

    def test_submitted_members_load_as_default_gru(self):
        from tempolens import temporal
        from tempolens.predict import default_models
        paths = default_models()
        self.assertEqual(len(paths), 5)
        for path in paths:      # committed, current feature version, no feature masked
            model, normaliser = temporal.load(path)
            self.assertEqual(model.config, temporal.DEFAULT_CONFIG)
            self.assertEqual(len(normaliser.drop), 0)


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


class ReviewStudyTests(unittest.TestCase):
    def test_split_bonus_with_equal_weights_is_the_plain_decoder(self):
        from tempolens.decoder import transition_matrix
        from tempolens.reviewstudy import decode_split
        table = _table()
        rng = np.random.default_rng(3)
        logp = np.concatenate([np.log(rng.dirichlet([1, 1, 1], len(table.period))),
                               rng.normal(0, 2, (len(table.period), 1))], axis=1)
        t = transition_matrix(2.0, 8.0, 8.0)
        np.testing.assert_array_equal(decode_split(table, logp, t, 0.5, 0.5),
                                      decode_table(table, logp, t, 0.5))
        self.assertFalse(np.array_equal(decode_split(table, logp, t, 3.0, 0.0),
                                        decode_split(table, logp, t, 0.0, 3.0)))

    def test_geometric_combination(self):
        from tempolens.reviewstudy import combine
        from tempolens.temporal import combine as plain
        rng = np.random.default_rng(4)
        one = np.concatenate([np.log(rng.dirichlet([1, 1, 1], 50)), rng.normal(size=(50, 1))], 1)
        np.testing.assert_allclose(combine([one, one], geometric=True), np.round(one, 4),
                                   atol=2e-4)
        two = one.copy()
        two[:, 3] += 1.0
        np.testing.assert_allclose(combine([one, two]), plain([one, two]))
        np.testing.assert_allclose(combine([one, two], max_change=True)[:, 3],
                                   np.round(two[:, 3], 4))


if __name__ == "__main__":
    unittest.main()
