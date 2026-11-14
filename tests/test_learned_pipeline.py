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
        from tempolens import temporal
        from tempolens.studies import soup
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
        from tempolens.studies.adaptseeds import decide
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
        paths = default_models()      # LCB-ADOPT: ENS5 then the 5 ms50 finals
        self.assertEqual([p.name for p in paths],
                         ["tempolens_{}_m{}.pt".format(kind, k) for kind in ("rnd4", "ms50")
                          for k in range(5)])
        for path in paths:      # committed, current feature version, no feature masked
            model, normaliser = temporal.load(path)
            self.assertEqual(model.config, temporal.DEFAULT_CONFIG)
            self.assertEqual(len(normaliser.drop), 0)

    def test_load_refuses_a_pickle_that_runs_code(self):
        import pickle
        import tempfile
        from pathlib import Path
        import torch
        from tempolens import temporal

        class Payload:      # unpickling this calls print: the weights_only allowlist must refuse it
            def __reduce__(self):
                return (print, ("loaded code from a .pt file",))

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "m.pt"
            torch.save({"normaliser": Payload()}, path)
            with self.assertRaises(pickle.UnpicklingError):
                temporal.load(path)


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
        # an exact tie in real numbers that rounds differently if the additions are reordered
        logits = np.array([[-12.6943, -7.6727, -50.0], [0.0, -50.0, -50.0]])
        transition, bonus = decoder.transition_matrix(4.0, 8.0, 8.0), np.array([0.0, -1.0216])
        with patch.object(decoder, "_kernel", None):
            np.testing.assert_array_equal(decoder.viterbi_matrix(logits, transition, bonus), [1, 0])
        if decoder._kernel is not None:
            np.testing.assert_array_equal(decoder.viterbi_matrix(logits, transition, bonus), [1, 0])

    def test_decode_writes_x_outside_periods_and_respects_evidence(self):
        table = _table()
        logp = np.log(np.tile([0.8, 0.1, 0.1], (table.n_frames, 1)))
        logp[120:190] = np.log([0.1, 0.8, 0.1])
        out = decode_table(table, logp, np.zeros((3, 3)))
        self.assertTrue(np.all(out[table.period == 0] == X))
        self.assertTrue(np.all(out[10:120] == H) and np.all(out[120:190] == A))


class WeightAveragingTests(unittest.TestCase):
    def test_swa_averages_the_last_epochs(self):
        import torch
        from tempolens import temporal
        table = SimulatorTests()._labelled(3000)
        seen = []
        keep = lambda e, m, n: seen.append({k: v.detach().double().cpu().clone()
                                            for k, v in m.state_dict().items()})
        config = temporal.TemporalConfig(hidden=16, swa=2)
        model, _ = temporal.train([table], epochs=3, device="cpu", verbose=False, config=config,
                                  on_epoch=keep)
        for k, v in model.state_dict().items():
            torch.testing.assert_close(v.double(), (seen[1][k] + seen[2][k]) / 2,
                                       atol=1e-6, rtol=0)
        self.assertEqual(config.tag(), "temporal_h16_swa2")


class HedgeHeadTests(unittest.TestCase):
    def test_hedge_head_trains_on_controlled_labels_and_is_not_decoded(self):
        from tempolens import temporal
        from tempolens.features import TableSample
        from tempolens.opendata import remove_flicker
        table = SimulatorTests()._labelled(3000)
        config = temporal.TemporalConfig(hidden=16, hedge=0.5)
        self.assertEqual(config.tag(), "temporal_h16_hd0.5")
        seqs, _ = temporal._sequences([table], config)
        sample = TableSample(table)
        y2 = remove_flicker(sample.labels, sample.runs(), temporal.HEDGE_FRAMES)
        start, stop = sample.runs()[0]
        expect = y2[start:stop].astype(np.int64)
        expect[expect == X] = -100
        np.testing.assert_array_equal(seqs[0]["y2"], expect)
        model, normaliser = temporal.train([table], epochs=1, device="cpu", verbose=False,
                                           config=config)
        self.assertEqual(model.head.out_features, 4)
        self.assertEqual(temporal.predict(model, normaliser, sample).shape[1], 4)


class LabelCalibrationTests(unittest.TestCase):
    def test_order_check_and_verdict(self):
        from tempolens.labelcal import (MACRO_ORDER, SPELL_ORDER, in_order, verdict)
        published_spell = {"raw": 0.0486, "smooth25": 0.1996, "smooth50": 0.2433,
                           "smooth125": 0.2242}
        published_macro = {"raw": 0.6532, "smooth25": 0.6752, "smooth50": 0.6724,
                           "smooth125": 0.6356}
        self.assertTrue(in_order(published_spell, SPELL_ORDER))
        self.assertTrue(in_order(published_macro, MACRO_ORDER))
        self.assertFalse(in_order(dict(published_spell, smooth125=0.19), SPELL_ORDER))
        self.assertFalse(in_order(dict(published_spell, smooth125=0.2433), SPELL_ORDER))
        self.assertEqual(verdict([])[0], None)
        self.assertEqual(verdict([1, 3])[0], None)          # 25 and 50: not adjacent
        self.assertEqual(verdict([2, 3, 4])[0], 37)         # adjacent: the smallest
        self.assertEqual(verdict([3])[0], 50)


class PffLoaderTests(unittest.TestCase):
    def _events(self):
        def ev(t, kind, team=1, piece=None, end=None):
            return {"startTime": t, "endTime": t if end is None else end,
                    "gameEvents": {"gameEventType": kind, "teamId": team, "setpieceType": piece}}
        return [ev(2.0, "FIRSTKICKOFF", 1, "K"), ev(4.0, "OTB", 2, "O"),     # away gains it
                ev(6.0, "OUT", 2), ev(7.0, "SUB", 1), ev(8.0, "OTB", 1, "T"),  # throw-in: home
                ev(9.0, "OTB", 1, "O", end=9.5), ev(10.0, "OFF", 2),
                ev(11.0, "OTB", 2, "F")]                                       # foul, free kick

    def test_labels_from_events(self):
        from tempolens.opendata import pff_labels
        labels, untagged = pff_labels(self._events(), 13, 1.0, home_id=1)
        #                frame: 0  1  2  3  4  5  6  7  8  9  10 11 12
        np.testing.assert_array_equal(labels, [D, D, H, H, A, A, D, D, H, H, D, A, A])
        self.assertEqual(untagged, 0)

    def test_test_subset_is_every_fourth_id(self):
        from tempolens.pffstudy import test_ids
        ids = [str(i) for i in range(10502, 10518)] + [str(i) for i in range(3812, 3860)]
        sub = test_ids(ids)
        self.assertEqual(len(sub), 16)
        self.assertEqual(sub[:3], ["3812", "3816", "3820"])
        self.assertEqual(sub[-4:], ["10502", "10506", "10510", "10514"])

    def test_team_ids_match_across_types_and_folds_change_labels(self):
        from tempolens.opendata import pff_labels, remove_flicker
        events = [{"startTime": t, "endTime": t, "gameEvents": {
            "gameEventType": k, "teamId": team, "setpieceType": piece}}
            for t, k, team, piece in ((0, "FIRSTKICKOFF", 372, "K"), (40, "OTB", 366, "O"),
                                      (70, "OTB", 372, "O"), (200, "OUT", 372, None))]
        labels, _ = pff_labels(events, 220, 1.0, home_id="372")   # events int, metadata str
        self.assertEqual(set(labels[:200].tolist()), {H, A})
        runs = [(0, 220)]
        folded10, folded50 = (remove_flicker(labels, runs, f) for f in (10, 50))
        np.testing.assert_array_equal(folded10, labels)          # the 30-frame away spell stays
        self.assertTrue((folded50[40:70] == H).all())             # folded at 50 frames
        self.assertFalse(np.array_equal(folded10, folded50))

    def test_loader_on_a_tiny_match(self):
        import bz2
        import json
        import tempfile
        from pathlib import Path
        from tempolens.opendata import pff_game
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for sub in ("metadata", "rosters", "tracking_data", "event_data/day"):
                (root / sub).mkdir(parents=True)
            fps = 30.0
            (root / "metadata" / "9.json").write_text(json.dumps([{
                "fps": fps, "homeTeam": {"id": 1, "name": "Home"},
                "awayTeam": {"id": 2, "name": "Away"},
                "stadium": {"pitches": [{"length": 105.0, "width": 68.0}]}}]))
            (root / "rosters" / "9.json").write_text(json.dumps(
                [{"team": {"id": 1}, "shirtNumber": "7"}, {"team": {"id": 2}, "shirtNumber": "4"}]))
            events = [{"startTime": e["startTime"] * 2, "endTime": e["endTime"] * 2,
                       "gameEvents": e["gameEvents"]} for e in self._events()]
            (root / "event_data" / "day" / "9.json").write_text(json.dumps(events))
            rows = []
            for f in range(int(30 * fps)):
                row = {"frameNum": f, "period": 1 if f >= 4 * fps else None,
                       "ballsSmoothed": None if f % 10 == 0 else {
                           "x": 1.0, "y": -2.0, "z": 0,
                           "visibility": "ESTIMATED" if f % 10 == 5 else "VISIBLE"},
                       "homePlayersSmoothed": [{"jerseyNum": 7, "x": 0.0, "y": 0.0}],
                       "awayPlayersSmoothed": [{"jerseyNum": 4, "x": 10.0, "y": 5.0},
                                               {"jerseyNum": 99, "x": 1.0, "y": 1.0}]}
                rows.append(json.dumps(row))
            with bz2.open(root / "tracking_data" / "9.jsonl.bz2", "wt") as handle:
                handle.write(chr(10).join(rows))
            table = pff_game("9", folder=root)
        self.assertEqual(table.n_frames, 25 * 30)
        np.testing.assert_array_equal(table.slot_team, [H, A])
        np.testing.assert_allclose(table.players[200, 0], [52.5, 34.0])
        np.testing.assert_allclose(table.players[200, 1], [62.5, 39.0])
        self.assertTrue(np.isnan(table.ball[0]).all())
        np.testing.assert_allclose(table.ball[101], [53.5, 32.0])
        self.assertTrue(np.isnan(table.ball[int(round(35 / 1.2))]).all())   # frame 35: estimated
        self.assertTrue((table.labels[:100] == X).all())      # before the period: ungraded
        self.assertEqual(table.labels[25 * 5], H)              # 5 s: home kicked off at 4 s
        self.assertEqual(table.labels[25 * 9], A)
        self.assertEqual(table.labels[25 * 13], D)             # out at 12 s, throw-in at 16 s
        self.assertEqual(table.labels[25 * 17], H)
        self.assertEqual(table.meta["kickoff_offset_s"], 0.0)


class ReviewStudyTests(unittest.TestCase):
    def test_split_bonus_with_equal_weights_is_the_plain_decoder(self):
        from tempolens.decoder import transition_matrix
        from tempolens.studies.reviewstudy import decode_split
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
        from tempolens.studies.reviewstudy import combine
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


class AtomicWriteTests(unittest.TestCase):
    def test_caches_are_replaced_whole_and_leave_no_temporary_file(self):
        import tempfile
        from pathlib import Path
        from tempolens.experiments import savez_atomic, write_atomic
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rows.json"
            write_atomic(path, "old")
            write_atomic(path, "[1, 2]")
            store = Path(folder) / "out.npz"
            savez_atomic(store, a=np.arange(3, dtype=np.float32))
            self.assertEqual(path.read_text(encoding="utf-8"), "[1, 2]")
            np.testing.assert_array_equal(np.load(store)["a"], [0, 1, 2])
            self.assertEqual(sorted(p.name for p in Path(folder).iterdir()), ["out.npz", "rows.json"])


class FinishLabelsTests(unittest.TestCase):
    BLOCKS = [(1, 2, 5), (2, 8, 11)]     # graded video frames 2..5 and 8..11 of 14

    def test_x_outside_the_halves_and_d_before_the_second_kickoff(self):
        from tempolens.predict import finish_labels
        states = np.array([0, 1, 0, 1, 2, 0, 3, 3, 1, 0, 1, 0, 1, 0])
        labels = finish_labels(states, self.BLOCKS, 10, 14)
        self.assertEqual("".join(labels), "XXHADHXXDDAHXX")
        self.assertEqual("".join(finish_labels(states, self.BLOCKS, None, 14)), "XXHADHXXAHAHXX")

    def test_contract_violations_raise(self):
        from tempolens.predict import finish_labels
        states = np.zeros(14, dtype=int)
        with self.assertRaises(RuntimeError):
            finish_labels(states, self.BLOCKS, None, 15)       # wrong length
        states[3] = 3
        with self.assertRaises(RuntimeError):
            finish_labels(states, self.BLOCKS, None, 14)       # X on a graded frame


class SwaAdoptTests(unittest.TestCase):
    @staticmethod
    def row(ens, gains, sc, ok=True):
        return {"ensemble": ens, "accepted": ok, "gain": dict({"-1": 0.0}, **gains),
                "conditions": {"-1": {"sc": sc}}}

    def test_pick_takes_the_highest_accepted_and_breaks_near_ties_by_the_weakest_member(self):
        from tempolens.studies.swaadopt import pick
        a = self.row(0.6476, {"0": 0.0001, "1": 0.003}, 0.010)
        b = self.row(0.6475, {"0": 0.0022, "1": 0.002}, 0.012)
        c = self.row(0.6490, {"0": 0.01, "1": 0.01}, 0.02, ok=False)
        self.assertIs(pick([a, b, c]), b)                 # within 0.0001: weakest member decides
        far = self.row(0.6470, {"0": 0.0022, "1": 0.002}, 0.012)
        self.assertIs(pick([far, a]), a)                  # a clear winner on the ensemble
        self.assertIsNone(pick([c]))

    def test_train_refuses_the_submitted_weights(self):
        from tempolens import train
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as folder:      # the guard's name, never the tracked file
            for name in ("tempolens_rnd4_m0.pt", "tempolens_ms50_m3.pt"):   # both halves of the mix
                out = Path(folder) / name
                with self.assertRaises(SystemExit):
                    train.main(["--out", str(out)])
                self.assertFalse(out.exists())

    def test_dead_weight_refuses_the_losses_that_would_drop_it(self):
        from tempolens import temporal
        table = SimulatorTests()._labelled(3000)
        for extra in ({"focal": 2.0}, {"seg_balance": 1.0}):
            config = temporal.TemporalConfig(hidden=8, dead_weight=2.0, **extra)
            with self.assertRaisesRegex(ValueError, "dead_weight"):
                temporal.train([table], epochs=1, verbose=False, config=config)
        temporal.train([table], epochs=1, verbose=False,          # alone it is fine
                       config=temporal.TemporalConfig(hidden=8, dead_weight=2.0))


class StudyReferenceTests(unittest.TestCase):
    """The B2 and B4 rules compare with this run's computed references, not rounded literals."""

    def test_b2_uses_the_computed_submission_and_no_adaptation_sim(self):
        import contextlib
        import io
        import itertools
        from unittest.mock import patch
        from tempolens.studies import hedgestudy as hs

        def table(blend_mean, blend_sim):
            t = {}
            for v, s, c in itertools.product(hs.VARIANTS, range(-1, 5), hs.B2_COSTS):
                t[(v, s, c)] = {"sim": 0.7700, "shift": 0.70, "sc": 0.40}
                if v == "tta+eata":                       # the submission: mean 0.6480
                    t[(v, s, c)] = {"sim": 0.7700, "shift": 0.7740, "sc": 0.40}
                if v == "blend tta":
                    shift = 3 * blend_mean - blend_sim - 0.40
                    t[(v, s, c)] = {"sim": blend_sim, "shift": shift, "sc": 0.40}
            return t

        def verdict(t):
            out = io.StringIO()
            with patch.object(hs, "collect", return_value=t), contextlib.redirect_stdout(out):
                hs.b2(1)
            return out.getvalue().strip().splitlines()[-1]

        # above the rounded 0.6475 but below this run's submission (0.6480): fails
        self.assertEqual(verdict(table(0.6477, 0.7800)), "B2: fails")
        # sim below the rounded 0.7727 but above this run's no-adaptation sim (0.7700): passes
        self.assertTrue(verdict(table(0.6490, 0.7710)).startswith("B2: passes"))

    def test_b4_gain_is_over_the_computed_equal_weights_score(self):
        import contextlib
        import io
        from unittest.mock import patch
        from tempolens.studies import reviewstudy as rs
        full = np.full(len(rs.SPLIT), 0.7000)
        e = next(i for i, s in enumerate(rs.SPLIT) if s[3] == s[4])
        b = next(i for i, s in enumerate(rs.SPLIT) if s[3] != s[4])
        full[e], full[b] = 0.7762, 0.7801            # +0.0041 over 0.7760, +0.0039 over 0.7762

        class Pool:
            @staticmethod
            def map(_, jobs):
                return [full for _ in jobs]

        outputs = {n: np.zeros((4, 4)) for n in rs.NAMES}
        out = io.StringIO()
        with patch.object(rs, "members", return_value=[outputs] * 5):
            with contextlib.redirect_stdout(out):
                rs.b4(Pool())
        self.assertTrue(out.getvalue().strip().splitlines()[-1].startswith("B4: fails (+0.0039"))


class CacheWriteTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)

    def tearDown(self):
        self.folder.cleanup()

    def sample(self):
        from unittest.mock import patch
        from tempolens.experiments import CachedSample
        sample = CachedSample.__new__(CachedSample)
        root = self.root
        return sample, patch.object(CachedSample, "_path", lambda self, swap, kind="f":
                                    root / "x_{}_{}.npy".format(kind, int(swap)))

    def test_a_final_written_meanwhile_by_another_process_is_kept(self):
        from unittest.mock import patch
        import tempolens.experiments as ex
        sample, path = self.sample()
        theirs = self.root / "x_k_1.npy"

        def compute(side):
            if side:                               # another process finishes side 1 first
                np.save(theirs, np.full(3, 7.0))
            return np.full(3, float(side))
        with path, patch.object(ex, "_replace_retry", wraps=ex._replace_retry) as replace:
            out = sample._cached("k", False, compute)
            np.testing.assert_array_equal(out, [0.0, 0.0, 0.0])
            self.assertEqual(replace.call_count, 1)          # side 0 only: side 1 skipped
            np.testing.assert_array_equal(np.load(theirs), [7.0, 7.0, 7.0])
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["x_k_0.npy", "x_k_1.npy"])

    def test_partial_and_tmp_names_carry_the_process_id(self):
        import os
        from unittest.mock import patch
        import tempolens.experiments as ex
        sample, path = self.sample()
        seen = []
        with path, patch.object(ex, "_replace_retry",
                                side_effect=lambda s, d: (seen.append(s.name), os.replace(s, d))):
            sample._cached("k", True, lambda side: np.zeros(2))
            ex.write_atomic(self.root / "t.json", "{}")
        pid = str(os.getpid())
        self.assertEqual(seen, ["x_k_0.{}.partial.npy".format(pid),
                                "x_k_1.{}.partial.npy".format(pid), "t.json.{}.tmp".format(pid)])

    def test_match_table_save_round_trips_through_the_atomic_writer(self):
        from unittest.mock import patch
        import tempolens.experiments as ex
        from tempolens.match import MatchTable
        table = _table()
        with patch.object(ex, "write_atomic", wraps=ex.write_atomic) as write:
            table.save(self.root / "m.npz")
            self.assertEqual(write.call_count, 1)
        back = MatchTable.load(self.root / "m.npz")
        np.testing.assert_array_equal(back.ball, table.ball)
        np.testing.assert_array_equal(back.period, table.period)
        self.assertEqual([p.name for p in self.root.iterdir()], ["m.npz"])


class LoaderVersionTests(unittest.TestCase):
    def test_table_cache_and_clean_features_name_the_loader_version(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        import tempolens.opendata as od
        from tempolens.experiments import CachedSample
        with tempfile.TemporaryDirectory() as folder, \
                patch.dict(od.LOADERS, {"fake": lambda: _table()}), \
                patch.object(od, "LOADER_VERSION", 7):
            od.load_open_match("fake", cache_dir=folder)
            self.assertEqual([p.name for p in Path(folder).iterdir()], ["fake_l7.npz"])
        sample = CachedSample.__new__(CachedSample)
        sample.name, sample.seed, sample.keeper, sample.repair = "x", None, False, None
        with patch("tempolens.experiments.open_data_dir", return_value=Path(".")):  # fresh clone
            self.assertIn("_clean_l{}_".format(od.LOADER_VERSION), sample._path(False).name)


class TrackerBallTests(unittest.TestCase):
    def test_smoothing_handles_runs_shorter_than_the_kernel(self):
        from tempolens.simulate import AtdNoise, _tracker_ball
        table = _table(n=40)
        table.period[:] = 0
        table.period[5:15] = 1                         # a 10-frame run, kernel is 25 frames
        table.period[20:40] = 2
        table.ball[8] = np.nan
        noise = AtdNoise(ball_noise_m=0.0)
        out = _tracker_ball(table, noise, np.random.default_rng(0))
        self.assertEqual(out.shape, table.ball.shape)
        self.assertTrue(np.isnan(out[8, 0]))
        seg, ok = table.ball[5:15].astype(np.float64), np.isfinite(table.ball[5:15, 0])
        half = int(4 * noise.ball_smooth_sd)
        kernel = np.exp(-0.5 * (np.arange(-half, half + 1) / noise.ball_smooth_sd) ** 2)
        for i in np.flatnonzero(ok):                  # each frame: the weighted mean of its run
            w = kernel[half - i:half - i + 10][ok]
            np.testing.assert_allclose(out[5 + i], (w[:, None] * seg[ok]).sum(0) / w.sum(),
                                       rtol=1e-5)

    def test_full_slice_equals_same_on_runs_of_the_kernel_length_or_more(self):
        rng = np.random.default_rng(3)
        half = 12
        kernel = np.exp(-0.5 * (np.arange(-half, half + 1) / 3.0) ** 2)
        for n in (25, 26, 27, 50, 333, 4000):
            x = rng.normal(size=n)
            np.testing.assert_array_equal(np.convolve(x, kernel, mode="full")[half:half + n],
                                          np.convolve(x, kernel, mode="same"))


class HalfSplitTests(unittest.TestCase):
    def test_back_to_back_halves_each_start_dead_until_their_kickoff(self):
        period = np.array([0] * 5 + [1] * 20 + [2] * 20 + [0] * 5)
        events = _events([(H, "SET PIECE", "KICK OFF", 6, 6), (H, "PASS", "", 10, 12),
                          (H, "PASS", "", 22, 24),           # first half runs to its end
                          (A, "PASS", "", 28, 30),           # second half: before its kickoff
                          (A, "SET PIECE", "KICK OFF", 32, 32), (A, "PASS", "", 35, 37)])
        labels = labels_from_events(events, period, 50)
        self.assertTrue(np.all(labels[period == 0] == X))
        self.assertTrue(np.all(labels[6:25] == H))
        self.assertTrue(np.all(labels[25:32] == D))          # dead from the half's first frame
        self.assertTrue(np.all(labels[32:38] == A))
        self.assertTrue(np.all(labels[38:45] == D))          # after the half's last event
        # the same with a gap between the halves gives the same labels inside them
        gap = period.copy()
        gap[25] = 0
        again = labels_from_events(events, gap, 50)
        self.assertEqual(again[25], X)
        np.testing.assert_array_equal(again[26:], labels[26:])


class AdaptMembersTests(unittest.TestCase):
    def test_predict_adaptation_equals_the_study_cell(self):
        import copy
        from tempolens import adaptcv, temporal
        from tempolens.features import TableSample
        from tempolens.predict import adapt_members
        table = SimulatorTests()._labelled(3000)
        model, normaliser = temporal.train([table], epochs=1, device="cpu", verbose=False,
                                           config=temporal.TemporalConfig(hidden=16))
        import torch
        model = model.double()
        with torch.no_grad():                # a sharper head, so EATA finds confident frames
            model.head.weight.mul_(30.0)
            model.head.bias.mul_(30.0)
        other = SimulatorTests()._labelled(3000)
        other.players = other.players * 0.8 + 5.0      # another match: other medians for TTA
        other.ball = other.ball * 0.8 + 5.0
        sample = TableSample(other)
        grid = adaptcv.member_grid([(model, normaliser)], sample, double=True,
                                   alphas=(0.0, 0.5), steps=(0, 10))
        for alpha, steps in ((0.5, 10), (0.5, 0), (0.0, 10), (0.0, 0)):
            (m, n), = adapt_members([(copy.deepcopy(model), normaliser)], sample, alpha, steps)
            np.testing.assert_allclose(temporal.predict(m, n, sample), grid[(alpha, steps)][0],
                                       atol=2e-4)
        for key in ((0.5, 0), (0.0, 10)):    # both adaptations really move the output
            self.assertGreater(np.abs(grid[key][0] - grid[(0.0, 0)][0]).max(), 1e-3, key)


class StudyMembersTests(unittest.TestCase):
    def test_a_cache_folder_refuses_other_members(self):
        import tempfile
        from pathlib import Path
        from tempolens.experiments import check_members
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            a, b = root / "a.pt", root / "b.pt"
            a.write_bytes(b"first")
            b.write_bytes(b"second")
            cache = root / "cache"
            first = check_members(cache, [a, b])                 # first use: recorded
            self.assertTrue((cache / "members.json").exists())
            self.assertEqual(check_members(cache, [a, b]), first)   # the same: fine
            with self.assertRaises(RuntimeError):
                check_members(cache, [b, a])                     # other order
            with self.assertRaises(RuntimeError):
                check_members(cache, [a])                        # fewer members
            b.write_bytes(b"retrained")                          # same name, other weights
            with self.assertRaises(RuntimeError):
                check_members(cache, [a, b])

    def test_collect_checks_the_sc_members_before_scoring(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from tempolens import adaptcv
        from tempolens.experiments import check_members
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            old, new = root / "old.pt", root / "new.pt"
            old.write_bytes(b"old")
            new.write_bytes(b"new")
            check_members(root, [old])
            with patch.object(adaptcv.temporal, "load", side_effect=AssertionError("loaded")):
                with self.assertRaises(RuntimeError):
                    adaptcv.collect(1, root, final_paths=[new])


class KickoffPriorTests(unittest.TestCase):
    BLOCKS = [(1, 2, 5), (2, 8, 11)]

    def test_forces_dead_from_the_half_start_up_to_the_kickoff_only(self):
        from tempolens.predict import kickoff_prior
        logp = np.log(np.tile([0.7, 0.2, 0.1, 0.0], (14, 1)))
        logp[:, 3] = 0.25                                # the change logit is not a probability
        out = kickoff_prior(logp, self.BLOCKS, 10)
        dead = np.log([1e-6, 1e-6, 1.0])
        np.testing.assert_array_equal(out[8:10, :3], np.tile(dead, (2, 1)))
        np.testing.assert_array_equal(out[:8], logp[:8])         # first half and the gap
        np.testing.assert_array_equal(out[10:], logp[10:])       # the kickoff frame itself on
        np.testing.assert_array_equal(out[:, 3], logp[:, 3])
        self.assertFalse(np.shares_memory(out, logp))
        np.testing.assert_array_equal(kickoff_prior(logp, self.BLOCKS, None), logp)
        np.testing.assert_array_equal(kickoff_prior(logp, self.BLOCKS, 8), logp)   # at the start

    def test_the_decoder_then_enters_the_half_dead(self):
        from tempolens.decoder import transition_matrix, viterbi_matrix
        from tempolens.predict import kickoff_prior
        logp = np.log(np.tile([0.6, 0.3, 0.1, 0.0], (14, 1)))[:, :4]
        out = kickoff_prior(logp, self.BLOCKS, 10)
        states = viterbi_matrix(out[8:12, :3], transition_matrix(4.0, 8.0, 8.0))
        np.testing.assert_array_equal(states[:2], [D, D])


class BothViewsTests(unittest.TestCase):
    def test_both_views_does_not_depend_on_which_club_is_home(self):
        from tempolens.features import TableSample
        from tempolens.temporal import Normaliser, adapt_normaliser
        sample = TableSample(_table(400))

        class Swapped:                        # the same match with the other club as home
            period, n_frames = sample.period, sample.n_frames

            @staticmethod
            def features(swap):
                return sample.features(not swap)

        base = Normaliser(np.asarray(sample.features(False))[sample.period > 0])
        one, other = (adapt_normaliser(base, s, 1.0, both_views=True) for s in (sample, Swapped))
        np.testing.assert_array_equal(one.median, other.median)
        np.testing.assert_array_equal(one.scale, other.scale)
        home, away = (adapt_normaliser(base, s, 1.0) for s in (sample, Swapped))
        self.assertFalse(np.array_equal(home.median, away.median))     # the default does depend
        # the default is the submitted recipe, unchanged
        own = Normaliser(np.asarray(sample.features(False))[sample.period > 0][::10])
        np.testing.assert_array_equal(home.median, own.median)


class NoisyCacheNameTests(unittest.TestCase):
    def test_noisy_feature_names_carry_a_bumped_loader_version(self):
        from unittest.mock import patch
        import tempolens.experiments as ex
        from pathlib import Path
        names = {}
        for version in (1, 2):
            with patch.object(ex, "LOADER_VERSION", version),                     patch.object(ex, "open_data_dir", return_value=Path(".")):    # fresh clone
                for kind in ("sim", "rnd", "rnf", "clean"):
                    s = ex.CachedSample.__new__(ex.CachedSample)
                    s.name, s.keeper, s.repair = "x", False, None
                    s.seed = None if kind == "clean" else 7
                    s.randomised, s.faults = kind in ("rnd", "rnf"), kind == "rnf"
                    names[(version, kind)] = s._path(False).name
        for kind in ("sim", "rnd", "rnf"):
            self.assertNotIn("_l1", names[(1, kind)])           # today's names, nothing recomputed
            self.assertIn("_l2_", names[(2, kind)])
            self.assertNotEqual(names[(1, kind)], names[(2, kind)])
        self.assertIn("_clean_l1_", names[(1, "clean")])
        self.assertIn("_clean_l2_", names[(2, "clean")])
        self.assertEqual(len(set(names.values())), 8)


class TableBuildTests(unittest.TestCase):
    def test_a_cache_miss_builds_the_noisy_table_once(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from tempolens.experiments import CachedSample
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            sample = CachedSample.__new__(CachedSample)
            calls = []
            with patch.object(CachedSample, "_path", lambda self, swap, kind="f":
                              root / "x_{}_{}.npy".format(kind, int(swap))), \
                    patch.object(CachedSample, "table",
                                 lambda self: calls.append(1) or _table()):
                for build in (sample.features, sample.tokens, sample.teamcue):
                    calls.clear()
                    build(False)
                    self.assertEqual(len(calls), 1, build.__name__)    # both views, one table
                    build(True)
                    self.assertEqual(len(calls), 1, build.__name__)    # a hit builds none


class CompleteRowsTests(unittest.TestCase):
    @staticmethod
    def rows(alphas, seed=None):
        out = [{"alpha": a, "steps": 0, "gate": False, "costs": [4.0, 8.0, 8.0, 1.0],
                "spellF1": 0.5 + a, "macroF1": 0.6} for a in alphas]
        return [dict(r, seed=seed) for r in out] if seed is not None else out

    def test_a_setting_missing_any_match_or_condition_raises(self):
        from tempolens import adaptcv
        from tempolens.studies import adaptseeds
        full = {"sim": {"m1": self.rows([0.0, 0.5]), "m2": self.rows([0.0, 0.5])},
                "sc": {"s1": self.rows([0.0, 0.5])}}
        self.assertEqual(len(adaptcv.table(full)), 2)
        sim_only = {"sim": full["sim"], "sc": {"s1": self.rows([0.0])}}    # 0.5 on sim only
        part = {"sim": {"m1": self.rows([0.0, 0.5]), "m2": self.rows([0.0])},
                "sc": full["sc"]}                                      # 0.5 missing m2
        for broken in (sim_only, part):
            with self.assertRaises(ValueError):
                adaptcv.table(broken)
        seeded = {c: {n: self.rows([0.0, 0.5], -1) + self.rows([0.0, 0.5], 0)
                      for n in m} for c, m in full.items()}
        self.assertEqual(len(adaptseeds.table(seeded)), 4)
        seeded["sc"]["s1"] = self.rows([0.0, 0.5], -1) + self.rows([0.0], 0)
        with self.assertRaises(ValueError):
            adaptseeds.table(seeded)

    def test_the_default_folder_names_the_swa_models(self):
        from unittest.mock import patch
        from tempolens import adaptcv
        seen = []

        class Stop(Exception):
            pass

        def check(folder, paths):
            seen.append(folder)
            raise Stop                        # the folder is all this test needs

        with patch.object(adaptcv, "check_members", side_effect=check):
            for swa in (0, 4):
                with self.assertRaises(Stop):
                    adaptcv.collect(5, final_paths=["x.pt"], swa=swa)
        self.assertEqual([f.name for f in seen], ["m5", "m5_swa4"])


class SingleTrainerTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)

    def tearDown(self):
        self.folder.cleanup()

    def test_a_held_lock_makes_the_job_wait_for_the_weights_not_train(self):
        from unittest.mock import patch
        from tempolens import adaptcv
        path = self.root / "f_m0.pt"
        (self.root / "f_m0.pt.lock").write_text("123")            # another job is training
        waits = []

        def sleep(_):
            waits.append(1)
            if len(waits) == 3:
                path.write_bytes(b"weights")                      # ... and finishes
        with patch.object(adaptcv.time, "sleep", side_effect=sleep), \
                patch.object(adaptcv.subprocess, "run", side_effect=AssertionError("trained")):
            self.assertEqual(adaptcv.train_once(path, ["train"], wait_s=1), path)
        self.assertEqual(len(waits), 3)
        self.assertTrue((self.root / "f_m0.pt.lock").exists())     # still the other job's

    def test_the_lock_is_released_after_training_or_a_failure(self):
        from unittest.mock import patch
        from tempolens import adaptcv
        path = self.root / "g_m0.pt"
        with patch.object(adaptcv.subprocess, "run",
                          side_effect=lambda cmd, check: path.write_bytes(b"w")) as run:
            adaptcv.train_once(path, ["train"])
            adaptcv.train_once(path, ["train"])                   # exists: no second training
        self.assertEqual(run.call_count, 1)
        other = self.root / "h_m0.pt"
        with patch.object(adaptcv.subprocess, "run", side_effect=OSError("crashed")):
            with self.assertRaises(OSError):
                adaptcv.train_once(other, ["train"])
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["g_m0.pt"])

    def test_a_stale_lock_is_reported(self):
        from unittest.mock import patch
        from tempolens import adaptcv
        path = self.root / "s_m0.pt"
        (self.root / "s_m0.pt.lock").write_text("1")
        with patch.object(adaptcv.time, "sleep"):
            with self.assertRaises(RuntimeError):
                adaptcv.train_once(path, ["train"], wait_s=10, timeout_s=30)

    def test_an_interrupted_save_leaves_no_final_file(self):
        from unittest.mock import patch
        from tempolens import temporal
        table = SimulatorTests()._labelled(3000)
        model, normaliser = temporal.train([table], epochs=1, device="cpu", verbose=False,
                                           config=temporal.TemporalConfig(hidden=8))
        torch = temporal._torch()
        real = torch.save

        def half(obj, f):
            open(f, "wb").write(b"partial")
            raise KeyboardInterrupt
        out = self.root / "m.pt"
        with patch.object(torch, "save", side_effect=half):
            with self.assertRaises(KeyboardInterrupt):
                temporal.save(model, normaliser, out)
        self.assertEqual(list(self.root.iterdir()), [])
        temporal.save(model, normaliser, out)
        self.assertEqual([p.name for p in self.root.iterdir()], ["m.pt"])
        temporal.load(out)
        self.assertIs(torch.save, real)

    def test_the_queue_sees_any_tempolens_job_but_itself(self):
        from unittest.mock import patch
        from tempolens import autoqueue
        lines = ["python -m tempolens.autoqueue", "python -u -m tempolens.studies.swaadopt study",
                 "python -m tempolens.train --member 1", "python -m pytest tests"]
        with patch.object(autoqueue, "python_processes", return_value=lines):
            self.assertEqual(autoqueue.other_runs(), lines[1:3])


class PffOffsetTests(unittest.TestCase):
    @staticmethod
    def events(shift):
        def ev(t, kind, team=1, piece=None):
            return {"startTime": t + shift, "endTime": t + shift,
                    "gameEvents": {"gameEventType": kind, "teamId": team, "setpieceType": piece}}
        return [ev(2.0, "FIRSTKICKOFF", 1, "K"), ev(4.0, "OTB", 2, "O"), ev(6.0, "OUT", 2),
                ev(8.0, "OTB", 1, "T"), ev(12.0, "SECONDKICKOFF", 2, "K"), ev(14.0, "OTB", 1, "O")]

    def test_shifted_event_times_land_on_the_tracking_frames(self):
        from tempolens.opendata import pff_labels, pff_offset
        period = np.zeros(20, dtype=np.int8)
        period[2:10], period[12:20] = 1, 2                  # tracking: kickoffs at 2 s and 12 s
        aligned, _ = pff_labels(self.events(0.0), 20, 1.0, home_id=1)
        late = self.events(1.0)                             # events stamped 1 s late
        offset = pff_offset(late, period, 1.0)
        self.assertAlmostEqual(offset, 1.0)
        shifted, _ = pff_labels(late, 20, 1.0, home_id=1, offset=offset)
        np.testing.assert_array_equal(shifted, aligned)
        unshifted, _ = pff_labels(late, 20, 1.0, home_id=1)
        self.assertFalse(np.array_equal(unshifted, aligned))

    def test_halves_that_disagree_raise_and_boundaries_hold(self):
        from tempolens.opendata import pff_offset
        period = np.zeros(20, dtype=np.int8)
        period[2:10], period[12:20] = 1, 2
        drift = self.events(0.0)
        drift[4] = dict(drift[4], startTime=12.5)           # exactly 0.5 s: still accepted
        self.assertAlmostEqual(pff_offset(drift, period, 1.0), 0.0)
        drift[4] = dict(drift[4], startTime=12.51)
        with self.assertRaises(ValueError):
            pff_offset(drift, period, 1.0)
        self.assertAlmostEqual(pff_offset(self.events(0.0)[:4], period, 1.0), 0.0)  # no 2nd half
        with self.assertRaises(ValueError):
            pff_offset(self.events(0.0)[1:], period, 1.0)    # no first kickoff

    def test_the_pff_table_cache_name_carries_its_version(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        import tempolens.opendata as od
        with tempfile.TemporaryDirectory() as folder:
            labelled = SimulatorTests()._labelled(3000)       # the loaders label their tables
            with patch.object(od, "pff_game", lambda game_id: labelled), \
                    patch.dict(od.LOADERS, {"dfl_x": lambda: labelled}):
                od.load_open_match("pff_9", cache_dir=folder)
                od.load_open_match("dfl_x", cache_dir=folder)
            self.assertEqual(sorted(p.name for p in Path(folder).iterdir()),
                             ["dfl_x_l{}.npz".format(od.LOADER_VERSION),
                              "pff_9_l{}_p{}.npz".format(od.LOADER_VERSION, od.PFF_VERSION)])


class KickoffTallyTests(unittest.TestCase):
    def test_the_first_half_is_the_reference_not_an_agreement(self):
        from tempolens import kickoffs
        from tempolens.match import MatchTable
        n = 130000
        ball = np.full((n, 2), np.nan, dtype=np.float32)
        for f in (11185, 113742, 60100):
            ball[f:f + 60] = [52.5, 34.0]
            ball[f + 25:f + 60, 0] = 52.5 + np.arange(35) * 0.2
        table = MatchTable("k", ball, np.zeros((n, 2, 2), dtype=np.float32),
                           np.array([H, A], dtype=np.int8), np.ones(n, dtype=np.int8))
        codes = np.full(n, D, dtype=np.int8)
        codes[11185:11300] = H
        codes[113742:113900] = H            # wrong: away should kick off the second half
        codes[60100:60200] = H              # wrong: home scored, away kicks off
        rows = kickoffs.check(codes, table, [(60000, A)])
        self.assertEqual(kickoffs.tally(rows), (0, 2, 0))
        codes[113742:113900] = A
        codes[60100:60200] = D              # nothing decoded after the goal
        self.assertEqual(kickoffs.tally(kickoffs.check(codes, table, [(60000, A)])), (1, 0, 1))
        self.assertEqual(kickoffs.tally([("half 1", H, H)]), (0, 0, 0))


class QuotedRuleTests(unittest.TestCase):
    def test_build_gate_c_and_d(self):
        from tempolens.studies import swaadopt as sa
        base = 0.5747761368684252
        for mean10, ok in ((base, True), (base - sa.VETO, True), (base - sa.VETO - 1e-9, False),
                           (base + 0.01, True), (0.0, False)):
            self.assertEqual(sa.veto_passes(mean10, base), ok, mean10)
        ref = 0.6475018131109088
        for cand, ok in ((ref, False), (ref + 1e-12, True), (ref - 1e-12, False),
                         (0.6466, False)):
            self.assertEqual(sa.beats(cand, ref), ok, cand)

    def test_build_stops_unless_c_and_d_pass(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from tempolens.studies import swaadopt as sa

        class Reached(Exception):
            pass
        cases = (({"pass": False}, True, 3, None),        # (c) fails: stop before (d)
                 ({"pass": True}, False, 3, {"d": False}),  # (d) fails: stop, recorded
                 ({"pass": True}, True, None, None))      # both pass: goes on to build
        for pff, beats, code, saved in cases:
            study = {"candidate": {"mean": 0.65}, "reference": {"mean": 0.64}, "beats": beats,
                     "alpha": 0.5, "steps": 0, "costs": [2.0, 8.0, 8.0, 0.5]}
            with tempfile.TemporaryDirectory() as folder:
                with patch.object(sa, "DIR", Path(folder)), \
                        patch.object(sa, "verdict", lambda step: study if step == "study" else pff), \
                        patch("tempolens.candidate.build", side_effect=Reached):
                    if code is None:
                        with self.assertRaises(Reached):          # handed to candidate.build
                            sa.build("bundle")
                        continue
                    with patch.object(sa, "save") as save, self.assertRaises(SystemExit) as stop:
                        sa.build("bundle")
                    self.assertEqual(stop.exception.code, code)
                    if saved:
                        save.assert_called_once_with("build", saved)
                    else:
                        save.assert_not_called()

    def test_pff_reversal_rule(self):
        from unittest.mock import patch
        from tempolens import pffstudy
        step = 0.0078125                              # exact in binary, so the boundary is exact
        with patch.object(pffstudy, "REVERSAL", step):
            for diffs, expected in (([step] * 16, (step, 16, False)),          # mean = limit
                                    ([2 * step] * 11 + [0.0] * 5, None),
                                    ([4 * step] * 11 + [-step] * 5, (None, 11, True)),
                                    ([4 * step] * 10 + [-step] * 6, (None, 10, False)),
                                    ([-4 * step] * 16, (None, 16, False))):     # wrong side
                mean, agree, reverses = pffstudy.reversal(diffs)
                if expected is None:
                    self.assertTrue(mean > step and agree == 11 and reverses)
                    continue
                if expected[0] is not None:
                    self.assertEqual(mean, expected[0])
                self.assertEqual((agree, reverses), expected[1:])


class OneRecipeTests(unittest.TestCase):
    def test_condition_sample_is_the_old_recipe_for_each_condition(self):
        from unittest.mock import patch
        from tempolens import adaptcv
        from tempolens.studies import hedgestudy
        from tempolens.experiments import SIM_TEST_SEED
        from tempolens.opendata import EXTRA_LOADERS, LOADERS

        def record(name, seed=None, randomised=False, faults=False, keeper=False, repair=None):
            return (name, seed, randomised, faults, keeper, repair)
        if not EXTRA_LOADERS:           # a fresh clone has no SkillCorner download
            patcher = patch.dict(EXTRA_LOADERS, {"sc_0": None})
            patcher.start()
            self.addCleanup(patcher.stop)
        names, sc = sorted(LOADERS), sorted(EXTRA_LOADERS)[0]
        old = {"sim": (names[3], SIM_TEST_SEED, False, False, False, None),
               "shift": (names[3], adaptcv.SHIFT_SEED + 3, True, True, False, None),
               "sc": (sc, None, False, False, False, None)}
        with patch.object(adaptcv, "CachedSample", side_effect=record):
            for c, expected in old.items():
                self.assertEqual(adaptcv.condition_sample(c, expected[0]), expected)
            made = {(c, n): make() for c, n, make, _ in hedgestudy.matches()}
        self.assertEqual(made[("sim", names[3])], old["sim"])
        self.assertEqual(made[("shift", names[3])], old["shift"])
        self.assertEqual(made[("sc", sc)], old["sc"])
        self.assertEqual(len(made), 2 * len(names) + len(EXTRA_LOADERS))
        with self.assertRaises(ValueError):
            adaptcv.condition_sample("public", names[0])

    def test_shared_constants_and_label_helpers_agree(self):
        from tempolens import cvrescore, kickoffs, labelcal, pffstudy, predict
        self.assertEqual(labelcal.E2A, (("provider", 0), ("controlled-1s", 25),
                                        ("controlled-2s", 50), ("controlled-3s", 75)))
        self.assertIs(labelcal.E2A, cvrescore.CONVENTIONS)
        self.assertEqual(kickoffs.HALF_KICKOFFS, (11185, 113742))
        self.assertEqual(kickoffs.HALF_KICKOFFS[1], predict.SECOND_HALF_KICKOFF["BAR-ATH"])
        from tempolens.opendata import remove_flicker
        table = SimulatorTests()._labelled(3000)
        labels = table.labels.copy()
        labels[500:510] = A                  # a 10-frame flicker inside a home spell
        table.labels = labels

        class S:
            pass
        s = S()
        s.labels, s.runs = labels, table.runs
        for f in (0, 10, 25, 50):            # labelcal: 0 is the provider, 10 refolds
            old = remove_flicker(labels, table.runs(), f) if f else labels
            np.testing.assert_array_equal(labelcal.labels_at(s, f), old)
        new = pffstudy.labels_by_convention(s)            # pffstudy: 10 is the loader's labels
        np.testing.assert_array_equal(new[10], labels)
        np.testing.assert_array_equal(new[50], remove_flicker(labels, table.runs(), 50))


class ShrinkTests(unittest.TestCase):
    def test_shrink_scales_the_unclipped_weight_and_keeps_the_clip(self):
        from tempolens.tagprior import MAX_WEIGHT, SHRINK, calibrate
        rng = np.random.default_rng(0)
        n = 5000
        side = rng.choice([H, A], n)
        logp = np.log(np.where(side[:, None] == np.array([H, A, 2]), 0.95, 0.025))
        inside = np.ones(n, dtype=bool)
        vote = np.where(side == H, 1, -1).astype(np.int8)
        noisy = vote.copy()
        noisy[rng.random(n) < 0.3] *= -1                         # r about 0.7: unclipped
        today = calibrate(logp, {"g": noisy}, inside)["g"]
        self.assertEqual(calibrate(logp, {"g": noisy}, inside, SHRINK)["g"], today)
        double = calibrate(logp, {"g": noisy}, inside, 1.0)["g"]
        self.assertAlmostEqual(double[0], 2 * today[0], places=12)
        self.assertTrue(0 < today[0] < double[0] < MAX_WEIGHT)
        sure = calibrate(logp, {"g": vote}, inside, 1.0)["g"]   # r near 1: clipped
        self.assertEqual(sure[0], MAX_WEIGHT)

    def test_self_calibrated_passes_the_shrink_through(self):
        from tempolens.tagprior import self_calibrated
        n = 3000
        side = np.tile([H] * 500 + [A] * 500, 3)
        logp = np.log(np.where(side[:, None] == np.array([H, A, 2]), 0.95, 0.025))
        tags = [(i, i + 500, "attack", H if (i // 500) % 2 == 0 else A) for i in range(0, n, 500)]
        tags[1] = (500, 1000, "attack", H)                      # one wrong: r = 5/6
        half = self_calibrated(logp, tags, np.ones(n, bool))[1]["attack"][0]
        full = self_calibrated(logp, tags, np.ones(n, bool), shrink=1.0)[1]["attack"][0]
        self.assertAlmostEqual(full, 2 * half, places=12)

    def test_shrink_rule_boundaries(self):
        from tempolens.studies.tagstudy import shrink_rule
        e = (0.0, 0.1, 0.2, 0.35, 0.5)
        base = dict(zip(e, (0.7820, 0.7772, 0.7763, 0.7733, 0.7730)))

        def row(*deltas):
            return {k: base[k] + d for k, d in zip(e, deltas)}
        q = 0.00390625                                       # 2^-8, exact in binary
        with __import__("unittest.mock").mock.patch("tempolens.studies.tagstudy.GAIN", q), \
                __import__("unittest.mock").mock.patch("tempolens.studies.tagstudy.LOSS", q):
            b = {k: 0.5 for k in e}                          # exact base for exact boundaries
            for deltas, ok in (((0, q, q, -q, -q), True),            # all exactly at the limits
                               ((0, q, q / 2, 0, 0), False),          # 20% short
                               ((0, q, q, -2 * q, 0), False),         # loses too much at 35%
                               ((0, q, q, 0, -2 * q), False)):        # ... at 50%
                scores = {0.5: b, 0.75: {k: 0.5 + d for k, d in zip(e, deltas)}}
                self.assertEqual(shrink_rule(scores)[1][0.75], ok, deltas)
        both = {0.5: base, 0.75: row(0, 0.004, 0.005, 0, 0), 1.0: row(0, 0.004, 0.004, 0, 0)}
        self.assertEqual(shrink_rule(both)[0], 0.75)          # larger gain at 20%
        self.assertIsNone(shrink_rule({0.5: base, 1.0: row(0, 0.01, 0.01, -0.01, 0)})[0])

    def test_predict_reads_tag_shrink_and_defaults_to_today(self):
        from unittest.mock import patch
        import games
        import tempolens.tagprior as tp
        from tempolens.predict import decode_learned
        table = _table()
        table.meta = {"home": "Home", "away": "Away"}
        logp = np.log(np.tile([0.5, 0.3, 0.2, 0.0], (table.n_frames, 1)))[:, :4]
        seen = []

        def fake(logp, instances, inside, groups=tp.GROUPS, shrink=tp.SHRINK):
            seen.append(shrink)
            return logp, {}
        settings = {"switch": 4.0, "alive_to_dead": 8.0, "dead_to_alive": 8.0, "tag_prior": "auto"}
        with patch.object(games, "require", return_value=["tags.xml"]), \
                patch.object(tp, "read_instances", return_value=[]), \
                patch.object(tp, "self_calibrated", side_effect=fake):
            decode_learned(table, logp, settings, "root", "BAR-ATH")
            decode_learned(table, logp, dict(settings, tag_shrink=1.0), "root", "BAR-ATH")
        self.assertEqual(seen, [tp.SHRINK, 1.0])


class CandidateTests(unittest.TestCase):
    def test_blend_is_the_hedge_composition_and_off_is_unchanged(self):
        import copy
        from tempolens import temporal
        from tempolens.features import TableSample
        from tempolens.predict import adapt_members, ensemble_logp
        table = SimulatorTests()._labelled(3000)
        model, normaliser = temporal.train([table], epochs=1, device="cpu", verbose=False,
                                           config=temporal.TemporalConfig(hidden=8))
        model = model.double()
        import torch
        with torch.no_grad():
            model.head.weight.mul_(30.0)
            model.head.bias.mul_(30.0)
        sample = TableSample(table)
        fresh = lambda: [(copy.deepcopy(model), normaliser), (copy.deepcopy(model), normaliser)]
        plain = temporal.predict_ensemble(fresh(), sample)
        adapted = temporal.predict_ensemble(adapt_members(fresh(), sample, 0.5, 10), sample)
        np.testing.assert_array_equal(ensemble_logp(fresh(), sample, 0.5, 10), adapted)
        one = temporal.predict(model, normaliser, sample)
        (m, n), = adapt_members([(copy.deepcopy(model), normaliser)], sample, 0.5, 10)
        mixed = temporal.combine([one, temporal.predict(m, n, sample)])
        np.testing.assert_array_equal(ensemble_logp(fresh(), sample, 0.5, 10, blend=True),
                                      temporal.combine([mixed, mixed]))
        self.assertFalse(np.array_equal(plain, adapted))

    def test_verdicts_are_read_and_failures_stop(self):
        import json
        import tempfile
        from pathlib import Path
        from tempolens import candidate
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.assertIsNone(candidate.ts_settings(root / "none.json"))
            (root / "v.json").write_text(json.dumps({"chosen": None}))
            self.assertIsNone(candidate.ts_settings(root / "v.json"))
            (root / "v.json").write_text(json.dumps({"chosen": 0.75}))
            self.assertEqual(candidate.ts_settings(root / "v.json"), {"tag_shrink": 0.75})
            self.assertIsNone(candidate.b2_settings(root))                 # no log yet
            (root / "b2.log").write_text("variant ...\nB2: fails\n")
            self.assertIsNone(candidate.b2_settings(root))
            (root / "b2.2.log").write_text(
                "B2: passes: blend tta+eata at (2.0, 8.0, 8.0, 0.5) (sim 0.7741)\n")
            self.assertEqual(candidate.b2_settings(root), {
                "blend": True, "tta_alpha": 0.5, "eata_steps": 10, "switch": 2.0,
                "alive_to_dead": 8.0, "dead_to_alive": 8.0, "change_weight": 0.5})
            (root / "b2_other.log").write_text("B2: passes: blend tta at (1, 1, 1, 1) (sim 1)\n")
            self.assertEqual(candidate.b2_settings(root)["eata_steps"], 10)   # not a b2 log
        with self.assertRaises(SystemExit) as stop:
            candidate.stop("x")
        self.assertEqual(stop.exception.code, 3)


class LearningCurveTests(unittest.TestCase):
    def test_subsets_are_fixed_nested_and_exclude_the_held_out(self):
        from tempolens import adaptcv
        from tempolens.studies import lcurve
        from tempolens.opendata import LOADERS
        first, again = lcurve.subsets(), lcurve.subsets()
        self.assertEqual(first, again)                          # fixed order
        self.assertEqual(len(first), 5)
        held, chosen = first[0]
        self.assertEqual(chosen[4], ["dfl_J03WOY", "dfl_J03WPY", "metrica1", "metrica3"])
        for held, chosen in first:
            self.assertEqual(chosen[6][:4], chosen[4])            # nested
            self.assertEqual(sorted(chosen[8]), sorted(set(LOADERS) - set(held)))
            self.assertFalse(set(chosen[8]) & set(held))
        self.assertEqual(lcurve.model_path(held, 8),
                         adaptcv.OUT / "models" / "{}_m0.pt".format("-".join(held)))
        self.assertNotEqual(lcurve.model_path(held, 6).parent, adaptcv.OUT / "models")


class Draw2Tests(unittest.TestCase):
    def test_summary_spreads_and_eata_count(self):
        import contextlib
        import io
        from tempolens.studies import draw2

        def rows(shift):
            out = []
            for seed in [-1] + list(range(5)):
                for name, (alpha, steps) in draw2.SETTINGS.items():
                    for costs in draw2.COSTS:
                        base = {"none": 0.70, "TTA 0.5": 0.71, "TTA 0.5 + EATA 10": 0.705}[name]
                        out.append({"seed": seed, "alpha": alpha, "steps": steps, "gate": False,
                                    "costs": list(costs), "spellF1": base + shift})
                out.append({"seed": seed, "alpha": 0.0, "steps": 30, "gate": True,   # ignored
                            "costs": [2.0, 2.0, 2.0, 0.5], "spellF1": 0.1})
            return out
        by_draw = {s: {"m1": rows(0.01 * i), "m2": rows(0.01 * i)}
                   for i, s in enumerate(draw2.SEEDS)}
        s = draw2.summarise(by_draw)
        v = s[(-1, "TTA 0.5", draw2.COSTS[0])]
        np.testing.assert_allclose(v, [0.71, 0.72, 0.73])
        self.assertNotIn((-1, "none", (2.0, 2.0, 2.0, 0.5)), s)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            draw2.report(s)
        text = out.getvalue()
        self.assertIn("vs none +0.0100 (sd 0.0000)", text)             # same gap on every draw
        self.assertIn("ensemble on 0 of 3 draws", text)                 # EATA below TTA alone
        self.assertIn("members on 0 of 15", text)

    def test_cache_names_carry_the_seed(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from tempolens import adaptcv
        with tempfile.TemporaryDirectory() as folder:
            adaptcv.cached(Path(folder), "sim1001", "m", lambda: [1])
            adaptcv.cached(Path(folder), "sim1002", "m", lambda: [2])
            self.assertEqual(adaptcv.cached(Path(folder), "sim1001", "m",
                                            lambda: self.fail("recomputed")), [1])
            self.assertEqual(sorted(p.name for p in Path(folder).iterdir()),
                             ["loader.json", "sim1001_m.json", "sim1002_m.json"])   # + the loader stamp


class TwoViewTests(unittest.TestCase):
    def test_rule_needs_the_mean_and_every_member(self):
        from tempolens import twoview
        ens = -1

        def table(ens_means, member_mean):
            t = {}
            for i, s in enumerate(twoview.STEPS):
                for j, c in enumerate(twoview.COSTS):
                    k = (0.5, s, False, c)
                    m = ens_means[2 * i + j]
                    t[(ens,) + k] = {c2: m for c2 in twoview.CONDITIONS}
                    for mem in range(5):
                        t[(mem,) + k] = {c2: member_mean[mem] for c2 in twoview.CONDITIONS}
            return t
        ref, refm = 0.6475, [0.64] * 5
        best, mean, _, ok = twoview.decide(table([0.6480, 0.6470, 0.6460, 0.6400], [0.64] * 5),
                                           ref, refm)
        self.assertEqual((best[1], best[3], ok), (10, (4.0, 8.0, 8.0, 1.0), True))
        self.assertFalse(twoview.decide(table([0.6475] * 4, [0.65] * 5), ref, refm)[3])  # equal
        low = [0.65, 0.65, 0.65, 0.65, 0.6399]
        self.assertFalse(twoview.decide(table([0.66] * 4, low), ref, refm)[3])   # one member
        tie = twoview.decide(table([0.6500, 0.6500, 0.6500, 0.6500], [0.64] * 5), ref, refm)
        self.assertEqual((tie[0][1], tie[0][3]), (10, (4.0, 8.0, 8.0, 1.0)))     # tie order
        best0 = twoview.decide(table([0.6476, 0.6477, 0.6490, 0.6400], [0.64] * 5), ref, refm)
        self.assertEqual((best0[0][1], best0[0][3]), (0, (4.0, 8.0, 8.0, 1.0)))

    def test_steps_stop_without_their_predecessor(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from tempolens import twoview
        with tempfile.TemporaryDirectory() as folder, patch.object(twoview, "DIR", Path(folder)):
            with self.assertRaises(SystemExit) as stop:
                twoview.pff()
            self.assertEqual(stop.exception.code, 3)
            (Path(folder) / "verdict_study.json").write_text(json.dumps({"pass": False}))
            with self.assertRaises(SystemExit) as stop:
                twoview.pff()
            self.assertEqual(stop.exception.code, 3)
            self.assertIsNone(twoview.settings())
            (Path(folder) / "verdict_study.json").write_text(json.dumps(
                {"pass": True, "steps": 0, "costs": [2.0, 8.0, 8.0, 0.5]}))
            (Path(folder) / "verdict_pff.json").write_text(json.dumps({"pass": True}))
            self.assertEqual(twoview.settings(), {
                "tta_both_views": True, "tta_alpha": 0.5, "eata_steps": 0, "switch": 2.0,
                "alive_to_dead": 8.0, "dead_to_alive": 8.0, "change_weight": 0.5})

    def test_both_views_reaches_the_study_and_predict(self):
        import copy
        from tempolens import adaptcv, temporal
        from tempolens.features import TableSample
        from tempolens.predict import adapt_members
        table = SimulatorTests()._labelled(3000)
        model, normaliser = temporal.train([table], epochs=1, device="cpu", verbose=False,
                                           config=temporal.TemporalConfig(hidden=8))
        model = model.double()
        other = SimulatorTests()._labelled(3000)
        other.players = other.players * 0.8 + 5.0
        other.ball = other.ball * 0.8 + 5.0
        other.slot_team = other.slot_team[::-1].copy()    # the views differ
        sample = TableSample(other)
        grid = {bv: adaptcv.member_grid([(model, normaliser)], sample, double=True,
                                        alphas=(0.5,), steps=(0,), both_views=bv)
                for bv in (False, True)}
        for bv in (False, True):
            (m, n), = adapt_members([(copy.deepcopy(model), normaliser)], sample, 0.5, 0, bv)
            np.testing.assert_allclose(temporal.predict(m, n, sample), grid[bv][(0.5, 0)][0],
                                       atol=2e-4)
        self.assertFalse(np.allclose(grid[False][(0.5, 0)][0], grid[True][(0.5, 0)][0]))


class LcbTests(unittest.TestCase):
    def test_a_finished_model_is_skipped_and_a_killed_one_left_absent(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from tempolens import adaptcv, lcb
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(lcb, "FINAL_DIR", root), patch.object(lcb, "FOLD_DIR", root):
                calls = []

                def killed(cmd, check):
                    calls.append(cmd)
                    out = Path(cmd[cmd.index("--out") + 1])
                    out.with_name(out.name + ".123.tmp").write_bytes(b"half")   # then killed
                    raise KeyboardInterrupt
                with patch.object(adaptcv.subprocess, "run", side_effect=killed):
                    with self.assertRaises(KeyboardInterrupt):
                        lcb.train("final", 0)
                self.assertFalse(lcb.final_path(0).exists())          # no final file
                self.assertFalse(Path(str(lcb.final_path(0)) + ".lock").exists())
                cmd = calls[0]
                self.assertIn("--min-spell", cmd)
                self.assertEqual(cmd[cmd.index("--min-spell") + 1], "50")
                self.assertNotIn("--matches", cmd)
                lcb.final_path(0).write_bytes(b"done")                # finished by a rerun
                with patch.object(adaptcv.subprocess, "run", side_effect=AssertionError("ran")):
                    self.assertEqual(lcb.train("final", 0), lcb.final_path(0))
                def trains(c, check):
                    calls.append(c)
                    Path(c[c.index("--out") + 1]).write_bytes(b"w")
                with patch.object(adaptcv.subprocess, "run", side_effect=trains):
                    lcb.train("fold", 2, 1)
                fold_cmd = calls[-1]
                held = __import__("tempolens.experiments", fromlist=["folds"]).folds(
                    sorted(__import__("tempolens.opendata", fromlist=["LOADERS"]).LOADERS))[1]
                matches = fold_cmd[fold_cmd.index("--matches") + 1:fold_cmd.index("--out")]
                self.assertFalse(set(matches) & set(held))
                self.assertEqual(len(matches), 8)
                self.assertTrue(fold_cmd[fold_cmd.index("--out") + 1].endswith(
                    "{}_m2.pt".format("-".join(held))))

    def test_missing_inputs_stop_with_their_name(self):
        from pathlib import Path
        from tempolens import lcb
        with self.assertRaises(SystemExit) as stop:
            lcb.require([Path(__file__), Path("no/such/model.pt")])
        self.assertEqual(stop.exception.code, 3)
        lcb.require([Path(__file__)])
        self.assertEqual(len(lcb.model_paths()), 30)
        self.assertEqual(len(set(lcb.model_paths())), 30)

    def test_the_table_keeps_sets_settings_conditions_and_conventions_apart(self):
        from tempolens import lcb
        results = {("sim", "m1"): {("ENS5", "s", 10): 0.6, ("mix", "s", 50): 0.8},
                   ("sim", "m2"): {("ENS5", "s", 10): 0.7, ("mix", "s", 50): 0.9},
                   ("pff", "g"): {("ENS5", "s", 10): 0.5}}
        t = lcb.table_of(results)
        self.assertAlmostEqual(t[("ENS5", "s", "sim", 10)], 0.65)
        self.assertAlmostEqual(t[("mix", "s", "sim", 50)], 0.85)
        self.assertAlmostEqual(t[("ENS5", "s", "pff", 10)], 0.5)
        self.assertEqual(len(t), 3)


class TrainOnceNoFileTests(unittest.TestCase):
    def test_a_command_that_writes_nothing_raises_instead_of_looping(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from tempolens import adaptcv
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "x.pt"
            with patch.object(adaptcv.subprocess, "run") as run:
                with self.assertRaises(RuntimeError):
                    adaptcv.train_once(path, ["train"])
            self.assertEqual(run.call_count, 1)
            self.assertEqual(list(Path(folder).iterdir()), [])          # lock released


class LcbMixMembersTests(unittest.TestCase):
    def test_the_mix_is_ens5_plus_ms50_whatever_decoder_json_lists(self):
        import tempfile
        from unittest.mock import patch
        from tempolens import adaptcv, lcb
        exp = adaptcv.ROOT / "models" / "exp"
        exp.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=exp) as folder:
            with patch.object(lcb, "FINAL_DIR", type(exp)(folder)):
                for k in range(lcb.MEMBERS):
                    lcb.final_path(k).write_bytes(b"w")
                ms50 = [lcb.final_path(k).relative_to(adaptcv.ROOT / "models").as_posix()
                        for k in range(lcb.MEMBERS)]
                ens5 = list(adaptcv.ENS5)
                ten = {"members": ens5 + ms50}      # decoder.json after the switch
                with patch("tempolens.predict.load_decoder", return_value=ten):
                    mix = lcb.members_for("lcb_mix")
                    self.assertEqual(mix, ens5 + ms50)
                    self.assertEqual(len(set(mix)), 10)
                    self.assertEqual(lcb.members_for("lcb_ms50"), ms50)
                lcb.final_path(4).unlink()
                self.assertIsNone(lcb.members_for("lcb_mix"))


class Ens5AfterAdoptTests(unittest.TestCase):
    TEN = {"members": ["tempolens_rnd4_m{}.pt".format(k) for k in range(5)]
           + ["tempolens_ms50_m{}.pt".format(k) for k in range(5)]}

    def test_adaptseeds_members_5_resolves_the_five_rnd4_files(self):
        import tempfile
        from unittest.mock import patch
        from tempolens import adaptcv

        class Seen(Exception):
            pass

        def stop_at_check(folder, paths):
            raise Seen([p.name for p in paths])
        with tempfile.TemporaryDirectory() as folder,                 patch("tempolens.predict.load_decoder", return_value=self.TEN),                 patch.object(adaptcv, "check_members", side_effect=stop_at_check):
            with self.assertRaises(Seen) as seen:        # adaptseeds: collect(5, folder, scorer)
                adaptcv.collect(5, folder, lambda *a: None)
        self.assertEqual(seen.exception.args[0], ["tempolens_rnd4_m{}.pt".format(k) for k in range(5)])
        for p in adaptcv.ens5_paths():
            self.assertTrue(p.exists(), p)               # committed, so a fresh clone has them

    def test_candidates_start_from_ens5_and_lcb_rows_keep_their_members(self):
        from tempolens import adaptcv, candidate
        with __import__("unittest.mock").mock.patch.object(
                candidate.DEFAULT_DECODER.__class__, "read_text",
                return_value=__import__("json").dumps({"switch": 4.0, **self.TEN})):
            b2 = candidate.config_for("b2", {"blend": True})
            mix = candidate.config_for("lcb_mix", {"members": self.TEN["members"]})
        self.assertEqual(b2["members"], list(adaptcv.ENS5))
        self.assertTrue(b2["blend"])
        self.assertEqual(b2["switch"], 4.0)
        self.assertEqual(mix["members"], self.TEN["members"])


class RecipeProvenanceTests(unittest.TestCase):
    def test_train_records_draws_and_min_spell_in_the_weights(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from tempolens import temporal, train
        saved = {}

        def fake_save(model, normaliser, path, recipe=None):
            saved.update(path=path, recipe=recipe)
        with tempfile.TemporaryDirectory() as folder,                 patch.object(train, "training_samples", return_value=[]),                 patch.object(temporal, "train", return_value=(None, None)),                 patch.object(temporal, "save", side_effect=fake_save):
            train.main(["--out", str(Path(folder) / "x.pt"), "--draws", "4", "--min-spell", "50",
                        "--matches", "metrica1"])
        self.assertEqual(saved["recipe"], {"draws": 4, "min_spell": 50, "matches": ["metrica1"]})

    def test_the_recipe_is_stored_and_load_ignores_it(self):
        import tempfile
        from pathlib import Path
        import torch
        from tempolens import temporal
        model = temporal.build_model(5, temporal.TemporalConfig(hidden=8))
        normaliser = temporal.Normaliser(np.random.default_rng(0).normal(size=(50, 5)))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "m.pt"
            temporal.save(model, normaliser, path, recipe={"draws": 4, "min_spell": 50})
            blob = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(blob["recipe"], {"draws": 4, "min_spell": 50})
            temporal.load(path)


class OofTagTests(unittest.TestCase):
    def test_every_training_input_names_the_cache_for_both_kinds(self):
        from unittest.mock import patch
        import tempolens.experiments as ex
        from tempolens.temporal import DEFAULT_CONFIG
        for kind in ("gbm", "temporal"):
            plain = ex.oof_tag(kind)
            variants = [ex.oof_tag(kind, draws=4), ex.oof_tag(kind, draws=4, faults=True),
                        ex.oof_tag(kind, min_spell=50), ex.oof_tag(kind, sc_visible=True)]
            self.assertEqual(len({plain, *variants}), 5, kind)
            with patch.object(ex, "LOADER_VERSION", 2):
                self.assertNotEqual(ex.oof_tag(kind), plain)
        # today's names, so no existing cache is recomputed
        self.assertEqual(ex.oof_tag("gbm"), "gbm")
        self.assertEqual(ex.oof_tag("temporal", draws=4, min_spell=50),
                         DEFAULT_CONFIG.tag() + ("" if ex.TRAIN_VERSION == 1 else "_t{}".format(
                             ex.TRAIN_VERSION)) + "_rnd4v{}_ms50".format(ex.RANDOM_VERSION))


class ScoreCacheVersionTests(unittest.TestCase):
    def test_pff_scores_ignore_a_stale_json_and_need_the_outputs(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from tempolens import pffstudy
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / "pff_g.json").write_text(json.dumps({"old labels": 1}))
            np.savez(folder / "pff_g_outputs.npz",
                     **{"a{}_s{}".format(*k): np.zeros((2, 5, 4)) for k in pffstudy.KEYS})
            with patch.object(pffstudy, "_cache", return_value=folder),                     patch.object(pffstudy, "CachedSample"),                     patch.object(pffstudy, "labels_by_convention", return_value={10: 0, 50: 0}),                     patch.object(pffstudy, "decode_table"),                     patch.object(pffstudy, "score", return_value={"spellF1": 0.5}):
                rows = pffstudy.score_match(None, "g")
                with self.assertRaises(FileNotFoundError):
                    pffstudy.score_match(None, "missing")
        self.assertEqual(set(rows), {label for label, _, _ in pffstudy.ROWS})
        self.assertEqual(rows[pffstudy.ROWS[0][0]], {"10": 0.5, "50": 0.5})

    def test_study_rows_refuse_another_loader_version(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from tempolens import adaptcv, opendata
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            self.assertEqual(adaptcv.cached(folder, "sim", "m", lambda: {"r": 1}), {"r": 1})
            self.assertEqual(adaptcv.cached(folder, "sim", "m", lambda: {"r": 2}), {"r": 1})
            with patch.object(opendata, "LOADER_VERSION", opendata.LOADER_VERSION + 1):
                with self.assertRaises(RuntimeError):
                    adaptcv.cached(folder, "sim", "m", lambda: {"r": 3})


class LockWaitMessageTests(unittest.TestCase):
    def test_a_waiter_names_the_lock_and_its_holder_at_once(self):
        import contextlib
        import io
        import tempfile
        from pathlib import Path
        from tempolens import adaptcv
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "m.pt"
            Path(folder, "m.pt.lock").write_text("4242")
            out = io.StringIO()
            with contextlib.redirect_stdout(out), self.assertRaises(RuntimeError):
                adaptcv.train_once(path, ["never-run"], wait_s=0, timeout_s=0)
        self.assertIn("m.pt.lock", out.getvalue())
        self.assertIn("4242", out.getvalue())


class PffVetoScoresTests(unittest.TestCase):
    def test_candidate_scores_cache_each_match_and_score_both_conventions(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from tempolens import pffstudy
        calls = []

        def grid(members, sample, alphas, steps, both_views=False):
            calls.append((alphas, steps, both_views))
            return {(alphas[0], steps[-1]): [np.zeros((5, 4)), np.zeros((5, 4))]}
        with tempfile.TemporaryDirectory() as folder,                 patch.object(pffstudy, "check_members"),                 patch.object(pffstudy, "CachedSample"),                 patch.object(pffstudy.temporal, "load", return_value=None),                 patch.object(pffstudy.adaptcv, "member_grid", side_effect=grid),                 patch.object(pffstudy, "decode_table"),                 patch.object(pffstudy, "labels_by_convention", return_value={10: 0, 50: 0}),                 patch.object(pffstudy, "score", return_value={"spellF1": 0.4}):
            args = (["a", "b"], Path(folder), ["m0", "m1"], 0.5, 10, (4, 8, 8, 1), "_s10")
            first = pffstudy.candidate_scores(*args, both_views=True)
            again = pffstudy.candidate_scores(*args, both_views=True)
            names = sorted(p.name for p in Path(folder).iterdir())
        self.assertEqual(first, {10: [0.4, 0.4], 50: [0.4, 0.4]})
        self.assertEqual(again, first)
        self.assertEqual(calls, [((0.5,), (0, 10), True)] * 2)    # computed once per match
        self.assertEqual(names, ["pff_a_s10.npz", "pff_b_s10.npz"])  # the old cache names


class GridFileTests(unittest.TestCase):
    def test_save_and_load_grid_round_trip_in_the_shared_format(self):
        import tempfile
        from pathlib import Path
        from tempolens import adaptcv
        keys = ((0.0, 0), (0.5, 10))
        grid = {k: [np.full((3, 4), i + j, dtype=np.float64) for j in range(2)]
                for i, k in enumerate(keys)}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "g.npz"
            adaptcv.save_grid(path, grid, keys)
            self.assertEqual(sorted(np.load(path).files), ["a0.0_s0", "a0.5_s10"])  # old names
            back = adaptcv.load_grid(path, keys)
        for k in keys:
            self.assertEqual(len(back[k]), 2)
            np.testing.assert_array_equal(np.stack(back[k]), np.stack(grid[k]))
            self.assertEqual(back[k][0].dtype, np.float64)


class GameConstantsTests(unittest.TestCase):
    def test_kickoff_constants_match_the_published_frames(self):
        from games import GAME, GAMES, half_video_blocks, source_frames
        from tempolens.kickoffs import HALF_KICKOFFS
        from tempolens.predict import SECOND_HALF_KICKOFF
        self.assertEqual(HALF_KICKOFFS, (11185, 113742))              # DATA.md / FAQ.md
        self.assertEqual(HALF_KICKOFFS[0], half_video_blocks(GAME)[0][1])   # first graded frame
        self.assertEqual(SECOND_HALF_KICKOFF[GAME], HALF_KICKOFFS[1])
        self.assertGreater(HALF_KICKOFFS[1], half_video_blocks(GAME)[1][1])
        self.assertEqual(source_frames(), 199319)
        self.assertEqual(GAMES[GAME]["n_video_frames"], 199319)


class SubmissionWriteTests(unittest.TestCase):
    def test_written_atomically_with_the_same_bytes(self):
        import tempfile
        from pathlib import Path
        from tempolens.predict import write_submission
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "Coding United.csv"
            write_submission(path, np.array(["X", "H", "A", "D"]))
            self.assertEqual(path.read_bytes(), b"frame,label\n0,X\n1,H\n2,A\n3,D\n")

            def crash():                    # labels that fail halfway through the write
                yield "H"
                raise RuntimeError("killed")
            with self.assertRaises(RuntimeError):
                write_submission(path, crash())
            self.assertEqual(path.read_bytes(), b"frame,label\n0,X\n1,H\n2,A\n3,D\n")  # untouched
            self.assertEqual([p.name for p in Path(folder).iterdir()], [path.name])   # no .tmp


if __name__ == "__main__":
    unittest.main()
