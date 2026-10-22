"""Grouped cross-validation on open data under simulated ATD noise, scored by the official metric.

This, not the public five minutes, is how versions of the pipeline are compared. Every match is
corrupted by `simulate.simulate_atd` -- calibrated so the official baseline reproduces its
published scores -- because clean open tracking hides the competition's real difficulty: the
ball is missing for 41% of the match, in long stretches.

    python -m tempolens.experiments                    # LightGBM, simulated ATD, 5 folds
    python -m tempolens.experiments --model temporal   # the recurrent model
    python -m tempolens.experiments --model ensemble   # its members averaged
    python -m tempolens.experiments --protocol clean   # no simulated noise (optimistic)
"""

import argparse
import hashlib
import os
import sys
import time

import numpy as np

from .decoder import transition_matrix, viterbi_matrix
from .features import FEATURE_VERSION, feature_names, frame_features, swap_labels
from .match import A, D, H, X
from .opendata import LOADERS, load_open_match, open_data_dir
from .simulate import AtdNoise, simulate_atd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

GBM_PARAMS = dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=63,
                  min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0, deterministic=True,
                  num_threads=8)
GBM_ROUNDS = 300
TRAIN_STRIDE = 5          # train on every 5th frame: neighbours are near-duplicates
SIM_TRAIN_SEEDS = (1, 2)  # two noise draws of every training match
SIM_TEST_SEED = 1000      # one fixed draw of every held-out match
N_FOLDS = 5
# The caches below are keyed by parameters, not by code. Bump SIM_VERSION when simulate.py's
# logic changes and TRAIN_VERSION when temporal.py's training changes (LR, CHUNK, losses...);
# version 1 keeps the historical file names, so existing caches stay valid.
SIM_VERSION = 2   # 2: Metrica's dead-time ball filled (opendata.fill_dead_ball)
TRAIN_VERSION = 1
NOISE_TAG = hashlib.md5((repr(AtdNoise()) + ("" if SIM_VERSION == 1 else str(SIM_VERSION)))
                        .encode()).hexdigest()[:8]


# -- scoring and decoding ---------------------------------------------------------------------

def score(labels, pred):
    """Official metrics for a predicted code array against true codes (X = ungraded)."""
    result = pm.evaluate(labels.astype(np.int8), pred.astype(np.int8))
    return {"spellF1": result["spell_f1_mean"], "macroF1": result["macro_f1"],
            "n_pred": result["n_pred_spells"], "n_gt": result["n_gt_spells"]}


def baseline_codes(table, window=50):
    """The official baseline on a MatchTable: nearest player's team, majority vote, D if no ball."""
    dist = np.hypot(table.players[..., 0] - table.ball[:, None, 0],
                    table.players[..., 1] - table.ball[:, None, 1])
    dist = np.where(np.isfinite(dist), dist, np.inf)
    nearest = np.argmin(dist, axis=1)
    have = np.isfinite(dist.min(axis=1))
    team = np.where(have, table.slot_team[nearest], -1)
    home = np.concatenate(([0], np.cumsum(team == H)))
    away = np.concatenate(([0], np.cumsum(team == A)))
    idx = np.arange(len(team))
    lo, hi = np.maximum(0, idx - window), np.minimum(len(team), idx + window + 1)
    h, a = home[hi] - home[lo], away[hi] - away[lo]
    out = np.where(h + a > 0, np.where(h >= a, H, A), D).astype(np.int8)
    out[table.period == 0] = X
    return out


def decode_table(table, logp, transition, change_weight=0.0):
    """Viterbi inside each period; X outside.

    `logp` is (n, 3) log-probabilities, or (n, 4) with a change logit in the last column, which
    then makes a state change `change_weight * logit` cheaper (dearer where it is negative).
    """
    out = np.full(table.n_frames, X, dtype=np.int8)
    for start, stop in table.runs():
        bonus = None
        if logp.shape[1] > 3 and change_weight:
            bonus = change_weight * logp[start:stop, 3]
        out[start:stop] = viterbi_matrix(logp[start:stop, :3], transition, bonus)
    return out


# -- open matches as cached samples -----------------------------------------------------------

class CachedSample:
    """An open match, optionally corrupted by one noise draw, with features cached on disk."""

    def __init__(self, name, seed=None):
        self.name, self.seed = name, seed
        clean = load_open_match(name)
        self.labels, self.period, self.n_frames = clean.labels, clean.period, clean.n_frames
        self._runs = clean.runs()

    def runs(self):
        return self._runs

    def table(self):
        clean = load_open_match(self.name)
        return clean if self.seed is None else simulate_atd(clean, seed=self.seed)

    def _path(self, swap, kind="f{}".format(FEATURE_VERSION)):
        tag = "clean" if self.seed is None else "sim{}_{}".format(NOISE_TAG, self.seed)
        return (open_data_dir() / "cache" / "features" /
                "{}_{}_{}_{}.npy".format(self.name, tag, kind, int(swap)))

    def _cached(self, kind, swap, compute):
        """np.load(mmap) of a per-perspective array, computing both perspectives once."""
        path = self._path(swap, kind)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            for side in (False, True):
                final = self._path(side, kind)
                partial = final.with_name(final.stem + ".partial.npy")
                np.save(partial, compute(side))
                os.replace(partial, final)
        return np.load(path, mmap_mode="r")

    def tokens(self, swap=False):
        from .features import player_tokens
        table = {}

        def compute(side):
            table.setdefault("t", self.table())
            return player_tokens(table["t"], swap=side)
        return self._cached("tok{}".format(FEATURE_VERSION), swap, compute)

    def ball_target(self, swap=False):
        from .features import hidden_ball_target
        tables = {}

        def compute(side):
            if not tables:
                tables["clean"], tables["noisy"] = load_open_match(self.name), self.table()
            return hidden_ball_target(tables["clean"], tables["noisy"], swap=side)
        return self._cached("ball{}".format(FEATURE_VERSION), swap, compute)

    def features(self, swap=False):
        path = self._path(swap)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            table = self.table()
            for side in (False, True):
                # Write then rename, so an interrupted run never leaves a truncated cache file.
                final = self._path(side)
                partial = final.with_name(final.stem + ".partial.npy")
                np.save(partial, frame_features(table, swap=side)[0])
                os.replace(partial, final)
        return np.load(path, mmap_mode="r")


def folds(names, k=N_FOLDS):
    """Deterministic grouped folds: every match is held out exactly once."""
    names = sorted(names)
    return [names[i::k] for i in range(min(k, len(names)))]


# -- LightGBM ---------------------------------------------------------------------------------

def training_rows(samples, stride=TRAIN_STRIDE):
    """Stacked features and labels from both team perspectives of every sample."""
    xs, ys = [], []
    for sample in samples:
        for swap in (False, True):
            x = sample.features(swap)
            y = swap_labels(sample.labels) if swap else sample.labels
            keep = (y != X) & (sample.period > 0)
            keep[np.arange(len(keep)) % stride != 0] = False
            xs.append(np.asarray(x[keep]))
            ys.append(y[keep])
    return np.concatenate(xs), np.concatenate(ys)


def train_gbm(samples, rounds=GBM_ROUNDS):
    import lightgbm as lgb
    x, y = training_rows(samples)
    return lgb.train(GBM_PARAMS, lgb.Dataset(x, y, feature_name=feature_names()), rounds)


def predict_logp(model, sample):
    """Log-probabilities averaged over both team perspectives (test-time symmetry)."""
    p = model.predict(np.asarray(sample.features(False)))
    ps = model.predict(np.asarray(sample.features(True)))[:, [A, H, D]]  # its team 0 = away
    return np.log(np.clip(0.5 * (p + ps), 1e-6, 1.0))


# -- cross-validated predictions --------------------------------------------------------------

ENSEMBLE_MEMBERS = 3


def member_noise_seeds(member):
    """Noise draws for ensemble member k: member 0 uses SIM_TRAIN_SEEDS, the others fresh draws,
    so members differ in training data as well as initialisation."""
    return SIM_TRAIN_SEEDS if member == 0 else (2 * member + 1, 2 * member + 2)


def oof_logp(names, kind="gbm", protocol="sim", refresh=False, member=0,
             members=ENSEMBLE_MEMBERS, config=None):
    """Held-out log-probabilities for every match, each from a model that never saw it.

    kind "temporal" trains the network described by `config` (default: the base GRU).
    kind "ensemble" averages the held-out outputs of temporal members 0..members-1, each of
    which is trained (or read from cache) like kind "temporal" with that member index.
    """
    if kind == "ensemble":
        from .temporal import combine
        outputs = [oof_logp(names, "temporal", protocol, refresh, k, config=config)
                   for k in range(members)]
        return {n: combine([out[n] for out in outputs]) for n in names}
    if kind == "temporal":
        from .temporal import DEFAULT_CONFIG
        config = config or DEFAULT_CONFIG
        kind_tag = config.tag() + ("" if TRAIN_VERSION == 1 else "_t{}".format(TRAIN_VERSION))
    else:
        kind_tag = kind
    cache = open_data_dir() / "cache" / "oof"
    cache.mkdir(parents=True, exist_ok=True)
    noise = NOISE_TAG if protocol == "sim" else "clean"
    train_seeds = member_noise_seeds(member) if protocol == "sim" else (None,)
    test_seed = SIM_TEST_SEED if protocol == "sim" else None
    tag = kind_tag if member == 0 else "{}{}".format(kind_tag, member)
    logps = {}
    for held in folds(names):
        key = "-".join(sorted(set(names) - set(held)))
        paths = {n: cache / "{}_{}_f{}_{}_train-{}.npy".format(
            tag, noise, FEATURE_VERSION, n, hashlib.md5(key.encode()).hexdigest()[:8])
            for n in held}
        if not refresh and all(p.exists() for p in paths.values()):
            logps.update({n: np.load(p) for n, p in paths.items()})
            continue
        t0 = time.time()
        train = [CachedSample(n, s) for n in sorted(set(names) - set(held)) for s in train_seeds]
        if kind == "gbm":
            model = train_gbm(train)
            predict = lambda sample: predict_logp(model, sample)
        else:
            from . import temporal
            model, normaliser = temporal.train(train, seed=temporal.SEED + member,
                                               config=config)
            predict = lambda sample: temporal.predict(model, normaliser, sample)
        for n in held:
            logps[n] = predict(CachedSample(n, test_seed))
            np.save(paths[n], logps[n])
        print("trained {} without {} in {:.0f}s".format(tag, ", ".join(held), time.time() - t0),
              flush=True)
    return logps


def _summary(rows, names):
    def mean(subset):
        values = [rows[n]["spellF1"] for n in subset]
        return np.mean(values) if values else float("nan")
    return (mean(names), mean([n for n in names if n.startswith("dfl_")]),
            mean([n for n in names if n.startswith("metrica")]))


def run(names, grid, kind="gbm", protocol="sim", refresh=False, member=0, config=None):
    test_seed = SIM_TEST_SEED if protocol == "sim" else None
    logps = oof_logp(names, kind, protocol, refresh, member, config=config)
    samples = {n: CachedSample(n, test_seed) for n in names}

    print("\n{:<14} {:>8} {:>8} {:>6} {:>6}".format("baseline50", "spellF1", "macroF1", "pred",
                                                   "gt"))
    base = {}
    for n in names:
        base[n] = score(samples[n].labels, baseline_codes(samples[n].table()))
        print("{:<14} {:>8.4f} {:>8.4f} {:>6} {:>6}".format(
            n, base[n]["spellF1"], base[n]["macroF1"], base[n]["n_pred"], base[n]["n_gt"]))
    print("baseline mean spellF1 all {:.4f}  dfl {:.4f}  metrica {:.4f}".format(
        *_summary(base, names)))

    has_change = all(logps[n].shape[1] > 3 for n in names)
    weights = (0.0, 0.5, 1.0) if has_change else (0.0,)
    results = []
    for costs in grid:
        transition = transition_matrix(*costs)
        for weight in weights:
            rows = {n: score(samples[n].labels,
                             decode_table(samples[n], logps[n], transition, weight))
                    for n in names}
            results.append((_summary(rows, names), costs + (weight,), rows))
    results.sort(key=lambda item: -item[0][0])
    print("\n{} on {} protocol; decoder costs switch/alive>dead/dead>alive/change".format(
        kind, protocol))
    for (overall, dfl, metrica), costs, rows in results[:5]:
        print("mean {:.4f} (dfl {:.4f}, metrica {:.4f})  costs {}".format(
            overall, dfl, metrica, costs))
    (_, _, _), costs, rows = results[0]
    print("\nbest per match:")
    for n in names:
        r = rows[n]
        print("  {:<14} spellF1 {:.4f}  macroF1 {:.4f}  pred {:>4}  gt {:>4}  (baseline {:.4f})"
              .format(n, r["spellF1"], r["macroF1"], r["n_pred"], r["n_gt"],
                      base[n]["spellF1"]))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matches", nargs="+", default=sorted(LOADERS))
    parser.add_argument("--model", default="gbm", choices=("gbm", "temporal", "ensemble"))
    parser.add_argument("--member", type=int, default=0,
                        help="temporal only: ensemble member index (seed, noise draws)")
    parser.add_argument("--arch", default="gru", choices=("gru", "lstm", "tcn", "transformer"))
    parser.add_argument("--encoder", default="features", choices=("features", "players"))
    parser.add_argument("--hidden", type=int, default=None, help="temporal width (default 96)")
    parser.add_argument("--epochs", type=int, default=None, help="temporal epochs (default 8)")
    parser.add_argument("--aux-ball", type=float, default=0.0,
                        help="weight of the hidden-ball auxiliary loss (0 = off)")
    parser.add_argument("--smooth", type=float, default=0.0,
                        help="weight of the MS-TCN++ smoothing loss (0 = off)")
    parser.add_argument("--seg-balance", type=float, default=0.0,
                        help="exponent of the run-length frame weight (0 = off)")
    parser.add_argument("--focal", type=float, default=0.0, help="focal-loss gamma (0 = off)")
    parser.add_argument("--protocol", default="sim", choices=("sim", "clean"))
    parser.add_argument("--refresh", action="store_true", help="retrain, ignoring the cache")
    args = parser.parse_args(argv)
    # A switch dearer than alive>dead + dead>alive is never taken directly (the decoder routes
    # through one dead frame instead, which spellF1 does not see), so the grid stops there.
    grid = [(s, ad, da) for s in (1.0, 1.5, 2.0, 4.0, 6.0, 8.0, 12.0) for ad in (2.0, 4.0, 8.0)
            for da in (2.0, 4.0, 8.0)]
    from .temporal import TemporalConfig
    sizes = {k: v for k, v in (("hidden", args.hidden), ("epochs", args.epochs)) if v}
    config = TemporalConfig(arch=args.arch, encoder=args.encoder, aux_ball=args.aux_ball,
                            smooth=args.smooth, seg_balance=args.seg_balance,
                            focal=args.focal, **sizes)
    run(args.matches, grid, args.model, args.protocol, args.refresh, args.member, config)


if __name__ == "__main__":
    main()
