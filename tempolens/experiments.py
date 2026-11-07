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
from .simulate import AtdNoise, TrackerFaults, add_tracker_faults, random_noise, simulate_atd

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
RANDOM_VERSION = 1   # simulate.random_noise's ranges
KEEPER_SWAPS = 2      # keeper-swap stretches per faulted test match (E1)
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

def _replace_retry(src, dst, attempts=20):
    """os.replace that waits out Windows' brief locks (antivirus scanning a file just written)."""
    for attempt in range(attempts):
        try:
            return os.replace(src, dst)
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.5)


class CachedSample:
    """An open match, optionally corrupted by one noise draw, with features cached on disk.
    `randomised`: the draw also picks its own noise setting (simulate.random_noise);
    `faults`: and adds tracker faults (identity swaps, duplicate and phantom tracks);
    `keeper`: the two keepers' tracks swap slots KEEPER_SWAPS times (1-5 min each, plan row E1);
    `repair`: then match.repair_keepers with these keyword settings."""

    def __init__(self, name, seed=None, randomised=False, faults=False, keeper=False, repair=None):
        self.name, self.seed, self.randomised, self.faults = name, seed, randomised, faults
        self.keeper, self.repair = keeper, repair
        clean = load_open_match(name)
        self.labels, self.period, self.n_frames = clean.labels, clean.period, clean.n_frames
        self._runs = clean.runs()

    def runs(self):
        return self._runs

    def table(self):
        clean = load_open_match(self.name)
        if self.seed is None:
            return clean
        noise = random_noise(self.seed) if self.randomised else AtdNoise()
        table = simulate_atd(clean, noise, seed=self.seed)
        table = add_tracker_faults(table, seed=self.seed) if self.faults else table
        if self.keeper:
            table = add_tracker_faults(table, TrackerFaults(
                swap_rate=0, duplicate_rate=0, phantom_rate=0, keeper_swaps=KEEPER_SWAPS),
                seed=self.seed)
        if self.repair:
            from .match import repair_keepers
            table = repair_keepers(table, **self.repair)
        return table

    def _path(self, swap, kind="f{}".format(FEATURE_VERSION)):
        tag = ("clean" if self.seed is None else "{}{}v{}_{}".format(
               "rnf" if self.faults else "rnd", NOISE_TAG, RANDOM_VERSION, self.seed)
               if self.randomised else "sim{}_{}".format(NOISE_TAG, self.seed))
        tag += "_kf{}".format(KEEPER_SWAPS) if self.keeper else ""
        if self.repair:
            tag += "_kr{window_s:g}-{occupancy:g}-{present:g}".format(**self.repair)
        return (open_data_dir() / "cache" / "features" /
                "{}_{}_{}_{}.npy".format(self.name, tag, kind, int(swap)))

    def _cached(self, kind, swap, compute):
        """np.load(mmap) of a per-perspective array, computing both perspectives once."""
        path = self._path(swap, kind)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            for side in (False, True):
                final = self._path(side, kind)
                if final.exists():     # keep a file this process may already have memory-mapped
                    continue
                partial = final.with_name(final.stem + ".partial.npy")
                np.save(partial, compute(side))
                _replace_retry(partial, final)
        return np.load(path, mmap_mode="r")

    def tokens(self, swap=False):
        from .features import player_tokens
        table = {}

        def compute(side):
            table.setdefault("t", self.table())
            return player_tokens(table["t"], swap=side)
        return self._cached("tok{}".format(FEATURE_VERSION), swap, compute)

    def teamcue(self, swap=False):
        """features.team_cue_features of this draw, cached like the features."""
        from .features import TEAMCUE_VERSION, team_cue_features
        table = {}

        def compute(side):
            table.setdefault("t", self.table())
            return team_cue_features(table["t"], swap=side)
        return self._cached("tc{}".format(TEAMCUE_VERSION), swap, compute)

    def touch_target(self, swap=False):
        """(n, 2) home / away touch targets from the match's events, from this side's view."""
        from .touches import touch_targets
        out = touch_targets(self.name, self.n_frames)
        return out[:, ::-1].copy() if swap else out

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
                # Only missing files: one already mapped by this process cannot be replaced.
                final = self._path(side)
                if final.exists():
                    continue
                partial = final.with_name(final.stem + ".partial.npy")
                np.save(partial, frame_features(table, swap=side)[0])
                _replace_retry(partial, final)
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


def training_samples(names, member, draws=0, faults=False):
    """Training draws of `names` for ensemble member k: the two fixed-noise draws, or with
    `draws` > 0 that many randomised-noise draws (seeds 100 + 10k + i, so fewer draws are a
    subset of more); `faults` adds tracker faults to those draws."""
    if draws:
        return [CachedSample(n, 100 + 10 * member + i, randomised=True, faults=faults)
                for n in sorted(names) for i in range(draws)]
    return [CachedSample(n, s) for n in sorted(names) for s in member_noise_seeds(member)]


def oof_logp(names, kind="gbm", protocol="sim", refresh=False, member=0,
             members=ENSEMBLE_MEMBERS, config=None, extra_train=(), draws=0, faults=False,
             min_spell=None, sc_visible=False):
    """Held-out log-probabilities for every match, each from a model that never saw it.

    kind "temporal" trains the network described by `config` (default: the base GRU).
    kind "ensemble" averages the held-out outputs of temporal members 0..members-1 (or the
    member indices in `members` when it is a list), each of which is trained (or read from
    cache) like kind "temporal" with that member index.
    `extra_train`: matches added to every fold's training set and never held out, so the
    held-out matches and their scores stay comparable with runs without them.
    `draws` > 0: train on that many randomised-noise draws per match (training_samples); the
    held-out matches keep the fixed test noise, so scores stay comparable.
    `min_spell`: the flicker threshold of the *training* labels, DFL and SkillCorner only (Metrica
    labels are never filtered; held-out labels keep opendata.DFL_MIN_SPELL). `sc_visible`: extra matches enter the loss only on frames
    where their ball is detected.
    """
    if kind == "ensemble":
        from .temporal import combine
        outputs = [oof_logp(names, "temporal", protocol, refresh, k, config=config,
                            extra_train=extra_train, draws=draws, faults=faults,
                            min_spell=min_spell, sc_visible=sc_visible)
                   for k in (range(members) if isinstance(members, int) else members)]
        return {n: combine([out[n] for out in outputs]) for n in names}
    if kind == "temporal":
        from .temporal import DEFAULT_CONFIG
        config = config or DEFAULT_CONFIG
        kind_tag = config.tag() + ("" if TRAIN_VERSION == 1 else "_t{}".format(TRAIN_VERSION))
        if draws:
            kind_tag += "_rnd{}v{}{}".format(draws, RANDOM_VERSION, "f" if faults else "")
        if config.touch:                # the touch targets' recipe is part of what was trained
            from .touches import TOUCH_VERSION
            kind_tag += "_tv{}".format(TOUCH_VERSION)
        if min_spell is not None:
            kind_tag += "_ms{}".format(min_spell)
        if sc_visible:
            kind_tag += "_scvis"
    else:
        kind_tag = kind
    cache = open_data_dir() / "cache" / "oof"
    cache.mkdir(parents=True, exist_ok=True)
    noise = NOISE_TAG if protocol == "sim" else "clean"
    train_seeds = member_noise_seeds(member) if protocol == "sim" else (None,)
    test_seed = SIM_TEST_SEED if protocol == "sim" else None
    # members of the default model keep their historical names (temporal1, temporal2); other
    # configs use a separator so that e.g. sb0.5 member 1 cannot read as sb0.51
    tag = (kind_tag if member == 0 else "{}{}".format(kind_tag, member) if kind_tag == "temporal"
           else "{}_m{}".format(kind_tag, member))
    logps = {}
    for held in folds(names):
        key = "-".join(sorted(set(names) - set(held)) + sorted(extra_train))
        paths = {n: cache / "{}_{}_f{}_{}_train-{}.npy".format(
            tag, noise, FEATURE_VERSION, n, hashlib.md5(key.encode()).hexdigest()[:8])
            for n in held}
        if not refresh and all(p.exists() for p in paths.values()):
            logps.update({n: np.load(p) for n, p in paths.items()})
            continue
        t0 = time.time()
        # extra matches carry real broadcast-tracking noise already: one noise draw each keeps
        # memory in bounds (~3.8 GB of training arrays with 20 SkillCorner matches)
        rest = set(names) - set(held)
        from . import opendata
        default_spell = opendata.DFL_MIN_SPELL
        if min_spell is not None:      # labels are read when a CachedSample is made
            opendata.DFL_MIN_SPELL = min_spell
        try:
            train = ((training_samples(rest, member, draws, faults) if protocol == "sim"
                      else [CachedSample(n) for n in sorted(rest)]))
            extra = [CachedSample(n, train_seeds[0]) for n in sorted(extra_train)]
        finally:
            opendata.DFL_MIN_SPELL = default_spell
        if sc_visible:
            for sample in extra:
                seen = np.isfinite(load_open_match(sample.name).ball[:, 0])
                sample.labels = np.where(seen, sample.labels, X).astype(sample.labels.dtype)
        train += extra
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


def run(names, grid, kind="gbm", protocol="sim", refresh=False, member=0, config=None,
        members=ENSEMBLE_MEMBERS, extra_train=(), draws=0, faults=False, min_spell=None,
        sc_visible=False):
    test_seed = SIM_TEST_SEED if protocol == "sim" else None
    logps = oof_logp(names, kind, protocol, refresh, member, members, config=config,
                     extra_train=extra_train, draws=draws, faults=faults,
                     min_spell=min_spell, sc_visible=sc_visible)
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


def drop_names(drop):
    """CLI --drop values as a config tuple; 'restart' stands for features.RESTART_FEATURES."""
    from .features import RESTART_FEATURES, feature_names
    names = tuple(sorted(n for d in drop for n in (RESTART_FEATURES if d == "restart" else (d,))))
    unknown = set(names) - set(feature_names())
    if unknown:
        raise SystemExit("unknown feature(s): {}".format(", ".join(sorted(unknown))))
    return names


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
    parser.add_argument("--mirror", action="store_true",
                        help="reflect half the training sequences across the long axis")
    parser.add_argument("--dead-weight", type=float, default=1.0,
                        help="cross-entropy weight of dead-ball frames (1 = off)")
    parser.add_argument("--touch", type=float, default=0.0,
                        help="weight of the auxiliary touch-head loss (0 = off)")
    parser.add_argument("--chunk", type=int, default=None,
                        help="temporal training sequence length in frames (default 750 = 30 s)")
    parser.add_argument("--member-set", type=int, nargs="+",
                        help="ensemble of exactly these member indices (overrides --members)")
    parser.add_argument("--members", type=int, default=ENSEMBLE_MEMBERS,
                        help="ensemble only: number of members averaged")
    parser.add_argument("--extra-train", choices=("skillcorner",),
                        help="add these open matches to every fold's training set only")
    parser.add_argument("--draws", type=int, default=0,
                        help="train on this many randomised-noise draws per match (0 = the two "
                             "fixed-noise draws)")
    parser.add_argument("--faults", action="store_true",
                        help="with --draws: add tracker faults to the training draws")
    parser.add_argument("--drop", nargs="+", default=(),
                        help="feature names masked out of the network's input ('restart' = "
                             "the v4 restart-spot features)")
    parser.add_argument("--protocol", default="sim", choices=("sim", "clean"))
    parser.add_argument("--teamcue", action="store_true",
                        help="append the team-cue feature pack (features.team_cue_features)")
    parser.add_argument("--hedge", type=float, default=0.0,
                        help="weight of a second head on 2 s-controlled labels (B5)")
    parser.add_argument("--swa", type=int, default=0,
                        help="average each member's weights over its last N epochs (B3)")
    parser.add_argument("--min-spell", type=int, default=None,
                        help="flicker threshold of the training labels, in frames, DFL and "
                             "SkillCorner only (Metrica unchanged; held-out labels unchanged)")
    parser.add_argument("--sc-visible", action="store_true",
                        help="--extra-train frames enter the loss only where the ball is detected")
    parser.add_argument("--refresh", action="store_true", help="retrain, ignoring the cache")
    args = parser.parse_args(argv)
    # A switch dearer than alive>dead + dead>alive is never taken directly (the decoder routes
    # through one dead frame instead, which spellF1 does not see), so the grid stops there.
    grid = [(s, ad, da) for s in (1.0, 1.5, 2.0, 4.0, 6.0, 8.0, 12.0) for ad in (2.0, 4.0, 8.0)
            for da in (2.0, 4.0, 8.0)]
    from .temporal import TemporalConfig
    sizes = {k: v for k, v in (("hidden", args.hidden), ("epochs", args.epochs),
                               ("chunk", args.chunk)) if v}
    config = TemporalConfig(arch=args.arch, encoder=args.encoder, aux_ball=args.aux_ball,
                            smooth=args.smooth, seg_balance=args.seg_balance,
                            focal=args.focal, mirror=args.mirror,
                            dead_weight=args.dead_weight, touch=args.touch,
                            teamcue=args.teamcue, swa=args.swa, hedge=args.hedge,
                            drop=drop_names(args.drop), **sizes)
    from .opendata import EXTRA_LOADERS
    extra = sorted(EXTRA_LOADERS) if args.extra_train == "skillcorner" else ()
    run(args.matches, grid, args.model, args.protocol, args.refresh, args.member, config,
        args.member_set or args.members, extra, args.draws, args.faults, args.min_spell,
        args.sc_visible)


if __name__ == "__main__":
    main()
