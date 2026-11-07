"""Cached-output studies from the independent review (plan of action rows B1, B4; aggregates only).

    python -m tempolens.reviewstudy b1    # member combination: geometric mean, max change logit
    python -m tempolens.reviewstudy b4    # change bonus split: H<->A switches vs live<->dead

Both read the cached held-out outputs of the 5 ENS5 members (experiments.oof_logp) and decode
the 10 open matches; nothing is trained. B4's split decoder lives here only; decoder.py is
unchanged.
"""

import argparse
from multiprocessing import Pool

import numpy as np

from .cvrescore import COSTS, NAMES, SETTINGS
from .decoder import transition_matrix
from .experiments import SIM_TEST_SEED, CachedSample, decode_table, oof_logp, score
from .match import X

ENS5 = (0.7760, 0.7650, 0.8017)        # all / DFL / Metrica at (2, 8, 8, 0.5)
WEIGHTS = (0.0, 0.5, 1.0, 1.5)
SPLIT = [c + (wt, wd) for c in COSTS for wt in WEIGHTS for wd in WEIGHTS]


def members():
    return [oof_logp(NAMES, "temporal", member=k, draws=4) for k in range(5)]


def combine(outputs, geometric=False, max_change=False):
    """temporal.combine, optionally with the geometric mean of the probabilities and/or the
    members' change logits max-pooled."""
    stack = np.stack(outputs)
    out = np.empty_like(stack[0])
    if geometric:
        lp = stack[:, :, :3].mean(axis=0)
        out[:, :3] = lp - np.logaddexp.reduce(lp, axis=1, keepdims=True)
    else:
        out[:, :3] = np.log(np.clip(np.exp(stack[:, :, :3]).mean(axis=0), 1e-6, 1.0))
    out[:, 3] = stack[:, :, 3].max(axis=0) if max_change else stack[:, :, 3].mean(axis=0)
    return np.round(out, 4)


def _split_loops(logits, transition, team_bonus, dead_bonus):
    """decoder._viterbi_loops with the change bonus split: `team_bonus` on H<->A, `dead_bonus`
    on every change into or out of D."""
    n = logits.shape[0]
    score_ = np.empty((n, 3))
    back = np.zeros((n, 3), dtype=np.int8)
    for s in range(3):
        score_[0, s] = logits[0, s]
    for t in range(1, n):
        for s in range(3):
            best, arg = -np.inf, 0
            for p in range(3):
                value = score_[t - 1, p] + transition[p, s]
                if p != s:
                    value += team_bonus[t] if (p < 2 and s < 2) else dead_bonus[t]
                if value > best:
                    best, arg = value, p
            back[t, s] = arg
            score_[t, s] = logits[t, s] + best
    states = np.empty(n, dtype=np.int8)
    last, arg = -np.inf, 0
    for s in range(3):
        if score_[n - 1, s] > last:
            last, arg = score_[n - 1, s], s
    states[n - 1] = arg
    for t in range(n - 2, -1, -1):
        states[t] = back[t + 1, states[t + 1]]
    return states


try:
    from numba import njit
    _split = njit(cache=True)(_split_loops)
except ImportError:
    _split = _split_loops


def decode_split(table, logp, transition, w_team, w_dead):
    """experiments.decode_table with separate change-bonus weights."""
    out = np.full(table.n_frames, X, dtype=np.int8)
    for start, stop in table.runs():
        if stop <= start:
            continue
        change = np.ascontiguousarray(logp[start:stop, 3], dtype=np.float64)
        out[start:stop] = _split(np.ascontiguousarray(logp[start:stop, :3], dtype=np.float64),
                                 np.ascontiguousarray(transition, dtype=np.float64),
                                 w_team * change, w_dead * change)
    return out


def _b1_match(job):
    """(settings, 3): spellF1, predicted spells, true spells of one match's output."""
    name, logp = job
    sample = CachedSample(name, SIM_TEST_SEED)
    out = np.zeros((len(SETTINGS), 3))
    for i, s in enumerate(SETTINGS):
        r = score(sample.labels, decode_table(sample, logp, transition_matrix(*s[:3]), s[3]))
        out[i] = r["spellF1"], r["n_pred"], r["n_gt"]
    return out


def _b4_match(job):
    name, logp = job
    sample = CachedSample(name, SIM_TEST_SEED)
    out = np.zeros(len(SPLIT))
    for i, s in enumerate(SPLIT):
        out[i] = score(sample.labels, decode_split(sample, logp, transition_matrix(*s[:3]),
                                                   s[3], s[4]))["spellF1"]
    return out


def _groups(per):
    dfl = [n for n in NAMES if n.startswith("dfl_")]
    met = [n for n in NAMES if n.startswith("metrica")]
    return [np.mean([per[n] for n in g], axis=0) for g in (NAMES, dfl, met)]


def b1(pool):
    outs = members()
    variants = {"probability mean (ENS5)": (False, False), "geometric mean": (True, False),
                "max change logit": (False, True), "geometric + max change": (True, True)}
    best = {}
    for label, (geo, mx) in variants.items():
        logps = {n: combine([o[n] for o in outs], geo, mx) for n in NAMES}
        per = dict(zip(NAMES, pool.map(_b1_match, [(n, logps[n]) for n in NAMES])))
        full, dfl, met = _groups(per)
        i = int(np.argmax(full[:, 0]))
        n_pred = sum(per[n][i, 1] for n in NAMES)
        n_gt = sum(per[n][i, 2] for n in NAMES)
        best[label] = (full[i, 0], dfl[i, 0], met[i, 0], n_pred, n_gt)
        print("{:<24} {:.4f} (DFL {:.4f}, Metrica {:.4f}) at {}; spells predicted {:.0f} / true "
              "{:.0f}".format(label, *best[label][:3], SETTINGS[i], n_pred, n_gt), flush=True)
    base = best["probability mean (ENS5)"]
    assert abs(base[0] - ENS5[0]) < 5e-4, "the plain combination must reproduce ENS5"
    passed = [k for k, v in best.items() if k != "probability mean (ENS5)"
              and v[0] - base[0] >= 0.004 and v[1] >= base[1] and v[2] >= base[2]
              and abs(v[3] - v[4]) < abs(base[3] - base[4])]
    for k, v in best.items():
        print("{:<24} {:+.4f} over ENS5 (DFL {:+.4f}, Metrica {:+.4f})".format(
            k, v[0] - base[0], v[1] - base[1], v[2] - base[2]))
    print("B1: {}".format("passes: " + max(passed, key=lambda k: best[k][0]) if passed
                          else "fails"))


def b4(pool):
    logps = {n: combine([o[n] for o in members()]) for n in NAMES}
    per = dict(zip(NAMES, pool.map(_b4_match, [(n, logps[n]) for n in NAMES])))
    full, dfl, met = _groups(per)
    equal = [i for i, s in enumerate(SPLIT) if s[3] == s[4]]
    e = max(equal, key=lambda i: full[i])
    i = int(np.argmax(full))
    print("equal weights (the plain decoder): {:.4f} at {}".format(full[e], SPLIT[e]))
    assert abs(full[e] - ENS5[0]) < 5e-4, "equal weights must reproduce ENS5"
    print("best split: {:.4f} (DFL {:.4f}, Metrica {:.4f}) at costs {} team {} dead {}".format(
        full[i], dfl[i], met[i], SPLIT[i][:3], SPLIT[i][3], SPLIT[i][4]))
    for wt in WEIGHTS:
        print("team {:<3} ".format(wt) + "  ".join(
            "dead {} {:.4f}".format(wd, max(full[k] for k, s in enumerate(SPLIT)
                                            if s[3] == wt and s[4] == wd)) for wd in WEIGHTS))
    print("B4: {} ({:+.4f} over 0.7760; rule +0.004)".format(
        "passes" if full[i] - ENS5[0] >= 0.004 else "fails", full[i] - ENS5[0]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("study", choices=("b1", "b4"))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)
    with Pool(args.workers) as pool:
        (b1 if args.study == "b1" else b4)(pool)


if __name__ == "__main__":
    main()
