"""Label-convention calibration to the published baseline order (plan of action row LC1;
open data and the organisers' published aggregates only).

    python -m tempolens.labelcal step1               # the 6 x 8 diagnostic and the rule
    python -m tempolens.labelcal step2 --frames M    # re-score cached CV outputs at M frames

A convention is the opendata.remove_flicker fold at `min_frames`, applied to the labels of all
10 open matches (as E2a). Step 1 scores the simulated official baseline (experiments.
baseline_codes, half-windows 0 / 25 / 50 / 125) on the standard test draw under each convention
and checks the order of the four against the order published in EVALUATION.md. The public
ground-truth CSV and the match are never read.
"""

import argparse
from multiprocessing import Pool

import numpy as np

from .cvrescore import CONVENTIONS, NAMES, SETTINGS
from .decoder import transition_matrix
from .experiments import (SIM_TEST_SEED, CachedSample, baseline_codes, decode_table, oof_logp,
                          score)
from .pffstudy import labels_by_convention
from .temporal import combine

FRAMES = (10, 25, 37, 50, 75, 100)
WINDOWS = (("raw", 0), ("smooth25", 25), ("smooth50", 50), ("smooth125", 125))
SPELL_ORDER = ("smooth50", "smooth125", "smooth25", "raw")     # EVALUATION.md, best first
MACRO_ORDER = ("smooth25", "smooth50", "raw", "smooth125")
E2A = CONVENTIONS     # the four E2a label versions: provider, controlled 1 / 2 / 3 s


def in_order(values, order):
    """True if values[order[0]] > values[order[1]] > ... strictly."""
    return all(values[a] > values[b] for a, b in zip(order, order[1:]))


def verdict(passing):
    """(calibrated frames or None, text) from the passing indices into FRAMES (the LC1 rule)."""
    if not passing:
        return None, "inconclusive: no value reproduces both orders; the convention stays at 10"
    if list(passing) != list(range(passing[0], passing[-1] + 1)):
        return None, "inconclusive: passing values {} are not adjacent; stays at 10".format(
            [FRAMES[i] for i in passing])
    return FRAMES[passing[0]], "PASS: {} reproduce both orders; calibrated value {}".format(
        [FRAMES[i] for i in passing], FRAMES[passing[0]])


def labels_at(sample, frames):
    """The labels refolded at `frames` (0: the provider's, untouched)."""
    return labels_by_convention(sample, (frames,), provider=0)[frames]


def _step1_match(name):
    """(len(FRAMES), 4 windows, 2) spellF1 / macroF1 of the simulated baseline."""
    sample = CachedSample(name, SIM_TEST_SEED)
    table = sample.table()
    preds = [baseline_codes(table, w) for _, w in WINDOWS]
    out = np.zeros((len(FRAMES), len(WINDOWS), 2))
    for i, f in enumerate(FRAMES):
        y = labels_at(sample, f)
        for j, p in enumerate(preds):
            r = score(y, p)
            out[i, j] = r["spellF1"], r["macroF1"]
    return out


def step1(pool):
    mean = np.mean(pool.map(_step1_match, NAMES), axis=0)
    names = [n for n, _ in WINDOWS]
    print("frames  " + "  ".join("spell {:<9}".format(n) for n in names) + "  " +
          "  ".join("macro {:<9}".format(n) for n in names) + "  orders (spell / macro)")
    passing = []
    for i, f in enumerate(FRAMES):
        spell = dict(zip(names, mean[i, :, 0]))
        macro = dict(zip(names, mean[i, :, 1]))
        ok = in_order(spell, SPELL_ORDER), in_order(macro, MACRO_ORDER)
        if all(ok):
            passing.append(i)
        print("{:<7} ".format(f) + "  ".join("{:<15.4f}".format(spell[n]) for n in names) + "  " +
              "  ".join("{:<15.4f}".format(macro[n]) for n in names) + "  {} / {}".format(
                  *("yes" if o else "no" for o in ok)), flush=True)
    print("published spellF1 0.0486 / 0.1996 / 0.2433 / 0.2242, macroF1 0.6532 / 0.6752 / "
          "0.6724 / 0.6356 (raw / 25 / 50 / 125)")
    print("LC1 step 1: " + verdict(passing)[1])


def _step2_match(job):
    name, logp, frames = job
    sample = CachedSample(name, SIM_TEST_SEED)
    ys = [labels_at(sample, f) for _, f in E2A] + [labels_at(sample, frames)]
    out = np.zeros((len(SETTINGS), len(ys)))
    for i, s in enumerate(SETTINGS):
        pred = decode_table(sample, logp, transition_matrix(*s[:3]), s[3])
        out[i] = [score(y, pred)["spellF1"] for y in ys]
    return out


def step2(pool, frames):
    ens5 = [oof_logp(NAMES, "temporal", member=k, draws=4) for k in range(5)]
    ms50 = [oof_logp(NAMES, "temporal", member=k, draws=4, min_spell=50) for k in range(3)]
    models = {"ENS5": {n: combine([m[n] for m in ens5]) for n in NAMES},
              "v7 3 members": {n: combine([m[n] for m in ens5[:3]]) for n in NAMES},
              "ms50 3 members": {n: combine([m[n] for m in ms50]) for n in NAMES},
              "ENS5 + ms50 (8)": {n: combine([m[n] for m in ens5 + ms50]) for n in NAMES}}
    models.update({"ms50 m{}".format(k): ms50[k] for k in range(3)})
    best = {}
    print("model              provider  controlled-{}  four-convention mean (one setting)".format(
        frames))
    for label, logps in models.items():
        per = np.mean(pool.map(_step2_match, [(n, logps[n], frames) for n in NAMES]), axis=0)
        mean4 = per[:, :4].mean(axis=1)
        best[label] = (per[:, 0].max(), per[:, 4].max(), mean4.max())
        k = int(np.argmax(per[:, 4]))
        print("{:<18} {:.4f}    {:.4f} at {}   {:.4f}".format(
            label, best[label][0], best[label][1], SETTINGS[k], best[label][2]), flush=True)
    gain = best["ms50 3 members"][1] - best["ENS5"][1]
    four = best["ms50 3 members"][2] - best["ENS5"][2]
    trig = gain >= 0.006 and four >= -0.006
    print("LC1 step 2: ms50 vs ENS5 at {} frames {:+.4f}, four-convention mean {:+.4f} -> step 3 "
          "{} (it starts only on the user's go-ahead)".format(
              frames, gain, four, "triggered" if trig else "not triggered"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("step1", "step2"))
    parser.add_argument("--frames", type=int, help="step2: the calibrated min_frames")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.step == "step2" and args.frames is None:
        parser.error("step2 needs --frames")
    with Pool(args.workers) as pool:
        if args.step == "step1":
            step1(pool)
        else:
            step2(pool, args.frames)


if __name__ == "__main__":
    main()
