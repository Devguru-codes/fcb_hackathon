"""Rescore cross-validated open-data outputs (plan of action rows E6, E2a, E2b; aggregates only).

    python -m tempolens.cvrescore temperature            # E6: dead-ball log-odds / T, ENS5
    python -m tempolens.cvrescore conventions            # E2a: ENS5 and v7 vs four label versions
    python -m tempolens.cvrescore conventions --min-spell 50   # E2b: its members vs v7's

Every model is decoded over the cross-validation's 189 decoder settings (63 costs x 3 change
bonuses); a "best" number is the best setting for that model. Label versions: "provider" is the
label the CV always uses; "controlled-Xs" also folds every possession shorter than X seconds that
sits between two possessions of the other team into them (opendata.remove_flicker), on all ten
matches.
"""

import argparse
from multiprocessing import Pool

import numpy as np

from .decoder import transition_matrix
from .experiments import COST_GRID as COSTS, SIM_TEST_SEED, CachedSample, decode_table, oof_logp, \
    score
from .match import A, D, H
from .opendata import LOADERS, remove_flicker

SETTINGS = [c + (w,) for c in COSTS for w in (0.0, 0.5, 1.0)]
CONVENTIONS = (("provider", 0), ("controlled-1s", 25), ("controlled-2s", 50),
               ("controlled-3s", 75))
NAMES = sorted(LOADERS)


def soften_dead(logp, temperature):
    """`logp` with its dead-ball log-odds divided by `temperature`; H:A ratio and the change
    logit kept."""
    out = logp.astype(np.float64).copy()
    alive = np.logaddexp(out[:, H], out[:, A])
    odds = (out[:, D] - alive) / temperature
    log_alive = -np.logaddexp(0.0, odds)
    out[:, D] = -np.logaddexp(0.0, -odds)
    out[:, H] += log_alive - alive
    out[:, A] += log_alive - alive
    return out


def _match_scores(job):
    """(settings, label versions) spellF1 of one match's output."""
    name, logp = job
    sample = CachedSample(name, SIM_TEST_SEED)
    labels = [remove_flicker(sample.labels, sample.runs(), f) if f else sample.labels
              for _, f in CONVENTIONS]
    out = np.zeros((len(SETTINGS), len(labels)))
    for i, setting in enumerate(SETTINGS):
        pred = decode_table(sample, logp, transition_matrix(*setting[:3]), setting[3])
        out[i] = [score(y, pred)["spellF1"] for y in labels]
    return out


def scores(logps, pool):
    """{name: (settings, label versions)} for one model's held-out outputs."""
    return dict(zip(NAMES, pool.map(_match_scores, [(n, logps[n]) for n in NAMES])))


def summarise(per):
    """(settings, label versions) means over all, DFL and Metrica matches."""
    groups = (NAMES, [n for n in NAMES if n.startswith("dfl_")],
              [n for n in NAMES if n.startswith("metrica")])
    return [np.mean([per[n] for n in g], axis=0) for g in groups]


def report(label, per):
    """Print a model's best setting per label version and for the mean of the four; returns
    (mean of four at its best setting, provider at its best setting, metrica provider)."""
    full, dfl, metrica = summarise(per)
    mean4 = full.mean(axis=1)
    k = int(np.argmax(mean4))
    p = int(np.argmax(full[:, 0]))
    print("{:<18} ".format(label) + "  ".join(
        "{} {:.4f}".format(c, full[:, j].max()) for j, (c, _) in enumerate(CONVENTIONS)) +
        "  | mean4 {:.4f} at {}  (provider dfl {:.4f} metrica {:.4f} at {})".format(
            mean4[k], SETTINGS[k], dfl[p, 0], metrica[p, 0], SETTINGS[p]), flush=True)
    return mean4[k], full[p, 0], metrica[p, 0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("study", choices=("temperature", "conventions"))
    parser.add_argument("--min-spell", type=int, default=None,
                        help="also score the members trained with this label threshold (E2b)")
    parser.add_argument("--swa", type=int, default=None,
                        help="also score the 5-member SWA ensemble of this many epochs (B3c)")
    parser.add_argument("--hedge", type=float, default=None,
                        help="also score the members trained with a hedge head of this weight (B5)")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)
    with Pool(args.workers) as pool:
        if args.study == "temperature":
            ens5 = oof_logp(NAMES, "ensemble", members=5, draws=4)
            best = {}
            for t in (1.0, 2.0, 4.0):
                full, dfl, metrica = summarise(scores(
                    {n: soften_dead(v, t) for n, v in ens5.items()}, pool))
                k = int(np.argmax(full[:, 0]))
                best[t] = full[k, 0]
                print("T {:<3} best {:.4f} (dfl {:.4f}, metrica {:.4f}) at {}".format(
                    t, full[k, 0], dfl[k, 0], metrica[k, 0], SETTINGS[k]), flush=True)
            t = max(best, key=best.get)
            print("E6: best T {} {:+.4f} over T 1 -> {}".format(
                t, best[t] - best[1.0], "passes (>= +0.004)" if best[t] - best[1.0] >= 0.004
                else "fails (< +0.004)"))
            return
        models = {"ENS5": oof_logp(NAMES, "ensemble", members=5, draws=4),
                  "v7 3 members": oof_logp(NAMES, "ensemble", members=3, draws=4)}
        if args.min_spell is not None:
            tag = "ms{}".format(args.min_spell)
            models["{} 3 members".format(tag)] = oof_logp(NAMES, "ensemble", members=3, draws=4,
                                                         min_spell=args.min_spell)
            for k in range(3):
                models["v7 m{}".format(k)] = oof_logp(NAMES, "temporal", member=k, draws=4)
                models["{} m{}".format(tag, k)] = oof_logp(NAMES, "temporal", member=k, draws=4,
                                                           min_spell=args.min_spell)
        if args.swa is not None:
            from .temporal import TemporalConfig
            models["swa 5 members"] = oof_logp(NAMES, "ensemble", members=5, draws=4,
                                               config=TemporalConfig(swa=args.swa))
        if args.hedge is not None:
            from .temporal import TemporalConfig
            config = TemporalConfig(hedge=args.hedge)
            models["hedge 3 members"] = oof_logp(NAMES, "ensemble", members=3, draws=4,
                                                 config=config)
            for k in range(3):
                models["hedge m{}".format(k)] = oof_logp(NAMES, "temporal", member=k, draws=4,
                                                         config=config)
        out = {label: report(label, scores(logps, pool)) for label, logps in models.items()}
        if args.hedge is not None:
            gain = out["hedge 3 members"][1] - out["v7 3 members"][1]
            v7 = (0.7576, 0.7677, 0.7633)
            ahead = sum(out["hedge m{}".format(k)][1] > v7[k] for k in range(3))
            ok = gain >= 0.006 and ahead >= 2 and out["hedge 3 members"][0] >= 0.7601
            print("B5: provider {:+.4f} over v7 (0.7716), {} of 3 members ahead, four-convention "
                  "mean {:.4f} (v7 0.7601) -> {}".format(gain, ahead, out["hedge 3 members"][0],
                                                         "passes" if ok else "fails"))
        if args.min_spell is not None:
            tag = "ms{}".format(args.min_spell)
            gain = out["{} 3 members".format(tag)][0] - out["v7 3 members"][0]
            ahead = sum(out["{} m{}".format(tag, k)][0] > out["v7 m{}".format(k)][0]
                        for k in range(3))
            drop = out["{} 3 members".format(tag)][1] - out["v7 3 members"][1]
            ok = gain >= 0.006 and ahead >= 2 and drop >= -0.006
            print("E2b: mean4 {:+.4f}, {} of 3 members ahead, provider {:+.4f} -> {}".format(
                gain, ahead, drop, "passes" if ok else "fails"))


if __name__ == "__main__":
    main()
