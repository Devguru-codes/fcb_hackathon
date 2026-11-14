"""Extra simulated test draws (plan of action row DRAW2). Report only, decides nothing.

    python -m tempolens.studies.draw2

ADseeds' ENS5 fold models on simulated test draws 1001 and 1002 of each held-out open match; draw
1000 (SIM_TEST_SEED) is ADseeds' cached sim rows (same models, same scorer). Per-match caches in
models/exp/draw2/, named by seed, so a rerun resumes.
"""

import numpy as np

from .. import adaptcv
from . import adaptseeds, swaadopt
from ..experiments import SIM_TEST_SEED, CachedSample, check_members, folds
from ..opendata import LOADERS

SEEDS = (SIM_TEST_SEED, 1001, 1002)
SETTINGS = {"none": (0.0, 0), "TTA 0.5": (0.5, 0), "TTA 0.5 + EATA 10": (0.5, 10)}
COSTS = ((4.0, 8.0, 8.0, 1.0), (2.0, 8.0, 8.0, 0.5))
OUT = adaptcv.OUT / "draw2"
MEMBERS = 5


def fold_path(held, k):
    return adaptcv.OUT / "models" / "{}_m{}.pt".format("-".join(held), k)


def score_match(members, sample):
    return adaptcv.score_match(members, sample, per_member=True, costs=COSTS, alphas=(0.0, 0.5),
                               steps=(0, 10), gates=(False,))


def rows_by_draw():
    """{seed: {match: rows}}; seed 1000 from ADseeds' cache, the others computed and cached."""
    out = {SIM_TEST_SEED: swaadopt.cached_results(swaadopt.ADSEEDS)["sim"]}
    names = sorted(LOADERS)
    check_members(OUT, [fold_path(held, k) for held in folds(names) for k in range(MEMBERS)])
    assert sorted(out[SIM_TEST_SEED]) == names, "ADseeds' sim rows are incomplete"
    for seed in SEEDS[1:]:
        out[seed] = {}
        for held in folds(names):
            members = None
            for n in held:
                def compute(n=n):
                    nonlocal members
                    members = members or adaptcv.fold_members(
                        held, sorted(set(names) - set(held)), MEMBERS)
                    return score_match(members, CachedSample(n, seed))
                out[seed][n] = adaptcv.cached(OUT, "sim{}".format(seed), n, compute)
            print("draw {}: scored {}".format(seed, ", ".join(held)), flush=True)
    return out


def summarise(by_draw):
    """{(member seed, setting name, costs): [10-match mean per draw]} from rows_by_draw()."""
    out = {}
    for draw in SEEDS:
        table = adaptseeds.table({"sim": by_draw[draw]})
        for (member, alpha, steps, gate, costs), conds in table.items():
            name = next((k for k, v in SETTINGS.items() if v == (alpha, steps)), None)
            if gate or name is None or costs not in COSTS:
                continue
            out.setdefault((member, name, costs), []).append(conds["sim"])
    return out


def report(s):
    ens = adaptseeds.ENSEMBLE
    print("\nensemble, mean over draws {} (sd of the draw means); difference from none (sd)".format(
        SEEDS))
    for costs in COSTS:
        for name in SETTINGS:
            v = np.array(s[(ens, name, costs)])
            d = v - np.array(s[(ens, "none", costs)])
            print("{:<18} {:<22} mean {:.4f} (sd {:.4f})   vs none {:+.4f} (sd {:.4f})   per draw "
                  "{}".format(name, str(costs), v.mean(), v.std(ddof=1), d.mean(), d.std(ddof=1),
                              " ".join("{:.4f}".format(x) for x in v)), flush=True)
    full, alone = "TTA 0.5 + EATA 10", "TTA 0.5"
    for costs in COSTS:
        e = np.array(s[(ens, full, costs)]) - np.array(s[(ens, alone, costs)])
        members = [np.array(s[(k, full, costs)]) - np.array(s[(k, alone, costs)])
                   for k in range(MEMBERS)]
        print("TTA + EATA >= TTA alone at {}: ensemble on {} of {} draws ({}); members on {} of {} "
              "member x draw cases".format(costs, int((e >= 0).sum()), len(e),
                                           " ".join("{:+.4f}".format(x) for x in e),
                                           int(sum((m >= 0).sum() for m in members)),
                                           MEMBERS * len(e)), flush=True)
    for costs in COSTS:
        spread = [np.std(np.array(s[(k, n, costs)]) - np.array(s[(k, "none", costs)]), ddof=1)
                  for k in range(MEMBERS) for n in (full, alone)]
        print("members at {}: sd over draws of the difference from none, mean {:.4f}, max {:.4f}"
              .format(costs, np.mean(spread), np.max(spread)), flush=True)


def main():
    report(summarise(rows_by_draw()))


if __name__ == "__main__":
    main()
