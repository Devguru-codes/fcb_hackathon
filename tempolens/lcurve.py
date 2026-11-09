"""Learning curve (plan of action row LCURVE). Report only, decides nothing.

    python -m tempolens.lcurve

One member (member 0, the submitted recipe) per CV fold, trained on the first 4 / 6 / 8 of the fold's
other matches in a fixed order (sorted, then permuted by default_rng(2027)); the 8-match models are
ADseeds' fold models. Each held-out match is scored on its standard test draw, no adaptation, at
the no-adaptation CV argmax costs.
"""

import sys

import numpy as np

from . import adaptcv, temporal
from .decoder import transition_matrix
from .experiments import SIM_TEST_SEED, CachedSample, decode_table, folds, score
from .opendata import LOADERS

SIZES = (4, 6, 8)
SEED = 2027
COSTS = (2.0, 8.0, 8.0, 0.5)
OUT = adaptcv.ROOT / "models" / "exp" / "lcurve"


def subsets(names=None):
    """[(held, {size: training matches})] for the 5 folds, nested prefixes of a fixed order."""
    names = sorted(names or LOADERS)
    out = []
    for held in folds(names):
        rest = sorted(set(names) - set(held))
        order = [rest[i] for i in np.random.default_rng(SEED).permutation(len(rest))]
        out.append((held, {k: order[:k] for k in SIZES}))
    return out


def model_path(held, size):
    if size == len(LOADERS) - len(held):      # every other match: ADseeds' fold model
        return adaptcv.OUT / "models" / "{}_m0.pt".format("-".join(held))
    return OUT / "{}_n{}_m0.pt".format("-".join(held), size)


def main():
    transition = transition_matrix(*COSTS[:3])
    scores = {k: [] for k in SIZES}
    for held, chosen in subsets():
        for size, matches in chosen.items():
            path = model_path(held, size)
            adaptcv.train_once(path, [sys.executable, "-m", "tempolens.train", "--member", "0",
                                      "--draws", str(adaptcv.DRAWS), "--out", str(path),
                                      "--matches", *matches])
            model, normaliser = temporal.load(path)
            for n in held:
                sample = CachedSample(n, SIM_TEST_SEED)
                pred = decode_table(sample, temporal.predict(model, normaliser, sample),
                                    transition, COSTS[3])
                scores[size].append(score(sample.labels, pred)["spellF1"])
            print("fold {} size {} done".format("-".join(held), size), flush=True)
    for size in SIZES:
        print("LCURVE: {} training matches -> mean spellF1 {:.4f} over {} held-out matches "
              "(one member, no adaptation, costs {})".format(size, np.mean(scores[size]),
                                                              len(scores[size]), COSTS), flush=True)


if __name__ == "__main__":
    main()
