"""Held-out spellF1 after every training epoch (plan of action row E5; open data, aggregates only).

    python -m tempolens.studies.epochstudy           # members 0 and 1, v7 recipe, 16 epochs, 5 folds

Each fold trains one 16-epoch v7-recipe member (`--draws 4`) and, after epochs 4..16, decodes
its held-out matches at costs (2, 8, 8, 0.5). The learning rate follows one cycle over all 16
epochs, so a snapshot after epoch k is not a model trained for k epochs. Results are cached per
(member, fold) in models/exp/epochstudy/.
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from .. import temporal
from ..decoder import transition_matrix
from ..experiments import SIM_TEST_SEED, CachedSample, decode_table, folds, score, \
    training_samples, write_atomic
from ..opendata import LOADERS

OUT = Path(__file__).resolve().parents[2] / "models" / "exp" / "epochstudy"
EPOCHS, FIRST, COSTS = 16, 4, (2.0, 8.0, 8.0, 0.5)


def fold_scores(member, held, rest):
    """{epoch: {match: spellF1}} for one fold, cached."""
    path = OUT / "m{}_{}.json".format(member, "-".join(held))
    if path.exists():
        return {int(k): v for k, v in json.loads(path.read_text(encoding="utf-8")).items()}
    tests = {n: CachedSample(n, SIM_TEST_SEED) for n in held}
    out = {}

    def on_epoch(epoch, model, normaliser):
        if epoch < FIRST:
            return
        snap = copy.deepcopy(model).cpu().eval()
        out[epoch] = {n: score(s.labels, decode_table(
            s, temporal.predict(snap, normaliser, s), transition_matrix(*COSTS[:3]),
            COSTS[3]))["spellF1"] for n, s in tests.items()}
        print("  m{} epoch {:>2}: {}".format(member, epoch, " ".join(
            "{:.4f}".format(v) for v in out[epoch].values())), flush=True)

    temporal.train(training_samples(rest, member, draws=4), seed=temporal.SEED + member,
                   config=temporal.TemporalConfig(epochs=EPOCHS), on_epoch=on_epoch)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(path, json.dumps(out))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--members", type=int, nargs="+", default=[0, 1])
    args = parser.parse_args(argv)
    names = sorted(LOADERS)
    means = {}
    for member in args.members:
        per = {}
        for held in folds(names):
            for epoch, rows in fold_scores(member, held, sorted(set(names) - set(held))).items():
                per.setdefault(epoch, {}).update(rows)
        means[member] = {e: float(np.mean(list(v.values()))) for e, v in per.items()
                         if len(v) == len(names)}
    epochs = sorted(set.intersection(*(set(m) for m in means.values())))
    print("\nepoch " + " ".join("m{:<7}".format(m) for m in means))
    for e in epochs:
        print("{:<5} ".format(e) + " ".join("{:.4f}  ".format(means[m][e]) for m in means))
    gains = {m: max(v.values()) - v[8] for m, v in means.items()}
    best = {m: max(v, key=v.get) for m, v in means.items()}
    print("best epoch per member {}; gain over epoch 8 {} -> {}".format(
        best, {m: round(g, 4) for m, g in gains.items()},
        "passes (>= +0.006 on every seed)" if all(g >= 0.006 for g in gains.values())
        else "fails"))


if __name__ == "__main__":
    main()
