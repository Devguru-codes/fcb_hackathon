"""Train the final temporal model on every open match under simulated ATD noise.

    python -m tempolens.train --out models/tempolens_gru.pt

Same recipe as the cross-validation in tempolens.experiments (two noise draws per match, both
team perspectives), on all matches. Training uses only open data (no competition asset), so the
weights may be committed. GPU training is not bit-reproducible, which is why the weights are
committed: the submission is regenerated from them by CPU inference, which is.
"""

import argparse
from pathlib import Path

from . import temporal
from .experiments import SIM_TRAIN_SEEDS, CachedSample
from .opendata import LOADERS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="models/tempolens_gru.pt")
    parser.add_argument("--matches", nargs="+", default=sorted(LOADERS))
    parser.add_argument("--epochs", type=int, default=temporal.EPOCHS)
    parser.add_argument("--seed", type=int, default=temporal.SEED)
    args = parser.parse_args(argv)
    samples = [CachedSample(name, seed) for name in args.matches for seed in SIM_TRAIN_SEEDS]
    model, normaliser = temporal.train(samples, epochs=args.epochs, seed=args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    temporal.save(model, normaliser, args.out)
    print("saved {} (trained on {}, noise draws {})".format(
        args.out, ", ".join(args.matches), SIM_TRAIN_SEEDS))


if __name__ == "__main__":
    main()
