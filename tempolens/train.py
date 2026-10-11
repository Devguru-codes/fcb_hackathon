"""Train the final temporal model on every open match under simulated ATD noise.

    python -m tempolens.train                 # member 0 -> models/tempolens_gru.pt
    python -m tempolens.train --member 1      # member 1 -> models/tempolens_gru_m1.pt

Same recipe as the cross-validation in tempolens.experiments (two noise draws per match, both
team perspectives), on all matches. Training uses only open data (no competition asset), so the
weights may be committed. GPU training is not bit-reproducible, which is why the weights are
committed: the submission is regenerated from them by CPU inference, which is.
"""

import argparse
from pathlib import Path

from . import temporal
from .experiments import CachedSample, member_noise_seeds
from .opendata import LOADERS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--member", type=int, default=0,
                        help="ensemble member: sets the network seed and the noise draws")
    parser.add_argument("--out", help="default models/tempolens_gru[_m<member>].pt")
    parser.add_argument("--matches", nargs="+", default=sorted(LOADERS))
    parser.add_argument("--epochs", type=int, default=temporal.EPOCHS)
    args = parser.parse_args(argv)
    out = args.out or ("models/tempolens_gru.pt" if args.member == 0
                       else "models/tempolens_gru_m{}.pt".format(args.member))
    seeds = member_noise_seeds(args.member)
    samples = [CachedSample(name, seed) for name in args.matches for seed in seeds]
    model, normaliser = temporal.train(samples, epochs=args.epochs,
                                       seed=temporal.SEED + args.member)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    temporal.save(model, normaliser, out)
    print("saved {} (member {}, trained on {}, noise draws {})".format(
        out, args.member, ", ".join(args.matches), seeds))


if __name__ == "__main__":
    main()
