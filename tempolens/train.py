"""Train the final temporal model on every open match under simulated ATD noise.

    python -m tempolens.train                 # member 0 -> models/retrained_m0.pt
    python -m tempolens.train --member 1      # member 1 -> models/retrained_m1.pt

Same recipe as the cross-validation in tempolens.experiments (two noise draws per match, both
team perspectives), on all matches. Training uses only open data (no competition asset), so the
weights may be committed. GPU training is not bit-reproducible, which is why the weights are
committed: the submission is regenerated from them by CPU inference, which is.
"""

import argparse
from pathlib import Path

from . import opendata, temporal
from .experiments import CachedSample, drop_names, member_noise_seeds, training_samples
from .opendata import EXTRA_LOADERS, LOADERS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--member", type=int, default=0,
                        help="ensemble member: sets the network seed and the noise draws")
    parser.add_argument("--out", help="default models/retrained_m<member>.pt (never the "
                                      "committed submission weights)")
    parser.add_argument("--matches", nargs="+", default=sorted(LOADERS))
    parser.add_argument("--epochs", type=int, default=temporal.EPOCHS)
    parser.add_argument("--hidden", type=int, default=temporal.HIDDEN)
    parser.add_argument("--touch", type=float, default=0.0, help="touch-head loss weight")
    parser.add_argument("--chunk", type=int, default=temporal.CHUNK,
                        help="training sequence length in frames")
    parser.add_argument("--draws", type=int, default=0,
                        help="randomised-noise draws per match (0 = the two fixed-noise draws)")
    parser.add_argument("--faults", action="store_true",
                        help="with --draws: add tracker faults to the training draws")
    parser.add_argument("--extra-train", choices=("skillcorner",),
                        help="also train on the SkillCorner matches (one noise draw each)")
    parser.add_argument("--min-spell", type=int, default=opendata.DFL_MIN_SPELL,
                        help="flicker threshold for the DFL/SkillCorner owner flag, in frames")
    parser.add_argument("--drop", nargs="+", default=(),
                        help="feature names masked out of the input ('restart' = the v4 "
                             "restart-spot features)")
    args = parser.parse_args(argv)
    opendata.DFL_MIN_SPELL = args.min_spell
    out = args.out or "models/retrained_m{}.pt".format(args.member)
    samples = training_samples(args.matches, args.member, args.draws, args.faults)
    if args.extra_train:     # as in the cross-validation: real broadcast noise, one draw each
        samples += [CachedSample(n, member_noise_seeds(args.member)[0])
                    for n in sorted(EXTRA_LOADERS)]
    config = temporal.TemporalConfig(hidden=args.hidden, epochs=args.epochs, chunk=args.chunk,
                                     touch=args.touch, drop=drop_names(args.drop))
    model, normaliser = temporal.train(samples, seed=temporal.SEED + args.member, config=config)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    temporal.save(model, normaliser, out)
    print("saved {} (member {}, trained on {}, {})".format(
        out, args.member, ", ".join(args.matches),
        "{} randomised draws each".format(args.draws) if args.draws else "fixed noise draws"))


if __name__ == "__main__":
    main()
