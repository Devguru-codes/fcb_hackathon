"""Train the final temporal model on every open match under simulated ATD noise.

    python -m tempolens.train                 # member 0 -> models/retrained_m0.pt
    python -m tempolens.train --member 1      # member 1 -> models/retrained_m1.pt

The default is the submitted recipe (models/tempolens_rnd4_m0..2.pt): 4 randomised noise draws
per match (domain randomisation), both team perspectives, all 10 open matches; `--draws 0` gives
the older two fixed-noise draws. Training uses only open data (no competition asset), so the
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
    parser.add_argument("--draws", type=int, default=4,
                        help="randomised-noise draws per match (default 4, the submitted recipe; "
                             "0 = the two fixed-noise draws)")
    parser.add_argument("--faults", action="store_true",
                        help="with --draws: add tracker faults to the training draws")
    parser.add_argument("--extra-train", choices=("skillcorner",),
                        help="also train on the SkillCorner matches (one noise draw each)")
    parser.add_argument("--min-spell", type=int, default=opendata.DFL_MIN_SPELL,
                        help="flicker threshold for the DFL/SkillCorner owner flag, in frames")
    parser.add_argument("--drop", nargs="+", default=(),
                        help="feature names masked out of the input ('restart' = the v4 "
                             "restart-spot features)")
    parser.add_argument("--swa", type=int, default=0,
                        help="average the weights of the last N epochs (plan rows B3 / B3c)")
    args = parser.parse_args(argv)
    opendata.DFL_MIN_SPELL = args.min_spell
    out = args.out or "models/retrained_m{}.pt".format(args.member)
    if Path(out).name.startswith("tempolens_rnd4_m"):
        parser.error("never overwrite the submitted weights ({})".format(out))
    samples = training_samples(args.matches, args.member, args.draws, args.faults)
    if args.extra_train:     # as in the cross-validation: real broadcast noise, one draw each
        samples += [CachedSample(n, member_noise_seeds(args.member)[0])
                    for n in sorted(EXTRA_LOADERS)]
    config = temporal.TemporalConfig(hidden=args.hidden, epochs=args.epochs, chunk=args.chunk,
                                     touch=args.touch, drop=drop_names(args.drop), swa=args.swa)
    model, normaliser = temporal.train(samples, seed=temporal.SEED + args.member, config=config)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    temporal.save(model, normaliser, out)
    print("saved {} (member {}, trained on {}, {})".format(
        out, args.member, ", ".join(args.matches),
        "{} randomised draws each".format(args.draws) if args.draws else "fixed noise draws"))


if __name__ == "__main__":
    main()
