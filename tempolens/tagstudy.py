"""Does the self-calibrated Smart Tagging prior help when tags are good, and stay harmless when
they are bad? Tested on open matches with synthetic tags, since real tags exist only for the
competition match.

    python -m tempolens.tagstudy

Synthetic tags imitate Smart Tagging's shape rather than its exact error profile, which is
unknown: phases attached to possessions, sloppy boundaries, phases that run on over the next
(opposing) possession, partial coverage, and a team error rate swept from perfect to random.
The model output is the cached cross-validated ensemble, so no match is scored by a model that
saw it.
"""

import argparse

import numpy as np

from .decoder import transition_matrix
from .experiments import (CachedSample, SIM_TEST_SEED, _summary, decode_table, oof_logp, pm,
                          score)
from .match import A, H
from .opendata import LOADERS
from .predict import load_decoder
from .tagprior import GROUPS, self_calibrated

COVER = 0.6          # share of possessions that get a phase tag
MERGE = 0.3          # share of tags that run on over the following possession
EDGE_SD = 25         # frames of boundary jitter
DEFENCE_SHARE = 0.3  # share of tags that are defensive codes (their polarity is pre-applied)


def synthetic_tags(labels, error, seed):
    """[(start, stop, group, side)] for one open match, from its true possessions."""
    rng = np.random.default_rng(seed)
    spells = pm.possession_spells(labels, pm.valid_runs(labels))
    groups = list(GROUPS)
    out = []
    for i, (start, stop, code) in enumerate(spells):
        if rng.random() >= COVER:
            continue
        side = code if rng.random() >= error else (A if code == H else H)
        if rng.random() < MERGE and i + 1 < len(spells):
            stop = spells[i + 1][1]
        start = int(start + rng.normal(0, EDGE_SD))
        stop = int(stop + rng.normal(0, EDGE_SD))
        group = groups[1] if rng.random() < DEFENCE_SHARE else groups[0]
        if stop > start:
            out.append((start, stop, group, side))
    return out


def run(errors, costs=None):
    names = sorted(LOADERS)
    samples = {n: CachedSample(n, SIM_TEST_SEED) for n in names}
    logps = oof_logp(names, "ensemble")
    settings = load_decoder()
    transition = transition_matrix(*(costs or (settings["switch"], settings["alive_to_dead"],
                                               settings["dead_to_alive"])))
    base = {n: score(samples[n].labels, decode_table(samples[n], logps[n], transition))
            for n in names}
    print("no prior              all {:.4f}  dfl {:.4f}  metrica {:.4f}".format(
        *_summary(base, names)))
    for error in errors:
        rows, weights = {}, []
        for k, n in enumerate(names):
            tags = synthetic_tags(samples[n].labels, error, seed=1000 * k + int(error * 100))
            adjusted, report = self_calibrated(logps[n], tags, samples[n].period > 0)
            rows[n] = score(samples[n].labels, decode_table(samples[n], adjusted, transition))
            weights.append([report[g][0] for g in GROUPS])
            reliability = [report[g][1] for g in GROUPS]
        print("tag error {:.2f}        all {:.4f}  dfl {:.4f}  metrica {:.4f}   mean weights {}"
              "  (last match reliability {})".format(
                  error, *_summary(rows, names),
                  np.round(np.mean(weights, axis=0), 2).tolist(),
                  np.round(reliability, 2).tolist()), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--errors", type=float, nargs="+", default=(0.0, 0.1, 0.2, 0.35, 0.5))
    args = parser.parse_args(argv)
    run(args.errors)


if __name__ == "__main__":
    main()
