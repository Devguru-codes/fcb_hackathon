"""Compare decoders on the cross-validated ensemble output: fixed Viterbi, minimum-duration
Viterbi, and minimum-Bayes-risk selection. Scored by the official spellF1 against open labels.

    python -m tempolens.studies.decodestudy
"""

import argparse
import itertools

import numpy as np

from ..decoder import transition_matrix
from ..experiments import CachedSample, SIM_TEST_SEED, _summary, decode_table, oof_logp, score
from ..mbr import decode_min_duration, mbr_decode
from ..opendata import LOADERS
from ..predict import load_decoder

_SETTINGS = load_decoder()   # the submitted costs, without the change bonus (the alternatives have none)
BASE = tuple(_SETTINGS[k] for k in ("switch", "alive_to_dead", "dead_to_alive"))


def evaluate(names, samples, decode):
    rows = {n: score(samples[n].labels, decode(n)) for n in names}
    return _summary(rows, names), rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--samples", type=int, default=24)
    args = parser.parse_args(argv)
    names = sorted(LOADERS)
    samples = {n: CachedSample(n, SIM_TEST_SEED) for n in names}
    logps = oof_logp(names, "ensemble")

    def show(label, summary):
        print("{:<46} all {:.4f}  dfl {:.4f}  metrica {:.4f}".format(label, *summary), flush=True)

    base, _ = evaluate(names, samples, lambda n: decode_table(
        samples[n], logps[n], transition_matrix(*BASE)))
    show("viterbi {} (3-member CV ensemble, submitted costs, no bonus)".format(BASE), base)

    # Minimum durations, with the switch cost re-tuned (a floor on run length replaces part of
    # what the switch cost was doing).
    best = None
    for (m, m_dead, switch) in itertools.product((3, 6, 10, 15), (1, 12), (1.0, 2.0, 3.0)):
        transition = transition_matrix(switch, *BASE[1:])
        summary, _ = evaluate(names, samples, lambda n: decode_min_duration(
            samples[n], logps[n], transition, (m, m, m_dead)))
        if best is None or summary[0] > best[0][0]:
            best = (summary, (m, m_dead, switch))
    show("min-duration best {}".format(best[1]), best[0])

    # MBR: per match, choose among decodings by expected spellF1 under the posterior.
    switches = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    for temperature in (0.5, 1.0):
        picks = []

        def decode(n):
            candidates = [decode_table(samples[n], logps[n], transition_matrix(s, *BASE[1:]))
                          for s in switches]
            chosen, index, _ = mbr_decode(samples[n], logps[n], candidates,
                                          transition_matrix(*BASE), args.samples, temperature,
                                          seed=7)
            picks.append(switches[index])
            return chosen
        summary, _ = evaluate(names, samples, decode)
        show("MBR over switch costs, temperature {}".format(temperature), summary)
        print("    switch chosen per match: {}".format(picks), flush=True)

    # Oracle: the best switch per match in hindsight -- the ceiling MBR's selection could reach.
    oracle_rows = {}
    for n in names:
        oracle_rows[n] = max((score(samples[n].labels, decode_table(
            samples[n], logps[n], transition_matrix(s, *BASE[1:]))) for s in switches),
            key=lambda r: r["spellF1"])
    show("oracle per-match switch (ceiling, not usable)", _summary(oracle_rows, names))


if __name__ == "__main__":
    main()
