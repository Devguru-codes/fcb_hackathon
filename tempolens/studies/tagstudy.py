"""Does the self-calibrated Smart Tagging prior help when tags are good, and stay harmless when
they are bad? Tested on open matches with synthetic tags, since real tags exist only for the
competition match.

    python -m tempolens.studies.tagstudy

Synthetic tags imitate Smart Tagging's shape rather than its exact error profile, which is
unknown: phases attached to possessions, sloppy boundaries, phases that run on over the next
(opposing) possession, partial coverage, and a team error rate swept from perfect to random.
The model output is the cached cross-validated ensemble, so no match is scored by a model that
saw it.
"""

import argparse

import numpy as np

from ..decoder import transition_matrix
from ..experiments import (CachedSample, ENSEMBLE_MEMBERS, SIM_TEST_SEED, _summary,
                          decode_table, oof_logp, pm, score)
from ..match import A, H
from ..opendata import LOADERS
from ..predict import load_decoder
from ..tagprior import FAMILIES, GROUPS, SHRINK, group_of, self_calibrated

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


def synthetic_coded_tags(labels, errors, seed):
    """Like synthetic_tags, but each tag is a code family (tagprior.FAMILIES) with its own team
    error rate (`errors`: {family: rate}): [(start, stop, family, side)]."""
    rng = np.random.default_rng(seed)
    spells = pm.possession_spells(labels, pm.valid_runs(labels))
    defence = [f for f in FAMILIES if group_of(FAMILIES[f][0]) == "defence"]
    attack = [f for f in FAMILIES if f not in defence]
    out = []
    for i, (start, stop, code) in enumerate(spells):
        if rng.random() >= COVER:
            continue
        family = (defence[rng.integers(len(defence))] if rng.random() < DEFENCE_SHARE
                  else attack[rng.integers(len(attack))])
        side = code if rng.random() >= errors[family] else (A if code == H else H)
        if rng.random() < MERGE and i + 1 < len(spells):
            stop = spells[i + 1][1]
        start = int(start + rng.normal(0, EDGE_SD))
        stop = int(stop + rng.normal(0, EDGE_SD))
        if stop > start:
            out.append((start, stop, family, side))
    return out


def error_profiles():
    """E7's tag error profiles: {name: {family: rate}}; "20% mixed" alternates 5% and 35% over
    the families (mean 20%), so reliability differs by code."""
    fams = list(FAMILIES)
    return {"perfect": dict.fromkeys(fams, 0.0),
            "20% mixed": {f: (0.05, 0.35)[i % 2] for i, f in enumerate(fams)},
            "20% uniform": dict.fromkeys(fams, 0.2),
            "random": dict.fromkeys(fams, 0.5)}


def run_per_code(draws=4, seeds=3, members=5):
    """E7: the two-group prior against one weight per code family, on the same synthetic tags,
    at the decoder's costs. Prints mean spellF1 per error profile and the rule's verdict."""
    names = sorted(LOADERS)
    samples = {n: CachedSample(n, SIM_TEST_SEED) for n in names}
    logps = oof_logp(names, "ensemble", draws=draws, members=members)
    settings = load_decoder()
    transition = transition_matrix(settings["switch"], settings["alive_to_dead"],
                                   settings["dead_to_alive"])
    change = settings.get("change_weight", 0.0)

    def spell(n, lp):
        return score(samples[n].labels, decode_table(samples[n], lp, transition, change))["spellF1"]

    none = np.mean([spell(n, logps[n]) for n in names])
    print("no prior {:.4f}  (costs {}, {}, {}, {})".format(
        none, settings["switch"], settings["alive_to_dead"], settings["dead_to_alive"], change))
    gains = {}
    for p, (name, errors) in enumerate(error_profiles().items()):
        two, per = [], []
        for k, n in enumerate(names):
            inside = samples[n].period > 0
            for s in range(seeds):
                tags = synthetic_coded_tags(samples[n].labels, errors,
                                            7000 + 1000 * k + 100 * p + s)
                grouped = [(a, b, group_of(FAMILIES[f][0]), side) for a, b, f, side in tags]
                two.append(spell(n, self_calibrated(logps[n], grouped, inside)[0]))
                per.append(spell(n, self_calibrated(logps[n], tags, inside, FAMILIES)[0]))
        gains[name] = np.mean(per) - np.mean(two)
        print("{:<12} two groups {:.4f}  per family {:.4f}  difference {:+.4f}".format(
            name, np.mean(two), np.mean(per), gains[name]), flush=True)
    ok = gains["perfect"] >= 0.003 and gains["20% mixed"] >= 0.003 and gains["random"] >= -0.003
    print("E7: {}".format("passes" if ok else "fails"))


def run(errors, costs=None, draws=0, seeds=1, members=ENSEMBLE_MEMBERS, shrink=SHRINK):
    """The decoder's costs and change bonus unless `costs` (switch, alive>dead, dead>alive,
    change) is given; `draws`: the ensemble's training recipe (experiments --draws); `seeds`:
    tag draws per match, averaged; `members`: ensemble size; `shrink`: the prior's. Returns
    {error: mean spellF1 over the matches}, with None for no prior."""
    names = sorted(LOADERS)
    samples = {n: CachedSample(n, SIM_TEST_SEED) for n in names}
    logps = oof_logp(names, "ensemble", draws=draws, members=members)
    settings = load_decoder()
    costs = costs or (settings["switch"], settings["alive_to_dead"], settings["dead_to_alive"],
                      settings.get("change_weight", 0.0))
    transition, change = transition_matrix(*costs[:3]), costs[3]
    base = {n: score(samples[n].labels, decode_table(samples[n], logps[n], transition, change))
            for n in names}
    print("no prior              all {:.4f}  dfl {:.4f}  metrica {:.4f}".format(
        *_summary(base, names)))
    out = {None: _summary(base, names)[0]}
    for error in errors:
        rows, weights = {}, []
        for k, n in enumerate(names):
            runs = []
            for s in range(seeds):      # seed 0 is the original study's tag draw
                tags = synthetic_tags(samples[n].labels, error,
                                      seed=1000 * k + int(error * 100) + 100000 * s)
                adjusted, report = self_calibrated(logps[n], tags, samples[n].period > 0,
                                                   shrink=shrink)
                runs.append(score(samples[n].labels,
                                  decode_table(samples[n], adjusted, transition, change)))
                weights.append([report[g][0] for g in GROUPS])
                reliability = [report[g][1] for g in GROUPS]
            rows[n] = {"spellF1": np.mean([r["spellF1"] for r in runs])}
        print("tag error {:.2f}        all {:.4f}  dfl {:.4f}  metrica {:.4f}   mean weights {}"
              "  (last match reliability {})".format(
                  error, *_summary(rows, names),
                  np.round(np.mean(weights, axis=0), 2).tolist(),
                  np.round(reliability, 2).tolist()), flush=True)
        out[error] = _summary(rows, names)[0]
    return out


GAIN, LOSS = 0.003, 0.003     # plan row TS


def shrink_rule(scores, base=SHRINK):
    """Plan row TS on {shrink: {error: mean spellF1}}: s passes if it beats `base` by >= GAIN at
    10% and 20% team error and loses <= LOSS at 35% and 50%; of those, the larger gain at 20%.
    Returns (chosen shrink or None, {shrink: passed})."""
    ref, passed = scores[base], {}
    for s, row in scores.items():
        if s == base:
            continue
        passed[s] = (all(row[e] - ref[e] >= GAIN for e in (0.1, 0.2))
                     and all(ref[e] - row[e] <= LOSS for e in (0.35, 0.5)))
    winners = [s for s, ok in passed.items() if ok]
    return (max(winners, key=lambda s: scores[s][0.2] - ref[0.2]) if winners else None), passed


def shrink_study(shrinks, errors, draws, seeds, members):
    """Each shrink on the same tag draws, the rule, and the verdict file for the candidate."""
    import json
    from ..experiments import write_atomic
    from ..adaptcv import ROOT
    assert SHRINK in shrinks and all(e in errors for e in (0.1, 0.2, 0.35, 0.5))
    scores = {}
    for s in shrinks:
        print("\nshrink {}".format(s), flush=True)
        scores[s] = run(errors, draws=draws, seeds=seeds, members=members, shrink=s)
    chosen, passed = shrink_rule(scores)
    print("\nshrink  " + "  ".join("{:>6.0%}".format(e) for e in errors) + "   (minus shrink 0.5)")
    for s in shrinks:
        print("{:<6}  ".format(s) + "  ".join("{:+.4f}".format(scores[s][e] - scores[SHRINK][e])
                                            for e in errors)
              + ("" if s == SHRINK else "   -> {}".format("passes" if passed[s] else "fails")))
    print("TS: {}".format("adopt shrink {}".format(chosen) if chosen is not None
                          else "none passes: shrink 0.5 stays"), flush=True)
    out = ROOT / "models" / "exp" / "ts"
    out.mkdir(parents=True, exist_ok=True)
    write_atomic(out / "verdict.json", json.dumps({
        "chosen": chosen, "passed": {str(k): v for k, v in passed.items()},
        "scores": {str(s): {str(e): v for e, v in row.items()} for s, row in scores.items()}},
        indent=1))
    return chosen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--errors", type=float, nargs="+", default=(0.0, 0.1, 0.2, 0.35, 0.5))
    parser.add_argument("--draws", type=int, default=0, help="ensemble recipe (experiments)")
    parser.add_argument("--seeds", type=int, default=1, help="synthetic tag draws per match")
    parser.add_argument("--members", type=int, default=ENSEMBLE_MEMBERS, help="ensemble size")
    parser.add_argument("--shrink", type=float, default=SHRINK, help="the prior's shrink")
    parser.add_argument("--shrinks", type=float, nargs="+",
                        help="plan row TS: these shrinks on the same draws, with the rule")
    parser.add_argument("--per-code", action="store_true",
                        help="E7: one weight per code family against the two groups")
    args = parser.parse_args(argv)
    if args.per_code:
        return run_per_code(args.draws or 4, args.seeds, args.members)
    if args.shrinks:
        return shrink_study(args.shrinks, args.errors, args.draws, args.seeds, args.members)
    run(args.errors, draws=args.draws, seeds=args.seeds, members=args.members, shrink=args.shrink)


if __name__ == "__main__":
    main()
