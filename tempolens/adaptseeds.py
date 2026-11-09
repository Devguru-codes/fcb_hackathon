"""Test-time adaptation per seed, on external data only (plan of action row ADseeds, 2026-11-07).

    python -m tempolens.adaptseeds --members 5

The same grid, conditions and fold models as `adaptcv` (sim / shift / sc), scored for the ensemble
(seed -1) and for each member alone (seed k: fold member k on sim and shift, final member k on sc).

Rule, fixed before the run: score = mean spellF1 over the three conditions. A setting is accepted
if, at the same costs, it beats no adaptation for the ensemble and for every seed; among accepted
settings the highest ensemble score wins; none accepted -> no adaptation. Each seed's sc-only gain
and the per-condition gains are reported but do not decide. Results are cached per match in
models/exp/adaptcv/seeds/m<members>/; the decision goes to decision.json there.
"""

import argparse
import json

import numpy as np

from . import adaptcv

COSTS = ((2.0, 8.0, 8.0, 0.5), (4.0, 8.0, 8.0, 1.0))
CONDITIONS = ("sim", "shift", "sc")
ENSEMBLE = -1


def score_match(members, sample):
    return adaptcv.score_match(members, sample, per_member=True, costs=COSTS)


def table(results):
    """{(seed, alpha, steps, gate, costs): {condition: mean spellF1 over matches}}."""
    acc, seen = {}, {}
    for condition, matches in results.items():
        for name, rows in matches.items():
            for r in rows:
                key = (r["seed"], r["alpha"], r["steps"], r["gate"], tuple(r["costs"]))
                acc.setdefault(key, {}).setdefault(condition, []).append(r["spellF1"])
                seen.setdefault(key, {}).setdefault(condition, set()).add(name)
    adaptcv.require_complete(seen, results)
    return {k: {c: float(np.mean(v)) for c, v in conds.items()} for k, conds in acc.items()}


def decide(summary):
    """The rule in the module docstring. Returns (rows sorted by ensemble gain, chosen row or None);
    each row: setting, ensemble mean, gain per seed, accepted, per-condition and sc gains per seed."""
    overall = {k: float(np.mean([v[c] for c in CONDITIONS])) for k, v in summary.items()}
    seeds = sorted({k[0] for k in summary})
    assert ENSEMBLE in seeds and len(seeds) > 1, "need the ensemble and at least one seed"
    rows = []
    for setting in sorted({k[1:] for k in summary if k[1:4] != (0.0, 0, False)}):
        none = (0.0, 0, False, setting[3])
        gain = {s: overall[(s,) + setting] - overall[(s,) + none] for s in seeds}
        conds = {s: {c: summary[(s,) + setting][c] - summary[(s,) + none][c] for c in CONDITIONS}
                 for s in seeds}
        rows.append({"alpha": setting[0], "steps": setting[1], "gate": setting[2],
                     "costs": list(setting[3]), "ensemble": overall[(ENSEMBLE,) + setting],
                     "none": overall[(ENSEMBLE,) + none],
                     "gain": {str(s): g for s, g in gain.items()},
                     "accepted": all(g > 0 for g in gain.values()),
                     "conditions": {str(s): c for s, c in conds.items()}})
    rows.sort(key=lambda r: -r["gain"][str(ENSEMBLE)])
    accepted = [r for r in rows if r["accepted"]]
    return rows, max(accepted, key=lambda r: r["ensemble"]) if accepted else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--members", type=int, default=5, help="ensemble size")
    args = parser.parse_args(argv)
    folder = adaptcv.OUT / "seeds" / "m{}".format(args.members)
    rows, chosen = decide(table(adaptcv.collect(args.members, folder, score_match)))
    seeds = [s for s in rows[0]["gain"] if s != str(ENSEMBLE)]
    print("\nalpha steps gate costs                  ens     gain    " +
          " ".join("s{:<6}".format(s) for s in seeds) + " ok   ens sim/shift/sc gains   sc gain per seed")
    for r in rows:
        print("{:<5} {:<5} {:<4} {:<22} {:.4f} {:+.4f} {} {:<4} {} {}".format(
            r["alpha"], r["steps"], "on" if r["gate"] else "off", str(tuple(r["costs"])),
            r["ensemble"], r["gain"][str(ENSEMBLE)],
            " ".join("{:+.4f}".format(r["gain"][s]) for s in seeds),
            "yes" if r["accepted"] else "no",
            "/".join("{:+.4f}".format(r["conditions"][str(ENSEMBLE)][c]) for c in CONDITIONS),
            " ".join("{:+.4f}".format(r["conditions"][s]["sc"]) for s in seeds)))
    if chosen:
        print("\nchosen: tta_alpha {} eata_steps {} gate {} costs {} -- ensemble {:.4f}, {:+.4f} "
              "over none at the same costs, every seed gains".format(
                  chosen["alpha"], chosen["steps"], "on" if chosen["gate"] else "off",
                  tuple(chosen["costs"]), chosen["ensemble"], chosen["gain"][str(ENSEMBLE)]))
    else:
        print("\nno setting beats no adaptation for the ensemble and every seed: adaptation stays off")
    (folder / "decision.json").write_text(json.dumps({"chosen": chosen, "rows": rows}, indent=1),
                                          encoding="utf-8")


if __name__ == "__main__":
    main()
