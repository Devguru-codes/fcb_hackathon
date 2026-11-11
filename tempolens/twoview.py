"""TTA fitted on both team views (plan of action row TTA-2V), one lane job per step.

    python -m tempolens.twoview study      # sim / shift / sc as ADseeds, rule (i) and (ii)
    python -m tempolens.twoview pff        # rule (iii), the pff-16 veto
    python -m tempolens.candidate tta2v --data-root <bundle>    # the candidate, if all pass

A step whose earlier step is missing or failed stops with exit 3 and computes nothing.
"""

import argparse
import json

import numpy as np

from . import adaptcv, adaptseeds, pffstudy, swaadopt
from .experiments import stop, write_atomic

MEMBERS = 5
STEPS = (10, 0)                                    # the tie order of the row
COSTS = ((4.0, 8.0, 8.0, 1.0), (2.0, 8.0, 8.0, 0.5))
SUBMITTED = (0.5, 10, False, (4.0, 8.0, 8.0, 1.0))
STUDY = adaptcv.OUT / "twoview" / "m5"
DIR = adaptcv.ROOT / "models" / "exp" / "twoview"
CONDITIONS = ("sim", "shift", "sc")


def finals():
    return adaptcv.ens5_paths()


def verdict(step):
    path = DIR / "verdict_{}.json".format(step)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def score_match(members, sample):
    return adaptcv.score_match(members, sample, per_member=True, costs=COSTS, alphas=(0.5,),
                               steps=(0,) + tuple(s for s in STEPS if s), gates=(False,),
                               both_views=True)


def mean3(cond):
    return float(np.mean([cond[c] for c in CONDITIONS]))


def decide(two, ref, ref_members):
    """Rules (i) and (ii) on adaptseeds tables: (best setting, its mean, member means, passed)."""
    order = [(0.5, s, False, c) for s in STEPS for c in COSTS]
    best = max(order, key=lambda k: (mean3(two[(adaptseeds.ENSEMBLE,) + k]), -order.index(k)))
    mean = mean3(two[(adaptseeds.ENSEMBLE,) + best])
    members = [mean3(two[(k,) + best]) for k in range(MEMBERS)]
    ok = mean > ref and all(m >= r for m, r in zip(members, ref_members))
    return best, mean, members, ok


def study():
    results = adaptcv.collect(MEMBERS, STUDY, score_match, final_paths=finals())
    two = adaptseeds.table(results)
    adseeds = adaptseeds.table(swaadopt.cached_results(swaadopt.ADSEEDS))
    ref = mean3(swaadopt.reference())
    ref_members = [mean3(adseeds[(k,) + SUBMITTED]) for k in range(MEMBERS)]
    for s in STEPS:
        for c in COSTS:
            k = (0.5, s, False, c)
            print("two-view TTA 0.5 eata {:<2} {:<22} ensemble {:.4f}  members {}".format(
                s, str(c), mean3(two[(adaptseeds.ENSEMBLE,) + k]),
                " ".join("{:.4f}".format(mean3(two[(m,) + k])) for m in range(MEMBERS))),
                flush=True)
    best, mean, members, ok = decide(two, ref, ref_members)
    print("best: eata {} costs {}: {:.4f} vs submission {:.4f} ({:+.4f}); members {} vs {} -> {}"
          .format(best[1], best[3], mean, ref, mean - ref,
                  " ".join("{:.4f}".format(m) for m in members),
                  " ".join("{:.4f}".format(m) for m in ref_members),
                  "PASS (i) and (ii)" if ok else "FAIL"), flush=True)
    DIR.mkdir(parents=True, exist_ok=True)
    write_atomic(DIR / "verdict_study.json", json.dumps(
        {"steps": best[1], "costs": list(best[3]), "mean": mean, "reference": ref,
         "members": members, "reference_members": ref_members, "pass": ok}, indent=1))


def pff():
    v = verdict("study")
    if v is None:
        stop("step 'study' has no result")
    if not v["pass"]:
        stop("rules (i) / (ii) failed in the study: no pff-16 run")
    steps, costs = v["steps"], tuple(v["costs"])
    ids = pffstudy.test_ids(pffstudy.pff_ids())
    assert len(ids) == 16 and ids[0] == "3812", "expected pff-16"
    ref = [r[swaadopt.SUBMITTED_PFF]["10"] for r in pffstudy.submitted_scores(ids)]
    got = pffstudy.candidate_scores(ids, pffstudy._cache() / "twoview", finals(), 0.5, steps,
                                    costs, "_s{}".format(steps), both_views=True)[10]
    mean, base = float(np.mean(got)), float(np.mean(ref))
    ok = swaadopt.veto_passes(mean, base)
    print("(iii) pff-16 two-view: 10 frames {:.4f} vs submission {:.4f} ({:+.4f}) -> {}".format(
        mean, base, mean - base, "PASS" if ok else "FAIL"), flush=True)
    write_atomic(DIR / "verdict_pff.json", json.dumps({"mean": mean, "submission": base,
                                                       "pass": ok}, indent=1))
    if not ok:
        stop("(iii) the pff-16 veto failed")


def settings():
    """The candidate's decoder overrides if every step passed, else None."""
    s, p = verdict("study"), verdict("pff")
    if not (s and s["pass"] and p and p["pass"]):
        return None
    c = s["costs"]
    return {"tta_both_views": True, "tta_alpha": 0.5, "eata_steps": s["steps"], "switch": c[0],
            "alive_to_dead": c[1], "dead_to_alive": c[2], "change_weight": c[3]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("study", "pff"))
    args = parser.parse_args(argv)
    return study() if args.step == "study" else pff()


if __name__ == "__main__":
    main()
