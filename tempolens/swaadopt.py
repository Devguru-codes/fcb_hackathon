"""B3c adoption pipeline (plan of action row B3c-adopt, pre-registered 2026-11-07 in c9110af).

    python -m tempolens.swaadopt final K   # (a) SWA final member K -> models/exp/swa4/
    python -m tempolens.swaadopt study     # (b) SWA fold models + restricted adaptation study
    python -m tempolens.swaadopt pff       # (c) pff-16 veto with the SWA finals
    python -m tempolens.swaadopt build --data-root <bundle>   # (d) check, then (e) candidate

Each step needs the previous one's result and stops (exit 3) when it is missing or failed, so
a failure stops the chain there. Never touches models/decoder.json or tempolens_rnd4_m*.pt.
Aggregates only.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

from . import adaptcv, adaptseeds, pffstudy, temporal
from .decoder import transition_matrix
from .experiments import CachedSample, decode_table, savez_atomic, score, write_atomic
from .predict import DEFAULT_DECODER

SWA, MEMBERS = 4, 5
DIR = adaptcv.ROOT / "models" / "exp" / "swa4"
FINALS = [DIR / "tempolens_swa4_m{}.pt".format(k) for k in range(MEMBERS)]
STUDY = adaptcv.OUT / "seeds_swa4" / "m5"
ADSEEDS = adaptcv.OUT / "seeds" / "m5"
ALPHAS, STEPS, COSTS = (0.0, 0.5), (0, 10), adaptseeds.COSTS
SUBMITTED = (0.5, 10, False, (4.0, 8.0, 8.0, 1.0))
SUBMITTED_PFF = "submitted: TTA 0.5 + EATA 10"
TIE = 0.0001
VETO = pffstudy.NOISE            # 0.006
CONDITIONS = adaptseeds.CONDITIONS
ENS = str(adaptseeds.ENSEMBLE)


def stop(message):
    print("STOPPED: " + message, flush=True)
    sys.exit(3)


def verdict(step):
    path = DIR / "verdict_{}.json".format(step)
    if not path.exists():
        stop("step '{}' has no result".format(step))
    return json.loads(path.read_text(encoding="utf-8"))


def save(step, result):
    DIR.mkdir(parents=True, exist_ok=True)
    write_atomic(DIR / "verdict_{}.json".format(step), json.dumps(result, indent=1))


def final(k):
    missing = [p.name for p in FINALS[:k] if not p.exists()]
    if missing:
        stop("earlier SWA finals missing: {}".format(", ".join(missing)))
    if FINALS[k].exists():
        print("exists: {}".format(FINALS[k].name))
        return
    subprocess.run([sys.executable, "-m", "tempolens.train", "--member", str(k), "--draws", "4",
                    "--swa", str(SWA), "--out", str(FINALS[k])], check=True)


def score_match(members, sample):
    return adaptcv.score_match(members, sample, per_member=True, costs=COSTS, alphas=ALPHAS,
                               steps=STEPS, gates=(False,))


def pick(rows):
    """ADseeds' rule plus the tie-break written before the run: the highest ensemble score among
    accepted settings; within TIE of the next, the larger smallest per-member gain, then the larger
    ensemble gain on sc. None accepted -> None."""
    accepted = sorted((r for r in rows if r["accepted"]), key=lambda r: -r["ensemble"])
    if not accepted:
        return None
    if len(accepted) > 1 and accepted[0]["ensemble"] - accepted[1]["ensemble"] <= TIE:
        return max(accepted[:2], key=lambda r: (
            min(g for s, g in r["gain"].items() if s != ENS), r["conditions"][ENS]["sc"]))
    return accepted[0]


def cached_results(folder):
    """{condition: {match: rows}} from a study folder's per-match caches."""
    out = {c: {} for c in CONDITIONS}
    for c in CONDITIONS:
        for path in sorted(folder.glob("{}_*.json".format(c))):
            out[c][path.stem[len(c) + 1:]] = json.loads(path.read_text(encoding="utf-8"))
    return out


def reference():
    """The current submission's per-condition means from ADseeds' cache (step d)."""
    results = cached_results(ADSEEDS)
    counts = {c: len(m) for c, m in results.items()}
    assert counts == {"sim": 10, "shift": 10, "sc": 20}, counts
    return adaptseeds.table(results)[(adaptseeds.ENSEMBLE,) + SUBMITTED]


def study():
    missing = [p.name for p in FINALS if not p.exists()]
    if missing:
        stop("step (a) incomplete: {}".format(", ".join(missing)))
    summary = adaptseeds.table(adaptcv.collect(MEMBERS, STUDY, score_match, final_paths=FINALS,
                                               swa=SWA))
    rows, _ = adaptseeds.decide(summary)
    print("\nalpha steps costs                  ens     gain    per-member gains              ok")
    for r in rows:
        print("{:<5} {:<5} {:<22} {:.4f} {:+.4f} {} {}".format(
            r["alpha"], r["steps"], str(tuple(r["costs"])), r["ensemble"], r["gain"][ENS],
            " ".join("{:+.4f}".format(g) for s, g in r["gain"].items() if s != ENS),
            "yes" if r["accepted"] else "no"), flush=True)
    chosen = pick(rows)
    if chosen:
        setting = (chosen["alpha"], chosen["steps"], False, tuple(chosen["costs"]))
        why = "accepted, highest ensemble score (tie-break applied if within {})".format(TIE)
    else:
        plain = {c: np.mean([summary[(adaptseeds.ENSEMBLE, 0.0, 0, False, c)][k]
                             for k in CONDITIONS]) for c in COSTS}
        setting = (0.0, 0, False, max(plain, key=plain.get))
        why = "no setting accepted: no adaptation at the better cost row"
    cand = summary[(adaptseeds.ENSEMBLE,) + setting]
    ref = reference()
    c_mean, r_mean = (float(np.mean([v[k] for k in CONDITIONS])) for v in (cand, ref))
    print("\n(b) chosen: tta_alpha {} eata_steps {} costs {} -- {}".format(
        setting[0], setting[1], setting[3], why))
    print("(d) numbers (checked by 'build'): candidate {:.4f} vs submission {:.4f} ({:+.4f}); "
          "sim / shift / sc {}".format(c_mean, r_mean, c_mean - r_mean, " / ".join(
              "{:+.4f}".format(cand[k] - ref[k]) for k in CONDITIONS)), flush=True)
    save("study", {"alpha": setting[0], "steps": setting[1], "costs": list(setting[3]),
                   "accepted": chosen is not None, "candidate": dict(cand, mean=c_mean),
                   "reference": dict(ref, mean=r_mean), "beats": c_mean > r_mean})


def pff():
    v = verdict("study")
    alpha, steps, costs = v["alpha"], v["steps"], tuple(v["costs"])
    ids = pffstudy.test_ids(pffstudy.pff_ids())
    assert len(ids) == 16 and ids[0] == "3812", "expected pff-16"
    ref = {c: [] for c in pffstudy.CONVENTIONS}
    for game_id in ids:     # the submission's cached pff-16 scores (row PFF)
        r = json.loads((pffstudy._cache() / "pff_{}.json".format(game_id)).read_text("utf-8"))
        for c in ref:
            ref[c].append(r[SUBMITTED_PFF][str(c)])
    members = [temporal.load(p) for p in FINALS]
    folder = pffstudy._cache() / "swa4"
    folder.mkdir(parents=True, exist_ok=True)
    key = "a{}_s{}".format(alpha, steps)
    got = {c: [] for c in pffstudy.CONVENTIONS}
    for game_id in ids:
        sample = CachedSample("pff_{}".format(game_id))
        store = folder / "pff_{}_{}.npz".format(game_id, key)
        if not store.exists():
            grid = adaptcv.member_grid(members, sample, alphas=(alpha,),
                                       steps=tuple(sorted({0, steps})))
            savez_atomic(store, out=np.stack(grid[(alpha, steps)]).astype(np.float32))
        logp = temporal.combine(list(np.load(store)["out"].astype(np.float64)))
        pred = decode_table(sample, logp, transition_matrix(*costs[:3]), costs[3])
        for c, y in pffstudy.labels_by_convention(sample).items():
            got[c].append(score(y, pred)["spellF1"])
        print("scored pff_{}".format(game_id), flush=True)
    mean = {c: float(np.mean(got[c])) for c in got}
    base = {c: float(np.mean(ref[c])) for c in ref}
    ok = mean[10] >= base[10] - VETO
    print("\n(c) pff-16, SWA finals at tta_alpha {} eata_steps {} costs {}: 10 frames {:.4f} "
          "(submission {:.4f}, {:+.4f}), 50 frames {:.4f} (submission {:.4f}, {:+.4f}) -> {}".format(
              alpha, steps, costs, mean[10], base[10], mean[10] - base[10], mean[50], base[50],
              mean[50] - base[50], "PASS" if ok else "FAIL (veto)"), flush=True)
    save("pff", {"mean": mean, "submission": base, "pass": ok})
    if not ok:
        stop("(c) pff-16 veto failed")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(data_root):
    s, c = verdict("study"), verdict("pff")
    if not c["pass"]:
        stop("(c) pff-16 veto failed")
    print("(d) candidate {:.4f} vs submission {:.4f} on the mean of sim / shift / sc -> {}".format(
        s["candidate"]["mean"], s["reference"]["mean"], "PASS" if s["beats"] else "FAIL"),
        flush=True)
    if not s["beats"]:
        save("build", {"d": False})
        stop("(d) the candidate does not beat the current submission")
    config = json.loads(DEFAULT_DECODER.read_text(encoding="utf-8"))
    config.update({
        "_provenance": "TEMPORARY candidate of plan row B3c-adopt, not the submission: the 5 SWA "
                       "finals with the setting from swaadopt study. models/decoder.json is "
                       "unchanged.",
        "members": [p.relative_to(adaptcv.ROOT / "models").as_posix() for p in FINALS],
        "tta_alpha": s["alpha"], "eata_steps": s["steps"],
        "switch": s["costs"][0], "alive_to_dead": s["costs"][1], "dead_to_alive": s["costs"][2],
        "change_weight": s["costs"][3]})
    out = adaptcv.ROOT / "derived" / "swa4"          # git-ignored
    out.mkdir(parents=True, exist_ok=True)
    decoder = out / "decoder_swa4.json"
    write_atomic(decoder, json.dumps(config, indent=2) + "\n")
    files = []
    for i in (1, 2):
        csv = out / "candidate_{}.csv".format(i)
        with open(out / "predict_{}.log".format(i), "w", encoding="utf-8") as log:
            subprocess.run([sys.executable, "-m", "tempolens.predict", "--data-root", data_root,
                            "--decoder", str(decoder), "--output", str(csv)],
                           check=True, stdout=log, stderr=subprocess.STDOUT)
        files.append(csv)
    valid = subprocess.run([sys.executable, str(adaptcv.ROOT / "evaluation" /
                                                "validate_submission.py"), str(files[0])],
                           capture_output=True, text=True).returncode == 0
    same = sha(files[0]) == sha(files[1])
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=adaptcv.ROOT,
                           capture_output=True, text=True, env=dict(os.environ, MTAH_E2E="1"))
    summary = (tests.stdout.strip().splitlines() or ["(no output)"])[-1]
    print("(e) candidate {}: valid {}, byte-identical on a second run {}; tests: {} (exit {})"
          .format(sha(files[0])[:16], valid, same, summary, tests.returncode), flush=True)
    print("submission unchanged: models/decoder.json sha {}".format(sha(DEFAULT_DECODER)[:16]))
    save("build", {"d": True, "sha": sha(files[0]), "valid": valid, "identical": same,
                   "tests": summary, "tests_exit": tests.returncode})
    if not (valid and same and tests.returncode == 0):
        stop("(e) the candidate build did not pass its checks")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("final", "study", "pff", "build"))
    parser.add_argument("member", nargs="?", type=int)
    parser.add_argument("--data-root")
    args = parser.parse_args(argv)
    if args.step == "final":
        return final(args.member)
    if args.step == "build":
        if not args.data_root:
            parser.error("build needs --data-root")
        return build(args.data_root)
    return study() if args.step == "study" else pff()


if __name__ == "__main__":
    main()
