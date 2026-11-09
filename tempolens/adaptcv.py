"""Choose the test-time adaptation on external data only (organisers' rules, 2026-11-03).

    python -m tempolens.adaptcv            # train/reuse fold models, score the grid, print the choice
    python -m tempolens.adaptcv --members 5   # the same with a 5-member ensemble (ENS5)

The competition match differs from the training data (simulated noise on clean tracking). The
adaptation settings -- `tta_alpha` (temporal.adapt_normaliser), `eata_steps` (temporal.adapt_head),
the team-evidence gate (predict.gate_team_evidence) and the decoder costs -- are scored here on
open matches under a domain shift, never on the provided match:

- sim:   each open match held out (5 folds, v7 recipe: 3 or --members members, 4 randomised draws), with the
         fixed calibrated test noise -- the standard cross-validation;
- shift: the same fold models on a different corruption: one randomised noise draw per match
         plus tracker faults (identity swaps, duplicate and phantom tracks), never trained on;
- sc:    the submitted members (trained on all 10 open matches) on the 20 SkillCorner matches as
         they are: real broadcast tracking, never seen in training.

Decision rule, fixed before the run: score = mean over the three conditions of the mean
spellF1. Adaptation (alpha, steps or the gate) is adopted only if its best setting beats no
adaptation at the same costs by at least NOISE and loses no condition by more than NOISE;
otherwise the costs alone are chosen. Results are cached per match in models/exp/adaptcv/
(m<members>/ for other ensemble sizes); fold models are shared.

EATA adapts the head on temporal.eata_inputs, the same fit set predict.py uses (both team views,
every 5th in-half frame); the study runs in float32, predict.py in float64.
"""

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from . import temporal
from .decoder import transition_matrix
from .experiments import SIM_TEST_SEED, CachedSample, check_members, decode_table, folds, score, \
    write_atomic
from .match import A, D, H
from .opendata import EXTRA_LOADERS, LOADERS
from .predict import DEFAULT_DECODER, gate_team_evidence, load_decoder

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models" / "exp" / "adaptcv"
ALPHAS = (0.0, 0.5, 1.0)
STEPS = (0, 10, 30, 60)
GATES = (False, True)
COSTS = ((2.0, 2.0, 2.0, 0.5), (2.0, 8.0, 8.0, 0.5), (4.0, 8.0, 8.0, 1.0))
SHIFT_SEED = 7000
MEMBERS, DRAWS = 3, 4
NOISE = 0.005


def fold_members(held, rest, count=MEMBERS, swa=0):
    """The v7-recipe members trained without `held` (with weight averaging over the last `swa`
    epochs if set), cached on disk."""
    members = []
    for k in range(count):
        folder = OUT / ("models_swa{}".format(swa) if swa else "models")
        path = folder / "{}_m{}.pt".format("-".join(held), k)
        if not path.exists():      # its own process: training arrays never pile up here
            path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([sys.executable, "-m", "tempolens.train", "--member", str(k),
                            "--draws", str(DRAWS), "--out", str(path), "--matches", *rest]
                           + (["--swa", str(swa)] if swa else []),
                           check=True)
        members.append(temporal.load(path))
    return members


def member_output(head, h0, h1, inside):
    """temporal.predict's output from head inputs of both views."""
    torch = temporal._torch()
    result = np.zeros((len(h0), 4))
    result[:, :3] = np.log(1.0 / 3)
    with torch.no_grad():
        for h, swap in ((h0, False), (h1, True)):
            raw = head(torch.from_numpy(h[inside]).to(head.weight.dtype)).numpy()[:, :4]
            p = np.exp(raw[:, :3] - raw[:, :3].max(axis=1, keepdims=True))
            p /= p.sum(axis=1, keepdims=True)
            if swap:
                p = p[:, [A, H, D]]
            if not swap:
                result[inside, :3], result[inside, 3] = 0.5 * p, 0.5 * raw[:, 3]
            else:
                result[inside, :3] += 0.5 * p
                result[inside, 3] += 0.5 * raw[:, 3]
    result[inside, :3] = np.log(np.clip(result[inside, :3], 1e-6, 1.0))
    return np.round(result, 4)


def member_grid(members, sample, double=False, alphas=ALPHAS, steps=STEPS):
    """{(alpha, steps): [one output per member]} for the grid on one match (float32, or float64
    like predict.py with `double`); `alphas` / `steps` restrict it (steps must include 0)."""
    per = {(a, s): [] for a in alphas for s in steps}
    for model, normaliser in members:
        model = copy.deepcopy(model)
        model = model.double() if double else model.float()
        for alpha in alphas:
            norm = temporal.adapt_normaliser(normaliser, sample, alpha) if alpha else normaliser
            h0, h1, inside, fit = temporal.eata_inputs(model, norm, sample)
            per[(alpha, 0)].append(member_output(model.head, h0, h1, inside))
            # one Adam run read at each step count equals a separate run of that many steps
            head, snaps = copy.deepcopy(model.head), {}
            for count in temporal.entropy_steps(head, fit, max(steps)):
                if count in steps:
                    snaps[count] = member_output(head, h0, h1, inside)
            for target in [s for s in steps if s]:      # stopped early (no confident frame): the last head
                per[(alpha, target)].append(snaps[target] if target in snaps
                                            else member_output(head, h0, h1, inside))
    return per


def score_match(members, sample, per_member=False, costs=COSTS, alphas=ALPHAS, steps=STEPS,
                gates=GATES):
    """Score rows for the whole grid on one match; with `per_member`, each member alone too, every
    row then carrying a `seed` (the ensemble is seed -1)."""
    rows = []
    features = np.asarray(sample.features(False))
    for (alpha, count), outs in member_grid(members, sample, alphas=alphas, steps=steps).items():
        variants = [(-1, temporal.combine(outs))] + (list(enumerate(outs)) if per_member else [])
        for seed, logp in variants:
            for gate in gates:
                lp = gate_team_evidence(logp, features) if gate else logp
                for c in costs:
                    r = score(sample.labels, decode_table(sample, lp, transition_matrix(*c[:3]),
                                                          c[3]))
                    row = {"alpha": alpha, "steps": count, "gate": gate, "costs": c,
                           "spellF1": r["spellF1"], "macroF1": r["macroF1"]}
                    rows.append(dict(row, seed=seed) if per_member else row)
    return rows


def cached(folder, condition, name, compute):
    path = folder / "{}_{}.json".format(condition, name)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    rows = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(path, json.dumps(rows))
    return rows


def collect(count=MEMBERS, folder=None, scorer=score_match, final_paths=None, swa=0):
    """`final_paths` (default: decoder.json's members) score sc; `swa` picks the fold models."""
    folder = folder or (OUT / "m{}_swa{}".format(count, swa) if swa
                        else OUT if count == MEMBERS else OUT / "m{}".format(count))
    final_paths = final_paths or [ROOT / "models" / p
                                  for p in load_decoder(DEFAULT_DECODER)["members"]]
    check_members(folder, final_paths)      # the sc rows are these members' outputs
    final = [temporal.load(p) for p in final_paths]
    assert len(final) == count, "decoder.json lists {} members, not {}".format(len(final), count)
    results = {"sim": {}, "shift": {}, "sc": {}}
    names = sorted(LOADERS)
    for held in folds(names):
        members = fold_members(held, sorted(set(names) - set(held)), count, swa)
        for n in held:
            results["sim"][n] = cached(folder, "sim", n, lambda: scorer(
                members, CachedSample(n, SIM_TEST_SEED)))
            seed = SHIFT_SEED + names.index(n)
            results["shift"][n] = cached(folder, "shift", n, lambda: scorer(
                members, CachedSample(n, seed, randomised=True, faults=True)))
            print("scored {}".format(n), flush=True)
    for n in sorted(EXTRA_LOADERS):
        results["sc"][n] = cached(folder, "sc", n, lambda: scorer(final, CachedSample(n)))
        print("scored {}".format(n), flush=True)
    return results


def require_complete(seen, results):
    """Raise unless every setting in `seen` ({setting: {condition: {matches}}}) is scored on every
    match of every condition in `results`: a mean over whatever rows exist is not the rule's."""
    for key, conditions in seen.items():
        for condition, matches in results.items():
            missing = set(matches) - conditions.get(condition, set())
            if missing:
                raise ValueError("setting {} has no {} score for {}".format(
                    key, condition, ", ".join(sorted(missing))))


def table(results):
    """{setting: {condition: (mean spellF1, mean macroF1)}}."""
    out, seen = {}, {}
    for condition, matches in results.items():
        acc = {}
        for name, rows in matches.items():
            for r in rows:
                key = (r["alpha"], r["steps"], r["gate"], tuple(r["costs"]))
                acc.setdefault(key, []).append((r["spellF1"], r["macroF1"]))
                seen.setdefault(key, {}).setdefault(condition, set()).add(name)
        for key, values in acc.items():
            out.setdefault(key, {})[condition] = tuple(np.mean(values, axis=0))
    require_complete(seen, results)
    return out


def choose(summary):
    """The decision rule in the module docstring. Returns (setting, overall, reason)."""
    overall = {k: np.mean([v[c][0] for c in v]) for k, v in summary.items()}
    plain = {k: v for k, v in overall.items() if k[:3] == (0.0, 0, False)}
    base = max(plain, key=plain.get)
    best = max(overall, key=overall.get)
    if best[:3] == (0.0, 0, False):
        return base, overall, "no adaptation is best"
    ref = (0.0, 0, False, best[3])
    gain = overall[best] - overall[ref]
    worst = min(summary[best][c][0] - summary[ref][c][0] for c in summary[best])
    if gain >= NOISE and worst >= -NOISE:
        return best, overall, "adaptation +{:.4f} over none at the same costs, worst condition {:+.4f}" \
            .format(gain, worst)
    return base, overall, "best adaptation +{:.4f} (worst condition {:+.4f}) is within noise: none" \
        .format(gain, worst)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--members", type=int, default=MEMBERS, help="ensemble size")
    args = parser.parse_args(argv)
    summary = table(collect(args.members))
    setting, overall, reason = choose(summary)
    print("\nalpha steps gate costs                 overall   sim (spell/macro)   shift    sc")
    for key in sorted(summary, key=lambda k: -overall[k])[:25]:
        v = summary[key]
        print("{:<5} {:<5} {:<4} {:<22} {:.4f}   {}".format(
            key[0], key[1], "on" if key[2] else "off", str(key[3]), overall[key],
            "  ".join("{:.4f}/{:.4f}".format(*v[c]) for c in ("sim", "shift", "sc"))))
    plain = (0.0, 0, False, setting[3])
    print("\nno adaptation at those costs: {:.4f}".format(overall[plain]))
    print("chosen: tta_alpha {} eata_steps {} gate {} costs {} -- {}".format(
        setting[0], setting[1], "on" if setting[2] else "off", setting[3], reason))


if __name__ == "__main__":
    main()
