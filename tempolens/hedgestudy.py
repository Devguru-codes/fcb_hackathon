"""Blend of adapted and un-adapted members, and a wider cost grid with adaptation on (plan of
action rows B2, E10; open data, aggregates only).

    python -m tempolens.hedgestudy b2  [--until "2026-11-08 09:00"]
    python -m tempolens.hedgestudy e10 [--until ...]

B2 runs the ADseeds conditions (sim / shift / sc; the adaptation study's fold members, the final
members on SkillCorner; float32) for tta_alpha {0, 0.5} x eata_steps {0, 10} and caches every
member's output per match in <open data>/cache/hedge/. A blend averages each member's un-adapted
output with its adapted one (temporal.combine on the pair) before the 5 members are combined.
E10 re-decodes the cached (0.5, 10) outputs over a wider cost grid. `--until`: no new match is
started after that time (caches stay; a re-run resumes).
"""

import argparse
import itertools
from datetime import datetime
from multiprocessing import Pool

import numpy as np

from . import adaptcv, temporal
from .decoder import transition_matrix
from .experiments import check_members, decode_table, folds, open_data_dir, savez_atomic, \
    score
from .opendata import EXTRA_LOADERS, LOADERS
from .predict import DEFAULT_DECODER, load_decoder

CONDITIONS = ("sim", "shift", "sc")
KEYS = ((0.0, 0), (0.5, 0), (0.5, 10))
B2_COSTS = ((4.0, 8.0, 8.0, 1.0), (2.0, 8.0, 8.0, 0.5))
SUBMITTED = (4.0, 8.0, 8.0, 1.0)
E10_COSTS = list(itertools.product((1.5, 2.0, 3.0, 4.0, 6.0, 8.0), (2.0, 4.0, 8.0),
                                   (2.0, 4.0, 8.0), (0.0, 0.5, 1.0)))
VARIANTS = ("none", "tta", "tta+eata", "blend tta", "blend tta+eata")


def _cache():
    folder = open_data_dir() / "cache" / "hedge"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def matches():
    """[(condition, name, sample factory, which members)] in the ADseeds order."""
    out = []
    for held in folds(sorted(LOADERS)):
        for n in held:
            for c in ("sim", "shift"):
                out.append((c, n, lambda c=c, n=n: adaptcv.condition_sample(c, n), held))
    out += [("sc", n, lambda n=n: adaptcv.condition_sample("sc", n), None)
            for n in sorted(EXTRA_LOADERS)]
    return out


def compute(until=None):
    """Cache each member's outputs for KEYS on every match; False if stopped by `until`."""
    names = sorted(LOADERS)
    paths = [adaptcv.ROOT / "models" / p for p in load_decoder(DEFAULT_DECODER)["members"]]
    check_members(_cache(), paths)          # the sc outputs are these members'
    final = [temporal.load(p) for p in paths]
    members_of = {}
    for condition, name, make, held in matches():
        path = _cache() / "{}_{}.npz".format(condition, name)
        if path.exists():
            continue
        if until and datetime.now() >= until:
            print("stopped at {:%H:%M} (deadline)".format(datetime.now()), flush=True)
            return False
        if held is None:
            members = final
        else:
            key = tuple(held)
            if key not in members_of:
                members_of = {key: adaptcv.fold_members(held, sorted(set(names) - set(held)), 5)}
            members = members_of[key]
        grid = adaptcv.member_grid(members, make(), alphas=(0.0, 0.5), steps=(0, 10))
        savez_atomic(path, **{"a{}_s{}".format(*k): np.stack(grid[k]).astype(np.float32)
                          for k in KEYS})
        print("cached {} {}".format(condition, name), flush=True)
    return True


def load(condition, name):
    z = np.load(_cache() / "{}_{}.npz".format(condition, name))
    return {k: [m.astype(np.float64) for m in z["a{}_s{}".format(*k)]] for k in KEYS}


def variants(out):
    """{variant: [per-member outputs]}."""
    none, tta, eata = out[(0.0, 0)], out[(0.5, 0)], out[(0.5, 10)]
    return {"none": none, "tta": tta, "tta+eata": eata,
            "blend tta": [temporal.combine([a, b]) for a, b in zip(none, tta)],
            "blend tta+eata": [temporal.combine([a, b]) for a, b in zip(none, eata)]}


def _score_job(job):
    """{(variant, seed, costs): spellF1} for one match; seed -1 = the ensemble."""
    condition, name, costs, keep = job
    sample = adaptcv.condition_sample(condition, name)
    rows = {}
    for variant, outs in variants(load(condition, name)).items():
        if variant not in keep:
            continue
        for seed, logp in [(-1, temporal.combine(outs))] + list(enumerate(outs)):
            for c in costs:
                rows[(variant, seed, c)] = score(sample.labels, decode_table(
                    sample, logp, transition_matrix(*c[:3]), c[3]))["spellF1"]
    return condition, rows


def collect(costs, keep, workers):
    jobs = [(c, n, costs, keep) for c, n, _, _ in matches()]
    acc = {}
    with Pool(workers) as pool:
        for condition, rows in pool.imap_unordered(_score_job, jobs):
            for k, v in rows.items():
                acc.setdefault(k, {}).setdefault(condition, []).append(v)
    return {k: {c: float(np.mean(v)) for c, v in conds.items()} for k, conds in acc.items()}


def mean3(cond):
    return float(np.mean([cond[c] for c in CONDITIONS]))


def b2(workers):
    t = collect(B2_COSTS, VARIANTS, workers)
    seeds = range(5)
    # the rule's references (0.6475, 0.7727 rounded) taken from this run, not from literals
    check = (mean3(t[("none", -1, SUBMITTED)]), mean3(t[("tta+eata", -1, SUBMITTED)]))
    sim_none = t[("none", -1, SUBMITTED)]["sim"]
    print("variant          costs                  sim    shift  sc     mean   seed gains over none")
    passed = []
    for c in B2_COSTS:
        for v in VARIANTS:
            e = t[(v, -1, c)]
            gains = [mean3(t[(v, s, c)]) - mean3(t[("none", s, c)]) for s in seeds]
            print("{:<16} {:<22} {:.4f} {:.4f} {:.4f} {:.4f} {}".format(
                v, str(c), e["sim"], e["shift"], e["sc"], mean3(e),
                " ".join("{:+.4f}".format(g) for g in gains)), flush=True)
            if v.startswith("blend") and mean3(e) >= check[1] and e["sim"] >= sim_none \
                    and all(g > 0 for g in gains):
                passed.append((e["sim"], v, c))
    print("ADseeds check at (4, 8, 8, 1.0): none {:.4f} (0.6448), tta+eata {:.4f} (0.6475)".format(
        *check))
    print("B2: {}".format("passes: {} at {} (sim {:.4f})".format(*max(passed)[1:], max(passed)[0])
                          if passed else "fails"))


def e10(workers):
    t = collect(E10_COSTS, ("tta+eata",), workers)
    ens = {c: mean3(t[("tta+eata", -1, c)]) for c in E10_COSTS}
    seed = {c: [mean3(t[("tta+eata", s, c)]) for s in range(5)] for c in E10_COSTS}
    ref = SUBMITTED
    print("submitted costs {}: ensemble {:.4f}".format(ref, ens[ref]))
    for c in sorted(E10_COSTS, key=lambda c: -ens[c])[:10]:
        print("{:<22} ensemble {:.4f} ({:+.4f}) seeds {}".format(
            str(c), ens[c], ens[c] - ens[ref],
            " ".join("{:+.4f}".format(a - b) for a, b in zip(seed[c], seed[ref]))))
    ok = [c for c in E10_COSTS if ens[c] - ens[ref] >= 0.002
          and all(a - b >= 0.002 for a, b in zip(seed[c], seed[ref]))]
    print("E10: {}".format("passes: costs {} ({:.4f})".format(max(ok, key=ens.get),
                                                            ens[max(ok, key=ens.get)])
                           if ok else "fails: costs stay (4, 8, 8, 1.0)"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("study", choices=("b2", "e10"))
    parser.add_argument("--until", default=None, help='"YYYY-MM-DD HH:MM": start nothing after')
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    until = datetime.strptime(args.until, "%Y-%m-%d %H:%M") if args.until else None
    if not compute(until):
        raise SystemExit(3)
    if until and datetime.now() >= until:
        print("deadline reached before scoring; caches kept", flush=True)
        raise SystemExit(3)
    (b2 if args.study == "b2" else e10)(args.workers)


if __name__ == "__main__":
    main()
