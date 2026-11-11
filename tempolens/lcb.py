"""The other label convention as a ready hedge (plan of action row LCB). Report only.

    python -m tempolens.lcb train final <k>          # one model per lane job (GPU lane)
    python -m tempolens.lcb train fold <f> <k>       # fold f = 0..4 in experiments.folds order
    python -m tempolens.lcb outputs                  # ms50 members' outputs, per match
    python -m tempolens.lcb score                    # the A / ms50 / mix table
    python -m tempolens.candidate lcb_ms50|lcb_mix --data-root <bundle>

Every model is written through adaptcv.train_once (lock, temporary file, replace) and a training
job whose model exists exits 0 at once, so the queue resumes where it stopped. A step that needs a
file that is missing stops with exit 3 and names it.
"""

import argparse
import sys

import numpy as np

from . import adaptcv, hedgestudy, pffstudy, temporal
from .decoder import transition_matrix
from .experiments import CachedSample, check_members, decode_table, folds, open_data_dir, \
    savez_atomic, score
from .opendata import LOADERS

MEMBERS, MIN_SPELL = 5, 50
FINAL_DIR = adaptcv.ROOT / "models"        # the committed ms50 finals (LCB-ADOPT)
FOLD_DIR = adaptcv.OUT / "models_ms50"
KEYS = ((0.0, 0), (0.5, 10))
SETTINGS = (("no adaptation", (0.0, 0), (4.0, 8.0, 8.0, 1.0)),
            ("no adaptation, CV argmax", (0.0, 0), (2.0, 8.0, 8.0, 0.5)),
            ("submitted: TTA 0.5 + EATA 10", (0.5, 10), (4.0, 8.0, 8.0, 1.0)))
CONDITIONS = ("sim", "shift", "sc", "pff")


def stop(message):
    print("STOPPED: " + message, flush=True)
    sys.exit(3)


def final_path(k):
    return FINAL_DIR / "tempolens_ms50_m{}.pt".format(k)


def fold_path(held, k):
    return FOLD_DIR / "{}_m{}.pt".format("-".join(held), k)


def model_paths():
    names = sorted(LOADERS)
    return ([final_path(k) for k in range(MEMBERS)]
            + [fold_path(held, k) for held in folds(names) for k in range(MEMBERS)])


def require(paths):
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        stop("missing {} file(s), first {}".format(len(missing), missing[0]))


def train(kind, k, fold=None):
    """One model; returns at once if it exists (the lane's resume)."""
    names = sorted(LOADERS)
    cmd = [sys.executable, "-m", "tempolens.train", "--member", str(k), "--draws",
           str(adaptcv.DRAWS), "--min-spell", str(MIN_SPELL)]
    if kind == "final":
        path = final_path(k)
    else:
        held = folds(names)[fold]
        path = fold_path(held, k)
        cmd += ["--matches", *sorted(set(names) - set(held))]
    if path.exists():
        print("exists: {}".format(path.name), flush=True)
        return path
    return adaptcv.train_once(path, cmd + ["--out", str(path)])


def _cache():
    folder = open_data_dir() / "cache" / "lcb"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def jobs():
    """[(condition, name, sample factory, held or None)]: the hedge study's matches plus pff-16."""
    out = list(hedgestudy.matches())
    out += [("pff", g, lambda g=g: CachedSample("pff_{}".format(g)), None)
            for g in pffstudy.test_ids(pffstudy.pff_ids())]
    return out


def store_path(condition, name):
    return _cache() / "{}_{}.npz".format(condition, name)


def outputs():
    require(model_paths())
    check_members(_cache(), model_paths())
    finals = [temporal.load(final_path(k)) for k in range(MEMBERS)]
    loaded = {}
    for condition, name, make, held in jobs():
        path = store_path(condition, name)
        if path.exists():
            continue
        if held is None:
            members = finals
        else:
            key = tuple(held)
            if key not in loaded:
                loaded = {key: [temporal.load(fold_path(held, k)) for k in range(MEMBERS)]}
            members = loaded[key]
        grid = adaptcv.member_grid(members, make(), alphas=(0.0, 0.5), steps=(0, 10))
        savez_atomic(path, **{"a{}_s{}".format(*k): np.stack(grid[k]).astype(np.float32)
                              for k in KEYS})
        print("outputs {} {}".format(condition, name), flush=True)


def ens5_outputs(condition, name):
    """{key: [member outputs]} of the submitted members (hedge cache, pff-16 cache)."""
    if condition == "pff":
        z = np.load(pffstudy._cache() / "pff_{}_outputs.npz".format(name))
        return {k: list(z["a{}_s{}".format(*k)].astype(np.float64)) for k in KEYS}
    out = hedgestudy.load(condition, name)
    return {k: out[k] for k in KEYS}


def ms50_outputs(condition, name):
    z = np.load(store_path(condition, name))
    return {k: list(z["a{}_s{}".format(*k)].astype(np.float64)) for k in KEYS}


def needed_files():
    out = []
    for condition, name, _, _ in jobs():
        out.append(store_path(condition, name))
        out.append(pffstudy._cache() / "pff_{}_outputs.npz".format(name) if condition == "pff"
                   else hedgestudy._cache() / "{}_{}.npz".format(condition, name))
    return out


def table_of(results):
    """{(set, setting, condition, convention): mean spellF1} from per-match result rows."""
    acc = {}
    for (condition, _), rows in results.items():
        for key, value in rows.items():
            acc.setdefault(key[:2] + (condition,) + key[2:], []).append(value)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def score_one(condition, name, make):
    sample = make()
    labels = pffstudy.labels_by_convention(sample)
    a, b = ens5_outputs(condition, name), ms50_outputs(condition, name)
    sets = {"ENS5": a, "ms50": b, "mix": {k: a[k] + b[k] for k in KEYS}}
    rows = {}
    for set_name, outs in sets.items():
        for label, key, costs in SETTINGS:
            pred = decode_table(sample, temporal.combine(outs[key]), transition_matrix(*costs[:3]),
                                costs[3])
            for c, y in labels.items():
                rows[(set_name, label, c)] = score(y, pred)["spellF1"]
    return rows


def report(table):
    for convention in pffstudy.CONVENTIONS:
        print("\nlabels at {} frames: mean spellF1 (sim / shift / sc / pff-16; mean of the first "
              "three)".format(convention))
        for set_name in ("ENS5", "ms50", "mix"):
            for label, _, costs in SETTINGS:
                v = [table[(set_name, label, c, convention)] for c in CONDITIONS]
                print("{:<5} {:<30} {:<22} {}   mean3 {:.4f}".format(
                    set_name, label, str(costs), " / ".join("{:.4f}".format(x) for x in v),
                    float(np.mean(v[:3]))), flush=True)
    print("\nLCB: report only; no rule picks the convention, nothing is switched", flush=True)


def score_all():
    require(needed_files())
    results = {}
    for condition, name, make, _ in jobs():
        results[(condition, name)] = score_one(condition, name, make)
        print("scored {} {}".format(condition, name), flush=True)
    report(table_of(results))


def members_for(row):
    """Decoder member lists (relative to models/) for the candidate rows; None if any is missing."""
    ms50 = [final_path(k) for k in range(MEMBERS)]
    if not all(p.exists() for p in ms50):
        return None
    rel = [p.relative_to(adaptcv.ROOT / "models").as_posix() for p in ms50]
    return rel if row == "lcb_ms50" else list(adaptcv.ENS5) + rel


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("train", "outputs", "score"))
    parser.add_argument("args", nargs="*")
    a = parser.parse_args(argv)
    if a.step == "train":
        if a.args[:1] == ["final"] and len(a.args) == 2:
            return train("final", int(a.args[1]))
        if a.args[:1] == ["fold"] and len(a.args) == 3:
            return train("fold", int(a.args[2]), int(a.args[1]))
        parser.error("train final <k> | train fold <f> <k>")
    return outputs() if a.step == "outputs" else score_all()


if __name__ == "__main__":
    main()
