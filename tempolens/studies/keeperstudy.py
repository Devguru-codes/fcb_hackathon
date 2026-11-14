"""Goalkeeper-swap fault and repair on open data (plan of action row E1; aggregates only).

    python -m tempolens.studies.keeperstudy

1. Thresholds, on positions only: every open match's test noise draw with KEEPER_SWAPS keeper
   swaps (1-5 min each) is repaired with each (window, occupancy) setting; the share of swapped
   frames restored and the frames changed that were not swapped (on the faulted and on the
   unfaulted draw) are counted. Chosen: the most restored with nothing else changed.
2. Cross-validation with the chosen setting: each match held out with its 5 fold members (the
   adaptation study's), no adaptation, costs (2, 8, 8, 0.5), under four versions of its test draw:
   clean, faulted, faulted + repaired, clean + repaired. Cached per match in models/exp/keeper/.
"""

import json
from pathlib import Path

import numpy as np

from .. import adaptcv, temporal
from ..decoder import transition_matrix
from ..experiments import SIM_TEST_SEED, CachedSample, decode_table, folds, score, write_atomic
from ..match import repair_keepers
from ..opendata import LOADERS

OUT = Path(__file__).resolve().parents[2] / "models" / "exp" / "keeper"
WINDOWS, OCCUPANCIES, PRESENT = (60.0, 30.0, 120.0), (0.8, 0.6, 0.7, 0.9), 0.5
COSTS = (2.0, 8.0, 8.0, 0.5)
VERSIONS = ("clean", "fault", "fault+repair", "clean+repair")


def _changed(a, b):
    """(n,) frames where any slot's position differs (NaN equal to NaN)."""
    return (~((a == b) | (np.isnan(a) & np.isnan(b)))).any(axis=(1, 2))


def thresholds(names):
    rows = {(w, o): [] for w in WINDOWS for o in OCCUPANCIES}
    for n in names:
        clean = CachedSample(n, SIM_TEST_SEED).table()
        fault = CachedSample(n, SIM_TEST_SEED, keeper=True).table()
        swapped = _changed(clean.players, fault.players)
        for (w, o), acc in rows.items():
            settings = dict(window_s=w, occupancy=o, present=PRESENT)
            fixed = repair_keepers(fault, **settings).players
            wrong = _changed(clean.players, fixed)
            acc.append((swapped.sum(), (swapped & ~wrong).sum(), (wrong & ~swapped).sum(),
                        _changed(clean.players, repair_keepers(clean, **settings).players).sum()))
        print("thresholds: {} done ({} swapped frames)".format(n, int(swapped.sum())), flush=True)
    print("\nwindow occupancy  restored  other frames changed (faulted / clean draw)")
    table = {}
    for (w, o), acc in rows.items():
        a = np.sum(acc, axis=0)
        table[(w, o)] = (a[1] / a[0], int(a[2]), int(a[3]))
        print("{:<6} {:<9} {:.3f}     {} / {}".format(w, o, *table[(w, o)]))
    safe = [k for k, v in table.items() if v[1] == 0 and v[2] == 0]
    best = max(safe or table, key=lambda k: table[k][0])   # ties: the first listed (60 s, 0.8)
    print("chosen: window {} s, occupancy {} ({})".format(
        best[0], best[1], "nothing else changed" if safe else "NO setting is clean"))
    return dict(window_s=best[0], occupancy=best[1], present=PRESENT)


def cv(names, settings):
    out = {}
    for held in folds(names):
        members = adaptcv.fold_members(held, sorted(set(names) - set(held)), 5)
        for n in held:
            path = OUT / "cv_{window_s:g}-{occupancy:g}-{present:g}_{name}.json".format(
                name=n, **settings)
            if path.exists():
                out[n] = json.loads(path.read_text(encoding="utf-8"))
                continue
            samples = {"clean": CachedSample(n, SIM_TEST_SEED),
                       "fault": CachedSample(n, SIM_TEST_SEED, keeper=True),
                       "fault+repair": CachedSample(n, SIM_TEST_SEED, keeper=True,
                                                    repair=settings),
                       "clean+repair": CachedSample(n, SIM_TEST_SEED, repair=settings)}
            out[n] = {}
            for v, s in samples.items():
                logp = temporal.predict_ensemble(members, s)
                out[n][v] = score(s.labels, decode_table(s, logp, transition_matrix(*COSTS[:3]),
                                                          COSTS[3]))["spellF1"]
            path.parent.mkdir(parents=True, exist_ok=True)
            write_atomic(path, json.dumps(out[n]))
            print("cv: {} {}".format(n, " ".join("{} {:.4f}".format(v, out[n][v])
                                                 for v in VERSIONS)), flush=True)
    return {v: float(np.mean([out[n][v] for n in names])) for v in VERSIONS}


def main():
    names = sorted(LOADERS)
    settings = thresholds(names)
    m = cv(names, settings)
    cost, gain = m["clean"] - m["fault"], m["fault+repair"] - m["fault"]
    side = m["clean+repair"] - m["clean"]
    print("\n" + "  ".join("{} {:.4f}".format(v, m[v]) for v in VERSIONS))
    print("fault costs {:+.4f}; repair recovers {:+.4f} ({:.0%}); on clean draws {:+.4f}".format(
        -cost, gain, gain / cost if cost > 0 else float("nan"), side))
    ok = cost > 0 and gain >= 0.5 * cost and abs(side) <= 0.002
    print("E1 open-data conditions: {} (the match diagnostic is the third condition)".format(
        "met" if ok else "not met"))
    (OUT / "summary.json").write_text(json.dumps({"settings": settings, "means": m}, indent=1),
                                      encoding="utf-8")


if __name__ == "__main__":
    main()
