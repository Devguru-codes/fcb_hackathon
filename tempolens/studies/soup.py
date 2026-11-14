"""Model soups of the 5 members (plan of action row E8; open data, aggregates only).

    python -m tempolens.studies.soup

A soup is one network whose weights are the average of several members'. The members scale their
inputs differently and flag different missing columns, so each member's first layer is first
re-expressed for one common normaliser (the members' mean median and scale; missing-value flags
on the union of their flagged columns, zero weight where a member has no flag) -- exact except
where the +/-10 clip binds. Uniform soup: all 5 fold members. Greedy soup: members in order of
their own score on two of the fold's training matches, each kept if that score does not drop
(never the held-out matches). Each soup is scored on the held-out matches over the CV's 189
decoder settings, against ENS5 (0.7760).
"""

import copy
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .. import adaptcv, cvrescore, temporal
from ..decoder import transition_matrix
from ..experiments import SIM_TEST_SEED, CachedSample, decode_table, folds, score
from ..opendata import LOADERS

OUT = Path(__file__).resolve().parents[2] / "models" / "exp" / "soup"


def soup(members):
    """(model, normaliser): the uniform weight average of (model, normaliser) members."""
    torch = temporal._torch()
    norms = [n for _, n in members]
    common = temporal.Normaliser.from_state({
        "median": np.mean([n.median for n in norms], axis=0),
        "scale": np.mean([n.scale for n in norms], axis=0),
        "flagged": np.unique(np.concatenate([n.flagged for n in norms])), "drop": ()})
    k = len(common.median)
    states = []
    for model, norm in members:
        state = {name: p.detach().double().clone() for name, p in model.state_dict().items()}
        w, b = state["inp.0.weight"], state["inp.0.bias"]
        ratio = torch.from_numpy(common.scale / norm.scale)
        shift = torch.from_numpy((common.median - norm.median) / norm.scale)
        new = torch.zeros((w.shape[0], k + len(common.flagged)), dtype=torch.float64)
        new[:, :k] = w[:, :k] * ratio
        b = b + w[:, :k] @ shift
        where = np.searchsorted(common.flagged, norm.flagged)
        new[:, k + where] = w[:, k:]
        state["inp.0.weight"], state["inp.0.bias"] = new, b
        states.append(state)
    out = copy.deepcopy(members[0][0]).double()
    if out.inp[0].in_features != k + len(common.flagged):
        layer = torch.nn.Linear(k + len(common.flagged), out.inp[0].out_features).double()
        out.inp[0] = layer
    out.load_state_dict({name: torch.stack([s[name] for s in states]).mean(dim=0)
                         for name in states[0]})
    return out.eval(), common


def spell(model_norm, name, costs=(2.0, 8.0, 8.0, 0.5)):
    sample = CachedSample(name, SIM_TEST_SEED)
    logp = temporal.predict(*model_norm, sample)
    return score(sample.labels, decode_table(sample, logp, transition_matrix(*costs[:3]),
                                             costs[3]))["spellF1"]


def main():
    names = sorted(LOADERS)
    logps = {"uniform": {}, "greedy": {}}
    picked = {}
    for held in folds(names):
        rest = sorted(set(names) - set(held))
        members = adaptcv.fold_members(held, rest, 5)
        val = rest[:2]
        single = [np.mean([spell((m.double(), n), v) for v in val]) for m, n in members]
        order = list(np.argsort(single)[::-1])
        chosen, best = [order[0]], single[order[0]]
        for i in order[1:]:
            trial = np.mean([spell(soup([members[j] for j in chosen + [i]]), v) for v in val])
            if trial >= best:
                chosen, best = chosen + [i], trial
        picked["-".join(held)] = [int(i) for i in chosen]
        for kind, idx in (("uniform", range(5)), ("greedy", chosen)):
            model_norm = soup([members[j] for j in idx])
            for n in held:
                logps[kind][n] = temporal.predict(*model_norm, CachedSample(n, SIM_TEST_SEED))
        print("fold {}: greedy soup keeps members {}".format(held, picked["-".join(held)]),
              flush=True)
    result = {}
    with Pool(4) as pool:
        for kind in ("uniform", "greedy"):
            full, dfl, metrica = cvrescore.summarise(cvrescore.scores(logps[kind], pool))
            i = int(np.argmax(full[:, 0]))
            result[kind] = float(full[i, 0])
            print("{} soup: {:.4f} (DFL {:.4f}, Metrica {:.4f}) at {}; vs ENS5 0.7760 {:+.4f}"
                  .format(kind, full[i, 0], dfl[i, 0], metrica[i, 0], cvrescore.SETTINGS[i],
                          full[i, 0] - 0.7760), flush=True)
    print("E8: {}".format("passes" if max(result.values()) >= 0.7760 + 0.004 else "fails"))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps({"result": result, "greedy": picked}, indent=1),
                                      encoding="utf-8")


if __name__ == "__main__":
    main()
