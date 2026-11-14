"""How much do tracker faults the simulator did not train on (identity swaps, duplicate tracks,
phantom tracks) cost the model? One cross-validation fold, held-out matches scored with and
without the faults.

    python -m tempolens.studies.robustness
"""

import numpy as np

from .. import temporal
from ..decoder import transition_matrix
from ..experiments import (CachedSample, SIM_TEST_SEED, SIM_TRAIN_SEEDS, baseline_codes,
                          decode_table, folds, score)
from ..features import TableSample
from ..opendata import LOADERS, load_open_match
from ..simulate import add_tracker_faults, simulate_atd

COSTS = (4.0, 8.0, 8.0)   # the single-GRU optimum (one model, not the submitted ensemble)


def main():
    names = sorted(LOADERS)
    held = folds(names)[0]
    train = [CachedSample(n, s) for n in sorted(set(names) - set(held)) for s in SIM_TRAIN_SEEDS]
    model, normaliser = temporal.train(train)
    transition = transition_matrix(*COSTS)
    for n in held:
        standard = simulate_atd(load_open_match(n), seed=SIM_TEST_SEED)
        faulty = add_tracker_faults(standard, seed=77)
        for label, table in (("standard noise", standard), ("+ tracker faults", faulty)):
            logp = temporal.predict(model, normaliser, TableSample(table))
            r = score(table.labels, decode_table(table, logp, transition))
            b = score(table.labels, baseline_codes(table))
            print("{:<12} {:<18} slots {:>3}  model spellF1 {:.4f}  baseline {:.4f}".format(
                n, label, table.players.shape[1], r["spellF1"], b["spellF1"]), flush=True)


if __name__ == "__main__":
    main()
