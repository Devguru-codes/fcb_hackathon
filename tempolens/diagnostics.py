"""Data-day checks on the competition match. Prints aggregate numbers only.

    python -m tempolens.diagnostics --data-root data

Everything printed is a count, a fraction or a score on the public five minutes -- no frame-level
values -- so the output can be shared (e.g. pasted to a hosted assistant) without disclosing the
licensed data. Do not extend it to print per-frame values.

Checks, each with the value it should have:
1. the tracking has one record per tactical video frame (199,319);
2. our loader reproduces the organisers' own baseline file on every frame, which proves frame
   alignment and the home/away mapping at once;
3. ball and player visibility match the published statistics (40.7% of graded frames without a
   ball, in ~150 long gaps; ~20.5 players tracked), i.e. the simulator's calibration holds;
4. every Smart Tagging team name resolves to a side;
5. public-window scores of the learned model, with bootstrap intervals (a sanity check, not a
   model selection signal).
"""

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
sys.path.insert(0, str(ROOT / "starter"))
import possession_metrics as pm  # noqa: E402

from games import GAME, require  # noqa: E402
from .experiments import baseline_codes  # noqa: E402
from .features import attack_sign  # noqa: E402
from .match import X, competition_table  # noqa: E402
from .tags import resolve_side  # noqa: E402


def _gaps(missing):
    edges = np.diff(np.concatenate(([0], missing.astype(np.int8), [0])))
    return np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)


def run(data_root, model_paths=None, decoder_path=None):
    report = {"note": "aggregate numbers only; safe to share"}
    table = competition_table(GAME, data_root)
    graded = table.period > 0

    # 1. alignment of the raw file
    import load_atd
    report["alignment"] = {k: (list(v) if isinstance(v, tuple) else v)
                           for k, v in load_atd.check_alignment_raw(GAME, data_root).items()}

    # 2. our loader against the organisers' baseline file
    official = pm.label_codes(pm.read_label_csv(ROOT / "evaluation/results/baseline_raw.csv"))
    ours = baseline_codes(table, window=0)
    ours[~graded] = official[~graded]   # outside the halves the file says something; we say X
    report["baseline_raw_reproduced_fraction"] = float(np.mean(ours == official))

    # 3. visibility against the published statistics and the simulator's calibration
    missing = np.isnan(table.ball[:, 0])
    gaps = np.concatenate([_gaps(missing[a:b]) for a, b in table.runs()])
    visible = np.isfinite(table.players[..., 0]).sum(axis=1)
    report["visibility"] = {
        "ball_missing_graded": round(float(missing[graded].mean()), 4),
        "ball_gaps": int(len(gaps)), "ball_gap_median_frames": float(np.median(gaps)),
        "ball_gap_mean_frames": round(float(gaps.mean()), 1),
        "players_visible_graded_mean": round(float(visible[graded].mean()), 2),
        "track_slots": int(table.players.shape[1]),
        "home": table.meta["home"], "away": table.meta["away"],
        "expected": "ball_missing 0.407, ~150 gaps of median ~268, ~20.5 players, 46 slots",
    }
    report["attack_sign_by_half"] = [float(attack_sign(table)[a]) for a, _ in table.runs()]

    # 4. Smart Tagging team names
    path = require(["smart_tagging"], GAME, data_root)[0]
    names = Counter(label.findtext("text") for label in ET.parse(path).getroot().iter("label")
                    if label.findtext("group") == "Team")
    report["tag_teams"] = {name: {"count": count,
                                  "side": resolve_side(name, table.meta["home"],
                                                       table.meta["away"])}
                           for name, count in names.items()}

    # 5. public-window scores
    truth = pm.label_codes(pm.read_label_csv(ROOT / "evaluation/gt/BAR-ATH_possession_gt_public.csv"))
    scores = {"baseline_smooth50": baseline_codes(table, window=50)}
    if model_paths:
        from .predict import learned_states
        scores["learned"], _ = learned_states(data_root, GAME, model_paths, decoder_path)
    report["public_window"] = {}
    for name, codes in scores.items():
        codes = np.where(codes == X, X, codes).astype(np.int8)
        result = pm.evaluate(truth, codes, bootstrap_n=1000)
        report["public_window"][name] = {
            "spellF1": round(result["spell_f1_mean"], 4),
            "spellF1_90pct": [round(result["spell_bootstrap_low"], 4),
                              round(result["spell_bootstrap_high"], 4)],
            "macroF1": round(result["macro_f1"], 4),
            "n_pred_spells": result["n_pred_spells"], "n_gt_spells": result["n_gt_spells"],
        }
    report["public_window_note"] = ("17 possessions: use for sanity only. baseline_smooth50 "
                                    "should read spellF1 0.3256, macroF1 0.6207 (README).")
    return report


def main(argv=None):
    from .predict import DEFAULT_DECODER, DEFAULT_MODELS
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model", nargs="+", default=[str(p) for p in DEFAULT_MODELS])
    parser.add_argument("--decoder", default=str(DEFAULT_DECODER))
    args = parser.parse_args(argv)
    print(json.dumps(run(args.data_root, args.model, args.decoder), indent=2, default=str))


if __name__ == "__main__":
    main()
