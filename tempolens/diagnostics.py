"""Data-day checks on the competition match. Prints aggregate numbers only.

    python -m tempolens.diagnostics --data-root data

Everything printed is a count or a fraction -- no frame-level values. Do not extend it to print
per-frame values. Nothing here reads a ground truth: under the organisers' rules (2026-11-03) no
setting is chosen with one, so the pipeline is not scored here.

Checks, each with the value it should have:
1. the tracking has one record per tactical video frame (199,319);
2. our loader reproduces the organisers' own baseline file on every frame, which proves frame
   alignment and the home/away mapping at once;
3. ball and player visibility match the published statistics (40.7% of graded frames without a
   ball, in ~150 long gaps; ~20.5 players tracked), i.e. the simulator's calibration holds;
4. every Smart Tagging team name resolves to a side;
5. the Smart Tagging prior's self-calibrated reliability (agreement with the confident model);
6. keeper-zone slots (plan row E0): per team, the share of graded frames on which one of its
   slots has stood for 5 s or more in the other team's keeper zone (within 20 m of that goal
   line, central 40 m), with the slots and minutes involved -- a sign of keeper slots filed
   under the wrong team.
"""

import argparse
import json
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
from .match import competition_table  # noqa: E402
from .tags import resolve_side  # noqa: E402


def _gaps(missing):
    edges = np.diff(np.concatenate(([0], missing.astype(np.int8), [0])))
    return np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)


def keeper_zone(table, depth=20.0, half_width=20.0, min_seconds=5.0):
    """Check 6, aggregates only: {team: share of graded frames, slots, minutes}."""
    graded = table.period > 0
    length, width = table.pitch
    # the goal a team defends sits at -sign * length / 2 (home) or +sign * length / 2 (away)
    sign = attack_sign(table)
    x = table.players[..., 0] - length / 2.0
    near_y = np.abs(table.players[..., 1] - width / 2.0) <= half_width
    out = {}
    for team, name in ((0, "home"), (1, "away")):
        other_goal = sign * length / 2.0 * (1.0 if team == 0 else -1.0)
        slots = np.flatnonzero(table.slot_team == team)
        flagged, involved = np.zeros(table.n_frames, dtype=bool), 0
        for k in slots:
            inside = graded & near_y[:, k] & (np.abs(x[:, k] - other_goal) <= depth)
            edges = np.diff(np.concatenate(([0], inside.astype(np.int8), [0])))
            starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
            long = (stops - starts) >= min_seconds * table.fps
            for a, b in zip(starts[long], stops[long]):
                flagged[a:b] = True
            involved += int(long.any())
        out[name] = {"share_of_graded": round(float(flagged[graded].mean()), 4),
                     "slots": involved, "slots_of_team": int(len(slots)),
                     "minutes": round(float(flagged.sum()) / table.fps / 60.0, 1)}
    return out


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
    report["keeper_zone"] = keeper_zone(table)

    # 4. Smart Tagging team names
    path = require(["smart_tagging"], GAME, data_root)[0]
    names = Counter(label.findtext("text") for label in ET.parse(path).getroot().iter("label")
                    if label.findtext("group") == "Team")
    report["tag_teams"] = {name: {"count": count,
                                  "side": resolve_side(name, table.meta["home"],
                                                       table.meta["away"])}
                           for name, count in names.items()}

    # 5. Smart Tagging reliability, measured against the confident model (no labels)
    if model_paths:
        from .predict import learned_logp, load_decoder
        from .tagprior import read_instances, self_calibrated
        from games import GAMES
        settings = load_decoder(decoder_path)
        _, logp, _ = learned_logp(data_root, GAME, model_paths)
        instances = read_instances(path, table.meta["home"], table.meta["away"], GAMES[GAME]["fps"])
        _, tag_report = self_calibrated(logp, instances, graded)
        report["tag_prior"] = {
            group: {"weight": round(w, 3), "reliability": round(r, 3), "confident_frames": n}
            for group, (w, r, n) in tag_report.items()}
        report["tag_prior_setting"] = settings.get("tag_prior", "off")
    return report


def main(argv=None):
    from .predict import DEFAULT_DECODER, default_models
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model", nargs="+", help="default: the decoder's members")
    parser.add_argument("--decoder", default=str(DEFAULT_DECODER))
    args = parser.parse_args(argv)
    models = args.model or default_models(args.decoder)
    print(json.dumps(run(args.data_root, models, args.decoder), indent=2, default=str))


if __name__ == "__main__":
    main()
