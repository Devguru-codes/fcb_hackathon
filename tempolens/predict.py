"""Generate the official More Than a Hack possession-file submission.

    python -m tempolens.predict --data-root data --output evaluation/results/<team>.csv

It runs the committed temporal ensemble (trained on open data only; the members are listed in
models/decoder.json, whose costs were tuned for exactly them) on CPU, averages it, and decodes
with the committed decoder settings, so the same data always gives the same file.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from games import GAME, half_video_blocks, source_frames

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DECODER = ROOT / "models" / "decoder.json"
# First alive frame of the second half. FAQ.md: grading starts at 110,808 but the kickoff is
# 113,742, and every graded frame before it is D. The model cannot see a whistle, so it is set here.
SECOND_HALF_KICKOFF = {"BAR-ATH": 113742}


def write_submission(path, labels):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["frame", "label"])
        writer.writerows(enumerate(labels))


def load_decoder(path=DEFAULT_DECODER):
    settings = json.loads(Path(path).read_text(encoding="utf-8"))
    return {key: value for key, value in settings.items() if not key.startswith("_")}


def default_models(decoder_path=DEFAULT_DECODER):
    """The ensemble the decoder settings were tuned for, named in the decoder JSON."""
    return [ROOT / "models" / name for name in load_decoder(decoder_path)["members"]]


def _paths(model_paths):
    if isinstance(model_paths, (str, Path)):
        return [Path(model_paths)]
    return [Path(path) for path in model_paths]


def learned_logp(data_root, game, model_paths, tta_alpha=0.0, eata_steps=0, blend=False,
                 both_views=False):
    """(competition MatchTable, ensemble output (n, 4), its TableSample). The slow part: run it
    once; the sample holds the features, computed once.

    `tta_alpha` > 0: test-time adaptation -- each member's feature scaling is moved that far
    toward this match's own medians and spreads (temporal.adapt_normaliser), unlabelled.
    `eata_steps` > 0: then each member's output layer adapts to the match (temporal.adapt_head).
    `blend`: each member's un-adapted and adapted outputs averaged before combining (plan row B2).
    """
    from . import temporal
    from .match import competition_table

    table = competition_table(game, data_root)
    paths = _paths(model_paths)
    if not paths:
        raise IOError("no model weights given and none found in models/")
    members = [temporal.load(path) for path in paths]
    from .features import TableSample
    sample = TableSample(table)      # features computed once, shared by every member
    return table, ensemble_logp(members, sample, tta_alpha, eata_steps, blend, both_views), sample


def ensemble_logp(members, sample, tta_alpha=0.0, eata_steps=0, blend=False, both_views=False):
    """The ensemble output after adaptation; with `blend`, each member's un-adapted output
    averaged with its adapted one first, as hedgestudy's blend variants (plan row B2)."""
    from . import temporal
    plain = [temporal.predict(model, normaliser, sample) for model, normaliser in members] \
        if blend else None              # before adapting: adapt_head changes the models
    members = adapt_members(members, sample, tta_alpha, eata_steps, both_views)
    if not blend:
        return temporal.predict_ensemble(members, sample)
    return temporal.combine([temporal.combine([a, temporal.predict(model, normaliser, sample)])
                             for a, (model, normaliser) in zip(plain, members)])


def adapt_members(members, sample, tta_alpha=0.0, eata_steps=0, both_views=False):
    """[(model, normaliser)] after test-time adaptation of the feature scaling (`tta_alpha`; from
    both team views with `both_views`, plan row TTA-2V), then of the output layer (`eata_steps`),
    both unlabelled; see learned_logp."""
    from . import temporal
    if tta_alpha:
        members = [(model, temporal.adapt_normaliser(normaliser, sample, tta_alpha, both_views))
                   for model, normaliser in members]
    if eata_steps:
        members = [(temporal.adapt_head(model, normaliser, sample, eata_steps), normaliser)
                   for model, normaliser in members]
    return members


def gate_team_evidence(logp, features, flight=0.2, loose=0.3, flight_speed=9.5, loose_m=1.35):
    """Damp the home-vs-away evidence where it is least trustworthy (after a rival, Suleyman):
    the log-odds of H against A are multiplied by `flight` where the seen ball moves faster than
    `flight_speed` m/s (in the air or passed: a nearby player is not on it) and by `loose` where
    nobody is within `loose_m` of it; P(alive) is unchanged. The decoder then keeps the current
    owner through such frames unless the evidence after them disagrees. `features`: the match's
    home-view feature matrix (TableSample.features(False))."""
    from .features import feature_names
    from .match import A, H
    col = {name: i for i, name in enumerate(feature_names())}
    seen = features[:, col["ball_seen"]] > 0.5
    factor = np.ones(len(logp))
    factor[seen & (np.nan_to_num(features[:, col["d_min"]]) > loose_m)] = loose
    factor[seen & (np.nan_to_num(features[:, col["ball_speed"]]) > flight_speed)] = flight
    out = logp.copy()
    p = np.exp(logp[:, :3])
    alive = p[:, H] + p[:, A]
    share = 1.0 / (1.0 + np.exp(-factor * (logp[:, H] - logp[:, A])))
    out[:, H] = np.log(np.clip(alive * share, 1e-6, 1))
    out[:, A] = np.log(np.clip(alive * (1 - share), 1e-6, 1))
    return out


def decode_learned(table, logp, settings, data_root, game, tag_prior=None):
    """(H/A/D codes per half with X outside, tag calibration report or None).

    `tag_prior` overrides settings["tag_prior"]: "auto" applies the self-calibrated Smart
    Tagging prior (tagprior.py), "off" does not.
    """
    from .decoder import transition_matrix
    from .experiments import decode_table

    mode = tag_prior or settings.get("tag_prior", "off")
    tag_report = None
    if mode == "auto":
        from games import GAMES, require
        from .tagprior import SHRINK, read_instances, self_calibrated
        path = require(["smart_tagging"], game, data_root)[0]
        instances = read_instances(path, table.meta["home"], table.meta["away"],
                                   GAMES[game]["fps"])
        logp, tag_report = self_calibrated(logp, instances, table.period > 0,     # plan row TS
                                           shrink=settings.get("tag_shrink", SHRINK))
    elif mode != "off":
        raise ValueError("tag_prior must be 'auto' or 'off', not {!r}".format(mode))
    transition = transition_matrix(settings["switch"], settings["alive_to_dead"],
                                   settings["dead_to_alive"])
    return decode_table(table, logp, transition, settings.get("change_weight", 0.0)), tag_report


def predict(data_root, output, model_paths=None, decoder_path=DEFAULT_DECODER, game=GAME):
    """Run the pipeline and write a validated-shape CSV. Returns the label array.

    `model_paths` defaults to the members listed in the decoder JSON.
    """
    model_paths = model_paths or default_models(decoder_path)
    settings = load_decoder(decoder_path)
    table, logp, sample = learned_logp(data_root, game, model_paths,
                                       settings.get("tta_alpha", 0.0),
                                       settings.get("eata_steps", 0),
                                       settings.get("blend", False),
                                       settings.get("tta_both_views", False))
    if settings.get("flight_gate") is not None:
        logp = gate_team_evidence(logp, sample.features(False), settings["flight_gate"],
                                  settings["loose_gate"])
    if settings.get("kickoff_prior", False):     # plan row KOP; off in the submission
        logp = kickoff_prior(logp, half_video_blocks(game), SECOND_HALF_KICKOFF.get(game))
    states, _ = decode_learned(table, logp, settings, data_root, game)
    labels = finish_labels(states, half_video_blocks(game), SECOND_HALF_KICKOFF.get(game),
                           source_frames("tactical", game))
    write_submission(output, labels)
    return labels


def kickoff_prior(logp, blocks, second_kickoff):
    """logp with the second half's graded frames before its kickoff (a FAQ fact) set to dead
    ball as decoder evidence; finish_labels still writes D there as the check."""
    out = logp.copy()
    if second_kickoff is not None:
        out[blocks[1][1]:second_kickoff, :3] = np.log([1e-6, 1e-6, 1.0])
    return out


def finish_labels(states, blocks, second_kickoff, n_frames):
    """Decoded states -> submission labels, with the contract checks (pure, unit-tested)."""
    labels = np.array(["H", "A", "D", "X"])[np.where(states > 2, 3, states)].astype("<U1")
    # Only the two halves are graded. X elsewhere is free, and says plainly that halftime was not
    # predicted.
    graded = np.zeros(len(labels), dtype=bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True
    labels[~graded] = "X"
    if second_kickoff is not None:
        labels[blocks[1][1]:second_kickoff] = "D"
    if np.any(labels[graded] == "X"):
        raise RuntimeError("abstention on a graded frame; the decoder must label every one")
    if len(labels) != n_frames:
        raise RuntimeError("prediction length does not match the tactical submission contract")
    return labels


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", required=True,
                        help="folder containing tactical/ and broadcast/ match assets")
    parser.add_argument("--output", required=True, help="target frame,label CSV")
    parser.add_argument("--model", nargs="+",
                        help="temporal model weights, averaged (default: the decoder's members)")
    parser.add_argument("--decoder", default=str(DEFAULT_DECODER), help="decoder settings JSON")
    args = parser.parse_args(argv)
    try:
        labels = predict(args.data_root, args.output, args.model, args.decoder)
    except (IOError, ImportError, KeyError, ValueError) as error:
        parser.error(str(error))
    counts = {label: int(np.sum(labels == label)) for label in ("H", "A", "D", "X")}
    print("wrote {}  H {H}  A {A}  D {D}  X {X}".format(args.output, **counts))


if __name__ == "__main__":
    main()
