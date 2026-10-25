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


def learned_logp(data_root, game, model_paths, video_ball=None, video_fill="alive",
                 tta_alpha=0.0):
    """(competition MatchTable, ensemble output (n, 4)). The slow part: run it once.

    `tta_alpha` > 0: test-time adaptation -- each member's feature scaling is moved that far
    toward this match's own medians and spreads (temporal.adapt_normaliser), unlabelled.

    `video_ball`: optional ball_filled.npz from tempolens.vision.run. Its detections fill the
    tracking's ball gaps -- everywhere ("all"), or ("alive") only where a first pass of the
    ensemble on the raw tracking says the ball is in play, so the model's "ball missing during
    a stoppage" cue is left intact. "alive" costs a second ensemble pass.
    """
    from dataclasses import replace
    from . import temporal
    from .match import D, competition_table

    table = competition_table(game, data_root)
    paths = _paths(model_paths)
    if not paths:
        raise IOError("no model weights given and none found in models/")
    members = [temporal.load(path) for path in paths]
    if tta_alpha:
        members = [(model, temporal.adapt_normaliser(normaliser, table, tta_alpha))
                   for model, normaliser in members]
    if video_ball is None:
        return table, temporal.predict_ensemble(members, table)
    filled = np.load(video_ball)
    from_video = filled["source"] == 2
    if len(from_video) != table.n_frames:
        raise ValueError("video ball file has {} frames, the match {}".format(
            len(from_video), table.n_frames))
    if video_fill == "alive":
        first = temporal.predict_ensemble(members, table)
        from_video &= np.exp(first[:, D]) < 0.5
    elif video_fill != "all":
        raise ValueError("video_fill must be 'alive' or 'all', not {!r}".format(video_fill))
    ball = table.ball.copy()
    ball[from_video] = filled["ball"][from_video]
    table = replace(table, ball=ball)
    return table, temporal.predict_ensemble(members, table)


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
        from .tagprior import read_instances, self_calibrated
        path = require(["smart_tagging"], game, data_root)[0]
        instances = read_instances(path, table.meta["home"], table.meta["away"],
                                   GAMES[game]["fps"])
        logp, tag_report = self_calibrated(logp, instances, table.period > 0)
    elif mode != "off":
        raise ValueError("tag_prior must be 'auto' or 'off', not {!r}".format(mode))
    transition = transition_matrix(settings["switch"], settings["alive_to_dead"],
                                   settings["dead_to_alive"])
    return decode_table(table, logp, transition, settings.get("change_weight", 0.0)), tag_report


def predict(data_root, output, model_paths=None, decoder_path=DEFAULT_DECODER, game=GAME,
            video_ball=None, video_fill="alive"):
    """Run the pipeline and write a validated-shape CSV. Returns the label array.

    `model_paths` defaults to the members listed in the decoder JSON.
    """
    model_paths = model_paths or default_models(decoder_path)
    settings = load_decoder(decoder_path)
    table, logp = learned_logp(data_root, game, model_paths, video_ball, video_fill,
                               settings.get("tta_alpha", 0.0))
    states, _ = decode_learned(table, logp, settings, data_root, game)
    labels = np.array(["H", "A", "D", "X"])[np.where(states > 2, 3, states)].astype("<U1")
    # Only the two halves are graded. X elsewhere is free, and says plainly that halftime was not
    # predicted.
    graded = np.zeros(len(labels), dtype=bool)
    for _, first, last in half_video_blocks(game):
        graded[first:last + 1] = True
    labels[~graded] = "X"
    if game in SECOND_HALF_KICKOFF:
        labels[half_video_blocks(game)[1][1]:SECOND_HALF_KICKOFF[game]] = "D"
    if np.any(labels[graded] == "X"):
        raise RuntimeError("abstention on a graded frame; the decoder must label every one")
    if len(labels) != source_frames("tactical", game):
        raise RuntimeError("prediction length does not match the tactical submission contract")
    write_submission(output, labels)
    return labels


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", required=True,
                        help="folder containing tactical/ and broadcast/ match assets")
    parser.add_argument("--output", required=True, help="target frame,label CSV")
    parser.add_argument("--model", nargs="+",
                        help="temporal model weights, averaged (default: the decoder's members)")
    parser.add_argument("--decoder", default=str(DEFAULT_DECODER), help="decoder settings JSON")
    parser.add_argument("--video-ball", help="ball_filled.npz from tempolens.vision.run "
                                             "(default: tracking ball only)")
    parser.add_argument("--video-fill", default="alive", choices=("alive", "all"),
                        help="fill video detections only where play looks live, or everywhere")
    args = parser.parse_args(argv)
    try:
        labels = predict(args.data_root, args.output, args.model, args.decoder,
                         video_ball=args.video_ball, video_fill=args.video_fill)
    except (IOError, ImportError, KeyError, ValueError) as error:
        parser.error(str(error))
    counts = {label: int(np.sum(labels == label)) for label in ("H", "A", "D", "X")}
    print("wrote {}  H {H}  A {A}  D {D}  X {X}".format(args.output, **counts))


if __name__ == "__main__":
    main()
