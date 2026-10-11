"""Generate the official More Than a Hack possession-file submission.

    python -m tempolens.predict --data-root data --output evaluation/results/<team>.csv

The default method runs the committed temporal models (every models/tempolens_gru*.pt, an
ensemble trained on open data only) on CPU, averages them, and decodes with the committed
decoder settings (models/decoder.json), so the same data always gives the same file. `--method heuristic` runs the original hand-tuned
TempoLens decoder for comparison.
"""

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from games import GAME, GAMES, half_video_blocks, source_frames

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = sorted((ROOT / "models").glob("tempolens_gru*.pt"))
DEFAULT_DECODER = ROOT / "models" / "decoder.json"


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


def _paths(model_paths):
    if isinstance(model_paths, (str, Path)):
        return [Path(model_paths)]
    return [Path(path) for path in model_paths]


def learned_logp(data_root, game, model_paths, video_ball=None, video_fill="alive"):
    """(competition MatchTable, ensemble output (n, 4)). The slow part: run it once.

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


def learned_states(data_root, game, model_paths, decoder_path, video_ball=None,
                   video_fill="alive"):
    """H/A/D codes from the temporal ensemble, decoded per half. X (3) outside the halves."""
    table, logp = learned_logp(data_root, game, model_paths, video_ball, video_fill)
    states, _ = decode_learned(table, logp, load_decoder(decoder_path), data_root, game)
    report = SimpleNamespace(ball_seen=np.isfinite(table.ball[:, 0]),
                             ball_x=table.ball[:, 0], ball_y=table.ball[:, 1])
    return states, report


def heuristic_states(data_root, game, config_path=None):
    """The original hand-tuned TempoLens decoder, kept as a reference."""
    from .config import load_config
    from .decoder import decode
    from .tags import tag_priors
    from .tracking import extract_features

    config = load_config(config_path)
    features, home_name, away_name = extract_features(game, data_root, config.close_control_m)
    tags = tag_priors(home_name, away_name, game, data_root, config.tag_team_weight,
                      config.tag_restart_weight)
    return decode(features, tags, config), features


def predict(data_root, output, method="learned", model_paths=DEFAULT_MODELS,
            decoder_path=DEFAULT_DECODER, config_path=None, report_dir=None, game=GAME,
            video_ball=None, video_fill="alive"):
    """Run the pipeline and write a validated-shape CSV. Returns the label array."""
    if method == "learned":
        states, report = learned_states(data_root, game, model_paths, decoder_path,
                                        video_ball, video_fill)
    else:
        states, report = heuristic_states(data_root, game, config_path)
    labels = np.array(["H", "A", "D", "X"])[np.where(states > 2, 3, states)].astype("<U1")
    # Only the two halves are graded. X elsewhere is free, and says plainly that halftime was not
    # predicted.
    graded = np.zeros(len(labels), dtype=bool)
    for _, first, last in half_video_blocks(game):
        graded[first:last + 1] = True
    labels[~graded] = "X"
    if np.any(labels[graded] == "X"):
        raise RuntimeError("abstention on a graded frame; the decoder must label every one")
    if len(labels) != source_frames("tactical", game):
        raise RuntimeError("prediction length does not match the tactical submission contract")
    write_submission(output, labels)
    if report_dir:
        from .report import write_report
        write_report(report_dir, np.where(states > 2, 2, states), report, GAMES[game]["fps"])
    return labels


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", required=True,
                        help="folder containing tactical/ and broadcast/ match assets")
    parser.add_argument("--output", required=True, help="target frame,label CSV")
    parser.add_argument("--method", default="learned", choices=("learned", "heuristic"))
    parser.add_argument("--model", nargs="+", default=[str(p) for p in DEFAULT_MODELS],
                        help="temporal model weights; several are averaged")
    parser.add_argument("--decoder", default=str(DEFAULT_DECODER), help="decoder settings JSON")
    parser.add_argument("--config", help="heuristic method only: DecoderConfig JSON")
    parser.add_argument("--report-dir", help="optional local transition-report directory")
    parser.add_argument("--video-ball", help="ball_filled.npz from tempolens.vision.run "
                                             "(default: tracking ball only)")
    parser.add_argument("--video-fill", default="alive", choices=("alive", "all"),
                        help="fill video detections only where play looks live, or everywhere")
    args = parser.parse_args(argv)
    try:
        labels = predict(args.data_root, args.output, args.method, args.model, args.decoder,
                         args.config, args.report_dir, video_ball=args.video_ball,
                         video_fill=args.video_fill)
    except (IOError, ImportError, ValueError) as error:
        parser.error(str(error))
    counts = {label: int(np.sum(labels == label)) for label in ("H", "A", "D", "X")}
    print("wrote {}  H {H}  A {A}  D {D}  X {X}".format(args.output, **counts))


if __name__ == "__main__":
    main()
