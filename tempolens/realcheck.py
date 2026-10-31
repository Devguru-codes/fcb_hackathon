"""Real-match checks of candidate submissions. Prints aggregates only, never per-frame data.

    python -m tempolens.realcheck public A.csv B.csv ...              # public window + dead ball
    python -m tempolens.realcheck setpiece --data-root data X.csv     # dead ball around set pieces
    python -m tempolens.realcheck top --data-root data A.csv B.csv --names a b --suffix _a_b

`public`: official metrics on the public five minutes, dead-ball recall and precision there, and
full-match dead-ball share and possession count. `setpiece`: share of frames labelled dead in the
seconds around each Smart Tagging SET PIECES tag, by restart type. `top`: the longest
disagreements between two files as a review sheet with reel times, matching the reel that
`tempolens.review` renders for the same pair (written under <data-root>/derived/review/).
"""

import argparse
import csv
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
import possession_metrics as pm  # noqa: E402

PUBLIC_GT = ROOT / "evaluation" / "gt" / "BAR-ATH_possession_gt_public.csv"
WINDOWS = ((-10, -5), (-5, 0), (0, 5), (5, 10))     # seconds around a set-piece tag's start


def codes(path):
    return pm.label_codes(pm.read_label_csv(path))


def public(paths):
    gt = codes(PUBLIC_GT)
    window = gt < 3
    for path in paths:
        pred = codes(path)
        r = pm.evaluate(gt, pred)
        hit = np.sum((gt == 2) & (pred == 2) & window)
        print("{:<30} spellF1 {:.4f} macroF1 {:.4f} spells {}/{} | D recall {:.3f} precision {:.3f}"
              " | full-match D {:.3f}, possessions {}".format(
                  Path(path).name, r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"],
                  r["n_gt_spells"], hit / np.sum((gt == 2) & window),
                  hit / max(np.sum((pred == 2) & window), 1), np.mean(pred[pred < 3] == 2),
                  len(pm.possession_spells(pred, pm.valid_runs(pred)))))


def setpiece(data_root, path):
    from games import GAME, require
    pred = codes(path)
    graded = pred < 3
    tags = []
    for inst in ET.parse(require(["smart_tagging"], GAME, data_root)[0]).getroot().iter("instance"):
        if (inst.findtext("code") or "").strip().upper() != "SET PIECES":
            continue
        start = int(round(float(inst.findtext("start")) * 25))
        kind = {l.findtext("group"): l.findtext("text") for l in inst.findall("label")}.get("Type", "?")
        if graded[min(start, len(graded) - 1)]:
            tags.append((kind, start))
    print("{}: share of frames labelled D around set-piece tags".format(Path(path).name))
    print("  {:<26}".format("type") + "".join("{:>10}".format("{}..{}s".format(a, b)) for a, b in WINDOWS))
    for kind in sorted({k for k, _ in tags}):
        starts = [s for k, s in tags if k == kind]
        cells = []
        for a, b in WINDOWS:
            frames = np.concatenate([np.arange(s + 25 * a, s + 25 * b) for s in starts])
            frames = frames[(frames >= 0) & (frames < len(pred))]
            cells.append("{:>10.2f}".format((pred[frames[graded[frames]]] == 2).mean()))
        print("  {:<26}".format("{} ({})".format(kind, len(starts))) + "".join(cells))


def top(data_root, a_path, b_path, names, suffix, n=20):
    from .review import FPS, NAMES, clock, disagreements, majority
    a, b = codes(a_path), codes(b_path)
    cores, spans = disagreements(a, b, 25, 125)

    def reel_second(frame):
        done = 0
        for lo, hi in spans:
            if lo <= frame < hi:
                return (done + frame - lo) // FPS
            done += hi - lo

    longest = sorted(enumerate(cores, 1), key=lambda kc: -(kc[1][1] - kc[1][0]))[:n]
    out = Path(data_root) / "derived" / "review" / "review_top{}{}.csv".format(n, suffix)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle)
        w.writerow(["n (reel counter)", "reel_time (mm:ss)", "match_video_time", "first_frame",
                    "last_frame", "seconds", names[0], names[1],
                    "verdict ({} / {} / both wrong / unclear)".format(*names), "notes"])
        for k, (s, e) in sorted(longest):
            r = reel_second(s)
            w.writerow([k, "{}:{:02d}".format(r // 60, r % 60), clock(s), s, e - 1,
                        round((e - s) / FPS, 1), NAMES[majority(a[s:e])], NAMES[majority(b[s:e])],
                        "", ""])
    print("{} disagreements >= 1 s ({} s in all); sheet of the {} longest: {}".format(
        len(cores), round(sum(e - s for s, e in cores) / FPS), len(longest), out.name))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("public", "setpiece", "top"))
    parser.add_argument("files", nargs="+")
    parser.add_argument("--data-root")
    parser.add_argument("--names", nargs=2, default=("A", "B"))
    parser.add_argument("--suffix", default="")
    args = parser.parse_args(argv)
    if args.mode == "public":
        public(args.files)
    elif args.mode == "setpiece":
        for path in args.files:
            setpiece(args.data_root, path)
    else:
        top(args.data_root, args.files[0], args.files[1], args.names, args.suffix)


if __name__ == "__main__":
    main()
