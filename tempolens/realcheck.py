"""Real-match checks of candidate submissions. Prints aggregates only, never per-frame data.

    python -m tempolens.realcheck public A.csv B.csv ...              # public window + dead ball
    python -m tempolens.realcheck setpiece --data-root data X.csv     # dead ball around set pieces

`public`: official metrics on the public five minutes, dead-ball recall and precision there, and
full-match dead-ball share and possession count. `setpiece`: share of frames labelled dead in the
seconds around each Smart Tagging SET PIECES tag, by restart type.
"""

import argparse
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("public", "setpiece"))
    parser.add_argument("files", nargs="+")
    parser.add_argument("--data-root")
    args = parser.parse_args(argv)
    if args.mode == "public":
        public(args.files)
    else:
        for path in args.files:
            setpiece(args.data_root, path)


if __name__ == "__main__":
    main()
