"""Render a review reel of the moments where two submissions disagree, for people to judge.

    python -m tempolens.review --data-root data --a v4.csv --b v5.csv

Reads the tactical video once, in order, and writes <data-root>/derived/review/reel.mp4
(960x540, 25 fps) with both submissions' labels drawn on every frame (and the public ground
truth inside its five minutes), plus review_sheet.csv: one row per disagreement with empty
columns for the reviewer's verdict. Everything stays under the data folder, since it is derived
from the licensed video; nothing here is sent anywhere. The submitted labels never come from this
review: it only validates them, which RULES.md allows ("hand-labelled data ... to validate").
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
import possession_metrics as pm  # noqa: E402

FPS = 25
SIZE = (960, 540)
COLOURS = {0: (200, 120, 0), 1: (40, 40, 220), 2: (128, 128, 128), 3: (0, 0, 0)}   # BGR
NAMES = {0: "H Barcelona", 1: "A Athletic", 2: "D dead ball", 3: "X"}


def disagreements(a, b, min_frames=25, context=125):
    """(core stretches, reel spans): graded runs where a != b lasting >= min_frames, and those
    runs padded by `context` frames each side, overlapping spans merged. Half-open intervals."""
    differ = (a != b) & (a < 3) & (b < 3)
    edges = np.diff(np.r_[0, differ.astype(np.int8), 0])
    cores = [(int(s), int(e)) for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
             if e - s >= min_frames]
    spans = []
    for s, e in cores:
        lo, hi = max(0, s - context), min(len(a), e + context)
        if spans and lo <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], hi))
        else:
            spans.append((lo, hi))
    return cores, spans


def clock(frame):
    seconds = frame // FPS
    return "{}:{:02d}:{:02d}".format(seconds // 3600, seconds // 60 % 60, seconds % 60)


def majority(codes):
    return int(np.bincount(codes, minlength=4).argmax())


def write_sheet(path, cores, a, b, names):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        out = csv.writer(handle)
        out.writerow(["n", "video_time", "first_frame", "last_frame", "seconds",
                      names[0], names[1], "verdict (A / B / both wrong / unclear)", "notes"])
        for k, (s, e) in enumerate(cores, start=1):
            out.writerow([k, clock(s), s, e - 1, round((e - s) / FPS, 1),
                          NAMES[majority(a[s:e])], NAMES[majority(b[s:e])], "", ""])


def render(video, out_path, a, b, truth, cores, spans, names):
    """Write the reel: every frame of every span, with three label bars and a counter."""
    import cv2
    needed = np.zeros(len(a), dtype=bool)
    for lo, hi in spans:
        needed[lo:hi] = True
    core_of = np.zeros(len(a), dtype=np.int32)
    for k, (s, e) in enumerate(cores, start=1):
        core_of[s:e] = k
    last = max(hi for _, hi in spans) if spans else 0
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise IOError("could not open video {}".format(video))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE)
    written = 0
    try:
        for frame_index in range(last):
            if not needed[frame_index]:
                if not capture.grab():          # decode without colour conversion: faster
                    raise IOError("video ended at frame {}".format(frame_index))
                continue
            ok, image = capture.read()
            if not ok:
                raise IOError("video ended at frame {}".format(frame_index))
            image = cv2.resize(image, SIZE)
            rows = [(names[0], a[frame_index]), (names[1], b[frame_index])]
            if truth[frame_index] < 3:
                rows.append(("truth", truth[frame_index]))
            for r, (label, code) in enumerate(rows):
                y = 8 + 30 * r
                cv2.rectangle(image, (8, y), (330, y + 26), COLOURS[int(code)], -1)
                cv2.putText(image, "{}: {}".format(label, NAMES[int(code)]), (14, y + 19),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            k = core_of[frame_index]
            caption = "{}  frame {}  {}".format("#{}/{}".format(k, len(cores)) if k else "context",
                                                frame_index, clock(frame_index))
            cv2.putText(image, caption, (8, SIZE[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2, cv2.LINE_AA)
            if k:                               # inside the disagreement itself: yellow frame
                cv2.rectangle(image, (0, 0), (SIZE[0] - 1, SIZE[1] - 1), (0, 220, 255), 4)
            writer.write(image)
            written += 1
    finally:
        capture.release()
        writer.release()
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--a", required=True, help="first submission CSV")
    parser.add_argument("--b", required=True, help="second submission CSV")
    parser.add_argument("--names", nargs=2, default=("A", "B"))
    parser.add_argument("--min-frames", type=int, default=25, help="shortest disagreement shown")
    parser.add_argument("--context", type=int, default=125, help="frames shown before and after")
    args = parser.parse_args(argv)
    from games import GAME, asset_paths
    a = pm.label_codes(pm.read_label_csv(args.a))
    b = pm.label_codes(pm.read_label_csv(args.b))
    truth = pm.label_codes(pm.read_label_csv(ROOT / "evaluation" / "gt"
                                            / "BAR-ATH_possession_gt_public.csv"))
    cores, spans = disagreements(a, b, args.min_frames, args.context)
    folder = Path(args.data_root) / "derived" / "review"
    folder.mkdir(parents=True, exist_ok=True)
    write_sheet(folder / "review_sheet.csv", cores, a, b, args.names)
    print("{} disagreements >= {} frames; reel {:.1f} min".format(
        len(cores), args.min_frames, sum(hi - lo for lo, hi in spans) / FPS / 60), flush=True)
    written = render(asset_paths(GAME, args.data_root)["video"], folder / "reel.mp4", a, b, truth,
                     cores, spans, args.names)
    print("wrote {} frames to {}".format(written, folder / "reel.mp4"), flush=True)


if __name__ == "__main__":
    main()
