"""Video ball pipeline, stage by stage. Everything it writes stays under <data-root>/derived/vision
(inside the gitignored data folder): it is derived from the licensed video.

    python -m tempolens.vision.run --data-root data labels    # one pass: crop training set
    python -m tempolens.vision.run --data-root data train     # fit the heatmap network
    python -m tempolens.vision.run --data-root data detect    # candidates where the ATD has no ball
    python -m tempolens.vision.run --data-root data fill      # link candidates, write ball_filled.npz
    python -m tempolens.vision.run --data-root data all

Every stage prints aggregate numbers only (counts, rates), safe to share.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from games import GAME, asset_paths, require
from ..match import competition_table
from . import ballnet
from .io import (CONTEXT, ball_pixels, build_training_set, choose_training_frames, iter_frames,
                 load_homographies)
from .track import candidates_on_pitch, fill_ball

AUDIT_FRAMES = 3000   # tracked frames held out of training, to score detection against the ATD


def _paths(data_root):
    folder = Path(data_root) / "derived" / "vision"
    folder.mkdir(parents=True, exist_ok=True)
    return {"dir": folder, "weights": folder / "ballnet.pt",
            "detections": folder / "detections.npz", "filled": folder / "ball_filled.npz",
            "meta": folder / "labels.json"}


def _context(data_root):
    table = competition_table(GAME, data_root)
    matrices = load_homographies(require(["homography"], GAME, data_root)[0])
    video = asset_paths(GAME, data_root)["video"]
    return table, matrices, video


def stage_labels(data_root):
    import cv2
    paths = _paths(data_root)
    table, matrices, video = _context(data_root)
    capture = cv2.VideoCapture(str(video))
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    capture.release()
    pixels = ball_pixels(table, matrices)
    graded = table.period > 0
    positive, negative = choose_training_frames(pixels, size, graded)
    # Hold some tracked frames out of training entirely, to audit detection against the ATD.
    rng = np.random.default_rng(1)
    tracked = np.setdiff1d(np.flatnonzero(np.isfinite(pixels[:, 0]) & graded),
                           np.concatenate([positive, negative]))
    audit = np.sort(rng.permutation(tracked)[:AUDIT_FRAMES])
    t0 = time.time()
    build_training_set(video, pixels, positive, negative, paths["dir"])
    meta = {"frame_size": size, "positives": int(len(positive)), "negatives": int(len(negative)),
            "audit_frames": audit.tolist(), "seconds": round(time.time() - t0)}
    paths["meta"].write_text(json.dumps(meta), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "audit_frames"}))


def stage_train(data_root):
    paths = _paths(data_root)
    model, report = ballnet.train(paths["dir"] / "crops.npy", paths["dir"] / "targets.npy")
    ballnet.save(model, paths["weights"])
    print(json.dumps(report))


def stage_detect(data_root, device=None):
    """Full-frame candidates on graded frames without a tracked ball, plus the audit frames."""
    import torch
    paths = _paths(data_root)
    table, _, video = _context(data_root)
    meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = ballnet.load(paths["weights"]).to(device)
    wanted = (table.period > 0) & ~np.isfinite(table.ball[:, 0])
    wanted[np.asarray(meta["audit_frames"], dtype=int)] = True
    targets = set(np.flatnonzero(wanted).tolist())
    k = 2 * CONTEXT + 1
    buffer, frames, peaks = [], [], []
    last = max(targets) + CONTEXT + 1 if targets else 0
    t0 = time.time()
    for index, frame in iter_frames(video, 0, last):
        buffer.append(frame)
        if len(buffer) > k:
            buffer.pop(0)
        centre = index - CONTEXT
        if centre in targets and len(buffer) == k:
            for x, y, p in ballnet.peaks(model, np.stack(buffer), device):
                frames.append(centre)
                peaks.append((x, y, p))
    peaks = np.asarray(peaks, dtype=np.float32).reshape(-1, 3)
    np.savez_compressed(paths["detections"], frames=np.asarray(frames, dtype=np.int64),
                        peaks=peaks)
    print(json.dumps({"frames_scanned": len(targets), "candidates": int(len(peaks)),
                      "seconds": round(time.time() - t0)}))


def load_detections(path):
    z = np.load(path)
    out = {}
    for frame, peak in zip(z["frames"].tolist(), z["peaks"].tolist()):
        out.setdefault(frame, []).append(tuple(peak))
    return out


def stage_fill(data_root):
    paths = _paths(data_root)
    table, matrices, _ = _context(data_root)
    meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
    detections = load_detections(paths["detections"])
    candidates = candidates_on_pitch(detections, matrices, table.pitch)

    # Audit: on tracked frames the network never trained on, how often is its best on-pitch
    # candidate within 1.5 m of the tracking's ball?
    audit = np.asarray(meta["audit_frames"], dtype=int)
    hits = found = 0
    for frame in audit:
        if frame in candidates:
            found += 1
            x, y, _ = max(candidates[frame], key=lambda c: c[2])
            hits += np.hypot(x - table.ball[frame, 0], y - table.ball[frame, 1]) <= 1.5
    ball, source = fill_ball(table, candidates)
    np.savez_compressed(paths["filled"], ball=ball, source=source)
    graded = table.period > 0
    gap = graded & (source != 1)
    print(json.dumps({
        "audit_frames": int(len(audit)),
        "audit_candidate_rate": round(found / max(len(audit), 1), 4),
        "audit_within_1p5m_rate": round(hits / max(len(audit), 1), 4),
        "graded_missing_before": round(float(np.mean(table.ball[graded, 0] != table.ball[graded, 0])), 4),
        "graded_missing_after": round(float(np.mean(source[graded] == 0)), 4),
        "gap_frames_filled_share": round(float(np.mean(source[gap] == 2)), 4),
    }))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", required=True)
    parser.add_argument("stage", choices=("labels", "train", "detect", "fill", "all"))
    args = parser.parse_args(argv)
    stages = {"labels": stage_labels, "train": stage_train, "detect": stage_detect,
              "fill": stage_fill}
    for name in (["labels", "train", "detect", "fill"] if args.stage == "all" else [args.stage]):
        print("== {}".format(name), flush=True)
        stages[name](args.data_root)


if __name__ == "__main__":
    main()
