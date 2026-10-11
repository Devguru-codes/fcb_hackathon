"""Video frames, homographies, ball pixel labels and the crop training set.

Frames are read strictly sequentially: seeking in an H.264 file is slow and not always exact,
and a one-frame slip would mislabel every crop. The decoded frame count is checked against the
expected count, as the FAQ asks.
"""

from pathlib import Path

import numpy as np

CROP = 192        # training crop size in pixels (the network is fully convolutional)
CONTEXT = 1       # frames before and after the labelled frame: input is 2*CONTEXT+1 frames


def iter_frames(path, start=0, stop=None):
    """(index, BGR uint8 frame) for every frame in [start, stop), read sequentially."""
    import cv2
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise IOError("could not open video {}".format(path))
    index = 0
    try:
        while stop is None or index < stop:
            ok, frame = capture.read()
            if not ok:
                break
            if index >= start:
                yield index, frame
            index += 1
    finally:
        capture.release()


def frame_count(path):
    """Frames actually decodable, by reading (the container's header count can be wrong)."""
    count = 0
    for count, _ in enumerate(iter_frames(path), start=1):
        pass
    return count


def load_homographies(path):
    """(n, 3, 3) matrices mapping normalised pitch (0-1) to pixels, NaN where invalid."""
    import pandas as pd
    table = pd.read_csv(path)
    columns = ["h{}{}".format(r, c) for r in range(3) for c in range(3)]
    matrices = table[columns].to_numpy(dtype=np.float64).reshape(-1, 3, 3)
    if "valid" in table:
        matrices[~table["valid"].to_numpy().astype(bool)] = np.nan
    return matrices


def project(matrices, xy):
    """Per-frame projection of (n, 2) points through (n, 3, 3) matrices; NaN propagates."""
    homogeneous = np.concatenate([xy, np.ones((len(xy), 1))], axis=1)
    out = np.einsum("nij,nj->ni", matrices, homogeneous)
    with np.errstate(invalid="ignore", divide="ignore"):
        return out[:, :2] / out[:, 2:3]


def unproject(matrices, pixels):
    """Pixels -> normalised pitch coordinates, per frame."""
    inverse = np.full_like(matrices, np.nan)
    ok = np.isfinite(matrices).all(axis=(1, 2))
    inverse[ok] = np.linalg.inv(matrices[ok])
    return project(inverse, pixels)


def ball_pixels(table, matrices):
    """(n, 2) pixel position of the tracked ball (NaN where the tracking has none)."""
    length, width = table.pitch
    normalised = table.ball.astype(np.float64) / np.array([length, width])
    return project(matrices, normalised)


def choose_training_frames(pixels, frame_size, graded, n_positive=12000, n_negative=6000,
                           seed=0):
    """(positive frames, negative frames): positives spread over the match where the ball is
    tracked and well inside the image; negatives are other tracked frames (cropped away from the
    ball later), so a negative crop is known to contain no ball."""
    rng = np.random.default_rng(seed)
    width, height = frame_size
    margin = CROP // 2
    inside = (np.isfinite(pixels[:, 0]) & graded
              & (pixels[:, 0] > margin) & (pixels[:, 0] < width - margin)
              & (pixels[:, 1] > margin) & (pixels[:, 1] < height - margin))
    candidates = np.flatnonzero(inside)
    candidates = candidates[(candidates >= CONTEXT) & (candidates < len(pixels) - CONTEXT)]
    if not len(candidates):
        return np.zeros(0, int), np.zeros(0, int)
    # Never use more than 85% of the candidates, so some tracked frames stay out of training for
    # the audit; scale positives and negatives down together when there are few.
    n_use = min(n_positive + n_negative, int(0.85 * len(candidates)))
    n_pos = int(round(n_use * n_positive / (n_positive + n_negative)))
    picked = rng.permutation(candidates)
    positive = np.sort(picked[:n_pos])
    negative = np.sort(picked[n_pos:n_use])
    return positive, negative


def _crop(stack, cx, cy):
    """CROP x CROP window of a (k, H, W, 3) stack, top-left (cx, cy), zero-padded at edges."""
    k, height, width, _ = stack.shape
    out = np.zeros((k, CROP, CROP, 3), dtype=np.uint8)
    x0, y0 = max(cx, 0), max(cy, 0)
    x1, y1 = min(cx + CROP, width), min(cy + CROP, height)
    if x1 > x0 and y1 > y0:
        out[:, y0 - cy:y1 - cy, x0 - cx:x1 - cx] = stack[:, y0:y1, x0:x1]
    return out


def build_training_set(video, pixels, positive, negative, out_dir, seed=0):
    """One sequential pass: save crops (N, 2*CONTEXT+1, CROP, CROP, 3) uint8 and ball offsets
    (N, 2) within the crop (NaN for negatives) as memmaps in `out_dir`. Returns their paths."""
    rng = np.random.default_rng(seed)
    wanted = {int(f): "pos" for f in positive}
    wanted.update({int(f): "neg" for f in negative})
    order = sorted(wanted)
    n, k = len(order), 2 * CONTEXT + 1
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    crops = np.lib.format.open_memmap(out_dir / "crops.npy", mode="w+", dtype=np.uint8,
                                      shape=(n, k, CROP, CROP, 3))
    targets = np.full((n, 2), np.nan, dtype=np.float32)
    position = {frame: i for i, frame in enumerate(order)}
    buffer = []
    last = order[-1] + CONTEXT if order else -1
    for index, frame in iter_frames(video, 0, last + 1):
        buffer.append(frame)
        if len(buffer) > k:
            buffer.pop(0)
        centre = index - CONTEXT
        if centre not in position or len(buffer) < k:
            continue
        stack = np.stack(buffer)
        height, width = frame.shape[:2]
        bx, by = pixels[centre]
        if wanted[centre] == "pos":
            # Ball anywhere in the crop's middle, so the network does not learn "ball = centre".
            cx = int(bx - rng.integers(CROP // 4, 3 * CROP // 4))
            cy = int(by - rng.integers(CROP // 4, 3 * CROP // 4))
            targets[position[centre]] = (bx - cx, by - cy)
        else:
            # A window at least one crop away from the tracked ball.
            for _ in range(20):
                cx, cy = int(rng.integers(0, width - CROP)), int(rng.integers(0, height - CROP))
                if abs(cx + CROP / 2 - bx) > CROP or abs(cy + CROP / 2 - by) > CROP:
                    break
        crops[position[centre]] = _crop(stack, cx, cy)
    crops.flush()
    np.save(out_dir / "targets.npy", targets)
    np.save(out_dir / "frames.npy", np.asarray(order))
    return out_dir / "crops.npy", out_dir / "targets.npy"
