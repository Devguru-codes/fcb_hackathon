"""Rehearse the video ball pipeline on a synthetic match video, before the real one exists.

    python -m tempolens.vision.rehearse --out <empty folder> [--seconds 240]

Renders a perspective "tactical camera" view of an open DFL match segment -- green pitch, players
as team-coloured discs, the true ball as a small white disc unless a player covers it -- next to
a fake ATD whose ball has the simulator's gaps. Then runs labels, train, detect and fill, and
scores the filled frames against the true ball. It proves the plumbing (frame alignment,
projections, training, linking) and gives a best-case fill rate; the real video will be harder.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from ..opendata import load_open_match
from ..simulate import simulate_atd

N_FRAMES = 199319
FIRST_GRADED = 11185
SIZE = (960, 540)
SLOTS = 46
CORNERS = [(70, 140), (890, 140), (950, 520), (10, 520)]   # pitch corners in pixels


def _homography():
    import cv2
    unit = np.float32([(0, 0), (1, 0), (1, 1), (0, 1)])
    return cv2.getPerspectiveTransform(unit, np.float32(CORNERS)).astype(np.float64)


def write_bundle(out, seconds, match="dfl_J03WMX", seed=0):
    """Fake tactical bundle in the real formats; returns (true ball (n, 2) metres, rendered mask)."""
    import cv2
    out = Path(out)
    tactical = out / "tactical"
    tactical.mkdir(parents=True)
    clean = load_open_match(match)
    noisy = simulate_atd(clean, seed=seed)
    n_render = seconds * 25
    src = slice(clean.runs()[0][0], clean.runs()[0][0] + n_render)
    dst = slice(FIRST_GRADED, FIRST_GRADED + n_render)
    length, width = clean.pitch

    players = np.full((N_FRAMES, SLOTS, 2), np.nan, dtype=np.float32)
    k = min(SLOTS, clean.players.shape[1])
    players[dst, :k] = noisy.players[src, :k]
    ball_atd = np.full((N_FRAMES, 2), np.nan, dtype=np.float32)
    ball_atd[dst] = noisy.ball[src]
    ball_true = np.full((N_FRAMES, 2), np.nan, dtype=np.float32)
    ball_true[dst] = clean.ball[src]
    teams = np.zeros(SLOTS, dtype=np.int8)
    teams[:k] = clean.slot_team[:k]
    teams[k:] = np.arange(SLOTS - k) % 2

    meta = "".join('<Player id="P{0}" teamId="{1}"><Name>Track {0}</Name></Player>'.format(
        i + 1, "T1" if teams[i] == 0 else "T2") for i in range(SLOTS))
    (tactical / "BAR-ATH_tactical_atd.xml").write_text(
        '<main><Teams><Team id="T1"><Name>Home</Name></Team><Team id="T2"><Name>Away</Name>'
        "</Team></Teams><Players>{}</Players></main>".format(meta), encoding="utf-8")
    nan = "NaN,NaN,NaN"
    with open(tactical / "BAR-ATH_tactical_atd.txt", "w", encoding="utf-8") as handle:
        for f in range(N_FRAMES):
            groups = [nan if not np.isfinite(p[0]) else "{:.5f},{:.5f},0".format(
                p[0] / length, p[1] / width) for p in players[f]]
            b = ball_atd[f]
            ball = nan if not np.isfinite(b[0]) else "{:.5f},{:.5f},0".format(b[0] / length,
                                                                            b[1] / width)
            handle.write("{}:{}:{}\n".format(f + 1, ";".join(groups), ball))

    matrix = _homography()
    row = ",".join("{:.8f}".format(v) for v in matrix.ravel())
    with open(tactical / "BAR-ATH_tactical_homography.csv", "w", encoding="utf-8") as handle:
        handle.write("frame,h00,h01,h02,h10,h11,h12,h20,h21,h22,reprojection_error,valid\n")
        for f in range(N_FRAMES):
            handle.write("{},{},0.2,1\n".format(f, row))
    (tactical / "BAR-ATH_tactical_smart_tagging.xml").write_text(
        "<file><ALL_INSTANCES></ALL_INSTANCES></file>", encoding="utf-8")

    def pix(xy):
        h = matrix @ np.array([xy[0] / length, xy[1] / width, 1.0])
        return int(round(h[0] / h[2])), int(round(h[1] / h[2]))

    rng = np.random.default_rng(seed)
    writer = cv2.VideoWriter(str(tactical / "BAR-ATH_tactical.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), 25, SIZE)
    base = np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8)
    base[:] = (40, 120, 40)
    cv2.polylines(base, [np.int32(CORNERS)], True, (230, 230, 230), 1)
    last = FIRST_GRADED + n_render
    for f in range(last):
        frame = base.copy()
        if dst.start <= f < dst.stop:
            frame = cv2.add(frame, rng.integers(0, 12, frame.shape, dtype=np.uint8))
            visible = clean.players[f - FIRST_GRADED + src.start]
            for p, team in zip(visible, clean.slot_team):
                if np.isfinite(p[0]):
                    cv2.circle(frame, pix(p), 6, (40, 40, 220) if team == 0 else (220, 200, 40), -1)
            b = ball_true[f]
            covered = np.nanmin(np.hypot(*(visible - b).T)) < 0.7 if np.isfinite(b[0]) else True
            if np.isfinite(b[0]) and not covered:
                cv2.circle(frame, pix(b), 2, (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    rendered = np.zeros(N_FRAMES, dtype=bool)
    rendered[dst] = True
    return ball_true, rendered


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="an empty folder for the fake bundle")
    parser.add_argument("--seconds", type=int, default=240)
    args = parser.parse_args(argv)
    from . import run
    ball_true, rendered = write_bundle(args.out, args.seconds)
    for stage in ("labels", "train", "detect", "fill"):
        print("== {}".format(stage), flush=True)
        getattr(run, "stage_" + stage)(args.out)
    filled = np.load(Path(args.out) / "derived" / "vision" / "ball_filled.npz")
    ball, source = filled["ball"], filled["source"]
    gap = rendered & (source != 1)
    video = gap & (source == 2)
    error = np.hypot(*(ball[video] - ball_true[video]).T)
    has_truth = gap & np.isfinite(ball_true[:, 0])
    print(json.dumps({
        "rendered_frames": int(rendered.sum()),
        "atd_gap_frames": int(gap.sum()),
        "gap_frames_filled": int(video.sum()),
        "fill_share_of_gap_frames_with_a_true_ball": round(
            float(video.sum() / max(has_truth.sum(), 1)), 4),
        "filled_within_1p5m": round(float(np.mean(error <= 1.5)) if len(error) else 0.0, 4),
        "filled_median_error_m": round(float(np.median(error)) if len(error) else float("nan"), 3),
    }, indent=2))


if __name__ == "__main__":
    main()
