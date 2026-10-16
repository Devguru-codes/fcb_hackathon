"""Link per-frame ball candidates into a physically possible track, and fill the ATD's gaps.

Single-frame detections confuse the ball with boots, heads and line markings. Inside each gap of
the tracking's ball, a Viterbi pass over {no ball, candidate 1..k} per frame keeps only paths
where the ball moves at most MAX_SPEED, stays on or near the pitch, and joins up with the
tracked ball at the gap's edges. Isolated false peaks cannot form such a path; a real ball can.
"""

import numpy as np

from .io import unproject

MAX_SPEED = 40.0       # m/s
SLACK = 1.0            # m allowed beyond MAX_SPEED * dt (detection jitter)
OFF_PITCH = 2.0        # m outside the lines a candidate may still be
START_COST = 3.0       # log-odds paid to start a detected stretch
END_COST = 3.0
STEP_COST = 0.05       # per metre moved between consecutive detections
FILL_MIN = 0.95        # minimum candidate probability for a frame to be filled
MIN_RUN = 5            # minimum consecutive video-filled frames
# On the synthetic rehearsal, 0.95 gave 47% coverage of gap frames at 100% within
# 1.5 m; 0.85 gave 91% coverage at only 56%. A wrong ball is worse than none.


def candidates_on_pitch(detections, matrices, pitch):
    """{frame: [(x_m, y_m, prob)]}: pixel peaks unprojected to metres, off-pitch ones dropped."""
    length, width = pitch
    out = {}
    for frame, peaks in detections.items():
        if not peaks:
            continue
        pixels = np.array([(x, y) for x, y, _ in peaks], dtype=np.float64)
        normalised = unproject(np.repeat(matrices[frame][None], len(pixels), axis=0), pixels)
        metres = normalised * np.array([length, width])
        kept = [(mx, my, p) for (mx, my), (_, _, p) in zip(metres, peaks)
                if np.isfinite(mx) and -OFF_PITCH <= mx <= length + OFF_PITCH
                and -OFF_PITCH <= my <= width + OFF_PITCH]
        if kept:
            out[frame] = kept
    return out


def _viterbi_gap(cands, before, after, fps):
    """Best path through one gap: list (per frame) of chosen candidate index or -1 (none)."""
    step_limit = MAX_SPEED / fps + SLACK
    states = [[None] + c for c in cands]            # index 0 = no ball
    score = [np.full(len(s), -np.inf) for s in states]
    back = [np.zeros(len(s), dtype=int) for s in states]
    for t, frame_states in enumerate(states):
        for j, cand in enumerate(frame_states):
            emit = 0.0 if cand is None else float(np.log(cand[2] / max(1e-6, 1 - cand[2])))
            if t == 0:
                if cand is None:
                    score[t][j] = 0.0
                elif before is not None and np.hypot(cand[0] - before[0],
                                                     cand[1] - before[1]) <= step_limit:
                    score[t][j] = emit                   # continues the tracked ball: no start
                else:
                    score[t][j] = emit - START_COST
                continue
            best, arg = -np.inf, 0
            for i, prev in enumerate(states[t - 1]):
                if not np.isfinite(score[t - 1][i]):
                    continue
                if cand is None:
                    value = score[t - 1][i] - (0.0 if prev is None else END_COST)
                elif prev is None:
                    value = score[t - 1][i] - START_COST
                else:
                    moved = np.hypot(cand[0] - prev[0], cand[1] - prev[1])
                    if moved > step_limit:
                        continue
                    value = score[t - 1][i] - STEP_COST * moved
                if value > best:
                    best, arg = value, i
            score[t][j], back[t][j] = best + emit, arg
    # Ending on a detection that joins the tracked ball after the gap costs nothing extra.
    final = score[-1].copy()
    for j, cand in enumerate(states[-1]):
        if cand is not None:
            joins = after is not None and np.hypot(cand[0] - after[0],
                                                   cand[1] - after[1]) <= step_limit
            final[j] -= 0.0 if joins else END_COST
    path = [int(np.argmax(final))]
    for t in range(len(states) - 1, 0, -1):
        path.append(int(back[t][path[-1]]))
    path.reverse()
    return [j - 1 for j in path]


def fill_ball(table, candidates, fill_mask=None, fps=25, fill_min=None, min_run=MIN_RUN):
    """(ball (n, 2) metres, source (n,) int8: 0 none / 1 tracking / 2 video).

    Frames already tracked keep the tracking's ball. Inside each gap (within a period), the
    linked path's detections fill frames where `fill_mask` allows it (default: everywhere),
    whose probability is at least `fill_min`, and that belong to a run of at least `min_run`
    consecutive filled frames -- a wrong ball is worse than no ball, so isolated fills go.
    """
    fill_min = FILL_MIN if fill_min is None else fill_min
    ball = table.ball.astype(np.float32).copy()
    source = np.where(np.isfinite(ball[:, 0]), 1, 0).astype(np.int8)
    allow = np.ones(table.n_frames, dtype=bool) if fill_mask is None else fill_mask
    for start, stop in table.runs():
        missing = ~np.isfinite(ball[start:stop, 0])
        edges = np.diff(np.concatenate(([0], missing.astype(np.int8), [0])))
        for g0, g1 in zip(np.flatnonzero(edges == 1) + start, np.flatnonzero(edges == -1) + start):
            cands = [candidates.get(f, []) for f in range(g0, g1)]
            if not any(cands):
                continue
            before = ball[g0 - 1] if g0 > start else None
            after = ball[g1] if g1 < stop else None
            path = _viterbi_gap(cands, before, after, fps)
            keep = np.array([j >= 0 and allow[g0 + k] and cands[k][j][2] >= fill_min
                             for k, j in enumerate(path)], dtype=bool)
            if min_run > 1:
                edges = np.diff(np.concatenate(([0], keep.astype(np.int8), [0])))
                for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                    if b - a < min_run:
                        keep[a:b] = False
            for offset in np.flatnonzero(keep):
                ball[g0 + offset] = cands[offset][path[offset]][:2]
                source[g0 + offset] = 2
    return ball, source
