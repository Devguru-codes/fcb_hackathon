"""Emission construction and a deterministic three-state Viterbi possession decoder."""

import numpy as np

from .config import DecoderConfig
from .tracking import TrackingFeatures

LABELS = np.asarray(["H", "A", "D"])


def emission_logits(features: TrackingFeatures, tag_evidence, config: DecoderConfig):
    """Create H/A/D log-likelihood proxies from geometry and soft tag evidence."""
    n_frames = len(features.ball_seen)
    logits = np.zeros((n_frames, 3), dtype=np.float64)
    home, away = features.nearest_home_m, features.nearest_away_m
    both_seen = np.isfinite(home) & np.isfinite(away) & features.ball_seen
    margin = np.zeros(n_frames, dtype=np.float64)
    margin[both_seen] = (away[both_seen] - home[both_seen]) / config.distance_scale_m
    logits[:, 0] += margin
    logits[:, 1] -= margin

    nearest = np.fmin(home, away)
    controlled = features.ball_seen & np.isfinite(nearest) & (nearest <= config.close_control_m)
    # A nearby player is affirmative evidence of live possession; no nearby player is evidence
    # for a stoppage only when the ball itself is actually observed.
    logits[controlled, 0] += 0.35
    logits[controlled, 1] += 0.35
    observed_far = features.ball_seen & (~np.isfinite(nearest) | (nearest > 9.0))
    logits[observed_far, 2] += 1.8
    empty_pitch = features.tracked_players <= 2
    logits[empty_pitch, 2] += config.no_player_dead_logit
    logits[~features.ball_seen, 2] += config.missing_ball_dead_logit

    # Do not make a one-frame missing ball choose a team; a near-uniform emission lets Viterbi
    # carry the previous spell. Very long occlusions receive only a small D nudge.
    long_missing = (~features.ball_seen) & (features.ball_age > 75)
    logits[long_missing, 2] += 0.45
    if tag_evidence is not None:
        logits += tag_evidence
    return logits


def viterbi(logits, config: DecoderConfig):
    """Decode the highest-scoring sequence under possession-aware transition costs."""
    transition = np.array([
        [0.0, -config.team_switch_cost, -config.alive_dead_cost],
        [-config.team_switch_cost, 0.0, -config.alive_dead_cost],
        [-config.alive_dead_cost, -config.alive_dead_cost, 0.0],
    ])
    return viterbi_matrix(logits, transition)


def transition_matrix(switch_cost, alive_to_dead, dead_to_alive):
    """Log-transition scores for H/A/D: staying is free, every change pays its cost."""
    return np.array([
        [0.0, -switch_cost, -alive_to_dead],
        [-switch_cost, 0.0, -alive_to_dead],
        [-dead_to_alive, -dead_to_alive, 0.0],
    ])


def viterbi_matrix(logits, transition, change_bonus=None):
    """Highest-scoring H/A/D path for per-frame log scores and a 3x3 log-transition matrix.

    `change_bonus`, optional (n,), is added to every state change INTO frame t: a boundary
    detector's evidence makes a change cheap exactly where one is likely, and dear elsewhere.
    """
    n_frames = len(logits)
    if n_frames == 0:
        return np.empty(0, dtype=np.int8)
    if _kernel is not None:
        bonus = (np.zeros(n_frames) if change_bonus is None
                 else np.ascontiguousarray(change_bonus, dtype=np.float64))
        return _kernel(np.ascontiguousarray(logits, dtype=np.float64),
                       np.ascontiguousarray(transition, dtype=np.float64), bonus)
    off_diagonal = 1.0 - np.eye(3)
    score = np.empty((n_frames, 3), dtype=np.float64)
    back = np.empty((n_frames, 3), dtype=np.int8)
    score[0] = logits[0]
    back[0] = 0
    for frame in range(1, n_frames):
        step = transition
        if change_bonus is not None:
            step = transition + change_bonus[frame] * off_diagonal
        options = score[frame - 1][:, None] + step
        back[frame] = np.argmax(options, axis=0)
        score[frame] = logits[frame] + options[back[frame], np.arange(3)]
    states = np.empty(n_frames, dtype=np.int8)
    states[-1] = int(np.argmax(score[-1]))
    for frame in range(n_frames - 2, -1, -1):
        states[frame] = back[frame + 1, states[frame + 1]]
    return states


def _viterbi_loops(logits, transition, bonus):
    """The same recursion as the numpy path, as explicit loops for numba. Ties pick the lowest
    previous state, exactly like np.argmax, so both paths return identical sequences."""
    n = logits.shape[0]
    score = np.empty((n, 3))
    back = np.zeros((n, 3), dtype=np.int8)
    for s in range(3):
        score[0, s] = logits[0, s]
    for t in range(1, n):
        for s in range(3):
            best, arg = -np.inf, 0
            for p in range(3):
                value = score[t - 1, p] + transition[p, s]
                if p != s:
                    value += bonus[t]
                if value > best:
                    best, arg = value, p
            back[t, s] = arg
            score[t, s] = logits[t, s] + best
    states = np.empty(n, dtype=np.int8)
    last, arg = -np.inf, 0
    for s in range(3):
        if score[n - 1, s] > last:
            last, arg = score[n - 1, s], s
    states[n - 1] = arg
    for t in range(n - 2, -1, -1):
        states[t] = back[t + 1, states[t + 1]]
    return states


try:
    from numba import njit
    _kernel = njit(cache=True)(_viterbi_loops)
except ImportError:   # numba is optional; the numpy path gives the same answer, slower
    _kernel = None


def merge_short_runs(states, config: DecoderConfig):
    """Remove isolated flicker only when it is bracketed by the same state.

    This is intentionally not a generic median filter: a real H→D→A transition must remain
    possible, while a two-frame H blip inside an A spell should never create a new possession.
    """
    result = states.copy()
    changed = True
    while changed:
        changed = False
        starts = np.r_[0, np.flatnonzero(result[1:] != result[:-1]) + 1]
        ends = np.r_[starts[1:], len(result)]
        for index in range(1, len(starts) - 1):
            start, end = starts[index], ends[index]
            state = int(result[start])
            minimum = config.min_dead_run_frames if state == 2 else config.min_alive_run_frames
            if end - start < minimum and result[starts[index - 1]] == result[starts[index + 1]]:
                result[start:end] = result[starts[index - 1]]
                changed = True
                break
    return result


def decode(features, tag_evidence, config):
    states = viterbi(emission_logits(features, tag_evidence, config), config)
    return merge_short_runs(states, config)
