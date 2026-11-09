"""Deterministic three-state (H/A/D) Viterbi possession decoder."""

import numpy as np


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
    off_diagonal = ~np.eye(3, dtype=bool)
    score = np.empty((n_frames, 3), dtype=np.float64)
    back = np.empty((n_frames, 3), dtype=np.int8)
    score[0] = logits[0]
    back[0] = 0
    for frame in range(1, n_frames):
        # numba's order of additions (score + transition, then the bonus), so exact ties agree
        options = score[frame - 1][:, None] + transition
        if change_bonus is not None:
            options[off_diagonal] += change_bonus[frame]
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
