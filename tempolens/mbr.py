"""Decoding aimed squarely at spellF1: a minimum-duration Viterbi and minimum-Bayes-risk selection.

- `viterbi_min_duration`: every H, A or D run must last at least m frames, enforced by expanding
  each label into a chain of m counter states (a semi-Markov decoder with a hard minimum).
- `sample_paths`: segmentations drawn from the chain-CRF posterior defined by the model's
  per-frame log-probabilities and the decoder's transition costs (forward filtering, backward
  sampling).
- `mbr_decode`: among candidate decodings, pick the one with the highest *expected* spellF1
  against those samples. The switch cost then adapts to how uncertain each match is, instead of
  one value tuned on other matches.
- `fast_spell_f1`: the official spellF1 (possession_metrics.spell_scores), same greedy
  IoU-descending one-to-one matching and tie-break, but only comparing spells that overlap in
  time -- fast enough to call thousands of times.
"""

import numpy as np

from .decoder import viterbi_matrix
from .match import A, D, H, X

IOUS = (0.3, 0.4, 0.5, 0.6, 0.7)


# -- spellF1, fast -----------------------------------------------------------------------------

def _spells(codes, runs):
    """(starts, stops, labels) of maximal H/A runs inside each run, in time order."""
    starts, stops, labels = [], [], []
    for start, stop in runs:
        w = codes[start:stop]
        if not len(w):
            continue
        bounds = np.r_[0, np.flatnonzero(w[1:] != w[:-1]) + 1, len(w)]
        lab = w[bounds[:-1]]
        keep = (lab == H) | (lab == A)
        starts.append(bounds[:-1][keep] + start)
        stops.append(bounds[1:][keep] + start)
        labels.append(lab[keep])
    if not starts:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0, int)
    return np.concatenate(starts), np.concatenate(stops), np.concatenate(labels)


def fast_spell_f1(gt_codes, pred_codes, runs, ious=IOUS):
    """Mean spellF1 over `ious`, identical to possession_metrics.spell_scores()["mean_f1"]."""
    gs, ge, gl = _spells(gt_codes, runs)
    ps, pe, pl = _spells(pred_codes, runs)
    if not len(gs) or not len(ps):
        return 0.0
    lo = np.searchsorted(pe, gs, side="right")     # first pred ending after the gt start
    hi = np.searchsorted(ps, ge, side="left")      # first pred starting at/after the gt stop
    gi, pj = [], []
    for i in range(len(gs)):
        for j in range(lo[i], hi[i]):
            if pl[j] == gl[i]:
                gi.append(i)
                pj.append(j)
    if not gi:
        return 0.0
    gi, pj = np.asarray(gi), np.asarray(pj)
    inter = np.minimum(ge[gi], pe[pj]) - np.maximum(gs[gi], ps[pj])
    union = np.maximum(ge[gi], pe[pj]) - np.minimum(gs[gi], ps[pj])
    iou = inter / union.astype(np.float64)
    order = np.lexsort((pj, gi, -iou))                # official tie-break: (-iou, gt, pred)
    iou, gi, pj = iou[order], gi[order], pj[order]
    f1s = []
    for threshold in ious:
        used_g = np.zeros(len(gs), dtype=bool)
        used_p = np.zeros(len(ps), dtype=bool)
        tp = 0
        for k in range(len(iou)):
            if iou[k] < threshold:
                break
            if used_g[gi[k]] or used_p[pj[k]]:
                continue
            used_g[gi[k]] = used_p[pj[k]] = True
            tp += 1
        precision, recall = tp / len(ps), tp / len(gs)
        f1s.append(2 * precision * recall / (precision + recall) if tp else 0.0)
    return float(np.mean(f1s))


# -- minimum-duration Viterbi ------------------------------------------------------------------

def _min_duration_loops(logits, transition, lengths):
    n = logits.shape[0]
    offsets = np.zeros(4, dtype=np.int64)
    for s in range(3):
        offsets[s + 1] = offsets[s] + lengths[s]
    m = offsets[3]
    label = np.zeros(m, dtype=np.int64)
    for s in range(3):
        for k in range(lengths[s]):
            label[offsets[s] + k] = s
    score = np.full((n, m), -np.inf)
    back = np.zeros((n, m), dtype=np.int32)
    for s in range(3):
        score[0, offsets[s]] = logits[0, s]
    for t in range(1, n):
        for s in range(3):
            first, last = offsets[s], offsets[s + 1] - 1
            # entering s: from the last (matured) counter of another label
            best, arg = -np.inf, 0
            for r in range(3):
                if r == s:
                    continue
                v = score[t - 1, offsets[r + 1] - 1] + transition[r, s]
                if v > best:
                    best, arg = v, offsets[r + 1] - 1
            if lengths[s] == 1:
                # a 1-frame minimum: staying (matured) competes with entering
                stay = score[t - 1, first] + transition[s, s]
                if stay > best:
                    best, arg = stay, first
                score[t, first], back[t, first] = best + logits[t, s], arg
                continue
            score[t, first], back[t, first] = best + logits[t, s], arg
            for k in range(1, lengths[s]):
                state = first + k
                prev = state - 1
                value = score[t - 1, prev] + transition[s, s]
                if k == lengths[s] - 1:
                    stay = score[t - 1, last] + transition[s, s]
                    if stay > value:
                        value, prev = stay, last
                score[t, state], back[t, state] = value + logits[t, s], prev
    states = np.empty(n, dtype=np.int64)
    best, arg = -np.inf, 0
    for state in range(m):
        if score[n - 1, state] > best:
            best, arg = score[n - 1, state], state
    states[n - 1] = arg
    for t in range(n - 2, -1, -1):
        states[t] = back[t + 1, states[t + 1]]
    out = np.empty(n, dtype=np.int8)
    for t in range(n):
        out[t] = label[states[t]]
    return out


try:
    from numba import njit
    _min_duration_kernel = njit(cache=True)(_min_duration_loops)
except ImportError:
    _min_duration_kernel = _min_duration_loops


def viterbi_min_duration(logits, transition, lengths=(1, 1, 1)):
    """Best H/A/D path in which every run of label s lasts at least lengths[s] frames. Only the
    last run may be shorter (cut off by the end of the sequence); the first run of a half must
    still reach the minimum. lengths (1, 1, 1) is plain Viterbi."""
    lengths = np.asarray([max(1, int(v)) for v in lengths], dtype=np.int64)
    return _min_duration_kernel(np.ascontiguousarray(logits[:, :3], dtype=np.float64),
                                np.ascontiguousarray(transition, dtype=np.float64), lengths)


def decode_min_duration(sample, logp, transition, lengths):
    out = np.full(sample.n_frames, X, dtype=np.int8)
    for start, stop in sample.runs():
        out[start:stop] = viterbi_min_duration(logp[start:stop], transition, lengths)
    return out


# -- posterior sampling and MBR ----------------------------------------------------------------

def _ffbs_loops(logits, transition, uniforms):
    """Forward filtering, backward sampling for a 3-state chain CRF; one path per uniforms row."""
    n = logits.shape[0]
    alpha = np.empty((n, 3))
    for s in range(3):
        alpha[0, s] = logits[0, s]
    for t in range(1, n):
        for s in range(3):
            m = -np.inf
            for r in range(3):
                v = alpha[t - 1, r] + transition[r, s]
                if v > m:
                    m = v
            total = 0.0
            for r in range(3):
                total += np.exp(alpha[t - 1, r] + transition[r, s] - m)
            alpha[t, s] = m + np.log(total) + logits[t, s]
    k = uniforms.shape[0]
    paths = np.empty((k, n), dtype=np.int8)
    w = np.empty(3)
    for path in range(k):
        for t in range(n - 1, -1, -1):
            for s in range(3):
                w[s] = alpha[t, s] if t == n - 1 else alpha[t, s] + transition[s, paths[path, t + 1]]
            m = max(w[0], max(w[1], w[2]))
            total = 0.0
            for s in range(3):
                w[s] = np.exp(w[s] - m)
                total += w[s]
            u, acc, pick = uniforms[path, t] * total, 0.0, 2
            for s in range(3):
                acc += w[s]
                if u <= acc:
                    pick = s
                    break
            paths[path, t] = pick
    return paths


try:
    _ffbs_kernel = njit(cache=True)(_ffbs_loops)
except NameError:
    _ffbs_kernel = _ffbs_loops


def sample_paths(sample, logp, transition, n_samples=24, temperature=1.0, seed=0):
    """(n_samples, n) label paths from the posterior, X outside the runs. Deterministic per seed."""
    rng = np.random.default_rng(seed)
    out = np.full((n_samples, sample.n_frames), X, dtype=np.int8)
    for start, stop in sample.runs():
        logits = np.ascontiguousarray(temperature * logp[start:stop, :3], dtype=np.float64)
        uniforms = rng.random((n_samples, stop - start))
        out[:, start:stop] = _ffbs_kernel(logits, np.ascontiguousarray(transition, np.float64),
                                          uniforms)
    return out


def mbr_decode(sample, logp, candidates, sampling_transition, n_samples=24, temperature=1.0,
               seed=0):
    """(chosen decoding, index of the chosen candidate, expected spellF1 per candidate).

    `candidates` are full decodings (n,) to choose from. Each is scored by its mean spellF1
    against posterior samples treated as ground truth.
    """
    paths = sample_paths(sample, logp, sampling_transition, n_samples, temperature, seed)
    runs = sample.runs()
    expected = [np.mean([fast_spell_f1(path, c, runs) for path in paths]) for c in candidates]
    best = int(np.argmax(expected))
    return candidates[best], best, expected
