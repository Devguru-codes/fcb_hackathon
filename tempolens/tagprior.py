"""Smart Tagging as a self-calibrated prior on which team has the ball.

Smart Tagging exists only for the competition match, so its weight cannot be learned on open
data, and the public five minutes are far too small to tune it on. Instead the prior calibrates
itself on the match, without labels:

1. each tag becomes a vote for the side in possession over its interval (defensive codes point at
   the other side -- see tags.py), grouped into code families;
2. on frames where the model is already very confident who has the ball, the agreement rate r of
   each family's votes with the model estimates that family's reliability;
3. the family's weight is the shrunk log-odds of r, zero when r is no better than chance, so
   unreliable tags cost nothing;
4. the weighted vote s is added as +s/2 to log P(H) and -s/2 to log P(A), log P(D) untouched and
   nothing renormalised: H gains s against A, and the favoured side also gains s/2 against D.

Deterministic: the same model output and tags always give the same weights, so `predict`
reproduces it exactly. Whether it helps is tested by `tagstudy.py` on synthetic tags.
"""

import xml.etree.ElementTree as ET

import numpy as np

from .match import A, H
from .tags import _normalise, possessing_side, resolve_side

GROUPS = {
    "attack": ("build up", "progression", "creating chances", "attacking transition",
               "long ball", "ball in final third", "ball in the box", "players in the box"),
    "defence": ("defending in attacking third", "defending in middle third",
                "defending in defensive third", "defensive transition"),
}
# Plan row E7: one weight per code family instead of the two groups above (set pieces stay out).
FAMILIES = {"build up": ("build up",), "progression": ("progression",),
            "creating chances": ("creating chances",),
            "attacking transition": ("attacking transition",), "long ball": ("long ball",),
            "final third / box": ("ball in final third", "ball in the box", "players in the box"),
            **{code: (code,) for code in GROUPS["defence"]}}
CONFIDENT = 0.9      # model probability of a side above which a frame calibrates the tags
SHRINK = 0.5         # fraction of the full naive-Bayes log-odds actually applied
MAX_WEIGHT = 2.0
MIN_FRAMES = 500     # a family with fewer confident frames under it gets no weight


def group_of(code, groups=GROUPS):
    code = _normalise(code)
    for group, codes in groups.items():
        if code in codes:
            return group
    return None


def read_instances(path, home, away, fps=25, groups=GROUPS):
    """[(start_frame, stop_frame, group, side)] for every tag naming a usable team."""
    out = []
    for instance in ET.parse(path).getroot().iter("instance"):
        try:
            start = int(round(float(instance.findtext("start")) * fps))
            stop = int(round(float(instance.findtext("end")) * fps)) + 1
        except (TypeError, ValueError):
            continue
        code = instance.findtext("code")
        group = group_of(code, groups)
        labels = {label.findtext("group"): label.findtext("text")
                  for label in instance.findall("label")}
        side = possessing_side(code, resolve_side(labels.get("Team"), home, away))
        if group is not None and side is not None and stop > start:
            out.append((start, stop, group, side))
    return out


def votes(instances, n_frames, groups=GROUPS):
    """{group: (n,) int8} with +1 where the family says home has the ball, -1 away, 0 none."""
    tally = {group: np.zeros(n_frames, dtype=np.int32) for group in groups}
    for start, stop, group, side in instances:
        start, stop = max(0, start), min(n_frames, stop)
        if stop <= start:      # wholly outside the match; a negative stop would index from the end
            continue
        tally[group][start:stop] += 1 if side == H else -1
    return {group: np.sign(t).astype(np.int8) for group, t in tally.items()}


def calibrate(logp, group_votes, inside, shrink=SHRINK):
    """{group: (weight, reliability, n_confident_frames)} from agreement with a confident model;
    the weight is `shrink` times the family's log-odds, clipped to [0, MAX_WEIGHT]."""
    p = np.exp(logp[:, :3])
    side = np.where(p[:, H] > p[:, A], 1, -1)
    confident = inside & (np.maximum(p[:, H], p[:, A]) > CONFIDENT)
    out = {}
    for group, vote in group_votes.items():
        frames = confident & (vote != 0)
        n = int(frames.sum())
        if n < MIN_FRAMES:
            out[group] = (0.0, float("nan"), n)
            continue
        r = float(np.mean(vote[frames] == side[frames]))
        r = min(max(r, 1e-3), 1 - 1e-3)
        weight = float(np.clip(shrink * np.log(r / (1 - r)), 0.0, MAX_WEIGHT))
        out[group] = (weight, r, n)
    return out


def apply(logp, group_votes, weights):
    """logp with half of the summed weighted votes added to log P(H) and subtracted from log P(A).

    log P(D) is untouched and nothing is renormalised, so the favoured side gains against D as
    well as against the other side (the decoder reads the scores as they are)."""
    out = logp.copy()
    shift = np.zeros(len(logp))
    for group, vote in group_votes.items():
        shift += weights.get(group, (0.0,))[0] * vote
    out[:, H] += 0.5 * shift
    out[:, A] -= 0.5 * shift
    return out


def self_calibrated(logp, instances, inside, groups=GROUPS, shrink=SHRINK):
    """(adjusted logp, calibration report) -- the whole prior in one call."""
    group_votes = votes(instances, len(logp), groups)
    weights = calibrate(logp, group_votes, inside, shrink)
    return apply(logp, group_votes, weights), weights
