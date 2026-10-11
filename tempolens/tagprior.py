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
4. weighted votes shift the H-versus-A log-odds only; P(dead ball) is left to the model.

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
CONFIDENT = 0.9      # model probability of a side above which a frame calibrates the tags
SHRINK = 0.5         # fraction of the full naive-Bayes log-odds actually applied
MAX_WEIGHT = 2.0
MIN_FRAMES = 500     # a family with fewer confident frames under it gets no weight


def group_of(code):
    code = _normalise(code)
    for group, codes in GROUPS.items():
        if code in codes:
            return group
    return None


def read_instances(path, home, away, fps=25):
    """[(start_frame, stop_frame, group, side)] for every tag naming a usable team."""
    out = []
    for instance in ET.parse(path).getroot().iter("instance"):
        try:
            start = int(round(float(instance.findtext("start")) * fps))
            stop = int(round(float(instance.findtext("end")) * fps)) + 1
        except (TypeError, ValueError):
            continue
        code = instance.findtext("code")
        group = group_of(code)
        labels = {label.findtext("group"): label.findtext("text")
                  for label in instance.findall("label")}
        side = possessing_side(code, resolve_side(labels.get("Team"), home, away))
        if group is not None and side is not None and stop > start:
            out.append((start, stop, group, side))
    return out


def votes(instances, n_frames):
    """{group: (n,) int8} with +1 where the family says home has the ball, -1 away, 0 none."""
    tally = {group: np.zeros(n_frames, dtype=np.int32) for group in GROUPS}
    for start, stop, group, side in instances:
        start, stop = max(0, start), min(n_frames, stop)
        tally[group][start:stop] += 1 if side == H else -1
    return {group: np.sign(t).astype(np.int8) for group, t in tally.items()}


def calibrate(logp, group_votes, inside):
    """{group: (weight, reliability, n_confident_frames)} from agreement with a confident model."""
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
        weight = float(np.clip(SHRINK * np.log(r / (1 - r)), 0.0, MAX_WEIGHT))
        out[group] = (weight, r, n)
    return out


def apply(logp, group_votes, weights):
    """logp with each family's weighted vote added to the H-versus-A log-odds (D untouched)."""
    out = logp.copy()
    shift = np.zeros(len(logp))
    for group, vote in group_votes.items():
        shift += weights.get(group, (0.0,))[0] * vote
    out[:, H] += 0.5 * shift
    out[:, A] -= 0.5 * shift
    return out


def self_calibrated(logp, instances, inside):
    """(adjusted logp, calibration report) -- the whole prior in one call."""
    group_votes = votes(instances, len(logp))
    weights = calibrate(logp, group_votes, inside)
    return apply(logp, group_votes, weights), weights
