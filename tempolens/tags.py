"""Soft evidence from Smart Tagging, never a direct possession label."""

from difflib import SequenceMatcher
import xml.etree.ElementTree as ET

import numpy as np

from games import GAME, GAMES, require, source_frames


def _normalise(value):
    return " ".join((value or "").strip().casefold().split())


# Codes whose Team tag names the side WITHOUT the ball: they describe defending, or the moments
# straight after losing it. Every other code tags the side in possession (or about to restart).
DEFENSIVE_CODES = frozenset({
    "defending in attacking third",
    "defending in middle third",
    "defending in defensive third",
    "defensive transition",
})

# Below this similarity a tag's team is treated as unknown rather than forced onto a side.
MIN_NAME_SIMILARITY = 0.6


def resolve_side(team, home, away):
    """0 for home, 1 for away, None for N/A or an unrecognised name.

    Tag names and tracking-metadata names are not spelled identically -- the tagging writes
    "Atletic Club" -- so an exact comparison silently drops one side's tags. Match each name to
    the closer of the two clubs, and refuse when neither is close.
    """
    team, home, away = _normalise(team), _normalise(home), _normalise(away)
    if not team or team == "n/a":
        return None

    def similarity(name):
        if team in name or name in team:
            return 1.0
        return SequenceMatcher(None, team, name).ratio()

    home_score, away_score = similarity(home), similarity(away)
    if max(home_score, away_score) < MIN_NAME_SIMILARITY or home_score == away_score:
        return None
    return 0 if home_score > away_score else 1


def possessing_side(code, side):
    """Side in possession implied by one tag: the tagged side, inverted for defensive codes."""
    if side is None:
        return None
    return 1 - side if _normalise(code) in DEFENSIVE_CODES else side


def tag_priors(home_name, away_name, game=GAME, data_dir=None, team_weight=0.65,
               restart_weight=0.45):
    """Return additive H/A/D logits from automatic tactical tags.

    The values are deliberately modest: tags identify phases, not possessions. A tagged team can
    be wrong or its interval can span more than one possession, so tracking and temporal evidence
    can always override it.
    """
    n_frames = source_frames("tactical", game)
    priors = np.zeros((n_frames, 3), dtype=np.float32)
    path = require(["smart_tagging"], game, data_dir)[0]
    fps = GAMES[game]["fps"]
    for instance in ET.parse(path).getroot().iter("instance"):
        try:
            start = max(0, int(round(float(instance.findtext("start")) * fps)))
            end = min(n_frames, int(round(float(instance.findtext("end")) * fps)) + 1)
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        labels = {label.findtext("group"): label.findtext("text")
                  for label in instance.findall("label")}
        code = _normalise(instance.findtext("code"))
        side = possessing_side(code, resolve_side(labels.get("Team"), home_name, away_name))
        if side is not None:
            priors[start:end, side] += team_weight
        # A set-piece interval is mostly live once the restart happens, so make only its boundary
        # weakly dead-ball-like. The decoder's geometry still decides the final answer.
        if code == "set pieces":
            edge = min(end, start + fps)
            priors[start:edge, 2] += restart_weight
    return priors
