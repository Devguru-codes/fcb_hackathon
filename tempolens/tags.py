"""Which side a Smart Tagging tag says has the ball (used by tagprior.py)."""

from difflib import SequenceMatcher


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
