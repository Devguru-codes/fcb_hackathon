"""Kickoff-team consistency on the match (plan of action row E9). Prints counts only.

    python -m tempolens.kickoffs --data-root <bundle> --labels <submission csv>

Rules of the game, nothing tuned: the team that kicks off the second half is the one that did not
kick off the first; after a goal, the conceding team kicks off. The first-half kickoff team is
taken from the decoded labels themselves; goals and their restarts come from Smart Tagging
(GOALS with its Team; the instance labelled Type "Kick Off (After Goal)", which opens a few
seconds early), the exact
restart from the ball track. Each kickoff's decoded team is the majority of H/A over the first
2 s after the restart. No ground truth is read.
"""

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
sys.path.insert(0, str(ROOT / "starter"))

from games import GAME  # noqa: E402
from .match import A, H  # noqa: E402
from .predict import SECOND_HALF_KICKOFF  # noqa: E402
from .tags import _normalise, resolve_side  # noqa: E402

HALF_KICKOFFS = (11185, SECOND_HALF_KICKOFF[GAME])    # FAQ.md
WINDOW_S, SEARCH_S, CENTRE_M, MOVE_M = 2.0, 15.0, 3.0, 2.0


def restart_frame(table, approx, search_s=SEARCH_S, strict=False):
    """The first frame from `approx` on (within `search_s`) where the ball sits within CENTRE_M of
    the centre spot and has moved MOVE_M one second later; if none is found, `approx` (None with
    `strict`)."""
    fps, (length, width) = table.fps, table.pitch
    ball = table.ball
    for f in range(approx, min(table.n_frames - fps, approx + int(search_s * fps))):
        b, later = ball[f], ball[f + fps]
        if (np.isfinite(b).all() and np.isfinite(later).all()
                and np.hypot(b[0] - length / 2, b[1] - width / 2) <= CENTRE_M
                and np.hypot(*(later - b)) >= MOVE_M):
            return f
    return None if strict else approx


def decoded_team(codes, frame, fps):
    """H or A by majority over the first WINDOW_S after `frame`; None if neither appears."""
    window = codes[frame:frame + int(WINDOW_S * fps)]
    h, a = int((window == H).sum()), int((window == A).sum())
    return None if h == a == 0 else (H if h >= a else A)


def goal_restarts(path, home, away, fps):
    """[(approx restart frame, kicking-off side, GOALS tag start, tag end)] from GOALS + Kick Off
    (After Goal) tags."""
    goals, restarts = [], []
    for instance in ET.parse(path).getroot().iter("instance"):
        code = _normalise(instance.findtext("code") or "")
        try:
            start = int(round(float(instance.findtext("start")) * fps))
            end = int(round(float(instance.findtext("end")) * fps))
        except (TypeError, ValueError):
            continue
        labels = {l.findtext("group"): l.findtext("text") for l in instance.findall("label")}
        if code == "goals":
            goals.append((end, resolve_side(labels.get("Team"), home, away), start))
        elif _normalise(labels.get("Type") or "") == "kick off (after goal)":   # a Type label
            restarts.append(start)
    out = []
    for end, scorer, start in sorted(goals):     # a GOALS tag runs until just before its restart
        after = [r for r in sorted(restarts) if end - 10 * fps <= r <= end + 60 * fps]
        if after and scorer in (H, A):
            out.append((after[0], A if scorer == H else H, start, end))
    return out, len(goals), len(restarts)


def restarts(codes, table, goals):
    """[(kind, restart frame, implied side)] for every kickoff, in time order. The first half's
    side is the decoded one (the rules cannot name it); `goals` as goal_restarts returns them."""
    first = decoded_team(codes, restart_frame(table, HALF_KICKOFFS[0]), table.fps)
    events = [("half 1", HALF_KICKOFFS[0], first),
              ("half 2", HALF_KICKOFFS[1], None if first is None else (A if first == H else H))]
    events += [("goal", f, side) for f, side, *_ in goals]
    return [(kind, restart_frame(table, approx), implied)
            for kind, approx, implied in sorted(events, key=lambda e: e[1])]


def check(codes, table, goals):
    """[(kind, implied side, decoded side)] for every kickoff, in time order."""
    return [(kind, implied, decoded_team(codes, frame, table.fps))
            for kind, frame, implied in restarts(codes, table, goals)]


def tally(rows):
    """(agree, disagree, undecided) over the kickoffs the rules can check. The first half's
    implied team is read from the decoded labels, so it agrees by construction: it is the
    reference, not a check."""
    checked = [r for r in rows if r[0] != "half 1"]
    agree = sum(i is not None and d is not None and i == d for _, i, d in checked)
    undecided = sum(i is None or d is None for _, i, d in checked)
    return agree, len(checked) - agree - undecided, undecided


def main(argv=None):
    import possession_metrics as pm
    from games import GAME, GAMES, require
    from .match import competition_table
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--labels", required=True, help="a full-match frame,label CSV")
    args = parser.parse_args(argv)
    table = competition_table(GAME, args.data_root)
    codes = pm.label_codes(pm.read_label_csv(args.labels))
    path = require(["smart_tagging"], GAME, args.data_root)[0]
    goals, n_goals, n_restarts = goal_restarts(path, table.meta["home"], table.meta["away"],
                                               GAMES[GAME]["fps"])
    rows = check(codes, table, goals)
    agree, disagree, undecided = tally(rows)
    print("GOALS tags {}, Kick Off (After Goal) tags {}, goal restarts paired {}".format(
        n_goals, n_restarts, len(goals)))
    print("first-half kickoff: the reference (its team is the decoded one, {})".format(
        "found" if rows and rows[0][2] is not None else "none decoded"))
    print("checkable kickoffs {}: decoded team agrees with the rule {}, disagrees {}, "
          "undecided {}".format(len(rows) - 1, agree, disagree, undecided))
    print("per kickoff (time order): " + " ".join(
        "{}={}".format(k, "?" if d is None or i is None else "yes" if d == i else "no")
        for k, i, d in rows))


if __name__ == "__main__":
    main()
