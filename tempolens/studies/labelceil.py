"""Label-noise ceiling (plan of action row LCEIL). Report only, decides nothing.

    python -m tempolens.studies.labelceil

On the 7 DFL matches: the Sportec ball-owner / status flag labels we train on (10-frame fold)
against labels made from DFL's own events with the Metrica event rules (opendata.labels_from_events),
folded at 10 frames too. DFL has no ball-out event, so before every set piece other than a kickoff
the ball is taken as dead from the previous event: the event labels call the ball dead too early,
which makes the measured disagreement an upper bound on that count.
"""

import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from ..match import A, H, X

SET_PIECES = ("KickOff", "ThrowIn", "FreeKick", "CornerKick", "GoalKick", "Penalty")
FOLD = 10


def _team(element, team_of):
    for el in element.iter():
        if el.attrib.get("Team") in team_of:
            return team_of[el.attrib["Team"]]
    return -1


def event_rows(raw):
    """Normalised events (team, type, subtype, start, end, order) from [(frame, tag, team,
    child tags)] in DFL's terms, sorted or not; the row's mapping, with an inferred ball-out
    (no team) at the previous event's frame before every set piece other than a kickoff."""
    rows, previous = [], None
    for frame, tag, team, children in sorted(raw, key=lambda r: r[0]):
        if tag in SET_PIECES:
            if tag != "KickOff" and previous is not None:
                rows.append((-1, "BALL OUT", "INFERRED", previous, previous))
            row = (team, "SET PIECE", tag.upper(), frame, frame)
        elif tag in ("Play", "OtherBallAction"):
            row = (team, "PASS", "", frame, frame)
        elif tag == "ShotAtGoal":
            sub = "GOAL" if "SuccessfulShot" in children else "OUT" if "ShotWide" in children else ""
            row = (team, "SHOT", sub, frame, frame)
        elif tag == "BallClaiming":
            row = (team, "RECOVERY", "", frame, frame)
        elif tag == "Foul":
            row = (-1, "FAULT RECEIVED", "", frame, frame)
        elif tag == "Offside":
            row = (team, "BALL LOST", "OFFSIDE", frame, frame)
        elif tag == "FinalWhistle":
            row = (-1, "BALL LOST", "END HALF", frame, frame)
        else:
            continue
        rows.append(row)
        previous = frame
    frame = pd.DataFrame(rows, columns=["team", "type", "subtype", "start", "end"])
    frame["order"] = np.arange(len(frame))
    return frame


def dfl_raw(match_id):
    """[(table frame, tag, team, child tags)] of one DFL match's events inside its halves."""
    from ..opendata import _dfl_file
    from ..touches import _dfl_halves, _time
    info = _dfl_file("matchinformation", match_id).read_text(encoding="utf-8")
    general = dict(re.findall(r'(\w+)="([^"]*)"', re.search(r"<General [^>]*>", info).group(0)))
    team_of = {general["HomeTeamId"]: H, general["GuestTeamId"]: A}
    halves, out = _dfl_halves(match_id), []
    for event in ET.parse(_dfl_file("events_raw", match_id)).getroot().iter("Event"):
        child = next(iter(event), None)
        if child is None or "EventTime" not in event.attrib:
            continue
        t, offset = _time(event.attrib["EventTime"]), 0
        for start, n in halves:
            k = int(round((t - start) / 0.04))
            if 0 <= k < n:
                out.append((offset + k, child.tag, _team(child, team_of),
                            {el.tag for el in child.iter()}))
                break
            offset += n
    return out


def compare(flag, events, period):
    """Agreement of two label arrays on graded frames (both not X): frame agreement, alive-vs-dead
    agreement, and spellF1 / macroF1 of `events` scored against `flag`."""
    from ..experiments import score
    graded = (period > 0) & (flag != X) & (events != X)
    alive_f, alive_e = np.isin(flag, (H, A)), np.isin(events, (H, A))
    s = score(np.where(graded, flag, X), np.where(graded, events, X))
    return {"agree": float(np.mean(flag[graded] == events[graded])),
            "alive": float(np.mean(alive_f[graded] == alive_e[graded])),
            "spellF1": s["spellF1"], "macroF1": s["macroF1"]}


def main():
    from ..opendata import LOADERS, labels_from_events, load_open_match, remove_flicker
    rows = []
    for name in sorted(n for n in LOADERS if n.startswith("dfl_")):
        table = load_open_match(name)        # flag labels, folded at opendata.DFL_MIN_SPELL
        events = labels_from_events(event_rows(dfl_raw(name[4:])), table.period, table.n_frames)
        events = remove_flicker(events, table.runs(), FOLD)
        r = compare(table.labels, events, table.period)
        rows.append(r)
        print("{:<11} frame agreement {:.3f}  alive/dead agreement {:.3f}  spellF1 {:.4f}  "
              "macroF1 {:.4f}".format(name, r["agree"], r["alive"], r["spellF1"], r["macroF1"]),
              flush=True)
    print("LCEIL mean over {} DFL matches: frame agreement {:.3f}, alive/dead {:.3f}, spellF1 {:.4f}, "
          "macroF1 {:.4f} (event labels scored against the flag labels; report only)".format(
              len(rows), *(float(np.mean([r[k] for r in rows]))
                           for k in ("agree", "alive", "spellF1", "macroF1"))), flush=True)


if __name__ == "__main__":
    main()
