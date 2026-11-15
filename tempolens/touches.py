"""Who touches the ball, and when, on the open matches: per-frame targets for a touch head.

The oracle on cross-validation puts most of the remaining headroom in team calls during live
play, and short possessions are under-predicted. Possession labels say who *has* the ball; the
open matches' event data also say where a player of which team *plays* it -- passes, shots,
claims, tackles won, set-piece takes. Those frames, +/-2, are the targets of an auxiliary head
(TemporalConfig.touch), so the network has to learn where control happens.

- DFL: event timestamps mapped to frames through the ball track's own timestamps (25 Hz). The
  event clock is offset from the tracking by a per-match constant (-42 to +34 frames) and each
  event is logged by hand to within ~1 s, so `snap_to_contact` removes the match's offset and
  moves each event to the frame, within +/-1 s, where the acting team's nearest player is closest
  to the ball; events with no contact within 1.5 m are dropped. Only the open matches' clean
  tracking is used for this.
- Metrica: event start frames as given (they already sit on the contact: median 0.07 m).
"""

import re
import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np
import pandas as pd

from .match import A, H
from .opendata import _dfl_file, _metrica_dir, open_data_dir

TOUCH_VERSION = 2
SNAP = 25           # frames searched each side of an event for the contact
CONTACT_M = 1.5
DILATE = 2
_ATTR = re.compile(r'(\w+)="([^"]*)"')
_DFL_SET_PIECES = ("ThrowIn", "FreeKick", "GoalKick", "CornerKick", "KickOff", "Penalty")
_DFL_ON_BALL = ("Play", "ShotAtGoal", "BallClaiming", "OtherBallAction") + _DFL_SET_PIECES
_METRICA_ON_BALL = ("PASS", "SHOT", "RECOVERY", "SET PIECE", "BALL LOST")


def _time(text):
    return datetime.fromisoformat(text).timestamp()


def _dfl_halves(match_id):
    """[(first ball time, n frames)] per half, in table order, from the ball's FrameSet."""
    halves, inside, first, last = [], False, None, None
    with open(_dfl_file("positions_raw", match_id), encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith("<FrameSet "):
                inside = 'TeamId="BALL"' in line
            elif inside and line.startswith("<Frame "):
                attrs = dict(_ATTR.findall(line))
                n = int(attrs["N"])
                if first is None:
                    first = (n, _time(attrs["T"]))
                last = n
            elif inside and line.startswith("</FrameSet"):
                halves.append((first[1], last - first[0] + 1))
                inside, first = False, None
    return halves


def dfl_event_frame(t, halves):
    """Table frame of a DFL event time `t` (s), or None outside every half (`_dfl_halves`)."""
    offset = 0
    for start, n in halves:
        k = int(round((t - start) / 0.04))
        if 0 <= k < n:
            return offset + k
        offset += n
    return None


def dfl_touch_frames(match_id):
    """(frames, teams) of on-ball events, in the table's frame index."""
    info = _dfl_file("matchinformation", match_id).read_text(encoding="utf-8")
    general = dict(_ATTR.findall(re.search(r"<General [^>]*>", info).group(0)))
    team_of = {general["HomeTeamId"]: H, general["GuestTeamId"]: A}
    halves = _dfl_halves(match_id)
    frames, teams = [], []
    for event in ET.parse(_dfl_file("events_raw", match_id)).getroot().iter("Event"):
        child = next(iter(event), None)
        if child is None or "EventTime" not in event.attrib:
            continue
        if child.tag in _DFL_ON_BALL:
            team = child.attrib.get("Team")
        elif child.tag == "TacklingGame" and "ballcontact" in child.attrib.get("WinnerResult", "").lower():
            team = child.attrib.get("WinnerTeam")
        else:
            continue
        if team not in team_of:
            continue
        frame = dfl_event_frame(_time(event.attrib["EventTime"]), halves)
        if frame is not None:
            frames.append(frame)
            teams.append(team_of[team])
    return np.asarray(frames, dtype=np.int64), np.asarray(teams, dtype=np.int8)


def metrica_touch_frames(game):
    folder = _metrica_dir(game)
    if game in (1, 2):
        events = pd.read_csv(folder / "Sample_Game_{}_RawEventsData.csv".format(game))
        keep = events["Type"].astype(str).str.upper().isin(_METRICA_ON_BALL)
        events = events[keep & events["Team"].isin(["Home", "Away"])]
        return (events["Start Frame"].to_numpy(dtype=np.int64) - 1,
                np.where(events["Team"] == "Home", H, A).astype(np.int8))
    import json
    from .match import teams_from_epts_metadata
    _, home, _ = teams_from_epts_metadata(folder / "Sample_Game_3_metadata.xml")
    with open(folder / "Sample_Game_3_tracking.txt", encoding="utf-8") as handle:
        offset = int(handle.readline().split(":", 1)[0])
    frames, teams = [], []
    for event in json.loads((folder / "Sample_Game_3_events.json").read_text(encoding="utf-8"))["data"]:
        start = (event.get("start") or {}).get("frame")
        name = (event.get("team") or {}).get("name")
        if start is None or not name or event["type"]["name"].upper() not in _METRICA_ON_BALL:
            continue
        frames.append(int(start) - offset)
        teams.append(H if name == home else A)
    return np.asarray(frames, dtype=np.int64), np.asarray(teams, dtype=np.int8)


def _own_distance(table, frames, teams):
    """Distance from the ball to the acting team's nearest player at each (frame, team)."""
    f = np.clip(frames, 0, table.n_frames - 1)
    ball = table.ball[f]
    d = np.hypot(table.players[f, :, 0] - ball[:, None, 0], table.players[f, :, 1] - ball[:, None, 1])
    d = np.where((table.slot_team[None, :] == teams[:, None]) & np.isfinite(d), d, np.inf)
    out = d.min(axis=1)
    return np.where(np.isfinite(out), out, np.nan)


def snap_to_contact(table, frames, teams, snap=SNAP, contact=CONTACT_M):
    """Remove the match-wide offset (the shift with the smallest median contact distance), then
    move each event to its closest contact within +/-snap frames; drop events with none."""
    shifts = range(-100, 101, 2)
    offset = min(shifts, key=lambda k: np.nanmedian(_own_distance(table, frames + k, teams)))
    frames = frames + offset
    grid = np.arange(-snap, snap + 1)
    d = np.stack([_own_distance(table, frames + k, teams) for k in grid], axis=1)
    d = np.where(np.isfinite(d), d, np.inf)
    best = d.argmin(axis=1)
    keep = d[np.arange(len(d)), best] < contact
    return np.clip(frames + grid[best], 0, table.n_frames - 1)[keep], teams[keep]


def touch_frames(name):
    if name.startswith("dfl_"):
        from .opendata import load_open_match
        return snap_to_contact(load_open_match(name), *dfl_touch_frames(name[4:]))
    if name.startswith("metrica"):
        return metrica_touch_frames(int(name[-1]))
    raise KeyError("no event data for {}".format(name))


def touch_targets(name, n_frames, dilate=DILATE):
    """(n, 2) float32: 1 within +/-dilate frames of a home (column 0) / away (column 1) touch.
    Cached under the open-data folder."""
    path = open_data_dir() / "cache" / "touch{}_{}.npy".format(TOUCH_VERSION, name)
    if path.exists():
        out = np.load(path)
        if len(out) == n_frames:
            return out
    frames, teams = touch_frames(name)
    out = np.zeros((n_frames, 2), dtype=np.float32)
    for k in range(-dilate, dilate + 1):
        f = frames + k
        ok = (f >= 0) & (f < n_frames)
        out[f[ok], teams[ok]] = 1.0
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, out)
    return out
