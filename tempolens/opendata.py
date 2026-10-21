"""Open football data with known possession, converted to MatchTables for training and validation.

Metrica Sports' public sample games (github.com/metrica-sports/sample-data) are the backbone:
the same provider as the competition, synchronised tracking with the ball at 25 fps, and event
data from which a frame-level H/A/D label can be derived. Game 3 is in the same FIFA/EPTS format
as the competition ATD.

The labels are derived from events, not from corrected tracking as the competition's ground truth
is, so they are close to but not identical with the competition's definition. That difference is
documented in the technical document; the public five minutes are too few to calibrate it.

Set MTAH_OPEN_DATA_DIR to the folder holding `metrica_sample_data/`; by default the nearest
`external_data/` folder above the repository is used.
"""

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .match import A, D, H, MatchTable, X

PITCH = (105.0, 68.0)


def open_data_dir():
    """MTAH_OPEN_DATA_DIR, else the nearest `external_data/` folder above this package."""
    candidates = [os.environ.get("MTAH_OPEN_DATA_DIR")]
    candidates += [parent / "external_data" for parent in Path(__file__).resolve().parents]
    for candidate in candidates:
        if candidate and Path(candidate, "metrica_sample_data").is_dir():
            return Path(candidate)
    raise IOError("Metrica sample data not found; clone github.com/metrica-sports/sample-data "
                  "into <dir>/metrica_sample_data and set MTAH_OPEN_DATA_DIR=<dir>")


def _metrica_dir(game):
    return open_data_dir() / "metrica_sample_data" / "data" / "Sample_Game_{}".format(game)


# -- events -> frame labels ------------------------------------------------------------------

# A dead ball begins at these events. The value says which frame of the event the ball dies on.
_DEAD_AT_END = ("BALL OUT",)
_DEAD_BALL_LOST_SUBTYPES = ("OFFSIDE", "HAND BALL", "END HALF", "REFEREE HIT")
# Events that show the acting team has the ball at the event's start frame.
_POSSESSION_TYPES = ("PASS", "SHOT", "RECOVERY", "BALL LOST", "BALL OUT", "SET PIECE")


def _dead_frame(row):
    """Frame at which this event kills the ball, or None if it does not."""
    kind, sub = row.type, row.subtype or ""
    end = row.end if row.end >= row.start else row.start
    if kind in _DEAD_AT_END:
        return end
    if kind == "SHOT" and ("OUT" in sub or "GOAL" in sub):
        return end
    if kind == "BALL LOST" and any(token in sub for token in _DEAD_BALL_LOST_SUBTYPES):
        return end
    if kind == "FAULT RECEIVED":
        return row.start
    if kind == "CHALLENGE" and "FAULT" in sub and "ADVANTAGE" not in sub:
        return row.start
    return None


def labels_from_events(events, period, n_frames):
    """Frame labels H/A/D inside periods, X outside, from a normalised event table.

    `events` needs columns team (H/A), type, subtype, start, end (zero-based frames), sorted
    or not. The rules, in order of precedence at any frame:

    - each period is dead until its first set piece (the kickoff);
    - a set piece restarts play with the taking team in possession;
    - BALL OUT, a shot that goes out or in, offside/handball/end-of-half losses and fouls kill
      the ball (at the frame the ball crosses the line, or the whistle);
    - while alive, a possession event (pass, shot, recovery, loss, set piece) by the other team
      hands possession over at its start frame; a dead ball followed by a possession event with
      no set piece in between resumes play at that event;
    - after a period's last event the ball is dead.

    Challenges, cards and fouls received never confer possession.
    """
    labels = np.full(n_frames, X, dtype=np.int8)
    events = events.sort_values(["start", "order"], kind="stable")
    inside = period > 0
    bounds = np.flatnonzero(np.diff(np.concatenate(([0], inside.astype(np.int8), [0]))))
    for first, stop in zip(bounds[0::2], bounds[1::2]):
        labels[first:stop] = D
        window = events[(events.start >= first) & (events.start < stop)]
        state, since = D, first
        changes = []
        started = False
        for row in window.itertuples():
            if row.type == "SET PIECE":
                started = True
            if not started:
                continue
            dead = _dead_frame(row)
            if row.type in _POSSESSION_TYPES and row.team in (H, A):
                if row.type == "SET PIECE" or state != row.team:
                    changes.append((row.start, row.team))
                    state = row.team
            if dead is not None:
                changes.append((max(dead, row.start), D))
                state = D
        last_end = int(min(stop, window.end.max() + 1)) if len(window) else first
        changes.append((last_end, D))
        changes.sort(key=lambda change: change[0])
        for (frame, code), (next_frame, _) in zip(changes, changes[1:] + [(stop, D)]):
            frame, next_frame = max(frame, first), min(next_frame, stop)
            if next_frame > frame:
                labels[frame:next_frame] = code
    return labels


# -- Metrica sample games 1 and 2: CSV ---------------------------------------------------------

def _read_metrica_csv_team(path):
    """(period, frame, positions (n, players, 2) normalised, ball (n, 2) normalised)."""
    frame = pd.read_csv(path, skiprows=2, low_memory=False)
    columns = list(frame.columns)
    xs = [i for i, name in enumerate(columns) if name.startswith("Player")]
    ball_col = columns.index("Ball")
    values = frame.to_numpy(dtype=np.float64)
    positions = np.stack([values[:, [i, i + 1]] for i in xs], axis=1)
    return (values[:, 0].astype(np.int8), values[:, 1].astype(np.int64), positions,
            values[:, [ball_col, ball_col + 1]])


def _metrica_csv_events(path, n_frames):
    events = pd.read_csv(path)
    team = events["Team"].map({"Home": H, "Away": A}).fillna(-1).astype(int)
    start = events["Start Frame"].fillna(0).astype(int) - 1
    end = events["End Frame"].fillna(0).astype(int) - 1
    return pd.DataFrame({
        "team": team, "type": events["Type"].astype(str).str.upper(),
        "subtype": events["Subtype"].fillna("").astype(str).str.upper(),
        "start": start.clip(0, n_frames - 1), "end": end.clip(-1, n_frames - 1),
        "order": np.arange(len(events)),
    })


def metrica_csv_game(game):
    folder = _metrica_dir(game)
    stem = "Sample_Game_{}_RawTrackingData_{}_Team.csv"
    period, frames, home, ball = _read_metrica_csv_team(folder / stem.format(game, "Home"))
    _, frames_away, away, _ = _read_metrica_csv_team(folder / stem.format(game, "Away"))
    if not np.array_equal(frames, frames_away):
        raise ValueError("home/away tracking frames differ in game {}".format(game))
    if not np.array_equal(frames, np.arange(1, len(frames) + 1)):
        raise ValueError("game {} tracking frames are not 1..n".format(game))
    n = len(frames)
    scale = np.asarray(PITCH, dtype=np.float64)
    players = np.concatenate([home, away], axis=1) * scale
    slot_team = np.asarray([H] * home.shape[1] + [A] * away.shape[1], dtype=np.int8)
    events = _metrica_csv_events(folder / "Sample_Game_{}_RawEventsData.csv".format(game), n)
    period = np.nan_to_num(period).astype(np.int8)
    table = MatchTable(name="metrica{}".format(game), ball=(ball * scale).astype(np.float32),
                       players=players.astype(np.float32), slot_team=slot_team, period=period,
                       pitch=PITCH)
    table.labels = labels_from_events(events, period, n)
    return table


# -- Metrica sample game 3: FIFA/EPTS + JSON events --------------------------------------------

def metrica_epts_game3():
    from kloppy import metrica

    folder = _metrica_dir(3)
    dataset = metrica.load_tracking_epts(meta_data=str(folder / "Sample_Game_3_metadata.xml"),
                                         raw_data=str(folder / "Sample_Game_3_tracking.txt"))
    frame = dataset.to_df(engine="pandas")
    ids = frame["frame_id"].to_numpy()
    if not np.array_equal(ids, np.arange(ids[0], ids[0] + len(ids))):
        raise ValueError("game 3 tracking frames are not contiguous")
    n = len(frame)
    offset = int(ids[0])   # first frame_id; event frames use the same counter
    scale = np.asarray(PITCH, dtype=np.float64)
    slots, slot_team = [], []
    for code, team in ((H, dataset.metadata.teams[0]), (A, dataset.metadata.teams[1])):
        for player in team.players:
            column = "{}_x".format(player.player_id)
            if column in frame:
                slots.append(frame[[column, "{}_y".format(player.player_id)]].to_numpy())
                slot_team.append(code)
    players = (np.stack(slots, axis=1) * scale).astype(np.float32)
    ball = (frame[["ball_x", "ball_y"]].to_numpy(dtype=np.float64) * scale).astype(np.float32)
    period = frame["period_id"].fillna(0).to_numpy().astype(np.int8)

    raw = json.loads((folder / "Sample_Game_3_events.json").read_text(encoding="utf-8"))["data"]
    home_id = dataset.metadata.teams[0].team_id
    rows = []
    for order, event in enumerate(raw):
        team = (event.get("team") or {}).get("id")
        start = ((event.get("start") or {}).get("frame"))
        end = ((event.get("end") or {}).get("frame"))
        if start is None:
            continue
        subtypes = event.get("subtypes") or []
        if isinstance(subtypes, dict):
            subtypes = [subtypes]
        rows.append({
            "team": H if team == home_id else (A if team else -1),
            "type": event["type"]["name"].upper(),
            "subtype": "-".join(s["name"].upper() for s in subtypes),
            "start": int(start) - offset, "end": int(end if end is not None else start) - offset,
            "order": order,
        })
    events = pd.DataFrame(rows)
    events["start"] = events["start"].clip(0, n - 1)
    events["end"] = events["end"].clip(-1, n - 1)
    table = MatchTable(name="metrica3", ball=ball, players=players,
                       slot_team=np.asarray(slot_team, dtype=np.int8), period=period,
                       pitch=PITCH)
    table.labels = labels_from_events(events, period, n)
    return table


# -- DFL / Sportec open data (Bassek et al., Scientific Data, CC-BY 4.0) -----------------------
#
# Seven Bundesliga matches whose 25 Hz positions carry, on every ball frame, the provider's own
# BallStatus (alive/dead) and BallPossession (home/guest). That is the same kind of signal the
# competition's ground truth is derived from -- corrected tracking's dead flag and owning team --
# so these labels are closer to the target definition than labels derived from events. The ball
# is tracked through dead time too, which lets simulate.py drop it realistically in both states.

DFL_MATCHES = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")

_ATTR = re.compile(r'(\w+)="([^"]*)"')


def _dfl_file(kind, match_id):
    folder = open_data_dir() / "dfl_idsse"
    found = sorted(folder.glob("DFL_*{}*DFL-MAT-{}.xml*".format(kind, match_id)))
    if not found:
        raise IOError("DFL {} file for {} not found in {}".format(kind, match_id, folder))
    return found[0]


def dfl_game(match_id):
    info = _dfl_file("matchinformation", match_id).read_text(encoding="utf-8")
    general = dict(_ATTR.findall(re.search(r"<General [^>]*>", info).group(0)))
    home_id, away_id = general["HomeTeamId"], general["GuestTeamId"]

    halves = {}        # half -> {"ball": {N: (x, y, poss, status)}, "people": {pid: (team, {N: (x, y)})}}
    half = team = person = None
    with open(_dfl_file("positions_raw", match_id), encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith("<Frame "):
                if team is None:
                    continue
                n = int(line[line.index('N="') + 3:line.index('"', line.index('N="') + 3)])
                attrs = dict(_ATTR.findall(line))
                x, y = float(attrs["X"]), float(attrs["Y"])
                if team == "BALL":
                    halves[half]["ball"][n] = (x, y, int(attrs.get("BallPossession", 0)),
                                               int(attrs.get("BallStatus", 0)))
                else:
                    halves[half]["people"][person][1][n] = (x, y)
            elif line.startswith("<FrameSet "):
                attrs = dict(_ATTR.findall(line))
                half = attrs["GameSection"]
                halves.setdefault(half, {"ball": {}, "people": {}})
                raw_team = attrs["TeamId"]
                team = ("BALL" if raw_team == "BALL" else H if raw_team == home_id
                        else A if raw_team == away_id else None)
                person = attrs.get("PersonId")
                if team in (H, A):
                    halves[half]["people"].setdefault(person, (team, {}))

    order = [h for h in ("firstHalf", "secondHalf") if h in halves]
    people = sorted({pid for h in order for pid in halves[h]["people"]})
    slot_team = np.asarray([next(halves[h]["people"][pid][0] for h in order
                                 if pid in halves[h]["people"]) for pid in people], dtype=np.int8)
    length, width = PITCH
    balls, players, labels, periods = [], [], [], []
    for number, h in enumerate(order, start=1):
        frames = sorted(halves[h]["ball"])
        first, n = frames[0], frames[-1] - frames[0] + 1
        ball = np.full((n, 2), np.nan, dtype=np.float32)
        label = np.full(n, D, dtype=np.int8)
        for f, (x, y, poss, status) in halves[h]["ball"].items():
            ball[f - first] = (x + length / 2, y + width / 2)
            if status == 1 and poss in (1, 2):
                label[f - first] = H if poss == 1 else A
        pos = np.full((n, len(people), 2), np.nan, dtype=np.float32)
        for slot, pid in enumerate(people):
            if pid not in halves[h]["people"]:
                continue
            for f, (x, y) in halves[h]["people"][pid][1].items():
                if 0 <= f - first < n:
                    pos[f - first, slot] = (x + length / 2, y + width / 2)
        balls.append(ball)
        players.append(pos)
        labels.append(label)
        periods.append(np.full(n, number, dtype=np.int8))
    table = MatchTable(name="dfl_" + match_id, ball=np.concatenate(balls),
                       players=np.concatenate(players), slot_team=slot_team,
                       period=np.concatenate(periods), pitch=PITCH,
                       meta={"home": general["HomeTeamName"], "away": general["GuestTeamName"]})
    table.labels = np.concatenate(labels)
    return table


LOADERS = {
    "metrica1": lambda: metrica_csv_game(1),
    "metrica2": lambda: metrica_csv_game(2),
    "metrica3": metrica_epts_game3,
}
LOADERS.update({"dfl_" + m: (lambda m=m: dfl_game(m)) for m in DFL_MATCHES})


def remove_flicker(labels, runs, min_frames):
    """Fold possession spells shorter than `min_frames` that sit between two spells of the other
    team back into that team. Shorter than ~0.4 s, a provider's ball-owner flag is flickering
    rather than describing control; the public ground truth's shortest possession is 19 frames.
    """
    out = labels.copy()
    for start, stop in runs:
        w = out[start:stop]
        bounds = np.r_[0, np.flatnonzero(w[1:] != w[:-1]) + 1, len(w)]
        for i in range(1, len(bounds) - 2):
            a, b = bounds[i], bounds[i + 1]
            before, here, after = w[bounds[i - 1]], w[a], w[b]
            if (b - a < min_frames and here in (H, A) and before == after
                    and before in (H, A) and before != here):
                w[a:b] = before
    return out


# Flicker threshold applied to the DFL provider flag at load time (see remove_flicker).
DFL_MIN_SPELL = 10


def fill_dead_ball(table):
    """A copy of `table` whose missing ball during dead-ball stretches moves in a straight line
    from where it was last seen to where it is next seen (in the same period).

    Metrica's sample games almost never track the ball while it is dead, so on them "no ball"
    meant "dead" every time -- a shortcut the competition tracking breaks: there the ball is
    visible on ~2/3 of dead frames (the keeper fetching it, the throw-in taker carrying it).
    The simulator then loses whole stoppages at its calibrated rate, as for DFL.
    """
    from dataclasses import replace
    ball = table.ball.astype(np.float32).copy()
    for start, stop in table.runs():
        seg, lab = ball[start:stop], table.labels[start:stop]
        seen = np.flatnonzero(np.isfinite(seg[:, 0]))
        if len(seen) < 2:
            continue
        idx = np.arange(len(seg))
        fill = ~np.isfinite(seg[:, 0]) & (lab == D) & (idx > seen[0]) & (idx < seen[-1])
        for axis in (0, 1):
            seg[fill, axis] = np.interp(idx[fill], seen, seg[seen, axis])
    return replace(table, ball=ball)


def load_open_match(name, cache_dir=None):
    """A labelled open MatchTable, cached as .npz under the open-data folder."""
    cache = Path(cache_dir) if cache_dir else open_data_dir() / "cache"
    path = cache / "{}.npz".format(name)
    if path.exists():
        table = MatchTable.load(path)
    else:
        table = LOADERS[name]()
        table.save(path)
    if name.startswith("dfl_") and DFL_MIN_SPELL:
        table.labels = remove_flicker(table.labels, table.runs(), DFL_MIN_SPELL)
    if name.startswith("metrica"):
        table = fill_dead_ball(table)
    return table
