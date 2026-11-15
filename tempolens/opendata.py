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
    # one block per period: halves that follow each other without a gap are split too
    bounds = np.concatenate(([0], np.flatnonzero(np.diff(period) != 0) + 1, [len(period)]))
    for first, stop in zip(bounds[:-1], bounds[1:]):
        if period[first] <= 0:
            continue
        labels[first:stop] = D
        window = events[(events.start >= first) & (events.start < stop)]
        state = D
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
# Seven DFL matches (2 Bundesliga, 5 2. Bundesliga) whose 25 Hz positions carry, on every ball frame, the provider's own
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


# -- SkillCorner open data (A-League 2024/25, broadcast tracking, MIT licence) ----------------
#
# github.com/SkillCorner/opendata: 20 matches tracked from the TV broadcast at 10 fps -- off-screen
# players extrapolated, the ball missing when unseen, nothing at all during replays -- so its
# faults are real ones, closer to the competition ATD than clean DFL. Labels follow the
# competition's rule: phases of play exist only while the ball is in play (outside them: D);
# inside, the team the tracking names as owning the ball (`possession.group`), kept by the last
# owner through passes and loose balls, starting from the phase's team in possession.

SKILLCORNER_FPS = 10


def _upsample(values, n_out, ratio):
    """Linear interpolation of a (n, ...) array sampled every `ratio` output frames; NaN wherever
    either neighbour is NaN."""
    pos = np.arange(n_out) / ratio
    i0 = np.minimum(np.floor(pos).astype(int), len(values) - 1)
    i1 = np.minimum(i0 + 1, len(values) - 1)
    w = (pos - i0).reshape((-1,) + (1,) * (values.ndim - 1))
    return (1 - w) * values[i0] + w * values[i1]


def skillcorner_game(match_id, folder=None, fps=25):
    import csv
    folder = Path(folder) if folder else (open_data_dir() / "skillcorner" / "data" / "matches"
                                          / str(match_id))
    meta = json.loads((folder / "{}_match.json".format(match_id)).read_text(encoding="utf-8"))
    length, width = float(meta["pitch_length"]), float(meta["pitch_width"])
    home_id, away_id = meta["home_team"]["id"], meta["away_team"]["id"]
    people = [p for p in meta["players"] if p["team_id"] in (home_id, away_id)]
    slot = {p["id"]: k for k, p in enumerate(people)}
    slot_team = np.asarray([H if p["team_id"] == home_id else A for p in people], dtype=np.int8)

    rows = [json.loads(line) for line in
            (folder / "{}_tracking_extrapolated.jsonl".format(match_id)).open(encoding="utf-8")]
    n = max(r["frame"] for r in rows) + 1
    ball = np.full((n, 2), np.nan)
    players = np.full((n, len(people), 2), np.nan)
    owner = np.full(n, -1)
    tracked = np.zeros(n, dtype=bool)
    for r in rows:
        f = r["frame"]
        b = r.get("ball_data") or {}
        if b.get("is_detected") and b.get("x") is not None:   # an extrapolated ball is a guess
            ball[f] = (b["x"] + length / 2, b["y"] + width / 2)
        for p in r.get("player_data") or []:
            k = slot.get(p["player_id"])
            if k is not None and p.get("x") is not None:
                players[f, k] = (p["x"] + length / 2, p["y"] + width / 2)
        tracked[f] = bool(r.get("player_data"))
        group = (r.get("possession") or {}).get("group")
        owner[f] = H if group == "home team" else A if group == "away team" else -1

    period = np.zeros(n, dtype=np.int8)
    for p in meta["match_periods"]:
        period[p["start_frame"]:p["end_frame"] + 1] = p["period"]
    period[~tracked] = 0               # replays and close-ups: no tracking, so no label either

    labels = np.full(n, D, dtype=np.int8)
    with (folder / "{}_phases_of_play.csv".format(match_id)).open(encoding="utf-8") as handle:
        for phase in csv.DictReader(handle):
            current = H if int(phase["team_in_possession_id"]) == home_id else A
            for f in range(int(phase["frame_start"]), min(int(phase["frame_end"]), n)):
                current = owner[f] if owner[f] >= 0 else current
                labels[f] = current
    labels[period == 0] = X

    ratio = fps / SKILLCORNER_FPS
    n_out = int(round(n * ratio))
    nearest = np.minimum(np.round(np.arange(n_out) / ratio).astype(int), n - 1)
    table = MatchTable(name="sc_{}".format(match_id),
                       ball=_upsample(ball, n_out, ratio).astype(np.float32),
                       players=_upsample(players, n_out, ratio).astype(np.float32),
                       slot_team=slot_team, period=period[nearest], pitch=(length, width), fps=fps,
                       meta={"home": meta["home_team"]["name"], "away": meta["away_team"]["name"]})
    table.labels = labels[nearest]
    return table


def skillcorner_ids():
    """Match ids of the SkillCorner open data present on disk (empty if not cloned)."""
    try:
        folder = open_data_dir() / "skillcorner" / "data" / "matches"
    except IOError:
        return []
    return sorted(p.name for p in folder.iterdir() if p.is_dir()) if folder.is_dir() else []


KICKOFFS = ("FIRSTKICKOFF", "SECONDKICKOFF", "THIRDKICKOFF", "FOURTHKICKOFF")
# Bump whenever the PFF loader or its labels change: it is in the PFF table cache's name only, so
# the other open-data caches stay valid. 2: event times shifted by the measured kickoff offset.
PFF_VERSION = 2
MAX_OFFSET_DRIFT_S = 0.5


def pff_offset(events, period, video_fps):
    """Event time minus tracking time at the first kickoff (seconds); raises if the second
    half's kickoff disagrees by more than MAX_OFFSET_DRIFT_S, or if the first is missing."""
    def offset(kind, half):
        kick = next((e for e in events
                     if (e.get("gameEvents") or {}).get("gameEventType") == kind), None)
        frames = np.flatnonzero(period == half)
        if kick is None or not len(frames):
            return None
        return float(kick["startTime"]) - frames[0] / video_fps
    first, second = offset("FIRSTKICKOFF", 1), offset("SECONDKICKOFF", 2)
    if first is None:
        raise ValueError("no first kickoff or no period-1 frame")
    if second is not None and abs(second - first) > MAX_OFFSET_DRIFT_S:
        raise ValueError("kickoff offsets {:+.3f} s and {:+.3f} s disagree".format(first, second))
    return first


def pff_labels(events, n_frames, fps, home_id, offset=0.0):
    """(n_frames,) H/A/D at the tracking's video frame rate from PFF events (plan row PFF):
    alive possession = team of the latest OTB; dead from OUT / END to the next restart (a
    kickoff or an OTB with a set-piece type other than open play "O"), and from the end of the
    previous event to any set-piece restart (a foul has no OUT before its free kick); dead
    before the first kickoff. OFF and SUB change nothing. `offset`: seconds subtracted from every
    event time, so events land on the tracking's frames. Returns (labels, open-play OTBs that
    ended a dead stretch without a restart tag)."""
    def frame(t):
        return min(max(int(round((float(t) - offset) * fps)), 0), n_frames)

    events = sorted((e for e in events if e.get("startTime") is not None),
                    key=lambda e: float(e["startTime"]))
    labels = np.full(n_frames, D, dtype=np.int8)
    changes, gaps, untagged = [], [], 0
    dead, prev_end = True, 0
    for e in events:
        g = e.get("gameEvents") or {}
        kind, start = g.get("gameEventType"), frame(e["startTime"])
        side = H if str(g.get("teamId")) == str(home_id) else A   # ids: int in events, str in metadata
        restart = kind in KICKOFFS or (kind == "OTB" and g.get("setpieceType") not in (None, "O"))
        if kind in ("OUT", "END"):
            changes.append((start, D))
            dead = True
        elif kind in KICKOFFS or kind == "OTB":
            if restart and kind == "OTB":
                gaps.append((prev_end, start))
            elif dead and not restart:
                untagged += 1      # an open-play touch while dead: play has resumed regardless
            changes.append((start, side))
            dead = False
        if e.get("endTime") is not None:
            prev_end = frame(e["endTime"])
    for (start, value), nxt in zip(changes, changes[1:] + [(n_frames, None)]):
        labels[start:nxt[0]] = value
    for a, b in gaps:
        labels[a:b] = D
    return labels, untagged


def _pff_dir():
    return open_data_dir() / "pff_wc2022"


def pff_ids():
    """Game ids whose PFF tracking file is on disk (the download lands file by file)."""
    try:
        folder = _pff_dir() / "tracking_data"
    except IOError:
        return []
    return sorted(p.name.split(".")[0] for p in folder.glob("*.jsonl.bz2")) if folder.is_dir() \
        else []


def pff_game(game_id, folder=None, fps=25):
    """A labelled MatchTable from PFF FC's World Cup 2022 data: tracking_data/<id>.jsonl.bz2,
    metadata/<id>.json, rosters/<id>.json, event_data/*/<id>.json. Test condition only."""
    import bz2
    folder = Path(folder) if folder else _pff_dir()
    meta = json.loads((folder / "metadata" / "{}.json".format(game_id)).read_text(
        encoding="utf-8"))
    meta = meta[0] if isinstance(meta, list) else meta
    roster = json.loads((folder / "rosters" / "{}.json".format(game_id)).read_text(
        encoding="utf-8"))
    events_path = next((folder / "event_data").rglob("{}.json".format(game_id)))
    events = json.loads(events_path.read_text(encoding="utf-8"))
    home_id, away_id = meta["homeTeam"]["id"], meta["awayTeam"]["id"]
    pitch = meta["stadium"]["pitches"][0]
    length, width = float(pitch["length"]), float(pitch["width"])
    video_fps = float(meta["fps"])
    home_id, away_id = str(home_id), str(away_id)
    people = [(H if str(r["team"]["id"]) == home_id else A, int(r["shirtNumber"])) for r in roster
              if str(r["team"]["id"]) in (home_id, away_id)]
    people = sorted(set(people))
    slot = {p: k for k, p in enumerate(people)}
    slot_team = np.asarray([team for team, _ in people], dtype=np.int8)

    with bz2.open(folder / "tracking_data" / "{}.jsonl.bz2".format(game_id), "rt",
                  encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    n = max(int(r["frameNum"]) for r in rows) + 1
    ball = np.full((n, 2), np.nan)
    players = np.full((n, len(people), 2), np.nan)
    period = np.zeros(n, dtype=np.int8)
    for r in rows:
        f = int(r["frameNum"])
        if r.get("period") in (1, 2):
            period[f] = r["period"]
        b = r.get("ballsSmoothed") or {}
        # a smoothed ball that the broadcast does not show is a guess: missing, as for SkillCorner
        if b.get("x") is not None and b.get("y") is not None                 and b.get("visibility", "VISIBLE") == "VISIBLE":
            ball[f] = (float(b["x"]) + length / 2, float(b["y"]) + width / 2)
        for key, team in (("homePlayersSmoothed", H), ("awayPlayersSmoothed", A)):
            for p in r.get(key) or []:
                k = slot.get((team, int(p["jerseyNum"])))
                if k is not None and p.get("x") is not None and p.get("y") is not None:
                    players[f, k] = (float(p["x"]) + length / 2, float(p["y"]) + width / 2)

    try:
        offset = pff_offset(events, period, video_fps)
    except ValueError as error:
        raise ValueError("pff {}: {}".format(game_id, error))
    if abs(offset) > 2.0:
        raise ValueError("pff {}: events and tracking {:+.1f} s apart".format(game_id, offset))
    labels, untagged = pff_labels(events, n, video_fps, home_id, offset)
    labels[period == 0] = X

    ratio = fps / video_fps
    n_out = int(round(n * ratio))
    nearest = np.minimum(np.round(np.arange(n_out) / ratio).astype(int), n - 1)
    table = MatchTable(name="pff_{}".format(game_id), ball=ball[nearest].astype(np.float32),
                       players=players[nearest].astype(np.float32), slot_team=slot_team,
                       period=period[nearest], pitch=(length, width), fps=fps,
                       meta={"home": meta["homeTeam"]["name"], "away": meta["awayTeam"]["name"],
                             "kickoff_offset_s": round(offset, 3), "untagged_resumes": untagged})
    table.labels = labels[nearest]
    return table


LOADERS = {
    "metrica1": lambda: metrica_csv_game(1),
    "metrica2": lambda: metrica_csv_game(2),
    "metrica3": metrica_epts_game3,
}
LOADERS.update({"dfl_" + m: (lambda m=m: dfl_game(m)) for m in DFL_MATCHES})
# Extra training-only matches: never part of the default cross-validation folds, so scores stay
# comparable with runs made before they existed.
EXTRA_LOADERS = {"sc_" + i: (lambda i=i: skillcorner_game(i)) for i in skillcorner_ids()}


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


# Bump whenever a loader or the label code changes: it is in the table cache's and the clean
# features' file names, so stale tables and features are never read back.
LOADER_VERSION = 1


def load_open_match(name, cache_dir=None):
    """A labelled open MatchTable, cached as .npz under the open-data folder."""
    cache = Path(cache_dir) if cache_dir else open_data_dir() / "cache"
    path = cache / "{}_l{}{}.npz".format(
        name, LOADER_VERSION, "_p{}".format(PFF_VERSION) if name.startswith("pff_") else "")
    if path.exists():
        table = MatchTable.load(path)
    else:
        if name.startswith("pff_"):        # test condition only: never in LOADERS / EXTRA_LOADERS
            table = pff_game(name[4:])
        else:
            table = {**LOADERS, **EXTRA_LOADERS}[name]()
        table.save(path)
    if name.startswith(("dfl_", "sc_", "pff_")) and DFL_MIN_SPELL:
        table.labels = remove_flicker(table.labels, table.runs(), DFL_MIN_SPELL)
    if name.startswith("metrica"):
        table = fill_dead_ball(table)
    return table
