"""One per-frame representation of a match, shared by the competition ATD and open datasets.

Everything downstream -- the noise simulator, the features, the models -- reads a MatchTable and
nothing else, so a model trained on open data sees exactly the same inputs as the competition
match. Positions are metres on a pitch of `pitch` = (length, width); missing values are NaN.
Player slots are anonymous: a slot is a column, not a person.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET

import numpy as np

H, A, D, X = 0, 1, 2, 3
LABELS = np.array(["H", "A", "D", "X"])


@dataclass
class MatchTable:
    name: str
    ball: np.ndarray            # (n, 2) float32 metres
    players: np.ndarray         # (n, slots, 2) float32 metres
    slot_team: np.ndarray       # (slots,) int8, H or A
    period: np.ndarray          # (n,) int8, 1/2 inside a scored half, 0 elsewhere
    pitch: tuple = (105.0, 68.0)
    fps: int = 25
    labels: Optional[np.ndarray] = None   # (n,) int8 H/A/D, X where unknown
    meta: dict = field(default_factory=dict)

    @property
    def n_frames(self):
        return len(self.ball)

    def runs(self):
        """[(start, stop)] half-open spans of each period, in frame order."""
        inside = (self.period > 0).astype(np.int8)
        edges = np.diff(np.concatenate(([0], inside, [0])))
        starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        # Split where the period number changes without a gap between the halves.
        out = []
        for start, stop in zip(starts, stops):
            cuts = np.flatnonzero(np.diff(self.period[start:stop]) != 0) + 1 + start
            bounds = np.concatenate(([start], cuts, [stop]))
            out.extend(zip(bounds[:-1].tolist(), bounds[1:].tolist()))
        return out

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, name=self.name, ball=self.ball, players=self.players,
            slot_team=self.slot_team, period=self.period, pitch=np.asarray(self.pitch),
            fps=self.fps,
            labels=self.labels if self.labels is not None else np.zeros(0, dtype=np.int8))

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        labels = z["labels"]
        return cls(name=str(z["name"]), ball=z["ball"], players=z["players"],
                   slot_team=z["slot_team"], period=z["period"],
                   pitch=tuple(float(v) for v in z["pitch"]), fps=int(z["fps"]),
                   labels=labels if len(labels) else None)


def teams_from_epts_metadata(meta_path):
    """(slot -> H/A codes, home name, away name). The first team listed is home (EPTS)."""
    root = ET.parse(meta_path).getroot()
    teams = list(root.iter("Team"))
    if len(teams) != 2:
        raise ValueError("expected two teams in {}".format(meta_path))
    home_id = teams[0].get("id")
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError("no Player slots in {}".format(meta_path))
    codes = np.asarray([H if slot == home_id else A for slot in slots], dtype=np.int8)
    return codes, teams[0].findtext("Name") or "Home", teams[1].findtext("Name") or "Away"


def _xy(group):
    """x,y from one 'x,y[,speed]' group, or None when missing or malformed."""
    if not group or group[0] == "N":
        return None
    values = group.split(",")
    if len(values) < 2:
        return None
    try:
        return float(values[0]), float(values[1])
    except ValueError:
        return None


def read_atd(meta_path, raw_path, n_frames, pitch, name="match", period_blocks=()):
    """Stream a single-format EPTS export ('frame:p;p;...:ball', frame ids from 1) to a table.

    `period_blocks` is [(period, first_frame, last_frame)] in zero-based frames, inclusive;
    frames outside every block get period 0.
    """
    slot_team, home, away = teams_from_epts_metadata(meta_path)
    length, width = pitch
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    players = np.full((n_frames, len(slot_team), 2), np.nan, dtype=np.float32)
    read = np.zeros(n_frames, dtype=bool)
    with open(raw_path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            try:
                frame = int(parts[0]) - 1   # tracking counts from 1, frames from 0
            except ValueError:
                continue
            if not 0 <= frame < n_frames:
                continue
            read[frame] = True
            point = _xy(parts[2])
            if point is not None:
                ball[frame] = (point[0] * length, point[1] * width)
            for slot, group in enumerate(parts[1].split(";")[:len(slot_team)]):
                point = _xy(group)
                if point is not None:
                    players[frame, slot] = (point[0] * length, point[1] * width)
    # A truncated or reformatted file would otherwise become NaN rows and a normal-looking CSV.
    if not read.all():
        raise ValueError("{} has records for {} of {} frames (first missing: {})".format(
            raw_path, int(read.sum()), n_frames, int(np.flatnonzero(~read)[0])))
    period = np.zeros(n_frames, dtype=np.int8)
    for half, first, last in period_blocks:
        period[first:last + 1] = half
    return MatchTable(name=name, ball=ball, players=players, slot_team=slot_team,
                      period=period, pitch=tuple(pitch),
                      meta={"home": home, "away": away})


def competition_table(game=None, data_dir=None):
    """The tactical ATD of the competition match as a MatchTable (no labels)."""
    from games import GAME, GAMES, half_video_blocks, require, source_frames
    game = game or GAME
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    return read_atd(meta_path, raw_path, source_frames("tactical", game), GAMES[game]["pitch"],
                    name=game, period_blocks=half_video_blocks(game))


# -- goalkeeper slots (plan row E1) ------------------------------------------------------------

KEEPER_DEPTH_M, KEEPER_HALF_WIDTH_M = 20.0, 20.0   # within 20 m of the goal line, central 40 m
KEEPER_REPAIR = dict(window_s=60.0, occupancy=0.8, present=0.5)


def keeper_zones(table):
    """(n, slots, 2) bool: each slot inside the keeper zone of team 0's (home) / team 1's goal."""
    from .features import attack_sign
    length, width = table.pitch
    x = table.players[..., 0] - length / 2.0
    near_y = np.abs(table.players[..., 1] - width / 2.0) <= KEEPER_HALF_WIDTH_M
    home_goal = (-attack_sign(table) * length / 2.0)[:, None]   # home attacks +x when sign > 0
    return np.stack([near_y & (np.abs(x - home_goal) <= KEEPER_DEPTH_M),
                     near_y & (np.abs(x + home_goal) <= KEEPER_DEPTH_M)], axis=-1)


def keeper_windows(table, window_s=60.0, occupancy=0.8, present=0.5, zones=None):
    """[(start, stop, slot, goal_team)]: windows (consecutive, inside each half) in which a slot of
    the other team spends >= `occupancy` of the frames in goal_team's keeper zone while no slot of
    goal_team is there for >= `present` of them -- a keeper filed under the wrong team."""
    zones = keeper_zones(table) if zones is None else zones
    size, out = int(window_s * table.fps), []
    for a, b in table.runs():
        for s in range(a, b, size):
            e = min(b, s + size)
            occ = zones[s:e].mean(axis=0)
            for goal in (0, 1):
                own = table.slot_team == goal
                if (occ[own, goal] >= present).any():
                    continue
                out += [(s, e, int(k), goal) for k in np.flatnonzero(~own & (occ[:, goal] >= occupancy))]
    return out


def repair_keepers(table, window_s=60.0, occupancy=0.8, present=0.5):
    """A copy in which each `keeper_windows` track is given to the goal's team on the frames of its
    window where it stands in that goal's zone: it swaps with that team's slot sitting in the
    other goal's zone (a swapped keeper pair), else moves to a slot of that team with no data in
    the window, else is dropped there."""
    from dataclasses import replace
    zones = keeper_zones(table)
    players, done = table.players.copy(), set()
    for s, e, k, goal in keeper_windows(table, window_s, occupancy, present, zones):
        if (s, k) in done:
            continue
        team = table.slot_team == goal
        wrong = s + np.flatnonzero(zones[s:e, k, goal])
        occ = zones[s:e].mean(axis=0)[:, 1 - goal]
        pair = np.flatnonzero(team & (occ >= occupancy))
        free = np.flatnonzero(team & ~np.isfinite(players[s:e, :, 0]).any(axis=0))
        if len(pair):
            j = int(pair[np.argmax(occ[pair])])
            players[wrong[:, None], [k, j]] = players[wrong[:, None], [j, k]]
            done |= {(s, k), (s, j)}
        elif len(free):
            players[wrong, free[0]] = players[wrong, k]
            players[wrong, k] = np.nan
        else:
            players[wrong, k] = np.nan
    return replace(table, players=players)
