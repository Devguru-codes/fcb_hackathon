"""Low-memory feature extraction from Metrica's raw EPTS tracking export."""

from dataclasses import dataclass
import xml.etree.ElementTree as ET

import numpy as np

from games import GAME, GAMES, require, source_frames

HOME, AWAY = 0, 1


@dataclass
class TrackingFeatures:
    """Frame-aligned features. Distances are in metres; missing values are NaN."""

    ball_seen: np.ndarray
    ball_x: np.ndarray
    ball_y: np.ndarray
    ball_age: np.ndarray
    nearest_home_m: np.ndarray
    nearest_away_m: np.ndarray
    close_home: np.ndarray
    close_away: np.ndarray
    tracked_players: np.ndarray


def teams_from_metadata(meta_path):
    """Return (slot -> H/A code, home name, away name) from the supplied EPTS metadata."""
    root = ET.parse(meta_path).getroot()
    teams = list(root.iter("Team"))
    if len(teams) != 2:
        raise ValueError("expected two teams in {}".format(meta_path))
    home_id = teams[0].get("id")
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError("no Player slots in {}".format(meta_path))
    codes = np.asarray([HOME if slot == home_id else AWAY for slot in slots], dtype=np.int8)
    return codes, teams[0].findtext("Name") or "Home", teams[1].findtext("Name") or "Away"


def _point(group):
    """Read x,y from one EPTS group; return None for missing or malformed observations."""
    if not group or group.startswith("N"):
        return None
    values = group.split(",")
    if len(values) < 2:
        return None
    try:
        return float(values[0]), float(values[1])
    except ValueError:
        return None


def extract_features(game=GAME, data_dir=None, close_control_m=3.5):
    """Stream raw tracking once and derive features without loading the video or kloppy.

    Track slots are deliberately never treated as player identities. They can disappear and swap;
    each frame is reduced to team-level geometry around the ball.
    """
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    slot_teams, home_name, away_name = teams_from_metadata(meta_path)
    n_frames = source_frames("tactical", game)
    length, width = GAMES[game]["pitch"]

    ball_seen = np.zeros(n_frames, dtype=bool)
    ball_x = np.full(n_frames, np.nan, dtype=np.float32)
    ball_y = np.full(n_frames, np.nan, dtype=np.float32)
    nearest_home = np.full(n_frames, np.nan, dtype=np.float32)
    nearest_away = np.full(n_frames, np.nan, dtype=np.float32)
    close_home = np.zeros(n_frames, dtype=np.int8)
    close_away = np.zeros(n_frames, dtype=np.int8)
    tracked_players = np.zeros(n_frames, dtype=np.int8)

    with open(raw_path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split(":")
            if len(parts) != 3:
                continue
            try:
                frame = int(parts[0]) - 1  # tracking is one-indexed; submission is zero-indexed
            except ValueError:
                continue
            if frame < 0 or frame >= n_frames:
                continue
            ball = _point(parts[2])
            players = parts[1].split(";")
            home_distances, away_distances = [], []
            if ball is not None:
                bx, by = ball[0] * length, ball[1] * width
                ball_seen[frame], ball_x[frame], ball_y[frame] = True, bx, by
            for slot, group in enumerate(players):
                player = _point(group)
                if player is None:
                    continue
                tracked_players[frame] += 1
                if ball is None or slot >= len(slot_teams):
                    continue
                dx, dy = player[0] * length - bx, player[1] * width - by
                distance = float(np.hypot(dx, dy))
                (home_distances if slot_teams[slot] == HOME else away_distances).append(distance)
            if home_distances:
                nearest_home[frame] = min(home_distances)
                close_home[frame] = sum(d <= close_control_m for d in home_distances)
            if away_distances:
                nearest_away[frame] = min(away_distances)
                close_away[frame] = sum(d <= close_control_m for d in away_distances)

    # Number of frames since the ball was last observed. This lets the decoder distinguish a brief
    # occlusion from a prolonged absence without pretending either means a dead ball.
    ages = np.empty(n_frames, dtype=np.int32)
    last_seen = -1
    for frame in range(n_frames):
        if ball_seen[frame]:
            last_seen = frame
        ages[frame] = frame - last_seen if last_seen >= 0 else n_frames
    return TrackingFeatures(ball_seen, ball_x, ball_y, ages, nearest_home, nearest_away,
                            close_home, close_away, tracked_players), home_name, away_name
