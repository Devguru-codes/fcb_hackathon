"""Per-frame, identity-free, team-level features from a MatchTable.

Design rules:
- No feature depends on which slot a player is in: slots are tracklets, not people.
- Features are written from the point of view of "team 0" and "team 1". Calling
  `frame_features(table, swap=True)` swaps the two, which is how training is made symmetric
  (a model must not learn that one side of a particular match has the ball more).
- Coordinates are oriented so team 0 attacks towards +x in every period, inferred from where each
  team stands, not from metadata.
- Offline: the whole match is available, so context windows look both backwards and forwards.
"""

import warnings

import numpy as np

from .match import A, H

FEATURE_VERSION = 4       # bump whenever a feature changes, to invalidate on-disk caches
MAX_SPEED = 12.0          # m/s; faster "players" are identity switches or teleports
MAX_BALL_SPEED = 40.0     # m/s
IMPUTE_GAP = 50           # frames; ball gaps up to this long are linearly interpolated
WINDOWS = (12, 50, 125)   # half-widths in frames for centred context


# -- small numpy helpers -----------------------------------------------------------------------

def _velocity(xy, fps, max_speed):
    """Central-difference velocity of an (n, ..., 2) track; NaN across gaps and teleports."""
    v = np.full_like(xy, np.nan)
    v[1:-1] = (xy[2:] - xy[:-2]) * (fps / 2.0)
    speed = np.hypot(v[..., 0], v[..., 1])
    v[speed > max_speed] = np.nan
    return v


def _nan_window_mean(x, half):
    """Centred mean over +/-half frames ignoring NaN; NaN where the window has no data."""
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    cs = np.concatenate(([0.0], np.cumsum(np.where(finite, x, 0.0))))
    cn = np.concatenate(([0], np.cumsum(finite)))
    idx = np.arange(len(x))
    lo, hi = np.maximum(0, idx - half), np.minimum(len(x), idx + half + 1)
    count = cn[hi] - cn[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, (cs[hi] - cs[lo]) / np.maximum(count, 1), np.nan)


def _one_sided_mean(x, half, forward):
    """Mean over the next (forward) or previous `half` frames, excluding the frame itself."""
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    cs = np.concatenate(([0.0], np.cumsum(np.where(finite, x, 0.0))))
    cn = np.concatenate(([0], np.cumsum(finite)))
    idx = np.arange(len(x))
    if forward:
        lo, hi = np.minimum(len(x), idx + 1), np.minimum(len(x), idx + 1 + half)
    else:
        lo, hi = np.maximum(0, idx - half), idx
    count = cn[hi] - cn[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, (cs[hi] - cs[lo]) / np.maximum(count, 1), np.nan)


def _nan_window_max(x, half):
    """Centred max over +/-half frames ignoring NaN; NaN where the window has no data."""
    x = np.asarray(x, dtype=np.float64)
    padded = np.pad(x, half, constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN windows -> NaN, by design
        return np.nanmax(windows, axis=1)


def _carry(values, forward):
    """Last (forward) or next (backward) finite value at every frame; NaN before/after any."""
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    idx = np.arange(n)
    finite = np.isfinite(values)
    if forward:
        pick = np.maximum.accumulate(np.where(finite, idx, -1))
        ok = pick >= 0
    else:
        pick = np.minimum.accumulate(np.where(finite, idx, n)[::-1])[::-1]
        ok = pick < n
    out = np.full(n, np.nan)
    out[ok] = values[pick[ok]]
    return out


def _sorted_padded(values, k):
    """Each row's finite values sorted ascending, NaN-padded (or truncated) to k columns."""
    filled = np.where(np.isfinite(values), values, np.inf)
    ordered = np.sort(filled, axis=1)[:, :k]
    if ordered.shape[1] < k:
        ordered = np.pad(ordered, ((0, 0), (0, k - ordered.shape[1])), constant_values=np.inf)
    return np.where(np.isfinite(ordered), ordered, np.nan)


def _row_quantile(values, q):
    """Per-row quantile ignoring NaN, linear interpolation (numpy's default), all-NaN -> NaN.

    np.nanpercentile along an axis loops in Python per row; this sorts once instead.
    """
    ordered = np.sort(np.where(np.isfinite(values), values, np.inf), axis=1)
    count = np.isfinite(values).sum(axis=1)
    pos = q * np.maximum(count - 1, 0)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, np.maximum(count - 1, 0))
    rows = np.arange(len(values))
    a, b = ordered[rows, lo], ordered[rows, hi]
    with np.errstate(invalid="ignore"):
        out = a + (b - a) * (pos - lo)
    out[count == 0] = np.nan
    return out


def _gap_lengths(seen):
    """Per frame: length of the missing-ball gap it sits in (0 if seen), frames since/until seen."""
    n = len(seen)
    idx = np.arange(n)
    last = np.maximum.accumulate(np.where(seen, idx, -1))
    nxt = np.minimum.accumulate(np.where(seen, idx, n)[::-1])[::-1]
    since = np.where(last >= 0, idx - last, n).astype(np.float64)
    until = np.where(nxt < n, nxt - idx, n).astype(np.float64)
    gap = np.where(seen, 0, np.minimum(since + until - 1, n)).astype(np.float64)
    return gap, since, until


def impute_ball(ball, runs, max_gap=IMPUTE_GAP):
    """Linear interpolation of the ball across short gaps, never across a period boundary."""
    out = ball.astype(np.float64).copy()
    imputed = np.zeros(len(ball), dtype=bool)
    for start, stop in runs:
        seg = out[start:stop]
        seen = np.isfinite(seg[:, 0])
        if seen.sum() < 2:
            continue
        gap, _, _ = _gap_lengths(seen)
        idx = np.arange(len(seg))
        known = np.flatnonzero(seen)
        fill = (~seen) & (gap <= max_gap) & (idx > known[0]) & (idx < known[-1])
        for axis in (0, 1):
            seg[fill, axis] = np.interp(idx[fill], known, seg[known, axis])
        imputed[start:stop] = fill
    return out, imputed


def attack_sign(table):
    """Per frame +1 if team 0 (home) attacks towards +x in that period, else -1."""
    sign = np.ones(table.n_frames, dtype=np.float64)
    px = table.players[..., 0]
    home = table.slot_team == H
    for start, stop in table.runs():
        hx = np.nanmedian(np.nanmean(px[start:stop][:, home], axis=1))
        ax = np.nanmedian(np.nanmean(px[start:stop][:, ~home], axis=1))
        # A team stands, on average, in its own half: home defending low x means attacking +x.
        sign[start:stop] = 1.0 if hx < ax else -1.0
    return sign


# -- the feature table -------------------------------------------------------------------------

def frame_features(table, swap=False):
    """(features (n, k) float32, names). `swap` exchanges the roles of the two teams."""
    fps = table.fps
    length, width = table.pitch
    runs = table.runs()
    team0 = (table.slot_team == (A if swap else H))
    sign = attack_sign(table) * (-1.0 if swap else 1.0)

    players = table.players.astype(np.float64)
    pvel = _velocity(players, fps, MAX_SPEED)
    pspeed = np.hypot(pvel[..., 0], pvel[..., 1])

    seen = np.isfinite(table.ball[:, 0])
    ball, imputed = impute_ball(table.ball, runs)
    have = np.isfinite(ball[:, 0])
    bvel = _velocity(ball, fps, MAX_BALL_SPEED)
    bspeed = np.hypot(bvel[:, 0], bvel[:, 1])
    bacc = np.full(len(ball), np.nan)
    bacc[1:-1] = np.hypot(*(bvel[2:] - bvel[:-2]).T) * (fps / 2.0)
    gap, since, until = _gap_lengths(seen)

    # Oriented coordinates: x towards team 0's attacking goal, measured from the centre.
    bx = (ball[:, 0] - length / 2.0) * sign
    by = np.abs(ball[:, 1] - width / 2.0)

    dist = np.hypot(players[..., 0] - ball[:, None, 0], players[..., 1] - ball[:, None, 1])
    feats = {}
    feats["ball_seen"] = seen.astype(np.float64)
    feats["ball_imputed"] = imputed.astype(np.float64)
    feats["ball_have"] = have.astype(np.float64)
    feats["gap_len"] = np.minimum(gap, 2000)
    feats["since_seen"] = np.minimum(since, 2000)
    feats["until_seen"] = np.minimum(until, 2000)
    feats["ball_x"] = bx
    feats["ball_y_abs"] = by
    feats["ball_speed"] = bspeed
    feats["ball_acc"] = bacc
    feats["ball_out"] = (have & ((np.abs(ball[:, 0] - length / 2) > length / 2 + 0.5)
                                 | (np.abs(ball[:, 1] - width / 2) > width / 2 + 0.5))
                         ).astype(np.float64)
    feats["ball_to_touchline"] = width / 2.0 - by
    feats["ball_to_goalline"] = length / 2.0 - np.abs(bx)
    feats["ball_to_centre"] = np.hypot(bx, ball[:, 1] - width / 2.0)
    feats["n_tracked"] = np.isfinite(players[..., 0]).sum(axis=1).astype(np.float64)

    team_cols = {}
    for tag, mask in (("t0", team0), ("t1", ~team0)):
        d = dist[:, mask]
        d_sorted = np.sort(np.where(np.isfinite(d), d, np.inf), axis=1)
        d1 = np.where(np.isfinite(d_sorted[:, 0]), d_sorted[:, 0], np.nan)
        d2 = (np.where(np.isfinite(d_sorted[:, 1]), d_sorted[:, 1], np.nan)
              if d.shape[1] > 1 else np.full(len(d), np.nan))
        nearest = np.argmin(np.where(np.isfinite(d), d, np.inf), axis=1)
        rows = np.arange(len(d))
        # With no ball (or nobody visible) argmin picks the team's first slot, an arbitrary
        # tracklet: blank those frames rather than report a stranger's speed.
        known = np.isfinite(d1)
        nv = pvel[:, mask][rows, nearest]
        rel = np.where(known, np.hypot(nv[:, 0] - bvel[:, 0], nv[:, 1] - bvel[:, 1]), np.nan)
        px = (players[:, mask, 0] - length / 2.0) * sign[:, None]
        feats[tag + "_d1"] = d1
        feats[tag + "_d2"] = d2
        for r in (1.5, 3.0, 6.0):
            feats["{}_within{}".format(tag, r)] = (d <= r).sum(axis=1).astype(np.float64)
        feats[tag + "_near_speed"] = np.where(known, pspeed[:, mask][rows, nearest], np.nan)
        feats[tag + "_near_rel_speed"] = rel
        feats[tag + "_centroid_x"] = np.nanmean(px, axis=1)
        feats[tag + "_spread_x"] = np.nanstd(px, axis=1)
        feats[tag + "_spread_y"] = np.nanstd(players[:, mask, 1], axis=1)
        feats[tag + "_mean_speed"] = np.nanmean(pspeed[:, mask], axis=1)
        feats[tag + "_centroid_vx"] = np.nanmean(pvel[:, mask, 0], axis=1) * sign
        # Share of the *visible* players: a boolean mean would divide by the slot count, which
        # is ~15 per team in open data and 46 in all on the competition ATD.
        own = np.where(np.isfinite(px), (px < 0) if tag == "t0" else (px > 0), np.nan)
        feats[tag + "_own_half"] = np.nanmean(own, axis=1)
        feats[tag + "_n"] = np.isfinite(players[:, mask, 0]).sum(axis=1).astype(np.float64)
        # Formation, slot-free: each team's players sorted along each axis. Who is pushed up,
        # who is spread wide, and who is running forwards, readable without the ball.
        sx = _sorted_padded(px, 11)
        sy = _sorted_padded(players[:, mask, 1] - width / 2.0, 11)
        for i in range(11):
            feats["{}_x{}".format(tag, i)] = sx[:, i]
            feats["{}_y{}".format(tag, i)] = sy[:, i]
        feats[tag + "_block_len"] = np.nanmax(px, axis=1) - np.nanmin(px, axis=1)
        feats[tag + "_block_wid"] = (np.nanmax(players[:, mask, 1], axis=1)
                                     - np.nanmin(players[:, mask, 1], axis=1))
        vx = pvel[:, mask, 0] * sign[:, None]
        for q in (10, 50, 90):
            feats["{}_vx_q{}".format(tag, q)] = _row_quantile(vx, q / 100.0)
        team_cols[tag] = d1

    d0, d1_ = team_cols["t0"], team_cols["t1"]
    feats["d_margin"] = d1_ - d0                       # > 0: team 0 is nearer
    feats["nearest_is_t0"] = np.where(np.isfinite(d0) & np.isfinite(d1_),
                                      (d0 < d1_).astype(np.float64), np.nan)
    feats["d_min"] = np.fmin(d0, d1_)
    feats["mean_speed_all"] = np.nanmean(pspeed, axis=1)
    feats["d_min_ddt"] = np.r_[np.nan, np.diff(feats["d_min"])] * fps

    # Context: centred windows, plus the second before and after, of the most telling signals.
    for key in ("nearest_is_t0", "d_margin", "ball_seen", "ball_speed", "mean_speed_all",
                "d_min", "t0_centroid_vx", "t1_centroid_vx", "ball_x"):
        for half in WINDOWS:
            feats["{}_w{}".format(key, half)] = _nan_window_mean(feats[key], half)
    for key in ("nearest_is_t0", "d_margin", "mean_speed_all", "ball_seen"):
        feats[key + "_past25"] = _one_sided_mean(feats[key], 25, forward=False)
        feats[key + "_next25"] = _one_sided_mean(feats[key], 25, forward=True)

    # Memory across ball gaps, which last ~10 s on the competition tracking: who was nearest and
    # where the ball was just before it vanished, and just after it comes back.
    near_smooth = np.where(seen, _nan_window_mean(np.where(seen, feats["nearest_is_t0"], np.nan),
                                                  12), np.nan)
    ball_x_seen = np.where(seen, bx, np.nan)
    ball_y_seen = np.where(seen, by, np.nan)
    def carry_in_runs(values, forward):
        out = np.full(len(values), np.nan)
        for start, stop in runs:
            out[start:stop] = _carry(values[start:stop], forward)
        return out

    for suffix, forward in (("last", True), ("next", False)):
        feats["nearest_" + suffix] = carry_in_runs(near_smooth, forward)
        feats["ball_x_" + suffix] = carry_in_runs(ball_x_seen, forward)
        feats["ball_y_" + suffix] = carry_in_runs(ball_y_seen, forward)
    feats["gap_progress"] = np.where(gap > 0, since / np.maximum(gap, 1), 0.0)

    # Restarts. The stoppages missed on the real match are set pieces with the ball visible and
    # still: placed in the goal area for a goal kick, held on the touchline for a throw-in, at a
    # corner flag. Fastest ball within +/-1 s (a placed ball stays slow until the kick), the
    # restart spots, and how recently the ball was on or over a line (it went out of play).
    feats["ball_speed_max_w25"] = _nan_window_max(bspeed, 25)
    feats["ball_in_goal_area"] = (have & (np.abs(bx) > length / 2.0 - 5.5)
                                  & (by < 9.16)).astype(np.float64)
    feats["ball_at_corner"] = (have & (np.abs(bx) > length / 2.0 - 2.0)
                               & (by > width / 2.0 - 2.0)).astype(np.float64)
    on_line = have & ((np.abs(bx) > length / 2.0 - 0.5) | (by > width / 2.0 - 0.5))
    _, since_line, until_line = _gap_lengths(on_line)
    feats["since_on_line"] = np.minimum(since_line, 750)
    feats["until_on_line"] = np.minimum(until_line, 750)

    # Frames outside every period are never scored or trained on; blank them. Windows can still
    # straddle two adjacent halves for a few seconds, which only touches the halves' dead edges.
    names = list(feats)
    matrix = np.column_stack([feats[name] for name in names]).astype(np.float32)
    matrix[table.period == 0] = np.nan
    return matrix, names


PLAYERS_PER_TEAM = 12      # token slots per team; 11 on the pitch plus one for a duplicate track
TOKEN_FIELDS = ("present", "x", "y", "vx", "vy", "dx_ball", "dy_ball", "d_ball", "ball_seen")


def player_tokens(table, swap=False, per_team=PLAYERS_PER_TEAM):
    """(n, 2*per_team, len(TOKEN_FIELDS)) float16: each frame's players as a set.

    The first `per_team` tokens are the perspective team's players, the rest the opponents',
    each sorted by oriented x and zero-padded with present = 0. Scaled to roughly unit range.
    A per-frame set/graph encoder reads these instead of hand-made formation features; slots
    carry no identity, so nothing here assumes a track is a person.
    """
    fps = table.fps
    length, width = table.pitch
    sign = attack_sign(table) * (-1.0 if swap else 1.0)
    players = table.players.astype(np.float64)
    vel = _velocity(players, fps, MAX_SPEED)
    ball = table.ball.astype(np.float64)
    seen = np.isfinite(ball[:, 0])
    team0 = table.slot_team == (A if swap else H)
    n = table.n_frames
    out = np.zeros((n, 2 * per_team, len(TOKEN_FIELDS)), dtype=np.float32)
    for block, mask in enumerate((team0, ~team0)):
        px = (players[:, mask, 0] - length / 2.0) * sign[:, None]
        py = players[:, mask, 1] - width / 2.0
        present = np.isfinite(px)
        order = np.argsort(np.where(present, px, np.inf), axis=1)[:, :per_team]
        take = lambda a: np.take_along_axis(a, order, axis=1)
        pres = take(present)
        dx = take(players[:, mask, 0] - ball[:, None, 0]) * sign[:, None]
        dy = take(players[:, mask, 1] - ball[:, None, 1])
        fields = [pres, take(px) / 52.5, take(py) / 34.0,
                  take(vel[:, mask, 0] * sign[:, None]) / 8.0, take(vel[:, mask, 1]) / 8.0,
                  dx / 20.0, dy / 20.0, np.hypot(dx, dy) / 20.0,
                  np.repeat(seen[:, None], pres.shape[1], axis=1)]
        k = pres.shape[1]
        for f, values in enumerate(fields):
            values = np.where(pres, np.nan_to_num(values, nan=0.0), 0.0)
            out[:, block * per_team:block * per_team + k, f] = values
    out[table.period == 0] = 0.0
    return np.clip(out, -10, 10).astype(np.float16)


def hidden_ball_target(clean, noisy, swap=False):
    """(n, 2) float32: the true ball, oriented like the features, where the noisy copy hid it.

    NaN wherever the clean ball is unknown or the noisy copy still shows it -- the supervision
    for an auxiliary head that learns to place a ball it cannot see from how the players move.
    """
    length, width = clean.pitch
    sign = attack_sign(noisy) * (-1.0 if swap else 1.0)
    target = np.full((clean.n_frames, 2), np.nan, dtype=np.float32)
    hidden = np.isfinite(clean.ball[:, 0]) & ~np.isfinite(noisy.ball[:, 0]) & (clean.period > 0)
    target[hidden, 0] = (clean.ball[hidden, 0] - length / 2.0) * sign[hidden] / 52.5
    target[hidden, 1] = (clean.ball[hidden, 1] - width / 2.0) / 34.0
    return target


class TableSample:
    """A match as the models see it: labels, periods, and features from either team's view."""

    def __init__(self, table):
        self.table = table
        self.labels = table.labels
        self.period = table.period
        self.n_frames = table.n_frames
        self._cache = {}

    def runs(self):
        return self.table.runs()

    def features(self, swap=False):
        """Computed once per perspective: ensemble members share them."""
        if swap not in self._cache:
            self._cache[swap] = frame_features(self.table, swap=swap)[0]
        return self._cache[swap]

    def tokens(self, swap=False):
        key = ("tokens", swap)
        if key not in self._cache:
            self._cache[key] = player_tokens(self.table, swap=swap)
        return self._cache[key]

    def ball_target(self, swap=False):
        """No hidden truth for a real match: an all-NaN target (the head is simply unused)."""
        return np.full((self.n_frames, 2), np.nan, dtype=np.float32)

    def touch_target(self, swap=False):
        """No event data for an arbitrary table: all-NaN, so the touch loss ignores it."""
        return np.full((self.n_frames, 2), np.nan, dtype=np.float32)


def feature_names():
    """Column names of frame_features, without computing a real match."""
    from .match import MatchTable
    tiny = MatchTable(name="names", ball=np.zeros((4, 2), np.float32),
                      players=np.zeros((4, 2, 2), np.float32),
                      slot_team=np.array([H, A], np.int8), period=np.ones(4, np.int8))
    return frame_features(tiny)[1]


def swap_labels(labels):
    """H <-> A, D and X unchanged: the label array for `frame_features(..., swap=True)`."""
    out = labels.copy()
    out[labels == H], out[labels == A] = A, H
    return out
