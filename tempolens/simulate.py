"""Make clean tracking look like Metrica's automatic, uncorrected tracking (ATD).

Why this exists: open tracking is clean. In Metrica's sample games the ball is missing almost
only while it is dead, and in the DFL data it is never missing. A model trained on that learns
"no ball = dead ball", or never learns to work without the ball at all.

What the competition ATD actually looks like is measurable from the organisers' public files:
`evaluation/results/baseline_raw.csv` is D exactly where the ATD has no ball. Over the graded
frames the ball is missing 40.7% of the time in only 150 gaps -- median 268 frames, mean 431,
and essentially every missing frame is inside a gap of 5 s or more. So the ATD loses the ball in
long blind stretches, and for ~40% of the match possession must be read from the players alone.

Simulated, per match:
- stoppage gaps: each dead-ball stretch is lost, edges jittered, with probability `dead_cover`
  (the tracker loses a ball that leaves play and finds it again around the restart);
- ball gaps that start at a per-frame hazard (optionally `dead_bias` times higher while the ball
  is dead) and last a log-normal number of frames fitted to those 150 gaps, topping the
  in-period missing fraction up to `ball_missing`; plus rare short flickers;

Calibration: the official baseline has published full-match scores for four smoothing windows
(EVALUATION.md). On DFL match J03WMX, dead_cover ~0.67 brings their macroF1 within ~0.03
in the published order (under spellF1 smooth25 and smooth50 tie and smooth125 falls below them); uniform gaps (dead_cover 0) leave macroF1 ~0.2 too
low whatever the dead-ball bias. Clean DFL also matches the competition's 78% agreement between
nearest player's team and owning team, so no extra team noise is added.
- player dropout as a two-state chain: ~4% per player, which matches the real share of frames
  with 21+ players visible (77%); the real mean of 20.5 also includes rare long blackouts during
  stoppages, which the model already reads as dead ball;
- tracker measurement noise on the ball, smoothed like a tracker's output: 0.2 m, Gaussian over
  ~3 frames. It reproduces the real ball's acceleration profile (median ~5.7 m/s^2, vs ~3 on the
  clean open data) while leaving its speeds unchanged. Independent per-frame jitter must never be
  used (see player_jitter_m).
"""

from dataclasses import dataclass, replace

import numpy as np

from .match import D


@dataclass(frozen=True)
class AtdNoise:
    ball_missing: float = 0.407      # in-period fraction without a ball (measured, see above)
    gap_median: float = 268.0        # frames; log-normal fit to the measured gaps
    gap_sigma: float = 0.95          # gives a mean of ~420 frames against a measured 431
    dead_bias: float = 1.0           # gap-start hazard while dead / while alive
    dead_cover: float = 0.67         # chance a whole stoppage is lost (ball out of play)
    dead_lag: float = 12.0           # frames after the stoppage starts before the ball is lost
    dead_edge_sd: float = 15.0       # jitter of both edges of a stoppage gap, frames
    flicker_rate: float = 2e-4       # per frame: short 1-5 frame losses
    ball_jitter_m: float = 0.0      # see player_jitter_m
    ball_noise_m: float = 0.2        # tracker noise on the ball, then smoothed (ball_smooth_sd)
    ball_smooth_sd: float = 3.0      # frames
    player_missing: float = 0.04     # real ATD: 77% of graded frames show 21+ players
    player_mean_gap: float = 60.0
    # Per-frame independent jitter is NOT tracker noise: through 25 fps velocities it added
    # ~2.6 m/s of fake speed (players 3.8 m/s vs 1.5 on the real ATD), so the model read real
    # play as a stoppage. The real ATD is smooth; jitter is off.
    player_jitter_m: float = 0.0


def _two_state_dropout(n, hazard, recover, rng):
    """Boolean 'lost' sequence of a two-state chain with constant rates."""
    u = rng.random(n)
    lost = np.zeros(n, dtype=bool)
    state = False
    for t in range(n):
        state = (u[t] >= recover) if state else (u[t] < hazard)
        lost[t] = state
    return lost


def _ball_gaps(hazard, durations, period):
    """Missing mask from per-frame gap-start hazards (already Bernoulli-drawn) and durations."""
    n = len(hazard)
    lost = np.zeros(n, dtype=bool)
    t = 0
    while t < n:
        if period[t] > 0 and hazard[t]:
            stop = min(n, t + int(durations[t]))
            # A gap never runs on into the next period.
            same = np.flatnonzero(period[t:stop] != period[t])
            stop = t + int(same[0]) if len(same) else stop
            lost[t:stop] = True
            t = stop + 1   # the ball is seen at least one frame between two gaps
        else:
            t += 1
    return lost


def _stoppage_gaps(table, noise, rng):
    """Lose the ball through whole stoppages: shortly after it dies until about the restart."""
    lost = np.zeros(table.n_frames, dtype=bool)
    dead = table.labels == D
    # Per half: two halves back to back (DFL) must not merge into one stoppage.
    for first, last in table.runs():
        edges = np.diff(np.concatenate(([0], dead[first:last].astype(np.int8), [0])))
        for start, stop in zip(first + np.flatnonzero(edges == 1), first + np.flatnonzero(edges == -1)):
            if rng.random() >= noise.dead_cover:
                continue
            a = start + int(round(noise.dead_lag + noise.dead_edge_sd * rng.standard_normal()))
            b = stop + int(round(noise.dead_edge_sd * rng.standard_normal()))
            a, b = max(start, a), min(last, b)
            if b > a:
                lost[a:b] = True
    return lost


@dataclass(frozen=True)
class TrackerFaults:
    """Tracker failures beyond dropout, applied on top of simulate_atd (robustness checks)."""
    swap_rate: float = 0.002        # per frame: two nearby same-team tracks exchange slots
    swap_radius_m: float = 3.0
    duplicate_rate: float = 0.0005  # per frame: someone is tracked twice for a while
    duplicate_frames: int = 75
    phantom_rate: float = 0.0003    # per frame: a non-player track near the ball, given a team
    phantom_frames: int = 125
    keeper_swaps: int = 0           # per match: the two keepers' tracks exchange slots (E1)
    keeper_frames: tuple = (1500, 7500)   # each for 1-5 min


def add_tracker_faults(table, faults=TrackerFaults(), seed=0):
    """A copy of `table` with identity swaps, duplicate tracks and phantom tracks added.

    Extra slots are appended for duplicates and phantoms, so the slot count grows as it does in
    the real ATD (46 slots for 22 players). Labels and periods are untouched.
    """
    rng = np.random.default_rng(seed)
    n = table.n_frames
    players = table.players.astype(np.float32).copy()
    teams = list(table.slot_team)
    extra = []

    # Identity swaps: exchange the rest of two close same-team tracks from frame t on, for a
    # stretch, which is what a tracker re-acquiring the wrong person looks like.
    for t in np.flatnonzero(rng.random(n) < faults.swap_rate):
        team = int(rng.integers(0, 2))
        slots = np.flatnonzero((np.asarray(teams[:players.shape[1]]) == team)
                               & np.isfinite(players[t, :, 0]))
        if len(slots) < 2:
            continue
        a = int(rng.choice(slots))
        d = np.hypot(*(players[t, slots] - players[t, a]).T)
        near = slots[(d > 0) & (d < faults.swap_radius_m)]
        if not len(near):
            continue
        b = int(rng.choice(near))
        stop = min(n, t + int(rng.integers(50, 500)))
        players[t:stop, [a, b]] = players[t:stop, [b, a]]

    # Duplicates: a second, slightly offset track of an existing player.
    for t in np.flatnonzero(rng.random(n) < faults.duplicate_rate):
        visible = np.flatnonzero(np.isfinite(players[t, :, 0]))
        if not len(visible):
            continue
        slot = int(rng.choice(visible))
        stop = min(n, t + faults.duplicate_frames)
        track = np.full((n, 2), np.nan, dtype=np.float32)
        track[t:stop] = players[t:stop, slot] + rng.normal(0, 0.5, 2).astype(np.float32)
        extra.append((track, teams[slot]))

    # Phantoms: something that is not a player (a referee, a coach on the line) tracked near
    # the ball and assigned to a random team.
    ball = table.ball.astype(np.float32)
    for t in np.flatnonzero(rng.random(n) < faults.phantom_rate):
        if not np.isfinite(ball[t, 0]):
            continue
        stop = min(n, t + faults.phantom_frames)
        offset = rng.normal(0, 4.0, 2).astype(np.float32)
        anchor = ball[t:stop].copy()
        anchor[~np.isfinite(anchor[:, 0])] = ball[t]
        track = np.full((n, 2), np.nan, dtype=np.float32)
        # lag the ball like a referee does, smoothed over a second
        smooth = np.array([np.nanmean(anchor[max(0, i - 25):i + 1], axis=0)
                           for i in range(len(anchor))])
        track[t:stop] = smooth + offset
        extra.append((track, int(rng.integers(0, 2))))

    # Keeper swaps: each keeper's track is filed under the other team's keeper slot, for minutes
    # (drawn last, so the faults above keep their draws).
    if faults.keeper_swaps:
        from .match import keeper_zones
        occ = keeper_zones(table).mean(axis=0)
        keepers = [int(np.flatnonzero(table.slot_team == t)[np.argmax(occ[table.slot_team == t, t])])
                   for t in (0, 1)]
        inside = np.flatnonzero(table.period > 0)
        for _ in range(faults.keeper_swaps):
            t = int(rng.choice(inside))
            stop = min(n, t + int(rng.integers(*faults.keeper_frames)))
            players[t:stop, keepers] = players[t:stop, keepers[::-1]]

    if extra:
        players = np.concatenate([players, np.stack([e[0] for e in extra], axis=1)], axis=1)
        teams = teams + [e[1] for e in extra]
    return replace(table, name=table.name + "~faults{}".format(seed), players=players,
                   slot_team=np.asarray(teams, dtype=np.int8))


def _tracker_ball(table, noise, rng):
    """The clean ball as a tracker reports it: measurement noise, then Gaussian smoothing within
    each period over the frames where the ball exists (normalised convolution)."""
    ball = table.ball.astype(np.float64)
    if noise.ball_noise_m <= 0 and noise.ball_smooth_sd <= 0:
        return ball.astype(np.float32)
    ball = ball + rng.normal(0, noise.ball_noise_m, ball.shape) if noise.ball_noise_m > 0 else ball
    if noise.ball_smooth_sd > 0:
        half = int(4 * noise.ball_smooth_sd)
        kernel = np.exp(-0.5 * (np.arange(-half, half + 1) / noise.ball_smooth_sd) ** 2)
        for start, stop in table.runs():
            seg = ball[start:stop]
            ok = np.isfinite(seg[:, 0])
            # "full" then the centred slice: "same" returns the kernel's length on runs shorter
            # than the kernel; on longer runs the two are identical
            weight = np.convolve(ok.astype(np.float64), kernel, mode="full")[half:half + len(seg)]
            for axis in (0, 1):
                smooth = np.convolve(np.where(ok, seg[:, axis], 0.0), kernel,
                                     mode="full")[half:half + len(seg)]
                seg[ok, axis] = (smooth / np.maximum(weight, 1e-9))[ok]
    return ball.astype(np.float32)


def random_noise(seed, base=AtdNoise()):
    """A noise setting drawn around the calibrated one (domain randomisation): training on many
    trackers keeps the model from fitting one simulator's habits, which the real ATD may not
    share. Bump experiments.RANDOM_VERSION when these ranges change."""
    rng = np.random.default_rng(10_000 + seed)
    return replace(base, ball_missing=rng.uniform(0.30, 0.50),
                   dead_cover=rng.uniform(0.40, 0.90),
                   gap_median=float(np.exp(rng.uniform(np.log(150), np.log(450)))),
                   player_missing=rng.uniform(0.02, 0.08),
                   ball_noise_m=rng.uniform(0.1, 0.4))


def simulate_atd(table, noise=AtdNoise(), seed=0):
    """A corrupted copy of `table`. Labels and periods are untouched."""
    rng = np.random.default_rng(seed)
    n = table.n_frames
    inside = table.period > 0
    ball = _tracker_ball(table, noise, rng)
    players = table.players.astype(np.float32).copy()

    # Relative gap-start hazard per frame; dead frames can be made likelier to start a gap.
    weight = np.ones(n)
    if table.labels is not None and noise.dead_bias != 1.0:
        weight[table.labels == D] = noise.dead_bias
    durations = np.exp(np.log(noise.gap_median) + noise.gap_sigma * rng.standard_normal(n))
    u = rng.random(n)
    already = np.isnan(ball[:, 0]) & inside
    if table.labels is not None and noise.dead_cover > 0:
        already |= _stoppage_gaps(table, noise, rng)
    # Stationary missing fraction ~ h*L / (1 + h*L) for mean gap L: solve for the hazard, then
    # correct it twice against the realised fraction (gaps are truncated at period ends).
    target = max(0.0, noise.ball_missing - already.sum() / max(1, inside.sum()))
    mean_gap = float(np.mean(np.minimum(durations, 3000)))
    h = target / (mean_gap * max(1e-6, 1.0 - target))
    lost = np.zeros(n, dtype=bool)
    for _ in range(3):
        if target <= 0:
            break
        rate = np.clip(h * weight / np.mean(weight[inside]), 0, 1)
        lost = _ball_gaps(u < rate, durations, table.period)
        extra = (lost & ~already)[inside].mean()
        if extra <= 0:
            break
        h *= target / extra
    lost |= already
    flicker_start = (rng.random(n) < noise.flicker_rate) & inside
    for t in np.flatnonzero(flicker_start):
        lost[t:t + int(rng.integers(1, 6))] = True
    ball[lost] = np.nan
    ball += rng.normal(0, noise.ball_jitter_m, ball.shape).astype(np.float32)

    # Players: independent dropout per slot.
    h_player = noise.player_missing / (noise.player_mean_gap * (1 - noise.player_missing))
    for slot in range(players.shape[1]):
        gone = _two_state_dropout(n, h_player, 1.0 / noise.player_mean_gap, rng)
        players[gone, slot] = np.nan
    players += rng.normal(0, noise.player_jitter_m, players.shape).astype(np.float32)

    return replace(table, name=table.name + "~atd{}".format(seed), ball=ball, players=players,
                   meta=dict(table.meta, simulated=seed))
