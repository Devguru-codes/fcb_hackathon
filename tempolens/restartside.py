"""Restart-side diagnostic (plan of action row RSD). Counts only.

    python -m tempolens.restartside match --data-root <bundle> --labels <full-match csv>
    python -m tempolens.restartside open

By the laws a corner restarts with the attacking side and a goal kick with the defending side.
Restarts come from the ball track alone (ball still at a corner point or in a goal area, then
moved); each team's defended goal per half from where its players stand at the half's start. `match`
reads the labels, the tracking and nothing else; `open` measures the same on the 10 open matches'
cached ENS5 out-of-fold outputs, with the labels beside, and tries the row's post-pass.
"""

import argparse
import warnings

import numpy as np

from .match import A, H, X

CORNER_M, STILL_M, MOVE_M = 2.0, 1.0, 2.0
STILL_S, SKIP_S, WINDOW_S, START_S = 0.5, 10.0, 2.0, 2.0
AREA_DEPTH, AREA_HALF = 5.5, 9.16          # goal area: 5.5 m deep, 18.32 m wide
MAX_FILL = 75
COSTS = (4.0, 8.0, 8.0, 1.0)
GAIN, RIGHT = 0.003, 0.70                  # the row's rule


def zone(point, pitch):
    """("corner" | "goal kick", end) for a ball position, end 0 = the goal at x 0, 1 = at x L;
    None elsewhere or when unseen."""
    length, width = pitch
    x, y = point
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    end = 0 if x < length / 2 else 1
    gx = 0.0 if end == 0 else length
    if min(np.hypot(x - gx, y), np.hypot(x - gx, y - width)) <= CORNER_M:
        return "corner", end
    if abs(x - gx) <= AREA_DEPTH and abs(y - width / 2) <= AREA_HALF:
        return "goal kick", end
    return None


def find_restarts(table):
    """[(frame, kind, end)] in time order: the ball still for STILL_S in a restart zone (at least
    half of those frames seen, all within STILL_M), then at least MOVE_M away one second later;
    the search skips SKIP_S after each find."""
    fps, ball, out = table.fps, table.ball, []
    still, f = int(STILL_S * fps), 0
    last = table.n_frames - fps
    while f < last:
        z = zone(ball[f], table.pitch) if table.period[f] > 0 else None
        if z is not None and np.isfinite(ball[f + fps]).all() \
                and np.hypot(*(ball[f + fps] - ball[f])) >= MOVE_M:
            before = ball[max(0, f - still):f]
            seen = before[np.isfinite(before).all(axis=1)]
            if 2 * len(seen) >= still and np.all(np.hypot(*(seen - ball[f]).T) <= STILL_M):
                out.append((f, z[0], z[1]))
                f += int(SKIP_S * fps)
                continue
        f += 1
    return out


def defended_ends(table):
    """{period: {H: end, A: end}} from the teams' mean x over the period's first START_S."""
    out = {}
    for period in sorted(set(table.period[table.period > 0].tolist())):
        first = int(np.flatnonzero(table.period == period)[0])
        x = table.players[first:first + int(START_S * table.fps), :, 0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)       # an all-NaN slice
            home = np.nanmean(x[:, table.slot_team == H])
            away = np.nanmean(x[:, table.slot_team == A])
        if np.isfinite(home) and np.isfinite(away) and home != away:
            out[period] = {H: 0, A: 1} if home < away else {H: 1, A: 0}
    return out


def implied_side(kind, end, ends):
    """The side the laws give the restart: corner -> attacking `end`, goal kick -> defending it."""
    if ends is None:
        return None
    defending = H if ends[H] == end else A
    return defending if kind == "goal kick" else (A if defending == H else H)


def decoded_side(codes, frame, fps):
    window = codes[frame:frame + int(WINDOW_S * fps)]
    h, a = int((window == H).sum()), int((window == A).sum())
    return None if h == a == 0 else (H if h >= a else A)


def restarts_with_sides(table):
    ends = defended_ends(table)
    return [(f, kind, implied_side(kind, end, ends.get(int(table.period[f]))))
            for f, kind, end in find_restarts(table)]


def tally(codes, rows, fps):
    """{kind: [found, agree, disagree, undecided]} plus "all"."""
    out = {k: [0, 0, 0, 0] for k in ("corner", "goal kick", "all")}
    for f, kind, implied in rows:
        got = decoded_side(codes, f, fps)
        slot = 3 if got is None or implied is None else (1 if got == implied else 2)
        for k in (kind, "all"):
            out[k][0] += 1
            out[k][slot] += 1
    return out


def fill(codes, rows):
    """(codes with each short stretch not labelled the implied side, from the restart, set to it;
    stretches changed): only a stretch of at most MAX_FILL frames that ends at the implied side."""
    out, changed = codes.copy(), 0
    for f, _, implied in rows:
        if implied is None:
            continue
        rest = out[f:f + MAX_FILL + 1]
        stop = np.flatnonzero((rest == implied) | (rest == X))
        if not len(stop) or stop[0] == 0 or rest[stop[0]] != implied:
            continue
        out[f:f + stop[0]] = implied
        changed += 1
    return out, changed


def _print_tally(label, t):
    print("{}: ".format(label) + "; ".join(
        "{} {} found, {} agree, {} disagree, {} undecided".format(k, *t[k])
        for k in ("corner", "goal kick", "all")), flush=True)


def match(data_root, labels_path):
    from games import GAME
    from .experiments import pm          # puts evaluation/ on the path
    from .match import competition_table
    table = competition_table(GAME, data_root)
    codes = pm.label_codes(pm.read_label_csv(labels_path))
    rows = restarts_with_sides(table)
    _print_tally("match, decoded", tally(codes, rows, table.fps))
    print("match: the post-pass would change {} stretch(es); {} restart(s) without a side".format(
        fill(codes, rows)[1], sum(r[2] is None for r in rows)), flush=True)


def open_study():
    from .decoder import transition_matrix
    from .experiments import SIM_TEST_SEED, CachedSample, decode_table, oof_logp, score
    from .opendata import LOADERS
    names = sorted(LOADERS)
    logps = oof_logp(names, "ensemble", members=5, draws=4)
    dec = {k: [0, 0, 0, 0] for k in ("corner", "goal kick", "all")}
    lab = {k: [0, 0, 0, 0] for k in dec}
    before, after = [], []
    changed = right = stretches = 0
    for name in names:
        sample = CachedSample(name, SIM_TEST_SEED)
        table = sample.table()
        pred = decode_table(sample, logps[name], transition_matrix(*COSTS[:3]), COSTS[3])
        rows = restarts_with_sides(table)
        for acc, codes in ((dec, pred), (lab, sample.labels)):
            for k, v in tally(codes, rows, table.fps).items():
                acc[k] = [a + b for a, b in zip(acc[k], v)]
        filled, n = fill(pred, rows)
        diff = filled != pred
        changed += int(diff.sum())
        right += int((filled[diff] == sample.labels[diff]).sum())
        stretches += n
        before.append(score(sample.labels, pred)["spellF1"])
        after.append(score(sample.labels, filled)["spellF1"])
    _print_tally("open, decoded", dec)
    _print_tally("open, labels ", lab)
    gain = float(np.mean(after) - np.mean(before))
    share = right / changed if changed else float("nan")
    print("post-pass: {} stretches, {} frames changed, {} right / {} wrong ({:.1%}); CV spellF1 "
          "{:.4f} -> {:.4f} ({:+.4f})".format(stretches, changed, right, changed - right, share,
                                             np.mean(before), np.mean(after), gain), flush=True)
    ok = gain >= GAIN and changed > 0 and share >= RIGHT
    print("RSD rule (CV change >= +{} and right on >= {:.0%} of changed frames): {}".format(
        GAIN, RIGHT, "TRIGGERS: a flagged post-pass may be built; adoption is the user's call"
        if ok else "does not trigger: report only"), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="step", required=True)
    m = sub.add_parser("match")
    m.add_argument("--data-root", required=True)
    m.add_argument("--labels", required=True)
    sub.add_parser("open")
    args = parser.parse_args(argv)
    if args.step == "match":
        return match(args.data_root, args.labels)
    return open_study()


if __name__ == "__main__":
    main()
