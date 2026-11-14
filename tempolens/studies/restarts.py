"""Restart diagnostics (plan of action row RST: Fable round 4, G3-1 and G3-2). Counts only.

    python -m tempolens.studies.restarts match --data-root <bundle> --labels <full-match csv>
    python -m tempolens.studies.restarts open

G3-1: how long after each kickoff the decoded labels stay D, and what back-filling a short leading
D run with the team the laws name (second half: the team that did not kick off the first; after a
goal: the conceding team) does. G3-2: the decoded alive share between a goal and its restart, where
the laws say the ball is dead. `match` reads only the labels, the tracking and Smart Tagging; `open`
measures the same on the 10 open matches' cached ENS5 out-of-fold outputs at the submitted costs.
"""

import argparse
import json
import xml.etree.ElementTree as ET

import numpy as np

from .. import kickoffs
from ..match import A, D, H

WINDOW = 75                  # 3 s at 25 fps: the longest leading D run a back-fill may fill
POST_M = 3.66 + 0.5          # half the goal mouth plus a margin for the ball track
LEAD_S, SEARCH_S = 3.0, 10.0  # open data: the ball-track search starts this long before the event
COSTS = (4.0, 8.0, 8.0, 1.0)


def leading_dead(codes, frame, window=WINDOW):
    """Length of the D run starting at `frame` if it ends at an H or A frame within `window`
    frames; 0 if `frame` is not D; None if the run is longer or ends at X / the end."""
    if codes[frame] != D:
        return 0
    rest = codes[frame:frame + window + 1]
    stop = np.flatnonzero(rest != D)
    if not len(stop) or rest[stop[0]] not in (H, A):
        return None
    return int(stop[0])


def backfill(codes, restarts, window=WINDOW):
    """(codes with each restart's short leading D run set to its implied team, stretches changed).
    `restarts`: [(frame, implied team or None)]."""
    out, changed = codes.copy(), 0
    for frame, team in restarts:
        run = leading_dead(out, frame, window)
        if team in (H, A) and run:
            out[frame:frame + run] = team
            changed += 1
    return out, changed


def start_delay(codes, frame, stop):
    """Frames from `frame` to the first H or A before `stop`; None if there is none."""
    alive = np.flatnonzero(np.isin(codes[frame:stop], (H, A)))
    return int(alive[0]) if len(alive) else None


def goal_moment(table, start, stop):
    """First frame in [start, stop) at which the ball track is beyond a goal line between the
    posts; None if the ball track never shows that."""
    length, width = table.pitch
    ball = table.ball[start:stop]
    with np.errstate(invalid="ignore"):
        inside = (((ball[:, 0] < 0) | (ball[:, 0] > length))
                  & (np.abs(ball[:, 1] - width / 2) <= POST_M))
    hit = np.flatnonzero(inside)
    return start + int(hit[0]) if len(hit) else None


def alive_share(codes):
    return float(np.isin(codes, (H, A)).mean()) if len(codes) else float("nan")


# -- the match -------------------------------------------------------------------------------

def match(data_root, labels_path):
    import possession_metrics as pm
    from games import GAME, GAMES, require
    from ..match import competition_table
    table = competition_table(GAME, data_root)
    codes = pm.label_codes(pm.read_label_csv(labels_path))
    fps = GAMES[GAME]["fps"]
    path = require(["smart_tagging"], GAME, data_root)[0]
    goals, n_goals, n_restarts = kickoffs.goal_restarts(path, table.meta["home"],
                                                        table.meta["away"], fps)
    rows = kickoffs.restarts(codes, table, goals)
    print("G3-1: {} restarts ({} GOALS tags, {} paired)".format(len(rows), n_goals, len(goals)))
    total = 0
    for kind, frame, implied in rows:
        dead = int((codes[frame:frame + WINDOW] == D).sum())
        run = leading_dead(codes, frame)
        total += dead
        print("  {:<6} D frames in the first 3 s {:>3}, leading D run {}, implied team {}".format(
            kind, dead, ">75 or open" if run is None else run,
            "known" if implied in (H, A) else "unknown"))
    _, changed = backfill(codes, [(f, i) for _, f, i in rows])
    print("G3-1 total: {} D frames in the first 3 s; a back-fill would change {} stretch(es) "
          "(at most 6 allowed)".format(total, changed))
    print("G3-2: goal moment -> restart")
    fallback = 0
    for (_, _, tag_start, tag_end), (_, frame, _) in zip(
            sorted(goals), [r for r in rows if r[0] == "goal"]):
        g = goal_moment(table, tag_start, frame)
        if g is None:
            fallback += 1
        g = min(tag_end, frame) if g is None else g
        print("  window {:6.1f} s, alive share {:.1%}".format(
            (frame - g) / fps, alive_share(codes[g:frame])))
    print("G3-2: {} of {} windows fell back to the tag end".format(fallback, len(goals)))


# -- open data -------------------------------------------------------------------------------

def _dfl_kickoffs(match_id):
    import re
    from ..opendata import _dfl_file
    from ..touches import _dfl_halves, _time
    info = _dfl_file("matchinformation", match_id).read_text(encoding="utf-8")
    general = dict(re.findall(r'(\w+)="([^"]*)"', re.search(r"<General [^>]*>", info).group(0)))
    team_of = {general["HomeTeamId"]: H, general["GuestTeamId"]: A}
    halves, out = _dfl_halves(match_id), []
    for event in ET.parse(_dfl_file("events_raw", match_id)).getroot().iter("Event"):
        kick = event.find("KickOff")
        play = None if kick is None else kick.find("Play")
        if play is None or play.attrib.get("Team") not in team_of:
            continue
        t, offset = _time(event.attrib["EventTime"]), 0
        for start, n in halves:
            k = int(round((t - start) / 0.04))
            if 0 <= k < n:
                out.append((offset + k, team_of[play.attrib["Team"]]))
                break
            offset += n
    return out


def _metrica_kickoffs(game, n_frames):
    from ..opendata import _metrica_csv_events, _metrica_dir
    folder = _metrica_dir(game)
    if game in (1, 2):
        events = _metrica_csv_events(folder / "Sample_Game_{}_RawEventsData.csv".format(game),
                                     n_frames)
        keep = ((events.type == "SET PIECE") & events.subtype.str.contains("KICK OFF")
                & events.team.isin([H, A]))
        return list(zip(events.start[keep].astype(int), events.team[keep].astype(int)))
    from ..match import teams_from_epts_metadata
    _, home, _ = teams_from_epts_metadata(folder / "Sample_Game_3_metadata.xml")
    with open(folder / "Sample_Game_3_tracking.txt", encoding="utf-8") as handle:
        offset = int(handle.readline().split(":", 1)[0])
    out = []
    for event in json.loads((folder / "Sample_Game_3_events.json").read_text(
            encoding="utf-8"))["data"]:
        subtypes = event.get("subtypes") or []
        subtypes = [subtypes] if isinstance(subtypes, dict) else subtypes
        name = (event.get("team") or {}).get("name")
        start = (event.get("start") or {}).get("frame")
        if (event["type"]["name"].upper() == "SET PIECE" and name and start is not None
                and any(s["name"].upper() == "KICK OFF" for s in subtypes)):
            out.append((int(start) - offset, H if name == home else A))
    return out


def open_kickoffs(name, table):
    """[(kind, restart frame, team)] in time order; kind "half" for a period's first kickoff,
    else "goal"; and how many events had no ball-track restart."""
    raw = (_dfl_kickoffs(name[4:]) if name.startswith("dfl_")
           else _metrica_kickoffs(int(name[-1]), table.n_frames))
    out, missing = [], 0
    fps = table.fps
    for approx, team in sorted(raw):
        frame = kickoffs.restart_frame(table, max(0, approx - int(LEAD_S * fps)), SEARCH_S,
                                       strict=True)
        if frame is None or table.period[frame] == 0:
            missing += 1
            continue
        out.append((frame, team))
    rows, seen = [], set()
    for frame, team in sorted(set(out)):
        period = int(table.period[frame])
        rows.append(("goal" if period in seen else "half", frame, team))
        seen.add(period)
    return rows, missing


def open_study():
    from ..decoder import transition_matrix
    from ..experiments import SIM_TEST_SEED, CachedSample, decode_table, oof_logp, score
    from ..opendata import LOADERS
    names = sorted(LOADERS)
    logps = oof_logp(names, "ensemble", members=5, draws=4)
    delays = {"half": [], "goal": []}
    truth_delays, before, after, windows = [], [], [], []
    changed = correct = stretches = missing = skipped = 0
    for name in names:
        sample = CachedSample(name, SIM_TEST_SEED)
        table = sample.table()
        pred = decode_table(sample, logps[name], transition_matrix(*COSTS[:3]), COSTS[3])
        rows, miss = open_kickoffs(name, table)
        missing += miss
        for kind, frame, team in rows:
            stop = next(e for s, e in sample.runs() if s <= frame < e)
            d = start_delay(pred, frame, stop)
            delays[kind].append(min(d, WINDOW * 4) if d is not None else WINDOW * 4)
            t = start_delay(sample.labels, frame, stop)
            truth_delays.append(min(t, WINDOW * 4) if t is not None else WINDOW * 4)
            if kind == "goal":
                alive = np.flatnonzero(np.isin(sample.labels[:frame], (H, A)))
                g = int(alive[-1]) + 1 if len(alive) else 0
                if frame - g >= table.fps:
                    windows.append(alive_share(pred[g:frame]))
                else:
                    skipped += 1
        filled, n = backfill(pred, [(f, t) for _, f, t in rows])
        diff = filled != pred
        changed += int(diff.sum())
        correct += int((filled[diff] == sample.labels[diff]).sum())
        stretches += n
        before.append(score(sample.labels, pred)["spellF1"])
        after.append(score(sample.labels, filled)["spellF1"])
    every = delays["half"] + delays["goal"]
    print("open: {} kickoffs ({} half, {} goal), {} without a ball-track restart".format(
        len(every), len(delays["half"]), len(delays["goal"]), missing))
    for label, values in (("all", every), ("half", delays["half"]), ("goal", delays["goal"]),
                          ("labels (all)", truth_delays)):
        print("start delay {:<13} mean {:6.1f} median {:5.1f} frames (capped at {})".format(
            label, np.mean(values), np.median(values), WINDOW * 4))
    gain = np.mean(after) - np.mean(before)
    print("back-fill: {} stretches, {} frames changed, {} correct / {} wrong; CV spellF1 {:.4f} -> "
          "{:.4f} ({:+.4f})".format(stretches, changed, correct, changed - correct,
                                     np.mean(before), np.mean(after), gain))
    ok = gain >= 0 and np.mean(every) >= 10
    print("G3-1 rule (CV change >= 0 and mean start delay >= 10 frames): {}".format(
        "TRIGGERS: a flagged post-pass may be built; adoption is the user's call" if ok
        else "does not trigger: report only"))
    n_windows, windows = len(windows), windows or [float("nan")]
    print("G3-2 open: {} goal windows (skipped under 1 s: {}), decoded alive share mean {:.1%}, "
          "max {:.1%}, windows > 5%: {}".format(n_windows, skipped, np.mean(windows),
                                                np.max(windows), sum(w > 0.05 for w in windows)))


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
