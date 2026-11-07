"""PFF FC World Cup 2022 as an external test condition (plan of action row PFF; aggregates only).

    python -m tempolens.pffstudy [--until "YYYY-MM-DD HH:MM"] [--wait]

Every PFF match on disk (tracking, metadata, roster, events) is scored with the submitted final
members (decoder.json), never trained on: each member's output for tta_alpha {0, 0.5} x
eata_steps {0, 10} (adaptcv.member_grid, float32 as in ADseeds) is cached per match in
<open data>/cache/pff/, then the pre-registered settings are scored at the 10- and 50-frame
label conventions. `--wait` keeps polling for newly landed files until `--until`. A match that
cannot be read yet (file still downloading, metadata missing) is retried; one whose events and
tracking disagree is skipped and logged.
"""

import argparse
import json
import time
from datetime import datetime

import numpy as np

from . import adaptcv, temporal
from .decoder import transition_matrix
from .experiments import CachedSample, decode_table, open_data_dir, score
from .opendata import _pff_dir, pff_ids, remove_flicker
from .predict import DEFAULT_DECODER, load_decoder

KEYS = ((0.0, 0), (0.5, 0), (0.5, 10))
ROWS = (("submitted: TTA 0.5 + EATA 10", (0.5, 10), (4.0, 8.0, 8.0, 1.0)),
        ("TTA 0.5 alone", (0.5, 0), (4.0, 8.0, 8.0, 1.0)),
        ("no adaptation", (0.0, 0), (4.0, 8.0, 8.0, 1.0)),
        ("no adaptation, CV argmax", (0.0, 0), (2.0, 8.0, 8.0, 0.5)))
CONVENTIONS = (10, 50)
NOISE = 0.006


def _cache():
    folder = open_data_dir() / "cache" / "pff"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def ready(game_id, settle_s=120):
    """All four files present, the tracking file not written to in the last `settle_s`."""
    root = _pff_dir()
    track = root / "tracking_data" / "{}.jsonl.bz2".format(game_id)
    return (track.exists() and time.time() - track.stat().st_mtime > settle_s
            and (root / "metadata" / "{}.json".format(game_id)).exists()
            and (root / "rosters" / "{}.json".format(game_id)).exists()
            and any((root / "event_data").rglob("{}.json".format(game_id))))


def score_match(members, game_id):
    """{row label: {convention: spellF1}} for one match (cached)."""
    path = _cache() / "pff_{}.json".format(game_id)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    sample = CachedSample("pff_{}".format(game_id))
    grid = adaptcv.member_grid(members, sample, alphas=(0.0, 0.5), steps=(0, 10))
    labels = {c: (sample.labels if c == 10 else remove_flicker(sample.labels, sample.runs(), c))
              for c in CONVENTIONS}
    out = {}
    for label, key, costs in ROWS:
        pred = decode_table(sample, temporal.combine(grid[key]), transition_matrix(*costs[:3]),
                            costs[3])
        out[label] = {str(c): score(y, pred)["spellF1"] for c, y in labels.items()}
    path.write_text(json.dumps(out), encoding="utf-8")
    return out


def report(results, skipped):
    print("\npff: {} matches scored, {} skipped".format(len(results), len(skipped)))
    means = {}
    for label, _, costs in ROWS:
        means[label] = {c: float(np.mean([r[label][str(c)] for r in results.values()]))
                        for c in CONVENTIONS}
        print("{:<30} {:<22} 10 frames {:.4f}   50 frames {:.4f}".format(
            label, str(costs), means[label][10], means[label][50]), flush=True)
    sub = means[ROWS[0][0]][10]
    rivals = {k: v[10] - sub for k, v in means.items() if k in (ROWS[1][0], ROWS[2][0])}
    worst = max(rivals, key=rivals.get)
    print("PFF use rule (c): {} is {:+.4f} against the submission at 10 frames -> {}".format(
        worst, rivals[worst], "REVERSAL beyond the noise floor: goes to the user"
        if rivals[worst] > NOISE else "no reversal; the submission stands"), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--until", default=None, help='"YYYY-MM-DD HH:MM": start nothing after')
    parser.add_argument("--wait", action="store_true", help="poll for new files until --until")
    args = parser.parse_args(argv)
    until = datetime.strptime(args.until, "%Y-%m-%d %H:%M") if args.until else None
    members = [temporal.load(adaptcv.ROOT / "models" / p)
               for p in load_decoder(DEFAULT_DECODER)["members"]]
    results, skipped = {}, {}
    while True:
        new = 0
        for game_id in pff_ids():
            if game_id in results or game_id in skipped or not ready(game_id):
                continue
            if until and datetime.now() >= until:
                break
            try:
                results[game_id] = score_match(members, game_id)
                new += 1
                print("scored pff_{} ({:%H:%M})".format(game_id, datetime.now()), flush=True)
            except ValueError as error:            # events and tracking disagree: never usable
                skipped[game_id] = str(error)
                print("skipped pff_{}: {}".format(game_id, error), flush=True)
            except (OSError, EOFError, KeyError, StopIteration) as error:   # retry next round
                print("not readable yet pff_{}: {!r}".format(game_id, error), flush=True)
        if new and results:
            report(results, skipped)
        if not args.wait or (until and datetime.now() >= until):
            break
        time.sleep(600)
    if results:
        report(results, skipped)
    else:
        print("pff: no match scored yet")


if __name__ == "__main__":
    main()
