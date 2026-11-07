"""PFF FC World Cup 2022 as an external test condition (plan of action row PFF; aggregates only).

    python -m tempolens.pffstudy gate    # the organisers' four baselines on pff-16 (run first)
    python -m tempolens.pffstudy test    # the pre-registered comparisons on pff-16

Only pff-16 (test_ids: every 4th game id) is ever scored; the other 48 are reserved. Each match is
scored with the submitted final
members (decoder.json), never trained on: each member's output for tta_alpha {0, 0.5} x
eata_steps {0, 10} (adaptcv.member_grid, float32 as in ADseeds) is cached per match in
<open data>/cache/pff/, then the pre-registered settings are scored at the 10- and 50-frame
label conventions. A match whose events and tracking disagree is skipped and logged.
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
GATE_WINDOWS = {"raw": 0, "smooth25": 25, "smooth50": 50, "smooth125": 125}


def _cache():
    folder = open_data_dir() / "cache" / "pff"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def ready(game_id, settle_s=120):
    """All four files present and none written to in the last `settle_s` (still downloading)."""
    root = _pff_dir()
    events = list((root / "event_data").rglob("{}.json".format(game_id)))         if (root / "event_data").is_dir() else []
    paths = [root / "tracking_data" / "{}.jsonl.bz2".format(game_id),
             root / "metadata" / "{}.json".format(game_id),
             root / "rosters" / "{}.json".format(game_id)] + events[:1]
    return len(paths) == 4 and all(p.exists() and time.time() - p.stat().st_mtime > settle_s
                                   for p in paths)


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


def test_ids(ids):
    """pff-16 (plan row PFF, amendment): ids sorted numerically, every 4th from the first. The
    other 48 are reserved for a fine-tune and never scored as a test."""
    ids = sorted(ids, key=int)
    return [ids[i] for i in range(0, len(ids), 4)]


def _gate_match(game_id):
    from .experiments import baseline_codes
    sample = CachedSample("pff_{}".format(game_id))
    table = sample.table()
    return [[score(sample.labels, baseline_codes(table, w))[k] for w in GATE_WINDOWS.values()]
            for k in ("spellF1", "macroF1")]


def gate(ids):
    from .labelcal import MACRO_ORDER, SPELL_ORDER, in_order
    rows = {}
    for game_id in ids:
        rows[game_id] = _gate_match(game_id)
        print("gate: pff_{} done ({:%H:%M})".format(game_id, datetime.now()), flush=True)
    mean = np.mean(list(rows.values()), axis=0)
    names = list(GATE_WINDOWS)
    spell, macro = dict(zip(names, mean[0])), dict(zip(names, mean[1]))
    print(chr(10) + "pff-16 baselines ({} matches, 10 frames):".format(len(rows)))
    for n in names:
        print("{:<10} spellF1 {:.4f}   macroF1 {:.4f}".format(n, spell[n], macro[n]))
    print("published spellF1 0.0486 / 0.1996 / 0.2433 / 0.2242, macroF1 0.6532 / 0.6752 / "
          "0.6724 / 0.6356 (raw / 25 / 50 / 125)")
    print("macroF1 order (reported only): {}".format(
        "reproduced" if in_order(macro, MACRO_ORDER) else "not reproduced"))
    print("PFF gate: {}".format("PASS (spellF1 order reproduced)" if in_order(spell, SPELL_ORDER)
                                else "FAIL (spellF1 order not reproduced)"), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("gate", "test"))
    args = parser.parse_args(argv)
    ids = test_ids(pff_ids())
    assert len(ids) == 16 and ids[0] == "3812", "expected the 64-match download"
    missing = [g for g in ids if not ready(g, settle_s=0)]
    assert not missing, "pff-16 files missing: {}".format(missing)
    if args.step == "gate":
        return gate(ids)
    members = [temporal.load(adaptcv.ROOT / "models" / p)
               for p in load_decoder(DEFAULT_DECODER)["members"]]
    results, skipped = {}, {}
    for game_id in ids:
        try:
            results[game_id] = score_match(members, game_id)
            print("scored pff_{} ({:%H:%M})".format(game_id, datetime.now()), flush=True)
        except ValueError as error:            # events and tracking disagree: never usable
            skipped[game_id] = str(error)
            print("skipped pff_{}: {}".format(game_id, error), flush=True)
    report(results, skipped)


if __name__ == "__main__":
    main()
