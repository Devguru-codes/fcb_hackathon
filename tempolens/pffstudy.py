"""PFF FC World Cup 2022 as an external test condition (plan of action row PFF; aggregates only).

    python -m tempolens.pffstudy gate    # the organisers' baselines at 6 folds on pff-16 (first)
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
NOISE = 0.006                # the veto for adoptions
REVERSAL, AGREE = 0.010, 11  # a reversal of the adaptation choice (amendment 2)


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
    sub = ROWS[0][0]
    for rival in (ROWS[1][0], ROWS[2][0]):     # amendment 2 (iii): > 0.010 and >= 11 of 16
        diffs = np.array([r[rival]["10"] - r[sub]["10"] for r in results.values()])
        agree = int((diffs > 0).sum()) if diffs.mean() > 0 else int((diffs < 0).sum())
        reversal = diffs.mean() > REVERSAL and agree >= AGREE
        print("PFF use rule (c): {} {:+.4f} against the submission at 10 frames, {} of {} "
              "matches agree in sign -> {}".format(
                  rival, diffs.mean(), agree, len(diffs),
                  "REVERSAL: goes to the user" if reversal else "no reversal"), flush=True)


def test_ids(ids):
    """pff-16 (plan row PFF, amendment): ids sorted numerically, every 4th from the first. The
    other 48 are reserved for a fine-tune and never scored as a test."""
    ids = sorted(ids, key=int)
    return [ids[i] for i in range(0, len(ids), 4)]


def _diag_match(game_id):
    """(len(FRAMES), 4 windows, 2) spellF1 / macroF1 of the organisers' baselines on one match."""
    from .experiments import baseline_codes
    from .labelcal import FRAMES, WINDOWS, labels_at
    sample = CachedSample("pff_{}".format(game_id))
    table = sample.table()
    preds = [baseline_codes(table, w) for _, w in WINDOWS]
    out = np.zeros((len(FRAMES), len(WINDOWS), 2))
    for i, f in enumerate(FRAMES):
        y = labels_at(sample, f)
        for j, pred in enumerate(preds):
            r = score(y, pred)
            out[i, j] = r["spellF1"], r["macroF1"]
    return out, sample


def gate(ids):
    """Amendment 2 (ii): LC1's six-value baseline-order diagnostic on pff-16; the gate is its
    10- and 50-frame rows, read as a diagnostic."""
    from .labelcal import FRAMES, MACRO_ORDER, SPELL_ORDER, WINDOWS, in_order, verdict
    from .experiments import pm
    per, lengths = [], []
    for game_id in ids:
        out, sample = _diag_match(game_id)
        per.append(out)
        lengths += [b - a for a, b, _ in pm.possession_spells(sample.labels, sample.runs())]
        print("diag: pff_{} done ({:%H:%M})".format(game_id, datetime.now()), flush=True)
    mean = np.mean(per, axis=0)
    names = [n for n, _ in WINDOWS]
    print(chr(10) + "pff-16, organisers' baselines ({} matches); spellF1 raw / 25 / 50 / 125 | "
          "macroF1 raw / 25 / 50 / 125 | orders spell / macro".format(len(per)))
    passing = []
    for i, f in enumerate(FRAMES):
        spell, macro = dict(zip(names, mean[i, :, 0])), dict(zip(names, mean[i, :, 1]))
        ok = in_order(spell, SPELL_ORDER), in_order(macro, MACRO_ORDER)
        if all(ok):
            passing.append(i)
        print("{:<4} {} | {} | {} / {}".format(
            f, " ".join("{:.4f}".format(spell[n]) for n in names),
            " ".join("{:.4f}".format(macro[n]) for n in names),
            *("yes" if o else "no" for o in ok)), flush=True)
    spell_ok = {f: in_order(dict(zip(names, mean[i, :, 0])), SPELL_ORDER)
                for i, f in enumerate(FRAMES)}
    print("gate (spellF1 order): 10 frames {}, 50 frames {}".format(
        *("PASS" if spell_ok[f] else "FAIL" for f in (10, 50))))
    if spell_ok[10]:
        q = np.percentile(np.asarray(lengths) / 25.0, [10, 25, 50, 75, 90])
        print("pass at 10: pff-16 possession lengths (s, at 10 frames) p10/25/50/75/90 " +
              " / ".join("{:.1f}".format(v) for v in q) + ", mean {:.1f}".format(
                  np.mean(lengths) / 25.0))
    print("LC1 rule on pff-16 (both orders): " + verdict(passing)[1])
    replicated = passing == [FRAMES.index(50)] or (
        FRAMES.index(50) in passing and verdict(passing)[0] is not None)
    print("replicated (50 passes on both datasets, no non-adjacent value on PFF; DFL passed at 50 "
          "only): {}".format("yes" if replicated else "no"))
    reading = ("passes at 50 only: PFF replicates LC1" if spell_ok[50] and not spell_ok[10] else
               "passes at 10: PFF's labels are longer than DFL's" if spell_ok[10] else
               "passes at neither: ball visibility dominates; PFF stays a test condition")
    print("reading: " + reading, flush=True)


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
