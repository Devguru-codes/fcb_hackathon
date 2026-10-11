"""Rehearsal of data day: a fake bundle in the real file formats through predict and diagnostics.

Slow (about a minute), so opt-in:

    MTAH_E2E=1 python -m pytest tests/test_end_to_end.py
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))

N_FRAMES = 199319
N_SLOTS = 46


def write_fake_bundle(folder, seed=0):
    """tactical/ ATD metadata + raw text + Smart Tagging, random-walk players and a ball."""
    rng = np.random.default_rng(seed)
    tactical = Path(folder) / "tactical"
    tactical.mkdir(parents=True)
    players = "".join('<Player id="P{0}" teamId="{1}"><Name>Track {0}</Name>'
                      '<ShirtNumber>{0}</ShirtNumber></Player>'.format(i + 1,
                                                                       "T1" if i % 2 else "T2")
                      for i in range(N_SLOTS))
    (tactical / "BAR-ATH_tactical_atd.xml").write_text(
        '<main><Metadata><Teams><Team id="T1"><Name>FC Barcelona</Name></Team>'
        '<Team id="T2"><Name>Athletic Club</Name></Team></Teams>'
        "<Players>{}</Players></Metadata></main>".format(players), encoding="utf-8")

    active = 22
    pos = rng.uniform(0.1, 0.9, size=(active, 2))
    owner = 0
    nan = "NaN,NaN,NaN"
    with open(tactical / "BAR-ATH_tactical_atd.txt", "w", encoding="utf-8") as handle:
        for frame in range(N_FRAMES):
            pos = np.clip(pos + rng.normal(0, 0.002, pos.shape), 0.01, 0.99)
            if frame % 300 == 0:
                owner = int(rng.integers(active))
            groups = ["{:.5f},{:.5f},1.0".format(x, y) for x, y in pos]
            groups += [nan] * (N_SLOTS - active)
            ball = (nan if (frame // 400) % 3 == 0
                    else "{:.5f},{:.5f},0.0".format(*(pos[owner] + 0.004)))
            handle.write("{}:{}:{}\n".format(frame + 1, ";".join(groups), ball))

    instances = "".join(
        "<instance><ID>{}</ID><start>{:.2f}</start><end>{:.2f}</end><code>{}</code>"
        "<label><group>Team</group><text>{}</text></label></instance>".format(
            i, 500.0 + 40 * i, 520.0 + 40 * i, code, team)
        for i, (code, team) in enumerate([("BUILD UP", "Barcelona"),
                                          ("DEFENDING IN MIDDLE THIRD", "Atletic Club"),
                                          ("SET PIECES", "N/A")] * 5))
    (tactical / "BAR-ATH_tactical_smart_tagging.xml").write_text(
        "<file><ALL_INSTANCES>{}</ALL_INSTANCES></file>".format(instances), encoding="utf-8")


def tiny_model(path):
    """A one-epoch temporal model on a synthetic labelled table: only the plumbing is tested."""
    from tempolens import temporal
    from tempolens.match import A, D, H, MatchTable
    rng = np.random.default_rng(1)
    n = 6000
    period = np.ones(n, dtype=np.int8)
    period[3000:] = 2
    table = MatchTable(name="tiny", ball=rng.uniform([0, 0], [105, 68], (n, 2)).astype(np.float32),
                       players=rng.uniform([0, 0], [105, 68], (n, 22, 2)).astype(np.float32),
                       slot_team=np.array([H, A] * 11, dtype=np.int8), period=period,
                       labels=np.tile(np.repeat(np.array([H, D, A], np.int8), 250), n // 750))
    model, normaliser = temporal.train([table], epochs=1, device="cpu", verbose=False)
    temporal.save(model, normaliser, path)


@unittest.skipUnless(os.environ.get("MTAH_E2E"), "slow; set MTAH_E2E=1")
class EndToEndTests(unittest.TestCase):
    def test_predict_writes_a_valid_submission_and_diagnostics_run(self):
        import possession_metrics as pm
        from tempolens import diagnostics, predict
        with tempfile.TemporaryDirectory() as folder:
            write_fake_bundle(folder)
            model, decoder = Path(folder) / "m.pt", Path(folder) / "d.json"
            tiny_model(model)
            decoder.write_text(json.dumps({"switch": 6.0, "alive_to_dead": 4.0,
                                           "dead_to_alive": 4.0, "change_weight": 0.5,
                                           "tag_prior": "auto"}))
            output = Path(folder) / "team.csv"
            labels = predict.predict(folder, output, "learned", model, decoder)
            frame = pm.read_label_csv(output)
            self.assertIsNone(pm.validate_submission(frame, N_FRAMES))
            graded = np.zeros(N_FRAMES, dtype=bool)
            graded[11185:90652] = graded[110808:190198] = True
            self.assertTrue(np.all(labels[~graded] == "X"))
            self.assertFalse(np.any(labels[graded] == "X"))

            report = diagnostics.run(folder, model, decoder)
            self.assertEqual(report["visibility"]["track_slots"], N_SLOTS)
            sides = {name: entry["side"] for name, entry in report["tag_teams"].items()}
            self.assertEqual(sides, {"Barcelona": 0, "Atletic Club": 1, "N/A": None})
            self.assertIn("learned", report["public_window"])
            self.assertIn("learned+tags", report["public_window"])
            self.assertEqual(set(report["tag_prior"]), {"attack", "defence"})


if __name__ == "__main__":
    unittest.main()
