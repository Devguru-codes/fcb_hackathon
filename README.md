# TempoLens — team Coding United, More than a Hack 2027 Phase 1

TempoLens labels every frame of FC Barcelona vs Athletic Club with who has the ball (`H` / `A` /
`D` / `X`). A 10-member ensemble of bidirectional GRUs reads ~140 identity-free features of the
automatic tracking. Its members were trained only on open football data (Metrica's 3 sample games
and 7 DFL/Sportec matches), with tracking noise simulated to match the competition's. Viterbi then
decodes the ensemble's output into whole possessions, the unit that `spellF1` scores. Test-time
feature normalisation and a Smart Tagging prior adjust it to the match without labels; no weight
is trained or updated on the match. Every
setting was chosen on open data only: no hand labels, no checks against the video, nothing tuned
on the provided match.

**Documents:** the short technical document [docs/Coding United.md](docs/Coding%20United.md) ·
the long appendix [docs/technical_document.md](docs/technical_document.md) · every experiment and
decision rule [docs/plan_of_action.md](docs/plan_of_action.md) · pipeline and commands
[README_Tempolens.md](README_Tempolens.md) · the official challenge README, unchanged:
[README_challenge.md](README_challenge.md).

## Reproduce the submission

```bash
pip install -r requirements-tempolens.txt      # or requirements-predict.txt: the submission only
# unpack the organisers' data bundle into data/ (data/tactical/..., see data/README.md)
python -m tempolens.predict --data-root data --output "Coding United.csv"
python evaluation/validate_submission.py "Coding United.csv"
```

- **Expected output:** 199,319 rows plus the header. sha256
  `9cc22752a2e280cb782823134b1209fa44be8a81228d607f8799b95e7878a61c` (TTA-only, `eata_steps` 0).
- **Runtime:** about 7 minutes (432 s) on a laptop CPU (Intel Core i7-12700H, 14 cores, 16 GB RAM) with the 10 members and TTA, measured while another job was using the machine. With EATA it took 14–24 minutes.
- **Environment:** Python 3.11 (tested on 3.11.3), the pinned versions in the requirements file.
  Only CPU is needed: inference runs in float64 in a fixed order, so the same data gives the same
  file byte for byte. A GPU is not used even if one is present. Byte identity was verified from
  fresh clones on Windows 11, x86-64. Other platforms should agree, since float64 and a fixed order
  leave little room to differ, but that has not been checked.
- **Inputs:** the tactical tracking (`data/tactical/BAR-ATH_tactical_atd.*`) and the Smart Tagging
  file. No video, no homography, no broadcast feed, no open data, no network access.
- The weights (`models/tempolens_rnd4_m*.pt`, `models/tempolens_ms50_m*.pt`) and every decoder
  setting (`models/decoder.json`) are committed. GPU training is not bit-reproducible, so
  retraining gives close but different weights. See [README_Tempolens.md](README_Tempolens.md).

The repository holds no match data: `data/` ships with its README only, and every output is
git-ignored.

## Open data (only needed to retrain or rerun the studies)

Point `MTAH_OPEN_DATA_DIR` at a folder holding these:

| dataset | folder | use | licence / terms |
|---|---|---|---|
| Metrica Sports sample data, games 1–3: `git clone https://github.com/metrica-sports/sample-data metrica_sample_data` | `metrica_sample_data/` | training and cross-validation | public sample data; Metrica asks that public use acknowledge the source (this README and the documents do) |
| DFL/Sportec open tracking and event data, 7 Bundesliga matches: Bassek, M., Rein, R., Weber, H. & Memmert, D. (2025), "An integrated dataset of spatiotemporal and event data in elite soccer", *Scientific Data* 12, 195. figshare article 28196177: the `positions_raw`, `matchinformation` and `events_raw` files | `dfl_idsse/` | training and cross-validation | CC-BY 4.0, used with this attribution |
| SkillCorner open data, 20 A-League matches, broadcast tracking: `git clone https://github.com/SkillCorner/opendata skillcorner` | `skillcorner/` | test only (real-tracking condition), never in the submitted models' training | MIT |
| PFF FC World Cup 2022 tracking and events (16 of 64 matches scored) | `pff_wc2022/` | test only (one extra check), never in training | PFF FC's terms of use, accepted at download |

```bash
export MTAH_OPEN_DATA_DIR=/path/to/external_data       # PowerShell: $env:MTAH_OPEN_DATA_DIR = "..."
python -m tempolens.train --member 0 --draws 4 --out models/retrained_m0.pt                   # ENS5 recipe
python -m tempolens.train --member 0 --draws 4 --min-spell 50 --out models/retrained_ms50_m0.pt  # ms50 recipe
```

## Tests

```bash
pip install pytest
python -m pytest -q tests                  # skips the one slow end-to-end test
MTAH_E2E=1 python -m pytest -q tests       # all of them; PowerShell: $env:MTAH_E2E = "1" first
```

The tests run on synthetic files: they need neither the match data nor the open data. The
end-to-end test runs `tempolens.predict` on a fake bundle in the real file formats and checks that
two runs write the same bytes; it is the only test of the path from the tracking files to the CSV.

## Licence

[LICENSE](LICENSE) (MIT) covers the contents of this repository. As its scope note says, it does
not cover the match data, which is licensed for the competition only and is not in this repository.
