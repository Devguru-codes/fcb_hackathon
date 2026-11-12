# TempoLens

A learned possession-reconstruction pipeline for **More than a Hack 2027**: a temporal model
trained on open football data under realistic simulated tracking noise, decoded for whole
possessions (`spellF1`), not for single frames.

The submission is `Coding United.csv` (sha256 `33ad4559…`); the short technical document is
[docs/Coding United.md](docs/Coding%20United.md). Method and results in full: [docs/technical_document.md](docs/technical_document.md). Every experiment
tried, running and planned, with its measured result and the decision rules:
[docs/plan_of_action.md](docs/plan_of_action.md).

## Why it is built this way

Two facts about the competition tracking (ATD), both measurable from the organisers' public
files, shape everything:

- **The ball is missing for 41% of the match, in long blind stretches**: ~150 gaps with a median
  of 10.7 s. Possession has to be read from how the 22 players move for a large part of the
  game, so ball-free inference is the core of the problem.
- **The public 5 minutes are unrepresentative**: only ~12% of their frames lack the ball, and
  they hold 17 possessions. They can check a pipeline, not choose between versions of it.

So models are trained and compared on **open matches with known possession** — Metrica's 3
sample games and the 7-match DFL/Sportec dataset (CC-BY 4.0), and tested on real broadcast tracking
(20 SkillCorner matches, 16 PFF FC matches) never trained on — after corrupting their tracking
with a **simulator calibrated so the official baseline reproduces its published scores**.

## Pipeline

| stage | module | what it does |
|---|---|---|
| data | `match.py`, `opendata.py` | one per-frame table (metres, anonymous slots) for the ATD and every open match; open labels from DFL's ball status/owner flag or Metrica events |
| noise | `simulate.py` | stoppage-aligned and long log-normal ball gaps, player dropout (no position jitter: the real tracking is smooth) |
| features | `features.py` | ~140 identity-free, team-symmetric features: control geometry, ball state, formation, memory across ball gaps, context windows |
| model | `temporal.py` | bidirectional GRU: P(H/A/D) per frame plus a possession-change head, **ten members** trained with domain randomisation (each noise draw its own simulator settings): five on the provider labels with the 10-frame fold (ENS5, `tempolens_rnd4_m*.pt`) and five with DFL labels folded at 50 frames, Metrica's unchanged (ms50, `tempolens_ms50_m*.pt`; the "controlled possession" hedge, plan row LCB-ADOPT); LightGBM in `experiments.py` as a comparison |
| adaptation | `temporal.py`, `adaptcv.py`, `adaptseeds.py` | test-time adaptation, **on** in the submission: feature re-scaling halfway to the match's own statistics (`tta_alpha` 0.5) and 10 EATA steps on the output layer, gate off. Chosen on open data only (`python -m tempolens.adaptseeds --members 5`) by a rule fixed before the run: beat no adaptation for the 5-member ensemble and for each member alone. Two settings passed and tied within 0.0001 (TTA alone 0.6476, TTA + EATA 10 0.6475); the rule's tie-break picks TTA alone, and the team chose TTA + EATA 10 after seeing the results, for its larger smallest per-member gain and its larger gain on real tracking (technical document §7c). Chosen on ENS5 and kept unchanged for the 10-member mix |
| decoding | `decoder.py` | Viterbi with switch/stoppage costs (4, 8, 8) and a possession-change bonus (1.0), chosen with the adaptation on open data (the cross-validated argmax without adaptation is (2, 8, 8, 0.5)); dead ball before the second-half kickoff (FAQ) |
| tags | `tags.py`, `tagprior.py` | Smart Tagging as a prior on the side in possession, **on** in the submission, weighted without labels by how often the tags agree with confident frames; defensive codes point at the other team |
| checks | `realcheck.py` | automatic real-match checks that print aggregates only (format, label shares) |

## Run

```bash
pip install -r requirements-predict.txt     # the submission only: numpy, pandas, torch, numba
pip install -r requirements-tempolens.txt   # training and studies (adds lightgbm, kloppy)

# The submission, from committed weights, CPU-only and deterministic:
python -m tempolens.predict --data-root data --output "Coding United.csv"
python evaluation/validate_submission.py "Coding United.csv"

# Data-day checks; prints aggregate numbers only:
python -m tempolens.diagnostics --data-root data

# Research loop on open data (MTAH_OPEN_DATA_DIR -> folder with metrica_sample_data/, dfl_idsse/):
python -m tempolens.experiments --model gbm
python -m tempolens.experiments --model temporal
python -m tempolens.train --member 0 --draws 4 --out models/retrained_m0.pt                    # ENS5; members 1-4 likewise
python -m tempolens.train --member 0 --draws 4 --min-spell 50 --out models/retrained_ms50_m0.pt  # ms50; members 1-4 likewise
python -m tempolens.predict --data-root data --output retrained.csv --model models/retrained_m0.pt ... models/retrained_ms50_m4.pt   # the ten files
# Never write retrained weights over the submitted members (models/tempolens_rnd4_m*.pt, models/tempolens_ms50_m*.pt): GPU training is not bit-reproducible.
```

`predict` uses the ensemble members listed in `models/decoder.json` (the costs there were tuned for
exactly them); `--model` overrides it. Weights record the feature version they were trained on and
refuse to load against another.
The same file sets the tag prior and the test-time adaptation (`tag_prior`, `tta_alpha`, `eata_steps`).
Every setting there is chosen on open data only, as the organisers' rules of 2026-11-03 require
(technical document §7c). Everything runs on the
CPU in float64 with a fixed order, so the same data gives the same file byte for byte; a full run
takes 14–24 minutes on a laptop CPU with the 10 members (EATA adds the head inputs of both team views).

## Open data

- Metrica Sports sample data: `git clone https://github.com/metrica-sports/sample-data
  external_data/metrica_sample_data`
- DFL/Sportec (Bassek et al., *Scientific Data*, CC-BY 4.0), figshare article 28196177: the
  `positions_raw`, `matchinformation` and `events_raw` files into `external_data/dfl_idsse/`.
- SkillCorner open data (MIT), test only: `git clone https://github.com/SkillCorner/opendata
  external_data/skillcorner`
- PFF FC World Cup 2022 (PFF's terms of use), test only: tracking, event, metadata and roster files
  into `external_data/pff_wc2022/`.

## Data safety

Competition data never enters the repository (`data/` is gitignored) and must not be sent to a
hosted LLM. Trained weights come from open data only. `tempolens.diagnostics` prints aggregates
only; nothing in this package exports frame-level tracking.

`experiments/queue.json` is the history of the research queue (`tempolens.autoqueue`), kept as run, with
its paths made repo-relative (`data`, `derived/`, `$MTAH_OPEN_DATA_DIR`; `$SCRATCH` for one-off
scripts that are not in the repository). It includes public-window scoring runs from before the
rules clarification (technical document §7b). It holds no data and the submission does not read it.
