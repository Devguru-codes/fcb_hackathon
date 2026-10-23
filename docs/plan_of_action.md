# TempoLens — plan of action and experiment tracker

The single list of everything tried, running and planned, with the measured result. Updated
whenever an experiment finishes. Numbers are aggregates only (no frame-level data: the repository
becomes public at submission).

**Legend:** ✅ works — adopted in the pipeline · ❌ tested — did not help, not adopted ·
⏳ running · ⬜ to do · 🚫 blocked (reason given) · 👤 needs a person (team)

## Decision rules

1. **Model selection = cross-validation (CV)** on 10 open matches (7 DFL, 3 Metrica) under the
   simulated competition tracking, scored with the official `possession_metrics.py`. Compare only
   runs on the **same simulator version** (v1 and v2 numbers are not comparable).
2. **Noise floor (measured, Q10):** the same model retrained with seeds 0/1/2 scores 0.7543 /
   0.7556 / 0.7599 (all; sd 0.0029, range 0.0056) and 0.7456 / 0.7439 / 0.7524 (DFL; sd 0.0045).
   A single-seed gain must exceed ~0.006 overall (2 sd); smaller gains need ≥ 2 seeds agreeing.
3. **Real match = sanity, not tuning:** the public 5 minutes (17 possessions) and aggregate
   real-match diagnostics (dead-ball share around Smart Tagging restarts, sim-vs-real feature
   shift) may veto a change, never select between close variants.
4. **Reproduction gate before any upload:** a clean checkout of the commit must rebuild the
   submission byte for byte, and `validate_submission.py` must pass.

## Current submission

| | |
|---|---|
| version | **v4** — features v3, simulator v2, 3-GRU ensemble, decoder switch 4 / 8 / 8, change bonus 1.0 |
| file | `submissions/tempolens_v4b.csv` (outside git), sha256 `e23d5c40…`, validated, reproduced byte for byte from commit `32b3a36` |
| CV (simulator v2) | 0.764 all / 0.754 DFL |
| public window (sanity) | spellF1 0.573 / macroF1 0.790, 20 possessions vs 17 (official baseline 0.326 / 0.621) |
| full match | 322 possessions, mean 11.5 s; dead ball 41.9% of graded frames |

## A. Pipeline milestones

| | item | result |
|---|---|---|
| ✅ | Learned pipeline: open data + ATD simulator + identity-free features + bi-GRU + Viterbi | replaces the hand-tuned Codex decoder (removed) |
| ✅ | ATD simulator calibrated to the official baseline's published full-match scores | reproduces their ranking order |
| ✅ | 3-member ensemble (seeds × noise draws) | +0.011 DFL over one GRU (simulator v1) |
| ✅ | Self-calibrated Smart Tagging prior, built and kept **off** | real tags 83%/94% reliable, but lowered public spellF1 (0.43 vs 0.59) |
| ✅ | Diagnostics on the real data (aggregates only) | alignment exact; organisers' baseline reproduced on 100% of frames |
| ✅ | Float64 inference + 4-decimal rounding | same file on other CPUs; on this machine 0 of 199,319 frames changed |
| ✅ | Batched inference | submission rebuild 26 → 6.5 min, byte-identical |
| ✅ | Tracking loader refuses files with missing frames | — |
| ✅ | Two-run byte-identity test (second run with the numpy decoder) | passes |
| ✅ | Weights record their feature version; `decoder.json` names its ensemble members | prevents silent mismatches |
| ✅ | Experiment caches keyed by `SIM_VERSION` / `TRAIN_VERSION` | no stale CV results |

## B. Bugs found and fixed (each found on the real data, invisible to CV)

| | bug | effect | fix |
|---|---|---|---|
| ✅ | Simulator per-frame jitter = fake speed (sim players 3.8 m/s vs real 1.5) | model called 88% of the match dead; public spellF1 0.13 | jitter off → 0.59 |
| ✅ | Own-half share divided by the slot count (46 real slots vs ~15 per team) | shifted input on every real frame | features v3 |
| ✅ | Nearest-player speed taken from an arbitrary slot when the ball is missing | noise on 41% of real frames | features v3 |
| ✅ | Metrica never tracks the dead ball → "no ball = dead" shortcut | dead ball missed at goal kicks (30–49%) and throw-ins | simulator v2: dead-time ball filled; goal kicks 56–71%, public macroF1 0.73 → 0.79 |
| ✅ | Wrong possession-count reasoning ("we merge possessions") | — | corrected: ~275–322 predicted vs ~280 implied by the public mean length |

## C. Experiments tested

| | experiment | result (DFL / all spellF1 unless noted) | verdict |
|---|---|---|---|
| ❌ | LightGBM on the same features | 0.566 DFL vs GRU 0.719 (sim v1) | GRU kept |
| ❌ | Player-set attention encoder (PathCRF-style) | 0.714 vs 0.719 alone; 0.726 vs 0.730 in ensemble | slower, no gain |
| ❌ | TCN | 0.717; ensemble 0.727 | no gain |
| ❌ | Bi-LSTM | 0.707; ensemble 0.729 | no gain |
| ❌ | Bigger GRU (128 units, 16 epochs) | 0.715; ensemble 0.731 | within noise |
| ❌ | Transformer over tracking | 0.656; ensemble 0.724 | too coarse |
| ❌ | Hidden-ball auxiliary head | 0.699 / 0.710 | hurts |
| ❌ | Minimum-duration Viterbi | +0.001 (sim v2) | not the bottleneck |
| ❌ | Minimum-Bayes-risk decoding | +0.0007 (sim v2) | not the bottleneck |
| ❌ | Oracle per-match switch cost (ceiling) | +0.0024 | decoding is not the bottleneck |
| ❌ | Decoder switch cost 1.0 / 1.5 | worse than 2–4 | grid kept |
| ✅ | Change-head bonus in the decoder | helps from v4 on (weight 1.0) | adopted |
| ❌ | MS-TCN++ smoothing loss 0.15 | 0.750 / 0.754 vs 0.746 / 0.754 | tie |
| ❌ | Video ball detector (TrackNet-style) on the real video | 57% within 15 px, median 12.6 px | too imprecise, slow, reproduction risk |
| ❌ | Broadcast tracking as second ball source | aligns 39.5% of broadcast frames; fills 5.3% of ball gaps | low value |
| ✅ | Simulator recalibration from the sim-vs-real feature audit | ball-acceleration shift 1.01 → 0.38; player count 1.40 → 0.70 | in simulator v2 |
| ❌ | Smart Tagging set-piece starts as restart timing | tag start ≈ set-up phase, median 4.6 s before our restart | not a timing cue |
| ❌ | Q1 segment-length-balanced loss (exponent 0.5) | 0.751 / 0.755 vs 0.746 / 0.754 (DFL +0.005, Metrica −0.010) | no gain beyond noise (seed-1 replicate queued) |
| ❌ | Q2 focal loss (γ = 2) | seed 0 −0.0023, seed 1 +0.0030 (mean +0.0004) | noise |

## GitHub scan for data and code (2026-10-23)

| | source | finding |
|---|---|---|
| ✅ | SkillCorner/opendata (MIT) | 20 matches with per-frame owner + phases of play — adopted as H1 |
| ❌ | DFL `positions_raw_observed` | already used; the ball is never missing (no raw faults to learn from) |
| ❌ | yangpeduto/symmetrica | rule-based "control with memory" on 2 Metrica games, 93% frame accuracy — weaker than ours |
| ❌ | GeonHeeJo2000/following-the-ball | event detection on the same 7 DFL matches — no new data |
| 🚫 | hyunsungkim-ds/ballradar, pathcrf | need persistent player identities (the ATD has none) |
| ⬜ | hyunsungkim-ds/elastic, ML-KULeuven/ETSY, databallpy | event–tracking synchronisation: only useful with event data we do not have for the match |

## D. Queue (ranked by expected gain ÷ cost)

| | id | hypothesis | why (measured weakness) | test | cost |
|---|---|---|---|---|---|
| ✅ | Q10 | **Multi-seed variance check** — done: sd 0.0029 all / 0.0045 DFL (see rule 2); seed-1 replicates of each variant still queued | decisions rest on 0.001–0.005 differences; noise floor unknown | score members 0–2 separately from cached CV outputs; then paired per-match comparison of any variant over ≥3 seeds | minutes (cached), then GPU per seed |
| ❌ | H1 | **More training data**: SkillCorner open data (20 A-League matches, broadcast tracking, labels from phases of play + owner flag). Training-only extra data, same CV folds: seed 0 −0.0006, seed 1 −0.0116 (mean −0.006; Metrica −0.023). Not adopted. Likely cause: SkillCorner teaches longer *controlled* possessions (median 5–7 s) while the DFL/Metrica test labels keep short ones — CV cannot tell which definition the private truth uses. PFF World Cup 2022 🚫: form not accessible | only 10 training matches | CV with vs without, 2 seeds | done |
| ⬜ | H1b | SkillCorner-trained ensemble checked on the real match only (public window + dead-ball diagnostics), since CV labels may not share the truth's definition of control — sanity-level evidence only (17 possessions) | H1 tie/loss may be a label-definition artefact | needs `train --extra-train`; compare with v4 | ~3 h |
| ⬜ | H3 | Training labels match "control": minimum possession 19 frames (public truth's shortest) instead of 10 | short possessions recalled 45% (27% of all possessions) | retrain; per-duration recall + public sanity | hours |
| ❌ | H6 | Mirror augmentation (flip pitch y, as in TacticAI): seed 0 +0.0023, seed 1 −0.0039 (mean −0.0008) | data scarcity | CV, 2 seeds | done |
| ⬜ | H5 | Restart-spot features (stationary ball at corner arc / touchline / 6-yard box, players walking) | real dead-ball recall 0.48 | CV + goal-kick/throw-in diagnostic | half a day + retrain |
| ❌ | H7 | 5 members instead of 3: 0.7647 vs 0.7638 (+0.0009). Stochastic weight averaging not tried (same expected size) | small, safe | CV | done |
| ⬜ | H2 | Adapt to the real match (domain-adversarial à la SSTDA, or self-training on confident frames) | real dead ball looks like live play | public + diagnostics | ~1 day · 🚫 until organisers answer whether weights derived from match data may be committed |
| ⬜ | H8 | Use the ATD's own per-object speed | tracker-smoothed speeds unused | needs simulating for open data | small |
| ⬜ | H9 | Full HSMM with learned duration distributions | — | decode study | low value (decoder ceiling +0.002) |
| 🚫 | H10 | Label-shift correction (Saerens EM, logit adjustment) | — | — | assumes classes look alike in train and test; real dead balls do not |
| 🚫 | V1 | Stronger video ball detector (WASB) | ball missing on 41% of frames | held-out audit | blocked on organiser question #3 (weights from the match video) |
| 🚫 | V2 | Video event spotting (ball out, throw-ins) | dead-ball onsets most missed | — | no labels for the tactical camera; weeks |
| 🚫 | V3 | Player re-identification → identity-based models (PathCRF, Ball Radar) | — | — | weeks |

## How experiments run unattended

`python -m tempolens.autoqueue` (started in the background; works offline) runs the jobs in
`experiments/queue.json` one after another, waits while any other training/CV run is active,
parses each result and rewrites table F below. To add a test, append a job to the queue file;
to stop, create `experiments/STOP`. Logs: `experiments/logs/` (not committed). It stops when the
laptop sleeps, reboots or the Claude app restarts (2026-10-23 03:25: one job lost 2 folds; finished folds
are cached, so a restart resumes). Restart from the repository root:

    python -m tempolens.autoqueue

## E. People and admin

| | task | owner | deadline |
|---|---|---|---|
| 👤 | Register the team (3–5 members, same team name everywhere) | team | **2026-11-10** |
| 👤 | Post `docs/organiser_questions.md` on the challenge repo | team | asap |
| 👤 | Upload the possession file (named `<team_name>.csv`), repository URL, technical document; edit later if a better version lands | team | **2026-11-17** |
| 👤 | Make the repository public at submission time (history checked: no match data) | team | at submission |
| 👤 | Optional: hand-label 20–30 min for validation only (allowed by RULES.md), locally, never sent to a cloud LLM | team | — |

## F. Automatic experiment queue

<!-- autoqueue:start -->
Written by `python -m tempolens.autoqueue` from `experiments/queue.json` (✅ here = finished; adoption still follows the decision rules above).

| | id | experiment | all | DFL | metrica | best costs | vs baseline | finished |
|---|---|---|---|---|---|---|---|---|
| ✅ | Q1 | segment-balanced loss 0.5 (seed 0) | 0.7551 | 0.7510 | 0.7645 | (8.0, 4.0, 8.0, 0.5) | no gain (+0.0008) | 2026-10-22 21:50 |
| ✅ | Q2 | focal loss gamma 2 (seed 0) | 0.7520 | 0.7398 | 0.7805 | (1.0, 4.0, 4.0, 0.0) | no gain (-0.0023) | 2026-10-22 21:54 |
| ✅ | Q10a | seed check: default GRU, member 1 | 0.7556 | 0.7439 | 0.7828 | (4.0, 8.0, 8.0, 0.5) | measured | 2026-10-22 21:57 |
| ✅ | Q10b | seed check: default GRU, member 2 | 0.7599 | 0.7524 | 0.7773 | (4.0, 8.0, 8.0, 0.0) | measured | 2026-10-22 22:01 |
| ✅ | H1sc | + 20 SkillCorner matches in training (seed 0) | 0.7537 | 0.7448 | 0.7744 | (2.0, 4.0, 8.0, 0.0) | no gain (-0.0006) | 2026-10-22 22:55 |
| ✅ | H1sc_s1 | + 20 SkillCorner matches in training (seed 1) | 0.7440 | 0.7375 | 0.7594 | (4.0, 2.0, 4.0, 0.5) | no gain (-0.0116) | 2026-10-22 23:49 |
| ✅ | Q1s1 | segment-balanced loss 0.5 (seed 1) | 0.7592 | 0.7508 | 0.7789 | (8.0, 8.0, 8.0, 1.0) | no gain (+0.0036) | 2026-10-23 00:21 |
| ✅ | Q2s1 | focal loss gamma 2 (seed 1) | 0.7586 | 0.7484 | 0.7824 | (1.5, 8.0, 8.0, 0.0) | no gain (+0.0030) | 2026-10-23 00:52 |
| ✅ | H6 | mirror augmentation (seed 0) | 0.7566 | 0.7490 | 0.7743 | (6.0, 8.0, 8.0, 0.5) | no gain (+0.0023) | 2026-10-23 01:26 |
| ✅ | H6s1 | mirror augmentation (seed 1) | 0.7517 | 0.7415 | 0.7756 | (8.0, 8.0, 8.0, 0.5) | no gain (-0.0039) | 2026-10-23 01:59 |
| ✅ | H7 | 5-member ensemble (vs 3) | 0.7647 | 0.7554 | 0.7864 | (1.5, 8.0, 8.0, 0.0) | no gain (+0.0009) | 2026-10-23 03:06 |
| ⏳ | Q1s2 | segment-balanced loss 0.5 (seed 2) |  |  |  |  |  |  |
<!-- autoqueue:end -->
