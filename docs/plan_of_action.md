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
2. **Noise floor:** a variant must beat its baseline by more than the seed-to-seed spread
   (to be measured, see Q10; until then ~+0.005 mean spellF1) without losing elsewhere.
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
| ⏳ | Q1 segment-length-balanced loss (exponent 0.5) | baseline member 0: 0.746 DFL / 0.754 all | running |
| ⏳ | Q2 focal loss (γ = 2) | same baseline | queued after Q1 |

## D. Queue (ranked by expected gain ÷ cost)

| | id | hypothesis | why (measured weakness) | test | cost |
|---|---|---|---|---|---|
| ⬜ | Q10 | **Multi-seed variance check** | decisions rest on 0.001–0.005 differences; noise floor unknown | score members 0–2 separately from cached CV outputs; then paired per-match comparison of any variant over ≥3 seeds | minutes (cached), then GPU per seed |
| ⬜ | H1 | **More training data**: PFF FIFA World Cup 2022 (64 matches), SkillCorner open data (10 matches, possession events) | only 10 training matches; broadcast-tracked data is noisy like the real ATD | loaders + labels; CV on the same DFL folds with/without | 1–2 days · 👤 PFF form needs an email |
| ⬜ | H3 | Training labels match "control": minimum possession 19 frames (public truth's shortest) instead of 10 | short possessions recalled 45% (27% of all possessions) | retrain; per-duration recall + public sanity | hours |
| ⬜ | H6 | Mirror augmentation (flip pitch y), reflection symmetry as in TacticAI | data scarcity | CV | ~2 h |
| ⬜ | H5 | Restart-spot features (stationary ball at corner arc / touchline / 6-yard box, players walking) | real dead-ball recall 0.48 | CV + goal-kick/throw-in diagnostic | half a day + retrain |
| ⬜ | H7 | 5 members instead of 3; stochastic weight averaging | small, safe | CV | GPU only |
| ⬜ | H2 | Adapt to the real match (domain-adversarial à la SSTDA, or self-training on confident frames) | real dead ball looks like live play | public + diagnostics | ~1 day · 🚫 until organisers answer whether weights derived from match data may be committed |
| ⬜ | H8 | Use the ATD's own per-object speed | tracker-smoothed speeds unused | needs simulating for open data | small |
| ⬜ | H9 | Full HSMM with learned duration distributions | — | decode study | low value (decoder ceiling +0.002) |
| 🚫 | H10 | Label-shift correction (Saerens EM, logit adjustment) | — | — | assumes classes look alike in train and test; real dead balls do not |
| 🚫 | V1 | Stronger video ball detector (WASB) | ball missing on 41% of frames | held-out audit | blocked on organiser question #3 (weights from the match video) |
| 🚫 | V2 | Video event spotting (ball out, throw-ins) | dead-ball onsets most missed | — | no labels for the tactical camera; weeks |
| 🚫 | V3 | Player re-identification → identity-based models (PathCRF, Ball Radar) | — | — | weeks |

## E. People and admin

| | task | owner | deadline |
|---|---|---|---|
| 👤 | Register the team (3–5 members, same team name everywhere) | team | **2026-11-10** |
| 👤 | Post `docs/organiser_questions.md` on the challenge repo | team | asap |
| 👤 | Upload the possession file (named `<team_name>.csv`), repository URL, technical document; edit later if a better version lands | team | **2026-11-17** |
| 👤 | Make the repository public at submission time (history checked: no match data) | team | at submission |
| 👤 | Optional: hand-label 20–30 min for validation only (allowed by RULES.md), locally, never sent to a cloud LLM | team | — |
