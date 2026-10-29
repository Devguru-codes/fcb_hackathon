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
| version | **v7+TTA+EATA** — features v4, simulator v2, 3-GRU ensemble trained with domain randomisation (4 noise draws per match, each with its own simulator settings); on the match: feature scaling re-fitted (`tta_alpha` 1), then each member's output layer adapted, EATA-style (`eata_steps` 30); decoder switch 4 / 8 / 8, change bonus 1.0 |
| file | `submissions/tempolens_final.csv` (outside git), sha256 `b483d0a3…`, validated, reproduced byte for byte by `python -m tempolens.predict` from a clean checkout of commit `88bed2e` |
| public window (sanity) | spellF1 0.647 / macroF1 0.8725, 17 possessions vs 17, dead-ball recall 0.728 (precision 0.876) — v7+TTA 0.629 / 0.849; v5 0.594 / 0.819 |
| human review | vs v7+TTA: all 24 disagreements, EATA right 20, v7+TTA 4 (20 longest 16 vs 4). v7+TTA vs v5: 45 vs 33 |
| full match | 301 possessions, mean 11.2 s |
| previous | v7+TTA `tempolens_v7_final.csv` (sha256 `83d89f66…`, commit `c3ba16f`); v5 + kickoff fix (commit `1b061bb`) |

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
| ❌ | Q1 segment-length-balanced loss (exponent 0.5) | seeds 0/1/2: +0.0008 / +0.0036 / −0.0001 (mean +0.0014) | noise |
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

## Research report review (agent report `github_repo.md`, 2026-10-23)

Links checked through the GitHub API and web search. Kept: Lang et al. (Sci. Rep. 2022, in/out of
play from player positions alone, 92% frame accuracy) → K; pitch control (LaurieOnTracking, MIT)
→ next; domain adaptation (AdaTime/SSTDA, MIT) = H2; 19-frame label minimum = H3. Discarded: the
report's code references (emission bonus, `merge_short_runs`, `tracking.py`) describe code removed
long ago; its gain estimates are unmeasured; "Renner et al. arXiv:2610.11571" was not found; ASRF's
dynamic switch cost is what our change bonus already does.

## D. Queue (ranked by expected gain ÷ cost)

| | id | hypothesis | why (measured weakness) | test | cost |
|---|---|---|---|---|---|
| ✅ | Q10 | **Multi-seed variance check** — done: sd 0.0029 all / 0.0045 DFL (see rule 2); seed-1 replicates of each variant still queued | decisions rest on 0.001–0.005 differences; noise floor unknown | score members 0–2 separately from cached CV outputs; then paired per-match comparison of any variant over ≥3 seeds | minutes (cached), then GPU per seed |
| ❌ | H1 | **More training data**: SkillCorner open data (20 A-League matches, broadcast tracking, labels from phases of play + owner flag). Training-only extra data, same CV folds: seed 0 −0.0006, seed 1 −0.0116 (mean −0.006; Metrica −0.023). Not adopted. Likely cause: SkillCorner teaches longer *controlled* possessions (median 5–7 s) while the DFL/Metrica test labels keep short ones — CV cannot tell which definition the private truth uses. PFF World Cup 2022 🚫: form not accessible | only 10 training matches | CV with vs without, 2 seeds | done |
| ⬜ | H1b | SkillCorner-trained ensemble checked on the real match only (public window + dead-ball diagnostics), since CV labels may not share the truth's definition of control — sanity-level evidence only (17 possessions) | H1 tie/loss may be a label-definition artefact | needs `train --extra-train`; compare with v4 | ~3 h |
| ⬜ | H3 | Training labels match "control": minimum possession 19 frames (public truth's shortest) instead of 10 | short possessions recalled 45% (27% of all possessions) | retrain; per-duration recall + public sanity | hours |
| ❌ | H6 | Mirror augmentation (flip pitch y, as in TacticAI): seed 0 +0.0023, seed 1 −0.0039 (mean −0.0008) | data scarcity | CV, 2 seeds | done |
| ✅ | H5 | Restart-spot features (features v4, branch `exp/restart-features`, kept): CV seed 0 −0.0031, seed 1 +0.0015 (mean −0.0008, neutral). Real match mixed: public spellF1 0.594 vs 0.573, macroF1 0.819 vs 0.790, 18 vs 20 possessions (17 true), dead-ball recall 0.534 vs 0.478 — but dead ball before goal-kick tags fell (0.37/0.59 vs 0.56/0.71), the very case targeted. Rule 3 says a CV tie cannot be decided by the public window; **adopted by the team's call** (gains on all four public numbers, 15-min switch, reversible) | real dead-ball recall 0.48 | CV 2 seeds + real-match checks | done |
| ❌ | H7 | 5 members instead of 3: 0.7647 vs 0.7638 (+0.0009). Stochastic weight averaging not tried (same expected size) | small, safe | CV | done |
| ⬜ | H2 | Adapt to the real match (domain-adversarial à la SSTDA, or self-training on confident frames) | real dead ball looks like live play | public + diagnostics | ~1 day · 🚫 until organisers answer whether weights derived from match data may be committed |
| ❌ | K | Collective kinematics features (features v5, branch `exp/kinematics`): fastest player's speed, 90th-percentile speed, share of players still, 2 s means — after Lang et al. 2022 | real dead-ball recall 0.53 | CV +0.004 / −0.012 (2 seeds disagree); public spellF1 0.573 vs v5 0.594, macroF1 0.810 vs 0.819, 20/17 spells, dead-ball recall unchanged 0.534 | not adopted; v5 stays |
| ✅ | B1 | Second half's pre-kickoff frames 110,808–113,741 are graded and all D (FAQ.md); the model labelled only 22% of them D | one false ~2-min possession plus ~2,300 wrong frames on the private truth; invisible on the public window | post-pass in `predict.py`, asserted in the end-to-end test | done |
| ❌ | DW | Dead-ball frames weigh 2x in the loss (`--dead-weight 2`, the simplest dead-ball specialist) | real dead-ball recall 0.53 | CV seed 0 −0.0010, seed 1 −0.0015 | not adopted |
| ❌ | KP | Keeper-hold features (branch `exp/keeper`): ball distance to each team's deepest player, ball in a penalty box | keeper catches called dead ball (video review) | CV seed 0 −0.0075, seed 1 −0.0055 | not adopted |
| ❌ | BF | Boundary features after the best public rival (branch `exp/boundary`): frames since/until each team touched the ball, nearest-team flips within ±1/±2 s | rival's macroF1 0.874 vs 0.819 | CV seed 0 −0.0023, seed 1 −0.0054 | not adopted |
| ⏸ | BC | Broadcast ball fill, one constant offset per half (rival's variant): offsets 4014 / 26,920 frames, 65% of 4 s blocks verified on players, fills 9.7% of ball gaps (missing ball 40.7% -> 36.7% of graded); where both see the ball, median 0.77 m apart, 74% within 2 m | ball missing on 41% of graded frames | public window unchanged (spellF1 0.5943 both; only 159 filled frames fall inside it). Full match: fill-all changes 2.1% of graded frames (35 stretches >= 1 s, 321 possessions vs 313); fill-alive 1.4% (20 stretches, 316) | needs a human look at the changed stretches (review reel); no open-data CV possible |
| ✅ | SC | Simulator calibration: where the ball goes missing, real vs simulated | CV 0.75 vs real public 0.59 | Ball missing on 40.7% of graded frames in both, and the share inside gaps of 20 s or more matches (real 0.68, sim 0.65–0.78). Around the 59 set-piece tags (5 s before / 3 s after) the real ball is missing 0.73 / 0.65 of the time vs sim P(missing given dead) 0.70. Implied real P(missing given live) is about 0.20 if the match is ~42% dead, vs sim 0.19. The public window is atypical (ball missing only ~12%; its dead frames missing 0.34) | no large mismatch: no retrain. Possible small follow-up: train with fewer lost stoppages (dead_cover < 0.67) so dead ball is learned with the ball visible; judged on the real match only |
| ❌ | RN | **Domain randomisation** (`--draws 4`): each training match gets 4 noise draws with their own simulator settings (ball missing 30–50%, stoppages lost 40–90%, gap median 6–18 s, player dropout 2–8%, ball noise 0.1–0.4 m); held-out matches keep the fixed test noise | CV 0.75 vs real 0.59 | **CV: single members +0.0064 / +0.0106 / +0.0130 (seeds 0/1/2); 3-member ensemble 0.7716 vs v5 ensemble 0.7636 (+0.0080, DFL +0.0071, 8 of 10 matches); ahead at every decoder cost tried (+0.006 to +0.013), at v5's submitted costs (4, 8, 8, 1.0) 0.7682 vs 0.7591.** 2 draws: −0.004 (mean); 4 draws + width 160: +0.005. Real match, v7 at v5's costs (`tempolens_v7_rnd4.csv`): public spellF1 0.582 vs 0.594 (16 vs 18 possessions, truth 17; bootstrap 95% CI 0.41–0.74 vs 0.42–0.74), macroF1 0.820 vs 0.819, dead-ball recall 0.542 vs 0.534, 292 vs 313 possessions; less dead ball before goal kicks (0.48 vs 0.59 over the last 5 s) and corners (0.82 vs 0.91), more before kick-offs. v7 at its CV-tuned costs (2, 2, 2, 0.5): public 0.549, so cheap switches do not transfer | **not adopted: full human review of all 71 v5/v7 disagreements: v5 right 38, v7 right 29** (v7 merges across real turnovers). Follow-ups: + tracker faults seed 0 −0.0052 vs v7 recipe; 6 draws seed 0 −0.0038 (plateau); seed-1 runs cancelled. Lesson: a CV gain under the simulator did not transfer to the real match |
| ✅ | HR | Human review of every disagreement of 1 s or more (user, 2026-10-25) | the real match is the only truth that counts | v4 vs v5: v5 right 34/62, v4 24, both wrong 3, unclear 1. v5 vs v7: v5 right 38/71, v7 29. Failure modes: v4 splits open play with false dead stretches; v5 misses goal-kick and restart dead ball; v7 merges across turnovers; both-wrong clips are restarts and aerial handovers | v5 confirmed as the submission; next target: dead ball at restarts |
| ✅ | TTA | **Test-time re-normalisation** (`temporal.adapt_normaliser`): each member's feature scaling moved to the target match's own medians and spreads, no labels (AdaBN-style; organisers allow test-time adaptation, per the user) | sim-to-real gap; dead ball at restarts | Open data, paired (same weights, no shift to correct): alpha 0.5 −0.006, alpha 1 −0.019. **Real match, alpha 1:** v7 models: public spellF1 0.629 vs v5 0.594 (CI 0.46–0.79), macroF1 0.849 vs 0.819, dead-ball recall 0.647 vs 0.534 (precision 0.88), last 5 s before goal kicks dead 0.77 vs 0.59, corners 0.99 vs 0.91; v5 models: spellF1 0.594, macroF1 0.851, dead-ball recall 0.680. Full match dead 0.455–0.465 vs 0.421 | real match improves on every measure we have; open data cannot show a shift it does not contain. **adopted with v7 (submission v7+TTA)**: human review of all 88 disagreements 45 vs 33 for v7+TTA |
| ✅ | EATA | Entropy-gated adaptation of each member's output layer on the real match (EATA, Niu et al. 2022; from the rival sweep), on top of test-time re-normalisation: full-batch Adam on frames with entropy < 0.4 ln 3, weighted exp(E0 − E), halves only, CPU float64 | tie-break gap, dead ball | public: 10 steps 0.629 / 0.852, **30 steps spellF1 0.647, macroF1 0.8725, 17 of 17 possessions, dead-ball recall 0.728 (precision 0.876)**; submitted 0.629 / 0.849 / 0.647 | **adopted**: human review 20 vs 4 (all 24 disagreements); `eata_steps` 30 in `decoder.json` |
| ❌ | BD | Turnover back-dating (Remontada / Polyamal / INVICTA): move each H↔A change back to the new team's first close contact | rivals' causal rules switch late | open-data CV, v7 ensemble: 50-frame lookback 0.7066 vs 0.7682, 25 frames 0.7356; v5 the same. Our bidirectional model plus Viterbi has a median offset of 0 frames on ~4,000 open-data boundaries | not adopted |
| ❌ | BD8 | Back-dating limited to 8 frames (rival sweep VII) | — | open-data CV 0.7622 vs 0.7682, macroF1 0.9378 vs 0.9414 | not adopted |
| ❌ | SG | Penalty-box scramble guard: never dead ball while 8+ players are in a box (rival sweep VII) | claimed false dead ball in scrambles | real match: crowded box with the ball missing is 9.9% of graded frames; inside the public window the **truth is dead ball on 71%** of them (submission 70%) — mostly set pieces being taken, so the guard would flip real stoppages | not adopted |
| ❌ | TCN | CoreOrigin-style residual corrector: frozen v7 ensemble, a 6-block dilated TCN (64 channels, dilations 1–32) reading the ensemble's held-out output plus features adds a zero-initialised residual; short possessions weighted ×3 | rivals' 'boundary smear' claim | open-data CV (5 folds, corrector trained on held-out outputs): **0.7373 vs 0.7682**, worse on 9 of 10 matches; real match on v7+TTA: public 0.582 / 0.820 vs 0.629 / 0.849, **380 possessions vs 302** (fragments play) | not adopted |
| ❌ | OB | Out-of-pitch bridge (INVICTA's OFF_BRIDGE): dead ball starts where the seen ball was last outside the pitch, up to 7 s before | stoppage onset 45 frames late on the public window | open-data CV −0.002; real: public spellF1 unchanged, macroF1 0.8513 vs 0.8489, dead-ball recall 0.655 | changes only 4 stretches (5 s) of the match against v7+TTA: negligible. Not adopted |
| ❌ | MIX | v5 + v7 members (6) with test-time adaptation | v5 better on turnovers, v7 on dead ball | public 0.589 / 0.824 vs 0.629 / 0.849, dead-ball precision 0.847 | not adopted |
| ⏸ | A75 | Milder adaptation (`tta_alpha` 0.75) | live restarts called dead (review) | public 0.612 / 0.836, 17 of 17 possessions, dead-ball recall 0.586, precision 1.00 | reel review pending |
| ❌ | ENV | Smart Tagging restart envelope: +1.5 / +3 to log P(D) from 5–10 s before each SET PIECES tag opens until 2–3.5 s after (tags are a fixed 35 s long, so only the start informs) | dead ball at restarts | changes 0.6–1.2% of graded frames; public spellF1 unchanged (0.594), macroF1 up to 0.823 (variant d), dead-ball recall up to 0.554 (precision 0.977); the set-piece table improves by construction, so it cannot judge this | on top of v7+TTA it adds nothing (macroF1 0.8497 vs 0.8489, same possessions): adaptation already covers it. Not adopted |
| ❌ | MIND | Minimum possession length 19 / 25 frames on the real match (v5 ensemble, `mbr.decode_min_duration`) | short false possessions in the review | public spellF1 0.583 vs 0.594 (one boundary worse), macroF1 0.817; 309 vs 313 possessions | not adopted (matches the open-data +0.001) |
| ❌ | CH | 60 s training windows (`--chunk 1500`; inference reads 120 s) | train/infer context mismatch | CV seed 0 +0.0013, seed 1 −0.0064. Real match (v5 decoder): public spellF1 0.635 (17 of 17 possessions) but macroF1 0.794 and dead-ball recall 0.460 (v5 0.819 / 0.534); with test-time adaptation 0.582 / 0.801 | not adopted: CV flat and the real match mixed (spellF1 up on one lucky count, tie-break and dead ball down) |
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
| ✅ | Q1s2 | segment-balanced loss 0.5 (seed 2) | 0.7598 | 0.7535 | 0.7745 | (12.0, 8.0, 2.0, 0.5) | no gain (-0.0001) | 2026-10-23 07:47 |
| ✅ | K0 | collective kinematics features, CV seed 0 (vs v5 seed 0) | 0.7554 | 0.7461 | 0.7770 | (4.0, 4.0, 4.0, 0.5) | no gain (+0.0042) | 2026-10-23 21:03 |
| ✅ | K1 | collective kinematics features, CV seed 1 (vs v5 seed 1) | 0.7448 | 0.7310 | 0.7771 | (4.0, 4.0, 4.0, 0.0) | no gain (-0.0123) | 2026-10-23 21:42 |
| ✅ | Ktrain0 | kinematics: train member 0 |  |  |  |  |  | 2026-10-23 21:48 |
| ✅ | Ktrain1 | kinematics: train member 1 |  |  |  |  |  | 2026-10-23 21:54 |
| ✅ | Ktrain2 | kinematics: train member 2 |  |  |  |  |  | 2026-10-23 22:04 |
| ✅ | Kpredict | kinematics: build tempolens_v6_kin.csv |  |  |  |  |  | 2026-10-23 22:09 |
| ✅ | Kvalidate | kinematics: official validator |  |  |  |  |  | 2026-10-23 22:09 |
| ✅ | Kpublic | kinematics: public window + dead-ball recall vs v5 |  |  |  |  |  | 2026-10-23 22:09 |
| ✅ | Ksetpiece | kinematics: dead-ball share around set-piece tags |  |  |  |  |  | 2026-10-23 22:10 |
| ✅ | R1 | kickoff fix: regenerate v5 from a clean predict run |  |  |  |  |  | 2026-10-24 01:16 |
| ✅ | R2 | kickoff fix: regenerated file byte-identical to tempolens_v5_kickoff.csv |  |  |  |  |  | 2026-10-24 01:16 |
| ✅ | DW0 | dead-ball loss weight 2, CV seed 0 (vs v5 seed 0) | 0.7502 | 0.7419 | 0.7693 | (2.0, 2.0, 2.0, 0.5) | no gain (-0.0010) | 2026-10-24 01:39 |
| ✅ | KP0 | keeper-hold features (keeper distance, ball in box), CV seed 0 (vs v5 seed 0) | 0.7437 | 0.7337 | 0.7668 | (4.0, 8.0, 8.0, 1.0) | no gain (-0.0075) | 2026-10-24 02:04 |
| ✅ | BF0 | touch-recency + scramble features, CV seed 0 (vs v5 seed 0) | 0.7489 | 0.7398 | 0.7701 | (4.0, 8.0, 8.0, 1.0) | no gain (-0.0023) | 2026-10-24 02:28 |
| ✅ | DW1 | dead-ball loss weight 2, CV seed 1 (vs v5 seed 1) | 0.7556 | 0.7419 | 0.7875 | (6.0, 2.0, 8.0, 1.0) | no gain (-0.0015) | 2026-10-24 02:51 |
| ✅ | KP1 | keeper-hold features (keeper distance, ball in box), CV seed 1 (vs v5 seed 1) | 0.7516 | 0.7423 | 0.7733 | (2.0, 4.0, 4.0, 0.0) | no gain (-0.0055) | 2026-10-24 03:15 |
| ✅ | BF1 | touch-recency + scramble features, CV seed 1 (vs v5 seed 1) | 0.7517 | 0.7418 | 0.7748 | (2.0, 8.0, 8.0, 0.0) | no gain (-0.0054) | 2026-10-24 03:38 |
| ✅ | BCALLpred | broadcast ball fill (all): predict |  |  |  |  |  | 2026-10-24 16:52 |
| ✅ | BCALLval | broadcast ball fill (all): official validator |  |  |  |  |  | 2026-10-24 16:52 |
| ✅ | BCALLpub | broadcast ball fill (all): public window + dead-ball recall vs v5 |  |  |  |  |  | 2026-10-24 16:52 |
| ✅ | BCALLset | broadcast ball fill (all): dead-ball share around set pieces |  |  |  |  |  | 2026-10-24 16:52 |
| ✅ | BCLIVEpred | broadcast ball fill (alive): predict |  |  |  |  |  | 2026-10-24 16:56 |
| ✅ | BCLIVEval | broadcast ball fill (alive): official validator |  |  |  |  |  | 2026-10-24 16:56 |
| ✅ | BCLIVEpub | broadcast ball fill (alive): public window + dead-ball recall vs v5 |  |  |  |  |  | 2026-10-24 16:56 |
| ✅ | BCLIVEset | broadcast ball fill (alive): dead-ball share around set pieces |  |  |  |  |  | 2026-10-24 16:56 |
| ✅ | RN2s0 | randomised noise, 2 draws/match, CV seed 0 (vs v5) | 0.7529 | 0.7453 | 0.7705 | (2.0, 8.0, 8.0, 0.0) | no gain (+0.0017) | 2026-10-25 05:16 |
| ✅ | RN4s0 | randomised noise, 4 draws/match, CV seed 0 (vs v5) | 0.7576 | 0.7467 | 0.7830 | (4.0, 2.0, 4.0, 0.5) | candidate (+0.0064) | 2026-10-25 05:50 |
| ✅ | RN4Hs0 | randomised noise, 4 draws, width 160, 12 epochs, CV seed 0 (vs v5) | 0.7592 | 0.7526 | 0.7746 | (6.0, 2.0, 8.0, 0.0) | candidate (+0.0080) | 2026-10-25 08:46 |
| ✅ | RN2s1 | randomised noise, 2 draws/match, CV seed 1 (vs v5) | 0.7479 | 0.7408 | 0.7646 | (4.0, 8.0, 8.0, 0.5) | no gain (-0.0092) | 2026-10-25 09:08 |
| ✅ | RN4s1 | randomised noise, 4 draws/match, CV seed 1 (vs v5) | 0.7677 | 0.7538 | 0.8003 | (6.0, 8.0, 8.0, 1.0) | candidate (+0.0106) | 2026-10-25 09:45 |
| ✅ | RN4Hs1 | randomised noise, 4 draws, width 160, 12 epochs, CV seed 1 (vs v5) | 0.7591 | 0.7526 | 0.7744 | (6.0, 8.0, 8.0, 1.0) | no gain (+0.0020) | 2026-10-25 12:45 |
| ✅ | RNtrain0 | randomised noise 4 draws: train member 0 (models/exp/, not committed) |  |  |  |  |  | 2026-10-25 12:52 |
| ✅ | RNtrain1 | randomised noise 4 draws: train member 1 (models/exp/, not committed) |  |  |  |  |  | 2026-10-25 12:58 |
| ✅ | RNtrain2 | randomised noise 4 draws: train member 2 (models/exp/, not committed) |  |  |  |  |  | 2026-10-25 13:11 |
| ✅ | RNpred | randomised noise 4 draws: build tempolens_v7_rnd4.csv |  |  |  |  |  | 2026-10-25 13:13 |
| ✅ | RNval | randomised noise: official validator |  |  |  |  |  | 2026-10-25 13:13 |
| ✅ | RNpub | randomised noise: public window + dead-ball recall vs v5 |  |  |  |  |  | 2026-10-25 13:13 |
| ✅ | RNset | randomised noise: dead-ball share around set pieces |  |  |  |  |  | 2026-10-25 13:13 |
| ✅ | V5s2 | v5 recipe: CV member 2 (completes the v5 ensemble) | 0.7503 | 0.7394 | 0.7756 | (2.0, 8.0, 8.0, 0.0) | measured | 2026-10-25 16:10 |
| ✅ | RN4s2 | randomised noise 4 draws: CV member 2 (completes the v7 ensemble) | 0.7633 | 0.7529 | 0.7877 | (6.0, 2.0, 4.0, 0.0) | measured | 2026-10-25 16:52 |
| ✅ | V5ens | v5 recipe: 3-member ensemble CV (decoder costs tuned) | 0.7636 | 0.7526 | 0.7892 | (2.0, 2.0, 8.0, 0.0) | measured | 2026-10-25 16:54 |
| ✅ | RN4ens | randomised noise 4 draws: 3-member ensemble CV (vs v5 ensemble) | 0.7716 | 0.7597 | 0.7994 | (2.0, 2.0, 2.0, 0.5) | candidate (+0.0080) | 2026-10-25 16:57 |
| ✅ | RN4dec | v7: decoder settings from RN4ens's best costs |  |  |  |  |  | 2026-10-25 16:57 |
| ✅ | RN4pred | v7: build tempolens_v7_rnd4_tuned.csv with its own decoder costs |  |  |  |  |  | 2026-10-25 17:01 |
| ✅ | RN4val | v7 tuned: official validator |  |  |  |  |  | 2026-10-25 17:01 |
| ✅ | RN4pub | v7 tuned: public window + dead-ball recall vs v5 |  |  |  |  |  | 2026-10-25 17:01 |
| ✅ | RN4set | v7 tuned: dead-ball share around set pieces |  |  |  |  |  | 2026-10-25 17:01 |
| ✅ | V5set | v5 (submitted): dead-ball share around set pieces, for comparison |  |  |  |  |  | 2026-10-25 17:01 |
| ✅ | RF4s0 | randomised noise 4 draws + tracker faults, CV seed 0 (vs RN4s0) | 0.7524 | 0.7406 | 0.7801 | (4.0, 2.0, 2.0, 0.0) | no gain (-0.0052) | 2026-10-25 18:47 |
| ✅ | RN6s0 | randomised noise 6 draws, CV seed 0 (vs RN4s0) | 0.7538 | 0.7415 | 0.7826 | (4.0, 8.0, 8.0, 0.5) | no gain (-0.0038) | 2026-10-25 19:48 |
| ❌ | RF4s1 | randomised noise 4 draws + tracker faults, CV seed 1 (vs RN4s1) |  |  |  |  |  | 2026-10-25 19:53 |
| ? | RN6s1 | randomised noise 6 draws, CV seed 1 (vs RN4s1) |  |  |  |  |  |  |
| ✅ | TTAcv | test-time re-normalisation on open data, paired (alpha 0 / 0.5 / 1) |  |  |  |  |  | 2026-10-25 20:56 |
| ✅ | TTAv7a1 | TTA alpha 1 on the real match, v7 models |  |  |  |  |  | 2026-10-25 21:00 |
| ✅ | TTAv7a05 | TTA alpha 0.5 on the real match, v7 models |  |  |  |  |  | 2026-10-25 21:04 |
| ✅ | TTAv5a1 | TTA alpha 1 on the real match, v5 models |  |  |  |  |  | 2026-10-25 21:09 |
| ✅ | TTApub | TTA: public window + dead-ball recall |  |  |  |  |  | 2026-10-25 21:09 |
| ✅ | TTAset | TTA alpha 1 (v7): dead-ball share around set pieces |  |  |  |  |  | 2026-10-25 21:09 |
| ✅ | ENV | Smart Tagging restart envelope on v5: 4 variants |  |  |  |  |  | 2026-10-25 21:13 |
| ✅ | ENVpub | restart envelope: public window + dead-ball recall vs v5 |  |  |  |  |  | 2026-10-25 21:13 |
| ✅ | ENVseta | restart envelope a: dead-ball share around set pieces |  |  |  |  |  | 2026-10-25 21:13 |
| ✅ | ENVsetb | restart envelope b: dead-ball share around set pieces |  |  |  |  |  | 2026-10-25 21:13 |
| ✅ | ENVsetc | restart envelope c: dead-ball share around set pieces |  |  |  |  |  | 2026-10-25 21:13 |
| ✅ | ENVsetd | restart envelope d: dead-ball share around set pieces |  |  |  |  |  | 2026-10-25 21:13 |
| ✅ | MIND | minimum possession length 19 / 25 frames on the real match (v5 ensemble) |  |  |  |  |  | 2026-10-25 21:13 |
| ✅ | MINDpub | minimum length: public window + dead-ball recall vs v5 |  |  |  |  |  | 2026-10-25 21:13 |
| ✅ | CH0 | 60 s training windows (chunk 1500), CV seed 0 (vs v5) | 0.7525 | 0.7442 | 0.7718 | (4.0, 8.0, 8.0, 0.0) | no gain (+0.0013) | 2026-10-25 21:51 |
| ✅ | CH1 | 60 s training windows (chunk 1500), CV seed 1 (vs v5) | 0.7507 | 0.7454 | 0.7633 | (2.0, 8.0, 8.0, 0.0) | no gain (-0.0064) | 2026-10-25 22:21 |
| ✅ | CHtrain0 | 60 s windows: train member 0 (models/exp/) |  |  |  |  |  | 2026-10-25 22:25 |
| ✅ | CHtrain1 | 60 s windows: train member 1 (models/exp/) |  |  |  |  |  | 2026-10-25 22:30 |
| ✅ | CHtrain2 | 60 s windows: train member 2 (models/exp/) |  |  |  |  |  | 2026-10-25 22:33 |
| ✅ | CHpred | 60 s windows: build tempolens_v8_c1500.csv (v5 decoder) |  |  |  |  |  | 2026-10-25 22:37 |
| ✅ | CHtta | 60 s windows + test-time re-normalisation (alpha 1) |  |  |  |  |  | 2026-10-25 22:41 |
| ✅ | CHpub | 60 s windows: public window + dead-ball recall |  |  |  |  |  | 2026-10-25 22:41 |
| ✅ | CHset | 60 s windows (no TTA): dead-ball share around set pieces |  |  |  |  |  | 2026-10-25 22:41 |
| ✅ | CHsettta | 60 s windows + TTA: dead-ball share around set pieces |  |  |  |  |  | 2026-10-25 22:41 |
| ✅ | TTRpred | v7+TTA through predict.py (decoder JSON tta_alpha 1) |  |  |  |  |  | 2026-10-26 01:10 |
| ✅ | TTRcmp | v7+TTA: predict.py output byte-identical to the scratch-built tempolens_v7_tta1.csv |  |  |  |  |  | 2026-10-26 01:10 |
| ✅ | ETTA | v7+TTA plus restart envelope (b, d) |  |  |  |  |  | 2026-10-26 01:14 |
| ✅ | ETTApub | v7+TTA +/- envelope: public window + dead-ball recall |  |  |  |  |  | 2026-10-26 01:14 |
| ✅ | MIX | mixed ensemble v5 + v7 members (6), TTA alpha 1 |  |  |  |  |  | 2026-10-29 19:54 |
| ✅ | A75 | v7 members, milder adaptation alpha 0.75 |  |  |  |  |  | 2026-10-29 19:59 |
| ✅ | MApub | mixed ensemble and alpha 0.75: public window + dead-ball recall vs v7+TTA |  |  |  |  |  | 2026-10-29 19:59 |
| ✅ | MIXset | mixed ensemble: dead-ball share around set pieces |  |  |  |  |  | 2026-10-29 19:59 |
| ✅ | A75set | alpha 0.75: dead-ball share around set pieces |  |  |  |  |  | 2026-10-29 19:59 |
| ✅ | REELmix | review reel v7+TTA vs mix |  |  |  |  |  | 2026-10-29 20:06 |
| ✅ | TOPmix | top-20 review sheet v7+TTA vs mix |  |  |  |  |  | 2026-10-29 20:06 |
| ✅ | REELa075 | review reel v7+TTA vs a075 |  |  |  |  |  | 2026-10-29 20:12 |
| ✅ | TOPa075 | top-20 review sheet v7+TTA vs a075 |  |  |  |  |  | 2026-10-29 20:12 |
| ✅ | EATA | EATA head adaptation (10 / 30 steps) and out-of-pitch bridge on the real match |  |  |  |  |  | 2026-10-29 20:57 |
| ✅ | EApub | EATA and bridge: public window + dead-ball recall vs v7+TTA |  |  |  |  |  | 2026-10-29 20:57 |
| ✅ | EAset | EATA 10: dead-ball share around set pieces |  |  |  |  |  | 2026-10-29 20:57 |
| ✅ | REELeata10 | review reel v7+TTA vs eata10 |  |  |  |  |  | 2026-10-29 21:02 |
| ✅ | TOPeata10 | top-20 sheet v7+TTA vs eata10 |  |  |  |  |  | 2026-10-29 21:02 |
| ✅ | EAset30 | EATA 30: dead-ball share around set pieces |  |  |  |  |  | 2026-10-29 21:02 |
| ✅ | REELeata30 | review reel v7+TTA vs eata30 |  |  |  |  |  | 2026-10-29 21:08 |
| ✅ | TOPeata30 | top-20 sheet v7+TTA vs eata30 |  |  |  |  |  | 2026-10-29 21:08 |
| ✅ | REELbridge | review reel v7+TTA vs bridge |  |  |  |  |  | 2026-10-29 21:12 |
| ✅ | TOPbridge | top-20 sheet v7+TTA vs bridge |  |  |  |  |  | 2026-10-29 21:12 |
| ✅ | TCNcv | TCN residual corrector on the frozen v7 ensemble: 5-fold CV on open data |  |  |  |  |  | 2026-10-29 21:25 |
| ✅ | TCNreal | TCN corrector trained on all open matches, applied to v7+TTA on the real match |  |  |  |  |  | 2026-10-29 21:32 |
| ✅ | TCNpub | TCN corrector: public window + dead-ball recall vs v7+TTA |  |  |  |  |  | 2026-10-29 21:32 |
| ✅ | REELtcn | review reel v7+TTA vs TCN corrector |  |  |  |  |  | 2026-10-29 21:44 |
| ✅ | TOPtcn | top-20 sheet v7+TTA vs TCN corrector |  |  |  |  |  | 2026-10-29 21:44 |
<!-- autoqueue:end -->
