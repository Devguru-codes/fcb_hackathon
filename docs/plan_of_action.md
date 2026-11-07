# TempoLens — plan of action and experiment tracker

The single list of everything tried, running and planned, with the measured result. Updated
whenever an experiment finishes. Numbers are aggregates only (no frame-level data: the repository
becomes public at submission).

**Legend:** ✅ works — adopted in the pipeline · ❌ tested — did not help, not adopted ·
⏳ running · ⬜ to do · 🚫 blocked (reason given) · 👤 needs a person (team)

## Organisers' rules clarification (official mail, 2026-11-03)

This mail takes precedence over the documentation, earlier emails and individual answers.

- **Not allowed:**
  - checking the output against the video;
  - hand labels, human-produced data or any other data created manually on the provided game;
  - uploading the provided data to AI services;
  - training, fitting or calibrating the algorithm on the ground-truth possession data or on the
    provided video.
- **Allowed:**
  - automatic checks using only the provided data;
  - inspecting the data and the video to understand what counts as possession;
  - the video, the automatic tracking, the Smart Tagging or other generated data as inputs, as long
    as the algorithm was not trained on the ground truth or the video;
  - files of intermediate steps;
  - any dataset we have the rights to use.
- **Training and tuning:** use only external data. Parameters that need tuning are tuned on
  external data, never on the ground-truth possession data or the video.

**Consequences for this repository:**
- The earlier submission v7+TTA+EATA+gate (`tempolens_final.csv`) is withdrawn. Its decoder costs,
  `tta_alpha`, `eata_steps` and gate were chosen with the public window's ground truth and with
  video reviews of our output.
- The submission is rebuilt with open-data choices only (section *Current submission*).
- Every human-review, public-window and reel row below records exploration done before the
  clarification. None of it selects anything in the submitted configuration.
- Blocked for good: a video ball detector trained on the match video (WASB), and hand-labelled
  windows (DEF).

## Decision rules

1. **Model selection = cross-validation (CV)** on 10 open matches (7 DFL, 3 Metrica) under the
   simulated competition tracking, scored with the official `possession_metrics.py`. Compare only
   runs on the **same simulator version** (v1 and v2 numbers are not comparable).
2. **Noise floor (measured, Q10):** the same model retrained with seeds 0/1/2 scores 0.7543 /
   0.7556 / 0.7599 (all; sd 0.0029, range 0.0056) and 0.7456 / 0.7439 / 0.7524 (DFL; sd 0.0045).
   A single-seed gain must exceed ~0.006 overall (2 sd); smaller gains need ≥ 2 seeds agreeing.
3. **The provided match decides nothing (organisers' rules, 2026-11-03).** No setting is chosen,
   tuned or vetoed with its ground truth or its video. Only automatic checks on its tracking and
   Smart Tagging that use no ground truth (format validation, label shares) may be run.
4. **Reproduction gate before any upload:** a clean checkout of the commit must rebuild the
   submission byte for byte, and `validate_submission.py` must pass.

## Current submission

| | |
|---|---|
| version | **final ENS5 + TTA + EATA** (2026-11-07): features v4, simulator v2, 5-GRU ensemble trained on the 10 open matches with domain randomisation (`train --draws 4`); test-time adaptation `tta_alpha` 0.5 and EATA 10 steps with decoder costs (4, 8, 8, 1.0), gate off (ADseeds, team's choice between the two settings that passed); Smart Tagging prior on (`auto`, TPcv5) |
| evidence | open-data CV ensemble 0.7760 (DFL 0.7650, Metrica 0.8017) vs 3 members 0.7716 and 0.7636 without domain randomisation; tag prior +0.0081 to −0.0006 (TPcv5); adaptation +0.0027 on the mean of sim / shift / SkillCorner (sim −0.0029, shift −0.0013, sc +0.0122), positive for the ensemble and for each of the 5 members, +0.0022 to +0.0063 (ADseeds, rule fixed before the run after the +0.005 bar was dropped). Two settings passed the pre-registered every-seed rule and tied within 0.0001 on the ensemble (TTA 0.5: 0.6476; TTA 0.5 + EATA 10: 0.6475). The rule's tie-break would pick TTA alone. The team chose TTA 0.5 + EATA 10 after seeing the results, because its smallest per-seed gain is larger (+0.0022 vs +0.0001) and its gain on the real-tracking SkillCorner condition is larger (+0.0122 vs +0.0108). The public window was not used for this choice. Not scored on the public window except the one ENS5 sanity check that preceded the rule change and decided nothing |
| file | `derived/adseeds/eata10_run1.csv` (outside git), sha256 **`SHA_NEW`**, validated; identical over two runs |
| reproduction | RERUN_LINE |
| rule's own pick | TTA alone (`tta_alpha` 0.5, EATA 0, costs (4, 8, 8, 1.0)): `derived/adseeds/adseeds_run1.csv`, sha256 `554fcb55fb5d2f024f4b8e84bdf999506c208106fd6a07fa75b596176bb98b0a`, validated, reproduced from a clean checkout of `2b468e8` |
| superseded | ENS5 without adaptation `tempolens_final_ens5.csv` (`553eb29c…`, reproduced from `52b9fad`); 3-member `tempolens_final_clean.csv` (`c89c7fee…`, reproduced from `df62475` and `f3efbaf`); `tempolens_clean.csv` (`704cfd62…`, same without the tag prior) |
| withdrawn | v7+TTA+EATA+gate `tempolens_final.csv` (`9f7dab1a…`): settings chosen on the provided match (see the rules clarification above) |

## A. Pipeline milestones

| | item | result |
|---|---|---|
| ✅ | Learned pipeline: open data + ATD simulator + identity-free features + bi-GRU + Viterbi | replaces the hand-tuned Codex decoder (removed) |
| ✅ | ATD simulator calibrated to the official baseline's published full-match scores | reproduces their ranking order |
| ✅ | 3-member ensemble (seeds × noise draws) | +0.011 DFL over one GRU (simulator v1) |
| ✅ | Self-calibrated Smart Tagging prior, **on** in the final configuration (TPcv) | never hurts on open data with synthetic tags (+0.011 perfect, −0.0001 random); it was first switched off on the public window, a choice the 2026-11-03 rules forbid |
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
| ⏸ | BCF | Broadcast ball fill (one offset per half, 9.7% of gaps) on top of the final (v7+TTA+EATA) | ball missing on 41% of graded frames | public window unchanged (0.6471); fill-all macroF1 0.8731 vs 0.8725, dead-ball precision 0.894 vs 0.876, but **310 possessions vs 301** (more splits), 28 disagreements (81 s); fill-alive: 9 disagreements (42 s), otherwise identical | negligible on every measure; adopt only if the user's reel review clearly favours it |
| ❌ | FRA | Flank-aware run absorption (ra189zor): a H/A run shorter than 12 / 25 frames with the other team on both sides joins them | split penalty | open-data CV +0.0013 (12) / −0.0078 (25); on the real final it changes **0 frames** at 12 and 22 frames at 25 (public unchanged): the decoder already removes such runs | not adopted |
| ✅ | ORC | Oracle diagnostic on open-data CV (v7 ensemble): true dead/live with our team calls vs our dead/live with true team calls | where is the headroom? | ours 0.7682; **perfect dead/live 0.7830 (+0.015); perfect team calls 0.9635 (+0.195)**. We also predict 3,464 possessions where there are 4,069 (−15%): short possessions are missed. Opposite of ra189zor's finding for their system. Caveat: simulated data; on the real match dead ball still mattered (TTA/EATA) | headroom is in who-has-the-ball within live play and short possessions, not in dead ball |
| ❌ | SNAP | Forward contact snapping (Suleyman): a H↔A change with the ball seen and the new team out of reach moves forward up to 40 frames to first contact (< 1.65 m), or the spell is deleted | false turnovers in flight | open-data CV 0.7632 vs 0.7682; real final: public 0.588 vs 0.647 | not adopted |
| ✅ | GATE | In-flight / loose-ball attenuation (Suleyman): H-vs-A log-odds × 0.2 when the seen ball moves > 9.5 m/s, × 0.3 when nobody is within 1.35 m; P(alive) kept | false turnovers in flight | open-data CV **0.7437 vs 0.7682**; real final: public spellF1 0.647 unchanged, macroF1 0.881 vs 0.8725, 291 vs 301 possessions, 509 frames changed | **adopted**: human review 7 of 8 for the gate; `flight_gate` / `loose_gate` in `decoder.json`, reproduced byte for byte |
| ❌ | TCH | **Touch head**: auxiliary per-frame home/away touch targets from the open matches' events, DFL times snapped to the physical contact (median 0.2–0.46 m), trained beside possession (`--touch`) | oracle: perfect team calls → 0.964 | CV vs the v7 recipe: weight 0.5 seed 0 +0.0053, seed 1 −0.0060 (noise); weight 1.0 seed 0 +0.0079. Real match (0.5, with TTA + EATA + gate): public 0.632 / 0.875 vs 0.647 / 0.881, 14 possessions vs 17 (merges), dead-ball recall 0.709 vs 0.756; 45 disagreements (109 s, `reel_final_touch.mp4`) | not adopted |
| ❌ | ST | **Self-training on the real match** (`temporal.self_train`, last GRU layer + head, CPU float64): A — pseudo-labels on 75% of graded frames (confident interiors of spells ≥ 3 s); B — A plus 3,625 frames of physical anchors that disagree with the decoder | adapting to the real match is what worked | A: public 0.647 / 0.877 vs 0.647 / 0.881, 286 possessions, 7 disagreements (15 s); B: 0.629 / 0.853, 294 possessions, 19 disagreements (53 s). ~5 min per variant | not adopted (A neutral, B worse) |
| ⬜ | WASB | **Strong video ball detector** (WASB-class, Tarashima et al. BMVC 2023) to fill the 41% of frames with no ball | our TrackNet-style detector: 57% within 15 px on the real video | prerequisites: (1) organisers confirm that weights trained on the licensed video may be committed, or another way to make the submission reproducible; (2) GPU inference over the gaps must be deterministic and fit the organisers' re-run; then: train on ATD-projected ball crops (`tempolens.vision`), gate fills on precision (≥ 95% within 1.5 m), check on the public window + a reel | blocked on (1) — added to organiser questions |
| ⬜ | DEF | **Training labels in the competition's definition of possession** (Metrica corrected tracking: control, not touch) | open labels only approximate it (DFL owner flag, Metrica events) — the structural ceiling | only legitimate route: honest hand-labelled windows by eye (small amounts may *train or validate*, RULES.md), used to fine-tune the output layer or to calibrate the open labels (flicker folding, owner hand-over timing); prerequisite: the user labels 2–3 windows of 5 min by the control rule | blocked on hand labels |
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
| ⏸ | H1b | SkillCorner-trained ensemble checked on the real match only (`train --extra-train skillcorner`, 3 members, final decoder) | H1 tie/loss may be a label-definition artefact | real match, vs final: **public spellF1 0.658, macroF1 0.911, 14 of 17 possessions, dead-ball recall 0.944** (precision 0.910); but **189 vs 291 possessions** in the match (mean 16.9 s): it merges across turnovers, as the touch head did. 153 disagreements of 1 s+ (462 s) | reel review pending (`reel_final_skillcorner.mp4`); risky: the oracle says we already find 15% too few possessions |
| ❌ | CMB | Final's team calls + H1b's dead/live (A: H1b's team where final says D and H1b live; B: final's last team carried forward) | H1b's dead-ball recall 0.944 without its merged possessions | 6,973 frames changed (5,632 to D, 1,341 to live); public window, A and B identical: **spellF1 0.618**, macroF1 0.915, 16 of 17 possessions, dead-ball recall 0.944; 257 / 275 possessions in the match. The main metric falls below both parents: H1b's dead stretches cut final's possessions into pieces | not adopted (spellF1 is the ranking) |
| ✅ | TPcv | **Smart Tagging prior on open data, current recipe** (`tagstudy --draws 4 --seeds 3`: v7 ensemble's held-out outputs, clean costs (2, 2, 2, 0.5), 3 synthetic tag draws per match; tag shape from the study's written design, not from the match) | it was switched off on the public window, a ground-truth decision the 2026-11-03 rules forbid | no prior 0.7716; tag error 0% **0.7825 (+0.011)**, 10% 0.7767 (+0.005), 20% 0.7748 (+0.003), 35% 0.7736 (+0.002), random 0.7715 (−0.0001): it never hurts and its weights fall to ~0.05 for random tags. Self-calibrated on the match without labels (agreement of the tags with confident frames of the ensemble) | **adopt** `tag_prior` "auto" in the final configuration |
| ❌ | MIXcv | v5 + v7 members (6) on open-data CV, decoder costs searched (81 settings) for each | the earlier mix test was judged on the public window | v7 0.7716 (2, 2, 2, 0.5); mix 0.7725 (2, 8, 8, 0.0): **+0.0009**, DFL +0.004, Metrica −0.006; rule (fixed before the run) needed +0.006 | not adopted: v7 alone |
| ❌ | FV3cv | **Feature set on open data only**: the v7 recipe (3 members, `--draws 4`) with the five v4 restart-spot features masked out of the input (`--drop restart`, = the v3 feature set) against the submitted v4 ensemble (CV 0.7716) | v4 was kept over v3 before the 2026-11-03 rules on public-window numbers, while CV had v3 ahead (0.764 vs 0.7591, v5 recipe) | **Rule, fixed before the run (2026-11-05):** 5-fold CV, simulator v2, 3-member ensemble, each variant at its own best decoder costs over the standard grid; the higher mean spellF1 over the 10 matches wins, an exact tie keeps v4. Per-member (seed) paired scores and the DFL/Metrica split are reported, not deciding. If the masked set wins: retrain the 3 final members on all 10 matches with `--drop restart`, take its CV-argmax costs, re-run the tag study and the adaptation study on the new members with the same rules, then regenerate, validate and reproduce | **Result:** masked 0.7674 (DFL 0.7584, Metrica 0.7886) at its best costs (1.5, 8, 8, 0) vs v4 **0.7716** (DFL 0.7597, Metrica 0.7994) at (2, 2, 2, 0.5): **−0.0042**. Per member (seed), each at v4's costs: 0.7553 / 0.7585 / 0.7563 vs 0.7569 / 0.7629 / 0.7591, v4 ahead on all three | not adopted: **v4 kept on open-data evidence**; submission unchanged |
| ✅ | ENS5 | **5-member ensemble**: v7 members 3 and 4 (fresh seeds and noise draws) added to the 3 (`--model ensemble --draws 4 --members 5`) | averaging more members is the one change that cannot overfit the simulator; 1→3 members gave +0.011 (sim v1) | **Rule, fixed before the run (2026-11-06), the same for all three:** 5-fold CV, simulator v2, v7 recipe (`--draws 4`), each candidate at its own best decoder costs over the standard grid. A candidate is adopted only if (a) its ensemble beats v7's 0.7716 by **≥ +0.004** and (b) for the 3-member candidates, at least 2 of 3 members beat the v7 member with the same index at v7's costs (2, 2, 2, 0.5). If several pass, only the highest is adopted. Adoption means: train the final members, take the CV-argmax costs, re-run the tag study and the adaptation study with their rules, then regenerate, validate and reproduce. **Result:** 0.7760 (DFL 0.7650, Metrica 0.8017) at (2, 8, 8, 0.5), **+0.0044**; the surface is flat (0.7754–0.7756 at the next four costs, including (2, 2, 2, 0) at +0.0038). Members 0–2 are the v7 members themselves, so the gain is the two added members alone | **adopt** (passes (a); (b) is for the 3-member candidates). The autoqueue's own label says "no gain" because its built-in bar is +0.005; the rule fixed before the run is +0.004 |
| ❌ | H128 | **Wider GRU (128 units) under domain randomisation** (`--hidden 128`, 3 members) | 4 randomised draws per match give twice the training data that tied 96 vs 128 units under simulator v1 | same rule. 0.7704 (DFL 0.7616, Metrica 0.7911) at (1.5, 8, 8, 0), −0.0012 | fails (a); 96 units kept |
| ❌ | EP12 | **12 training epochs instead of 8** under domain randomisation (`--epochs 12`, 3 members) | the epoch count was set for 2 fixed draws; randomised draws are harder to fit | same rule. 0.7723 (DFL 0.7630, Metrica 0.7942) at (2, 2, 8, 0.5), +0.0007 | fails (a); 8 epochs kept |
| ✅ | ENS5fin | **Adopt ENS5**: final members 3 and 4 (`train --member 3/4 --draws 4 --out models/tempolens_rnd4_m3/m4.pt`, all 10 open matches); `decoder.json` gets 5 members and the ENS5 CV-argmax costs (2, 8, 8, 0.5) | ENS5 passed its rule | **Rules, fixed before these runs (2026-11-06):** (1) the costs are the ENS5 CV argmax, whatever the studies below find. (2) **TPcv5** (`tagstudy --draws 4 --seeds 3 --members 5`, costs (2, 8, 8, 0.5)): keep `tag_prior` "auto" unless random tags (50% error) lose more than 0.003 against no prior or perfect tags gain less than 0.003; otherwise switch it off. (3) **ADcv5** (`adaptcv --members 5`, costs {(2, 2, 2, 0.5), (2, 8, 8, 0.5), (4, 8, 8, 1.0)}, fold members 3 and 4 trained like 0–2): the ADcv rule unchanged, i.e. adaptation only with ≥ +0.005 mean over the three conditions at the same costs and no condition below −0.005. (4) Then regenerate, validate, and reproduce byte for byte from a clean checkout | ~1 h GPU + ~5 h |
| ✅ | TPcv5 | Smart Tagging prior on the 5-member ensemble (`tagstudy --draws 4 --seeds 3 --members 5`, costs (2, 8, 8, 0.5)) | ENS5fin rule (2) | no prior 0.7760; tag error 0% **0.7841 (+0.0081)**, 10% 0.7790, 20% 0.7781, 35% 0.7770, random 0.7754 (**−0.0006**); weights fall to ~0.05 for random tags **Re-run at the submitted costs (4, 8, 8, 1.0), 2026-11-07** (same command, decoder costs; cached non-adapted CV outputs, so TTA/EATA are not in it): no prior **0.7727**; tag error 0% 0.7820 (+0.0093), 10% 0.7772, 20% 0.7763, 35% 0.7733, 50% 0.7730 (+0.0003). Keep rule met: `auto` stays | **keep** `tag_prior` "auto" (perfect ≥ +0.003, random no worse than −0.003) |
| ❌ | ADcv5 | Adaptation study on the 5-member ensemble (`adaptcv --members 5`; fold members 3 and 4 trained like 0–2; costs {(2, 2, 2, 0.5), (2, 8, 8, 0.5), (4, 8, 8, 1.0)}) | ENS5fin rule (3) | none at (2, 8, 8, 0.5): sim 0.7760, shift 0.7721, sc 0.3878, mean **0.6453** (also the best plain costs, matching the CV argmax). Best: `tta_alpha` 0.5 at (4, 8, 8, 1.0) 0.6476, **+0.0028** over none at the same costs, worst condition −0.0018; `tta_alpha` 0.5 helps real tracking (sc +0.011) but costs sim/shift; EATA and the gate never help | not adopted (under +0.005): no adaptation, as with 3 members. 10.4 h |
| ✅ | ADseeds | **Adaptation per seed on open data** (`python -m tempolens.adaptseeds --members 5`): every adaptation setting, `tta_alpha` {0, 0.5, 1} × `eata_steps` {0, 10, 30, 60} × gate {off, on} minus no adaptation, at costs (2, 8, 8, 0.5) and (4, 8, 8, 1.0), scored for the 5-member ensemble and for each of the 5 seeds alone (fold member k on sim / shift, final member k on sc); same conditions and fold models as ADcv5 | ADcv5's best (`tta_alpha` 0.5, +0.0028) failed only the +0.005 bar. **The team dropped that bar on 2026-11-07, after a one-off public-window sanity check of ENS5 (spellF1 0.4750, macroF1 0.8204); that check decides nothing here** | **Rule, fixed before the run (2026-11-07):** score = mean spellF1 over sim / shift / sc. A setting is accepted if, at the same costs, it beats no adaptation for the 5-member ensemble **and** for every one of the 5 seeds scored alone; no minimum size. Among accepted settings the highest ensemble mean wins; none accepted → adaptation stays off. Reported but not deciding: each seed's SkillCorner-only gain and the per-condition gains. If accepted: `decoder.json` gets the setting and its costs, the submission is regenerated, validated and reproduced byte for byte. No further conditions are searched for EATA **Result:** **`tta_alpha` 0.5, EATA 0, gate off, costs (4, 8, 8, 1.0)**: ensemble mean 0.6476 vs 0.6448 without adaptation, **+0.0028**; every seed gains (+0.0008 / +0.0009 / +0.0024 / +0.0060 / +0.0001). Ensemble per condition sim −0.0006, shift −0.0018, sc +0.0108; sc gain per seed +0.0048 to +0.0189. Runner-up, also accepted: `tta_alpha` 0.5 + EATA 10 at the same costs 0.6475 (+0.0027, every seed +0.0022 to +0.0063), a 0.0001 near-tie (per condition sim −0.0029, shift −0.0013, sc +0.0122). 8 of the 46 settings pass the every-seed test, all with the gate off; every gate setting fails. The ensemble numbers reproduce ADcv5 (0.6476 / 0.6448), so the refactor changed nothing. The rule's own pick (TTA alone) was built overnight with a separate decoder file, `554fcb55…`, validated, byte-identical over two runs and from a clean checkout of `2b468e8`. **Tie-break:** Two settings passed the pre-registered every-seed rule and tied within 0.0001 on the ensemble (TTA 0.5: 0.6476; TTA 0.5 + EATA 10: 0.6475). The rule's tie-break would pick TTA alone. The team chose TTA 0.5 + EATA 10 after seeing the results, because its smallest per-seed gain is larger (+0.0022 vs +0.0001) and its gain on the real-tracking SkillCorner condition is larger (+0.0122 vs +0.0108). The public window was not used for this choice. **Adopted: TTA 0.5 + EATA 10, gate off, costs (4, 8, 8, 1.0)** (`77bf088`). Caveat: the study's EATA adapts on per-frame head inputs over the whole match, `predict.py` on overlapping inference windows (same update loop, see `adaptcv`). 8.2 h CPU | ~10 h |
| ❌ | E6 | **Dead-ball temperature** (`python -m tempolens.cvrescore temperature`): the dead-ball log-odds divided by T ∈ {1, 2, 4} before Viterbi (H:A ratio and change logit kept), on the cached ENS5 cross-validated outputs, costs re-searched (189 settings) for each T | jekkiechan's softening; the decoder has little headroom (oracle switch +0.0024), so a fail is expected | **Rule, fixed before the run (2026-11-07):** passes only if the best T beats T 1 (which must reproduce 0.7760) by ≥ +0.004 at its own best costs. Reported for the team; `decoder.json` is not changed tonight **Result:** T 1 **0.7760** at (2, 8, 8, 0.5) (reproduces ENS5); T 2 **0.7763** at (2, 2, 8, 0.5), +0.0003 (DFL −0.0008, Metrica +0.0027); T 4 0.7758 at (2, 2, 4, 0), −0.0002. Fails: no temperature | minutes |
| ✅ | E2a | **Label-convention rescoring, report only** (`python -m tempolens.cvrescore conventions`): the cached ENS5 and v7 3-member CV outputs scored against four label versions of the same 10 open matches: provider (the CV's labels) and controlled-1/2/3 s (`opendata.remove_flicker` at 25/50/75 frames on all 10 matches: a possession shorter than X sandwiched by the other team folds into it); each model at its own best setting per version | the private truth is "controlled" possession; our labels follow the provider | **Rule (2026-11-07):** report the four numbers per model; if ENS5 is ahead of v7 under all four, record that the ensemble choice is robust to the definition. Decides nothing **Result:** best setting per version, ENS5 vs v7 3 members: provider **0.7760 / 0.7716**; controlled-1 s 0.8004 / 0.7962; controlled-2 s 0.7705 / 0.7669; controlled-3 s 0.7233 / 0.7186; mean of the four at one setting 0.7623 (12, 2, 8, 1.0) / 0.7601 (8, 8, 8, 1.0). ENS5 ahead under all four (+0.0036 to +0.0047): the ensemble choice is robust to the definition. Longer minimum possessions cost both models alike (−0.05 at 3 s) | minutes |
| ✅ | E0 | **Keeper-slot diagnostic, aggregates only** (`tempolens.diagnostics` check 6, `keeper_zone`): per team, the share of graded frames on which one of its slots has stood ≥ 5 s in the *other* team's keeper zone (within 20 m of that goal line, central 40 m), plus slots and minutes involved; run locally on the match, only these counts printed, no labels read | five rivals report ATD keeper slots filed under the wrong team for minutes; TempoLens fixes each slot's team for the whole match | **Rule (2026-11-07):** under 0.5% of graded frames for both teams → drop E1; otherwise E1 is queued (not run tonight). Note: the zone also catches attackers waiting in the box, so a high share is a necessary, not sufficient, sign **Result:** home 20.2% of graded frames (22 of 24 slots, 21.3 min), away 22.7% (21 of 22 slots, 24.0 min). Over 0.5%, so E1 is queued by the rule; but nearly every slot is involved, i.e. the zone is dominated by attackers waiting in the box, and this measure cannot tell a mis-filed keeper from them. E1 should start with a keeper-specific check (e.g. a slot of one team spending most of a minute within 6 m of the other team's goal) | minutes |
| ⬜ | E1 | **Goalkeeper repair** (keeper-slot fault in `simulate.add_tracker_faults`, repair in `match.py`, thresholds on open matches only), queued by E0's rule | E0 over 0.5% | **Not run** (2026-11-07): it was not built overnight. E0's measure is not keeper-specific, so a keeper-specific check should come first | — |
| ❌ | E2b | **Controlled-2 s training labels** (`experiments --model ensemble --draws 4 --members 3 --min-spell 50`): the v7 recipe with DFL/SkillCorner training labels at a 50-frame flicker threshold (held-out labels unchanged); then `cvrescore conventions --min-spell 50` | E2a's hedge; generalises H3's `--min-spell 19` | **Rule (2026-11-07):** compare with v7 3 members on the mean over the four E2a label versions, each model at its best setting for that mean; passes only if the gain is ≥ +0.006, at least 2 of 3 members (each alone, same measure) are ahead, and provider-label spellF1 (each at its own best costs) falls by no more than 0.006. Reported for the team **Result:** mean of the four versions **0.7519** vs v7 3 members 0.7601, **−0.0082**; 0 of 3 members ahead (−0.0097 / −0.0066 / −0.0132); provider **0.7111** vs 0.7716, −0.0605 (DFL 0.6888). It wins only on the 2 s and 3 s versions it was trained for (0.7911 / 0.7697 vs 0.7669 / 0.7186). Fails on all three counts. 1.2 h GPU | ~2 h GPU |
| ❌ | E3 | **SkillCorner extra training, visible ball only** (`experiments --model ensemble --draws 4 --members 3 --extra-train skillcorner --sc-visible`): H1 again under the v7 recipe, but SkillCorner frames enter the loss only where its ball is detected (`is_detected`); tested on the 10 standard matches | H1 lost on Metrica (−0.023), likely from SkillCorner's long controlled spells while the ball is unseen | **Rule (2026-11-07):** passes only if ≥ +0.006 over v7 3 members (0.7716) at its own best costs, at least 2 of 3 members ahead (each alone at its own best costs), and the Metrica subset falls by no more than 0.006. Reported for the team **Result:** **0.7630** (DFL 0.7548, Metrica 0.7822) at (1, 2, 2, 0) vs v7 3 members 0.7716 (Metrica 0.7994): **−0.0086**, Metrica −0.0172; members alone 0.7595 / 0.7500 / 0.7603 vs 0.7576 / 0.7677 / 0.7633: 1 of 3 ahead. Fails. Visible-ball-only SkillCorner halves H1's Metrica loss but still costs. 1.8 h GPU | ~2.5 h GPU |
| ❌ | E5 | **Per-epoch held-out spellF1** (if GPU time remains before 11:00): one 16-epoch v7-recipe run per seed (members 0 and 1), 5 folds, held-out spellF1 logged after epochs 4–16 at costs (2, 8, 8, 0.5) | CoreOrigin: frame-level fit and spellF1 diverge | **Rule (2026-11-07):** a change of epoch count is proposed only if the best epoch beats epoch 8 by ≥ +0.006 on both seeds. Caveat recorded before the run: the learning rate follows one cycle over all 16 epochs, so a mid-run snapshot is not a model trained for that many epochs; a pass only queues a full run at that count (as EP12) **Result:** snapshots at (2, 8, 8, 0.5): member 0 epoch 8 0.7443, epoch 16 0.7562; member 1 epoch 8 0.7523, best epoch 10 0.7581, epoch 16 0.7534. **Correction (2026-11-07):** the epoch-8 snapshot is a model halfway through a 16-epoch one-cycle schedule, not an 8-epoch model, so "+0.0119 over epoch 8" compared the wrong thing. The valid comparison is the end of the run (a fully trained 16-epoch model) against the real 8-epoch v7 members at the same costs (cached CV outputs re-decoded): member 0 **0.7562 vs 0.7567 (−0.0005)**, member 1 **0.7534 vs 0.7638 (−0.0104)**. 16 epochs are no better on either seed; mid-run snapshots cannot stand for other epoch counts. **Fails**: 8 epochs kept. 2.7 h GPU | ~2.5 h GPU |
| ❌ | ENS7 / ENS9 | **7 and 9 members** (`--model ensemble --draws 4 --members 7` / `9`): members 5–8 added | 3→5 gave +0.0044; is the curve flat yet? | **Rule, fixed before the run (2026-11-06):** adopt only if ≥ **+0.004** over ENS5's 0.7760 at its own best costs (same grid); if both pass, the higher; adoption then repeats ENS5fin's steps (final members, CV-argmax costs, TPcv/ADcv reruns, regenerate, reproduce). **Result:** ENS7 **0.7766** (DFL 0.7663, Metrica 0.8008) at (2, 2, 2, 0), +0.0006; ENS9 **0.7756** (DFL 0.7655, Metrica 0.7992) at (2, 2, 2, 0), −0.0004: the curve is flat after 5 members | not adopted: 5 members kept |
| ✅ | SUB5 | **Is the 5-member gain a lucky pair?** Three more 5-member sets from the same cached members (`--member-set`): {0, 1, 2, 5, 6}, {0, 1, 2, 7, 8}, {4, 5, 6, 7, 8} | ENS5's +0.0044 rests on one pair of extra members | **Report only, fixed before the run:** no setting changes. If the mean of the four 5-member sets (ENS5 and these three) is under 0.7736 (+0.002 over 3 members), the technical document calls the 5-member gain uncertain. **Result:** {0, 1, 2, 5, 6} 0.7744; {0, 1, 2, 7, 8} 0.7719; {4, 5, 6, 7, 8} 0.7717. Mean of the four 5-member sets (with ENS5's 0.7760) **0.7735**, just under the 0.7736 line | per the rule, the technical document calls the 5-member gain uncertain: the expected gain over 3 members is ~+0.002, and ENS5's +0.0044 partly reflects a good pair of extra members. No setting changes (report-only rule); more members cannot hurt in expectation |
| ❌ | ADcv | **Test-time adaptation on external data only** (`python -m tempolens.adaptcv`): `tta_alpha` {0, 0.5, 1} × `eata_steps` {0, 10, 30, 60} × gate {off, on} × costs {(2, 2, 2, 0.5), (4, 8, 8, 1.0)}; three conditions: sim (5 folds, fixed test noise), shift (same fold models, a randomised noise draw plus identity swaps, duplicates and phantoms), sc (submitted members on the 20 SkillCorner matches as they are: real broadcast tracking, never trained on) | TTA, EATA and the gate were adopted on the provided match, which the 2026-11-03 rules forbid | none (2, 2, 2, 0.5): sim 0.7716, shift 0.7679, sc 0.3848, mean **0.6415**. Best: `tta_alpha` 0.5, costs (4, 8, 8, 1.0) 0.6454, **+0.0043** over none at the same costs, worst condition +0.0022; `tta_alpha` 0.5 + EATA 10 0.6451. `tta_alpha` 1 −0.002 to −0.007 on every condition; the gate loses 0.008–0.011 on sim and shift; the withdrawn setting (1, 30, gate, 4/8/8/1) 0.7388 / 0.7370 on sim / shift. Rule fixed before the run: adopt only with ≥ +0.005 mean and no condition below −0.005 | not adopted: mild re-scaling helps a little everywhere (most on real tracking, sc +0.013) but stays under the bar; no adaptation |
| ⏸ | H3 | Training labels match "control": DFL/SkillCorner flicker threshold 19 frames (public truth's shortest) instead of 10 (`train --min-spell 19`, 3 members, final decoder) | short possessions recalled 45% (27% of all possessions) | real match, vs final 0.647 / 0.881 / 17 of 17: **public spellF1 0.680, macroF1 0.873, 13 of 17 possessions**, dead-ball recall 0.726, 283 vs 291 possessions; 24 disagreements of 1 s+ (52 s). spellF1 up on fewer possessions found (the CH pattern: a lucky count on 17), tie-break down | reel review pending (`reel_final_minspell19.mp4`) |
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
| ⚠️ | W123 | Extra validation windows W1–W3 (3 × 5 min, frames 45,000–52,499, 127,500–134,999, 172,500–179,999) | the public window is only 17 possessions | **labels turned out to be algorithmic** (a nearest-player control rule plus Smart Tagging), not human, so they are not truth: their strict control rule finds few possessions (43 in 15 min), which made every version look like it over-splits. **Not used for any decision**; the earlier reel reviews were manual and stand | discarded |
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
| ✅ | CHKA | check A: submission with 1 torch thread equals the 4-thread file |  |  |  |  |  | 2026-11-01 16:45 |
| ✅ | CHKB | check B: fresh clone + new venv + pinned requirements + one command = same file |  |  |  |  |  | 2026-11-01 16:51 |
| ✅ | TC5s0 | touch head 0.5 on the v7 recipe, CV seed 0 (vs RN4s0) | 0.7629 | 0.7544 | 0.7828 | (6.0, 4.0, 8.0, 1.0) | candidate (+0.0053) | 2026-11-01 17:39 |
| ✅ | TC5s1 | touch head 0.5 on the v7 recipe, CV seed 1 (vs RN4s1) | 0.7617 | 0.7535 | 0.7808 | (4.0, 4.0, 8.0, 0.5) | no gain (-0.0060) | 2026-11-01 18:18 |
| ✅ | CHKB2 | fresh clone + new venv reproduces the gated submission byte for byte |  |  |  |  |  | 2026-11-01 18:28 |
| ✅ | TC1s0 | touch head 1.0 on the v7 recipe, CV seed 0 (vs RN4s0) | 0.7655 | 0.7551 | 0.7898 | (2.0, 8.0, 8.0, 0.5) | candidate (+0.0079) | 2026-11-01 19:07 |
| ✅ | TCtrain0 | touch head 0.5: train member 0 (models/exp/) |  |  |  |  |  | 2026-11-01 19:14 |
| ✅ | TCtrain1 | touch head 0.5: train member 1 (models/exp/) |  |  |  |  |  | 2026-11-01 19:21 |
| ✅ | TCtrain2 | touch head 0.5: train member 2 (models/exp/) |  |  |  |  |  | 2026-11-01 19:28 |
| ✅ | TCdec | touch head: decoder settings = submission's, members = touch models |  |  |  |  |  | 2026-11-01 19:28 |
| ✅ | TCpred | touch head: full real-match prediction with TTA + EATA |  |  |  |  |  | 2026-11-01 19:36 |
| ✅ | TCpub | touch head: public window + dead ball vs final |  |  |  |  |  | 2026-11-01 19:36 |
| ✅ | TCset | touch head: set pieces |  |  |  |  |  | 2026-11-01 19:36 |
| ✅ | TCreel | touch head: review reel vs final |  |  |  |  |  | 2026-11-01 19:43 |
| ✅ | TCtop | touch head: top-20 review sheet vs final |  |  |  |  |  | 2026-11-01 19:43 |
| ✅ | ST | self-training on the real match, variants A (confident spells) and B (+ physical anchors) |  |  |  |  |  | 2026-11-01 20:08 |
| ✅ | STpub | self-training: public window + dead ball vs final |  |  |  |  |  | 2026-11-01 20:08 |
| ✅ | STset | self-training: set pieces |  |  |  |  |  | 2026-11-01 20:08 |
| ✅ | STreelA | self-training A: review reel vs final |  |  |  |  |  | 2026-11-01 20:13 |
| ✅ | STtopA | self-training A: top-20 sheet vs final |  |  |  |  |  | 2026-11-01 20:13 |
| ✅ | STreelB | self-training B: review reel vs final |  |  |  |  |  | 2026-11-01 20:18 |
| ✅ | STtopB | self-training B: top-20 sheet vs final |  |  |  |  |  | 2026-11-01 20:18 |
| ✅ | H3train0 | H3: train member 0 (minspell19) |  |  |  |  |  | 2026-11-01 23:35 |
| ✅ | H3train1 | H3: train member 1 (minspell19) |  |  |  |  |  | 2026-11-01 23:42 |
| ✅ | H3train2 | H3: train member 2 (minspell19) |  |  |  |  |  | 2026-11-01 23:49 |
| ✅ | H3dec | H3: decoder file |  |  |  |  |  | 2026-11-01 23:49 |
| ✅ | H3pred | H3: predict the real match |  |  |  |  |  | 2026-11-01 23:56 |
| ✅ | H3val | H3: validate |  |  |  |  |  | 2026-11-01 23:56 |
| ✅ | H3pub | H3: public window |  |  |  |  |  | 2026-11-01 23:56 |
| ✅ | H3set | H3: set-piece table |  |  |  |  |  | 2026-11-01 23:56 |
| ✅ | H3reel | H3: review reel |  |  |  |  |  | 2026-11-02 00:02 |
| ✅ | H3top | H3: top disagreements |  |  |  |  |  | 2026-11-02 00:02 |
| ✅ | H1btrain0 | H1b: train member 0 (skillcorner) |  |  |  |  |  | 2026-11-02 00:16 |
| ✅ | H1btrain1 | H1b: train member 1 (skillcorner) |  |  |  |  |  | 2026-11-02 00:30 |
| ✅ | H1btrain2 | H1b: train member 2 (skillcorner) |  |  |  |  |  | 2026-11-02 00:43 |
| ✅ | H1bdec | H1b: decoder file |  |  |  |  |  | 2026-11-02 00:43 |
| ✅ | H1bpred | H1b: predict the real match |  |  |  |  |  | 2026-11-02 00:51 |
| ✅ | H1bval | H1b: validate |  |  |  |  |  | 2026-11-02 00:51 |
| ✅ | H1bpub | H1b: public window |  |  |  |  |  | 2026-11-02 00:51 |
| ✅ | H1bset | H1b: set-piece table |  |  |  |  |  | 2026-11-02 00:51 |
| ✅ | H1breel | H1b: review reel |  |  |  |  |  | 2026-11-02 01:06 |
| ✅ | H1btop | H1b: top disagreements |  |  |  |  |  | 2026-11-02 01:06 |
| ✅ | CMB | final teams + H1b dead/live (A, B) |  |  |  |  |  | 2026-11-02 02:01 |
| ✅ | CMBvalA | CMB A: validate |  |  |  |  |  | 2026-11-02 02:01 |
| ✅ | CMBvalB | CMB B: validate |  |  |  |  |  | 2026-11-02 02:01 |
| ✅ | CMBpub | CMB: public window |  |  |  |  |  | 2026-11-02 02:01 |
| ✅ | CMBset | CMB: set-piece table |  |  |  |  |  | 2026-11-02 02:01 |
| ✅ | CMBtopA | CMB A: top disagreements |  |  |  |  |  | 2026-11-02 02:01 |
| ✅ | CMBtopB | CMB B: top disagreements |  |  |  |  |  | 2026-11-02 02:01 |
| ✅ | CMBreelA | CMB A: review reel |  |  |  |  |  | 2026-11-02 02:16 |
| ✅ | CMBreelB | CMB B: review reel |  |  |  |  |  | 2026-11-02 02:31 |
| ✅ | ADcv | adaptation study on open data (tempolens.adaptcv) |  |  |  |  |  | 2026-11-05 06:08 |
| ✅ | FV3cv | v7 recipe without the restart-spot features (v3 set), 3-member CV | 0.7674 | 0.7584 | 0.7886 | (1.5, 8.0, 8.0, 0.0) | no gain (-0.0042) | 2026-11-05 21:56 |
| ✅ | ENS5 | v7 recipe, 5-member ensemble | 0.7760 | 0.7650 | 0.8017 | (2.0, 8.0, 8.0, 0.5) | no gain (+0.0044) | 2026-11-06 03:14 |
| ✅ | H128 | v7 recipe, GRU 128 units, 3-member CV | 0.7704 | 0.7616 | 0.7911 | (1.5, 8.0, 8.0, 0.0) | no gain (-0.0012) | 2026-11-06 04:40 |
| ✅ | EP12 | v7 recipe, 12 epochs, 3-member CV | 0.7723 | 0.7630 | 0.7942 | (2.0, 2.0, 8.0, 0.5) | no gain (+0.0007) | 2026-11-06 06:18 |
| ✅ | TPcv5 | tag study, 5-member ensemble, costs (2, 8, 8, 0.5) |  |  |  |  |  | 2026-11-06 06:38 |
| ✅ | ADcv5 | adaptation study, 5-member ensemble (tempolens.adaptcv --members 5) |  |  |  |  |  | 2026-11-06 17:03 |
| ✅ | ENS7 | v7 recipe, 7-member ensemble | 0.7766 | 0.7663 | 0.8008 | (2.0, 2.0, 2.0, 0.0) | no gain (+0.0006) | 2026-11-06 19:14 |
| ✅ | SUB5b | 5-member set {0,1,2,5,6} | 0.7744 | 0.7645 | 0.7975 | (2.0, 2.0, 2.0, 0.5) | no gain (+0.0028) | 2026-11-06 19:16 |
| ✅ | ENS9 | v7 recipe, 9-member ensemble | 0.7756 | 0.7655 | 0.7992 | (2.0, 2.0, 2.0, 0.0) | no gain (-0.0004) | 2026-11-06 20:24 |
| ✅ | SUB5c | 5-member set {0,1,2,7,8} | 0.7719 | 0.7620 | 0.7949 | (1.5, 2.0, 8.0, 0.0) | no gain (+0.0003) | 2026-11-06 20:27 |
| ✅ | SUB5d | 5-member set {4,5,6,7,8} | 0.7717 | 0.7623 | 0.7936 | (2.0, 4.0, 8.0, 0.5) | no gain (+0.0001) | 2026-11-06 20:30 |
<!-- autoqueue:end -->
