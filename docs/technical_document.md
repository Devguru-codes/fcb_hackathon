# TempoLens — technical document (draft)

*More than a Hack 2027, Phase 1: ball possession. Draft: numbers marked TBD are filled when the
final model is frozen.*

## 1. Approach in one paragraph

We treat possession as a sequence-labelling problem learned from **open football data with
known possession**, not as a hand-tuned rule. A bidirectional recurrent network reads ~140
identity-free, team-symmetric features per frame and outputs P(home), P(away), P(dead ball) and
the probability that possession changes. A Viterbi decoder turns those into whole possessions
under switch and stoppage costs chosen by cross-validated **spellF1**. Because open tracking is
clean while the competition tracking is not, every training and validation match is first
corrupted by a **simulator of the automatic tracking, calibrated so the official baseline
reproduces its published scores**.

## 2. What the data told us before modelling

From the organisers' public files only (`evaluation/results/baseline_raw.csv` is D exactly where
the automatic tracking has no ball; the public ground truth covers five minutes):

| fact | value | consequence |
|---|---|---|
| ball missing, graded frames | 40.7% | half the problem is ball-free |
| ball gaps | ~150, median 268 frames (10.7 s), mean 431 | losses are long blind stretches, not flicker |
| ball missing in the public window | ~12% | the public window badly under-represents the hard part |
| nearest player's team = owner, ball visible and alive | 77.8% | the nearest-player rule is a weak signal on its own |
| baseline macroF1 0.65 with ball-missing = D | — | missing-ball gaps are aligned with stoppages |

Checked on the released data with `python -m tempolens.diagnostics`, which prints aggregates only:
the tracking has one record per tactical video frame; our loader reproduces the organisers'
`baseline_raw.csv` on 100% of frames (frame alignment and home/away mapping); 40.7% of graded
frames have no ball, in 150 gaps of median 268 frames, as the simulator assumes; every Smart
Tagging team name resolves (the tagging spells the away side "Atletic Club").

Two bugs only the real tracking could reveal, both fixed:

- **Simulated jitter read as play stopping.** The first simulator added independent per-frame
  position noise. Through 25 fps velocities that is ~2.6 m/s of fake speed: simulated players ran
  at a median 3.8 m/s against 1.5 m/s on the real tracking, so the model learned "everyone slow =
  stoppage" and called 88% of the real match dead (public spellF1 0.13). The real tracking is
  smooth; jitter is off, and simulated speed statistics now match (1.65 vs 1.47 m/s).
- **Features that saw the slot count.** The real tracking spreads ~20 visible players over 46
  anonymous slots; open data has ~15 per team. A team's "share in its own half" was divided by
  the slot count, and "nearest player's speed" fell back to an arbitrary slot when the ball was
  missing. Cross-validation could not see either (every open match has few slots). Features v3
  fixes both; a test pads and shuffles slots and requires identical features.
- **Dead ball missed at goal kicks.** Around Smart Tagging's goal-kick tags the model predicted
  dead ball on only 30–49% of frames, against ~76% around goal kicks on open data. Cause: Metrica's
  sample games never track the ball while it is dead, so 3 of 10 training matches taught "no
  ball = dead ball", while on the real match the ball stays visible through much of a stoppage
  (on the public window, 66% of dead frames). Simulator v2 fills Metrica's dead-time ball (a
  straight line from where it was last seen to where play restarts) before the usual stoppage
  losses. After retraining, dead-ball share before goal kicks rose to 56–71%, after throw-ins
  to 88%.

A per-feature comparison of simulated and real features (quantiles over all frames, nothing
frame-level) guided two further simulator changes: tracker ball noise (0.2 m, smoothed over ~3
frames) brings the ball-acceleration profile in line (shift score 1.01 → 0.38, speeds unchanged),
and 4% player dropout instead of 7% matches the real share of frames with 21+ players (shift
1.40 → 0.70). Real tracking also has rare long blackouts during stoppages (≤14 players on 5% of
frames), which the model already labels dead ball (96%).

## 3. Assets used and dropped

| asset | use |
|---|---|
| tactical ATD | the only input to the model: player and ball positions in metres, slots treated as anonymous tracklets |
| Smart Tagging | optional self-calibrated prior on which team has the ball (§6a); defensive codes (`DEFENDING IN …`, `DEFENSIVE TRANSITION`) name the side *without* the ball and are inverted; team names matched fuzzily ("Atletic Club"); set-piece tags used only as a check of dead ball around restarts, and tested as a restart prior (§7b) |
| homography, tactical video | tried, not used: a ball detector trained on ATD-projected labels was too imprecise on the real video to fill gaps safely (§6b) |
| broadcast feed | tried, not used so far: aligned to the tactical tracking with one constant offset per half, its ball fills 9.7% of the tactical ball gaps, but a filled position is off by more than 2 m about a quarter of the time and the public window cannot judge it (§6b, §7b) |
| open data | Metrica Sports sample games 1–3 (labels from events), DFL/Sportec seven matches (Bassek et al., CC-BY 4.0; labels from the provider's per-frame ball status and owner) |

## 4. Labels on open data, and the possession rules they encode

- **DFL**: dead where the provider's `BallStatus` is 0; otherwise the team in `BallPossession`.
  This is the same kind of signal the competition ground truth is derived from. Owner flips
  shorter than 10 frames sandwiched by the other team are folded back (flicker, not control).
- **Metrica**: each half dead until the kickoff; a set piece restarts play for its taker; ball
  out, shots out or in, offside, handball and fouls kill the ball; while alive, a possession
  event (pass, shot, recovery, loss, set piece) by the other side hands possession over.
  **Challenges, single touches and fouls received never confer possession** — the
  competition's "touch without control is not a change".

## 5. Simulating the automatic tracking

Per match: each stoppage is lost with probability 0.67 (its edges jittered by ~0.6 s); further
log-normal gaps (median 268 frames, σ 0.95) top the missing fraction up to 40.7%; players drop
out so ~20.5 are visible. Positions are not jittered (see §2). Calibration on DFL match J03WMX
against the published full-match baseline scores (measured before jitter was removed; the
baseline reads only the nearest player, which the jitter barely moved):

| | raw | smooth25 | smooth50 | smooth125 |
|---|---|---|---|---|
| published macroF1 | 0.653 | 0.675 | 0.672 | 0.636 |
| simulated | 0.628 | 0.648 | 0.644 | 0.610 |
| published spellF1 | 0.049 | 0.200 | 0.243 | 0.224 |
| simulated | 0.046 | 0.257 | 0.256 | 0.198 |

Ranking order is reproduced under both metrics; uniform (non-stoppage) gaps cannot get macroF1
within 0.08 of the published values at any bias.

## 6. Model and decoder

- Features: control geometry (nearest/second-nearest per team, counts within radii, relative
  speeds), ball state (imputed across gaps ≤2 s, speed, out of play, distance to lines/spots),
  team shape (sorted player coordinates, block length/width, velocity quantiles), memory across
  ball gaps (who was nearest and where the ball was before it vanished and after it returned),
  and ±0.5/2/5 s context windows. Coordinates oriented to each team's attacking direction,
  inferred from positions. Computed from both teams' points of view; training uses both and
  inference averages them, so the model cannot learn a side.
- Network: 2-layer bidirectional GRU (96 units), 30 s training windows, cross-entropy on H/A/D
  plus a weighted change-detection loss. LightGBM on the same features as a comparison.
- Decoder: Viterbi over H/A/D, costs for team switch, alive→dead and dead→alive; optional bonus
  from the change head.

## 6a. Smart Tagging as a self-calibrated prior

Tags exist only for the competition match, so their weight cannot be learned from open data,
and the public window is too small to tune it. The prior calibrates itself without labels:
each tag votes for the side in possession over its interval (two families: attacking and
defensive codes); on frames where the ensemble is >90% sure which side has the ball, the
family's agreement rate r with the ensemble estimates its reliability; the family's weight is
0.5·logit(r), clipped to [0, 2], and shifts only the home-versus-away log-odds.

Tested on the open matches with synthetic tags (phases on 60% of possessions, ±1 s boundaries,
30% running on over the next possession, team error swept), using the cross-validated ensemble:

| tag team error | DFL spellF1 | learned weights (attack / defence) |
|---|---|---|
| no prior | 0.730 | — |
| 0% | 0.743 | 0.82 / 0.62 |
| 10% | 0.736 | 0.59 / 0.45 |
| 20% | 0.735 | 0.40 / 0.33 |
| 35% | 0.729 | 0.20 / 0.28 |
| 50% (random) | 0.730 | 0.05 / 0.07 |

It helps when tags are informative and costs nothing when they are not. Shrinkage 1.0 gained
more with perfect tags (+0.017) but lost 0.004 at 35% error, so 0.5 was kept. Caveat: real tag
errors may correlate with the model's own (both read the same tracking), which synthetic tags do
not capture; the reliability measured on the real match is reported by `tempolens.diagnostics`
before the prior is switched on (`"tag_prior": "auto"` in `models/decoder.json`).

On the real match the tags agree with a confident model 83% (attacking codes) and 94%
(defensive codes) of the time, yet the prior lowered the public-window spellF1 (0.43 vs 0.59 for
a single GRU): a tag spans a whole phase and swallows short counter-possessions. It is off.

## 7. Validation

5-fold grouped cross-validation over the 10 open matches, every match corrupted by the
simulator (two noise draws per training match, a fixed draw per test match), scored with the
official `possession_metrics.py`. The public five minutes are used only as an alignment check.

| method | DFL mean spellF1 | all mean spellF1 |
|---|---|---|
| official baseline (smooth50), simulator v1 | 0.267 | 0.291 |
| GRU, features v2, simulator v1 | 0.759 | 0.784 |
| 3 GRUs (seeds × noise draws), features v2, simulator v1 | 0.768 | 0.792 |
| 3 GRUs, features v3, simulator v1 | 0.770 | 0.794 |
| 3 GRUs, features v3, simulator v2 (v4) | 0.754 | 0.764 |
| 3 GRUs, features v4 (+ restart spots), simulator v2 (v5) | 0.7482 | 0.7591 |
| **3 GRUs, v5 recipe + domain randomisation (v7); submitted with test-time adaptation** | 0.7555 | 0.7682 |

Simulator v2 is harder and closer to the real tracking (visible dead-time ball, tracker ball
noise), so its scores are not comparable with v1's; the choice between them rests on the real
match. On the public five minutes (17 possessions; a sanity check, never used for tuning):
official baseline smooth50 0.326 spellF1 / 0.621 macroF1; v3 ensemble 0.518 / 0.732 (17
possessions); v4 0.573 / 0.790 (20); **submitted v5 0.594 / 0.819** (18 possessions; dead-ball
recall 0.534 vs 0.478). Full match: ~324 possessions of mean 11.6 s.

Features v4 add five restart-spot features — fastest ball within ±1 s, ball in the goal area,
ball at a corner flag, frames since/until the ball was on or over a line — aimed at the
real-match stoppages still missed (dead-ball recall 0.48 on the public window). They tie in
cross-validation; v5 was chosen on the real-match sanity numbers, a judgement call recorded as such.

Earlier experiments ran under the first simulator (with jitter), so their absolute values are
lower; they decided relative choices only: baseline 0.262 DFL, LightGBM 0.566, GRU 0.719, 3-GRU
ensemble 0.730. Tried and not adopted there (DFL mean spellF1 of the model alone, then with the
3-GRU ensemble):

| variant | alone | in ensemble | verdict |
|---|---|---|---|
| player-set encoder (per-frame attention over players, PathCRF-style) | 0.714 | 0.726 | slightly worse, 4× slower |
| TCN (dilated convolutions) | 0.717 | 0.727 | close, no ensemble gain |
| bidirectional LSTM | 0.707 | 0.729 | worse alone, no ensemble gain |
| bigger GRU (128 units, 16 epochs) | 0.715 | 0.731 | tie alone; +0.001 in ensemble, within noise |
| transformer (5 fps tokens + full-rate skip) | 0.656 | 0.724 | worse: too coarse for short possessions |
| hidden-ball auxiliary head, weight 1.0 / 0.1 | 0.699 / 0.710 | — / 0.725 | hurts: competes with the possession objective |
| MS-TCN++ truncated smoothing loss, weight 0.15 (simulator v2; base 0.746 DFL / 0.754 all) | 0.750 DFL / 0.754 all | — | tie: the decoder's switch cost already does this job |

Decoding alternatives, on the cross-validated ensemble output under the first simulator
(`python -m tempolens.decodestudy`):

| decoder | DFL | all |
|---|---|---|
| Viterbi, switch 2 (submitted) | 0.7295 | 0.7592 |
| minimum-duration Viterbi (best: 6-frame minimum) | 0.7297 | 0.7593 |
| minimum-Bayes-risk choice of switch cost per match | 0.7271 | 0.7574 |
| oracle: best switch per match in hindsight (ceiling) | 0.7314 | 0.7610 |

Decoding is not the bottleneck: even the hindsight-optimal per-match decoder gains 0.002. MBR
picked the lowest switch cost everywhere, because chain-CRF samples over per-frame probabilities
are more fragmented than real possession sequences.

Robustness to tracker faults absent from training (`python -m tempolens.robustness`): one
cross-validation fold, held-out matches re-scored after adding identity swaps between nearby
team-mates, duplicate tracks and phantom (non-player) tracks near the ball — 70–80 extra slots,
up to 24 tracked objects per frame as in the real ATD. spellF1 0.7087 → 0.7114 and
0.7134 → 0.7157: no loss, so the features' slot-free design absorbs these faults.

## 7b. After v5: the real match decides

Simulated cross-validation chose every version up to v5. After v5 it stopped being enough, and
every candidate is now also judged on the real match. Three checks are used, none of which feeds
back into training: the public window (spellF1, macroF1, dead-ball recall); dead-ball share in the
seconds around Smart Tagging's 59 set-piece tags; and a **human review of every disagreement of
1 s or more** between two full-match submissions. The review is a reel rendered from the tactical
video with both submissions' labels drawn on it (`python -m tempolens.review`), and a person marks
which one is right. Hand labels only validate, as the rules allow.

- **Pre-kickoff frames.** The second half is graded from frame 110,808 but kicks off at 113,742
  (FAQ). Every graded frame before that is dead ball, but the model labelled only 22% of them D.
  `predict.py` now sets them after decoding, and the end-to-end test asserts it.
- **Human review.** v5 was right on 34 of the 62 disagreements with v4, against 24 for v4 (3 both
  wrong, 1 unclear). v4 splits open play with false dead stretches. v5 misses dead ball at goal
  kicks and restarts. The clips both get wrong are restarts and aerial hand-overs.
- **Domain randomisation** (v7): each training match gets 4 noise draws, each with its own
  simulator settings (ball missing 30–50%, stoppages lost 40–90%, gap median 6–18 s, player
  dropout 2–8%, ball noise 0.1–0.4 m). It was the clearest cross-validation gain we measured: the
  ensemble scored 0.7716 vs 0.7636, and the gain held at every decoder setting and for three seeds.
  **It did not transfer:** on the review, v5 was right 38 times and v7 29 times, because v7 merges
  across real turnovers. Not adopted on its own. Since this result, no simulated gain is adopted
  without real-match evidence.
- **Test-time adaptation** (allowed by the organisers): each member's feature scaling is re-fitted
  to the real match's own medians and spreads before predicting, with no labels
  (`temporal.adapt_normaliser`, set by `tta_alpha` in the decoder settings). On open data it costs
  0.019 spellF1, as expected, because simulated test matches have no domain shift to correct. On
  the real match it lifts every measure we have. For v7 with adaptation: public spellF1 0.629 vs
  0.594, macroF1 0.849 vs 0.819, dead-ball recall 0.647 vs 0.534 (precision 0.88), and dead ball
  in the last 5 s before goal kicks 0.77 vs 0.59. **Submitted:** on human review of all 88
  disagreements with v5, v7 with adaptation was right 45 times and v5 33 (4 both wrong, 6
  unclear), and 14 to 4 on the 20 longest. Most of its wins drop v5's stale "live" labels during
  stoppages (keepers holding the ball, set pieces being prepared, injuries, ball out). Its
  remaining weakness is the reverse: live balls at restarts (corner deliveries, balls a keeper
  releases) called dead ball.
- **Output-layer adaptation (EATA, Niu et al. 2022)**, after the re-scaling: each member's
  output layer (192 × 4 weights plus 4 biases; the rest of the network frozen) takes 30
  full-batch Adam steps that lower the H/A/D entropy of frames that are already confident
  (entropy < 0.4 ln 3), weighted exp(E0 − E), over every 5th frame inside the halves and both
  team views (`temporal.adapt_head`, `eata_steps` in the decoder settings). It is CPU float64 in a
  fixed order, so it is deterministic. Public window 0.647 / 0.8725 (17 of 17 possessions),
  dead-ball recall 0.728 (precision 0.876). Human review of all 24 disagreements with the
  version without it: 20 to 4. **Submitted.** Against v5, over all 96 disagreements of 1 s or
  more, the submission was right 54 times (277 s) and v5 33 times (93 s); 4 both wrong, 5 unclear.
- **Simulator check.** Ball missing on 40.7% of graded frames in both simulation and the real
  match, with the same share in gaps of 20 s or more (0.68 real, 0.65–0.78 simulated) and the same
  missing rate around stoppages (0.65–0.73 real, 0.70 simulated). The public window is atypical:
  the ball is missing only ~12% of the time there.

Tried after v5 and not adopted:

| idea | result |
|---|---|
| collective kinematics features (fastest player, share of still players) | CV +0.004 / −0.012; public 0.573 |
| keeper-hold features (ball near each team's deepest player, ball in the box) | CV −0.0075 / −0.0055 |
| touch-recency and nearest-team flip features (after the best public rival) | CV −0.0023 / −0.0054 |
| dead-ball frames weighted 2× in the loss | CV −0.0010 / −0.0015 |
| randomisation plus identity swaps, duplicates and phantoms; 6 draws instead of 4 | CV −0.0052; −0.0038 |
| 60 s training windows (inference reads 120 s) | CV +0.0013 / −0.0064; real match mixed (spellF1 0.635, macroF1 0.794) |
| minimum possession length 19 / 25 frames on the real match | public 0.583 vs 0.594 |
| Smart Tagging restart prior (log P(D) raised from 5 s before each set-piece tag to 3.5 s after) | public spellF1 unchanged, macroF1 +0.004 |
| broadcast ball, one constant offset per half (4014 / 26,920 frames) | missing ball 40.7% → 36.7% of graded frames; public window unchanged; 2.1% of the match relabelled, with no evidence of which version is right |

## 6b. Video ball detector (built, tested on the real video, not used)

A TrackNet-style heatmap network (3 stacked frames in, ball probability per pixel out) trained
on labels the ATD itself provides: wherever the tracking has the ball, its pitch position through
that frame's homography is the ball's pixel position — no hand labelling. Candidates in the ATD's
gaps are linked by a Viterbi pass (≤40 m/s, on or near the pitch, joined to the tracked ball at
the gap edges) and only confident, sustained detections fill the gap; by default only where the
first ensemble pass says play is live, so the "ball missing at a stoppage" cue survives.

Rehearsed end to end on a synthetic tactical video rendered from a DFL match (players as discs,
the ball a 2 px dot hidden when covered): held-out crops 82% within 5 px; filled positions at the
default threshold (p ≥ 0.95, ≥5 consecutive frames) covered 47% of gap frames with 100% within
1.5 m (median 0.16 m) — measured on the same clip the threshold was chosen on, so an optimistic
figure, not a held-out one; on the real video the threshold is set from the data-day audit
(`audit_confident_within_1p5m`). At p ≥ 0.85 coverage was 91% but only 56% within 1.5 m — precision is
what matters, a wrong ball being worse than none.

On the real tactical video (12,000 positive and 6,000 negative crops, labels from the ATD as
above) the held-out crops were 22% within 5 px, 57% within 15 px (about 1 m), median error
12.6 px; only the 5% most confident detections were reliable (95% within 15 px). At the fill
threshold that covers a small share of the gaps, a full scan of the gap frames takes hours, and
weights trained on the licensed video may not be publishable, which the submission's exact
reproduction would need. It is therefore not used. A stronger detector (e.g. WASB, Tarashima et
al., BMVC 2023) is the obvious upgrade.

**Broadcast tracking as a second ball source.** The broadcast feed's tracking is not
frame-aligned to the tactical one, but can be aligned from the tracking alone: broadcast frames
matched to the tactical frame with the same ball position and velocity, verified by player
positions (median distance < 1.5 m), then grouped into constant-offset segments that also pass
the player check every second. 39.5% of broadcast frames align. Yet its ball fills only 5.3% of
the tactical ball gaps (about 2% of graded frames): both cameras lose the ball in the same hard
moments. A second attempt uses one constant offset per half (4014 and 26,920 frames, voted by
the verified pairs; about 65% of 4 s blocks pass the player check, replays and close-ups fail).
It fills 9.7% of the gaps, but where both cameras see the ball they disagree by more than 2 m
about a quarter of the time. Not used so far (§7b).

PathCRF (Kim et al., KDD 2026) was reviewed but its released weights were not used: it needs 22
persistent player identities (the ATD has anonymous, fragmented tracklets), models only live
play, and was trained on five of the seven DFL matches used in our validation.

Under the first simulator, recall of true possessions at IoU 0.5, GRU against LightGBM: 1–2 s
possessions 0.51 vs 0.21; possessions with the ball visible under 20% of the time 0.33 vs 0.25.
Decoder for the submitted ensemble: switch 4, alive→dead 8, dead→alive 8, change-head bonus 1.0
(the first ensemble where the boundary head helps). Re-running the decoder study on its
cross-validated output: minimum-duration Viterbi +0.001, MBR +0.0007, hindsight-optimal switch
per match +0.0024 — decoding is still not the bottleneck.

## 8. Limitations

- Open-data labels approximate, but are not, Metrica's corrected-tracking definition; the
  public window is too small to calibrate the definition of "control" precisely.
- The simulator matches published aggregates, not the joint structure of the real errors.
- Very short possessions (<2 s) remain the weakest part. The model predicts ~85% of the true
  possession count on open data. On the competition match it predicts ~275 possessions of mean
  14.5 s; the public window's mean possession (14.2 s) over our ~3,985 s of live play implies
  ~280, so there is no sign of systematic merging there.
- Transition timing on the public window shows no constant lag (median offset −1 frame) but a
  typical error of ~0.8 s, and dead-ball onsets are the most often missed.
- Cross-validation cannot see faults that open tracking lacks (see §2): the slot-count bugs were
  found by reading the code, not by a score.
- A gain under the simulator can fail to transfer (domain randomisation, §7b). Real-match
  evidence is the public window (17 possessions, a 95% interval roughly ±0.16 wide), the set-piece
  tags and human review, all small samples.

## 9. Reproduction

```bash
pip install -r requirements-tempolens.txt
python -m tempolens.predict --data-root data --output <team>.csv     # committed weights, CPU
python evaluation/validate_submission.py <team>.csv
```

The ensemble members are listed in `models/decoder.json`, whose costs were tuned for exactly
them; each weights file records its feature version and refuses to load against another. The same
file carries `tta_alpha`, the test-time re-normalisation strength (0 = off), and the end-to-end
test checks that an adapted run reproduces byte for byte.
Inference runs on the CPU in float64 and rounds to 4 decimals, so a different CPU (e.g. AVX2 vs
AVX-512) gives the same file. Retraining from open data: `python -m tempolens.train --member k`
(GPU training is not bit-reproducible, which is why the weights used for the submission are
committed).
