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

## 3. Assets used and dropped

| asset | use |
|---|---|
| tactical ATD | the only input to the model: player and ball positions in metres, slots treated as anonymous tracklets |
| Smart Tagging | optional decoder prior; defensive codes (`DEFENDING IN …`, `DEFENSIVE TRANSITION`) name the side *without* the ball and are inverted; team names matched fuzzily ("Atletic Club") — weight TBD |
| homography, videos | not used so far; the tracking is already in pitch coordinates. A local video ball detector to fill tracking gaps is the main untried extension |
| broadcast feed | not used: unsynchronised, a third of frames without homography |
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

Per match: each stoppage is lost with probability 0.67 (edges jittered); further log-normal gaps
(median 268 frames, σ 0.95) top the missing fraction up to 40.7%; players drop out so ~20.5 are
visible; positions are jittered. Calibration on DFL match J03WMX against the published
full-match baseline scores:

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

## 7. Validation

5-fold grouped cross-validation over the 10 open matches, every match corrupted by the
simulator (two noise draws per training match, a fixed draw per test match), scored with the
official `possession_metrics.py`. The public five minutes are used only as an alignment check.

| method | DFL mean spellF1 | all mean spellF1 |
|---|---|---|
| official baseline (smooth50) | 0.262 | 0.285 |
| LightGBM, features v1 | 0.558 | 0.585 |
| LightGBM, features v2 | 0.566 | 0.591 |
| GRU, features v2 | 0.719 | 0.749 |
| **3 GRUs (seeds × noise draws), averaged — submitted** | **0.730** | **0.759** |

Tried and not adopted (same protocol; DFL mean spellF1 of the model alone, then with the
3-GRU ensemble):

| variant | alone | in ensemble | verdict |
|---|---|---|---|
| player-set encoder (per-frame attention over players, PathCRF-style) | 0.714 | 0.726 | slightly worse, 4× slower |
| TCN (dilated convolutions) | 0.717 | 0.727 | close, no ensemble gain |
| transformer (5 fps tokens + full-rate skip) | 0.656 | 0.724 | worse: too coarse for short possessions |
| hidden-ball auxiliary head, weight 1.0 / 0.1 | 0.699 / 0.710 | — / 0.725 | hurts: competes with the possession objective |

PathCRF (Kim et al., KDD 2026) was reviewed but its released weights were not used: it needs 22
persistent player identities (the ATD has anonymous, fragmented tracklets), models only live
play, and was trained on five of the seven DFL matches used in our validation.

Per held-out DFL match the GRU scores 0.705–0.758 (baseline 0.248–0.284). Recall of true
possessions at IoU 0.5, GRU against LightGBM: 1–2 s possessions 0.51 vs 0.21; possessions with
the ball visible under 20% of the time 0.33 vs 0.25. Decoder for the ensemble: switch 2,
alive→dead 4, dead→alive 8, no change-head bonus (it did not help at any weight tried).

## 8. Limitations

- Open-data labels approximate, but are not, Metrica's corrected-tracking definition; the
  public window is too small to calibrate the definition of "control" precisely.
- The simulator matches published aggregates, not the joint structure of the real errors.
- Very short possessions (<2 s) remain the weakest part.

## 9. Reproduction

```bash
pip install -r requirements-tempolens.txt
python -m tempolens.predict --data-root data --output <team>.csv     # committed weights, CPU
python evaluation/validate_submission.py <team>.csv
```

Retraining from open data: `python -m tempolens.train` (GPU training is not bit-reproducible,
which is why the weights used for the submission are committed).
