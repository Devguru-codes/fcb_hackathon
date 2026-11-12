---
title: "Coding United — TempoLens"
subtitle: "More than a Hack 2027, Phase 1: ball possession. Short technical document"
---

| | |
|---|---|
| team | Coding United |
| possession file | `Coding United.csv`: 199,319 rows, passes `evaluation/validate_submission.py`, sha256 `33ad45598f4790d7f6fd2193c4bd9ed825c2e6a77882f90724b94df07b0581a6` |
| method | TempoLens: 10 bidirectional GRUs trained on open data under simulated tracking noise, test-time adaptation, a Smart Tagging prior, Viterbi decoding |
| reproduction | `python -m tempolens.predict --data-root data --output "Coding United.csv"` (section 8) |
| long appendix | `docs/technical_document.md`; every experiment and its decision rule: `docs/plan_of_action.md` |

## 1. Approach, and why

**What the competition tracking looks like.** The organisers' public files and our own aggregate
checks of the automatic tracking (ATD) show two things. First, the ball is missing on 40.7% of
graded frames, in about 150 gaps with a median of 10.7 s. So for much of the match, possession has
to be read from how the 22 players move. Second, the public 5-minute window is not representative:
the ball is missing on only ~12% of its frames, and it holds 17 possessions. It can check that a
pipeline is wired correctly. It cannot choose between versions of one.

**So we learn possession from open matches whose possession is known.** We use Metrica's three
sample games and seven DFL/Sportec Bundesliga matches. Before training or validation, every one is
corrupted by a simulator of the automatic tracking. The simulator loses most stoppages, adds long
log-normal ball gaps up to the competition's 40.7%, drops players, and adds tracker-like ball
noise. Its parameters were calibrated so that the organisers' four baselines reproduce their
published full-match scores (EVALUATION.md). The members are trained on four noise draws per
match, each with its own simulator settings (domain randomisation).

**Pipeline** (`tempolens/`):

1. **Features** (`features.py`): about 140 per frame. They have no player identities and are
   symmetric between the teams: control geometry (nearest and second-nearest player per team,
   counts within radii, relative speeds), ball state with gaps of up to 2 s imputed, team shape,
   memory across ball gaps (who was nearest and where the ball was before it vanished and after it
   came back), restart-spot cues, and context windows of ±0.5, 2 and 5 s. The features are computed
   from both teams' points of view and averaged, so the model cannot learn a side.
2. **Model** (`temporal.py`): a 2-layer bidirectional GRU with 96 units, trained on 30 s windows
   with cross-entropy on home/away/dead plus a possession-change head. The submission averages
   **10 members**:
   - five trained on the provider's labels, with owner flips under 10 frames folded ("ENS5",
     `tempolens_rnd4_m0..4.pt`);
   - five with the same recipe whose DFL labels fold spells under 50 frames, about 2 s ("ms50",
     `tempolens_ms50_m0..4.pt`). Section 5 says why.
3. **Test-time adaptation** without labels: each member's feature scaling moves halfway toward the
   match's own statistics (`tta_alpha` 0.5). Then its output layer takes 10 entropy-minimisation
   steps on the match (EATA 10).
4. **Smart Tagging prior** (`tagprior.py`): each tag votes for the side in possession over its
   interval. Defensive codes name the side without the ball, so they are inverted. Each tag family
   is weighted by how often it agrees with the ensemble on frames where the ensemble is over 90%
   sure. This is self-calibration on the match, with no labels. On the submission the attacking
   family gets weight 0.729 and the defensive family 1.219.
5. **Decoder** (`decoder.py`): Viterbi over home/away/dead with a switch cost of 4, alive→dead and
   dead→alive costs of 8, and a bonus of 1.0 × the change head's logit. Then `X` is written outside
   the halves, and `D` on the second half's graded frames before its kickoff (frame 113,742, from
   the FAQ).

**Why this and not something else.** Under the first simulator, LightGBM on the same features
scored 0.566 DFL spellF1 against the GRU's 0.719. Several alternatives were no better than the
GRU, and some were worse: a TCN, an LSTM, a transformer, a per-frame player-set encoder, a bigger
GRU, and auxiliary hidden-ball or smoothing losses. Smarter decoding was not the bottleneck either.
Minimum-duration Viterbi and MBR gained at most 0.001, and even the best switch cost per match,
picked in hindsight, gained only 0.002. Domain randomisation was the clearest gain we measured
(0.7716 against 0.7636 for 3 members).

## 2. Assets used and dropped

| asset | status |
|---|---|
| tactical ATD | **used**, the model's only tracking input. Positions in metres; the 46 slots are treated as anonymous tracklets |
| Smart Tagging | **used** as the self-calibrated prior above. Its set-piece tags served only as a check of the dead-ball share around restarts |
| tactical video + homography | **dropped**. A TrackNet-style ball detector, trained on ATD-projected positions (no hand labels), reached only 22% of crops within 5 px on the real video. A detector trained on the match video is also out under the rules mail |
| broadcast feed and its ATD | **dropped**. We aligned it to the tactical tracking from the tracking alone, but its ball fills only 9.7% of the tactical ball gaps: both cameras lose the ball in the same moments |
| PathCRF released weights | **dropped**: they need persistent player identities and were trained on DFL matches in our validation set |
| SkillCorner, PFF FC | **test only** (section 3), never used to train the submitted models |

## 3. Training and evaluation data

No label from the provided game was used to train, tune or validate anything.

| dataset | source and licence | who labelled it, and how | amount | role |
|---|---|---|---|---|
| Metrica Sports sample games 1–3 | github.com/metrica-sports/sample-data; public sample data, source acknowledged | Metrica's own event data shipped with the games. Our rules turn the events into per-frame labels (section 4) | 3 full matches | training and cross-validation |
| DFL/Sportec open data | Bassek, Rein, Weber & Memmert (2025), *Scientific Data* 12, 195; figshare 28196177; CC-BY 4.0 | the provider's per-frame ball status and ball-possession flags, as published with the dataset | 7 full Bundesliga matches | training and cross-validation |
| SkillCorner open data | github.com/SkillCorner/opendata; MIT | SkillCorner's owner field and phases of play, from broadcast tracking | 20 A-League matches | test only: the "sc" real-tracking condition |
| PFF FC World Cup 2022 | PFF FC's data release, under the terms accepted at download | PFF's event data, aligned to its tracking | 16 of 64 matches, picked by a fixed rule | test only: the "pff-16" condition |

No hand labels were used to train, tune or validate the submission. The video reviews from
before the rules mail (section 6) choose nothing in it. Model selection used 5-fold grouped
cross-validation over the 10 training matches, scored with the official `possession_metrics.py`,
under three conditions:

- **sim**: the simulator, on a fixed test draw;
- **shift**: a second draw with tracker faults never trained on (identity swaps, duplicate and
  phantom tracks);
- **sc**: SkillCorner as it is.

`mean3` is the mean of the three. pff-16 is a fourth, later check.

## 4. The possession rules implemented

The rules live in the labels the model learns from, plus a few hard constraints at decoding.

- **DFL.** The ball is dead where the provider's `BallStatus` is 0. Otherwise the label is the team
  in `BallPossession`. An owner flip shorter than the fold threshold, with the other team on both
  sides of it, is folded back: it is a touch, not control. The threshold is 10 frames for ENS5 and
  50 frames for ms50.
- **Metrica.** Each half is dead until its kickoff. A set piece restarts play for the team taking
  it. A ball out, a shot that goes out or in, an offside, a handball or a foul kills the ball. While
  the ball is alive, a possession event by the other side hands possession over: a pass, shot,
  recovery, loss or set piece.
- **Loose balls** stay with the last team in control until the other side makes a possession
  event. **Challenges, single touches, deflections and fouls received never confer possession**,
  which is the competition's "a touch without control is not a change".
- **Restarts.** The ball is dead from the stoppage to the restart. Metrica's sample games do not
  track the ball while it is dead, so the simulator fills that ball with a straight line from where
  it was last seen to the restart spot. Otherwise the model would learn "no ball = dead ball". At
  decoding, the second half's graded frames before its kickoff are `D` (FAQ), and frames outside
  the halves are `X`.
- **Decoder costs** make short flips and short stoppages expensive. The change head makes a switch
  cheaper where a change of possession is likely.

**Audit item 7, the exact scope of the 50-frame fold.** `--min-spell 50` folds only the DFL labels
(and the SkillCorner and PFF labels, which are not used in training). Metrica's event-derived
labels are never folded. So the five ms50 members saw 50-frame labels on the 7 DFL matches and
unchanged labels on the 3 Metrica games. The models were not retrained to change this.

## 5. Results on open data, and the label convention

Mean spellF1 at the submitted setting (TTA 0.5 + EATA 10, costs 4/8/8/1.0; plan rows LCB and
LCB-ADOPT). Each row is scored against two versions of the same open labels: the provider's
10-frame fold, and the same labels refolded at 50 frames.

| ensemble | labels | sim | shift | sc | pff-16 | mean3 |
|---|---|---|---|---|---|---|
| **submitted: 10 members (ENS5 + ms50)** | **10-frame** | **0.7493** | **0.7476** | **0.4177** | **0.5717** | **0.6382** |
| **submitted: 10 members (ENS5 + ms50)** | **50-frame** | **0.7832** | **0.7870** | **0.4594** | **0.6417** | **0.6765** |
| ENS5 only (the earlier 5-member file) | 10-frame | 0.7698 | 0.7694 | 0.4033 | 0.5748 | 0.6475 |
| ENS5 only | 50-frame | 0.7610 | 0.7658 | 0.4344 | 0.6365 | 0.6537 |
| ms50 only | 10-frame | 0.7120 | 0.7192 | 0.4260 | 0.5647 | 0.6191 |
| ms50 only | 50-frame | 0.7871 | 0.7964 | 0.4749 | 0.6391 | 0.6861 |

For scale, the official `baseline_smooth50` scored 0.291 under the first simulator. On the real
match the submission predicts 262 possessions, of mean length 13.5 s.

**Label convention.** CHALLENGE.md defines a change of possession as the opponent gaining
*controlled* possession. A touch, a deflection, or an aerial duel without control does not count,
and "keeps the ball for a few seconds" is one proxy it names. The provider's 10-frame spells
include short uncontrolled ones. The 50-frame refold is our own surrogate for "controlled".

- SkillCorner is real tracking whose labels have longer, controlled spells. It favours the
  controlled-trained models under both label versions: ms50 beats ENS5 by +0.0227 at 10 frames and
  +0.0405 at 50 frames.
- On mean3, the mix loses 0.0093 against ENS5 if the grader's convention is the 10-frame one, and
  gains 0.0228 if it is the 50-frame one. It therefore breaks even at
  P(50-frame convention) = 0.0093 / 0.0321 = **0.29**. The team's independent review estimated
  that probability at about **0.6**, and the team agreed.
- **+0.0228 is an upper bound**, because the 50-frame labels are our surrogate, not the grader's.
- DATA.md's public aggregate (mean possession 14.2 s in the public window) is consistent with long
  controlled spells. It was corroboration only, never the basis. No public-window score was
  computed for this decision.

**Noise floor.** The same recipe retrained with three seeds varies by sd 0.003. Scoring the same
models on two more simulated test draws (plan row DRAW2) gives a draw-to-draw sd of 0.003–0.004
for a difference between settings. Sim differences smaller than that are not evidence.

## 6. Rules compliance and disclosed choices

We follow the organisers' rules mail of 3 November 2026:

- **No hand labels and no human-made data on the provided game.** The only labels are the open
  datasets' own (section 3).
- **No checks against the video.** No output of the submitted pipeline was compared with the
  video.
- **Nothing tuned on the provided match.** Every setting was chosen on external data, each with a
  decision rule written before its run (`docs/plan_of_action.md`): the ensemble, the features, the
  decoder costs, the adaptation, the tag prior and the 10-member mix.
- **No match data sent to AI services.** Only aggregate numbers left the team's machine.

On the match itself, we used only automatic checks that read no ground truth: format validation,
label shares, the tags' agreement with the model's confident frames, and dead-ball shares around
set-piece tags. Earlier versions, before the mail, were also judged on the public window and on
video reviews of our output. Every setting chosen that way was withdrawn and re-decided on open
data. Appendix §7b records that history.

**Post-hoc choices, disclosed:**

- **The EATA tie-break (2026-11-07).** Two adaptation settings passed the pre-registered every-seed
  rule and tied within 0.0001 on mean3: TTA 0.5 alone (0.6476) and TTA 0.5 + EATA 10 (0.6475). The
  rule's tie-break picks TTA alone. The team chose TTA + EATA after seeing the results, for its
  larger smallest per-member gain (+0.0022 against +0.0001) and its larger real-tracking gain
  (SkillCorner +0.0122 against +0.0108).
- **The rule change before that study.** The earlier rule's +0.005 bar was dropped after a one-off
  sanity check of the 5-member ensemble on the public window. That check decided nothing.
- **LCB-ADOPT (2026-11-11).** The study that built the 10-member mix (row LCB) was registered as
  report-only. The switch was a separate decision, made on the definition text and the SkillCorner
  evidence in section 5, after the numbers were known. It was conditional on a byte-identical
  rebuild from a clean clone, which passed.
- **Restart-spot features.** They were designed after inspecting which real-match stoppages the
  model missed, which the rules allow. Their weights are learned on open data, and keeping them was
  re-decided on open data: 0.7716 against 0.7674 without them.

## 7. Limitations, and what we would do with more time

- **The label convention bounds every number.** On the 7 DFL matches we built a second labelling
  from DFL's own events with the Metrica rules (plan row LCEIL). Scored against the flag labels we
  train on, it reaches only 0.46 spellF1, with 81% frame agreement and 90% on alive versus dead.
  DFL has no ball-out event, so those ball-outs were inferred, and this overstates the gap somewhat.
  No open dataset pins the grader's convention. On PFF the published baseline order holds at every
  fold from 10 to 75 frames, so it does not single one out. The mix is a hedge between the two
  conventions, not a resolution.
- **The noise floor.** Sim differences under about 0.003–0.004 are within one test draw's noise
  (DRAW2). Several adopted settings, adaptation among them, won by less than that on sim. They were
  taken for their real-tracking (SkillCorner) gains.
- **No check on the real match.** The simulator matches published aggregates, not the joint
  structure of the real errors. A gain under it can fail to transfer, and the rules leave no way to
  check the submission on the match. The shift and SkillCorner conditions are the closest
  stand-ins; the competition's ATD is neither.
- **Short possessions** (under 2 s) are the weakest part. On open data the model predicts 85% of the
  true possession count, and most of the remaining headroom is in team calls, not decoding.
- **Dead-ball timing.** Dead-ball onsets may be called late and restarts early. This was seen on the
  public window before the rules mail, on few stoppages, and is not offset on open data.
- **Adaptation is one-sided.** Test-time adaptation re-centres the features on the home team's view
  only. A version pooling both views was studied and did not help (plan row TTA-2V).
- **Audit item 7.** The ms50 members' Metrica labels are unfolded (section 4). Their "controlled"
  training signal comes from the DFL matches only.
- **With more time:** hand-label a few other matches under the controlled-possession definition
  (allowed by the rules) to pin the convention; a stronger ball detector trained on other matches'
  video, to fill the ATD's gaps; and a fine-tune on the 48 PFF matches held back for it.

## 8. Reproduction

```bash
pip install -r requirements-tempolens.txt     # Python 3.11; pinned numpy, pandas, torch, numba, ...
# unpack the organisers' bundle into data/ (data/tactical/...)
python -m tempolens.predict --data-root data --output "Coding United.csv"
python evaluation/validate_submission.py "Coding United.csv"
```

Everything needed is committed: the 10 weight files, and `models/decoder.json`, which lists the
members and every setting. Inference runs on the CPU in float64 in a fixed order, so the same data
gives the same file byte for byte (sha256 above). A full run takes 14–24 minutes on a laptop CPU (Intel Core i7-12700H, 14 cores, 16 GB RAM) with the 10 members, TTA and EATA. The three measurements (822, 1356 and 1409 s, two of them from fresh clones) all had another job sharing the CPU, so an idle machine should be faster. A fresh clone of the
submitted commit rebuilt the file byte for byte and validated, and its 151 tests pass (`MTAH_E2E=1`, including the end-to-end test). Retraining
needs the open data and a GPU (`python -m tempolens.train --member k --draws 4 [--min-spell 50]`);
GPU training is not bit-reproducible, which is why the weights are committed.
