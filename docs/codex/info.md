# More Than a Hack 2027 — Complete Working Information

## Event at a glance

| Item | Detail |
|---|---|
| Competition | More Than a Hack 2027 — Phase 1: Ball Possession |
| Organizers | Metrica Sports, FC Barcelona / Barça Innovation Hub, Mobile World Capital Barcelona |
| Audience | Football analysts, data scientists, engineers, developers, tech enthusiasts |
| Event context | Talent Arena, Barcelona |
| Team size | 3–5 members required for Innovation Challenge and Final Pitch eligibility |
| Phase-1 registration | 1 October–10 November 2026 |
| Phase-1 data release | 20 October 2026 |
| Phase-1 deadline | 17 November 2026 |
| Top-20 announcement | Before 26 November 2026 |
| Innovation Challenge | 26 November 2026–14 January 2027 |
| Final pitch | 2 March 2027, on-site at Talent Arena |
| Top-5 travel/accommodation | Covered for final pitch according to the landing page |

Authoritative sources: [landing page](https://landing.mobileworldcapital.com/more-than-a-hack-2027),
[repository](https://github.com/metrica-sports/more-than-a-hack-2027).

## The exact Phase-1 task

For every frame of the **tactical** recording of FC Barcelona vs Athletic Club (LaLiga 2024–25,
matchday 2), output one label:

| Label | Meaning |
|---|---|
| `H` | Barcelona/home team has the ball while alive |
| `A` | Athletic Club/away team has the ball while alive |
| `D` | Ball is dead/not in play |
| `X` | Abstain/no answer; valid but handled differently by each metric |

The delivery is a UTF-8 CSV with exactly 199,319 data rows:

```csv
frame,label
0,X
1,X
...
199318,X
```

`frame` must be precisely `0…199318`, in order. Labels may only be `H`, `A`, `D`, or `X`.
Run `python evaluation/validate_submission.py <file>.csv` before upload.

## Dataset

### Tactical reference source

| Property | Detail |
|---|---|
| Frames / FPS | 199,319 / 25 fps |
| Resolution | 1920 × 1080 |
| Submission alignment | Output row `i` is tactical video frame `i` |
| Tracking | 199,319 records; player/ball positions and speed |
| Homography | 199,319 rows; 100% usable |
| Smart Tagging | 638 automatic tactical annotations |
| Ball visible | 56.9% overall; 59.3% of graded frames |

### Broadcast source

The broadcast video has 164,488 frames, its own tracking and homography, camera cuts/replays, and
only 65.8% usable homography. It is **not synchronized** to the tactical timeline. It is permitted
but optional; a submission must still be tactical-indexed.

### Tracking caveats

- 46 track slots represent temporary tracklets, not persistent player IDs.
- Player positions, ball positions, and speeds are normalized; pitch size is 105.3 × 68.0 metres.
- The raw tracking counter is one-indexed; submission frames are zero-indexed:
  `video_frame = frame_id - 1`.
- The ball is missing frequently; missing values are `NaN`.

### Smart Tagging details

It identifies build-up, progression, transitions, set pieces, long balls, final-third/box activity,
defensive phases, and goals. Tags include team, half, set-piece type, side, and box-player count.
They are automatic predictions, not ground truth. Time is measured in seconds from tactical-video
start: `round(seconds * 25)` gives video-frame index.

## Graded timeline

| Segment | Frame range | Scored? |
|---|---:|---|
| Before match | 0–11,184 | No |
| First half | 11,185–90,651 | Yes |
| Half-time | 90,652–110,807 | No |
| Second-half graded window | 110,808–190,197 | Yes |
| After match | 190,198–199,318 | No |

Important: second-half grading begins at frame 110,808, but the second-half kickoff is frame
113,742. The preceding graded frames are dead ball.

## Evaluation

### Ranking metric: `spellF1`

Each live `H` or `A` run is a possession spell. Predicted and true spells are matched one-to-one by
temporal IoU. F1 is computed at IoU thresholds 0.3, 0.4, 0.5, 0.6, and 0.7, then averaged. This
metric determines the ranking.

- A one-frame offset costs nothing at the published illustration; a one-second error costs greatly.
- Constant/random answers score 0.0.
- Small spells/flicker are harmful because they create spurious possessions.
- Published full-match baseline to beat: `baseline_smooth50`, `spellF1 = 0.2433`.

### Tie-break: `macroF1`

The unweighted mean of frame-level F1 for `H`, `A`, and `D`. It is not the leaderboard metric but
is the first tie-break.

### How `X` is treated

- On ground-truth-ungraded frames, `X` is ignored.
- For `macroF1`, `X` on a graded frame is a miss.
- For `spellF1`, the scorer fills `X` with the previous non-`X` label in the same half. An `X` in
  the middle of a true spell is safer than a flicker; an `X` across a possession change can be bad.

### Public ground truth

The repository includes labels only for the first five minutes of play: 7,500 graded frames,
frames 11,185–18,684. It has 17 possessions with mean duration 14.2 seconds. It is a pipeline
check, not a reliable leaderboard: even the supplied baselines change order between that window and
the private full match.

## Required deliverables

1. Possession CSV.
2. Public code repository with source, pinned dependencies, setup/run directions, and the exact
   command needed to regenerate the CSV from the official participant bundle.
3. Short technical document (PDF or Markdown) describing approach, assets used/dropped, possession
   rules, limitations, and reproduction.

The submission form is emailed to registered participants. It requires a Google account for uploads;
only the last team response before the deadline is scored.

## Rules and safety

### Allowed

- Any inspectable method, language, framework, public/commercial/open model, or external data.
- Improving/replacing supplied tracking.
- Small human-labeled data for training or validation.

### Forbidden or required

- Do not hand-label or hand-correct the submitted possession file.
- Do not use private/full-match ground truth or anything derived from it.
- Do not submit output that the public repository cannot reproduce exactly.
- Do not commit/distribute match video, tracking exports, clipped video, or reconstructable derived
  data. The data license is separate from the code license.
- Do not upload competition datasets, videos, confidential information, or extracts to hosted/cloud
  LLMs.

## TempoLens commands

```powershell
pip install -r requirements-tempolens.txt
python -m tempolens.predict --data-root D:\mtah2027 --output evaluation\results\tempolens.csv --report-dir tempolens\output
python evaluation\validate_submission.py evaluation\results\tempolens.csv
python evaluation\run_evaluation.py --results evaluation\results
python -m tempolens.diagnostics --data-root D:\mtah2027
python -m tempolens.dashboard --report tempolens\output\transition_report.json --output tempolens\output\atlas.html
```

Optional CPU-only scene-change cues:

```powershell
pip install -r requirements-video.txt
python -m tempolens.predict --data-root D:\mtah2027 --output evaluation\results\tempolens.csv --video-cues
```

## Separate Talent Arena Call for Technologies

This is different from the hackathon. It seeks functional interactive technology experiences,
including AI/XR/data-science installations and real-time visualization. Selected exhibitors receive
the exhibition space structure, basic furniture, internet, and power, but pay their own development
and production costs. At least one project representative must staff the stand for all opening
hours. Talent Arena 2027 runs 1–3 March in Barcelona.

Best use of this call: submit the validated Transition Risk Atlas as an interactive applied-AI
experience after the Phase-1 solution is working.
