# Research Basis — TempoLens Strategy

Research date: 10 October 2026 (Asia/Kolkata).

## Primary sources

1. [More Than a Hack 2027 landing page](https://landing.mobileworldcapital.com/more-than-a-hack-2027)
2. [Official challenge repository](https://github.com/metrica-sports/more-than-a-hack-2027)
3. [Challenge definition](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/CHALLENGE.md)
4. [Data specification](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/DATA.md)
5. [Evaluation specification](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/EVALUATION.md)
6. [Competition rules](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/RULES.md)
7. [Submission contract](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/SUBMISSION.md)
8. [Talent Arena Call for Technologies](https://talentarena.tech/call-for-technologies-2027/)

## What the research establishes

### 1. This is a ranking problem before it is a product problem

The 2027 format deliberately replaced the prior open-ended, jury-evaluated hackathon with one
common task and a private ground truth. Phase 1 is automatic and reproducible: every team outputs a
single possession CSV and is ranked by `spellF1`. A polished dashboard, LLM use, or presentation
does not improve Phase-1 rank unless it improves that CSV.

**Strategy consequence:** invest first in temporal possession reconstruction; delay the product
experience until the team reaches Phase 2.

### 2. The metric rewards coherent intervals, not raw accuracy

`spellF1` treats each maximal home/away possession as one object, matches predicted and real spells
by temporal IoU, and averages F1 at IoU thresholds 0.3–0.7. It is the ranking metric. `macroF1`
is frame-level and only breaks ties.

The repository's own baselines prove the mismatch: on the private full match, the smoother
50-frame variant reaches `spellF1` 0.2433, while a raw nearest-player classifier has a materially
better-looking frame-level result than its 0.0486 spell score suggests. The public five-minute slice
can choose the wrong smoothing setting.

**Strategy consequence:** decode a sequence with explicit transition costs and suppress flicker;
never optimize a per-frame classifier in isolation.

### 3. Missing-ball tracking is the decisive technical challenge

The tactical ball track is missing in 41% of graded frames. Missingness is concentrated in hard
football moments: occlusions, aerial balls, tackles, and crowded boxes. The supplied tracking also
has gaps, fragmented tracklets, identity changes, and no player identities.

**Strategy consequence:** reduce each frame to team-level geometry rather than assume track slots
are players. During short ball absences, retain uncertainty and let temporal continuity preserve the
current spell; do not imitate the baseline's "missing ball = dead ball" shortcut.

### 4. Smart Tagging is valuable but must not be mistaken for truth

Smart Tagging provides 638 automated phase/set-piece intervals. It carries a team field in many
instances and identifies restarts, but it is generated automatically, contains errors, and its
intervals are not possession intervals. Throw-ins have `N/A` teams.

**Strategy consequence:** use tags as small additive priors, particularly at tactical-phase and
restart boundaries. Never assign labels by copying a tag interval.

### 5. A CPU-first hybrid is appropriate for this team

The tactical source is the only output timeline: 199,319 frames at 25 fps. Broadcast video is not
frame-synchronized and includes cuts/replays. The supplied tactical tracking and homography are
already aligned and usable on every frame.

**Strategy consequence:** do not spend early competition time rebuilding all tracking or aligning
the broadcast feed. Add local video evidence only at candidate transitions if it demonstrably helps.
This gives the team a realistic high-upside route without depending on GPU training.

### 6. Legality and reproducibility are gating constraints

Any method, external data, local/open/commercial model, and tracking replacement is allowed only if
the organizers can access/inspect it and the technical document explains it. Hand-producing the
submission labels is forbidden. The private ground truth is forbidden. The public repository must
regenerate the exact submitted file, while licensed source data must not be committed or
redistributed. Competition materials must not be uploaded to a hosted/cloud LLM.

**Strategy consequence:** TempoLens is deterministic, local, and standard-library/NumPy based.
Its report/dashboard contains only derived possession summaries. The repository `.gitignore` blocks
the data bundle and generated outputs.

## Why the Phase-2 concept follows naturally

Possession segmentation is a necessary first layer for meaningful football analytics. Once a spell
has a reliable start/end, the team can compute recoveries, losses, duration, location, team phase,
and post-recovery outcomes. A Transition Risk Atlas makes this directly useful to an analyst without
changing the Phase-1 objective.

## Talent Arena Call fit

The separate Call for Technologies explicitly welcomes functional AI/data-science experiences,
interactive data visualization, applied research, and socially/educationally meaningful technology.
It provides the exhibition structure, basic furniture, power, and internet, but not development or
production costs; a project representative must staff the stand during opening hours.

The Atlas can therefore become a visitor-facing interaction: choose a moment, see sparse/noisy
observations turn into a possession decision, then explore what the recovered transition means
tactically. This is credible only after the competition model works.
