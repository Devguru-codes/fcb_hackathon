# TempoLens — Winning Plan for More Than a Hack 2027

## Objective

Win the **More Than a Hack 2027** football-data competition by placing in the top 20 in Phase 1,
then turn the core possession engine into a compelling transition-analysis product for the
Innovation Challenge and the Barcelona final.

The immediate scoring goal is not a presentation or dashboard. Phase 1 is ranked automatically on
`spellF1`: the quality of recovered whole possession spells. `macroF1` is only the first tie-break.

## Chosen product direction

**TempoLens** is an uncertainty-aware, CPU-first system that answers, for every tactical-video
frame, whether Barcelona (`H`), Athletic Club (`A`), or neither (`D`) has possession.

It uses a hybrid, competition-first approach:

1. Automatic Tracking Data is the backbone: nearest team-to-ball distance, control density,
   visibility, and lengths of missing-ball intervals.
2. Smart Tagging supplies only weak tactical priors for team phases and restarts. It is never
   copied as an answer.
3. A three-state temporal decoder uses transition penalties and run-aware smoothing to recover
   complete possessions rather than make isolated frame decisions.
4. Optional local CPU-only video scene-change evidence can verify a limited set of likely
   boundaries. It does not upload source assets and does not replace tracking.

## Implementation plan

### Phase 0 — now through data release

- Keep a 3–5 person team; this is mandatory for the Innovation Challenge and final pitch.
- Register before **10 November 2026** and accept the data-use terms.
- Install the pinned TempoLens environment and run the supplied scorer/baselines before the gated
  assets arrive.
- Agree one shared operating rule: no competition video, tracking export, or derived frame-level
  asset goes into a public repository or hosted AI service.

### Phase 1 — possession ranking

- Download the participant bundle, preserving its `tactical/` and `broadcast/` directory layout.
- Run the deterministic pipeline:

  ```powershell
  python -m tempolens.predict --data-root D:\mtah2027 --output evaluation\results\tempolens.csv --report-dir tempolens\output
  python evaluation\validate_submission.py evaluation\results\tempolens.csv
  python evaluation\run_evaluation.py --results evaluation\results
  ```

- Diagnose before tuning: confirm 199,319 output rows, exact tactical-frame alignment, team mapping,
  missing-ball behavior, and the expected `X` labels outside graded match blocks.
- Compare against the supplied `baseline_smooth50` floor. Use the five-minute public ground truth
  only to validate pipeline behavior; it is too small to choose production hyperparameters.
- Run ablations: geometry only, geometry plus tags, and optional local video cues. Keep the simplest
  approach that is stable under synthetic extra ball occlusion.
- Submit the exact validated CSV, public source repository, and technical document by
  **17 November 2026**.

### Phase 2 — Transition Risk Atlas

If TempoLens reaches the top 20, use the provided ten-match dataset to create the **Transition Risk
Atlas**:

- show recovered possessions and their confidence/ball-visibility context;
- identify possession recoveries and losses;
- summarize five-second transition outcomes by pitch zone and tactical phase;
- let an analyst compare where a team escapes pressure or creates danger after recoveries.

The local `atlas.html` preview is a foundation for this work. It contains only derived summaries,
not licensed match video or raw tracking.

## Roles

| Role | Primary responsibility |
|---|---|
| Data/model lead | Features, decoder behavior, ablations, evaluation discipline |
| Data engineer | Reliable ingestion, frame alignment, reproducible submission command |
| Football analyst | Operational definition of control, failure review, Phase-2 insight framing |
| Product/story lead | Technical document, Atlas interaction design, final pitch |

One person can own more than one role, but all roles need an explicit owner.

## Definition of done

- `python -m tempolens.predict` regenerates the submitted CSV from a clean checkout plus the
  participant data bundle.
- The official validator accepts that exact file.
- The source is deterministic, dependency-pinned, and contains no restricted competition assets.
- The technical document describes the method, assets used/dropped, possession rules, limitations,
  and reproduction command.
- Final model selection is supported by stability/ablation evidence, not the public five-minute
  score alone.

## Separate Talent Arena exhibition call

The Call for Technologies is a distinct selection process. Reuse the Phase-2 Atlas as an interactive
demonstration only after the core competition system is proven. It has separate production costs,
selection, and in-person staffing requirements.
