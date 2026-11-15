# Questions for the organisers (answered; never posted)

**Status (2026-11-07).** This draft was never posted: the challenge repository's issues are
now off. The organisers' Q&A answers #2, #10, #11 and #16 (as quoted in 7dracoder's
`CoreOrigin.md`, lines 506–513) cover questions 2 and 3 below. Question 1, the licence of
Metrica's sample data, stays open: the data is used with attribution, as its repository asks
(README.md, "Open data"; docs/Coding United.md, section 3). The submission uses no component trained on the provided video
(question 3 and the last section): the video ball detector was built and tested before the rules mail, never used, and its code
has been removed (technical document §6b). The
text below is the original draft, kept as history.

RULES.md asks participants to ask before building on anything near a line. Three questions:

**Title:** Clarifications: open datasets for training, committed model weights, Smart Tagging

1. **Metrica's own sample data for training.** We would like to train on Metrica's public sample
   games (github.com/metrica-sports/sample-data) alongside the CC-BY 4.0 Sportec/DFL open dataset
   (Bassek et al., 2025). The sample-data repository names no licence and asks for responsible
   use with attribution. Is using it to train a model for this competition acceptable, with
   attribution in the technical document?

2. **Reproduction with committed weights.** Our model is trained only on open data (no competition
   asset). GPU training is not bit-reproducible, so the repository commits the trained weights
   (a few MB) and the submission command runs CPU inference from them, which reproduces the
   submitted file exactly; the training command is documented too. Does "running your repository
   against the provided data must reproduce your submission" accept committed weights in this
   way?

3. **Weights derived from the provided video (only if used).** If a component is trained on the
   provided match itself — for example a ball detector whose labels are the ATD's own ball
   positions projected through the provided homography (no hand labelling) — may its weights be
   committed to the public repository, or should the repository instead retrain it from the data
   at reproduction time? (Retraining on a GPU is not bit-reproducible, so only committed weights
   let the submission file be reproduced exactly; without them we would leave the component out.)

Thank you!

## Model weights trained on the tactical video

A ball detector trained on crops of the tactical video (with labels projected from the ATD, no hand
labelling) would need its weights in the repository for the submission to be regenerated. May
weights trained on the licensed video be committed to the public repository? If not, is a
submission whose pipeline trains such a detector during the run acceptable?
