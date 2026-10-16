# Questions for the organisers (draft — post as a GitHub issue on metrica-sports/more-than-a-hack-2027)

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
   at reproduction time?

Thank you!
