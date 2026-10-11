"""A video ball detector that fills the ATD's ball gaps, trained on labels the ATD itself provides.

Wherever the automatic tracking sees the ball, its pitch position projected through that frame's
homography is the ball's pixel position: ~113k algorithmically labelled frames of exactly this
camera. A small heatmap network learns from them and is applied where the tracking lost the ball.
Everything runs locally; nothing derived from the video leaves the machine.
"""
