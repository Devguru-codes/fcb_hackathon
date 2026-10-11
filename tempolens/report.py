"""Portable transition report used by the Phase-2 dashboard foundation."""

import csv
import json
from pathlib import Path

import numpy as np

from .decoder import LABELS


def spells_from_states(states, features, fps=25):
    starts = np.r_[0, np.flatnonzero(states[1:] != states[:-1]) + 1]
    ends = np.r_[starts[1:], len(states)]
    rows = []
    for start, end in zip(starts, ends):
        state = int(states[start])
        if state == 2:
            continue
        observed = features.ball_seen[start:end]
        xs, ys = features.ball_x[start:end][observed], features.ball_y[start:end][observed]
        rows.append({
            "start_frame": int(start), "end_frame": int(end - 1),
            "team": str(LABELS[state]), "duration_s": round((end - start) / fps, 3),
            "ball_observed_fraction": round(float(np.mean(observed)), 3),
            "mean_ball_x_m": round(float(np.mean(xs)), 3) if len(xs) else None,
            "mean_ball_y_m": round(float(np.mean(ys)), 3) if len(ys) else None,
        })
    return rows


def write_report(output_dir, states, features, fps=25):
    """Write JSON and CSV reports without copying any source tracking/video data."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    spells = spells_from_states(states, features, fps)
    (output / "transition_report.json").write_text(
        json.dumps({"schema_version": 1, "fps": fps, "spells": spells}, indent=2) + "\n",
        encoding="utf-8")
    with (output / "possessions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(spells[0]) if spells else ["start_frame", "end_frame", "team", "duration_s", "ball_observed_fraction", "mean_ball_x_m", "mean_ball_y_m"])
        writer.writeheader()
        writer.writerows(spells)
    return spells
