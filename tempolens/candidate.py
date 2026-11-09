"""Candidate builds for plan rows TS and B2 (outside git; models/decoder.json is never changed).

    python -m tempolens.candidate ts|b2|combo --data-root <bundle>

Each reads its row's verdict and stops with exit 3 if the rule did not pass (combo: both). Otherwise
it writes a temporary decoder file to derived/<name>/, runs predict twice, checks the two files are
byte-identical, validates the format, runs the full tests with MTAH_E2E=1, and prints the sha,
validity, test counts and the possession count (H/A runs) -- nothing else of the output.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from .experiments import write_atomic
from .predict import DEFAULT_DECODER

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "experiments" / "logs" / "nov07"      # the lanes' logs
TS_VERDICT = ROOT / "models" / "exp" / "ts" / "verdict.json"
B2_LINE = re.compile(r"^B2: passes: (blend tta\+eata|blend tta) at \(([^)]*)\)")
BLEND = {"blend tta": (0.5, 0), "blend tta+eata": (0.5, 10)}


def stop(message):
    print("STOPPED: " + message, flush=True)
    sys.exit(3)


def ts_settings(path=TS_VERDICT):
    """The decoder settings plan row TS adopts, or None if it adopted nothing."""
    if not path.exists():
        return None
    chosen = json.loads(path.read_text(encoding="utf-8"))["chosen"]
    return None if chosen is None else {"tag_shrink": float(chosen)}


def b2_settings(logs=LOGS):
    """The decoder settings plan row B2 adopts, from the lane's latest try that reached its
    verdict (b2.log is try 1, b2.<n>.log try n); None if it failed or never finished."""
    tries = {}
    for p in logs.glob("b2*.log"):
        m = re.fullmatch(r"b2(?:\.(\d+))?\.log", p.name)
        if m:
            tries[int(m.group(1) or 1)] = p
    for _, path in sorted(tries.items(), reverse=True):
        lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.startswith("B2: ")]
        if not lines:
            continue
        m = B2_LINE.match(lines[-1])
        if not m:
            return None
        alpha, steps = BLEND[m.group(1)]
        switch, to_dead, to_alive, change = (float(x) for x in m.group(2).split(","))
        return {"blend": True, "tta_alpha": alpha, "eata_steps": steps, "switch": switch,
                "alive_to_dead": to_dead, "dead_to_alive": to_alive, "change_weight": change}
    return None


def possessions(path):
    sys.path.insert(0, str(ROOT / "evaluation"))
    import possession_metrics as pm
    codes = pm.label_codes(pm.read_label_csv(path))
    alive = np.isin(codes, (0, 1))                    # H, A
    return int((alive & np.concatenate(([True], codes[1:] != codes[:-1]))).sum())


def build(name, overrides, data_root):
    out = ROOT / "derived" / name                     # git-ignored
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads(DEFAULT_DECODER.read_text(encoding="utf-8"))
    config.update(overrides)
    config["_provenance"] = ("TEMPORARY candidate {} ({}), not the submission; models/decoder.json "
                             "is unchanged.".format(name, json.dumps(overrides)))
    decoder = out / "decoder_{}.json".format(name)
    write_atomic(decoder, json.dumps(config, indent=2) + "\n")
    files = []
    for i in (1, 2):
        csv = out / "candidate_{}.csv".format(i)
        with open(out / "predict_{}.log".format(i), "w", encoding="utf-8") as log:
            subprocess.run([sys.executable, "-m", "tempolens.predict", "--data-root", data_root,
                            "--decoder", str(decoder), "--output", str(csv)],
                           check=True, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        files.append(csv)
    sha = [hashlib.sha256(f.read_bytes()).hexdigest()[:16] for f in files]
    valid = subprocess.run([sys.executable, str(ROOT / "evaluation" / "validate_submission.py"),
                            str(files[0])], capture_output=True, text=True, cwd=ROOT).returncode == 0
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                            "tests"], cwd=ROOT, capture_output=True, text=True,
                           env=dict(os.environ, MTAH_E2E="1"))
    summary = (tests.stdout.strip().splitlines() or ["(no output)"])[-1]
    report = {"name": name, "overrides": overrides, "sha": sha[0], "identical": sha[0] == sha[1],
              "valid": valid, "tests": summary, "tests_exit": tests.returncode,
              "possessions": possessions(files[0])}
    write_atomic(out / "report.json", json.dumps(report, indent=1))
    print("candidate {} {}: sha {}, byte-identical on a second run {}, valid {}, possessions {}; "
          "tests: {} (exit {})".format(name, json.dumps(overrides), report["sha"],
                                       report["identical"], valid, report["possessions"], summary,
                                       tests.returncode), flush=True)
    return 0 if report["identical"] and valid and tests.returncode == 0 else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("row", choices=("ts", "b2", "combo"))
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args(argv)
    ts, b2 = ts_settings(), b2_settings()
    if args.row == "ts" and ts is None:
        stop("TS adopted no shrink")
    if args.row == "b2" and b2 is None:
        stop("B2 did not pass (or has no verdict)")
    if args.row == "combo" and (ts is None or b2 is None):
        stop("the combination needs both TS and B2 to pass")
    overrides = {"ts": ts, "b2": b2}.get(args.row) or dict(b2, **ts)
    return build(args.row, overrides, args.data_root)


if __name__ == "__main__":
    sys.exit(main())
