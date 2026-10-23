"""Run queued experiments one after another, unattended and offline, and record the results.

    python -m tempolens.autoqueue           # loop: run pending jobs; when idle, re-check every 5 min
    python -m tempolens.autoqueue --once    # run what is pending, then exit

The queue is experiments/queue.json, a list of jobs:

    {"id": "Q2", "title": "focal loss gamma 2", "args": ["--model", "temporal", "--focal", "2"],
     "baseline": {"all": 0.7543, "dfl": 0.7456} | null, "baseline_from": "<job id>" | null}

Each job runs `python -m tempolens.experiments <args>` with its log in experiments/logs/<id>.log.
A job may instead give "cmd" (arguments after `python`, e.g. ["-m", "tempolens.train"] or a script
path), "cwd" (a branch worktree) and "env" (extra variables); such a job is done when it exits 0.
Its best "mean A (dfl D, metrica M)  costs (...)" line is parsed into the job, compared with the
baseline, and the results table in docs/plan_of_action.md (between the autoqueue markers) is
rewritten. Before starting a job the runner waits while any other tempolens.experiments or
tempolens.train process is running, so manual runs and the queue never share the GPU. A failed
job is marked and skipped. Create experiments/STOP to end the loop after the current job.
Nothing is committed or pushed.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "experiments" / "queue.json"
LOGS = ROOT / "experiments" / "logs"
STOP = ROOT / "experiments" / "STOP"
PLAN = ROOT / "docs" / "plan_of_action.md"
START, END = "<!-- autoqueue:start -->", "<!-- autoqueue:end -->"
NOISE = 0.005        # a gain in mean spellF1 below this is noise until the seed check says more
IDLE_SECONDS = 300
RESULT = re.compile(r"^mean ([0-9.]+) \(dfl ([0-9.]+), metrica ([0-9.]+)\)\s+costs (\(.*\))")


def load():
    return json.loads(QUEUE.read_text(encoding="utf-8")) if QUEUE.exists() else []


def save(jobs):
    tmp = QUEUE.with_suffix(".tmp")
    tmp.write_text(json.dumps(jobs, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, QUEUE)


def now():
    return datetime.now().strftime("%H:%M")


def python_processes():
    """Command lines of all running python processes (works offline)."""
    if os.name == "nt":
        cmd = ["powershell", "-NoProfile", "-Command",
               "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
               "ForEach-Object { $_.CommandLine }"]
    else:
        cmd = ["ps", "-eo", "args"]
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.splitlines()
    except (OSError, subprocess.SubprocessError):
        return []


def other_runs():
    """Running tempolens.experiments / tempolens.train processes."""
    return [line for line in python_processes()
            if "tempolens.experiments" in line or "tempolens.train" in line]


def parse(log_text):
    """Best result of an experiments run: the first 'mean ...' line after its header."""
    for line in log_text.splitlines():
        m = RESULT.match(line.strip())
        if m:
            return {"all": float(m.group(1)), "dfl": float(m.group(2)),
                    "metrica": float(m.group(3)), "costs": m.group(4)}
    return None


def baseline_of(job, jobs):
    if job.get("baseline_from"):
        ref = next((j for j in jobs if j["id"] == job["baseline_from"]), None)
        return ref.get("result") if ref else None
    return job.get("baseline")


def verdict(job, jobs):
    result, base = job.get("result"), baseline_of(job, jobs)
    if job["status"] != "done" or not result:
        return ""
    if not base:
        return "measured"
    d_all, d_dfl = result["all"] - base["all"], result["dfl"] - base["dfl"]
    if d_all >= NOISE and d_dfl >= -0.002:
        return "candidate (+{:.4f})".format(d_all)
    return "no gain ({:+.4f})".format(d_all)


def write_plan(jobs):
    mark = {"pending": "⬜", "running": "⏳", "done": "✅", "failed": "❌"}
    rows = ["| | id | experiment | all | DFL | metrica | best costs | vs baseline | finished |",
            "|---|---|---|---|---|---|---|---|---|"]
    for job in jobs:
        r = job.get("result") or {}
        rows.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            mark.get(job["status"], "?"), job["id"], job["title"],
            "{:.4f}".format(r["all"]) if r else "", "{:.4f}".format(r["dfl"]) if r else "",
            "{:.4f}".format(r["metrica"]) if r else "", r.get("costs", ""),
            verdict(job, jobs), job.get("finished", "")))
    table = "\n".join(rows)
    note = ("Written by `python -m tempolens.autoqueue` from `experiments/queue.json` "
            "(✅ here = finished; adoption still follows the decision rules above).")
    block = "{}\n{}\n\n{}\n{}".format(START, note, table, END)
    text = PLAN.read_text(encoding="utf-8")
    if START in text and END in text:
        text = text[:text.index(START)] + block + text[text.index(END) + len(END):]
    else:
        text = text.rstrip() + "\n\n## F. Automatic experiment queue\n\n" + block + "\n"
    tmp = PLAN.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, PLAN)


def run_job(job, jobs):
    while other_runs():
        print("{}  waiting: another training/CV run is active".format(now()), flush=True)
        time.sleep(IDLE_SECONDS)
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / "{}.log".format(job["id"])
    job.update(status="running", started=datetime.now().strftime("%Y-%m-%d %H:%M"))
    save(jobs)
    write_plan(jobs)
    cmd = job.get("cmd") or ["-m", "tempolens.experiments"] + job["args"]
    print("{}  start {}: {}".format(now(), job["id"], " ".join(cmd)), flush=True)
    with log.open("w", encoding="utf-8") as handle:
        code = subprocess.call([sys.executable, "-u", "-W", "ignore"] + cmd,
                               cwd=job.get("cwd") or ROOT, env=dict(os.environ, **job.get("env", {})),
                               stdout=handle, stderr=subprocess.STDOUT)
    jobs = load()                       # the queue may have been edited meanwhile
    job = next(j for j in jobs if j["id"] == job["id"])
    result = parse(log.read_text(encoding="utf-8", errors="replace"))
    finished_ok = code == 0 and (result or job.get("cmd"))
    job.update(status="done" if finished_ok else "failed", result=result,
               finished=datetime.now().strftime("%Y-%m-%d %H:%M"), exit_code=code)
    save(jobs)
    write_plan(jobs)
    print("{}  {} {}: {}".format(now(), job["status"], job["id"], verdict(job, jobs)), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--once", action="store_true", help="exit when nothing is pending")
    args = parser.parse_args(argv)
    if sum("tempolens.autoqueue" in line for line in python_processes()) > 1:
        print("another autoqueue is already running; exiting")
        return
    while not STOP.exists():
        jobs = load()
        for job in jobs:
            if job["status"] == "running":      # interrupted earlier (reboot, crash): run again
                job["status"] = "pending"
        pending = [j for j in jobs if j["status"] == "pending"]
        if pending:
            run_job(pending[0], jobs)
            continue
        write_plan(jobs)
        if args.once:
            break
        time.sleep(IDLE_SECONDS)
    print("autoqueue stopped", flush=True)


if __name__ == "__main__":
    main()
