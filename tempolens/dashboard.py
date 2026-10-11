"""Build a self-contained local Phase-2 Transition Risk Atlas preview."""

import argparse
import html
import json
from pathlib import Path


HTML = """<!doctype html><html lang=\"en\"><meta charset=\"utf-8\"><title>TempoLens — Transition Risk Atlas</title>
<style>body{background:#09141d;color:#e8f0f4;font:16px system-ui;margin:2rem}h1{color:#65dfb1}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1rem}.card{background:#122532;padding:1rem;border-radius:.6rem}.bar{height:12px;background:#203a49;border-radius:6px}.fill{height:100%;background:#65dfb1;border-radius:6px}table{border-collapse:collapse;width:100%;margin-top:2rem}td,th{padding:.55rem;border-bottom:1px solid #28404d;text-align:left}</style>
<h1>TempoLens <small>Transition Risk Atlas</small></h1><p>Local analytical preview — source match assets are not embedded or uploaded.</p><div class=\"grid\" id=\"stats\"></div><table><thead><tr><th>Team</th><th>Start</th><th>End</th><th>Duration</th><th>Ball observed</th></tr></thead><tbody id=\"rows\"></tbody></table>
<script>const report=__REPORT__;const spells=report.spells;const total=spells.reduce((n,s)=>n+s.duration_s,0);document.querySelector('#stats').innerHTML=`<div class=card><b>${spells.length}</b><br>recovered possessions</div><div class=card><b>${total.toFixed(1)} s</b><br>live possession time</div><div class=card><b>${(spells.reduce((n,s)=>n+s.ball_observed_fraction,0)/Math.max(spells.length,1)*100).toFixed(0)}%</b><br>mean ball visibility</div>`;document.querySelector('#rows').innerHTML=spells.map(s=>`<tr><td>${s.team}</td><td>${s.start_frame}</td><td>${s.end_frame}</td><td><div class=bar><div class=fill style=\"width:${Math.min(100,s.duration_s/40*100)}%\"></div></div>${s.duration_s}s</td><td>${(s.ball_observed_fraction*100).toFixed(0)}%</td></tr>`).join('');</script>"""


def build(report_path, output):
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    payload = json.dumps(report).replace("</", "<\\/")
    Path(output).write_text(HTML.replace("__REPORT__", payload), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    build(args.report, args.output)
    print("wrote {}".format(args.output))


if __name__ == "__main__":
    main()
