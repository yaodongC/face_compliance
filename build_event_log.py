"""Populate the external-memory event log from the harness outputs.

Turns the (stateless, per-window) perception/verdict stream + operator events into
a durable, debounced INCIDENT TIMELINE: state changes, screen installs, coverage
milestones, operator-in-danger-zone violations and near-misses. This is the record
a compliance audit / incident review reads.

Usage: python3 build_event_log.py [--analysis data/full_cycle_analysis.json]
       [--index data/full_cycle.idx] [--operator data/operator_events.json]
       [--out data/event_log.jsonl]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import event_log as EL
from coverage import build_coverage

# verdict -> (severity, is it the safe/compliant state)
VERDICT_SEV = {"SUPPORTED": EL.INFO, "DRILLING": EL.INFO, "NOT VERIFIED": EL.WARNING,
               "UNSUPPORTED": EL.WARNING, "DANGER": EL.VIOLATION}


def _cycle_mapper(index_path):
    if not index_path or not Path(index_path).exists():
        return lambda t: t
    idx = {int(f): float(c) for f, c in
           (l.split(",") for l in Path(index_path).read_text().splitlines()[1:])}
    keys = sorted(idx)

    def m(tsec):
        fr = int(round(tsec * 15))
        k = min(keys, key=lambda x: abs(x - fr))
        return idx[k]
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default="data/full_cycle_analysis.json")
    ap.add_argument("--index", default="data/full_cycle.idx")
    ap.add_argument("--operator", default="data/operator_events.json")
    ap.add_argument("--out", default="data/event_log.jsonl")
    a = ap.parse_args()
    lg = EL.EventLogger(a.out, reset=True)
    cyc = _cycle_mapper(a.index)

    # 1) verdict-state timeline -> state_change + face_supported milestones
    if Path(a.analysis).exists():
        steps = sorted(json.loads(Path(a.analysis).read_text())["steps"], key=lambda s: s["t_sec"])
        prev = None
        for s in steps:
            v = s["verdict"]
            if v != prev:
                cs = cyc(s["t_sec"])
                lg.log(EL.STATE_CHANGE, cs, severity=VERDICT_SEV.get(v, EL.INFO),
                       description=f"verdict -> {v}", source="face_harness", verdict=v)
                if v == "SUPPORTED":
                    lg.log(EL.FACE_SUPPORTED, cs, severity=EL.INFO,
                           description="face screened + booms parked (supported state)",
                           source="face_harness")
                prev = v

    # 2) operator events -> danger incidents, near-misses, screen installs
    if Path(a.operator).exists():
        ops = json.loads(Path(a.operator).read_text())["events"]
        danger = EL.IncidentDebouncer(lg, EL.OPERATOR_IN_ZONE, EL.VIOLATION,
                                      source="operator_safety", min_frames=1)
        for e in sorted(ops, key=lambda x: x["cycle_sec"]):
            cs = e["cycle_sec"]
            is_danger = e["verdict"] == "DANGER"
            danger.update(is_danger, cs,
                          description=f"operator in front while boom MOVING "
                                      f"(motion {e.get('arm_motion')}): {e.get('action','')}",
                          bbox=e.get("person_bbox"),
                          evidence=f"data/operator_frames/op_{int(cs):05d}.png")
            if e["verdict"] == "OK_LOADING" and e.get("person_bbox"):
                act = (e.get("action") or "").lower()
                if any(k in act for k in ("screen", "mesh", "bolt", "fit", "load", "install")):
                    lg.log(EL.SCREEN_INSTALLED, cs, severity=EL.INFO,
                           description=f"operator installing (drill stopped): {e.get('action','')}",
                           bbox=e.get("person_bbox"), source="operator_safety",
                           evidence=f"data/operator_frames/op_{int(cs):05d}.png")

        # 3) coverage milestones from accumulated install sites
        prog, covered = build_coverage([e for e in ops if e.get("person_bbox")], cols=10)
        last_cov = 0.0
        for p in prog:
            if p["coverage"] > last_cov:
                lg.log(EL.SCREEN_INSTALLED, p["cycle_sec"], severity=EL.INFO,
                       description=f"face coverage advanced to {p['coverage']*100:.0f}%",
                       source="coverage", coverage=p["coverage"])
                last_cov = p["coverage"]
        if covered and sum(covered) / len(covered) >= 0.9:
            lg.log(EL.COVERAGE_FULL, prog[-1]["cycle_sec"], severity=EL.INFO,
                   description="face fully covered by overlapping screens (compliant coverage)",
                   source="coverage")

    # report
    s = lg.summary()
    print(f"=== EVENT LOG written to {a.out} ===")
    print("incidents:", s["total"], "| by severity:", s["by_severity"],
          "| violations:", s["violations"], "| near-misses:", s["near_misses"])
    print("\nTimeline:")
    for e in lg.events():
        cs = int(e["cycle_sec"])
        print(f"  [{e['severity']:9}] {cs//60:02d}:{cs%60:02d} {e['type']:22} {e['description'][:62]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
