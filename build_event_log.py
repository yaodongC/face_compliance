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

    # 2) operator RELOAD SESSIONS: the operator must enter the zone to reload - that
    #    is normal. Non-compliant ONLY if the boom was still operating at ENTRY.
    if Path(a.operator).exists():
        from operator_safety import classify_sessions
        ops = json.loads(Path(a.operator).read_text())["events"]
        for s in classify_sessions(ops):
            ev = f"data/operator_frames/op_{int(s['start']):05d}.png"
            if s["verdict"] == "NON_COMPLIANT_ENTRY":
                lg.log(EL.OPERATOR_IN_ZONE, s["end"], severity=EL.VIOLATION,
                       description=f"operator ENTERED danger zone while boom STILL OPERATING "
                                   f"(entry motion {s['entry_motion']:.3f}): {s['action']}",
                       bbox=s["person_bbox"], evidence=ev, source="operator_safety",
                       started_at=s["start"], duration_sec=round(s["end"] - s["start"], 1))
            else:
                lg.log(EL.SCREEN_INSTALLED, s["start"], severity=EL.INFO,
                       description=f"operator reloaded with boom STOPPED (compliant entry): {s['action']}",
                       bbox=s["person_bbox"], evidence=ev, source="operator_safety")

        # 3) face-SEGMENT coverage milestones (4 segments; per-mesh boxes are not
        #    reliable, so coverage is tracked as 4 coarse face quarters)
        from coverage import segment_coverage
        seg_times = segment_coverage(ops, n=4)
        for i, st in sorted(enumerate(seg_times), key=lambda x: (x[1] is None, x[1] or 0)):
            if st is not None:
                lg.log(EL.SCREEN_INSTALLED, st, severity=EL.INFO,
                       description=f"face segment Q{i+1} of 4 covered", source="coverage",
                       segment=i + 1)
        if all(st is not None for st in seg_times):
            lg.log(EL.COVERAGE_FULL, max(t for t in seg_times if t is not None),
                   severity=EL.INFO, description="entire face covered (all 4 segments)",
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
