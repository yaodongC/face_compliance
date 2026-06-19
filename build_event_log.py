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

    # danger-zone ENTRIES (dense classical orange detection; classification
    #    reconciled with the VLM scan). The operator must enter to reload - that is
    #    normal; non-compliant ONLY when the boom was still moving at entry.
    ep = Path("data/operator_entries.json")
    if ep.exists():
        for e in json.loads(ep.read_text())["entries"]:
            if e["verdict"] == "NON_COMPLIANT_ENTRY":
                lg.log(EL.OPERATOR_IN_ZONE, e.get("end", e["time"]), severity=EL.VIOLATION,
                       description=f"operator entered danger zone while boom MOVING "
                                   f"(motion {e['boom_motion']})", source="entries",
                       started_at=e["time"])
            else:
                lg.log(EL.OPERATOR_IN_ZONE, e["time"], severity=EL.INFO,
                       description="operator entered danger zone to reload — boom stopped",
                       source="entries")

    # 3) mesh-install milestones (ESTIMATE: a new mesh = a new TEMPORAL install
    #    episode; the total depends on face size and is not assumed).
    if Path(a.operator).exists():
        from coverage import mesh_installs
        ops = json.loads(Path(a.operator).read_text())["events"]
        for k, ins in enumerate(mesh_installs(ops)):
            lg.log(EL.SCREEN_INSTALLED, ins["time"], severity=EL.INFO,
                   description=f"mesh #{k+1} installed (estimate)", source="coverage",
                   mesh=k + 1, cx=ins["cx"])

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
