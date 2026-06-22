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
import hashlib
import json
from pathlib import Path
import event_log as EL
from task import active_task, task_dir

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


def _log_init(lg, geom="data/face_geometry.json"):
    """Emit startup messages explaining the size-derived support requirement: the Lidar
    measured the face, and the Vale standards turn that size into the # of meshes/bolts.
    Reads data/face_geometry.json; degrades to the configured default if it is absent."""
    g = json.loads(Path(geom).read_text()) if Path(geom).exists() else {}
    try:
        import progress_tracker as pt
        tg = pt.load_targets()
        nm, nb = tg["meshes_required"], tg["bolts_required"]
        fw = g.get("face_width") or tg.get("face_width")
    except Exception:
        nm, nb, fw = g.get("meshes_required", 4), g.get("bolts_required", 16), g.get("face_width")
    fh = g.get("face_height")
    src = "lidar+vale" if g.get("face_width") else "default(no lidar)"
    dim = f"{fw:.1f} x {fh:.1f} m" if (fw and fh) else (f"{fw:.1f} m wide" if fw else "size n/a")
    sheet = g.get("sheet_w_ft", 6.0)
    lg.log(EL.SYSTEM_INIT, 0.0, severity=EL.INFO, source="lidar",
           description=f"Front Livox Mid360 measured end face: {dim} (gravity-leveled scan)")
    lg.log(EL.SYSTEM_INIT, 0.0, severity=EL.INFO, source="vale",
           description=f"Vale CMTS-2015-001/Div6: {int(sheet)}ft screens, 1ft overlap "
                       f"-> {nm} mesh panels for this face")
    lg.log(EL.SYSTEM_INIT, 0.0, severity=EL.INFO, source="vale",
           description=f"Bolt pattern 4'x5' (4 bolts/screen) -> {nb} bolts required  [{src}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default="data/full_cycle_analysis.json")
    ap.add_argument("--index", default="data/full_cycle.idx")
    ap.add_argument("--operator", default="data/operator_events.json")
    ap.add_argument("--out", default="data/event_log.jsonl")
    a = ap.parse_args()
    lg = EL.EventLogger(a.out, reset=True)
    cyc = _cycle_mapper(a.index)

    # --- SYSTEM INIT: record WHY this face needs N meshes / M bolts, grounded in the
    # Lidar measurement + the Vale documents, at the start of the timeline (cycle 0). ---
    _log_init(lg)

    # operator danger-zone ENTRIES from the VLM operator scan (confirms a PERSON, so
    # no orange-colour false positives). The operator must enter to reload - that is
    # normal; non-compliant ONLY when the boom was still moving at entry
    # (boom_motion_thresh in config.yaml).
    if Path(a.operator).exists():
        from operator_safety import classify_sessions
        from coverage import mesh_installs
        from rule_config import RULES
        from rules_engine import decide_traced
        ops = json.loads(Path(a.operator).read_text())["events"]
        for s in classify_sessions(ops):
            ev = f"data/operator_frames/op_{int(s['start']):05d}.png"
            # provenance: which rule row produced this verdict (audit trail)
            _, ridx = decide_traced(RULES["operator_entry"], {"boom_moving_at_entry": s["entry_boom_moving"]})
            rule = {"table": "operator_entry", "index": ridx}
            if s["verdict"] == "NON_COMPLIANT_ENTRY":
                lg.log(EL.OPERATOR_IN_ZONE, s["end"], severity=EL.VIOLATION,
                       description=f"operator entered danger zone while boom MOVING "
                                   f"(entry motion {s['entry_motion']:.3f})",
                       bbox=s["person_bbox"], evidence=ev, source="operator_safety",
                       rule=rule, started_at=s["start"], duration_sec=round(s["end"] - s["start"], 1))
            else:
                lg.log(EL.OPERATOR_IN_ZONE, s["start"], severity=EL.INFO,
                       description="operator entered danger zone to reload — boom stopped",
                       bbox=s["person_bbox"], evidence=ev, source="operator_safety", rule=rule)
        # mesh-install milestones (a new mesh = a new TEMPORAL install episode)
        for k, ins in enumerate(mesh_installs(ops)):
            lg.log(EL.SCREEN_INSTALLED, ins["time"], severity=EL.INFO,
                   description=f"mesh #{k+1} installed (estimate)", source="coverage",
                   mesh=k + 1, cx=ins["cx"])

    # run-provenance sidecar (audit: which task bundle + exact rules/params/prompts
    # produced this log). Separate file -> does not change the event log itself.
    td = task_dir()

    def _sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else None

    meta = {"task": active_task(), "events": lg.summary()["total"],
            "rules_sha": _sha(td / "rules.yaml"), "params_sha": _sha(td / "params.yaml"),
            "prompts_sha": _sha(td / "prompts.yaml")}
    Path(a.out).with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))

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
