"""Track installed mesh/screen panels from operator install events.

A fully-bolted mesh blends into the face and cannot be detected statically, so we
record WHERE each screen is installed (the operator's location while loading a
screen / fitting a bolt with the drill stopped) and REMEMBER it. Each distinct
install location becomes a persistent mesh panel with its own colour, so the GUI
can show the coverage building up panel-by-panel and keep showing it afterwards.

Input : data/operator_events.json (confirmed operators with bboxes).
Output: data/mesh_events.json  [{mesh_id, bbox, installed_at, color, label}]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

# distinct BGR colours per mesh panel
PALETTE = [(75, 180, 60), (48, 130, 245), (200, 130, 0), (230, 50, 240),
           (240, 240, 70), (60, 60, 240), (145, 30, 180), (255, 200, 0),
           (0, 200, 120), (200, 0, 200), (90, 160, 255), (160, 220, 40)]

# the face band the screens are bolted onto (fractions); a panel is ~PANEL_W wide
FACE_Y = (0.20, 0.80)
PANEL_W = 0.14


def _install_x(e):
    bb = e.get("person_bbox")
    if not bb:
        return None
    return (bb[0] + bb[2]) / 2.0


def build_meshes(events):
    """Cluster confirmed install events by horizontal location into mesh panels.
    Only events where the operator is actually installing (drill stopped =
    OK_LOADING, or an install action) seed a panel."""
    meshes = []
    for e in sorted(events, key=lambda x: x["cycle_sec"]):
        x = _install_x(e)
        if x is None:
            continue
        # Every confirmed operator-in-front site is a screen-install location (the
        # worker is at the face installing), whether the install was done safely
        # (OK_LOADING) or dangerously (boom moving). Coverage tracks WHERE screens
        # went in; the danger/compliance of HOW is recorded separately in the log.
        # same panel if within ~half a panel width of an existing centre
        hit = next((m for m in meshes if abs(m["_cx"] - x) < PANEL_W * 0.6), None)
        if hit is None:
            i = len(meshes)
            meshes.append({"mesh_id": i + 1, "_cx": x,
                           "bbox": [round(max(0.0, x - PANEL_W / 2), 3), FACE_Y[0],
                                    round(min(1.0, x + PANEL_W / 2), 3), FACE_Y[1]],
                           "installed_at": e["cycle_sec"],
                           "color": list(PALETTE[i % len(PALETTE)]),
                           "label": f"mesh {i + 1}"})
    for m in meshes:
        m.pop("_cx", None)
    return meshes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="data/operator_events.json")
    ap.add_argument("--out", default="data/mesh_events.json")
    a = ap.parse_args()
    ops = json.loads(Path(a.events).read_text())["events"]
    meshes = build_meshes([e for e in ops if e.get("person_bbox")])
    Path(a.out).write_text(json.dumps({"meshes": meshes}, indent=2))
    print(f"=== {len(meshes)} mesh panels tracked -> {a.out} ===")
    for m in meshes:
        cs = int(m["installed_at"])
        print(f"  {m['label']:8} installed {cs//60:02d}:{cs%60:02d}  bbox={m['bbox']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
