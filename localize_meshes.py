"""Localise mesh panels by VLM-grounding the screen during each install session.

For each operator reload session, grab the frame and ground the wire-mesh screen
being installed (clamped to the face band). This is more faithful than the operator
position, but APPROXIMATE - a bolted mesh blends into the face and cannot be
localised precisely (labels are marked 'est').

Input : data/operator_events.json, data/full_cycle.mp4 + .idx (frames by cycle time)
Output: data/mesh_events.json [{mesh_id,bbox,installed_at,color,label}]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import yaml
import operator_safety as osf

PALETTE = [(75, 180, 60), (48, 130, 245), (200, 130, 0), (230, 50, 240),
           (240, 240, 70), (60, 60, 240), (145, 30, 180), (255, 200, 0),
           (0, 200, 120), (200, 0, 200), (90, 160, 255), (160, 220, 40)]


def _iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _frame_at(cap, idx, cycle_sec):
    fr = min(idx, key=lambda k: abs(idx[k] - cycle_sec))
    cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
    ok, f = cap.read()
    return f if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--events", default="data/operator_events.json")
    ap.add_argument("--video", default="data/full_cycle.mp4")
    ap.add_argument("--index", default="data/full_cycle.idx")
    ap.add_argument("--out", default="data/mesh_events.json")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    ops = json.loads(Path(a.events).read_text())["events"]
    idx = {int(f): float(c) for f, c in
           (l.split(",") for l in Path(a.index).read_text().splitlines()[1:])}
    cap = cv2.VideoCapture(a.video)

    meshes = []
    for s in osf.classify_sessions(ops):
        # use a mid-session moment (screen most likely in hand / on the face)
        tc = (s["start"] + s["end"]) / 2.0
        frame = _frame_at(cap, idx, tc)
        if frame is None:
            continue
        det = osf.detect_screen(frame, cfg)
        bb = det.get("screen_bbox")
        if not bb:
            continue
        hit = next((m for m in meshes if _iou(m["bbox"], bb) > 0.4), None)
        if hit is None:
            i = len(meshes)
            meshes.append({"mesh_id": i + 1, "bbox": bb, "installed_at": s["start"],
                           "color": list(PALETTE[i % len(PALETTE)]),
                           "label": f"mesh {i + 1} (est)"})
    cap.release()
    Path(a.out).write_text(json.dumps({"meshes": meshes}, indent=2))
    print(f"=== {len(meshes)} mesh panels localised (VLM screen grounding, est) -> {a.out} ===")
    for m in meshes:
        cs = int(m["installed_at"])
        print(f"  {m['label']:14} installed {cs//60:02d}:{cs%60:02d}  bbox={m['bbox']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
