"""Render 'what the GUI shows' to an MP4 (headless, no Qt).

Composites each video frame with the verdict banner, the fail-safe checklist, and
the scene text from a cached analysis.json -- i.e. a screen-recording of the GUI,
produced directly. Works on the full-cycle time-lapse to give a GUI time-lapse of
the whole development cycle.

Usage:
  python3 render_gui.py --video data/full_cycle.mp4 --analysis data/full_cycle_analysis.json \
      --out data/full_cycle_gui.mp4 [--index data/full_cycle.idx] [--fps 15]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import yaml

# BGR colours
BANNER = {"DANGER": (40, 20, 120), "UNSUPPORTED": (44, 44, 192),
          "DRILLING": (0, 102, 204), "NOT VERIFIED": (30, 150, 200),
          "SUPPORTED": (60, 160, 60)}
SUBTITLE = {
    "DANGER": "Person under unsupported ground - clear the area",
    "UNSUPPORTED": "End face not screened - treat as UNSUPPORTED",
    "DRILLING": "Active face drilling - work in progress, not the supported state",
    "NOT VERIFIED": "Face support not confirmed - human inspection required",
    "SUPPORTED": "Face screened + booms parked, drilling done (assistive - still verify)",
}
ITEM_COLOR = {"verified": (70, 160, 70), "violation": (44, 44, 192), "not_verified": (130, 130, 130)}
ITEM_MARK = {"verified": "[x]", "violation": "[!]", "not_verified": "[ ]"}


def _load_index(path):
    if not path or not Path(path).exists():
        return None
    rows = [l.split(",") for l in Path(path).read_text().splitlines()[1:]]
    return {int(f): float(c) for f, c in rows}


def _wrap(text, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= width:
            cur = (cur + " " + w).strip()
        else:
            lines.append(cur); cur = w
    if cur:
        lines.append(cur)
    return lines


SEV_COLOR = {"INFO": (200, 200, 200), "WARNING": (40, 170, 220),
             "VIOLATION": (44, 44, 220), "CRITICAL": (40, 20, 200)}


def _load_events(path):
    if not path or not Path(path).exists():
        return []
    out = []
    for line in Path(path).read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return sorted(out, key=lambda e: e.get("cycle_sec", 0))


def _load_meshes(path):
    if not path or not Path(path).exists():
        return []
    return json.loads(Path(path).read_text()).get("meshes", [])


def render(video, analysis, out, index_path=None, fps=15.0, face_crop=None,
           events_path=None, meshes_path=None):
    data = json.loads(Path(analysis).read_text())
    steps = sorted(data["steps"], key=lambda s: s["t_sec"])
    items = data["meta"]["items"]
    index = _load_index(index_path)
    events = _load_events(events_path)
    meshes = _load_meshes(meshes_path)
    cap = cv2.VideoCapture(video)
    vfps = cap.get(cv2.CAP_PROP_FPS) or fps

    W, H = 1320, 760
    vid_w, vid_h = 860, 484
    panel_x = vid_w + 20
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    def cur_step(t):
        c = steps[0]
        for s in steps:
            if s["t_sec"] <= t:
                c = s
            else:
                break
        return c

    fno = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = fno / vfps
        csec = index.get(fno, t) if index else t
        step = cur_step(t)
        v = step["verdict"]
        canvas = np.full((H, W, 3), 30, dtype=np.uint8)

        # banner
        col = BANNER.get(v, (90, 90, 90))
        cv2.rectangle(canvas, (0, 0), (W, 56), col, -1)
        cv2.putText(canvas, v, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(canvas, SUBTITLE.get(v, ""), (16, 78), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (210, 210, 210), 1, cv2.LINE_AA)

        # video (with the analysed face-crop region drawn so the viewer can see
        # exactly what the model judges)
        fr = cv2.resize(frame, (vid_w, vid_h))
        if face_crop:
            x0, y0, x1, y1 = face_crop
            cv2.rectangle(fr, (int(x0 * vid_w), int(y0 * vid_h)),
                          (int(x1 * vid_w), int(y1 * vid_h)), (0, 220, 220), 2)
            cv2.putText(fr, "model view: end face", (int(x0 * vid_w) + 4, int(y0 * vid_h) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 220), 1, cv2.LINE_AA)
        # persistent installed-mesh panels: each mesh keeps its own colour and stays
        # drawn once installed (the coverage building up panel-by-panel)
        for m in meshes:
            if m.get("installed_at", 0) <= csec + 0.1:
                bx0, by0, bx1, by1 = m["bbox"]
                col = tuple(int(c) for c in m["color"])
                cv2.rectangle(fr, (int(bx0 * vid_w), int(by0 * vid_h)),
                              (int(bx1 * vid_w), int(by1 * vid_h)), col, 2)
                cv2.putText(fr, m["label"], (int(bx0 * vid_w) + 3, int(by0 * vid_h) + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)
        canvas[92:92 + vid_h, 10:10 + vid_w] = fr

        # cycle clock
        cv2.putText(canvas, f"cycle {int(csec)//60:02d}:{int(csec)%60:02d}", (16, 92 + vid_h + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 60), 2, cv2.LINE_AA)

        # checklist panel
        cv2.putText(canvas, "Face-support checklist", (panel_x, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 2, cv2.LINE_AA)
        snap = step["checklist_snapshot"]
        y = 150
        for it in items:
            st = snap.get(it["id"], "not_verified")
            c = ITEM_COLOR.get(st, (130, 130, 130))
            cv2.putText(canvas, ITEM_MARK.get(st, "[ ]"), (panel_x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA)
            for i, ln in enumerate(_wrap(it["label"], 30)):
                cv2.putText(canvas, ln, (panel_x + 44, y + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, c, 1, cv2.LINE_AA)
            y += 26 + 22 * max(1, len(_wrap(it["label"], 30)))

        # scene text
        y += 8
        cv2.putText(canvas, "What the camera sees:", (panel_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        y += 24
        for ln in _wrap(step.get("scene", ""), 34)[:5]:
            cv2.putText(canvas, ln, (panel_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 200, 220), 1, cv2.LINE_AA)
            y += 20

        # event-log panel (external memory) -- events up to the current cycle time,
        # ordered by time, most recent at the bottom
        ey0 = 92 + vid_h + 44
        cv2.line(canvas, (10, ey0 - 16), (W - 10, ey0 - 16), (90, 90, 90), 1)
        cv2.putText(canvas, "EVENT LOG (external memory) - ordered by time", (12, ey0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 2, cv2.LINE_AA)
        shown = [e for e in events if e.get("cycle_sec", 0) <= csec + 0.1]
        for i, e in enumerate(shown[-7:]):
            cs = int(e.get("cycle_sec", 0))
            col = SEV_COLOR.get(e.get("severity", "INFO"), (200, 200, 200))
            line = (f"{cs//60:02d}:{cs%60:02d}  [{e.get('severity','INFO'):9}] "
                    f"{e.get('type',''):20} {e.get('description','')[:70]}")
            cv2.putText(canvas, line, (14, ey0 + 24 + i * 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, col, 1, cv2.LINE_AA)

        # disclaimer footer
        cv2.putText(canvas, "ASSISTIVE DEMO - NOT A CERTIFIED SAFETY SYSTEM. Always physically verify.",
                    (12, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (60, 200, 255), 1, cv2.LINE_AA)
        writer.write(canvas)
        fno += 1
    cap.release()
    writer.release()
    print(f"wrote {out} ({fno} frames)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--video")
    ap.add_argument("--analysis")
    ap.add_argument("--out", required=True)
    ap.add_argument("--index")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--face-crop-config", default="config.yaml",
                    help="config to read the face_crop box from (for the overlay)")
    ap.add_argument("--events", default="data/event_log.jsonl",
                    help="event-log JSONL to display in the GUI panel")
    ap.add_argument("--meshes", default="data/mesh_events.json",
                    help="tracked mesh panels to draw persistently on the video")
    a = ap.parse_args()
    video, analysis = a.video, a.analysis
    face_crop = None
    cfgpath = a.config or a.face_crop_config
    if cfgpath and Path(cfgpath).exists():
        cfg = yaml.safe_load(Path(cfgpath).read_text())
        video = video or cfg["paths"]["video"]
        analysis = analysis or cfg["paths"]["analysis"]
        face_crop = cfg.get("face_crop")
    render(video, analysis, a.out, a.index, a.fps, face_crop, a.events, a.meshes)


if __name__ == "__main__":
    main()
