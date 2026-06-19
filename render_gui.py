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
from coverage import mesh_installs, FACE_X

# BGR colours. Compliance is COVERAGE-driven: COMPLIANT only when the WHOLE face is
# covered by OVERLAPPING bolted meshes; partial coverage is NOT supported.
BANNER = {"DANGER": (40, 20, 120), "NOT SUPPORTED": (44, 44, 192),
          "COMPLIANT": (60, 160, 60)}
SUBTITLE = {
    "DANGER": "Operator in front while boom MOVING - drilling not stopped",
    "NOT SUPPORTED": "Full mesh coverage not auto-confirmed - human inspection required",
    "COMPLIANT": "Entire face covered by overlapping bolted meshes (assistive - still verify)",
}


def _danger_windows(events):
    """Operator-in-danger-zone incident windows [start, end] from the event log."""
    wins = []
    for e in events:
        if e.get("type") == "operator_in_danger_zone" and e.get("started_at") is not None:
            wins.append((e["started_at"], e.get("cycle_sec", e["started_at"])))
    return wins


def _danger_active(dwins, csec):
    return any(a - 0.1 <= csec <= b + 0.1 for a, b in dwins)
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


def render(video, analysis, out, index_path=None, fps=15.0, face_crop=None,
           events_path=None, operator_path=None):
    data = json.loads(Path(analysis).read_text())
    steps = sorted(data["steps"], key=lambda s: s["t_sec"])
    index = _load_index(index_path)
    events = _load_events(events_path)
    dwins = _danger_windows(events)
    # continuous face-width coverage from operator install sites (per-mesh boxes are
    # not reliable; the number of screens depends on face size and is NOT assumed)
    ops = []
    if operator_path and Path(operator_path).exists():
        ops = json.loads(Path(operator_path).read_text()).get("events", [])
    installs = mesh_installs(ops)   # estimated mesh-install times + locations
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
        n_mesh = sum(1 for i in installs if i["time"] <= csec + 0.1)
        danger = _danger_active(dwins, csec)
        v = "DANGER" if danger else "NOT SUPPORTED"
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
        # Meshes installed: a running COUNT (no status bar - the total number of
        # screens depends on face size and is unknown). Thin ticks mark each counted
        # mesh's approximate location (estimate, not a precise box).
        for k, ins in enumerate(installs):
            if ins["time"] <= csec + 0.1:
                mx = int(ins["cx"] * vid_w)
                cv2.line(fr, (mx, vid_h - 40), (mx, vid_h - 12), (70, 210, 70), 2, cv2.LINE_AA)
                cv2.putText(fr, f"M{k+1}", (mx - 12, vid_h - 44), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (70, 210, 70), 1, cv2.LINE_AA)
        cv2.rectangle(fr, (8, vid_h - 34), (224, vid_h - 8), (35, 35, 35), -1)
        cv2.putText(fr, f"Meshes installed: {n_mesh}", (14, vid_h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (90, 235, 90), 2, cv2.LINE_AA)
        canvas[92:92 + vid_h, 10:10 + vid_w] = fr

        # cycle clock
        cv2.putText(canvas, f"cycle {int(csec)//60:02d}:{int(csec)%60:02d}", (16, 92 + vid_h + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 60), 2, cv2.LINE_AA)

        # compliance checklist (COVERAGE-driven, not booms-parked)
        cv2.putText(canvas, "Compliance: full overlapping mesh coverage", (panel_x, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 2, cv2.LINE_AA)
        cov_items = [
            ("Full mesh coverage - requires human check", "not_verified"),
            ("Operator clear of moving boom", "violation" if danger else "not_verified"),
        ]
        y = 150
        for label, st in cov_items:
            c = ITEM_COLOR.get(st, (130, 130, 130))
            cv2.putText(canvas, ITEM_MARK.get(st, "[ ]"), (panel_x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA)
            for i, ln in enumerate(_wrap(label, 30)):
                cv2.putText(canvas, ln, (panel_x + 44, y + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, c, 1, cv2.LINE_AA)
            y += 26 + 22 * max(1, len(_wrap(label, 30)))
        cv2.putText(canvas, f"Meshes installed so far: {n_mesh}  (estimate)",
                    (panel_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 235, 90), 2, cv2.LINE_AA)
        y += 28

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
        cv2.putText(canvas, "Event log", (12, ey0),
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
    ap.add_argument("--operator", default="data/operator_events.json",
                    help="operator events -> 4-segment face-coverage bar")
    a = ap.parse_args()
    video, analysis = a.video, a.analysis
    face_crop = None
    cfgpath = a.config or a.face_crop_config
    if cfgpath and Path(cfgpath).exists():
        cfg = yaml.safe_load(Path(cfgpath).read_text())
        video = video or cfg["paths"]["video"]
        analysis = analysis or cfg["paths"]["analysis"]
        face_crop = cfg.get("face_crop")
    render(video, analysis, a.out, a.index, a.fps, face_crop, a.events, a.operator)


if __name__ == "__main__":
    main()
