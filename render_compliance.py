"""Render the compliance-milestone monitor over the full cycle to an MP4.

Self-contained OpenCV overlay (no font-file dependency, kept separate from the
e2e-locked render_gui.compose). For each frame it shows LIVE, code-computed progress:
bolts x/16 (16-segment bar), screens x/4, face coverage %, the recent milestone log, and
a full-frame COMPLIANCE COMPLETE banner once the latched moment is reached.

  python3 render_compliance.py [--video data/full_cycle.mp4] [--index data/full_cycle.idx]
                               [--out data/compliance_cycle_gui.mp4] [--width 1280] [--fps 30]
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import bisect
import cv2
import progress_tracker as pt

BG = (24, 22, 20); SURF = (40, 36, 32); HAIR = (70, 64, 58)
INK = (225, 225, 225); MUTED = (150, 150, 150)
GREEN = (90, 200, 110); AMBER = (60, 180, 240); RED = (60, 70, 230); BLUE = (235, 180, 70)


def _load_index(path):
    rows = list(csv.reader(Path(path).read_text().splitlines()))[1:]
    return [int(r[0]) for r in rows if r], [float(r[1]) for r in rows if r]


def _seg_bar(img, x, y, w, h, n, total, color):
    gap = 3
    sw = (w - gap * (total - 1)) / total
    for i in range(total):
        x0 = int(x + i * (sw + gap))
        c = color if i < n else SURF
        cv2.rectangle(img, (x0, y), (int(x0 + sw), y + h), c, -1)
        cv2.rectangle(img, (x0, y), (int(x0 + sw), y + h), HAIR, 1)


def _text(img, s, x, y, scale=0.6, color=INK, thick=1):
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def render(video, index, out, complete_at, milestones, width=1280, fps=30.0):
    tl, ev, cls = pt.load_evidence()
    targets = pt.load_targets()
    NB, NM = targets["bolts_required"], targets["meshes_required"]
    frames_idx, times = _load_index(index)
    cap = cv2.VideoCapture(str(video))
    sw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); sh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = width / sw
    vw, vh = width, int(sh * scale)
    PANEL = 360
    W, H = vw + PANEL, vh
    wr = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    nf = len(frames_idx)
    for fi in range(nf):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frames_idx[fi])
        ok, fr = cap.read()
        if not ok:
            continue
        csec = times[fi]
        feed = cv2.resize(fr, (vw, vh))
        canvas = cv2.copyMakeBorder(feed, 0, 0, 0, PANEL, cv2.BORDER_CONSTANT, value=BG)
        prog = pt.progress_at(tl, ev, csec, cls, targets)
        nb, nm = prog["bolts_target"], prog["screens_target"]
        done = complete_at is not None and csec >= complete_at
        px = vw + 20
        # header
        _text(canvas, "LoopX FACE-SUPPORT COMPLIANCE", px, 34, 0.62, INK, 2)
        _text(canvas, f"CYCLE {int(csec)//60:02d}:{int(csec)%60:02d}", px, 62, 0.5, MUTED)
        fw = prog.get("face_width")
        if fw:
            _text(canvas, f"face {fw}m -> need {nm} meshes / {nb} bolts", px, 82, 0.42, MUTED)
        # bolts
        _text(canvas, "BOLTS", px, 116, 0.6, MUTED)
        _text(canvas, f"{prog['bolts']}/{nb}", px + 250, 116, 0.7, GREEN if prog['bolts'] >= nb else AMBER, 2)
        _seg_bar(canvas, px, 128, PANEL - 42, 22, prog['bolts'], nb, GREEN)
        # screens
        _text(canvas, "SCREENS", px, 196, 0.6, MUTED)
        _text(canvas, f"{prog['screens']}/{nm}", px + 250, 196, 0.7, GREEN if prog['screens'] >= nm else AMBER, 2)
        _seg_bar(canvas, px, 208, PANEL - 42, 26, prog['screens'], nm, BLUE)
        # coverage
        cov = prog['coverage']
        _text(canvas, "FACE COVERAGE", px, 282, 0.6, MUTED)
        _text(canvas, f"{cov*100:.0f}%", px + 250, 282, 0.7, GREEN if cov >= pt.FULL_COVER_FRAC else AMBER, 2)
        cv2.rectangle(canvas, (px, 294), (px + PANEL - 42, 316), SURF, -1)
        cv2.rectangle(canvas, (px, 294), (int(px + (PANEL - 42) * min(1.0, cov)), 316), GREEN, -1)
        cv2.rectangle(canvas, (px, 294), (px + PANEL - 42, 316), HAIR, 1)
        # recent milestones
        _text(canvas, "MILESTONES", px, 372, 0.55, MUTED)
        recent = [m for m in milestones if m["cycle_sec"] <= csec + 0.1][-7:]
        for i, m in enumerate(recent):
            s = int(m["cycle_sec"])
            col = GREEN if m["type"] == "compliance_complete" else INK
            _text(canvas, f"{s//60:02d}:{s%60:02d} {m['description'][:34]}", px, 398 + i * 22, 0.42, col)
        # status chip
        chip = (12, 90, 24) if done else (40, 50, 70)
        cv2.rectangle(canvas, (px, H - 60), (px + PANEL - 42, H - 24), chip, -1)
        _text(canvas, "COMPLIANCE COMPLETE" if done else "INSTALLING SUPPORT...",
              px + 14, H - 36, 0.6, (120, 255, 150) if done else AMBER, 2)
        # full-frame banner on completion
        if done:
            cv2.rectangle(canvas, (0, 0), (vw, vh), (60, 200, 90), 6)
            ov = canvas.copy()
            cv2.rectangle(ov, (0, vh // 2 - 46), (vw, vh // 2 + 46), (10, 70, 20), -1)
            cv2.addWeighted(ov, 0.55, canvas, 0.45, 0, canvas)
            _text(canvas, "FACE SUPPORT COMPLIANT", 40, vh // 2 + 2, 1.2, (130, 255, 160), 3)
            _text(canvas, f"{NM}/{NM} screens - {NB}/{NB} bolts - full coverage - VLM confirmed",
                  40, vh // 2 + 34, 0.55, INK)
        _text(canvas, "ASSISTIVE MONITOR - NOT A CERTIFIED SAFETY SYSTEM - VERIFY ON SITE",
              20, H - 12, 0.42, MUTED)
        wr.write(canvas)
        if fi % 200 == 0:
            print(f"[render] {fi}/{nf}", flush=True)
    cap.release(); wr.release()
    print(f"[render] wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/full_cycle.mp4")
    ap.add_argument("--index", default="data/full_cycle.idx")
    ap.add_argument("--result", default="data/compliance_result.json")
    ap.add_argument("--out", default="data/compliance_cycle_gui.mp4")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--fps", type=float, default=30.0)
    a = ap.parse_args()
    res = json.loads(Path(a.result).read_text())
    render(a.video, a.index, a.out, res.get("complete_at"), res.get("milestones", []),
           a.width, a.fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
