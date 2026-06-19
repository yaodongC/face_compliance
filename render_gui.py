"""Render the face-support compliance MONITOR to an MP4 (headless, Pillow-composited).

A mining safety control-room UI: the live camera feed is the hero; an Operator-Safety
status card and a Mesh-Install progress card sit to the right; a full-width event-log
timeline runs along the bottom. Calm when clear, the whole frame turns to alarm on a
DANGER (operator under a moving boom). Each fact appears exactly once.

Usage:
  python3 render_gui.py --video data/full_cycle.mp4 --analysis data/full_cycle_analysis.json \
      --out data/full_cycle_gui.mp4 --index data/full_cycle.idx \
      --events data/event_log.jsonl --operator data/operator_events.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml
from coverage import mesh_installs, FACE_X

# ---------------------------------------------------------------- theme (RGB)
BG = (13, 17, 23)
SURFACE = (22, 28, 36)
SURFACE_HI = (28, 35, 45)
DANGER_BG = (38, 18, 20)
HAIR = (40, 49, 61)
INK = (233, 238, 243)
MUTED = (132, 143, 156)
FAINT = (92, 102, 114)
CLEAR = (52, 211, 153)
DANGER = (255, 86, 86)
AMBER = (255, 176, 32)
WARN = (228, 161, 8)
SEV = {"INFO": MUTED, "WARNING": WARN, "VIOLATION": DANGER, "CRITICAL": (255, 130, 130)}

_UB = "/usr/share/fonts/truetype/ubuntu/"
_LIB = "/usr/share/fonts/truetype/liberation/"
_FONTMAP = {
    "b": [_UB + "Ubuntu-B.ttf", _LIB + "LiberationSans-Bold.ttf"],
    "m": [_UB + "Ubuntu-M.ttf", _LIB + "LiberationSans-Bold.ttf"],
    "r": [_UB + "Ubuntu-R.ttf", _LIB + "LiberationSans-Regular.ttf"],
    "mono": [_UB + "UbuntuMono-R.ttf", _LIB + "LiberationMono-Regular.ttf"],
    "monob": [_UB + "UbuntuMono-B.ttf", _LIB + "LiberationMono-Bold.ttf"],
}


def _font(weight, size):
    for p in _FONTMAP[weight]:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


# current-activity labels from the (informational) face-perception verdict
NOW = {"DRILLING": "Face drilling in progress", "SUPPORTED": "Booms parked at face",
       "UNSUPPORTED": "Bare face exposed", "NOT VERIFIED": "Assessing face support"}


def _load_index(path):
    if not path or not Path(path).exists():
        return None
    rows = [l.split(",") for l in Path(path).read_text().splitlines()[1:]]
    return {int(f): float(c) for f, c in rows}


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


def _danger_windows(events):
    return [(e["started_at"], e.get("cycle_sec", e["started_at"])) for e in events
            if e.get("type") == "operator_in_danger_zone" and e.get("started_at") is not None]


def _danger_active(dwins, csec):
    return any(a - 0.1 <= csec <= b + 0.1 for a, b in dwins)


# ---------------------------------------------------------------- draw helpers
def _tracked(d, pos, text, font, fill, track=2):
    x, y = pos
    for ch in text:
        d.text((x, y), ch, font=font, fill=fill)
        x += d.textlength(ch, font=font) + track
    return x


def _right(d, right, y, text, font, fill):
    d.text((right - d.textlength(text, font=font), y), text, font=font, fill=fill)


def _wrap(d, text, font, maxw):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if d.textlength(t, font=font) <= maxw:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def render(video, analysis, out, index_path=None, fps=15.0, face_crop=None,
           events_path=None, operator_path=None):
    data = json.loads(Path(analysis).read_text())
    steps = sorted(data["steps"], key=lambda s: s["t_sec"])
    index = _load_index(index_path)
    events = _load_events(events_path)
    dwins = _danger_windows(events)
    ops = []
    if operator_path and Path(operator_path).exists():
        ops = json.loads(Path(operator_path).read_text()).get("events", [])
    installs = mesh_installs(ops)

    cap = cv2.VideoCapture(video)
    vfps = cap.get(cv2.CAP_PROP_FPS) or fps

    # fonts
    f_num = _font("b", 60); f_verdict = _font("b", 42); f_h1 = _font("b", 22)
    f_eye = _font("m", 12); f_body = _font("r", 16); f_small = _font("r", 14)
    f_mono = _font("mono", 14); f_monob = _font("monob", 15); f_title = _font("b", 18)

    # geometry
    W, H = 1376, 788
    M = 24
    vx, vy, vw, vh = M, 86, 860, 484
    rx = vx + vw + M
    rw = W - rx - M
    safe_box = (rx, vy, rx + rw, vy + 232)
    mesh_box = (rx, vy + 248, rx + rw, vy + 484)
    log_box = (M, 586, W - M, 748)

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
        accent = DANGER if danger else CLEAR

        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)

        # top alarm strip (only on danger) -- the one bold reaction
        if danger:
            d.rectangle([0, 0, W, 4], fill=DANGER)

        # ---- header
        d.rounded_rectangle([M, 30, M + 12, 42], radius=3, fill=AMBER)
        d.text((M + 24, 27), "FACE-SUPPORT MONITOR", font=f_title, fill=INK)
        cs = int(csec)
        _right(d, W - M, 30, f"CYCLE {cs//60:02d}:{cs%60:02d}", f_monob, INK)
        _right(d, W - M - 132, 31, "● REC", f_small, DANGER if (fno // 8) % 2 else (120, 40, 40))
        d.line([M, 68, W - M, 68], fill=HAIR, width=1)

        # ---- camera feed (hero)
        fr = cv2.cvtColor(cv2.resize(frame, (vw, vh)), cv2.COLOR_BGR2RGB)
        img.paste(Image.fromarray(fr), (vx, vy))
        # caption strip (semi-transparent) along the feed's bottom
        strip = Image.new("RGBA", (vw, 30), (8, 11, 16, 175))
        img.paste(strip, (vx, vy + vh - 30), strip)
        d.text((vx + 12, vy + vh - 24), "FRONT RGB  ·  END FACE", font=f_small, fill=(190, 198, 208))
        now = "Operator at face — boom moving" if danger else NOW.get(step["verdict"], "Monitoring")
        _right(d, vx + vw - 12, vy + vh - 24, now, f_small, accent if danger else (190, 198, 208))
        # status-coloured frame around the feed
        d.rectangle([vx - 2, vy - 2, vx + vw + 1, vy + vh + 1], outline=accent, width=3 if danger else 2)

        # ---- operator-safety card
        d.rounded_rectangle(safe_box, radius=14, fill=DANGER_BG if danger else SURFACE,
                            outline=DANGER if danger else HAIR, width=1)
        d.rounded_rectangle([safe_box[0], safe_box[1] + 14, safe_box[0] + 4, safe_box[3] - 14],
                            radius=2, fill=accent)
        _tracked(d, (rx + 22, safe_box[1] + 22), "OPERATOR SAFETY", f_eye, MUTED, 2)
        word = "DANGER" if danger else "CLEAR"
        d.text((rx + 20, safe_box[1] + 44), word, font=f_verdict, fill=accent)
        reason = ("Operator in the danger zone while the boom is still moving — drilling must stop."
                  if danger else "No operator under a moving boom. Bolting cycle proceeding normally.")
        ry = safe_box[1] + 108
        for ln in _wrap(d, reason, f_body, rw - 44):
            d.text((rx + 22, ry), ln, font=f_body, fill=INK if danger else MUTED)
            ry += 24
        # small footnote: coverage is not auto-certified
        d.text((rx + 22, safe_box[3] - 30), "Full mesh coverage requires on-site inspection.",
               font=f_small, fill=FAINT)

        # ---- mesh-install progress card
        d.rounded_rectangle(mesh_box, radius=14, fill=SURFACE, outline=HAIR, width=1)
        _tracked(d, (rx + 22, mesh_box[1] + 22), "MESH INSTALLATION", f_eye, MUTED, 2)
        d.text((rx + 20, mesh_box[1] + 40), str(n_mesh), font=f_num, fill=AMBER)
        numw = d.textlength(str(n_mesh), font=f_num)
        d.text((rx + 28 + numw, mesh_box[1] + 62), "meshes", font=f_h1, fill=INK)
        d.text((rx + 28 + numw, mesh_box[1] + 88), "installed (est.)", font=f_small, fill=MUTED)
        # location lane: where each installed mesh sits across the face width
        lane_y = mesh_box[1] + 150
        lx0, lx1 = rx + 22, mesh_box[2] - 22
        d.line([lx0, lane_y, lx1, lane_y], fill=HAIR, width=2)
        fx0, fx1 = FACE_X
        for k, ins in enumerate(installs):
            if ins["time"] <= csec + 0.1:
                fxr = min(max((ins["cx"] - fx0) / (fx1 - fx0), 0.0), 1.0)
                mx = int(lx0 + fxr * (lx1 - lx0))
                d.ellipse([mx - 5, lane_y - 5, mx + 5, lane_y + 5], fill=AMBER)
                d.text((mx - 8, lane_y + 10), f"M{k+1}", font=f_small, fill=AMBER)
        # install times, monospaced
        times = "   ".join(f"{int(i['time'])//60:02d}:{int(i['time'])%60:02d}"
                           for i in installs if i["time"] <= csec + 0.1) or "—"
        d.text((rx + 22, mesh_box[3] - 30), times, font=f_mono, fill=MUTED)

        # ---- event-log timeline (full width)
        d.rounded_rectangle(log_box, radius=14, fill=SURFACE, outline=HAIR, width=1)
        _tracked(d, (M + 22, log_box[1] + 20), "EVENT LOG", f_eye, MUTED, 2)
        nviol = sum(1 for e in events if e.get("severity") == "VIOLATION" and e.get("cycle_sec", 0) <= csec + 0.1)
        _right(d, W - M - 22, log_box[1] + 16, f"{nviol} violations", f_small, DANGER if nviol else MUTED)
        shown = [e for e in events if e.get("cycle_sec", 0) <= csec + 0.1][-5:]
        ey = log_box[1] + 48
        for e in shown:
            ec = int(e.get("cycle_sec", 0))
            col = SEV.get(e.get("severity", "INFO"), MUTED)
            d.text((M + 22, ey), f"{ec//60:02d}:{ec%60:02d}", font=f_monob, fill=MUTED)
            d.ellipse([M + 92, ey + 5, M + 102, ey + 15], fill=col)
            d.text((M + 116, ey), e.get("description", "")[:96], font=f_body,
                   fill=INK if e.get("severity") in ("VIOLATION", "WARNING") else MUTED)
            ey += 22

        # ---- footer
        _tracked(d, (M, H - 22), "ASSISTIVE MONITOR  ·  NOT A CERTIFIED SAFETY SYSTEM  ·  VERIFY ON SITE",
                 f_small, FAINT, 1)

        writer.write(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
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
    ap.add_argument("--face-crop-config", default="config.yaml")
    ap.add_argument("--events", default="data/event_log.jsonl")
    ap.add_argument("--operator", default="data/operator_events.json")
    a = ap.parse_args()
    video, analysis = a.video, a.analysis
    cfgpath = a.config or a.face_crop_config
    if cfgpath and Path(cfgpath).exists():
        cfg = yaml.safe_load(Path(cfgpath).read_text())
        video = video or cfg["paths"]["video"]
        analysis = analysis or cfg["paths"]["analysis"]
    render(video, analysis, a.out, a.index, a.fps, None, a.events, a.operator)


if __name__ == "__main__":
    main()
