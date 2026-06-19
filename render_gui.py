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
    ops = []
    if operator_path and Path(operator_path).exists():
        ops = json.loads(Path(operator_path).read_text()).get("events", [])
    installs = mesh_installs(ops)
    # operator danger-zone entries from the VLM operator scan, which confirms a
    # PERSON (no orange-colour false positives). One entry per reload visit.
    from operator_safety import classify_sessions
    entries = [{"time": s["start"], "end": s["end"], "verdict": s["verdict"]}
               for s in classify_sessions(ops)] if ops else []
    # hold the alarm for the realistic length of a reload visit (~14 s)
    dwins = [(e["time"] - 1, max(e["end"], e["time"] + 13)) for e in entries
             if e["verdict"] == "NON_COMPLIANT_ENTRY"]

    cap = cv2.VideoCapture(video)
    vfps = cap.get(cv2.CAP_PROP_FPS) or fps
    t_end = max(list(index.values()) if index else [s["t_sec"] for s in steps] or [1.0])
    t_end = max(t_end, 1.0)

    # fonts
    f_num = _font("b", 60); f_verdict = _font("b", 42); f_h1 = _font("b", 22)
    f_eye = _font("m", 12); f_body = _font("r", 16); f_small = _font("r", 14)
    f_mono = _font("mono", 14); f_monob = _font("monob", 15); f_title = _font("b", 18)
    f_cap = _font("b", 19)   # camera-feed caption (kept large + bold for legibility)

    # geometry
    W, H = 1376, 788
    M = 24
    vx, vy, vw, vh = M, 86, 860, 484
    rx = vx + vw + M
    rw = W - rx - M
    safe_box = (rx, vy, rx + rw, vy + 176)
    mesh_box = (rx, vy + 192, rx + rw, vy + 484)
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
        d.text((M + 24, 27), "LoopX Safety AI-Agent (VLM)", font=f_title, fill=INK)
        cs = int(csec)
        _right(d, W - M, 30, f"CYCLE {cs//60:02d}:{cs%60:02d}", f_monob, INK)
        _right(d, W - M - 132, 31, "● REC", f_small, DANGER if (fno // 8) % 2 else (120, 40, 40))
        d.line([M, 68, W - M, 68], fill=HAIR, width=1)

        # ---- camera feed (hero)
        fr = cv2.cvtColor(cv2.resize(frame, (vw, vh)), cv2.COLOR_BGR2RGB)
        img.paste(Image.fromarray(fr), (vx, vy))
        # caption strip (taller + more opaque so the text reads over the footage)
        ch = 46
        strip = Image.new("RGBA", (vw, ch), (6, 9, 13, 212))
        img.paste(strip, (vx, vy + vh - ch), strip)
        cy = vy + vh - 33
        # amber tick + bright label on the left
        d.rounded_rectangle([vx + 14, cy + 4, vx + 20, cy + 16], radius=2, fill=AMBER)
        d.text((vx + 30, cy), "FRONT RGB · END FACE", font=f_cap, fill=INK)
        # current activity on the right, in the live status colour
        now = "OPERATOR AT FACE — BOOM MOVING" if danger else NOW.get(step["verdict"], "Monitoring").upper()
        _right(d, vx + vw - 14, cy, now, f_cap, accent)
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
        ry = safe_box[1] + 92
        for ln in _wrap(d, reason, f_small, rw - 44):
            d.text((rx + 22, ry), ln, font=f_small, fill=INK if danger else MUTED)
            ry += 19
        d.text((rx + 22, safe_box[3] - 26), "Full mesh coverage requires on-site inspection.",
               font=f_small, fill=FAINT)

        # ---- mesh-install card: count + TWO shared-axis timelines (installs above,
        #      danger-zone entries below)
        d.rounded_rectangle(mesh_box, radius=14, fill=SURFACE, outline=HAIR, width=1)
        ax0, ax1 = rx + 22, mesh_box[2] - 22
        _tracked(d, (ax0, mesh_box[1] + 20), "MESH INSTALLATION", f_eye, MUTED, 2)
        d.text((ax0, mesh_box[1] + 36), str(n_mesh), font=_font("b", 44), fill=AMBER)
        numw = d.textlength(str(n_mesh), font=_font("b", 44))
        d.text((ax0 + numw + 14, mesh_box[1] + 50), "meshes installed", font=f_body, fill=INK)
        d.text((ax0 + numw + 14, mesh_box[1] + 72), "estimate", font=f_small, fill=MUTED)

        def _track(y, label, marks, info):
            _tracked(d, (ax0, y - 22), label, f_eye, MUTED, 1)
            _right(d, ax1, y - 21, info, f_small, MUTED)
            d.line([ax0, y, ax1, y], fill=HAIR, width=2)
            for mk in marks:
                if mk["t"] <= csec + 0.1:
                    px = ax0 + min(mk["t"] / t_end, 1.0) * (ax1 - ax0)
                    d.line([px, y - 7, px, y + 7], fill=mk["c"], width=2)
                    d.ellipse([px - 3, y - 3, px + 3, y + 3], fill=mk["c"])
                    if mk.get("lab"):
                        d.text((px - 6, y + 9), mk["lab"], font=f_small, fill=mk["c"])
            ph = ax0 + min(csec / t_end, 1.0) * (ax1 - ax0)
            d.line([ph, y - 11, ph, y + 11], fill=(205, 211, 219), width=1)

        ml = [{"t": i["time"], "lab": f"M{k+1}", "c": AMBER} for k, i in enumerate(installs)]
        _track(mesh_box[1] + 142, "INSTALLS", ml, f"{n_mesh} so far")
        el = [{"t": e["time"], "c": DANGER if e["verdict"] == "NON_COMPLIANT_ENTRY" else WARN}
              for e in entries]
        n_in = sum(1 for e in entries if e["time"] <= csec + 0.1)
        n_dz = sum(1 for e in entries if e["time"] <= csec + 0.1 and e["verdict"] == "NON_COMPLIANT_ENTRY")
        _track(mesh_box[1] + 224, "DANGER-ZONE ENTRIES", el, f"{n_in} entries · {n_dz} boom moving")
        d.text((ax0, mesh_box[3] - 22), "00:00", font=f_mono, fill=FAINT)
        _right(d, ax1, mesh_box[3] - 22, f"{int(t_end)//60:02d}:{int(t_end)%60:02d}", f_mono, FAINT)

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
