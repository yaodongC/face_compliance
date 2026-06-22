"""Render the LoopX Safety AI-Agent (VLM) monitor UI (headless, Pillow-composited).

A mining safety control-room UI: the camera feed is the hero; an Operator-Safety
status card and a Mesh-Install progress card sit to the right; a full-width event-log
timeline runs along the bottom. Calm when clear, the whole frame turns to alarm on a
DANGER (operator under a moving boom).

`compose()` draws one frame from a state dict and is shared by the offline renderer
here and the live monitor (live_gui.py), so both look identical.

Usage (offline, from a recorded MP4 + cached analysis):
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
from PIL import Image, ImageDraw
import yaml
from coverage import mesh_installs
from compliance_checklist import evaluate_checklist
from gui_theme import (BG, SURFACE, DANGER_BG, HAIR, INK, MUTED, FAINT, CLEAR, DANGER,
                       AMBER, WARN, SEV, NOW, build_fonts, geometry,
                       _tracked, _right, _wrap)


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


def _danger_active(dwins, csec):
    return any(a - 0.1 <= csec <= b + 0.1 for a, b in dwins)


def _build_checklist_signals(installs, entries):
    """Assemble the Vale-checklist signals from the fused harness evidence (mesh installs
    from operator tracking, bolt count from the IMU, required counts + completion from the
    compliance milestone). Degrades gracefully if an artifact is absent."""
    sig = {"screen_times": [i["time"] for i in installs],
           "n_screens_req": len(installs) or 4,
           "bolt_times": [], "n_bolts_req": 16,
           "danger_times": [e["time"] for e in entries if e.get("verdict") == "NON_COMPLIANT_ENTRY"],
           "complete_at": None}
    try:
        import progress_tracker as pt
        tg = pt.load_targets()
        sig["n_screens_req"] = tg["meshes_required"]
        sig["n_bolts_req"] = tg["bolts_required"]
        tl = pt.load_evidence()[0]
        sig["bolt_times"] = [b["set_at"] for b in pt.bolt_episodes(tl)]
    except Exception as e:
        print(f"[render_gui] checklist: bolt/target signals unavailable ({e})")
    try:
        res = json.loads(Path("data/compliance_result.json").read_text())
        sig["complete_at"] = res.get("complete_at")
    except Exception:
        pass
    return sig


def _draw_checklist(d, box, items, F):
    """Vale-grounded compliance checklist card: one row per regulation-cited check, with a
    tick + completion time when done; an overall COMPLIANT banner at the bottom."""
    from gui_theme import SURFACE, HAIR, INK, MUTED, FAINT, CLEAR, AMBER
    x0, y0, x1, y1 = box
    d.rounded_rectangle(box, radius=14, fill=SURFACE, outline=HAIR, width=1)
    _tracked(d, (x0 + 22, y0 + 14), "COMPLIANCE CHECKLIST", F["eye"], MUTED, 2)
    _right(d, x1 - 20, y0 + 13, "Vale CMTS-2015-001 / Div 6", F["small"], FAINT)
    rows = [it for it in items if not it.get("overall")]
    overall = next((it for it in items if it.get("overall")), None)
    ry = y0 + 40
    for it in rows:
        done = it["done"]
        bx, by = x0 + 22, ry + 1
        d.rounded_rectangle([bx, by, bx + 13, by + 13], radius=3,
                            fill=CLEAR if done else None, outline=CLEAR if done else HAIR, width=1)
        if done:                                   # drawn check mark (font-independent)
            d.line([bx + 3, by + 7, bx + 6, by + 10], fill=(12, 22, 16), width=2)
            d.line([bx + 6, by + 10, bx + 11, by + 3], fill=(12, 22, 16), width=2)
        d.text((x0 + 44, ry), it["label"], font=F["small"], fill=INK if done else MUTED)
        if done and it.get("done_time") is not None:
            t = int(it["done_time"])
            _right(d, x1 - 20, ry, f"{t//60:02d}:{t%60:02d}", F["small"], CLEAR)
        elif done:
            _right(d, x1 - 20, ry, "ok", F["small"], CLEAR)
        else:
            _right(d, x1 - 20, ry, it.get("detail", ""), F["small"],
                   AMBER if it.get("detail") else MUTED)
        ry += 17
    if overall is not None:
        oy = y1 - 30
        odone = overall["done"]
        d.rounded_rectangle([x0 + 14, oy, x1 - 14, oy + 23], radius=6,
                            fill=(12, 42, 28) if odone else (44, 36, 12))
        nd = sum(1 for it in rows if it["done"])
        if odone and overall.get("done_time") is not None:
            t = int(overall["done_time"])
            txt = f"FACE SUPPORT COMPLIANT  ·  {t//60:02d}:{t%60:02d}"
        else:
            txt = f"IN PROGRESS  ·  {nd}/{len(rows)} checks"
        d.text((x0 + 26, oy + 3), txt, font=F["body"], fill=CLEAR if odone else AMBER)


def compose(frame_bgr, state, F, g):
    """Draw one monitor frame. state keys: csec, danger, n_mesh, installs, entries,
    events, activity, t_end, blink. Returns a BGR ndarray."""
    W, H, M = g["W"], g["H"], g["M"]
    vx, vy, vw, vh = g["vx"], g["vy"], g["vw"], g["vh"]
    rx, rw = g["rx"], g["rw"]
    safe_box, mesh_box, log_box = g["safe_box"], g["mesh_box"], g["log_box"]
    checklist_box = g["checklist_box"]
    csec, danger, t_end = state["csec"], state["danger"], max(state["t_end"], 1.0)
    installs, entries, events = state["installs"], state["entries"], state["events"]
    n_mesh = state["n_mesh"]
    accent = DANGER if danger else CLEAR

    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    if danger:
        d.rectangle([0, 0, W, 4], fill=DANGER)

    # header
    d.rounded_rectangle([M, 30, M + 12, 42], radius=3, fill=AMBER)
    d.text((M + 24, 27), "LoopX Safety AI-Agent (VLM)", font=F["title"], fill=INK)
    cs = int(csec)
    _right(d, W - M, 30, f"CYCLE {cs//60:02d}:{cs%60:02d}", F["monob"], INK)
    _right(d, W - M - 132, 31, "● REC", F["small"], DANGER if (state["blink"] // 8) % 2 else (120, 40, 40))
    d.line([M, 68, W - M, 68], fill=HAIR, width=1)

    # camera feed (hero)
    fr = cv2.cvtColor(cv2.resize(frame_bgr, (vw, vh)), cv2.COLOR_BGR2RGB)
    img.paste(Image.fromarray(fr), (vx, vy))
    ch = 46
    strip = Image.new("RGBA", (vw, ch), (6, 9, 13, 212))
    img.paste(strip, (vx, vy + vh - ch), strip)
    cy = vy + vh - 33
    d.rounded_rectangle([vx + 14, cy + 4, vx + 20, cy + 16], radius=2, fill=AMBER)
    d.text((vx + 30, cy), "FRONT RGB · END FACE", font=F["cap"], fill=INK)
    now = "OPERATOR AT FACE — BOOM MOVING" if danger else state["activity"].upper()
    _right(d, vx + vw - 14, cy, now, F["cap"], accent)
    d.rectangle([vx - 2, vy - 2, vx + vw + 1, vy + vh + 1], outline=accent, width=3 if danger else 2)

    # operator-safety card
    d.rounded_rectangle(safe_box, radius=14, fill=DANGER_BG if danger else SURFACE,
                        outline=DANGER if danger else HAIR, width=1)
    d.rounded_rectangle([safe_box[0], safe_box[1] + 14, safe_box[0] + 4, safe_box[3] - 14],
                        radius=2, fill=accent)
    _tracked(d, (rx + 22, safe_box[1] + 22), "OPERATOR SAFETY", F["eye"], MUTED, 2)
    d.text((rx + 20, safe_box[1] + 44), "DANGER" if danger else "CLEAR", font=F["verdict"], fill=accent)
    reason = ("Operator in the danger zone while the boom is still moving — drilling must stop."
              if danger else "No operator under a moving boom. Bolting cycle proceeding normally.")
    ry = safe_box[1] + 92
    for ln in _wrap(d, reason, F["small"], rw - 44):
        d.text((rx + 22, ry), ln, font=F["small"], fill=INK if danger else MUTED)
        ry += 19
    d.text((rx + 22, safe_box[3] - 26), "Full mesh coverage requires on-site inspection.",
           font=F["small"], fill=FAINT)

    # mesh-install card: count + two shared-axis timelines
    d.rounded_rectangle(mesh_box, radius=14, fill=SURFACE, outline=HAIR, width=1)
    ax0, ax1 = rx + 22, mesh_box[2] - 22
    _tracked(d, (ax0, mesh_box[1] + 20), "MESH INSTALLATION", F["eye"], MUTED, 2)
    d.text((ax0, mesh_box[1] + 36), str(n_mesh), font=F["num44"], fill=AMBER)
    numw = d.textlength(str(n_mesh), font=F["num44"])
    d.text((ax0 + numw + 14, mesh_box[1] + 50), "meshes installed", font=F["body"], fill=INK)
    d.text((ax0 + numw + 14, mesh_box[1] + 72), "estimate", font=F["small"], fill=MUTED)

    def _track(y, label, marks, info):
        _tracked(d, (ax0, y - 22), label, F["eye"], MUTED, 1)
        _right(d, ax1, y - 21, info, F["small"], MUTED)
        d.line([ax0, y, ax1, y], fill=HAIR, width=2)
        for mk in marks:
            if mk["t"] <= csec + 0.1:
                px = ax0 + min(mk["t"] / t_end, 1.0) * (ax1 - ax0)
                d.line([px, y - 7, px, y + 7], fill=mk["c"], width=2)
                d.ellipse([px - 3, y - 3, px + 3, y + 3], fill=mk["c"])
                if mk.get("lab"):
                    d.text((px - 6, y + 9), mk["lab"], font=F["small"], fill=mk["c"])
        ph = ax0 + min(csec / t_end, 1.0) * (ax1 - ax0)
        d.line([ph, y - 11, ph, y + 11], fill=(205, 211, 219), width=1)

    ml = [{"t": i["time"], "lab": f"M{k+1}", "c": AMBER} for k, i in enumerate(installs)]
    _track(mesh_box[1] + 142, "INSTALLS", ml, f"{n_mesh} so far")
    el = [{"t": e["time"], "c": DANGER if e["verdict"] == "NON_COMPLIANT_ENTRY" else WARN}
          for e in entries]
    n_in = sum(1 for e in entries if e["time"] <= csec + 0.1)
    n_dz = sum(1 for e in entries if e["time"] <= csec + 0.1 and e["verdict"] == "NON_COMPLIANT_ENTRY")
    _track(mesh_box[1] + 224, "DANGER-ZONE ENTRIES", el, f"{n_in} entries · {n_dz} boom moving")
    d.text((ax0, mesh_box[3] - 22), "00:00", font=F["mono"], fill=FAINT)
    _right(d, ax1, mesh_box[3] - 22, f"{int(t_end)//60:02d}:{int(t_end)%60:02d}", F["mono"], FAINT)

    # event-log timeline (narrowed to the camera width; checklist card sits to its right)
    lx0, lx1 = log_box[0], log_box[2]
    d.rounded_rectangle(log_box, radius=14, fill=SURFACE, outline=HAIR, width=1)
    _tracked(d, (lx0 + 22, log_box[1] + 20), "EVENT LOG", F["eye"], MUTED, 2)
    nviol = sum(1 for e in events if e.get("severity") == "VIOLATION" and e.get("cycle_sec", 0) <= csec + 0.1)
    _right(d, lx1 - 22, log_box[1] + 16, f"{nviol} violations", F["small"], DANGER if nviol else MUTED)
    shown = [e for e in events if e.get("cycle_sec", 0) <= csec + 0.1][-5:]
    ey = log_box[1] + 48
    logw = int(lx1 - lx0)
    for e in shown:
        ec = int(e.get("cycle_sec", 0))
        col = SEV.get(e.get("severity", "INFO"), MUTED)
        d.text((lx0 + 22, ey), f"{ec//60:02d}:{ec%60:02d}", font=F["monob"], fill=MUTED)
        d.ellipse([lx0 + 92, ey + 5, lx0 + 102, ey + 15], fill=col)
        d.text((lx0 + 116, ey), e.get("description", "")[:max(20, (logw - 140) // 8)], font=F["body"],
               fill=INK if e.get("severity") in ("VIOLATION", "WARNING") else MUTED)
        ey += 22

    # compliance-checklist card (Vale-grounded) — below the mesh-installation card
    _draw_checklist(d, checklist_box, state.get("checklist", []), F)

    _tracked(d, (M, H - 22), "ASSISTIVE MONITOR  ·  NOT A CERTIFIED SAFETY SYSTEM  ·  VERIFY ON SITE",
             F["small"], FAINT, 1)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


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
    from operator_safety import classify_sessions
    entries = [{"time": s["start"], "end": s["end"], "verdict": s["verdict"]}
               for s in classify_sessions(ops)] if ops else []
    dwins = [(e["time"] - 1, max(e["end"], e["time"] + 13)) for e in entries
             if e["verdict"] == "NON_COMPLIANT_ENTRY"]
    checklist_sig = _build_checklist_signals(installs, entries)

    cap = cv2.VideoCapture(video)
    vfps = cap.get(cv2.CAP_PROP_FPS) or fps
    t_end = max((list(index.values()) if index else [s["t_sec"] for s in steps]) or [1.0])
    F, g = build_fonts(), geometry()
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (g["W"], g["H"]))

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
        danger = _danger_active(dwins, csec)
        state = {"csec": csec, "danger": danger, "t_end": t_end,
                 "n_mesh": sum(1 for i in installs if i["time"] <= csec + 0.1),
                 "installs": installs, "entries": entries, "events": events,
                 "activity": NOW.get(cur_step(t)["verdict"], "Monitoring"), "blink": fno,
                 "checklist": evaluate_checklist(checklist_sig, csec)}
        writer.write(compose(frame, state, F, g))
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
