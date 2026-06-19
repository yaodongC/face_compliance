"""Live LoopX Safety monitor: read from a CONFIGURABLE input (MP4 file OR RTSP
camera) and run the harness perception in real time, rendering the same GUI.

The input is set in config.yaml (`input:`) or with --input; both file paths and
rtsp:// URLs work via live_source. Perception (the slow VLM + operator detection)
runs in a background thread so the feed stays smooth; the main loop composes the GUI
at display rate from the latest shared state.

  python3 live_gui.py                      # uses config.yaml `input:`
  export RTSP_USER=... RTSP_PASS=...        # credentials via env, never committed
  python3 live_gui.py --input 'rtsp://${RTSP_USER}:${RTSP_PASS}@10.20.30.40:554/cam0_0' --display
  python3 live_gui.py --input data/full_cycle.mp4 --seconds 60 --out data/live.mp4
"""
from __future__ import annotations
import argparse
import os
import threading
import time
from pathlib import Path
import cv2
import requests
import yaml
from live_source import open_source
from render_gui import compose
from gui_theme import build_fonts, geometry, NOW
import vlm_client as V
import operator_safety as osf
from coverage import mesh_installs
from operator_safety import classify_sessions
import event_log as EL


def _activity(p):
    if p.get("drill_active"):
        return NOW["DRILLING"]
    if p.get("arms_parked"):
        return NOW["SUPPORTED"]
    if not p.get("face_screened"):
        return NOW["UNSUPPORTED"]
    return NOW["NOT VERIFIED"]


class LiveState:
    def __init__(self, events_path):
        self.lock = threading.Lock()
        self.frame = None
        self.activity = "Assessing face support"
        self.ops, self.installs, self.entries, self.dwins, self.events = [], [], [], [], []
        self.lg = EL.EventLogger(events_path, reset=True)


def perception_worker(state, cfg, t0, every, stop):
    sess = requests.Session()
    prev = None
    while not stop.is_set():
        with state.lock:
            frame = None if state.frame is None else state.frame.copy()
        if frame is None:
            time.sleep(0.2)
            continue
        csec = time.time() - t0
        try:
            perc = V.analyze_window([frame], cfg, session=sess)
        except Exception:
            perc = {}
        try:
            op = osf.detect_person(frame, cfg, session=sess)
        except Exception:
            op = {"person_in_front": False, "person_bbox": None}
        motion = osf.arm_motion(prev, frame, op.get("person_bbox")) if prev is not None else 0.0
        prev = frame
        moving = motion > osf.MOTION_FRAC_THRESH
        with state.lock:
            if perc:
                state.activity = _activity(perc)
            if op.get("person_in_front"):
                state.ops.append({"cycle_sec": csec, "person_bbox": op["person_bbox"],
                                  "arm_motion": motion,
                                  "verdict": "DANGER" if moving else "OK_LOADING",
                                  "action": op.get("action", "")})
                state.entries = [{"time": s["start"], "end": s["end"], "verdict": s["verdict"]}
                                 for s in classify_sessions(state.ops)]
                state.installs = mesh_installs(state.ops)
                if moving:
                    state.dwins.append((csec - 1, csec + 13))
                    state.lg.log(EL.OPERATOR_IN_ZONE, csec, severity=EL.VIOLATION,
                                 description=f"operator entered danger zone while boom MOVING "
                                             f"(motion {motion:.3f})", started_at=csec)
                else:
                    state.lg.log(EL.OPERATOR_IN_ZONE, csec, severity=EL.INFO,
                                 description="operator entered danger zone to reload — boom stopped")
                state.events = state.lg.events()
        time.sleep(max(0.1, every))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--input", help="MP4 path or rtsp:// URL (overrides config `input`)")
    ap.add_argument("--out", help="record the monitor to this MP4")
    ap.add_argument("--out-fps", type=float, default=12.0)
    ap.add_argument("--seconds", type=float, default=0, help="stop after N s (0 = until stream ends)")
    ap.add_argument("--every", type=float, default=5.0, help="perception interval (s)")
    ap.add_argument("--display", action="store_true", help="cv2.imshow window (needs a display)")
    ap.add_argument("--snapshot", default="data/live_frame.png", help="latest frame for headless monitoring")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    inp = a.input or cfg.get("input")
    if not inp:
        raise SystemExit("no input: set `input:` in config.yaml or pass --input")
    print(f"input: {inp}  ({'live RTSP' if inp.startswith('rtsp://') else 'file'})")

    state = LiveState("data/live_events.jsonl")
    F, g = build_fonts(), geometry()
    src = open_source(inp)
    t0 = time.time()
    stop = threading.Event()
    threading.Thread(target=perception_worker, args=(state, cfg, t0, a.every, stop),
                     daemon=True).start()
    writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.out_fps,
                             (g["W"], g["H"])) if a.out else None
    blink, n = 0, 0
    try:
        for ts, frame in src.frames():
            csec = time.time() - t0
            with state.lock:
                state.frame = frame
                danger = any(lo <= csec <= hi for lo, hi in state.dwins)
                st = {"csec": csec, "danger": danger, "t_end": max(csec, 120.0),
                      "n_mesh": sum(1 for i in state.installs if i["time"] <= csec + 0.1),
                      "installs": list(state.installs), "entries": list(state.entries),
                      "events": list(state.events), "activity": state.activity, "blink": blink}
            canvas = compose(frame, st, F, g)
            if writer:
                writer.write(canvas)
            if a.display:
                cv2.imshow("LoopX Safety AI-Agent (VLM)", canvas)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if a.snapshot and n % 12 == 0:
                cv2.imwrite(a.snapshot, canvas)
            blink += 1
            n += 1
            if a.seconds and csec >= a.seconds:
                break
    finally:
        stop.set()
        src.release()
        if writer:
            writer.release()
        if a.display:
            cv2.destroyAllWindows()
        print(f"stopped after {n} frames ({int(time.time()-t0)}s)")


if __name__ == "__main__":
    main()
