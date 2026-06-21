"""Classify each IMU work-window with the VLM — the fusion step.

The IMU says WHEN the machine drilled (16 sustained episodes); the VLM looks at frames
in each window and says WHAT it was (bolt_install / screen_load / production_drill /
mucking / ...). This (a) cross-checks the IMU "16 bolts" claim with vision, and (b)
produces a veto map so progress_tracker drops any sustained episode the VLM judges is
NOT face work — keeping the bolt count honest and fused, not IMU-only.

  python3 classify_episodes.py [--out data/episode_class.json]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import yaml
import imu_analyzer as ia
import progress_tracker as pt
import vlm_client
from cycle_frames import FrameGrabber


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--imu", default="data/imu_timeline.json")
    ap.add_argument("--out", default="data/episode_class.json")
    ap.add_argument("--frames", type=int, default=3)
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    tl = ia.load_timeline(a.imu)
    bolts = pt.bolt_episodes(tl)
    grab = FrameGrabber()
    out = []
    BOLTISH = {"bolt_install", "screen_load", "scaling"}   # legitimate face-support work
    for i, b in enumerate(bolts, 1):
        mid = (b["start"] + b["set_at"]) / 2.0
        frames = grab.around(mid, n=a.frames, span=min(b["dur"], 30.0))
        c = vlm_client.classify_episode(frames, cfg)
        c["start"] = b["start"]
        c["mid"] = round(mid, 1)
        c["is_face_work"] = c["activity"] in BOLTISH or c["at_face"]
        out.append(c)
        m = int(b["start"])
        print(f"  ep{i:2d} {m//60:02d}:{m%60:02d}  activity={c['activity']:16s} "
              f"at_face={c['at_face']} plate={c['plate_visible']} region={c['region']} "
              f"-> face_work={c['is_face_work']}", flush=True)
    grab.release()
    n_face = sum(1 for c in out if c["is_face_work"])
    res = {"episodes": out, "n_episodes": len(out), "n_face_work": n_face}
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"\n{n_face}/{len(out)} IMU episodes VLM-confirmed as face-support work -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
