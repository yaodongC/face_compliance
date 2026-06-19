"""Scan the full session for operator-in-front events and verify the arm/boom is
fully stopped while the operator works in front of the jumbo.

For each bag (streamed, bounded memory): every `--person-every` seconds run the VLM
person+bbox check; when a worker is in front, compute the person-MASKED arm motion
over the surrounding ~2 s and classify DANGER (arm moving) / OK_LOADING (stopped).

Writes data/operator_events.json and prints a summary.
"""
from __future__ import annotations
import argparse
import collections
import glob
import json
from pathlib import Path
import cv2
import yaml
import operator_safety as osf
from extract_video import iter_frames


def bagpath(base, n):
    g = glob.glob(f"{base}/*_{n}.bag")
    return Path(g[0]) if g else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--base", default="/home/nvidia/rosbags/vale/20260611_115532")
    ap.add_argument("--bags", default="0-56")
    ap.add_argument("--person-every", type=float, default=8.0)
    ap.add_argument("--out", default="data/operator_events.json")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    topic = cfg["camera_topic"]
    lo, hi = (int(x) for x in a.bags.split("-")) if "-" in a.bags else (int(a.bags), int(a.bags))

    events = []
    t0 = None
    frames_dir = Path("data/operator_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)
    for n in range(lo, hi + 1):
        p = bagpath(a.base, n)
        if not p:
            continue
        buf = collections.deque(maxlen=40)   # ~2 s at ~19 fps
        last_check_t = None
        fps_est = 19.0
        for ts, frame in iter_frames([p], topic, 0.0, None):
            if t0 is None:
                t0 = ts
            cyc = (ts - t0) / 1e9
            buf.append(frame)
            if last_check_t is None or (ts - last_check_t) / 1e9 >= a.person_every:
                last_check_t = ts
                per = osf.detect_person(frame, cfg)
                if per["person_in_front"]:
                    bb = per.get("person_bbox")
                    frames = list(buf)
                    mots = [osf.arm_motion(frames[i - 1], frames[i], bb)
                            for i in range(1, len(frames))]
                    peak = max(mots) if mots else 0.0
                    verdict = osf.classify(True, peak)
                    events.append({"bag": n, "cycle_sec": round(cyc, 1),
                                   "verdict": verdict, "arm_motion": round(peak, 3),
                                   "hi_vis": per["hi_vis"], "person_bbox": bb,
                                   "action": per.get("action", ""), "note": per["note"][:60]})
                    ann = osf.annotate(frame, bb, verdict, per.get("action", ""), peak, cyc)
                    cv2.imwrite(str(frames_dir / f"op_{int(cyc):05d}.png"), ann)
                    print(f"[op] bag{n:02d} cycle {int(cyc)//60:02d}:{int(cyc)%60:02d} "
                          f"{verdict:10} motion={peak:.3f} | {per.get('action','')[:40]}", flush=True)
        print(f"  ...bag {n} scanned", flush=True)

    dist = collections.Counter(e["verdict"] for e in events)
    danger = [e for e in events if e["verdict"] == "DANGER"]
    out = {"events": events, "summary": dict(dist),
           "danger_count": len(danger),
           "danger_times": [f"{int(e['cycle_sec'])//60:02d}:{int(e['cycle_sec'])%60:02d}" for e in danger]}
    Path(a.out).write_text(json.dumps(out, indent=2))
    print("\n=== OPERATOR-SAFETY SUMMARY ===")
    print("operator-in-front checks:", len(events), "->", dict(dist))
    print(f"DANGER (operator in front while boom MOVING): {len(danger)}")
    for e in danger:
        print(f"   cycle {int(e['cycle_sec'])//60:02d}:{int(e['cycle_sec'])%60:02d}  arm_motion={e['arm_motion']}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
