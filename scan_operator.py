"""Scan the full session for operator-in-front events and verify the machine is fully
stopped while the operator works in front of the jumbo.

For each bag (streamed, bounded memory): every `--person-every` seconds run the VLM
person+bbox check. When a worker is detected in front we judge the danger with TWO
robust facts, NOT the old vision frame-diff (which fired on the operator walking / dust /
lighting -- 14/17 of its DANGERs were machine-quiet false positives):

  * operator PRESENT  -- the VLM person must PERSIST across `person_persist_n` nearby frames
    (a real worker is in ~all of them; a boom-hallucination flickers in 1-4/8).
  * machine ACTIVE    -- the front Livox IMU accelerometer energy in a +/-imu_win_sec window
    (the jumbo physically vibrates when drilling/booming; idle ~0.005, active ~0.03+).

  verdict = classify_zone(present, seen, machine_active):
    DANGER (persistent operator + machine running), OK_LOADING (operator + machine stopped),
    REVIEW (machine running + flickering presence -> audit), NO_PERSON.

Writes data/operator_events.json and prints a summary.
"""
from __future__ import annotations
import argparse
import collections
import glob
import json
from pathlib import Path
import cv2
import numpy as np
import yaml
from rosbags.highlevel import AnyReader
import operator_safety as osf
from extract_video import iter_frames


def bagpath(base, n):
    g = glob.glob(f"{base}/*_{n}.bag")
    return Path(g[0]) if g else None


def read_imu(bag, topic):
    """Read the IMU topic for a bag -> (ts_ns array, accel (N,3) array). Reads only the IMU
    connection (no image decode), so it is cheap relative to the frame stream."""
    ts, acc = [], []
    try:
        with AnyReader([Path(bag)]) as r:
            conns = [c for c in r.connections if c.topic == topic]
            for c, t, raw in r.messages(connections=conns):
                m = r.deserialize(raw, c.msgtype)
                la = m.linear_acceleration
                ts.append(t); acc.append((la.x, la.y, la.z))
    except Exception as e:                       # missing/!damaged IMU must not crash the scan
        print(f"[imu] {Path(bag).name}: {e}")
    return np.array(ts, dtype=np.int64), np.array(acc, dtype=float)


def machine_active_at(imu_ts, imu_acc, ts_ns, win_sec):
    """IMU machine-motion std in [ts-win, ts+win]; returns (std, active_bool)."""
    if imu_ts.size == 0:
        return 0.0, False
    w = int(win_sec * 1e9)
    sel = (imu_ts >= ts_ns - w) & (imu_ts <= ts_ns + w)
    std = osf.machine_motion(imu_acc[sel]) if sel.sum() >= 2 else 0.0
    return std, std > osf.IMU_ACTIVE_THR


def operator_persistence(frames, cfg, n):
    """Re-run detect_person on up to n evenly-spaced frames -> fraction confirmed in front."""
    if not frames:
        return 0.0
    idx = np.linspace(0, len(frames) - 1, min(n, len(frames))).round().astype(int)
    hits = sum(osf.detect_person(frames[i], cfg)["person_in_front"] for i in idx)
    return hits / len(idx)


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
        imu_ts, imu_acc = read_imu(p, osf.IMU_TOPIC)     # physical machine-motion for this bag
        buf = collections.deque(maxlen=40)               # ~2 s at ~19 fps
        last_check_t = None
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
                    frames_l = list(buf)
                    # legacy vision motion (audit + IMU-missing fallback)
                    mots = [osf.arm_motion(frames_l[i - 1], frames_l[i], bb)
                            for i in range(1, len(frames_l))]
                    peak = max(mots) if mots else 0.0
                    # robust facts: persistent operator + physical machine motion
                    persist = operator_persistence(frames_l, cfg, osf.PERSON_PERSIST_N)
                    present, seen = osf.operator_present(persist)
                    if imu_ts.size > 0:
                        imu_std, m_active = machine_active_at(imu_ts, imu_acc, ts, osf.IMU_WIN_SEC)
                        degraded = False
                    else:
                        # FAIL-SAFE: no IMU for this bag -> fall back to the legacy vision motion
                        # so safety DEGRADES (still able to raise DANGER) rather than silently
                        # disabling it (machine_active stuck False would never alarm).
                        imu_std, m_active, degraded = -1.0, (peak > osf.MOTION_FRAC_THRESH), True
                    verdict = osf.classify_zone(present, seen, m_active)
                    events.append({"bag": n, "cycle_sec": round(cyc, 1),
                                   "verdict": verdict, "imu_std": round(imu_std, 4),
                                   "machine_active": bool(m_active), "imu_degraded": degraded,
                                   "operator_persist": round(persist, 3),
                                   "operator_present": bool(present),
                                   "arm_motion": round(peak, 3),
                                   "hi_vis": per["hi_vis"], "person_bbox": bb,
                                   "action": per.get("action", ""), "note": per["note"][:60]})
                    ann = osf.annotate(frame, bb, verdict, per.get("action", ""), peak, cyc)
                    cv2.imwrite(str(frames_dir / f"op_{int(cyc):05d}.png"), ann)
                    print(f"[op] bag{n:02d} cycle {int(cyc)//60:02d}:{int(cyc)%60:02d} "
                          f"{verdict:10} imu={imu_std:.3f}({'ACTIVE' if m_active else 'idle'}) "
                          f"persist={persist:.2f} | {per.get('action','')[:34]}", flush=True)
        print(f"  ...bag {n} scanned", flush=True)

    dist = collections.Counter(e["verdict"] for e in events)
    danger = [e for e in events if e["verdict"] == "DANGER"]
    review = [e for e in events if e["verdict"] == "REVIEW"]
    out = {"events": events, "summary": dict(dist),
           "danger_count": len(danger), "review_count": len(review),
           "danger_times": [f"{int(e['cycle_sec'])//60:02d}:{int(e['cycle_sec'])%60:02d}" for e in danger]}
    Path(a.out).write_text(json.dumps(out, indent=2))
    print("\n=== OPERATOR-SAFETY SUMMARY (IMU-fused) ===")
    print("operator-in-front checks:", len(events), "->", dict(dist))
    print(f"DANGER (persistent operator in front while machine ACTIVE): {len(danger)}")
    for e in danger:
        print(f"   cycle {int(e['cycle_sec'])//60:02d}:{int(e['cycle_sec'])%60:02d}  "
              f"imu={e['imu_std']} persist={e['operator_persist']}")
    print(f"REVIEW (machine active, presence uncertain): {len(review)}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
