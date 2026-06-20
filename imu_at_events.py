"""Does machine IMU energy separate the FALSE-POSITIVE operator-DANGER events from reality?

For each operator event (data/operator_events.json), compute the front-IMU acceleration
energy in a +/-WIN window around the event, aligned to the SAME global clock the events use
(cam t0 = first front-camera frame of bag 0). If the known false positives (cyc=1529, 2225)
are IMU-quiet while the pipeline called them DANGER, IMU is a real machine-motion veto.
"""
from __future__ import annotations
import glob, json
from pathlib import Path
import numpy as np
from rosbags.highlevel import AnyReader

BASE = "/home/nvidia/rosbags/vale/20260611_115532"
CAM = "/sensing/front/rgb/image_raw/compressed"
IMU = "/sensing/front/livox/imu"
WIN = 3.0  # seconds each side


def bag(n):
    g = glob.glob(f"{BASE}/*_{n}.bag")
    return g[0] if g else None


def cam_t0_bag0():
    with AnyReader([Path(bag(0))]) as r:
        cc = [c for c in r.connections if c.topic == CAM]
        for c, ts, raw in r.messages(connections=cc):
            return ts
    return None


def imu_timeline(n, t0):
    """Return (global_cyc[], accel_xyz[]) for bag n on the global clock."""
    cyc, acc = [], []
    with AnyReader([Path(bag(n))]) as r:
        ic = [c for c in r.connections if c.topic == IMU]
        for c, ts, raw in r.messages(connections=ic):
            m = r.deserialize(raw, c.msgtype)
            cyc.append((ts - t0) / 1e9)
            la = m.linear_acceleration
            acc.append((la.x, la.y, la.z))
    return np.array(cyc), np.array(acc)


def main():
    ev = json.load(open("data/operator_events.json"))["events"]
    t0 = cam_t0_bag0()
    bags = sorted({e["bag"] for e in ev})
    tl = {}
    for n in bags:
        tl[n] = imu_timeline(n, t0)
    print(f"{'cyc':>6} {'bag':>3} {'verdict':10} {'vis_mot':>7} {'imu_std':>8} {'imu_p95':>8}  note")
    rows = []
    for e in sorted(ev, key=lambda x: x["cycle_sec"]):
        cyc, acc = tl[e["bag"]]
        if len(cyc) == 0:
            continue
        sel = (cyc >= e["cycle_sec"] - WIN) & (cyc <= e["cycle_sec"] + WIN)
        if sel.sum() < 10:
            astd = ap95 = float("nan")
        else:
            mag = np.linalg.norm(acc[sel], axis=1)
            astd = float(mag.std())
            # high-freq: deviation from local mean, 95th pct
            ap95 = float(np.percentile(np.abs(mag - mag.mean()), 95))
        tag = ""
        if int(e["cycle_sec"]) in (1529, 2225):
            tag = "<-- KNOWN FALSE POSITIVE"
        print(f"{e['cycle_sec']:6.0f} {e['bag']:3d} {e['verdict']:10} {e['arm_motion']:7.3f} "
              f"{astd:8.4f} {ap95:8.4f}  {tag}")
        rows.append((e["verdict"], e["arm_motion"], astd))
    # summary: IMU energy distribution by pipeline verdict
    import statistics as st
    for v in ("DANGER", "OK_LOADING"):
        xs = [a for vv, m, a in rows if vv == v and a == a]
        if xs:
            print(f"\n{v}: n={len(xs)} imu_std min={min(xs):.4f} median={st.median(xs):.4f} max={max(xs):.4f}")


if __name__ == "__main__":
    main()
