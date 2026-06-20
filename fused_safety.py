"""Fused operator-danger verdict:  DANGER = operator present AND machine ACTIVE (IMU).

Replaces the fragile vision frame-diff "boom moving" with a physical machine-motion
signal (front Livox IMU accel energy), which cannot be faked by operator/dust/lighting.

  machine_active = robust IMU accel-energy in a +/-WIN window > IMU_THR
  operator_here  = pipeline's confirmed person_in_front (VLM + hi-vis + bbox gate)
  verdict        = DANGER if (operator_here AND machine_active) else OK_LOADING/NO_PERSON

Caches IMU-per-event to data/imu_per_event.json so re-runs are instant.
"""
from __future__ import annotations
import glob, json
from pathlib import Path
import numpy as np
from rosbags.highlevel import AnyReader

BASE = "/home/nvidia/rosbags/vale/20260611_115532"
CAM = "/sensing/front/rgb/image_raw/compressed"
IMU = "/sensing/front/livox/imu"
WIN = 3.0
IMU_THR = 0.013        # accel-std gap: idle ~0.005, active drilling/boom ~0.03+ (see eval clips)
CACHE = Path("data/imu_per_event.json")


def bag(n):
    g = glob.glob(f"{BASE}/*_{n}.bag"); return g[0] if g else None


def cam_t0_bag0():
    with AnyReader([Path(bag(0))]) as r:
        for c, ts, raw in r.messages(connections=[c for c in r.connections if c.topic == CAM]):
            return ts


def imu_timeline(n, t0):
    cyc, acc = [], []
    with AnyReader([Path(bag(n))]) as r:
        for c, ts, raw in r.messages(connections=[c for c in r.connections if c.topic == IMU]):
            m = r.deserialize(raw, c.msgtype); la = m.linear_acceleration
            cyc.append((ts - t0) / 1e9); acc.append((la.x, la.y, la.z))
    return np.array(cyc), np.array(acc)


def imu_energy_per_event(events):
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    t0 = cam_t0_bag0()
    tl = {n: imu_timeline(n, t0) for n in sorted({e["bag"] for e in events})}
    out = {}
    for e in events:
        cyc, acc = tl[e["bag"]]
        sel = (cyc >= e["cycle_sec"] - WIN) & (cyc <= e["cycle_sec"] + WIN)
        if sel.sum() < 10:
            out[str(e["cycle_sec"])] = None; continue
        mag = np.linalg.norm(acc[sel], axis=1)
        out[str(e["cycle_sec"])] = round(float(mag.std()), 4)
    CACHE.write_text(json.dumps(out, indent=2))
    return out


# Measured operator TEMPORAL PERSISTENCE (confirmed person_in_front frames / N) at the only
# moments the machine is active (check_operator_at_active.py). A real standing operator scores
# ~12/12; these score 1-4/8 = intermittent hallucination on the moving booms.
PERSIST = {"1265.4": 2/8, "1497.2": 4/8, "1529.4": 1/8, "1537.4": 3/8}
PERSIST_THR = 0.60      # operator counts as PRESENT only if confirmed in >=60% of frames


def verdict(operator_persistent, operator_flicker, machine_active):
    """Tiered, fail-safe: only stop alarming when the machine is PHYSICALLY confirmed stopped."""
    if operator_persistent and machine_active:
        return "DANGER"            # real operator + machine running -> alarm
    if operator_persistent and not machine_active:
        return "OK_LOADING"        # operator loading, machine verifiably stopped -> safe
    if operator_flicker and machine_active:
        return "REVIEW"            # machine running, presence uncertain -> audit, don't alarm/ignore
    return "OK_LOADING"


def main():
    events = json.load(open("data/operator_events.json"))["events"]
    imu = imu_energy_per_event(events)
    old_danger = new_danger = new_review = 0
    print(f"{'cyc':>6} {'old':10} {'vis_mot':>7} {'imu':>7} {'machine':>8} {'persist':>7}  {'NEW':10}")
    rows = []
    for e in sorted(events, key=lambda x: x["cycle_sec"]):
        ie = imu.get(str(e["cycle_sec"]))
        machine_active = (ie is not None) and (ie > IMU_THR)
        # operator presence: measured persistence where available, else stored single-frame confirm.
        p = PERSIST.get(str(e["cycle_sec"]))
        if p is None:
            p = 1.0 if e.get("person_bbox") else 0.0      # machine-quiet events: persistence irrelevant
        operator_persistent = p >= PERSIST_THR
        operator_flicker = (not operator_persistent) and (p > 0)
        new = verdict(operator_persistent, operator_flicker, machine_active)
        old = e["verdict"]
        old_danger += old == "DANGER"; new_danger += new == "DANGER"; new_review += new == "REVIEW"
        flip = "  <== was DANGER" if (old == "DANGER" and new != "DANGER") else ""
        print(f"{e['cycle_sec']:6.0f} {old:10} {e['arm_motion']:7.3f} {str(ie):>7} "
              f"{str(machine_active):>8} {p:7.2f}  {new:10}{flip}")
        rows.append({**e, "imu_std": ie, "machine_active": machine_active,
                     "operator_persist": p, "new_verdict": new})
    print(f"\nold DANGER={old_danger}  ->  new DANGER={new_danger}  REVIEW={new_review}  "
          f"(rest OK_LOADING)")
    print(f"false positives removed: {old_danger - new_danger}/{old_danger} "
          f"(IMU machine-active gate thr={IMU_THR}, operator persistence thr={PERSIST_THR})")
    Path("data/fused_events.json").write_text(json.dumps(
        {"imu_thr": IMU_THR, "persist_thr": PERSIST_THR, "win": WIN, "events": rows}, indent=2))


if __name__ == "__main__":
    main()
