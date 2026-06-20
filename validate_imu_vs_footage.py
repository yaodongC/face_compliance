"""Validate the IMU interpretation against the actual video.

Claim to test: high IMU accel-energy == machine/boom actually moving; low == still.
For bag 25 (IMU-LOUD window incl. cyc 1497/1529/1537) and bag 37 (IMU-QUIET, cyc 2225):
  - print the IMU accel-std timeline in 1s bins (global cyc)
  - save a contact sheet of frames across each target moment so we can SEE if the boom moves
"""
from __future__ import annotations
import glob
from pathlib import Path
import cv2, numpy as np
from rosbags.highlevel import AnyReader
from extract_video import decode_compressed

BASE = "/home/nvidia/rosbags/vale/20260611_115532"
CAM = "/sensing/front/rgb/image_raw/compressed"
IMU = "/sensing/front/livox/imu"


def bag(n):
    g = glob.glob(f"{BASE}/*_{n}.bag"); return g[0] if g else None


def cam_t0_bag0():
    with AnyReader([Path(bag(0))]) as r:
        for c, ts, raw in r.messages(connections=[c for c in r.connections if c.topic == CAM]):
            return ts


def imu_bins(n, t0, bin_s=1.0):
    cyc, mag = [], []
    with AnyReader([Path(bag(n))]) as r:
        for c, ts, raw in r.messages(connections=[c for c in r.connections if c.topic == IMU]):
            m = r.deserialize(raw, c.msgtype); la = m.linear_acceleration
            cyc.append((ts - t0) / 1e9); mag.append((la.x**2 + la.y**2 + la.z**2) ** 0.5)
    cyc, mag = np.array(cyc), np.array(mag)
    bins = {}
    for ti, mi in zip(cyc, mag):
        bins.setdefault(int(ti // bin_s), []).append(mi)
    return {k * bin_s: float(np.std(v)) for k, v in sorted(bins.items())}


def burst(n, t0, center_cyc, half=1.5, k=6, tag=""):
    frames = []
    with AnyReader([Path(bag(n))]) as r:
        for c, ts, raw in r.messages(connections=[c for c in r.connections if c.topic == CAM]):
            cyc = (ts - t0) / 1e9
            if cyc < center_cyc - half:
                continue
            if cyc > center_cyc + half:
                break
            frames.append((cyc, decode_compressed(bytes(r.deserialize(raw, c.msgtype).data))))
    if len(frames) < 2:
        print(f"  burst {tag}: only {len(frames)} frames"); return
    idx = np.linspace(0, len(frames) - 1, min(k, len(frames))).round().astype(int)
    tiles = []
    for i in idx:
        cyc, fr = frames[i]
        fr = cv2.resize(fr, (360, 203))
        cv2.putText(fr, f"{cyc:.2f}s", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        tiles.append(fr)
    out = f"data/operator_frames/VAL_{tag}.png"
    cv2.imwrite(out, np.hstack(tiles))
    # also report consecutive full-frame diff (whole frame, no ROI) across the burst
    g = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for _, f in frames]
    d = [float((cv2.absdiff(g[i - 1], g[i]) > 25).mean()) for i in range(1, len(g))]
    print(f"  burst {tag}: {len(frames)} frames, full-frame motion med={np.median(d):.3f} max={np.max(d):.3f} -> {out}")


def main():
    t0 = cam_t0_bag0()
    for n, label in ((25, "LOUD"), (37, "QUIET")):
        b = imu_bins(n, t0)
        ks = sorted(b)
        print(f"\n=== bag {n} ({label}) IMU accel-std per 1s (global cyc {ks[0]:.0f}-{ks[-1]:.0f}) ===")
        print("  " + " ".join(f"{b[k]:.3f}" for k in ks))
    print("\n--- bursts ---")
    burst(25, t0, 1497, tag="b25_cyc1497_imuLOUD_visOK")
    burst(25, t0, 1531, tag="b25_cyc1531_imuLOUD_visDANGER")
    burst(37, t0, 2225, tag="b37_cyc2225_imuQUIET_visDANGER")


if __name__ == "__main__":
    main()
