"""IMU machine-activity analyzer — the physical "is the jumbo working" tool.

The front Livox IMU vibrates only when the machine is physically drilling/booming
(idle ~0.005, active ~0.03+ accel-mag std — a clean gap; see operator_safety.machine_motion).
Unlike vision frame-diff it cannot be faked by an operator walking, dust, or lighting.

This module turns the raw 200 Hz IMU stream over the whole cycle into:
  * a per-second machine-motion ENVELOPE (std of accel magnitude over +/-win),
  * discrete machine-ACTIVE EPISODES (contiguous active runs, small gaps merged),

An episode is one sustained burst of machine work — drilling a bolt hole, setting a
bolt, slewing a boom. It is the physical, IMU-grounded counter the harness uses for
"bolting progress" (each support bolt requires a drilled hole = one drilling episode)
and the machine-active gate for operator danger. WHICH episodes are bolt-installs vs
production face-drilling is decided by fusing with the camera/VLM — the IMU only says
the machine was physically running, and for how long.

CLI:  python3 imu_analyzer.py [--bags 0-56] [--out data/imu_timeline.json]
Tool: load_timeline(); machine_active_at(tl, t); episodes_until(tl, t)
"""
from __future__ import annotations
import argparse
import glob
import json
from pathlib import Path
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.highlevel.anyreader import AnyReaderError
import operator_safety as osf

DEFAULT_BASE = "/home/nvidia/rosbags/vale/20260611_115532"
CAM_TOPIC = "/sensing/front/rgb/image_raw/compressed"

# episode segmentation defaults (seconds)
ENV_WIN = 1.5        # +/- window for the per-second motion envelope (finer than the 3 s danger win)
ENV_STEP = 1.0       # envelope resolution
MERGE_GAP = 5.0      # gaps <= this between active seconds are one episode (drill->set pause)
MIN_DUR = 4.0        # drop bursts shorter than this (transients / single jolts)


def bagpath(base, n):
    g = sorted(glob.glob(f"{base}/*_{n}.bag"))
    return g[0] if g else None


def first_image_ts(bag) -> int | None:
    """Global cycle t0 = first front-camera frame ts (matches extract_video / the GUI clock)."""
    try:
        with AnyReader([Path(bag)]) as r:
            conns = [c for c in r.connections if c.topic == CAM_TOPIC]
            for _c, t, _raw in r.messages(connections=conns):
                return int(t)
    except AnyReaderError:
        return None
    return None


def read_imu(bag, topic=osf.IMU_TOPIC):
    """(ts_ns, accel(N,3)) for one bag's IMU. Reads only the IMU connection (cheap)."""
    ts, acc = [], []
    try:
        with AnyReader([Path(bag)]) as r:
            conns = [c for c in r.connections if c.topic == topic]
            for c, t, raw in r.messages(connections=conns):
                m = r.deserialize(raw, c.msgtype)
                la = m.linear_acceleration
                ts.append(t)
                acc.append((la.x, la.y, la.z))
    except (AnyReaderError, Exception) as e:        # missing/damaged IMU must not crash the scan
        print(f"[imu] {Path(bag).name}: {e}")
    return np.array(ts, dtype=np.int64), np.array(acc, dtype=float)


def envelope(cyc_s, accel, win=ENV_WIN, step=ENV_STEP):
    """Per-`step`-second machine-motion envelope: std of accel-magnitude in [t-win, t+win].

    cyc_s: (N,) sample times in cycle seconds; accel: (N,3). Returns (t, std) arrays."""
    if cyc_s.size < 2:
        return np.array([]), np.array([])
    mag = np.linalg.norm(accel, axis=1)
    lo, hi = int(np.floor(cyc_s.min())), int(np.ceil(cyc_s.max()))
    ts, sd = [], []
    order = np.argsort(cyc_s)
    cyc_s, mag = cyc_s[order], mag[order]
    for s in np.arange(lo, hi + 1, step):
        i0 = np.searchsorted(cyc_s, s - win)
        i1 = np.searchsorted(cyc_s, s + win)
        if i1 - i0 >= 2:
            ts.append(float(s))
            sd.append(float(mag[i0:i1].std()))
    return np.array(ts), np.array(sd)


def segment_episodes(t, std, thr=None, merge_gap=MERGE_GAP, min_dur=MIN_DUR):
    """Discrete machine-active episodes from the envelope.

    active = std > thr; contiguous active seconds within `merge_gap` are one episode;
    episodes shorter than `min_dur` are dropped (transients). Each episode is a sustained
    burst of machine work. Returns [{start, end, dur, peak, mean}]."""
    thr = osf.IMU_ACTIVE_THR if thr is None else thr
    if t.size == 0:
        return []
    active = std > thr
    eps, cur = [], None
    for i in range(len(t)):
        if active[i]:
            if cur is None:
                cur = [t[i], t[i]]
            else:
                if t[i] - cur[1] <= merge_gap:
                    cur[1] = t[i]
                else:
                    eps.append(cur)
                    cur = [t[i], t[i]]
    if cur is not None:
        eps.append(cur)
    out = []
    for a, b in eps:
        sel = (t >= a) & (t <= b)
        if b - a + 1 >= min_dur or std[sel].max() > 2 * thr:   # keep short-but-strong bursts
            out.append({"start": round(float(a), 1), "end": round(float(b), 1),
                        "dur": round(float(b - a), 1),
                        "peak": round(float(std[sel].max()), 4),
                        "mean": round(float(std[sel].mean()), 4)})
    return out


def build_timeline(base=DEFAULT_BASE, lo=0, hi=56, topic=osf.IMU_TOPIC):
    t0 = None
    for n in range(lo, hi + 1):
        bp = bagpath(base, n)
        if bp:
            t0 = first_image_ts(bp)
            if t0 is not None:
                break
    if t0 is None:
        raise RuntimeError("no front-camera frame found to anchor cycle t0")
    allc, alla = [], []
    for n in range(lo, hi + 1):
        bp = bagpath(base, n)
        if not bp:
            continue
        ts, acc = read_imu(bp, topic)
        if ts.size:
            allc.append((ts - t0) / 1e9)
            alla.append(acc)
            print(f"[imu] bag{n:02d} {ts.size} msgs  cyc {(ts[0]-t0)/1e9:6.0f}..{(ts[-1]-t0)/1e9:6.0f}", flush=True)
    cyc = np.concatenate(allc)
    acc = np.concatenate(alla)
    t, std = envelope(cyc, acc)
    eps = segment_episodes(t, std)
    return {"t0_ns": int(t0), "thr": osf.IMU_ACTIVE_THR,
            "env_win": ENV_WIN, "merge_gap": MERGE_GAP, "min_dur": MIN_DUR,
            "envelope": [[round(float(a), 1), round(float(b), 5)] for a, b in zip(t, std)],
            "episodes": eps}


# ---- tool API (for the harness / VLM) ----
def load_timeline(path="data/imu_timeline.json") -> dict:
    return json.loads(Path(path).read_text())


def machine_active_at(tl, t, win=osf.IMU_WIN_SEC) -> dict:
    """Is the machine physically running at cycle-time t? (envelope max in +/-win)."""
    env = np.array(tl["envelope"], dtype=float)
    if env.size == 0:
        return {"active": False, "std": 0.0}
    sel = (env[:, 0] >= t - win) & (env[:, 0] <= t + win)
    std = float(env[sel, 1].max()) if sel.any() else 0.0
    return {"active": std > tl["thr"], "std": round(std, 4)}


def episodes_until(tl, t) -> list:
    """Machine-active episodes whose work has occurred by cycle-time t."""
    return [e for e in tl["episodes"] if e["start"] <= t + 0.1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--bags", default="0-56")
    ap.add_argument("--out", default="data/imu_timeline.json")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.bags.split("-")) if "-" in a.bags else (int(a.bags), int(a.bags))
    tl = build_timeline(a.base, lo, hi)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(tl, indent=1))
    eps = tl["episodes"]
    tot = sum(1 for _ in tl["envelope"])
    act = sum(1 for _, s in tl["envelope"] if s > tl["thr"])
    print(f"\n=== IMU machine-activity timeline -> {a.out} ===")
    print(f"envelope: {tot} s, active {act} s ({100*act/max(1,tot):.0f}%);  {len(eps)} episodes")
    for i, e in enumerate(eps, 1):
        m = int(e["start"])
        print(f"  ep{i:2d}  {m//60:02d}:{m%60:02d}  dur={e['dur']:5.1f}s  peak={e['peak']:.3f}")


if __name__ == "__main__":
    raise SystemExit(main())
