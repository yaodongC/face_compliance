"""Livox front-Lidar analyzer — geometric face/boom context tool.

The front Livox lidar (livox_ros_driver2/CustomMsg, ~10 Hz, ~20k pts/scan) looks at the
end face the crew is supporting. It cannot tell mesh from bare rock (a screen is thin and
conforms to the rock), so it does NOT measure mesh-vs-rock. What it CAN measure robustly is
GEOMETRY, which the harness uses as physical cross-checks:

  * face_dist     — median forward range to the wall ahead (jumbo standoff from the face).
  * struct_front  — density of close points in the central cone (a boom/operator/structure
                    right in front of the camera). High while a boom is worked at the face;
                    lower when the booms are folded clear.
  * lat_spread    — lateral spread of the near points (work happening across the face width).

Features are aggregated over several scans per sample (single scans are noisy). The lidar
signal is CONTEXT/cross-check, not a primary compliance gate — its limits are documented.

CLI:  python3 lidar_analyzer.py [--bags 0-56] [--per-bag 6] [--out data/lidar_timeline.json]
Tool: load_timeline(); face_state_at(tl, t)
"""
from __future__ import annotations
import argparse
import glob
import json
from pathlib import Path
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.highlevel.anyreader import AnyReaderError

DEFAULT_BASE = "/home/nvidia/rosbags/vale/20260611_115532"
CAM_TOPIC = "/sensing/front/rgb/image_raw/compressed"
LIDAR_TOPIC = "/sensing/front/livox/lidar"

CONE_Y = 1.2          # |y| < CONE_Y is the central cone (boresight column)
NEAR_M = 1.8          # points closer than this are "structure right in front"
FWD_MIN = 0.2         # forward hemisphere


def bagpath(base, n):
    g = sorted(glob.glob(f"{base}/*_{n}.bag"))
    return g[0] if g else None


def first_image_ts(bag) -> int | None:
    try:
        with AnyReader([Path(bag)]) as r:
            conns = [c for c in r.connections if c.topic == CAM_TOPIC]
            for _c, t, _raw in r.messages(connections=conns):
                return int(t)
    except AnyReaderError:
        return None
    return None


def _xyz(msg) -> np.ndarray:
    pts = msg.points
    xyz = np.empty((len(pts), 3), dtype=np.float32)
    for i, p in enumerate(pts):
        xyz[i, 0] = p.x
        xyz[i, 1] = p.y
        xyz[i, 2] = p.z
    d = np.linalg.norm(xyz, axis=1)
    return xyz[d > 0.1]


def scan_features(xyz) -> dict:
    if xyz.shape[0] < 50:
        return {}
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    d = np.linalg.norm(xyz, axis=1)
    fwd = x > FWD_MIN
    cen = fwd & (np.abs(y) < CONE_Y)
    near_cen = cen & (d < NEAR_M)
    return {
        "face_dist": float(np.percentile(x[fwd], 50)) if fwd.any() else 0.0,
        "face_dist_p25": float(np.percentile(x[fwd], 25)) if fwd.any() else 0.0,
        "struct_front": int(near_cen.sum()),
        "lat_spread": float(y[fwd].std()) if fwd.any() else 0.0,
        "near_y": float(np.median(y[near_cen])) if near_cen.any() else 0.0,
    }


def build_timeline(base=DEFAULT_BASE, lo=0, hi=56, per_bag=6):
    t0 = None
    for n in range(lo, hi + 1):
        bp = bagpath(base, n)
        if bp:
            t0 = first_image_ts(bp)
            if t0 is not None:
                break
    if t0 is None:
        raise RuntimeError("no front-camera frame to anchor cycle t0")
    samples = []
    for n in range(lo, hi + 1):
        bp = bagpath(base, n)
        if not bp:
            continue
        try:
            with AnyReader([Path(bp)]) as r:
                conns = [c for c in r.connections if c.topic == LIDAR_TOPIC]
                raws = list(r.messages(connections=conns))     # (conn, ts, raw) — cheap, no decode
                if not raws:
                    continue
                idxs = np.linspace(0, len(raws) - 1, min(per_bag, len(raws))).round().astype(int)
                for j in idxs:
                    c, t, raw = raws[int(j)]
                    cyc = (t - t0) / 1e9
                    f = scan_features(_xyz(r.deserialize(raw, c.msgtype)))
                    if f:
                        f["t"] = round(float(cyc), 1)
                        samples.append(f)
            print(f"[lidar] bag{n:02d} sampled {len(idxs)}  (cum {len(samples)})", flush=True)
        except (AnyReaderError, Exception) as e:
            print(f"[lidar] {Path(bp).name}: {e}")
    samples.sort(key=lambda s: s["t"])
    return {"t0_ns": int(t0), "cone_y": CONE_Y, "near_m": NEAR_M, "samples": samples}


def accumulate(base=DEFAULT_BASE, bags=(55,), max_scans=400):
    """Accumulate Livox Mid360 scans into ONE dense, high-resolution cloud.

    The Mid360 has a NON-REPETITIVE scan pattern: each ~20k-pt scan samples different
    directions, so stacking many scans while the jumbo is PARKED (chassis fixed during the
    support cycle) fills in coverage and yields a far denser cloud than any single scan —
    enough to measure the face precisely and inspect its surface. Returns (xyz Nx3, refl N).
    Assumes the lidar pose is fixed over the window (valid while parked at the face)."""
    xs, rs = [], []
    taken = 0
    for n in bags:
        bp = bagpath(base, int(n))
        if not bp:
            continue
        try:
            with AnyReader([Path(bp)]) as r:
                conns = [c for c in r.connections if c.topic == LIDAR_TOPIC]
                raws = list(r.messages(connections=conns))
                step = max(1, len(raws) // max(1, (max_scans - taken)))
                for c, _t, raw in raws[::step]:
                    m = r.deserialize(raw, c.msgtype)
                    pts = m.points
                    a = np.empty((len(pts), 4), dtype=np.float32)
                    for i, p in enumerate(pts):
                        a[i] = (p.x, p.y, p.z, p.reflectivity)
                    d = np.linalg.norm(a[:, :3], axis=1)
                    a = a[d > 0.1]
                    xs.append(a[:, :3])
                    rs.append(a[:, 3])
                    taken += 1
                    if taken >= max_scans:
                        break
        except (AnyReaderError, Exception) as e:
            print(f"[lidar] accumulate {Path(bp).name}: {e}")
        if taken >= max_scans:
            break
    if not xs:
        return np.empty((0, 3)), np.empty((0,))
    return np.vstack(xs), np.concatenate(rs)


def face_heightmap(xyz, shell=(1.2, 4.5), ny=160, nz=120):
    """Project the accumulated face cloud to a (z,y) standoff image: pixel = nearest forward
    distance (depth to the face). Reveals the face surface; bolt plates/booms read as closer
    blobs. Returns a uint8 image (or None)."""
    if xyz.shape[0] < 500:
        return None
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    sel = (x >= shell[0]) & (x < shell[1])
    x, y, z = x[sel], y[sel], z[sel]
    if x.size < 500:
        return None
    y0, y1 = np.percentile(y, 1), np.percentile(y, 99)
    z0, z1 = np.percentile(z, 1), np.percentile(z, 99)
    img = np.full((nz, ny), np.nan)
    iy = ((y - y0) / (y1 - y0 + 1e-9) * (ny - 1)).astype(int).clip(0, ny - 1)
    iz = ((z - z0) / (z1 - z0 + 1e-9) * (nz - 1)).astype(int).clip(0, nz - 1)
    for j, k, dx in zip(iy, iz, x):
        if np.isnan(img[k, j]) or dx < img[k, j]:
            img[k, j] = dx
    valid = ~np.isnan(img)
    if valid.sum() < 100:
        return None
    lo, hi = np.percentile(img[valid], 2), np.percentile(img[valid], 98)
    out = np.zeros((nz, ny), dtype=np.uint8)
    out[valid] = (255 * (1 - (img[valid] - lo) / (hi - lo + 1e-9))).clip(0, 255).astype(np.uint8)
    return np.flipud(out)   # z up


IMU_TOPIC = "/sensing/front/livox/imu"
RGB_INFO_TOPIC = "/sensing/front/rgb/camera_info"


def camera_hfov(base=DEFAULT_BASE, bag=55):
    """Front-camera horizontal/vertical FOV (deg) from camera_info intrinsics — an
    INDEPENDENT modality to cross-check the Lidar face width."""
    import math
    bp = bagpath(base, bag)
    if not bp:
        return None
    try:
        with AnyReader([Path(bp)]) as r:
            conns = [c for c in r.connections if c.topic == RGB_INFO_TOPIC]
            for c, _t, raw in r.messages(connections=conns):
                m = r.deserialize(raw, c.msgtype)
                K = np.array(m.K, float).reshape(3, 3)
                fx, fy, W, H = K[0, 0], K[1, 1], m.width, m.height
                return {"hfov_deg": round(2 * math.degrees(math.atan(W / (2 * fx))), 1),
                        "vfov_deg": round(2 * math.degrees(math.atan(H / (2 * fy))), 1),
                        "width_px": int(W), "height_px": int(H)}
    except (AnyReaderError, Exception) as e:
        print(f"[lidar] camera_info {Path(bp).name}: {e}")
    return None


def camera_crosscheck(face_width_m, face_dist_m, base=DEFAULT_BASE, bag=55):
    """Cross-check the Lidar face width against the camera FOV: a face of width W at standoff
    d subtends 2*atan(W/2d); if the camera is ~co-located/forward-aligned it should fill that
    fraction of the HFOV. >~105% means the Lidar width is too big (walls would be off-frame)."""
    import math
    cam = camera_hfov(base, bag)
    if not cam or not face_dist_m:
        return {"have": False}
    subtend = 2 * math.degrees(math.atan((face_width_m / 2) / face_dist_m))
    frac = subtend / cam["hfov_deg"]
    return {"have": True, "hfov_deg": cam["hfov_deg"], "face_subtend_deg": round(subtend, 1),
            "image_fill_frac": round(frac, 2),
            "consistent": bool(frac <= 1.05)}


def read_gravity(bag):
    """Mean front-IMU acceleration over a bag = the gravity vector in the lidar frame.
    The jumbo is parked (static) during the support cycle, so the mean accel is gravity;
    it reveals the lidar's pitch/roll so the cloud can be levelled before measuring."""
    acc = []
    try:
        with AnyReader([Path(bag)]) as r:
            conns = [c for c in r.connections if c.topic == IMU_TOPIC]
            for c, _t, raw in r.messages(connections=conns):
                la = r.deserialize(raw, c.msgtype).linear_acceleration
                acc.append((la.x, la.y, la.z))
    except (AnyReaderError, Exception) as e:
        print(f"[lidar] gravity {Path(bag).name}: {e}")
    if not acc:
        print(f"[lidar] gravity: no IMU messages in {Path(bag).name}")
        return None
    return np.array(acc).mean(axis=0)


def gravity_align_R(g):
    """Rotation that levels the cloud: maps lidar axes -> (X=forward-horizontal, Y=left,
    Z=true-up) using the IMU gravity vector. Corrects the lidar mount pitch/roll so width
    is measured horizontally and height vertically."""
    up = -np.asarray(g, float)
    up = up / np.linalg.norm(up)
    fwd = np.array([1.0, 0.0, 0.0]) - up[0] * up      # lidar +x projected onto the horizontal
    nf = np.linalg.norm(fwd)
    if nf < 0.1:                                       # lidar nearly vertical -> no horizontal forward
        raise ValueError(f"gravity_align_R: lidar axis ~vertical, degenerate forward (g={g})")
    fwd = fwd / nf
    left = np.cross(up, fwd)
    return np.vstack([fwd, left, up])                 # rows = new axes


def _face_start_x(X, Y):
    """Forward distance where the END FACE begins = end of the boom/equipment gap.
    Booms cluster near the lidar, then a low-density gap, then the face; find the near boom
    cluster, then the first empty run after it — the face starts at its end.

    NOTE: a 'farthest-significant-cluster' variant was tried to generalise to large standoffs
    (review finding L1), but on the real bags it DESTABILISED the gap (4.4 m -> 2.0-4.4 m, and
    a bag misfired at 7.2 m), because boom/face clusters merge irregularly across scans. The
    near-window detector below is stable on this rig; generalising to other standoffs needs
    real far-standoff data to tune against, so L1 stays a documented limitation."""
    cen = np.abs(Y) < 1.2
    hist, edges = np.histogram(X[cen & (X > 0.3)], bins=45, range=(0, 9))
    if hist.sum() < 200:
        return 1.5
    peak = hist[:15].max()                       # boom cluster density (first ~3 m)
    i = int(np.argmax(hist[:15]))
    while i < len(hist) and hist[i] > 0.10 * peak:   # walk past the boom cluster
        i += 1
    while i < len(hist) and hist[i] < 0.10 * peak:   # walk to the end of the gap
        i += 1
    if i >= len(hist):                           # no clear gap found -> face is near the edge,
        return 1.5                               # not at 9 m (which would empty the face mask)
    return float(edges[i])


def _wall_pos(vals, side, slab=0.4, edge_pct=2.0):
    """Robust position of a bounding surface = median of the points within `slab` m of the
    extreme. Robust to flare/outliers (a few corner points beyond the wall plane don't move
    the median), unlike a raw percentile extent. side: 'lo' or 'hi'."""
    if vals.size < 30:
        return None, 0.0
    if side == "lo":
        edge = np.percentile(vals, edge_pct)
        sel = vals < edge + slab
    else:
        edge = np.percentile(vals, 100 - edge_pct)
        sel = vals > edge - slab
    w = vals[sel]
    return float(np.median(w)), float(w.std())


def _measure_one(base, bag, max_scans, spring_band, edge_pct):
    g = read_gravity(bagpath(base, bag))
    if g is None:
        return None
    R = gravity_align_R(g)
    pitch = float(np.degrees(np.arctan2(g[0], -g[2])))
    xyz, _ = accumulate(base, bags=(bag,), max_scans=max_scans)
    if xyz.shape[0] < 1000:
        return None
    P = (R @ xyz.T).T
    X, Y, Z = P[:, 0], P[:, 1], P[:, 2]
    gap = _face_start_x(X, Y)
    if gap > 7.0:
        print(f"[lidar] {Path(bagpath(base, bag)).name}: boom-gap at {gap:.1f} m — face may not be in view")
    face = (X > gap) & (X < 9.0)
    if face.sum() < 500:
        return None
    # camera-to-face standoff = characteristic forward distance of the face wall (central
    # column), NOT the gap START — used by the camera FOV cross-check.
    cc = face & (np.abs(Y) < 1.5)
    face_dist = float(np.median(X[cc if cc.sum() > 100 else face]))
    # HEIGHT: robust floor->crown of the end face (central column, gravity-aligned Z).
    # Uses the FAR region (face = X>gap), which excludes the near boom cluster — important
    # because the booms sit ~3.6 m higher than the true floor and would corrupt the floor Z.
    cen = face & (np.abs(Y) < 1.5)
    zf, _sf = _wall_pos(Z[cen if cen.sum() > 200 else face], "lo", slab=0.3, edge_pct=edge_pct)
    zc, _sc = _wall_pos(Z[cen if cen.sum() > 200 else face], "hi", slab=0.3, edge_pct=edge_pct)
    height = float(zc - zf)
    # FLOOR GRADE (gravity-aligned): fit the far-floor slab; ~0 = flat drift, steep = ramp
    # (on a real ramp the floor-to-back height would need an along-grade correction).
    floor_grade = 0.0
    fl = face & (Z < zf + 0.4) & (np.abs(Y) < 2.5)
    if fl.sum() > 300:
        slope = float(np.polyfit(X[fl], Z[fl], 1)[0])
        floor_grade = float(np.degrees(np.arctan(slope)))
    # WIDTH: distance between the two SIDE-WALL planes at the springline (robust to flare).
    # Side walls are uniform along the drift, so measure over the forward span for density.
    zmid = (zf + zc) / 2
    spring = (X > 0.5) & (X < min(7.5, gap + 4.0)) & (Z > zmid - spring_band) & (Z < zmid + spring_band)
    Ys = Y[spring if spring.sum() > 200 else face]
    lw, lstd = _wall_pos(Ys, "lo", slab=0.4, edge_pct=edge_pct)
    rw, rstd = _wall_pos(Ys, "hi", slab=0.4, edge_pct=edge_pct)
    if lw is None or rw is None:
        return None
    width = float(rw - lw)
    return {"width": width, "height": height, "pitch": pitch, "gap": gap,
            "face_dist": face_dist,
            "n_face": int(face.sum()), "wall_planarity": round((lstd + rstd) / 2, 3),
            "floor_grade_deg": round(floor_grade, 1)}


def face_profile(base=DEFAULT_BASE, bag=55, max_scans=300, n_bands=14, band=0.25):
    """The face is ARCHED, not a rectangle. Return its cross-sectional profile = wall-to-wall
    width at each height (gravity-aligned), the true cross-sectional AREA, and the max
    (springline) width that drives the mesh count. Single parked bag (dense accumulation)."""
    g = read_gravity(bagpath(base, bag))
    if g is None:
        return {"have": False}
    R = gravity_align_R(g)
    xyz, _ = accumulate(base, bags=(bag,), max_scans=max_scans)
    if xyz.shape[0] < 1000:
        return {"have": False}
    P = (R @ xyz.T).T
    X, Y, Z = P[:, 0], P[:, 1], P[:, 2]
    gap = _face_start_x(X, Y)
    face = (X > gap) & (X < 9.0)
    Yf, Zf = Y[face], Z[face]
    # robust floor/crown (same estimator as measure_face_precise, so height/area agree)
    zf, _z0 = _wall_pos(Zf, "lo", slab=0.3, edge_pct=1.0)
    zc, _z1 = _wall_pos(Zf, "hi", slab=0.3, edge_pct=1.0)
    if zf is None or zc is None or zc - zf < 0.5:
        return {"have": False}
    bands, area = [], 0.0
    dz = (zc - zf) / n_bands
    band = min(band, dz / 2)            # non-overlapping integration strips (no double-count)
    for k in range(n_bands):
        z = zf + (k + 0.5) * dz
        sel = (Zf > z - band) & (Zf < z + band)
        if sel.sum() < 80:
            continue
        lo, _ = _wall_pos(Yf[sel], "lo", slab=0.4)
        hi, _ = _wall_pos(Yf[sel], "hi", slab=0.4)
        if lo is None or hi is None:
            continue
        w = hi - lo
        area += w * dz
        bands.append({"h_frac": round((z - zf) / (zc - zf), 2), "height_m": round(z - zf, 2),
                      "width_m": round(float(w), 2)})
    widths = [b["width_m"] for b in bands]
    return {"have": True, "floor_z": round(float(zf), 2), "crown_z": round(float(zc), 2),
            "height_m": round(float(zc - zf), 2), "max_width_m": round(float(max(widths)), 2) if widths else 0,
            "area_m2": round(float(area), 1), "n_bands": len(bands), "profile": bands}


def measure_face_precise(base=DEFAULT_BASE, bags=(50, 53, 55), max_scans=300,
                         spring_band=0.6, edge_pct=1.0):
    """PRECISE end-face size from accumulated, gravity-levelled Lidar.

    Per parked bag: accumulate many Mid360 scans (dense), level with the IMU gravity
    vector, drop the near boom cluster (gap detection), then width = horizontal (Y) extent
    at the springline and height = floor->crown (Z). Measured INDEPENDENTLY per bag (the
    chassis may shift between bags) and combined by MEDIAN for robustness."""
    per = [m for m in (_measure_one(base, b, max_scans, spring_band, edge_pct) for b in bags) if m]
    if not per:
        return {"have": False, "reason": "no usable bag"}
    w = float(np.median([m["width"] for m in per]))
    h = float(np.median([m["height"] for m in per]))
    return {"have": True, "face_width": round(w, 2), "face_height": round(h, 2),
            "face_width_ft": round(w / 0.3048, 1), "face_height_ft": round(h / 0.3048, 1),
            "width_spread": round(float(np.ptp([m["width"] for m in per])), 2),
            "lidar_pitch_deg": round(float(np.median([m["pitch"] for m in per])), 1),
            "floor_grade_deg": round(float(np.median([m["floor_grade_deg"] for m in per])), 1),
            "boom_gap_m": round(float(np.median([m["gap"] for m in per])), 1),
            "face_dist_m": round(float(np.median([m["face_dist"] for m in per])), 1),
            "n_bags": len(per), "bags": list(bags),
            "per_bag_width": [round(m["width"], 2) for m in per],
            "wall_planarity_m": round(float(np.median([m["wall_planarity"] for m in per])), 3),
            "n_face_points": int(np.median([m["n_face"] for m in per])),
            "method": "per-bag accumulated + IMU-gravity-levelled, side-wall-plane width"}


# ---- tool API ----
def load_timeline(path="data/lidar_timeline.json") -> dict:
    return json.loads(Path(path).read_text())


def face_state_at(tl, t, win=8.0) -> dict:
    """Aggregate lidar features around cycle-time t (median over +/-win)."""
    s = [x for x in tl["samples"] if abs(x["t"] - t) <= win]
    if not s:
        return {"have": False}
    return {"have": True, "n": len(s),
            "face_dist": round(float(np.median([x["face_dist"] for x in s])), 2),
            "struct_front": int(np.median([x["struct_front"] for x in s])),
            "lat_spread": round(float(np.median([x["lat_spread"] for x in s])), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--bags", default="0-56")
    ap.add_argument("--per-bag", type=int, default=6)
    ap.add_argument("--out", default="data/lidar_timeline.json")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.bags.split("-")) if "-" in a.bags else (int(a.bags), int(a.bags))
    tl = build_timeline(a.base, lo, hi, a.per_bag)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(tl, indent=1))
    s = tl["samples"]
    print(f"\n=== Lidar timeline -> {a.out}  ({len(s)} samples) ===")
    for x in s[::max(1, len(s)//40)]:
        m = int(x["t"])
        print(f"  {m//60:02d}:{m%60:02d}  dist={x['face_dist']:.2f}m  struct_front={x['struct_front']:4d}  lat={x['lat_spread']:.2f}")


if __name__ == "__main__":
    raise SystemExit(main())
