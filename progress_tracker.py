"""Fused face-support progress — bolts(t), screens(t), coverage(t).

Combines the physical IMU work-windows with the operator install positions (and,
when present, per-episode VLM confirmations) into the three quantities the compliance
milestone needs:

  * bolts_installed(t)   — primary counter = IMU SUSTAINED drilling episodes completed by
                           time t (each support bolt needs a drilled hole; the cycle has
                           exactly 16 such episodes). A VLM episode-confirmation map can
                           VETO an episode that is not face-bolting (mucking/tramming),
                           keeping the count conservative (fail-safe: never over-count).
  * screens_installed(t) — Vale pattern is ~bolts_per_screen bolts per screen, so screen
                           progress follows the bolt count; cross-checked against the
                           number of distinct operator screen-load clusters.
  * face_coverage(t)     — worked-width fraction across the face band from operator
                           install x-positions (independent spatial check) AND the
                           screen-fraction; coverage is full when the whole face is worked.

The REQUIRED mesh/bolt count is NOT fixed at 4/16 — it is DERIVED from the MEASURED
face size (front Lidar): a bigger face needs more panels. `derive_counts` /
`load_targets` turn the measured face width into (meshes_required, bolts_required).

Pure functions over cached evidence (data/imu_timeline.json, data/operator_events.json,
optional data/episode_class.json, data/face_geometry.json). No model calls here.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

try:
    from harness_config import PARAMS
    _CV = PARAMS["coverage"]
except Exception:                       # tests / standalone without a bundle
    _CV = {}

# --- bolt episode gates (an IMU episode that is a real support-bolt drilling) ---
BOLT_MIN_DUR = float(_CV.get("bolt_min_dur", 40.0))    # s of sustained drilling per bolt hole
BOLT_MIN_PEAK = float(_CV.get("bolt_min_peak", 0.05))  # accel-mag std peak (idle 0.005, thr 0.013)
BOLTS_PER_SCREEN = int(_CV.get("bolts_per_screen", 4)) # Vale face pattern: ~4 bolts hold a screen

# --- size-dependent support requirement ---
FACE_WIDTH_DEFAULT = float(_CV.get("face_width_default", 6.0))   # fallback if no Lidar measure


def derive_counts(face_width, panel_w=None, overlap=None, bolts_per_screen=None):
    """Meshes/bolts for a face of `face_width` m, using the SINGLE Vale-document model in
    vale_support (6' sheets, 1' overlap, 4 bolts/sheet leading-edge minimum) — the count is
    emergent from the face size. Kept here as a thin wrapper so every path uses ONE model.
    (legacy panel_w/overlap/bolts_per_screen args are ignored — vale_support is authoritative.)"""
    import vale_support as vs
    nm = vs.meshes_required(face_width / vs.FT)      # width-only (mesh count is width-driven)
    return {"face_width": round(float(face_width), 2), "meshes_required": nm,
            "bolts_required": nm * vs.CMTS_MIN_BOLTS_PER_SHEET,
            "bolts_per_screen": vs.CMTS_MIN_BOLTS_PER_SHEET,
            "bolts_required_div6": nm * vs.DIV6_BOLTS_PER_SHEET,
            "panel_w": round(vs.SHEET_W_FT * vs.FT, 2), "overlap": round(vs.OVERLAP_FT * vs.FT, 2)}


def load_targets(geom="data/face_geometry.json"):
    """Required (meshes, bolts) for THIS face. Prefers the Vale-grounded counts written by
    face_geometry.py (precise lidar measure + vale_support); else derives from the measured
    width; else the configured fallback (logged via 'source')."""
    p = Path(geom)
    if p.exists():
        g = json.loads(p.read_text())
        if g.get("meshes_required") and g.get("bolts_required"):     # Vale-grounded (preferred)
            import vale_support as vs
            bps = int(g.get("bolts_per_screen", BOLTS_PER_SCREEN))
            return {"face_width": g.get("face_width"), "meshes_required": int(g["meshes_required"]),
                    "bolts_required": int(g["bolts_required"]), "bolts_per_screen": bps,
                    "bolts_required_div6": g.get("bolts_required_div6"),
                    "panel_w": round(vs.SHEET_W_FT * vs.FT, 2), "overlap": round(vs.OVERLAP_FT * vs.FT, 2),
                    "source": f"vale-doc + lidar({g.get('n_bags','?')} bags, {g.get('face_width')} m)"}
        if g.get("face_width"):
            t = derive_counts(g["face_width"])
            t["source"] = f"lidar({g.get('face_width')} m)"
            return t
    t = derive_counts(FACE_WIDTH_DEFAULT)
    t["source"] = "default(no lidar measurement)"
    return t


# back-compat module defaults (derived from the configured fallback face width)
_DEF = derive_counts(FACE_WIDTH_DEFAULT)
TARGET_BOLTS = _DEF["bolts_required"]
TARGET_SCREENS = _DEF["meshes_required"]

# --- coverage (operator install positions across the face band) ---
FACE_X = (0.12, 0.92)      # face width band (fractions); excludes wall edges
COVER_PANEL = 0.18         # how wide a worked position covers (reach), fraction
FULL_COVER_FRAC = 0.85


def load_evidence(imu="data/imu_timeline.json", ops="data/operator_events.json",
                  episode_class="data/episode_class.json"):
    """Load cached evidence. The VLM episode-classification map is used as a veto when
    present (auto-detected) so the bolt count is vision-cross-checked, not IMU-only."""
    tl = json.loads(Path(imu).read_text())
    ev = json.loads(Path(ops).read_text())
    cls = json.loads(Path(episode_class).read_text()) if episode_class and Path(episode_class).exists() else None
    return tl, ev, cls


def bolt_episodes(tl, episode_cls=None):
    """IMU sustained drilling episodes that count as support bolts. Each gets a
    'set_at' = end of drilling. An optional VLM classification map {start: label}
    can VETO an episode whose label is clearly non-bolting (fail-safe)."""
    veto = set()
    if episode_cls:
        for c in episode_cls.get("episodes", []):
            if c.get("activity") in ("mucking", "tramming", "idle", "other_clear"):
                veto.add(round(float(c["start"]), 1))
    out = []
    for e in tl["episodes"]:
        if e["dur"] >= BOLT_MIN_DUR and e["peak"] >= BOLT_MIN_PEAK:
            if round(float(e["start"]), 1) in veto:
                continue
            out.append({"start": e["start"], "set_at": round(e["start"] + e["dur"], 1),
                        "dur": e["dur"], "peak": e["peak"]})
    return out


def bolts_installed(bolts, t):
    return sum(1 for b in bolts if b["set_at"] <= t + 0.1)


def screens_installed(bolts, t, bolts_per_screen=BOLTS_PER_SCREEN, meshes_required=TARGET_SCREENS):
    """Screen progress from the bolts-per-screen pattern (a screen is held once its bolt
    pattern is in), capped at the (size-derived) number of meshes the face needs."""
    return min(meshes_required, bolts_installed(bolts, t) // bolts_per_screen)


def screen_load_clusters(ev, gap=90.0):
    """Distinct operator screen-load episodes (cross-check on screen count). Confirmed
    operator-in-front visits within `gap` s are one screen-load."""
    visits = sorted([e for e in ev["events"] if e.get("person_bbox")], key=lambda x: x["cycle_sec"])
    clusters, cur = [], None
    for v in visits:
        if cur is None or v["cycle_sec"] - cur[-1]["cycle_sec"] > gap:
            cur = [v]
            clusters.append(cur)
        else:
            cur.append(v)
    return clusters


def worked_coverage(ev, t, face_x=FACE_X, panel=COVER_PANEL):
    """Worked-width fraction of the face band from operator install x-positions up to t."""
    fx0, fx1 = face_x
    span = fx1 - fx0
    nb = 64
    binw = span / nb
    hit = [False] * nb
    for e in ev["events"]:
        if not e.get("person_bbox") or e["cycle_sec"] > t + 0.1:
            continue
        bb = e["person_bbox"]
        cx = (bb[0] + bb[2]) / 2.0
        b0 = int((max(fx0, cx - panel / 2) - fx0) / binw)
        b1 = int((min(fx1, cx + panel / 2) - fx0) / binw)
        for b in range(max(0, b0), min(nb, b1 + 1)):
            hit[b] = True
    return sum(hit) / nb


def face_coverage(bolts, ev, t, meshes_required=TARGET_SCREENS, bolts_per_screen=BOLTS_PER_SCREEN):
    """Fused face coverage at t: max of (a) screen-fraction (all required meshes = whole
    face) and (b) operator worked-width. Reported AND used as a milestone gate."""
    screen_frac = screens_installed(bolts, t, bolts_per_screen, meshes_required) / max(1, meshes_required)
    worked = worked_coverage(ev, t)
    return {"screen_frac": round(screen_frac, 3), "worked_frac": round(worked, 3),
            "coverage": round(max(screen_frac, worked), 3)}


def progress_at(tl, ev, t, episode_cls=None, targets=None):
    """Progress at cycle-time t. `targets` (from load_targets) carry the SIZE-DERIVED
    requirement; defaults to the configured fallback if not supplied."""
    if targets is None:
        targets = load_targets()
    nm, bps = targets["meshes_required"], targets["bolts_per_screen"]
    bolts = bolt_episodes(tl, episode_cls)
    nb = bolts_installed(bolts, t)
    ns = screens_installed(bolts, t, bps, nm)
    cov = face_coverage(bolts, ev, t, nm, bps)
    return {"t": round(t, 1), "bolts": nb, "screens": ns,
            "coverage": cov["coverage"], "coverage_detail": cov,
            "bolts_target": targets["bolts_required"], "screens_target": nm,
            "face_width": targets.get("face_width"), "target_source": targets.get("source")}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--imu", default="data/imu_timeline.json")
    ap.add_argument("--ops", default="data/operator_events.json")
    ap.add_argument("--episode-class", default=None)
    ap.add_argument("--step", type=float, default=60.0)
    a = ap.parse_args()
    tl, ev, cls = load_evidence(a.imu, a.ops, a.episode_class)
    tg = load_targets()
    bolts = bolt_episodes(tl, cls)
    clusters = screen_load_clusters(ev)
    NB, NM = tg["bolts_required"], tg["meshes_required"]
    print(f"=== Progress evidence ===")
    print(f"face width: {tg.get('face_width')} m  ->  REQUIRED {NM} meshes / {NB} bolts  "
          f"[{tg.get('source')}]")
    print(f"IMU bolt episodes (dur>={BOLT_MIN_DUR}s, peak>={BOLT_MIN_PEAK}): {len(bolts)} (need {NB})")
    print(f"operator screen-load clusters (cross-check): {len(clusters)} (need {NM})")
    tmax = bolts[-1]["set_at"] + 60 if bolts else 3400
    print(f"\n  cycle    bolts   screens  coverage(scr/worked)")
    t = 0.0
    while t <= tmax:
        p = progress_at(tl, ev, t, cls, tg)
        d = p["coverage_detail"]
        print(f"  {int(t)//60:02d}:{int(t)%60:02d}   {p['bolts']:2d}/{NB}   {p['screens']}/{NM}   "
              f"{p['coverage']:.2f} ({d['screen_frac']:.2f}/{d['worked_frac']:.2f})")
        t += a.step
    def first_reach(pred):
        tt = 0.0
        while tt <= tmax:
            if pred(progress_at(tl, ev, tt, cls, tg)):
                return tt
            tt += 2.0
        return None
    bN = first_reach(lambda p: p["bolts"] >= NB)
    sN = first_reach(lambda p: p["screens"] >= NM)
    print(f"\nbolts>={NB} first at: {None if bN is None else f'{int(bN)//60:02d}:{int(bN)%60:02d}'}")
    print(f"screens>={NM} first at: {None if sN is None else f'{int(sN)//60:02d}:{int(sN)%60:02d}'}")


if __name__ == "__main__":
    raise SystemExit(main())
