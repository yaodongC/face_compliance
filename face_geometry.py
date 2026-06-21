"""Measure the face PRECISELY (lidar) and compute the meshes/bolts needed per the VALE
documents. The required count is size-dependent, derived — not a fixed 4/16.

Pipeline:
  1. lidar_analyzer.measure_face_precise — accumulate Livox Mid360 scans (parked,
     non-repetitive -> dense), level with the IMU gravity vector, drop the boom cluster,
     measure face width x height robustly (median across parked bags).
  2. vale_support.calc — turn the measured size into meshes_required + bolt counts using
     the Vale standards (CMTS-2015-001 Rev5 + Division 6 Creighton).
Writes data/face_geometry.json (read by progress_tracker.load_targets) + a height-map PNG.

  python3 face_geometry.py [--bags 48,50,53,55] [--out data/face_geometry.json]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import lidar_analyzer as la
import vale_support as vs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=la.DEFAULT_BASE)
    ap.add_argument("--bags", default="48,50,53,55", help="parked bags to measure (median)")
    ap.add_argument("--accumulate-bag", type=int, default=55, help="bag for the dense height-map")
    ap.add_argument("--out", default="data/face_geometry.json")
    ap.add_argument("--heightmap", default="data/face_heightmap.png")
    a = ap.parse_args()
    bags = tuple(int(x) for x in a.bags.split(","))

    # 1) PRECISE face size (gravity-levelled, dense accumulation, median across bags)
    m = la.measure_face_precise(a.base, bags=bags)
    if not m.get("have"):
        raise SystemExit(f"could not measure face size from Lidar: {m}")
    face_w, face_h = m["face_width"], m["face_height"]

    # 2) Vale-document requirement (+ count confidence using the measured width spread)
    v = vs.calc(face_w, face_h)
    unc = max(0.05, (m.get("width_spread") or 0.13))
    v["count_confidence"] = vs.mesh_count_confidence(face_w, width_unc_m=round(unc, 2))

    # 2b) independent camera cross-check of the measured width (use the camera-to-face
    # standoff = median face-wall distance, NOT the boom-gap start)
    xcheck = la.camera_crosscheck(face_w, m.get("face_dist_m") or m.get("boom_gap_m") or 4.1,
                                  a.base, a.accumulate_bag)

    # 3) dense accumulated cloud -> height-map artifact
    xyz, _ = la.accumulate(a.base, bags=(a.accumulate_bag,), max_scans=400)
    hm = la.face_heightmap(xyz) if xyz.shape[0] else None
    if hm is not None:
        big = cv2.applyColorMap(cv2.resize(hm, (640, 480), interpolation=cv2.INTER_NEAREST),
                                cv2.COLORMAP_TURBO)
        cv2.putText(big, f"ACCUMULATED FACE {xyz.shape[0]:,} pts  {face_w}x{face_h} m -> {v['meshes_required']} meshes",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
        Path(a.heightmap).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(a.heightmap, big)

    # 3b) concrete mesh/bolt layout + schematic
    try:
        import render_mesh_layout as rml
        lay = rml.render(face_w, face_h, out="data/face_mesh_layout.png",
                         bg_heightmap=a.heightmap, meta="lidar-measured")
        layout_png = "data/face_mesh_layout.png"
    except Exception as e:
        print(f"[face_geometry] layout render skipped: {e}")
        lay, layout_png = vs.mesh_layout(face_w, face_h), None

    # 3c) arched cross-section profile + true area (the face is an arch, not a rectangle)
    prof = la.face_profile(a.base, a.accumulate_bag)
    prof_png = None
    if prof.get("have"):
        try:
            import render_face_profile as rfp
            rfp.render(prof, "data/face_profile.png")
            prof_png = "data/face_profile.png"
        except Exception as e:
            print(f"[face_geometry] profile render skipped: {e}")

    out = {
        # measurement
        "face_width": face_w, "face_height": face_h,
        "face_width_ft": m["face_width_ft"], "face_height_ft": m["face_height_ft"],
        "lidar_pitch_deg": m["lidar_pitch_deg"], "floor_grade_deg": m.get("floor_grade_deg"),
        "boom_gap_m": m["boom_gap_m"], "face_dist_m": m.get("face_dist_m"),
        "width_spread": m.get("width_spread"), "per_bag_width": m.get("per_bag_width"),
        "wall_planarity_m": m.get("wall_planarity_m"),
        "n_bags": m["n_bags"], "n_face_points": m["n_face_points"],
        "measure_method": m["method"], "camera_crosscheck": xcheck,
        "face_area_m2": prof.get("area_m2"), "cross_section": prof.get("profile"),
        "cross_section_png": prof_png,
        # Vale requirement
        "meshes_required": v["meshes_required"],
        "bolts_per_screen": v["bolts_per_sheet_min"],         # compliance counter (leading-edge min)
        "bolts_required": v["bolts_required_min"],            # = meshes * 4 (CMTS min / observed)
        "bolts_required_div6": v["bolts_required_div6"],      # Creighton full standard (6/sheet)
        "sheet_w_ft": v["sheet_w_ft"], "overlap_ft": v["overlap_ft"], "pattern": v["pattern"],
        "bolt_grid": v["bolt_grid"], "count_confidence": v["count_confidence"],
        "sources": v["sources"],
        "mesh_panels": lay["panels"], "n_bolts_grid": len(lay["bolts"]),
        "heightmap": a.heightmap if hm is not None else None,
        "mesh_layout_png": layout_png,
    }
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"=== FACE GEOMETRY (lidar) -> {a.out} ===")
    print(f"  measured: {face_w} x {face_h} m  ({m['face_width_ft']} x {m['face_height_ft']} ft)  "
          f"pitch {m['lidar_pitch_deg']} deg (mount), floor-grade {m.get('floor_grade_deg')} deg, "
          f"spread {m.get('width_spread')} m, wall-planarity {m.get('wall_planarity_m')} m, {m['n_bags']} bags")
    if xcheck.get("have"):
        print(f"  camera cross-check: face subtends {xcheck['face_subtend_deg']} deg of "
              f"{xcheck['hfov_deg']} deg HFOV -> fills {xcheck['image_fill_frac']*100:.0f}% of frame "
              f"({'CONSISTENT' if xcheck['consistent'] else 'INCONSISTENT (width too large?)'})")
    print(vs._fmt(v))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
