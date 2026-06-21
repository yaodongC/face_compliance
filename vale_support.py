"""Vale face-support requirement calculator — number of meshes/bolts for a measured face.

Pure rules distilled from the two Vale documents (citations inline). Given the precisely
measured face size (from lidar_analyzer.measure_face_precise), it computes how many screen
sheets and bolts the face needs, per the standard. No assumed 4/16 — it scales with size.

SOURCES
  [A] All Mines Face Support Guidelines, CMTS-2015-001 Rev5:
      - "Minimum bolting pattern is 4'x5' on leading edge of screen. Install additional
         bolts as required."  -> leading-edge minimum = 4 corner bolts per screen.
      - "Screen should overlap a minimum of 3 squares" (#6/#4 gauge welded wire mesh).
      - Extents: bolts <=1.5' from walls; top row <=2' from back; bottom screen bolts to
         within 5' of BOR (Scenario 1, moderate seismic / gravity).
  [B] Division 6 Lateral Development Support Standard (Creighton), Rev4:
      - FACE: "4'x5' PATTERN (6 BOLTS PER SHEET, 3-0-3 PATTERN) WITH 6.5' FS46 AND 4GA MESH.
         WALL BOLTS MAXIMUM 1' FROM FACE."  -> site face standard = 6 bolts/sheet.
      - "1' OVERLAP (3 SQUARES ...). SAME OVERLAP APPLIES TO ... FACE SUPPORT."
      - Detail: two sheets span 11' at 1' overlap -> screen sheet ~6' wide.
"""
from __future__ import annotations
import math
import numpy as np

FT = 0.3048  # m per foot

# --- screen geometry (doc [B] detail + [A]/[B] overlap) ---
SHEET_W_FT = 6.0          # welded-wire screen sheet width (11' = two sheets at 1' overlap)
OVERLAP_FT = 1.0          # 3 squares of #6/#4 mesh ~ 1' (both docs)
# --- bolting (doc [A] minimum vs doc [B] Creighton face standard) ---
PATTERN = "4'x5'"         # face bolt pattern (both docs)
CMTS_MIN_BOLTS_PER_SHEET = 4   # [A] leading-edge 4'x5' minimum = 4 corner bolts
DIV6_BOLTS_PER_SHEET = 6       # [B] Creighton face: 6 bolts/sheet, 3-0-3
# --- bolt-grid extents (doc [A], Scenario 1) for the informational full-pattern count ---
WALL_OFFSET_FT = 1.5
BACK_OFFSET_FT = 2.0
BOR_OFFSET_FT = 5.0
H_SPACING_FT = 5.0        # 4'x5' -> 5' across the face (max)
V_SPACING_FT = 4.0        # 4'x5' -> 4' up the face


def meshes_required(face_width_ft, sheet_w=SHEET_W_FT, overlap=OVERLAP_FT):
    """Screen sheets to span the face WIDTH with the required overlap. Face screens are
    full-height vertical strips (cross-sections show one face screen wrapped brow->BOR),
    so the count is driven by width. EMERGENT from the measured size."""
    advance = max(0.1, sheet_w - overlap)
    return max(1, math.ceil((face_width_ft - overlap) / advance))


def bolt_grid(face_width_ft, face_height_ft):
    """Informational full 4'x5' bolt grid over the face within doc [A] extents."""
    usable_w = max(0.0, face_width_ft - 2 * WALL_OFFSET_FT)
    cols = int(math.ceil(usable_w / H_SPACING_FT)) + 1
    usable_h = max(0.0, face_height_ft - BACK_OFFSET_FT - BOR_OFFSET_FT)
    rows = int(math.ceil(usable_h / V_SPACING_FT)) + 1
    return {"cols": cols, "rows": rows, "grid_bolts": cols * rows}


def mesh_layout(face_width_m, face_height_m, sheet_w=SHEET_W_FT, overlap=OVERLAP_FT):
    """Concrete screen layout for a measured face (feet, face-front coords, origin = bottom-left):
    the panel x-spans across the width (overlapping per the doc) and the 4'x5' bolt grid within
    the doc extents. Makes 'N meshes' inspectable, not just a count."""
    w_ft, h_ft = face_width_m / FT, face_height_m / FT
    nm = meshes_required(w_ft, sheet_w, overlap)
    advance = sheet_w - overlap
    panels = []
    for i in range(nm):
        x0 = i * advance
        x1 = min(x0 + sheet_w, w_ft)
        if i == nm - 1:                      # last panel pinned to the far wall
            x1 = w_ft
            x0 = max(0.0, x1 - sheet_w)
        panels.append({"i": i + 1, "x0": round(x0, 2), "x1": round(x1, 2)})
    # 4'x5' bolt grid within extents: <=1.5' from walls (H_SPACING across), top <=2' from
    # back and bottom to within 5' of BOR (V_SPACING up).
    g = bolt_grid(w_ft, h_ft)
    xs = list(np.linspace(WALL_OFFSET_FT, w_ft - WALL_OFFSET_FT, g["cols"])) if g["cols"] > 1 else [w_ft / 2]
    ys = list(np.linspace(BOR_OFFSET_FT, h_ft - BACK_OFFSET_FT, g["rows"])) if g["rows"] > 1 else [h_ft / 2]
    bolts = [{"x": round(x, 2), "y": round(y, 2)} for y in ys for x in xs]
    return {"face_width_ft": round(w_ft, 1), "face_height_ft": round(h_ft, 1),
            "n_meshes": nm, "sheet_w_ft": sheet_w, "overlap_ft": overlap,
            "panels": panels, "bolt_cols": g["cols"], "bolt_rows": g["rows"], "bolts": bolts}


def mesh_count_confidence(face_width_m, width_unc_m=0.13, sheet_w=SHEET_W_FT, overlap=OVERLAP_FT):
    """How confident is the mesh count, given the measurement uncertainty? Returns the face-
    width BAND that yields the same count and the margin from the measured width to the
    nearest band edge — `robust` if that margin exceeds 2x the measurement uncertainty.
    A precise measurement is only as useful as its margin to the next count boundary."""
    advance = sheet_w - overlap
    nm = meshes_required(face_width_m / FT, sheet_w, overlap)   # meshes_required takes FEET
    lo_m = ((nm - 1) * advance + overlap) * FT      # exclusive lower edge of this count's band
    hi_m = (nm * advance + overlap) * FT            # inclusive upper edge
    margin = min(face_width_m - lo_m, hi_m - face_width_m)
    return {"meshes": nm, "width_band_m": [round(lo_m, 2), round(hi_m, 2)],
            "margin_m": round(float(margin), 2), "width_unc_m": width_unc_m,
            "robust": bool(margin > 2 * width_unc_m)}


def calc(face_width_m, face_height_m):
    """Full requirement for a measured face. Returns meshes + bolt counts under both the
    CMTS minimum (leading-edge) and the Div6 Creighton face standard, plus the bolt grid."""
    w_ft, h_ft = face_width_m / FT, face_height_m / FT
    nm = meshes_required(w_ft)
    grid = bolt_grid(w_ft, h_ft)
    return {
        "face_width_m": round(face_width_m, 2), "face_height_m": round(face_height_m, 2),
        "face_width_ft": round(w_ft, 1), "face_height_ft": round(h_ft, 1),
        "meshes_required": nm,
        "sheet_w_ft": SHEET_W_FT, "overlap_ft": OVERLAP_FT, "pattern": PATTERN,
        # leading-edge minimum (CMTS-2015-001) — matches the observed/operator definition
        "bolts_per_sheet_min": CMTS_MIN_BOLTS_PER_SHEET,
        "bolts_required_min": nm * CMTS_MIN_BOLTS_PER_SHEET,
        # Creighton Div6 face standard (3-0-3)
        "bolts_per_sheet_div6": DIV6_BOLTS_PER_SHEET,
        "bolts_required_div6": nm * DIV6_BOLTS_PER_SHEET,
        # informational full grid from the 4'x5' pattern + extents
        "bolt_grid": grid,
        # confidence: margin from the measured width to the next mesh-count boundary
        "count_confidence": mesh_count_confidence(face_width_m),
        "sources": ["CMTS-2015-001 Rev5 (All Mines Face Support)",
                    "Division 6 Lateral Development Support Standard Rev4 (Creighton)"],
    }


def _fmt(r):
    cc = r["count_confidence"]
    return (f"face {r['face_width_m']}x{r['face_height_m']} m "
            f"({r['face_width_ft']}x{r['face_height_ft']} ft)\n"
            f"  meshes needed : {r['meshes_required']}  "
            f"(6' sheets, 1' overlap -> {SHEET_W_FT-OVERLAP_FT}' advance)\n"
            f"  count confidence: {'ROBUST' if cc['robust'] else 'MARGINAL'} "
            f"(band {cc['width_band_m'][0]}-{cc['width_band_m'][1]} m, "
            f"margin {cc['margin_m']} m vs +/-{cc['width_unc_m']} m measure)\n"
            f"  bolts (CMTS-2015-001 min, 4/sheet) : {r['bolts_required_min']}\n"
            f"  bolts (Div6 Creighton face, 6/sheet 3-0-3) : {r['bolts_required_div6']}\n"
            f"  full 4'x5' grid (informational) : {r['bolt_grid']['cols']}x{r['bolt_grid']['rows']} "
            f"= {r['bolt_grid']['grid_bolts']} bolts")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Vale face-support requirement from a face size")
    ap.add_argument("--width", type=float, help="face width (m); omit to measure with lidar")
    ap.add_argument("--height", type=float, default=5.66)
    ap.add_argument("--bags", default="48,50,53,55")
    a = ap.parse_args()
    if a.width is None:
        import lidar_analyzer as la
        m = la.measure_face_precise(bags=tuple(int(x) for x in a.bags.split(",")))
        if not m.get("have"):
            raise SystemExit(f"lidar measure failed: {m}")
        w, h = m["face_width"], m["face_height"]
        print(f"[lidar] measured {w}x{h} m  (pitch {m['lidar_pitch_deg']} deg, "
              f"spread {m.get('width_spread')} m, {m['n_bags']} bags)")
    else:
        w, h = a.width, a.height
    print(_fmt(calc(w, h)))
