"""Draw the computed mesh/bolt layout on the measured face — makes 'N meshes' inspectable.

Reads data/face_geometry.json (precise lidar size) + vale_support rules and renders a
to-scale face-front schematic: the overlapping screen panels and the 4'x5' bolt grid, with
the optional accumulated-lidar height-map behind it. -> data/face_mesh_layout.png
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import vale_support as vs

FT = 0.3048
PANEL_COLORS = [(75, 180, 60), (48, 130, 245), (200, 130, 0), (230, 50, 240),
                (240, 200, 70), (60, 60, 240), (150, 30, 180), (0, 200, 120)]


def render(face_w_m, face_h_m, out="data/face_mesh_layout.png", bg_heightmap=None, meta=""):
    lay = vs.mesh_layout(face_w_m, face_h_m)
    W_ft, H_ft = lay["face_width_ft"], lay["face_height_ft"]
    scale = 760 / W_ft                       # px per foot
    fw, fh = int(W_ft * scale), int(H_ft * scale)
    pad = 70
    img = np.full((fh + 2 * pad, fw + 2 * pad, 3), 28, np.uint8)
    if bg_heightmap and Path(bg_heightmap).exists():
        hm = cv2.imread(bg_heightmap)
        if hm is not None:
            img[pad:pad + fh, pad:pad + fw] = cv2.resize(hm, (fw, fh))

    def X(xf):  # face-x (ft, from left) -> px
        return int(pad + xf * scale)

    def Y(yf):  # face-y (ft, from BOR up) -> px (image y inverted)
        return int(pad + fh - yf * scale)

    # face outline
    cv2.rectangle(img, (pad, pad), (pad + fw, pad + fh), (210, 210, 210), 2)
    # overlapping panels (translucent fills + outlines)
    for p in lay["panels"]:
        c = PANEL_COLORS[(p["i"] - 1) % len(PANEL_COLORS)]
        ov = img.copy()
        cv2.rectangle(ov, (X(p["x0"]), pad), (X(p["x1"]), pad + fh), c, -1)
        cv2.addWeighted(ov, 0.22, img, 0.78, 0, img)
        cv2.rectangle(img, (X(p["x0"]), pad + 3), (X(p["x1"]), pad + fh - 3), c, 2)
        cv2.putText(img, f"mesh {p['i']}", (X(p['x0']) + 8, pad + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2, cv2.LINE_AA)
    # bolts (4'x5' grid)
    for b in lay["bolts"]:
        cv2.circle(img, (X(b["x"]), Y(b["y"])), 7, (0, 230, 255), -1)
        cv2.circle(img, (X(b["x"]), Y(b["y"])), 7, (20, 20, 20), 1)
    # labels
    cv2.putText(img, f"FACE {W_ft:.1f} ft x {H_ft:.1f} ft  ({face_w_m:.2f} x {face_h_m:.2f} m)",
                (pad, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f"{lay['n_meshes']} meshes (6ft sheets, 1ft overlap) | "
                f"{len(lay['bolts'])} bolts ({lay['bolt_cols']}x{lay['bolt_rows']} 4'x5' grid) | {meta}",
                (pad, pad + fh + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out, img)
    return lay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geom", default="data/face_geometry.json")
    ap.add_argument("--out", default="data/face_mesh_layout.png")
    ap.add_argument("--width", type=float, help="override face width (m)")
    ap.add_argument("--height", type=float, help="override face height (m)")
    a = ap.parse_args()
    w, h, hm, meta = a.width, a.height, None, ""
    if (w is None or h is None) and Path(a.geom).exists():
        g = json.loads(Path(a.geom).read_text())
        w = w or g["face_width"]
        h = h or g["face_height"]
        hm = g.get("heightmap")
        meta = g.get("source", "") or g.get("measure_method", "")
    if w is None:
        raise SystemExit("need a face size: --width/--height or a data/face_geometry.json")
    lay = render(w, h, a.out, bg_heightmap=hm, meta="lidar-measured")
    print(f"=== mesh layout -> {a.out} ===")
    print(f"  face {lay['face_width_ft']}x{lay['face_height_ft']} ft -> {lay['n_meshes']} meshes, "
          f"{len(lay['bolts'])} bolts ({lay['bolt_cols']}x{lay['bolt_rows']})")
    for p in lay["panels"]:
        print(f"   mesh {p['i']}: x {p['x0']}-{p['x1']} ft")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
