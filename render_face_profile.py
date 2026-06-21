"""Draw the Lidar-measured face cross-section (arched profile) — true size, not a bounding box.
  python3 render_face_profile.py [--bag 55] [--out data/face_profile.png]
"""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import numpy as np
import lidar_analyzer as la


def render(prof, out="data/face_profile.png"):
    if not prof.get("have"):
        raise SystemExit(f"no profile: {prof}")
    H = prof["height_m"]
    bands = prof["profile"]
    scale = 90.0                        # px per metre
    pad = 60
    maxw = max(b["width_m"] for b in bands)
    W_px, H_px = int(maxw * scale), int(H * scale)
    img = np.full((H_px + 2 * pad, W_px + 2 * pad, 3), 26, np.uint8)
    cx = pad + W_px // 2

    def Y(hm):  # height(m from floor) -> px (inverted)
        return int(pad + H_px - hm / H * H_px)
    # arch outline (left + right walls from the width profile)
    left = [(cx - int(b["width_m"] / 2 * scale), Y(b["height_m"])) for b in bands]
    right = [(cx + int(b["width_m"] / 2 * scale), Y(b["height_m"])) for b in bands]
    poly = np.array(left + right[::-1], np.int32)
    ov = img.copy(); cv2.fillPoly(ov, [poly], (70, 55, 40)); cv2.addWeighted(ov, 0.5, img, 0.5, 0, img)
    cv2.polylines(img, [poly], True, (90, 200, 255), 2, cv2.LINE_AA)
    for b in bands:                     # width ticks
        y = Y(b["height_m"]); xl = cx - int(b["width_m"] / 2 * scale); xr = cx + int(b["width_m"] / 2 * scale)
        cv2.line(img, (xl, y), (xr, y), (60, 60, 70), 1)
        cv2.putText(img, f"{b['width_m']:.1f}", (xr + 6, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (170, 170, 170), 1, cv2.LINE_AA)
    cv2.putText(img, f"FACE CROSS-SECTION (lidar)  {prof['max_width_m']:.2f} m wide x {H:.2f} m high",
                (pad, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f"arched: area {prof['area_m2']} m2 (vs {prof['max_width_m']*H:.1f} m2 bounding box)  "
                f"| mesh count uses max(springline) width",
                (pad, H_px + 2 * pad - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 190, 190), 1, cv2.LINE_AA)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out, img)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", type=int, default=55)
    ap.add_argument("--out", default="data/face_profile.png")
    a = ap.parse_args()
    prof = la.face_profile(bag=a.bag)
    render(prof, a.out)
    print(f"=== face cross-section -> {a.out} ===")
    print(f"  height {prof['height_m']} m, max width {prof['max_width_m']} m, area {prof['area_m2']} m2 "
          f"(bounding {prof['max_width_m']*prof['height_m']:.1f} m2)")
    for b in prof["profile"]:
        print(f"   h={b['h_frac']*100:3.0f}%  width {b['width_m']} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
