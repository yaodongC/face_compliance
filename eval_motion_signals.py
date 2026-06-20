"""Test candidate 'boom moving' signals on labelled eval clips.

MOVING (active drilling): drill_b0,b3,b40,b50    -> signal should be HIGH
PARKED (booms to sides)  : sup_b54,b55,b56        -> signal should be LOW

Signals (all in the danger ROI, over frame pairs ~DT apart):
  peak_diff   : max  fraction of pixels changed (>25)  [current pipeline style]
  med_diff    : median fraction changed              [robust replacement]
  p80_diff    : 80th pct fraction changed
  yel_diff    : median fraction of YELLOW pixels that changed (boom intensity motion)
  yel_xor     : median (yellow_prev XOR yellow_cur)/union  -> boom SILHOUETTE displacement
  yel_cov     : mean yellow coverage in ROI (sanity: is the boom even in the ROI?)
"""
from __future__ import annotations
import statistics as st
import cv2
import numpy as np

ROI = (0.45, 1.0, 0.20, 0.80)      # operator_safety.DANGER_ROI
PX = 25                             # per-pixel abs-diff threshold (motion_px_thresh)
DT = 0.25                           # seconds between the two frames of a pair
YLO = (18, 60, 60)                 # yellow boom HSV (tuned below)
YHI = (40, 255, 255)

CLIPS = {
    "drill_b0": "MOVING", "drill_b3": "MOVING", "drill_b40": "MOVING", "drill_b50": "MOVING",
    "sup_b54": "PARKED", "sup_b55": "PARKED", "sup_b56": "PARKED",
}


def roi(img):
    h, w = img.shape[:2]
    y0, y1, x0, x1 = ROI
    return img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


def yellow(bgr_roi):
    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    return (cv2.inRange(hsv, YLO, YHI) > 0)


def signals_for_clip(path, max_pairs=60):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    step = max(1, int(round(DT * fps)))
    frames = []
    ok, fr = cap.read()
    while ok:
        frames.append(fr)
        ok, fr = cap.read()
    cap.release()
    if len(frames) < step + 1:
        return None
    diffs, yel_diffs, yel_xors, yel_covs = [], [], [], []
    idxs = list(range(0, len(frames) - step, step))
    if len(idxs) > max_pairs:
        idxs = [idxs[i] for i in np.linspace(0, len(idxs) - 1, max_pairs).round().astype(int)]
    for i in idxs:
        a, b = roi(frames[i]), roi(frames[i + step])
        ga, gb = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        changed = (cv2.absdiff(ga, gb) > PX)
        diffs.append(float(changed.mean()))
        ya, yb = yellow(a), yellow(b)
        union = (ya | yb).sum()
        yel_covs.append(float(yb.mean()))
        if yb.sum() > 0:
            yel_diffs.append(float((changed & yb).sum()) / float(yb.sum()))
        if union > 0:
            yel_xors.append(float((ya ^ yb).sum()) / float(union))
    def pct(v, p): return float(np.percentile(v, p)) if v else 0.0
    return {
        "n_pairs": len(idxs),
        "peak_diff": max(diffs) if diffs else 0.0,
        "med_diff": st.median(diffs) if diffs else 0.0,
        "p80_diff": pct(diffs, 80),
        "yel_diff": st.median(yel_diffs) if yel_diffs else 0.0,
        "yel_xor": st.median(yel_xors) if yel_xors else 0.0,
        "yel_cov": st.mean(yel_covs) if yel_covs else 0.0,
        # determinism check: median over 1st half vs 2nd half of pairs
        "med_h1": st.median(diffs[:len(diffs)//2]) if len(diffs) > 3 else 0.0,
        "med_h2": st.median(diffs[len(diffs)//2:]) if len(diffs) > 3 else 0.0,
        "xor_h1": st.median(yel_xors[:len(yel_xors)//2]) if len(yel_xors) > 3 else 0.0,
        "xor_h2": st.median(yel_xors[len(yel_xors)//2:]) if len(yel_xors) > 3 else 0.0,
    }


def main():
    rows = {}
    for name, label in CLIPS.items():
        r = signals_for_clip(f"eval/{name}.mp4")
        if r is None:
            print(f"{name}: no frames"); continue
        rows[name] = (label, r)
        print(f"{name:10} {label:7} pairs={r['n_pairs']:3} | "
              f"peak={r['peak_diff']:.3f} med={r['med_diff']:.3f} p80={r['p80_diff']:.3f} | "
              f"yel_diff={r['yel_diff']:.3f} yel_xor={r['yel_xor']:.3f} yel_cov={r['yel_cov']:.2f}")

    print("\n=== SEPARABILITY (min over MOVING vs max over PARKED; want MOVING >> PARKED) ===")
    for sig in ("peak_diff", "med_diff", "p80_diff", "yel_diff", "yel_xor"):
        mv = [r[sig] for l, r in rows.values() if l == "MOVING"]
        pk = [r[sig] for l, r in rows.values() if l == "PARKED"]
        if not mv or not pk:
            continue
        lo_mv, hi_pk = min(mv), max(pk)
        margin = lo_mv / hi_pk if hi_pk > 1e-6 else float("inf")
        ok = "PASS" if lo_mv > hi_pk else "FAIL"
        print(f"  {sig:10}: min(MOVING)={lo_mv:.3f}  max(PARKED)={hi_pk:.3f}  margin={margin:5.1f}x  [{ok}]")

    print("\n=== DETERMINISM (half1 vs half2 of same clip; want ~equal) ===")
    for name, (label, r) in rows.items():
        print(f"  {name:10} {label:7} med: {r['med_h1']:.3f}/{r['med_h2']:.3f}   xor: {r['xor_h1']:.3f}/{r['xor_h2']:.3f}")

    # dump yellow-mask overlays to verify the boom is captured and operator/rock are not
    for name in ("drill_b40", "sup_b55"):
        cap = cv2.VideoCapture(f"eval/{name}.mp4")
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) // 2))
        ok, fr = cap.read(); cap.release()
        if not ok:
            continue
        h, w = fr.shape[:2]
        y0, y1, x0, x1 = ROI
        vis = fr.copy()
        ym = np.zeros((h, w), np.uint8)
        ry = yellow(roi(fr)).astype(np.uint8) * 255
        ym[int(y0*h):int(y1*h), int(x0*w):int(x1*w)] = ry
        vis[ym > 0] = (0, 0, 255)
        cv2.rectangle(vis, (int(x0*w), int(y0*h)), (int(x1*w), int(y1*h)), (0, 220, 220), 2)
        cv2.imwrite(f"data/operator_frames/YELLOWMASK_{name}.png", vis)
        print(f"yellow-mask overlay -> data/operator_frames/YELLOWMASK_{name}.png")


if __name__ == "__main__":
    main()
