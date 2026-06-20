"""One-off diagnostic: is the VLM person bbox STABLE across consecutive frames?

Picks a known operator event, extracts a short burst of CONSECUTIVE raw frames,
runs detect_person on each, and reports how much the bbox jitters frame-to-frame.
If the operator is standing still but the box jumps around, the masked motion is
an artifact of the bbox, not the boom.
"""
from __future__ import annotations
import glob
import statistics as st
from pathlib import Path
import cv2
import numpy as np
import yaml
import operator_safety as osf
from extract_video import iter_frames

CFG = yaml.safe_load(Path("config.yaml").read_text())
TOPIC = CFG["camera_topic"]
BASE = "/home/nvidia/rosbags/vale/20260611_115532"


def bagpath(n):
    g = glob.glob(f"{BASE}/*_{n}.bag")
    return g[0] if g else None


def first_ts(bag):
    for ts, _ in iter_frames([bag], TOPIC, 0.0, None):
        return ts
    return None


def probe(target_cyc, bag_n, n_frames=12, half_window=0.7):
    bag0, bagN = bagpath(0), bagpath(bag_n)
    t0 = first_ts(bag0)
    tN = first_ts(bagN)
    bagN_start_global = (tN - t0) / 1e9
    local = target_cyc - bagN_start_global
    print(f"\n===== EVENT cyc={target_cyc}s  bag={bag_n}  (bag starts at global {bagN_start_global:.1f}s, local offset {local:.1f}s) =====")

    frames = []
    for ts, fr in iter_frames([bagN], TOPIC, max(0.0, local - half_window), 2 * half_window):
        frames.append(fr)
    if len(frames) < 2:
        print(f"  only {len(frames)} frames in window -- skip")
        return
    # subsample evenly to n_frames
    idx = np.linspace(0, len(frames) - 1, min(n_frames, len(frames))).round().astype(int)
    sub = [frames[i] for i in idx]
    print(f"  {len(frames)} raw frames in {2*half_window:.1f}s window; probing {len(sub)} of them")

    dets = []
    for i, fr in enumerate(sub):
        d = osf.detect_person(fr, CFG)
        dets.append(d)
        bb = d.get("person_bbox")
        bbs = "None" if not bb else "[" + ",".join(f"{x:.3f}" for x in bb) + "]"
        print(f"  f{i:02d}: person_in_front={int(d['person_in_front'])} vlm={int(d['vlm_person'])} "
              f"orange={d['orange_frac']:.3f} bbox={bbs}")

    # jitter stats over frames that produced a confirmed bbox
    boxes = [d["person_bbox"] for d in dets if d.get("person_bbox")]
    print(f"\n  confirmed person_in_front: {sum(d['person_in_front'] for d in dets)}/{len(dets)} frames")
    if len(boxes) >= 2:
        cx = [(b[0] + b[2]) / 2 for b in boxes]
        cy = [(b[1] + b[3]) / 2 for b in boxes]
        ww = [b[2] - b[0] for b in boxes]
        hh = [b[3] - b[1] for b in boxes]
        print(f"  center_x: mean={st.mean(cx):.3f} std={st.pstdev(cx):.3f} range={max(cx)-min(cx):.3f}")
        print(f"  center_y: mean={st.mean(cy):.3f} std={st.pstdev(cy):.3f} range={max(cy)-min(cy):.3f}")
        print(f"  width   : mean={st.mean(ww):.3f} std={st.pstdev(ww):.3f} range={max(ww)-min(ww):.3f}")
        print(f"  height  : mean={st.mean(hh):.3f} std={st.pstdev(hh):.3f} range={max(hh)-min(hh):.3f}")

    # how much does the MOTION reading depend on the bbox?
    shared = next((b for b in boxes), None)
    mot_nomask, mot_shared, mot_self = [], [], []
    for i in range(1, len(sub)):
        mot_nomask.append(osf.arm_motion(sub[i - 1], sub[i], None))
        mot_shared.append(osf.arm_motion(sub[i - 1], sub[i], shared))
        bb_i = dets[i].get("person_bbox")
        mot_self.append(osf.arm_motion(sub[i - 1], sub[i], bb_i))
    def pk(v): return max(v) if v else 0.0
    print(f"\n  arm_motion PEAK over the burst:")
    print(f"    no mask          : {pk(mot_nomask):.3f}")
    print(f"    shared 1st bbox  : {pk(mot_shared):.3f}   (what the pipeline does)")
    print(f"    per-frame bbox   : {pk(mot_self):.3f}")
    print(f"    threshold        : {osf.MOTION_FRAC_THRESH}  -> pipeline calls it "
          f"{'MOVING/DANGER' if pk(mot_shared) > osf.MOTION_FRAC_THRESH else 'stopped/OK'}")

    # overlay all boxes on the middle frame for visual jitter check
    mid = sub[len(sub) // 2].copy()
    h, w = mid.shape[:2]
    y0, y1, x0, x1 = osf.DANGER_ROI
    cv2.rectangle(mid, (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h)), (0, 220, 220), 2)
    cols = [(0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0),
            (255, 0, 0), (255, 0, 255), (200, 200, 200), (128, 0, 255), (0, 128, 255),
            (128, 255, 0), (255, 128, 0)]
    for i, d in enumerate(dets):
        bb = d.get("person_bbox")
        if not bb:
            continue
        c = cols[i % len(cols)]
        cv2.rectangle(mid, (int(bb[0] * w), int(bb[1] * h)), (int(bb[2] * w), int(bb[3] * h)), c, 2)
    out = f"data/operator_frames/JITTER_cyc{int(target_cyc)}.png"
    cv2.imwrite(out, mid)
    print(f"  overlay of all {len(dets)} bboxes -> {out}")


if __name__ == "__main__":
    # a real operator at left + booms (FP, motion 0.145), and the boom-hallucination case
    probe(2225, 37)
    probe(1529, 25)
