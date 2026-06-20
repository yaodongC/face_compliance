"""Safety-critical test: at the IMU-ACTIVE moments, is a REAL operator in front?

If machine-active never coincides with a confirmed (temporally persistent) operator,
then there is no true danger in this session and all 17 vision-DANGERs are false positives.
For each active moment: extract a burst, run detect_person on every frame (temporal
persistence), and save a contact sheet to eyeball.
"""
from __future__ import annotations
import glob
from pathlib import Path
import cv2, numpy as np, yaml
from rosbags.highlevel import AnyReader
from extract_video import decode_compressed
import operator_safety as osf

BASE = "/home/nvidia/rosbags/vale/20260611_115532"
CAM = "/sensing/front/rgb/image_raw/compressed"
CFG = yaml.safe_load(Path("config.yaml").read_text())


def bag(n):
    g = glob.glob(f"{BASE}/*_{n}.bag"); return g[0] if g else None


def cam_t0_bag0():
    with AnyReader([Path(bag(0))]) as r:
        for c, ts, raw in r.messages(connections=[c for c in r.connections if c.topic == CAM]):
            return ts


def check(n, t0, center, tag, half=2.0, k=8):
    frames = []
    with AnyReader([Path(bag(n))]) as r:
        for c, ts, raw in r.messages(connections=[c for c in r.connections if c.topic == CAM]):
            cyc = (ts - t0) / 1e9
            if cyc < center - half:
                continue
            if cyc > center + half:
                break
            frames.append((cyc, decode_compressed(bytes(r.deserialize(raw, c.msgtype).data))))
    if not frames:
        print(f"{tag}: no frames"); return
    idx = np.linspace(0, len(frames) - 1, min(k, len(frames))).round().astype(int)
    tiles, n_person, n_vlm = [], 0, 0
    for i in idx:
        cyc, fr = frames[i]
        d = osf.detect_person(fr, CFG)
        n_person += int(d["person_in_front"]); n_vlm += int(d["vlm_person"])
        col = (40, 40, 220) if d["person_in_front"] else ((0, 165, 255) if d["vlm_person"] else (60, 200, 60))
        t = cv2.resize(fr, (360, 203))
        if d.get("person_bbox"):
            b = d["person_bbox"]; h, w = t.shape[:2]
            cv2.rectangle(t, (int(b[0]*w), int(b[1]*h)), (int(b[2]*w), int(b[3]*h)), col, 2)
        cv2.putText(t, f"{cyc:.1f} P{int(d['person_in_front'])}V{int(d['vlm_person'])} o{d['orange_frac']:.2f}",
                    (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        tiles.append(t)
    out = f"data/operator_frames/ACTIVE_{tag}.png"
    cv2.imwrite(out, np.hstack(tiles))
    print(f"{tag}: {len(idx)} frames | confirmed person_in_front={n_person}/{len(idx)} | vlm_person={n_vlm}/{len(idx)} -> {out}")


def main():
    t0 = cam_t0_bag0()
    for n, center, tag in ((21, 1265, "b21_cyc1265"), (25, 1497, "b25_cyc1497"),
                           (25, 1529, "b25_cyc1529"), (25, 1537, "b25_cyc1537")):
        check(n, t0, center, tag)


if __name__ == "__main__":
    main()
