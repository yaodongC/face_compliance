"""Full-res look at the IMU-active moments: is the flickering 'person' a real worker or
a hallucination on the moving booms? Saves full-resolution annotated frames where the VLM
fired, so we can judge real-vs-hallucination for the safety claim.
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


def inspect(n, t0, center, tag, half=2.0):
    saved = 0
    with AnyReader([Path(bag(n))]) as r:
        for c, ts, raw in r.messages(connections=[c for c in r.connections if c.topic == CAM]):
            cyc = (ts - t0) / 1e9
            if cyc < center - half:
                continue
            if cyc > center + half:
                break
            fr = decode_compressed(bytes(r.deserialize(raw, c.msgtype).data))
            d = osf.detect_person(fr, CFG)
            if d["person_in_front"] and saved < 2:
                img = osf.annotate(fr, d.get("person_bbox"), "REVIEW", d.get("action", ""), None, cyc)
                out = f"data/operator_frames/FULLRES_{tag}_{cyc:.1f}.png"
                cv2.imwrite(out, img)
                print(f"  {tag} cyc={cyc:.1f}: person bbox={d.get('person_bbox')} "
                      f"orange={d['orange_frac']:.3f} act='{d.get('action','')}' -> {out}")
                saved += 1
    if saved == 0:
        print(f"  {tag}: no confirmed-person frame to save in window")


def main():
    t0 = cam_t0_bag0()
    inspect(21, t0, 1265, "b21_1265")
    inspect(25, t0, 1497, "b25_1497")


if __name__ == "__main__":
    main()
