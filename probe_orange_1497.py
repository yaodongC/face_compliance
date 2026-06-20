"""Is the cyc=1497 DANGER a hi-vis worker or a yellow-boom hallucination? Print the VLM
orange fraction at a few frames -> does raising min_orange separate it from a real vest?"""
from __future__ import annotations
import glob
from pathlib import Path
import yaml
from rosbags.highlevel import AnyReader
from extract_video import decode_compressed
import operator_safety as osf

BASE = "/home/nvidia/rosbags/vale/20260611_115532"
CAM = "/sensing/front/rgb/image_raw/compressed"
CFG = yaml.safe_load(Path("config.yaml").read_text())


def bag(n):
    return glob.glob(f"{BASE}/*_{n}.bag")[0]


def cam_t0():
    with AnyReader([Path(bag(0))]) as r:
        for c, ts, raw in r.messages(connections=[c for c in r.connections if c.topic == CAM]):
            return ts


def main():
    t0 = cam_t0()
    n = 0
    with AnyReader([Path(bag(25))]) as r:
        for c, ts, raw in r.messages(connections=[c for c in r.connections if c.topic == CAM]):
            cyc = (ts - t0) / 1e9
            if cyc < 1495.5:
                continue
            if cyc > 1499.0:
                break
            fr = decode_compressed(bytes(r.deserialize(raw, c.msgtype).data))
            d = osf.detect_person(fr, CFG)
            if d["vlm_person"]:
                print(f"cyc={cyc:.1f} vlm={int(d['vlm_person'])} confirmed={int(d['person_in_front'])} "
                      f"orange={d['orange_frac']:.3f} bbox={d.get('person_bbox')} act='{d.get('action','')}'")
                n += 1
            if n >= 6:
                break
    print(f"\ncurrent min_orange gate = {osf.MIN_ORANGE}")
    print("(real vest measured ~0.11-0.20 at cyc=2225; boom hallucinations ~0.02-0.04 at cyc=1529)")


if __name__ == "__main__":
    main()
