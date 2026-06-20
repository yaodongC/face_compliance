"""Orange fraction at CONFIRMED real-operator OK_LOADING moments (high persistence), to set a
safe min_orange that rejects boom hallucinations (~0.02) without rejecting real vests."""
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
# (bag, cyc) of confirmed real operators (OK_LOADING, persist>=0.83 in the IMU-fused run)
TARGETS = [(12, 757), (14, 837), (16, 997)]


def bag(n):
    return glob.glob(f"{BASE}/*_{n}.bag")[0]


def cam_t0():
    with AnyReader([Path(bag(0))]) as r:
        for c, ts, raw in r.messages(connections=[c for c in r.connections if c.topic == CAM]):
            return ts


def main():
    t0 = cam_t0()
    for n, target in TARGETS:
        got = 0
        with AnyReader([Path(bag(n))]) as r:
            for c, ts, raw in r.messages(connections=[c for c in r.connections if c.topic == CAM]):
                cyc = (ts - t0) / 1e9
                if cyc < target - 0.5:
                    continue
                if cyc > target + 2.5 or got >= 3:
                    break
                fr = decode_compressed(bytes(r.deserialize(raw, c.msgtype).data))
                d = osf.detect_person(fr, CFG)
                if d["vlm_person"]:
                    print(f"bag{n} cyc={cyc:.1f} REAL-OP orange={d['orange_frac']:.3f} "
                          f"confirmed={int(d['person_in_front'])} bbox={d.get('person_bbox')}")
                    got += 1
    print(f"\ncurrent min_orange={osf.MIN_ORANGE} | halluc 1497=0.018, 1529~0.02-0.04 | proposed ~0.06")


if __name__ == "__main__":
    main()
