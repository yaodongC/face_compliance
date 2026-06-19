"""Build a 2x real-time video of the full cycle from the bags (smooth, not the
28x timelapse), plus a re-based analysis so the GUI renderer lines up.

At 2x and out_fps=15 we keep one frame every 2/15 s of real footage (decoding only
the kept frames -- the timestamp is checked before the JPEG is decoded, so the other
~5/6 of frames are skipped cheaply). The frame index stores cycle seconds directly,
and the analysis steps are re-timed to the 2x output clock (t_out = cycle / speed).
"""
from __future__ import annotations
import argparse
import glob
import json
import re
from pathlib import Path
import cv2
from rosbags.highlevel import AnyReader
from rosbags.highlevel.anyreader import AnyReaderError
from extract_video import decode_compressed


def _bags(base):
    bags = glob.glob(f"{base}/*.bag")
    return sorted(bags, key=lambda p: int(re.search(r"_(\d+)\.bag$", p).group(1)))


def extract(base, topic, out_mp4, out_idx, speed, out_fps, width):
    dt = speed / out_fps
    t0 = last = None
    n = 0
    writer = None
    Path(out_mp4).parent.mkdir(parents=True, exist_ok=True)
    with open(out_idx, "w") as idx:
        idx.write("frame_no,cycle_sec\n")
        for bag in _bags(base):
            try:
                r = AnyReader([Path(bag)]); r.open()
            except AnyReaderError as e:
                print(f"skip {Path(bag).name}: {e}", flush=True)
                continue
            try:
                conns = [c for c in r.connections if c.topic == topic]
                for conn, ts_ns, raw in r.messages(connections=conns):
                    if t0 is None:
                        t0 = ts_ns
                    cyc = (ts_ns - t0) / 1e9
                    if last is not None and cyc - last < dt - 1e-3:
                        continue                       # skip without decoding
                    last = cyc
                    frame = decode_compressed(bytes(r.deserialize(raw, conn.msgtype).data))
                    h, w = frame.shape[:2]
                    if width and w != width:
                        frame = cv2.resize(frame, (width, int(h * width / w)))
                    if writer is None:
                        H, W = frame.shape[:2]
                        writer = cv2.VideoWriter(str(out_mp4),
                                                 cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (W, H))
                    writer.write(frame)
                    idx.write(f"{n},{cyc:.3f}\n")
                    n += 1
                    if n % 500 == 0:
                        print(f"  {n} frames, cycle {int(cyc)//60:02d}:{int(cyc)%60:02d}", flush=True)
            except AnyReaderError as e:
                print(f"truncated {Path(bag).name}: {e}", flush=True)
            finally:
                r.close()
    if writer:
        writer.release()
    print(f"wrote {n} frames to {out_mp4} ({n/out_fps:.0f}s at {out_fps}fps)", flush=True)
    return n


def rebase_analysis(analysis, tl_index, out, speed):
    """Re-time analysis steps onto the 2x output clock using the TIMELAPSE index to
    convert each step's timelapse-time -> cycle-time -> output-time."""
    data = json.loads(Path(analysis).read_text())
    idx = {int(f): float(c) for f, c in
           (l.split(",") for l in Path(tl_index).read_text().splitlines()[1:])}
    keys = sorted(idx)
    for s in data["steps"]:
        fr = int(round(s["t_sec"] * 15))                  # timelapse fps = 15
        k = min(keys, key=lambda x: abs(x - fr))
        s["t_sec"] = round(idx[k] / speed, 3)             # -> 2x output seconds
    Path(out).write_text(json.dumps(data))
    print(f"re-based {len(data['steps'])} analysis steps -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/home/nvidia/rosbags/vale/20260611_115532")
    ap.add_argument("--topic", default="/sensing/front/rgb/image_raw/compressed")
    ap.add_argument("--speed", type=float, default=2.0)
    ap.add_argument("--out-fps", type=float, default=15.0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--out", default="data/full_cycle_2x.mp4")
    ap.add_argument("--index", default="data/full_cycle_2x.idx")
    ap.add_argument("--analysis", default="data/full_cycle_analysis.json")
    ap.add_argument("--tl-index", default="data/full_cycle.idx")
    ap.add_argument("--out-analysis", default="data/full_cycle_2x_analysis.json")
    a = ap.parse_args()
    extract(a.base, a.topic, a.out, a.index, a.speed, a.out_fps, a.width)
    rebase_analysis(a.analysis, a.tl_index, a.out_analysis, a.speed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
