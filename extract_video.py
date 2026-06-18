"""Extract the front RGB camera stream from ROS1 bags into an MP4.

ROS2's rosbag2 cannot read ROS1 .bag files; we use the pure-Python `rosbags`
library's AnyReader. CompressedImage payloads are JPEG -> cv2.imdecode.
"""
from __future__ import annotations
import argparse
import glob
from pathlib import Path
import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.highlevel.anyreader import AnyReaderError


def decode_compressed(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cv2 could not decode compressed image")
    return img


def find_camera_topic(reader, prefer: str = "front") -> str:
    img_conns = [c for c in reader.connections
                 if "CompressedImage" in c.msgtype or "Image" in c.msgtype]
    if not img_conns:
        raise ValueError("no image topics in bag")
    preferred = [c for c in img_conns if prefer in c.topic and "Compressed" in c.msgtype]
    chosen = preferred or sorted(img_conns, key=lambda c: -c.msgcount)
    return chosen[0].topic


def iter_frames(bags, topic, start_sec=0.0, duration_sec=None):
    """Yield (bag_time_ns, bgr) for `topic` across bags in order.

    Skips a bag whose index is damaged/truncated and continues with the next.
    `start_sec`/`duration_sec` are measured from the first bag's start time.
    """
    t0 = None
    for bag in bags:
        try:
            reader_cm = AnyReader([Path(bag)])
            reader_cm.open()
        except AnyReaderError as e:
            print(f"[extract] skipping {Path(bag).name}: {e}")
            continue
        try:
            conns = [c for c in reader_cm.connections if c.topic == topic]
            if not conns:
                continue
            for conn, ts_ns, raw in reader_cm.messages(connections=conns):
                if t0 is None:
                    t0 = ts_ns
                elapsed = (ts_ns - t0) / 1e9
                if elapsed < start_sec:
                    continue
                if duration_sec is not None and elapsed > start_sec + duration_sec:
                    return
                msg = reader_cm.deserialize(raw, conn.msgtype)
                yield ts_ns, decode_compressed(bytes(msg.data))
        except AnyReaderError as e:
            print(f"[extract] truncated read on {Path(bag).name}: {e}")
        finally:
            reader_cm.close()


def extract(bags, topic, out_mp4, frame_index, start_sec=0.0, duration_sec=None, fps=15.0):
    out_mp4 = Path(out_mp4); out_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    n = 0
    with open(frame_index, "w") as idx:
        idx.write("frame_no,bag_time_ns\n")
        for ts_ns, frame in iter_frames(bags, topic, start_sec, duration_sec):
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(out_mp4),
                                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(frame)
            idx.write(f"{n},{ts_ns}\n")
            n += 1
            if n % 100 == 0:
                print(f"[extract] {n} frames")
    if writer is not None:
        writer.release()
    print(f"[extract] wrote {n} frames to {out_mp4}")
    return n


def _expand_bags(spec: list[str]) -> list[Path]:
    out = []
    for s in spec:
        matches = sorted(glob.glob(s))
        out.extend(Path(m) for m in matches) if matches else out.append(Path(s))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bags", nargs="+", required=True,
                    help="bag paths or globs, in order")
    ap.add_argument("--topic", default="/sensing/front/rgb/image_raw/compressed")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--out", default="data/front_camera.mp4")
    ap.add_argument("--index", default="data/frames.idx")
    args = ap.parse_args()
    extract(_expand_bags(args.bags), args.topic, args.out, args.index,
            args.start, args.duration, args.fps)


if __name__ == "__main__":
    main()
