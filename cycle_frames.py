"""Grab hi-res frames at a given cycle-time from the full-cycle video + index.

full_cycle.mp4 is the full-resolution (2560x1440) time-lapse of the whole session;
full_cycle.idx maps frame_no -> cycle_sec. This grabber returns the frame(s) nearest a
requested cycle-time so the VLM tools (final confirmation, per-episode classification)
can look at the exact moment the physical signals flag.
"""
from __future__ import annotations
import csv
from pathlib import Path
import bisect
import cv2


class FrameGrabber:
    def __init__(self, video="data/full_cycle.mp4", index="data/full_cycle.idx"):
        self.cap = cv2.VideoCapture(str(video))
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open {video}")
        rows = list(csv.reader(Path(index).read_text().splitlines()))[1:]
        self.frames = [int(r[0]) for r in rows if r]
        self.times = [float(r[1]) for r in rows if r]

    def _frame_no(self, cycle_sec):
        i = bisect.bisect_left(self.times, cycle_sec)
        if i <= 0:
            return self.frames[0]
        if i >= len(self.times):
            return self.frames[-1]
        lo, hi = self.times[i - 1], self.times[i]
        return self.frames[i - 1] if abs(cycle_sec - lo) <= abs(hi - cycle_sec) else self.frames[i]

    def at(self, cycle_sec):
        fn = self._frame_no(cycle_sec)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ok, fr = self.cap.read()
        return fr if ok else None

    def around(self, cycle_sec, n=3, span=20.0):
        """n frames spread over +/- span/2 seconds around cycle_sec (for consensus)."""
        if n <= 1:
            f = self.at(cycle_sec)
            return [f] if f is not None else []
        out = []
        for k in range(n):
            t = cycle_sec - span / 2 + span * k / (n - 1)
            f = self.at(max(0.0, t))
            if f is not None:
                out.append(f)
        return out

    def release(self):
        self.cap.release()
