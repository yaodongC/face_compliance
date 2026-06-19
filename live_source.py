"""Unified frame source for the harness: an MP4 file OR a live RTSP camera.

The harness was built to read MP4s; this adds a live pipeline so the same perception
code can run on a camera. RTSP is decoded with the Jetson hardware decoder
(`nvh264dec`) via GStreamer, dropping to the latest frame (low latency) and
auto-reconnecting on drop.

    src = open_source("rtsp://${RTSP_USER}:${RTSP_PASS}@10.20.30.40:554/cam0_0")   # live
    src = open_source("data/full_cycle.mp4")                       # file
    for ts, frame in src.frames():
        ...
"""
from __future__ import annotations
import os
import time
from pathlib import Path
import cv2


def rtsp_pipeline(url: str, latency: int = 200, decoder: str = "nvh264dec") -> str:
    """GStreamer pipeline string -> BGR appsink for OpenCV. Falls back to software
    `avdec_h264` if the hardware decoder is unavailable on a given box."""
    return (f"rtspsrc location={url} protocols=tcp latency={latency} ! "
            f"rtph264depay ! h264parse ! {decoder} ! videoconvert ! "
            "video/x-raw,format=BGR ! appsink drop=1 max-buffers=1 sync=false")


class LiveSource:
    """Frame source over an MP4 path or an rtsp:// URL. `.frames()` yields
    (wall_time, bgr); for RTSP it reconnects on failure."""

    def __init__(self, spec: str, latency: int = 200, reconnect: bool = True):
        # expand ${RTSP_USER}/${RTSP_PASS} etc. from the environment so credentials
        # are never committed in config; the literal URL is only ever in memory
        self.spec = os.path.expandvars(spec)
        self.is_rtsp = self.spec.startswith("rtsp://")
        self.latency = latency
        self.reconnect = reconnect and self.is_rtsp
        self.cap = None

    def _open(self):
        if self.is_rtsp:
            cap = cv2.VideoCapture(rtsp_pipeline(self.spec, self.latency), cv2.CAP_GSTREAMER)
            if not cap.isOpened():               # fall back to software decode
                cap = cv2.VideoCapture(rtsp_pipeline(self.spec, self.latency, "avdec_h264"),
                                       cv2.CAP_GSTREAMER)
            return cap
        return cv2.VideoCapture(self.spec)

    def open(self):
        self.cap = self._open()
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(f"could not open source: {self.spec}")
        return self

    def read(self):
        ok, frame = self.cap.read()
        if not ok and self.reconnect:
            self.cap.release()
            time.sleep(1.0)
            self.cap = self._open()
            ok, frame = self.cap.read()
        return ok, frame

    def frames(self, max_frames=None, warmup=3):
        if self.cap is None:
            self.open()
        for _ in range(warmup):                  # discard initial buffered/black frames
            self.cap.read()
        n = 0
        while max_frames is None or n < max_frames:
            ok, frame = self.read()
            if not ok:
                if self.is_rtsp and self.reconnect:
                    continue
                break
            yield time.time(), frame
            n += 1

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def open_source(spec: str, **kw) -> LiveSource:
    return LiveSource(spec, **kw).open()
