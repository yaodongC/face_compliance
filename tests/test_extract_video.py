import os
from pathlib import Path
import numpy as np
import pytest
from extract_video import decode_compressed, iter_frames

ROOT = Path(__file__).resolve().parents[1]
BAG = Path("/home/nvidia/rosbags/vale/20260611_115532/_2026-06-11-11-55-36_0.bag")
TOPIC = "/sensing/front/rgb/image_raw/compressed"


def test_decode_compressed_roundtrip():
    import cv2
    img = (np.random.rand(32, 48, 3) * 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    out = decode_compressed(buf.tobytes())
    assert out.shape[2] == 3 and out.dtype == np.uint8


@pytest.mark.skipif(not BAG.exists(), reason="sample bag not present")
def test_iter_first_front_frame_from_real_bag():
    gen = iter_frames([BAG], TOPIC, start_sec=0.0, duration_sec=2.0)
    ts, frame = next(gen)
    assert isinstance(ts, int) and frame.ndim == 3 and frame.shape[2] == 3
