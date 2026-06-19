import numpy as np
from operator_safety import classify, arm_motion, MOTION_FRAC_THRESH


def test_classify_danger_person_and_moving():
    assert classify(True, 0.10) == "DANGER"


def test_classify_ok_loading_person_and_stopped():
    assert classify(True, 0.001) == "OK_LOADING"


def test_classify_no_person():
    assert classify(False, 0.20) == "NO_PERSON"


def test_motion_threshold_boundary():
    # at/below threshold with a person = stopped/compliant, just above = danger
    assert classify(True, MOTION_FRAC_THRESH) == "OK_LOADING"
    assert classify(True, MOTION_FRAC_THRESH + 0.001) == "DANGER"


def test_arm_motion_zero_for_identical_frames():
    img = (np.random.rand(120, 200, 3) * 255).astype(np.uint8)
    assert arm_motion(img, img) == 0.0


def test_arm_motion_high_for_changed_frames():
    a = np.zeros((120, 200, 3), dtype=np.uint8)
    b = np.full((120, 200, 3), 255, dtype=np.uint8)  # whole frame flips
    assert arm_motion(a, b) > 0.5
