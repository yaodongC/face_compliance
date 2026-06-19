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


def test_bbox_ok_rejects_thin_horizontal_boom():
    from operator_safety import _bbox_ok
    assert not _bbox_ok([0.40, 0.50, 0.62, 0.54])   # wide + thin = boom line


def test_bbox_ok_accepts_upright_person():
    from operator_safety import _bbox_ok
    assert _bbox_ok([0.35, 0.55, 0.45, 0.78])        # taller than wide


def test_orange_fraction_high_for_orange_low_for_yellow():
    import numpy as np, cv2
    from operator_safety import hi_vis_orange_fraction
    orange = np.zeros((100, 100, 3), np.uint8); orange[:] = (20, 120, 240)   # BGR orange
    yellow = np.zeros((100, 100, 3), np.uint8); yellow[:] = (20, 220, 240)   # BGR yellow
    bb = [0.0, 0.0, 1.0, 1.0]
    assert hi_vis_orange_fraction(orange, bb) > 0.5
    assert hi_vis_orange_fraction(yellow, bb) < 0.2


def test_coverage_state_partial_vs_full():
    from coverage import coverage_state, FACE_X
    # two overlapping panels spanning the whole face band -> COMPLIANT
    full = [{"bbox": [FACE_X[0], 0.2, 0.55, 0.8], "installed_at": 1},
            {"bbox": [0.50, 0.2, FACE_X[1], 0.8], "installed_at": 2}]
    s = coverage_state(full, 10)
    assert s["full"] and s["overlaps"] and s["verdict"] == "COMPLIANT"
    # one small panel -> NOT SUPPORTED (partial)
    part = [{"bbox": [0.3, 0.2, 0.44, 0.8], "installed_at": 1}]
    assert coverage_state(part, 10)["verdict"] == "NOT SUPPORTED"


def test_classify_sessions_entry_based():
    from operator_safety import classify_sessions
    evs = [
        {"cycle_sec": 100, "arm_motion": 0.10, "person_bbox": [0.3, 0.5, 0.4, 0.7], "action": "a"},
        {"cycle_sec": 108, "arm_motion": 0.001, "person_bbox": [0.3, 0.5, 0.4, 0.7], "action": "a"},
        {"cycle_sec": 300, "arm_motion": 0.001, "person_bbox": [0.5, 0.5, 0.6, 0.7], "action": "b"},
        {"cycle_sec": 308, "arm_motion": 0.05, "person_bbox": [0.5, 0.5, 0.6, 0.7], "action": "b"},
    ]
    sess = classify_sessions(evs, gap=20)
    assert len(sess) == 2
    assert sess[0]["verdict"] == "NON_COMPLIANT_ENTRY"   # entered while boom moving
    assert sess[1]["verdict"] == "SAFE_RELOAD"           # entered while boom stopped
