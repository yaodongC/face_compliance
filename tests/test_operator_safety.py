import numpy as np
from operator_safety import (classify, arm_motion, MOTION_FRAC_THRESH,
                             machine_motion, machine_active, operator_present,
                             classify_zone, IMU_ACTIVE_THR)


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


def test_machine_motion_zero_for_still_machine():
    # an idle machine: accelerometer reads steady gravity, tiny noise
    still = np.tile([0.0, 0.0, 1.0], (200, 1)) + np.random.randn(200, 3) * 0.001
    assert machine_motion(still) < IMU_ACTIVE_THR


def test_machine_motion_high_for_vibrating_machine():
    # drilling: strong high-frequency acceleration about gravity
    t = np.linspace(0, 1, 200)
    vib = np.stack([0.05 * np.sin(2 * np.pi * 40 * t),
                    0.05 * np.cos(2 * np.pi * 40 * t),
                    1.0 + 0.05 * np.sin(2 * np.pi * 55 * t)], axis=1)
    assert machine_motion(vib) > IMU_ACTIVE_THR
    assert machine_active(vib)
    assert not machine_active(np.tile([0.0, 0.0, 1.0], (200, 1)))


def test_machine_motion_handles_degenerate_input():
    assert machine_motion([]) == 0.0
    assert machine_motion([[0, 0, 1]]) == 0.0          # single sample -> 0


def test_operator_present_persistence_gate():
    # a real operator confirmed in ~all frames is PRESENT; a flickering hallucination is not
    assert operator_present(12 / 12) == (True, True)
    assert operator_present(4 / 8) == (False, True)    # flicker -> seen but not present
    assert operator_present(0.0) == (False, False)


def test_classify_zone_tiers():
    # persistent operator + machine running -> DANGER (the real hazard)
    assert classify_zone(True, True, True) == "DANGER"
    # persistent operator, machine verifiably stopped -> safe reload
    assert classify_zone(True, True, False) == "OK_LOADING"
    # machine running but presence only a flicker -> REVIEW, not a false alarm
    assert classify_zone(False, True, True) == "REVIEW"
    # flicker + machine stopped, or nothing -> NO_PERSON
    assert classify_zone(False, True, False) == "NO_PERSON"
    assert classify_zone(False, False, False) == "NO_PERSON"


def test_classify_zone_imu_vetoes_vision_style_false_positive():
    # the cyc=2225 case: vision cried DANGER, but machine quiet + (real) operator present
    # -> the fused verdict is the safe OK_LOADING, not DANGER
    assert classify_zone(operator_is_present=True, operator_seen=True, machine_is_active=False) == "OK_LOADING"


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


def test_mesh_installs_counts_temporal_episodes_not_drift():
    from coverage import mesh_installs, mesh_count
    evs = [
        # mesh 1: a burst, operator DRIFTS across the mesh width (0.30 -> 0.50)
        {"cycle_sec": 10, "person_bbox": [0.25, 0.5, 0.35, 0.7]},
        {"cycle_sec": 18, "person_bbox": [0.30, 0.5, 0.40, 0.7]},
        {"cycle_sec": 26, "person_bbox": [0.45, 0.5, 0.55, 0.7]},
        # big gap (reload a fresh screen) -> mesh 2
        {"cycle_sec": 320, "person_bbox": [0.55, 0.5, 0.65, 0.7]},
        {"cycle_sec": 328, "person_bbox": [0.50, 0.5, 0.60, 0.7]},
        {"cycle_sec": 336, "person_bbox": [0.60, 0.5, 0.70, 0.7]},
    ]
    ins = mesh_installs(evs, gap=240, min_events=3)
    assert len(ins) == 2                 # drift within a mesh does NOT add a count
    assert mesh_count(evs, 100) == 1     # only mesh 1 by t=100
    assert mesh_count(evs, 999) == 2


def test_mesh_installs_filters_brief_blip():
    from coverage import mesh_installs
    evs = [{"cycle_sec": 10 + i * 8, "person_bbox": [0.3, 0.5, 0.4, 0.7]} for i in range(4)]
    evs += [{"cycle_sec": 600, "person_bbox": [0.3, 0.5, 0.4, 0.7]},      # 2-event blip
            {"cycle_sec": 608, "person_bbox": [0.3, 0.5, 0.4, 0.7]}]
    assert len(mesh_installs(evs, gap=240, min_events=3)) == 1   # blip filtered
