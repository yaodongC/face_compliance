"""Unit tests for the compliance-milestone pipeline (pure logic, no VLM / no bags).

Covers: IMU episode segmentation, fused bolt/screen/coverage counting, the latched
fail-safe milestone state machine, and the cycle-frame nearest-time mapping.
"""
import numpy as np
import imu_analyzer as ia
import progress_tracker as pt
import compliance_milestone as cm


# ---------- IMU episode segmentation ----------
def test_segment_episodes_merges_gaps_and_drops_transients():
    # envelope: a long active burst (0..50s), a short 1s transient (80s), another burst (100..160)
    t = np.arange(0, 200, 1.0)
    std = np.full_like(t, 0.005)
    std[(t >= 0) & (t <= 50)] = 0.06        # 50 s burst -> kept
    std[(t == 80)] = 0.06                     # 1 s transient -> dropped (too short, peak not >2*thr? it is)
    std[(t >= 100) & (t <= 160)] = 0.06     # 60 s burst -> kept
    eps = ia.segment_episodes(t, std, thr=0.013, merge_gap=5.0, min_dur=4.0)
    longs = [e for e in eps if e["dur"] >= 40]
    assert len(longs) == 2
    assert longs[0]["start"] == 0.0 and longs[0]["end"] == 50.0


def test_segment_episodes_merges_within_gap():
    t = np.arange(0, 100, 1.0)
    std = np.full_like(t, 0.005)
    std[(t >= 0) & (t <= 20)] = 0.06
    std[(t >= 23) & (t <= 60)] = 0.06       # 3 s gap < merge_gap 5 -> one episode
    eps = ia.segment_episodes(t, std, thr=0.013, merge_gap=5.0, min_dur=4.0)
    assert len(eps) == 1 and eps[0]["start"] == 0.0 and eps[0]["end"] == 60.0


# ---------- size-dependent required counts (single Vale-doc model in vale_support) ----------
def test_derive_counts_scales_with_face_size():
    # derive_counts delegates to vale_support (6' sheets, 1' overlap -> 5' advance, 4 bolts/sheet)
    assert pt.derive_counts(6.2)["meshes_required"] == 4     # this face
    assert pt.derive_counts(6.2)["bolts_required"] == 16
    assert pt.derive_counts(4.0)["meshes_required"] == 3     # smaller face -> fewer
    assert pt.derive_counts(9.6)["meshes_required"] == 7     # bigger face -> more
    assert pt.derive_counts(9.6)["bolts_required"] == 28


def test_derive_counts_matches_vale_support():
    import vale_support as vs
    for w in (4.0, 6.0, 6.2, 8.0, 11.0):
        assert pt.derive_counts(w)["meshes_required"] == vs.calc(w, w)["meshes_required"]


def test_derive_counts_monotone_in_face_width():
    widths = [3, 4, 5, 6, 7, 8, 10, 12]
    meshes = [pt.derive_counts(w)["meshes_required"] for w in widths]
    assert meshes == sorted(meshes)         # non-decreasing with face size
    assert pt.derive_counts(1.0)["meshes_required"] >= 1   # at least one mesh


def test_bolts_required_is_meshes_times_pattern():
    for w in (4.0, 6.2, 8.0, 11.0):
        c = pt.derive_counts(w)
        assert c["bolts_required"] == c["meshes_required"] * c["bolts_per_screen"]


# ---------- progress fusion ----------
def _synth_timeline(n_bolt=16):
    """A timeline with n_bolt sustained drilling episodes spaced 180 s apart."""
    eps = [{"start": float(180 * i + 60), "end": float(180 * i + 160),
            "dur": 100.0, "peak": 0.08, "mean": 0.05} for i in range(n_bolt)]
    return {"episodes": eps, "thr": 0.013, "envelope": []}


def _synth_ops(n=8):
    return {"events": [{"cycle_sec": float(300 * i + 100),
                        "person_bbox": [0.1 + 0.1 * (i % 6), 0.6, 0.2 + 0.1 * (i % 6), 0.95]}
                       for i in range(n)]}


def test_bolt_episodes_filtered_by_dur_and_peak():
    tl = {"episodes": [{"start": 10, "end": 60, "dur": 50, "peak": 0.08},   # bolt
                       {"start": 100, "end": 110, "dur": 10, "peak": 0.08},  # too short
                       {"start": 200, "end": 260, "dur": 60, "peak": 0.02}], # too weak
          "thr": 0.013, "envelope": []}
    b = pt.bolt_episodes(tl)
    assert len(b) == 1 and b[0]["start"] == 10


def test_bolts_and_screens_count_monotone():
    tl = _synth_timeline(16)
    bolts = pt.bolt_episodes(tl)
    assert len(bolts) == 16
    # bolt set at end of each episode
    assert pt.bolts_installed(bolts, 0) == 0
    assert pt.bolts_installed(bolts, 160) == 1
    assert pt.bolts_installed(bolts, 10_000) == 16
    # screens follow the 4-bolts-per-screen Vale pattern
    assert pt.screens_installed(bolts, 160) == 0
    assert pt.screens_installed(bolts, bolts[3]["set_at"]) == 1
    assert pt.screens_installed(bolts, 10_000) == 4


def test_episode_class_can_veto_non_bolt():
    tl = _synth_timeline(16)
    cls = {"episodes": [{"start": tl["episodes"][0]["start"], "activity": "mucking"}]}
    assert len(pt.bolt_episodes(tl, cls)) == 15   # one vetoed


def test_coverage_full_only_when_all_screens_in():
    tl = _synth_timeline(16)
    ev = _synth_ops(8)
    bolts = pt.bolt_episodes(tl)
    early = pt.face_coverage(bolts, ev, bolts[3]["set_at"])   # 1 screen
    late = pt.face_coverage(bolts, ev, 10_000)                # 4 screens
    assert early["coverage"] < late["coverage"]
    assert late["coverage"] >= cm.COVER_THR


# ---------- latched milestone state machine ----------
def _prog(bolts, screens, coverage):
    return {"bolts": bolts, "screens": screens, "coverage": coverage,
            "bolts_target": 16, "screens_target": 4}


def test_milestone_does_not_fire_early():
    ms = cm.ComplianceMilestone()
    for t, b, s in [(100, 4, 1), (200, 8, 2), (300, 12, 3), (400, 15, 3)]:
        ms.update(t, _prog(b, s, b / 16), confirm={"supported": True})
    assert ms.phase != cm.COMPLETE
    assert ms.complete_at is None


def test_milestone_fires_when_all_gates_met():
    ms = cm.ComplianceMilestone()
    ms.update(400, _prog(16, 4, 1.0), confirm={"supported": True})
    assert ms.phase == cm.COMPLETE
    assert ms.complete_at == 400


def test_milestone_failsafe_without_vlm_confirmation():
    ms = cm.ComplianceMilestone()
    # physical gates met but VLM says NOT supported -> stays AWAIT, never COMPLETE
    ms.update(400, _prog(16, 4, 1.0), confirm={"supported": False})
    assert ms.phase == cm.AWAIT
    assert ms.complete_at is None


def test_milestone_latches_and_logs_each_bolt():
    ms = cm.ComplianceMilestone()
    ms.update(100, _prog(4, 1, 0.25), confirm=None)
    ms.update(400, _prog(16, 4, 1.0), confirm={"supported": True})
    assert ms.phase == cm.COMPLETE
    bolt_logs = [m for m in ms.log if m["type"] == "bolt_installed"]
    screen_logs = [m for m in ms.log if m["type"] == "screen_installed"]
    assert len(bolt_logs) == 16 and len(screen_logs) == 4
    # latched: a later non-ready update does not revert
    ms.update(500, _prog(0, 0, 0.0), confirm={"supported": False})
    assert ms.phase == cm.COMPLETE and ms.complete_at == 400


# ---------- cycle frame nearest-time mapping ----------
def test_frame_nearest_time(tmp_path):
    import csv
    from cycle_frames import FrameGrabber
    # build a tiny fake index; monkeypatch cv2 open by subclassing
    idx = tmp_path / "f.idx"
    with open(idx, "w") as f:
        w = csv.writer(f)
        w.writerow(["frame", "cycle_sec"])
        for i, t in enumerate([0.0, 10.0, 20.0, 30.0]):
            w.writerow([i, t])

    class _G(FrameGrabber):
        def __init__(self, index):
            rows = list(csv.reader(open(index)))[1:]
            self.frames = [int(r[0]) for r in rows]
            self.times = [float(r[1]) for r in rows]
    g = _G(str(idx))
    assert g._frame_no(0) == 0
    assert g._frame_no(9) == 1        # closer to 10 than 0
    assert g._frame_no(14) == 1       # closer to 10 than 20
    assert g._frame_no(16) == 2       # closer to 20
    assert g._frame_no(999) == 3      # clamps to last
