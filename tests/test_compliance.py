from pathlib import Path
from compliance import (ComplianceTracker, ChecklistItem, load_regulation,
                        PENDING, IN_PROGRESS, SATISFIED, OK, VIOLATION)

ROOT = Path(__file__).resolve().parents[1]


def make_tracker(lock_after=1):
    items = [
        ChecklistItem("p1", "process", "scale", "..."),
        ChecklistItem("p2", "process", "screen", "..."),
        ChecklistItem("S1", "safety", "zone", "..."),
    ]
    return ComplianceTracker(items, lock_after=lock_after)


def test_loads_regulation_into_items():
    items = load_regulation(ROOT / "regulation.yaml")
    assert all(isinstance(i, ChecklistItem) for i in items)
    assert any(i.id == "p1" for i in items)


def test_process_defaults_pending_safety_defaults_ok():
    t = make_tracker()
    snap = t.snapshot()
    assert snap["p1"] == PENDING
    assert snap["S1"] == OK


def test_in_progress_then_satisfied_lock_after_1():
    t = make_tracker(lock_after=1)
    t.update(1.0, [{"item_id": "p1", "status": "in_progress", "evidence": "x"}], [])
    assert t.snapshot()["p1"] == IN_PROGRESS
    t.update(2.0, [{"item_id": "p1", "status": "satisfied", "evidence": "x"}], [])
    assert t.snapshot()["p1"] == SATISFIED


def test_smoothing_lock_after_2():
    t = make_tracker(lock_after=2)
    t.update(1.0, [{"item_id": "p1", "status": "satisfied", "evidence": "x"}], [])
    assert t.snapshot()["p1"] == IN_PROGRESS  # one sighting not enough
    t.update(2.0, [{"item_id": "p1", "status": "satisfied", "evidence": "x"}], [])
    assert t.snapshot()["p1"] == SATISFIED


def test_satisfied_is_monotonic():
    t = make_tracker(lock_after=1)
    t.update(1.0, [{"item_id": "p1", "status": "satisfied", "evidence": "x"}], [])
    t.update(2.0, [{"item_id": "p1", "status": "in_progress", "evidence": "x"}], [])
    assert t.snapshot()["p1"] == SATISFIED  # not downgraded


def test_safety_flag_sets_sticky_violation():
    t = make_tracker()
    t.update(3.0, [], [{"id": "S1", "severity": "high", "note": "person under brow"}])
    assert t.snapshot()["S1"] == VIOLATION
    t.update(4.0, [], [])  # no flag this step
    assert t.snapshot()["S1"] == VIOLATION  # sticky
    v = t.violations()
    assert v and v[0]["id"] == "S1" and v[0]["since_t"] == 3.0


def test_process_violation_sets_violation():
    t = make_tracker()
    t.update(5.0, [{"item_id": "p2", "status": "violation", "evidence": "drilling unscreened face"}], [])
    assert t.snapshot()["p2"] == VIOLATION


def test_verdict_in_progress_then_at_risk_then_compliant():
    t = make_tracker(lock_after=1)
    assert t.verdict() == "IN PROGRESS"
    t.update(1.0, [], [{"id": "S1", "severity": "high", "note": "x"}])
    assert t.verdict() == "AT-RISK"
    # clear is not automatic; build a fresh tracker for compliant path
    t2 = make_tracker(lock_after=1)
    t2.update(1.0, [{"item_id": "p1", "status": "satisfied", "evidence": "x"}], [])
    t2.update(2.0, [{"item_id": "p2", "status": "satisfied", "evidence": "x"}], [])
    assert t2.verdict() == "COMPLIANT"  # all process satisfied, no violation
