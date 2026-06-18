from pathlib import Path
from compliance import (ComplianceTracker, ChecklistItem, load_regulation,
                        PENDING, IN_PROGRESS, SATISFIED, OK, VIOLATION)

ROOT = Path(__file__).resolve().parents[1]


def items():
    return [ChecklistItem("p1", "process", "scale", "..."),
            ChecklistItem("p2", "process", "screen", "..."),
            ChecklistItem("S1", "safety", "zone", "...")]


def obs(iid, status):
    return {"item_id": iid, "status": status, "evidence": "e"}


def flag(iid, sev):
    return {"id": iid, "severity": sev, "note": "n"}


def test_loads_regulation_into_items():
    its = load_regulation(ROOT / "regulation.yaml")
    assert all(isinstance(i, ChecklistItem) for i in its)
    assert any(i.id == "p1" for i in its)


def test_process_defaults_pending_safety_defaults_ok():
    t = ComplianceTracker(items())
    assert t.snapshot()["p1"] == PENDING
    assert t.snapshot()["S1"] == OK


def test_satisfied_lock_after_one():
    t = ComplianceTracker(items(), confirm_satisfied=1)
    t.update(1, [obs("p1", "satisfied")], [])
    assert t.snapshot()["p1"] == SATISFIED


def test_satisfied_smoothing_two():
    t = ComplianceTracker(items(), confirm_satisfied=2)
    t.update(1, [obs("p1", "satisfied")], [])
    assert t.snapshot()["p1"] == IN_PROGRESS
    t.update(2, [obs("p1", "satisfied")], [])
    assert t.snapshot()["p1"] == SATISFIED


def test_satisfied_is_monotonic_under_progress():
    t = ComplianceTracker(items(), confirm_satisfied=1)
    t.update(1, [obs("p1", "satisfied")], [])
    t.update(2, [obs("p1", "in_progress")], [])
    assert t.snapshot()["p1"] == SATISFIED


def test_violation_debounce_requires_two_windows():
    t = ComplianceTracker(items(), confirm_violation=2)
    t.update(1, [obs("p1", "violation")], [])
    assert t.snapshot()["p1"] != VIOLATION       # one sighting not enough
    t.update(2, [obs("p1", "violation")], [])
    assert t.snapshot()["p1"] == VIOLATION


def test_single_noisy_violation_does_not_lock():
    t = ComplianceTracker(items(), confirm_violation=2)
    t.update(1, [obs("p1", "violation")], [])
    t.update(2, [obs("p1", "satisfied")], [])    # noise interrupted by compliant window
    assert t.snapshot()["p1"] == SATISFIED


def test_low_severity_flag_is_advisory_not_violation():
    t = ComplianceTracker(items(), min_severity="med", confirm_violation=1)
    t.update(1, [], [flag("S1", "low")])
    assert t.snapshot()["S1"] == OK


def test_high_severity_flag_is_violation():
    t = ComplianceTracker(items(), min_severity="med", confirm_violation=1)
    t.update(1, [], [flag("S1", "high")])
    assert t.snapshot()["S1"] == VIOLATION


def test_safety_violation_clears_after_compliant_windows():
    t = ComplianceTracker(items(), confirm_violation=1, confirm_clear=2)
    t.update(1, [], [flag("S1", "high")])
    assert t.snapshot()["S1"] == VIOLATION
    t.update(2, [], [])                          # 1 clear window — not yet
    assert t.snapshot()["S1"] == VIOLATION
    t.update(3, [], [])                          # 2 clear windows — clears
    assert t.snapshot()["S1"] == OK


def test_process_violation_clears_to_satisfied():
    t = ComplianceTracker(items(), confirm_violation=1, confirm_clear=1)
    t.update(1, [obs("p1", "violation")], [])
    assert t.snapshot()["p1"] == VIOLATION
    t.update(2, [obs("p1", "satisfied")], [])
    assert t.snapshot()["p1"] == SATISFIED


def test_since_t_recorded_then_reset_on_clear():
    t = ComplianceTracker(items(), confirm_violation=1, confirm_clear=1)
    t.update(5, [], [flag("S1", "high")])
    v = t.violations()
    assert v and v[0]["id"] == "S1" and v[0]["since_t"] == 5
    t.update(6, [], [])
    assert t.violations() == []


def test_non_consecutive_flags_do_not_lock():
    t = ComplianceTracker(items(), confirm_violation=2)
    t.update(1, [], [flag("S1", "high")])        # viol_count 1
    assert t.snapshot()["S1"] == OK
    t.update(2, [], [])                          # compliant resets count
    t.update(3, [], [flag("S1", "high")])        # viol_count 1 again, not 2
    assert t.snapshot()["S1"] == OK


def test_verdict_in_progress_at_risk_then_back_to_compliant():
    t = ComplianceTracker(items(), confirm_satisfied=1, confirm_violation=1, confirm_clear=1)
    assert t.verdict() == "IN PROGRESS"
    t.update(1, [obs("p1", "satisfied"), obs("p2", "satisfied")], [])
    assert t.verdict() == "COMPLIANT"
    t.update(2, [], [flag("S1", "high")])
    assert t.verdict() == "AT-RISK"
    t.update(3, [obs("p1", "satisfied"), obs("p2", "satisfied")], [])  # S1 absent -> clears
    assert t.verdict() == "COMPLIANT"
