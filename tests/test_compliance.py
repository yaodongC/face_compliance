from pathlib import Path
from compliance import (SafetyTracker, ChecklistItem, load_regulation,
                        NOT_VERIFIED, VERIFIED, VIOLATION,
                        DANGER, UNSUPPORTED, DRILLING, NOT_VERIFIED_VERDICT, SUPPORTED)

ROOT = Path(__file__).resolve().parents[1]


def perc(screened=False, drill=False, parked=False, danger=False):
    return {"scene": "s", "face_screened": screened, "drill_active": drill,
            "arms_parked": parked, "person_in_danger": danger, "note": "n"}


SUP = perc(screened=True, parked=True)         # compliant: screened + parked, no drill
DRILL = perc(screened=False, drill=True)       # active drilling on a bare face
BARE = perc(screened=False)                    # not screened, not drilling


def test_loads_regulation_items():
    items = load_regulation(ROOT / "regulation.yaml")
    assert all(isinstance(i, ChecklistItem) for i in items)
    assert any(i.id == "face_screen" for i in items)


def test_defaults_not_verified():
    t = SafetyTracker(support_window=3)
    assert set(t.snapshot().values()) == {NOT_VERIFIED}
    assert t.verdict() != SUPPORTED


def test_sustained_supported_is_supported():
    t = SafetyTracker(support_window=2)
    t.update(1, SUP)
    assert t.verdict() != SUPPORTED          # one window not enough
    t.update(2, SUP)
    assert t.verdict() == SUPPORTED
    snap = t.snapshot()
    assert snap["face_screen"] == VERIFIED and snap["arms_parked"] == VERIFIED


def test_one_drilling_window_blocks_supported():
    t = SafetyTracker(support_window=2)
    t.update(1, SUP)
    t.update(2, DRILL)
    assert t.verdict() != SUPPORTED


def test_sustained_drilling_is_DRILLING():
    t = SafetyTracker(support_window=3, hazard_confirm=2)
    t.update(1, DRILL)
    t.update(2, DRILL)
    assert t.verdict() == DRILLING
    assert t.snapshot()["no_active_drilling"] == VIOLATION


def test_single_drill_frame_debounced():
    # one drilling frame must not raise DRILLING when hazard_confirm=2 (but it does
    # block SUPPORTED)
    t = SafetyTracker(support_window=3, hazard_confirm=2)
    t.update(1, DRILL)
    assert t.verdict() != DRILLING
    assert t.verdict() != SUPPORTED


def test_unscreened_face_is_unsupported():
    t = SafetyTracker(support_window=2)
    t.update(1, BARE)
    t.update(2, BARE)
    assert t.verdict() == UNSUPPORTED


def test_person_in_danger_raises_danger():
    t = SafetyTracker(support_window=3, hazard_confirm=1)
    t.update(1, perc(danger=True))
    assert t.verdict() == DANGER
    assert t.snapshot()["worker_safe"] == VIOLATION


def test_person_danger_debounced():
    t = SafetyTracker(support_window=3, hazard_confirm=2)
    t.update(1, perc(danger=True))
    assert t.verdict() != DANGER
    t.update(2, perc(danger=True))
    assert t.verdict() == DANGER


def test_supported_requires_parked_and_screened():
    # screened but booms NOT parked -> not the compliant rest-state
    t = SafetyTracker(support_window=2)
    screened_only = perc(screened=True, parked=False)
    t.update(1, screened_only)
    t.update(2, screened_only)
    assert t.verdict() != SUPPORTED
