from pathlib import Path
from compliance import (SafetyTracker, ChecklistItem, load_regulation,
                        NOT_VERIFIED, VERIFIED, VIOLATION,
                        DANGER, UNSUPPORTED, NOT_VERIFIED_VERDICT, SUPPORTED)

ROOT = Path(__file__).resolve().parents[1]


def perc(mesh=False, bolts=False, gss="none_visible", call="UNSUPPORTED",
         activity="none", danger=False, people=False, scene="s", note="n"):
    return {"scene": scene, "activity": activity, "people_visible": people,
            "person_in_danger": danger, "mesh_visible": mesh, "bolts_visible": bolts,
            "ground_support_state": gss, "safety_call": call, "note": note}


SUPPORTED_P = perc(mesh=True, bolts=True, gss="full", call="SUPPORTED")
BARE_P = perc()  # unsupported bare face


def test_loads_regulation_items():
    items = load_regulation(ROOT / "regulation.yaml")
    assert all(isinstance(i, ChecklistItem) for i in items)
    assert any(i.id == "support" for i in items)
    assert all(i.label for i in items)


def test_defaults_are_not_verified():
    t = SafetyTracker(support_window=3)
    snap = t.snapshot()
    assert set(snap.values()) == {NOT_VERIFIED}
    assert t.verdict() != SUPPORTED


def test_bare_face_is_unsupported_and_nothing_verified():
    t = SafetyTracker(support_window=3)
    for i in range(3):
        t.update(i, BARE_P)
    assert t.verdict() == UNSUPPORTED
    assert set(t.snapshot().values()) == {NOT_VERIFIED}


def test_sustained_support_is_verified():
    t = SafetyTracker(support_window=2)
    t.update(1, SUPPORTED_P)
    assert t.verdict() != SUPPORTED        # one window is not enough
    t.update(2, SUPPORTED_P)
    assert t.verdict() == SUPPORTED
    snap = t.snapshot()
    assert snap["bolts"] == VERIFIED and snap["mesh"] == VERIFIED and snap["support"] == VERIFIED


def test_any_unsupported_in_window_drags_down():
    t = SafetyTracker(support_window=3)
    t.update(1, SUPPORTED_P)
    t.update(2, SUPPORTED_P)
    t.update(3, BARE_P)                    # one bad window in the buffer
    assert t.verdict() == UNSUPPORTED
    assert t.snapshot()["support"] == NOT_VERIFIED


def test_mesh_requires_bolts_conjunction():
    t = SafetyTracker(support_window=2)
    mesh_only = perc(mesh=True, bolts=False, gss="none_visible", call="UNSUPPORTED")
    t.update(1, mesh_only)
    t.update(2, mesh_only)
    snap = t.snapshot()
    assert snap["mesh"] == NOT_VERIFIED
    assert snap["bolts"] == NOT_VERIFIED
    assert t.verdict() == UNSUPPORTED


def test_person_in_danger_raises_danger_and_violation():
    t = SafetyTracker(support_window=3, hazard_confirm=1)
    t.update(1, perc(people=True, danger=True))
    assert t.verdict() == DANGER
    assert t.snapshot()["worker_safe"] == VIOLATION
    assert t.hazard_note()


def test_drilling_unsupported_raises_danger():
    t = SafetyTracker(support_window=3, hazard_confirm=1)
    t.update(1, perc(activity="drilling"))
    assert t.verdict() == DANGER
    assert t.snapshot()["drill_safe"] == VIOLATION


def test_single_hallucinated_hazard_is_debounced():
    # one drilling frame must NOT raise DANGER when hazard_confirm=2
    t = SafetyTracker(support_window=3, hazard_confirm=2)
    t.update(1, perc(activity="drilling"))
    assert t.verdict() != DANGER
    assert t.snapshot()["drill_safe"] != VIOLATION
    t.update(2, perc(activity="drilling"))   # sustained -> now it fires
    assert t.verdict() == DANGER


def test_partial_support_is_treated_as_unsupported():
    # partial support is NOT good enough for a face you may work under -> unsafe
    t = SafetyTracker(support_window=2)
    partial = perc(mesh=True, bolts=True, gss="partial", call="PARTIAL")
    t.update(1, partial)
    t.update(2, partial)
    assert t.verdict() == UNSUPPORTED
    assert t.snapshot()["support"] == NOT_VERIFIED


def test_not_verified_only_when_genuinely_uncertain():
    # model cannot tell (no positive support, no clear 'unsupported') -> NOT VERIFIED
    t = SafetyTracker(support_window=2)
    uncertain = perc(mesh=False, bolts=False, gss="cannot_tell", call="CANNOT_VERIFY")
    t.update(1, uncertain)
    t.update(2, uncertain)
    assert t.verdict() == NOT_VERIFIED_VERDICT


def test_drilling_when_supported_is_not_a_hazard():
    t = SafetyTracker(support_window=2)
    t.update(1, SUPPORTED_P)
    t.update(2, perc(mesh=True, bolts=True, gss="full", call="SUPPORTED", activity="drilling"))
    assert t.verdict() == SUPPORTED
    assert t.snapshot()["drill_safe"] != VIOLATION
