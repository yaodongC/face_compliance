"""Tests for the Vale-grounded compliance checklist (compliance_checklist.py)."""
from compliance_checklist import evaluate_checklist, CHECKLIST_SPEC, n_done

SIG = {
    "screen_times": [757, 1377, 2017, 2517], "n_screens_req": 4,
    "bolt_times": [385, 558, 859, 1035, 1119, 1523, 1688, 1883,
                   2045, 2256, 2410, 2548, 2736, 2909, 3072, 3284], "n_bolts_req": 16,
    "danger_times": [], "complete_at": 3290,
}


def test_every_item_cites_a_vale_document():
    items = evaluate_checklist(SIG, 9999)
    for it in items:
        assert it["source"], f"{it['key']} has no Vale citation"
        if it["key"] != "compliant":
            assert any(it["key"] == k for k, _l, _s in CHECKLIST_SPEC)
        # citations reference the actual Vale standards
        assert ("CMTS-2015-001" in it["source"]) or ("Div6" in it["source"]) or ("Division 6" in it["source"])


def test_only_maintained_safety_check_done_at_start():
    # at t=0 the progressive items are pending; the continuous "workers clear" safety check
    # is ✓ (maintained — no danger has occurred yet); overall compliance is NOT met.
    items = evaluate_checklist(SIG, 0)
    by = {i["key"]: i for i in items}
    assert by["workers"]["done"] and by["workers"]["done_time"] is None
    assert not by["screened"]["done"] and not by["bolted"]["done"]
    assert not by["coverage"]["done"] and not by["ordering"]["done"]
    assert not by["compliant"]["done"]                 # not compliant at t=0
    assert n_done(items) == 1                           # only the maintained safety check


def test_progress_midcycle():
    # at 35:00 (2100s): 3 screens (757,1377,2017), 8 bolts set
    items = evaluate_checklist(SIG, 2100)
    by = {i["key"]: i for i in items}
    assert by["screened"]["detail"] == "3/4 panels"
    assert not by["screened"]["done"]
    assert by["bolted"]["detail"].endswith("bolts") and not by["bolted"]["done"]
    assert by["workers"]["done"] and by["workers"]["detail"] == "clear"   # no danger yet
    assert not by["compliant"]["done"]


def test_all_complete_at_end():
    items = evaluate_checklist(SIG, 3300)
    by = {i["key"]: i for i in items}
    assert by["screened"]["done"] and by["screened"]["done_time"] == 2517
    assert by["bolted"]["done"] and by["bolted"]["done_time"] == 3284
    assert by["coverage"]["done"]
    assert by["ordering"]["done"] and by["ordering"]["done_time"] == 3290
    assert by["compliant"]["done"] and by["compliant"]["done_time"] == 3290
    assert n_done(items) == 5


def test_danger_unchecks_workers_item():
    sig = dict(SIG, danger_times=[1500])
    items = evaluate_checklist(sig, 1600)
    by = {i["key"]: i for i in items}
    assert not by["workers"]["done"]
    assert "danger" in by["workers"]["detail"]
    # and overall compliance cannot be true while a safety item is open
    items_end = evaluate_checklist(sig, 3300)
    assert not {i["key"]: i for i in items_end}["compliant"]["done"]


def test_completion_times_are_monotone_visible():
    # an item shows its done_time only once csec passes it
    items_before = {i["key"]: i for i in evaluate_checklist(SIG, 2500)}
    items_after = {i["key"]: i for i in evaluate_checklist(SIG, 2600)}
    assert items_before["screened"]["done_time"] is None      # 4th screen at 2517 not yet
    assert items_after["screened"]["done_time"] == 2517
