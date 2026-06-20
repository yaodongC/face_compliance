"""Verdict-rules engine + the face_support decision tables (the configurable rules)."""
from rules_engine import decide, decide_traced
from rule_config import RULES


def test_decide_traced_reports_which_rule_fired():
    t = {"rules": [{"when": {"a": True}, "verdict": "X"},
                   {"when": {"b": True}, "verdict": "Y"}], "default": "D"}
    assert decide_traced(t, {"a": True}) == ("X", 0)
    assert decide_traced(t, {"a": False, "b": True}) == ("Y", 1)
    assert decide_traced(t, {}) == ("D", -1)          # fail-safe default
    # decide() and decide_traced() agree on the verdict
    assert decide(t, {"a": True}) == decide_traced(t, {"a": True})[0]


def test_decide_first_match_then_failsafe_default():
    t = {"rules": [{"when": {"a": True, "b": True}, "verdict": "X"},
                   {"when": {"a": True}, "verdict": "Y"}], "default": "D"}
    assert decide(t, {"a": True, "b": True}) == "X"    # first match wins
    assert decide(t, {"a": True, "b": False}) == "Y"
    assert decide(t, {"a": False}) == "D"              # nothing matches -> default
    assert decide(t, {}) == "D"


def test_face_support_tables_reproduce_current_mappings():
    assert decide(RULES["operator_entry"], {"boom_moving_at_entry": True}) == "NON_COMPLIANT_ENTRY"
    assert decide(RULES["operator_entry"], {"boom_moving_at_entry": False}) == "SAFE_RELOAD"
    assert decide(RULES["operator_live"], {"person_in_front": True, "boom_moving": True}) == "DANGER"
    assert decide(RULES["operator_live"], {"person_in_front": True, "boom_moving": False}) == "OK_LOADING"
    assert decide(RULES["operator_live"], {"person_in_front": False, "boom_moving": True}) == "NO_PERSON"
    # IMU-fused tiered operator-zone table
    assert decide(RULES["operator_zone"], {"operator_present": True, "operator_seen": True, "machine_active": True}) == "DANGER"
    assert decide(RULES["operator_zone"], {"operator_present": True, "operator_seen": True, "machine_active": False}) == "OK_LOADING"
    assert decide(RULES["operator_zone"], {"operator_present": False, "operator_seen": True, "machine_active": True}) == "REVIEW"
    assert decide(RULES["operator_zone"], {"operator_present": False, "operator_seen": False, "machine_active": False}) == "NO_PERSON"
    assert decide(RULES["coverage_overlap"], {"full": True, "overlaps": True}) == "COMPLIANT"
    assert decide(RULES["coverage_overlap"], {"full": True, "overlaps": False}) == "NOT SUPPORTED"
    assert decide(RULES["coverage_full"], {"full": True}) == "COMPLIANT"
    assert decide(RULES["coverage_full"], {"full": False}) == "NOT SUPPORTED"


def test_every_table_has_failsafe_default():
    for name, table in RULES.items():
        assert "default" in table, f"rules table {name} has no fail-safe default"
        assert isinstance(table.get("rules"), list)
