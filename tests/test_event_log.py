from event_log import (EventLogger, IncidentDebouncer,
                       INFO, WARNING, VIOLATION, CRITICAL,
                       OPERATOR_IN_ZONE, NEAR_MISS, SCREEN_INSTALLED)


def test_append_read_and_seq(tmp_path):
    lg = EventLogger(tmp_path / "ev.jsonl", reset=True, wall_clock=1000.0)
    lg.log(SCREEN_INSTALLED, 10.0, description="screen 1")
    lg.log(OPERATOR_IN_ZONE, 12.0, severity=VIOLATION)
    evs = lg.events()
    assert [e["seq"] for e in evs] == [1, 2]
    assert evs[0]["type"] == SCREEN_INSTALLED
    assert evs[1]["severity"] == VIOLATION


def test_persistence_continues_seq(tmp_path):
    p = tmp_path / "ev.jsonl"
    EventLogger(p, reset=True, wall_clock=1.0).log("a", 1.0)
    lg2 = EventLogger(p, wall_clock=2.0)          # reopen, append-only
    lg2.log("b", 2.0)
    evs = lg2.events()
    assert [e["seq"] for e in evs] == [1, 2]      # external memory survives reopen


def test_summary_counts(tmp_path):
    lg = EventLogger(tmp_path / "ev.jsonl", reset=True, wall_clock=1.0)
    lg.log(OPERATOR_IN_ZONE, 1.0, severity=VIOLATION)
    lg.log(NEAR_MISS, 2.0, severity=WARNING)
    lg.log("accident", 3.0, severity=CRITICAL)
    s = lg.summary()
    assert s["total"] == 3
    assert s["violations"] == 2          # VIOLATION + CRITICAL
    assert s["near_misses"] == 1


def test_debouncer_logs_one_incident_with_duration(tmp_path):
    lg = EventLogger(tmp_path / "ev.jsonl", reset=True, wall_clock=1.0)
    dz = IncidentDebouncer(lg, OPERATOR_IN_ZONE, VIOLATION, min_frames=2)
    dz.update(True, 10.0)        # 1st active - not yet (min_frames=2)
    assert lg.events() == []
    dz.update(True, 11.0)        # 2nd active -> START
    dz.update(True, 12.0)        # still active -> no new event
    dz.update(False, 14.0)       # ends -> END with duration
    evs = lg.events()
    assert len(evs) == 2
    assert "[START]" in evs[0]["description"]
    assert evs[1]["duration_sec"] == 3.0    # 14 - 11
