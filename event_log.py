"""Persistent external memory for the safety harness (append-only event log).

The VLM is STATELESS - it sees one frame at a time. Compliance, near-misses,
non-compliant operations and accidents are properties of the PROCESS OVER TIME, so
the system must hold the memory, not the model. This module is that memory: a
durable, append-only JSONL log of discrete, timestamped, typed, severity-rated
EVENTS, plus a small state machine that decides WHEN to emit an event (on
transitions / thresholds) rather than every frame.

Design goals:
  * Append-only + flushed per write  -> crash-safe, never loses an event.
  * Each event carries: seq, cycle_sec, wall_time, type, severity, description,
    location (bbox), evidence (frame path), and a source detector.
  * Debounced: a sustained condition logs ONE start event (+ optional end), not
    one per frame -> a readable incident timeline, not noise.
  * Queryable: full timeline, violations, near-misses, summary report.
"""
from __future__ import annotations
import json
import time
from pathlib import Path

# severities (ascending)
INFO = "INFO"
WARNING = "WARNING"        # near-miss / advisory
VIOLATION = "VIOLATION"    # non-compliant operation
CRITICAL = "CRITICAL"      # accident / imminent harm

# event types
SYSTEM_INIT = "system_init"                        # startup: face measurement + Vale requirement
SCREEN_INSTALLED = "screen_installed"
COVERAGE_FULL = "coverage_full"
FACE_SUPPORTED = "face_supported"
OPERATOR_IN_ZONE = "operator_in_danger_zone"      # operator in front + boom moving
NEAR_MISS = "near_miss"
NON_COMPLIANT = "non_compliant_operation"
ACCIDENT = "accident"
STATE_CHANGE = "state_change"
DOMAIN_ABSTAIN = "domain_abstain"                 # OOD guard suspended monitoring


class EventLogger:
    def __init__(self, path, *, reset=False, wall_clock=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if reset and self.path.exists():
            self.path.unlink()
        self._seq = self._last_seq()
        self._wall = wall_clock  # injectable for determinism in tests

    def _last_seq(self) -> int:
        if not self.path.exists():
            return 0
        last = 0
        for line in self.path.read_text().splitlines():
            try:
                last = max(last, json.loads(line).get("seq", 0))
            except json.JSONDecodeError:
                continue
        return last

    def log(self, etype, cycle_sec, *, severity=INFO, description="",
            bbox=None, evidence=None, source="", **extra) -> dict:
        self._seq += 1
        ev = {"seq": self._seq, "cycle_sec": round(float(cycle_sec), 1),
              "wall_time": (self._wall if self._wall is not None else time.time()),
              "type": etype, "severity": severity, "description": description,
              "bbox": bbox, "evidence": evidence, "source": source}
        ev.update(extra)
        with self.path.open("a") as f:
            f.write(json.dumps(ev) + "\n")
            f.flush()
        return ev

    # --- queries (always read from disk: the log is the source of truth) ---
    def events(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def by_severity(self, *sev) -> list[dict]:
        s = set(sev)
        return [e for e in self.events() if e["severity"] in s]

    def summary(self) -> dict:
        evs = self.events()
        by_type, by_sev = {}, {}
        for e in evs:
            by_type[e["type"]] = by_type.get(e["type"], 0) + 1
            by_sev[e["severity"]] = by_sev.get(e["severity"], 0) + 1
        return {"total": len(evs), "by_type": by_type, "by_severity": by_sev,
                "violations": by_sev.get(VIOLATION, 0) + by_sev.get(CRITICAL, 0),
                "near_misses": by_sev.get(WARNING, 0)}


class IncidentDebouncer:
    """Turns a per-frame boolean condition into START/END incident events, so a
    sustained danger logs one incident (with duration), not one event per frame."""

    def __init__(self, logger, etype, severity, source="", min_frames=1):
        self.logger = logger
        self.etype = etype
        self.severity = severity
        self.source = source
        self.min_frames = min_frames
        self._active = False
        self._start_cyc = None
        self._streak = 0

    def update(self, active: bool, cycle_sec, *, description="", bbox=None,
               evidence=None, **extra):
        if active:
            self._streak += 1
            if not self._active and self._streak >= self.min_frames:
                self._active = True
                self._start_cyc = cycle_sec
                self.logger.log(self.etype, cycle_sec, severity=self.severity,
                                description=description + " [START]", bbox=bbox,
                                evidence=evidence, source=self.source, **extra)
        else:
            self._streak = 0
            if self._active:
                self._active = False
                dur = round(cycle_sec - (self._start_cyc or cycle_sec), 1)
                self.logger.log(self.etype, cycle_sec, severity=INFO,
                                description=f"{self.etype} ended (duration {dur}s)",
                                source=self.source, duration_sec=dur,
                                started_at=self._start_cyc)
