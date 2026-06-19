"""Fail-safe compliance/safety state machine for the face-support demo.

SAFETY-CRITICAL: a small VLM hallucinates ground support that is not there, so
this layer NEVER trusts a single frame and NEVER treats absence of evidence as
safety. It consumes conservative PERCEPTION dicts from vlm_client and aggregates
them with a SAFETY BIAS:

  * every item defaults to NOT_VERIFIED (not "ok", not "satisfied"),
  * an item becomes VERIFIED only on positive, SUSTAINED, conjunctive evidence,
  * any recent UNSUPPORTED reading drags the overall verdict down,
  * SUPPORTED requires a clean streak of `support_window` agreeing windows,
  * active hazards (a person under unsupported rock, or drilling an unsupported
    face) raise DANGER immediately.

No I/O except load_regulation().
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import yaml

# Per-item states
NOT_VERIFIED = "not_verified"   # default — we cannot confirm this; treat as unsafe
VERIFIED = "verified"           # positive sustained evidence
VIOLATION = "violation"         # active hazard / clear non-compliance

# Overall verdicts (worst-case biased)
DANGER = "DANGER"
UNSUPPORTED = "UNSUPPORTED"
NOT_VERIFIED_VERDICT = "NOT VERIFIED"
SUPPORTED = "SUPPORTED"


@dataclass
class ChecklistItem:
    id: str
    label: str


def load_regulation(path) -> list[ChecklistItem]:
    data = yaml.safe_load(Path(path).read_text())
    return [ChecklistItem(it["id"], it.get("label", it["id"])) for it in data["items"]]


def _is_supported(p: dict) -> bool:
    """A window counts as 'supported' only with the full conjunction."""
    return (bool(p.get("mesh_visible")) and bool(p.get("bolts_visible"))
            and p.get("ground_support_state") == "full"
            and p.get("safety_call") == "SUPPORTED")


def _is_unsupported(p: dict) -> bool:
    # Anything short of clearly-full support is treated as unsupported (unsafe).
    return (p.get("safety_call") in ("UNSUPPORTED", "PARTIAL")
            or p.get("ground_support_state") in ("none_visible", "partial"))


class SafetyTracker:
    """Aggregates perception windows into a fail-safe checklist + verdict.

    support_window: number of recent windows considered. SUPPORTED / a VERIFIED
                    support item requires ALL of the last `support_window`
                    windows to agree; any UNSUPPORTED in the buffer wins.
    """

    # The fixed set of items this tracker reasons about (perception-grounded).
    ITEM_IDS = ("bolts", "mesh", "support", "drill_safe", "worker_safe")

    def __init__(self, items=None, support_window: int = 3, hazard_confirm: int = 2):
        self.support_window = max(1, support_window)
        # a hazard must be seen in this many windows of the buffer before it fires,
        # so a single hallucinated drilling/person frame cannot raise DANGER.
        self.hazard_confirm = max(1, hazard_confirm)
        self.items = items or [ChecklistItem(i, i) for i in self.ITEM_IDS]
        self._buf: deque[dict] = deque(maxlen=self.support_window)
        self._last_hazard_note = ""
        self._last_scene = ""
        self._last_note = ""

    def update(self, t_sec, perception: dict) -> None:
        p = perception or {}
        self._buf.append(p)
        self._last_scene = p.get("scene", "") or self._last_scene
        self._last_note = p.get("note", "")
        haz, note = self._hazard()
        self._last_hazard_note = note if haz else ""

    # --- internal aggregation over the rolling buffer ---
    def _full(self) -> bool:
        return len(self._buf) >= self.support_window

    def _all(self, pred) -> bool:
        return self._full() and all(pred(p) for p in self._buf)

    def _any(self, pred) -> bool:
        return any(pred(p) for p in self._buf)

    def _hazard_counts(self) -> tuple[int, int]:
        """Count windows in the buffer showing a person-in-danger / drilling hazard."""
        person_n = drill_n = 0
        for p in self._buf:
            if p.get("person_in_danger"):
                person_n += 1
            elif p.get("activity") == "drilling" and not _is_supported(p):
                drill_n += 1
        return person_n, drill_n

    def _hazard(self) -> tuple[bool, str]:
        person_n, drill_n = self._hazard_counts()
        if person_n >= self.hazard_confirm:
            return True, "person under unsupported ground"
        if drill_n >= self.hazard_confirm:
            return True, "drilling on an unsupported face"
        return False, ""

    def snapshot(self) -> dict[str, str]:
        """Per-item state. Defaults NOT_VERIFIED; only sustained positive
        conjunctive evidence yields VERIFIED; hazards yield VIOLATION."""
        bolts_ok = self._all(lambda p: bool(p.get("bolts_visible")))
        mesh_ok = self._all(lambda p: bool(p.get("mesh_visible")) and bool(p.get("bolts_visible")))
        support_ok = self._all(_is_supported)

        snap = {iid: NOT_VERIFIED for iid in self.ITEM_IDS}
        if bolts_ok:
            snap["bolts"] = VERIFIED
        if mesh_ok:
            snap["mesh"] = VERIFIED
        if support_ok:
            snap["support"] = VERIFIED

        # Safety items: VIOLATION on a CONFIRMED hazard; otherwise NOT_VERIFIED
        # (never auto-ok — we do not certify the absence of a hazard).
        person_n, drill_n = self._hazard_counts()
        if person_n >= self.hazard_confirm:
            snap["worker_safe"] = VIOLATION
        if drill_n >= self.hazard_confirm:
            snap["drill_safe"] = VIOLATION
        return snap

    def verdict(self) -> str:
        haz, _ = self._hazard()
        if haz:
            return DANGER
        if self._any(_is_unsupported):
            return UNSUPPORTED
        if self._all(_is_supported):
            return SUPPORTED
        return NOT_VERIFIED_VERDICT

    def hazard_note(self) -> str:
        return self._last_hazard_note

    def scene(self) -> str:
        return self._last_scene
