"""Fail-safe compliance/safety state machine (face-focused).

Consumes PERCEPTION dicts {face_screened, drill_active, arms_parked,
person_in_danger, scene, note} from vlm_client and aggregates them with a SAFETY
BIAS over a rolling window:

  * the compliant SUPPORTED state (face screened + booms parked + no active
    drilling) must hold UNANIMOUSLY across the whole window before it is reported,
  * active drilling or an unscreened face immediately blocks SUPPORTED,
  * a person under unsupported ground raises DANGER,
  * everything defaults to NOT_VERIFIED.

No I/O except load_regulation().
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import yaml

# Per-item states
NOT_VERIFIED = "not_verified"
VERIFIED = "verified"
VIOLATION = "violation"

# Overall verdicts (worst-case biased; SUPPORTED is the only "safe/compliant" one)
DANGER = "DANGER"
UNSUPPORTED = "UNSUPPORTED"
DRILLING = "DRILLING"
NOT_VERIFIED_VERDICT = "NOT VERIFIED"
SUPPORTED = "SUPPORTED"


@dataclass
class ChecklistItem:
    id: str
    label: str


def load_regulation(path) -> list[ChecklistItem]:
    data = yaml.safe_load(Path(path).read_text())
    return [ChecklistItem(it["id"], it.get("label", it["id"])) for it in data["items"]]


def _window_state(p: dict) -> str:
    """Classify a single perception window. Priority: danger > drilling >
    supported > unsupported > uncertain."""
    if p.get("person_in_danger"):
        return "danger"
    if p.get("drill_active"):
        return "drilling"
    if p.get("face_screened") and p.get("arms_parked"):
        return "supported"
    if not p.get("face_screened"):
        return "unsupported"
    return "uncertain"


class SafetyTracker:
    ITEM_IDS = ("face_screen", "no_active_drilling", "arms_parked", "worker_safe")

    def __init__(self, items=None, support_window: int = 3, hazard_confirm: int = 2):
        self.support_window = max(1, support_window)
        self.hazard_confirm = max(1, hazard_confirm)
        self.items = items or [ChecklistItem(i, i) for i in self.ITEM_IDS]
        self._buf: deque[dict] = deque(maxlen=self.support_window)
        self._last_scene = ""
        self._last_note = ""

    def update(self, t_sec, perception: dict) -> None:
        p = perception or {}
        self._buf.append(p)
        self._last_scene = p.get("scene", "") or self._last_scene
        self._last_note = p.get("note", "")

    # --- aggregation over the rolling buffer ---
    def _full(self) -> bool:
        return len(self._buf) >= self.support_window

    def _all(self, pred) -> bool:
        return self._full() and all(pred(p) for p in self._buf)

    def _count(self, pred) -> int:
        return sum(1 for p in self._buf if pred(p))

    def _states(self) -> list[str]:
        return [_window_state(p) for p in self._buf]

    def verdict(self) -> str:
        states = self._states()
        person_n = self._count(lambda p: p.get("person_in_danger"))
        drill_n = self._count(lambda p: p.get("drill_active"))
        if person_n >= self.hazard_confirm:
            return DANGER
        if "unsupported" in states:
            return UNSUPPORTED
        if drill_n >= self.hazard_confirm:
            return DRILLING
        if self._full() and all(s == "supported" for s in states):
            return SUPPORTED
        return NOT_VERIFIED_VERDICT

    def snapshot(self) -> dict[str, str]:
        person_n = self._count(lambda p: p.get("person_in_danger"))
        drill_n = self._count(lambda p: p.get("drill_active"))
        snap = {iid: NOT_VERIFIED for iid in self.ITEM_IDS}
        if self._all(lambda p: p.get("face_screened")):
            snap["face_screen"] = VERIFIED
        if self._all(lambda p: p.get("arms_parked")):
            snap["arms_parked"] = VERIFIED
        if drill_n >= self.hazard_confirm:
            snap["no_active_drilling"] = VIOLATION
        elif self._full() and drill_n == 0:
            snap["no_active_drilling"] = VERIFIED
        if person_n >= self.hazard_confirm:
            snap["worker_safe"] = VIOLATION
        return snap

    def hazard_note(self) -> str:
        person_n = self._count(lambda p: p.get("person_in_danger"))
        drill_n = self._count(lambda p: p.get("drill_active"))
        if person_n >= self.hazard_confirm:
            return "person under unsupported ground"
        if drill_n >= self.hazard_confirm:
            return "active drilling at the face"
        return ""

    def scene(self) -> str:
        return self._last_scene
