"""Pure compliance state machine for the face-support demo.

Accumulates per-step VLM observations into a progressively-filling checklist
and computes an overall verdict. No I/O except load_regulation().
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import yaml

PENDING = "pending"
IN_PROGRESS = "in_progress"
SATISFIED = "satisfied"
OK = "ok"
VIOLATION = "violation"


@dataclass
class ChecklistItem:
    id: str
    kind: str       # "process" | "safety"
    label: str
    evidence: str


@dataclass
class _ItemState:
    status: str
    since_t: float | None = None
    note: str = ""
    sat_count: int = 0


def load_regulation(path) -> list[ChecklistItem]:
    data = yaml.safe_load(Path(path).read_text())
    return [ChecklistItem(it["id"], it["kind"], it["label"], it["evidence"])
            for it in data["items"]]


class ComplianceTracker:
    def __init__(self, items: list[ChecklistItem], lock_after: int = 1):
        self.items = items
        self.lock_after = max(1, lock_after)
        self._state: dict[str, _ItemState] = {}
        for it in items:
            self._state[it.id] = _ItemState(OK if it.kind == "safety" else PENDING)
        self._kind = {it.id: it.kind for it in items}

    def update(self, t_sec, observations, safety_flags):
        # Process/observation updates
        for obs in observations or []:
            iid = obs.get("item_id")
            st = self._state.get(iid)
            if st is None:
                continue
            status = obs.get("status")
            if status == VIOLATION:
                if st.status != VIOLATION:
                    st.status = VIOLATION
                    st.since_t = t_sec
                    st.note = obs.get("evidence", "")
            elif st.status in (VIOLATION, SATISFIED):
                continue  # sticky; do not downgrade
            elif status == SATISFIED:
                st.sat_count += 1
                st.status = SATISFIED if st.sat_count >= self.lock_after else IN_PROGRESS
            elif status == IN_PROGRESS:
                st.status = IN_PROGRESS
        # Safety flags (sticky violations)
        for fl in safety_flags or []:
            iid = fl.get("id")
            st = self._state.get(iid)
            if st is None:
                continue
            if st.status != VIOLATION:
                st.status = VIOLATION
                st.since_t = t_sec
                st.note = fl.get("note", "")

    def snapshot(self) -> dict[str, str]:
        return {iid: s.status for iid, s in self._state.items()}

    def violations(self) -> list[dict]:
        return [{"id": iid, "since_t": s.since_t, "note": s.note}
                for iid, s in self._state.items() if s.status == VIOLATION]

    def verdict(self) -> str:
        if any(s.status == VIOLATION for s in self._state.values()):
            return "AT-RISK"
        process_ids = [iid for iid, k in self._kind.items() if k == "process"]
        if process_ids and all(self._state[i].status == SATISFIED for i in process_ids):
            return "COMPLIANT"
        return "IN PROGRESS"
