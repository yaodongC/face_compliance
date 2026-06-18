"""Pure compliance state machine for the face-support demo.

Accumulates per-step VLM observations into a progressively-filling checklist and
computes an overall verdict. Designed for NOISY real-model input and an ONGOING
bolting process:
  * a violation locks only after `confirm_violation` consecutive violation windows
    (debounce against single-frame misreads),
  * safety flags below `min_severity` are advisories, not violations,
  * a locked violation CLEARS after `confirm_clear` compliant windows (a flagged
    item can become compliant again as the work progresses).
Setting confirm_* to 1 and min_severity to "low" reproduces immediate-and-sticky
behavior. No I/O except load_regulation().
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml

PENDING = "pending"
IN_PROGRESS = "in_progress"
SATISFIED = "satisfied"
OK = "ok"
VIOLATION = "violation"

_SEVERITY_RANK = {"low": 1, "med": 2, "medium": 2, "high": 3, "critical": 4}


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
    viol_count: int = 0
    clear_count: int = 0


def load_regulation(path) -> list[ChecklistItem]:
    data = yaml.safe_load(Path(path).read_text())
    return [ChecklistItem(it["id"], it["kind"], it["label"], it["evidence"])
            for it in data["items"]]


class ComplianceTracker:
    """Debounced, clearable compliance accumulator.

    confirm_satisfied: satisfied windows before a process item locks SATISFIED.
    confirm_violation: consecutive violation windows before an item locks VIOLATION.
    confirm_clear:     compliant windows before a locked VIOLATION clears.
    min_severity:      minimum safety-flag severity ("low"|"med"|"high") that counts
                       as a violation.
    """

    def __init__(self, items, confirm_satisfied=1, confirm_violation=2,
                 confirm_clear=2, min_severity="med"):
        self.items = items
        self.confirm_satisfied = max(1, confirm_satisfied)
        self.confirm_violation = max(1, confirm_violation)
        self.confirm_clear = max(1, confirm_clear)
        self.min_rank = _SEVERITY_RANK.get(str(min_severity).lower(), 2)
        self._kind = {it.id: it.kind for it in items}
        self._state = {it.id: _ItemState(OK if it.kind == "safety" else PENDING)
                       for it in items}

    def update(self, t_sec, observations, safety_flags):
        # 1) derive this window's per-item signal: "violation" | "compliant" | "progress"
        signal: dict[str, str] = {}
        note: dict[str, str] = {}
        for o in observations or []:
            iid = o.get("item_id")
            if iid not in self._state:
                continue
            s = o.get("status")
            if s == VIOLATION:
                signal[iid] = "violation"
                note[iid] = o.get("evidence", "")
            elif s == SATISFIED:
                if signal.get(iid) != "violation":
                    signal[iid] = "compliant"
            elif s == IN_PROGRESS:
                if signal.get(iid) not in ("violation", "compliant"):
                    signal[iid] = "progress"
            elif self._kind.get(iid) == "safety" and s in ("ok", "not_applicable"):
                if signal.get(iid) != "violation":
                    signal[iid] = "compliant"
        # 2) severity-gated safety flags force a violation signal
        for fl in safety_flags or []:
            iid = fl.get("id")
            if iid not in self._state:
                continue
            rank = _SEVERITY_RANK.get(str(fl.get("severity", "")).lower(), 0)
            if rank >= self.min_rank:
                signal[iid] = "violation"
                note[iid] = fl.get("note", "")
        # 3) a safety item with no violation signal this window is compliant
        for iid, kind in self._kind.items():
            if kind == "safety" and signal.get(iid) != "violation":
                signal.setdefault(iid, "compliant")
        # 4) apply transitions
        for iid, sig in signal.items():
            self._apply(iid, self._state[iid], sig, t_sec, note.get(iid, ""))

    def _apply(self, iid, st, sig, t_sec, note):
        kind = self._kind[iid]
        if sig == "violation":
            st.viol_count += 1
            st.clear_count = 0
            if st.status != VIOLATION and st.viol_count >= self.confirm_violation:
                st.status = VIOLATION
                st.since_t = t_sec
                st.note = note
            return
        # non-violation signal: "compliant" or "progress"
        st.viol_count = 0
        st.clear_count += 1
        if st.status == VIOLATION:
            if st.clear_count >= self.confirm_clear:
                st.since_t = None
                st.note = ""
                if kind == "safety":
                    st.status = OK
                else:
                    st.status = SATISFIED if sig == "compliant" else IN_PROGRESS
            return
        if kind == "safety":
            st.status = OK
            return
        if sig == "compliant":
            st.sat_count += 1
            if st.sat_count >= self.confirm_satisfied:
                st.status = SATISFIED
            elif st.status != SATISFIED:
                st.status = IN_PROGRESS
        elif sig == "progress" and st.status != SATISFIED:
            st.status = IN_PROGRESS

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
