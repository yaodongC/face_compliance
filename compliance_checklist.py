"""Vale-grounded face-support compliance checklist for the monitor GUI.

Each checklist item is a requirement distilled DIRECTLY from the Vale documents (cited
inline) that the harness can track from its sensors. evaluate_checklist() marks each item
done / pending at a given cycle-time and records WHEN it was completed.

Grounded in:
  [A] All Mines Face Support Guidelines, CMTS-2015-001 Rev5 (GENERAL PRACTICES + Scenario 1)
  [B] Division 6 Lateral Development Support Standard, Rev4 (Creighton — face: 4'x5', 6 bolts/
      sheet 3-0-3, 6.5' FS46, 4GA mesh, 1' overlap)

The signals that satisfy each item come from the fused harness (mesh installs from operator
tracking, bolt count from the IMU, coverage + completion from the compliance milestone),
i.e. the checklist is the human-readable, regulation-cited view of the same evidence.
"""
from __future__ import annotations

# (key, short label, Vale requirement it enforces) — order = display order
CHECKLIST_SPEC = [
    ("screened", "End face screened",
     "CMTS-2015-001: install screen on the face, wrapped to brow & side walls; overlap ≥ 3 squares"),
    ("bolted", "Bolt pattern installed",
     "CMTS-2015-001 / Div6: minimum 4'×5' face pattern (6.5' FS46) on the screen"),
    ("coverage", "Screen tight to walls & BOR",
     "CMTS-2015-001: bolts ≤ 1.5' from walls, top row ≤ 2' from back, bottom ≤ 5' from BOR"),
    ("workers", "Workers clear of bad ground",
     "CMTS-2015-001: keep operators out of the high-risk (unsupported) zone"),
    ("ordering", "Support done before drilling",
     "CMTS-2015-001: install face support BEFORE drilling the face"),
]


def _fmt(sec):
    if sec is None:
        return ""
    s = int(sec)
    return f"{s//60:02d}:{s%60:02d}"


def evaluate_checklist(sig, csec):
    """Evaluate the Vale checklist at cycle-time `csec`.

    sig fields (all times in cycle-seconds):
      screen_times   : list of mesh-install times (operator tracking)
      n_screens_req  : required mesh panels for this face (size-derived)
      bolt_times     : list of bolt-set times (IMU drilling episodes)
      n_bolts_req    : required bolts (= screens x 4'x5' pattern)
      danger_times   : times a worker was under a moving boom (operator safety)
      complete_at    : latched compliance-complete time (or None)

    Returns list of {key,label,source,done,done_time,detail} + an overall 'compliant' item.
    """
    items = []

    def add(key, label, source, done, done_time, detail):
        items.append({"key": key, "label": label, "source": source,
                      "done": bool(done), "done_time": done_time, "detail": detail})

    src = {k: s for k, _l, s in CHECKLIST_SPEC}
    lab = {k: l for k, l, _s in CHECKLIST_SPEC}

    # 1. End face screened — all required mesh panels installed
    st = sorted(sig.get("screen_times", []))
    req_s = int(sig.get("n_screens_req", 0)) or len(st)
    n_scr = sum(1 for t in st if t <= csec + 0.1)
    s_done_t = st[req_s - 1] if len(st) >= req_s and req_s > 0 else None
    s_done = s_done_t is not None and csec >= s_done_t - 0.1
    add("screened", lab["screened"], src["screened"], s_done, s_done_t if s_done else None,
        f"{min(n_scr, req_s)}/{req_s} panels")

    # 2. Bolt pattern installed — required bolts set (4'x5')
    bt = sorted(sig.get("bolt_times", []))
    req_b = int(sig.get("n_bolts_req", 0)) or len(bt)
    n_blt = sum(1 for t in bt if t <= csec + 0.1)
    b_done_t = bt[req_b - 1] if len(bt) >= req_b and req_b > 0 else None
    b_done = b_done_t is not None and csec >= b_done_t - 0.1
    add("bolted", lab["bolted"], src["bolted"], b_done, b_done_t if b_done else None,
        f"{min(n_blt, req_b)}/{req_b} bolts")

    # 3. Screen tight to walls & BOR — full coverage (achieved once all panels are in)
    c_done_t = s_done_t
    c_done = s_done
    add("coverage", lab["coverage"], src["coverage"], c_done, c_done_t if c_done else None,
        "full" if c_done else "partial")

    # 4. Workers clear of bad ground — no operator under a moving boom so far
    dgr = [t for t in sig.get("danger_times", []) if t <= csec + 0.1]
    w_ok = len(dgr) == 0
    add("workers", lab["workers"], src["workers"], w_ok, None,
        "clear" if w_ok else f"{len(dgr)} danger entr{'y' if len(dgr)==1 else 'ies'}")

    # 5. Support done before drilling — booms parked / support complete (latched)
    ca = sig.get("complete_at")
    o_done = ca is not None and csec >= ca - 0.1
    add("ordering", lab["ordering"], src["ordering"], o_done, ca if o_done else None, "")

    # overall — FACE SUPPORT COMPLIANT when every item is satisfied
    all_done = all(i["done"] for i in items)
    overall_t = ca if (all_done and ca is not None) else None
    items.append({"key": "compliant", "label": "FACE SUPPORT COMPLIANT",
                  "source": "All items above per CMTS-2015-001 / Division 6",
                  "done": all_done, "done_time": overall_t, "detail": "", "overall": True})
    return items


def n_done(items):
    return sum(1 for i in items if i["done"] and not i.get("overall"))
