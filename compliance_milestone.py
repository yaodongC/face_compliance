"""Latched face-support compliance milestone (fail-safe).

Recognises the MOMENT the face becomes compliant: 4 screens installed AND 16 bolts set
AND the face fully covered AND a VLM hi-res look confirms screen+plates+booms-parked.
The decision is made HERE in code (the VLM only reports what it sees) and is LATCHED —
once COMPLETE it never reverts; it can never fire during the active-work phase because
the physical bolt/screen counters are below target there.

Gates (all required, in order):
  S  screens_installed >= TARGET_SCREENS (4)
  B  bolts_installed   >= TARGET_BOLTS   (16)   [primary = IMU sustained drilling episodes]
  C  face_coverage     >= cover_thr      (whole face worked)
  V  VLM final confirmation: face_screened AND plates_visible AND booms_parked
       (asymmetric consensus; fail-safe default not-supported)

Run:  python3 compliance_milestone.py            # full (calls the VLM at the candidate moment)
      python3 compliance_milestone.py --no-vlm   # physical-only (confirmation auto-passed)
      python3 compliance_milestone.py --checkpoints  # also probe VLM mid-cycle (shows it says NO)
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import yaml
import progress_tracker as pt

PRE = "PRE_SUPPORT"
INSTALLING = "INSTALLING"
AWAIT = "AWAITING_CONFIRMATION"   # physical gates met, VLM not yet confirmed
COMPLETE = "COMPLIANCE_COMPLETE"

COVER_THR = 0.85


class ComplianceMilestone:
    """State machine fed (t, progress, confirm). Latches COMPLETE; logs each milestone."""

    def __init__(self, cover_thr=COVER_THR):
        self.cover_thr = cover_thr
        self.phase = PRE
        self.complete_at = None
        self.confirm = None
        self.log = []           # milestone events
        self._bolts = 0
        self._screens = 0

    def _emit(self, t, kind, msg, **kw):
        self.log.append({"cycle_sec": round(t, 1), "type": kind, "description": msg, **kw})

    def physical_ready(self, prog):
        return (prog["screens"] >= prog["screens_target"] and prog["bolts"] >= prog["bolts_target"]
                and prog["coverage"] >= self.cover_thr)

    def update(self, t, prog, confirm=None):
        """confirm: dict from vlm_client.confirm_supported (or None = not evaluated;
        {'supported': True} bypass for --no-vlm)."""
        if self.phase == COMPLETE:
            return self.phase
        nb, nm = prog["bolts_target"], prog["screens_target"]
        # progress milestones (log the first time each bolt / screen lands)
        if prog["bolts"] > self._bolts:
            for n in range(self._bolts + 1, prog["bolts"] + 1):
                self._emit(t, "bolt_installed", f"bolt {n}/{nb} set", bolt=n)
            self._bolts = prog["bolts"]
        if prog["screens"] > self._screens:
            for n in range(self._screens + 1, prog["screens"] + 1):
                self._emit(t, "screen_installed", f"screen {n}/{nm} bolted", screen=n)
            self._screens = prog["screens"]
        if self.phase == PRE and (prog["bolts"] > 0 or prog["screens"] > 0):
            self.phase = INSTALLING
        # compliance gate
        if self.physical_ready(prog):
            if confirm is None:
                self.phase = AWAIT
            elif confirm.get("supported"):
                self.phase = COMPLETE
                self.complete_at = round(t, 1)
                self.confirm = confirm
                self._emit(t, "compliance_complete",
                           f"FACE SUPPORT COMPLETE — {prog['screens']}/{nm} screens, "
                           f"{prog['bolts']}/{nb} bolts, coverage {prog['coverage']:.0%}, VLM-confirmed",
                           confirm=confirm)
            else:
                self.phase = AWAIT     # physical done but VLM not satisfied -> stay fail-safe
        return self.phase


def run(cfg, *, use_vlm=True, checkpoints=False, step=10.0):
    tl, ev, cls = pt.load_evidence()
    tg = pt.load_targets()                     # SIZE-DERIVED requirement for THIS face
    bolts = pt.bolt_episodes(tl, cls)
    tmax = (bolts[-1]["set_at"] + 90) if bolts else 3400
    ms = ComplianceMilestone()
    grab = None
    confirm_cache = {}
    if use_vlm:
        from cycle_frames import FrameGrabber
        grab = FrameGrabber()

    def confirm_at(t):
        if not use_vlm:
            return {"supported": True, "note": "physical-only (--no-vlm)"}
        if t in confirm_cache:
            return confirm_cache[t]
        import vlm_client
        frames = grab.around(t, n=cfg.get("confirm_frames", 3), span=24.0)
        c = vlm_client.confirm_supported(frames, cfg)
        confirm_cache[t] = c
        return c

    t = 0.0
    last_log = None
    while t <= tmax:
        prog = pt.progress_at(tl, ev, t, cls, tg)
        confirm = None
        if ms.physical_ready(prog):
            confirm = confirm_at(t)
        elif checkpoints and use_vlm and prog["bolts"] in (4, 8, 12) and prog["bolts"] != last_log:
            c = confirm_at(t)   # demonstrate fail-safe: VLM says NOT supported mid-cycle
            print(f"  [checkpoint {int(t)//60:02d}:{int(t)%60:02d} bolts={prog['bolts']}] "
                  f"VLM supported={c.get('supported')} ({c.get('note','')[:40]})")
            last_log = prog["bolts"]
        phase = ms.update(t, prog, confirm)
        if phase != last_log and phase in (AWAIT, COMPLETE):
            pass
        if ms.phase == COMPLETE:
            break
        t += step

    result = {"phase": ms.phase, "complete_at": ms.complete_at,
              "complete_mmss": (None if ms.complete_at is None
                                else f"{int(ms.complete_at)//60:02d}:{int(ms.complete_at)%60:02d}"),
              "confirm": ms.confirm, "n_bolts": ms._bolts, "n_screens": ms._screens,
              "milestones": ms.log,
              "face_width": tg.get("face_width"), "target_source": tg.get("source"),
              "params": {"cover_thr": ms.cover_thr,
              "target_bolts": tg["bolts_required"], "target_screens": tg["meshes_required"],
              "mesh_panel_w": tg.get("panel_w"), "mesh_overlap": tg.get("overlap")},
              "used_vlm": use_vlm}
    if grab:
        grab.release()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--no-vlm", action="store_true", help="physical-only (confirmation auto-passed)")
    ap.add_argument("--checkpoints", action="store_true", help="also probe VLM mid-cycle")
    ap.add_argument("--step", type=float, default=10.0)
    ap.add_argument("--out", default="data/compliance_result.json")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    res = run(cfg, use_vlm=not a.no_vlm, checkpoints=a.checkpoints, step=a.step)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print("\n=== COMPLIANCE MILESTONE ===")
    print(f"phase           : {res['phase']}")
    print(f"complete_at     : {res['complete_mmss']}  ({res['complete_at']}s)")
    print(f"bolts / screens : {res['n_bolts']}/16  {res['n_screens']}/4")
    if res["confirm"]:
        c = res["confirm"]
        print(f"VLM confirm     : screened={c.get('face_screened')} plates={c.get('plates_visible')} "
              f"parked={c.get('booms_parked')} conf={c.get('confidence')}")
    print(f"milestones      : {len(res['milestones'])} logged -> {a.out}")
    for m in res["milestones"]:
        s = int(m["cycle_sec"])
        print(f"   {s//60:02d}:{s%60:02d}  {m['type']:20s} {m['description']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
