"""Evaluate the compliance-milestone detector against the hand-labelled full-cycle GT.

Runs the milestone (physical signals; deterministic) over the cycle and scores the
fail-safe criteria from COMPLIANCE_DETECTION.md against eval/cycle_gt.json:

  F1  no COMPLIANCE_COMPLETE at any GT not_supported point   (== old false-safe = 0)
  F2  COMPLIANCE_COMPLETE by every GT supported point
  F3  bolts/screens progress is monotone non-decreasing
  point accuracy, and |t* - park_transition|

Default uses physical-only confirmation (reproducible, no VLM); --vlm runs the real
VLM confirmation at the candidate moment.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import yaml
import progress_tracker as pt
import compliance_milestone as cm


def run_eval(cfg, use_vlm=False, step=10.0):
    gt = json.loads(Path("eval/cycle_gt.json").read_text())
    res = cm.run(cfg, use_vlm=use_vlm, step=step)
    t_star = res["complete_at"]

    # predicted complete at a GT time = (t_star is set and time >= t_star)
    def predicted_complete(t):
        return t_star is not None and t >= t_star

    rows, false_safe, correct = [], 0, 0
    for p in gt["points"]:
        t, truth = p["cycle_sec"], p["truth"]
        pred = "supported" if predicted_complete(t) else "not_supported"
        ok = (pred == truth)
        correct += ok
        if pred == "supported" and truth == "not_supported":
            false_safe += 1
        rows.append((t, truth, pred, ok))

    # F3 monotone progress
    tl, ev, cls = pt.load_evidence()
    last_b = last_s = -1
    mono = True
    tt = 0.0
    while tt <= 3400:
        pr = pt.progress_at(tl, ev, tt, cls)
        if pr["bolts"] < last_b or pr["screens"] < last_s:
            mono = False
            break
        last_b, last_s = pr["bolts"], pr["screens"]
        tt += step

    park = gt["park_transition_sec"]
    f1 = (false_safe == 0)
    f2 = all(pred == "supported" for (_t, truth, pred, _ok) in rows if truth == "supported")
    acc = correct / len(rows) if rows else 0.0
    summary = {
        "t_star": t_star, "park_transition": park,
        "t_star_minus_park": (None if t_star is None else round(t_star - park, 1)),
        "F1_no_false_safe": f1, "false_safe_count": false_safe,
        "F2_detects_supported": f2,
        "F3_monotone_progress": mono,
        "point_accuracy": round(acc, 3), "n_points": len(rows),
        "n_bolts": res["n_bolts"], "n_screens": res["n_screens"],
        "used_vlm": use_vlm,
        "pass": bool(f1 and f2 and mono and acc >= 0.90 and t_star is not None),
    }
    return summary, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--vlm", action="store_true")
    ap.add_argument("--step", type=float, default=10.0)
    ap.add_argument("--out", default="data/compliance_eval.json")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    summary, rows = run_eval(cfg, use_vlm=a.vlm, step=a.step)
    Path(a.out).write_text(json.dumps({"summary": summary, "points": rows}, indent=2))
    print("=== COMPLIANCE EVAL (vs eval/cycle_gt.json) ===")
    for t, truth, pred, ok in rows:
        print(f"  {int(t)//60:02d}:{int(t)%60:02d}  truth={truth:14s} pred={pred:14s} {'OK' if ok else 'XX'}")
    print("\n--- summary ---")
    for k, v in summary.items():
        print(f"  {k:24s}: {v}")
    print("\n" + ("ALL CRITERIA MET ✅" if summary["pass"] else "NOT YET PASSING ❌"))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
