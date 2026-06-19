"""Validation harness: score the safety harness against hand-labelled ground truth.

Metrics map directly to SUCCESS_CRITERIA.md:
  C1 false_safe == 0   : a not-supported face is NEVER called SUPPORTED (critical).
  C2 positive recall   : truly-supported (bolted) faces ARE called SUPPORTED.
  C3 negatives flagged : not-supported faces get a non-safe verdict.
  C4 accuracy >= 0.90  : band(pred) == band(truth), every error biased to over-caution.

Usage: python3 eval.py [--config config.yaml] [--labels eval/labels.json] [--sampling 2.0]
Requires the local VLM server (offline at inference).
"""
from __future__ import annotations
import argparse
import collections
import copy
import json
from pathlib import Path
import yaml
import analyze

SUPPORTED = "SUPPORTED"
NOT_SUPPORTED = {"UNSUPPORTED", "DANGER", "NOT VERIFIED"}


def _run_sequence(cfg, clip, out_json):
    c = copy.deepcopy(cfg)
    c["paths"]["video"] = clip
    c["paths"]["analysis"] = out_json
    result = analyze.run_analysis(c, stub=False)
    return [s["verdict"] for s in result["steps"]], result["steps"]


def evaluate(cfg, labels):
    rows = []
    for seq in labels["sequences"]:
        clip = seq["clip"]
        if not Path(clip).exists():
            rows.append({"name": seq["name"], "truth": seq["truth"], "error": "clip missing"})
            continue
        verdicts, steps = _run_sequence(cfg, clip, f"eval/{seq['name']}.analysis.json")
        dist = collections.Counter(verdicts)
        any_supported = SUPPORTED in dist
        final = verdicts[-1] if verdicts else "NOT VERIFIED"
        truth = seq["truth"]
        if truth == SUPPORTED:
            false_safe = False
            correct = (final == SUPPORTED)
        else:  # not-supported ground truth
            false_safe = any_supported            # a SUPPORTED window here is a false-safe
            correct = not any_supported           # must never certify support
        rows.append({"name": seq["name"], "truth": truth, "final": final,
                     "dist": dict(dist), "any_supported": any_supported,
                     "false_safe": false_safe, "correct": correct,
                     "windows": len(steps)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--labels", default="eval/labels.json")
    ap.add_argument("--sampling", type=float, default=2.0,
                    help="override sampling_sec so the support_window streak is reachable in a clip")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.sampling:
        cfg["sampling_sec"] = args.sampling
    labels = json.loads(Path(args.labels).read_text())

    rows = evaluate(cfg, labels)
    scored = [r for r in rows if "error" not in r]

    print(f"\n=== Evaluation (model={cfg['model']}, votes={cfg.get('votes',1)}, "
          f"frame_max_width={cfg.get('frame_max_width')}, face_crop={bool(cfg.get('face_crop'))}, "
          f"sampling={cfg['sampling_sec']}s) ===")
    print(f"{'seq':10} {'truth':12} {'final':12} {'correct':8} {'false_safe':10} verdicts")
    for r in rows:
        if "error" in r:
            print(f"{r['name']:10} {r['truth']:12} ERROR {r['error']}")
            continue
        fs = "  <-- FALSE-SAFE!" if r["false_safe"] else ""
        print(f"{r['name']:10} {r['truth']:12} {r['final']:12} "
              f"{str(r['correct']):8} {str(r['false_safe']):10} {r['dist']}{fs}")

    false_safe_n = sum(r["false_safe"] for r in scored)
    sup = [r for r in scored if r["truth"] == SUPPORTED]
    neg = [r for r in scored if r["truth"] != SUPPORTED]
    pos_recall = sum(r["final"] == SUPPORTED for r in sup) / len(sup) if sup else float("nan")
    neg_correct = sum(r["correct"] for r in neg)
    accuracy = sum(r["correct"] for r in scored) / len(scored) if scored else 0.0

    print("\n--- scorecard (vs SUCCESS_CRITERIA.md) ---")
    print(f"C1 false-safe count : {false_safe_n}   {'PASS' if false_safe_n == 0 else 'FAIL — UNSAFE'}")
    print(f"C2 positive recall  : {pos_recall:.2f} ({sum(r['final']==SUPPORTED for r in sup)}/{len(sup)} supported->SUPPORTED)   "
          f"{'PASS' if sup and pos_recall >= 0.8 else 'FAIL'}")
    print(f"C3 negatives flagged: {neg_correct}/{len(neg)}   {'PASS' if neg_correct == len(neg) else 'FAIL'}")
    print(f"C4 accuracy         : {accuracy:.2f}   {'PASS' if accuracy >= 0.90 else 'FAIL'}")
    overall = (false_safe_n == 0 and sup and pos_recall >= 0.8
               and neg_correct == len(neg) and accuracy >= 0.90)
    print(f"\nOVERALL: {'PASS' if overall else 'NOT YET'}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
