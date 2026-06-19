"""Validation harness: score the safety harness against hand-labelled ground truth.

For each labelled sequence it runs the full perception+consensus+aggregation
pipeline and compares the result to the human label. The metric that matters for
a safety-critical system is FALSE-SAFE: a truly-unsupported/hazardous face that
the system reports as SUPPORTED. That count MUST be zero. Over-caution (reporting
UNSUPPORTED/NOT VERIFIED when nothing is wrong) is acceptable.

Usage: python3 eval.py [--config config.yaml] [--labels eval/labels.json]
Requires the VLM server to be up (uses the real model, with K-vote consensus).
"""
from __future__ import annotations
import argparse
import collections
import json
from pathlib import Path
import copy
import yaml
import analyze

UNSAFE_TRUTHS = {"UNSUPPORTED", "DANGER", "NOT VERIFIED"}


def _run_sequence(cfg, clip, out_json):
    c = copy.deepcopy(cfg)
    c["paths"]["video"] = clip
    c["paths"]["analysis"] = out_json
    result = analyze.run_analysis(c, stub=False)
    return [s["verdict"] for s in result["steps"]], result["steps"]


def evaluate(cfg, labels):
    rows = []
    false_safe = 0
    for seq in labels["sequences"]:
        clip = seq["clip"]
        if not Path(clip).exists():
            rows.append({"name": seq["name"], "truth": seq["truth"],
                         "error": "clip missing"})
            continue
        out_json = f"eval/{seq['name']}.analysis.json"
        verdicts, steps = _run_sequence(cfg, clip, out_json)
        dist = collections.Counter(verdicts)
        any_supported = "SUPPORTED" in dist
        final = verdicts[-1] if verdicts else "NOT VERIFIED"
        # how badly did the raw model hallucinate support on this clip?
        bolt_fp = sum(1 for s in steps if s["perception"].get("bolts_visible"))
        truth = seq["truth"]
        fs = truth in UNSAFE_TRUTHS and any_supported
        false_safe += int(fs)
        rows.append({"name": seq["name"], "truth": truth, "final": final,
                     "dist": dict(dist), "any_supported": any_supported,
                     "false_safe": fs, "bolt_fp": f"{bolt_fp}/{len(steps)}"})
    return rows, false_safe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--labels", default="eval/labels.json")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    labels = json.loads(Path(args.labels).read_text())

    rows, false_safe = evaluate(cfg, labels)

    print(f"\n=== Safety-harness evaluation (model={cfg['model']}, votes={cfg.get('votes',1)}) ===")
    print(f"{'seq':6} {'truth':12} {'final':12} {'any_SUPPORTED':14} {'bolt_halluc':12} verdicts")
    for r in rows:
        if "error" in r:
            print(f"{r['name']:6} {r['truth']:12} ERROR: {r['error']}")
            continue
        flag = "  <-- FALSE-SAFE!" if r["false_safe"] else ""
        print(f"{r['name']:6} {r['truth']:12} {r['final']:12} "
              f"{str(r['any_supported']):14} {r['bolt_fp']:12} {r['dist']}{flag}")

    scored = [r for r in rows if "error" not in r]
    correct = sum(1 for r in scored
                  if r["truth"] in UNSAFE_TRUTHS and not r["any_supported"])
    print("\n--- scorecard ---")
    print(f"sequences scored : {len(scored)}")
    print(f"FALSE-SAFE (truly-unsafe called SUPPORTED): {false_safe}   "
          f"{'PASS (0)' if false_safe == 0 else 'FAIL — UNSAFE'}")
    print(f"flagged-unsafe correctly (no SUPPORTED on an unsafe face): {correct}/{len(scored)}")
    print("overall:", "PASS" if false_safe == 0 else "FAIL")
    return 0 if false_safe == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
