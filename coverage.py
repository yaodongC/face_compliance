"""Screen-coverage tracking by INSTALLATION LOCATION.

A fully-bolted mesh blends into the face and is nearly impossible to detect
statically (validated: both the VLM and classical CV fail). But we have the whole
process, so instead we TRACK WHERE each screen is installed: every time a confirmed
operator is working in front of the face (loading a screen / fitting a bolt), the
location they are working becomes 'covered'. Accumulating these installation sites
over the cycle builds the coverage map; the face is compliant once the whole face
width has been worked.

This module post-processes data/operator_events.json (confirmed operators only,
with bboxes) into a coverage progression + a final coverage map.

Usage: python3 coverage.py [--events data/operator_events.json] [--cols 10]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

# face region of the frame that gets screened (fractions): exclude floor/edges
FACE_X = (0.20, 0.85)


def _col_span(bbox, cols, reach_frac=0.6):
    """Columns (over the FACE_X band) that an operator at `bbox` is installing.
    The operator reaches up and to the sides, so cover their own column plus a
    little spread."""
    fx0, fx1 = FACE_X
    cx = (bbox[0] + bbox[2]) / 2.0
    # operator half-width spread in face-column units
    spread = max((bbox[2] - bbox[0]), 0.06) * reach_frac
    lo = max(fx0, cx - spread)
    hi = min(fx1, cx + spread)
    c0 = int((lo - fx0) / (fx1 - fx0) * cols)
    c1 = int((hi - fx0) / (fx1 - fx0) * cols)
    return range(max(0, c0), min(cols, c1 + 1))


def build_coverage(events, cols=10):
    """Return (progression, covered_cols). progression = list of
    {cycle_sec, coverage} as columns get worked over time."""
    covered = [False] * cols
    progression = []
    for e in sorted(events, key=lambda x: x["cycle_sec"]):
        bb = e.get("person_bbox")
        if not bb:
            continue
        for c in _col_span(bb, cols):
            covered[c] = True
        progression.append({"cycle_sec": e["cycle_sec"],
                            "coverage": sum(covered) / cols,
                            "covered_cols": covered.copy()})
    return progression, covered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="data/operator_events.json")
    ap.add_argument("--cols", type=int, default=10)
    ap.add_argument("--full-frac", type=float, default=0.9,
                    help="coverage fraction considered 'fully covered / compliant'")
    a = ap.parse_args()
    data = json.loads(Path(a.events).read_text())
    events = [e for e in data["events"] if e.get("person_bbox")]
    prog, covered = build_coverage(events, a.cols)
    final = sum(covered) / a.cols if a.cols else 0.0

    print(f"=== Screen-coverage by installation tracking ({len(events)} confirmed operator sites) ===")
    print(f"face split into {a.cols} columns; coverage grows as the operator works across the face\n")
    last = -1
    for p in prog:
        cov = p["coverage"]
        if cov != last:
            bar = "".join("#" if c else "." for c in p["covered_cols"])
            cs = int(p["cycle_sec"])
            print(f"  cycle {cs//60:02d}:{cs%60:02d}  [{bar}]  {cov*100:3.0f}%")
            last = cov
    print(f"\nFINAL coverage: {final*100:.0f}%  -> "
          f"{'COMPLIANT (face fully covered)' if final >= a.full_frac else 'NOT fully covered'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
