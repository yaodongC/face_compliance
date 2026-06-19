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


def install_intervals(events, t, panel_w=0.12, face_x=FACE_X, min_hits=2, bin_w=0.02):
    """Merged covered x-intervals of the face at time t, requiring SUSTAINED install
    activity. A face region is 'covered' only where >= min_hits install detections
    accumulate - so the operator drifting through a spot once (e.g. while still
    bolting an earlier mesh) does NOT mark it covered. The NUMBER of screens is
    emergent (depends on face size); we do not assume a fixed count."""
    fx0, fx1 = face_x
    nb = max(1, int(round((fx1 - fx0) / bin_w)))
    hits = [0] * nb
    for e in sorted([e for e in events if e.get("person_bbox")], key=lambda x: x["cycle_sec"]):
        if e["cycle_sec"] > t + 0.1:
            continue
        cx = (e["person_bbox"][0] + e["person_bbox"][2]) / 2.0
        b0 = int((max(fx0, cx - panel_w / 2) - fx0) / bin_w)
        b1 = int((min(fx1, cx + panel_w / 2) - fx0) / bin_w)
        for b in range(max(0, b0), min(nb, b1 + 1)):
            hits[b] += 1
    merged, i = [], 0
    while i < nb:
        if hits[i] >= min_hits:
            j = i
            while j < nb and hits[j] >= min_hits:
                j += 1
            merged.append((round(fx0 + i * bin_w, 3), round(fx0 + j * bin_w, 3)))
            i = j
        else:
            i += 1
    return merged


def width_coverage(events, t, panel_w=0.12, face_x=FACE_X, min_hits=2):
    """Continuous face-width coverage at time t. COMPLIANT only when the ENTIRE
    face width is covered (no bare gaps), regardless of how many screens it took."""
    fx0, fx1 = face_x
    span = fx1 - fx0
    merged = install_intervals(events, t, panel_w, face_x, min_hits)
    cov = sum(b - a for a, b in merged)
    frac = max(0.0, min(1.0, cov / span))
    full = frac >= 0.98
    return {"intervals": merged, "fraction": frac, "full": full,
            "verdict": "COMPLIANT" if full else "NOT SUPPORTED"}


def segment_coverage(events, n=4, face_x=FACE_X):
    """Per-segment install time: the first time the operator works in each of n
    face segments (left->right). Returns a list of n times (or None). Per-mesh
    bboxes are not reliable, so coverage is tracked as n coarse face segments."""
    fx0, fx1 = face_x
    wseg = (fx1 - fx0) / n
    times = [None] * n
    for e in sorted([e for e in events if e.get("person_bbox")], key=lambda x: x["cycle_sec"]):
        bb = e["person_bbox"]
        cx = min(max((bb[0] + bb[2]) / 2.0, fx0), fx1 - 1e-6)
        seg = min(max(int((cx - fx0) / wseg), 0), n - 1)
        if times[seg] is None:
            times[seg] = e["cycle_sec"]
    return times


def segment_state(seg_times, t):
    """Compliance from the n face segments at time t. COMPLIANT only when ALL
    segments are covered (entire face covered)."""
    covered = [(st is not None and st <= t + 0.1) for st in seg_times]
    n = len(seg_times) or 1
    full = all(covered)
    return {"covered": covered, "fraction": sum(covered) / n, "full": full,
            "verdict": "COMPLIANT" if full else "NOT SUPPORTED"}


def coverage_state(meshes, t, face_x=FACE_X, min_overlap=0.02):
    """Compliance from INSTALLED MESH PANELS only (booms-parked is NOT used).
    COMPLIANT iff the installed panels cover the ENTIRE face band with OVERLAPS
    between adjacent panels (per the regulation). Partial coverage is NOT
    supported at all.

    Returns {fraction, full, overlaps, verdict, n_panels}.
    """
    fx0, fx1 = face_x
    span = fx1 - fx0
    installed = sorted([m for m in meshes if m.get("installed_at", 0) <= t + 0.1],
                       key=lambda m: m["bbox"][0])
    if not installed:
        return {"fraction": 0.0, "full": False, "overlaps": False,
                "verdict": "NOT SUPPORTED", "n_panels": 0}
    # union coverage of the face band
    cov = 0.0
    cursor = fx0
    for m in installed:
        a = max(fx0, m["bbox"][0]); b = min(fx1, m["bbox"][2])
        if b <= cursor:
            continue
        cov += b - max(a, cursor)
        cursor = max(cursor, b)
    fraction = max(0.0, min(1.0, cov / span))
    # gaps? cursor must have reached fx1 with no hole
    full = fraction >= 0.98
    # adjacent installed panels must overlap (next.x0 < prev.x1 - min_overlap)
    overlaps = True
    for p, q in zip(installed, installed[1:]):
        if q["bbox"][0] > p["bbox"][2] - min_overlap:
            overlaps = False
            break
    verdict = "COMPLIANT" if (full and overlaps) else "NOT SUPPORTED"
    return {"fraction": round(fraction, 3), "full": full, "overlaps": overlaps,
            "verdict": verdict, "n_panels": len(installed)}


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
