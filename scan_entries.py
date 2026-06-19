"""Dense, classical detection of operator DANGER-ZONE ENTRIES.

The operator's hi-vis ORANGE in the danger zone is a cheap per-frame signal (no VLM,
no sampling gaps), so we can catch EVERY entry - including the brief bolt-reload
visits the 8 s VLM scan missed. Each entry is classified by the BOOM motion at entry
(operator-pixels masked out): boom still moving => non-compliant; boom stopped =>
compliant reload. Output: data/operator_entries.json.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import operator_safety as osf

ORANGE_LO, ORANGE_HI = (3, 110, 110), (20, 255, 255)
PRESENT_TH = 0.012      # orange fraction in the danger ROI => operator present
MOTION_TH = 0.020       # masked boom-motion fraction => boom moving
MERGE_GAP = 25.0        # s; flicker within this is the same presence


def _roi(img):
    h, w = img.shape[:2]
    y0, y1, x0, x1 = osf.DANGER_ROI
    return img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


def _orange_mask(roi_bgr):
    return cv2.inRange(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV), ORANGE_LO, ORANGE_HI)


def _boom_motion(prev_roi, roi):
    """Fraction of danger-ROI pixels that changed, EXCLUDING operator (orange) px."""
    g0 = cv2.GaussianBlur(cv2.cvtColor(prev_roi, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    g1 = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    moving = cv2.absdiff(g0, g1) > 25
    op = cv2.dilate((_orange_mask(prev_roi) | _orange_mask(roi)), np.ones((9, 9), np.uint8)) > 0
    boom = moving & ~op
    return float(boom.mean())


def scan(video, index_path):
    idx = {int(f): float(c) for f, c in
           (l.split(",") for l in Path(index_path).read_text().splitlines()[1:])}
    cap = cv2.VideoCapture(video)
    prev_roi = None
    sig = []   # (t, orange, boom_motion)
    for fr in sorted(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
        ok, im = cap.read()
        if not ok:
            continue
        roi = _roi(im)
        orange = float(_orange_mask(roi).mean() / 255)
        motion = _boom_motion(prev_roi, roi) if prev_roi is not None else 0.0
        sig.append((idx[fr], orange, motion))
        prev_roi = roi
    cap.release()
    return sig


def entries_from_signal(sig, present_th=PRESENT_TH, motion_th=MOTION_TH, merge_gap=MERGE_GAP):
    # presence intervals (debounced)
    intervals, start, last = [], None, None
    for t, orange, _ in sig:
        if orange >= present_th:
            if start is None:
                start = t
            elif t - last > merge_gap:
                intervals.append((start, last))
                start = t
            last = t
    if start is not None:
        intervals.append((start, last))
    # classify each entry by boom motion in a small window around entry
    bymotion = {t: m for t, _, m in sig}
    times = [t for t, _, _ in sig]
    out = []
    for a, b in intervals:
        win = [bymotion[t] for t in times if a - 4 <= t <= a + 4]
        mv = max(win) if win else 0.0
        out.append({"time": round(a, 1), "end": round(b, 1), "boom_motion": round(mv, 3),
                    "verdict": "NON_COMPLIANT_ENTRY" if mv > motion_th else "SAFE_RELOAD"})
    return out


def _classify(motion, thresh):
    return "NON_COMPLIANT_ENTRY" if motion > thresh else "SAFE_RELOAD"


def reconcile(entries, operator_path, thresh, window=35.0):
    """Produce the UNION of the dense classical entries and the VLM operator
    sessions (each scan misses some), and classify every entry by `thresh`. For the
    boom-motion VALUE prefer the VLM's bag-level `entry_motion` (reliable) when a
    session is within `window` s; else use the classical timelapse motion."""
    if not Path(operator_path).exists():
        for e in entries:
            e["entry_motion"] = e["boom_motion"]
            e["source"] = "classical"
            e["verdict"] = _classify(e["boom_motion"], thresh)
        return entries
    from operator_safety import classify_sessions
    sessions = classify_sessions(json.loads(Path(operator_path).read_text())["events"])
    used = set()
    for e in entries:
        near = [(i, s) for i, s in enumerate(sessions) if abs(s["start"] - e["time"]) <= window]
        if near:
            i, s = min(near, key=lambda x: abs(x[1]["start"] - e["time"]))
            used.add(i)
            e["entry_motion"] = round(s["entry_motion"], 3)
            e["source"] = "vlm"
        else:
            e["entry_motion"] = e["boom_motion"]
            e["source"] = "classical"
        e["verdict"] = _classify(e["entry_motion"], thresh)
    # add VLM sessions no classical entry covered (the orange scan missed them)
    for i, s in enumerate(sessions):
        if i not in used:
            m = round(s["entry_motion"], 3)
            entries.append({"time": round(s["start"], 1), "end": round(s["end"], 1),
                            "boom_motion": m, "entry_motion": m, "source": "vlm",
                            "verdict": _classify(m, thresh)})
    entries.sort(key=lambda e: e["time"])
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/full_cycle.mp4")
    ap.add_argument("--index", default="data/full_cycle.idx")
    ap.add_argument("--operator", default="data/operator_events.json")
    ap.add_argument("--out", default="data/operator_entries.json")
    ap.add_argument("--config", default="config.yaml")
    a = ap.parse_args()
    import yaml
    cfg = yaml.safe_load(Path(a.config).read_text()) if Path(a.config).exists() else {}
    thresh = float(cfg.get("boom_motion_thresh", 0.035))
    sig = scan(a.video, a.index)
    entries = reconcile(entries_from_signal(sig), a.operator, thresh)
    nviol = sum(1 for e in entries if e["verdict"] == "NON_COMPLIANT_ENTRY")
    print(f"(boom-moving threshold = {thresh})")
    Path(a.out).write_text(json.dumps({"entries": entries}, indent=2))
    print(f"=== {len(entries)} danger-zone entries ({nviol} entered while boom moving) -> {a.out} ===")
    for e in entries:
        ts = int(e["time"])
        tag = "DANGER " if e["verdict"] == "NON_COMPLIANT_ENTRY" else "reload "
        print(f"  {ts//60:02d}:{ts%60:02d}  {tag} motion={e.get('entry_motion', e['boom_motion'])}"
              f"  ({e.get('source', '')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
