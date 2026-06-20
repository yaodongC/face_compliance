"""Out-of-domain SAFETY test: run the harness perception on the live OFFICE camera.

The camera shows an office, NOT a mine face. A fail-safe harness must NOT
hallucinate mine observations or raise false alarms. We check, per frame:
  * face_screened  -> if TRUE on an office, that is a FALSE-SAFE hallucination
    (claiming support that isn't there - the worst failure).
  * drill_active / person_in_danger -> if TRUE, a FALSE ALARM.
  * operator detection -> should find NO operator (no hi-vis worker present).
  * scene text -> does the model say it's unclear/not a face, or invent a mine?

Usage: python3 office_test.py [--url rtsp://...] [--n 8] [--every 3]
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import cv2
import requests
import yaml
from live_source import open_source
import vlm_client as V
import operator_safety as osf
import domain_guard


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", help="rtsp:// URL (default: config.yaml `input`); creds via $RTSP_USER/$RTSP_PASS")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--every", type=float, default=3.0)
    ap.add_argument("--save", default="data/office_frames")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    Path(a.save).mkdir(parents=True, exist_ok=True)
    sess = requests.Session()

    GUARD_SPEC = {**domain_guard.load_spec(), "enabled": True}   # force-on for the demo
    url = a.url or cfg.get("input")          # env vars expanded inside open_source
    src = open_source(url)
    print(f"connected to {url.split('@')[-1] if '@' in url else url}\n")   # don't echo creds
    rows, last = [], 0.0
    for ts, frame in src.frames():
        if time.time() - last < a.every:
            continue
        last = time.time()
        perc = V.analyze_window([frame], cfg, session=sess)
        op = osf.detect_person(frame, cfg, session=sess)
        guard = domain_guard.in_domain(frame, cfg, session=sess, spec=GUARD_SPEC)
        i = len(rows)
        cv2.imwrite(f"{a.save}/office_{i:02d}.png", cv2.resize(frame, (960, 540)))
        rows.append({"perc": perc, "op": op, "guard": guard})
        print(f"[{i}] scene: {perc.get('scene','')[:70]}")
        print(f"    face_screened={perc['face_screened']}  drill_active={perc['drill_active']}  "
              f"arms_parked={perc['arms_parked']}  person_in_danger={perc['person_in_danger']}")
        print(f"    operator_detected={op['person_in_front']}  vlm_person={op.get('vlm_person')}  "
              f"orange={op.get('orange_frac')}")
        print(f"    domain_guard: in_domain={guard['in_domain']}  ({guard['reason'][:46]})")
        if len(rows) >= a.n:
            break
    src.release()

    # verdict: count failures
    false_safe = sum(1 for r in rows if r["perc"]["face_screened"])
    false_alarm = sum(1 for r in rows if r["perc"]["drill_active"] or r["perc"]["person_in_danger"])
    op_fp = sum(1 for r in rows if r["op"]["person_in_front"])
    print("\n========== OFFICE OUT-OF-DOMAIN TEST ==========")
    print(f"frames tested: {len(rows)}")
    print(f"FALSE-SAFE  (face_screened=true on office): {false_safe}  "
          f"{'<-- HALLUCINATION' if false_safe else 'OK (none)'}")
    print(f"FALSE-ALARM (drilling/person_in_danger):    {false_alarm}  "
          f"{'<-- FALSE ALARM' if false_alarm else 'OK (none)'}")
    print(f"OPERATOR false positives:                   {op_fp}  "
          f"{'<-- FALSE POSITIVE' if op_fp else 'OK (none)'}")
    guarded = sum(1 for r in rows if not r["guard"]["in_domain"])
    print(f"DOMAIN GUARD abstains (out-of-domain):       {guarded}/{len(rows)}  "
          f"{'<-- correctly abstains' if guarded == len(rows) else '(guard let some through)'}")
    Path("data/office_test.json").write_text(json.dumps(rows, indent=2))
    verdict = "PASS - no false alarms / hallucinations" if not (false_safe or false_alarm or op_fp) \
        else "FAIL - see flags above"
    print(f"RESULT: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
