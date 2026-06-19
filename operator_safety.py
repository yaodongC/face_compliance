"""Operator-in-front-while-drilling danger detector (hybrid CV + VLM).

The lethal scenario: after each screen is installed the jumbo operator walks IN
FRONT of the machine to load a new screen + friction bolt onto a boom. Drilling /
boom movement MUST be fully stopped while they do this. We detect:

  * arm_motion  -- classical-CV frame differencing in the lower-centre 'danger
                   zone' ROI (booms + the space the operator stands in). Reliable,
                   no hallucination. ~0.00 = still, >motion_thresh = moving.
  * person_in_front -- VLM check (a worker on foot in front of the jumbo).

Rule (fail-safe):
  DANGER          = person_in_front AND arm moving (drilling NOT stopped)
  OK_LOADING      = person_in_front AND arm stopped (compliant: drilling stopped)
  (no person)     = motion is informational only

This is the strongest, most safety-relevant signal in the harness: it never needs
the VLM to make a fine judgement, only to spot a person; the motion is measured.
"""
from __future__ import annotations
import base64
import json
import re
import cv2
import numpy as np
import requests

# danger-zone ROI as fractions [y0,y1,x0,x1] -- lower centre (booms + operator)
DANGER_ROI = (0.45, 1.0, 0.20, 0.80)
MOTION_PX_THRESH = 25      # per-pixel abs-diff threshold
MOTION_FRAC_THRESH = 0.02  # fraction of ROI pixels changed => "arm moving"

PERSON_PROMPT = ('Underground mine, camera on a drill jumbo facing the rock face. '
                 'JSON only: {"person_in_front":bool (a person/worker on foot in front '
                 'of the jumbo, near the face or booms),"hi_vis":bool (orange/yellow '
                 'hi-vis visible),"note":"<short>"}')


def roi_gray(img):
    h, w = img.shape[:2]
    y0, y1, x0, x1 = DANGER_ROI
    crop = img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
    return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)


def arm_motion(prev_img, img) -> float:
    """Fraction of danger-zone pixels that changed -- proxy for boom/arm movement."""
    d = cv2.absdiff(roi_gray(prev_img), roi_gray(img))
    return float((d > MOTION_PX_THRESH).mean())


def detect_person(img, cfg, *, session=None) -> dict:
    h, w = img.shape[:2]
    c = cv2.resize(img, (1100, int(h * 1100 / w)))
    ok, buf = cv2.imencode(".jpg", c, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    b64 = base64.b64encode(buf.tobytes()).decode()
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": PERSON_PROMPT},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}]
    sess = session or requests
    r = sess.post(f"{cfg['endpoint']}/chat/completions",
                  json={"model": cfg["model"], "messages": msgs, "max_tokens": 120,
                        "temperature": 0.0}, timeout=120).json()
    t = r["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", t, re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        d = {}
    return {"person_in_front": bool(d.get("person_in_front")),
            "hi_vis": bool(d.get("hi_vis")), "note": d.get("note", "")}


def classify(person_in_front: bool, motion: float,
             motion_thresh: float = MOTION_FRAC_THRESH) -> str:
    """Fail-safe operator-zone verdict."""
    moving = motion > motion_thresh
    if person_in_front and moving:
        return "DANGER"          # operator in front while the arm is moving
    if person_in_front and not moving:
        return "OK_LOADING"      # operator in front, drilling stopped (compliant)
    return "NO_PERSON"
