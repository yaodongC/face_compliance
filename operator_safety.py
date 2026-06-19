"""Operator-in-front-while-drilling danger detector (hybrid CV + VLM).

The lethal scenario: after each screen is installed the jumbo operator walks IN
FRONT of the machine to load a new screen + friction bolt onto a boom. Drilling /
boom movement MUST be fully stopped while they do this. We detect:

  * person_in_front + person_bbox -- VLM (Qwen grounding) spots the worker and
    locates them.
  * arm_motion -- classical-CV frame differencing in the lower-centre danger-zone
    ROI, with the OPERATOR'S OWN region MASKED OUT (so we measure the boom/arm
    moving, not the operator walking). Reliable, no hallucination.

Rule (fail-safe):
  DANGER      = person_in_front AND arm (boom) moving  -> drilling NOT stopped
  OK_LOADING  = person_in_front AND arm stopped         -> compliant
  NO_PERSON   = nobody in front; motion is informational only
"""
from __future__ import annotations
import base64
import json
import re
import cv2
import numpy as np
import requests
from harness_config import PARAMS

_OP = PARAMS["operator"]   # single source of truth (config.yaml params.operator)
# danger-zone ROI as fractions [y0,y1,x0,x1] -- lower centre (booms + operator)
DANGER_ROI = tuple(_OP["danger_roi"])
MOTION_PX_THRESH = _OP["motion_px_thresh"]      # per-pixel abs-diff threshold
# Fraction of danger-zone pixels that must change for the boom to count as MOVING
# (data clusters <=0.023 stopped vs >=0.046 moving). See params.operator.
MOTION_FRAC_THRESH = _OP["boom_motion_thresh"]
_ORANGE_LO = tuple(_OP["orange_hsv_lo"])
_ORANGE_HI = tuple(_OP["orange_hsv_hi"])

PERSON_PROMPT = (
    'Underground mine, camera on a drill jumbo facing the rock face. Find the '
    'WORKER on foot in front of the jumbo (hi-vis). Return JSON only: '
    '{"person_in_front":bool,"hi_vis":bool,'
    '"person_bbox":[x0,y0,x1,y1] as fractions 0-1 of the image (or null),'
    '"action":"<if a worker is in front, what are they doing? e.g. loading a '
    'screen/mesh onto a boom, fitting a friction bolt, reaching up to the face, '
    'walking, standing - else empty>","note":"<short>"}')


def _roi_gray(img):
    h, w = img.shape[:2]
    y0, y1, x0, x1 = DANGER_ROI
    return cv2.cvtColor(img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)],
                        cv2.COLOR_BGR2GRAY)


def arm_motion(prev_img, img, person_bbox=None, pad=12) -> float:
    """Fraction of danger-zone pixels that changed, with the operator's bbox
    masked out so the result reflects BOOM/ARM movement, not the operator."""
    h, w = img.shape[:2]
    diff = (cv2.absdiff(_roi_gray(prev_img), _roi_gray(img)) > MOTION_PX_THRESH).astype(np.uint8)
    if person_bbox:
        y0r, _, x0r, _ = DANGER_ROI
        zy0, zx0 = int(y0r * h), int(x0r * w)
        zh, zw = diff.shape
        bx0, by0, bx1, by1 = person_bbox
        px0 = max(0, int(bx0 * w) - zx0 - pad)
        px1 = min(zw, int(bx1 * w) - zx0 + pad)
        py0 = max(0, int(by0 * h) - zy0 - pad)
        py1 = min(zh, int(by1 * h) - zy0 + pad)
        if px1 > px0 and py1 > py0:
            diff[py0:py1, px0:px1] = 0
    return float(diff.mean())


def hi_vis_orange_fraction(img, bbox) -> float:
    """Fraction of bbox pixels that are hi-vis ORANGE (worker vest). The booms /
    jumbo are YELLOW, not orange, so this rejects boom-arm false positives that the
    VLM mislabels as an operator."""
    if not bbox:
        return 0.0
    h, w = img.shape[:2]
    x0, y0, x1, y1 = bbox
    crop = img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, _ORANGE_LO, _ORANGE_HI)
    return float((mask > 0).mean())


def _bbox_ok(bbox, min_area=_OP["bbox"]["min_area"], max_area=_OP["bbox"]["max_area"],
             max_aspect_wh=_OP["bbox"]["max_aspect"]) -> bool:
    """Reject degenerate / non-person bboxes (thin boom lines, whole-frame, etc.).
    People are roughly upright, so width/height should not be very large."""
    if not bbox:
        return False
    x0, y0, x1, y1 = bbox
    bw, bh = x1 - x0, y1 - y0
    if bw <= 0 or bh <= 0:
        return False
    area = bw * bh
    return (min_area <= area <= max_area) and (bw / bh <= max_aspect_wh)


# minimum hi-vis-orange fraction in the bbox to accept a person (classical gate)
MIN_ORANGE = _OP["min_orange"]


def detect_person(img, cfg, *, session=None) -> dict:
    h, w = img.shape[:2]
    c = cv2.resize(img, (1000, int(h * 1000 / w)))
    ok, buf = cv2.imencode(".jpg", c, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    b64 = base64.b64encode(buf.tobytes()).decode()
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": PERSON_PROMPT},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}]
    sess = session or requests
    r = sess.post(f"{cfg['endpoint']}/chat/completions",
                  json={"model": cfg["model"], "messages": msgs, "max_tokens": 140,
                        "temperature": 0.0}, timeout=120).json()
    t = r["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", t, re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        d = {}
    bb = d.get("person_bbox")
    if not (isinstance(bb, (list, tuple)) and len(bb) == 4 and all(isinstance(x, (int, float)) for x in bb)):
        bb = None
    vlm_person = bool(d.get("person_in_front"))
    # CLASSICAL GATE: a real operator must be hi-vis ORANGE in a person-shaped bbox.
    # This rejects the booms/equipment the VLM mislabels as an operator.
    orange = hi_vis_orange_fraction(img, bb)
    confirmed = vlm_person and _bbox_ok(bb) and orange >= MIN_ORANGE
    return {"person_in_front": confirmed,
            "vlm_person": vlm_person, "orange_frac": round(orange, 3),
            "hi_vis": bool(d.get("hi_vis")), "person_bbox": bb if confirmed else None,
            "action": str(d.get("action", "") or "") if confirmed else "",
            "note": d.get("note", "")}


def annotate(frame, person_bbox=None, verdict="NO_PERSON", action="",
             motion=None, cycle_sec=None):
    """Draw the danger-zone ROI, the person bbox, and the verdict/action onto a
    copy of the frame so a human can verify the model's detection."""
    img = frame.copy()
    h, w = img.shape[:2]
    # danger-zone ROI (yellow)
    y0, y1, x0, x1 = DANGER_ROI
    cv2.rectangle(img, (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h)), (0, 220, 220), 2)
    cv2.putText(img, "danger zone", (int(x0 * w) + 6, int(y0 * h) + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 220), 2, cv2.LINE_AA)
    # person bbox: red if DANGER else green
    col = (40, 40, 220) if verdict == "DANGER" else (60, 200, 60)
    if person_bbox:
        bx0, by0, bx1, by1 = person_bbox
        cv2.rectangle(img, (int(bx0 * w), int(by0 * h)), (int(bx1 * w), int(by1 * h)), col, 3)
        cv2.putText(img, "operator", (int(bx0 * w), max(20, int(by0 * h) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
    # banner
    cv2.rectangle(img, (0, 0), (w, 64), col if verdict != "NO_PERSON" else (60, 60, 60), -1)
    txt = verdict + (f"   boom motion={motion:.3f}" if motion is not None else "")
    if cycle_sec is not None:
        txt = f"cycle {int(cycle_sec)//60:02d}:{int(cycle_sec)%60:02d}   " + txt
    cv2.putText(img, txt, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    if action:
        cv2.putText(img, "operator: " + action[:70], (14, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


# face band the screens are bolted onto (clamp grounding to here): [x0,y0,x1,y1]
FACE_BAND = tuple(_OP["face_band"])
_SCREEN_SEND_W = _OP["screen_send_w"]


def detect_screen(img, cfg, *, session=None):
    """VLM-ground the wire-mesh SCREEN being installed (most visible during the
    install, before it blends into the face). Returns a fractional bbox clamped to
    the face band. APPROXIMATE - a bolted mesh cannot be localised precisely."""
    h, w = img.shape[:2]
    rh = int(h * _SCREEN_SEND_W / w)
    c = cv2.resize(img, (_SCREEN_SEND_W, rh))
    ok, buf = cv2.imencode(".jpg", c, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    b64 = base64.b64encode(buf.tobytes()).decode()
    prompt = (f"Underground mine face, image is {_SCREEN_SEND_W}x{rh} px. A worker/boom is "
              "installing a WIRE-MESH SCREEN panel on the rock face. Give the pixel bounding box "
              "of the screen panel being installed/handled right now. JSON only: "
              '{"screen_visible":bool,"screen_bbox_px":[x0,y0,x1,y1]}')
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}]
    sess = session or requests
    r = sess.post(f"{cfg['endpoint']}/chat/completions",
                  json={"model": cfg["model"], "messages": msgs, "max_tokens": 120,
                        "temperature": 0.0}, timeout=120).json()
    txt = r["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", txt, re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        d = {}
    bb = d.get("screen_bbox_px")
    if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
        return {"screen_visible": False, "screen_bbox": None}
    frac = [bb[0] / _SCREEN_SEND_W, bb[1] / rh, bb[2] / _SCREEN_SEND_W, bb[3] / rh]
    fx0, fy0, fx1, fy1 = FACE_BAND
    cl = [min(max(frac[0], fx0), fx1), min(max(frac[1], fy0), fy1),
          min(max(frac[2], fx0), fx1), min(max(frac[3], fy0), fy1)]
    if cl[2] - cl[0] < 0.03 or cl[3] - cl[1] < 0.03:   # degenerate
        return {"screen_visible": False, "screen_bbox": None}
    return {"screen_visible": bool(d.get("screen_visible", True)),
            "screen_bbox": [round(x, 3) for x in cl]}


def classify_sessions(events, gap=_OP["session_gap"], motion_thresh=MOTION_FRAC_THRESH):
    """Group operator-in-front detections into reload SESSIONS and judge each by the
    boom state AT ENTRY. The operator MUST enter the zone to reload meshes/bolts -
    that is normal. It is non-compliant ONLY if the boom was still operating when he
    entered (drilling not stopped first). One verdict per session, not per frame.
    """
    evs = sorted([e for e in events if e.get("person_bbox")], key=lambda x: x["cycle_sec"])
    sessions, cur = [], []
    for e in evs:
        if cur and e["cycle_sec"] - cur[-1]["cycle_sec"] > gap:
            sessions.append(cur); cur = []
        cur.append(e)
    if cur:
        sessions.append(cur)
    out = []
    for s in sessions:
        entry = s[0]
        entry_moving = entry.get("arm_motion", 0.0) > motion_thresh
        out.append({"start": s[0]["cycle_sec"], "end": s[-1]["cycle_sec"],
                    "entry_motion": entry.get("arm_motion", 0.0),
                    "entry_boom_moving": entry_moving,
                    "verdict": "NON_COMPLIANT_ENTRY" if entry_moving else "SAFE_RELOAD",
                    "action": entry.get("action", ""), "n_frames": len(s),
                    "person_bbox": entry.get("person_bbox")})
    return out


def classify(person_in_front: bool, arm_motion_value: float,
             motion_thresh: float = MOTION_FRAC_THRESH) -> str:
    """Fail-safe operator-zone verdict (arm_motion_value should be the
    person-masked motion)."""
    moving = arm_motion_value > motion_thresh
    if person_in_front and moving:
        return "DANGER"          # operator in front while the boom is moving
    if person_in_front and not moving:
        return "OK_LOADING"      # operator in front, drilling stopped (compliant)
    return "NO_PERSON"
