"""OpenAI-compatible client for the local vLLM-served VLM (offline at inference).

SAFETY-CRITICAL, FACE-FOCUSED PERCEPTION. Learnings that shaped this:
  * The arched BACK/walls are mesh+bolted in EVERY frame, so "is there mesh" does
    not discriminate compliance. The reliable, safety-meaningful signal is the
    state of the FLAT END FACE and the drill: a face being drilled is active work
    (not the supported rest-state); a fully screened face with the booms parked is
    the compliant, supported end-state.
  * Fine detail (mesh/bolts) is invisible at low resolution, so we send a
    high-resolution CROP of the face/centre region (see crop_region + config).
  * The model is asked only for grounded PERCEPTION; the compliance decision is
    made in code with conservative, asymmetric K-vote aggregation.
On any parse failure the perception defaults to "not screened" (the fail-safe
direction — never invent support).
"""
from __future__ import annotations
import base64
import json
import cv2
import numpy as np
import requests
from prompt_config import PROMPTS

# Conservative defaults: never assume the face is supported.
SAFE_DEFAULT = {
    "scene": "",
    "face_screened": False,   # END FACE covered with mesh AND bolt plates
    "drill_active": False,    # drill rods pushed into the face (active drilling)
    "arms_parked": False,     # booms folded to the sides, not working the face
    "person_in_danger": False,
    "note": "",
}

# externalized to prompts/face_support.yaml (config, not code)
SYSTEM_PROMPT = PROMPTS["system"]


def build_system_prompt(items=None) -> str:
    return SYSTEM_PROMPT


def crop_region(bgr, box):
    """Crop to a [x0,y0,x1,y1] fractional box (the face/centre region). Focusing
    the model on the end face at high resolution is what lets it resolve screen,
    bolt plates and drill state; full-frame views confuse it."""
    if not box:
        return bgr
    h, w = bgr.shape[:2]
    x0, y0, x1, y1 = box
    crop = bgr[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
    return crop if crop.size else bgr


def encode_frame(bgr, max_width: int) -> str:
    h, w = bgr.shape[:2]
    if w > max_width:
        scale = max_width / float(w)
        bgr = cv2.resize(bgr, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise ValueError("jpeg encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def build_messages(frames_b64, system_prompt: str) -> list[dict]:
    content = [{"type": "text",
                "text": "Inspect the end face and drill state. Return JSON only."}]
    for b64 in frames_b64:
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    return [{"role": "system", "content": system_prompt},
            {"role": "user", "content": content}]


def _coerce_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "1")
    return bool(v)


def parse_response(text: str) -> dict:
    """Extract the perception JSON. ALWAYS returns every SAFE_DEFAULT key; on any
    failure returns the conservative default (face not screened)."""
    out = dict(SAFE_DEFAULT)
    if not text:
        return out
    s = text.strip()
    if "```" in s:
        for part in s.split("```"):
            p = part.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("{"):
                s = p
                break
    start = s.find("{")
    if start == -1:
        return out
    depth = 0
    obj = None
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(s[start:i + 1])
                except json.JSONDecodeError:
                    return out
                break
    if not isinstance(obj, dict):
        return out
    out["scene"] = str(obj.get("scene", "") or "")
    out["note"] = str(obj.get("note", "") or "")
    out["face_screened"] = _coerce_bool(obj.get("face_screened", False))
    out["drill_active"] = _coerce_bool(obj.get("drill_active", False))
    out["arms_parked"] = _coerce_bool(obj.get("arms_parked", False))
    out["person_in_danger"] = _coerce_bool(obj.get("person_in_danger", False))
    return out


def fuse_perceptions(perceptions: list[dict]) -> dict:
    """Asymmetric K-vote fusion. SUPPORT signals are UNANIMOUS (face_screened and
    arms_parked require every vote), while HAZARD/ACTIVITY signals fire if ANY vote
    saw them (drill_active, person_in_danger). Hard to earn 'supported', easy to
    flag work/hazard."""
    ps = [p for p in perceptions if p]
    if not ps:
        return dict(SAFE_DEFAULT)
    out = dict(SAFE_DEFAULT)
    out["scene"] = ps[0].get("scene", "")
    out["note"] = ps[0].get("note", "")
    out["face_screened"] = all(bool(p.get("face_screened")) for p in ps)
    out["arms_parked"] = all(bool(p.get("arms_parked")) for p in ps)
    out["drill_active"] = any(bool(p.get("drill_active")) for p in ps)
    out["person_in_danger"] = any(bool(p.get("person_in_danger")) for p in ps)
    out["votes"] = len(ps)
    return out


def analyze_window(frames, cfg, *, session=None, temperature=None) -> dict:
    box = cfg.get("face_crop")
    frames_b64 = [encode_frame(crop_region(f, box), cfg["frame_max_width"]) for f in frames]
    temp = cfg.get("temperature", 0.0) if temperature is None else temperature
    payload = {"model": cfg["model"],
               "messages": build_messages(frames_b64, SYSTEM_PROMPT),
               "max_tokens": cfg["max_tokens"],
               "temperature": temp}
    sess = session or requests
    resp = sess.post(f"{cfg['endpoint']}/chat/completions", json=payload, timeout=180)
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"]
    return parse_response(text)


def analyze_window_consensus(frames, cfg, *, session=None) -> dict:
    """Run K independent queries (sampled at vote_temperature) and fuse them
    asymmetrically. K=1 falls back to a single deterministic query."""
    k = max(1, int(cfg.get("votes", 1)))
    if k == 1:
        return analyze_window(frames, cfg, session=session)
    temp = cfg.get("vote_temperature", 0.6)
    votes = [analyze_window(frames, cfg, session=session, temperature=temp) for _ in range(k)]
    return fuse_perceptions(votes)
