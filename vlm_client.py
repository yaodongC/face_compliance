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


# ===========================================================================
# Compliance-milestone perception: final SUPPORT confirmation + episode class
# These are the VLM "tools" the milestone harness calls. The VLM only reports
# what it sees; the latched compliance decision is made in compliance_milestone.
# ===========================================================================

_CONFIRM_PROMPT = PROMPTS.get("confirm",
    "You are a SAFETY INSPECTOR. These are HIGH-RESOLUTION crops of the END FACE of an "
    "underground mine heading (overview, then left/centre/right thirds) at the end of the "
    "ground-support cycle. IGNORE the arched back and side walls (always meshed). Judge ONLY "
    "the flat END FACE straight ahead. Confirm whether FACE SUPPORT IS COMPLETE: the WHOLE end "
    "face is covered with wire-mesh SCREEN, it has ROCK-BOLT PLATES (round/square steel plates "
    "~15 cm) across it, and the drill booms are PARKED/stopped (not drilling). Report ONLY what "
    "is clearly visible; if any part of the end face shows bare rock, or you cannot clearly see "
    "screen+plates, report false (fail-safe). JSON only: "
    '{"face_screened":bool,"plates_visible":bool,"booms_parked":bool,'
    '"all_regions_covered":bool,"confidence":0.0,"note":"<short>"}')

_EPISODE_PROMPT = PROMPTS.get("episode",
    "Underground mine, front camera on a drill jumbo at the rock face. During this moment the "
    "machine is physically running (IMU-confirmed). Classify the DOMINANT activity at the END "
    "FACE. JSON only: "
    '{"activity":"bolt_install|screen_load|production_drill|mucking|tramming|scaling|other_clear",'
    '"at_face":bool,"plate_visible":bool,"region":"L|C|R","note":"<short>"}')

_CONFIRM_DEFAULT = {"face_screened": False, "plates_visible": False, "booms_parked": False,
                    "all_regions_covered": False, "confidence": 0.0, "note": "", "scene": ""}
_EPISODE_DEFAULT = {"activity": "other_clear", "at_face": False, "plate_visible": False,
                    "region": "C", "note": ""}


def _region_crops(bgr, box):
    """Overview face crop + left/centre/right thirds, all at native resolution (the
    resolution-is-a-safety-parameter lever — fine mesh/plates need detail)."""
    face = crop_region(bgr, box)
    h, w = face.shape[:2]
    thirds = [face[:, 0:int(w * 0.40)], face[:, int(w * 0.30):int(w * 0.70)],
              face[:, int(w * 0.60):w]]
    return [face] + [t for t in thirds if t.size]


def _parse_json(text, default) -> dict:
    out = dict(default)
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
    depth, obj = 0, None
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
    for k, v in default.items():
        if k not in obj:
            continue
        out[k] = _coerce_bool(obj[k]) if isinstance(v, bool) else (
            float(obj[k]) if isinstance(v, float) and not isinstance(obj[k], str) else
            (obj[k] if not isinstance(v, str) else str(obj[k])))
    return out


def _vlm_json(images_bgr, cfg, prompt, default, *, session=None, temperature=None) -> dict:
    frames_b64 = [encode_frame(im, cfg["frame_max_width"]) for im in images_bgr]
    content = [{"type": "text", "text": "Inspect and return JSON only."}]
    for b64 in frames_b64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    msgs = [{"role": "system", "content": prompt}, {"role": "user", "content": content}]
    temp = cfg.get("temperature", 0.0) if temperature is None else temperature
    payload = {"model": cfg["model"], "messages": msgs,
               "max_tokens": cfg.get("max_tokens", 420), "temperature": temp}
    sess = session or requests
    resp = sess.post(f"{cfg['endpoint']}/chat/completions", json=payload, timeout=180)
    resp.raise_for_status()
    return _parse_json(resp.json()["choices"][0]["message"]["content"], default)


def confirm_supported(frames, cfg, *, session=None) -> dict:
    """FINAL compliance confirmation at the candidate moment. Sends hi-res face region
    crops; fail-safe consensus = supported ONLY if EVERY vote confirms screen+plates+parked
    (asymmetric, like the rest of the harness). Returns fused dict + 'supported' bool."""
    box = cfg.get("face_crop")
    k = max(1, int(cfg.get("confirm_votes", cfg.get("votes", 1))))
    temp = cfg.get("vote_temperature", 0.6) if k > 1 else cfg.get("temperature", 0.0)
    # one representative frame -> region crops (overview + thirds); <=8 images/prompt server cap.
    # consensus comes from K votes on these crops, not from stacking many frames' crops.
    fl = list(frames or [])
    if not fl:
        return {**_CONFIRM_DEFAULT, "supported": False}
    images = _region_crops(fl[len(fl) // 2], box)[:8]
    votes = [_vlm_json(images, cfg, _CONFIRM_PROMPT, _CONFIRM_DEFAULT,
                       session=session, temperature=(temp if k > 1 else None)) for _ in range(k)]
    fused = {
        "face_screened": all(v["face_screened"] for v in votes),
        "plates_visible": all(v["plates_visible"] for v in votes),
        "booms_parked": all(v["booms_parked"] for v in votes),
        "all_regions_covered": all(v["all_regions_covered"] for v in votes),
        "confidence": round(min(v["confidence"] for v in votes), 3),
        "note": votes[0]["note"], "votes": k,
    }
    fused["supported"] = (fused["face_screened"] and fused["plates_visible"]
                          and fused["booms_parked"])
    return fused


def classify_episode(frames, cfg, *, session=None) -> dict:
    """Classify what the machine is doing during an IMU work-window (bolt vs screen-load
    vs production drill vs mucking...). Used to VETO non-bolting episodes (keeps the bolt
    count honest). Majority over the window's frames."""
    box = cfg.get("face_crop")
    results = []
    for f in (frames or []):
        results.append(_vlm_json(_region_crops(f, box), cfg, _EPISODE_PROMPT, _EPISODE_DEFAULT,
                                 session=session))
    if not results:
        return dict(_EPISODE_DEFAULT)
    from collections import Counter
    act = Counter(r["activity"] for r in results).most_common(1)[0][0]
    reg = Counter(r["region"] for r in results).most_common(1)[0][0]
    return {"activity": act, "region": reg,
            "at_face": sum(r["at_face"] for r in results) >= max(1, len(results) // 2),
            "plate_visible": any(r["plate_visible"] for r in results),
            "note": results[0]["note"]}
