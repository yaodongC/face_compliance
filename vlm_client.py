"""OpenAI-compatible client for the vLLM-served Cosmos-Reason2 VLM.

SAFETY-CRITICAL DESIGN: a small (2B) VLM hallucinates ground support that is not
there. This module therefore asks the model only for GROUNDED PERCEPTION (what is
literally visible), with a strict prompt that forces a conservative, conjunctive
reading and defaults to UNSUPPORTED when uncertain. The compliance decision itself
is made in code (compliance.py), never trusted to a single model boolean. On any
parse failure the perception defaults to "cannot verify / nothing supported" — the
fail-safe direction.
"""
from __future__ import annotations
import base64
import json
import cv2
import numpy as np
import requests

# Conservative defaults: absence of evidence is NEVER treated as support.
SAFE_DEFAULT = {
    "scene": "",
    "activity": "none",
    "people_visible": False,
    "person_in_danger": False,
    "mesh_visible": False,
    "bolts_visible": False,
    "ground_support_state": "cannot_tell",
    "safety_call": "CANNOT_VERIFY",
    "note": "",
}

_ACTIVITIES = {"none", "scaling", "screening", "bolting", "drilling", "other"}
_SUPPORT_STATES = {"none_visible", "partial", "full", "cannot_tell"}
_SAFETY_CALLS = {"UNSUPPORTED", "PARTIAL", "SUPPORTED", "CANNOT_VERIFY"}

# Conservative rank: lower = less support assumed. Consensus takes the MIN so a
# positive (SUPPORTED / full) reading requires every vote to agree.
_GSS_RANK = {"none_visible": 0, "cannot_tell": 1, "partial": 2, "full": 3}
_CALL_RANK = {"UNSUPPORTED": 0, "CANNOT_VERIFY": 1, "PARTIAL": 2, "SUPPORTED": 3}
# Activity priority for fusion: the most hazardous activity any vote saw wins.
_ACT_PRIORITY = ["drilling", "bolting", "screening", "scaling", "other", "none"]

SYSTEM_PROMPT = """You are a SAFETY INSPECTOR in an underground hard-rock mine, viewing the front camera pointed at a development face (the rock wall at the end of a tunnel). Miners are CRUSHED TO DEATH by unsupported rock. Reporting ground support that is not actually there KILLS people. When you are unsure, you MUST report UNSUPPORTED. Your job is NOT to certify that the face is safe; it is to report only what is unmistakably visible and to default to "unsupported / not verified".

HARD RULES:
1. Report ONLY what is directly, unmistakably visible in THIS frame. Never assume, infer, predict, or imagine activity, people, or equipment you cannot point to.
2. EXPECT a bare rock face: brown/grey rock, sometimes with painted survey lines (yellow/red) and a pile of broken rock (muck) on the floor. That bare face is the normal UNSUPPORTED state and is what you should report unless you clearly see support installed.
3. Wire mesh / screen is a FLAT, REGULAR metal GRID of small squares lying tight against the rock, covering it like a net. Loose HOSES, CABLES, WIRES, a drill boom, or rock texture are NOT mesh. If you do not clearly see a tight regular grid of squares ON the rock, mesh_visible = false.
4. Rock bolts are bright/shiny round steel PLATES or washers (~15 cm) pressed flat against the rock in a clear repeating grid. Rough rock, shadows, wet patches, survey paint, bolt-heads of equipment, and dark spots are NOT rock bolts. If you do not clearly see a pattern of shiny round plates ON the rock face, bolts_visible = false. On a bare development face there are usually NO bolts.
5. CONSERVATIVE CONJUNCTION: a face is only "supported" if you clearly see BOTH mesh AND bolts. If mesh_visible is false OR bolts_visible is false, then ground_support_state must be "none_visible" and safety_call must be "UNSUPPORTED".
6. A parked machine, a boom held near the face, hoses, or cables are NOT work. Report activity="drilling" ONLY if you can see a steel drill rod actively pushed into the rock; report "bolting" ONLY if a tool is setting a bolt. If nothing is clearly moving against the face, activity = "none".
7. person_in_danger = true only if you can see a person standing/working directly under bare, unsupported rock.
8. When in ANY doubt: mesh_visible=false, bolts_visible=false, ground_support_state="none_visible", safety_call="UNSUPPORTED".

Respond with ONLY this JSON object, no prose:
{"scene":"<one sentence: literally what is visible>","activity":"none|scaling|screening|bolting|drilling|other","people_visible":true|false,"person_in_danger":true|false,"mesh_visible":true|false,"bolts_visible":true|false,"ground_support_state":"none_visible|partial|full|cannot_tell","safety_call":"UNSUPPORTED|PARTIAL|SUPPORTED|CANNOT_VERIFY","note":"<plain-language caveat about what you could not verify>"}"""


def build_system_prompt(items=None) -> str:
    """Return the safety-inspector system prompt. `items` is accepted for
    backward compatibility but is unused: the perception schema is fixed."""
    return SYSTEM_PROMPT


def encode_frame(bgr, max_width: int) -> str:
    h, w = bgr.shape[:2]
    if w > max_width:
        scale = max_width / float(w)
        bgr = cv2.resize(bgr, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        raise ValueError("jpeg encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def build_messages(frames_b64, system_prompt: str) -> list[dict]:
    content = [{"type": "text",
                "text": ("These are consecutive frames from the face camera (oldest "
                         "first). Inspect the most recent state and return JSON only.")}]
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
    """Extract the perception JSON. ALWAYS returns every SAFE_DEFAULT key. On any
    failure or ambiguity it returns the conservative default (nothing supported,
    CANNOT_VERIFY) — never a 'supported' reading invented from a bad parse."""
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
    act = str(obj.get("activity", "none") or "none").lower()
    out["activity"] = act if act in _ACTIVITIES else "other"
    out["people_visible"] = _coerce_bool(obj.get("people_visible", False))
    out["person_in_danger"] = _coerce_bool(obj.get("person_in_danger", False))
    out["mesh_visible"] = _coerce_bool(obj.get("mesh_visible", False))
    out["bolts_visible"] = _coerce_bool(obj.get("bolts_visible", False))
    gss = str(obj.get("ground_support_state", "cannot_tell") or "cannot_tell").lower()
    out["ground_support_state"] = gss if gss in _SUPPORT_STATES else "cannot_tell"
    sc = str(obj.get("safety_call", "CANNOT_VERIFY") or "CANNOT_VERIFY").upper()
    out["safety_call"] = sc if sc in _SAFETY_CALLS else "CANNOT_VERIFY"
    # Defense in depth: enforce the conjunction rule even if the model violated it.
    # "Supported" requires BOTH mesh and bolts; otherwise downgrade to the safe call.
    if not (out["mesh_visible"] and out["bolts_visible"]):
        if out["safety_call"] in ("SUPPORTED", "PARTIAL"):
            out["safety_call"] = "UNSUPPORTED"
        if out["ground_support_state"] in ("full", "partial"):
            out["ground_support_state"] = "none_visible"
    return out


def analyze_window(frames, cfg, *, session=None, temperature=None) -> dict:
    frames_b64 = [encode_frame(f, cfg["frame_max_width"]) for f in frames]
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


def fuse_perceptions(perceptions: list[dict]) -> dict:
    """Asymmetric K-vote fusion. SUPPORT signals are taken UNANIMOUSLY (mesh/bolts
    require every vote; ground_support_state / safety_call take the most
    conservative rank), while HAZARD signals fire if ANY vote saw them. This makes
    a positive 'supported' reading hard to earn and a hazard easy to raise."""
    ps = [p for p in perceptions if p]
    if not ps:
        return dict(SAFE_DEFAULT)
    out = dict(SAFE_DEFAULT)
    out["scene"] = ps[0].get("scene", "")
    out["note"] = ps[0].get("note", "")
    # support: unanimous
    out["mesh_visible"] = all(bool(p.get("mesh_visible")) for p in ps)
    out["bolts_visible"] = all(bool(p.get("bolts_visible")) for p in ps)
    out["ground_support_state"] = min(
        (p.get("ground_support_state", "cannot_tell") for p in ps),
        key=lambda g: _GSS_RANK.get(g, 1))
    out["safety_call"] = min(
        (p.get("safety_call", "CANNOT_VERIFY") for p in ps),
        key=lambda c: _CALL_RANK.get(c, 1))
    # hazards: any vote
    out["people_visible"] = any(bool(p.get("people_visible")) for p in ps)
    out["person_in_danger"] = any(bool(p.get("person_in_danger")) for p in ps)
    acts = {p.get("activity", "none") for p in ps}
    out["activity"] = next((a for a in _ACT_PRIORITY if a in acts), "none")
    # re-apply the conjunction defense to the fused result
    if not (out["mesh_visible"] and out["bolts_visible"]):
        if out["safety_call"] in ("SUPPORTED", "PARTIAL"):
            out["safety_call"] = "UNSUPPORTED"
        if out["ground_support_state"] in ("full", "partial"):
            out["ground_support_state"] = "none_visible"
    out["votes"] = len(ps)
    return out


def analyze_window_consensus(frames, cfg, *, session=None) -> dict:
    """Run K independent queries (sampled at vote_temperature for diversity) and
    fuse them asymmetrically. K=1 falls back to a single deterministic query."""
    k = max(1, int(cfg.get("votes", 1)))
    if k == 1:
        return analyze_window(frames, cfg, session=session)
    temp = cfg.get("vote_temperature", 0.6)
    votes = [analyze_window(frames, cfg, session=session, temperature=temp) for _ in range(k)]
    return fuse_perceptions(votes)
