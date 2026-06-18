"""OpenAI-compatible client for the vLLM-served Cosmos-Reason2 VLM."""
from __future__ import annotations
import base64
import json
import cv2
import numpy as np
import requests

_SCHEMA = (
    '{"narration": "<one or two sentences>", '
    '"current_activity": "<item id like p5, or other>", '
    '"observations": [{"item_id": "p#", "status": "in_progress|satisfied|violation", "evidence": "<what you see>"}], '
    '"safety_flags": [{"id": "S#", "severity": "low|med|high", "note": "<what you see>"}], '
    '"confidence": 0.0}'
)


def build_system_prompt(items) -> str:
    process = [i for i in items if i.kind == "process"]
    safety = [i for i in items if i.kind == "safety"]
    lines = ["You are a mining ground-control compliance observer. You watch a short burst of",
             "consecutive frames from the FRONT camera of a vehicle at an underground hard-rock",
             "mining face where a crew installs face support (wire screen/mesh and rock bolts)",
             "following the Vale Face Support Guidelines.",
             "",
             "Do two things: (1) narrate concisely what the crew/equipment is doing now;",
             "(2) report only the checklist items you can actually see evidence for.",
             "",
             "PROCESS items (the bolting sequence):"]
    for i in process:
        lines.append(f"  {i.id}: {i.label} -- {i.evidence}")
    lines.append("")
    lines.append("SAFETY items (report as safety_flags if you see a problem):")
    for i in safety:
        lines.append(f"  {i.id}: {i.label} -- {i.evidence}")
    lines += ["",
              "You are looking at ONE RGB camera: judge qualitatively, never invent measured",
              "distances or counts. Only report items with visible evidence. For process item p8,",
              "a drill drilling an unscreened/unbolted face is a 'violation'.",
              "",
              "In each observation's 'evidence', describe what you ACTUALLY SEE in these frames",
              "(e.g. 'a bolter boom pressed against the mesh on the left wall') -- do NOT copy the",
              "checklist wording above. If you cannot see clear evidence for an item, omit it",
              "rather than guessing. Keep 'narration' to one or two sentences about what is",
              "happening right now in the frames.",
              "",
              "Respond with ONLY a single JSON object, no prose, in exactly this shape:",
              _SCHEMA]
    return "\n".join(lines)


def encode_frame(bgr, max_width: int) -> str:
    h, w = bgr.shape[:2]
    if w > max_width:
        scale = max_width / float(w)
        bgr = cv2.resize(bgr, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise ValueError("jpeg encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def build_messages(frames_b64, system_prompt: str) -> list[dict]:
    content = [{"type": "text",
                "text": "Here are consecutive frames (oldest first). Analyze and return JSON."}]
    for b64 in frames_b64:
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    return [{"role": "system", "content": system_prompt},
            {"role": "user", "content": content}]


def parse_response(text: str) -> dict:
    default = {"narration": "", "current_activity": "other",
               "observations": [], "safety_flags": [], "confidence": 0.0}
    if not text:
        return default
    s = text.strip()
    if "```" in s:                       # strip code fences
        parts = s.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("{"):
                s = p
                break
    start = s.find("{")
    if start == -1:
        return default
    depth = 0
    for i in range(start, len(s)):       # find matching closing brace
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(s[start:i + 1])
                except json.JSONDecodeError:
                    return default
                out = dict(default)
                out.update({k: obj.get(k, default[k]) for k in default})
                if not isinstance(out["observations"], list):
                    out["observations"] = []
                if not isinstance(out["safety_flags"], list):
                    out["safety_flags"] = []
                return out
    return default


def analyze_window(frames, cfg, *, session=None) -> dict:
    from compliance import load_regulation
    items = load_regulation(cfg["paths"]["regulation"])
    sp = build_system_prompt(items)
    frames_b64 = [encode_frame(f, cfg["frame_max_width"]) for f in frames]
    payload = {"model": cfg["model"],
               "messages": build_messages(frames_b64, sp),
               "max_tokens": cfg["max_tokens"],
               "temperature": cfg["temperature"]}
    sess = session or requests
    resp = sess.post(f"{cfg['endpoint']}/chat/completions", json=payload, timeout=120)
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"]
    return parse_response(text)
