"""Out-of-distribution / DOMAIN GUARD — a cheap front gate that asks 'is this even the
inspection scene?' BEFORE the expensive compliance perception.

An out-of-domain frame (an office, a vehicle cab, a surface yard) must ABSTAIN — the
harness emits no compliance verdict on a scene it was not built for, rather than
guessing. This is the runtime answer to SOTIF's 'unknown-unsafe' region.

Opt-in per task via tasks/<task>/domain.yaml ({enabled, question, send_w}); default
disabled, so existing behaviour is unchanged. FAIL-SAFE: if the check errors or is
unparseable, the frame is treated as OUT of domain (abstain), never silently in.
"""
from __future__ import annotations
import base64
import json
import re
import cv2
import requests
import yaml
from task import task_dir


def load_spec(task: str | None = None) -> dict:
    p = task_dir(task) / "domain.yaml"
    return yaml.safe_load(p.read_text()) if p.exists() else {"enabled": False}


SPEC = load_spec()


def in_domain(frame, cfg, *, session=None, spec=None) -> dict:
    """Return {in_domain, checked, reason}. When the guard is disabled, in_domain=True
    (checked=False) and perception proceeds unchanged."""
    spec = SPEC if spec is None else spec
    if not spec.get("enabled"):
        return {"in_domain": True, "checked": False, "reason": "guard disabled"}
    h, w = frame.shape[:2]
    sw = int(spec.get("send_w", 768))
    c = cv2.resize(frame, (sw, int(h * sw / w)))
    ok, buf = cv2.imencode(".jpg", c, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    b64 = base64.b64encode(buf.tobytes()).decode()
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": spec["question"]},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}]
    sess = session or requests
    try:
        r = sess.post(f"{cfg['endpoint']}/chat/completions",
                      json={"model": cfg["model"], "messages": msgs,
                            "max_tokens": 60, "temperature": 0.0}, timeout=60).json()
        txt = r["choices"][0]["message"]["content"]
        m = re.search(r"\{.*\}", txt, re.S)
        d = json.loads(m.group(0)) if m else {}
    except Exception as e:
        return {"in_domain": False, "checked": True, "reason": f"guard error -> abstain: {e}"[:80]}
    # fail-safe: only IN-domain when the model explicitly says so
    return {"in_domain": bool(d.get("in_domain", False)), "checked": True,
            "reason": str(d.get("reason", ""))[:80]}
