"""Externalized prompts, loaded from prompts/<task>.yaml.

Prompts are config/data, not code: they can be edited, reviewed and versioned
without touching the engine, and gated against the golden eval set. `PROMPTS` is the
default task (face_support); the screen prompt is a template whose `<W>`/`<H>` tokens
are substituted with the sent-image size at call time.
"""
from __future__ import annotations
from pathlib import Path
import yaml

_DIR = Path(__file__).resolve().parent / "prompts"
_REQUIRED = {"system", "person", "screen"}


def load(task: str = "face_support") -> dict:
    """Load + validate a prompt bundle. A safety system must NOT run with missing or
    malformed prompts, so fail LOUDLY at startup with a clear message rather than a
    bare FileNotFoundError / KeyError deep in a worker."""
    p = _DIR / f"{task}.yaml"
    if not p.exists():
        raise RuntimeError(f"SAFETY: prompt bundle not found: {p} — cannot start perception")
    try:
        data = yaml.safe_load(p.read_text())
    except yaml.YAMLError as e:
        raise RuntimeError(f"SAFETY: malformed prompt bundle {p}: {e}") from e
    missing = _REQUIRED - set(data or {})
    if missing:
        raise RuntimeError(f"SAFETY: prompt bundle {p} missing required keys: {sorted(missing)}")
    return data


PROMPTS = load()
