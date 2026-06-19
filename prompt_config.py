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


def load(task: str = "face_support") -> dict:
    return yaml.safe_load((_DIR / f"{task}.yaml").read_text())


PROMPTS = load()
