"""Active inspection-TASK selector.

A task BUNDLE is a directory tasks/<name>/ holding the declarative definition of one
inspection task: params.yaml (thresholds/ROIs), prompts.yaml (VLM prompts), and
rules.yaml (verdict decision tables). The same engine runs ANY task by swapping the
bundle. The active task is config.yaml `task:` (or env HARNESS_TASK), default
face_support. Feature-extractor CODE is still per-domain; the bundle is the config.
"""
from __future__ import annotations
import os
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent


def active_task() -> str:
    name = os.environ.get("HARNESS_TASK")
    if name:
        return name
    p = ROOT / "config.yaml"
    cfg = yaml.safe_load(p.read_text()) if p.exists() else {}
    return (cfg or {}).get("task", "face_support")


def task_dir(name: str | None = None) -> Path:
    return ROOT / "tasks" / (name or active_task())
