"""Active inspection-TASK selector.

A task BUNDLE is a directory tasks/<name>/ holding the declarative definition of one
inspection task: params.yaml (thresholds/ROIs), prompts.yaml (VLM prompts), and
rules.yaml (verdict decision tables). The same engine runs ANY task by swapping the
bundle. The active task is config.yaml `task:` (or env HARNESS_TASK), default
face_support. Feature-extractor CODE is still per-domain; the bundle is the config.
"""
from __future__ import annotations
import functools
import os
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent


@functools.lru_cache(maxsize=1)
def active_task() -> str:
    """The task in force for the life of the process (cached so config is fixed for
    the run). env HARNESS_TASK wins over config.yaml; an override is logged because it
    is invisible in the config file and must appear in the audit trail."""
    name = os.environ.get("HARNESS_TASK")
    if name:
        print(f"[harness] SAFETY: HARNESS_TASK override active -> task={name}", file=sys.stderr)
        return name
    p = ROOT / "config.yaml"
    cfg = yaml.safe_load(p.read_text()) if p.exists() else {}
    return (cfg or {}).get("task", "face_support")


def task_dir(name: str | None = None) -> Path:
    """Resolve a task bundle dir, failing LOUDLY if it doesn't exist — a typo in
    config.yaml `task:` / $HARNESS_TASK must NOT silently start the default task."""
    d = ROOT / "tasks" / (name or active_task())
    if not d.is_dir():
        raise RuntimeError(f"SAFETY: task bundle not found: {d} — check config.yaml task: / $HARNESS_TASK")
    return d
