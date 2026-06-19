"""Externalized verdict RULES, loaded from rules/<task>.yaml.

The condition->verdict decision tables are config/data (editable, versionable,
reviewable). The engine (rules_engine.decide) and the feature computation stay in
code. Each table is validated to have a fail-safe `default` at load, so a malformed
rules bundle fails LOUDLY at startup rather than emitting a wrong/blank verdict.
"""
from __future__ import annotations
from pathlib import Path
import yaml

_DIR = Path(__file__).resolve().parent / "rules"


def load(task: str = "face_support") -> dict:
    p = _DIR / f"{task}.yaml"
    if not p.exists():
        raise RuntimeError(f"SAFETY: rules bundle not found: {p} — cannot decide verdicts")
    data = yaml.safe_load(p.read_text()) or {}
    for name, table in data.items():
        if not (isinstance(table, dict) and isinstance(table.get("rules"), list) and "default" in table):
            raise RuntimeError(f"SAFETY: rules table '{name}' must have a 'rules' list and a fail-safe 'default'")
    return data


RULES = load()
