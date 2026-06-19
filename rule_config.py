"""Externalized verdict RULES, loaded from the active task bundle (tasks/<task>/rules.yaml).

The condition->verdict decision tables are config/data (editable, versionable,
reviewable). The engine (rules_engine.decide) and the feature computation stay in
code. Each table is validated to have a fail-safe `default` at load, so a malformed
rules bundle fails LOUDLY at startup rather than emitting a wrong/blank verdict.
"""
from __future__ import annotations
import yaml
from task import task_dir


def load(task: str | None = None) -> dict:
    p = task_dir(task) / "rules.yaml"
    if not p.exists():
        raise RuntimeError(f"SAFETY: rules bundle not found: {p} — cannot decide verdicts")
    data = yaml.safe_load(p.read_text()) or {}
    for name, table in data.items():
        if not (isinstance(table, dict) and isinstance(table.get("rules"), list) and "default" in table):
            raise RuntimeError(f"SAFETY: rules table '{name}' must have a 'rules' list and a fail-safe 'default'")
        if table["default"] is None:
            raise RuntimeError(f"SAFETY: rules table '{name}' default is null — must be an explicit verdict")
        for i, rule in enumerate(table["rules"]):
            if not isinstance(rule.get("when"), dict):
                raise RuntimeError(f"SAFETY: rules table '{name}' rule[{i}] needs a 'when' mapping")
            if "verdict" not in rule:
                raise RuntimeError(f"SAFETY: rules table '{name}' rule[{i}] needs a 'verdict'")
    return data


RULES = load()
