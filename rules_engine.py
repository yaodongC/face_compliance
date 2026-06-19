"""Tiny deterministic decision-table engine.

A table is {"rules": [{"when": {field: value, ...}, "verdict": X}, ...],
"default": Y}. The FIRST rule whose `when` fully matches the facts wins; if none
match, the fail-safe `default` is returned. The facts are computed by certified code
(motion, coverage, mesh episodes); only the condition->verdict MAPPING lives in the
editable, auditable rules data. Verdicts never depend on the model directly.
"""
from __future__ import annotations


def decide(table: dict, facts: dict):
    for row in table["rules"]:
        if all(facts.get(k) == v for k, v in row["when"].items()):
            return row["verdict"]
    return table["default"]          # fail-safe (validated present at load)
