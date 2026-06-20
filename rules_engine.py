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


def decide_traced(table: dict, facts: dict):
    """Like decide() but also returns which rule fired: (verdict, rule_index), where
    rule_index is the matched 0-based row or -1 for the fail-safe default. For audit
    provenance — recording which rule produced a verdict."""
    for i, row in enumerate(table["rules"]):
        if all(facts.get(k) == v for k, v in row["when"].items()):
            return row["verdict"], i
    return table["default"], -1
