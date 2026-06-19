"""PHASE-0 EQUIVALENCE LOCK.

The deterministic rule/aggregation layer must reproduce the golden outputs captured
from the known-good harness, exactly. This is the GATE for every behaviour-preserving
refactor (centralise config, externalise prompts/rules, generalise the engine): if a
refactor changes a session verdict, a mesh count, or a coverage state, this fails.

Regenerate golden ONLY with an explicit, reviewed behaviour change:
    python3 -c "<the generator in git history>"   # never silently
"""
import json
from pathlib import Path
from operator_safety import classify_sessions
from coverage import (mesh_installs, mesh_count, width_coverage, install_intervals,
                      segment_coverage, segment_state, coverage_state)

HERE = Path(__file__).resolve().parent
OPS = json.loads((HERE / "fixtures" / "operator_events.json").read_text())["events"]
GOLD = json.loads((HERE / "golden" / "rules.json").read_text())
TS = [600, 1300, 2000, 2600, 3400]
_FULL = [{"bbox": [0.20, 0.2, 0.55, 0.8], "installed_at": 1},
         {"bbox": [0.50, 0.2, 0.85, 0.8], "installed_at": 2}]
_PART = [{"bbox": [0.30, 0.2, 0.44, 0.8], "installed_at": 1}]


def _norm(x):
    return json.loads(json.dumps(x))   # tuples -> lists, stable compare


def test_sessions_equivalence():
    assert _norm(classify_sessions(OPS)) == GOLD["sessions"]


def test_mesh_installs_equivalence():
    assert _norm(mesh_installs(OPS)) == GOLD["mesh_installs"]


def test_mesh_count_equivalence():
    assert _norm({str(t): mesh_count(OPS, t) for t in TS}) == GOLD["mesh_count"]


def test_width_coverage_equivalence():
    assert _norm({str(t): width_coverage(OPS, t) for t in TS}) == GOLD["width_coverage"]


def test_install_intervals_equivalence():
    assert _norm({str(t): install_intervals(OPS, t) for t in TS}) == GOLD["install_intervals"]


def test_segment_equivalence():
    assert _norm(segment_coverage(OPS)) == GOLD["segment_coverage"]
    assert _norm(segment_state(segment_coverage(OPS), 99999)) == GOLD["segment_state_final"]


def test_coverage_state_equivalence():
    assert _norm(coverage_state(_FULL, 10)) == GOLD["coverage_state_full"]
    assert _norm(coverage_state(_PART, 10)) == GOLD["coverage_state_part"]
