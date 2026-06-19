"""Automated END-TO-END behaviour gate: the rendered full-cycle GUI + event log must
hash to the committed baseline. Skipped if the (gitignored) data artifacts aren't
present, so a clean checkout still runs the unit suite. Run the offline pipeline
(build_event_log + render_gui) first to refresh the artifacts."""
import json
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2e_signature import signature

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
BASE = json.loads((HERE / "golden" / "e2e_signature.json").read_text())

_have = (DATA / "full_cycle_gui.mp4").exists() and (DATA / "event_log.jsonl").exists()


@pytest.mark.skipif(not _have, reason="full-cycle data artifacts not present (gitignored)")
def test_e2e_full_cycle_unchanged():
    sig = signature(str(DATA / "event_log.jsonl"), str(DATA / "full_cycle_gui.mp4"))
    assert sig == BASE, "end-to-end output changed vs golden baseline"
