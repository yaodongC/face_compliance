# Face Support Compliance VLM Demo

Watches the front RGB camera from underground mining rosbags with the
Cosmos-Reason2-2B VLM (served by vLLM) and shows a running narration plus a
progressively-filling face-support compliance checklist (Vale CMTS-2015-001).

## Pipeline
1. **Extract** front camera to MP4:
   `python3 extract_video.py --bags ../_2026-06-11-11-55-36_0.bag ... --duration 220`
2. **Serve** the model: see `../install.md`.
3. **Analyze**: `python3 analyze.py`  (or `--stub` to run without the model)
4. **Replay GUI**: `python3 gui.py`

## Compliance logic
The checklist fills in progressively as the bolting process advances. Violations
are **debounced** (must persist `confirm_violation` windows before locking),
**severity-gated** (safety flags below `min_severity` are advisories), and
**clearable** (a locked violation resolves after `confirm_clear` compliant windows,
since bolting is ongoing and a flagged item can become compliant again). All knobs
live in `config.yaml`; setting them to 1 / `low` reproduces immediate-and-sticky
behavior. The overall verdict (`IN PROGRESS` / `AT-RISK` / `COMPLIANT`) can return
from `AT-RISK` to `COMPLIANT` as violations clear.

## Analysis timelines
`gui.py` replays whatever is at `data/analysis.json`. Three are kept:
- `data/analysis_stub.json` — canned sequence; demonstrates the full lifecycle
  (`IN PROGRESS → AT-RISK → IN PROGRESS → COMPLIANT`). This is the default.
- `data/analysis_real.json` — the real Cosmos-Reason2-2B run over the clip.
- `data/analysis.json` — whatever was generated last (currently the stub).

Swap which one the GUI shows:
```bash
cp data/analysis_stub.json data/analysis.json   # clean lifecycle demo (default)
cp data/analysis_real.json data/analysis.json    # genuine VLM output
```

## Notes
- Compliance is observational, not measured: distance/pattern items are
  qualitative judgements from a single RGB camera, not metrology.
- On this footage the real model is safety-conservative: it repeatedly flags p1
  (scaling) and sometimes p8 (drilling) as violations, so the real timeline stays
  largely `AT-RISK`. That is the model's genuine read, not a logic error — the
  stub shows the intended progressive UX more cleanly.
- Bags 15 and 33 are truncated; the extractor skips damaged bags.
- Tests: `python3 -m pytest -v`  (26 tests)
