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

## Notes
- Compliance is observational, not measured: distance/pattern items are
  qualitative judgements from a single RGB camera, not metrology.
- Bags 15 and 33 are truncated; the extractor skips damaged bags.
- Tests: `python3 -m pytest -v`
