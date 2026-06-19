# Face-Support Compliance VLM Harness (safety-critical)

Watches the front camera of a drill jumbo in an underground heading and decides
whether the **face is supported to compliance**, using a local VLM
(Qwen2.5-VL-32B served by vLLM on the Jetson Thor). Fully offline at inference.

## What it decides
- **SUPPORTED** (compliant): the end face is screened + bolted, drilling is
  complete, and the drill booms are parked — the safe rest-state.
- **DRILLING / UNSUPPORTED / DANGER**: active face drilling, an unscreened face,
  or a person under unsupported ground — not the supported state.
- **NOT VERIFIED**: can't confirm — treat as unsafe, human inspection required.

## Design (why it is built this way)
Hard-won lessons, encoded in the harness:
1. **The arched back/walls are mesh-bolted in every frame**, so "is there mesh"
   does NOT indicate compliance. The reliable, safety-meaningful signal is the
   **end FACE + drill state** (screened? drilling? booms parked?).
2. **Resolution is a safety parameter.** Mesh/bolt/drill detail is invisible at
   low resolution and full-frame views confuse the model — so we send a
   **high-res crop of the end-face/centre region** (`face_crop` in config).
3. **Fail-safe aggregation.** SUPPORTED requires the face screened in *every*
   window of a rolling buffer (the reliable signal) + at least one "booms parked"
   sighting (booms that are drilling are never parked) + no sustained drilling.
   Everything defaults to NOT VERIFIED. A single noisy frame can never certify
   support. The result: it is hard to earn SUPPORTED, easy to fall to unsafe.
4. **Grounded perception, decision in code.** The VLM only reports what it sees
   (`face_screened, drill_active, arms_parked, person_in_danger`); compliance is
   decided by `compliance.py`, never by the model's own verdict.

## Validation — meets SUCCESS_CRITERIA.md
`python3 eval.py` scores the harness against hand-labelled ground truth
(`eval/labels.json`). Latest run (Qwen2.5-VL-32B, 7 clips from 7 bags):
- **C1 false-safe = 0** (never certified a drilling/bare face) — the critical one
- **C2 recall = 1.00** (all parked-supported clips → SUPPORTED)
- **C3 negatives 4/4**, **C4 accuracy 1.00**, **C7 28 unit tests pass**
- **C5 offline** (localhost-only inference, weights cached), **C6 deterministic**

## Pipeline
```bash
# 1. serve the model (offline once cached): see ../install.md
# 2. extract footage to MP4:
python3 extract_video.py --bags ../<bag>.bag ... --duration 220
# 3. analyze -> data/analysis.json (real model; or --stub for no server):
python3 analyze.py
# 4. replay GUI (video + scene + fail-safe checklist + verdict banner):
python3 gui.py
# 5. score against ground truth:
python3 eval.py
```

## Safety notes
- ASSISTIVE demo, NOT a certified safety system. SUPPORTED is advisory — always
  physically verify ground support before anyone approaches a face.
- Validation is on one mine session / one camera. The fail-safe design holds by
  construction, but generalisation to other headings/cameras is not yet proven —
  expand `eval/labels.json` as more labelled footage becomes available.
- Tests: `python3 -m pytest -v` (28).
