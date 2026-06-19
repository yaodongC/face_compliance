# LoopX Safety AI-Agent (VLM) — underground face-support monitor

A safety-critical harness that watches the front camera of a drill jumbo in an
underground heading and monitors **ground-support compliance** during the
screen-and-bolt cycle, using a local VLM (Qwen2.5-VL-32B served by vLLM on the
Jetson Thor). Fully offline at inference. Reads from a **recorded MP4 or a live
RTSP camera** — same code path.

## What it reports
- **Operator safety** (the real-time, reliable signal): **CLEAR** vs **DANGER** —
  DANGER when a worker enters the danger zone in front of the jumbo *while the boom
  is still moving* (drilling not stopped before entry).
- **Mesh installation**: a running **count of screens installed** over the cycle,
  with an install timeline and a danger-zone-entry timeline.
- **Event log**: a durable, append-only timeline of entries, violations and mesh
  installs — the system's external memory (the VLM is stateless).
- It **never auto-certifies "supported."** Full mesh coverage can't be measured
  reliably from this footage, so coverage stays *assistive* and defers to on-site
  inspection.

## Design (hard-won, encoded in the harness)
1. **Operator danger is entry-based.** The operator *must* enter the zone to reload
   mesh + bolts — that is normal. It's non-compliant only if the boom was still
   moving at entry. `operator_safety.classify_sessions` judges each reload visit by
   the boom motion at entry (`boom_motion_thresh` in config; data clusters ≤0.023
   stopped vs ≥0.046 moving, so 0.035 sits in the gap).
2. **Person-confirmed, not colour-confirmed.** Operators are detected by the VLM
   confirming a *person*, then gated by a classical hi-vis-**orange** check (workers
   are orange, booms are yellow) — colour alone false-positives on equipment.
3. **Mesh count by temporal episode, not position.** One mesh is bolted over a
   sustained burst of visits (the operator drifts across its width); a *new* mesh
   starts only after a long gap to reload a fresh screen. The number of screens is
   **emergent** (depends on face size) — never assumed. Per-panel localisation is
   not reliable (a bolted mesh blends into the face), so we count, not outline.
4. **Resolution is a safety parameter.** Detail is invisible at low resolution and
   full-frame views confuse the model — perception runs on a high-res face crop
   (`face_crop`).
5. **Grounded perception, decision in code.** The VLM only reports what it sees;
   compliance/danger is decided in code, never by the model's own verdict. Defaults
   are the *unsafe/unverified* answer.

## Inputs — configurable (file or live RTSP)
`live_source.py` is one frame source for both:
```python
open_source("data/full_cycle.mp4")                              # recorded file
open_source("rtsp://${RTSP_USER}:${RTSP_PASS}@10.20.30.40:554/cam0_0")  # live (HW decode)
```
Set the live input in **config.yaml**; **credentials come from the environment**, not
the committed config:
```yaml
input: rtsp://${RTSP_USER}:${RTSP_PASS}@10.20.30.40:554/cam0_0   # or a file path
```
```bash
export RTSP_USER=<user> RTSP_PASS=<pass>   # expanded at runtime by live_source
```
`${...}` placeholders in `input` are expanded from env vars, so the literal URL is
only ever in memory. (Camera passwords must never be committed; rotate any that have
been.)

## Run it
```bash
# serve the model (offline once weights are cached): see ../install.md
# --- LIVE monitor (reads config.yaml `input:` — file or RTSP) ---
python3 live_gui.py                      # headless: writes data/live_frame.png + optional --out
python3 live_gui.py --display            # on a display (DISPLAY=:0)
python3 live_gui.py --input data/full_cycle.mp4 --seconds 60 --out data/live.mp4

# --- OFFLINE pipeline on recorded bags ---
python3 extract_video.py --bags ../<bag>.bag ...          # bags -> MP4 + index
python3 make_2x.py                                        # smooth 2x real-time video of the full cycle
python3 scan_operator.py --bags 0-56                      # full-session operator scan -> data/operator_events.json
python3 build_event_log.py                               # -> data/event_log.jsonl (external memory)
python3 render_gui.py --video data/full_cycle.mp4 \
  --analysis data/full_cycle_analysis.json --index data/full_cycle.idx \
  --events data/event_log.jsonl --operator data/operator_events.json \
  --out data/full_cycle_gui.mp4                          # render the monitor to MP4
```

## Validation
- **In-domain** (`python3 eval.py` vs `eval/labels.json`): false-safe = 0 (never
  certified a drilling/bare face), recall 1.00 on the labelled clips.
- **Out-of-domain fail-safe** (`python3 office_test.py`): pointed at a live **office**
  camera, the harness produced **0 false-safe, 0 false-alarm, 0 operator false
  positives** — the VLM described it accurately ("an office environment with a
  desk, cables, and equipment") and refused to invent mine observations. The
  fail-safe design holds on completely out-of-domain input.
- Unit tests: `python3 -m pytest -q`.

## Tasks — one engine, many inspections
An inspection task is a **declarative bundle** under `tasks/<name>/`, loaded by the
same task-agnostic engine:
```
tasks/face_support/
  params.yaml    # thresholds / ROIs (merged over harness_config.DEFAULTS, validated at load)
  prompts.yaml   # VLM prompts (system / person / screen)
  rules.yaml     # verdict decision tables (rules_engine: first-match, fail-safe default)
```
The active task is `config.yaml` `task:` (or env `HARNESS_TASK`). The engine
(perception client, feature extractors, rules engine, event log, GUI) is shared;
swapping the bundle runs a different inspection **with no code change** — see
`tasks/demo/` (a PPE example) and `tests/test_task_bundle.py`. Per-domain *feature
extractors* (what computes the booleans a rule consumes) are still code; the bundle
is everything declarative. **Every bundle must be validated against its own golden
eval set before go-live** — a well-formed but wrong rule is a hazard.

> Safety-critical refactors here are gated by **output equivalence**: the rule layer
> is locked by `tests/test_equivalence.py` (golden fixtures) and the full pipeline by
> `tests/test_e2e.py` (rendered-frame + event-log hash). A change that alters any
> verdict fails the build.

## Components
| file | role |
|---|---|
| `live_source.py` | unified frame source: MP4 file or RTSP camera (HW decode, reconnect) |
| `live_gui.py` | **live** monitor — configurable input, real-time perception, GUI |
| `render_gui.py` / `gui_theme.py` | **offline** monitor renderer (`compose()`) + shared theme |
| `vlm_client.py` | VLM face perception (grounded) |
| `operator_safety.py` | operator detection + entry-based danger classification |
| `scan_operator.py` | full-session operator scan → `operator_events.json` |
| `coverage.py` | mesh-install counting (temporal episodes) |
| `event_log.py` / `build_event_log.py` | external-memory event log |
| `office_test.py` | out-of-domain false-alarm / hallucination test |
| `config.yaml` | input source, `face_crop`, `boom_motion_thresh`, endpoint |

## Safety notes
- **ASSISTIVE demo, NOT a certified safety system.** Coverage is advisory and the
  harness never auto-certifies support — always physically verify ground support
  before anyone approaches a face.
- Per-mesh localisation and exact counts are estimates; the reliable output is the
  operator-danger detection. Validation is on one mine session / one camera —
  generalisation to other headings is not yet proven.
