# LoopX Safety AI-Agent (VLM) — underground face-support monitor

A safety-critical harness that watches the front camera of a drill jumbo in an
underground heading and monitors **ground-support compliance** during the
screen-and-bolt cycle, using a local VLM (Qwen2.5-VL-32B served by vLLM on the
Jetson Thor). Fully offline at inference. Reads from a **recorded MP4, ROS bags, or a
live RTSP camera** — same code path.

It now also includes a **Lidar-based face-measurement tool** and a **compliance-
milestone detector**: it measures the precise face size from the front Livox Mid360,
computes how many screens/bolts the face needs **from the Vale standards**, tracks
bolting progress from the IMU, and recognises the moment the face becomes compliant
(4 screens + 16 bolts for this heading). See **`COMPLIANCE_DETECTION.md`** for the full
design, grounding, and evaluation.

## System at a glance

```mermaid
flowchart LR
    subgraph IN["Inputs (one frame source)"]
        BAGS["ROS bags<br/>camera + 200&nbsp;Hz IMU"]
        MP4["recorded MP4"]
        RTSP["live RTSP camera"]
    end
    BAGS -->|extract_video.py| MP4
    MP4 --> SRC["live_source<br/>unified frame source"]
    RTSP --> SRC

    SRC --> VLM["VLM perception<br/>Qwen2.5-VL via vLLM<br/>(grounded: reports what it sees)"]
    BAGS -->|IMU accel energy| IMU["machine-motion signal<br/>(physical)"]

    VLM --> OSF["operator_safety<br/>person + persistence + orange"]
    IMU --> OSF
    VLM --> COV["coverage<br/>mesh-install episodes"]

    OSF --> RULES["rules engine<br/>decision tables (code)"]
    COV --> RULES
    RULES --> LOG["event_log.jsonl<br/>external memory"]

    OSF --> GUI["render_gui / live_gui<br/>monitor UI → MP4"]
    COV --> GUI
    LOG --> GUI
```

The VLM only **reports what it sees**; every compliance/danger **decision is made in
code** with classical gates and decision tables, never by the model's own verdict.

## What it reports
- **Operator safety** (the real-time, reliable signal): **CLEAR** vs **DANGER** —
  DANGER when a worker is in the danger zone in front of the jumbo *while the machine
  (drill/boom) is physically running* (drilling not stopped before entry). A
  **REVIEW** tier flags machine-active moments where presence is uncertain.
- **Mesh installation**: a running **count of screens installed** over the cycle,
  with an install timeline and a danger-zone-entry timeline.
- **Required support from the Lidar + Vale docs**: the precise **face size**
  (≈6.0 m × 5.5 m here, measured from accumulated, gravity-levelled Mid360 scans) and the
  **meshes/bolts the face needs** derived from the Vale standards (4 meshes / 16 bolts here;
  it scales with face size — a 4 m face → 3, an 8 m face → 5).
- **Bolting progress + compliance milestone**: a live `bolts x/16` · `screens x/4` ·
  `coverage %`, and a single **latched COMPLIANCE-COMPLETE** event when all are met and a
  VLM hi-res look confirms the screened+bolted face (fail-safe; cannot fire early because the
  physical IMU bolt-count holds it back). On the recorded cycle this fires at the right
  moment (≈3290 s) with false-safe = 0.
- **Event log**: a durable, append-only timeline of entries, violations and mesh
  installs — the system's external memory (the VLM is stateless).
- Coverage from vision alone stays *assistive*; the **certifiable** quantities are the
  Lidar-measured size, the IMU bolt-count, and the doc-derived requirement — always verify
  ground support on site before anyone approaches a face.

## Lidar face-measurement + Vale mesh count + compliance milestone

The required support is **not assumed** — it is measured and derived per heading:

```mermaid
flowchart LR
    LID["front Livox Mid360<br/>(non-repetitive scan)"] -->|accumulate while parked| DENSE["dense cloud<br/>(~6 M pts)"]
    IMU2["front IMU gravity"] -->|level pitch/roll| DENSE
    DENSE --> MEAS["measure_face_precise<br/>side-wall-plane width + floor→crown"]
    MEAS --> SIZE["face size 6.0×5.5 m<br/>+ arched area + camera cross-check"]
    SIZE --> VALE["vale_support<br/>(CMTS-2015-001 + Div6 rules)"]
    VALE --> REQ["4 meshes / 16 bolts<br/>(scales with size, confidence ROBUST)"]
    IMU3["IMU drilling episodes"] --> PROG["progress_tracker<br/>bolts(t)/screens(t)/coverage(t)"]
    REQ --> PROG
    PROG --> MILE["compliance_milestone<br/>(latched, fail-safe)"]
    VLMc["VLM hi-res confirm"] --> MILE
    MILE --> OUT["COMPLIANCE COMPLETE<br/>+ event log + GUI"]
```

- **Precise face size** (`lidar_analyzer.measure_face_precise`): accumulate ~300 Mid360
  scans (parked) → level with the IMU gravity vector (the Lidar is pitched 23.5°) → drop the
  boom cluster → width = distance between the fitted **side-wall planes** (robust to corner
  flare), height = robust floor→crown. Result **5.99 m × 5.5 m**, ±0.13 m across bags, walls
  planar to 0.13 m. A **camera-FOV cross-check** (`camera_crosscheck`) independently confirms
  it (face fills ~73% of the HFOV at the 6.2 m standoff). The arched cross-section + true area
  (29.9 m² vs 34.4 m² box) come from `face_profile`.
- **Meshes from the Vale documents** (`vale_support`): 6′ screen sheets, 1′ overlap (3
  squares), 4′×5′ bolt pattern → `meshes = ceil((W_ft − 1)/5)` = **4**; bolts **16** (CMTS
  leading-edge minimum, = the 16 IMU drilling episodes) or **24** (Div6 Creighton 3-0-3). It
  reports `mesh_layout` (panel x-spans + bolt grid) and a count-confidence margin.
- **Milestone** (`compliance_milestone`): latches **COMPLIANCE COMPLETE** only when
  `screens ≥ required ∧ bolts ≥ required ∧ coverage ∧ VLM confirms` — fail-safe, and it
  cannot fire during active work because the IMU bolt-count is below target there.

```bash
python3 imu_analyzer.py    --bags 0-56     # -> data/imu_timeline.json (16 bolt episodes)
python3 face_geometry.py                   # precise lidar measure + camera-check + Vale rules
python3 vale_support.py                    # (standalone) meshes/bolts for the measured face
python3 render_mesh_layout.py              # -> data/face_mesh_layout.png (panels + bolt grid)
python3 render_face_profile.py             # -> data/face_profile.png (arched cross-section)
python3 classify_episodes.py               # VLM cross-check of the IMU bolt windows
python3 compliance_milestone.py            # -> data/compliance_result.json (latched t*)
python3 eval_compliance.py [--vlm]         # F1-F3 + accuracy vs eval/cycle_gt.json
```

> Validated on one heading/session: false-safe = 0, fires at the parked compliant end
> (≈3290 s), accuracy 1.00 on the labelled points. Generalisation to other face sizes is
> built-in (the count scales with the measured size) but not yet confirmed on a second
> heading; the boom-gap detector is tuned to this rig's standoff. See `COMPLIANCE_DETECTION.md`.

## How operator danger is decided

The operator **must** enter the zone to reload mesh + bolts — that is normal. It is
only dangerous if the **machine is still operating** when they are in front. The
verdict fuses three robust signals and is **tiered and fail-safe**:

```mermaid
flowchart TD
    A["frame checked every ~8 s"] --> B{"VLM: person in front?"}
    B -- no --> NP["NO_PERSON"]
    B -- yes --> C{"hi-vis ORANGE ≥ min_orange<br/>AND person-shaped bbox?"}
    C -- "no (boom is yellow)" --> NP
    C -- yes --> D{"operator PERSISTS<br/>≥ 60% of nearby frames?"}
    D -- "no (flicker / hallucination)" --> E{"machine ACTIVE?<br/>IMU energy > thr"}
    D -- yes --> F{"machine ACTIVE?<br/>IMU energy > thr"}
    E -- yes --> REV["REVIEW<br/>audit — don't alarm, don't ignore"]
    E -- no --> NP
    F -- "yes (drill/boom running)" --> DAN["DANGER<br/>alarm"]
    F -- "no (machine stopped)" --> OK["OK_LOADING<br/>safe reload"]

    classDef danger fill:#5a1410,stroke:#e0533f,color:#fff;
    classDef ok fill:#10331c,stroke:#3fb463,color:#fff;
    classDef review fill:#4a3a10,stroke:#d6a13a,color:#fff;
    class DAN danger;
    class OK ok;
    class REV review;
```

### Why "machine moving" is read from the IMU, not the camera

Judging boom motion by **vision frame-differencing** is confounded: it fires on the
**operator walking**, on dust/water, and on lighting changes — not just the boom.
(In the recorded session, 14 of 17 vision-flagged DANGERs occurred while the machine
was physically stopped.) The jumbo's **Livox IMU** vibrates only when the machine is
actually drilling/booming (idle ≈ 0.005, active ≈ 0.03+ accel-std — a clean gap), so
it is the physical ground truth for "is the machine running." Vision remains a
fail-safe fallback if the IMU stream is missing.

```mermaid
flowchart LR
    subgraph S["signals"]
        I["IMU accel energy<br/>physical machine vibration"]
        V["vision frame-diff<br/>(confounded by operator/dust/light)"]
        P["VLM person + persistence + orange"]
    end
    I -->|machine_active| Z["classify_zone()"]
    V -. "fallback if IMU missing" .-> Z
    P -->|operator_present / seen| Z
    Z --> OUT["DANGER · OK_LOADING · REVIEW · NO_PERSON"]
```

## Design (hard-won, encoded in the harness)
1. **Operator danger is entry-based, gated on physical machine motion.** Reloading in
   front is normal; it is non-compliant only if the machine is running at entry.
   `operator_safety.classify_zone` / `classify_sessions` judge each reload visit by the
   **IMU machine-motion** at entry (`imu_active_thr`), with the vision frame-diff
   (`boom_motion_thresh`) kept only as a fallback.
2. **Person-confirmed, persistent, and colour-gated.** Operators are detected by the
   VLM confirming a *person*, then gated by (a) a classical hi-vis-**orange** check
   (workers are orange ≈ 0.09–0.27, booms are yellow ≈ 0.02–0.04 → `min_orange`),
   (b) a person-shaped bbox, and (c) **temporal persistence** across nearby frames —
   which rejects the VLM intermittently hallucinating a "worker" onto a moving boom.
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

```mermaid
sequenceDiagram
    participant Cam as Camera frame
    participant VLM as VLM (Qwen2.5-VL)
    participant Code as Feature extractor (code)
    participant Rules as Rules engine
    Cam->>VLM: high-res face crop + prompt
    VLM-->>Code: JSON {person_in_front, bbox, action}
    Note over Code: classical gates — orange, bbox shape,<br/>persistence, IMU machine-motion
    Code->>Rules: booleans {operator_present, machine_active}
    Rules-->>Code: verdict (first-match rule, fail-safe default)
    Note over Rules: the model never decides the verdict
```

## Inputs — configurable (file or live RTSP)
`live_source.py` is one frame source for both the live monitor and the renderers:

```mermaid
flowchart LR
    F["MP4 file<br/>data/full_cycle.mp4"] --> OPEN["open_source()"]
    R["rtsp://...<br/>(HW decode, auto-reconnect)"] --> OPEN
    OPEN --> APP["live_gui.py / render"]
```

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

## Make a monitor MP4 from an input MP4

There are **two ways** to turn a camera MP4 into the monitor UI video, depending on
whether you want live perception or a deterministic offline replay:

```mermaid
flowchart TD
    INMP4["input MP4<br/>(your camera clip)"]
    INMP4 --> Q{"need precomputed<br/>analysis / event log?"}
    Q -- "no — quick, runs VLM live" --> LG["live_gui.py --input IN.mp4 --out GUI.mp4"]
    Q -- "yes — deterministic replay" --> PIPE["offline pipeline<br/>(analysis + operator scan + event log)"]
    LG --> OUT1["GUI MP4"]
    PIPE --> RG["render_gui.py --video IN.mp4 ..."]
    RG --> OUT2["GUI MP4"]
```

### A. Quick — live monitor over any MP4 (one command)
Runs the real perception pipeline frame-by-frame on the MP4 and writes the composed
monitor UI to an MP4. Needs the VLM server running.
```bash
python3 live_gui.py --input data/full_cycle.mp4 --out data/full_cycle_gui.mp4
# options: --seconds N  (limit duration)   --display  (also show on DISPLAY=:0)
```
Use this for an arbitrary clip when you don't already have cached analysis.

### B. Offline render from cached analysis (deterministic, fast replay)
`render_gui.py` composites the UI from a video plus **precomputed** artifacts
(scene analysis, operator events, event log). Use this to re-render the canonical
cycle, or any MP4 for which you have generated those inputs:
```bash
python3 render_gui.py \
  --video    data/full_cycle.mp4 \              # the input MP4 (the camera feed shown)
  --analysis data/full_cycle_analysis.json \    # scene/step analysis (drill/support state)
  --index    data/full_cycle.idx \              # frame → cycle-time map (optional)
  --operator data/operator_events.json \        # operator-safety scan results
  --events   data/event_log.jsonl \             # the incident timeline (bottom bar)
  --out      data/full_cycle_gui.mp4            # output monitor MP4
```
`--analysis` and `--operator`/`--events` are what drive the overlays; `--video` is just
the feed that gets composited behind them. To render a **new** MP4 end-to-end, generate
those inputs first (see the offline pipeline below).

## Run live on an RTSP camera (real-time)

The live monitor decodes the camera in real time, runs the slow perception (VLM +
operator detection) in a **background thread** so the feed stays smooth, and composes
the same GUI at display rate from the latest shared state.

```mermaid
flowchart LR
    CAM["RTSP camera<br/>rtsp://...:554/stream"] --> LS["live_source<br/>HW decode + auto-reconnect"]
    LS --> FEED["display loop<br/>compose() @ display rate"]
    LS --> PW["perception thread<br/>every ~5 s (--every)"]
    PW --> VLM["VLM: face perception + person"]
    PW --> DET["operator_safety gates<br/>orange · bbox · motion"]
    DET --> ST["shared state + event log"]
    ST --> FEED
    FEED --> OUT["data/live_frame.png · --display window · --out MP4"]
```

1. **Start the VLM server** (vLLM; offline once weights are cached) — see `../install.md`.
   Confirm it answers: `curl http://localhost:8000/v1/models`.
2. **Point the input at the camera**, keeping credentials in the environment (never in
   the committed config):
   ```yaml
   # config.yaml
   input: rtsp://${RTSP_USER}:${RTSP_PASS}@10.20.30.40:554/cam0_0
   ```
   ```bash
   export RTSP_USER=<user> RTSP_PASS=<pass>   # expanded at runtime by live_source
   ```
3. **Run the monitor:**
   ```bash
   python3 live_gui.py                          # headless: updates data/live_frame.png continuously
   python3 live_gui.py --display                # on a screen (DISPLAY=:0); press q to quit
   python3 live_gui.py --out data/live.mp4      # also record the monitor to an MP4
   # override the camera without editing config, and tune the perception cadence:
   python3 live_gui.py --input 'rtsp://${RTSP_USER}:${RTSP_PASS}@<ip>:554/<stream>' --every 5 --display
   ```
   **Flags:** `--input` (URL/file), `--display`, `--out` (record), `--out-fps`,
   `--every` (perception interval, s), `--seconds` (auto-stop), `--snapshot`
   (headless latest-frame PNG, default `data/live_frame.png`).
4. **Watch it headless:** `data/live_frame.png` refreshes continuously; the live
   incident timeline is appended to `data/live_events.jsonl`. The feed auto-reconnects
   if the camera drops.

> **IMU note:** an RTSP stream carries **video only**, so the physical IMU
> machine-motion gate is not available live — the live path falls back to the
> vision boom-motion signal (the VLM **person + hi-vis-orange + bbox** gates still
> apply, which already reject the boom-as-operator hallucinations). For the full
> IMU-fused logic on a live rig, wire the jumbo's IMU/telemetry into the perception
> thread alongside the camera. The **offline bag pipeline uses the IMU** end-to-end.

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
python3 scan_operator.py --bags 0-56                      # full-session operator scan (VLM + IMU) -> data/operator_events.json
python3 build_event_log.py                               # -> data/event_log.jsonl (external memory)
python3 render_gui.py --video data/full_cycle.mp4 \
  --analysis data/full_cycle_analysis.json --index data/full_cycle.idx \
  --events data/event_log.jsonl --operator data/operator_events.json \
  --out data/full_cycle_gui.mp4                          # render the monitor to MP4
```

### Offline pipeline — artifacts and order

```mermaid
flowchart TD
    BAGS["ROS bags<br/>(camera + IMU)"]
    BAGS -->|extract_video.py| MP4["full_cycle.mp4 + .idx"]
    BAGS -->|analyze.py| AN["full_cycle_analysis.json<br/>(drill/support steps)"]
    BAGS -->|"scan_operator.py<br/>(VLM person + IMU + orange + persistence)"| OPS["operator_events.json"]
    OPS -->|build_event_log.py| EL["event_log.jsonl"]
    AN  -->|build_event_log.py| EL
    MP4 --> RG["render_gui.py"]
    AN  --> RG
    OPS --> RG
    EL  --> RG
    RG --> OUT["full_cycle_gui.mp4"]
```

## The monitor UI

```mermaid
flowchart TB
    H["Header — LoopX Safety AI-Agent · CYCLE mm:ss · ● REC"]
    CAM["CAMERA FEED (hero) — danger-zone ROI + operator bbox; turns red on DANGER"]
    OSC["OPERATOR SAFETY card — CLEAR / DANGER + reason"]
    MESH["MESH INSTALLATION card — count + INSTALLS timeline + DANGER-ZONE ENTRIES timeline"]
    LOG["EVENT LOG — durable incident timeline (recent entries + violation count)"]
    FOOT["ASSISTIVE MONITOR · NOT A CERTIFIED SAFETY SYSTEM · VERIFY ON SITE"]
    H --> CAM
    CAM --> OSC
    OSC --> MESH
    CAM --> LOG
    MESH --> LOG
    LOG --> FOOT
```

`compose()` in `render_gui.py` draws one frame from a state dict and is shared by the
offline renderer and the live monitor (`live_gui.py`), so both look identical. The
whole frame turns to an alarm style on a DANGER.

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
  params.yaml    # thresholds / ROIs / IMU + orange gates (merged over harness_config.DEFAULTS, validated at load)
  prompts.yaml   # VLM prompts (system / person / screen)
  rules.yaml     # verdict decision tables (rules_engine: first-match, fail-safe default)
```

```mermaid
flowchart TD
    subgraph BUNDLE["tasks/&lt;name&gt;/ — declarative (swappable, no code change)"]
        P["params.yaml<br/>thresholds / ROIs / gates"]
        PR["prompts.yaml<br/>VLM prompts"]
        RU["rules.yaml<br/>verdict decision tables"]
    end
    subgraph ENGINE["task-agnostic engine"]
        HC["harness_config"]
        PC["prompt_config"]
        RE["rules_engine.decide<br/>(first-match, fail-safe default)"]
    end
    subgraph CODE["feature extractors — per-domain CODE"]
        OSF["operator_safety.py"]
        COVf["coverage.py"]
    end
    P --> HC
    PR --> PC
    RU --> RE
    CODE --> RE
    HC --> V["verdicts"]
    RE --> V
```

The active task is `config.yaml` `task:` (or env `HARNESS_TASK`; an override is logged
and a missing bundle fails loudly). **Scope of "no code change":** the
*thresholds, prompts and verdict rules* are swappable per task without touching the
engine (see `tasks/demo/` + `tests/test_task_bundle.py`). The **feature extractors**
(`operator_safety.py`, `coverage.py`) and the perception fields they produce are
**face-support-specific code** — a genuinely new domain (e.g. the demo PPE check)
needs its own extractor module that turns frames into the booleans its rules consume.
The bundle is the declarative half; the perception half is code. **Every bundle must
be validated against its own golden eval set before go-live** — a well-formed but
wrong rule is a hazard. A run records its task + bundle hashes in
`data/event_log.meta.json` (audit provenance).

> Safety-critical refactors here are gated by **output equivalence**: the rule layer
> is locked by `tests/test_equivalence.py` (golden fixtures) and the full pipeline by
> `tests/test_e2e.py` (rendered-frame + event-log hash). A change that alters any
> verdict fails the build.

## Components
| file | role |
|---|---|
| `live_source.py` | unified frame source: MP4 file or RTSP camera (HW decode, reconnect) |
| `extract_video.py` | ROS bags → MP4 + frame-index; shared frame iterator |
| `live_gui.py` | **live** monitor — configurable input, real-time perception, GUI; `--input MP4 --out MP4` |
| `render_gui.py` / `gui_theme.py` | **offline** monitor renderer (`compose()`) + shared theme |
| `vlm_client.py` | VLM face perception (grounded) |
| `operator_safety.py` | operator detection + IMU-fused, entry-based danger classification (`classify_zone`, `machine_active`, `operator_present`, `classify_sessions`) |
| `scan_operator.py` | full-session operator scan (VLM person + IMU machine-motion) → `operator_events.json` |
| `coverage.py` | mesh-install counting (temporal episodes) |
| `imu_analyzer.py` | IMU machine-activity envelope + drilling-episode segmentation → `imu_timeline.json` (bolt counter) |
| `lidar_analyzer.py` | Mid360 scan accumulation, IMU-gravity levelling, **precise face measure** (`measure_face_precise`), arched profile (`face_profile`), camera FOV cross-check |
| `vale_support.py` | **mesh/bolt count + layout from the Vale documents** (`meshes_required`, `mesh_layout`, `mesh_count_confidence`, `calc`) |
| `face_geometry.py` | end-to-end tool: lidar measure → camera-check → Vale rules → `data/face_geometry.json` (+ PNGs) |
| `progress_tracker.py` | fused `bolts(t)` / `screens(t)` / `coverage(t)`; size-derived targets via `load_targets` |
| `compliance_milestone.py` | latched, fail-safe COMPLIANCE-COMPLETE state machine |
| `classify_episodes.py` | VLM classification of each IMU work-window (bolt-install cross-check) |
| `eval_compliance.py` | F1–F3 + accuracy of the milestone vs `eval/cycle_gt.json` |
| `render_compliance.py` / `render_mesh_layout.py` / `render_face_profile.py` | compliance GUI MP4, mesh-panel layout, arched cross-section |
| `event_log.py` / `build_event_log.py` | external-memory event log |
| `office_test.py` | out-of-domain false-alarm / hallucination test |
| `config.yaml` / `tasks/<t>/params.yaml` | input source, `face_crop`, `imu_active_thr`, `min_orange`, `boom_motion_thresh`, endpoint |

## Safety notes
- **ASSISTIVE demo, NOT a certified safety system.** The compliance-complete signal is
  fail-safe (latched; requires the IMU bolt-count, screen count, coverage AND a VLM
  confirmation; it cannot fire during active work) but it is still a demo — always
  physically verify ground support before anyone approaches a face.
- The Lidar face **size** is metric-accurate (direct ranging) and the mesh requirement
  follows the Vale documents; the **bolt count** assumes one sustained IMU drilling episode
  per bolt (16 here, VLM-cross-checked). Per-mesh visual localisation from the camera is NOT
  reliable (the VLM over-counts panels), which is why the count comes from Lidar size + docs.
- Validation is on **one mine session / one heading / one camera**: false-safe = 0, the
  milestone fires at the parked compliant end, accuracy 1.00 on labelled points. The count
  scales with measured face size by construction, but this is **not yet confirmed on a second
  heading**, and the boom-gap detector is tuned to this rig's standoff (a far-standoff
  variant was tried and reverted — see `_face_start_x` / `COMPLIANCE_DETECTION.md`).
- The IMU machine-motion gate is validated to remove vision false positives on the
  recorded session; **recall of a genuine "operator present while machine moving"
  event is not yet validated end-to-end** (no such event occurs in the recording —
  operators only enter when the machine is stopped). A silent (non-vibrating) boom
  slew with an operator present remains a documented residual risk.
