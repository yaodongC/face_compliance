# Harness design review — toward a configurable, general, safety-critical inspection engine

**Goal:** make the harness's *configuration* (thresholds, prompts, rules) and *task*
externalized and reusable — **without changing any compliance behaviour**. The
current decisions are good; we are changing *how it is configured*, not *what it
decides*. Every step is gated by an output-equivalence test (below).

This review is grounded in 2024–2026 best practice for safety-critical agentic /
VLM inspection systems (guardrails frameworks, functional-safety standards,
policy-as-code, structured output, task generalisation). Key external anchors are
cited inline.

---

## 1. How the current harness already scores (it's strong)

A safety-critical inspection harness is, in regulatory terms, a **high-risk AI
system** and a **SOTIF/ISO 21448** problem (harm from functional insufficiency, not
breakage). Scored against the distilled best-practice principles:

| Principle (best practice) | Status today |
|---|---|
| **VLM is an untrusted sensor; deterministic code owns the verdict** | ✅ already core — VLM reports `face_screened/drill_active/person…`; `coverage.py`/`operator_safety.py` decide. (IEC 61508-3: "AI not recommended above SIL 1" → keep the model out of the decision.) |
| **Fail-safe defaults (uncertain → unsafe, never silent PASS)** | ✅ `SAFE_DEFAULT` all-false; never auto-certifies coverage |
| **Conservative aggregation** | ✅ entry-based danger, sustained-activity mesh count, worst-case fusion |
| **Automatic audit log / traceability (EU AI Act Art. 12)** | ✅ `event_log.py` append-only JSONL (external memory) |
| **Validated out-of-distribution behaviour** | ✅ office test: 0 false-safe / 0 false-alarm / 0 operator-FP |
| **Centralised, single-source config** | ⚠️ scattered + **drift bug** (below) |
| **Externalised, versioned prompts** | ❌ hardcoded in `vlm_client.py`, `operator_safety.py` |
| **Rules separated from engine (data vs code)** | ❌ verdict logic is in code |
| **Schema-validated structured perception output** | ⚠️ JSON parsed with regex + fallbacks, no strict schema |
| **Explicit OOD / domain-guard gate** | ❌ none (robust by luck of grounding, not by design) |
| **One engine, many tasks (generalisable)** | ❌ single hardcoded domain |

The bottom rows are the work. The top rows are why the results are good and must
not regress.

---

## 2. The four requested improvements, concretely

### (a) Centralised config — *and a drift bug to fix*
Constants are spread across `operator_safety.py` (`MOTION_FRAC_THRESH`, `MIN_ORANGE`,
`DANGER_ROI`, `FACE_BAND`, orange HSV ranges, `_SCREEN_SEND_W`, `_bbox_ok` params)
and `coverage.py` (`FACE_X`, mesh `gap/min_events`, `panel_w/min_hits`,
`min_overlap`). **Drift bug:** `boom_motion_thresh: 0.035` exists in `config.yaml`
*and* `operator_safety.MOTION_FRAC_THRESH = 0.035` exists as a separate hardcoded
default that `classify_sessions()` uses unless explicitly passed the config value —
two sources of truth for one safety knob. Centralising forces one.

→ One `config.yaml` block per task; every module reads the loaded config object,
no module-level constants. (Validate the loaded config against a schema at startup —
a malformed threshold should fail loudly, not silently default.)

### (b) Configurable rules — with the safety-correct boundary
Best practice (OPA, DMN, json-rules-engine) is unanimous: **separate rules (data)
from engine (code)**, version them, unit-test them. *But* the same sources are
explicit that **derived facts, counts, and branching must stay in code, not data** —
pushing them into YAML produces the "Inner-Platform Effect," a worse language that
defeats the auditability you wanted. So split into **two layers**:

1. **Feature extractors (code, certifiable, config-driven thresholds):** the things
   that *compute* derived facts — motion fraction, orange fraction, mesh temporal
   episodes, coverage union. These are exactly the "computed facts" that must remain
   in the certified engine; only their **thresholds** become config.
2. **Verdict rules (YAML data, decision-table style):** the *mapping* from
   grounded/derived booleans to a verdict, e.g.
   ```yaml
   rules:
     - when: {operator_in_zone: true, boom_moving: true}   ->  DANGER
     - when: {coverage_full: true, overlaps: true}         ->  COMPLIANT
     - default: NOT_SUPPORTED          # fail-safe default is explicit and first-class
   ```
   A small interpreter evaluates these with an explicit **hit policy** (first-match,
   fail-safe default last). When a rule needs cross-rule precedence or "≥2 people in
   zone without spotter," escalate that rule to DMN/code — don't grow YAML into a
   language.

This gives you editable rules *and* keeps the part a regulator would have to certify
small and deterministic.

### (c) Configurable prompts
Move `SYSTEM_PROMPT` / `PERSON_PROMPT` / screen-prompt into `prompts/<task>.yaml`
(YAML frontmatter + templated body, à la Microsoft Prompty), loaded at runtime.
Treat prompts as code: version them, and gate edits on a golden eval set
(`eval/labels.json` already exists) so a silent prompt change can't degrade a
safety-critical perception.

### (d) Generalise the engine — task plug-in bundles
Define each inspection task as a **declarative bundle**:
```
tasks/face_support/
  task.yaml        # perception schema (fields + types + `unknown` + confidence)
  prompts.yaml     # system + per-detector prompts
  thresholds.yaml  # all numeric knobs (was the scattered constants)
  rules.yaml       # verdict decision table (fail-safe default)
```
A **task-agnostic engine** loads a bundle by name and runs the same pipeline:
`source → perceive (VLM, structured) → extract features (code) → rules → verdict →
event log → GUI`. `face_support` is the default bundle and reproduces today exactly.
A new inspection (e.g. PPE/helmet check) is a new bundle, **no engine change** —
validated against *its own* golden set before go-live (a syntactically valid but
wrong bundle is a hazard).

---

## 3. Cheap safety hardening the research surfaced (optional, high value)
- **Strict structured output.** vLLM supports guided/grammar-constrained decoding;
  emit the perception object against a JSON Schema so a missing/renamed field or a
  hallucinated enum is impossible — this is the keystone of perception/decision
  separation. Add `confidence` and an explicit **`unknown`** value per field that
  routes to REVIEW (not coerced to `false`).
- **OOD / domain-guard front gate.** A cheap check ("is this even the inspection
  scene?") *before* the expensive VLM, that abstains on out-of-domain frames. The
  office test passed by grounding; a guard makes it pass *by design*.
- **Rule/prompt versioning in the event log.** Log the task-bundle version + rule id
  fired with each verdict — closes the Art. 12 traceability loop (frame → perception
  → rule → verdict → versions).

---

## 4. Behaviour-preserving migration plan (each phase gated by equivalence)

**Phase 0 — Equivalence lock (do first).** Snapshot today's outputs on the full
cycle (operator_events, mesh count + install times, entries + verdicts, event log,
coverage state) as golden fixtures; add a test asserting the refactored pipeline
reproduces them exactly. This + the existing 45 unit tests are the gate for every
later phase. *No refactor merges unless these are bit-identical.*

**Phase 1 — Centralise config.** Pull every scattered constant into `config.yaml`;
modules read the config object; delete module-level constants; fix the
`boom_motion_thresh` drift. Behaviour identical (same numbers).

**Phase 2 — Externalise prompts.** `prompts/face_support.yaml`, loaded at runtime;
prompt text byte-identical.

**Phase 3 — Structured perception schema.** Declare the perception schema; optionally
enable vLLM guided decoding (same fields). Add `unknown`/confidence as additive,
non-breaking.

**Phase 4 — Rules layer.** Extract verdict mappings into `rules.yaml` + a tiny
interpreter; feature extractors stay in code with config thresholds. The decision
table is written to reproduce current verdicts exactly (Phase 0 proves it).

**Phase 5 — Task bundle + engine.** Package the above as `tasks/face_support/`; the
engine loads it by name. Prove generality with one second trivial task.

**Phase 6 — Safety hardening.** OOD gate, version stamping in the log, golden-set CI
for prompts/rules.

---

## 5. Anti-patterns to avoid (from the research)
- **Don't let the VLM decide.** Keep it a sensor; the deterministic layer owns the
  verdict. (Already true — preserve it.)
- **Don't push derived-fact/branching logic into YAML.** Thresholds + simple
  condition→verdict rows only; computation stays in certified code.
- **Don't treat config conformance as correctness.** A well-formed but wrong rule
  bundle is more dangerous than a missing one — every task gets its own golden eval.
- **Don't change behaviour in this refactor.** Output equivalence is the contract.

---

### Primary sources
Anthropic *Building Effective Agents* (workflow vs agent); *Design Patterns for
Securing LLM Agents* (arXiv 2506.08837); *Safety Monitoring of ML Perception
Functions* (arXiv 2412.06869); EU AI Act Arts. 9/12/14/15; IEC 61508-3 (AI ≤ SIL 1);
OpenAI Structured Outputs / function-calling; OPA & DMN (policy-as-code, hit
policies); Microsoft Prompty / Langfuse (prompt-as-config); InspectVLM (arXiv
2508.01921) & MMAD (general VLM inspectors unreliable without grounding); NIST AI RMF.
