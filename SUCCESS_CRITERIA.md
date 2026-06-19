# Success criteria — safety-critical face-support compliance harness

The harness watches the front-camera view of an underground development face and
decides whether the FACE is supported to standard (per the operator's definition:
**SUPPORTED = the face itself is covered with wire mesh/screen AND has the rock-bolt
plate pattern**; back/wall mesh alone does NOT count; a face being drilled with no
face screen is NOT supported).

It is "done" when ALL of the following hold on a held-out, human-labelled test set
drawn from the actual rosbags (sequences spanning the full cycle: bare/drilling
face → screened+bolted face).

## Must-pass (safety)
- **C1 — Zero false-safe.** The harness NEVER outputs `SUPPORTED` (or any "compliant"
  signal) on a sequence whose ground truth is not-supported. `false_safe_count == 0`.
  This is non-negotiable: a false "supported" can get a miner crushed.

## Must-pass (usefulness — otherwise it's a constant-"unsafe" alarm nobody trusts)
- **C2 — Detects real support.** On ground-truth SUPPORTED sequences (e.g. the bolted
  end-state, bags ~50-56), the harness outputs `SUPPORTED`. Positive recall ≥ 0.8,
  and it MUST get the clear fully-bolted end-state right.
- **C3 — Correct on negatives.** On not-supported sequences (drilling / bare face) it
  outputs a non-safe verdict (`UNSUPPORTED` / `NOT VERIFIED` / `DANGER`).

## Quantitative target on the test set
- **C4 — Accuracy ≥ 0.90** over all labelled sequences, where "correct" = predicted
  band matches truth (SUPPORTED↔supported; any non-safe verdict↔not-supported), AND
  C1 holds (every error is over-caution, never false-safe).
- Test set: ≥ 8 sequences, ≥ 3 SUPPORTED and ≥ 3 not-supported, from ≥ 6 different bags.

## Operational
- **C5 — Fully offline.** No network access at inference: model weights local, vLLM on
  localhost, no external URLs in the code path. Verified by (a) grep of the code path
  and (b) a run with outbound network blocked.
- **C6 — Stable & reproducible.** Sequence verdict is stable across a window (no
  per-frame flicker) and reproducible across repeated runs.
- **C7 — All unit tests pass.**

## Method notes (how we get there — "resolution is a safety parameter")
- Mesh + bolt plates are FINE detail, invisible at 768px / thumbnails. Perception must
  use HIGH resolution and/or ZOOM into regions (face centre, brow, walls) — analyse
  each region at native detail, then aggregate. This is the primary accuracy lever.
- Asymmetric K-vote consensus: unanimous for SUPPORTED, any-vote for hazards.
- Fail-safe defaults: NOT_VERIFIED unless positively, repeatedly verified.

## Current status (update each iteration)
- [ ] C1 false-safe = 0 on test set
- [ ] C2 bolted end-state -> SUPPORTED ; positive recall >= 0.8
- [ ] C3 negatives -> non-safe
- [ ] C4 accuracy >= 0.90
- [ ] C5 offline verified
- [ ] C6 stable/reproducible
- [ ] C7 unit tests pass
