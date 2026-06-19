# Success metrics — harness over a FULL development cycle

The harness is run over the entire recorded cycle (~56 min, bags 0-56): start with
the booms drilling/bolting the face → end with the face fully bolted and the drill
booms parked to the sides (the compliant rest-state). A full-cycle time-lapse
(`data/full_cycle.mp4`, 1 frame / 2 s real) is analysed every ~30 s of real time.

## Ground truth (hand-labelled at high resolution)
- Almost the entire cycle (~0–3250 s) = ACTIVE WORK: booms drilling/bolting the
  face, not parked → NOT the supported rest-state (label: not-supported).
- The very end (~3300–3374 s, booms parked, face fully bolted) = SUPPORTED.

## Metrics (the harness is "optimised for the full cycle" when ALL hold)
- **M1 — Zero false-safe over the whole cycle (CRITICAL).** The harness outputs
  `SUPPORTED` ONLY in the parked end-state; never during the active-work phase.
  `false_safe_windows == 0`. A false "supported" mid-cycle could kill someone.
- **M2 — Detects the compliant end.** `SUPPORTED` is reached during the parked
  end-state (the last ~1–2 min). It must not be uselessly never-supported.
- **M3 — Point accuracy ≥ 0.90** on the hand-labelled GT points (≥15 points
  spanning the cycle), every error biased to over-caution (never false-safe).
- **M4 — Temporal coherence.** `SUPPORTED` appears as ONE coherent block at the
  end, not isolated spikes scattered through the work phase (≤1 isolated SUPPORTED
  run before the end).
- **M5 — Full coverage.** A verdict is produced for the whole cycle with no
  crash/gap; the run is reproducible (temp 0).

## Deliverable after metrics pass
- Save **what the GUI shows** over the full cycle to an MP4 (render the video +
  verdict banner + checklist + scene text per frame), i.e. a time-lapse of the
  GUI for the whole cycle.

## Status (update each iteration)
- [ ] M1 false-safe over cycle = 0
- [ ] M2 SUPPORTED reached at parked end
- [ ] M3 point accuracy >= 0.90
- [ ] M4 temporal coherence (one SUPPORTED block at end)
- [ ] M5 full coverage, reproducible
- [ ] GUI-over-cycle MP4 saved
