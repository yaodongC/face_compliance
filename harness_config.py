"""Single source of truth for harness PARAMETERS (thresholds, ROIs, gaps).

Every tunable that used to be a scattered module-level constant now lives here once.
`DEFAULTS` holds the safe built-in values (equal to the historical constants, so
behaviour is unchanged if config.yaml is absent); `config.yaml` `params:` overrides
them. Modules read `PARAMS[...]` instead of defining their own constants, so there is
exactly one place to change a safety knob — no drift.
"""
from __future__ import annotations
from pathlib import Path
import yaml

# Built-in defaults == the historical hardcoded constants (behaviour-preserving).
DEFAULTS = {
    "operator": {
        "danger_roi": [0.45, 1.0, 0.20, 0.80],   # [y0,y1,x0,x1] fractions
        "motion_px_thresh": 25,                   # per-pixel abs-diff
        "boom_motion_thresh": 0.035,              # fraction of ROI px changed => boom MOVING
        "min_orange": 0.015,                      # min hi-vis-orange fraction to accept a person
        "orange_hsv_lo": [3, 110, 110],
        "orange_hsv_hi": [20, 255, 255],
        "bbox": {"min_area": 0.004, "max_area": 0.25, "max_aspect": 1.6},
        "session_gap": 20.0,                      # s; group operator detections into a visit
        "face_band": [0.15, 0.10, 0.90, 0.85],    # clamp screen grounding to here
        "screen_send_w": 1000,                    # px width sent to the VLM for screen grounding
        "person_send_w": 1000,                    # px width sent to the VLM for person detection
        "person_max_tokens": 140,                 # VLM response budget for person detection
        "screen_max_tokens": 120,                 # VLM response budget for screen grounding
    },
    "coverage": {
        "face_x": [0.20, 0.85],                   # face width band (fractions)
        "mesh_gap": 240,                          # s; gap between mesh-install episodes
        "mesh_min_events": 3,                     # detections needed for a real mesh episode
        "width_panel_w": 0.12,                    # screen-coverage width per install
        "width_min_hits": 2,                      # sustained hits to count a region covered
        "width_bin_w": 0.02,
        "min_overlap": 0.02,                      # adjacent panels must overlap by this
        "full_coverage_frac": 0.98,               # face-width fraction that counts as full
        "segments": 4,
    },
}


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def validate(p):
    """Fail LOUDLY at startup on a misconfigured safety knob, rather than silently
    producing a false-safe verdict (e.g. boom_motion_thresh=0 never raises DANGER) or
    crashing a worker thread later (e.g. a string threshold). Conservative checks."""
    o, c = p["operator"], p["coverage"]
    # positive scalar thresholds (a zero/negative/non-number here disables safety)
    for key, src in [("boom_motion_thresh", o), ("motion_px_thresh", o),
                     ("session_gap", o), ("mesh_gap", c), ("mesh_min_events", c),
                     ("width_panel_w", c), ("width_min_hits", c), ("screen_send_w", o),
                     ("person_send_w", o), ("person_max_tokens", o), ("screen_max_tokens", o)]:
        if not (_num(src[key]) and src[key] > 0):
            raise ValueError(f"config params: {key} must be a positive number, got {src[key]!r}")
    if not (_num(o["min_orange"]) and o["min_orange"] >= 0):
        raise ValueError(f"config params.operator.min_orange must be >= 0, got {o['min_orange']!r}")
    # fractions in [0,1]
    for key, src in [("full_coverage_frac", c)]:
        if not (_num(src[key]) and 0 < src[key] <= 1):
            raise ValueError(f"config params: {key} must be in (0,1], got {src[key]!r}")
    # ROIs: 4 fractions in [0,1], non-degenerate (positive span)
    roi = o["danger_roi"]                          # [y0,y1,x0,x1]
    if not (len(roi) == 4 and all(_num(v) and 0 <= v <= 1 for v in roi) and roi[1] > roi[0] and roi[3] > roi[2]):
        raise ValueError(f"config params.operator.danger_roi invalid (need [y0,y1,x0,x1] fractions, y1>y0, x1>x0): {roi}")
    fx = c["face_x"]
    if not (len(fx) == 2 and all(_num(v) and 0 <= v <= 1 for v in fx) and fx[1] > fx[0]):
        raise ValueError(f"config params.coverage.face_x invalid (need [x0,x1], x1>x0): {fx}")
    return p


def _merge(base, over):
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load(path=None):
    p = Path(path) if path else Path(__file__).resolve().parent / "config.yaml"
    cfg = yaml.safe_load(p.read_text()) if p.exists() else {}
    return validate(_merge(DEFAULTS, (cfg or {}).get("params", {})))


PARAMS = load()
# Modules read PARAMS at import and bind values into constants / default args, so a
# config.yaml change takes effect on process RESTART (no hot-reload). Intentional for
# a safety system: the config in force is fixed for the life of the run.
