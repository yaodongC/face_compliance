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
        "screen_send_w": 1000,
    },
    "coverage": {
        "face_x": [0.20, 0.85],                   # face width band (fractions)
        "mesh_gap": 240,                          # s; gap between mesh-install episodes
        "mesh_min_events": 3,                     # detections needed for a real mesh episode
        "width_panel_w": 0.12,                    # screen-coverage width per install
        "width_min_hits": 2,                      # sustained hits to count a region covered
        "width_bin_w": 0.02,
        "min_overlap": 0.02,                      # adjacent panels must overlap by this
        "segments": 4,
    },
}


def _merge(base, over):
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load(path=None):
    p = Path(path) if path else Path(__file__).resolve().parent / "config.yaml"
    cfg = yaml.safe_load(p.read_text()) if p.exists() else {}
    return _merge(DEFAULTS, (cfg or {}).get("params", {}))


PARAMS = load()
