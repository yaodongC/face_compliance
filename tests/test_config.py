"""Config single-source-of-truth + startup-validation gates."""
import pytest
import harness_config as hc
import operator_safety as osf
import coverage as cov


def test_module_constants_bound_to_params():
    # the centralised params ARE the source the modules use (no drift)
    assert osf.MOTION_FRAC_THRESH == hc.PARAMS["operator"]["boom_motion_thresh"]
    assert osf.MIN_ORANGE == hc.PARAMS["operator"]["min_orange"]
    assert osf.MOTION_PX_THRESH == hc.PARAMS["operator"]["motion_px_thresh"]
    assert tuple(cov.FACE_X) == tuple(hc.PARAMS["coverage"]["face_x"])


def test_validate_accepts_current_config():
    hc.validate(hc.load())   # the shipped config.yaml must pass


@pytest.mark.parametrize("bad", [
    {"operator": {"boom_motion_thresh": 0}},        # never raises DANGER
    {"operator": {"boom_motion_thresh": "0.035"}},  # string -> TypeError at compare
    {"operator": {"boom_motion_thresh": -1}},
    {"operator": {"min_orange": -1}},               # disables the hi-vis gate
    {"operator": {"danger_roi": [0, 0, 0, 0]}},     # zero-area ROI -> no motion
    {"operator": {"danger_roi": [0.5, 0.4, 0.2, 0.8]}},  # y1<y0
    {"coverage": {"face_x": [0.8, 0.2]}},           # x1<x0
    {"coverage": {"full_coverage_frac": 1.5}},      # out of (0,1]
    {"coverage": {"mesh_min_events": 0}},
])
def test_validate_rejects_unsafe_config(bad):
    with pytest.raises(ValueError):
        hc.validate(hc._merge(hc.DEFAULTS, bad))
