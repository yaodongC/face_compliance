"""Tests for the Vale-document mesh/bolt calculator and the lidar gravity alignment."""
import numpy as np
import vale_support as vs
import lidar_analyzer as la


# ---- mesh count from Vale rules (size-dependent) ----
def test_meshes_scale_with_width():
    # 6' sheets, 1' overlap -> 5' advance; meshes = ceil((W_ft - 1)/5)
    assert vs.meshes_required(20.3) == 4      # this face (~6.2 m)
    assert vs.meshes_required(13.1) == 3      # ~4 m
    assert vs.meshes_required(26.0) == 5      # ~8 m wider face -> more
    assert vs.meshes_required(5.0) == 1       # tiny face -> at least one


def test_meshes_monotone():
    ms = [vs.meshes_required(w) for w in range(10, 40, 2)]
    assert ms == sorted(ms)


def test_calc_bolts_follow_docs():
    r = vs.calc(6.2, 5.69)                     # measured face
    assert r["meshes_required"] == 4
    assert r["bolts_required_min"] == 16       # CMTS-2015-001 leading-edge 4/sheet
    assert r["bolts_required_div6"] == 24      # Div6 Creighton face 6/sheet (3-0-3)
    assert r["bolts_required_min"] == r["meshes_required"] * vs.CMTS_MIN_BOLTS_PER_SHEET
    assert r["bolts_required_div6"] == r["meshes_required"] * vs.DIV6_BOLTS_PER_SHEET
    assert set(["face_width_ft", "bolt_grid", "sources"]).issubset(r)


def test_calc_scales_to_bigger_face():
    small, big = vs.calc(4.0, 5.0), vs.calc(9.0, 5.0)
    assert big["meshes_required"] > small["meshes_required"]
    assert big["bolts_required_min"] > small["bolts_required_min"]


# ---- concrete mesh/bolt layout ----
def test_mesh_layout_panels_cover_face_with_overlap():
    lay = vs.mesh_layout(5.99, 5.50)
    assert lay["n_meshes"] == 4
    p = lay["panels"]
    assert p[0]["x0"] == 0.0                       # first panel at the left wall
    assert abs(p[-1]["x1"] - lay["face_width_ft"]) < 0.15  # last panel at the right wall (rounding)
    for a, b in zip(p, p[1:]):
        assert b["x0"] < a["x1"]                   # adjacent panels overlap
        assert (a["x1"] - b["x0"]) >= vs.OVERLAP_FT - 0.01  # by at least the required overlap


def test_mesh_layout_bolts_within_extents():
    W, H = 5.99, 5.50
    lay = vs.mesh_layout(W, H)
    w_ft, h_ft = W / vs.FT, H / vs.FT
    assert len(lay["bolts"]) == lay["bolt_cols"] * lay["bolt_rows"]
    for b in lay["bolts"]:
        assert vs.WALL_OFFSET_FT - 0.01 <= b["x"] <= w_ft - vs.WALL_OFFSET_FT + 0.01
        assert vs.BOR_OFFSET_FT - 0.01 <= b["y"] <= h_ft - vs.BACK_OFFSET_FT + 0.01


def test_mesh_layout_scales():
    assert vs.mesh_layout(4.0, 5.0)["n_meshes"] < vs.mesh_layout(9.0, 5.0)["n_meshes"]


# ---- mesh-count confidence (precision adequacy) ----
def test_count_confidence_robust_mid_band():
    c = vs.mesh_count_confidence(5.99, width_unc_m=0.13)   # this face, well inside the band
    assert c["meshes"] == 4
    assert c["width_band_m"] == [4.88, 6.40]
    assert c["robust"] is True
    assert c["margin_m"] > 2 * 0.13


def test_count_confidence_marginal_near_boundary():
    c = vs.mesh_count_confidence(6.38, width_unc_m=0.13)   # just under the 6.40 m -> 5-mesh edge
    assert c["robust"] is False                            # margin < 2x uncertainty
    assert c["meshes"] == 4


def test_count_confidence_measured_width_inside_its_band():
    for w in (4.0, 5.99, 7.5, 9.0):
        c = vs.mesh_count_confidence(w)
        lo, hi = c["width_band_m"]
        assert lo <= w <= hi


# ---- arched cross-section render (the face is not a rectangle) ----
def test_face_profile_render_arched(tmp_path):
    import render_face_profile as rfp
    # synthetic arched profile: narrow floor -> wide springline -> tapered crown
    prof = {"have": True, "height_m": 5.7, "max_width_m": 6.0, "area_m2": 30.9,
            "profile": [{"h_frac": f, "height_m": round(f * 5.7, 2),
                         "width_m": w} for f, w in
                        [(0.05, 3.6), (0.3, 5.4), (0.5, 6.0), (0.7, 6.0), (0.9, 5.5), (0.97, 4.8)]]}
    out = tmp_path / "p.png"
    rfp.render(prof, str(out))
    import cv2
    img = cv2.imread(str(out))
    assert img is not None and img.shape[0] > 100 and img.shape[1] > 100


# ---- robust wall-position (the precision upgrade) ----
def test_wall_pos_robust_to_flare():
    # a planar wall at Y=-3.0 (dense) plus a few flared corner points out to -3.6.
    wall = np.full(2000, -3.0) + np.random.RandomState(0).normal(0, 0.05, 2000)
    flare = np.linspace(-3.6, -3.1, 40)
    vals = np.concatenate([wall, flare, np.full(2000, 3.0)])  # + right wall at +3.0
    pos, std = la._wall_pos(vals, "lo", slab=0.4)
    assert abs(pos - (-3.0)) < 0.1          # locks onto the wall plane, not the flare tip
    assert std < 0.15                        # tight (planar)
    # raw percentile would be pulled toward the flare:
    assert np.percentile(vals, 1) < pos      # i.e. the 1pct extent is more negative (inflated)


def test_wall_pos_width_matches_truth():
    rs = np.random.RandomState(1)
    left = -2.99 + rs.normal(0, 0.05, 3000)
    right = 2.99 + rs.normal(0, 0.05, 3000)
    vals = np.concatenate([left, right])
    lo, _ = la._wall_pos(vals, "lo")
    hi, _ = la._wall_pos(vals, "hi")
    assert abs((hi - lo) - 5.98) < 0.1       # recovers the true 5.98 m separation


# ---- gravity alignment (pure geometry) ----
def test_gravity_align_levels_up():
    # a lidar pitched ~24 deg forward (matches the real rig): gravity has +x, -z
    g = np.array([0.40, 0.02, -0.91])
    R = la.gravity_align_R(g)
    up = -g / np.linalg.norm(g)
    z_new = R @ up
    assert np.allclose(z_new, [0, 0, 1], atol=1e-6)    # gravity-up maps to +Z
    # R is a proper rotation (orthonormal, det +1)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-6)


def test_gravity_align_horizontal_forward():
    # the new forward axis must be horizontal (zero vertical component along true up)
    g = np.array([0.40, 0.0, -0.91])
    R = la.gravity_align_R(g)
    up = -g / np.linalg.norm(g)
    fwd_new = R[0]                                       # first new axis = forward
    assert abs(fwd_new @ up) < 1e-6                      # forward is perpendicular to up


# ---- regression guards for the code-review fixes ----
def test_gravity_align_degenerate_raises():
    # M2: lidar pointing ~straight along gravity (no horizontal forward) must raise, not NaN
    import pytest
    for g in (np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0])):
        with pytest.raises(ValueError):
            la.gravity_align_R(g)


def test_face_start_x_fallback_when_no_gap():
    # H1: if the central column is one solid cluster (no boom gap), return the near-edge
    # fallback (~1.5 m), NOT 9.0 m (which would empty the face mask downstream).
    rs = np.random.RandomState(0)
    X = rs.uniform(0.3, 8.5, 40000)          # uniformly filled, no low-density gap
    Y = rs.uniform(-1.0, 1.0, 40000)
    assert la._face_start_x(X, Y) <= 1.6


def test_face_start_x_finds_gap():
    # near boom cluster (0.5-2.5 m) + empty gap + far face (4.5-7 m) -> face starts ~after gap
    rs = np.random.RandomState(1)
    boom_x = rs.uniform(0.5, 2.5, 20000)
    face_x = rs.uniform(4.5, 7.0, 20000)
    X = np.concatenate([boom_x, face_x])
    Y = rs.uniform(-1.0, 1.0, X.size)
    gap = la._face_start_x(X, Y)
    assert 2.5 <= gap <= 4.7                  # in/after the empty band, before the face
    # NOTE: a far-standoff (>6 m) boom case is NOT handled — see _face_start_x docstring (L1).
    # That generalisation needs real far-standoff data to tune against, not a synthetic guess.
