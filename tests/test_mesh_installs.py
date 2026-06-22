"""Regression tests for mesh-install counting (coverage.mesh_installs).

Bug: a real screen install that is detected SPARSELY late in the cycle (operator seen
only 1-2 times per visit, fragmented by the mesh_gap) was dropped entirely by the
min_events filter, so 4 installed meshes were counted as 3. The fix accumulates
consecutive sub-threshold episodes into one install when they collectively reach
min_events — recovering the sparsely-detected install without over-counting.
"""
from coverage import mesh_installs


def _ev(t, cx=0.35):
    return {"cycle_sec": float(t), "person_bbox": [cx - 0.05, 0.6, cx + 0.05, 0.95]}


def test_sparse_trailing_install_is_counted():
    # 3 DENSE early clusters (>=3 each) + a 4th mesh detected SPARSELY (1+1+2 events,
    # each fragment <3 and split by the 240 s gap) -> must count as 4, not 3.
    ev = []
    ev += [_ev(757), _ev(765), _ev(833)]                      # mesh 1 (dense)
    ev += [_ev(1377), _ev(1385), _ev(1477)]                   # mesh 2 (dense)
    ev += [_ev(2017), _ev(2217), _ev(2225)]                   # mesh 3 (dense)
    ev += [_ev(2517), _ev(2873), _ev(3201), _ev(3209)]        # mesh 4 (sparse, fragmented)
    inst = mesh_installs(ev, gap=240, min_events=3)
    assert len(inst) == 4, f"expected 4 installs, got {len(inst)}: {inst}"
    # the 4th install is anchored at the start of the sparse run
    assert abs(inst[-1]["time"] - 2517) < 1


def test_dense_episodes_unchanged():
    # well-detected installs are unaffected (backward compatibility)
    ev = [_ev(757), _ev(765), _ev(833),
          _ev(1377), _ev(1385), _ev(1477),
          _ev(2017), _ev(2217), _ev(2225)]
    inst = mesh_installs(ev, gap=240, min_events=3)
    assert len(inst) == 3


def test_isolated_sparse_blip_not_overcounted():
    # a single isolated late detection (1 event) below threshold and not reaching the
    # accumulated min_events must NOT create a spurious install
    ev = [_ev(757), _ev(765), _ev(833), _ev(3209)]
    inst = mesh_installs(ev, gap=240, min_events=3)
    assert len(inst) == 1
