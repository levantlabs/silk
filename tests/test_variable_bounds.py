"""TASK-390: out-of-bounds values raise; off-grid values snap with a warning."""
import warnings

import pytest

from silk.circuit.variable import Variable


@pytest.mark.parametrize("res", [None, 0.005])
def test_assign_out_of_bounds_raises_and_keeps_value(res):
    v = Variable('w', 1.0, bounds=(0.42, 20.0), resolution=res)
    for bad in (0.1, 50):
        with pytest.raises(ValueError, match=r"'w'.*" + str(bad) + r".*\(0\.42, 20\.0\)"):
            v.value = bad
        assert v.value == 1.0


def test_none_bound_end_is_unbounded():
    v = Variable('w', 1.0, bounds=(0.42, None), resolution=0.005)
    v.value = 1e6
    with pytest.raises(ValueError):
        v.value = 0.1
    assert v.value == 1e6


@pytest.mark.parametrize("bounds", [(0.42, 20.0), (0.42, None)])
def test_constructor_out_of_bounds_raises(bounds):
    with pytest.raises(ValueError, match="'w'"):
        Variable('w', 0.1, bounds=bounds)


def test_off_grid_snaps_with_warning():
    v = Variable('w', 1.0, bounds=(0.42, None), resolution=0.005)
    with pytest.warns(UserWarning, match=r"'w'.*1\.0013.*1\.0"):
        v.value = 1.0013
    assert v.value == 1.0
    with pytest.warns(UserWarning, match="'w2'"):
        Variable('w2', 1.0013, bounds=(0.42, None), resolution=0.005)


def test_on_grid_and_unenforced_do_not_warn():
    v = Variable('w', 1.0, bounds=(0.42, None), resolution=0.005)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        v.value = 0.435
        Variable('c', 1.0013, resolution=0.005, enforce_resolution=False)


def test_clip_still_clips_and_space_apply_unchanged():
    from silk.optimize import VariableSpace
    v = Variable('w', 1.0, bounds=(0.42, 20.0), resolution=0.005)
    assert v.clip(0.1) == 0.42
    sp = VariableSpace([v])
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        sp.apply({'w': 1.0013})
    assert v.value == 1.0
    with pytest.raises(ValueError, match="Invalid value"):
        sp.apply({'w': 50})
