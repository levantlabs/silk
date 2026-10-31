import pytest
from silk.faults.sugar import (
    stuck_open, short_ds, short_gs, short_gd,
    w_variation, l_variation, vth_shift,
    FaultFactory,
)
from silk.faults.types import Fault


class _MockHandle:
    """Simulates a SubcircuitInstance device handle with a hierarchical path."""
    def __init__(self, path: str):
        self._silk_fault_target_path = path


# --- stuck_open ---

def test_stuck_open_with_path_string():
    f = stuck_open('xcomp1.mn_tail')
    assert isinstance(f, Fault)
    assert f.target == 'xcomp1.mn_tail'
    assert f.fault_type == 'stuck_open'
    assert f.params == {'per_finger': True}


def test_stuck_open_with_handle():
    h = _MockHandle('xcomp1.mn_tail')
    f = stuck_open(h)
    assert f.target == 'xcomp1.mn_tail'


def test_stuck_open_per_finger_false():
    f = stuck_open('xcomp1.mn_tail', per_finger=False)
    assert f.params == {'per_finger': False}


def test_stuck_open_zero_arg_returns_factory():
    factory = stuck_open()
    assert isinstance(factory, FaultFactory)
    f = factory('xcomp1.mn_tail')
    assert f.fault_type == 'stuck_open'


# --- short_ds ---

def test_short_ds_with_path():
    f = short_ds('xcomp1.mn_in_p')
    assert f.fault_type == 'short_ds'
    assert f.target == 'xcomp1.mn_in_p'
    assert f.params == {'per_finger': True}


def test_short_ds_zero_arg_factory():
    factory = short_ds()
    assert factory('x.y').fault_type == 'short_ds'


# --- short_gs / short_gd parallel tests ---

def test_short_gs():
    assert short_gs('x.y').fault_type == 'short_gs'


def test_short_gd():
    assert short_gd('x.y').fault_type == 'short_gd'


# --- w_variation ---

def test_w_variation_requires_sigma():
    with pytest.raises(TypeError):
        w_variation('x.y')  # sigma is required


def test_w_variation_stores_sigma():
    f = w_variation('x.y', sigma=0.02)
    assert f.fault_type == 'w_variation'
    assert f.params['sigma'] == 0.02


def test_w_variation_zero_arg_requires_sigma():
    with pytest.raises(TypeError):
        w_variation()  # sigma is required even in factory form


def test_w_variation_factory_with_sigma():
    factory = w_variation(sigma=0.02)
    # sigma captured, produces a Fault when called with a path
    f = factory('x.y')
    assert f.params['sigma'] == 0.02


# --- l_variation (symmetric with w) ---

def test_l_variation():
    f = l_variation('x.y', sigma=0.03)
    assert f.fault_type == 'l_variation'
    assert f.params['sigma'] == 0.03


# --- vth_shift ---

def test_vth_shift_requires_delta():
    with pytest.raises(TypeError):
        vth_shift('x.y')


def test_vth_shift_stores_delta_and_sigma():
    f = vth_shift('x.y', delta=0.03, sigma=0.005)
    assert f.fault_type == 'vth_shift'
    assert f.params == {'delta': 0.03, 'sigma': 0.005}


def test_vth_shift_sigma_defaults_to_zero():
    f = vth_shift('x.y', delta=0.03)
    assert f.params['sigma'] == 0.0


def test_public_api_importable_from_silk_faults():
    from silk.faults import (
        Fault, AppliedFault, FaultManifest, FaultFactory,
        stuck_open, short_ds, short_gs, short_gd,
        w_variation, l_variation, vth_shift,
    )
    assert stuck_open('x.y').fault_type == 'stuck_open'


@pytest.mark.parametrize('helper,options', [
    (stuck_open, {}), (short_ds, {}), (short_gs, {}), (short_gd, {}),
    (w_variation, {'sigma': .02}), (l_variation, {'sigma': .02}),
    (vth_shift, {'delta': .03}),
])
@pytest.mark.parametrize('as_handle', [False, True])
def test_all_helpers_keep_legacy_targets_in_direct_and_factory_forms(helper, options, as_handle):
    target = 'xstage[8].bank[5]'
    value = _MockHandle(target) if as_handle else target
    assert helper(value, **options).target == target
    assert helper(**options)(value).target == target
