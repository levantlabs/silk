"""The params view a ports-form model receives (design 3.1)."""
import pytest

from silk import PortDir
from silk.circuit.modules import Module


def _view(params):
    from silk.simulation._behavioral_views import ParamsView
    return ParamsView(params)


def test_attribute_and_bracket_read_the_same_entry():
    params = _view({'gain': 2.0, 'i_tail': 1e-6})
    assert params.gain == params['gain'] == 2.0
    assert params.i_tail == params['i_tail'] == 1e-6


def test_the_view_is_live_not_a_copy():
    raw = {'gain': 2.0}
    params = _view(raw)
    raw['gain'] = 3.0
    raw['offset'] = 0.1
    assert params.gain == 3.0 and params['offset'] == 0.1


def test_calibration_writes_are_visible(pdk, tmp_path):
    mod = Module('PDK_skywater130', str(tmp_path), 'amp.cir', pdk_details=pdk,
                 module_name='amp')
    mod.define_module_ports({'a': PortDir.IN, 'y': PortDir.OUT})
    mod.set_behavioral_model(lambda ports, t, params: None, params={'gain': 2.0},
                             execution='vector')
    params = _view(mod._behavioral_params)
    mod.calibrate_behavioral('gain', 5.0)
    assert params.gain == 5.0


def test_attribute_lookup_exposes_parameters_only():
    params = _view({'get': 1.0, 'items': 2.0, 'keys': 3.0})
    assert params.get == params['get'] == 1.0
    assert params.items == params['items'] == 2.0
    assert params.keys == params['keys'] == 3.0
    for name in ('get', 'items', 'keys', 'values'):
        with pytest.raises(AttributeError):
            getattr(_view({'gain': 2.0}), name)


def test_a_missing_name_raises_naming_it():
    params = _view({'gain': 2.0})
    with pytest.raises(AttributeError, match=r"'gian'.*params\['gian'\]|params\['gian'\].*'gian'"):
        params.gian
    with pytest.raises(KeyError):
        params['gian']


def test_monte_carlo_per_path_dict_is_viewed_directly():
    resolved = {'offset': 1e-3}
    params = _view(resolved)
    assert params.offset == 1e-3
    resolved['offset'] = 2e-3
    assert params.offset == 2e-3


def test_an_empty_view_stands_in_for_no_params():
    params = _view(None)
    assert len(params) == 0 and list(params) == []
    with pytest.raises(AttributeError):
        params.gain
