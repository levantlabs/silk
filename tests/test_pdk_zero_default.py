"""A catalog parameter whose explicit default is 0 keeps it (TASK-602)."""
import pytest

from silk.elements.factory import _build_constraints
from silk.pdk.basepdk import PDKDevice


def _spec(**nf):
    return {'paramList': {'nf': {'minval': 1, 'maxval': 10, **nf}}}


@pytest.mark.parametrize('zero', [0, 0.0])
def test_factory_keeps_zero_default(zero):
    assert _build_constraints(_spec(default=zero))['nf'].default == zero


def test_factory_absent_default_falls_back_to_minval():
    assert _build_constraints(_spec())['nf'].default == 1


@pytest.mark.parametrize('zero', [0, 0.0])
def test_pdkdevice_keeps_zero_default(zero):
    dev = PDKDevice.from_dict('x', {'paramList': _spec(default=zero)['paramList'],
                                    'pinList': {'d': 'drain'}, 'PDKDevice': 'x'})
    assert dev.param_list['nf'].default == zero


def test_pdkdevice_absent_default_falls_back_to_minval():
    dev = PDKDevice.from_dict('x', {'paramList': _spec()['paramList'],
                                    'pinList': {'d': 'drain'}, 'PDKDevice': 'x'})
    assert dev.param_list['nf'].default == 1

