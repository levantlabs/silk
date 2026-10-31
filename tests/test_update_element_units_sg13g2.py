"""Catalog/emission validation only; SG13G2 remains a preview (TASK-598.2).

The shipped catalog has four MOS devices, no dimensional passives or area
parameters, and no W/L bounds. Count bounds are exercised without inventing
geometry limits. All tests need only packaged catalog data.
"""
import pytest

from silk.circuit.modules import Module
from silk.circuit.variable import Variable
from silk.exceptions import CircuitError
from silk.pdk import load_pdk
from silk.utils._utils import _parse_si


DEVICES = ('nmos_lv', 'pmos_lv', 'nmos_hv', 'pmos_hv')
MODES = (None, 'si', 'pdk')
PINS = dict(d='out', g='in', s='0', b='0')
DEFAULT_LENGTH = dict(nmos_lv=3.4e-7, pmos_lv=2.8e-7,
                      nmos_hv=3.4e-7, pmos_hv=2.8e-7)


def _build(tmp_path, kind, mode, initial=None, legacy=False):
    pdk = load_pdk('sg13g2', **({} if mode is None else {'user_units': mode}))
    mod = Module('sg13g2', str(tmp_path), 'unused.cir', pdk_details=pdk)
    if legacy:
        mod.add_element(pdk._devices[kind], 'mn', nets=dict(PINS),
                        parameters=initial)
    else:
        mod.add(getattr(pdk, kind)('mn', **PINS, **(initial or {})))
    return mod


def _device(mod):
    return mod.elem_dict['mn']['device']


def _params(mod, legacy=False):
    line = (mod.netlister.generate_device_instances(mod.elem_dict)
            if legacy else _device(mod).to_spice())
    return {k: _parse_si(v) for token in line.strip().splitlines()[-1].split() if '=' in token
            for k, v in [token.split('=', 1)]}


def _authored(mode, microns):
    # Independent physical expectation: SG13G2 takes metres, manifest authors
    # write microns, explicit SI and native PDK authors both write metres.
    return microns if mode is None else microns * 1e-6


def test_shipped_catalog_coverage_is_explicit():
    pdk = load_pdk('sg13g2')
    assert set(pdk._device_classes) == set(DEVICES)
    for name in DEVICES:
        cls = pdk._device_classes[name]
        assert set(cls._param_names) == {'w', 'l', 'm', 'ng'}
        for field in ('w', 'l'):
            constraint = cls._param_constraints[field]
            assert constraint.pdk_unit == 'base'
            assert getattr(constraint, 'quantity', None) in (None, 'length')
            assert str(constraint.minval) == str(constraint.maxval) == '-1'
        for field in ('m', 'ng'):
            constraint = cls._param_constraints[field]
            assert not constraint.pdk_unit
            assert float(constraint.minval) == 1
            assert float(constraint.resolution) == 1


@pytest.mark.parametrize('kind', DEVICES)
@pytest.mark.parametrize('mode', MODES, ids=['shipped-u', 'si', 'native-pdk'])
@pytest.mark.parametrize('supplied', [False, True], ids=['defaulted', 'supplied'])
def test_update_matches_constructor_and_assignment(tmp_path, kind, mode, supplied):
    initial = ({'w': _authored(mode, 1), 'l': _authored(mode, 0.5),
                'm': 2, 'ng': 3} if supplied else {})
    target = dict(w=_authored(mode, 4), l=_authored(mode, 0.8), m=4, ng=2)
    updated = _build(tmp_path, kind, mode, initial)
    assigned = _build(tmp_path, kind, mode, initial)
    built = _build(tmp_path, kind, mode, target)
    for _ in range(3):
        updated.update_element('mn', list(target), list(target.values()))
        for key, value in target.items():
            setattr(_device(assigned), key, value)
        for mod in (updated, assigned, built):
            assert _params(mod) == pytest.approx(
                dict(w=4e-6, l=0.8e-6, m=4, ng=2), rel=1e-12, abs=0)
        assert _device(updated).to_spice() == _device(assigned).to_spice()
        assert _device(updated).to_spice() == _device(built).to_spice()


@pytest.mark.parametrize('kind', DEVICES)
@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('field', ['w', 'l'])
def test_single_field_update_preserves_unrelated_defaults(tmp_path, kind, mode, field):
    mod = _build(tmp_path, kind, mode)
    before = _params(mod)
    assert before['w'] == pytest.approx(0.35e-6, rel=1e-12, abs=0)
    assert before['l'] == pytest.approx(DEFAULT_LENGTH[kind], rel=1e-12, abs=0)
    mod.update_element('mn', [field], [_authored(mode, 4)])
    after = _params(mod)
    assert after[field] == pytest.approx(4e-6, rel=1e-12, abs=0)
    assert {k: v for k, v in after.items() if k != field} == {
        k: v for k, v in before.items() if k != field}


@pytest.mark.parametrize('kind', DEVICES)
@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('field', ['w', 'l', 'm', 'ng'])
def test_live_variable_updates(tmp_path, kind, mode, field):
    mod = _build(tmp_path, kind, mode)
    dimensional = field in ('w', 'l')
    variable = Variable('live', value=_authored(mode, 4) if dimensional else 4)
    mod.update_element('mn', [field], [variable])
    assert _device(mod).params[field] is variable
    assert _params(mod)[field] == pytest.approx(4e-6 if dimensional else 4,
                                               rel=1e-12, abs=0)
    variable.value = _authored(mode, 6) if dimensional else 6
    assert _device(mod).params[field] is variable
    assert _params(mod)[field] == pytest.approx(6e-6 if dimensional else 6,
                                               rel=1e-12, abs=0)


@pytest.mark.parametrize('kind', DEVICES)
@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('field', ['m', 'ng'])
@pytest.mark.parametrize('bad,reason', [(0, 'minimum'), (1.5, 'integer')])
def test_declared_count_constraints(tmp_path, kind, mode, field, bad, reason):
    for path in ('update', 'assignment', 'constructor'):
        mod = _build(tmp_path, kind, mode, {field: bad} if path == 'constructor' else {})
        if path == 'update':
            mod.update_element('mn', [field], [bad])
        elif path == 'assignment':
            setattr(_device(mod), field, bad)
        with pytest.raises(CircuitError, match='invalid parameter value') as error:
            _params(mod)
        assert field in str(error.value)
        assert reason in str(error.value)
    mod = _build(tmp_path, kind, mode)
    mod.update_element('mn', [field], [2])
    assert _params(mod)[field] == 2


@pytest.mark.parametrize('kind', DEVICES)
@pytest.mark.parametrize('mode', MODES)
def test_string_literals_remain_verbatim(tmp_path, kind, mode):
    mod = _build(tmp_path, kind, mode)
    mod.update_element('mn', ['w', 'l', 'm', 'ng'], ['4u', '800n', '2', '3'])
    assert _device(mod).params['w'] == '4u'
    line = _device(mod).to_spice()
    assert 'w=4u' in line and 'l=800n' in line
    assert _params(mod) == pytest.approx(dict(w=4e-6, l=0.8e-6, m=2, ng=3),
                                        rel=1e-12, abs=0)


@pytest.mark.parametrize('kind', ['nmos_lv', 'pmos_hv'])
@pytest.mark.parametrize('mode', MODES)
def test_legacy_pdkdevice_updates(tmp_path, kind, mode):
    initial = dict(w=_authored(mode, 1), l=_authored(mode, 0.5), m=1, ng=1)
    mod = _build(tmp_path, kind, mode, initial, legacy=True)
    live = Variable('legacy_width', value=_authored(mode, 4))
    mod.update_element('mn', ['w', 'l', 'm', 'ng'],
                       [live, _authored(mode, 0.8), 2, 3])
    assert _device(mod).param_list['w'].value is live
    assert _params(mod, legacy=True) == pytest.approx(
        dict(w=4e-6, l=0.8e-6, m=2, ng=3), rel=1e-12, abs=0)
    live.value = _authored(mode, 6)
    assert _params(mod, legacy=True)['w'] == pytest.approx(6e-6, rel=1e-12, abs=0)
