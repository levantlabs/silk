"""update_element follows assignment semantics for factory devices (TASK-598).

``update_element`` used to write ``AnalogElement.params`` directly, skipping
the caller-supplied bookkeeping that ``mn.w = v`` maintains. Emission unit-
converts and constraint-checks exactly the caller-supplied parameters, so a
defaulted parameter updated through ``update_element`` was emitted unscaled
(``w=4u`` under SI units) and never validated.
"""
import pytest

from silk.circuit.modules import Module
from silk.circuit.variable import Variable
from silk.exceptions import CircuitError
from silk.pdk import load_pdk

_UNITS = [('si', 4e-6), ('u', 4.0), ('pdk', 4.0), (None, 4.0)]
_NETS = dict(d='a', g='b', s='0', b='0')


_OUT = {}


@pytest.fixture(autouse=True)
def _out_dir(tmp_path):
    _OUT['dir'] = str(tmp_path)


def _build(units, **params):
    kw = {} if units is None else {'user_units': units}
    pdk = load_pdk('sky130', **kw)
    mod = Module('PDK_skywater130', _OUT['dir'], 't.cir', pdk_details=pdk)
    mod.add(pdk.nmos1p8('mn', **_NETS, **params))
    return mod


def _dev(mod):
    return mod.elem_dict['mn']['device']


def _line(mod):
    return _dev(mod).to_spice().strip()


@pytest.mark.parametrize('units,w', _UNITS)
@pytest.mark.parametrize('initial', [{}, {'w': 1.0, 'l': 0.15}])
def test_update_matches_construction_and_assignment(units, w, initial):
    scale = 1e-6 if units == 'si' else 1.0
    l = 0.5 * scale

    built = _build(units, w=w, l=l)

    updated = _build(units, **{k: v * scale for k, v in initial.items()})
    updated.update_element('mn', ['w', 'l'], [w, l])

    assigned = _build(units, **{k: v * scale for k, v in initial.items()})
    _dev(assigned).w = w
    _dev(assigned).l = l

    assert 'w=4 l=0.5' in _line(built)
    assert _line(updated) == _line(built) == _line(assigned)


@pytest.mark.parametrize('units,w', _UNITS)
def test_repeated_updates_never_double_scale_and_defaults_stay(units, w):
    mod = _build(units)
    untouched = _line(mod)
    assert 'l=0.15' in untouched
    for _ in range(3):
        mod.update_element('mn', ['w'], [w])
        assert 'w=4 ' in _line(mod) and 'l=0.15' in _line(mod)


def test_pinned_si_default_width_emits_4():
    mod = _build('si')
    mod.update_element('mn', ['w'], [4e-6])
    assert ' w=4 ' in _line(mod)


@pytest.mark.parametrize('units,bad', [('u', -1.0), ('si', -1e-6), (None, -1.0),
                                       ('u', 0.01)])
def test_updated_geometry_is_validated_like_assignment(units, bad):
    mod = _build(units)
    mod.update_element('mn', ['w'], [bad])
    with pytest.raises(CircuitError, match='invalid parameter value'):
        _line(mod)

    assigned = _build(units)
    _dev(assigned).w = bad
    with pytest.raises(CircuitError, match='invalid parameter value'):
        _line(assigned)


def test_updated_count_must_be_integer_valid_controls_pass():
    mod = _build('u')
    mod.update_element('mn', ['m'], [1.5])
    with pytest.raises(CircuitError, match='integer'):
        _line(mod)
    mod.update_element('mn', ['m'], [2])
    assert 'm=2' in _line(mod)
    mod.update_element('mn', ['w'], [1.0])
    assert ' w=1 ' in _line(mod)


def test_update_does_not_write_pins():
    mod = _build('u')
    before = _line(mod)
    mod.update_element('mn', ['d'], ['other'])
    assert _dev(mod).nets['d'] == 'a'
    assert _line(mod) == before


@pytest.mark.parametrize('units,w1,w2', [('si', 4e-6, 6e-6), ('u', 4.0, 6.0)])
def test_variable_identity_and_linkage_survive(units, w1, w2):
    mod = _build(units)
    var = Variable('wv', value=w1)
    mod.update_element('mn', ['w'], [var])
    assert _dev(mod).params['w'] is var
    assert ' w=4 ' in _line(mod)
    var.value = w2
    assert _dev(mod).params['w'] is var
    assert ' w=6 ' in _line(mod)


def test_structure_hash_and_schema_agree_with_assignment():
    from silk.circuit.circuit_elements import _module_structure_hash
    updated = _build('si')
    updated.update_element('mn', ['w'], [4e-6])
    assigned = _build('si')
    _dev(assigned).w = 4e-6
    assert _module_structure_hash(updated) == _module_structure_hash(assigned)
    assert updated.export_schema() == assigned.export_schema()


def test_string_literal_and_ideal_element_updates_keep_semantics():
    mod = _build('u')
    mod.update_element('mn', ['w'], ['2'])
    assert _dev(mod).params['w'] == '2'

    from silk.elements.passives import Resistor
    r = Resistor('r1', 'a', 'b', 1e3)
    mod.add(r)
    mod.update_element('r1', ['value'], [2e3])
    assert mod.elem_dict['r1']['device'].params['value'] == 2e3


def _footprint(mod):
    from silk.layout.area_model import compute_device_area
    dev = _dev(mod)
    return (mod.pdk_details.device_geometry_um(dev),
            compute_device_area(dev, 2.5, pdk=mod.pdk_details))


@pytest.mark.parametrize('units,w', _UNITS)
def test_footprint_agrees_across_build_paths(units, w):
    scale = 1e-6 if units == 'si' else 1.0
    want = ((4.0, 0.15, 1), pytest.approx(1.5))

    def check(mod):
        geom, area = _footprint(mod)
        assert geom == pytest.approx(want[0]) and area == want[1]

    check(_build(units, w=w, l=0.15 * scale))             # constructor

    assigned = _build(units)
    _dev(assigned).w = w                                   # direct assignment
    check(assigned)

    updated = _build(units)
    for _ in range(2):                                     # repeated update
        updated.update_element('mn', ['w'], [w])
    check(updated)

    live = _build(units)
    var = Variable('wv', value=w * 1.5)
    live.update_element('mn', ['w'], [var])
    var.value = w                                          # live linkage
    check(live)
