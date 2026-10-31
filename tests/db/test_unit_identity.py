"""Unit-aware structure identity uses emission semantics, without simulation."""

import copy
import json

import pytest

from silk import Module, Port, load_pdk
from silk.circuit.variable import Variable
from silk.db.circuit_import import CircuitExporter
from silk.db.identity import _fingerprint_module_structure, _fingerprint_structure
from silk.db.structure_tracker import CircuitStructureTracker
from silk.exceptions import CircuitError, EvidenceMismatchError


@pytest.fixture(params=['factory', 'legacy'])
def api(request):
    return request.param


@pytest.fixture(params=['sky130', 'gf180mcu'])
def catalog(request):
    return request.param


@pytest.fixture
def build(tmp_path):
    modules = []

    def make(api, catalog='sky130', mode='u', params=None, *, device=None,
             name='amp', db_path=None, reverse=False):
        pdk = load_pdk(catalog, user_units=mode, library_location='/fake/lib')
        mod = Module(catalog, str(tmp_path), f'{name}.cir', pdk_details=pdk,
                     module_name=name, db_path=db_path)
        modules.append(mod)
        ports = [('a', Port.analog_input()), ('b', Port.analog_output()),
                 ('vss', Port.ground())]
        mod.define_module_ports(dict(reversed(ports) if reverse else ports))
        device = device or ('nmos1p8' if catalog == 'sky130' else 'nmos3p3')
        cls = getattr(pdk, device)
        nets = ({'d': 'b', 'g': 'a', 's': 'vss', 'b': 'vss'}
                if 'd' in cls._pin_names else {'a': 'a', 'k': 'vss'})
        params = dict(params or {})
        if reverse:
            params = dict(reversed(list(params.items())))
        if api == 'factory':
            mod.add(cls('mn', **nets, **params))
        else:
            mod.add_element(cls, 'mn', nets=nets, parameters=params)
        return mod

    yield make
    for mod in modules:
        mod.close()


def structure(mod):
    return CircuitStructureTracker(None, None).extract_structure(mod)


def identities(mod):
    tracker = CircuitStructureTracker(None, None)
    return (_fingerprint_structure(tracker.extract_structure(mod))[0],
            _fingerprint_module_structure(tracker._extract_module_structure(mod)))


def test_identical_numbers_with_different_units_are_distinct(build, api, catalog):
    micron = build(api, catalog, 'u', {'w': 1.0, 'l': 0.3})
    metres = build(api, catalog, 'si', {'w': 1.0, 'l': 0.3})
    assert all(a != b for a, b in zip(identities(micron), identities(metres)))


@pytest.mark.parametrize('length', [0.3, 0.66, 1.22])
def test_equivalent_units_have_identical_fingerprints(build, api, catalog, length):
    micron = build(api, catalog, 'u', {'w': 2.2, 'l': length})
    metres = build(api, catalog, 'si', {'w': 2.2e-6, 'l': length * 1e-6})
    assert micron.generate_netlist() == metres.generate_netlist()
    assert identities(micron) == identities(metres)


def test_roundoff_and_declaration_order_are_canonical(build, api):
    original = build(api, params={'w': 1.0, 'l': 0.22, 'm': 2})
    for _ in range(2):
        rebuilt = build(api, mode='si', params={'w': 1e-6, 'l': 0.22e-6, 'm': 2},
                        reverse=True)
        assert identities(original) == identities(rebuilt)


def test_defaults_and_explicit_physical_defaults_agree(build, api, catalog):
    defaults = build(api, catalog)
    metre_defaults = build(api, catalog, 'si')
    # GF180's width default is a string literal, deliberately left literal;
    # the numeric length default is 280 nm. No guessing/evaluating strings.
    length_um = 0.15 if catalog == 'sky130' else 0.28
    explicit_um = build(api, catalog, 'u', {'l': length_um})
    explicit_si = build(api, catalog, 'si', {'l': length_um * 1e-6})
    for mod in (metre_defaults, explicit_um, explicit_si):
        assert defaults.generate_netlist() == mod.generate_netlist()
        assert identities(defaults) == identities(mod)


def test_assigned_variables_match_numeric_values_without_mutation(build, api, catalog):
    variable = build(api, catalog, 'si', {
        'w': Variable('width', value=2.2e-6),
        'l': Variable('length', value=0.66e-6),
    })
    plain = build(api, catalog, 'u', {'w': 2.2, 'l': 0.66})
    dev = variable.elem_dict['mn']['device']
    if api == 'factory':
        values = dict(dev.params)
        supplied = set(dev._caller_supplied)
        scales = dict(type(dev)._param_scales)
        defaults = copy.deepcopy(type(dev)._param_defaults)
    else:
        objects = dict(dev.param_list)
        values = {key: obj.value for key, obj in objects.items()}
        constraints = copy.deepcopy({key: vars(obj) for key, obj in objects.items()})
    pdk_state = copy.deepcopy({key: val for key, val in vars(variable.pdk_details).items()
                               if key in ('user_units', 'library_location', 'model_list')})
    before = variable.generate_netlist()
    assert identities(variable) == identities(plain)
    assert variable.generate_netlist() == before == plain.generate_netlist()
    assert {key: val for key, val in vars(variable.pdk_details).items()
            if key in pdk_state} == pdk_state
    if api == 'factory':
        assert all(dev.params[key] is val for key, val in values.items())
        assert dev._caller_supplied == supplied
        assert type(dev)._param_scales == scales
        assert type(dev)._param_defaults == defaults
    else:
        assert all(dev.param_list[key] is obj for key, obj in objects.items())
        assert all(dev.param_list[key].value is val for key, val in values.items())
        for key, obj in objects.items():
            assert {k: v for k, v in vars(obj).items() if k != 'value'} == {
                k: v for k, v in constraints[key].items() if k != 'value'}
    assert values['w'].value == 2.2e-6
    assert values['l'].value == 0.66e-6


@pytest.mark.parametrize('literal', ['1u', 'width_param', '{width_param*2}', '1e-05'])
def test_literals_bypass_units_and_dimensionless_mutation_matters(build, api, catalog, literal):
    params = {'w': literal, 'l': 'length_param', 'm': 2}
    micron = build(api, catalog, 'u', params)
    metres = build(api, catalog, 'si', params)
    before = micron.generate_netlist()
    assert identities(micron) == identities(metres)
    assert structure(micron)['transistors'][0]['parameters']['w'] == literal
    assert micron.generate_netlist() == before == metres.generate_netlist()
    changed = build(api, catalog, 'si', {**params, 'm': 3})
    assert all(a != b for a, b in zip(identities(micron), identities(changed)))


@pytest.mark.parametrize(('catalog', 'device', 'um', 'si', 'expected'), [
    ('sky130', 'diode_std', {'area': 0.22, 'perim': 2.2},
     {'area': 0.22e-12, 'perim': 2.2e-6}, {'area': '0.22', 'perim': '2.2'}),
    ('gf180mcu', 'diode_nd2ps_3p3', {'area': 2.2, 'pj': 6.6},
     {'area': 2.2e-12, 'pj': 6.6e-6}, {'area': '2.2p', 'pj': '6.6u'}),
])
def test_all_unit_bearing_parameters_use_quantity_metadata(build, api, catalog, device,
                                                          um, si, expected):
    micron = build(api, catalog, 'u', um, device=device)
    metres = build(api, catalog, 'si', si, device=device)
    assert micron.generate_netlist() == metres.generate_netlist()
    assert identities(micron) == identities(metres)
    params = structure(micron)['transistors'][0]['parameters']
    assert {key: params[key] for key in expected} == expected


def wrap(tmp_path, child, name, *, db_path=None):
    # Deliberately different from the legacy leaf's owning PDK/unit context.
    parent = Module('none', str(tmp_path), f'{name}.cir', module_name=name,
                    pdk_details=None, db_path=db_path)
    parent.define_module_ports({'a': Port.analog_input(), 'b': Port.analog_output(),
                                'vss': Port.ground()})
    parent.add_module(child, 'u', nets={'a': 'a', 'b': 'b', 'vss': 'vss'})
    return parent


def test_nested_leaf_units_propagate_to_every_ancestor(build, api, catalog, tmp_path):
    original = build(api, catalog, 'u', {'w': 2.2, 'l': 0.66})
    equivalent = build(api, catalog, 'si', {'w': 2.2e-6, 'l': 0.66e-6})
    changed = build(api, catalog, 'si', {'w': 4.4e-6, 'l': 0.66e-6})
    for name in ('middle', 'root', 'outer'):
        original, equivalent, changed = [wrap(tmp_path, mod, name)
                                          for mod in (original, equivalent, changed)]
        assert identities(original) == identities(equivalent)
        assert all(a != b for a, b in zip(identities(original), identities(changed)))
        assert len(structure(original)['transistors']) == 1
        assert original.generate_netlist() == equivalent.generate_netlist()


def test_equivalent_definition_reuse_preserves_emitted_text_guard(build, api, tmp_path):
    first = build(api, params={'w': 1.0, 'l': 0.22})
    equivalent = build(api, mode='si', params={'w': 1e-6, 'l': 0.22e-6})
    different = build(api, mode='si', params={'w': 2e-6, 'l': 0.22e-6})
    parent = wrap(tmp_path, first, 'parent')
    parent.add_module(equivalent, 'second', nets={'a': 'a', 'b': 'b', 'vss': 'vss'})
    assert parent.generate_netlist().count('.subckt amp ') == 1
    parent.add_module(different, 'third', nets={'a': 'a', 'b': 'b', 'vss': 'vss'})
    with pytest.raises(CircuitError, match='two structurally different circuits'):
        parent.generate_netlist()
    assert parent.elem_dict['third']['device'] is different


def test_sqlite_reuse_and_reopened_evidence_isolation(build, api, catalog, tmp_path):
    path = str(tmp_path / 'units.db')
    original = build(api, catalog, 'u', {'w': 2.2, 'l': 0.66}, db_path=path)
    first = original.create_circuit_version()
    original.store_result('sentinel', 0.666, 'V/V')
    original.close()
    equivalent = build(api, catalog, 'si', {'w': 2.2e-6, 'l': 0.66e-6}, db_path=path)
    assert equivalent.create_circuit_version() == first
    assert [r.value for r in equivalent.query_results(parameter='sentinel')] == [0.666]
    equivalent.close()
    different = build(api, catalog, 'si', {'w': 2.2, 'l': 0.66}, db_path=path)
    second = different.create_circuit_version()
    assert second != first
    with pytest.raises(EvidenceMismatchError, match='incompatible structure'):
        different.query_results(parameter='sentinel')
    different.close()
    reopened = build(api, catalog, 'si', {'w': 2.2, 'l': 0.66}, db_path=path)
    assert reopened.create_circuit_version() == second
    with pytest.raises(EvidenceMismatchError, match='incompatible structure'):
        reopened.query_results(parameter='sentinel')


@pytest.mark.parametrize(('catalog', 'params', 'expected'), [
    ('sky130', {'w': 1e-6, 'l': 0.22e-6}, {'w': '1', 'l': '0.22', 'm': '1'}),
    ('gf180mcu', {'w': 2.2e-6, 'l': 0.66e-6}, {'w': '2.2u', 'l': '660n', 'm': '1'}),
    ('gf180mcu', {'w': 'width_param', 'l': '280n'},
     {'w': 'width_param', 'l': '280n', 'm': '1'}),
])
def test_stored_and_exported_parameters_are_pdk_native_text(build, api, catalog,
                                                           params, expected, tmp_path):
    leaf = build(api, catalog, 'si', params, db_path=str(tmp_path / 'store.db'))
    before = leaf.generate_netlist()
    leaf.create_circuit_version()
    db = leaf._tracker.db
    assert json.loads(db.conn.execute('SELECT properties FROM devices').fetchone()[0]) == expected
    projection = tmp_path / 'projection.json'
    CircuitExporter(db).export_to_json('amp', str(projection))
    transistor = json.loads(projection.read_text())['circuits'][0]['transistors'][0]
    assert {key: transistor[key] for key in ('width', 'length', 'multiplier')} == {
        'width': expected['w'], 'length': expected['l'], 'multiplier': expected['m']}
    assert leaf.generate_netlist() == before

    parent = wrap(tmp_path, leaf, 'parent', db_path=str(tmp_path / 'nested.db'))
    try:
        parent.create_circuit_version()
        payload = json.loads(parent._tracker.db.conn.execute(
            "SELECT structure FROM modules WHERE module_name = 'amp'").fetchone()[0])
        assert payload['devices'][0]['parameters'] == expected
        assert _fingerprint_module_structure(payload) == identities(leaf)[1]
    finally:
        parent.close()


@pytest.mark.parametrize('extractor', ['extract_structure', '_extract_module_structure'])
def test_normalization_failure_is_actionable_not_partial(build, api, extractor, monkeypatch):
    mod = build(api, mode='si', params={'w': 1e-6, 'l': 0.22e-6})
    dev = mod.elem_dict['mn']['device']
    if api == 'factory':
        monkeypatch.setitem(type(dev)._param_scales, 'w', object())
    else:
        monkeypatch.setattr(dev.param_list['w'], 'quantity', 'unknown-quantity')
    with pytest.raises(CircuitError, match=r"mn.*w.*[Pp]arameter|mn.*w.*[Pp][Dd][Kk]"):
        getattr(CircuitStructureTracker(None, None), extractor)(mod)


@pytest.mark.parametrize('kind', ['numeric', 'default', 'literal'])
def test_extraction_preserves_raw_values_and_pdk_bookkeeping(build, api, catalog, kind):
    params = {'numeric': {'w': 2.2e-6, 'l': 0.66e-6}, 'default': {},
              'literal': {'w': 'width_param', 'l': '280n'}}[kind]
    mod = build(api, catalog, 'si', params)
    dev = mod.elem_dict['mn']['device']
    pdk = mod.pdk_details
    pdk_before = (pdk.user_units, pdk.library_location, pdk.model_list)
    before = mod.generate_netlist()
    if api == 'factory':
        raw = dict(dev.params)
        supplied = set(dev._caller_supplied)
        scales = dict(type(dev)._param_scales)
    else:
        raw = {key: dict(vars(value)) for key, value in dev.param_list.items()}
        objects = dict(dev.param_list)
    identities(mod)
    assert mod.generate_netlist() == before
    assert (pdk.user_units, pdk.library_location, pdk.model_list) == pdk_before
    if api == 'factory':
        assert dev.params == raw
        assert dev._caller_supplied == supplied
        assert type(dev)._param_scales == scales
    else:
        assert {key: vars(value) for key, value in dev.param_list.items()} == raw
        assert all(dev.param_list[key] is value for key, value in objects.items())


@pytest.mark.parametrize('mode', ['n', 'pdk'])
def test_other_supported_unit_modes_use_same_physical_identity(build, api, catalog, mode):
    original = build(api, catalog, 'u', {'w': 2.2, 'l': 0.66})
    if mode == 'n':
        params = {'w': 2200, 'l': 660}
    elif catalog == 'gf180mcu':
        params = {'w': 2.2e-6, 'l': 0.66e-6}
    else:
        params = {'w': 2.2, 'l': 0.66}
    rebuilt = build(api, catalog, mode, params)
    assert original.generate_netlist() == rebuilt.generate_netlist()
    assert identities(original) == identities(rebuilt)
