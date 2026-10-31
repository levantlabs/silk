"""The JSON projection carries passives and sources; the importer never skips a record."""
import json

import pytest

from silk import Module, Port, load_pdk, sources
from silk.db import Database
from silk.db.circuit_import import CircuitExporter, CircuitImporter
from silk.elements import elements
from silk.exceptions import DatabaseError


def _tracked(tmp_path, with_source=False):
    pdk = load_pdk('sky130', library_location='/fake/lib')
    m = Module('sky130', str(tmp_path), 'amp.cir', pdk_details=pdk,
               module_name='amp', db_path=str(tmp_path / 'src.db'))
    m.define_module_ports({'a': Port.analog_input(), 'b': Port.analog_output(),
                           'vss': Port.ground()})
    m.add(pdk.nmos1p8('mn', d='b', g='a', s='vss', b='vss', w=4.0, l=0.5))
    m.add(elements.resistor('r1', p='b', n='vss', value=1000))
    m.add(elements.capacitor('c1', p='b', n='vss', value=0))
    m.add(elements.inductor('l1', p='a', n='b', value=1e-9))
    if with_source:
        m.add_source(sources.VoltageSource('vbias', 'a', 'vss', dc=0.9))
        m.add_source(sources.CurrentSource('ibias', 'b', 'vss', dc=1e-6))
    m.create_circuit_version()
    return m


def _devices(db):
    summary = db.get_circuit_summary('amp')
    return {d['name']: (d['type'], d['model'], json.loads(d['properties']))
            for d in summary['devices']}


def _roundtrip(tmp_path, m):
    out = tmp_path / 'p.json'
    CircuitExporter(m._tracker.db).export_to_json('amp', str(out))
    dst = Database(str(tmp_path / 'dst.db'))
    CircuitImporter(dst).import_from_json(str(out))
    return out, dst


def test_passives_and_transistor_survive_export_and_import(tmp_path):
    m = _tracked(tmp_path)
    out, dst = _roundtrip(tmp_path, m)
    src = _devices(m._tracker.db)
    got = _devices(dst)
    assert set(got) == set(src) == {'mn', 'r1', 'c1', 'l1'}
    for name in ('r1', 'c1', 'l1'):
        assert got[name] == src[name]       # kind, value (0 included), nets
    assert got['c1'][2]['value'] == 0
    assert got['mn'][2]['width'] == src['mn'][2]['w'] == '4'   # text kept as stored
    data = json.loads(out.read_text())['circuits'][0]
    assert {p['name'] for p in data['passives']} == {'r1', 'c1', 'l1'}
    assert [t['name'] for t in data['transistors']] == ['mn']


def test_sources_survive_export_and_import(tmp_path):
    m = _tracked(tmp_path, with_source=True)
    _out, dst = _roundtrip(tmp_path, m)
    src = {k: v for k, v in _devices(m._tracker.db).items() if v[0] == 'silk_source'}
    got = {k: v for k, v in _devices(dst).items() if v[0] == 'silk_source'}
    assert set(src) == {'vbias', 'ibias'}
    assert got == src


@pytest.mark.parametrize('bad, field', [
    ({'name': 'x', 'kind': 'Resistor', 'pos': 'a', 'neg': 'b'}, 'value'),
    ({'name': 'x', 'kind': 'Resistor', 'value': 1, 'neg': 'b'}, 'pos'),
    ({'kind': 'Resistor', 'value': 1, 'pos': 'a', 'neg': 'b'}, 'name'),
])
def test_unusable_passive_raises_naming_record_and_field(tmp_path, bad, field):
    f = tmp_path / 'bad.json'
    f.write_text(json.dumps({'circuits': [{'name': 'amp', 'passives': [bad],
                                           'transistors': [], 'nets': []}]}))
    with Database(str(tmp_path / 'd.db')) as db:
        with pytest.raises(DatabaseError, match=field):
            CircuitImporter(db).import_from_json(str(f))


def test_transistor_without_dimensions_raises_instead_of_skipping(tmp_path):
    f = tmp_path / 'bad.json'
    f.write_text(json.dumps({'circuits': [{'name': 'amp', 'nets': [], 'transistors': [
        {'name': 'm1', 'type': 'nmos', 'width': None, 'length': 0.5}]}]}))
    with Database(str(tmp_path / 'd.db')) as db:
        with pytest.raises(DatabaseError, match=r"m1.*width"):
            CircuitImporter(db).import_from_json(str(f))


def test_spice_imported_voltage_and_current_sources_round_trip(tmp_path):
    netlist = tmp_path / 'amp.cir'
    netlist.write_text("Vsupply vdd 0 DC 1.8V AC 1\nIbias vdd out DC 0.0001A\n"
                       "R1 out 0 1k\n")
    with Database(str(tmp_path / 'one.db')) as one:
        CircuitImporter(one).import_from_spice(str(netlist))
        out = tmp_path / 'p.json'
        CircuitExporter(one).export_to_json('amp', str(out))
        first = _devices(one)
    with Database(str(tmp_path / 'two.db')) as two:
        CircuitImporter(two).import_from_json(str(out))
        second = _devices(two)
    assert {k: v[0] for k, v in first.items()} == {
        'Vsupply': 'voltage_source', 'Ibias': 'current_source',
        'R1': 'resistor'}
    assert first['Vsupply'][2] == {'positive': 'vdd', 'negative': '0',
                                   'dc': 1.8, 'ac': 1.0}
    assert first['Ibias'][2]['dc'] == 0.0001
    assert second == first


@pytest.mark.parametrize('bad, field', [
    ({'name': 'vbias', 'type': 'VoltageSource',
      'parameters': {'dc': '0.9', 'ac': None}}, 'nets'),
    ({'name': 'vbias', 'kind': 'voltage', 'dc': 0.9}, 'positive'),
    ({'name': 'vbias', 'kind': 'voltage', 'positive': 'a', 'dc': 0.9},
     'negative'),
])
def test_malformed_source_raises_naming_record_and_field(tmp_path, bad, field):
    f = tmp_path / 'bad.json'
    f.write_text(json.dumps({'circuits': [{'name': 'amp', 'sources': [bad],
                                           'transistors': [], 'nets': []}]}))
    with Database(str(tmp_path / 'd.db')) as db:
        with pytest.raises(DatabaseError, match=rf"vbias.*{field}"):
            CircuitImporter(db).import_from_json(str(f))
