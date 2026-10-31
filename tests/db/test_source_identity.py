"""DUT-owned sources must prevent reuse of electrically different evidence."""
import copy
import json

import pytest

from silk import Module, sources
from silk.circuit.variable import Variable
from silk.db import Database, DesignStage
from silk.db.identity import (
    _fingerprint_module_structure, _fingerprint_structure,
    _require_fingerprint_scheme, STRUCTURE_FINGERPRINT_SCHEME,
)
from silk.db.models import Fingerprint
from silk.db.structure_tracker import CircuitStructureTracker
from silk.elements import elements
from silk.exceptions import CircuitError, EvidenceMismatchError, EvidenceNotFoundError, UnsupportedFingerprintSchemeError


def module(tmp_path, *, dc=None, name='bias', db=None, variants=False):
    m = Module('none', str(tmp_path), name + '.cir', pdk_details=None,
               module_name=name, db_path=None if db is None else str(db), variants=variants)
    m.define_module_ports({'out': 'out', 'vss': 'ground', 'in': 'in'})
    m.add(elements.resistor('load', p='out', n='vss', value=1000))
    if dc is not None:
        m.add_source(sources.VoltageSource('bias', 'out', 'vss', dc=dc))
    return m


def structure(m):
    return CircuitStructureTracker(None, None).extract_structure(m)


def identities(m):
    t = CircuitStructureTracker(None, None)
    return (_fingerprint_structure(t.extract_structure(m))[0],
            _fingerprint_module_structure(t._extract_module_structure(m)))


FACTORIES = [
    lambda v: sources.VoltageSource('stim', 'out', 'vss', dc=v),
    lambda v: sources.CurrentSource('stim', 'out', 'vss', dc=v),
    lambda v: sources.VCVS('stim', 'out', 'vss', 'in', 'vss', v),
    lambda v: sources.VCCS('stim', 'out', 'vss', 'in', 'vss', v),
    lambda v: sources.CCVS('stim', 'out', 'vss', v, 'sense'),
    lambda v: sources.CCCS('stim', 'out', 'vss', v, 'sense'),
    lambda v: sources.BehavioralVoltage('stim', 'out', 'vss', f'{v}*V(in)'),
    lambda v: sources.BehavioralCurrent('stim', 'out', 'vss', f'{v}*V(in)'),
]


@pytest.mark.parametrize('factory', FACTORIES, ids=['V', 'I', 'E', 'G', 'H', 'F', 'BV', 'BI'])
@pytest.mark.parametrize('depth', [0, 1, 2])
def test_every_source_family_changes_circuit_and_definition_identity(tmp_path, factory, depth):
    a, b = module(tmp_path), module(tmp_path)
    a.add_source(factory(1))
    b.add_source(factory(2))
    assert a.sig_sources_dict['stim'].to_spice() != b.sig_sources_dict['stim'].to_spice()
    for level in range(depth):
        parents = [module(tmp_path, name=f'layer{level}') for _ in range(2)]
        for parent, child in zip(parents, (a, b)):
            parent.add_module(child, 'u', nets={'out': 'out', 'vss': 'vss', 'in': 'in'})
        a, b = parents
    assert all(x != y for x, y in zip(identities(a), identities(b)))
    assert identities(a) == identities(copy.deepcopy(a))


@pytest.mark.parametrize('factory', FACTORIES[:2], ids=['V', 'I'])
def test_independent_source_variables_match_literals_and_change_with_values(tmp_path, factory):
    v = Variable('bias', 1.)
    a, b = module(tmp_path), module(tmp_path)
    a.add_source(factory(1.))
    b.add_source(factory(v))
    assert identities(a) == identities(b)
    before = identities(b)
    v.value = 2.
    assert all(x != y for x, y in zip(before, identities(b)))


@pytest.mark.parametrize('edit', ['output', 'control', 'reference', 'type', 'ac'])
def test_connections_type_reference_and_ac_are_identity(tmp_path, edit):
    a = module(tmp_path, dc=1)
    a.add_source(sources.VCVS('amp', 'out', 'vss', 'in', 'vss', 2))
    a.add_source(sources.CCCS('copy', 'out', 'vss', 2, 'sense'))
    b = copy.deepcopy(a)
    if edit == 'output':
        b.sig_sources_dict['bias'].positive = 'in'
    elif edit == 'control':
        b.sig_sources_dict['amp'].ctrl_pos = 'vss'
    elif edit == 'reference':
        b.sig_sources_dict['copy'].ctrl_vname = 'other'
    elif edit == 'type':
        b.sig_sources_dict.pop('bias')
        b.add_source(sources.CurrentSource('bias', 'out', 'vss', dc=1))
    else:
        b.sig_sources_dict['bias'].ac = 2
    assert all(x != y for x, y in zip(identities(a), identities(b)))


@pytest.mark.parametrize('wave', ['sine', 'pwl', 'pulse', 'exponential', 'sffm', 'am', 'trnoise', 'trrandom', 'bits', 'prbs'])
def test_waveform_fields_are_identity(tmp_path, wave):
    def shape(value):
        return {
            'sine': lambda: sources.Sine(vampl=value, freq=1),
            'pwl': lambda: sources.PWL([(0, 0), (1, value)], r=0, td=0),
            'pulse': lambda: sources.Pulse(0, value, tr=.1, tf=.1, pw=.2, per=1),
            'exponential': lambda: sources.Exponential(0, value, 0, .1, 1, .1),
            'sffm': lambda: sources.SFFM(0, value, 1, 2, 1),
            'am': lambda: sources.AM(value, .5, 1, 2),
            'trnoise': lambda: sources.TransientNoise(value, .1),
            'trrandom': lambda: sources.RandomSource(1, .1, 0, value, 0),
            'bits': lambda: sources.BitPattern('01', v1=0, v2=1, ui=1, tr=.1, tf=.1, repeat=value),
            'prbs': lambda: sources.PRBS(7, length=8, seed=value, v1=0, v2=1, ui=1, tr=.1, tf=.1),
        }[wave]()
    a, b = module(tmp_path), module(tmp_path)
    a.add_source(sources.VoltageSource('stim', 'out', 'vss', waveform=shape(1)))
    b.add_source(sources.VoltageSource('stim', 'out', 'vss', waveform=shape(2)))
    assert all(x != y for x, y in zip(identities(a), identities(b)))


@pytest.mark.parametrize('kind', ['sine', 'pwl', 'E', 'G', 'H', 'F'])
def test_source_precision_and_variable_resolution(tmp_path, kind):
    def build(value):
        m = module(tmp_path)
        if kind == 'sine':
            src = sources.VoltageSource('stim', 'out', 'vss', sine=sources.Sine(vampl=value, freq=1))
        elif kind == 'pwl':
            src = sources.VoltageSource('stim', 'out', 'vss', pwl=sources.PWL([(0, 0), (1, value)]))
        else:
            src = FACTORIES[['E', 'G', 'H', 'F'].index(kind) + 2](value)
        m.add_source(src)
        return m
    v = Variable('level', 1.0000001)
    a, b = build(1.0000001), build(v)
    assert identities(a) == identities(b)
    before = identities(b)
    v.value = 1.0000002
    assert all(x != y for x, y in zip(before, identities(b)))
    assert identities(build(v * 2)) == identities(build(2.0000004))


@pytest.mark.parametrize('bad', [Variable('unset'), float('nan'), float('inf'), object()])
def test_bad_source_values_refuse_persistent_identity(tmp_path, bad):
    m = module(tmp_path, dc=bad)
    with pytest.raises(CircuitError, match='source|Source'):
        identities(m)


def test_add_remove_and_nested_mutation_propagate_to_ancestors(tmp_path):
    leaf = module(tmp_path, dc=1, name='leaf')
    sibling = module(tmp_path, name='sibling')
    parent = module(tmp_path, name='parent')
    parent.add_module(leaf, 'left', nets={'out': 'out', 'vss': 'vss', 'in': 'in'})
    parent.add_module(sibling, 'right', nets={'out': 'out', 'vss': 'vss', 'in': 'in'})
    root = module(tmp_path, name='root')
    root.add_module(parent, 'stage', nets={'out': 'out', 'vss': 'vss', 'in': 'in'})
    placed = root.elem_dict['stage']['device']
    before = identities(root), identities(placed), identities(placed.elem_dict['right']['device'])
    placed.elem_dict['left']['device'].sig_sources_dict['bias'].dc = 2
    assert identities(root) != before[0]
    assert identities(placed) != before[1]
    assert identities(placed.elem_dict['right']['device']) == before[2]
    empty = module(tmp_path)
    baseline = identities(empty)
    empty.add_source(sources.VoltageSource('bias', 'out', 'vss', dc=1))
    assert identities(empty) != baseline
    empty.sig_sources_dict.pop('bias')
    assert identities(empty) == baseline


def test_external_registered_bench_is_not_dut_structure(tmp_path):
    from silk.circuit.module_test import ModuleTestbench
    from silk.db.identity import _canonicalize_characterization_profile
    m = module(tmp_path, dc=1)
    baseline = identities(m)
    tb = ModuleTestbench(m, 'level', parameters={})
    tb.add_module_under_test({'out': 'out', 'vss': '0', 'in': 'in'})
    tb.add_source(sources.VoltageSource('stimulus', 'out', '0', dc=1))
    profile = _canonicalize_characterization_profile(tb._build_context('tt', 27, 1.8)).fingerprint
    campaign = tb._structure_hash()
    tb.netlist.sig_sources_dict['stimulus'].dc = 2
    assert identities(m) == baseline
    assert _canonicalize_characterization_profile(tb._build_context('tt', 27, 1.8)).fingerprint == profile
    assert tb._structure_hash() != campaign
    m.sig_sources_dict['bias'].dc = 2
    assert identities(m) != baseline


def test_public_database_reproduction_and_bound_mutation(tmp_path):
    path = tmp_path / 'db.sqlite'
    first, changed, same = [module(tmp_path, dc=v, db=path) for v in (1, 2, 1)]
    v1 = first.create_circuit_version()
    first.store_result('out_V', 1., 'V')
    assert same.create_circuit_version() == v1
    assert same.query_results(parameter='out_V')[0].value == 1.
    assert changed.create_circuit_version() != v1
    with pytest.raises(EvidenceMismatchError):
        changed.query_results(parameter='out_V')
    first.sig_sources_dict['bias'].dc = 3
    with pytest.raises(EvidenceMismatchError):
        first.query_results(parameter='out_V')
    with pytest.raises(EvidenceMismatchError):
        first.store_result('out_V', 3., 'V')


def test_root_nested_facts_reopen_promotion_and_lossy_export(tmp_path):
    from silk.db.circuit_import import CircuitExporter, CircuitImporter
    path = tmp_path / 'db.sqlite'
    root = module(tmp_path, dc=1, db=path, name='root')
    root.add_module(module(tmp_path, dc=2, name='child'), 'u', nets={'out': 'out', 'vss': 'vss', 'in': 'in'})
    root.create_circuit_version()
    db = root._tracker.db
    summary = db.get_circuit_summary('root')
    rows = [r for r in summary['devices'] if r['type'] == 'silk_source']
    assert {r['name'] for r in rows} == {'bias', 'u.bias'}
    facts = {r['name']: json.loads(r['properties']) for r in rows}
    assert all(p['kind'] == 'source' for p in facts.values())
    assert facts['bias']['parameters']['dc'] == '1'
    assert facts['u.bias']['parameters']['dc'] == '2'
    assert len(summary['devices']) == 4  # two loads, two source placements
    assert summary['modules'][0]['device_count'] == 0  # ideal load is a separate passive
    assert summary['modules'][0]['structure']['sources'][0]['parameters']['dc'] == '2'
    promoted = root.promote('stable', lock=False)
    promoted_rows = db.conn.execute('SELECT device_id,name,properties FROM devices WHERE circuit_version_id=? AND type=?', (promoted, 'silk_source')).fetchall()
    assert {r['name']: json.loads(r['properties']) for r in promoted_rows} == facts
    assert {r['device_id'] for r in promoted_rows}.isdisjoint(r['device_id'] for r in rows)
    root.close()
    reopened = Database(str(path))
    assert {r['name']: json.loads(r['properties']) for r in reopened.get_circuit_summary('root')['devices'] if r['type'] == 'silk_source'} == facts
    export = tmp_path / 'projection.json'
    CircuitExporter(reopened).export_to_json('root', str(export))
    data = json.loads(export.read_text())['circuits'][0]
    assert not any(r['type'] == 'silk_source' for r in data['transistors'])
    assert {r['name'] for r in data['sources']} == {'bias', 'u.bias'}
    from silk.db.circuit_import import _import_structure
    assert {r['name'] for r in _import_structure(data)['sources']} == {'bias', 'u.bias'}
    reopened.close()


def test_old_v3_equal_digest_cannot_authorize_v4_reuse(tmp_path):
    current = identities(module(tmp_path))[0]
    assert current.scheme == 'silk.structure/v4'
    with pytest.raises(UnsupportedFingerprintSchemeError, match='Preserve.*rebuild'):
        _require_fingerprint_scheme(Fingerprint('silk.structure/v3', current.digest), STRUCTURE_FINGERPRINT_SCHEME)


@pytest.mark.parametrize('field', ['seed', 'length', 'order', 'repeat'])
def test_discrete_waveform_fields_stay_exact(tmp_path, field):
    m = module(tmp_path)
    if field == 'repeat':
        wave = sources.BitPattern('01', v1=0, v2=1, ui=1, tr=.1, tf=.1, repeat=2**60)
    else:
        wave = sources.PRBS(7, length=8, seed=1, v1=0, v2=1, ui=1, tr=.1, tf=.1)
    m.add_source(sources.VoltageSource('stim', 'out', 'vss', waveform=wave))
    before = identities(m)
    setattr(wave, field, {'seed': 2, 'length': 9, 'order': 9, 'repeat': 2**60 + 1}[field])
    assert all(x != y for x, y in zip(before, identities(m)))


@pytest.mark.parametrize('where', ['root', 'module', 'direct_module_fresh', 'direct_module_present'])
@pytest.mark.parametrize('tamper', ['value', 'pop', 'empty'])
def test_pending_source_payload_tamper_is_refused(tmp_path, where, tamper):
    m = module(tmp_path, dc=1, db=':memory:')
    if where != 'root':
        m.add_module(module(tmp_path, dc=2, name='child'), 'u', nets={'out': 'out', 'vss': 'vss', 'in': 'in'})
    tracker = m._ensure_tracker()
    payload, hashes, identity, _ = tracker._prepare_snapshot(m)
    if where.startswith('direct_module'):
        local = copy.deepcopy(payload['modules']['child'])
        digest = local.pop('hash')
        present = where == 'direct_module_present'
        if present:
            tracker.db.insert_or_get_module('child', 'child', digest, structure=local)
        if tamper == 'value':
            local['sources'][0]['parameters']['dc'] = '99'
        elif tamper == 'pop':
            local.pop('sources')
        else:
            local['sources'] = []
        with pytest.raises(EvidenceMismatchError):
            tracker.db.insert_or_get_module('child', 'child', digest, structure=local)
        assert tracker.db.conn.execute('SELECT COUNT(*) FROM modules').fetchone()[0] == int(present)
        return
    version = tracker.db.create_circuit('pending', identity=identity)
    target = payload if where == 'root' else payload['modules']['child']
    if tamper == 'value':
        target['sources'][0]['parameters']['dc'] = '99'
    elif tamper == 'pop':
        target.pop('sources')
    else:
        target['sources'] = []
    with pytest.raises(EvidenceMismatchError):
        CircuitStructureTracker(tracker.db, version).store_structure(payload, identity.structure.digest, hashes)
    assert tracker.db.conn.execute('SELECT COUNT(*) FROM devices').fetchone()[0] == 0


@pytest.mark.parametrize('hand_built', [False, True])
def test_direct_source_free_payload_inserts_and_reuses(tmp_path, hand_built):
    if hand_built:
        payload = {'devices': [{'name': 'M1', 'type': 'nmos'}],
                   'nets': ['out', 'gnd'], 'submodules': {}, 'passives': []}
    else:
        definition = CircuitStructureTracker(None, None)._extract_module_structure(module(tmp_path))
        payload = {key: definition[key] for key in ('devices', 'nets', 'submodules', 'passives')}
        assert payload['passives']
    assert not payload.get('sources')
    digest = _fingerprint_module_structure(payload).digest
    with Database(':memory:') as db:
        first = db.insert_or_get_module('bias', 'bias', digest, structure=payload)
        assert db.insert_or_get_module('bias', 'bias', digest, structure=payload) == first
        assert db.conn.execute('SELECT COUNT(*) FROM modules').fetchone()[0] == 1


@pytest.mark.parametrize('payload', [{}, {'devices': [], 'nets': ['out'], 'submodules': {}, 'passives': []}])
@pytest.mark.parametrize('present', [False, True])
def test_direct_source_free_payload_must_match_even_on_reuse(payload, present):
    with Database(':memory:') as db:
        if present:
            db.insert_or_get_module('bias', 'bias', 'hand-built-digest')
        with pytest.raises(EvidenceMismatchError):
            db.insert_or_get_module('bias', 'bias', 'hand-built-digest', structure=payload)
        assert db.conn.execute('SELECT COUNT(*) FROM modules').fetchone()[0] == int(present)


@pytest.mark.parametrize('field,value,cause', [
    ('devices', None, TypeError),
    ('devices', ['M1'], AttributeError),
    ('submodules', [], AttributeError),
    ('nets', ['out', 1], TypeError),
], ids=['devices-none', 'devices-strings', 'submodules-list', 'nets-mixed'])
@pytest.mark.parametrize('present', [False, True])
def test_malformed_module_payload_has_typed_field_error_without_db_changes(field, value, cause, present):
    valid = {'devices': [{'name': 'M1', 'type': 'nmos'}],
             'nets': ['out', 'gnd'], 'submodules': {}, 'passives': []}
    digest = _fingerprint_module_structure(valid).digest
    malformed = copy.deepcopy(valid)
    malformed[field] = value
    with Database(':memory:') as db:
        if present:
            db.insert_or_get_module('bias', 'bias', digest, structure=valid)
        before = list(db.conn.iterdump())
        with pytest.raises(CircuitError, match=field) as caught:
            db.insert_or_get_module('bias', 'bias', digest, structure=malformed)
        assert isinstance(caught.value.__cause__, cause)
        assert list(db.conn.iterdump()) == before


@pytest.mark.parametrize('action', ['spec_read', 'spec_restore', 'spec_write', 'current_read', 'current_write', 'characterization_restore'])
def test_source_mutation_rejects_spec_current_and_restore_evidence(tmp_path, action):
    from silk.circuit.spec_report import SpecReport
    from silk.circuit.modules_results import CurrentAnnotationReport
    from silk.db.context import CharacterizationContext
    m = module(tmp_path, dc=1, db=tmp_path / 'evidence.sqlite')
    spec = m.declare_spec('Level', 'out_V', '>=', .5, units='V', key='level')
    version = m.create_circuit_version()
    context = CharacterizationContext(owner_path='$', test_name='level', parameters={}, corner='tt', temperature=27., voltage=1.8)
    report = SpecReport('level', 'bias', {'out_V': 1.}, owner_path='$', parameters={}, characterization_context=context)
    report.check('Level', 'out_V', '>=', .5, spec_key='level', owner_path='$')
    m.store_report(report)
    snapshot = m._tracker.db._snapshot_row(version)
    m._tracker.db.replace_net_current_set(version, corner='tt', temperature=27., voltage=1.8,
        structure_scheme=snapshot['structure_scheme'], structure_hash=snapshot['structure_hash'], context=context,
        rows=[{'net_name': 'out', 'pin_id': '', 'avg_a': .001, 'rms_a': None, 'peak_a': None, 'source': 'spice', 'current_vector': None}])
    assert m.restore_spec_measurements() == 1
    assert m.select_characterization_evidence(context=context).current_set.currents[0].avg_a == .001
    m.sig_sources_dict['bias'].dc = 2
    operations = {
        'spec_read': lambda: m.select_characterization_evidence(context=context),
        'spec_restore': lambda: m.restore_spec_measurements(),
        'spec_write': lambda: m.store_report(report),
        'current_read': lambda: m.load_net_currents('tt', force=True),
        'current_write': lambda: m.store_net_currents(CurrentAnnotationReport(results_file='unused.raw', analysis='transient', window=()), 'tt'),
        'characterization_restore': lambda: m.restore_characterization_evidence(context=context, force=True),
    }
    with pytest.raises(EvidenceMismatchError):
        operations[action]()


def test_historical_v3_snapshot_rows_stay_tagged_and_refuse_rebuild(tmp_path):
    path = tmp_path / 'history.sqlite'
    m = module(tmp_path, db=path)
    version = m.create_circuit_version()
    m.store_result('out_V', 1., 'V')
    digest = identities(m)[0].digest
    m._tracker.db.conn.execute("UPDATE circuit_versions SET structure_scheme='silk.structure/v3' WHERE version_id=?", (version,))
    m._tracker.db.conn.commit()
    m.close()
    rebuilt = module(tmp_path, db=path)
    with pytest.raises(UnsupportedFingerprintSchemeError, match='rebuild'):
        rebuilt.create_circuit_version()
    row = rebuilt._tracker.db.conn.execute('SELECT structure_scheme,structure_hash FROM circuit_versions WHERE version_id=?', (version,)).fetchone()
    assert tuple(row) == ('silk.structure/v3', digest)
    assert rebuilt._tracker.db.conn.execute('SELECT value FROM simulation_results').fetchone()[0] == 1.


def test_b_source_does_not_treat_symbolic_variable_expression_as_resolved_literal(tmp_path):
    m = module(tmp_path)
    variable = Variable('gain', 1.)
    m.add_source(sources.BehavioralVoltage('behavior', 'out', 'vss', variable * 2))
    # B emission uses literal expression text; it does not resolve this object.
    with pytest.raises(CircuitError, match='expression'):
        identities(m)


@pytest.mark.parametrize('bad', [object(), float('nan'), Variable('unresolved')])
@pytest.mark.parametrize('kind', ['module', 'circuit'])
def test_pending_source_facts_cannot_hash_unsupported_values_as_repr(tmp_path, bad, kind):
    m = module(tmp_path, dc=1)
    tracker = CircuitStructureTracker(None, None)
    payload = tracker._extract_module_structure(m) if kind == 'module' else tracker.extract_structure(m)
    payload['sources'][0]['parameters']['dc'] = bad
    fingerprint = _fingerprint_module_structure if kind == 'module' else _fingerprint_structure
    with pytest.raises(CircuitError, match='source|Source'):
        fingerprint(payload)


def test_invalid_source_variable_arithmetic_has_typed_refusal(tmp_path):
    m = module(tmp_path, dc=1 / Variable('denominator', 0.))
    with pytest.raises(CircuitError, match='source.*division'):
        identities(m)
