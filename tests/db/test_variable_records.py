"""TASK-402: a Variable reaches stored and reported records as its value.

A sweep changes a Variable run by run. Every record of a run — the
characterization profile, the devices rows, the structure identity, the
report outputs, the fault manifest and the faulted deck — must hold the
number the run used, never the Variable object, its repr, or the text of
an expression built from it.
"""
import json
import sqlite3

import pytest

from silk import Module, Variable, load_pdk
from silk.circuit.spec_report import SpecReport
from silk.db import Database
from silk.db.context import CharacterizationContext
from silk.db.identity import _canonicalize_characterization_profile
from silk.db.structure_tracker import CircuitStructureTracker
from silk.elements import elements
from silk.exceptions import CharacterizationContextError, CircuitError
from tests.db_snapshot_helpers import (
    create_building_snapshot,
    create_sealed_snapshot,
)

# Parameter names are legitimate keys; a value must never be the object's
# repr or an expression's text.
TRACES = ('Variable(', '_VarExpr', '2*r_load', '2*cload')


def _assert_no_traces(text):
    found = [t for t in TRACES if t in text]
    assert not found, f'Variable traces {found} in {text[:400]!r}'


@pytest.fixture(scope='module')
def pdk():
    return load_pdk('sky130')


def _context(parameters):
    return CharacterizationContext(
        owner_path='$', test_name='t', parameters=parameters,
        corner='tt', temperature=27.0, voltage=1.8)


# ---------------------------------------------------------------------------
# AC #1: declared test parameters
# ---------------------------------------------------------------------------

def test_context_records_variable_parameters_as_their_values():
    cl = Variable('cload', 1e-12)
    ctx = _context({'cload': cl, 'twice': 2 * cl, 'nested': {'a': [cl, 3]},
                    'label': 'x'})
    assert ctx.parameters == {'cload': 1e-12, 'twice': 2e-12,
                              'nested': {'a': [1e-12, 3]}, 'label': 'x'}
    assert type(ctx.parameters['cload']) is float
    # The record is the value at build time; a later assignment leaves it.
    cl.value = 5e-12
    assert ctx.parameters['cload'] == 1e-12


def test_variable_parameter_profile_equals_the_literal_and_tracks_the_value():
    cl = Variable('cload', 1e-12)
    a = _canonicalize_characterization_profile(_context({'cload': cl}))
    literal = _canonicalize_characterization_profile(_context({'cload': 1e-12}))
    cl.value = 5e-12
    b = _canonicalize_characterization_profile(_context({'cload': cl}))
    assert a.fingerprint == literal.fingerprint
    assert a.parameters_json == literal.parameters_json
    assert a.fingerprint != b.fingerprint


def test_unassigned_variable_parameter_raises_a_typed_error():
    with pytest.raises(CharacterizationContextError, match="'cload'"):
        _context({'cload': Variable('cload')})


def test_two_scoped_runs_at_two_values_store_two_profiles():
    cl = Variable('cload', 1e-12)
    with Database(':memory:') as db:
        version_id = create_sealed_snapshot(db)
        first = db.store_characterization_profile(version_id, _context({'cload': cl}))
        cl.value = 5e-12
        second = db.store_characterization_profile(version_id, _context({'cload': cl}))
        assert first.profile_id != second.profile_id
        assert dict(first.parameters) == {'cload': 1e-12}
        assert dict(second.parameters) == {'cload': 5e-12}
        rows = db.conn.execute(
            'SELECT * FROM characterization_profiles').fetchall()
        _assert_no_traces(json.dumps([tuple(r) for r in rows], default=str))


def test_testbench_context_and_report_hold_values_while_the_bench_stays_live(
        pdk, tmp_path):
    m = Module('PDK_skywater130', str(tmp_path), 'rc.cir', pdk_details=pdk,
               module_name='rc')
    cl = Variable('cload', 1e-12)
    m.add(elements.resistor('r1', p='inp', n='out', value=1e3))
    m.add(elements.capacitor('c1', p='out', n='gnd', value=cl))
    tb = m.create_test('t', parameters={'cload': cl})
    cl.value = 5e-12
    # TASK-400: the declared map stays live ...
    assert tb.parameters['cload'] is cl
    # ... and the run's records take its value.
    ctx = tb._build_context('tt', 27.0, 1.8)
    assert ctx.parameters == {'cload': 5e-12}
    assert _canonicalize_characterization_profile(ctx).parameters_json
    report = SpecReport(test_name='t', module_name='rc', metrics={},
                        parameters=tb.parameters)
    assert report.parameters == {'cload': 5e-12}


# ---------------------------------------------------------------------------
# AC #2: device rows and structure identity
# ---------------------------------------------------------------------------

def _dut(pdk, tmp_path, w, r, cl):
    m = Module('PDK_skywater130', str(tmp_path), 'dut.cir', pdk_details=pdk,
               module_name='dut')
    m.define_module_nets({'in': None, 'out': None, 'vss': None})
    m.add(pdk.nmos1p8('mn', d='out', g='in', s='vss', b='vss', w=w, l=0.15, m=1))
    m.add(elements.resistor('rl', p='out', n='vss', value=r))
    m.add(elements.capacitor('cl', p='out', n='vss', value=cl))
    return m


def _hash(module):
    t = CircuitStructureTracker(None, None)
    return t.compute_structure_hash(t.extract_structure(module))[0]


def test_passive_values_are_extracted_as_numbers(pdk, tmp_path):
    r = Variable('r_load', 1e3)
    m = _dut(pdk, tmp_path, 2.0, 2 * r, Variable('cload', 1e-12))
    passives = {p['name']: p['value']
                for p in CircuitStructureTracker(None, None)
                .extract_structure(m)['passives']}
    assert passives == {'rl': 2e3, 'cl': 1e-12}


def test_structure_hash_follows_a_passive_variable_like_a_width_variable(
        pdk, tmp_path):
    w = Variable('w_n', 2.0)
    r = Variable('r_load', 1e3)
    cl = Variable('cload', 1e-12)
    m = _dut(pdk, tmp_path, w, 2 * r, cl)
    base = _hash(m)
    w.value = 3.0
    assert _hash(m) != base
    w.value = 2.0
    assert _hash(m) == base
    r.value = 5e3
    assert _hash(m) != base, 'expression value must enter the structure hash'
    r.value = 1e3
    cl.value = 2e-12
    assert _hash(m) != base
    cl.value = 1e-12
    assert _hash(m) == base


def test_variable_circuit_shares_identity_with_its_literal_twin(pdk, tmp_path):
    var = _dut(pdk, tmp_path / 'a', Variable('w_n', 2.0),
               2 * Variable('r_load', 1e3), Variable('cload', 1e-12))
    lit = _dut(pdk, tmp_path / 'b', 2.0, 2000.0, 1e-12)
    assert _hash(var) == _hash(lit)


def test_unassigned_passive_variable_raises_circuit_error(pdk, tmp_path):
    m = _dut(pdk, tmp_path, 2.0, Variable('r_load'), 1e-12)
    with pytest.raises(CircuitError, match="'rl'"):
        CircuitStructureTracker(None, None).extract_structure(m)


def test_stored_device_rows_hold_numbers(pdk, tmp_path):
    m = _dut(pdk, tmp_path, Variable('w_n', 2.0), 2 * Variable('r_load', 1e3),
             Variable('cload', 1e-12))
    with Database(':memory:') as db:
        structure = CircuitStructureTracker(None, None).extract_structure(m)
        version_id = create_building_snapshot(db, structure=structure)
        tracker = CircuitStructureTracker(db, version_id)
        h, mh = tracker.compute_structure_hash(structure)
        tracker.store_structure(structure, h, mh)
        rows = {name: json.loads(props) for name, props in db.conn.execute(
            'SELECT name, properties FROM devices WHERE circuit_version_id = ?',
            (version_id,))}
    assert rows['rl']['value'] == 2000.0
    assert rows['cl']['value'] == 1e-12
    assert rows['mn']['w'] == '2'
    _assert_no_traces(json.dumps(rows))


# ---------------------------------------------------------------------------
# AC #3: written report outputs
# ---------------------------------------------------------------------------

def test_circuit_report_outputs_hold_no_variable_text(pdk, tmp_path):
    from silk.circuit.report import generate_report
    cl = Variable('cload', 1e-12)
    m = _dut(pdk, tmp_path, Variable('w_n', 2.0), 2 * Variable('r_load', 1e3), cl)
    rep = SpecReport(test_name='t', module_name='dut', metrics={'x': 1.0},
                     parameters={'cload': cl, 'twice': 2 * cl},
                     characterization_context=_context({'cload': cl}))
    cr = generate_report(m, auto_discover_config=False, spec_reports=[rep])
    cr.save_json(str(tmp_path / 'r.json'))
    cr.save_html(str(tmp_path / 'r.html'))
    _assert_no_traces((tmp_path / 'r.json').read_text())
    _assert_no_traces((tmp_path / 'r.html').read_text())
    _assert_no_traces(rep.summary())


def test_fault_manifest_and_report_hold_numbers(tmp_path):
    from silk.faults.report import CampaignReport
    from silk.faults.types import AppliedFault, FaultManifest
    sigma = Variable('sigma_w', 0.05)
    manifest = FaultManifest(seed=0, input_spec={'faults': [{'sigma': sigma}]},
                             applied=[AppliedFault(
                                 target='xdut.mn', fault_type='w_variation',
                                 declared_params={'sigma': sigma},
                                 realized_params={'w_nominal': 2 * sigma},
                                 netlist_comment='')])
    data = manifest.to_json()
    assert data['applied'][0]['declared_params'] == {'sigma': 0.05}
    assert data['applied'][0]['realized_params'] == {'w_nominal': 0.1}
    assert data['input_spec'] == {'faults': [{'sigma': 0.05}]}

    from silk.faults import CampaignResult, FaultSet, RunResult
    run = RunResult(run_id=0, seed=0, status='passed', manifest=manifest)
    result = CampaignResult('c', 'dut', 1, FaultSet(), 0, [run], str(tmp_path))
    CampaignReport(result).save_json(str(tmp_path / 'f.json'))
    text = (tmp_path / 'f.json').read_text()
    assert 'Variable(' not in text and 'sigma_w' not in text


def test_faulted_deck_writes_a_variable_width_as_its_value(pdk, tmp_path):
    # Through the public path since TASK-415 removed the entry adapter: a
    # faulted device is emitted by its own writer, which resolves a
    # Variable to its value, both when the fault leaves w alone and when it
    # perturbs w.
    import re
    import warnings
    from silk.faults import FaultSet, short_gd, w_variation
    w = Variable('w_n', 2.0)
    for i, fault in enumerate((short_gd('mn'), w_variation('mn', sigma=0.05))):
        mod = Module('sky130', str(tmp_path), f'fv{i}.cir', pdk_details=pdk)
        for net in ('out', 'in', 'vss'):
            mod.add_net(net)
        with mod:
            pdk.nmos1p8('mn', d='out', g='in', s='vss', b='vss', w=w,
                        l=0.15, m=1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            deck_path = mod.generate_netlist(fault_set=FaultSet([fault]), seed=7)
        with open(deck_path) as f:
            text = f.read()
        assert 'Variable' not in text and 'w_n' not in text
        widths = re.findall(r'\bw=([^\s]+)', text)
        assert widths, text
        for value in widths:
            float(value.rstrip('munpf'))
        if i == 0:
            assert any(v in ('2', '2u', '2.0') for v in widths), widths
