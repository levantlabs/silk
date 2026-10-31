"""Silk runs record corner none and supply none from the public proofs (TASK-406).

Covers corner none, labels versus evidence, the run_corners warning,
supply none, and confirmation against a snapshot.
"""
import os
from pathlib import Path
import subprocess
import warnings

import pytest

from tests._mock_rawfile import write_complete_rawfile

from silk import Module, load_pdk, sources
from silk.circuit.ports import PortDir
from silk.elements import elements
from silk.exceptions import CharacterizationContextError
from silk.simulation import ConditionState, Transient
from silk.simulation.conditions import (
    OperatingConditions, ConditionSweep, _TemperatureRequest,
    confirm_conditions, observed_conditions, storage_conditions,
)
from silk.specs import Measurement
from silk.specs.spec import _measurement_slot_key
from tests._env import needs_ngspice, needs_pdk

NONE = ConditionState.NONE
_FIXTURE = Path(__file__).parent / 'fixtures/rawfiles/e_ng_tran.raw'


def _rc(tmp_path, *, name='rc'):
    """The ideal RC low-pass: no PDK device, a PULSE stimulus at dc=0."""
    dut = Module('sky130', str(tmp_path), f'{name}.cir', load_pdk('sky130'),
                 module_name=name)
    dut.define_module_ports({'a': PortDir.IN, 'b': PortDir.OUT})
    with dut:
        elements.resistor('r1', p='a', n='b', value=1e3)
        elements.capacitor('c1', p='b', n='0', value=1e-9)
    tb = dut.create_test('rc')
    tb.add_module_under_test({'a': 'vin', 'b': 'vout'})
    tb.add_source(sources.VoltageSource(
        'vs', positive='vin', negative='0', dc=0.0,
        pulse=sources.Pulse(v1=0.0, v2=1.0, td=0.0, tr=1e-9, tf=1e-9,
                            pw=5e-6, per=10e-6)))
    tb.add_sim_type(Transient(step=1e-8, stop=1e-5))
    return tb


def _with_spec(tb, signal):
    tb.add_metric('final', lambda c: float(c.load_signal(signal).y[-1]))
    tb.add_spec('final', 'final', '>', -1e9, key='final')
    return tb


def _nmos_bench(tmp_path):
    """A bench with one sky130 device, so its deck keeps the PDK includes."""
    dut = Module('sky130', str(tmp_path), 'amp.cir', load_pdk('sky130'),
                 module_name='amp')
    dut.define_module_ports({'d': PortDir.INOUT, 'g': PortDir.IN})
    dut.add(dut.pdk_details.nmos1p8('m1', d='d', g='g', s='0', b='0', w=1.0, l=0.15))
    tb = dut.create_test('amp')
    tb.add_module_under_test({'d': 'vdd', 'g': 'vg'})
    tb.add_source(sources.VoltageSource('vdd', positive='vdd', negative='0', dc=1.8))
    tb.add_source(sources.VoltageSource('vg', positive='vg', negative='0', dc=0.9))
    tb.add_sim_type(Transient(step=1e-9, stop=1e-8))
    return tb


@pytest.fixture
def launch(monkeypatch, mock_ngspice_version):
    """Stub the simulator: copy a fixture rawfile, log the requested TEMP."""
    state = {'temperatures': (27.0,), 'calls': 0}

    def run(argv, **kwargs):
        state['calls'] += 1
        raw = Path(argv[argv.index('-r') + 1])
        log = Path(argv[argv.index('-o') + 1])
        log.write_text(''.join(f'Doing analysis at TEMP = {t} and TNOM = 27.000000\n'
                               for t in state['temperatures']))
        write_complete_rawfile(raw, Path(argv[-1]).read_text(), template=_FIXTURE)
        return subprocess.CompletedProcess(argv, 0, stdout='', stderr='')

    monkeypatch.setattr(subprocess, 'run', run)
    return state


# ---------------------------------------------------------------------------
# Corner none and supply none at launch (stubbed simulator)
# ---------------------------------------------------------------------------

def test_pdk_free_run_records_corner_none_supply_none_and_profiles(tmp_path, launch):
    tb = _with_spec(_rc(tmp_path), 'out')
    tb.set_temperature(85)
    launch['temperatures'] = (85.0,)
    report = tb.run()
    c = report.operating_conditions
    assert (c.corner_kind, c.corner_basis, c.corner) == ('none', 'proven-no-pdk-device', None)
    assert (c.temperature_kind, c.temperature_c) == ('fixed', 85.0)
    assert (c.supply_kind, c.supply_basis, c.supply_voltage_v) == ('none', 'proven-no-supply', None)
    assert c.scalar('corner') is NONE and c.scalar('supply') is NONE
    ctx = report.characterization_context
    assert (ctx.corner, ctx.temperature, ctx.voltage) == (NONE, 85.0, NONE)
    # The requested corner (the PDK default here) stays the label.
    assert report.corner == 'tt'
    assert tb.netlist._last_sim_corner is NONE
    assert tb.netlist._last_sim_label == 'tt'
    # The Measurement takes the evidence, not the label.
    m = next(iter(tb._specs[0]['spec'].measurements.values()))
    assert (m.corner, m.temperature, m.voltage) == (NONE, 85.0, NONE)


def test_explicit_corner_on_pdk_free_deck_is_label_not_evidence(tmp_path, launch):
    tb = _rc(tmp_path)
    report = tb.run('ff')
    assert report.corner == 'ff'
    assert report.operating_conditions.corner_kind == 'none'
    assert report.operating_conditions.corner is None
    # The deck said so: no PDK library reached the simulator.
    assert '.lib' not in Path(report.run.deck_path).read_text()


def test_pdk_free_dc_source_keeps_supply_unknown(tmp_path, launch):
    tb = _rc(tmp_path)
    tb.add_source(sources.VoltageSource('bias', positive='vb', negative='0', dc=1.2))
    c = tb.run().operating_conditions
    assert c.corner_kind == 'none'
    assert (c.supply_kind, c.supply_basis) == ('unknown', 'unknown')


def test_configured_supply_on_pdk_free_deck_is_a_value(tmp_path, launch):
    tb = _rc(tmp_path)
    tb.add_source(sources.VoltageSource('vdd', positive='vdd', negative='0', dc=1.8))
    tb.set_supply_voltage(1.8, source='vdd')
    report = tb.run()
    c = report.operating_conditions
    assert (c.supply_kind, c.supply_voltage_v, c.supply_basis) == (
        'fixed', 1.8, 'configured-and-emitted')
    assert c.corner_kind == 'none'
    # Temperature 27 is backend-observed, so the run scopes (none, 27, 1.8).
    ctx = report.characterization_context
    assert (ctx.corner, ctx.temperature, ctx.voltage) == (NONE, 27.0, 1.8)


def test_pdk_run_records_configured_and_emitted_corner(tmp_path, launch, monkeypatch):
    tb = _nmos_bench(tmp_path)
    monkeypatch.setattr(tb.pdk_details, 'library_location', os.devnull)
    report = tb.run('ff')
    c = report.operating_conditions
    assert (c.corner_kind, c.corner_basis, c.corner) == ('fixed', 'configured-and-emitted', 'ff')
    assert report.corner == 'ff'
    # Supply as today: a DC source silk was not told about is unknown.
    assert (c.supply_kind, c.supply_basis) == ('unknown', 'unknown')
    assert tb.netlist._last_sim_corner == 'ff'


def test_temperature_sweep_is_configured_with_corner_from_proof(tmp_path, launch):
    from silk.simulation import DC
    tb = _rc(tmp_path)
    tb.add_sim_type(DC(source='temp', start=-40, stop=125, step=5))
    report = tb.run()
    c = report.operating_conditions
    assert (c.temperature_kind, c.temperature_basis) == ('sweep', 'configured')
    assert c.corner_kind == 'none' and c.supply_kind == 'none'
    assert report.characterization_context is None


# ---------------------------------------------------------------------------
# run_corners: label keys, evidence none, one warning
# ---------------------------------------------------------------------------

def test_run_corners_on_pdk_free_deck_warns_once_and_keeps_labels(tmp_path, launch):
    tb = _rc(tmp_path)
    with pytest.warns(UserWarning, match='no PDK device') as record:
        results = tb.run_corners(('tt', 'ff', 'ss'))
    assert len([w for w in record if 'no PDK device' in str(w.message)]) == 1
    assert list(results) == ['tt', 'ff', 'ss']
    for label, report in results.items():
        assert report.corner == label
        assert report.operating_conditions.corner_kind == 'none'
        assert Path(report.rawfile_path).parent.parent.name == label


@pytest.mark.parametrize('corners', [('tt',), ('tt', 'tt')])
def test_run_corners_with_one_distinct_corner_does_not_warn(tmp_path, launch, corners):
    tb = _rc(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        tb.run_corners(corners)


def test_run_corners_on_pdk_deck_does_not_warn(tmp_path, launch, monkeypatch):
    tb = _nmos_bench(tmp_path)
    monkeypatch.setattr(tb.pdk_details, 'library_location', os.devnull)
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        results = tb.run_corners(('tt', 'ff'))
    assert [r.operating_conditions.corner for r in results.values()] == ['tt', 'ff']


# ---------------------------------------------------------------------------
# Confirming declarations against none evidence (design section 3.5)
# ---------------------------------------------------------------------------

def test_reloaded_none_evidence_confirms_none_and_refuses_a_corner(tmp_path, launch):
    tb = _rc(tmp_path)
    tb.run('tt')
    assert tb.evaluate().corner == 'tt'          # omitted: the run's label
    assert tb.evaluate('tt').corner == 'tt'      # the label itself
    confirmed = tb.evaluate(NONE)                # a declaration of none
    assert confirmed.corner == 'tt'
    assert confirmed.operating_conditions.corner_kind == 'none'
    with pytest.raises(CharacterizationContextError,
                       match="corner='ss' conflicts with operating conditions"):
        tb.evaluate('ss')


def test_reloaded_fixed_evidence_refuses_a_none_declaration(tmp_path, launch, monkeypatch):
    tb = _nmos_bench(tmp_path)
    monkeypatch.setattr(tb.pdk_details, 'library_location', os.devnull)
    tb.run('tt')
    with pytest.raises(CharacterizationContextError, match='conflicts with operating conditions'):
        tb.evaluate(NONE)
    with pytest.raises(CharacterizationContextError, match='conflicts with operating conditions'):
        tb.evaluate('ss')


def _none_snapshot():
    return OperatingConditions('silk', 'ngspice', None, 'fixed', 85.0,
                               'configured-and-observed',
                               corner_kind='none', corner_basis='proven-no-pdk-device',
                               supply_kind='none', supply_basis='proven-no-supply')


def test_confirm_and_storage_conditions_read_scalars():
    snapshot = _none_snapshot()
    confirm_conditions(snapshot, corner=NONE, temperature=85.0, voltage=NONE)
    for kwargs in ({'corner': 'tt'}, {'voltage': 1.8}, {'temperature': NONE}):
        with pytest.raises(CharacterizationContextError, match='conflicts with operating conditions'):
            confirm_conditions(snapshot, **kwargs)
    assert storage_conditions(snapshot, None, None, None, None) == (NONE, 85.0, NONE)
    fixed = OperatingConditions('silk', 'ngspice', 'tt', 'fixed', 27.0)
    with pytest.raises(CharacterizationContextError):
        confirm_conditions(fixed, corner=NONE)
    assert storage_conditions(fixed, None, 'tt', None, None) == ('tt', 27.0, None)


def test_observed_conditions_takes_the_corner_proof():
    request = _TemperatureRequest(backend='ngspice', temperature=None, basis='unknown',
                                  sweep=ConditionSweep(-40.0, 125.0, 5.0), custom=(),
                                  opaque=False)
    swept = observed_conditions(request, 'ss', None, no_pdk_device=False)
    assert (swept.corner, swept.corner_basis, swept.temperature_basis) == (
        'ss', 'configured-and-emitted', 'configured')
    swept = observed_conditions(request, 'ss', None, no_pdk_device=True)
    assert (swept.corner, swept.corner_kind, swept.corner_basis) == (
        None, 'none', 'proven-no-pdk-device')


def test_unscoped_none_corner_measurement_key_is_a_tuple():
    m = Measurement(source='spice', value=1.0, corner=NONE)
    assert _measurement_slot_key(m) == ('spice', NONE)
    assert _measurement_slot_key(Measurement(source='spice', value=1.0, corner='none')) == 'spice@none'


def test_simulate_run_fallback_snapshot_takes_corner_kind_from_the_proof(tmp_path, monkeypatch):
    from silk.circuit.module_test import ModuleTestbench, _RunSpec
    tb = _rc(tmp_path)

    def simulate(corner=None, **kwargs):
        tb.netlist.operating_conditions = None
    monkeypatch.setattr(tb.netlist, 'simulate', simulate)
    spec = _RunSpec(netlist=tb.netlist, corner_str='ff', tb_dir=str(tmp_path),
                    invocation=None, test_name='rc', cache_key=None)
    result = ModuleTestbench._simulate_run(spec)
    assert result.conditions.corner_kind == 'none'
    assert result.conditions.corner_basis == 'proven-no-pdk-device'
    assert result.corner_str == 'ff'


# ---------------------------------------------------------------------------
# Real simulator
# ---------------------------------------------------------------------------

@pytest.mark.slow
@needs_ngspice
def test_ideal_rc_at_85c_profiles_as_none_85_none(tmp_path):
    tb = _with_spec(_rc(tmp_path), 'vout')
    tb.set_temperature(85)
    report = tb.run()
    c = report.operating_conditions
    assert (c.origin, c.backend) == ('silk', 'ngspice')
    assert (c.corner_kind, c.corner_basis) == ('none', 'proven-no-pdk-device')
    assert (c.temperature_kind, c.temperature_c, c.temperature_basis) == (
        'fixed', 85.0, 'configured-and-observed')
    assert (c.supply_kind, c.supply_basis) == ('none', 'proven-no-supply')
    ctx = report.characterization_context
    assert (ctx.corner, ctx.temperature, ctx.voltage) == (NONE, 85.0, NONE)
    assert report.corner == 'tt'
    m = next(iter(tb._specs[0]['spec'].measurements.values()))
    assert (m.corner, m.temperature, m.voltage) == (NONE, 85.0, NONE)


@pytest.mark.slow
@needs_ngspice
@needs_pdk
def test_sky130_run_records_corner_fixed_configured_and_emitted(tmp_path):
    tb = _nmos_bench(tmp_path)
    tb.set_supply_voltage(1.8, source='vdd')
    report = tb.run('tt')
    c = report.operating_conditions
    assert (c.corner_kind, c.corner, c.corner_basis) == ('fixed', 'tt', 'configured-and-emitted')
    assert (c.supply_kind, c.supply_voltage_v, c.supply_basis) == (
        'fixed', 1.8, 'configured-and-emitted')
    assert report.characterization_context.corner == 'tt'


# ---------------------------------------------------------------------------
# Storage (schema v5, TASK-409): the evidence is stored, the label accepted
# ---------------------------------------------------------------------------

def _stored_rc(tmp_path):
    dut = Module('sky130', str(tmp_path), 'rc.cir', load_pdk('sky130'),
                 module_name='rc', db_path=str(tmp_path / 'evidence.db'))
    dut.define_module_ports({'a': PortDir.IN, 'b': PortDir.OUT})
    dut.add(elements.resistor('r1', p='a', n='b', value=1e3))
    tb = dut.create_test('rc')
    tb.add_module_under_test({'a': 'vin', 'b': 'out'})
    tb.add_source(sources.VoltageSource(
        'vs', positive='vin', negative='0', dc=0.0,
        pulse=sources.Pulse(v1=0.0, v2=1.0, td=0.0, tr=1e-9, tf=1e-9,
                            pw=5e-6, per=10e-6)))
    tb.add_sim_type(Transient(step=1e-8, stop=1e-5))
    return _with_spec(tb, 'out')


def test_store_report_accepts_the_label_and_stores_corner_none(tmp_path, launch):
    import json
    from silk.exceptions import EvidenceMismatchError
    tb = _stored_rc(tmp_path)
    # A DC bias keeps the supply unknown, so the run stays unscoped.
    tb.add_source(sources.VoltageSource('bias', positive='vb', negative='0', dc=1.2))
    report = tb.run('ff')
    assert report.characterization_context is None
    mod = tb.module
    mod.create_circuit_version()
    assert mod.store_report(report, corner='ff')['specs_stored'] == 1
    rows = mod._tracker.db.conn.execute('SELECT * FROM simulation_results').fetchall()
    assert rows and all((r['corner'], r['corner_state']) == (None, 'none') for r in rows)
    marker = json.loads(rows[0]['conditions'])['silk_operating_conditions']
    assert marker['corner_kind'] == 'none' and marker['supply_kind'] == 'unknown'
    with pytest.raises(EvidenceMismatchError, match='conflicts with operating conditions'):
        mod.store_report(report, corner='ss')
    mod.close()


def test_store_report_stores_a_none_scoped_profile(tmp_path, launch):
    tb = _stored_rc(tmp_path)
    tb.set_temperature(85)
    launch['temperatures'] = (85.0,)
    report = tb.run()
    ctx = report.characterization_context
    assert (ctx.corner, ctx.temperature, ctx.voltage) == (NONE, 85.0, NONE)
    tb.module.create_circuit_version()
    tb.module.store_report(report)
    rows = tb.module._tracker.db.conn.execute(
        'SELECT corner, corner_state, temperature, voltage, voltage_state, profile_id '
        'FROM simulation_results').fetchall()
    assert rows
    for r in rows:
        assert (r['corner'], r['corner_state'], r['temperature'],
                r['voltage'], r['voltage_state']) == (None, 'none', 85.0, None, 'none')
        assert r['profile_id'] is not None
    tb.module.close()


# ---------------------------------------------------------------------------
# Measurement slot keys (TASK-422, design section 4.3)
# ---------------------------------------------------------------------------

def test_none_conditions_key_without_the_enum_text():
    from silk.simulation.conditions import ConditionState
    m = Measurement(source='spice', value=1.0, corner=NONE,
                    temperature=ConditionState.SWEEP, voltage=NONE)
    key = _measurement_slot_key(m)
    assert key == ('spice', NONE)
    assert not any(isinstance(part, str) and 'ConditionState' in part for part in key)
    scoped = Measurement(source='spice', value=1.0, corner=NONE, temperature=NONE,
                         voltage=NONE, profile_id='p1')
    assert _measurement_slot_key(scoped) == ('profile', 'p1', 'spice', NONE, NONE, NONE)


def test_existing_measurement_keys_are_unchanged():
    assert _measurement_slot_key(Measurement(source='spice', value=1.0)) == 'spice'
    assert _measurement_slot_key(Measurement(source='spice', value=1.0, corner='tt')) == 'spice@tt'
    assert _measurement_slot_key(Measurement(source='behavioral', value=1.0, corner='ss')) == 'behavioral@ss'
    fixed = Measurement(source='spice', value=1.0, corner='tt', temperature=27.0,
                        voltage=1.8, profile_id='p1')
    assert _measurement_slot_key(fixed) == ('profile', 'p1', 'spice', 'tt', 27.0, 1.8)


def test_store_report_none_corner_measurement_keys_by_tuple_and_reloads(tmp_path, launch):
    tb = _stored_rc(tmp_path)
    tb.add_source(sources.VoltageSource('bias', positive='vb', negative='0', dc=1.2))
    report = tb.run('ff')
    mod = tb.module
    spec = tb._specs[0]['spec']
    mod.create_circuit_version()
    mod.store_report(report, corner='ff')
    assert list(spec.measurements) == [('spice', NONE)]
    spec.measurements.clear()
    mod._tracker.restore_spec_measurements()
    assert list(spec.measurements) == [('spice', NONE)]
    assert spec.measurements[('spice', NONE)].corner is NONE
    mod.close()
