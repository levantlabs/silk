"""Temperature configuration must describe executed evidence, including failures."""
from dataclasses import FrozenInstanceError
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
from unittest.mock import Mock

import pytest

from tests._mock_rawfile import write_complete_rawfile

from silk import Module, load_pdk, sources
from silk.elements import elements
from silk.exceptions import (
    CharacterizationContextError, EvidenceMismatchError, NotSupportedError,
    SimulationError,
)
from silk.simulation import DC, Transient
from silk.signals.rawfile import RawfileError
from tests._env import needs_ngspice, needs_pdk


@pytest.fixture
def bench(tmp_path):
    mod = Module('sky130', str(tmp_path), 'dut.cir', load_pdk('sky130'),
                 module_name='load', db_path=str(tmp_path / 'evidence.db'))
    mod.define_module_ports({'out': 'out', 'vss': 'ground'})
    mod.add(elements.resistor('rload', p='out', n='vss', value=1000))
    tb = mod.create_test('level')
    tb.add_module_under_test({'out': 'out', 'vss': '0'})
    tb.add_source(sources.VoltageSource('supply', positive='out', negative='0', dc=1.8))
    tb.add_sim_type(Transient(step=1e-6, stop=2e-6))
    tb.add_metric('supply', lambda c: float(c.load_signal('out').y[-1]))
    tb.add_spec('supply present', 'supply', '>', 0.5, key='supply')
    yield tb
    mod.close()


@pytest.fixture
def launch(monkeypatch, bench, mock_ngspice_version):
    # The stub supplies a fixture rawfile and never loads transistor models.
    monkeypatch.setattr(bench.pdk_details, 'library_location', os.devnull)
    state = {'temperatures': (27.0,), 'calls': 0}
    fixture = Path(__file__).parent / 'fixtures/rawfiles/e_ng_tran.raw'

    def run(argv, **kwargs):
        state['calls'] += 1
        raw = Path(argv[argv.index('-r') + 1])
        log = Path(argv[argv.index('-o') + 1])
        log.write_text(''.join(f'Doing analysis at TEMP = {t} and TNOM = 27.000000\n'
                               for t in state['temperatures']))
        write_complete_rawfile(raw, Path(argv[-1]).read_text(), template=fixture)
        return subprocess.CompletedProcess(argv, 0, stdout='', stderr='')

    monkeypatch.setattr(subprocess, 'run', run)
    return state


@pytest.mark.parametrize('value,error', [(True, TypeError), ('85', TypeError),
                                        (complex(2), TypeError), (float('nan'), ValueError),
                                        (float('inf'), ValueError), (-273.15, ValueError),
                                        (-300, ValueError)])
def test_typed_temperature_domain_is_atomic(bench, value, error):
    bench.set_temperature(85)
    with pytest.raises(error):
        bench.set_temperature(value)
    assert bench.netlist.simulator_options['temp'] == 85


def test_setter_native_share_one_value_and_clear_only_temp(bench):
    assert bench.set_temperature(85) is bench
    bench.netlist.set_simulator_options(temp='35', tnom=20)
    assert bench.netlist.set_temperature(-40) is bench.netlist
    bench_netlist_deck_path = bench.netlist.generate_netlist()
    deck = Path(bench_netlist_deck_path).read_text()
    assert deck.count('temp=') == 1 and 'tnom=20' in deck
    bench.set_temperature(None)
    assert bench.netlist.simulator_options == {'tnom': 20}
    # TASK-168: the DUT owns this test, so it may hold their default -- and
    # doing so still never writes into the test's own executing netlist; the
    # test opted out above and stays out.
    assert bench.module.set_temperature(85) is bench.module
    assert bench.netlist.simulator_options == {'tnom': 20}
    assert bench.netlist._temperature_explicit is True


@pytest.mark.parametrize('native', [False, True])
def test_fixed_run_snapshot_and_partial_storage(bench, launch, native):
    launch['temperatures'] = (85, 85)
    if native:
        bench.netlist.set_simulator_options(temp='85')
    else:
        bench.set_temperature(85)
    report = bench.run()
    conditions = report.operating_conditions
    assert conditions.temperature_c == 85
    assert conditions.temperature_basis == ('native-and-observed' if native else 'configured-and-observed')
    assert bench.netlist.operating_conditions is conditions
    assert report.characterization_context is None
    with pytest.raises(FrozenInstanceError):
        conditions.temperature_c = 35
    bench.module.create_circuit_version()
    bench.module.set_corner('ss', temperature=-40, voltage=9)
    bench.module.store_report(report, conditions={'author': 'test'})
    rows = bench.module._tracker.db.conn.execute('SELECT * FROM simulation_results').fetchall()
    assert rows
    for row in rows:
        # The bench is ideal: corner none is stored as NULL, never the
        # 'tt' label (TASK-406); the marker below carries the kind.
        assert (row['corner'], row['temperature'], row['voltage'], row['profile_id']) == (None, 85, None, None)
        assert json.loads(row['conditions'])['silk_operating_conditions'] == conditions.to_dict()
    stored = bench.module._tracker.db.conn.execute('SELECT * FROM spec_measurements').fetchone()
    assert stored['temperature'] == 85 and stored['voltage'] is None
    summary = report.summary()
    assert 'corner none (' in summary and 'supply ' in summary
    assert 'primary supply declared' not in summary


@pytest.mark.parametrize('observations', [(), ('garbage',), (35,), (85, 35)])
def test_failed_confirmation_invalidates_live_results_before_metrics(bench, launch, observations):
    first = bench.run()
    original = first.operating_conditions
    bench.set_temperature(85)
    launch['temperatures'] = observations
    extractor = Mock(side_effect=AssertionError('must not extract'))
    bench.add_metric('forbidden', extractor)
    with pytest.raises(SimulationError, match='Configured TEMP'):
        bench.run()
    assert bench.netlist.operating_conditions is None
    assert bench.netlist.characterization_context is None
    assert not bench.netlist.sim_results_loaded
    assert bench.results_file is None
    extractor.assert_not_called()
    assert first.operating_conditions is original
    assert original.temperature_c == 27


@pytest.mark.parametrize('observations', [(), ('garbage',), (27, 35)])
def test_unknown_not_defaulted_by_storage(bench, launch, observations):
    launch['temperatures'] = observations
    report = bench.run(voltage=1.8)
    assert report.operating_conditions.temperature_kind == 'unknown'
    assert report.characterization_context is None
    bench.module.create_circuit_version()
    bench.module.set_corner('ss', temperature=85, voltage=5)
    bench.module.store_report(report)
    rows = bench.module._tracker.db.conn.execute('SELECT * FROM simulation_results').fetchall()
    assert all(r['temperature'] is None and r['voltage'] == 1.8 for r in rows)
    before = len(rows)
    for kwargs in ({'temperature': 27}, {'voltage': 5}, {'corner': 'ss'},
                   {'conditions': {'silk_operating_conditions': {'invented': True}}}):
        with pytest.raises(EvidenceMismatchError):
            bench.module.store_report(report, **kwargs)
    assert bench.module._tracker.db.conn.execute('SELECT count(*) FROM simulation_results').fetchone()[0] == before


def test_legacy_assertion_requires_setup_before_allocation(bench, launch):
    first = bench.run(run_label='good')
    path = bench.results_file
    for setup in (None, 35):
        bench.set_temperature(setup)
        with pytest.warns(DeprecationWarning), pytest.raises(CharacterizationContextError, match='set_temperature'):
            bench.run(temperature=85, run_label='bad')
    assert launch['calls'] == 1
    assert bench.results_file == path
    assert bench.netlist.operating_conditions is first.operating_conditions
    bench.set_temperature(85)
    launch['temperatures'] = (85,)
    with pytest.warns(DeprecationWarning):
        report = bench.run(temperature=85)
    assert report.characterization_context is None


@pytest.mark.parametrize('mode', ['overwrite', 'new'])
def test_prepared_temperature_changes_require_regeneration(bench, launch, mode):
    c = bench.netlist
    c.set_output_options(mode)
    c.set_temperature(85)
    c.generate_netlist()
    c.set_temperature(35)
    with pytest.raises(SimulationError, match='generate_netlist'):
        c.run_simulator()
    assert launch['calls'] == 0
    c.generate_netlist()
    launch['temperatures'] = (35,)
    c.run_simulator()
    assert c.operating_conditions.temperature_c == 35
    c.generate_netlist()
    with open(c._run_files.netlist_path, 'a') as f:
        f.write('\n.temp 85\n')
    with pytest.raises(SimulationError, match='generate_netlist'):
        c.run_simulator()


@pytest.mark.parametrize('custom', ['.include "unknown.lib"',
                                   '.control\nset temp=85\n.endc'])
def test_opaque_custom_decks_cannot_confirm_managed_temperature(bench, launch, custom):
    bench.add_custom_line(custom)
    report = bench.run()
    assert report.operating_conditions.temperature_kind == 'unknown'
    bench.set_temperature(85)
    with pytest.raises(NotSupportedError, match='custom SPICE'):
        bench.run()
    assert launch['calls'] == 1


def test_cache_preserves_snapshot_and_rejects_changed_conditions(bench, launch):
    bench._cache_enabled = True
    first = bench.run()
    assert bench.run(run_label='other') is first
    for action in (lambda: bench.set_temperature(85), lambda: bench.set_temperature(None)):
        action()
        if bench.netlist.simulator_options.get('temp') is not None:
            with pytest.raises(SimulationError, match='clear_cache'):
                bench.run()
    with pytest.raises(SimulationError, match='clear_cache'):
        bench.run(voltage=1.8)
    assert launch['calls'] == 1
    bench.clear_cache()
    report = bench.run(voltage=1.8)
    assert report.characterization_context.temperature == 27
    assert first.operating_conditions.supply_voltage_v is None


def test_memory_confirms_silk_but_external_file_redeclares(bench, launch):
    report = bench.run('ff', voltage=1.8)
    assert bench.evaluate().operating_conditions is report.operating_conditions
    assert bench.evaluate('ff', temperature=27, voltage=1.8).characterization_context == report.characterization_context
    for kwargs in ({'corner': 'ss'}, {'temperature': 85}, {'voltage': 1.62}):
        with pytest.raises(CharacterizationContextError):
            bench.evaluate(**kwargs)
    bench.set_temperature(85)
    external = bench.evaluate('ss', rawfile=bench.results_file, temperature=35, voltage=1.62)
    assert external.operating_conditions.origin == 'external'
    assert external.operating_conditions.temperature_basis == 'caller-declared'
    assert external.characterization_context.temperature == 35
    assert launch['calls'] == 1


@needs_ngspice
@pytest.mark.parametrize('temperature', [27, 85])
def test_real_ngspice_plain_module_temperature(bench, temperature):
    c = bench.netlist
    c.set_temperature(temperature)
    c.simulate('tt')
    assert c.operating_conditions.temperature_c == temperature
    assert c.characterization_context is None
    assert float(c.load_signal('out').y[-1]) == pytest.approx(1.8)
    log = (Path(c.results_file).parent / 'output.log').read_text()
    assert f'TEMP = {temperature:.6f}' in log and 'TNOM = 27.000000' in log


@needs_ngspice
@needs_pdk
@pytest.mark.slow
@pytest.mark.parametrize('temperature', [27, 85])
def test_real_sky130_report_and_sqlite_roundtrip(bench, temperature, tmp_path):
    # A transistor ensures the process model library is exercised, too.
    c = bench.netlist
    c.add(bench.pdk_details.nmos1p8('load', d='out', g='out', s='0', b='0', w=1, l=0.15))
    bench.set_temperature(temperature)
    bench.module.create_circuit_version()
    report = bench.module.run_test('level', corner='ff', voltage=1.8, store=True)
    assert report.metrics['supply'] == pytest.approx(1.8)
    assert report.operating_conditions.temperature_c == temperature
    assert report.characterization_context.temperature == temperature
    deck = Path(c._run_files.netlist_path).read_text()
    assert deck.count('temp=') == 1 and ' ff' in deck
    log = (Path(c.results_file).parent / 'output.log').read_text()
    assert f'TEMP = {temperature:.6f}' in log and 'TNOM = 27.000000' in log
    bench.module.close()
    with sqlite3.connect(tmp_path / 'evidence.db') as db:
        for table in ('simulation_results', 'spec_measurements'):
            rows = db.execute(f'SELECT corner, temperature, voltage, profile_id FROM {table}').fetchall()
            assert rows and all(r[:3] == ('ff', temperature, 1.8) and r[3] is not None for r in rows)


@needs_ngspice
def test_real_sweep_overrides_initialization_and_external_scalar_is_rejected(bench):
    bench.set_temperature(85)
    bench.netlist.sim_analysis_dict.clear()
    bench.add_sim_type(DC(source='temp', start=-40, stop=125, step=5))
    with pytest.warns(DeprecationWarning), pytest.raises(CharacterizationContextError):
        bench.run(temperature=85)
    report = bench.run(voltage=1.8)
    c = report.operating_conditions
    assert c.temperature_kind == 'sweep' and c.temperature_c is None
    assert (c.temperature_sweep.start, c.temperature_sweep.stop, c.temperature_sweep.step) == (-40, 125, 5)
    assert report.characterization_context is None
    path = bench.results_file
    with pytest.raises(CharacterizationContextError, match='sweep'):
        bench.evaluate(rawfile=path, temperature=85, voltage=1.8)
    assert bench.netlist.operating_conditions is None
    with pytest.raises(CharacterizationContextError, match='sweep'):
        bench.evaluate(rawfile=path, temperature=85)
    external = bench.evaluate(rawfile=path)
    assert external.operating_conditions.temperature_kind == 'sweep'


@pytest.mark.parametrize('mode', ['overwrite', 'new'])
def test_repeated_setters_clear_and_wrappers_preserve_prior_conditions(bench, launch, mode):
    bench.module.set_output_options(mode)
    snapshots = []
    for temperature in (85, 35, 85, None):
        bench.set_temperature(temperature)
        launch['temperatures'] = (27 if temperature is None else temperature,)
        report = bench.module.run_test('level', voltage=1.8, run_label='point')
        snapshots.append(report.operating_conditions)
    assert [c.temperature_c for c in snapshots] == [85, 35, 85, 27]
    assert snapshots[-1].temperature_basis == 'backend-observed'
    # An ideal bench: each corner is a label, the evidence is none (TASK-406).
    with pytest.warns(UserWarning, match='no PDK device'):
        reports = bench.run_corners(('tt', 'ff'), voltage=1.8)
    assert [r.corner for r in reports.values()] == ['tt', 'ff']
    assert [r.operating_conditions.corner_kind for r in reports.values()] == ['none', 'none']
    all_reports = bench.module.run_all_tests(corner='ss', voltage=1.8)
    only = next(iter(all_reports.values()))
    assert (only.corner, only.operating_conditions.corner_kind) == ('ss', 'none')


def test_mixed_sweep_has_no_scalar_pvt_in_storage(bench, launch):
    bench.add_sim_type(DC(source='temp', start=-40, stop=125, step=5))
    bench.set_temperature(85)
    report = bench.run(voltage=1.8)
    assert report.operating_conditions.temperature_kind == 'sweep'
    assert report.characterization_context is None
    bench.module.create_circuit_version()
    bench.module.store_report(report)
    rows = bench.module._tracker.db.conn.execute('SELECT * FROM simulation_results').fetchall()
    assert all(r['temperature'] is None and r['profile_id'] is None for r in rows)
    marker = json.loads(rows[0]['conditions'])['silk_operating_conditions']
    assert marker['temperature_sweep'] == {'start': -40, 'stop': 125, 'step': 5}


def test_behavioral_temperature_request_fails_before_attempt(bench, launch):
    bench.set_temperature(85)
    with pytest.raises(NotSupportedError, match='behavioral'):
        bench.netlist.simulate_behavioral(1e-6, 2e-6)
    assert launch['calls'] == 0


def test_direct_external_deck_has_unknown_temperature_and_setter_requires_preparation(bench, launch):
    c = bench.netlist
    c_deck_path = c.generate_netlist()
    external = Module('sky130', str(Path(c_deck_path).parent), Path(c_deck_path).name, c.pdk_details)
    external.set_output_options(mode='overwrite')
    external.run_simulator()
    assert external.operating_conditions.temperature_kind == 'unknown'
    external.set_temperature(85)
    with pytest.raises(SimulationError, match='generate_netlist'):
        external.run_simulator()
    assert launch['calls'] == 1


def test_new_external_load_and_failure_clear_silk_snapshot(bench, launch, tmp_path):
    first = bench.run(voltage=1.8)
    missing = tmp_path / 'missing.raw'
    bench.netlist.load_file(missing)
    assert bench.netlist.operating_conditions is None
    with pytest.raises(RawfileError, match="missing.raw"):
        bench.netlist.read_raw_file()
    assert not bench.netlist.sim_results_loaded
    assert bench.netlist.characterization_context is None
    assert first.operating_conditions.origin == 'silk'


def test_fractional_setting_retains_precision_after_log_confirmation(bench, launch):
    bench._cache_enabled = True
    bench.set_temperature(85.12345678)
    launch['temperatures'] = (85.123457,)
    with pytest.warns(DeprecationWarning):
        report = bench.run(temperature=85.12345678)
    assert report.operating_conditions.temperature_c == 85.12345678
    with pytest.warns(DeprecationWarning):
        assert bench.run(temperature=85.12345678) is report


def test_external_memory_corner_redeclaration_updates_the_complete_descriptor(bench):
    fixture = Path(__file__).parent / 'fixtures/rawfiles/e_ng_tran.raw'
    bench.evaluate('ff', rawfile=fixture, temperature=85, voltage=1.8)
    updated = bench.evaluate('ss')
    assert updated.corner == updated.operating_conditions.corner == updated.characterization_context.corner == 'ss'


def test_spice_numeric_sweep_bounds_are_numeric_provenance(bench, launch):
    bench.add_sim_type(DC(source='temp', start='-40', stop='125', step='5'))
    c = bench.run().operating_conditions
    assert c.to_dict()['temperature_sweep'] == {'start': -40.0, 'stop': 125.0, 'step': 5.0}


def test_missing_overwrite_deck_is_actionable_preflight_failure(bench, launch):
    c = bench.netlist
    c.generate_netlist()
    Path(c._run_files.netlist_path).unlink()
    with pytest.raises(SimulationError, match='generate_netlist'):
        c.run_simulator()
    assert launch['calls'] == 0


@pytest.mark.parametrize('custom', ['.param vbias=0.9', '.ic v(out)=0.5',
                                    '.nodeset v(mid)=0.4'])
def test_unrelated_custom_text_keeps_the_backend_observation(bench, launch, custom):
    """An unrelated custom line must not erase an honest observation.

    Custom text with no temperature effect leaves the runtime observation
    available, including alongside an explicit managed setting.
    """
    bench.add_custom_line(custom)
    report = bench.run()
    assert report.operating_conditions.temperature_kind == 'fixed'
    assert report.operating_conditions.temperature_basis == 'backend-observed'
    # A managed setting is likewise not refused: nothing in this text can carry
    # a temperature, so the run reaches the ordinary observation check.
    bench.set_temperature(27)
    confirmed = bench.run()
    assert confirmed.operating_conditions.temperature_c == 27
    assert confirmed.operating_conditions.temperature_basis == 'configured-and-observed'


@pytest.mark.parametrize('custom,options,temperature', [
    ('.temp 35', {}, 35),
    ('.TeMp\t+8.5e1', {}, 85),
    ('.param vbias=0.9\n.TeMp 85\n.ic v(out)=0.5', {}, 85),
    ('.temp\n+ 85', {}, 85),
    (['.temp', '* comment', '+ 85'], {}, 85),
    ('.OpTiOnS reltol=1e-5\n+ TeMp = 85\n+ tnom=20', {}, 85),
    ('.option temp=85', {}, 85),
    ('.temp 85 $ ignored temp=35', {}, 85),
    ('.temp 85 ; ignored .temp 35', {}, 85),
    ('.temp 85 // ignored "temp=35"', {}, 85),
    ('.temp 85000m', {}, 85),
    (None, {'TEMP': '85'}, 85),
    (None, {'reltol': '1e-5 TeMp = 85 tnom=20'}, 85),
])
def test_raw_literals_preserve_deck_and_confirm_report(bench, launch, custom, options, temperature):
    c = bench.netlist
    lines = custom if isinstance(custom, list) else ([] if custom is None else [custom])
    for line in lines:
        bench.add_custom_line(line)
    c.set_simulator_options(**options)
    c_deck_path = c.generate_netlist()
    before = Path(c_deck_path).read_bytes()
    for line in lines:
        assert line.encode() in before
    for key, value in options.items():
        assert f'{key}={value}'.encode() in before
    launch['temperatures'] = (temperature, temperature)
    with pytest.warns(DeprecationWarning):
        report = bench.run(temperature=temperature, voltage=1.8)
    assert Path(c_deck_path).read_bytes() == before
    assert c.simulator_options == options
    assert report.operating_conditions.temperature_c == temperature
    assert report.operating_conditions.temperature_basis == 'raw-and-observed'
    assert report.characterization_context.temperature == temperature


def test_multiline_option_value_is_refused(bench):
    """TASK-593: an option value is written into one options card, so a
    continuation line inside it is refused. Multi-line .options text belongs
    in add_custom_line(), which is written verbatim."""
    from silk.exceptions import DeckTextError
    with pytest.raises(DeckTextError, match='line break'):
        bench.netlist.set_simulator_options(reltol='1e-5\n+ TEMP = 85')


@pytest.mark.parametrize('custom,options,labels', [
    ('.temp 35', {'temp': 85}, ['.temp 35', 'temp=85', 'native options']),
    ('.temp 85', {'temp': 85}, ['.temp 85', 'temp=85']),
    ('.temp 35\n.temp 35', {}, ['custom_lines[0] line 1', 'custom_lines[0] line 2']),
    (['.temp 35', '.options temp=85'], {}, ['custom_lines[0]', 'custom_lines[1]', 'temp=85']),
    ('.options temp=85\n+ TEMP=85', {}, ['temp=85', 'TEMP=85', 'line 2']),
    (None, {'temp': 85, 'TEMP': 85}, ['temp=85', 'TEMP=85']),
    (None, {'reltol': '1e-5 temp=35 TEMP=85'}, ['temp=35', 'TEMP=85']),
    ('.options reltol=1e-5 temp={hot}', {'temp': 85}, ['temp={hot}', 'temp=85']),
])
def test_duplicate_declarations_fail_before_output_or_launch(bench, launch, custom, options, labels):
    first = bench.run()
    path = Path(first.run.deck_path)
    before = path.read_bytes()
    lines = custom if isinstance(custom, list) else ([] if custom is None else [custom])
    for line in lines:
        bench.add_custom_line(line)
    bench.netlist.set_simulator_options(**options)
    with pytest.raises(SimulationError, match='Multiple explicit temperature') as error:
        bench.run(run_label='conflict')
    for label in labels:
        assert label in str(error.value)
    assert 'Remove' in str(error.value)
    assert launch['calls'] == 1
    assert path.read_bytes() == before
    assert bench.netlist.operating_conditions is first.operating_conditions
    assert not (path.parent / 'conflict').exists()


@pytest.mark.parametrize('custom', [
    '.temp -273.15', '.temp -300', '.temp nan', '.temp inf',
    '.temp 1e999', '.options temp=-273.15', '.options TEMP=-inf',
])
def test_invalid_raw_temperature_domain_fails_preflight(bench, launch, custom):
    bench.add_custom_line(custom)
    with pytest.raises(SimulationError, match='Invalid temperature.*finite.*absolute zero'):
        bench.run()
    assert launch['calls'] == 0


@pytest.mark.parametrize('custom', [
    '.temp {35}', ".temp '35'", '.temp 35 + 5', '.temp 35 85', '.temp 35\n+ 85',
    '.options temp={hot}', '.options reltol=1e-5 temp=35+5',
    '.options temp=35 85', '.control\nset temp=85\n.endc',
    '.control\naltermod nmod tnom=85\n.endc',
    '.include "caller.lib"', '.INC "caller.lib"', '.lib "caller.lib" hot',
    '.temp 35\n.include "caller.lib"',
    '.if (1)\n.temp 35\n.endif',
    '.subckt ignored a b\n.temp 35\n.ends ignored',
    '.options label="this is temp=85"',
])
def test_nonliteral_or_unexpanded_content_has_unknown_provenance(bench, launch, custom):
    launch['temperatures'] = (35,)
    bench.add_custom_line(custom)
    report = bench.run(voltage=1.8)
    assert report.operating_conditions.temperature_kind == 'unknown'
    assert report.operating_conditions.temperature_basis == 'unknown'
    assert report.operating_conditions.temperature_c is None
    assert report.characterization_context is None


def test_comments_and_tnom_are_not_ambient_declarations(bench, launch):
    bench.add_custom_line('* .temp 85\n.options tnom=20 ; temp=35')
    report = bench.run()
    assert report.operating_conditions.temperature_c == 27
    assert report.operating_conditions.temperature_basis == 'backend-observed'
    bench.set_temperature(27)
    assert bench.run().operating_conditions.temperature_basis == 'configured-and-observed'


def test_raw_literal_requires_runtime_confirmation(bench, launch):
    bench.add_custom_line('.temp 85')
    with pytest.raises(SimulationError, match='observed'):
        bench.run()
    assert launch['calls'] == 1
    assert bench.netlist.operating_conditions is None


def test_changed_raw_native_text_invalidates_preparation_and_cache(bench, launch):
    c = bench.netlist
    c.set_simulator_options(reltol='1e-5 temp=27')
    c.generate_netlist()
    c.set_simulator_options(reltol='1e-5 temp=85')
    with pytest.raises(SimulationError, match='generate_netlist'):
        c.run_simulator()
    assert launch['calls'] == 0
    launch['temperatures'] = (85,)
    bench._cache_enabled = True
    first = bench.run()
    c.set_simulator_options(reltol='1e-5 temp=35')
    with pytest.raises(SimulationError, match='clear_cache'):
        bench.run()
    assert first.operating_conditions.temperature_c == 85


@pytest.mark.parametrize('custom', ['.temp 85', '.options temp=85', '.options device temp=85'])
def test_raw_xyce_temperature_never_borrows_ngspice_proof(bench, custom):
    from silk.simulation.backend import RawResult
    from silk.simulation.conditions import observed_conditions, prepare_temperature
    bench.netlist.simulator = 'xyce'
    bench.add_custom_line(custom)
    request = prepare_temperature(bench.netlist)
    raw = RawResult(native=None, source_path='unused', ambient_temperatures=(85,))
    c = observed_conditions(request, 'tt', raw)
    assert c.temperature_c is None and c.temperature_basis == 'unknown'


@needs_ngspice
@needs_pdk
@pytest.mark.parametrize('custom,options,temperature', [
    ('.temp 35', {}, 35),
    ('.options TEMP=85', {}, 85),
    ('.OpTiOnS reltol=1e-5\n+ TeMp=85', {}, 85),
    (None, {'reltol': '1e-5 TEMP=85'}, 85),
])
def test_real_ngspice_raw_temperature_matches_log(bench, custom, options, temperature):
    if custom:
        bench.add_custom_line(custom)
    bench.netlist.set_simulator_options(**options)
    report = bench.run(voltage=1.8)
    assert report.operating_conditions.temperature_basis == 'raw-and-observed'
    assert report.operating_conditions.temperature_c == temperature
    assert report.characterization_context.temperature == temperature
    log = (Path(bench.results_file).parent / 'output.log').read_text()
    assert f'TEMP = {temperature:.6f}' in log
    assert report.metrics['supply'] == pytest.approx(1.8)


@needs_ngspice
@needs_pdk
def test_real_ngspice_audit_conflict_is_rejected_then_reports_actual_card(bench, tmp_path):
    # Independently reproduce the old audit deck, bypassing Silk preflight.
    deck = tmp_path / 'audit.cir'
    deck.write_text('audit\nV1 out 0 1.8\nR1 out 0 1000\n.temp 35\n.options temp=85\n.op\n.end\n')
    result = subprocess.run([os.environ.get('SILK_NGSPICE_BIN', 'ngspice'), '-b', str(deck)],
                            capture_output=True, text=True, check=True)
    assert 'TEMP = 35.000000' in result.stdout
    bench.add_custom_line('.temp 35')
    bench.netlist.set_simulator_options(temp=85)
    with pytest.raises(SimulationError, match='Multiple explicit temperature'):
        bench.run()
    assert bench.results_file is None
    bench.set_temperature(None)
    report = bench.run()
    log = (Path(bench.results_file).parent / 'output.log').read_text()
    assert 'TEMP = 35.000000' in log
    assert report.operating_conditions.temperature_basis == 'raw-and-observed'
    assert report.operating_conditions.temperature_c == 35


@needs_pdk
@pytest.mark.slow
@pytest.mark.requires_xyce
@pytest.mark.skipif(not shutil.which(os.environ.get('SILK_XYCE_BIN', 'Xyce')),
                    reason='requires Xyce ($SILK_XYCE_BIN)')
def test_real_xyce_ignored_temp_card_retains_unknown_report(bench):
    bench.module.simulator = 'xyce'
    bench.add_custom_line('.temp 85')
    report = bench.run()
    assert report.operating_conditions.temperature_c is None
    assert report.operating_conditions.temperature_basis == 'unknown'
    assert report.metrics['supply'] == pytest.approx(1.8)


@pytest.mark.parametrize('designation', [None, 'supply'])
def test_designated_supply_sweep_records_range_not_nominal(bench, launch, designation):
    from silk.simulation import ConditionState
    bench.set_supply_voltage(1.8, **({'source': designation} if designation else {}))
    bench.netlist.sim_analysis_dict.clear()
    bench.add_sim_type(DC('VSupply', 0, 2, 0.5))
    report = bench.run()
    c = report.operating_conditions
    assert c.supply_kind == 'sweep'
    assert c.supply_voltage_v is None and c.supply_source == 'supply'
    assert c.scalar('supply') is ConditionState.SWEEP
    assert (c.supply_sweep.start, c.supply_sweep.stop, c.supply_sweep.step) == (0, 2, 0.5)
    assert report.characterization_context is None
    assert 'supply sweep' in report.summary()
    assert 'DC 1.8' in Path(report.run.deck_path).read_text()


def test_supply_sweep_refuses_deprecated_scalar_even_at_nominal(bench, launch):
    from silk.exceptions import SimulationConfigurationError
    bench.set_supply_voltage(1.8, source='supply')
    bench.add_sim_type(DC('supply', 0, 2, 0.5))
    with pytest.warns(DeprecationWarning), pytest.raises(SimulationConfigurationError, match='sweep'):
        bench.run(voltage=1.8)
    assert launch['calls'] == 0


@pytest.mark.parametrize('dc_last', [False, True])
def test_behavioral_supply_sweep_uses_dc_plot_and_automatic_designation(bench, dc_last):
    from silk.simulation.conditions import prepare_supply
    bench.netlist.simulator = 'behavioral'
    bench.set_supply_voltage(1.8)
    # The ideal load is supported without any model or external dependency.
    bench.netlist.sim_analysis_dict.clear()
    analyses = [DC('supply', 0, 2, 0.5), Transient(step=1e-6, stop=2e-6)]
    for analysis in (reversed(analyses) if dc_last else analyses):
        bench.add_sim_type(analysis)
    before = bench.netlist.sig_sources_dict['supply'].dc
    request = prepare_supply(bench.netlist, apply=False)
    assert request.source == 'supply'
    assert bench.netlist.sig_sources_dict['supply'].dc == before
    bench.netlist.simulate()
    c = bench.netlist.operating_conditions
    assert c.supply_kind == 'sweep' and c.supply_voltage_v is None
    assert c.supply_source == 'supply'
    assert c.supply_sweep.stop == 2
    import numpy as np
    sig = bench.netlist.load_signal('out', analysis='dc')
    np.testing.assert_array_equal(sig.y, sig.x)


@pytest.mark.parametrize('target', ['other', 'temp'])
def test_other_source_or_temperature_sweep_keeps_supply_fixed(bench, launch, target):
    bench.set_supply_voltage(1.8, source='supply')
    bench.add_source(sources.VoltageSource('other', positive='input', negative='0', dc=0))
    bench.netlist.sim_analysis_dict.clear()
    bench.add_sim_type(DC(target, 0, 2, 0.5))
    report = bench.run()
    assert report.operating_conditions.supply_kind == 'fixed'
    assert report.operating_conditions.supply_sweep is None
    assert report.operating_conditions.supply_voltage_v == 1.8


def test_supply_analysis_change_invalidates_preparation(bench, launch):
    bench.set_supply_voltage(1.8, source='supply')
    bench.add_sim_type(DC('supply', 0, 2, 0.5))
    bench.netlist.generate_netlist()
    bench.add_sim_type(DC('supply', 0, 3, 0.5))
    with pytest.raises(SimulationError, match='generate_netlist'):
        bench.netlist.run_simulator()
    assert launch['calls'] == 0


def test_opaque_supply_sweep_does_not_claim_configured_evidence(bench, launch):
    bench.set_supply_voltage(1.8, source='supply')
    bench.add_sim_type(DC('supply', 0, 2, 0.5))
    bench.add_custom_line('* opaque caller content')
    report = bench.run()
    assert report.operating_conditions.supply_kind == 'unknown'
    assert report.operating_conditions.supply_sweep is None


def test_behavioral_sweep_validation_refuses_missing_or_wrong_axis_values(bench):
    from silk.simulation.conditions import prepare_supply, driven_supply
    import numpy as np
    bench.set_supply_voltage(1.8, source='supply')
    request = prepare_supply(bench.netlist)
    for driven in ({}, {'out': np.array([0, 1])}, {'out': np.array([])}):
        with pytest.raises(SimulationError, match='no waveform|expected supply drive'):
            driven_supply(request, bench.netlist, driven, axis=np.array([0, 2]))
    driven_supply(request, bench.netlist, {'out': np.array([0, 2])}, axis=np.array([0, 2]))


@needs_ngspice
def test_real_ngspice_supply_sweep_records_range_and_waveform(bench):
    import numpy as np
    bench.set_supply_voltage(1.8, source='supply')
    bench.netlist.sim_analysis_dict.clear()
    bench.add_sim_type(DC('supply', 0, 2, 0.5))
    report = bench.run()
    assert report.operating_conditions.supply_kind == 'sweep'
    assert report.operating_conditions.supply_sweep.stop == 2
    signal = bench.netlist.load_signal('out')
    np.testing.assert_allclose(signal.y, [0, 0.5, 1, 1.5, 2])


def test_supply_sweep_stores_unscoped_without_reusing_fixed_profile(bench, launch):
    bench.set_supply_voltage(1.8, source='supply')
    fixed = bench.run()
    assert fixed.characterization_context is not None
    bench.module.create_circuit_version()
    bench.module.store_report(fixed)
    bench.netlist.sim_analysis_dict.clear()
    bench.add_sim_type(DC('supply', 0, 2, 0.5))
    swept = bench.run()
    assert swept.characterization_context is None
    bench.module.store_report(swept)
    db = bench.module._tracker.db.conn
    for table in ('simulation_results', 'spec_measurements'):
        row = db.execute(f'SELECT voltage, voltage_state, profile_id FROM {table} '
                         "WHERE voltage_state='sweep'").fetchone()
        assert tuple(row) == (None, 'sweep', None)
    assert db.execute('SELECT count(*) FROM characterization_profiles').fetchone()[0] == 1


def test_indexed_supply_target_uses_backend_designer_and_emitted_resolution(bench):
    from silk.simulation.conditions import prepare_supply
    bench.add_source(sources.VoltageSource('rail[0]', positive='indexed', negative='0', dc=1.8))
    bench.set_supply_voltage(1.8, source='rail[0]')
    bench.add_sim_type(DC('RAIL[0]', 0, 2, 0.5))
    designer = prepare_supply(bench.netlist, apply=False)
    source = bench.netlist.sig_sources_dict['rail[0]']
    bench.add_sim_type(DC(source.instance_name.lower(), 0, 2, 0.5))
    emitted = prepare_supply(bench.netlist, apply=False)
    assert designer.sweep == emitted.sweep
    assert designer.sweep.stop == 2 and emitted.source == 'rail[0]'


def test_temp_target_bypasses_a_supply_source_with_that_designer_name(bench):
    from silk.simulation.conditions import prepare_supply
    bench.add_source(sources.VoltageSource('temp', positive='rail', negative='0', dc=1.8))
    bench.set_supply_voltage(1.8, source='temp')
    bench.add_sim_type(DC('TEMP', 0, 2, 0.5))
    assert prepare_supply(bench.netlist).sweep is None


def test_behavioral_automatic_fixed_supply_remains_verified(bench):
    bench.netlist.simulator = 'behavioral'
    bench.set_supply_voltage(1.8)
    bench.netlist.simulate()
    c = bench.netlist.operating_conditions
    assert c.supply_kind == 'fixed' and c.supply_voltage_v == 1.8
    assert c.supply_basis == 'configured-and-driven' and c.supply_sweep is None


@pytest.mark.parametrize('with_builtin', [False, True])
def test_dc_subclass_makes_supply_unknown_even_beside_builtin_sweep(bench, launch, with_builtin):
    class NamedDC(DC):
        pass

    bench.set_supply_voltage(1.8, source='supply')
    bench.netlist.sim_analysis_dict.clear()
    if with_builtin:
        bench.add_sim_type(DC('supply', 0, 2, 0.5))
    bench.add_sim_type(NamedDC('supply', 0, 2, 0.5))
    from silk.exceptions import SimulationConfigurationError
    with pytest.warns(DeprecationWarning), pytest.raises(
            SimulationConfigurationError, match='custom DC analysis.*one supply voltage'):
        bench.run(voltage=1.8)
    assert launch['calls'] == 0
    report = bench.run()
    deck = Path(report.run.deck_path).read_text()
    assert '.dc Vsupply 0 2 0.5' in deck
    assert bench.netlist._run_files.completion is None
    c = report.operating_conditions
    assert c.supply_kind == 'unknown'
    assert c.supply_voltage_v is None and c.supply_sweep is None
    assert report.characterization_context is None


def test_opaque_carried_voltage_refuses_in_run_simulator_before_launch(bench, launch):
    from silk.exceptions import SimulationConfigurationError
    class NamedDC(DC):
        pass

    bench.set_supply_voltage(1.8, source='supply')
    bench.netlist.sim_analysis_dict.clear()
    bench.add_sim_type(NamedDC('supply', 0, 2, 0.5))
    c = bench.netlist
    c.generate_netlist()
    c._declared_supply = 1.8
    with pytest.raises(SimulationConfigurationError, match='custom DC analysis.*one supply voltage'):
        c.run_simulator()
    assert launch['calls'] == 0
    assert c._declared_supply is None
    c.run_simulator()
    assert launch['calls'] == 1
    assert c.operating_conditions.supply_kind == 'unknown'
    assert c.operating_conditions.supply_voltage_v is None
    assert c.operating_conditions.supply_sweep is None


def test_zero_card_supply_proof_does_not_certify_custom_dc_launch(bench, launch):
    from silk.simulation import has_no_supply
    class NamedDC(DC):
        pass

    bench.netlist.sig_sources_dict['supply'].dc = 0.0
    bench.netlist.sim_analysis_dict.clear()
    bench.add_sim_type(NamedDC('supply', 0, 2, 0.5))
    assert has_no_supply(bench.netlist) is True
    report = bench.run()
    assert launch['calls'] == 1
    c = report.operating_conditions
    assert c.supply_kind == 'unknown'
    assert c.supply_voltage_v is None and c.supply_sweep is None
    assert report.characterization_context is None
