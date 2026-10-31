"""``ModuleTestbench.evaluate(corner, rawfile=...)`` (TASK-113).

Pre-fix there was NO public path from a rawfile to metrics + spec
measurements: ``ModuleTestbench._simulate`` hard-wired
``self.netlist.simulate(corner)`` on its first line and then ran the
extraction/spec tail, so a user who generated the deck here, ran ngspice or
Xyce elsewhere, and loaded the rawfile with ``tb.netlist.load_file(path);
tb.netlist.read_raw_file()`` could only reach the tail through the private
``tb._metrics``/``tb._specs`` and a copy of ``_simulate``'s body.

The proof that matters (reviewer ruling): ``run(corner)`` and
``evaluate(corner, rawfile=<the rawfile run() just produced>)`` must be equal
field-for-field, and the Spec measurements they record must be identical.
The fast legs stand ``netlist.simulate`` in for the simulator by loading a
committed TASK-74 fixture (real ngspice-42 / Xyce 7.10 output); the slow leg
goes through real ngspice + sky130.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from tests._temperature import mark_simulated

from silk import sources
from silk.circuit.circuit_elements import PortDir
from silk.circuit.modules import Module
from silk.circuit.module_test import ModuleTestbench
from silk.circuit.spec_report import SpecReport
from silk.exceptions import CharacterizationContextError, SimulationAborted, SimulationError
from silk.pdk.basepdk import BasePDK
from silk.signals.rawfile import RawfileError, read_rawfile
from silk.simulation import analysis
from silk.simulation.backend import SimulatorBackend

FIXTURES = Path(__file__).resolve().parent / 'fixtures' / 'rawfiles'
_PDK_YAML = os.path.join(os.path.dirname(__file__), '..', 'silk', 'pdk', 'data',
                         'skywater_130_device_details.yaml')
_SKY130_LIB = os.path.expanduser(
    '~/pdk/sky130A/libs.tech/ngspice/sky130.lib.spice')


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _dut(pdk, tmp_dir, *, db_path=None):
    """One-transistor DUT with ports — the DUT's structure is irrelevant to
    the fixture legs (no deck is simulated); it exists so
    add_module_under_test() is a real call and evaluate() runs on a
    testbench built exactly the way a user builds one."""
    dut = Module('PDK_skywater130', tmp_dir, 'dut.cir',
                 pdk_details=pdk, module_name='dut', db_path=db_path)
    dut.define_module_ports({'a': PortDir.IN, 'b': PortDir.OUT,
                             'vdd': PortDir.POWER, 'vss': PortDir.GROUND})
    with dut:
        pdk.nmos1p8('mn', d='b', g='a', s='vss', b='vss', w=1.0, l=0.15)
    return dut


def _fixture_tb(pdk, tmp_dir, name='fix'):
    """Testbench whose extractors read the ``e_*_tran`` fixture's node
    names (``v(out)``, ``v(clk)``, ``i(v1)``; gen_e.py). Two specs: one
    that passes on the fixture (out_max = 1.372 > 1) and one that fails
    (clk_max = 1.8 is not < 1), so the passed flags are exercised both
    ways."""
    dut = _dut(pdk, tmp_dir)
    tb = dut.create_test(name, parameters={'cload_f': 2e-12})
    tb.add_module_under_test({'a': 'step', 'b': 'out', 'vdd': 'clk', 'vss': '0'})
    # A real rail on the DUT's declared POWER port, so a run that declares a
    # supply has one to be checked against (TASK-158). It is never emitted
    # here — every run in this file loads a fixture rawfile instead.
    tb.add_source(sources.VoltageSource('vsup', positive='clk', negative='0', dc=1.8))
    tb.add_metric('out_max', lambda ckt: float(np.max(ckt.load_signal('out').y)))
    tb.add_metric('clk_max', lambda ckt: float(np.max(ckt.load_signal('clk').y)))
    tb.add_metric('i_v1_end', lambda ckt: float(ckt.load_signal('v1', 'I').y[-1]))
    tb.add_spec('out_max > 1 V', 'out_max', '>', 1.0)
    tb.add_spec('clk_max < 1 V', 'clk_max', '<', 1.0)
    return tb


def _stub_simulate_with(tb, rawfile, voltage=None):
    """Stand ``tb.netlist.simulate`` in for the simulator: load *rawfile*
    through the public loader and record it as the run's results file,
    exactly what a real ``simulate()`` leaves behind — so ``run()`` exercises
    its real tail on real simulator output without launching anything.

    *voltage* is the supply such a run would have established from the deck
    it emitted; a stub emits none, so it is stated here instead (TASK-158)."""
    def fake_simulate(corner=None, **_):
        tb.netlist.load_file(str(rawfile))
        tb.netlist.read_raw_file()
        tb.netlist.results_file = str(rawfile)
        mark_simulated(tb.netlist, corner, voltage=voltage,
                       supply_source='vsup')
        return tb.netlist
    return patch.object(tb.netlist, 'simulate', side_effect=fake_simulate)


def _snapshot_measurements(tb):
    """{spec key: {slot: Measurement fields}} for every spec on the tb."""
    return {
        s['spec'].key: {
            slot: (m.source, m.value, m.passes, m.corner)
            for slot, m in s['spec'].measurements.items()
        }
        for s in tb._specs
    }


def _assert_reports_identical(a: SpecReport, b: SpecReport):
    """Field-for-field equality: metrics, results (incl. passed), owner_path,
    parameters, context, names; both point at the same loaded Module.
    Operating-condition origin/basis deliberately differ for external files."""
    assert a.test_name == b.test_name
    assert a.module_name == b.module_name
    assert a.owner_path == b.owner_path
    assert a.parameters == b.parameters
    assert a.characterization_context == b.characterization_context
    assert a.metrics == b.metrics
    assert a.passed == b.passed
    assert a.results == b.results
    # Each extraction has its own observation identity and precise time.
    # Public check results remain identical for identical waveform data.
    for left, right in zip(a.results, b.results):
        if hasattr(left, '_observation') and hasattr(right, '_observation'):
            original, repeated = left._observation[-1], right._observation[-1]
            assert original.attempt_id != repeated.attempt_id
            assert original.measurement.value == repeated.measurement.value
            assert original.measurement.passes == repeated.measurement.passes
    assert a.sim_results is b.sim_results


# ---------------------------------------------------------------------------
# AC3 — identity: run() vs evaluate(rawfile=<run()'s rawfile>), fast legs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('fixture', ['e_ng_tran.raw', 'e_xy_tran.raw'])
def test_run_and_evaluate_on_same_rawfile_are_identical(pdk, tmp_dir, fixture):
    """Pre-fix: no evaluate() existed. run() (simulator stood in by the
    TASK-74 fixture — real ngspice-42 / Xyce 7.10 output) and
    evaluate('tt', rawfile=<that same file>) must yield field-for-field
    equal SpecReports: out_max 1.372 (pass), clk_max 1.8 (fail),
    i_v1_end -1.64e-10 — and the Measurement each Spec carries afterwards
    must equal the one run() recorded in value/passes/corner/source."""
    tb = _fixture_tb(pdk, tmp_dir)
    tb.set_temperature(27)
    rawfile = FIXTURES / fixture

    with _stub_simulate_with(tb, rawfile):
        report_a = tb.run('tt')
    after_run = _snapshot_measurements(tb)
    assert report_a.metrics['out_max'] == pytest.approx(1.3723, abs=1e-3)
    assert [r.passed for r in report_a.results] == [True, False]

    with patch.object(tb.netlist, 'simulate',
                      side_effect=AssertionError('evaluate() launched a simulation')):
        report_b = tb.evaluate('tt', rawfile=str(rawfile))
    after_evaluate = _snapshot_measurements(tb)

    _assert_reports_identical(report_a, report_b)
    # Spec.measurements is keyed by (source, corner) slot: the second
    # 'spice'@'tt' Measurement lands in the SAME slot as run()'s, so the
    # spec holds one live Measurement per slot — identical in every field
    # but the timestamp.
    assert after_evaluate == after_run
    assert all(set(slots) == {'spice@tt'} for slots in after_run.values())
    assert tb.results_file == str(rawfile)


def test_evaluate_with_context_matches_run_with_context(pdk, tmp_dir):
    """The characterization context goes through the same _build_context as
    run(): evaluate('tt', rawfile=..., temperature=27, voltage=1.8) carries
    the same CharacterizationContext (all six fields) as the scoped run(),
    on the report and on tb.netlist."""
    tb = _fixture_tb(pdk, tmp_dir)
    tb.set_temperature(27)
    rawfile = FIXTURES / 'e_ng_tran.raw'
    with _stub_simulate_with(tb, rawfile, voltage=1.8):
        report_a = tb.run('tt', temperature=27.0, voltage=1.8)
    report_b = tb.evaluate('tt', rawfile=str(rawfile), temperature=27.0, voltage=1.8)
    _assert_reports_identical(report_a, report_b)
    ctx = report_b.characterization_context
    assert ctx is not None and ctx.corner == 'tt' and ctx.temperature == 27.0
    assert tb.netlist.characterization_context is ctx


def test_evaluate_temperature_only_records_temperature_unscoped(pdk, tmp_dir):
    """A temperature declared without a supply is recorded: temperature fixed
    and caller-declared, supply unknown, no scoped context."""
    tb = _fixture_tb(pdk, tmp_dir)
    path = str(FIXTURES / 'e_ng_tran.raw')
    with patch.object(tb.netlist, 'simulate',
                      side_effect=AssertionError('evaluate() launched a simulation')):
        report = tb.evaluate('tt', rawfile=path, temperature=27.0)
    c = report.operating_conditions
    assert c.origin == 'external'
    assert (c.temperature_kind, c.temperature_c, c.temperature_basis) == (
        'fixed', 27.0, 'caller-declared')
    assert (c.supply_voltage_v, c.supply_basis) == (None, 'unknown')
    assert report.characterization_context is None
    assert tb.netlist.characterization_context is None


def test_evaluate_voltage_only_records_supply_unscoped(pdk, tmp_dir):
    tb = _fixture_tb(pdk, tmp_dir)
    report = tb.evaluate('tt', rawfile=str(FIXTURES / 'e_ng_tran.raw'), voltage=1.8)
    c = report.operating_conditions
    assert (c.supply_voltage_v, c.supply_basis) == (1.8, 'caller-declared')
    assert (c.temperature_kind, c.temperature_c) == ('unknown', None)
    assert report.characterization_context is None


def test_evaluate_both_or_neither_declared_unchanged(pdk, tmp_dir):
    tb = _fixture_tb(pdk, tmp_dir)
    path = str(FIXTURES / 'e_ng_tran.raw')
    both = tb.evaluate('tt', rawfile=path, temperature=27.0, voltage=1.8)
    assert both.characterization_context is not None
    assert both.characterization_context.temperature == 27.0
    assert both.characterization_context.voltage == 1.8
    assert both.operating_conditions.supply_basis == 'caller-declared'
    neither = tb.evaluate('tt', rawfile=path)
    c = neither.operating_conditions
    assert (c.temperature_kind, c.temperature_c, c.supply_voltage_v) == ('unknown', None, None)
    assert neither.operating_conditions.supply_basis == 'unknown'
    assert neither.characterization_context is None


def test_evaluate_partial_redeclaration_on_loaded_external_results(pdk, tmp_dir):
    """Without a rawfile, evaluate() redeclares external evidence under the
    same rules; an omitted fact keeps what the loaded evidence recorded."""
    tb = _fixture_tb(pdk, tmp_dir)
    tb.evaluate('tt', rawfile=str(FIXTURES / 'e_ng_tran.raw'))
    only_t = tb.evaluate('tt', temperature=85.0)
    assert only_t.operating_conditions.temperature_c == 85.0
    assert only_t.operating_conditions.supply_voltage_v is None
    assert only_t.characterization_context is None
    # Adding the voltage now completes the pair and builds the scoped context.
    both = tb.evaluate('tt', voltage=1.8)
    assert both.characterization_context.temperature == 85.0
    assert both.characterization_context.voltage == 1.8
    only_v = tb.evaluate('tt', rawfile=str(FIXTURES / 'e_ng_tran.raw'), voltage=1.62)
    assert only_v.operating_conditions.supply_voltage_v == 1.62
    assert only_v.characterization_context is None
    again = tb.evaluate('tt', voltage=1.8)
    assert again.operating_conditions.supply_voltage_v == 1.8
    assert again.operating_conditions.temperature_kind == 'unknown'


def test_temperature_only_external_report_stores_unscoped_with_temperature(pdk, tmp_path):
    tb, spec, _ = _external_tb(pdk, tmp_path, db_path=str(tmp_path / 'evidence.db'))
    try:
        tb.module.create_circuit_version()
        report = tb.evaluate('ff', rawfile=str(FIXTURES / 'e_ng_tran.raw'), temperature=27.0)
        assert tb.module.store_report(report)['specs_stored'] == 1
        conn = tb.module._tracker.db.conn
        rows = conn.execute(
            'SELECT corner, temperature, voltage, profile_id FROM spec_measurements').fetchall()
        assert [tuple(r) for r in rows] == [('ff', 27.0, None, None)]
    finally:
        tb.module.close()


def test_evaluate_unscoped_after_scoped_run_is_unscoped(pdk, tmp_dir):
    """evaluate() builds its context from ITS arguments: omitting both after
    a scoped run() yields an unscoped report (context None), not the
    previous run's context replayed."""
    tb = _fixture_tb(pdk, tmp_dir)
    tb.set_temperature(27)
    rawfile = FIXTURES / 'e_ng_tran.raw'
    with _stub_simulate_with(tb, rawfile, voltage=1.8):
        tb.run('tt', temperature=27.0, voltage=1.8)
    report = tb.evaluate('tt', rawfile=str(rawfile))
    assert report.characterization_context is None
    assert tb.netlist.characterization_context is None


# ---------------------------------------------------------------------------
# AC1 — evaluate(rawfile=...) launches no simulator
# ---------------------------------------------------------------------------

def test_evaluate_rawfile_launches_no_simulator(pdk, tmp_dir, monkeypatch):
    """Every way a simulator could start is armed to raise — subprocess.run,
    every registered backend's run(), the netlist's own simulate()/
    run_simulator() — and evaluate(rawfile=...) still returns a SpecReport
    with a Measurement(source='spice', corner='ss') on every attached
    Spec."""
    def boom(*a, **k):
        raise AssertionError('a simulator was launched')

    monkeypatch.setattr(subprocess, 'run', boom)
    monkeypatch.setattr(subprocess, 'Popen', boom)
    for backend_cls in SimulatorBackend._registry.values():
        monkeypatch.setattr(backend_cls, 'run', boom)

    tb = _fixture_tb(pdk, tmp_dir)
    tb.set_temperature(27)
    monkeypatch.setattr(tb.netlist, 'simulate', boom)
    monkeypatch.setattr(tb.netlist, 'run_simulator', boom)

    report = tb.evaluate('ss', rawfile=str(FIXTURES / 'e_ng_tran.raw'))
    assert isinstance(report, SpecReport)
    assert report.metrics['clk_max'] == pytest.approx(1.8)
    for s in tb._specs:
        m = s['spec'].latest()
        assert m is not None
        assert (m.source, m.corner) == ('spice', 'ss')
        assert m.passes == next(r.passed for r in report.results if r.name == s['name'])


def test_evaluate_relative_rawfile_resolves_against_testbench_dir(pdk, tmp_dir):
    """A bare filename resolves against the per-test directory (load_file's
    contract), so `evaluate('tt', rawfile='rawfile.raw')` finds the file a
    remote run dropped next to the deck."""
    tb = _fixture_tb(pdk, tmp_dir)
    tb.set_temperature(27)
    os.makedirs(tb.netlist.dir_folder, exist_ok=True)
    shutil.copy(FIXTURES / 'e_ng_tran.raw', os.path.join(tb.netlist.dir_folder, 'rawfile.raw'))
    report = tb.evaluate('tt', rawfile='rawfile.raw')
    assert report.metrics['out_max'] == pytest.approx(1.3723, abs=1e-3)
    assert tb.results_file == os.path.join(tb.netlist.dir_folder, 'rawfile.raw')


# ---------------------------------------------------------------------------
# AC4 — nothing loaded, no rawfile: typed error naming both entry points
# ---------------------------------------------------------------------------

def test_evaluate_without_results_raises_typed_error(pdk, tmp_dir):
    """Pre-fix there was nothing to call; the private tail on an unloaded
    netlist would have swallowed every extractor failure into
    'Error: ...' strings and reported specs as failed. evaluate() with
    nothing loaded raises SimulationError naming generate_netlist(),
    read_raw_file() and evaluate(rawfile=...)."""
    tb = _fixture_tb(pdk, tmp_dir)
    tb.set_temperature(27)
    assert not tb.netlist.sim_results_loaded
    with pytest.raises(SimulationError) as excinfo:
        tb.evaluate('tt')
    msg = str(excinfo.value)
    for token in ('generate_netlist', 'read_raw_file', 'rawfile=', 'run()'):
        assert token in msg
    # Nothing was recorded on the specs.
    assert all(s['spec'].measurements == {} for s in tb._specs)


def test_evaluate_before_add_module_under_test_raises(pdk, tmp_dir):
    tb = _dut(pdk, tmp_dir).create_test('empty')
    with pytest.raises(ValueError, match='add_module_under_test'):
        tb.evaluate('tt', rawfile=str(FIXTURES / 'e_ng_tran.raw'))


def test_evaluate_on_already_loaded_results(pdk, tmp_dir):
    """The read_raw_file() entry point: load through the public Module API
    first, then evaluate() with no rawfile runs the tail on what is
    loaded."""
    tb = _fixture_tb(pdk, tmp_dir)
    tb.set_temperature(27)
    tb.netlist.load_file(str(FIXTURES / 'e_xy_tran.raw'))
    tb.netlist.read_raw_file()
    report = tb.evaluate('ff')
    assert report.metrics['out_max'] == pytest.approx(1.3723, abs=1e-3)
    assert all(s['spec'].latest().corner == 'ff' for s in tb._specs)


def test_evaluate_undeclared_corner_is_unknown_not_pdk_default(pdk, tmp_dir):
    """TASK-407: an external file with no corner= and no set_corner() records
    corner unknown; the PDK default is not assumed."""
    tb = _fixture_tb(pdk, tmp_dir)
    tb.set_temperature(27)
    report = tb.evaluate(rawfile=str(FIXTURES / 'e_ng_tran.raw'))
    assert report.operating_conditions.corner_kind == 'unknown'
    assert all(s['spec'].latest().corner is None for s in tb._specs)


def test_evaluate_set_corner_is_the_declared_external_corner(pdk, tmp_dir):
    tb = _fixture_tb(pdk, tmp_dir)
    tb.set_corner('ff')
    tb.evaluate(rawfile=str(FIXTURES / 'e_ng_tran.raw'))
    assert all(s['spec'].latest().corner == 'ff' for s in tb._specs)


# ---------------------------------------------------------------------------
# AC2 — run() is simulate + the shared tail (one copy); its guards, caching
# and context behaviour are pinned by the existing suite
# (tests/test_sim_cache.py, tests/test_characterization_context_propagation.py).
# ---------------------------------------------------------------------------

def test_run_calls_simulate_then_public_evaluate_once(pdk, tmp_dir):
    """run() = netlist.simulate(corner) followed by exactly one call of the
    PUBLIC evaluate() (TASK-115: _simulate has no private _evaluate_loaded
    twin to call instead), with the corner and the context run() attached
    to the netlist reaching evaluate() unchanged."""
    tb = _fixture_tb(pdk, tmp_dir)
    tb.set_temperature(27)
    calls = []
    real_evaluate = tb.evaluate

    def spy(corner=None, **kwargs):
        calls.append((corner, kwargs, tb.netlist.characterization_context))
        return real_evaluate(corner, **kwargs)

    with _stub_simulate_with(tb, FIXTURES / 'e_ng_tran.raw',
                            voltage=1.8) as sim, \
            patch.object(tb, 'evaluate', side_effect=spy):
        report = tb.run('ff', temperature=27.0, voltage=1.8)
    assert sim.call_count == 1
    assert len(calls) == 1
    assert calls[0][0] == 'ff'
    # run() calls evaluate(corner) with no rawfile/temperature/voltage of
    # its own — the context reaching evaluate() is exactly the one run()
    # already attached, reused rather than rebuilt (see evaluate()'s
    # docstring "Context reuse").
    assert calls[0][1] == {}
    assert calls[0][2] is report.characterization_context


# ---------------------------------------------------------------------------
# AC3 slow leg — real ngspice + sky130: run() then evaluate() on the rawfile
# run() wrote
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(not os.path.exists(_SKY130_LIB),
                    reason='sky130 ngspice models not installed')
@pytest.mark.skipif(shutil.which('ngspice') is None, reason='ngspice not installed')
@pytest.mark.parametrize('entrypoint', ['run', 'run_test'])
def test_slow_run_then_evaluate_real_ngspice(tmp_path, entrypoint):
    """sky130 inverter driven by a pulse, .tran through real ngspice: run('tt')
    then evaluate('tt', rawfile=tb.results_file) — same rawfile ngspice
    wrote — must be field-for-field identical, with vout swinging to the
    1.8 V rail (plus edge overshoot) and the two specs (one pass, one fail)
    carrying identical Measurements."""
    pdk = BasePDK(_PDK_YAML, library_location=_SKY130_LIB)
    inv = Module('PDK_skywater130', str(tmp_path), 'inv.cir',
                 pdk_details=pdk, module_name='inv')
    inv.define_module_ports({'in': PortDir.IN, 'out': PortDir.OUT,
                             'vdd': PortDir.POWER, 'vss': PortDir.GROUND})
    with inv:
        pdk.nmos1p8('mn', d='out', g='in', s='vss', b='vss', w=1.0, l=0.15)
        pdk.pmos1p8('mp', d='out', g='in', s='vdd', b='vdd', w=2.0, l=0.15)

    tb = inv.create_test('tran')
    tb.add_module_under_test({'in': 'vin', 'out': 'vout', 'vdd': 'vdd', 'vss': '0'})
    tb.add(sources.vdc('vdd', positive='vdd', negative='0', dc=1.8))
    tb.add(sources.vpulse('vin', positive='vin', negative='0', v1=0.0, v2=1.8,
                          td=1e-9, tr=100e-12, tf=100e-12, pw=5e-9, per=10e-9))
    tb.add_sim_type(analysis.Transient(step=10e-12, stop=20e-9))
    tb.add_metric('vout_max', lambda ckt: float(np.max(ckt.load_signal('vout').y)))
    tb.add_metric('vout_min', lambda ckt: float(np.min(ckt.load_signal('vout').y)))
    tb.add_spec('swings high', 'vout_max', '>', 1.7)
    tb.add_spec('impossible', 'vout_min', '>', 1.0)

    with patch('silk.signals.rawfile.read_rawfile', wraps=read_rawfile) as reader, \
            patch('silk.simulation.runner.read_rawfile', reader), \
            patch('subprocess.run', wraps=subprocess.run) as runner:
        if entrypoint == 'run':
            report_a = tb.run('tt')
        else:
            report_a = inv.run_test('tran', corner='tt')
        # The once-per-binary version probe is not a simulation.
        launches = [c for c in runner.call_args_list if '-b' in c.args[0]]
        assert len(launches) == 1
        assert reader.call_count == 1
    rawfile = tb.results_file
    assert rawfile and os.path.exists(rawfile)
    after_run = _snapshot_measurements(tb)
    # No load cap: the 100 ps input edge couples through Cgd, so the rail-
    # to-rail swing overshoots (1.88 V measured on ngspice-42 / sky130A).
    assert report_a.metrics['vout_max'] == pytest.approx(1.8, abs=0.15)
    assert [r.passed for r in report_a.results] == [True, False]

    with patch('silk.signals.rawfile.read_rawfile', wraps=read_rawfile) as reader, \
            patch.object(tb.netlist, 'simulate',
                         side_effect=AssertionError('evaluate() launched a simulation')):
        report_b = tb.evaluate('tt', rawfile=rawfile)
        assert reader.call_count == 1
    _assert_reports_identical(report_a, report_b)
    assert _snapshot_measurements(tb) == after_run
    assert tb.results_file == rawfile


# ---------------------------------------------------------------------------
# TASK-116 — the no-data gate on the LOAD path: evaluate(rawfile=...) must not
# measure an aborted run's constants plot
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('fixture', ['d_constants_only.raw',
                                     'd_multiplot_constants_last.raw'])
def test_evaluate_on_aborted_rawfile_raises_and_records_nothing(pdk, tmp_dir, fixture):
    """Pre-fix (TASK-116): _require_data guarded only the three run() paths,
    so evaluate(rawfile=<aborted run's rawfile>) loaded ngspice's synthetic
    'constants' plot (npoints=1, no analysis data) as results and RETURNED a
    SpecReport — metrics extracted from it and a source='spice' Measurement
    recorded on every attached Spec, i.e. an aborted run scored green/red
    against the specs. Now the gate is in backend.load_raw, so evaluate()
    raises SimulationAborted (data_rows=0) before any metric runs and every
    Spec's measurements dict is still empty."""
    from silk.exceptions import SimulationAborted

    tb = _fixture_tb(pdk, tmp_dir)
    tb.set_temperature(27)
    rawfile = FIXTURES / fixture
    with pytest.raises(SimulationAborted) as excinfo:
        tb.evaluate('tt', rawfile=str(rawfile))

    err = excinfo.value
    assert err.data_rows == 0
    # load_file's simulatorType hint, defaulted: evaluate() does not ask the
    # user which simulator ran the deck elsewhere.
    assert err.simulator == 'ngspice'
    assert err.reason is None
    # The message says the file was handed to silk, not produced by a run here.
    assert 'Supplied rawfile' in str(err) and str(rawfile) in str(err)
    # NOTHING was measured: no Measurement on any Spec, and the netlist did
    # not come up as "results loaded".
    assert all(s['spec'].measurements == {} for s in tb._specs)
    assert not tb.netlist.sim_results_loaded


def test_evaluate_abort_carries_explicit_reason(pdk, tmp_dir):
    """A caller that ran the simulator elsewhere can hand over the reason it
    already scraped: evaluate(rawfile=..., reason=...) puts it on the
    exception and in the message. Pre-fix there was no exception at all."""
    from silk.exceptions import SimulationAborted

    tb = _fixture_tb(pdk, tmp_dir)
    tb.set_temperature(27)
    with pytest.raises(SimulationAborted) as excinfo:
        tb.evaluate('tt', rawfile=str(FIXTURES / 'd_constants_only.raw'),
                    reason='remote runner: convergence failure')
    assert excinfo.value.reason == 'remote runner: convergence failure'
    assert 'convergence failure' in str(excinfo.value)
    assert all(s['spec'].measurements == {} for s in tb._specs)


def test_evaluate_abort_scans_log_the_way_run_does(pdk, tmp_dir, tmp_path):
    """log=<the simulator's log where it ran> is scanned by the same
    abort_reason_from_log run() uses on ngspice's output.log — the sentinel
    line comes back on the exception, and log_path is carried for a UI."""
    from silk.exceptions import SimulationAborted

    log = tmp_path / 'output.log'
    log.write_text('doAnalyses: Timestep too small; time = 1.2e-18\n'
                   'run simulation(s) aborted\n')
    tb = _fixture_tb(pdk, tmp_dir)
    tb.set_temperature(27)
    with pytest.raises(SimulationAborted) as excinfo:
        tb.evaluate('tt', rawfile=str(FIXTURES / 'd_constants_only.raw'),
                    log=str(log))
    assert 'Timestep too small' in (excinfo.value.reason or '')
    assert excinfo.value.log_path == str(log)
    assert all(s['spec'].measurements == {} for s in tb._specs)


def test_read_raw_file_on_aborted_rawfile_raises_with_hint_simulator(pdk, tmp_dir):
    """The gate lives in backend.load_raw, so the OTHER public entry point —
    load_file() + read_raw_file() — is covered too, and load_file's
    simulatorType hint (dead until TASK-116) names the simulator on the
    exception. Pre-fix read_raw_file() set sim_results_loaded=True on the
    constants plot."""
    from silk.exceptions import SimulationAborted

    tb = _fixture_tb(pdk, tmp_dir)
    tb.set_temperature(27)
    tb.netlist.load_file(str(FIXTURES / 'd_constants_only.raw'), 'xyce')
    with pytest.raises(SimulationAborted) as excinfo:
        tb.netlist.read_raw_file()
    assert excinfo.value.simulator == 'xyce'
    assert excinfo.value.data_rows == 0
    assert not tb.netlist.sim_results_loaded


def test_load_raw_without_a_hint_attributes_the_abort_to_the_rawfile():
    """A bare backend.load_raw(path) — no Module, no hint — still gates; with
    nothing to name the simulator the abort is attributed to 'rawfile'."""
    from silk.exceptions import SimulationAborted

    be = SimulatorBackend.get('ngspice')
    with pytest.raises(SimulationAborted) as excinfo:
        be.load_raw(str(FIXTURES / 'd_constants_only.raw'))
    assert excinfo.value.simulator == 'rawfile'
    assert excinfo.value.data_rows == 0


@pytest.mark.parametrize('fixture', ['e_ng_op.raw', 'e_xy_op.raw',
                                     'e_ng_noise.raw', 'd_healthy_transient.raw',
                                     'd_multiplot_constants_first.raw'])
def test_gate_lets_real_results_through(fixture):
    """The gate must not cost a legitimate load. A .op rawfile is a REAL
    one-point plot (npoints=1, like the constants plot) — _has_data_rows
    checks the plotname first, so it loads; so do noise, transient, and a
    multiplot whose constants plot is not last."""
    raw = SimulatorBackend.get('ngspice').load_raw(str(FIXTURES / fixture))
    assert raw.native is not None and raw.native.npoints >= 1


def test_gate_skips_a_xyce_noise_only_result():
    """A Xyce noise-only run has no rawfile at all (results are CSV): its
    RawResult.native is None and the gate leaves it alone, so the noise
    signal still loads."""
    from silk.simulation.backend import RawResult, _require_data

    raw = RawResult(native=None,
                    source_path=str(FIXTURES / 'e_xy_noise' / 'rawfile.raw'))
    _require_data(raw, simulator='xyce', supplied=True)   # must not raise
    sig = SimulatorBackend.get('xyce').load_signal(raw, 'noise_out')
    assert len(sig.y) > 0


def test_rebuilding_declared_target_changes_fresh_report_with_cache_disabled(
    pdk, tmp_dir,
):
    def build(target):
        dut = _dut(pdk, tmp_dir)
        tb = ModuleTestbench(dut, 'target_check', cache=False)
        tb.add_module_under_test(
            {'a': 'step', 'b': 'out', 'vdd': 'clk', 'vss': '0'},
        )
        tb.add_metric('out_max', lambda ckt: float(np.max(ckt.load_signal('out').y)))
        spec = tb.add_spec('Output threshold', 'out_max', '>', target, key='out')
        return tb, spec

    first_tb, first_spec = build(1.0)
    rawfile = FIXTURES / 'e_ng_tran.raw'
    with _stub_simulate_with(first_tb, rawfile) as simulator:
        first_report = first_tb.run('tt')
        first_measurement = first_spec.latest()
        with pytest.raises(AttributeError, match='target.*declaration.*rebuild'):
            first_spec.target = 2.0
        repeated_report = first_tb.run('tt')
        assert simulator.call_count == 2
    assert repeated_report is not first_report
    assert first_spec.latest() is not first_measurement
    assert first_report.passed is True and repeated_report.passed is True
    assert first_report.results[0].target == 1.0
    assert first_spec.target == 1.0

    rebuilt_tb, rebuilt_spec = build(2.0)
    with _stub_simulate_with(rebuilt_tb, rawfile) as simulator:
        rebuilt_report = rebuilt_tb.run('tt')
        assert simulator.call_count == 1
    assert rebuilt_report.passed is False
    assert rebuilt_report.results[0].target == 2.0
    assert rebuilt_spec.key == first_spec.key == 'out'

    for tb, spec, report, passes in (
        (first_tb, first_spec, repeated_report, True),
        (rebuilt_tb, rebuilt_spec, rebuilt_report, False),
    ):
        measured = spec.latest()
        assert measured.value == pytest.approx(1.372346, abs=1e-6)
        assert measured.passes is passes
        assert spec.evaluate(measured) is passes
        reconciled = spec.module.reconcile_specs().outcomes[0]
        assert reconciled.status == ('PASS' if passes else 'FAIL')
        assert measured.passes is passes
        evaluated = tb.evaluate('tt', rawfile=str(rawfile))
        _assert_reports_identical(report, evaluated)

    assert first_report.results[0].target == 1.0
    assert first_report.passed is True
    assert first_measurement.passes is True


def _external_tb(pdk, tmp_path, *, db_path=None):
    """Only measure the output shared by both healthy replacement fixtures."""
    dut = _dut(pdk, str(tmp_path), db_path=db_path)
    tb = dut.create_test('external')
    tb.add_module_under_test({'a': 'step', 'b': 'out', 'vdd': 'clk', 'vss': '0'})
    samples = []

    def out_max(ckt):
        signal = ckt.load_signal('out')
        samples.append(signal.y.copy())
        return float(np.max(signal.y))

    tb.add_metric('out_max', out_max)
    spec = tb.add_spec('Output threshold', 'out_max', '>', 1.0, key='out')
    return tb, spec, samples


@pytest.mark.parametrize('fixture', ['e_ng_tran.raw', 'e_xy_tran.raw'])
@pytest.mark.parametrize('path_kind', ['absolute', 'relative', 'different'])
def test_explicit_evaluate_reads_replacement_signals_and_measurements(
    pdk, tmp_path, fixture, path_kind,
):
    """Same-path reuse must not turn the replacement's 0.991 V into a PASS."""
    tb, spec, samples = _external_tb(pdk, tmp_path)
    path = Path(tb.netlist.dir_folder) / 'external.raw'
    shutil.copyfile(FIXTURES / fixture, path)
    argument = path.name if path_kind == 'relative' else str(path)

    with patch('silk.signals.rawfile.read_rawfile', wraps=read_rawfile) as reader, \
            patch.object(tb.netlist, 'simulate', side_effect=AssertionError('simulation launched')):
        first = tb.evaluate('ff', rawfile=argument, temperature=27.0, voltage=1.8)
        first_measurement = spec.latest()
        assert reader.call_count == 1
        unchanged = tb.evaluate('ff', rawfile=argument, temperature=27.0, voltage=1.8)
        unchanged_reads = reader.call_count
        _assert_reports_identical(first, unchanged)

        if path_kind == 'different':
            path = path.with_name('replacement.raw')
            argument = str(path)
        shutil.copyfile(FIXTURES / 'd_healthy_transient.raw', path)
        second = tb.evaluate('ff', rawfile=argument)
        replacement_reads = reader.call_count

    assert first.metrics['out_max'] == pytest.approx(1.372346, abs=1e-6)
    assert second.metrics['out_max'] == pytest.approx(0.991456, abs=1e-6)
    assert unchanged_reads == 2  # Even an unchanged file must be opened.
    assert replacement_reads == 3
    assert np.max(samples[0]) == pytest.approx(1.372346, abs=1e-6)
    assert np.max(samples[-1]) == pytest.approx(0.991456, abs=1e-6)
    assert np.max(second.sim_results.load_signal('out').y) == pytest.approx(0.991456, abs=1e-6)
    assert first.passed is True and second.passed is False
    assert first.results[0].target == second.results[0].target == spec.target == 1.0
    assert first_measurement.value == pytest.approx(1.372346, abs=1e-6)
    assert first_measurement.passes is True
    measured = spec.latest()
    assert measured is not first_measurement
    assert measured.value == pytest.approx(0.991456, abs=1e-6)
    assert (measured.source, measured.corner, measured.passes) == ('spice', 'ff', False)
    assert first.characterization_context.corner == 'ff'
    assert second.characterization_context is None
    assert tb.netlist.characterization_context is None
    assert second.corner == 'ff'
    assert tb.results_file == str(path)


@pytest.mark.parametrize('replacement,diagnostics,error', [
    ('constants', 'none', SimulationAborted),
    ('constants', 'reason', SimulationAborted),
    ('constants', 'log', SimulationAborted),
    ('constants', 'both', SimulationAborted),
    ('deleted', 'none', RawfileError),
    ('malformed', 'none', RawfileError),
    ('directory', 'none', RawfileError),
])
def test_failed_explicit_reload_clears_live_results_but_preserves_evidence(
    pdk, tmp_path, replacement, diagnostics, error,
):
    """A failed import must neither expose old samples nor write new evidence."""
    tb, spec, samples = _external_tb(pdk, tmp_path, db_path=str(tmp_path / 'evidence.db'))
    path = tmp_path / 'external.raw'
    shutil.copyfile(FIXTURES / 'e_ng_tran.raw', path)
    try:
        tb.module.create_circuit_version()
        first = tb.evaluate('ff', rawfile=str(path), temperature=27.0, voltage=1.8)
        assert tb.module.store_report(first)['specs_stored'] == 1
        measurements = dict(spec.measurements)
        conn = tb.module._tracker.db.conn
        stored = list(conn.iterdump())

        if replacement == 'constants':
            shutil.copyfile(FIXTURES / 'd_constants_only.raw', path)
        elif replacement == 'malformed':
            path.write_text('external simulator returned a broken rawfile\n')
        else:
            path.unlink()
            if replacement == 'directory':
                path.mkdir()  # A path that exists but cannot be read as a file.

        kwargs = {}
        if diagnostics in ('log', 'both'):
            log = tmp_path / 'output.log'
            log.write_text('doAnalyses: Timestep too small; time = 1.2e-18\n')
            kwargs['log'] = str(log)
        if diagnostics in ('reason', 'both'):
            kwargs['reason'] = 'remote runner: convergence failure'

        with pytest.raises(error) as excinfo:
            tb.evaluate('ss', rawfile=str(path), **kwargs)
        assert str(path) in str(excinfo.value)
        if replacement == 'constants':
            assert excinfo.value.data_rows == 0
            assert excinfo.value.simulator == 'ngspice'
            if 'reason' in kwargs:
                assert excinfo.value.reason == kwargs['reason']
            elif 'log' in kwargs:
                assert 'Timestep too small' in excinfo.value.reason
            else:
                assert excinfo.value.reason is None
            if 'log' in kwargs:
                assert excinfo.value.log_path == kwargs['log']

        assert not tb.netlist.sim_results_loaded
        assert tb.netlist._raw is None
        assert tb.netlist.simResultsAll is None
        assert tb.results_file is None
        assert tb.netlist.dataFile == str(path)
        assert tb.netlist.characterization_context is None
        with pytest.raises(SimulationError, match='no simulation results'):
            tb.evaluate('ss')
        with pytest.raises(SimulationError, match='not Loaded'):
            first.sim_results.load_signal('out')
        assert len(samples) == 1  # Failed imports never reach the extractor.
        assert spec.measurements == measurements
        assert all(spec.measurements[key] is value for key, value in measurements.items())
        assert list(conn.iterdump()) == stored
        assert first.passed is True
        assert first.metrics['out_max'] == pytest.approx(1.372346, abs=1e-6)
        assert first.characterization_context.corner == 'ff'

        if path.is_dir():
            path.rmdir()
        shutil.copyfile(FIXTURES / 'd_healthy_transient.raw', path)
        recovered = tb.evaluate('ss', rawfile=str(path), temperature=85.0, voltage=1.6)
        assert recovered.metrics['out_max'] == pytest.approx(0.991456, abs=1e-6)
        assert recovered.passed is False
        assert recovered.characterization_context.corner == 'ss'
        assert tb.netlist.characterization_context is recovered.characterization_context
        assert tb.results_file == str(path)
        assert len(samples) == 2
        assert list(conn.iterdump()) == stored  # evaluate() still never auto-stores.
    finally:
        tb.module.close()


def test_omitted_rawfile_uses_memory_after_the_file_is_deleted(pdk, tmp_path):
    tb, spec, _ = _external_tb(pdk, tmp_path)
    path = tmp_path / 'external.raw'
    shutil.copyfile(FIXTURES / 'e_ng_tran.raw', path)
    first = tb.evaluate('ff', rawfile=str(path), temperature=27.0, voltage=1.8)
    path.unlink()
    with patch.object(tb.netlist, 'read_raw_file', side_effect=AssertionError('disk read')), \
            patch.object(tb.netlist, 'simulate', side_effect=AssertionError('simulation launched')):
        repeated = tb.evaluate('ff')
        _assert_reports_identical(first, repeated)
        assert repeated.characterization_context is first.characterization_context
        updated = tb.evaluate('ss', temperature=85.0, voltage=1.6)
    assert updated.metrics == first.metrics
    assert updated.characterization_context.corner == spec.latest().corner == 'ss'
    assert updated.characterization_context.temperature == 85.0


def test_rejected_duplicate_preserves_handle_spec_and_retained_evidence(pdk, tmp_dir):
    tb = _fixture_tb(pdk, tmp_dir)
    original = tb.add_metric('gain', lambda ckt: float(np.max(ckt.load_signal('out').y)))
    derived = tb.add_metric('double_gain', original * 2)
    spec = tb.add_spec('original gain', original, '>', 1.0)
    report = tb.evaluate(rawfile=str(FIXTURES / 'e_ng_tran.raw'))
    declarations = list(tb._metrics)
    measurements = _snapshot_measurements(tb)
    with pytest.raises(ValueError, match='fix.*gain.*distinct name'):
        tb.add_metric('gain', lambda ckt: pytest.fail('replacement must never run'))
    assert tb._metrics == declarations
    assert _snapshot_measurements(tb) == measurements
    assert original.value == report.metrics['gain'] == pytest.approx(1.3723464265661118)
    assert derived.value == report.metrics['double_gain'] == pytest.approx(2.7446928531322236)
    assert report.sim_results.load_signal('out').y.max() == pytest.approx(1.3723464265661118)
    repeated = tb.evaluate()
    assert repeated.metrics == report.metrics
    assert repeated.results[-1].value == pytest.approx(1.3723464265661118)
    assert spec.metric == 'gain'
