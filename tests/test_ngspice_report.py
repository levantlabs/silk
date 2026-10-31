"""Tests for ngspice-aware _build_simulation — CircuitReport now
populates SignalStats from module.simResultsAll after simulate()
(not only from Net.voltage after simulate_behavioral())."""

import types

import numpy as np
import pytest

from silk.circuit.modules import Module
from silk.circuit.report import (
    _build_simulation,
    SimulationSummary,
    generate_report,
)


# ---------------------------------------------------------------------------
# Fake rawfile — mimics the shape silk expects from silk.signals.rawfile
# ---------------------------------------------------------------------------

def _fake_vector(name, data):
    return types.SimpleNamespace(name=name, data=np.asarray(data))


def _fake_raw(vectors):
    return types.SimpleNamespace(vectors=vectors, nvars=len(vectors))


def _mount_ngspice_result(module, vectors):
    """Simulate what read_raw_file() would do after a real ngspice run."""
    module.simResultsAll = _fake_raw(vectors)
    module.sim_results_loaded = True


# ---------------------------------------------------------------------------
# Core: ngspice-only SimulationSummary
# ---------------------------------------------------------------------------

class TestNgspiceSimulationSummary:

    def test_returns_none_when_no_sim(self, pdk, tmp_dir):
        mod = Module('PDK_skywater130', tmp_dir, 'm.cir', pdk_details=pdk)
        assert _build_simulation(mod) is None

    def test_populates_from_rawfile(self, pdk, tmp_dir):
        mod = Module('PDK_skywater130', tmp_dir, 'm.cir', pdk_details=pdk)
        time = np.linspace(0, 1e-6, 11)
        vout = np.linspace(0.0, 1.8, 11)
        vin  = np.full(11, 0.9)
        _mount_ngspice_result(mod, [
            _fake_vector('time',   time),
            _fake_vector('v(vout)', vout),
            _fake_vector('v(vin)',  vin),
        ])

        summary = _build_simulation(mod)
        assert summary is not None
        assert summary.t_start == pytest.approx(0.0)
        assert summary.t_stop  == pytest.approx(1e-6)
        assert summary.n_points == 11
        names = [s.name for s in summary.signals]
        assert 'vout' in names
        assert 'vin'  in names

    def test_signal_stats_correct(self, pdk, tmp_dir):
        mod = Module('PDK_skywater130', tmp_dir, 'm.cir', pdk_details=pdk)
        vout = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        _mount_ngspice_result(mod, [
            _fake_vector('time',    np.linspace(0, 4e-6, 5)),
            _fake_vector('v(vout)', vout),
        ])

        summary = _build_simulation(mod)
        s = next(s for s in summary.signals if s.name == 'vout')
        assert s.min  == pytest.approx(0.0)
        assert s.max  == pytest.approx(2.0)
        assert s.mean == pytest.approx(1.0)

    def test_hierarchical_net_names_preserved(self, pdk, tmp_dir):
        """ngspice uses names like 'v(xfcamp.vbiasn)' for subcircuit nets.
        _build_simulation must keep the path, not drop to just 'vbiasn'."""
        mod = Module('PDK_skywater130', tmp_dir, 'm.cir', pdk_details=pdk)
        _mount_ngspice_result(mod, [
            _fake_vector('time',              np.linspace(0, 1e-6, 3)),
            _fake_vector('v(xfcamp.vbiasn)',  np.array([0.0, 0.7, 1.44])),
            _fake_vector('v(vout)',           np.array([0.0, 0.4, 0.89])),
        ])
        summary = _build_simulation(mod)
        names = [s.name for s in summary.signals]
        assert 'xfcamp.vbiasn' in names
        assert 'vout'          in names

    def test_current_signals_kept_with_i_prefix(self, pdk, tmp_dir):
        """Currents (e.g. i(vsupply)) should stay distinguishable from
        voltages in the stats list."""
        mod = Module('PDK_skywater130', tmp_dir, 'm.cir', pdk_details=pdk)
        _mount_ngspice_result(mod, [
            _fake_vector('time',          np.linspace(0, 1e-6, 3)),
            _fake_vector('v(vout)',       np.array([0.0, 1.0, 0.5])),
            _fake_vector('i(vsupply)',    np.array([-1e-3, -1e-3, -1e-3])),
        ])
        summary = _build_simulation(mod)
        names = [s.name for s in summary.signals]
        assert 'vout' in names
        assert any('i(' in n for n in names), \
            "current signals should keep an 'i(...)' marker"

    def test_frequency_axis_ac_sim(self, pdk, tmp_dir):
        """A 'frequency' x-axis lands in f_start/f_stop, not t_start/t_stop.

        Pre-TASK-82 this test asserted ``t_start == 1.0`` and ``t_stop ==
        1e6`` — pinning the defect that a 1 Hz..1 MHz sweep was reported
        through fields named for time. A real-valued frequency-axis plot is
        the 'noise' kind; its vectors are listed as magnitudes.
        """
        mod = Module('PDK_skywater130', tmp_dir, 'm.cir', pdk_details=pdk)
        _mount_ngspice_result(mod, [
            _fake_vector('frequency', np.logspace(0, 6, 7)),
            _fake_vector('v(vout)',   np.ones(7)),
        ])
        summary = _build_simulation(mod)
        assert summary is not None
        assert summary.analysis == 'noise'
        assert summary.t_start is None and summary.t_stop is None
        assert summary.f_start == pytest.approx(1.0)
        assert summary.f_stop  == pytest.approx(1e6)
        assert summary.signals == []
        assert [m.name for m in summary.magnitudes] == ['vout']

    def test_sim_results_loaded_false_is_noop(self, pdk, tmp_dir):
        """Even with simResultsAll attached, if sim_results_loaded is False
        _build_simulation must not return stale stats."""
        mod = Module('PDK_skywater130', tmp_dir, 'm.cir', pdk_details=pdk)
        mod.simResultsAll = _fake_raw([
            _fake_vector('time',    np.linspace(0, 1e-6, 3)),
            _fake_vector('v(vout)', np.ones(3)),
        ])
        # deliberately leave sim_results_loaded == False
        assert _build_simulation(mod) is None

    def test_empty_rawfile_returns_none(self, pdk, tmp_dir):
        mod = Module('PDK_skywater130', tmp_dir, 'm.cir', pdk_details=pdk)
        _mount_ngspice_result(mod, [])
        assert _build_simulation(mod) is None


# ---------------------------------------------------------------------------
# Behavioral path still works (no regression)
# ---------------------------------------------------------------------------

class TestBehavioralStillWorks:

    def test_behavioral_path_still_returns_summary(self, pdk, tmp_dir):
        """_build_simulation's behavioral branch must remain intact —
        regression guard for anyone hitting simulate_behavioral()."""
        mod = Module('PDK_skywater130', tmp_dir, 'm.cir', pdk_details=pdk)
        # Auto-create net via add_net and attach a voltage array like the
        # behavioral engine would.
        mod.add_net('vout')
        t = np.linspace(0, 1e-6, 11)
        mod.net_dict['vout'].voltage = np.linspace(0.0, 1.8, 11)
        mod._behavioral_t = t
        mod._behavioral_mode = True

        summary = _build_simulation(mod)
        assert summary is not None
        assert summary.n_points == 11
        assert any(s.name == 'vout' for s in summary.signals)


# ---------------------------------------------------------------------------
# End-to-end: generate_report() picks it up
# ---------------------------------------------------------------------------

class TestEndToEnd:

    def test_generate_report_shows_ngspice_sim(self, pdk, tmp_dir):
        """After simulate() + read_raw_file, generate_report().simulation
        is populated — previously None for ngspice runs."""
        mod = Module('PDK_skywater130', tmp_dir, 'm.cir', pdk_details=pdk)
        _mount_ngspice_result(mod, [
            _fake_vector('time',    np.linspace(0, 1e-6, 11)),
            _fake_vector('v(vout)', np.linspace(0.0, 1.0, 11)),
            _fake_vector('v(vin)',  np.full(11, 0.5)),
        ])
        rpt = generate_report(mod)
        assert rpt.simulation is not None
        assert rpt.simulation.n_points == 11
        assert len(rpt.simulation.signals) >= 2


# ---------------------------------------------------------------------------
# print_summary() — optional sections for sim op-point and spec metrics
# ---------------------------------------------------------------------------

class TestPrintSummarySections:
    """Sections auto-appear when data is present, stay silent otherwise."""

    def test_operating_point_section_absent_without_sim(self, pdk, tmp_dir, capsys):
        mod = Module('PDK_skywater130', tmp_dir, 'm.cir', pdk_details=pdk)
        generate_report(mod).print_summary()
        out = capsys.readouterr().out
        assert 'Operating point' not in out
        assert 'Metrics:' not in out

    def test_operating_point_section_prints_for_module_nets(
            self, pdk, tmp_dir, capsys):
        """Only nets present in module.net_dict appear — subcircuit-internal
        ngspice signals like 'xfcamp.vbiasn' should be filtered out."""
        mod = Module('PDK_skywater130', tmp_dir, 'm.cir', pdk_details=pdk)
        mod.add_net('vout')
        mod.add_net('vinp')
        _mount_ngspice_result(mod, [
            _fake_vector('time',             np.linspace(0, 1e-6, 5)),
            _fake_vector('v(vout)',          np.array([0.0, 0.5, 0.9, 0.92, 0.93])),
            _fake_vector('v(vinp)',          np.full(5, 0.9)),
            _fake_vector('v(xfcamp.vint)',   np.linspace(0, 1.8, 5)),
        ])
        generate_report(mod).print_summary()
        out = capsys.readouterr().out
        assert 'Operating point (module nets):' in out
        assert 'vout' in out
        assert 'vinp' in out
        assert 'xfcamp.vint' not in out

    def test_operating_point_shows_mean_value(self, pdk, tmp_dir, capsys):
        mod = Module('PDK_skywater130', tmp_dir, 'm.cir', pdk_details=pdk)
        mod.add_net('vout')
        _mount_ngspice_result(mod, [
            _fake_vector('time',    np.linspace(0, 1e-6, 5)),
            _fake_vector('v(vout)', np.array([0.0, 0.5, 1.0, 1.5, 2.0])),
        ])
        generate_report(mod).print_summary()
        out = capsys.readouterr().out
        # mean of [0, 0.5, 1.0, 1.5, 2.0] == 1.0
        assert '1.000 V' in out
        # min/max shown as parenthetical range
        assert '0.000' in out and '2.000' in out

    def test_metrics_section_prints_when_specs_declared(
            self, pdk, tmp_dir, capsys):
        from silk.specs.spec import Measurement

        mod = Module('PDK_skywater130', tmp_dir, 'm.cir', pdk_details=pdk)
        s = mod.declare_spec(name='gain_dc', metric='gain_dc', op='>=',
                             target=60.0, units='dB')
        s.record_measurement(
            Measurement(source='spice', value=62.3, timestamp=0.0))
        generate_report(mod).print_summary()
        out = capsys.readouterr().out
        assert 'Metrics:' in out
        assert 'gain_dc' in out
        assert '62.3' in out
        assert 'PASS' in out

    def test_metrics_section_absent_without_specs(self, pdk, tmp_dir, capsys):
        mod = Module('PDK_skywater130', tmp_dir, 'm.cir', pdk_details=pdk)
        generate_report(mod).print_summary()
        out = capsys.readouterr().out
        assert 'Metrics:' not in out
