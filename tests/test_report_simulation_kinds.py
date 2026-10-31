"""``SimulationSummary`` across analysis kinds (TASK-82).

Every rawfile here is REAL simulator output — the TASK-74 slice-E fixtures
(``tests/fixtures/rawfiles/gen_e.py``; ngspice-42 and Xyce 7.10) — loaded
through the real path (``Module.dataFile`` -> ``Module.read_raw_file()`` ->
``backend.load_raw`` -> ``read_rawfile``), then summarised by
``generate_report``. Each docstring records what ``_build_simulation``
returned for the same fixture BEFORE the fix, measured on main at the time
this file was written, and why that value was wrong.

``sweep_name`` is case-folded the way ``NgspiceSignal.from_rawfile`` folds
``xlabel`` (TASK-87), so the Xyce cells assert the same canonical
``'time'`` / ``'frequency'`` spelling as the ngspice ones.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from silk.circuit.modules import Module
from silk.circuit.report import (
    CircuitReport, MagnitudeStats, SignalStats, SimulationSummary,
    _build_simulation, _simulation_from_dict, generate_report,
)

FIXTURES = Path(__file__).resolve().parent / 'fixtures' / 'rawfiles'

# gen_e.py's closed forms for the AC fixture: A0 = 1000, poles at 10 kHz,
# 3 MHz, 30 MHz; v(p1) is the single-pole node, v(out) the 3-pole output.
_A0, _FP1, _FP2, _FP3 = 1000.0, 10e3, 3e6, 30e6
_F_LO, _F_HI = 10.0, 100e6


def _h1(f):
    return _A0 / np.sqrt(1 + (f / _FP1) ** 2)


def _h3(f):
    return (_A0 / np.sqrt(1 + (f / _FP1) ** 2)
            / np.sqrt(1 + (f / _FP2) ** 2) / np.sqrt(1 + (f / _FP3) ** 2))


def _report_for(pdk, tmp_dir, fixture, nets=()):
    """generate_report on a module whose results are *fixture*, loaded
    through the real load path, with every warning captured. *nets* are
    declared on the module so the terminal op-point table (which filters
    to module nets) has something to show."""
    mod = Module('PDK_skywater130', tmp_dir, 'm.cir', pdk_details=pdk)
    for net in nets:
        mod.add_net(net)
    mod.dataFile = str(FIXTURES / fixture)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        mod.read_raw_file()
        report = generate_report(mod, auto_discover_config=False)
    return report, caught


def _complex_warnings(caught):
    return [w for w in caught if issubclass(w.category, np.exceptions.ComplexWarning)]


def _by_name(entries, name):
    return next(e for e in entries if e.name.lower() == name)


# ---------------------------------------------------------------------------
# Transient: field names and values preserved
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('fixture', ['e_ng_tran.raw', 'e_xy_tran.raw'])
def test_transient_keeps_time_fields_and_values(pdk, tmp_dir, fixture):
    """Pre-fix (both simulators): t_start=0, t_stop=5e-05, clk min=0
    max=1.8, out max=1.37235 — correct, and every one of those field names
    and values must survive so no transient JSON consumer changes."""
    report, caught = _report_for(pdk, tmp_dir, fixture)
    sim = report.simulation
    assert sim is not None
    assert not _complex_warnings(caught)
    assert sim.analysis == 'transient'
    assert sim.sweep_name == 'time'          # Xyce writes 'TIME'; folded
    assert sim.t_start == pytest.approx(0.0)
    assert sim.t_stop == pytest.approx(50e-6)
    assert (sim.sweep_start, sim.sweep_stop) == (sim.t_start, sim.t_stop)
    assert sim.f_start is None and sim.f_stop is None
    assert sim.magnitudes == []
    clk = _by_name(sim.signals, 'clk')
    assert clk.max == pytest.approx(1.8)
    assert abs(clk.min) < 1e-9
    out = _by_name(sim.signals, 'out')
    assert out.max == pytest.approx(1.37235, abs=1e-4)   # 37.2 % overshoot
    # Currents keep their i(...) tag so they cannot be read as voltages.
    assert any(s.name.lower().startswith('i(') for s in sim.signals)


def test_ngspice_sweep_name_is_unchanged_by_the_fold(pdk, tmp_dir):
    """ngspice already writes lower-case axis names; folding must be a
    no-op for them — 'time', 'frequency', 'v(v-sweep)' exactly."""
    for fixture, name in [('e_ng_tran.raw', 'time'), ('e_ng_ac.raw', 'frequency'),
                          ('e_ng_dc.raw', 'v(v-sweep)')]:
        report, _ = _report_for(pdk, tmp_dir, fixture)
        assert report.simulation.sweep_name == name


def test_transient_t_step_is_mean_spacing(pdk, tmp_dir):
    """``.tran 20n 50u`` on an adaptive grid: 2629 points over 50 us gives a
    MEAN spacing of 19.03 ns, which is what t_step reports (and reported
    before the fix); the docstring now says it is a mean, not the deck's
    max step."""
    report, _ = _report_for(pdk, tmp_dir, 'e_ng_tran.raw')
    sim = report.simulation
    assert sim.n_points == 2629
    assert sim.t_step == pytest.approx(50e-6 / 2628)
    assert sim.sweep_step == sim.t_step


# ---------------------------------------------------------------------------
# AC: magnitude statistics, frequency fields, no ComplexWarning
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('fixture', ['e_ng_ac.raw', 'e_xy_ac.raw'])
def test_ac_reports_magnitudes_under_frequency_fields(pdk, tmp_dir, fixture):
    """Pre-fix (both simulators): 9 ComplexWarnings per report, and
    ``out: min=-3.2831 max=999.999 mean=427.379`` — the REAL PART of a
    complex transfer ratio (a 3-pole amplifier's Re(H) goes negative past
    the phase crossover; |H| never does), while ``t_start=10
    t_stop=1e8`` presented the 10 Hz..100 MHz sweep as seconds.

    Post-fix the vectors are summarised by magnitude: |out| min = 8.62e-4
    (= |H3| at 100 MHz), max = 999.999 (= A0 at 10 Hz), |p1| min = 0.1
    (= A0*fp1/f at 100 MHz exactly), under f_start/f_stop, with t_* None.
    """
    report, caught = _report_for(pdk, tmp_dir, fixture)
    sim = report.simulation
    assert sim is not None
    assert _complex_warnings(caught) == []
    assert sim.analysis == 'ac'
    assert sim.sweep_name == 'frequency'     # Xyce writes 'FREQUENCY'; folded
    assert sim.t_start is None and sim.t_stop is None and sim.t_step is None
    assert sim.f_start == pytest.approx(_F_LO)
    assert sim.f_stop == pytest.approx(_F_HI)
    assert (sim.sweep_start, sim.sweep_stop) == (sim.f_start, sim.f_stop)
    assert sim.sweep_step is None            # decade sweep: no single step
    assert sim.n_points == 701
    assert sim.signals == []                 # nothing here is a voltage
    assert sim.n_signals == 3

    out = _by_name(sim.magnitudes, 'out')
    assert out.mag_min == pytest.approx(_h3(_F_HI), rel=1e-6)
    assert out.mag_max == pytest.approx(_h3(_F_LO), rel=1e-6)
    assert out.mag_min > 0                  # a magnitude; Re(H) was -3.28
    p1 = _by_name(sim.magnitudes, 'p1')
    assert p1.mag_min == pytest.approx(_h1(_F_HI), rel=1e-6)   # 0.1
    assert p1.mag_max == pytest.approx(_h1(_F_LO), rel=1e-6)   # ~1000
    stim = _by_name(sim.magnitudes, 'in')
    assert (stim.mag_min, stim.mag_max, stim.mag_mean) == (1.0, 1.0, 1.0)


def test_ac_magnitude_mean_is_mean_of_abs_not_abs_of_mean(pdk, tmp_dir):
    """mean(|H|) over the 701-point decade grid is 471.4 for v(out);
    the pre-fix ``mean=427.379`` was mean(Re(H)), which a consumer could
    not tell apart from a voltage. Pinned so the reduction cannot silently
    revert to a real-part or |mean| form."""
    report, _ = _report_for(pdk, tmp_dir, 'e_ng_ac.raw')
    out = _by_name(report.simulation.magnitudes, 'out')
    f = np.logspace(1, 8, 701)
    assert out.mag_mean == pytest.approx(_h3(f).mean(), rel=1e-4)
    assert out.mag_mean != pytest.approx(427.379, abs=1)


# ---------------------------------------------------------------------------
# DC sweep: a summary at all, keyed off vector 0
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('fixture, axis', [
    ('e_ng_dc.raw', 'v(v-sweep)'),
    ('e_xy_dc.raw', 'sweep'),
])
def test_dc_sweep_gets_a_summary(pdk, tmp_dir, fixture, axis):
    """Pre-fix (both simulators): ``report.simulation is None`` — the axis
    whitelist {'time', 'frequency'} matched neither 'v(v-sweep)' (ngspice)
    nor 'sweep' (Xyce), so a 401-point .dc sweep was reported as 'not run'.

    Post-fix: analysis 'dc', sweep_name = the axis vector's own name,
    -2 V..+2 V in 0.01 V steps, and tanh(2V) spanning -0.9993..+0.9993.
    """
    report, caught = _report_for(pdk, tmp_dir, fixture)
    sim = report.simulation
    assert sim is not None, 'DC sweep must produce a summary'
    assert not _complex_warnings(caught)
    assert sim.analysis == 'dc'
    assert sim.sweep_name == axis
    assert sim.t_start is None and sim.f_start is None
    assert sim.sweep_start == pytest.approx(-2.0)
    assert sim.sweep_stop == pytest.approx(2.0)
    assert sim.n_points == 401
    assert sim.sweep_step == pytest.approx(0.01)
    assert sim.magnitudes == []
    names = [s.name.lower() for s in sim.signals]
    assert names == ['in', 'out']            # axis excluded from the list
    out = _by_name(sim.signals, 'out')
    assert out.min == pytest.approx(-np.tanh(4.0), abs=2e-3)
    assert out.max == pytest.approx(np.tanh(4.0), abs=2e-3)
    assert abs(out.mean) < 1e-3              # odd transfer, symmetric sweep


# ---------------------------------------------------------------------------
# Operating point: every vector listed, none taken as the axis (TASK-102)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('fixture, names', [
    ('e_ng_op.raw', ['i(v1)', 'in', 'mid']),
    ('e_xy_op.raw', ['i(v1)', 'in', 'mid', 'sweep']),
])
def test_op_lists_every_vector_and_has_no_axis(pdk, tmp_dir, fixture, names):
    """Pre-fix (measured on main): ngspice's .op classified 'dc' with
    ``sweep_name='v(in)'``, ``sweep_start=sweep_stop=1.5`` and
    ``signals=['i(v1)', 'mid']`` — the 1.5 V input node was swallowed as the
    swept axis and missing from the table; Xyce's classified 'dc' with
    ``sweep_name='sweep'`` and bounds 0 → 0 (the placeholder's value).

    Post-fix: analysis 'op', no axis (``sweep_name=''``, every bound
    ``None``, every step ``None``), one point, and EVERY vector of the plot
    listed with min == max == mean == its value — v(in) = 1.5, v(mid) =
    1.0 (= 1.5 · 2k/3k), i(v1) = −0.5 mA (= −1.5/3k). Xyce's placeholder
    ``sweep`` (0.0) is a vector of the plot and is listed as one."""
    report, caught = _report_for(pdk, tmp_dir, fixture)
    sim = report.simulation
    assert sim is not None
    assert not _complex_warnings(caught)
    assert sim.analysis == 'op'
    assert sim.sweep_name == ''
    assert (sim.sweep_start, sim.sweep_stop) == (None, None)
    assert (sim.t_start, sim.t_stop, sim.f_start, sim.f_stop) == (None,) * 4
    assert sim.t_step is None and sim.sweep_step is None
    assert sim.n_points == 1
    assert sim.magnitudes == []
    assert [s.name.lower() for s in sim.signals] == names
    for s in sim.signals:
        assert s.min == s.max == s.mean
    assert _by_name(sim.signals, 'in').mean == pytest.approx(1.5)
    assert _by_name(sim.signals, 'mid').mean == pytest.approx(1.0)
    assert _by_name(sim.signals, 'i(v1)').mean == pytest.approx(-0.5e-3)


def test_op_renderers_label_an_operating_point(pdk, tmp_dir, capsys):
    """Pre-fix every writer treated the .op as a DC sweep: 'Simulation: dc,
    v(in) 1.5 → 1.5, 1 pts, 2 signals', a 'Sweep range' line, a 'Sweep
    statistics (dc, ...)' table without the input node, and a Min/Max/Mean
    table repeating one number three times. Now each writer says operating
    point, prints one value per vector, and never prints a range."""
    report, _ = _report_for(pdk, tmp_dir, 'e_ng_op.raw', nets=('in', 'mid'))
    report.print_summary()
    out = capsys.readouterr().out
    assert 'Simulation: op, one point, no swept axis, 1 pts, 3 signals' in out
    assert 'Operating point (.op, module nets):' in out
    assert 'in     1.500 V' in out and 'mid    1.000 V' in out
    assert 'Sweep' not in out and '→' not in out

    md = Path(tmp_dir) / 'r.md'
    html = Path(tmp_dir) / 'r.html'
    report.save(str(md))
    report.save_html(str(html))
    for text in (md.read_text(), html.read_text()):
        sim_section = text[text.index('Behavioral Simulation'):]
        assert 'Operating point' in sim_section
        assert 'Value (V)' in sim_section
        assert 'Min (V)' not in sim_section
        assert 'Sweep range' not in sim_section and ' ns' not in sim_section
        assert '1.5000' in sim_section and '-0.0005' in sim_section
    assert '| Signal | Value (V) |\n|---|---|\n' in md.read_text()
    assert '<th>Signal</th><th>Value (V)</th>' in html.read_text()


def test_op_summary_json_round_trip(pdk, tmp_dir):
    report, _ = _report_for(pdk, tmp_dir, 'e_xy_op.raw')
    path = Path(tmp_dir) / 'r.json'
    report.save_json(str(path))
    on_disk = json.loads(path.read_text())['simulation']
    assert on_disk['analysis'] == 'op'
    assert on_disk['sweep_name'] == ''
    assert on_disk['sweep_start'] is None and on_disk['t_start'] is None
    loaded = CircuitReport.load_json(str(path))
    assert loaded.simulation == report.simulation


# ---------------------------------------------------------------------------
# Noise: real spectral densities under frequency fields
# ---------------------------------------------------------------------------

def test_noise_reports_densities_as_magnitudes_under_frequency_fields(
        pdk, tmp_dir):
    """Pre-fix: the noise plot is real, so no ComplexWarning — but
    ``t_start=1 t_stop=1e6`` labelled the 1 Hz..1 MHz sweep as seconds and
    ``onoise_spectrum: min=1.43e-07 max=4.53e-06`` sat in a table headed
    'Min (V)' although the unit is V/sqrt(Hz).

    Post-fix: analysis 'noise', f_start=1, f_stop=1e6, and the densities
    are listed as magnitudes (the same numbers — |x| = x for a density —
    under field names that no longer claim volts).
    """
    report, caught = _report_for(pdk, tmp_dir, 'e_ng_noise.raw')
    sim = report.simulation
    assert sim is not None
    assert not _complex_warnings(caught)
    assert sim.analysis == 'noise'
    assert sim.sweep_name == 'frequency'
    assert sim.t_start is None
    assert sim.f_start == pytest.approx(1.0)
    assert sim.f_stop == pytest.approx(1e6)
    assert sim.n_points == 121
    assert sim.signals == []
    onoise = _by_name(sim.magnitudes, 'onoise_spectrum')
    assert onoise.mag_min == pytest.approx(1.43355e-07, rel=1e-4)
    assert onoise.mag_max == pytest.approx(4.53244e-06, rel=1e-4)
    # The ngspice per-device breakdown vectors ride along, as they do in
    # the rawfile; nine vectors besides the axis.
    assert sim.n_signals == 9


# ---------------------------------------------------------------------------
# Consumers: every renderer, every kind
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('fixture, label, needle, absent', [
    ('e_ng_tran.raw', 'Time range', '50000.0 ns', 'Hz'),
    ('e_ng_ac.raw', 'Frequency range', '1e+08 Hz', ' ns'),
    ('e_ng_dc.raw', 'Sweep range', 'v(v-sweep) -2 → 2', ' ns'),
    ('e_ng_noise.raw', 'Frequency range', '1e+06 Hz', ' ns'),
])
def test_renderers_name_the_axis_present(pdk, tmp_dir, fixture, label,
                                         needle, absent):
    """Pre-fix the Markdown and HTML writers printed every kind as
    ``**Time range:** X ns → Y ns`` — '10.0 ns → 100000000000.0 ns' for the
    AC fixture's 10 Hz..100 MHz sweep. Each writer now labels the axis it
    holds, and never prints a nanosecond figure for a non-time axis."""
    report, _ = _report_for(pdk, tmp_dir, fixture)
    md = Path(tmp_dir) / 'r.md'
    html = Path(tmp_dir) / 'r.html'
    report.save(str(md))
    report.save_html(str(html))
    for text in (md.read_text(), html.read_text()):
        sim_section = text[text.index('Behavioral Simulation'):]
        sim_section = sim_section[:sim_section.find('Figures')] \
            if 'Figures' in sim_section else sim_section
        assert label in sim_section
        assert needle in sim_section
        assert absent not in sim_section.split('Points')[0]
        assert 'Analysis:' in sim_section


def test_markdown_and_html_tables_say_magnitude_for_ac(pdk, tmp_dir):
    """Pre-fix the AC table was headed 'Min (V) | Max (V) | Mean (V)' over
    real-part numbers. Now the header says magnitude and the value column
    carries |H|."""
    report, _ = _report_for(pdk, tmp_dir, 'e_ng_ac.raw')
    md = Path(tmp_dir) / 'r.md'
    html = Path(tmp_dir) / 'r.html'
    report.save(str(md))
    report.save_html(str(html))
    for text in (md.read_text(), html.read_text()):
        assert 'Magnitude min' in text
        assert 'Min (V)' not in text
        assert '999.9995' in text            # |out| at 10 Hz, 4 dp
        assert '-3.2831' not in text         # the pre-fix real part


def test_markdown_table_keeps_volt_headers_for_transient(pdk, tmp_dir):
    """The transient table is unchanged: 'Min (V)' headers, ns axis."""
    report, _ = _report_for(pdk, tmp_dir, 'e_ng_tran.raw')
    md = Path(tmp_dir) / 'r.md'
    report.save(str(md))
    text = md.read_text()
    assert '| Signal | Min (V) | Max (V) | Mean (V) |' in text
    assert '**Time range:** 0.0 ns → 50000.0 ns' in text
    assert 'step: 19025.9 ps' in text


def test_print_summary_states_kind_and_axis(pdk, tmp_dir, capsys):
    """Pre-fix ``print_summary`` printed 'Simulation: 100000000000.0 ns,
    701 pts, 3 signals' for the AC fixture. It now names the kind and the
    axis, and the op-point table is replaced by a magnitude table headed
    as such (a transfer ratio has no operating point)."""
    report, _ = _report_for(pdk, tmp_dir, 'e_ng_ac.raw', nets=('in', 'out', 'p1'))
    report.print_summary()
    out = capsys.readouterr().out
    assert 'Simulation: ac, 10 Hz → 1e+08 Hz, 701 pts, 3 signals' in out
    assert '100000000000.0 ns' not in out
    assert 'Operating point' not in out
    assert 'Signal magnitude (ac, module nets)' in out


def test_print_summary_dc_sweep_no_longer_not_run(pdk, tmp_dir, capsys):
    """Pre-fix: 'Simulation: not run' for a 401-point DC sweep."""
    report, _ = _report_for(pdk, tmp_dir, 'e_ng_dc.raw', nets=('in', 'out'))
    report.print_summary()
    out = capsys.readouterr().out
    assert 'Simulation: not run' not in out
    assert 'Simulation: dc, v(v-sweep) -2 → 2, 401 pts, 2 signals' in out
    # A sweep has no single operating point; the table says what it holds.
    assert 'Operating point' not in out
    assert 'Sweep statistics (dc, module nets):' in out


# ---------------------------------------------------------------------------
# Terminal signal table: Xyce case-folding (TASK-109)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('fixture, nets, section, lines', [
    ('e_xy_tran.raw', ('clk', 'out'), 'Operating point (module nets):',
     ['CLK    0.918 V   (-0.000 – 1.800)', 'OUT    0.961 V   (0.000 – 1.372)']),
    ('e_xy_ac.raw', ('in', 'out', 'p1'), 'Signal magnitude (ac, module nets):',
     ['IN     1.000     (1.000 – 1.000)',
      'OUT  471.407     (0.001 – 999.999)',
      'P1   471.610     (0.100 – 1000.000)']),
    ('e_xy_dc.raw', ('in', 'out'), 'Sweep statistics (dc, module nets):',
     ['IN     0.000 V   (-2.000 – 2.000)', 'OUT    0.000 V   (-0.999 – 0.999)']),
    ('e_xy_op.raw', ('in', 'mid'), 'Operating point (.op, module nets):',
     ['IN     1.500 V', 'MID    1.000 V']),
])
def test_xyce_terminal_table_lists_module_nets(pdk, tmp_dir, capsys,
                                               fixture, nets, section, lines):
    """Pre-fix (measured on main for all four kinds): ``_render_signals_
    terminal`` filtered ``r[0] in module_nets`` with an exact, case-
    sensitive match; Xyce's rawfile vectors are upper-cased ('V(IN)' ->
    stored as 'IN'), which never equals the lower-case net names declared
    on the module ('in'), so the section header (``'Operating point
    (module nets):'`` / ``'Signal magnitude (ac, module nets):'`` /
    ``'Sweep statistics (dc, module nets):'`` / ``'Operating point (.op,
    module nets):'``) never printed at all — the terminal table was
    entirely absent, for every analysis kind, on every Xyce report.

    Post-fix the section appears with the circuit's nets and their values,
    stored-name case preserved verbatim (Xyce's upper-case, unlike
    ngspice's lower-case) since only the comparison is folded, not the
    data.
    """
    report, _ = _report_for(pdk, tmp_dir, fixture, nets=nets)
    report.print_summary()
    out = capsys.readouterr().out
    assert section in out
    for line in lines:
        assert line in out


def test_xyce_terminal_table_keeps_stored_names_verbatim(pdk, tmp_dir, capsys):
    """The membership test is case-folded; the printed name is not. A
    Xyce report prints 'CLK'/'OUT' (upper-case, as Xyce wrote them), never
    folded to 'clk'/'out' to match the ngspice spelling — TASK-109 fixes
    the comparison, not the data."""
    report, _ = _report_for(pdk, tmp_dir, 'e_xy_tran.raw', nets=('clk', 'out'))
    report.print_summary()
    out = capsys.readouterr().out
    assert 'CLK    0.918 V' in out and 'OUT    0.961 V' in out
    # Never folded to the ngspice spelling of the same two nets.
    assert 'clk    0.918 V' not in out and 'out    0.961 V' not in out


def test_ngspice_terminal_table_unchanged_by_the_xyce_fix(pdk, tmp_dir, capsys):
    """ngspice's vector names are already lower-case and already matched
    ``module.net_dict`` exactly, so the TASK-109 case-fold must be a no-op
    for it: the terminal table's signal lines are byte-identical to what
    the pre-fix, case-sensitive comparison already produced."""
    report, _ = _report_for(pdk, tmp_dir, 'e_ng_tran.raw', nets=('clk', 'out'))
    report.print_summary()
    out = capsys.readouterr().out
    assert 'Operating point (module nets):' in out
    assert 'clk    0.920 V   (0.000 – 1.800)' in out
    assert 'out    0.960 V   (0.000 – 1.372)' in out


# ---------------------------------------------------------------------------
# JSON round-trip and old-file compatibility
# ---------------------------------------------------------------------------

def test_ac_summary_json_round_trip(pdk, tmp_dir):
    """save_json -> load_json preserves the kind, the frequency fields and
    the magnitude list; t_* serialise as null, never as a frequency."""
    report, _ = _report_for(pdk, tmp_dir, 'e_ng_ac.raw')
    path = Path(tmp_dir) / 'r.json'
    report.save_json(str(path))
    on_disk = json.loads(path.read_text())['simulation']
    assert on_disk['t_start'] is None and on_disk['t_stop'] is None
    assert on_disk['f_start'] == pytest.approx(10.0)
    assert on_disk['analysis'] == 'ac'
    assert on_disk['signals'] == []
    assert {'mag_min', 'mag_max', 'mag_mean'} <= set(on_disk['magnitudes'][0])
    loaded = CircuitReport.load_json(str(path))
    assert loaded.simulation == report.simulation


def test_dc_summary_json_round_trip(pdk, tmp_dir):
    report, _ = _report_for(pdk, tmp_dir, 'e_ng_dc.raw')
    path = Path(tmp_dir) / 'r.json'
    report.save_json(str(path))
    loaded = CircuitReport.load_json(str(path))
    assert loaded.simulation == report.simulation
    assert loaded.simulation.sweep_name == 'v(v-sweep)'


def test_pre_task_82_dict_reads_as_transient():
    """A summary written before TASK-82 has only the four transient keys.
    It must load, classify as transient, and mirror t_* into the generic
    bounds so new consumers see a complete summary."""
    old = {
        't_start': 0.0, 't_stop': 1e-6, 'n_points': 11,
        'signals': [{'name': 'vout', 'min': 0.0, 'max': 1.8, 'mean': 0.9}],
    }
    sim = _simulation_from_dict(old)
    assert sim.analysis == 'transient'
    assert sim.sweep_name == 'time'
    assert (sim.sweep_start, sim.sweep_stop) == (0.0, 1e-6)
    assert sim.f_start is None
    assert sim.magnitudes == []
    assert sim.t_step == pytest.approx(1e-7)
    assert sim.signals == [SignalStats('vout', 0.0, 1.8, 0.9)]


def test_positional_constructor_still_builds_a_transient_summary():
    """The four original fields remain positional, in their original order."""
    sim = SimulationSummary(0.0, 2e-9, 3, [])
    assert sim.analysis == 'transient'
    assert sim.sweep_stop == 2e-9
    assert sim.t_step == pytest.approx(1e-9)


def test_magnitude_stats_never_carry_value_field_names():
    """MagnitudeStats has no ``min``/``max``/``mean`` — a consumer that
    reads those off an AC entry gets an AttributeError, not a number it
    could mistake for a voltage."""
    m = MagnitudeStats('out', 1.0, 2.0, 1.5)
    for attr in ('min', 'max', 'mean'):
        assert not hasattr(m, attr)


# ---------------------------------------------------------------------------
# Hand-built results: the dtype guard behind the header flag
# ---------------------------------------------------------------------------

def test_complex_vectors_without_header_flag_are_still_magnitudes(pdk, tmp_dir):
    """A result stub with complex data but no ``real`` flag (the shape the
    older report tests mount) must not be real-cast either: the classifier
    walks dtypes when the header flag is absent."""
    import types
    mod = Module('PDK_skywater130', tmp_dir, 'm.cir', pdk_details=pdk)
    f = np.logspace(0, 3, 4)
    h = np.array([10 + 0j, 6 + 8j, 0 + 10j, -10 + 0j])   # |h| = 10 throughout
    vectors = [types.SimpleNamespace(name='frequency', data=f),
               types.SimpleNamespace(name='v(out)', data=h)]
    mod.simResultsAll = types.SimpleNamespace(vectors=vectors, nvars=2)
    mod.sim_results_loaded = True
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        sim = _build_simulation(mod)
    assert _complex_warnings(caught) == []
    assert sim.analysis == 'ac'
    assert sim.signals == []
    (out,) = sim.magnitudes
    assert (out.mag_min, out.mag_max) == pytest.approx((10.0, 10.0))
    # Lexicographic np.min would have picked -10+0j -> real part -10.
    assert out.mag_min != -10.0
