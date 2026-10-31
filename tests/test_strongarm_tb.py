"""Smoke tests for silk/examples/strongarm_tb.py — the canonical
testbench-as-part-of-module example.

The end-to-end SPICE run is marked @pytest.mark.slow; the import-and-build
smoke tests run in the default suite.
"""
import pytest
from tests._env import needs_pdk


def test_module_imports():
    """The example module imports without raising."""
    from silk.examples import strongarm_tb
    assert hasattr(strongarm_tb, 'make_comparator_strongarm')
    assert hasattr(strongarm_tb, 'make_clock_delay')


def test_flash_adc_still_imports():
    """The flash ADC example imports and exposes its shared comparator,
    clock-delay builder and analog supply constant.
    """
    import silk.examples.flash_adc_example as fa
    assert hasattr(fa, 'make_comparator_strongarm')
    assert hasattr(fa, 'make_clock_delay')
    assert hasattr(fa, 'AVDD')


@needs_pdk


def test_rig_applies_to_testbench():
    """add_strongarm_rig adds sources and a clock-delay submodule to a TB."""
    from silk.examples.strongarm_tb import (
        _pdk, make_comparator_strongarm, add_strongarm_rig, OUT_DIR,
    )
    from silk.circuit.module_test import ModuleTestbench
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    pdk = _pdk()
    comp = make_comparator_strongarm(pdk, OUT_DIR)
    tb = ModuleTestbench(comp, name='rig_smoke')
    tb.add_module_under_test({
        'vp': 'vp', 'vm': 'vm', 'clkb': 'clkb_d',
        'vdd': 'vdd', 'vss': 'gnd', 'vout': 'vout',
    })
    add_strongarm_rig(tb)

    # Three voltage sources registered (vdd, clk, clkb)
    assert 'vdd_src' in tb.netlist.sig_sources_dict
    assert 'clk_src' in tb.netlist.sig_sources_dict
    assert 'clkb_src' in tb.netlist.sig_sources_dict
    # Clock-delay submodule registered
    assert 'clkdel1' in tb.netlist.elem_dict


@needs_pdk


def test_pvt_threshold_declares_its_corners():
    """The PVT bench declares its corner-set with set_corners."""
    from silk.examples.strongarm_tb import (
        _pdk, make_comparator_strongarm, make_pvt_threshold_tb, OUT_DIR,
    )
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    pdk = _pdk()
    comp = make_comparator_strongarm(pdk, OUT_DIR)
    pvt = make_pvt_threshold_tb(comp)
    assert pvt.corners == ('tt', 'ff', 'ss', 'sf', 'fs')


def test_metric_extractors_importable():
    """The threshold extractor (no silk equivalent) is defined."""
    from silk.examples.strongarm_tb import _extract_threshold
    assert callable(_extract_threshold)


@needs_pdk


def test_threshold_sweep_specs_have_stable_context_keys():
    from silk.examples.strongarm_tb import (
        _pdk, make_comparator_strongarm, make_threshold_tb, OUT_DIR,
    )
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    comp = make_comparator_strongarm(_pdk(), OUT_DIR)
    tb = make_threshold_tb(comp, vm_dc=0.85)

    assert [entry['spec'].key for entry in tb._specs] == [
        'threshold_vm0850mV.threshold_upper',
        'threshold_vm0850mV.threshold_lower',
    ]


@needs_pdk


def test_register_all_testbenches():
    """All four TB factories register the expected names on the comparator."""
    from silk.examples.strongarm_tb import (
        _pdk, make_comparator_strongarm,
        make_threshold_tb, make_decision_time_tb,
        make_offset_tb, make_pvt_threshold_tb,
        VM_VALUES, OVERDRIVE_VALUES, OUT_DIR,
    )
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    pdk = _pdk()
    comp = make_comparator_strongarm(pdk, OUT_DIR)
    for vm in VM_VALUES:
        make_threshold_tb(comp, vm_dc=vm)
    for od in OVERDRIVE_VALUES:
        make_decision_time_tb(comp, overdrive=od)
    make_offset_tb(comp)
    make_pvt_threshold_tb(comp)

    names = set(comp.list_tests())
    expected = {
        'threshold_vm0850mV', 'threshold_vm1000mV',
        'threshold_vm1150mV', 'threshold_vm1300mV',
        'decision_time_od0001mV', 'decision_time_od0005mV',
        'decision_time_od0020mV', 'decision_time_od0100mV',
        'offset', 'pvt_threshold',
    }
    assert names == expected, f'unexpected TB set: {names ^ expected}'
    assert len(comp._tests) == 10


def test_plotters_and_printer_importable():
    """The four plotters and the summary printer are defined and callable."""
    from silk.examples.strongarm_tb import (
        _plot_threshold_sweep, _plot_decision_time,
        _plot_offset_kickback, _plot_pvt_threshold,
        _print_summary_table,
    )
    assert callable(_plot_threshold_sweep)
    assert callable(_plot_decision_time)
    assert callable(_plot_offset_kickback)
    assert callable(_plot_pvt_threshold)
    assert callable(_print_summary_table)


def test_summary_table_handles_empty_results(capsys):
    """_print_summary_table prints a non-crashing header for empty results."""
    from silk.examples.strongarm_tb import _print_summary_table
    _print_summary_table({})
    captured = capsys.readouterr()
    assert 'StrongArm Comparator' in captured.out
    assert 'Total: 0/0' in captured.out


@pytest.mark.slow
def test_end_to_end_run():
    """End-to-end: build comp, register 10 TBs, run them all, verify PNGs.

    Marked @pytest.mark.slow because it invokes ngspice for every TB.
    Skipped by default; run with: pytest -m slow tests/test_strongarm_tb.py
    """
    from silk.examples.strongarm_tb import (
        _pdk, make_comparator_strongarm, _build_and_register_all,
        _print_summary_table,
        _plot_threshold_sweep, _plot_decision_time,
        _plot_offset_kickback, _plot_pvt_threshold,
        OUT_DIR, VM_VALUES, OVERDRIVE_VALUES,
    )
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    pdk  = _pdk()
    comp = make_comparator_strongarm(pdk, OUT_DIR)
    _build_and_register_all(comp)

    assert len(comp.list_tests()) == 10
    results = comp.run_all_tests()

    # Every registered test ran and produced a result
    assert len(results) == 10
    # PVT result is CornerResults (5 corners); others are SpecReports
    pvt = results[f'{comp.module_name}.pvt_threshold']
    from silk.circuit.spec_report import CornerResults, SpecReport
    assert isinstance(pvt, CornerResults)
    assert len(list(pvt.items())) == 5

    threshold_report = results[f'{comp.module_name}.threshold_vm1000mV']
    assert isinstance(threshold_report, SpecReport)

    # Plots — call each, verify the PNG was written
    threshold_reports = {
        name: results[f'{comp.module_name}.{name}']
        for name in (f'threshold_vm{int(v*1000):04d}mV' for v in VM_VALUES)
    }
    decision_reports = {
        name: results[f'{comp.module_name}.{name}']
        for name in (f'decision_time_od{int(o*1000):04d}mV'
                     for o in OVERDRIVE_VALUES)
    }
    # The 100 mV decision uses the intended evaluate edge after startup.
    decision_100mv = decision_reports['decision_time_od0100mV']
    assert 0 < decision_100mv.metrics['t_decide'] < 1e-9
    assert decision_100mv.passed
    _plot_threshold_sweep(threshold_reports,
                          os.path.join(OUT_DIR, 'strongarm_threshold.png'))
    _plot_decision_time(decision_reports,
                        os.path.join(OUT_DIR, 'strongarm_decision_time.png'))
    _plot_offset_kickback(
        results[f'{comp.module_name}.offset'],
        os.path.join(OUT_DIR, 'strongarm_offset.png'))
    _plot_pvt_threshold(
        results[f'{comp.module_name}.pvt_threshold'],
        os.path.join(OUT_DIR, 'strongarm_pvt.png'))

    for name in ('strongarm_threshold.png', 'strongarm_decision_time.png',
                 'strongarm_offset.png', 'strongarm_pvt.png'):
        assert os.path.exists(os.path.join(OUT_DIR, name)), \
            f'expected {name} in {OUT_DIR}'

    # Print summary — also a smoke test that the printer handles real results
    _print_summary_table(results)


# ══════════════════════════════════════════════════════════════════════════════
#  _extract_threshold — regression tests for the reset-glitch bug
#
#  These run on synthetic traces, so they need no simulator. The trace shape is
#  the one the real testbench produces: uic=True means vout starts HIGH, the
#  comparator's reset pulls it low, and only once vp crosses vm does vout pulse
#  high once per clock.
# ══════════════════════════════════════════════════════════════════════════════
def _synthetic_threshold_trace(vm, *, start_high=True, glitch_at=None):
    """Build (t, vp, vout) mimicking a threshold-sweep run.

    Parameters
    ----------
    vm : float
        Common-mode level; vp ramps vm−VP_HALF_RANGE → vm+VP_HALF_RANGE.
    start_high : bool
        If True, vout starts at the rail (the ``uic=True`` behaviour that caused
        the bug). If False, vout starts low.
    glitch_at : float or None
        Optional time (s) at which to inject a single-sample high spike, to
        check the persistence guard rejects it.
    """
    import numpy as np
    from silk.examples.strongarm_tb import VDD, VP_HALF_RANGE, CLK_PER, T_STOP

    dt = CLK_PER / 50.0
    t = np.arange(0.0, T_STOP, dt)
    vp = np.linspace(vm - VP_HALF_RANGE, vm + VP_HALF_RANGE, t.size)
    vout = np.zeros_like(t)

    # One evaluate pulse per clock cycle, but only once vp has passed vm.
    phase = np.mod(t, CLK_PER)
    evaluating = phase > CLK_PER / 2
    vout[(vp > vm) & evaluating] = VDD

    if start_high:                      # the uic=True artifact: rail at t=0
        vout[:2] = 0.98 * VDD
    if glitch_at is not None:
        vout[int(glitch_at / dt)] = VDD

    return t, vp, vout


class _FakeCkt:
    """Minimal stand-in for the simulated netlist: just ``load_signal``."""

    def __init__(self, t, **traces):
        self._t = t
        self._traces = traces

    def load_signal(self, name):
        from silk.signals.base import Signal
        return Signal(f'v({name})', 'voltage', 'time', 'voltage',
                      self._t, self._traces[name])


def test_extract_threshold_ignores_leading_reset_transition():
    """A trace that starts HIGH must not yield vp[0].

    This is the exact bug: with ``uic=True`` the first vout transition is the
    reset at t≈0, so taking the first flip of *either* polarity returned the vp
    ramp's starting value (vm − 50 mV) on every run.
    """
    from silk.examples.strongarm_tb import _extract_threshold, VP_HALF_RANGE

    vm = 1.0
    t, vp, vout = _synthetic_threshold_trace(vm, start_high=True)
    assert vout[0] > 0.9, 'trace precondition: vout must start high'

    ckt = _FakeCkt(t, vp=vp, vout=vout)
    got = _extract_threshold(ckt.load_signal('vout'), ckt.load_signal('vp'), vm)

    assert abs(got - vp[0]) > VP_HALF_RANGE / 2, (
        f'extractor returned {got}, which is the ramp start vp[0]={vp[0]} — '
        'it read the reset glitch instead of a decision')
    assert abs(got - vm) < 0.010, f'threshold {got} not within 10 mV of vm={vm}'


def test_extract_threshold_agrees_whether_or_not_vout_starts_high():
    """The leading reset must not change the answer at all."""
    from silk.examples.strongarm_tb import _extract_threshold

    vm = 1.15
    t, vp, hi = _synthetic_threshold_trace(vm, start_high=True)
    _, _, lo = _synthetic_threshold_trace(vm, start_high=False)

    ckt_hi = _FakeCkt(t, vp=vp, vout=hi)
    ckt_lo = _FakeCkt(t, vp=vp, vout=lo)
    a = _extract_threshold(ckt_hi.load_signal('vout'), ckt_hi.load_signal('vp'), vm)
    b = _extract_threshold(ckt_lo.load_signal('vout'), ckt_lo.load_signal('vp'), vm)
    assert a == b, f'start-high gave {a}, start-low gave {b}'


def test_extract_threshold_rejects_single_sample_glitch():
    """A one-sample spike after settling is not a decision."""
    from silk.examples.strongarm_tb import _extract_threshold, CLK_PER

    vm = 1.0
    # Glitch early in the sweep, well before vp reaches vm.
    t, vp, vout = _synthetic_threshold_trace(
        vm, start_high=True, glitch_at=10 * CLK_PER)

    ckt = _FakeCkt(t, vp=vp, vout=vout)
    got = _extract_threshold(ckt.load_signal('vout'), ckt.load_signal('vp'), vm)
    assert abs(got - vm) < 0.010, (
        f'threshold {got} is off by more than 10 mV — the persistence guard '
        'let a single-sample glitch through')


def test_extract_threshold_caps_when_no_decision():
    """No high decision reports the upper sweep limit, never a zero offset."""
    import numpy as np
    from silk.examples.strongarm_tb import _extract_threshold, T_STOP, CLK_PER

    vm = 1.0
    t = np.arange(0.0, T_STOP, CLK_PER / 50.0)
    vp = np.linspace(vm - 0.1, vm + 0.1, t.size)
    vout = np.zeros_like(t)
    ckt = _FakeCkt(t, vp=vp, vout=vout)
    assert _extract_threshold(ckt.load_signal('vout'), ckt.load_signal('vp'), vm) == pytest.approx(1.1)


@pytest.mark.parametrize('pattern, expected', [
    ('low', 1.1), ('high', 0.9), ('single', 1.0),
    ('multiple', float('nan')), ('lower_edge', 0.9), ('upper_edge', 1.1),
])
def test_offset_threshold_decisions(pattern, expected):
    """Reset pulses must not hide constant decisions or extra reversals."""
    import numpy as np
    from silk.examples import strongarm_tb as s
    from silk.signals.base import Signal

    t = np.linspace(0, 100 * s.CLK_PER, 10001)
    vp = 0.9 + 0.2 * t / t[-1]
    decision = {
        'low': np.zeros(t.size, dtype=bool),
        'high': np.ones(t.size, dtype=bool),
        'single': vp >= 1.0,
        'multiple': ((vp >= 0.96) & (vp < 1.0)) | (vp >= 1.04),
        'lower_edge': vp >= 0.901,
        'upper_edge': vp >= 1.099,
    }[pattern]
    evaluating = np.mod(t, s.CLK_PER) > 0.5 * s.CLK_PER
    vout = np.where(decision & evaluating, s.VDD, 0.)
    out = Signal('vout', 'voltage', 'time', 'voltage', t, vout)
    inp = Signal('vp', 'voltage', 'time', 'voltage', t, vp)
    got = s._extract_threshold(out, inp, 1.0)
    if np.isnan(expected):
        assert np.isnan(got)
    else:
        assert got == pytest.approx(expected, abs=0.002 if pattern == 'single' else 1e-12)
    if pattern == 'low':
        from silk.circuit.spec_report import SpecReport
        report = SpecReport('offset', 'comparator', {'offset_v': got - 1.0})
        report.check('offset < +5 mV', 'offset_v', '<', 0.005)
        assert not report.passed


@pytest.mark.parametrize('kind', ['strongarm', 'flash_clocked', 'flash_dc'])
@pytest.mark.parametrize('high, side', [(False, 1), (True, -1)])
def test_capped_offset_report(tmp_path, capsys, kind, high, side):
    """Synthetic rawfiles exercise registered metrics, flags, specs and text."""
    import numpy as np
    from silk.examples import strongarm_tb as s, flash_adc_example as f
    from silk.signals.rawfile import RawFile, RawPlot, RawVector, write_rawfile

    factory = f.make_comparator if kind == 'flash_dc' else s.make_comparator_strongarm
    comp = factory(s._pdk(), str(tmp_path))
    tb = (s.make_offset_tb(comp) if kind == 'strongarm'
          else f.make_comparator_offset_tb(comp, strongarm=kind == 'flash_clocked'))
    x = np.linspace(0, 100 * s.CLK_PER, 10001)
    vp = np.linspace(.9, 1.1, x.size)
    out = np.full(x.size, s.VDD if high else 0.)
    clocked = kind != 'flash_dc'
    if clocked:
        out[np.mod(x, s.CLK_PER) <= .5 * s.CLK_PER] = 0
    raw = RawFile([RawPlot(
        {'plotname': 'Transient Analysis' if clocked else 'DC transfer characteristic',
         'command': 'ngspice'},
        [RawVector('time' if clocked else 'v(v-sweep)', 'time' if clocked else 'voltage', data=x if clocked else vp),
         RawVector('v(vp)', 'voltage', data=vp),
         RawVector('v(vout)', 'voltage', data=out)],
        npoints=x.size, real=True, padded=True)])
    path = tmp_path / 'offset.raw'
    write_rawfile(path, raw)
    report = tb.evaluate(rawfile=str(path))
    assert report.metrics['offset_v'] == pytest.approx(side * .1)
    assert report.metrics['offset_cap_side'] == side
    assert not report.passed
    if kind == 'strongarm':
        s._print_summary_table({'comparator.offset': report})
        assert f'offset capped at {side * 100:+.0f} mV' in capsys.readouterr().out


def test_offset_deck_saves_only_observed_voltages(tmp_path):
    """Avoid retaining unused device internals throughout the long ramp."""
    from pathlib import Path
    from silk.examples.strongarm_tb import (
        _pdk, make_comparator_strongarm, make_offset_tb,
    )

    comp = make_comparator_strongarm(_pdk(), str(tmp_path))
    tb = make_offset_tb(comp)
    tb_netlist_deck_path = tb.netlist.generate_netlist()
    deck = Path(tb_netlist_deck_path).read_text()
    saves = [line.split() for line in deck.splitlines()
             if line.strip().lower().startswith(('save ', '.save '))]
    assert len(saves) == 1
    assert saves[0][0] == 'save'
    assert sorted(saves[0][1:]) == ['"v(vm)"', '"v(vout)"', '"v(vp)"']
    assert 'save all' not in deck.lower()


@pytest.mark.parametrize('startup_glitch', [True, False])
@pytest.mark.parametrize('overdrive', [1e-3, 5e-3, 20e-3, 100e-3])
def test_decision_time_uses_intended_evaluation_edge(
        tmp_path, monkeypatch, startup_glitch, overdrive):
    """Startup clock pulses must not lengthen the registered decision delay."""
    import numpy as np
    from silk.examples import strongarm_tb as s

    comp = s.make_comparator_strongarm(s._pdk(), str(tmp_path))
    tb = s.make_decision_time_tb(comp, overdrive=overdrive)
    # Linear ramps cross half supply at 1 ns (startup) and 26 ns
    # (evaluation); the output reaches 90% supply at 27.9 ns.
    t = np.array([0, 2, 4, 25, 27, 28, 30], dtype=float) * 1e-9
    clock = np.array([0, 1, 0, 0, 1, 1, 1], dtype=float) * s.VDD
    if not startup_glitch:
        clock[1] = 0
    ckt = _FakeCkt(t, clkb_d=clock,
                   vout=np.array([0, 0, 0, 0, 0, 1, 1]) * s.VDD,
                   vp=np.ones_like(t))
    monkeypatch.setattr(tb.netlist, 'load_signal',
                        lambda name, *args, **kwargs: ckt.load_signal(name))
    monkeypatch.setattr(tb.netlist, 'sim_results_loaded', True)
    metric = next(m for m in tb._metrics if m['name'] == 't_decide')
    assert metric['extractor'](tb.netlist) == pytest.approx(1.9e-9, abs=1e-15)


def test_decision_time_plot_marks_intended_evaluation_edge(tmp_path, monkeypatch):
    """The actual plot must anchor both markers to evaluation after startup."""
    import numpy as np
    from types import SimpleNamespace
    from silk.examples import strongarm_tb as s

    t = np.array([0, 2, 4, 25, 27, 28, 30], dtype=float) * 1e-9
    ckt = _FakeCkt(t,
                   clkb_d=np.array([0, 1, 0, 0, 1, 1, 1]) * s.VDD,
                   vout=np.array([0, 0, 0, 0, 0, 1, 1]) * s.VDD,
                   vp=np.ones_like(t))
    report = SimpleNamespace(sim_results=ckt, metrics={'t_decide': 1.9e-9})
    # Retain the real figure for inspection after the plotter saves it.
    monkeypatch.setattr(s.plt, 'close', lambda fig: None)
    try:
        s._plot_decision_time({'decision_time_od0100mV': report},
                              str(tmp_path / 'decision.png'))
        ax = s.plt.gcf().axes[0]
        markers = [line.get_xdata()[0] for line in ax.lines[2:]]
        assert markers == pytest.approx([26.0, 27.9])
        assert ax.lines[-1].get_label() == 't_decide = 1900 ps'
    finally:
        monkeypatch.undo()
        s.plt.close('all')
