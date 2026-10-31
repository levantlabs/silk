"""SAR control-deck checks and an opt-in SPICE trip-gain regression."""
from pathlib import Path
import re

import numpy as np
import pytest

from silk.examples import sar_adc_spice as sar
from silk.utils._utils import _parse_si
from tests._env import needs_pdk, needs_ngspice


@pytest.mark.parametrize('sample_input', [False, True])
def test_cap_dac_equations_replay_irregular_grid(sample_input):
    """The array kernel preserves reset, sampling and charge across bit trials."""
    t = np.array([0., .3, .9, 1.1, 1.9, 3.2, 3.3]) * 1e-9
    high = sar.VDD
    p, n = sar._cap_dac_voltages(
        2, 1e-15, sample_input, t,
        smp=np.array([high, high, 0., 0., 0., high, 0.]),
        vcm=np.full(t.size, .9), vrefp=np.full(t.size, .9),
        vrefn=np.zeros(t.size), vinp=np.full(t.size, 1.05),
        vinn=np.full(t.size, .75),
        bits=np.array([[0., 0., 0., high, high, 0., high],
                       [0., 0., 0., 0., high, 0., 0.]]))
    # Binary caps 1+2 and a 1-unit dummy; sample/reset top plates at 0.9 V.
    expected_p = ([.9, .9, .1125, .3375, .7875, .9, .3375] if sample_input
                  else [.9, .9, .9, 1.125, 1.575, .9, 1.125])
    expected_n = ([.9, .9, 1.0125, .7875, .3375, .9, .7875] if sample_input
                  else [.9, .9, .9, .675, .225, .9, .675])
    np.testing.assert_allclose(p, expected_p, rtol=0, atol=1e-15)
    np.testing.assert_allclose(n, expected_n, rtol=0, atol=1e-15)


@pytest.mark.parametrize('bench', ['dac', 'loop', 'continuous'])
def test_controls_hold_between_events(tmp_path, monkeypatch, bench):
    """Missing old-level points must not turn digital controls into phase ramps."""
    decks = []

    class DeckReady(Exception):
        pass

    def emit_only(module, **kwargs):
        module.generate_netlist(**kwargs)
        decks.append(Path(module._run_files.netlist_path).read_text())
        raise DeckReady

    # Only the external solve is replaced: build and emit the real circuit.
    monkeypatch.setattr(sar.Module, 'simulate', emit_only)
    with pytest.raises(DeckReady):
        if bench == 'dac':
            sar._convert_one(3, 5, sar._pdk(), str(tmp_path))
        elif bench == 'loop':
            sar._sar_decision(3, 0.6, 0.3, 5, sar._pdk(), str(tmp_path))
        else:
            # Hand-derived binary search trials for vin=0.3 V: 4, 6, 5.
            sar.convert_spice_continuous(3, 0.3, sar._pdk(), str(tmp_path))

    controls = {}
    for line in decks[0].splitlines():
        match = re.match(r'\S+\s+(smp|clkb|b[0-2]b?)\s+\S+\s+.*?PWL\(([^)]*)\)',
                         line, re.IGNORECASE)
        if match:
            controls[match[1]] = np.array([_parse_si(x) for x in match[2].split()]).reshape(-1, 2)

    expected = {'smp': [(0, 1.8), (4e-9, 0)]}
    trials = [(4e-9, 5)] if bench != 'continuous' else [(4e-9, 4), (12e-9, 6), (20e-9, 5)]
    for k in range(3):
        expected[f'b{k}'] = [(0, 0)] + [(t, 1.8 if code & (1 << k) else 0) for t, code in trials]
        expected[f'b{k}b'] = [(0, 1.8 if bench == 'dac' else 0)] + [
            (t, 0 if code & (1 << k) else 1.8) for t, code in trials]
    if bench == 'loop':
        expected['clkb'] = [(0, 0), (14e-9, 1.8)]
    assert controls.keys() == expected.keys()
    for name, events in expected.items():
        times, values = controls[name].T
        assert np.all(np.diff(times) > 0), name
        for (start, old), (end, new) in zip(events, events[1:]):
            # Hold until T_RISE before the event, then finish the edge at it.
            assert end - start >= sar.T_RISE, name
            assert np.interp((start + end) / 2, times, values) == pytest.approx(old), name
            assert np.interp(end - sar.T_RISE, times, values) == pytest.approx(old, abs=1e-9), name
            assert np.interp(end - sar.T_RISE / 2, times, values) == pytest.approx((old + new) / 2), name
            if old != new:
                changing = np.flatnonzero(np.diff(values) != 0)
                edge = changing[np.argmin(abs(times[changing + 1] - end))]
                assert times[edge + 1] - times[edge] == pytest.approx(0.2e-9, abs=1e-15), name
            assert np.interp(end, times, values) == pytest.approx(new, abs=1e-9), name
            assert np.interp(end + sar.T_RISE, times, values) == pytest.approx(new, abs=1e-9), name
        assert np.interp(events[-1][0] + 1e-9, times, values) == pytest.approx(events[-1][1]), name
    assert np.interp(2e-9, *controls['smp'].T) == pytest.approx(1.8)
    if bench == 'continuous':
        # The already-correct clock uses a pulse, not the broken PWL helper.
        pulse = re.search(r'^\S+\s+clkb\s+\S+\s+.*?PULSE\(([^)]*)\)', decks[0], re.M | re.I)
        assert pulse
        fields = [_parse_si(x) for x in pulse[1].split()]
        assert fields[:7] == pytest.approx([0, 1.8, 8.8e-9, 0.1e-9, 0.1e-9, 2.8e-9, 8e-9], abs=1e-15)


def test_step_pwl_default_preserves_sine_edge_timing():
    points = list(sar._step_pwl([(0.0, 0.0), (4e-9, 1.8), (8e-9, 0.0)]))
    assert points == [(0.0, 0.0), (4e-9 - 1e-12, 0.0), (4e-9, 1.8),
                      (8e-9 - 1e-12, 1.8), (8e-9, 0.0)]


def test_twin_dac_gain_default_is_ideal_and_scales_levels():
    from silk.examples import sar_adc_example as twin
    w, wtot = twin.unit_cap_array(8, sigma=0.0)
    vin = np.linspace(-0.81, 0.81, 7)
    ideal = twin.sar_codes(vin, w, wtot, vref=0.9)
    assert list(ideal) == [12, 51, 89, 128, 166, 204, 243]
    assert list(twin.sar_codes(vin, w, wtot, vref=0.9, dac_gain=1.0)) == list(ideal)
    # A gain below one shrinks every DAC level, so the codes spread wider.
    wide = twin.sar_codes(vin, w, wtot, vref=0.9, dac_gain=0.97)
    assert wide[0] < ideal[0] and wide[-1] > ideal[-1]


def test_loop_twin_defaults_to_schematic_dummy_terms(monkeypatch):
    """The recorded 28 µm loop codes at the seven default inputs (TASK-577)."""
    loop = {-0.81: 12, -0.54: 51, -0.27: 89, 0.0: 127, 0.27: 165, 0.54: 203, 0.81: 242}
    monkeypatch.setattr(sar, '_pdk', lambda: None)
    monkeypatch.setattr(sar, 'example_output_dir', lambda: None)
    monkeypatch.setattr(sar, 'convert_spice',
                        lambda n, v, pdk, od, **kw: loop[round(v, 2)])
    vd, csa, cid, ct = sar.sar_transfer_spice(8)
    assert cid is None
    assert sar.LOOP_DAC_GAIN == 256 / 255
    assert sar.LOOP_TRIP_OFFSET == 0.9 / 255
    assert list(ct) == [12, 51, 89, 127, 165, 204, 242]
    assert np.abs(csa - ct).max() == 1
    _, _, _, ideal = sar.sar_transfer_spice(8, dac_gain=1.0, trip_offset=0.0)
    assert list(ideal) == [12, 51, 89, 128, 166, 204, 243]


def test_measure_trip_points_bisects_the_decision(monkeypatch):
    trip = {13: -0.7835, 128: 0.0105}
    calls = []

    def decision(n, vinp, vinn, code, pdk, out_dir, *, ideal_comp=False):
        calls.append(ideal_comp)
        return (vinp - vinn) < trip[code]          # DAC above the input -> clear

    monkeypatch.setattr(sar, '_pdk', lambda: None)
    monkeypatch.setattr(sar, 'example_output_dir', lambda: None)
    monkeypatch.setattr(sar, '_sar_decision', decision)
    codes, v_dac, v_trip = sar.measure_trip_points(8, (13, 128), tol=0.5e-3)
    assert list(codes) == [13, 128]
    assert v_dac == pytest.approx([-0.80859375, 0.0])
    assert v_trip == pytest.approx([-0.7835, 0.0105], abs=0.25e-3)
    assert not any(calls)
    with pytest.raises(RuntimeError, match='widen span'):
        sar.measure_trip_points(8, (13,), span=10e-3)


@pytest.mark.slow
@needs_pdk
@needs_ngspice
def test_strongarm_trip_gain_matches_dummy_cap_schematic():
    """Kickback must not restore the old code-dependent decision gain."""
    codes, vd, trips = sar.measure_trip_points(codes=(13, 128, 243))
    gain, _ = np.polyfit(vd, trips, 1)
    # Half the final 0.46875 mV brackets, projected onto the fitted slope.
    slope_resolution = 0.234375e-3 * np.abs(vd).sum() / np.square(vd).sum()
    assert gain == pytest.approx(256 / 255, abs=slope_resolution)


def test_reference_switch_width_override_reaches_both_arrays(tmp_path):
    """Sizing a high reference must not also resize low-reference/sample paths."""
    dut = sar.build_cap_dac(3, sar._pdk(), str(tmp_path), sample_input=True,
                            sw_refp_w=12.5)
    deck = dut.generate_netlist()
    switches = {}
    for line in deck.splitlines():
        match = re.match(r'XM(\w+)\s+.*\bw=(\S+)', line, re.I)
        if match:
            switches[match[1]] = _parse_si(match[2])
    for k in range(3):
        assert switches[f'swpp{k}'] == 12.5
        assert switches[f'swnp{k}'] == 12.5
        for name in ('swpn', 'swnn', 'smpp', 'smpn'):
            assert switches[f'{name}{k}'] == 2.0
    for name in ('rstp', 'rstn', 'rsteq'):
        assert switches[name] == 2.0


def test_static_linearity_covers_terminal_interior_code():
    from silk.examples import sar_adc_example as twin
    w, wtot = twin.unit_cap_array(4, sigma=0.0)
    dnl, inl, transitions = twin.static_dnl_inl(4, w, wtot)
    assert len(dnl) == 14
    assert np.isfinite(transitions[1:]).all()
    assert np.abs(dnl).max() < 0.03
    assert np.abs(inl).max() < 0.03


@pytest.mark.parametrize('n', [4, 6])
def test_loop_dummy_correction_scales_with_resolution(monkeypatch, n):
    from silk.examples import sar_adc_example as twin
    monkeypatch.setattr(sar, '_pdk', lambda: None)
    monkeypatch.setattr(sar, 'example_output_dir', lambda: None)
    monkeypatch.setattr(sar, 'convert_spice', lambda *args, **kw: 0)
    inputs = np.linspace(-0.7, 0.7, 101)
    _, _, _, actual = sar.sar_transfer_spice(n, inputs)
    w, wtot = twin.unit_cap_array(n, sigma=0.0)
    expected = twin.sar_codes(inputs, w, wtot, vref=sar.VREFP,
                              dac_gain=2**n/(2**n-1),
                              comp_offset=-sar.VREFP/(2**n-1))
    np.testing.assert_array_equal(actual, expected)


def test_static_linearity_does_not_turn_uncovered_width_into_missing_code():
    from silk.examples import sar_adc_example as twin
    # A slightly smaller dummy raises the final transition above the old ramp end.
    w = np.array([1., 2., 4., 8.])
    dnl, _, transitions = twin.static_dnl_inl(4, w, 15.99)
    assert np.isfinite(transitions[1:]).all()
    assert dnl[-1] > -1
