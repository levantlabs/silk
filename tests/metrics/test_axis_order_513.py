"""Shared coordinate contracts, including real descending simulation sweeps."""
import hashlib
import inspect
import json
from pathlib import Path
import warnings

import numpy as np
import pytest

import silk.metrics as T
from silk.metrics import ac, converter as dc, noise, rf, spectral
from silk.signals.base import Signal
from silk.signals.spice import NgspiceSignal
from silk.signals.rawfile import read_rawfile
from silk.signals.schema import export_metric_schema


def signal(x, y=None, xlabel='time', ylabel='V'):
    x = np.asarray(x, dtype=float)
    return Signal('axis_test', 'voltage', xlabel, ylabel, x,
                  np.asarray(y if y is not None else 2 + np.sin(x)))


def reverse(s):
    return signal(s.x[::-1], s.y[::-1], s.xlabel, s.ylabel)


WINDOW_METRICS = 'average rms stddev variance peak peak_to_peak minimum maximum argmin_x argmax_x integral'.split()
TIME_METRICS = '''rise_time fall_time rise_times fall_times slew_rate slew_rates
settling_time propagation_delay propagation_delays overshoot undershoot
average_period average_frequency cycle_period cycle_frequency average_duty cycle_duty
pulse_width pulse_widths period_jitter period_jitter_pp cycle_to_cycle_jitter
cycle_to_cycle_jitter_pp period_deviation cycle_to_cycle_difference crossing_jitter_rms
average_power dynamic_energy_per_cycle hysteresis on_time off_time histogram eye_diagram
eye_height eye_width tie long_term_jitter allan_deviation'''.split()
SPECTRAL_METRICS = '''fft sndr sinad sfdr sfdr_db enob thd thd_percent noise_floor_dbfs
harmonic_levels_db harmonic_level_dbc imd3_db oip3_dbm iip3_dbm psd spectral_power
harmonic mix_product snr harmonic_freqs welch_psd coherence evm_constellation evm_percent'''.split()
AC_METRICS = '''passband_ripple_db stopband_attenuation_db q_from_peak group_delay
gain_bw_product psrr_db dc_gain_dB peak_gain_dB gain_at_freq bandwidth_3dB unity_gain_freq
phase_at_freq phase_margin gain_margins gain_margin cmrr_dB'''.split()
NOISE_METRICS = '''one_over_f_corner integrated_noise noise_density_at_freq
noise_figure_at_freq band_noise_figure noise_figure_vs_freq'''.split()
DC_METRICS = '''inl dnl static_errors is_monotonic missing_codes inl_max_lsb dnl_max_lsb
offset_error_lsb gain_error_lsb full_scale_error_lsb missing_code_count non_monotonic_steps p1db extract_code_transitions'''.split()
GENERIC_METRICS = WINDOW_METRICS + ['derivative', 'value_at', 'final_value', 'sample',
                                    'error_norm', 'relative_error_norm', 'crossing', 'crossings']


def invoke(module, name, s):
    """Supply ordinary, valid requests; only source axis order varies."""
    fn = getattr(module, name)
    signals = {'sig', 'clock', 'a', 'b', 'goal', 'sig_in', 'sig_out',
               'sig_in_rising', 'sig_out_rising', 'sig_in_falling', 'sig_out_falling',
               'out_sig', 'in_sig', 'i_sig', 'q_sig', 'v_supply', 'v_out',
               'vdiff', 'vcm', 'noise_sig', 'noise_in', 'noise_out', 'v_in', 'code'}
    values = dict(initial=0., target=1., period=4., bins=4, threshold=0.5,
                  tclk=1., window=1., f1_hz=0.0625, f2_hz=0.09375, k=2, m=1, n=2,
                  symbol_period=2., modulation='qpsk', x=2., kind='rms', floor=0.,
                  f_pass=4., f_stop=4., freq=4., f1=1., f2=16., rs=50.,
                  temp_sim_k=290., full_scale=64., n_bits=6)
    kwargs = {}
    for p in inspect.signature(fn).parameters.values():
        if p.name in signals:
            kwargs[p.name] = s
        elif p.name in ('sig_v', 'sig_i'):
            kwargs[p.name] = signal(s.x, s.y, s.xlabel, 'V' if p.name == 'sig_v' else 'A')
        elif p.default is inspect.Parameter.empty:
            kwargs[p.name] = values[p.name]
    if name == 'moving_average':
        kwargs['window'] = 1.
    if name == 'sample':
        kwargs['n'] = 8
    if name == 'evm_constellation':
        kwargs['modulation'] = 'qpsk'
    return fn(**kwargs)


def test_descending_point_and_sweep_endpoint():
    s = signal([1.8, .9, 0], [4., 2., 1.], 'voltage')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        assert T.value_at(s, .9) == 2.
        assert T.value_at(s, .45) == 1.5
        assert T.final_value(s) == 1.
        assert T.final_value(reverse(s)) == 4.
        assert T.final_value(signal([0], [7.])) == 7.
    assert not caught


@pytest.mark.parametrize('version', ['42', '47'])
def test_real_descending_dc_endpoints(version):
    path = Path(__file__).parents[1] / 'fixtures/device_param_probe' / f'ngspice{version}_dc_sweep_pfet.raw'
    s = NgspiceSignal.from_rawfile(read_rawfile(path), 'vgp', 'V')
    assert np.all(np.diff(s.x) < 0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        assert T.value_at(s, s.x[len(s.x)//2]) == pytest.approx(s.y[len(s.x)//2])
        assert T.final_value(s) == pytest.approx(s.y[-1])
        x = (s.x[4] + s.x[5])/2
        assert T.value_at(s, x) == pytest.approx((s.y[4] + s.y[5])/2)
    assert not caught


@pytest.mark.parametrize('name', WINDOW_METRICS)
@pytest.mark.parametrize('interpolate', [True, False])
def test_generic_window_metric_descending(name, interpolate):
    s = signal([0, 1, 2, 3, 4], [4, 1, 0, 1, 4], 'voltage')
    fn = getattr(T, name)
    kwargs = dict(start=.5, end=3.5, interpolate=interpolate)
    if name in ('argmin_x', 'argmax_x'):
        kwargs['refine'] = True
    assert fn(reverse(s), **kwargs) == pytest.approx(fn(s, **kwargs))
    assert fn(reverse(s)) == pytest.approx(fn(s))


def test_foldback_path_and_mask_remain_supported():
    s = signal([0, 2, 1, 3], [1., 1., 1., 1.])
    np.testing.assert_array_equal(T.running_integral(s).y, [0, 2, 1, 3])
    np.testing.assert_array_equal(T.window(s, .5, 2.5).x, [2, 1])


@pytest.mark.parametrize('name', ['running_integral', 'derivative', 'sample', 'window'])
def test_order_preserving_transform_descending(name):
    s = signal([3, 2, 1, 0], [7., 5., 3., 1.])
    if name == 'running_integral':
        out = T.running_integral(s)
        np.testing.assert_allclose(out.y, [0, -6, -10, -12])
    elif name == 'derivative':
        out = T.derivative(s)
        np.testing.assert_allclose(out.y, 2.)
    elif name == 'sample':
        out = T.sample(s, n=7)
        np.testing.assert_allclose(out.y, 2*out.x+1)
    else:
        out = T.window(s, .5, 2.5)
        np.testing.assert_array_equal(out.y, [5, 3])
    assert np.all(np.diff(out.x) < 0)


@pytest.mark.parametrize('name', ['error_norm', 'relative_error_norm'])
@pytest.mark.parametrize('reverse_sig,reverse_goal', [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize('kind', ['max', 'rms', 'mean_abs'])
def test_error_norm_axis_orders(name, reverse_sig, reverse_goal, kind):
    s = signal(np.linspace(0, 4, 9), np.ones(9)*2j)
    g = signal([0, 2, 4], [1j, -1j, 1j])
    kwargs = dict(kind=kind, start=.5, end=3.5)
    if name == 'relative_error_norm':
        kwargs['floor'] = .1
    fn = getattr(T, name)
    assert fn(reverse(s) if reverse_sig else s, reverse(g) if reverse_goal else g, **kwargs) == pytest.approx(fn(s, g, **kwargs))


@pytest.mark.parametrize('xlabel', ['time', 'voltage'])
@pytest.mark.parametrize('edge,expected', [('rising', [2.5, .5]), ('falling', [3.5, 1.5]), ('any', [3.5, 2.5, 1.5, .5])])
def test_crossings_follow_acquisition_order(xlabel, edge, expected):
    s = signal([4, 3, 2, 1, 0], [1., 0., 1., 0., 1.], xlabel)
    np.testing.assert_allclose(T.crossings(s, threshold=.5, edge=edge), expected)
    assert T.crossing(s, .5, edge=edge, n=1) == expected[0]
    assert T.crossing(s, .5, edge=edge, n=-1) == expected[-1]
    np.testing.assert_allclose(T.crossings(s, threshold=.5, edge=edge, start=1., end=3.), [v for v in expected if 1 <= v <= 3])
    # Auto threshold and a window cutting a segment must use the same support.
    assert T.crossing(s, edge='falling', start=3.25, end=3.75) == 3.5


REFUSAL_CASES = ([(T, n) for n in TIME_METRICS + ['moving_average', 'convolve', 'cross_correlation']]
                 + [(inspect.getmodule(getattr(T, n)), n) for n in SPECTRAL_METRICS])


@pytest.mark.parametrize('module,name', REFUSAL_CASES, ids=[n for _, n in REFUSAL_CASES])
def test_increasing_only_metric_rejects_descending(module, name):
    with pytest.raises(ValueError, match=r'axis.*(down|descending).*needs.*(ris|increas)'):
        invoke(module, name, signal(np.arange(64.)[::-1]))


ALL_CASES = (REFUSAL_CASES + [(T, n) for n in GENERIC_METRICS]
             + [(ac, n) for n in AC_METRICS] + [(noise, n) for n in NOISE_METRICS]
             + [(rf if n == "p1db" else dc, n) for n in DC_METRICS])


@pytest.mark.parametrize('module,name', ALL_CASES, ids=[f'{m.__name__.split(".")[-1]}.{n}' for m, n in ALL_CASES])
@pytest.mark.parametrize('bad_axis', ['foldback', 'duplicate', 'nonfinite'])
def test_nonmonotonic_axis_rejected(module, name, bad_axis):
    if name == 'p1db' and bad_axis == 'duplicate':
        pytest.skip('Repeated power coordinates are explicitly supported.')
    x = np.arange(1., 65.)
    x[4] = {'foldback': 2., 'duplicate': 4., 'nonfinite': np.nan}[bad_axis]
    xlabel = 'frequency' if module in (ac, noise) else 'code' if module == dc and name != 'enob' else 'time'
    if name == 'p1db':
        xlabel = 'power'
    elif name == 'extract_code_transitions':
        xlabel = 'time'
    y = np.sqrt(1 + 4/np.arange(1., 65.)) if module == noise else 2 + np.sin(np.arange(64.))
    s = signal(x, y, xlabel, 'V/sqrt(Hz)' if module == noise else 'V')
    with pytest.raises(ValueError, match=rf'^{name}:.*axis.*needs'):
        invoke(module, name, s)


@pytest.mark.parametrize('module,name', [(ac, n) for n in AC_METRICS] + [(noise, n) for n in NOISE_METRICS], ids=AC_METRICS+NOISE_METRICS)
def test_frequency_metric_descending(module, name):
    x = np.geomspace(1., 64., 65)
    y = np.sqrt(1 + 8/x) if module == noise else 10/(1+1j*x/4)**3
    if name == 'q_from_peak':
        y = np.exp(-np.log(x/8)**2)*10
    s = signal(x, y, 'frequency', 'V/sqrt(Hz)' if module == noise else 'V')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        expected = invoke(module, name, s)
        actual = invoke(module, name, reverse(s))
    if isinstance(expected, Signal):
        np.testing.assert_array_equal(actual.x, expected.x[::-1])
        np.testing.assert_allclose(actual.y, expected.y[::-1], equal_nan=True)
    else:
        np.testing.assert_allclose(actual, expected, equal_nan=True)


def test_repeated_power_coordinates_preserved():
    s = signal([0, 0, 1, 2], [10., 10., 10.5, 10.], 'power')
    assert rf.p1db(s) == pytest.approx(4/3)


@pytest.mark.parametrize('name', ['value_at', 'final_value'])
def test_descending_point_nan_support(name):
    s = signal([2, 1, 0], [np.nan, 3., 2.])
    with warnings.catch_warnings(record=True) as caught:
        result = T.value_at(s, 1.) if name == 'value_at' else T.final_value(s)
    assert result == (3. if name == 'value_at' else 2.)
    assert not caught
    with pytest.warns(UserWarning):
        assert np.isnan(T.value_at(s, 1.5))
    with pytest.warns(UserWarning):
        assert np.isnan(T.value_at(s, -1.))


def test_metric_catalog_unchanged():
    payload = json.dumps(export_metric_schema(), sort_keys=True, separators=(',', ':')).encode()
    assert hashlib.sha256(payload).hexdigest() == 'f47ca45cb4a0666deb6a955dfe495e17888bfa63308be96cb15e27966e65c040'


def test_clock_sample_count_matches_endpoint_tolerance():
    from silk.metrics.spectral import _sample_at_clock
    t = np.linspace(0, .7, 71)
    s = signal(t, t)
    result = _sample_at_clock(s, metric='fft', tclk=.01, n_samples=71)
    assert len(result) == 71
    assert result[-1] == .7
    with pytest.raises(ValueError, match='exceeds trace end'):
        _sample_at_clock(s, metric='fft', tclk=.01, n_samples=72)


def test_resonance_plateau_uses_lowest_frequency_in_either_order():
    s = signal([1, 2, 3, 4], [0., 10., 10., 0.], 'frequency')
    assert ac.q_from_peak(s) == 2.
    assert ac.q_from_peak(reverse(s)) == 2.


@pytest.mark.parametrize('name', ['window', 'running_integral'])
@pytest.mark.parametrize('fault', ['nonfinite', 'shape'])
def test_path_metrics_validate_finite_matching_samples(name, fault):
    s = signal([0, 2, 1], [1., 2., 3.])
    if fault == 'nonfinite':
        s.x[1] = np.nan
    else:
        s.y = s.y[:-1]
    with pytest.raises(ValueError, match=rf'{name}:.*axis.*needs.*matching 1-D'):
        T.window(s, 0, 2) if name == 'window' else T.running_integral(s)


@pytest.mark.parametrize('call', [
    lambda good, bad: T.error_norm(good, bad),
    lambda good, bad: T.relative_error_norm(good, bad, kind='rms', floor=0),
    lambda good, bad: T.propagation_delay(good, bad),
    lambda good, bad: T.convolve(good, bad),
    lambda good, bad: T.cross_correlation(good, bad),
    lambda good, bad: rf.evm_constellation(good, bad, symbol_period=1, modulation='qpsk'),
    lambda good, bad: rf.evm_percent(good, bad, symbol_period=1, modulation='qpsk'),
    lambda good, bad: spectral.coherence(good, bad, tclk=1, n_samples=8),
])
def test_secondary_input_axis_is_validated(call):
    good = signal(np.arange(16.))
    bad = signal(np.arange(16.))
    bad.x[4] = 2.
    with pytest.raises(ValueError, match='axis.*up then down.*needs'):
        call(good, bad)


def test_crossings_dc_nan_diagnostic_uses_coordinate_units():
    s = signal([1.8, .9, 0], [1., np.nan, 0.], 'voltage')
    with pytest.warns(UserWarning) as caught:
        assert np.isnan(T.crossings(s, threshold=.5, edge='falling')).all()
    details = caught[0].message.details
    assert details['x_label'] == 'voltage'
    assert details['x_unit'] == 'V'
    assert details['x_start'] == 1.8
    assert details['x_stop'] == 0
    assert not any(key.endswith('_s') for key in details)


@pytest.mark.parametrize('name', DC_METRICS)
def test_dc_metric_rejects_descending_with_public_name(name):
    xlabel = ('time' if name == 'extract_code_transitions' else
              'power' if name == 'p1db' else 'code')
    s = signal(np.arange(64.)[::-1], xlabel=xlabel)
    with pytest.raises(ValueError, match=rf'^{name}:.*axis.*down.*needs.*(ris|ascend|increas)'):
        invoke(rf if name == "p1db" else dc, name, s)
