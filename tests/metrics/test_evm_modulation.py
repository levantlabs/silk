"""TASK-369: EVM against a named modulation with a fitted complex gain."""
import math
import warnings

import numpy as np
import pytest

import silk.metrics as M
from silk.metrics.diagnostics import MetricWarning
from silk.metrics.rf import MODULATIONS
from silk.signals import schema as S
from silk.signals.base import Signal

SPP = 4
SIZES = {'bpsk': 2, 'qpsk': 4, '8psk': 8, '16qam': 16, '64qam': 64,
         '256qam': 256, 'pam4': 4, 'pam8': 8}


def _iq(symbols):
    """Zero-order-hold I/Q transients, SPP samples per unit symbol period."""
    t = np.arange(symbols.size * SPP) / SPP
    return (Signal('i', 'v', 'time', 'V', t, np.repeat(symbols.real, SPP)),
            Signal('q', 'v', 'time', 'V', t, np.repeat(symbols.imag, SPP)))


def _symbols(modulation, n, seed=0):
    grid = M.modulation_grid(modulation)
    return grid[np.random.default_rng(seed).integers(0, grid.size, n)]


def test_supported_set_is_the_literal():
    assert MODULATIONS == tuple(SIZES)


@pytest.mark.parametrize('modulation', SIZES)
def test_grid_is_unit_rms_and_distinct(modulation):
    grid = M.modulation_grid(modulation)
    assert grid.size == SIZES[modulation]
    assert np.mean(np.abs(grid) ** 2) == pytest.approx(1.0, rel=1e-12)
    assert np.all(np.abs(grid) > 0)
    assert np.unique(np.round(grid, 12)).size == grid.size
    if modulation in ('bpsk', 'pam4', 'pam8'):
        assert np.all(grid.imag == 0)


@pytest.mark.parametrize('modulation', SIZES)
@pytest.mark.parametrize('gain', [0.37 * np.exp(0.4j), 2.5 * np.exp(-2.9j)])
def test_scaled_and_rotated_constellation_reads_zero(modulation, gain):
    received = gain * _symbols(modulation, 600)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', MetricWarning)
        r = M.evm_constellation(*_iq(received), symbol_period=1.0,
                                modulation=modulation)
        scalar = M.evm_percent(*_iq(received), symbol_period=1.0,
                               modulation=modulation)
    assert r.evm_percent == pytest.approx(0.0, abs=1e-9)
    assert abs(r.fitted_gain) == pytest.approx(abs(gain), rel=1e-12)
    np.testing.assert_allclose(r.assigned_ideals, received, atol=1e-12)
    assert scalar == pytest.approx(0.0, abs=1e-9)
    if modulation == '256qam':
        assert len(caught) == 2  # One per call, including the scalar wrapper.
        for warning in caught:
            assert warning.message.code == 'incomplete_constellation'
            assert warning.message.details['covered_points'] == 231
            assert warning.message.details['total_points'] == 256
    else:
        assert not caught


@pytest.mark.parametrize('modulation', SIZES)
def test_known_noise_gives_the_analytic_evm(modulation):
    # Complex Gaussian noise of per-axis sigma on a unit-RMS grid scaled by A:
    # EVM = sqrt(2)*sigma (as a fraction of A). sigma is small enough that no
    # decision is wrong on the densest grid (256-QAM half-spacing ~0.077).
    n, amp, sigma = 20000, 0.8, 0.005
    rng = np.random.default_rng(7)
    noise = sigma * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    received = amp * np.exp(1.1j) * (_symbols(modulation, n, seed=3) + noise)
    evm = M.evm_percent(*_iq(received), symbol_period=1.0, modulation=modulation)
    assert evm == pytest.approx(100 * math.sqrt(2) * sigma, rel=0.03)


def test_ideal_points_override_is_unchanged_and_unfitted():
    ideals = M.modulation_grid('qpsk')
    received = 1.1 * _symbols('qpsk', 64)       # 10 % radial error, not fitted
    r = M.evm_constellation(*_iq(received), symbol_period=1.0,
                            ideal_points=ideals)
    assert r.evm_percent == pytest.approx(10.0, rel=1e-9)
    assert r.fitted_gain is None
    fitted = M.evm_constellation(*_iq(received), symbol_period=1.0,
                                 modulation='qpsk')
    assert fitted.evm_percent == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize('kwargs,word', [
    ({'modulation': 'qpsk', 'ideal_points': [1, -1]}, 'both'),
    ({}, 'neither'),
])
def test_exactly_one_reference_input(kwargs, word):
    with pytest.raises(ValueError, match=f'exactly one of modulation.*{word}'):
        M.evm_constellation(*_iq(_symbols('qpsk', 8)), symbol_period=1.0,
                            **kwargs)


@pytest.mark.parametrize('call', [
    lambda i, q: M.evm_constellation(i, q, symbol_period=1.0, modulation='32qam'),
    lambda i, q: M.evm_percent(i, q, symbol_period=1.0, modulation='32qam'),
    lambda i, q: M.modulation_grid('32qam'),
])
def test_unknown_modulation_names_the_supported_set(call):
    with pytest.raises(ValueError, match="'32qam'.*bpsk, qpsk, 8psk, 16qam, "
                                         "64qam, 256qam, pam4, pam8"):
        call(*_iq(_symbols('qpsk', 8)))


def test_reference_requires_ideal_points():
    received = _symbols('qpsk', 8)
    with pytest.raises(ValueError, match='reference requires ideal_points'):
        M.evm_constellation(*_iq(received), symbol_period=1.0,
                            modulation='qpsk', reference=received)


def test_dead_signal_is_nan_with_diagnostic():
    with pytest.warns(MetricWarning, match='zero_gain'):
        r = M.evm_constellation(*_iq(np.zeros(16, dtype=complex)),
                                symbol_period=1.0, modulation='16qam')
    assert math.isnan(r.evm_percent)
    assert r.fitted_gain == 0


def test_nan_symbols_are_nan():
    received = _symbols('qpsk', 16).copy()
    received[3] = complex(np.nan, 0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        evm = M.evm_percent(*_iq(received), symbol_period=1.0, modulation='qpsk')
    assert math.isnan(evm)
    assert any(getattr(w.message, 'code', None) == 'nan_symbols' for w in caught)


def test_evm_percent_schema_entry_is_scalar_and_string_only():
    entry = S.export_metric_schema()['evm_percent']
    assert entry['unit'] == '%'
    assert [s['name'] for s in entry['signal_inputs']] == ['i_sig', 'q_sig']
    params = {p['name']: p for p in entry['parameters']}
    assert list(params) == ['symbol_period', 'modulation', 'tstart', 'n_symbols']
    assert {p['type'] for p in params.values()} <= {'float', 'int', 'str'}
    assert params['modulation']['type'] == 'str'
    assert params['modulation']['required'] is True
    assert params['modulation']['optional'] is False
    assert params['modulation']['choices'] == list(MODULATIONS)
    assert 'ideal_points' not in params


@pytest.mark.parametrize('modulation', SIZES)
def test_every_declared_choice_is_accepted(modulation):
    M.evm_percent(*_iq(_symbols(modulation, 32)), symbol_period=1.0,
                  modulation=modulation)


@pytest.mark.parametrize('name', ['evm_constellation', 'evm_percent'])
@pytest.mark.parametrize('count', [None, 4])
def test_subnanosecond_endpoint_roundoff(name, count):
    t = np.linspace(0, .7e-9, 8)
    i = Signal('i', 'voltage', 'time', 'V', t, np.ones_like(t))
    q = Signal('q', 'voltage', 'time', 'V', t, np.ones_like(t))
    result = getattr(M, name)(i, q, symbol_period=.2e-9,
                              modulation='qpsk', n_symbols=count)
    assert (result.evm_percent if name == 'evm_constellation' else result) == pytest.approx(0., abs=1e-10)


@pytest.mark.parametrize('name', ['evm_constellation', 'evm_percent'])
@pytest.mark.parametrize('overrun', [.5e-9, 2e-9])
def test_endpoint_tolerance_boundary(name, overrun):
    period = .2e-9
    # Relative overrun measured in symbol periods, not in seconds.
    t = np.linspace(0, (3.5-overrun)*period, 8)
    i = Signal('i', 'voltage', 'time', 'V', t, np.ones_like(t))
    q = Signal('q', 'voltage', 'time', 'V', t, np.ones_like(t))
    call = lambda: getattr(M, name)(i, q, symbol_period=period,
                                    modulation='qpsk', n_symbols=4)
    if overrun > 1e-9:
        with pytest.raises(ValueError, match='only|exceed'):
            call()
    else:
        result = call()
        assert np.isfinite(result.evm_percent if name == 'evm_constellation' else result)


@pytest.mark.parametrize('name', ['evm_constellation', 'evm_percent'])
def test_endpoint_tolerance_rejects_real_overrun(name):
    t = np.linspace(0, .7e-9, 8)
    i = Signal('i', 'voltage', 'time', 'V', t, np.ones_like(t))
    with pytest.raises(ValueError, match='only|exceed'):
        getattr(M, name)(i, i, symbol_period=.2e-9, modulation='qpsk', n_symbols=5)


@pytest.mark.parametrize('name', ['evm_constellation', 'evm_percent'])
def test_evm_explicit_count_must_fit_both_actual_spans(name):
    ti = np.linspace(0, .7e-9, 8)
    # Within the grid tolerance, but the fourth center is outside Q support.
    tq = np.linspace(0, .7e-9 - 1e-15, 8)
    i = Signal('i', 'voltage', 'time', 'V', ti, np.ones(8))
    q = Signal('q', 'voltage', 'time', 'V', tq, np.ones(8))
    with pytest.raises(ValueError, match='only 3|exceeds'):
        getattr(M, name)(i, q, symbol_period=.2e-9, modulation='qpsk', n_symbols=4)


def test_evm_default_count_uses_common_span():
    ti = np.linspace(0, .7e-9, 8)
    # Within the grid tolerance, but the fourth center is outside Q support.
    tq = np.linspace(0, .7e-9 - 1e-15, 8)
    i = Signal('i', 'voltage', 'time', 'V', ti, np.ones(8))
    q = Signal('q', 'voltage', 'time', 'V', tq, np.ones(8))
    result = M.evm_constellation(i, q, symbol_period=.2e-9, modulation='qpsk')
    assert len(result.received) == 3


def test_q_endpoint_roundoff_uses_the_same_period_tolerance():
    ti = np.linspace(0, .7e-9, 8)
    tq = np.linspace(0, .7e-9 - .5e-9*.2e-9, 8)
    i = Signal('i', 'voltage', 'time', 'V', ti, np.ones(8))
    q = Signal('q', 'voltage', 'time', 'V', tq, np.ones(8))
    result = M.evm_constellation(i, q, symbol_period=.2e-9,
                                 modulation='qpsk', n_symbols=4)
    assert len(result.received) == 4


@pytest.mark.parametrize('gain', [.4 * np.exp(.3j), 3 * np.exp(1.1j),
                                  .05 * np.exp(2.6j)])
@pytest.mark.parametrize('shuffle', [False, True])
def test_uneven_full_constellation_fits_actual_samples(gain, shuffle):
    grid = M.modulation_grid('64qam')
    symbols = np.r_[grid, np.repeat(grid[0], 32)]
    if shuffle:
        symbols = np.random.default_rng(24).permutation(symbols)
    received = gain * symbols
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = M.evm_constellation(*_iq(received), symbol_period=1.,
                                     modulation='64qam')
        scalar = M.evm_percent(*_iq(received), symbol_period=1., modulation='64qam')
    assert not caught
    assert result.evm_percent == pytest.approx(0., abs=1e-10)
    assert scalar == pytest.approx(0., abs=1e-10)
    assert abs(result.fitted_gain) == pytest.approx(abs(gain), rel=1e-12)
    assert (result.fitted_gain / gain) ** 4 == pytest.approx(1., abs=1e-12)
    np.testing.assert_allclose(result.received, received, atol=1e-14)
    np.testing.assert_allclose(result.assigned_ideals, received, atol=1e-14)


def test_uneven_noisy_capture_matches_known_symbol_least_squares():
    grid = M.modulation_grid('64qam')
    symbols = np.r_[grid, np.repeat(grid[0], 32)]
    rng = np.random.default_rng(20261023)
    noise = .005 * (rng.standard_normal(96) + 1j * rng.standard_normal(96))
    received = .4 * np.exp(.3j) * (symbols + noise)
    # Independent oracle: transmitted identities are known in this test only.
    oracle = np.vdot(symbols, received) / np.vdot(symbols, symbols)
    errors = received - oracle * symbols
    expected = 100 * np.sqrt(np.mean(abs(errors) ** 2)) / abs(oracle)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = M.evm_constellation(*_iq(received), symbol_period=1.,
                                     modulation='64qam')
        scalar = M.evm_percent(*_iq(received), symbol_period=1., modulation='64qam')
    assert not caught
    assert result.evm_percent == pytest.approx(expected, rel=1e-12)
    assert scalar == pytest.approx(expected, rel=1e-12)
    assert abs(result.fitted_gain) == pytest.approx(abs(oracle), rel=1e-12)


def test_peak_outlier_keeps_the_better_existing_fit():
    grid = M.modulation_grid('16qam')
    symbols = np.r_[grid, np.repeat(grid[0], 32)]
    symbols[0] *= 1.5
    received = .4 * np.exp(.3j) * symbols
    result = M.evm_constellation(*_iq(received), symbol_period=1.,
                                 modulation='16qam')
    # Frozen pre-change residual; a peak-only seed gives 0.019913194444444445.
    assert np.mean(abs(result.errors) ** 2) <= .0014633152173913037 + 1e-16
    assert abs(result.fitted_gain) == pytest.approx(.40489130434782616)


def test_equal_residual_retains_the_original_orientation():
    from silk.metrics.rf import _fit_complex_gain
    # Original seed gives +1, peak seed -1: both fit exactly.
    assert _fit_complex_gain(np.array([1., -1.]),
                             np.array([-1., 1.]), 2) == 1.


@pytest.mark.parametrize('modulation', SIZES)
def test_complete_grid_has_no_coverage_warning(modulation):
    received = .4 * np.exp(.3j) * M.modulation_grid(modulation)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = M.evm_constellation(*_iq(received), symbol_period=1.,
                                     modulation=modulation)
    assert not caught
    assert result.evm_percent == pytest.approx(0., abs=1e-10)


@pytest.mark.parametrize('shape,covered', [('missing_one', 63), ('inner', 16),
                                          ('constant', 1)])
@pytest.mark.parametrize('name', ['evm_constellation', 'evm_percent'])
def test_incomplete_capture_warns_once_and_retains_result(shape, covered, name):
    grid = M.modulation_grid('64qam')
    symbols = (grid[:-1] if shape == 'missing_one' else
               grid[(abs(grid.real) < .5) & (abs(grid.imag) < .5)]
               if shape == 'inner' else np.repeat(grid[0], 32))
    received = .4 * np.exp(.3j) * symbols
    with pytest.warns(MetricWarning) as caught:
        result = getattr(M, name)(*_iq(received), symbol_period=1., modulation='64qam')
    assert len(caught) == 1
    warning = caught[0].message
    assert warning.metric == 'evm_constellation'
    assert warning.code == 'incomplete_constellation'
    assert warning.details == dict(modulation='64qam', n_symbols=symbols.size,
                                   covered_points=covered, total_points=64,
                                   proximity_fraction=.5)
    assert 'longer' in warning.suggestion.lower()
    if name == 'evm_constellation':
        assert np.isfinite(result.evm_percent)
        np.testing.assert_allclose(result.received, received)
        np.testing.assert_allclose(result.errors, received - result.assigned_ideals)
    else:
        assert np.isfinite(result)


@pytest.mark.parametrize('factor,covered', [(1 - 1e-9, 2), (1., 2), (1 + 1e-9, 1),
                                          (10., 1)])
@pytest.mark.parametrize('gain', [1., .4 * np.exp(.3j)])
def test_coverage_uses_distance_in_fitted_units(monkeypatch, factor, covered, gain):
    from silk.metrics import rf
    # Isolate proximity from fitting: BPSK spacing is 2, radius 1. A point
    # outside +1 is assigned to +1 even when it is too far to cover it.
    monkeypatch.setattr(rf, '_fit_complex_gain', lambda *args: gain)
    received = gain * np.array([-1., 1. + factor])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = M.evm_constellation(*_iq(received), symbol_period=1., modulation='bpsk')
    if covered == 2:
        assert not caught
    else:
        assert len(caught) == 1
        assert caught[0].message.details['covered_points'] == 1
    np.testing.assert_allclose(result.assigned_ideals, gain * np.array([-1., 1.]))


def test_incomplete_coverage_obeys_warning_policy():
    signals = _iq(np.ones(8, dtype=complex))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('ignore', MetricWarning)
        value = M.evm_percent(*signals, symbol_period=1., modulation='bpsk')
    assert not caught and value == pytest.approx(0.)
    with warnings.catch_warnings():
        warnings.simplefilter('error', MetricWarning)
        with pytest.raises(MetricWarning, match='incomplete_constellation'):
            M.evm_percent(*signals, symbol_period=1., modulation='bpsk')


@pytest.mark.parametrize('received,code', [(np.zeros(8, complex), 'zero_gain'),
                                         (np.full(8, complex(np.nan, 0)), 'nan_symbols')])
def test_unavailable_fit_does_not_add_coverage_warning(received, code):
    with pytest.warns(MetricWarning) as caught:
        value = M.evm_percent(*_iq(received), symbol_period=1., modulation='bpsk')
    assert math.isnan(value)
    assert [w.message.code for w in caught] == [code]


@pytest.mark.parametrize('known_reference', [False, True])
def test_explicit_points_do_not_warn_about_incomplete_coverage(known_reference):
    received = np.ones(8, complex)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = M.evm_constellation(*_iq(received), symbol_period=1.,
                                     ideal_points=[-1., 1.],
                                     reference=received if known_reference else None)
    assert not caught
    assert result.evm_percent == 0.
    assert result.fitted_gain is None
