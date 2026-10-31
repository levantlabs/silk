"""Noise warning coverage: consumed data, warning policy and numeric contracts."""
import math
import warnings

import numpy as np
import pytest

from silk.metrics import noise
from silk.metrics.diagnostics import MetricWarning
from silk.signals.base import Signal


def density(y, x=None, unit='V/sqrt(Hz)'):
    if x is None:
        x = np.arange(1, len(y) + 1)
    return Signal('density', 'noise', 'frequency', unit,
                  np.asarray(x, dtype=float), np.asarray(y, dtype=float))


LEDGER = [
    ('one_over_f_corner', (density([2, np.nan, 1, 1]),), 'nan_density'),
    ('one_over_f_corner', (density([0, 0, 0, 0]),), 'nonpositive_white_floor'),
    ('integrated_noise', (density([1, np.nan, 1]), 1, 3), 'nan_density'),
    ('noise_density_at_freq', (density([1, 2, 3], [1, 10, 100]), 1000),
     'frequency_outside_sweep'),
    ('noise_density_at_freq', (density([1, np.nan, 3], [1, 10, 100]), 10),
     'nan_density'),
]


@pytest.mark.parametrize('name,args,code', LEDGER)
@pytest.mark.parametrize('policy', ['always', 'ignore', 'error'])
def test_ledger_warning_policies_preserve_float_nan(name, args, code, policy):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        warnings.simplefilter(policy, MetricWarning)
        if policy == 'error':
            with pytest.raises(MetricWarning) as error:
                getattr(noise, name)(*args)
            diagnostic = error.value
        else:
            result = getattr(noise, name)(*args)
            assert type(result) is float and math.isnan(result)
            if policy == 'ignore':
                assert caught == []
                return
            assert len(caught) == 1
            diagnostic = caught[0].message
    assert type(diagnostic) is MetricWarning
    assert diagnostic.metric == name
    assert diagnostic.code == code
    assert diagnostic.message and diagnostic.suggestion
    assert 'frequency_min_hz' in diagnostic.details
    assert 'frequency_max_hz' in diagnostic.details
    assert 'sample_count' in diagnostic.details
    assert diagnostic.details['density_unit'] == args[0].ylabel
    assert all(v is None or isinstance(v, (str, float, int, bool))
               for v in diagnostic.details.values())


@pytest.mark.parametrize('unit', ['V/sqrt(Hz)', 'A/sqrt(Hz)'])
@pytest.mark.parametrize('nan_index', [0, 3])
def test_corner_nan_in_knee_or_floor_preserves_density_units(unit, nan_index):
    y = np.array([4., 3., 2., 1.])
    y[nan_index] = np.nan
    with pytest.warns(MetricWarning) as caught:
        assert math.isnan(noise.one_over_f_corner(density(y, [1, 10, 100, 1000], unit)))
    d = caught[0].message.details
    assert d['sample_count'] == 4 and d['nan_sample_count'] == 1
    assert d['frequency_min_hz'] == 1 and d['frequency_max_hz'] == 1000
    assert d['nan_frequency_min_hz'] == d['nan_frequency_max_hz'] == 10 ** nan_index
    assert d['density_min'] == np.nanmin(y)
    assert d['density_max'] == np.nanmax(y)
    assert d['density_unit'] == unit


@pytest.mark.parametrize('floor', [0, -2])
def test_nonpositive_floor_reports_only_estimator_samples_with_sign(floor):
    with pytest.warns(MetricWarning) as caught:
        assert math.isnan(noise.one_over_f_corner(density(
            [100, 50, floor, floor], [1, 10, 100, 1000], unit='A/sqrt(Hz)')))
    assert len(caught) == 1
    d = caught[0].message.details
    assert d['white_floor'] == floor
    assert d['density_min'] == d['density_max'] == floor
    assert d['density_unit'] == 'A/sqrt(Hz)'
    assert d['sample_count'] == 2 and d['available_sample_count'] == 4
    assert d['frequency_min_hz'] == 100 and d['frequency_max_hz'] == 1000


@pytest.mark.parametrize('unit', ['V/sqrt(Hz)', 'A/sqrt(Hz)'])
@pytest.mark.parametrize('y', [[np.nan, -2], [np.nan, np.nan]])
def test_integration_warning_describes_selected_band_only(unit, y):
    sig = density([999, *y, np.nan], [1, 2, 3, 4], unit)
    with pytest.warns(MetricWarning) as caught:
        assert math.isnan(noise.integrated_noise(sig, 1.5, 3.5))
    assert len(caught) == 1
    d = caught[0].message.details
    assert d['frequency_min_hz'] == 2 and d['frequency_max_hz'] == 3
    assert d['band_min_hz'] == 1.5 and d['band_max_hz'] == 3.5
    assert d['sample_count'] == 2 and d['nan_sample_count'] == np.isnan(y).sum()
    assert d['density_min'] == d['density_max'] == (None if np.isnan(y[1]) else -2)
    assert d['density_unit'] == unit


@pytest.mark.parametrize('level,expected', [(1, 1), (0, 0), (np.inf, np.inf)])
def test_integration_ignores_unused_nan_and_preserves_zero_infinity(level, expected):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        value = noise.integrated_noise(density([level, level, np.nan]), 1, 2)
    assert type(value) is float and value == expected
    assert caught == []


def test_nested_input_referred_noise_keeps_one_originating_explanation():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        value = noise.input_referred_noise(
            noise.integrated_noise(density([1, np.nan, 1]), 1, 3), 10)
    assert type(value) is float and math.isnan(value)
    assert len(caught) == 1
    assert caught[0].message.metric == 'integrated_noise'
    assert caught[0].message.code == 'nan_density'


def test_successful_corner_remains_unchanged():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        corner = noise.one_over_f_corner(density([8, 4, 1, 1], [1, 10, 100, 1000]))
    assert type(corner) is float and corner == 100
    assert caught == []


@pytest.mark.parametrize('freq,expected', [(1, 1), (10, 2), (100, 3), (37, 2)])
def test_density_at_freq_nearest_sample_remains_quiet(freq, expected):
    """37 Hz is nearer the grid point at 10 than the one at 100 (log grid)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        value = noise.noise_density_at_freq(density([1, 2, 3], [1, 10, 100]), freq,
                                            interpolate=False)
    assert type(value) is float and value == expected
    assert caught == []


@pytest.mark.parametrize('name,args,match', [
    ('integrated_noise', (density([np.nan] * 3), 1, 1),
     r'integrated_noise: band \[1, 1\] Hz contains 1 of 3 sweep points'),
    ('noise_density_at_freq', (density([1, 2, 3], [1, 10, 100]), float('nan')),
     r'noise_density_at_freq: freq must be finite, got nan Hz'),
    ('noise_density_at_freq', (density([1, 2, 3], [1, 10, 100]), float('inf')),
     r'noise_density_at_freq: freq must be finite, got inf Hz'),
])
def test_existing_caller_errors_precede_diagnostics(name, args, match):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('error', MetricWarning)
        with pytest.raises(ValueError, match=match):
            getattr(noise, name)(*args)
    assert caught == []
