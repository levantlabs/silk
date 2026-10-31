"""Tests for silk.metrics.fom — figures of merit (closed-form scalars)."""
import math
import warnings

import numpy as np
import pytest

from silk.metrics.fom import (
    walden_fom, schreier_fom, iip3_extrapolation,
    pae, dc_to_rf_efficiency,
)


class TestWaldenFom:
    def test_known_value(self):
        # P=1mW, fs=1MS/s, ENOB=10 → 1e-3 / (1024 · 1e6) ≈ 977 fJ/conv
        fom = walden_fom(power_w=1e-3, fs_hz=1e6, enob_bits=10.0)
        assert math.isclose(fom, 1e-3 / (1024.0 * 1e6), rel_tol=1e-9)

    @pytest.mark.parametrize('fs_hz', [0, -1.5])
    def test_invalid_fs(self, fs_hz):
        with pytest.raises(ValueError) as caught:
            walden_fom(1e-3, fs_hz, 10)
        assert type(caught.value) is ValueError
        assert str(caught.value) == (
            f'walden_fom: fs_hz must be > 0 Hz, got {fs_hz} Hz')


class TestSchreierFom:
    def test_known_value(self):
        # SNDR=70, BW=10 MHz, P=1 mW → 70 + 100 = 170 dB
        assert math.isclose(schreier_fom(70.0, 10e6, 1e-3), 170.0, abs_tol=1e-9)

    @pytest.mark.parametrize('bw_hz', [0, -1.5])
    def test_invalid_bw(self, bw_hz):
        with pytest.raises(ValueError) as caught:
            schreier_fom(70, bw_hz, 1e-3)
        assert type(caught.value) is ValueError
        assert str(caught.value) == (
            f'schreier_fom: bw_hz must be > 0 Hz, got {bw_hz} Hz')

    @pytest.mark.parametrize('power_w', [0, -1.5])
    def test_invalid_power(self, power_w):
        with pytest.raises(ValueError) as caught:
            schreier_fom(70, 1e6, power_w)
        assert type(caught.value) is ValueError
        assert str(caught.value) == (
            f'schreier_fom: power_w must be > 0 W, got {power_w} W')

    def test_bandwidth_guard_precedes_power_guard(self):
        with pytest.raises(ValueError, match='^schreier_fom: bw_hz'):
            schreier_fom(70, 0, 0)


class TestIip3Extrapolation:
    def test_known(self):
        # P_in=-30, P_fund=-30, P_imd3=-90 → IIP3 = 0 dBm
        assert math.isclose(iip3_extrapolation(-30.0, -30.0, -90.0),
                            0.0, abs_tol=1e-12)


class TestImports:
    def test_top_level(self):
        from silk.metrics import walden_fom, schreier_fom, iip3_extrapolation
        assert callable(walden_fom)


# ---------------------------------------------------------------------------
# Tier 3 — vendor-parity FoM additions
# ---------------------------------------------------------------------------

class TestPae:
    def test_known_value(self):
        # Pout=1W, Pin=0.1W, Pdc=2W → PAE = 0.45
        assert math.isclose(pae(p_rf_out_w=1.0, p_rf_in_w=0.1, p_dc_w=2.0),
                            0.45, rel_tol=1e-9)

    def test_zero_input_drive(self):
        # No input drive → PAE collapses to drain efficiency
        assert math.isclose(pae(p_rf_out_w=1.0, p_rf_in_w=0.0, p_dc_w=2.0),
                            0.5, rel_tol=1e-9)

    @pytest.mark.parametrize('p_dc_w', [0, -1.5])
    def test_invalid_dc(self, p_dc_w):
        with pytest.raises(ValueError) as caught:
            pae(1.0, 0.1, p_dc_w)
        assert type(caught.value) is ValueError
        assert str(caught.value) == (
            f'pae: p_dc_w must be > 0 W, got {p_dc_w} W')


class TestDcToRfEfficiency:
    def test_known(self):
        assert math.isclose(dc_to_rf_efficiency(p_rf_w=1.0, p_dc_w=2.5),
                            0.4, rel_tol=1e-9)

    @pytest.mark.parametrize('p_dc_w', [0, -1.5])
    def test_invalid_dc(self, p_dc_w):
        with pytest.raises(ValueError) as caught:
            dc_to_rf_efficiency(1.0, p_dc_w)
        assert type(caught.value) is ValueError
        assert str(caught.value) == (
            f'dc_to_rf_efficiency: p_dc_w must be > 0 W, got {p_dc_w} W')


class TestTier3FomImports:
    def test_top_level(self):
        from silk.metrics import pae, dc_to_rf_efficiency
        assert callable(pae)


@pytest.mark.parametrize(('metric', 'args'), [
    (walden_fom, (1e-3, 1e6, 10)),
    (schreier_fom, (70, 1e6, 1e-3)),
    (iip3_extrapolation, (-30, -30, -90)),
    (pae, (1, 0.1, 2)),
    (dc_to_rf_efficiency, (1, 2)),
])
def test_nan_operands_propagate_without_warnings(metric, args):
    for index in range(len(args)):
        operands = list(args)
        operands[index] = math.nan
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            assert math.isnan(metric(*operands))
        assert not caught


@pytest.mark.parametrize(('metric', 'args', 'expected'), [
    (walden_fom, (0, 1e6, 10), 0),
    (walden_fom, (-1, 1, 0), -1),
    (walden_fom, (math.inf, 1e6, 10), math.inf),
    (walden_fom, (1, math.inf, 10), 0),
    (walden_fom, (math.inf, math.inf, 10), math.nan),
    (schreier_fom, (0, 1, 1), 0),
    (schreier_fom, (-10, 1, 1), -10),
    (schreier_fom, (70, math.inf, 1), math.inf),
    (schreier_fom, (70, 1, math.inf), -math.inf),
    (schreier_fom, (70, math.inf, math.inf), math.nan),
    (iip3_extrapolation, (-30, -30, -90), 0),
    (iip3_extrapolation, (math.inf, -30, -90), math.inf),
    (iip3_extrapolation, (-30, math.inf, math.inf), math.nan),
    (pae, (1, 1, 2), 0),
    (pae, (0, 1, 2), -0.5),
    (pae, (math.inf, 1, 2), math.inf),
    (pae, (1, 0.1, math.inf), 0),
    (pae, (math.inf, math.inf, 2), math.nan),
    (dc_to_rf_efficiency, (0, 2), 0),
    (dc_to_rf_efficiency, (-1, 2), -0.5),
    (dc_to_rf_efficiency, (math.inf, 2), math.inf),
    (dc_to_rf_efficiency, (1, math.inf), 0),
    (dc_to_rf_efficiency, (math.inf, math.inf), math.nan),
])
def test_arithmetic_limits_without_new_warnings(metric, args, expected):
    # log10(0) already emits a NumPy warning; only new warnings are forbidden.
    with np.errstate(divide='ignore'):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            actual = metric(*args)
    if math.isnan(expected):
        assert math.isnan(actual)
    else:
        assert actual == expected
    assert not caught
