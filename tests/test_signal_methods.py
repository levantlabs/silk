"""Tests for the magnitude / phase / dB / real / imag methods on Signal.

These methods were added 2026-04-25 (`mag`, `phase_deg`, `phase_rad`,
`phase_deg_unwrapped`, `phase_rad_unwrapped`, `db10`, `db20`, `dbm`, `real`, `imag`).
"""
import math
import warnings

import numpy as np
import pytest

from silk.signals.base import Signal


def _ac_sig(freqs, complex_y):
    """Build a complex AC Signal."""
    return Signal('h', 'voltage', 'frequency', 'V',
                  np.asarray(freqs, dtype=float),
                  np.asarray(complex_y))


# ---------------------------------------------------------------------------
# Each method returns a Signal of the same concrete subclass
# ---------------------------------------------------------------------------

class TestSubclassPreservation:
    def test_mag_returns_signal(self):
        sig = _ac_sig([1.0, 10.0], [1+1j, 2+2j])
        result = sig.mag()
        assert isinstance(result, Signal)
        assert type(result) is type(sig)

    def test_db20_returns_signal(self):
        sig = _ac_sig([1.0, 10.0], [1+1j, 2+2j])
        assert isinstance(sig.db20(), Signal)


# ---------------------------------------------------------------------------
# mag / real / imag
# ---------------------------------------------------------------------------

class TestMag:
    def test_complex_magnitude(self):
        sig = _ac_sig([1.0, 10.0, 100.0],
                      [3+4j, 5+12j, 8+15j])
        assert np.allclose(sig.mag().y, [5.0, 13.0, 17.0])

    def test_real_signal(self):
        sig = _ac_sig([1.0, 2.0], [-3.0, 4.0])
        assert np.allclose(sig.mag().y, [3.0, 4.0])

    def test_x_unchanged(self):
        sig = _ac_sig([1.0, 10.0, 100.0], [1+1j, 2+2j, 3+3j])
        assert np.allclose(sig.mag().x, sig.x)


class TestReal:
    def test_complex(self):
        sig = _ac_sig([1.0, 2.0], [3+4j, 5-6j])
        assert np.allclose(sig.real().y, [3.0, 5.0])


class TestImag:
    def test_complex(self):
        sig = _ac_sig([1.0, 2.0], [3+4j, 5-6j])
        assert np.allclose(sig.imag().y, [4.0, -6.0])


# ---------------------------------------------------------------------------
# Phase
# ---------------------------------------------------------------------------

class TestPhaseRad:
    def test_known_values(self):
        sig = _ac_sig([1.0, 2.0, 3.0],
                      [1+0j, 0+1j, -1+0j])
        # angles: 0, π/2, π
        assert np.allclose(sig.phase_rad().y, [0.0, math.pi/2, math.pi])

    def test_ylabel_set(self):
        sig = _ac_sig([1.0], [1+0j])
        assert sig.phase_rad().ylabel == 'rad'

    def test_type_set(self):
        sig = _ac_sig([1.0], [1+0j])
        assert sig.phase_rad().type == 'phase'


class TestPhaseDeg:
    def test_known_values(self):
        sig = _ac_sig([1.0, 2.0, 3.0],
                      [1+0j, 0+1j, -1+0j])
        assert np.allclose(sig.phase_deg().y, [0.0, 90.0, 180.0])

    def test_ylabel_set(self):
        sig = _ac_sig([1.0], [1+0j])
        assert sig.phase_deg().ylabel == 'deg'


class TestPhaseUnwrapped:
    def test_unwrap_radians(self):
        # angles wrap from π → -π without unwrap; unwrap should preserve continuity
        # Build a signal whose argument grows continuously past π
        n = 100
        f = np.linspace(1, 100, n)
        # exp(j·linear_phase): unwrapped angle = linear_phase
        linear_phase = np.linspace(0, 4 * math.pi, n)
        sig = _ac_sig(f, np.exp(1j * linear_phase))
        unwrapped = sig.phase_rad_unwrapped().y
        # Should approximate the linear ramp without wrapping
        assert np.allclose(unwrapped, linear_phase, atol=1e-9)

    def test_unwrap_degrees(self):
        n = 100
        f = np.linspace(1, 100, n)
        linear_phase = np.linspace(0, 4 * math.pi, n)
        sig = _ac_sig(f, np.exp(1j * linear_phase))
        unwrapped = sig.phase_deg_unwrapped().y
        assert np.allclose(unwrapped, np.degrees(linear_phase), atol=1e-7)


# ---------------------------------------------------------------------------
# dB / dBm
# ---------------------------------------------------------------------------

class TestDb20:
    def test_unity_is_zero_db(self):
        sig = _ac_sig([1.0, 10.0], [1+0j, 1+0j])
        assert np.allclose(sig.db20().y, [0.0, 0.0])

    def test_factor_10_is_20_db(self):
        sig = _ac_sig([1.0], [10+0j])
        assert math.isclose(float(sig.db20().y[0]), 20.0, abs_tol=1e-12)

    def test_complex_uses_magnitude(self):
        sig = _ac_sig([1.0], [3+4j])   # |y| = 5
        assert math.isclose(float(sig.db20().y[0]),
                            20.0 * math.log10(5.0), rel_tol=1e-12)

    def test_ylabel_set(self):
        sig = _ac_sig([1.0], [1+0j])
        assert sig.db20().ylabel == 'dB'


class TestDb10:
    def test_unity_is_zero(self):
        sig = _ac_sig([1.0, 10.0], [1.0, 1.0])
        assert np.allclose(sig.db10().y, [0.0, 0.0])

    def test_factor_10_is_10_db(self):
        sig = _ac_sig([1.0], [10.0])
        assert math.isclose(float(sig.db10().y[0]), 10.0, abs_tol=1e-12)


class TestDbm:
    def test_known_at_50_ohm(self):
        # 1 V across 50 Ω → P = 1²/50 = 20 mW = 13.01 dBm
        sig = _ac_sig([1.0], [1+0j])
        assert math.isclose(float(sig.dbm(50).y[0]),
                            10.0 * math.log10(0.02 / 1e-3),
                            rel_tol=1e-9)

    def test_default_z0_is_50(self):
        sig = _ac_sig([1.0], [1+0j])
        assert sig.dbm().y[0] == sig.dbm(z0=50).y[0]

    def test_invalid_z0(self):
        sig = _ac_sig([1.0], [1+0j])
        with pytest.raises(ValueError, match='z0 must be > 0'):
            sig.dbm(z0=0)


# ---------------------------------------------------------------------------
# Chainability and back-compat
# ---------------------------------------------------------------------------

class TestChainability:
    def test_mag_then_db20(self):
        # Real signal: mag() is identity, db20() works
        sig = _ac_sig([1.0, 2.0], [10.0, 100.0])
        chained = sig.mag().db20()
        assert np.allclose(chained.y, [20.0, 40.0])

    def test_db20_directly_on_complex(self):
        # db20 takes magnitude internally; should match mag().db20()
        sig = _ac_sig([1.0], [3+4j])
        a = sig.db20().y[0]
        b = sig.mag().db20().y[0]
        assert math.isclose(float(a), float(b), rel_tol=1e-12)


class TestBackwardCompat:
    """Existing arithmetic / plot path must still work after _new gained
    an optional ylabel kwarg (default None preserves old behaviour).
    """

    def test_add_still_works(self):
        a = _ac_sig([1.0, 2.0], [1+0j, 2+0j])
        b = _ac_sig([1.0, 2.0], [3+0j, 4+0j])
        c = a + b
        assert np.allclose(c.y, [4+0j, 6+0j])
        assert c.ylabel == 'V'   # inherited unchanged

    def test_sub_still_works(self):
        a = _ac_sig([1.0, 2.0], [5+0j, 7+0j])
        b = _ac_sig([1.0, 2.0], [2+0j, 3+0j])
        c = a - b
        assert np.allclose(c.y, [3+0j, 4+0j])

    def test_mul_still_works(self):
        a = _ac_sig([1.0, 2.0], [2+0j, 3+0j])
        b = _ac_sig([1.0, 2.0], [4+0j, 5+0j])
        c = a * b
        assert np.allclose(c.y, [8+0j, 15+0j])


# ---------------------------------------------------------------------------
# Division operator
# ---------------------------------------------------------------------------

class TestTrueDiv:
    def test_basic(self):
        a = _ac_sig([1.0, 2.0], [10+0j, 12+0j])
        b = _ac_sig([1.0, 2.0], [2+0j, 3+0j])
        c = a / b
        assert np.allclose(c.y, [5+0j, 4+0j])

    def test_type_combined(self):
        a = _ac_sig([1.0, 2.0], [1+0j, 1+0j])
        b = _ac_sig([1.0, 2.0], [1+0j, 1+0j])
        c = a / b
        assert '/' in c.type

    def test_xlabel_check(self):
        a = _ac_sig([1.0, 2.0], [1+0j, 1+0j])
        b = Signal('s', 'voltage', 'time', 'V',
                   np.array([1.0, 2.0]), np.array([1+0j, 1+0j]))
        with pytest.raises(ValueError, match='different x-labels'):
            _ = a / b


# ---------------------------------------------------------------------------
# Waveform manipulation methods
# ---------------------------------------------------------------------------

def _trans(t, y):
    return Signal('s', 'voltage', 'time', 'V',
                  np.asarray(t, dtype=float), np.asarray(y, dtype=float))


class TestShift:
    def test_basic(self):
        t = np.linspace(0, 10, 1001)
        sig = _trans(t, t.copy())
        s = sig.shift(2.0)
        assert math.isclose(float(s.x[0]), -2.0, abs_tol=1e-12)
        assert np.allclose(s.y, sig.y)


class TestWindow:
    def test_basic(self):
        t = np.linspace(0, 10, 1001)
        sig = _trans(t, t.copy())
        w = sig.window(2.0, 5.0)
        assert w.x[0] >= 2.0
        assert w.x[-1] <= 5.0

    def test_invalid(self):
        sig = _trans([0, 1], [0, 1])
        with pytest.raises(ValueError, match='x2 > x1'):
            sig.window(0.5, 0.5)


class TestFlipMethod:
    def test_y_reversed(self):
        sig = _trans([0, 1, 2, 3], [10.0, 20.0, 30.0, 40.0])
        f = sig.flip()
        assert np.allclose(f.y, [40.0, 30.0, 20.0, 10.0])
        assert np.allclose(f.x, sig.x)

    def test_uniform_grid_bit_identical_to_unchanged_x(self):
        """TASK-313 AC#2: uniform grid -> x unchanged, bit-identical."""
        x = np.array([0, 1, 2, 3], dtype=float)
        for y in (
            [10.0, 20.0, 30.0, 40.0],
            [1 + 2j, 3 - 4j, 5j, -6 + 1j],
            [1.0, np.nan, 3.0, 4.0],
        ):
            sig = Signal('s', 'voltage', 'time', 'V', x, np.asarray(y))
            f = sig.flip()
            assert np.array_equal(f.x, x)
            assert np.array_equal(np.asarray(y)[::-1], f.y, equal_nan=True)

    def test_non_uniform_grid_mirrors_x(self):
        """TASK-313 AC#1: x mirrored to match reversed y, not left unchanged."""
        sig = _trans([0, 1, 1.1, 3], [0.0, 0.0, 1.0, 2.0])
        f = sig.flip()
        assert np.allclose(f.x, [0, 1.9, 2, 3])
        assert np.allclose(f.y, [2.0, 1.0, 0.0, 0.0])

    def test_uniform_grid_x_exactly_unchanged(self):
        """TASK-313 AC#2: exact (bit-identical) x on realistic uniform grids.

        The naive mirror formula ``x[0] + x[-1] - x[::-1]`` is NOT
        bit-identical to ``x`` on these grids (measured up to 1 ULP on
        ~90% of samples) — ``flip`` must detect the uniform grid and
        return it unchanged rather than recompute it.
        """
        for x in (
            np.linspace(0, 1e-6, 1001),
            np.arange(10) * 0.1,
            np.linspace(5e-9, 105e-9, 4097),
        ):
            sig = _trans(x, np.arange(len(x), dtype=float))
            f = sig.flip()
            assert np.array_equal(f.x, x)


class TestInterpolate:
    def test_known(self):
        sig = _trans([0.0, 1.0, 2.0], [0.0, 1.0, 4.0])
        result = sig.interpolate([0.5, 1.5])
        assert np.allclose(result.y, [0.5, 2.5])

    def test_complex_y_raises(self):
        """TASK-114 ruling: there is no single right answer for what "interpolate a complex trace"
        should mean, so a complex Signal handed to ``interpolate`` is a
        caller-error request (decision-1 category 2) — raise, naming the
        limitation and the workaround, rather than silently splitting into
        real/imag or dropping Im(y) as the pre-fix ``dtype=float`` cast
        did.
        """
        sig = _ac_sig([1.0, 2.0, 3.0], [1 + 1j, 2 + 2j, 3 + 3j])
        with pytest.raises(ValueError, match='complex'):
            sig.interpolate([1.5, 2.5])

    def test_real_and_imag_interpolate_separately(self):
        """The documented workaround actually works."""
        sig = _ac_sig([0.0, 1.0, 2.0], [0 + 0j, 1 + 2j, 4 + 8j])
        real_part = sig.real().interpolate([0.5, 1.5])
        imag_part = sig.imag().interpolate([0.5, 1.5])
        assert np.allclose(real_part.y, [0.5, 2.5])
        assert np.allclose(imag_part.y, [1.0, 5.0])


class TestDerivativeMethod:
    def test_linear_ramp_unity(self):
        t = np.linspace(0, 5, 1001)
        sig = _trans(t, t.copy())
        d = sig.derivative()
        assert np.allclose(d.y[5:-5], 1.0, rtol=1e-6)

    def test_real_input_byte_unchanged(self):
        """Real-input result is unchanged by the TASK-110 delegation.

        ``Signal.derivative`` now delegates to
        ``silk.metrics.waveform.derivative`` instead of computing
        ``np.gradient`` inline; on a real Signal the output array must
        be exactly (not just approximately) what the old inline
        ``np.asarray(y, dtype=float); np.gradient(y, x)`` produced.
        """
        t = np.linspace(0, 5, 101)
        y = t ** 2
        sig = _trans(t, y)
        d = sig.derivative()
        expected = np.gradient(np.asarray(y, dtype=float),
                                np.asarray(t, dtype=float))
        assert np.array_equal(d.y, expected)
        assert d.y.dtype == np.float64

    def test_complex_input_no_complexwarning_and_correct_gradient(self):
        """Before TASK-110, ``Signal.derivative`` did
        ``np.asarray(self.y, dtype=float)`` unconditionally (mirroring the
        pre-TASK-106 defect in ``silk.metrics.waveform.derivative``), so a
        complex Signal (e.g. an .ac result, H(jw)) silently lost Im(y) with
        only a ``numpy.exceptions.ComplexWarning`` — confirmed by
        reproducing it against the pre-fix code path (plain
        ``np.asarray(y, dtype=float)`` on the same input raises exactly
        this warning under ``warnings.simplefilter('error')``). Fixed by
        delegating to TASK-106's complex-aware
        ``silk.metrics.waveform.derivative``, which differentiates the
        real and imaginary parts together via ``np.gradient`` on the
        untouched complex array.
        """
        t = np.linspace(0, 1, 5)
        # y = t + 2j*t**2  =>  dy/dt = 1 + 4j*t
        y = t + 2j * t ** 2
        sig = Signal('h', 'voltage', 'time', 'V', t, y)

        # Pre-fix behaviour still raises under warnings-as-errors.
        with warnings.catch_warnings():
            warnings.simplefilter('error', np.exceptions.ComplexWarning)
            with pytest.raises(np.exceptions.ComplexWarning):
                np.asarray(sig.y, dtype=float)

        with warnings.catch_warnings():
            warnings.simplefilter('error', np.exceptions.ComplexWarning)
            d = sig.derivative()

        assert np.iscomplexobj(d.y)
        expected = np.gradient(y, t)
        assert np.allclose(d.y.real, expected.real, rtol=1e-9)
        assert np.allclose(d.y.imag, expected.imag, rtol=1e-9)

    def test_agrees_with_metrics_transient_derivative(self):
        """Signal.derivative and silk.metrics.waveform.derivative must
        agree exactly (TASK-110 delegates one to the other)."""
        from silk.metrics.waveform import derivative as metrics_derivative

        t = np.linspace(0, 2, 33)
        for y in (np.sin(t), t + 3j * t ** 2):
            sig = Signal('h', 'voltage', 'time', 'V', t, y)
            via_method = sig.derivative()
            via_function = metrics_derivative(sig)
            assert np.array_equal(via_method.x, via_function.x)
            assert np.array_equal(via_method.y, via_function.y)


class TestIntegralMethod:
    def test_constant(self):
        t = np.linspace(0, 5, 5001)
        sig = _trans(t, np.ones_like(t))
        i = sig.integral()
        assert math.isclose(float(i.y[-1]), 5.0, rel_tol=1e-9)
        assert i.y[0] == 0.0

    def test_real_input_byte_unchanged(self):
        """Real-input result is unchanged by the TASK-114 dtype-preserving
        fix — on a real Signal the output array must be exactly (not just
        approximately) what the old unconditional
        ``np.asarray(y, dtype=float)`` cumulative trapezoid produced."""
        t = np.linspace(0, 5, 101)
        y = t ** 2
        sig = _trans(t, y)
        i = sig.integral()

        x_ref = np.asarray(t, dtype=float)
        y_ref = np.asarray(y, dtype=float)
        expected = np.zeros_like(y_ref, dtype=float)
        expected[1:] = np.cumsum(
            0.5 * (y_ref[1:] + y_ref[:-1]) * (x_ref[1:] - x_ref[:-1])
        )
        assert np.array_equal(i.y, expected)
        assert i.y.dtype == np.float64

    def test_complex_input_no_complexwarning_and_correct_integral(self):
        """Before TASK-114, ``Signal.integral`` did
        ``np.asarray(self.y, dtype=float)`` unconditionally (the same
        defect TASK-106 fixed in ``silk.metrics.waveform.derivative`` and
        TASK-110 fixed in ``Signal.derivative``), so a complex Signal (e.g.
        an .ac result, H(jw)) silently lost Im(y) with only a
        ``numpy.exceptions.ComplexWarning`` — confirmed by reproducing it
        against the pre-fix code path (plain ``np.asarray(y, dtype=float)``
        on the same input raises exactly this warning under
        ``warnings.simplefilter('error')``). Fixed by keeping ``y``
        complex and letting the cumulative trapezoid integrate real and
        imaginary parts independently.
        """
        t = np.linspace(0, 1, 5)
        # y = t + 2j*t**2
        y = t + 2j * t ** 2
        sig = Signal('h', 'voltage', 'time', 'V', t, y)

        # Pre-fix behaviour still raises under warnings-as-errors.
        with warnings.catch_warnings():
            warnings.simplefilter('error', np.exceptions.ComplexWarning)
            with pytest.raises(np.exceptions.ComplexWarning):
                np.asarray(sig.y, dtype=float)

        with warnings.catch_warnings():
            warnings.simplefilter('error', np.exceptions.ComplexWarning)
            i = sig.integral()

        assert np.iscomplexobj(i.y)
        expected_real = np.zeros(5)
        expected_real[1:] = np.cumsum(
            0.5 * (y.real[1:] + y.real[:-1]) * (t[1:] - t[:-1])
        )
        expected_imag = np.zeros(5)
        expected_imag[1:] = np.cumsum(
            0.5 * (y.imag[1:] + y.imag[:-1]) * (t[1:] - t[:-1])
        )
        assert np.allclose(i.y.real, expected_real, rtol=1e-9)
        assert np.allclose(i.y.imag, expected_imag, rtol=1e-9)


class TestFftMethod:
    def test_returns_freq_signal(self):
        N = 1024
        TCLK = 1e-9
        t = np.arange(N) * TCLK
        sig = _trans(t, np.sin(2 * np.pi * 11 * np.arange(N) / N))
        spec = sig.fft(tclk=TCLK, n_samples=N)
        assert spec.xlabel == 'frequency'
        assert len(spec.x) == N // 2 + 1


class TestChainability:
    def test_window_then_derivative(self):
        t = np.linspace(0, 10, 1001)
        sig = _trans(t, t ** 2)
        # Derivative of x² is 2x; in window [2, 5], derivative ≈ 4, 6, 8, 10
        d = sig.window(2.0, 5.0).derivative()
        assert np.allclose(d.y[10:-10], 2.0 * d.x[10:-10], rtol=1e-3)


# ---------------------------------------------------------------------------
# TASK-214: scalar arithmetic, reflected operators, negation, exponent;
# complex passthrough; y-label warning fires only on type mismatch (never).
# ---------------------------------------------------------------------------

class TestScalarArithmetic:
    """v OP k and k OP v for +, -, *, / keep x and type unchanged."""

    def test_add_scalar_right(self):
        v = _trans([0.0, 1.0], [1.0, 2.0])
        r = v + 0.5
        assert np.allclose(r.y, [1.5, 2.5])
        assert r.type == 'voltage'
        assert np.allclose(r.x, v.x)
        assert r.name == 's+0.5'

    def test_add_scalar_left(self):
        v = _trans([0.0, 1.0], [1.0, 2.0])
        r = 0.5 + v
        assert np.allclose(r.y, [1.5, 2.5])
        assert r.type == 'voltage'
        assert r.name == '0.5+s'

    def test_sub_scalar_right(self):
        v = _trans([0.0, 1.0], [1.0, 2.0])
        r = v - 1
        assert np.allclose(r.y, [0.0, 1.0])
        assert r.type == 'voltage'
        assert r.name == 's-1'

    def test_sub_scalar_left(self):
        v = _trans([0.0, 1.0], [1.0, 2.0])
        r = 1 - v
        assert np.allclose(r.y, [0.0, -1.0])
        assert r.type == 'voltage'
        assert r.name == '1-s'

    def test_mul_scalar_right(self):
        v = _trans([0.0, 1.0], [1.0, 2.0])
        r = v * 0.5
        assert np.allclose(r.y, [0.5, 1.0])
        assert r.type == 'voltage'
        assert r.name == 's*0.5'

    def test_mul_scalar_left(self):
        v = _trans([0.0, 1.0], [1.0, 2.0])
        r = 0.5 * v
        assert np.allclose(r.y, [0.5, 1.0])
        assert r.type == 'voltage'
        assert r.name == '0.5*s'

    def test_div_scalar_right(self):
        v = _trans([0.0, 1.0], [1.0, 2.0])
        r = v / 2
        assert np.allclose(r.y, [0.5, 1.0])
        assert r.type == 'voltage'
        assert r.name == 's/2'

    def test_div_scalar_left(self):
        v = _trans([0.0, 1.0], [1.0, 2.0])
        r = 2 / v
        assert np.allclose(r.y, [2.0, 1.0])
        assert r.type == 'voltage'
        assert r.name == '2/s'

    def test_neg(self):
        v = _trans([0.0, 1.0], [1.0, -2.0])
        r = -v
        assert np.allclose(r.y, [-1.0, 2.0])
        assert r.type == 'voltage'
        assert r.name == '-s'

    def test_pow(self):
        v = _trans([0.0, 1.0], [2.0, 3.0])
        r = v ** 2
        assert np.allclose(r.y, [4.0, 9.0])
        assert r.type == 'voltage^2'
        assert r.name == '(s)**2'

    def test_pow_non_real_exponent_raises(self):
        v = _trans([0.0, 1.0], [2.0, 3.0])
        with pytest.raises(TypeError):
            v ** 'two'

    def test_pow_signal_exponent_raises(self):
        v = _trans([0.0, 1.0], [2.0, 3.0])
        w = _trans([0.0, 1.0], [1.0, 1.0])
        with pytest.raises(TypeError):
            v ** w

    def test_unsupported_operand_raises_typeerror(self):
        v = _trans([0.0, 1.0], [2.0, 3.0])
        with pytest.raises(TypeError):
            v + 'nope'
        with pytest.raises(TypeError):
            'nope' + v

    def test_signal_signal_still_requires_same_type(self):
        """Scalar support must not weaken the Signal-Signal type check."""
        v = _trans([0.0, 1.0], [2.0, 3.0])
        i = Signal('i', 'current', 'time', 'A',
                   np.asarray([0.0, 1.0]), np.asarray([1.0, 1.0]))
        with pytest.raises(ValueError, match='different types'):
            v + i


class TestSignalSignalArithmeticUnchanged:
    """Signal-with-Signal +, -, *, / behave byte-for-byte as before."""

    def test_add(self):
        a = _trans([0.0, 1.0], [1.0, 2.0])
        b = _trans([0.0, 1.0], [3.0, 4.0])
        r = a + b
        assert np.allclose(r.y, [4.0, 6.0])
        assert r.type == 'voltage'
        assert r.name == 's+s'

    def test_mul_composes_type(self):
        v = _trans([0.0, 1.0], [2.0, 3.0])
        i = Signal('i', 'current', 'time', 'A',
                   np.asarray([0.0, 1.0]), np.asarray([1.0, 1.0]))
        r = v * i
        assert r.type == 'voltage*current'
        assert np.allclose(r.y, [2.0, 3.0])


class TestComplexPassthrough:
    """AC-style complex samples pass through every operator unchanged;
    no magnitude is taken."""

    def test_mul_scalar_keeps_complex_dtype(self):
        v = _ac_sig([1.0, 2.0], [1 + 2j, 3 - 1j])
        r = v * 0.5
        assert r.y.dtype == np.complex128
        assert np.allclose(r.y, [0.5 + 1j, 1.5 - 0.5j])

    def test_add_scalar_keeps_complex_dtype(self):
        v = _ac_sig([1.0, 2.0], [1 + 2j, 3 - 1j])
        r = v + 1.0
        assert r.y.dtype == np.complex128
        assert np.allclose(r.y, [2 + 2j, 4 - 1j])

    def test_sub_scalar_keeps_complex_dtype(self):
        v = _ac_sig([1.0, 2.0], [1 + 2j, 3 - 1j])
        r = v - 1.0
        assert r.y.dtype == np.complex128
        assert np.allclose(r.y, [0 + 2j, 2 - 1j])

    def test_div_scalar_keeps_complex_dtype(self):
        v = _ac_sig([1.0, 2.0], [2 + 4j, 6 - 2j])
        r = v / 2.0
        assert r.y.dtype == np.complex128
        assert np.allclose(r.y, [1 + 2j, 3 - 1j])

    def test_neg_keeps_complex_dtype(self):
        v = _ac_sig([1.0, 2.0], [1 + 2j, 3 - 1j])
        r = -v
        assert r.y.dtype == np.complex128
        assert np.allclose(r.y, [-1 - 2j, -3 + 1j])

    def test_pow_keeps_complex_dtype(self):
        v = _ac_sig([1.0, 2.0], [1 + 1j, 2 + 0j])
        r = v ** 2
        assert r.y.dtype == np.complex128
        assert np.allclose(r.y, [(1 + 1j) ** 2, (2 + 0j) ** 2])

    def test_reflected_scalar_keeps_complex_dtype(self):
        v = _ac_sig([1.0, 2.0], [1 + 2j, 3 - 1j])
        r = 2.0 * v
        assert r.y.dtype == np.complex128
        assert np.allclose(r.y, [2 + 4j, 6 - 2j])


class TestYLabelWarningOnlyOnTypeMismatch:
    """The 'different y-labels' warning is gone: it used to fire on any
    same-type pair with different names (pure noise, TASK-214). A type
    mismatch still raises, exactly as before -- it never reaches a
    y-label check at all."""

    def test_same_type_different_names_no_warning(self):
        vin = Signal('vin', 'voltage', 'time', 'vin',
                     np.asarray([0.0, 1.0]), np.asarray([1.0, 2.0]))
        vout = Signal('vout', 'voltage', 'time', 'vout',
                      np.asarray([0.0, 1.0]), np.asarray([0.5, 1.5]))
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            result = vin + vout
        assert np.allclose(result.y, [1.5, 3.5])

    def test_voltage_plus_current_raises_as_before(self):
        v = Signal('v', 'voltage', 'time', 'V',
                   np.asarray([0.0, 1.0]), np.asarray([1.0, 2.0]))
        i = Signal('i', 'current', 'time', 'A',
                   np.asarray([0.0, 1.0]), np.asarray([1.0, 1.0]))
        with pytest.raises(ValueError, match='different types'):
            v + i


# ---------------------------------------------------------------------------
# TASK-600: Signal.interpolate shares the axis policy of silk.metrics.sample.
# Expected values are computed by hand from the piecewise-linear curve through
# (0, 1), (0.9, 2), (1.8, 4), not by either implementation.
# ---------------------------------------------------------------------------

_T600_TARGETS = [0.9, 1.35, 0.45, -1, 2, 0.0, 1.8]
_T600_EXPECTED = [2.0, 3.0, 1.5, 1.0, 4.0, 1.0, 4.0]


def _t600_pair(descending):
    x, y = [0.0, 0.9, 1.8], [1.0, 2.0, 4.0]
    if descending:
        x, y = x[::-1], y[::-1]
    return _trans(x, y)


def _t600_entry_points():
    from silk.metrics import sample
    return {
        'interpolate': lambda s, t: s.interpolate(t),
        'sample': lambda s, t: sample(s, x_new=np.asarray(t, dtype=float)),
    }


class TestInterpolateAxisPolicy600:
    @pytest.mark.parametrize('descending', [False, True])
    @pytest.mark.parametrize('entry', ['interpolate', 'sample'])
    def test_matrix(self, entry, descending):
        sig = _t600_pair(descending)
        x0, y0 = sig.x.copy(), sig.y.copy()
        r = _t600_entry_points()[entry](sig, _T600_TARGETS)
        np.testing.assert_allclose(r.y, _T600_EXPECTED)
        np.testing.assert_array_equal(r.x, _T600_TARGETS)  # order preserved
        np.testing.assert_array_equal(sig.x, x0)           # not mutated
        np.testing.assert_array_equal(sig.y, y0)

    def test_reported_failure(self):
        sig = _trans([1.8, 0.9, 0.0], [4.0, 2.0, 1.0])
        assert list(sig.interpolate([0.9]).y) == [2.0]

    def test_metadata_preserved(self):
        r = _t600_pair(True).interpolate([0.5])
        assert r.name == 'interp(s)'
        assert r.type == 'voltage'
        assert (r.xlabel, r.ylabel) == ('time', 'V')
        from silk.metrics import sample
        r = sample(_t600_pair(True), x_new=np.array([0.5]))
        assert r.name == 'sample(s)' and r.type == 'voltage'
        assert (r.xlabel, r.ylabel) == ('time', 'V')

    @pytest.mark.parametrize('x', [
        [0.0, 0.0, 1.0],            # repeated
        [0.0, 2.0, 1.0],            # fold-back
        [0.0, np.nan, 1.0],         # nonfinite
        [0.0, np.inf, 1.0],
    ])
    def test_malformed_axis_refused(self, x):
        sig = _trans(x, [1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match='interpolate'):
            sig.interpolate([0.5])

    def test_mismatched_and_empty_refused(self):
        with pytest.raises(ValueError, match='interpolate'):
            _trans([0.0, 1.0], [1.0, 2.0, 3.0]).interpolate([0.5])
        with pytest.raises(ValueError, match='interpolate'):
            _trans([], []).interpolate([0.5])

    def test_singleton_source(self):
        r = _trans([1.0], [7.0]).interpolate([0.0, 1.0, 2.0])
        np.testing.assert_array_equal(r.y, [7.0, 7.0, 7.0])

    def test_nan_sample_propagates(self):
        sig = _trans([0.0, 1.0, 2.0], [0.0, np.nan, 4.0])
        y = sig.interpolate([0.0, 0.5, 1.5, 2.0]).y
        assert y[0] == 0.0 and y[3] == 4.0
        assert np.isnan(y[1]) and np.isnan(y[2])

    def test_scalar_and_empty_targets_unchanged(self):
        sig = _t600_pair(True)
        assert float(sig.interpolate(0.45).y) == pytest.approx(1.5)
        assert len(sig.interpolate([]).y) == 0

    def test_complex_still_refused_without_numpy_claim(self):
        sig = _ac_sig([1.0, 2.0], [1 + 1j, 2 + 2j])
        with pytest.raises(ValueError, match='complex') as e:
            sig.interpolate([1.5])
        assert 'np.interp' not in str(e.value)
        from silk.metrics import sample
        assert sample(sig, x_new=np.array([1.5])).y[0] == pytest.approx(1.5 + 1.5j)

    @pytest.mark.parametrize('entry', ['interpolate', 'sample'])
    def test_descending_target_list(self, entry):
        r = _t600_entry_points()[entry](_t600_pair(True), [1.8, 1.35, 0.9, 0.45, 0.0])
        np.testing.assert_allclose(r.y, [4.0, 3.0, 2.0, 1.5, 1.0])

    @pytest.mark.parametrize('entry', ['interpolate', 'sample'])
    def test_non_1d_source_refused(self, entry):
        sig = _trans([0.0, 1.0, 2.0], [1.0, 2.0, 3.0])
        sig.y = np.ones((3, 2))
        with pytest.raises(ValueError, match=entry):
            _t600_entry_points()[entry](sig, [0.5])
        sig = _trans([0.0, 1.0, 2.0], [1.0, 2.0, 3.0])
        sig.x = np.zeros((3, 1))
        with pytest.raises(ValueError, match=entry):
            _t600_entry_points()[entry](sig, [0.5])

    def test_scalar_and_empty_targets_per_wrapper(self):
        from silk.metrics import sample
        sig = _t600_pair(True)
        assert float(sig.interpolate(0.45).y) == pytest.approx(1.5)
        assert len(sig.interpolate([]).y) == 0
        with pytest.raises(TypeError):
            sample(sig, x_new=0.45)
        with pytest.raises(ValueError, match='at least 1'):
            sample(sig, x_new=[])

    @pytest.mark.parametrize('version', ['42', '47'])
    @pytest.mark.parametrize('entry', ['interpolate', 'sample'])
    def test_saved_descending_dc_trace(self, entry, version):
        from pathlib import Path
        from silk.signals.spice import NgspiceSignal
        from silk.signals.rawfile import read_rawfile
        path = (Path(__file__).parent / 'fixtures/device_param_probe'
                / f'ngspice{version}_dc_sweep_pfet.raw')
        s = NgspiceSignal.from_rawfile(read_rawfile(path), 'vgp', 'V')
        assert np.all(np.diff(s.x) < 0)
        x0, y0 = s.x.copy(), s.y.copy()
        mid = (x0[4] + x0[5]) / 2
        r = _t600_entry_points()[entry](s, [x0[3], mid, x0[0] + 1, x0[-1] - 1])
        np.testing.assert_allclose(
            r.y, [y0[3], (y0[4] + y0[5]) / 2, y0[0], y0[-1]])
        np.testing.assert_array_equal(s.x, x0)
        np.testing.assert_array_equal(s.y, y0)
