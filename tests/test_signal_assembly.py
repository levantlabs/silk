"""TASK-211 — the supported hand-assembly of the Signal kinds no analysis produces.

Four of the eight kinds in ``_meta.signal_kinds`` have an empty
``produced_by``. :mod:`silk.signals.assemble` is the supported route for
the two of them that need more than a constructor call
(``code_transitions``, via a decimal-code transient, and ``power_sweep``),
and these tests pin three things:

1. the helpers produce the Signal the metrics actually accept — checked by
   running the metrics, not by inspecting an xlabel;
2. every refusal that protects against a *plausible wrong number* rather
   than a crash (bit order, bit weighting, sweep direction) really refuses;
3. the exported schema is **byte-identical** to its pre-helper baseline.
   The helpers are not analyses, so ``produced_by`` must stay empty and a
   consumer must keep refusing ``p1db`` and the transfer-curve family.
"""
import hashlib
import json
import warnings

import numpy as np
import pytest

from silk.signals import Signal, codes_from_bits, power_sweep
from silk.signals import export_metric_schema
from silk.signals.kinds import (
    CODE_TRANSITIONS, POWER_SWEEP, TRANSIENT, KindMatch, SIGNAL_KINDS,
)
from silk.metrics.converter import (
    extract_code_transitions,
    inl,
    dnl,
    is_monotonic,
    missing_codes,
)
from silk.metrics.rf import (
    p1db,
)
from silk.metrics.diagnostics import MetricWarning


@pytest.mark.parametrize('decoder', ['as_int', 'codes_from_bits'])
@pytest.mark.parametrize('msb_first', [False, True])
@pytest.mark.parametrize('width', [53, 54, 63, 64, 65, 1100])
def test_binary_exact_integer_oracle(decoder, msb_first, width):
    # Catch fixed-width wrap and floating-weight double rounding. The last
    # mixed word rounds differently if its low bits are rounded before adding
    # its high bit: 2**55 + 2**53 + 5 must round up to the next multiple of 8.
    words = [0, 5, 1 << min(width - 1, 64)]
    if width < 1100:
        words += [(1 << width) - 1]
    if width >= 56:
        words += [(1 << 55) + (1 << 53) + 5]
    x = np.arange(len(words), dtype=float)
    bits = [_bit(f'd[{i + 8}]', [(word >> i) & 1 for word in words], x)
            for i in range(width)]
    if msb_first:
        bits.reverse()
    bus = Signal._make_bus_handle('d', tuple(bits), indices=tuple(range(8, width + 8)))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = (bus.as_int(msb_first=msb_first) if decoder == 'as_int'
                  else codes_from_bits(bus, msb_first=msb_first))
    np.testing.assert_array_equal(result.y, [float(word) for word in words])
    np.testing.assert_array_equal(result.x, x)
    assert result.y.dtype == np.dtype(float)
    assert (result.type, result.xlabel, result.ylabel) == ('count', 'time', 'code')
    assert result.name == ('d.as_int' if decoder == 'as_int' else 'code')
    rounded = sum(int(float(word)) != word for word in words)
    assert len(caught) == bool(rounded)
    if rounded:
        diagnostic = caught[0].message
        assert isinstance(diagnostic, MetricWarning)
        assert diagnostic.metric == 'as_int'
        assert diagnostic.code == 'binary_code_precision_loss'
        assert diagnostic.details['rounded_sample_count'] == rounded
        assert diagnostic.details['sample_count'] == len(words)
        assert diagnostic.details['bit_width'] == width
        assert 'Keep the bit Signals to compare whole words exactly.' in str(diagnostic)


@pytest.mark.parametrize('decoder', ['as_int', 'codes_from_bits'])
def test_binary_float_overflow_names_sample(decoder):
    with pytest.raises(OverflowError):
        float(1 << 1024)
    x = np.array([0., 2.])
    bits = [_bit(f'd[{i}]', [0, i == 1024], x) for i in range(1025)]
    bus = Signal._make_bus_handle('wide', tuple(bits))
    with pytest.raises(OverflowError) as exc:
        if decoder == 'as_int':
            bus.as_int()
        else:
            codes_from_bits(bus, msb_first=False)
    assert type(exc.value) is OverflowError
    assert str(exc.value).startswith('as_int:')
    assert 'sample 1 (x=2.0)' in str(exc.value)
    assert '1025-bit' in str(exc.value)
    assert 'Keep the original bit Signals' in str(exc.value)


@pytest.mark.parametrize('decoder', ['as_int', 'codes_from_bits'])
def test_binary_precision_warning_filters(decoder):
    x = np.array([0., 1.])
    words = [1 << 53, (1 << 53) + 1]
    bits = [_bit(f'd[{i}]', [(word >> i) & 1 for word in words], x)
            for i in range(54)]
    bus = Signal._make_bus_handle('d', tuple(bits))
    def decode():
        return (bus.as_int() if decoder == 'as_int'
                else codes_from_bits(bus, msb_first=False))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', MetricWarning)
        np.testing.assert_array_equal(decode().y, [float(word) for word in words])
    with warnings.catch_warnings():
        warnings.simplefilter('error', MetricWarning)
        with pytest.raises(MetricWarning, match='9007199254740993 became 9007199254740992'):
            decode()


@pytest.mark.parametrize('width', [12, 65])
def test_binary_empty_samples(width):
    bits = [_bit(f'd[{i}]', [], np.array([])) for i in range(width)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        assert Signal._make_bus_handle('d', tuple(bits)).as_int().y.size == 0
        assert codes_from_bits(bits, msb_first=False).y.size == 0
    assert not caught


@pytest.mark.parametrize('decoder', ['as_int', 'codes_from_bits'])
def test_binary_exact_wide_values_do_not_warn(decoder):
    words = [0, 5, 1 << 53, 1 << 63, 1 << 64]
    x = np.arange(len(words), dtype=float)
    bits = tuple(_bit(f'd[{i}]', [(word >> i) & 1 for word in words], x)
                 for i in range(1100))
    bus = Signal._make_bus_handle('d', bits)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = (bus.as_int() if decoder == 'as_int'
                  else codes_from_bits(bus, msb_first=False))
    np.testing.assert_array_equal(result.y, [float(word) for word in words])
    assert not caught


@pytest.mark.parametrize('decoder', ['as_int', 'codes_from_bits'])
def test_binary_unavailable_samples_are_not_rounded_or_overflowed(decoder):
    x = np.array([0., 1., 2., 3.])
    words = [(1 << 53) + 1, (1 << 53) + 1, 1 << 1024, 5]
    bits = [_bit(f'd[{i}]', [(word >> i) & 1 for word in words], x)
            for i in range(1025)]
    bits[2].y[1:3] = np.nan
    bus = Signal._make_bus_handle('d', tuple(bits))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', MetricWarning)
        result = (bus.as_int() if decoder == 'as_int'
                  else codes_from_bits(bus, msb_first=False))
    np.testing.assert_array_equal(result.y, [float(words[0]), np.nan, np.nan, 5.])
    diagnostics = {w.message.code: w.message for w in caught}
    assert len(caught) == 2
    assert diagnostics['nan_bits'].details['affected_sample_count'] == 2
    assert diagnostics['binary_code_precision_loss'].details['rounded_sample_count'] == 1
    assert diagnostics['binary_code_precision_loss'].details['sample_count'] == 2
    assert '1 of 2 decoded samples were rounded' in str(diagnostics['binary_code_precision_loss'])


# ---------------------------------------------------------------------------
# Fixtures — synthetic bit Signals whose codes are known by construction
# ---------------------------------------------------------------------------

T = np.linspace(0.0, 7e-6, 8)


def _bit(name, values, x=None):
    """One digital trace, 0 V / 1 V, on the shared axis."""
    return Signal(name, 'voltage', 'time', 'V',
                  T if x is None else x, np.asarray(values, dtype=float))


def _binary_bits_msb_first():
    """d2,d1,d0 counting 0,1,2,...,7 — the code at sample i IS i."""
    return [
        _bit('d2', [0, 0, 0, 0, 1, 1, 1, 1]),
        _bit('d1', [0, 0, 1, 1, 0, 0, 1, 1]),
        _bit('d0', [0, 1, 0, 1, 0, 1, 0, 1]),
    ]


def _thermometer_bits():
    """Seven comparators of a 3-bit flash; the code at sample i is i."""
    return [_bit(f'vcmp{k}', [1.0 if i >= k else 0.0 for i in range(8)])
            for k in range(1, 8)]


# ---------------------------------------------------------------------------
# codes_from_bits — the decoded values
# ---------------------------------------------------------------------------

class TestCodesFromBitsValues:

    def test_binary_msb_first_counts_up(self):
        code = codes_from_bits(_binary_bits_msb_first(), msb_first=True)
        assert np.array_equal(code.y, np.arange(8, dtype=float))

    def test_bit_order_is_honoured_both_ways(self):
        bits = _binary_bits_msb_first()
        msb = codes_from_bits(bits, msb_first=True)
        lsb = codes_from_bits(list(reversed(bits)), msb_first=False)
        assert np.array_equal(msb.y, lsb.y)
        # ...and reading the SAME list the other way really is a different
        # answer, which is why msb_first has no default: 0,1,2,3.. becomes
        # the bit-reversal 0,4,2,6,1,5,3,7.
        wrong = codes_from_bits(bits, msb_first=False)
        assert np.array_equal(wrong.y, np.array([0, 4, 2, 6, 1, 5, 3, 7],
                                                dtype=float))

    def test_thermometer_counts_asserted_bits(self):
        code = codes_from_bits(_thermometer_bits(), weighting='thermometer')
        assert np.array_equal(code.y, np.arange(8, dtype=float))

    def test_thermometer_ignores_bit_order(self):
        bits = _thermometer_bits()
        a = codes_from_bits(bits, weighting='thermometer')
        b = codes_from_bits(list(reversed(bits)), weighting='thermometer')
        assert np.array_equal(a.y, b.y)

    def test_threshold_follows_the_digital_levels(self):
        """A 1.8 V rail read with a 0.9 V threshold, the flash example's case."""
        x4 = np.linspace(0.0, 3e-6, 4)
        bits = [_bit('vcmp1', [0.0, 0.8, 1.0, 1.8], x=x4),
                _bit('vcmp2', [0.0, 0.0, 0.0, 1.8], x=x4)]
        # default 0.5 V: the 0.8 V sample counts
        assert np.array_equal(
            codes_from_bits(bits, weighting='thermometer').y,
            np.array([0, 1, 1, 2], dtype=float))
        # 0.9 V: it does not
        assert np.array_equal(
            codes_from_bits(bits, weighting='thermometer', threshold=0.9).y,
            np.array([0, 0, 1, 2], dtype=float))

    def test_accepts_a_bus_handle(self):
        bits = _binary_bits_msb_first()
        handle = Signal._make_bus_handle(name='dout', underlying=tuple(bits))
        code = codes_from_bits(handle, msb_first=True)
        assert np.array_equal(code.y, np.arange(8, dtype=float))

    def test_delegates_the_binary_pack_to_as_int(self):
        """One implementation of the shift-and-add, not two (the whole point)."""
        bits = _binary_bits_msb_first()
        handle = Signal._make_bus_handle(name='dout', underlying=tuple(bits))
        assert np.array_equal(
            codes_from_bits(bits, msb_first=True).y,
            handle.as_int(msb_first=True).y)

    def test_name_defaults_to_code_and_is_overridable(self):
        bits = _binary_bits_msb_first()
        assert codes_from_bits(bits, msb_first=True).name == 'code'
        assert codes_from_bits(bits, msb_first=True, name='adc_code').name \
            == 'adc_code'
        assert codes_from_bits(_thermometer_bits(), weighting='thermometer',
                               name='flash_code').name == 'flash_code'


# ---------------------------------------------------------------------------
# codes_from_bits — the Signal it returns is a TRANSIENT, not a transfer curve
# ---------------------------------------------------------------------------

class TestCodesFromBitsKind:

    @pytest.mark.parametrize('kwargs', [
        dict(msb_first=True),
        dict(weighting='thermometer'),
    ])
    def test_result_is_a_transient_not_code_transitions(self, kwargs):
        bits = (_thermometer_bits() if kwargs.get('weighting') == 'thermometer'
                else _binary_bits_msb_first())
        code = codes_from_bits(bits, **kwargs)
        assert code.xlabel == 'time'
        assert TRANSIENT.check(code) is KindMatch.MATCHES
        # The code_transitions Signal is what extract_code_transitions
        # RETURNS; the decoded trace is its input, and handing it straight
        # to inl() must still be refused.
        assert CODE_TRANSITIONS.check(code) is KindMatch.CONFLICTS
        with pytest.raises(ValueError, match="xlabel='code'"):
            inl(code)

    def test_carries_the_bits_own_axis(self):
        """Not hardcoded to 'time' — a dc ramp capture keeps its own axis."""
        x = np.linspace(0.0, 1.8, 8)
        bits = [Signal('d0', 'voltage', 'sweep', 'V', x,
                       np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=float))]
        assert codes_from_bits(bits, msb_first=True).xlabel == 'sweep'

    def test_labels_and_type(self):
        code = codes_from_bits(_binary_bits_msb_first(), msb_first=True)
        assert (code.type, code.ylabel) == ('count', 'code')
        assert codes_from_bits(_thermometer_bits(),
                               weighting='thermometer').type == 'count'


# ---------------------------------------------------------------------------
# codes_from_bits — refusals
# ---------------------------------------------------------------------------

class TestCodesFromBitsRefusals:

    def test_msb_first_is_required_for_binary(self):
        with pytest.raises(TypeError, match='msb_first is required'):
            codes_from_bits(_binary_bits_msb_first())

    def test_msb_first_is_refused_for_thermometer(self):
        with pytest.raises(TypeError, match='meaningless'):
            codes_from_bits(_thermometer_bits(), weighting='thermometer',
                            msb_first=True)
        # including msb_first=False — a silently ignored argument would
        # read as an ordering that was honoured.
        with pytest.raises(TypeError, match='meaningless'):
            codes_from_bits(_thermometer_bits(), weighting='thermometer',
                            msb_first=False)

    def test_unknown_weighting(self):
        with pytest.raises(ValueError, match='weighting must be one of'):
            codes_from_bits(_binary_bits_msb_first(), weighting='gray',
                            msb_first=True)

    def test_empty_bits(self):
        with pytest.raises(ValueError, match='at least one bit'):
            codes_from_bits([], msb_first=True)

    def test_scalar_signal_is_not_a_bit_vector(self):
        with pytest.raises(TypeError, match='width 1'):
            codes_from_bits(_bit('d0', [0, 1, 0, 1, 0, 1, 0, 1]),
                            msb_first=True)

    def test_raw_arrays_are_refused(self):
        with pytest.raises(TypeError, match=r'bits\[0\] is ndarray'):
            codes_from_bits([np.zeros(8)], msb_first=True)

    def test_non_iterable(self):
        with pytest.raises(TypeError, match='bus Signal or a sequence'):
            codes_from_bits(3, msb_first=True)

    def test_bits_of_different_length(self):
        bits = [_bit('d1', [0, 1, 0, 1, 0, 1, 0, 1]),
                _bit('d0', [0, 1, 0], x=np.linspace(0.0, 2e-6, 3))]
        with pytest.raises(ValueError, match='share one x-axis'):
            codes_from_bits(bits, msb_first=True)

    def test_bits_on_a_different_grid(self):
        bits = [_bit('d1', [0, 1, 0, 1, 0, 1, 0, 1]),
                _bit('d0', [0, 1, 0, 1, 0, 1, 0, 1],
                     x=np.linspace(0.0, 9e-6, 8))]
        with pytest.raises(ValueError, match='different grid'):
            codes_from_bits(bits, msb_first=True)


# ---------------------------------------------------------------------------
# The whole recipe: bits -> code -> transitions -> inl/dnl/monotonic/missing
# ---------------------------------------------------------------------------

class TestCodeTransitionRecipeEndToEnd:
    """AC #1 — the documented route really does feed all five dc metrics."""

    @staticmethod
    def _ramp_capture(widths_lsb):
        """A converter whose code k has width ``widths_lsb[k]`` LSB.

        Returns ``(v_in, bits)`` for a 3-bit binary converter driven by a
        linear ramp, so the transition levels are known exactly.
        """
        lsb = 0.1
        edges = np.cumsum(np.asarray(widths_lsb, dtype=float)) * lsb
        # one sample per 1/50 LSB, so a transition is read within 0.02 LSB
        v = np.arange(0.0, edges[-1] + lsb, lsb / 50.0)
        code = np.searchsorted(edges, v, side='right').astype(int)
        code = np.clip(code, 0, 7)
        t = np.arange(len(v), dtype=float) * 1e-9
        v_in = Signal('vin', 'voltage', 'time', 'V', t, v)
        bits = [Signal(f'd{b}', 'voltage', 'time', 'V', t,
                       ((code >> b) & 1).astype(float))
                for b in (2, 1, 0)]
        return v_in, bits, lsb

    def test_a_perfect_converter_is_linear_and_monotonic(self):
        v_in, bits, _ = self._ramp_capture([1.0] * 8)
        trans = extract_code_transitions(
            v_in, codes_from_bits(bits, msb_first=True))
        assert trans.xlabel == 'code'
        assert CODE_TRANSITIONS.check(trans) is KindMatch.MATCHES
        assert np.array_equal(trans.x, np.arange(1.0, 8.0))
        assert dnl(trans).max_abs_lsb < 0.05
        assert inl(trans).max_abs_lsb < 0.05
        assert is_monotonic(trans)
        assert len(missing_codes(trans)) == 0

    def test_a_designed_in_mismatch_is_measured(self):
        """Code 3 is half an LSB wide, code 4 is one and a half."""
        widths = [1.0, 1.0, 1.0, 0.5, 1.5, 1.0, 1.0, 1.0]
        v_in, bits, _ = self._ramp_capture(widths)
        trans = extract_code_transitions(
            v_in, codes_from_bits(bits, msb_first=True))
        d = dnl(trans)
        # DNL rows are the widths of codes 1..6 (the span's interior).
        assert d.dnl_lsb[2] == pytest.approx(-0.5, abs=0.05)   # code 3
        assert d.dnl_lsb[3] == pytest.approx(+0.5, abs=0.05)   # code 4
        assert is_monotonic(trans)
        assert len(missing_codes(trans)) == 0

    def test_a_missing_code_is_found(self):
        widths = [1.0, 1.0, 0.0, 2.0, 1.0, 1.0, 1.0, 1.0]
        v_in, bits, _ = self._ramp_capture(widths)
        trans = extract_code_transitions(
            v_in, codes_from_bits(bits, msb_first=True))
        # widths[2] == 0 -> code 2 is the one that never appears
        assert 2 in missing_codes(trans)
        assert not is_monotonic(trans)

    def test_the_thermometer_route_reaches_the_same_metrics(self):
        """The flash case: seven comparators, no binary packing anywhere."""
        lsb = 0.1
        edges = np.arange(1, 8) * lsb
        v = np.arange(0.0, 8 * lsb, lsb / 50.0)
        t = np.arange(len(v), dtype=float) * 1e-9
        v_in = Signal('vin', 'voltage', 'time', 'V', t, v)
        therm = [Signal(f'vcmp{k}', 'voltage', 'time', 'V', t,
                        np.where(v > edges[k - 1], 1.8, 0.0))
                 for k in range(1, 8)]
        code = codes_from_bits(therm, weighting='thermometer', threshold=0.9)
        trans = extract_code_transitions(v_in, code)
        assert np.array_equal(trans.x, np.arange(1.0, 8.0))
        assert dnl(trans).max_abs_lsb < 0.05
        assert is_monotonic(trans)


# ---------------------------------------------------------------------------
# power_sweep
# ---------------------------------------------------------------------------

class TestPowerSweep:

    def test_p1db_accepts_what_power_sweep_builds(self):
        """AC #1 — a hand-built sweep with a compression point p1db finds.

        Gain is 20 dB up to p_in = -10 dBm, then the curve is bent so the
        gain at -5 dBm is exactly 19 dB: the 1 dB compression point is
        -5 dBm by construction.
        """
        p_in = [-30.0, -20.0, -10.0, -5.0, 0.0]
        p_out = [-10.0, 0.0, 10.0, 14.0, 17.0]
        sig = power_sweep(p_in, p_out)
        assert sig.xlabel == 'power'
        assert POWER_SWEEP.check(sig) is KindMatch.MATCHES
        assert p1db(sig) == pytest.approx(-5.0)

    def test_an_uncompressed_sweep_is_nan_not_an_error(self):
        sig = power_sweep([-30.0, -20.0, -10.0], [-10.0, 0.0, 10.0])
        assert np.isnan(p1db(sig))

    def test_fields(self):
        sig = power_sweep([-30.0, -20.0], [-10.0, 0.0], name='cs_amp_sweep')
        assert (sig.name, sig.type, sig.xlabel, sig.ylabel) == \
            ('cs_amp_sweep', 'power', 'power', 'p_out')
        assert sig.name != 'power_sweep'
        assert power_sweep([-30.0, -20.0], [-10.0, 0.0]).name == 'power_sweep'

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match='same length'):
            power_sweep([-30.0, -20.0, -10.0], [-10.0, 0.0])

    def test_single_point(self):
        with pytest.raises(ValueError, match='at least 2 sweep points'):
            power_sweep([-30.0], [-10.0])

    def test_not_one_dimensional(self):
        with pytest.raises(ValueError, match='must be 1-D'):
            power_sweep([[-30.0, -20.0]], [[-10.0, 0.0]])

    def test_descending_sweep_is_refused(self):
        """The one malformed sweep p1db itself cannot detect."""
        p_in = [0.0, -5.0, -10.0, -20.0, -30.0]
        p_out = [17.0, 14.0, 10.0, 0.0, -10.0]
        with pytest.raises(ValueError, match='sorted ascending'):
            power_sweep(p_in, p_out)

    def test_repeated_input_power_is_refused(self):
        with pytest.raises(ValueError, match='sorted ascending'):
            power_sweep([-30.0, -30.0, -20.0], [-10.0, -10.0, 0.0])

    def test_nan_passes_through_to_p1db(self):
        """A NaN is a measurement failure, not a malformed request: the
        assembler builds the Signal and p1db returns nan (decision-1)."""
        sig = power_sweep([-30.0, -20.0, -10.0, 0.0],
                          [-10.0, float('nan'), 10.0, 17.0])
        assert np.isnan(p1db(sig))


# ---------------------------------------------------------------------------
# The schema does not move
# ---------------------------------------------------------------------------

#: sha256 of ``json.dumps(export_metric_schema(), sort_keys=True,
#: separators=(',', ':'))``, re-measured after TASK-223 added
#: ``propagation_delay``'s ``out_edge`` param, TASK-210 removed
#: ``phase_noise_at_offset`` (and the ``phase_noise_sideband`` kind),
#: TASK-288 corrected ``integrated_noise``'s declared unit from the
#: hard-coded ``'V'`` (wrong for a current-noise probe) to the
#: input-relative ``'[y]*sqrt([x])'``, and TASK-326 added
#: ``bandwidth_3dB``'s ``reference={'dc','peak'}`` param (default 'dc',
#: BREAKING), and TASK-364 replaced x_when with crossing (threshold,
#: start/end). Measured, not asserted.
_SCHEMA_BASELINE_SHA256 = (
    # TASK-226.22: required scalar eye threshold; centering default True.
    'f47ca45cb4a0666deb6a955dfe495e17888bfa63308be96cb15e27966e65c040'  # TASK-417 period/c2c jitter params + _pp; TASK-418 long_term_jitter units/reference/phase/threshold/edge/start/end; TASK-412 removed input_referred_offset, kickback, hold_droop; TASK-379 variance; TASK-386 relative_error_norm; TASK-378 interpolate; TASK-399 static_errors wrappers; TASK-385 error_norm mean_abs + start/end; TASK-377 minimum/maximum + start/end; TASK-233 eight wrappers; TASK-323.3 final_value; TASK-364 crossing; TASK-369 evm_percent; TASK-365 rise/fall/slew edge params; TASK-363 propagation_delay; TASK-366 period/frequency; TASK-226.20 crossing_jitter_rms; TASK-367 average_duty, pulse_width, on/off_time; TASK-438 hysteresis n_rising/n_falling/start/end
)


class TestSchemaIsUnchanged:
    """AC — the helpers are NOT analyses, so nothing in the catalog moves.

    ``produced_by`` stays empty for the three remaining unproducible kinds
    and a consumer keeps refusing ``p1db``, ``inl``, ``dnl``,
    ``is_monotonic`` and ``missing_codes``. A helper a Python caller
    invokes is not something a schema consumer can wire a probe into.
    """

    def test_export_is_byte_identical_to_the_pre_helper_baseline(self):
        blob = json.dumps(export_metric_schema(), sort_keys=True,
                          separators=(',', ':')).encode()
        assert hashlib.sha256(blob).hexdigest() == _SCHEMA_BASELINE_SHA256, (
            'export_metric_schema() changed. TASK-211 adds HELPERS, not '
            'analyses and not metrics: signal_kinds[].produced_by must stay '
            'empty for code_transitions, power_sweep '
            'and reference_waveform, or every consumer starts offering '
            'p1db and the transfer-curve family with probes wired into '
            'slots no probe can fill. If a LATER task legitimately changes '
            'the catalog, re-measure this digest in that task and say so '
            'there.'
        )

    @pytest.mark.parametrize('kind_name', [
        'code_transitions', 'power_sweep',
        'reference_waveform',
    ])
    def test_produced_by_stays_empty(self, kind_name):
        meta = export_metric_schema()['_meta']['signal_kinds']
        assert meta[kind_name] == {'produced_by': []}
        assert SIGNAL_KINDS[kind_name].produced_by == ()

    def test_the_helpers_are_not_in_the_catalog(self):
        names = set(export_metric_schema()) - {'_meta'}
        assert 'codes_from_bits' not in names
        assert 'power_sweep' not in names


@pytest.mark.parametrize('decoder', ['as_int', 'binary', 'thermometer'])
@pytest.mark.parametrize('as_bus', [False, True])
@pytest.mark.parametrize('msb_first', [False, True])
@pytest.mark.parametrize('values', [
    [[np.nan, 0, 1, 1, 0, 1, 0, np.nan], [0, 1, 0, 1, 0, 1, 1, 0]],
    [[np.nan, 0, 1, np.nan, 0, 1, 0, 1], [0, np.nan, 0, 1, 0, 1, 1, np.nan]],
    [[np.nan] * 8, [np.nan] * 8],
    [[np.nan, 0, 1, 0.5, 0.6, -1, 1, np.nan]],
])
def test_decoders_preserve_unavailable_samples(decoder, as_bus, msb_first, values):
    import warnings
    from silk.metrics.diagnostics import MetricWarning

    bits = [_bit(f'd[{i + 7}]', row) for i, row in enumerate(values)]
    handle = Signal._make_bus_handle('d', bits, indices=range(7, 7 + len(bits)))
    inputs = handle if as_bus else bits
    missing = np.isnan(values).any(axis=0)
    clean_bits = (np.asarray(values) > 0.5).astype(int)
    weights = (np.ones(len(bits), dtype=int) if decoder == 'thermometer'
               else 2 ** np.arange(len(bits))[::-1 if msb_first else 1])
    expected = np.sum(clean_bits * weights[:, None], axis=0).astype(float)
    expected[missing] = np.nan
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        if decoder == 'as_int':
            result = handle.as_int(msb_first=msb_first)
        else:
            kwargs = {'msb_first': msb_first} if decoder == 'binary' else {}
            result = codes_from_bits(inputs, weighting=decoder, name='decoded', **kwargs)
    np.testing.assert_array_equal(result.y, expected)
    np.testing.assert_array_equal(result.x, T)
    assert result.y.dtype.kind == 'f'
    assert (result.type, result.xlabel, result.ylabel) == ('count', 'time', 'code')
    assert result.name == ('d.as_int' if decoder == 'as_int' else 'decoded')
    assert len(caught) == 1
    warning = caught[0].message
    assert isinstance(warning, MetricWarning)
    assert warning.metric == ('as_int' if decoder == 'as_int' or
                              (decoder == 'binary' and len(bits) > 1) else 'codes_from_bits')
    assert warning.code == 'nan_bits'
    assert warning.details['affected_sample_count'] == int(missing.sum())
    # Decoding never mutates its inputs.
    for s, row in zip(bits, values):
        np.testing.assert_array_equal(s.y, row)


@pytest.mark.parametrize('weighting', ['binary', 'thermometer'])
def test_clean_decoding_does_not_warn(weighting):
    import warnings
    bits = [_bit('d7', [0, .5, .6, 1, -1, 0, 1, 1]),
            _bit('d8', [1, 1, 0, 1, 0, 1, 0, 1])]
    handle = Signal._make_bus_handle('d', bits, indices=(7, 8))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        direct = handle.as_int(msb_first=False)
        result = codes_from_bits(handle, weighting=weighting,
                                 **({'msb_first': False} if weighting == 'binary' else {}))
    np.testing.assert_array_equal(direct.y, [2, 2, 1, 3, 0, 2, 1, 3])
    np.testing.assert_array_equal(result.y, direct.y if weighting == 'binary'
                                  else [1, 1, 1, 2, 0, 1, 1, 2])
    assert not caught


@pytest.mark.parametrize('weighting', ['binary', 'thermometer'])
@pytest.mark.parametrize('as_bus', [False, True])
@pytest.mark.parametrize('other_x', [
    [9e-9, 10e-9, 11e-9], [0., 2e-9, 4e-9],
    [0., np.nextafter(1e-9, 2e-9), 2e-9], [0.],
])
def test_decode_requires_exact_coordinates(weighting, as_bus, other_x):
    bits = [_bit('d8', [0, 1, 1], [0., 1e-9, 2e-9]),
            _bit('d9', np.ones(len(other_x)), other_x)]
    inputs = Signal._make_bus_handle('d', tuple(bits), indices=(8, 9)) if as_bus else bits
    kwargs = {'msb_first': False} if weighting == 'binary' else {}
    with pytest.raises(ValueError) as exc:
        codes_from_bits(inputs, weighting=weighting, **kwargs)
    message = str(exc.value)
    assert 'codes_from_bits' in message and 'share one x-axis' in message
    assert 'd8' in message and 'd9' in message
    assert '3' in message and str(len(other_x)) in message
    assert 'silk.metrics.sample' in message
    assert len(message) < 700
    if len(other_x) == 3:
        index = next(i for i, (a, b) in enumerate(zip(bits[0].x, other_x)) if a != b)
        assert f'index {index}' in message
        assert repr(float(other_x[index])) in message
    else:
        assert 'shape' in message and 'index' not in message


@pytest.mark.parametrize('weighting,expected', [('binary', [2., 1., 3.]),
                                               ('thermometer', [1., 1., 2.])])
@pytest.mark.parametrize('as_bus', [False, True])
def test_decode_accepts_copied_coordinates(weighting, expected, as_bus):
    bits = [_bit('d8', [0., .8, 1.], np.array([2, 1, 0])),
            _bit('d9', [1., .7, 1.], np.array([2., 1., 0.]))]
    inputs = Signal._make_bus_handle('d', tuple(bits), indices=(8, 9)) if as_bus else bits
    kwargs = {'msb_first': False} if weighting == 'binary' else {}
    result = codes_from_bits(inputs, weighting=weighting, threshold=.75, **kwargs)
    np.testing.assert_array_equal(result.y, expected)
    np.testing.assert_array_equal(result.x, [2., 1., 0.])


@pytest.mark.parametrize('weighting', ['binary', 'thermometer'])
def test_decode_does_not_round_coordinates_before_comparison(weighting):
    bits = [_bit('d0', [0., 1.], np.array([2**53, 2**53 + 2])),
            _bit('d1', [1., 0.], np.array([2**53 + 1, 2**53 + 2]))]
    kwargs = {'msb_first': False} if weighting == 'binary' else {}
    with pytest.raises(ValueError, match='index 0'):
        codes_from_bits(bits, weighting=weighting, **kwargs)
