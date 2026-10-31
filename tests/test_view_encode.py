"""Encoding round-trip tests for silk.signals.view.encode (design spec §9.1).

Pure numpy: base64 <-> array round trips, the float32 range escape hatch, the
decimation pyramid's pinned worked examples, and the envelope-correctness
properties the whole pyramid exists for.
"""
import base64
import math
from pathlib import Path

import numpy as np
import pytest

from silk.signals.rawfile import read_rawfile
from silk.signals.view.encode import (
    DECIMATE_BASE, DECIMATE_MIN_BUCKET, DECIMATE_MIN_POINTS,
    b64_f4, b64_f8, count_nonfinite, decide_y_dtype, decimation_strides,
    minmax_level, unit_for_type,
)

FIXTURE_DIR = Path(__file__).parent / 'fixtures' / 'rawfiles'


def _decode_f4(s):
    return np.frombuffer(base64.b64decode(s), dtype='<f4')


def _decode_f8(s):
    return np.frombuffer(base64.b64decode(s), dtype='<f8')


class TestB64RoundTrip:
    @pytest.mark.parametrize('arr', [
        np.zeros(5),
        np.array([-1.0, -2.5, 0.0, 3.5]),
        np.array([1e-15, -1e-15, 2e-15]),          # current-scale
        np.array([1e9, 2.4e9, -1e9]),               # frequency-scale
        np.linspace(0, 1, 1_000_000),
    ])
    def test_f4_exact_after_cast(self, arr):
        encoded = b64_f4(arr)
        decoded = _decode_f4(encoded)
        assert np.array_equal(decoded, arr.astype('<f4'))

    def test_f8_exact(self):
        arr = np.linspace(0, 1e-3, 1_000_001)  # includes a 1ns step
        decoded = _decode_f8(b64_f8(arr))
        assert np.array_equal(decoded, arr.astype('<f8'))

    def test_x_precision_survives_1ns_in_1ms(self):
        # A 1ns edge inside a 1ms span is 1e-6 of full scale; float64 must
        # resolve it to better than 1ps (the reason X is never float32).
        span = 1e-3
        edge = 5e-4
        x = np.array([0.0, edge, edge + 1e-9, span])
        decoded = _decode_f8(b64_f8(x))
        assert abs(decoded[2] - decoded[1] - 1e-9) < 1e-12

    def test_empty_array_encodes_to_empty_string(self):
        assert b64_f4(np.empty(0)) == ''
        assert b64_f8(np.empty(0)) == ''


class TestFloat32EscapeHatch:
    def test_ordinary_data_is_f32(self):
        assert decide_y_dtype(np.array([1e-15, 1.0, 1e9])) == 'f32'

    def test_overflow_promotes_to_f64(self):
        assert decide_y_dtype(np.array([3.1e38])) == 'f64'
        assert decide_y_dtype(np.array([3.0e38])) == 'f32'  # boundary itself is safe
        assert decide_y_dtype(np.array([3.0e38 + 1e30])) == 'f64'

    def test_underflow_promotes_to_f64(self):
        assert decide_y_dtype(np.array([5e-38])) == 'f64'
        assert decide_y_dtype(np.array([0.0, 5e-38])) == 'f64'

    def test_zero_never_promotes(self):
        assert decide_y_dtype(np.array([0.0, 0.0, 0.0])) == 'f32'

    def test_nonfinite_only_does_not_promote(self):
        assert decide_y_dtype(np.array([np.nan, np.inf, -np.inf])) == 'f32'

    def test_nan_inf_survive_round_trip(self):
        arr = np.array([1.0, np.nan, np.inf, -np.inf, -2.0])
        decoded = _decode_f4(b64_f4(arr))
        assert np.array_equal(decoded, arr.astype('<f4'), equal_nan=True)


class TestNonfiniteCount:
    def test_counts_nan_and_inf(self):
        assert count_nonfinite(np.array([1.0, np.nan, np.inf, -np.inf, 2.0])) == 3

    def test_zero_on_clean_array(self):
        assert count_nonfinite(np.array([1.0, 2.0, 3.0])) == 0

    def test_empty(self):
        assert count_nonfinite(np.empty(0)) == 0


class TestDecimationPyramidWorkedExamples:
    """Pinned exactly (spec §4.8, §9.1) — Lane B's samples must match these."""

    def test_n10_one_level(self):
        assert decimation_strides(10) == [1]

    def test_n8000_one_level(self):
        assert decimation_strides(8000) == [1]

    def test_n8001_two_levels(self):
        assert decimation_strides(8001) == [1, 8]

    def test_n16000_two_levels(self):
        assert decimation_strides(16000) == [1, 8]

    def test_n1000000_four_levels(self):
        strides = decimation_strides(1_000_000)
        assert strides == [1, 8, 64, 512]
        coarsest_buckets = math.ceil(1_000_000 / 512)
        assert coarsest_buckets == 1954
        # up to 2 points per bucket
        assert 2 * coarsest_buckets == 3908

    def test_max_levels_cap_honored(self):
        # With a lowered cap, a huge N still stops at max_levels entries.
        strides = decimation_strides(10_000_000, max_levels=3)
        assert len(strides) <= 3

    def test_defaults_match_module_constants(self):
        assert DECIMATE_MIN_POINTS == 8000
        assert DECIMATE_BASE == 8
        assert DECIMATE_MIN_BUCKET == 1000


class TestMinmaxLevel:
    def test_envelope_reproduces_exact_spike(self):
        n = 20000
        x = np.arange(n, dtype=np.float64)
        y = np.zeros(n, dtype=np.float64)
        spike_idx = 12345
        y[spike_idx] = 999.0
        strides = decimation_strides(n)
        assert len(strides) > 1
        for stride in strides[1:]:
            xs, ys = minmax_level(x, y, stride, n)
            assert 999.0 in ys

    def test_x_monotonic_non_decreasing_including_reversed_argmin_argmax(self):
        n = 24
        x = np.arange(n, dtype=np.float64)
        # Build y so some buckets have argmax before argmin (reversed order).
        y = np.zeros(n, dtype=np.float64)
        y[1] = 10.0   # max first in bucket [0,8)
        y[2] = -10.0  # min second in bucket [0,8)
        xs, ys = minmax_level(x, y, 8, n)
        assert np.all(np.diff(xs) >= 0)

    def test_all_nan_bucket_emits_single_pair(self):
        n = 16
        x = np.arange(n, dtype=np.float64)
        y = np.full(n, np.nan)
        xs, ys = minmax_level(x, y, 8, n)
        assert len(xs) == 2  # one point per bucket, not two
        assert np.all(np.isnan(ys))

    def test_mixed_nan_bucket_ignores_nan_in_minmax(self):
        n = 8
        x = np.arange(n, dtype=np.float64)
        y = np.array([np.nan, 1.0, np.nan, 5.0, np.nan, 2.0, np.nan, np.nan])
        xs, ys = minmax_level(x, y, 8, n)
        assert len(xs) == 2
        assert set(ys.tolist()) == {1.0, 5.0}

    def test_point_count_bound(self):
        n = 1000
        stride = 8
        x = np.arange(n, dtype=np.float64)
        y = np.random.RandomState(0).randn(n)
        xs, ys = minmax_level(x, y, stride, n)
        assert len(xs) <= 2 * math.ceil(n / stride)


# ---------------------------------------------------------------------------
# Pinned reference implementation — DO NOT "clean up" or optimize this. It is
# byte-for-byte the per-bucket Python loop that shipped as `minmax_level`
# before it was vectorized (encode.py, pre-2026-08-12). Its only job is to be
# an obviously-correct-by-inspection oracle that `minmax_level` is checked
# against below. If `minmax_level`'s semantics ever need to change,
# deliberately change this reference too — a silent divergence here defeats
# the whole point of the test.
# ---------------------------------------------------------------------------
def _reference_minmax_level(x, y, stride, n):
    nbuckets = math.ceil(n / stride)
    xs = []
    ys = []
    for b in range(nbuckets):
        lo = b * stride
        hi = min(lo + stride, n)
        seg = y[lo:hi]
        if np.all(np.isnan(seg)):
            xs.append(x[lo])
            ys.append(np.nan)
            continue
        imin = lo + int(np.nanargmin(seg))
        imax = lo + int(np.nanargmax(seg))
        if imin <= imax:
            xs.append(x[imin]); ys.append(y[imin])
            xs.append(x[imax]); ys.append(y[imax])
        else:
            xs.append(x[imax]); ys.append(y[imax])
            xs.append(x[imin]); ys.append(y[imin])
    return np.asarray(xs, dtype='<f8'), np.asarray(ys, dtype=np.asarray(y).dtype)


def _assert_minmax_matches_reference(x, y, stride, n):
    ref_x, ref_y = _reference_minmax_level(x, y, stride, n)
    got_x, got_y = minmax_level(x, y, stride, n)
    assert got_x.dtype == ref_x.dtype == np.dtype('<f8')
    assert got_y.dtype == ref_y.dtype
    assert len(got_x) == len(ref_x) == len(got_y) == len(ref_y)
    assert np.array_equal(got_x, ref_x, equal_nan=True)
    assert np.array_equal(got_y, ref_y, equal_nan=True)


class TestMinmaxLevelDifferential:
    """`minmax_level` (vectorized) vs `_reference_minmax_level` (the pinned
    per-bucket loop) — exact equality of X, Y, dtypes, and lengths.

    This is the gold-standard check for the numpy vectorization: every trap
    (ragged tail, all-NaN buckets needing compaction, +-Inf as legal data,
    argmin/argmax tie order) is exercised here against an oracle that is
    obviously correct by construction, not just against hand-picked
    properties.
    """

    RANDOM_WALK_CASES = [
        (10, 8), (10, 3), (999, 8), (999, 64), (1000, 8), (1000, 64),
        (1001, 8), (1001, 64), (8001, 8), (8001, 64), (8001, 512),
        (16000, 8), (16000, 64), (16000, 512),
    ]

    @pytest.mark.parametrize('n,stride', RANDOM_WALK_CASES)
    def test_random_walk(self, n, stride):
        rng = np.random.RandomState(n * 1000 + stride)
        x = np.arange(n, dtype=np.float64)
        y = np.cumsum(rng.randn(n))
        _assert_minmax_matches_reference(x, y, stride, n)

    @pytest.mark.parametrize('n,stride', [(999, 8), (8001, 64), (16000, 512)])
    def test_random_walk_float32_dtype_preserved(self, n, stride):
        rng = np.random.RandomState(n + stride)
        x = np.arange(n, dtype=np.float64)
        y = np.cumsum(rng.randn(n)).astype(np.float32)
        _assert_minmax_matches_reference(x, y, stride, n)

    @pytest.mark.parametrize('n,stride', [(10, 3), (1000, 8), (8001, 64)])
    def test_constant_array(self, n, stride):
        x = np.arange(n, dtype=np.float64)
        y = np.full(n, 3.5)
        _assert_minmax_matches_reference(x, y, stride, n)

    @pytest.mark.parametrize('n,stride', [(10, 3), (1000, 8), (8001, 64)])
    def test_single_sample_spike(self, n, stride):
        x = np.arange(n, dtype=np.float64)
        y = np.zeros(n)
        y[n // 2] = 12345.0
        _assert_minmax_matches_reference(x, y, stride, n)
        y2 = np.zeros(n)
        y2[-1] = -999.0  # spike inside the ragged tail bucket itself
        _assert_minmax_matches_reference(x, y2, stride, n)

    @pytest.mark.parametrize('n,stride', [(64, 8), (999, 8), (8001, 64)])
    def test_nan_scattered(self, n, stride):
        rng = np.random.RandomState(n + stride + 1)
        x = np.arange(n, dtype=np.float64)
        y = rng.randn(n)
        nan_idx = rng.choice(n, size=max(1, n // 5), replace=False)
        y[nan_idx] = np.nan
        _assert_minmax_matches_reference(x, y, stride, n)

    def test_nan_whole_bucket_in_the_middle(self):
        n, stride = 32, 8
        x = np.arange(n, dtype=np.float64)
        y = np.arange(n, dtype=np.float64)
        y[8:16] = np.nan  # bucket 1 entirely NaN, buckets 0/2/3 real
        _assert_minmax_matches_reference(x, y, stride, n)

    def test_nan_whole_bucket_in_ragged_tail(self):
        n, stride = 20, 8  # tail bucket [16, 20) is short
        x = np.arange(n, dtype=np.float64)
        y = np.arange(n, dtype=np.float64)
        y[16:20] = np.nan
        _assert_minmax_matches_reference(x, y, stride, n)

    @pytest.mark.parametrize('n,stride', [(16, 8), (17, 8), (8001, 64)])
    def test_all_nan(self, n, stride):
        x = np.arange(n, dtype=np.float64)
        y = np.full(n, np.nan)
        _assert_minmax_matches_reference(x, y, stride, n)

    @pytest.mark.parametrize('n,stride', [(20, 8), (999, 8), (8001, 64)])
    def test_pos_and_neg_inf_are_legal_data(self, n, stride):
        rng = np.random.RandomState(n * 7 + stride)
        x = np.arange(n, dtype=np.float64)
        y = rng.randn(n)
        # Scatter some +-Inf, including right at the end (inside whatever
        # the ragged tail bucket is) so a NaN-vs-Inf pad sentinel mixup would
        # bite here if it existed.
        y[0] = np.inf
        y[-1] = -np.inf
        if n > 5:
            y[n // 2] = np.inf
            y[n // 3] = -np.inf
        _assert_minmax_matches_reference(x, y, stride, n)

    @pytest.mark.parametrize('n,stride', [(20, 8), (999, 8), (8001, 64)])
    def test_inf_and_nan_mixed(self, n, stride):
        rng = np.random.RandomState(n * 13 + stride)
        x = np.arange(n, dtype=np.float64)
        y = rng.randn(n)
        y[::7] = np.nan
        y[::11] = np.inf
        y[::13] = -np.inf
        _assert_minmax_matches_reference(x, y, stride, n)

    @pytest.mark.parametrize('n', [10, 999, 1000, 1001, 8001, 16000])
    @pytest.mark.parametrize('stride', [8, 64, 512])
    def test_ragged_tail_grid_n_by_stride(self, n, stride):
        # Full grid required by the task: every (N, stride) combination,
        # multiples of stride and not, well below and well above stride.
        rng = np.random.RandomState(n * 100003 + stride)
        x = np.arange(n, dtype=np.float64)
        y = rng.randn(n)
        # Sprinkle a few NaN/Inf so the tail-bucket edge cases are covered
        # for every grid point, not just the dedicated NaN/Inf tests.
        if n >= 4:
            y[rng.choice(n, size=max(1, n // 10), replace=False)] = np.nan
            y[-1] = np.inf if (n + stride) % 2 else -np.inf
        _assert_minmax_matches_reference(x, y, stride, n)

    def test_argmin_argmax_tie_order_reversed(self):
        # Regression for the exact scenario in TestMinmaxLevel above: max
        # appears before min inside a bucket, so the reference's "else"
        # branch (swap) is exercised, not just the common case.
        n, stride = 24, 8
        x = np.arange(n, dtype=np.float64)
        y = np.zeros(n)
        y[1] = 10.0
        y[2] = -10.0
        _assert_minmax_matches_reference(x, y, stride, n)


def _fixture_names():
    names = sorted(p.name for p in FIXTURE_DIR.glob('*.raw'))
    assert len(names) >= 3, 'expected at least 3 rawfile fixtures'
    return names


class TestMinmaxLevelDifferentialFixtureCorpus:
    """Every real signal (X and Y vectors, real or complex) from every
    rawfile fixture, run through `minmax_level` at several strides
    (including ragged ones) and checked against the pinned reference.
    Complex vectors are reduced to a real array (magnitude) first —
    `minmax_level` itself only ever sees real float arrays in production
    (mag/phase expansion happens upstream in emit.py); this just gives more
    varied real-world data shapes to stress the vectorization with.
    """

    @pytest.mark.parametrize('fixture_name', _fixture_names())
    def test_every_signal_in_fixture(self, fixture_name):
        raw = read_rawfile(FIXTURE_DIR / fixture_name)
        for plot in raw.get_plots():
            n = plot.npoints
            if n == 0 or not plot.vectors:
                continue
            x_vec = plot.vectors[0]
            x_data = x_vec.data
            x = np.abs(x_data).astype(np.float64) if np.iscomplexobj(x_data) \
                else x_data.astype(np.float64)
            for vec in plot.vectors:
                y_data = vec.data
                y = np.abs(y_data).astype(np.float64) if np.iscomplexobj(y_data) \
                    else y_data.astype(np.float64)
                if len(y) == 0:
                    continue  # dims=0 vector (TASK-440): nothing to decimate
                for stride in (1, 2, 3, 5, 8):
                    if stride > n:
                        continue
                    _assert_minmax_matches_reference(x, y, stride, n)


class TestUnitsTable:
    @pytest.mark.parametrize('type_str,expected', [
        ('voltage', 'V'), ('current', 'A'), ('power', 'W'), ('charge', 'C'),
        ('capacitance', 'F'), ('notype', ''), ('', ''), ('time', 's'),
        ('frequency', 'Hz'), ('temp-sweep', '°C'), ('res-sweep', 'Ω'),
        ('impedance', 'Ω'), ('admittance', 'S'),
    ])
    def test_exact_table(self, type_str, expected):
        assert unit_for_type(type_str) == expected

    def test_case_insensitive(self):
        assert unit_for_type('VOLTAGE') == 'V'
        assert unit_for_type('Frequency') == 'Hz'

    def test_unknown_type_is_empty(self):
        assert unit_for_type('something-unheard-of') == ''
        assert unit_for_type(None) == ''
