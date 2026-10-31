"""Conformance suite for silk's nutmeg rawfile reader (`silk/signals/rawfile.py`).

Every expectation here derives from:

* the **published nutmeg/rawfile format** description (header keywords, the
  ``Flags:`` tokens, the ``Variables:`` line layout, the binary layout, and how
  consecutive plots are concatenated in one file);
* **real rawfiles** generated locally by ngspice-42 and Xyce 7.10 — see
  ``tests/fixtures/rawfiles/gen_{a,b,c,d}.py``, which record for every fixture
  whether it is REAL, REAL+DERIVED, or HAND-CRAFTED, and how it was produced;
* **ASCII-vs-binary cross-validation of the same simulator run** — the ASCII
  encoding renders every vector as full-precision decimal text, so it is a
  self-describing, human-readable oracle for the binary layout; and
* **circuit physics** — a 1k/1k divider outputs vin/2, a `.dc vin 0 1 0.1`
  sweep has exactly 11 points spanning 0..1, an RC lowpass rolls off at
  1/(2*pi*R*C).

Tests reach the reader only through :func:`tests.rawreader.read_raw`, so the
suite is written against the format, not against any one implementation.

Contract pinned here (what silk actually consumes):

* ``nvars`` / ``npoints`` / ``vectors[i].name`` / ``.type`` / ``.data``;
* ``current_plot.plotname`` — load-bearing for
  ``silk/simulation/backend.py:_has_data_rows``, which rejects a rawfile whose
  last/only plot is ``constants`` (the aborted-run false-green guard);
* **vector 0 is the independent axis** for every analysis — never detected by
  name, since a `.dc` sweep axis is ``v(v-sweep)``/``temp-sweep`` on ngspice and
  ``sweep`` on Xyce — so vector order is preserved exactly as written;
* ``plots`` — every plot in file order, ``current_plot`` being the last.
"""

import cmath
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from tests.rawreader import read_raw
from silk.simulation.backend import _has_data_rows

FIXTURES = Path(__file__).parent / "fixtures" / "rawfiles"


@pytest.fixture(scope="module", autouse=True, params=[False, True],
                ids=["eager", "lazy"])
def reader_mode(request):
    """Run the complete format corpus through both public reader modes."""
    from unittest.mock import patch
    from silk.signals import rawfile
    original = rawfile.read_rawfile

    def read(path, **kwargs):
        if request.param:
            kwargs.setdefault("lazy", True)
        return original(path, **kwargs)

    with patch.object(rawfile, "read_rawfile", read):
        yield request.param


# ==========================================================================
# ngspice binary + ASCII: transient (real) and AC (complex)
# (was tests/test_rawfile_reader_a.py)
# ==========================================================================

class TestTransientBinaryReal:
    """a_tran_bin.raw: single-plot transient run, ``Flags: real``.

    Netlist (see gen_a.py TRAN_NETLIST): V1 is a PULSE source with a 1us
    rise time and no falling edge inside the 0..5us simulation window,
    driving a 1k/1n RC low-pass. ``.tran 1u 5u 0 1u`` pins ngspice's
    internal step ceiling (Tmax) to the print step, so the run is a fixed,
    reproducible 22 points: 13 points where ngspice's initial step-doubling
    ramps up to Tmax (t=0 .. 1us, matching the source's rise time), then 9
    more at the 1us ceiling out to t=5us.
    """

    @pytest.fixture(scope="class")
    def result(self):
        return read_raw(FIXTURES / "a_tran_bin.raw")

    def test_top_level_counts(self, result):
        assert result.nvars == 4
        assert result.npoints == 22
        assert result.real is True
        # This fixture's Flags line is bare ("Flags: real", no "padded" or
        # "unpadded" token -- confirmed by inspecting the fixture bytes
        # directly). The published format does not define a default for the
        # undeclared case, so the reader reports None rather than assuming
        # one; nothing in silk consumes the value.
        assert result.padded is None

    def test_vector_order_and_names(self, result):
        # vectors[0] is the independent axis -- silk treats it as such for
        # every analysis type, so order here is load-bearing.
        names = [v.name for v in result.vectors]
        assert names == ["time", "v(in)", "v(out)", "i(v1)"]

    def test_vector_types(self, result):
        types = [v.type for v in result.vectors]
        assert types == ["time", "voltage", "voltage", "current"]

    def test_vector_dtype_and_shape(self, result):
        for v in result.vectors:
            assert v.data.dtype == np.float64
            assert v.data.shape == (22,)

    def test_t0_is_exact_zero(self, result):
        time, vin, vout, iv1 = (v.data for v in result.vectors)
        # t=0, before the source starts to rise: everything is exactly 0.
        assert time[0] == 0.0
        assert vin[0] == 0.0
        assert vout[0] == 0.0
        assert iv1[0] == 0.0

    def test_exact_values_at_rise_time_breakpoint(self, result):
        # index 13 is ngspice's forced breakpoint at t=1us, exactly where
        # the PULSE source finishes rising to 1V (values from the ASCII
        # sibling a_tran_ascii.raw, same run).
        time, vin, vout, iv1 = (v.data for v in result.vectors)
        assert time[13] == pytest.approx(1.000000000000000e-06, rel=1e-12)
        assert vin[13] == pytest.approx(1.0, rel=1e-12)
        assert vout[13] == pytest.approx(3.654990082396880e-01, rel=1e-12)
        assert iv1[13] == pytest.approx(-6.345009917603121e-04, rel=1e-12)

    def test_exact_values_at_final_point(self, result):
        time, vin, vout, iv1 = (v.data for v in result.vectors)
        assert time[-1] == pytest.approx(5.000000000000000e-06, rel=1e-12)
        assert vin[-1] == pytest.approx(1.0, rel=1e-12)
        assert vout[-1] == pytest.approx(9.914561752448197e-01, rel=1e-12)
        assert iv1[-1] == pytest.approx(-8.543824755180346e-06, rel=1e-12)

    def test_vout_charges_monotonically_toward_vin(self, result):
        # Physical sanity check independent of the exact literals above: an
        # RC low-pass step response is monotonically increasing toward the
        # final (source) value, never overshooting.
        vout = result.vectors[2].data
        assert np.all(np.diff(vout) >= 0.0)
        assert vout[-1] < 1.0

    def test_current_plot_metadata(self, result):
        cp = result.current_plot
        assert cp.title == "* rc charging transient - real vectors, binary rawfile"
        assert cp.plotname == "Transient Analysis"
        # cp.date is ngspice's wall-clock timestamp at generation time --
        # frozen into the committed fixture, but not meaningful to pin as a
        # literal (would only churn if the fixture is regenerated).
        assert isinstance(cp.date, str) and cp.date
        assert len(cp.dimensions) == 0  # not a nested/multi-dim sweep

    def test_current_plot_vectors_match_top_level(self, result):
        names = [v.name for v in result.current_plot.vectors]
        assert names == ["time", "v(in)", "v(out)", "i(v1)"]

    def test_plots_list_contains_this_single_plot(self, result):
        assert len(result.plots) == 1
        assert result.plots[-1] is result.current_plot
        assert result.current_plot.plotname == "Transient Analysis"


class TestACBinaryComplex:
    """a_ac_bin.raw: single-plot AC run, ``Flags: complex``.

    Netlist (see gen_a.py AC_NETLIST): V1 AC 1 driving the same 1k/1n RC
    low-pass, swept linearly (``.ac lin 5 1k 5k``) so the 5 frequency points
    are round numbers: 1k, 2k, 3k, 4k, 5k Hz.
    """

    @pytest.fixture(scope="class")
    def result(self):
        return read_raw(FIXTURES / "a_ac_bin.raw")

    def test_top_level_counts(self, result):
        assert result.nvars == 4
        assert result.npoints == 5
        assert result.real is False
        # Same bare-Flags-line reasoning as the transient fixture above:
        # "Flags: complex" with no padded/unpadded token present -> the
        # padding state is undeclared, so the reader reports None.
        assert result.padded is None

    def test_vector_order_and_names(self, result):
        names = [v.name for v in result.vectors]
        assert names == ["frequency", "v(in)", "v(out)", "i(v1)"]

    def test_vector_types(self, result):
        types = [v.type for v in result.vectors]
        assert types == ["frequency", "voltage", "voltage", "current"]

    def test_frequency_axis_is_real_valued(self, result):
        # nutmeg's binary "complex" plot layout stores every vector,
        # including the independent frequency axis, as an interleaved
        # (real, imag) double pair -- but frequency's imaginary half is
        # always exactly 0 (confirmed against the ASCII sibling, which
        # prints "1.000000...e+03,0.000000...e+00" for every frequency
        # point). The reader must hand back a real float64 array for it,
        # not a complex one, matching what every other vector-consuming
        # part of silk expects of an independent axis.
        freq = result.vectors[0]
        assert freq.data.dtype == np.float64
        assert freq.data.shape == (5,)

    def test_dependent_vectors_are_complex128(self, result):
        for v in result.vectors[1:]:
            assert v.data.dtype == np.complex128
            assert v.data.shape == (5,)

    def test_frequency_values(self, result):
        np.testing.assert_allclose(
            result.vectors[0].data,
            [1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
            rtol=1e-12,
        )

    def test_vin_is_unity_with_zero_phase(self, result):
        vin = result.vectors[1].data
        np.testing.assert_allclose(vin.real, [1.0] * 5, rtol=1e-12)
        np.testing.assert_allclose(vin.imag, [0.0] * 5, atol=1e-15)

    def test_vout_exact_complex_values(self, result):
        # Literal values transcribed from a_ac_ascii.raw (same run, ASCII).
        vout = result.vectors[2].data
        expected = np.array(
            [
                9.999605231408795e-01 - 6.282937266758386e-03j,
                9.998421112623725e-01 - 1.256438652596631e-02j,
                9.996448204389040e-01 - 1.884286094453970e-02j,
                9.993687440541039e-01 - 2.511687603638105e-02j,
                9.990140126903603e-01 - 3.138495083101297e-02j,
            ]
        )
        np.testing.assert_allclose(vout, expected, rtol=1e-12)

    def test_vout_has_nonzero_imaginary_parts(self, result):
        # This IS a complex signal, not a real one padded with zero
        # imaginary parts: an RC low-pass has nonzero phase everywhere
        # except at DC.
        vout = result.vectors[2].data
        assert np.all(vout.imag != 0.0)

    def test_vout_magnitude_rolls_off_and_phase_lags_with_frequency(self, result):
        # Physical sanity check: |Vout| decreases monotonically and the
        # (negative) phase grows in magnitude monotonically as frequency
        # increases, for a first-order RC low-pass below its corner.
        vout = result.vectors[2].data
        mag = np.abs(vout)
        phase = np.angle(vout)
        assert np.all(np.diff(mag) < 0.0)
        assert np.all(np.diff(phase) < 0.0)

    def test_current_plot_metadata(self, result):
        cp = result.current_plot
        assert cp.title == "* rc low-pass ac sweep - complex vectors, binary rawfile"
        assert cp.plotname == "AC Analysis"
        assert isinstance(cp.date, str) and cp.date
        assert len(cp.dimensions) == 0  # not a nested/multi-dim sweep

    def test_current_plot_vectors_match_top_level(self, result):
        names = [v.name for v in result.current_plot.vectors]
        assert names == ["frequency", "v(in)", "v(out)", "i(v1)"]

    def test_plots_list_contains_this_single_plot(self, result):
        assert len(result.plots) == 1
        assert result.plots[-1] is result.current_plot


class TestBinaryAsciiRoundTrip:
    """The SAME ngspice run, written once binary and once ASCII (via
    ``set filetype=ascii`` in the same ``.control`` block), must parse to
    numerically equal vectors. Slice A owns only this equivalence check;
    ASCII feature depth belongs to the other slice(s).
    """

    @pytest.mark.parametrize(
        "bin_name,ascii_name",
        [
            ("a_tran_bin.raw", "a_tran_ascii.raw"),
            ("a_ac_bin.raw", "a_ac_ascii.raw"),
        ],
    )
    def test_binary_and_ascii_agree(self, bin_name, ascii_name):
        rb = read_raw(FIXTURES / bin_name)
        ra = read_raw(FIXTURES / ascii_name)

        assert rb.nvars == ra.nvars
        assert rb.npoints == ra.npoints
        assert rb.real == ra.real

        names_b = [v.name for v in rb.vectors]
        names_a = [v.name for v in ra.vectors]
        assert names_b == names_a

        for vb, va in zip(rb.vectors, ra.vectors):
            assert vb.type == va.type
            assert vb.data.dtype == va.data.dtype
            # ASCII text carries ~15 significant digits, so the round-trip
            # agrees to a tight relative tolerance but not bit-for-bit.
            np.testing.assert_allclose(vb.data, va.data, rtol=1e-13, atol=1e-20)


# ==========================================================================
# ASCII depth: Values: layout, Offset:, per-variable attrs, Flags: combinations
# (was tests/test_rawfile_reader_b.py)
# ==========================================================================

# ---------------------------------------------------------------------------
# ASCII real Values: block (b_tran_real.raw - REAL ngspice transient)
# ---------------------------------------------------------------------------

class TestAsciiRealValues:
    """ASCII ``Values:`` parsing for a real (non-complex) plot.

    b_tran_real.raw: RC transient, ``Flags: real``, no Offset/padded keyword
    present in the header (also exercises the undeclared-padding case, see
    TestPaddedFlag). Values below are transcribed directly from the fixture
    file, which is REAL ngspice-42 output.
    """

    def test_header_fields(self):
        r = read_raw(FIXTURES / 'b_tran_real.raw')
        assert r.nvars == 4
        assert r.npoints == 62
        assert r.real is True

    def test_vector_order_and_identity(self):
        r = read_raw(FIXTURES / 'b_tran_real.raw')
        names = [v.name for v in r.vectors]
        types = [v.type for v in r.vectors]
        assert names == ['time', 'v(1)', 'v(2)', 'i(v1)']
        assert types == ['time', 'voltage', 'voltage', 'current']

    def test_vector_dtype_and_shape(self):
        r = read_raw(FIXTURES / 'b_tran_real.raw')
        for v in r.vectors:
            assert v.data.dtype == np.float64
            assert v.data.shape == (62,)

    def test_scale_values(self):
        """vectors[0] (time) is the independent axis silk treats as vector 0."""
        r = read_raw(FIXTURES / 'b_tran_real.raw')
        time = r.vectors[0].data
        assert_allclose(time[:4], [0.0, 4.0e-13, 8.0e-13, 1.6e-12])
        assert time[-1] == pytest.approx(4e-09)

    def test_dependent_vector_values(self):
        r = read_raw(FIXTURES / 'b_tran_real.raw')
        v1 = r.vectors[1].data
        assert_allclose(v1[:4], [0.0, 0.0004, 0.0008, 0.0016])
        assert v1[-1] == pytest.approx(1.0)

    def test_single_plot_file_current_plot_is_the_only_plot(self):
        """Multi-plot contract established by the lead: ``plots`` is the
        list of ALL plots in file order, ``current_plot`` is the LAST one.
        For a single-plot ASCII file that means ``plots`` has exactly one
        entry and ``current_plot`` IS it - grounded against the file's own
        ``Plotname:`` header line, not a guessed attribute.
        """
        r = read_raw(FIXTURES / 'b_tran_real.raw')
        assert len(r.plots) == 1
        assert r.current_plot is r.plots[0]
        assert r.current_plot.plotname == 'Transient Analysis'


# ---------------------------------------------------------------------------
# ASCII complex Values: block (b_ac_complex.raw - REAL ngspice AC sweep)
# ---------------------------------------------------------------------------

class TestAsciiComplexValues:
    """ASCII ``Values:`` parsing for a complex plot (``Flags: complex``).

    b_ac_complex.raw: RC lowpass, decade AC sweep (R=1k, C=1n, so the pole is
    at 1/(2*pi*R*C) ~= 159.15 kHz), 4 points. The frequency scale carries a
    real per-variable trailing attribute (``frequency grid=3``, added by
    ngspice specifically for decade/octave sweeps) - covered in
    TestTrailingAttributes; this class covers the complex Values: layout
    itself ("real,imag" per point, comma-separated, no surrounding spaces).
    """

    def test_header_fields(self):
        r = read_raw(FIXTURES / 'b_ac_complex.raw')
        assert r.nvars == 4
        assert r.npoints == 4
        assert r.real is False

    def test_current_plot_plotname(self):
        r = read_raw(FIXTURES / 'b_ac_complex.raw')
        assert r.current_plot.plotname == 'AC Analysis'

    def test_vector_order_and_identity(self):
        r = read_raw(FIXTURES / 'b_ac_complex.raw')
        names = [v.name for v in r.vectors]
        types = [v.type for v in r.vectors]
        assert names == ['frequency', 'v(1)', 'v(2)', 'i(v1)']
        assert types == ['frequency', 'voltage', 'voltage', 'current']

    def test_scale_is_real_valued_even_though_plot_is_complex(self):
        """Only the dependent vectors are complex; the frequency scale
        (vectors[0]) stays float64. Not a quirk of this one file - the same
        holds for a hand-crafted `Flags: complex` fixture
        (b_flags_complex_padded.raw, see TestPaddedFlag), and follows from
        the format itself: a complex plot stores every column as an (re, im)
        pair on disk, but the scale is a real physical quantity (time or
        frequency), so a reader has to fold it back down rather than hand
        back a scale with an always-zero imaginary part.
        """
        r = read_raw(FIXTURES / 'b_ac_complex.raw')
        assert r.vectors[0].data.dtype == np.float64
        for v in r.vectors[1:]:
            assert v.data.dtype == np.complex128

    def test_frequency_values_exact_decade_formula(self):
        """3 points/decade from 1k to 10k: f_i = 10**(3 + i/3)."""
        r = read_raw(FIXTURES / 'b_ac_complex.raw')
        expected = [10 ** (3 + i / 3) for i in range(4)]
        assert_allclose(r.vectors[0].data, expected, rtol=1e-12)

    def test_source_voltage_exactly_unity(self):
        """v(1) is the AC=1 source itself: exactly 1+0j at every point,
        independent of circuit dynamics.
        """
        r = read_raw(FIXTURES / 'b_ac_complex.raw')
        assert_allclose(r.vectors[1].data, [1 + 0j] * 4)

    def test_output_voltage_matches_rc_lowpass_physics(self):
        """v(2)/v(1) must equal the single-pole transfer function
        1 / (1 + j*2*pi*f*R*C) for R=1k, C=1n - what the circuit under test
        physically does, independent of any reader.
        """
        r = read_raw(FIXTURES / 'b_ac_complex.raw')
        rc = 1e3 * 1e-9
        freqs = r.vectors[0].data
        expected = [1 / (1 + 1j * 2 * cmath.pi * f * rc) for f in freqs]
        assert_allclose(r.vectors[2].data, expected, rtol=1e-6)

    def test_current_matches_ohms_law_with_spice_sign_convention(self):
        """i(v1) = -(v(1)-v(2))/R1 - SPICE's convention that current through
        an independent voltage source is reported flowing INTO its positive
        terminal from the external circuit.
        """
        r = read_raw(FIXTURES / 'b_ac_complex.raw')
        v1, v2 = r.vectors[1].data, r.vectors[2].data
        expected = -(v1 - v2) / 1e3
        assert_allclose(r.vectors[3].data, expected, rtol=1e-6)

    def test_complex_values_transcribed_from_file(self):
        """Belt-and-suspenders: the literal numbers as they appear in
        b_ac_complex.raw's Values: block, transcribed by hand, for the
        first point (f=1000 Hz).
        """
        r = read_raw(FIXTURES / 'b_ac_complex.raw')
        assert r.vectors[0].data[0] == pytest.approx(1.000000000000000e+03)
        assert r.vectors[1].data[0] == pytest.approx(1.000000000000000e+00 + 0.000000000000000e+00j)
        assert r.vectors[2].data[0] == pytest.approx(9.999605231408795e-01 - 6.282937266758386e-03j)
        assert r.vectors[3].data[0] == pytest.approx(-3.947685912062196e-08 - 6.282937266758386e-06j)


# ---------------------------------------------------------------------------
# The Offset: keyword
# ---------------------------------------------------------------------------

class TestOffsetKeyword:
    """``Offset:`` is a header keyword carrying a value that is ADDED to the
    scale (vectors[0]) values only, applying identically whether the plot is
    written in ASCII or binary.

    No installed ngspice actually writes this field - confirmed both
    empirically (``.tran step stop tstart`` with nonzero tstart, tried at the
    top level and inside a ``.control`` block, with and without ``uic``, and
    across a ``resume``d two-``tran`` run: none of them produce an
    ``Offset:`` line), so a real-ngspice fixture for it cannot be produced
    locally (see gen_b.py's module docstring). It is a real nutmeg-family
    field nonetheless: independent public documentation of the nutmeg rawfile
    structure describes it as marking "when the saving of data started"
    (spicelib's rawfile-structure docs,
    spicelib.readthedocs.io/en/latest/varia/raw_file.html), i.e. a start-time
    correction to be added back onto the recorded scale to recover absolute
    time - so a conforming reader still needs to apply it. The fixture is
    therefore REAL+DERIVED: a byte-for-byte real ngspice rawfile with one
    ``Offset:`` header line spliced in.

    "the scale" is defined positionally, not by name: the original SPICE3
    user's manual (T. Quarles, 1993, ss 5.3.13/5.3.14) states the scale
    vector "is always the first vector in a rawfile" and "The scale vector
    (time, frequency) is always in the first column" - covering both time
    and frequency axes by position, never by name. Combined with the above,
    that is why b_offset_nontime_scale.raw (Offset on a vector 0 named
    "frequency") is asserted to behave identically to the time-axis case,
    not empirically discovered behavior.

    Semantics asserted below (spec-derived per the above, hardcoded once
    pinned down - not compared against any reader at test time):
        offset present  -> vectors[0].data += offset   (elementwise)
        offset absent   -> vectors[0].data unchanged
        effect is purely positional (applies to whatever sits at
            vectors[0], regardless of its .name/.type)
        no other vector is affected
        identical behavior in binary rawfiles (b_tran_bin_offset.raw) -
            Offset: is a header-only keyword that precedes the
            Binary:/Values: data section, so its handling cannot depend on
            which data encoding follows
    """

    def test_ascii_offset_added_to_scale_only(self):
        baseline = read_raw(FIXTURES / 'b_offset_ascii_baseline.raw')
        offset = read_raw(FIXTURES / 'b_offset_ascii.raw')

        assert_allclose(baseline.vectors[0].data, [0.0, 1e-9, 2e-9])
        assert_allclose(offset.vectors[0].data, [5e-9, 6e-9, 7e-9])
        # exactly the additive semantics: diff == the Offset: value, uniformly
        assert_allclose(offset.vectors[0].data - baseline.vectors[0].data, [5e-9] * 3)

        # the dependent vector is untouched by Offset
        assert_allclose(offset.vectors[1].data, baseline.vectors[1].data)
        assert_allclose(offset.vectors[1].data, [1.0, 2.0, 3.0])

    def test_offset_is_positional_not_gated_on_scale_type(self):
        """Offset still applies to vectors[0] even when it is named/typed
        "frequency", not "time" - it is not conditioned on the scale being a
        time axis.
        """
        r = read_raw(FIXTURES / 'b_offset_nontime_scale.raw')
        assert r.vectors[0].name == 'frequency'
        assert_allclose(r.vectors[0].data, [1250.0, 2250.0, 3250.0])
        # dependent vector still untouched
        assert_allclose(r.vectors[1].data, [1.0, 2.0, 3.0])

    def test_offset_zero_is_a_noop(self):
        r = read_raw(FIXTURES / 'b_offset_zero.raw')
        assert_allclose(r.vectors[0].data, [1e-9, 2e-9])

    def test_offset_applies_identically_in_binary(self):
        """b_tran_bin_real.raw / b_tran_bin_offset.raw are REAL / REAL+DERIVED
        (see gen_b.py): the derived file is the real binary file with a
        single Offset: text line spliced into the header and the binary
        payload byte-for-byte untouched, so any difference in parsed vector 0
        values is attributable only to that line.
        """
        baseline = read_raw(FIXTURES / 'b_tran_bin_real.raw')
        offset = read_raw(FIXTURES / 'b_tran_bin_offset.raw')

        assert baseline.npoints == 22
        assert offset.npoints == 22
        assert_allclose(
            offset.vectors[0].data - baseline.vectors[0].data,
            np.full(22, 5e-9),
        )
        # non-scale vectors identical between the two
        assert_allclose(offset.vectors[1].data, baseline.vectors[1].data)


# ---------------------------------------------------------------------------
# Per-variable trailing attributes
# ---------------------------------------------------------------------------

class TestTrailingAttributes:
    """``Variables:`` block lines are
    ``\\t<index>\\t<name>\\t<type>[ <attr>...]``: index, name, and type are
    the only fields load-bearing for silk (see the module docstring's
    "contract pinned here"), with optional trailing tokens (``grid=``,
    ``dims=``, ``min=``, ``max=``) real ngspice emits for some analyses (see
    b_ac_complex.raw's real ``frequency grid=3``, produced by a decade AC
    sweep - confirmed empirically absent from an otherwise-identical `ac
    lin` sweep on the same circuit). This class asserts only what the
    contract requires: trailing tokens must not corrupt ``.name``/``.type``.
    What surface (if any) exposes the tokens themselves is an implementation
    choice, not asserted here.
    """

    def test_real_ngspice_grid_attribute_does_not_corrupt_name_or_type(self):
        """b_ac_complex.raw's frequency scale carries a REAL ``grid=3``
        attribute (ngspice attaches this specifically for decade/octave AC
        sweeps - confirmed absent for an otherwise-identical `ac lin` sweep).
        """
        r = read_raw(FIXTURES / 'b_ac_complex.raw')
        assert r.vectors[0].name == 'frequency'
        assert r.vectors[0].type == 'frequency'

    def test_hand_crafted_multi_attribute_line_does_not_corrupt_name_or_type(self):
        """b_attrs_trailing.raw stacks multiple attributes on one line
        (``dims=2 min=0 max=1`` on vector 0, ``grid=3`` on vector 1),
        space-separated after the type field exactly as real ngspice does
        for its single-attribute case.
        """
        r = read_raw(FIXTURES / 'b_attrs_trailing.raw')
        assert r.nvars == 2
        assert r.npoints == 2
        assert r.vectors[0].name == 'time'
        assert r.vectors[0].type == 'time'
        assert r.vectors[1].name == 'v(1)'
        assert r.vectors[1].type == 'voltage'
        assert_allclose(r.vectors[0].data, [0.0, 1e-9])
        assert_allclose(r.vectors[1].data, [1.0, 2.0])


# ---------------------------------------------------------------------------
# No. Points: 0 (empty Values: block) - abort-detection depends on this
# ---------------------------------------------------------------------------

class TestZeroPoints:
    """An ASCII plot with ``No. Points: 0`` and an empty ``Values:`` block
    must parse to ``npoints == 0`` and length-0 vector data without raising -
    silk's simulation abort-detection gate depends on being able to read this
    "no data" case rather than getting an exception.
    """

    def test_zero_points_no_exception(self):
        r = read_raw(FIXTURES / 'b_zero_points.raw')
        assert r.npoints == 0

    def test_zero_points_vectors_are_empty_not_missing(self):
        r = read_raw(FIXTURES / 'b_zero_points.raw')
        assert len(r.vectors) == 2
        for v in r.vectors:
            assert v.data.shape == (0,)
            assert v.data.dtype == np.float64


# ---------------------------------------------------------------------------
# Case-insensitive header keywords
# ---------------------------------------------------------------------------

class TestCaseInsensitiveKeywords:
    """Header keywords are matched case-insensitively, as the published
    nutmeg format description specifies (``Keyword: value``, keywords
    case-insensitive). b_lowercase_keywords.raw
    cases every keyword differently from ngspice's own
    Title:/Date:/Plotname:/.../Values: convention
    (TITLE:/date:/PlotName:/FLAGS:/.../values:) and must parse identically.
    """

    def test_mixed_case_header_parses_identically(self):
        r = read_raw(FIXTURES / 'b_lowercase_keywords.raw')
        assert r.nvars == 2
        assert r.npoints == 2
        assert r.real is True
        assert [v.name for v in r.vectors] == ['time', 'v(1)']
        assert_allclose(r.vectors[0].data, [0.0, 1e-9])
        assert_allclose(r.vectors[1].data, [1.0, 2.0])


# ---------------------------------------------------------------------------
# Flags: padded / unpadded
# ---------------------------------------------------------------------------

class TestPaddedFlag:
    """``Flags:`` may carry ``padded``/``unpadded`` in addition to
    ``real``/``complex``. Real ngspice output emits neither token (confirmed on
    two independent REAL ngspice files - a real transient and a complex AC
    sweep), and the published format defines no default for that case, so the
    reader reports the declared state only: True/False when a token is present,
    None when it is absent. No parsing decision depends on the value.
    """

    def test_padding_undeclared_when_absent_real(self):
        """b_tran_real.raw: REAL ngspice ASCII output, `Flags: real` only."""
        r = read_raw(FIXTURES / 'b_tran_real.raw')
        assert r.real is True
        assert r.padded is None

    def test_padding_undeclared_when_absent_complex(self):
        """b_ac_complex.raw: REAL ngspice ASCII output, `Flags: complex` only."""
        r = read_raw(FIXTURES / 'b_ac_complex.raw')
        assert r.real is False
        assert r.padded is None

    def test_explicit_real_padded(self):
        r = read_raw(FIXTURES / 'b_flags_real_padded.raw')
        assert r.real is True
        assert r.padded is True

    def test_explicit_real_unpadded(self):
        r = read_raw(FIXTURES / 'b_flags_real_unpadded.raw')
        assert r.real is True
        assert r.padded is False

    def test_explicit_complex_padded(self):
        r = read_raw(FIXTURES / 'b_flags_complex_padded.raw')
        assert r.real is False
        assert r.padded is True
        assert r.vectors[0].data.dtype == np.float64  # scale stays real
        assert r.vectors[1].data.dtype == np.complex128
        assert_allclose(r.vectors[1].data, [1 + 0.5j, 2 + 1j])

    def test_explicit_complex_unpadded(self):
        r = read_raw(FIXTURES / 'b_flags_complex_unpadded.raw')
        assert r.real is False
        assert r.padded is False
        assert_allclose(r.vectors[1].data, [1 + 0.5j, 2 + 1j])


# ==========================================================================
# Xyce format=raw: .dc sweep axis, upper-cased names, header deltas
# (was tests/test_rawfile_reader_c.py)
# ==========================================================================

def _vector(result, name):
    """Look up a vector by exact (case-sensitive) name."""
    for v in result.vectors:
        if v.name == name:
            return v
    raise KeyError(f"{name!r} not found among {[v.name for v in result.vectors]}")


# ---------------------------------------------------------------------------
# .tran, binary, real vectors — c_tran_bin.raw
# ---------------------------------------------------------------------------


class TestXyceTransientBinary:
    """``.print tran format=raw`` — silk's actual XyceNetlister/XyceRunner path."""

    @pytest.fixture(scope="class")
    def result(self):
        return read_raw(FIXTURES / "c_tran_bin.raw")

    def test_counts(self, result):
        assert result.nvars == 3
        assert result.npoints == 70
        assert result.real is True

    def test_vector_names_ordered_and_upper_cased_verbatim(self, result):
        # Order matters: vectors[0] is the independent axis for every
        # analysis type silk consumes.
        names = [v.name for v in result.vectors]
        assert names == ["TIME", "V(IN)", "V(OUT)"]
        # Xyce upper-cases; a reader that lower-cased or otherwise
        # normalized names would fail this.
        assert "time" not in names
        assert "v(in)" not in names
        assert "v(out)" not in names

    def test_vector_types(self, result):
        types = [v.type for v in result.vectors]
        assert types == ["time", "voltage", "voltage"]

    def test_dtypes_and_shapes(self, result):
        for v in result.vectors:
            assert v.data.dtype == np.float64
            assert v.data.shape == (70,)

    def test_time_axis_boundary_values(self, result):
        time = _vector(result, "TIME").data
        assert time[0] == pytest.approx(0.0, abs=1e-15)
        assert time[-1] == pytest.approx(5e-6, rel=1e-9)
        # strictly increasing
        assert np.all(np.diff(time) > 0)

    def test_input_and_output_boundary_values(self, result):
        # Real Xyce output values, byte-exact from the .raw file (see
        # gen_c.py TRAN_NETLIST: PULSE(0 1 0 1u 1u 8u 20u) into an RC low
        # pass, R1=1k C1=1n).
        vin = _vector(result, "V(IN)").data
        vout = _vector(result, "V(OUT)").data
        assert vin[0] == pytest.approx(0.0, abs=1e-12)
        assert vin[-1] == pytest.approx(1.0, rel=1e-9)  # settled at Vsrc
        assert vout[0] == pytest.approx(0.0, abs=1e-12)
        # RC (tau=1us) hasn't fully caught up to a 1us-ramped step by t=5us,
        # but it's close; pin the actual simulated value.
        assert vout[-1] == pytest.approx(0.9885794113870636, rel=1e-9)
        # output must be monotonically charging (never overshoots for an
        # RC low-pass with no capacitor pre-charge)
        assert np.all(np.diff(vout) >= 0)

    def test_current_plot_metadata(self, result):
        cp = result.current_plot
        assert cp.plotname == "Transient Analysis"
        assert len(cp.dimensions) == 0  # list or tuple, either is fine
        assert "RC charging transient" in cp.title

    def test_exactly_one_plot(self, result):
        # Xyce writes one plot per rawfile via .print format=raw (gen_c.py
        # finding #4) — no multi-plot accumulation to handle for this
        # simulator, unlike ngspice's sequential .control `write` calls.
        assert len(result.plots) == 1
        assert result.current_plot is result.plots[-1]

    def test_top_level_mirrors_current_plot(self, result):
        assert result.nvars == len(result.current_plot.vectors)
        assert result.npoints == result.current_plot.vectors[0].data.shape[0]
        assert [v.name for v in result.vectors] == [
            v.name for v in result.current_plot.vectors
        ]


# ---------------------------------------------------------------------------
# .dc sweep, binary, real vectors — c_dc_sweep_bin.raw
# ---------------------------------------------------------------------------


class TestXyceDcSweep:
    """The independent-axis-by-position case this slice exists to pin."""

    @pytest.fixture(scope="class")
    def result(self):
        return read_raw(FIXTURES / "c_dc_sweep_bin.raw")

    def test_counts(self, result):
        assert result.nvars == 2
        assert result.npoints == 6
        assert result.real is True

    def test_vector_zero_is_the_sweep_axis_named_sweep(self, result):
        # This is THE Xyce .dc quirk: the x-axis vector is literally named
        # "sweep" (all lowercase) — not "V(V1)", not "temp-sweep" (ngspice's
        # name for an equivalent sweep), not the source name. Silk must key
        # the x-axis off vector *position* (vectors[0]), never a name.
        names = [v.name for v in result.vectors]
        assert names[0] == "sweep"
        assert names == ["sweep", "V(OUT)"]

    def test_sweep_axis_type(self, result):
        assert result.vectors[0].type == "voltage"

    def test_sweep_values_exact(self, result):
        # V1 swept 0 -> 5V in 1V steps: .dc V1 0 5 1
        sweep = _vector(result, "sweep").data
        assert_allclose(sweep, np.arange(6, dtype=float))

    def test_output_values_exact_divider(self, result):
        # R1=R2=1k resistor divider -> V(out) = V(sweep)/2, exactly.
        sweep = _vector(result, "sweep").data
        vout = _vector(result, "V(OUT)").data
        assert_allclose(vout, sweep / 2.0, rtol=1e-9)

    def test_dtype(self, result):
        for v in result.vectors:
            assert v.data.dtype == np.float64

    def test_current_plot_metadata(self, result):
        assert result.current_plot.plotname == "DC transfer characteristic"

    def test_exactly_one_plot(self, result):
        assert len(result.plots) == 1


# ---------------------------------------------------------------------------
# .ac sweep, binary, complex vectors — c_ac_bin.raw
# ---------------------------------------------------------------------------


class TestXyceAcComplex:
    @pytest.fixture(scope="class")
    def result(self):
        return read_raw(FIXTURES / "c_ac_bin.raw")

    def test_counts(self, result):
        # .ac dec 10 1 1e6 -> 6 decades * 10 pts/decade + 1 = 61
        assert result.nvars == 2
        assert result.npoints == 61

    def test_flags_complex(self, result):
        assert result.real is False

    def test_vector_names_and_order(self, result):
        names = [v.name for v in result.vectors]
        assert names == ["FREQUENCY", "V(OUT)"]

    def test_vector_types(self, result):
        types = [v.type for v in result.vectors]
        assert types == ["frequency", "voltage"]

    def test_dependent_vector_dtype_is_complex128(self, result):
        # V(OUT) is a genuine complex quantity (magnitude + phase) -- must
        # come back complex128, not silently truncated to its real part.
        assert _vector(result, "V(OUT)").data.dtype == np.complex128

    def test_independent_axis_is_real_valued(self, result):
        # On the wire, FREQUENCY is stored as a (re, im) pair just like every
        # other vector in a `Flags: complex` plot -- but its imaginary part
        # is identically zero (it's a real physical quantity: a frequency),
        # so the reader hands it back as float64 rather than complex128.
        # This is silk's own consumption contract, not a nutmeg-format rule:
        # silk does `float(sig.x[i])` / np.interp over the independent axis,
        # so a real dtype there is the useful one. Confirmed the wire format
        # still carries the (re, im) pair by checking the raw byte count in
        # TestXyceHeaderQuirks.test_no_footer_after_the_binary_data_block.
        freq = _vector(result, "FREQUENCY").data
        assert freq.dtype == np.float64

    def test_frequency_axis_exact_log_sweep(self, result):
        # .ac dec 10 1 1e6: freq[i] = 10**(i/10) for i in 0..60.
        freq = _vector(result, "FREQUENCY").data
        expected = 10.0 ** (np.arange(61) / 10.0)
        assert_allclose(freq, expected, rtol=1e-9)

    def test_output_matches_rc_low_pass_theory(self, result):
        # R1=1k, C1=1n RC low-pass: H(f) = 1 / (1 + j*2*pi*f*R*C). This is
        # derived purely from the deck's own component values (gen_c.py
        # AC_NETLIST) and the sweep formula above -- independent of any
        # reader implementation.
        freq = _vector(result, "FREQUENCY").data.real
        vout = _vector(result, "V(OUT)").data
        R, C = 1e3, 1e-9
        expected = 1.0 / (1.0 + 1j * 2 * np.pi * freq * R * C)
        assert_allclose(vout, expected, rtol=1e-6)

    def test_current_plot_metadata(self, result):
        assert result.current_plot.plotname == "AC Analysis"

    def test_exactly_one_plot(self, result):
        assert len(result.plots) == 1


# ---------------------------------------------------------------------------
# ASCII rawfile via Xyce CLI -r/-a — c_tran_ascii_cli.raw
# ---------------------------------------------------------------------------


class TestXyceAsciiViaCli:
    """Xyce's ONLY ascii nutmeg path: CLI ``-r <file> -a``, not ``.print``.

    See gen_c.py finding #1: this is a genuinely different code path from
    ``.print ... format=raw`` (used for every OTHER fixture in this file
    and by silk's actual XyceNetlister/XyceRunner). It auto-dumps every
    circuit variable and uses its own naming convention -- lowercase
    ``time``, bare node names, and a branch-current pseudo-vector. Included
    for reader robustness/completeness, not because silk's pipeline
    exercises it today.
    """

    @pytest.fixture(scope="class")
    def result(self):
        return read_raw(FIXTURES / "c_tran_ascii_cli.raw")

    def test_counts(self, result):
        assert result.nvars == 4
        assert result.npoints == 70
        assert result.real is True

    def test_vector_names_differ_from_the_print_path(self, result):
        # Note the DIVERGENCE from TestXyceTransientBinary on the exact
        # SAME circuit: lowercase "time" (not "TIME"), bare "IN"/"OUT" (not
        # "V(IN)"/"V(OUT)"), plus an extra branch-current vector that never
        # appears via .print format=raw.
        names = [v.name for v in result.vectors]
        assert names == ["time", "IN", "OUT", "V1#branch"]

    def test_vector_types(self, result):
        types = [v.type for v in result.vectors]
        assert types == ["time", "voltage", "voltage", "current"]

    def test_ascii_and_binary_agree_on_the_same_physical_circuit(self, result):
        # Cross-check against c_tran_bin.raw: same RC circuit, same .tran
        # line -> the internal solver should take the identical time steps
        # and produce the identical node voltages regardless of which
        # output mechanism wrote them. This validates both fixtures against
        # each other without hand-deriving the PWL/RC waveform analytically.
        # rtol=1e-7 (not tighter): Xyce's ASCII writer prints ~9 significant
        # digits (e.g. "1.06254431e-08"), so it necessarily loses precision
        # relative to the binary fixture's full float64 values -- confirmed
        # empirically, not a reader bug.
        bin_result = read_raw(FIXTURES / "c_tran_bin.raw")
        assert_allclose(
            _vector(result, "time").data, _vector(bin_result, "TIME").data, rtol=1e-7
        )
        assert_allclose(
            _vector(result, "IN").data, _vector(bin_result, "V(IN)").data, rtol=1e-7
        )
        assert_allclose(
            _vector(result, "OUT").data,
            _vector(bin_result, "V(OUT)").data,
            rtol=1e-7,
            atol=1e-9,
        )

    def test_exactly_one_plot(self, result):
        assert len(result.plots) == 1


# ---------------------------------------------------------------------------
# Header-format quirks — pinned directly against fixture bytes, independent
# of the reader, so a regression in either the fixture or the reader's
# tolerance for this quirk is caught explicitly.
# ---------------------------------------------------------------------------


class TestXyceHeaderQuirks:
    @pytest.mark.parametrize(
        "fixture_name",
        [
            "c_tran_bin.raw",
            "c_dc_sweep_bin.raw",
            "c_ac_bin.raw",
            "c_tran_ascii_cli.raw",
        ],
    )
    def test_no_points_line_is_padded_to_fixed_width(self, fixture_name):
        # Xyce pads "No. Points: <n>" with trailing spaces to a fixed
        # 30-byte line, regardless of digit count -- confirmed to NOT
        # happen for ngspice-42 (its "No. Points: <n>" line has no trailing
        # padding at all). A reader that does a strict/exact-format match
        # on this header line rather than tolerating trailing whitespace
        # will break on Xyce output.
        data = (FIXTURES / fixture_name).read_bytes()
        for line in data.split(b"\n"):
            if line.startswith(b"No. Points"):
                assert len(line) == 30, (fixture_name, line)
                assert line.rstrip() != line, "expected trailing pad spaces"
                break
        else:
            pytest.fail(f"no 'No. Points' header line found in {fixture_name}")

    @pytest.mark.parametrize(
        "fixture_name",
        [
            "c_tran_bin.raw",
            "c_dc_sweep_bin.raw",
            "c_ac_bin.raw",
        ],
    )
    def test_no_footer_after_the_binary_data_block(self, fixture_name):
        # Xyce's "End of Xyce(TM) Simulation" banner goes to stdout/log
        # only; the .raw file itself is exactly header + npoints*nvars
        # doubles (real) or npoints*nvars*2 doubles (complex), no trailing
        # bytes.
        data = (FIXTURES / fixture_name).read_bytes()
        marker = b"Binary:\n"
        idx = data.find(marker)
        assert idx != -1
        header_len = idx + len(marker)
        result = read_raw(FIXTURES / fixture_name)
        itemsize = 16 if not result.real else 8
        expected_data_len = result.npoints * result.nvars * itemsize
        assert len(data) - header_len == expected_data_len

    def test_flags_line_never_carries_a_padded_or_unpadded_token(self):
        # None of our 4 Xyce fixtures write a `padded`/`unpadded` token on
        # the `Flags:` line at all -- it's always a bare `Flags: real` or
        # `Flags: complex` (checked directly against the fixture bytes, no
        # reader involved). `padded`/`unpadded` describe whether a
        # multi-dimensional data block is padded out to a rectangular array.
        # The published format defines no default for the absent case, so the
        # reader reports None (undeclared) rather than assuming one. The
        # binary-layout check above is what actually pins the data size, and
        # it is derived from `No. Points` x `No. Variables` and the bytes
        # present -- both observable, neither dependent on this flag.
        for fixture_name in (
            "c_tran_bin.raw",
            "c_dc_sweep_bin.raw",
            "c_ac_bin.raw",
            "c_tran_ascii_cli.raw",
        ):
            data = (FIXTURES / fixture_name).read_bytes()
            flags_line = next(
                line for line in data.split(b"\n") if line.startswith(b"Flags:")
            )
            assert flags_line.strip() in (b"Flags: real", b"Flags: complex"), (
                fixture_name,
                flags_line,
            )

            result = read_raw(FIXTURES / fixture_name)
            assert result.padded is None, fixture_name


# ==========================================================================
# Multi-plot and the constants-only abort plot
# (was tests/test_rawfile_reader_d.py)
# ==========================================================================

class TestMultiPlotTwoRealAnalyses:
    """d_multiplot_ac_tran(.raw / _ascii.raw): a REAL two-plot rawfile.

    Produced by ``ngspice -b -r <file> <deck>`` on a deck with two top-level
    analysis directives (``.ac dec 3 1 1k`` and ``.tran 1u 5u 0 1u`` — the
    latter pins ngspice's step ceiling for a small, deterministic point
    count, the same technique gen_a.py's fixtures use) and no ``.control``
    block — the genuine ngspice multi-plot mechanism (explicit ``write``
    inside ``.control`` cannot produce this: consecutive ``write`` calls to
    the same filename overwrite rather than append; see gen_d.py). ngspice
    executes AC before Transient regardless of the textual order of the two
    directives in the deck (confirmed by swapping them and rerunning:
    identical output), so the file order is ``[AC Analysis, Transient
    Analysis]``.
    """

    @pytest.fixture(
        scope="class",
        params=["d_multiplot_ac_tran.raw", "d_multiplot_ac_tran_ascii.raw"],
    )
    def result(self, request):
        return read_raw(FIXTURES / request.param)

    def test_plots_list_has_both_in_file_order(self, result):
        assert len(result.plots) == 2
        assert [p.plotname for p in result.plots] == ["AC Analysis", "Transient Analysis"]

    def test_current_plot_is_the_last_plot(self, result):
        assert result.current_plot.plotname == "Transient Analysis"
        assert result.current_plot is result.plots[-1]

    def test_top_level_mirrors_last_plot_not_a_flattened_union(self, result):
        # The defect this whole section exists to fix: a reader that
        # flattens multi-plot files and clobbers nvars. The new contract is
        # top-level == current_plot (the LAST plot), never a concatenation
        # of both plots' vectors.
        assert result.nvars == 4
        assert result.npoints == 22
        assert result.real is True
        assert len(result.vectors) == 4

    def test_top_level_vectors_are_the_transient_axis_not_ac(self, result):
        # silk/signals/spice.py:NgspiceSignal.from_rawfile reads vectors[0] as
        # the independent axis. If top-level vectors were the AC (first)
        # plot's, this would be 'frequency' and every downstream signal read
        # after a multi-plot run would silently get the wrong axis.
        assert result.vectors[0].name == "time"

    def test_first_plot_is_ac_analysis_complex(self, result):
        ac = result.plots[0]
        assert ac.plotname == "AC Analysis"
        assert ac.nvars == 4
        assert ac.npoints == 10
        assert ac.real is False
        assert [v.name for v in ac.vectors] == ["frequency", "v(in)", "v(out)", "i(v1)"]
        for v in ac.vectors:
            assert v.data.shape == (10,)

    def test_second_plot_is_transient_real(self, result):
        tran = result.plots[1]
        assert tran.plotname == "Transient Analysis"
        assert tran.nvars == 4
        assert tran.npoints == 22
        assert tran.real is True
        assert [v.name for v in tran.vectors] == ["time", "v(in)", "v(out)", "i(v1)"]
        for v in tran.vectors:
            assert v.data.dtype == np.float64
            assert v.data.shape == (22,)

    def test_plots_carry_independent_data_not_shared_state(self, result):
        # A flattening bug could pass this test's counts by accident if the
        # two plots secretly shared a vectors list. They must not.
        assert result.plots[0].vectors is not result.plots[1].vectors
        assert result.plots[0].vectors[0].data is not result.plots[1].vectors[0].data

    def test_transient_plot_exact_endpoint_values(self, result):
        # Concrete values from this real RC network (V1 is a slow-rising
        # PULSE with an AC magnitude on the same line, into a 1k/1n RC
        # lowpass — the same circuit shape as gen_a.py's a_tran_bin.raw): at
        # t=0 everything is exactly 0; the exact closing values are
        # transcribed from this same real run (stable across regeneration —
        # unlike the AC plot's frequency scale, see
        # test_ac_frequency_scale_source_noise_is_genuine below, the
        # transient waveform itself is ordinary physics, not memory noise).
        time, vin, vout, iv1 = (v.data for v in result.plots[1].vectors)
        assert time[0] == 0.0
        assert vin[0] == 0.0
        assert vout[0] == 0.0
        assert iv1[0] == 0.0
        assert time[-1] == pytest.approx(5e-06, rel=1e-9)
        assert vin[-1] == pytest.approx(1.0, rel=1e-12)
        assert vout[-1] == pytest.approx(0.9914561752448197, rel=1e-9)
        assert iv1[-1] == pytest.approx(-8.543824755180346e-06, rel=1e-9)

    def test_ac_plot_exact_vout_values(self, result):
        vout_ac = result.plots[0].vectors[2].data
        assert vout_ac[0] == pytest.approx(0.9999999999605216 - 6.2831853069315364e-06j, rel=1e-9)
        assert vout_ac[-1] == pytest.approx(0.9999605231408795 - 0.00628293726675839j, rel=1e-9)

    def test_ac_frequency_scale_realified_despite_stray_imaginary_noise(self, result):
        # Runs for BOTH the binary and the ASCII form of this fixture.
        #
        # The reader hands a complex plot's independent axis back as a real
        # float64 array — silk's metrics do float(sig.x[i]) and np.interp over
        # it, and np.interp raises outright on complex input. It must do so
        # even though THIS file's frequency scale is not cleanly real (see
        # test_ac_frequency_scale_source_noise_is_genuine below), which is why
        # _realify_scale tests for a negligible imaginary part rather than an
        # exactly-zero one.
        freq = result.plots[0].vectors[0].data
        assert freq.dtype == np.float64
        assert freq[0] == pytest.approx(1.0, rel=1e-9)
        assert freq[-1] == pytest.approx(1000.0, rel=1e-9)
        # Only the scale is realified; dependent vectors stay complex.
        assert result.plots[0].vectors[2].data.dtype == np.complex128

    def test_ac_frequency_scale_source_noise_is_genuine(self):
        # Evidence, read straight from the fixture bytes with no reader
        # involved, that the test above is guarding a real condition and not a
        # hypothetical one. A real, reproducible-in-KIND (not bit-for-bit)
        # ngspice-42 quirk of the "-r batch, multiple top-level directives"
        # code path: the AC frequency scale is written with a non-zero,
        # DENORMALIZED imaginary half. It differs between invocations
        # (uninitialized memory), so this asserts its existence and magnitude,
        # never an exact bit pattern.
        raw_bytes = (FIXTURES / "d_multiplot_ac_tran.raw").read_bytes()
        header_end = raw_bytes.index(b"Binary:\n") + len(b"Binary:\n")
        # AC plot is first; its scale is the first complex128 of each row.
        pairs = np.frombuffer(raw_bytes, dtype="<c16", count=10 * 4,
                              offset=header_end).reshape(10, 4)
        stray = np.abs(pairs[:, 0].imag)
        assert np.any(stray > 0.0), (
            "fixture no longer carries the stray imaginary noise this guards; "
            "if it was regenerated, _realify_scale's tolerance is now untested"
        )
        # Denormal: below the smallest normal float, ~300 orders of magnitude
        # under the frequencies themselves — hence separable by a relative test.
        assert np.max(stray) < np.finfo(np.float64).tiny

        # The ASCII sibling of the same run shows ngspice writing it as text,
        # confirming this is what ngspice emits and not our binary misalignment.
        ascii_text = (FIXTURES / "d_multiplot_ac_tran_ascii.raw").read_text()
        body = ascii_text.split("Values:", 1)[1]
        first_scale_token = body.split("\n")[1].split()[-1]
        assert float(first_scale_token.split(",")[1]) != 0.0


class TestMultiPlotDifferingNvars:
    """d_multiplot_nvars_differ.raw: HAND-CRAFTED concatenation of two
    independently-REAL plots that genuinely disagree on nvars.

    A single ngspice circuit's AC and transient plots always share the same
    variable set (same nodes/sources save-listed identically for both), so a
    real nvars mismatch across plots needs two different circuits. Plot 0 is
    a REAL AC run of a *different*, smaller circuit with an explicit
    ``.save v(out)`` (2 vars: frequency, v(out)); plot 1 is the same REAL
    Transient plot as the ac_tran fixture above (4 vars). Pins that the
    reader reports each plot's OWN nvars/npoints/dtype independently rather
    than reusing counts across plots.
    """

    @pytest.fixture(scope="class")
    def result(self):
        return read_raw(FIXTURES / "d_multiplot_nvars_differ.raw")

    def test_plot_order_and_names(self, result):
        assert [p.plotname for p in result.plots] == ["AC Analysis", "Transient Analysis"]

    def test_first_plot_has_2_vars(self, result):
        ac = result.plots[0]
        assert ac.nvars == 2
        assert ac.npoints == 10
        assert ac.real is False
        assert [v.name for v in ac.vectors] == ["frequency", "v(out)"]
        for v in ac.vectors:
            assert v.data.shape == (10,)

    def test_second_plot_has_4_vars(self, result):
        tran = result.plots[1]
        assert tran.nvars == 4
        assert tran.npoints == 22
        assert tran.real is True
        assert [v.name for v in tran.vectors] == ["time", "v(in)", "v(out)", "i(v1)"]

    def test_top_level_mirrors_the_4_var_last_plot(self, result):
        assert result.nvars == 4
        assert result.npoints == 22
        assert result.real is True

    def test_ac_vout_is_constant_half_scale_divider(self, result):
        # Physical sanity check independent of the AC-frequency-noise quirk
        # above: R1=R2=1k resistive divider (no reactive elements) -> v(out)
        # is exactly 0.5, flat across the whole sweep, real and imaginary.
        vout = result.plots[0].vectors[1].data
        np.testing.assert_allclose(vout, [0.5 + 0.0j] * 10, rtol=1e-9, atol=1e-12)


class TestMultiPlotOrderIsFileOrder:
    """d_multiplot_order_reversed.raw: HAND-CRAFTED — the SAME two real plot
    blocks as d_multiplot_ac_tran.raw, concatenated in the OPPOSITE order
    ([Transient Analysis, AC Analysis]).

    ngspice-42 itself will not produce this ordering (it always runs AC
    before Transient regardless of deck order — see gen_d.py), so this is
    the only way to prove the reader honors literal byte order in the file
    rather than a hardcoded "AC always precedes Transient" assumption.
    """

    @pytest.fixture(scope="class")
    def result(self):
        return read_raw(FIXTURES / "d_multiplot_order_reversed.raw")

    def test_plot_order_is_reversed_vs_the_other_fixture(self, result):
        assert [p.plotname for p in result.plots] == ["Transient Analysis", "AC Analysis"]

    def test_current_plot_is_ac_this_time(self, result):
        assert result.current_plot.plotname == "AC Analysis"
        assert result.current_plot is result.plots[-1]

    def test_top_level_mirrors_ac_not_transient(self, result):
        assert result.nvars == 4
        assert result.npoints == 10
        assert result.real is False
        assert result.vectors[0].name == "frequency"

    def test_first_plot_is_the_transient_one(self, result):
        assert result.plots[0].plotname == "Transient Analysis"
        assert result.plots[0].npoints == 22
        assert result.plots[0].real is True


class TestConstantsOnlyPlot:
    """d_constants_only(.raw / _ascii.raw): REAL — ngspice's synthetic
    fallback plot when a deck has no analysis directive at all and an
    explicit ``.control write`` runs anyway (the same mechanism a genuine
    abort leaves behind — see gen_d.py). This reproduces
    the binary fixture byte-for-byte: 513 bytes, 12 vars, ``No. Points: 1``.
    """

    @pytest.fixture(
        scope="class",
        params=["d_constants_only.raw", "d_constants_only_ascii.raw"],
    )
    def result(self, request):
        return read_raw(FIXTURES / request.param)

    def test_single_plot_named_constants(self, result):
        assert len(result.plots) == 1
        assert result.current_plot.plotname == "constants"

    def test_npoints_is_1(self, result):
        assert result.npoints == 1
        assert result.current_plot.npoints == 1

    def test_ordered_variable_names(self, result):
        names = [v.name for v in result.vectors]
        assert names == [
            "yes", "false", "true", "boltz", "c", "e",
            "echarge", "i", "kelvin", "no", "pi", "planck",
        ]
        assert result.nvars == 12

    def test_exact_constant_values(self, result):
        by_name = {v.name: v.data[0] for v in result.vectors}
        assert by_name["yes"] == pytest.approx(1.0)
        assert by_name["false"] == pytest.approx(0.0)
        assert by_name["true"] == pytest.approx(1.0)
        assert by_name["boltz"] == pytest.approx(1.38064852e-23, rel=1e-9)
        assert by_name["c"] == pytest.approx(299792458.0, rel=1e-12)
        assert by_name["e"] == pytest.approx(2.718281828459045, rel=1e-12)
        assert by_name["echarge"] == pytest.approx(1.6021766208e-19, rel=1e-9)
        assert by_name["i"] == pytest.approx(1j)
        assert by_name["kelvin"] == pytest.approx(-273.15, rel=1e-12)
        assert by_name["no"] == pytest.approx(0.0)
        assert by_name["pi"] == pytest.approx(3.141592653589793, rel=1e-12)
        assert by_name["planck"] == pytest.approx(6.62607004e-34, rel=1e-9)

    def test_flagged_complex_except_the_realified_scale(self, result):
        # Every constant except vector 0 ('yes') is stored as a complex pair
        # (Flags: complex covers the whole plot, including 'i' == 0+1j,
        # genuinely complex) — vector 0 alone is realified because its
        # imaginary part is identically zero (see _realify_scale).
        vectors = result.vectors
        assert vectors[0].data.dtype == np.float64
        for v in vectors[1:]:
            assert v.data.dtype == np.complex128

    def test_backend_guard_rejects_constants_only(self, result):
        # This is the actual safety net: _has_data_rows must say "no data"
        # for a constants-only rawfile even though npoints == 1 (would look
        # like data to a naive check).
        assert _has_data_rows(result) is False


class TestMultiPlotConstantsOrdering:
    """The two orderings the backend guard's docstring says can and cannot
    occur (silk/simulation/backend.py:_has_data_rows):

    * constants-then-real (an aborted first attempt whose leftover plot is
      superseded by a later successful run in the same file) -> the LAST
      plot is real data, so the run must NOT be flagged as aborted.
    * real-then-constants -- "a real analysis followed by constants cannot
      occur" per that docstring; this fixture exists as defense-in-depth so
      the guard's behavior is pinned even for a case ngspice itself would not
      produce, not just documented as impossible.
    """

    def test_constants_first_then_real_is_not_flagged_aborted(self):
        result = read_raw(FIXTURES / "d_multiplot_constants_first.raw")
        assert [p.plotname for p in result.plots] == ["constants", "Transient Analysis"]
        assert result.current_plot.plotname == "Transient Analysis"
        assert result.npoints == 22
        assert _has_data_rows(result) is True

    def test_real_then_constants_last_is_flagged_aborted(self):
        result = read_raw(FIXTURES / "d_multiplot_constants_last.raw")
        assert [p.plotname for p in result.plots] == ["Transient Analysis", "constants"]
        assert result.current_plot.plotname == "constants"
        assert result.npoints == 1
        assert _has_data_rows(result) is False


class TestHealthyTransientGuard:
    """d_healthy_transient.raw: REAL single-plot transient — the "normal run,
    do not false-positive" companion to TestConstantsOnlyPlot's rejection
    case. Both together are the real safety net for the abort-detection
    migration onto the new reader.
    """

    @pytest.fixture(scope="class")
    def result(self):
        return read_raw(FIXTURES / "d_healthy_transient.raw")

    def test_single_real_transient_plot(self, result):
        assert len(result.plots) == 1
        assert result.current_plot.plotname == "Transient Analysis"
        assert result.npoints == 22
        assert result.real is True

    def test_backend_guard_accepts_healthy_run(self, result):
        assert _has_data_rows(result) is True


# --- TASK-440: the ``dims=0`` zero-length-vector marker -------------------

_OP_FIXTURES = Path(__file__).parent / 'fixtures' / 'op_point'


def test_dims0_vector_reads_back_empty_not_zero():
    from silk.signals.rawfile import read_rawfile
    raw = read_rawfile(str(_OP_FIXTURES / 'ngspice42_op_probe.raw'))
    by_name = {v.name.lower(): v for v in raw.vectors}
    for param in ('cgso', 'cgdo'):
        vec = next(v for n, v in by_name.items() if n.endswith(f'[{param}]'))
        assert 'dims=0' in vec.attrs
        assert len(vec.data) == 0
    # a normally-filled vector in the same plot is untouched
    gm = next(v for n, v in by_name.items() if n.endswith('[gm]'))
    assert len(gm.data) == 1 and gm.data[0] != 0.0


def test_device_op_vectors_maps_dims0_to_nan_and_ngspice47_populated():
    from types import SimpleNamespace
    from silk.signals.rawfile import read_rawfile
    from silk.simulation.backend import NgspiceBackend
    be = NgspiceBackend.__new__(NgspiceBackend)
    for ver, populated in (('42', False), ('47', True)):
        raw = SimpleNamespace(native=read_rawfile(
            str(_OP_FIXTURES / f'ngspice{ver}_op_probe.raw')))
        out = be.device_op_vectors(raw, 'xdut.xmmmn', 'msky130_fd_pr__nfet_01v8')
        for param in ('cgso', 'cgdo'):
            if populated:
                assert out[param] > 0.0
            else:
                assert out[param] != out[param]          # nan, never 0.0


def test_emit_view_data_survives_dims0_vector_as_strict_json():
    """A host serializes emit_view_data's payload with json.dumps(allow_nan=False) and
    catches only ViewError: a zero-length vector must not break that."""
    import json
    from silk.signals.view.emit import emit_view_data
    payload = emit_view_data(str(_OP_FIXTURES / 'ngspice42_op_probe.raw'))
    json.dumps(payload, allow_nan=False)


# TASK-469.1: opt-in lazy reader ownership and bounded materialization.

def _lazy_fixture(tmp_path, binary=True, npoints=9, nvars=4):
    from silk.signals.rawfile import RawPlot, RawVector, write_rawfile
    path = tmp_path / "lazy.raw"
    vectors = [RawVector("time" if i == 0 else "duplicate", data=
                         np.arange(npoints, dtype=float) + i * 100)
               for i in range(nvars)]
    write_rawfile(path, RawPlot({"plotname": "Transient Analysis"}, vectors,
                              npoints, True, None), binary=binary)
    return path


@pytest.mark.parametrize("binary", [True, False])
def test_lazy_metadata_and_slot_cache(tmp_path, monkeypatch, binary):
    from silk.signals import rawfile
    path = _lazy_fixture(tmp_path, binary)
    calls = []
    original = rawfile._LazyStore.materialize

    def counted(self, *args):
        calls.append(args)
        return original(self, *args)

    monkeypatch.setattr(rawfile._LazyStore, "materialize", counted)
    raw = rawfile.read_rawfile(path, lazy=True)
    assert raw.npoints == 9
    assert len(raw.vectors[1]) == 9
    assert "npoints=9" in repr(raw.vectors[1])
    assert "nvars=4" in repr(raw)
    assert [v.name for v in raw.vectors] == ["time"] + ["duplicate"] * 3
    assert not any(v.name == "missing" for p in raw.get_plots() for v in p.vectors)
    assert calls == []
    first = raw.vectors[1].data
    assert raw.vectors[1].data is first
    assert len(calls) == 1
    np.testing.assert_array_equal(first, np.arange(9) + 100)
    np.testing.assert_array_equal(raw.vectors[2].data, np.arange(9) + 200)
    assert len(calls) == 2
    other = rawfile.read_rawfile(path, lazy=True)
    first[0] = 123
    assert other.vectors[1].data[0] == 100


@pytest.mark.parametrize("change", ["delete", "move", "replace", "truncate", "modify"])
def test_lazy_changed_file_preserves_copied_vectors(tmp_path, change):
    from silk.signals.rawfile import read_rawfile, RawfileError
    from silk.exceptions import SimulationError
    path = _lazy_fixture(tmp_path)
    raw = read_rawfile(path, lazy=True)
    cached = raw.vectors[0].data.copy()
    raw.vectors[0].data[0] = 123
    if change == "delete":
        path.unlink()
    elif change == "move":
        path.rename(path.with_suffix(".old"))
    elif change == "replace":
        replacement = path.with_suffix(".new")
        replacement.write_bytes(path.read_bytes())
        replacement.replace(path)
    elif change == "truncate":
        with path.open("r+b") as handle:
            handle.truncate(100)
    else:
        import os
        stat = path.stat()
        with path.open("r+b") as handle:
            handle.seek(-8, 2)
            handle.write(b"12345678")
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert raw.vectors[0].data[0] == 123
    np.testing.assert_array_equal(raw.vectors[0].data[1:], cached[1:])
    with pytest.raises(RawfileError) as error:
        _ = raw.vectors[1].data
    assert isinstance(error.value, SimulationError)
    assert str(path) in str(error.value)
    assert "duplicate" in str(error.value)


def test_lazy_missing_at_open(tmp_path):
    from silk.signals.rawfile import read_rawfile, RawfileError
    path = tmp_path / "missing.raw"
    with pytest.raises(RawfileError, match="missing.raw"):
        read_rawfile(path, lazy=True)


def test_lazy_deepcopy_aliases_and_writable_data_after_deletion(tmp_path):
    from copy import deepcopy
    from silk.signals.rawfile import read_rawfile, RawVector, RawfileError
    path = _lazy_fixture(tmp_path)
    raw = read_rawfile(path, lazy=True)
    y = raw.vectors[1].data
    y[0] = 123
    path.unlink()
    copied, alias, copied_y = deepcopy([raw, raw, y])
    assert copied is alias
    assert copied.vectors[1].data is copied_y
    assert copied_y[0] == 123
    copied_y[0] = 456
    assert y[0] == 123
    with pytest.raises(RawfileError):
        _ = copied.vectors[2].data
    replacement = np.array([7.0])
    copied.vectors[2].data = replacement
    assert copied.vectors[2].data is replacement
    assert len(copied.vectors[2]) == 1
    direct = RawVector("manual", data=replacement)
    assert direct.data is replacement
    direct.data = y
    assert direct.data is y


@pytest.mark.parametrize("binary", [True, False])
def test_lazy_bounded_io_and_streamed_fingerprint(tmp_path, monkeypatch, binary):
    import builtins
    import hashlib
    from silk.signals.rawfile import read_rawfile
    path = _lazy_fixture(tmp_path, binary, npoints=5000, nvars=40)
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    original = builtins.open
    reads = []
    allocations = []
    original_empty = np.empty

    def bounded_empty(shape, *args, **kwargs):
        size = int(np.prod(shape))
        assert size <= 5000, "allocated more than a single waveform column"
        allocations.append(size)
        return original_empty(shape, *args, **kwargs)

    class BoundedFile:
        def __init__(self, handle):
            self.handle = handle
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.handle.close()
        def __getattr__(self, name):
            return getattr(self.handle, name)
        def read(self, size=-1):
            assert 0 <= size <= 1024 * 1024
            reads.append(size)
            return self.handle.read(size)

    def bounded_open(file, *args, **kwargs):
        return BoundedFile(original(file, *args, **kwargs))

    monkeypatch.setattr(builtins, "open", bounded_open)
    monkeypatch.setattr(np, "empty", bounded_empty)
    monkeypatch.setattr(np, "memmap", lambda *a, **k: pytest.fail("external mmap"))
    raw = read_rawfile(path, lazy=True)
    assert raw._content_sha256 == expected
    assert not any(allocations)
    assert all(v._data is None for v in raw.vectors)
    np.testing.assert_array_equal(raw.vectors[1].data, np.arange(5000) + 100)
    assert sum(v._data is not None for v in raw.vectors) == 1
    assert reads
    assert allocations.count(5000) == 1


@pytest.mark.parametrize('binary', [False, True])
@pytest.mark.parametrize('points', [0, 1, 7])
def test_lazy_truncated_data_matches_eager(tmp_path, binary, points):
    from silk.signals.rawfile import read_rawfile
    path = _lazy_fixture(tmp_path, binary)
    content = path.read_bytes()
    if binary:
        start = content.index(b'Binary:\n') + len(b'Binary:\n')
        path.write_bytes(content[:start + points * 4 * 8 + 3])
    else:
        start = content.index(b'Values:\n') + len(b'Values:\n')
        lines = content[start:].splitlines(keepends=True)
        path.write_bytes(content[:start] + b''.join(lines[:points * 4 + 1]))
    eager = read_rawfile(path, lazy=False)
    lazy = read_rawfile(path, lazy=True)
    assert eager.npoints == lazy.npoints == points
    for a, b in zip(eager.vectors, lazy.vectors):
        assert len(a) == len(b)
        np.testing.assert_array_equal(a.data, b.data)


def test_lazy_mixed_encodings_and_empty_plot(tmp_path):
    from silk.signals.rawfile import read_rawfile
    binary = _lazy_fixture(tmp_path).read_bytes()
    ascii_data = _lazy_fixture(tmp_path, False).read_bytes()
    empty = _lazy_fixture(tmp_path, npoints=0).read_bytes()
    path = tmp_path / 'mixed.raw'
    path.write_bytes(binary + ascii_data + empty)
    eager = read_rawfile(path, lazy=False)
    lazy = read_rawfile(path, lazy=True)
    assert [p.npoints for p in lazy.plots] == [9, 9, 0]
    for a, b in zip(eager.plots, lazy.plots):
        assert a.header == b.header
        for x, y in zip(a.vectors, b.vectors):
            np.testing.assert_array_equal(x.data, y.data)


@pytest.mark.parametrize('content', [b'not a rawfile', b'Plotname: broken\n',
    b'Plotname: broken\nVariables:\n',
    b'Plotname: broken\nNo. Variables: 3\nVariables:\n0 time time\n'])
def test_lazy_malformed_header_matches_eager(tmp_path, content):
    from silk.signals.rawfile import read_rawfile, RawfileError
    path = tmp_path / 'broken.raw'
    path.write_bytes(content)
    for lazy in (False, True):
        with pytest.raises(RawfileError):
            read_rawfile(path, lazy=lazy)


@pytest.mark.parametrize('binary', [False, True])
def test_lazy_mutation_during_read_is_typed_and_not_cached(tmp_path, monkeypatch, binary):
    import builtins
    from silk.signals.rawfile import read_rawfile, RawfileError
    path = _lazy_fixture(tmp_path, binary)
    raw = read_rawfile(path, lazy=True)
    original = builtins.open

    class ChangingFile:
        def __init__(self, handle):
            self.handle = handle
            self.changed = False
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.handle.close()
        def __getattr__(self, name):
            return getattr(self.handle, name)
        def change(self):
            if not self.changed:
                self.changed = True
                with original(path, 'wb') as output:
                    output.write(b'broken')
        def read(self, size=-1):
            self.change()
            return self.handle.read(size)
        def readline(self, *args):
            self.change()
            return self.handle.readline(*args)

    monkeypatch.setattr(builtins, 'open', lambda *a, **k: ChangingFile(original(*a, **k)))
    with pytest.raises(RawfileError, match='duplicate'):
        _ = raw.vectors[1].data
    assert raw.vectors[1]._data is None


def test_lazy_preserves_symlink_parent_resolution_and_relative_path(tmp_path, monkeypatch):
    from silk.signals.rawfile import read_rawfile
    local = tmp_path / 'local'
    remote = tmp_path / 'remote'
    local.mkdir()
    remote.mkdir()
    (remote / 'child').mkdir()
    (local / 'link').symlink_to(remote / 'child', target_is_directory=True)
    _lazy_fixture(local, npoints=3)
    _lazy_fixture(remote, npoints=7)
    monkeypatch.chdir(local)
    path = Path('link/../lazy.raw')
    eager = read_rawfile(path, lazy=False)
    lazy = read_rawfile(path, lazy=True)
    assert lazy.npoints == eager.npoints == 7
    monkeypatch.chdir(tmp_path)
    np.testing.assert_array_equal(lazy.vectors[1].data, eager.vectors[1].data)


def test_lazy_change_during_index_names_source(tmp_path, monkeypatch):
    from silk.signals import rawfile
    path = _lazy_fixture(tmp_path)
    original = rawfile._parse_plot

    def change_then_parse(data, pos):
        path.write_bytes(b'Plotname: changed\nnot a header\n')
        return original(data, pos)

    monkeypatch.setattr(rawfile, '_parse_plot', change_then_parse)
    with pytest.raises(rawfile.RawfileError) as error:
        rawfile.read_rawfile(path, lazy=True)
    assert str(path) in str(error.value)
    assert '<index>' in str(error.value)
