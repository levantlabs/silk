"""
Tests for silk.signals — NgspiceSignal.from_rawfile() and Signal behavior.

Covers:
- NgspiceSignal.from_rawfile() success paths (voltage, current)
- NgspiceSignal.from_rawfile() error paths (not found, malformed names)
- Case-insensitive signal type matching
- Inheritance chain assertions
- Arithmetic on NgspiceSignal returns NgspiceSignal

Result loading is dispatched by the SimulatorBackend (load_signal), not a
Signal registry — see test_simulator_backend.py / test_xyce_netlister.py.

Does NOT duplicate arithmetic/plot/error tests already in test_analysis.py.
"""

from pathlib import Path

import pytest
import numpy as np
from unittest.mock import MagicMock

from silk.signals import Signal, NgspiceSignal
from silk.signals.rawfile import read_rawfile
from silk.exceptions import SimulationError
from tests._env import needs_ngspice


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_rawdata(signals):
    """Build a mock ngspice rawdata object.

    Parameters
    ----------
    signals : list of (name, data, type)
        Each tuple becomes one vector on the mock object.
    """
    raw = MagicMock()
    raw.nvars = len(signals)
    vecs = []
    for name, data, sig_type in signals:
        v = MagicMock()
        v.name = name
        v.data = np.asarray(data)
        v.type = sig_type
        vecs.append(v)
    raw.vectors = vecs
    return raw


def _make_transient_rawdata():
    """Return a minimal transient rawdata with time, v(out), and i(drain)."""
    t = np.linspace(0, 1e-6, 50)
    return make_rawdata([
        ('time',    t,                   'time'),
        ('v(out)',  np.sin(t * 1e6),     'voltage'),
        ('i(drain)', np.ones(50) * 1e-3, 'current'),
    ])


# ---------------------------------------------------------------------------
# NgspiceSignal.from_rawfile() — success paths
# ---------------------------------------------------------------------------

class TestFromRawfileSuccess:

    def test_extracts_voltage_signal_y_values(self):
        """from_rawfile() extracts correct y-values for a voltage signal."""
        raw = _make_transient_rawdata()
        sig = NgspiceSignal.from_rawfile(raw, 'out', 'V')
        expected_y = raw.vectors[1].data
        np.testing.assert_array_equal(sig.y, expected_y)

    def test_extracts_voltage_signal_x_values(self):
        """from_rawfile() sets x-values from the time vector."""
        raw = _make_transient_rawdata()
        sig = NgspiceSignal.from_rawfile(raw, 'out', 'V')
        np.testing.assert_array_equal(sig.x, raw.vectors[0].data)

    def test_extracted_signal_name(self):
        """from_rawfile() uses the bare signal name (without v(...) wrapper)."""
        raw = _make_transient_rawdata()
        sig = NgspiceSignal.from_rawfile(raw, 'out', 'V')
        assert sig.name == 'out'

    def test_extracts_current_signal(self):
        """from_rawfile() extracts a current signal when signal_type='I'."""
        raw = _make_transient_rawdata()
        sig = NgspiceSignal.from_rawfile(raw, 'drain', 'I')
        expected_y = raw.vectors[2].data
        np.testing.assert_array_equal(sig.y, expected_y)

    def test_current_signal_type_attribute(self):
        """Extracted current signal has type attribute from rawfile vector."""
        raw = _make_transient_rawdata()
        sig = NgspiceSignal.from_rawfile(raw, 'drain', 'I')
        assert sig.type == 'current'

    def test_returns_pyspice_signal_instance(self):
        """from_rawfile() returns a NgspiceSignal (not just Signal)."""
        raw = _make_transient_rawdata()
        sig = NgspiceSignal.from_rawfile(raw, 'out', 'V')
        assert type(sig) is NgspiceSignal

    def test_returned_signal_is_pysignal_instance(self):
        """NgspiceSignal from from_rawfile() is also an instance of Signal."""
        raw = _make_transient_rawdata()
        sig = NgspiceSignal.from_rawfile(raw, 'out', 'V')
        assert isinstance(sig, Signal)

    def test_default_signal_type_is_voltage(self):
        """from_rawfile() defaults to signal_type='V' when not specified."""
        raw = _make_transient_rawdata()
        sig = NgspiceSignal.from_rawfile(raw, 'out')
        np.testing.assert_array_equal(sig.y, raw.vectors[1].data)


# ---------------------------------------------------------------------------
# NgspiceSignal.from_rawfile() — error paths
# ---------------------------------------------------------------------------

class TestFromRawfileErrors:

    def test_missing_signal_raises_simulation_error(self):
        """from_rawfile() raises SimulationError when signal is not found."""
        raw = _make_transient_rawdata()
        with pytest.raises(SimulationError):
            NgspiceSignal.from_rawfile(raw, 'nonexistent_net', 'V')

    def test_error_message_includes_signal_name(self):
        """SimulationError message contains the requested signal name."""
        raw = _make_transient_rawdata()
        with pytest.raises(SimulationError, match='nonexistent_net'):
            NgspiceSignal.from_rawfile(raw, 'nonexistent_net', 'V')

    def test_wrong_type_raises_simulation_error(self):
        """from_rawfile() raises SimulationError when type doesn't match."""
        raw = _make_transient_rawdata()
        # 'out' exists as voltage, not current
        with pytest.raises(SimulationError):
            NgspiceSignal.from_rawfile(raw, 'out', 'I')

    def test_malformed_signal_name_is_skipped(self):
        """Vectors with no '(' in name are silently skipped (not matched)."""
        # Include a malformed vector name alongside valid ones
        t = np.linspace(0, 1e-6, 10)
        raw = make_rawdata([
            ('time',         t,              'time'),
            ('malformed',    np.zeros(10),   'voltage'),   # no '(' — skipped
            ('v(good)',      np.ones(10),    'voltage'),
        ])
        # 'good' should be found normally despite malformed sibling
        sig = NgspiceSignal.from_rawfile(raw, 'good', 'V')
        np.testing.assert_array_equal(sig.y, np.ones(10))

    def test_malformed_only_signals_raise_simulation_error(self):
        """When all non-xaxis vectors are malformed, SimulationError is raised."""
        t = np.linspace(0, 1e-6, 10)
        raw = make_rawdata([
            ('time',       t,            'time'),
            ('malformed1', np.zeros(10), 'voltage'),
            ('alsobroken', np.zeros(10), 'voltage'),
        ])
        with pytest.raises(SimulationError):
            NgspiceSignal.from_rawfile(raw, 'out', 'V')


# ---------------------------------------------------------------------------
# DC sweep x-axis — vector 0 is the independent axis for ANY analysis
# (the sweep axis is named neither 'time' nor 'frequency'; ngspice uses
# 'temp-sweep' / 'v(v-sweep)', Xyce uses 'sweep' — all are vector 0).
# ---------------------------------------------------------------------------

class TestDCSweepAxis:

    def test_dc_temp_sweep_bare_axis(self):
        """ngspice '.dc temp': axis 'temp-sweep' (bare) becomes x; node is y."""
        temp = np.linspace(-40, 125, 34)
        vref = np.linspace(1.2001, 1.2114, 34)
        raw = make_rawdata([
            ('temp-sweep', temp, 'temperature'),
            ('v(vref)',    vref, 'voltage'),
        ])
        sig = NgspiceSignal.from_rawfile(raw, 'vref', 'V')
        np.testing.assert_array_equal(sig.x, temp)   # x is the swept variable
        np.testing.assert_array_equal(sig.y, vref)   # y is the node

    def test_dc_voltage_sweep_wrapped_axis_not_confused_with_node(self):
        """ngspice '.dc vin': axis 'v(v-sweep)' (v(...)-wrapped) becomes x, and
        must NOT be mistaken for a node. A divider node stays the y-signal."""
        vsweep = np.linspace(0, 1, 11)
        vout = vsweep / 2.0                         # a real divider output
        raw = make_rawdata([
            ('v(v-sweep)', vsweep, 'voltage'),
            ('v(vout)',    vout,   'voltage'),
        ])
        sig = NgspiceSignal.from_rawfile(raw, 'vout', 'V')
        np.testing.assert_array_equal(sig.x, vsweep)
        np.testing.assert_array_equal(sig.y, vout)
        assert not np.array_equal(sig.y, vsweep)     # not a copy of the axis

    def test_xyce_dc_sweep_axis(self):
        """Xyce '.dc vin': axis is literally 'sweep' (does not end in -sweep),
        with upper-cased node names. Vector 0 is still the axis."""
        sweep = np.linspace(0, 1, 11)
        vout = sweep * 0.5
        raw = make_rawdata([
            ('sweep',  sweep, 'voltage'),
            ('V(OUT)', vout,  'voltage'),           # Xyce upper-cases
        ])
        sig = NgspiceSignal.from_rawfile(raw, 'out', 'V')   # case-insensitive
        np.testing.assert_array_equal(sig.x, sweep)
        np.testing.assert_array_equal(sig.y, vout)

    def test_transient_still_works(self):
        """Regression: the time axis (vector 0) still resolves as before."""
        raw = _make_transient_rawdata()
        sig = NgspiceSignal.from_rawfile(raw, 'out', 'V')
        np.testing.assert_array_equal(sig.x, raw.vectors[0].data)
        np.testing.assert_array_equal(sig.y, raw.vectors[1].data)


class TestOperatingPointSignal:
    """``from_rawfile`` on an .op plot (TASK-102): one sample, no axis."""

    FIXTURES = Path(__file__).resolve().parent / 'fixtures' / 'rawfiles'

    @pytest.mark.parametrize('fixture', ['e_ng_op.raw', 'e_xy_op.raw'])
    def test_every_node_readable_as_one_sample(self, fixture):
        """Pre-fix on the ngspice fixture: ``from_rawfile(raw, 'in')``
        raised SimulationError "Signal 'in' not found" — vector 0 IS v(in)
        on an ngspice .op and the scan started at index 1 — and
        ``from_rawfile(raw, 'mid')`` returned x = v(in) = [1.5] with
        xlabel 'v(in)', a node voltage posing as the axis. On the Xyce
        fixture x was the placeholder 'sweep' = [0.0].

        Post-fix, both backends: x = [0.0], xlabel 'op', y = the value."""
        raw = read_rawfile(self.FIXTURES / fixture)
        for node, value in (('in', 1.5), ('mid', 1.0)):
            sig = NgspiceSignal.from_rawfile(raw, node, 'V')
            assert sig.xlabel == 'op'
            np.testing.assert_array_equal(sig.x, [0.0])
            np.testing.assert_allclose(sig.y, [value])
        cur = NgspiceSignal.from_rawfile(raw, 'v1', 'I')
        np.testing.assert_allclose(cur.y, [-0.5e-3])
        assert cur.xlabel == 'op' and len(cur.x) == 1

    def test_op_signals_are_arithmetic_compatible(self):
        """Two nodes of one .op share the ``'op'`` axis, so ``in - mid``
        (the drop across R1: 0.5 V) works like any pair of Signals."""
        raw = read_rawfile(self.FIXTURES / 'e_ng_op.raw')
        vin = NgspiceSignal.from_rawfile(raw, 'in', 'V')
        vmid = NgspiceSignal.from_rawfile(raw, 'mid', 'V')
        np.testing.assert_allclose((vin - vmid).y, [0.5])

    def test_missing_node_on_op_still_raises(self):
        raw = read_rawfile(self.FIXTURES / 'e_ng_op.raw')
        with pytest.raises(SimulationError, match="'nowhere'.*not found"):
            NgspiceSignal.from_rawfile(raw, 'nowhere', 'V')


class TestErrorMessageSplit:
    """A missing x-axis and a missing y-signal must raise distinct errors —
    the conflated message is what historically mis-diagnosed the DC bug."""

    def test_missing_yaxis_names_the_signal(self):
        raw = _make_transient_rawdata()
        with pytest.raises(SimulationError, match=r"bogus_node"):
            NgspiceSignal.from_rawfile(raw, 'bogus_node', 'V')

    def test_missing_xaxis_reports_xaxis_not_signal(self):
        """An empty rawfile (no vectors) reports an x-axis problem, not a
        'signal not found' for a signal the user never got to look for."""
        raw = make_rawdata([])   # nvars == 0, no scale vector
        with pytest.raises(SimulationError, match=r"(?i)x-?axis"):
            NgspiceSignal.from_rawfile(raw, 'vout', 'V')


# ---------------------------------------------------------------------------
# Case-insensitive signal type matching
# ---------------------------------------------------------------------------

class TestCaseInsensitiveTypeMatch:

    def test_lowercase_v_matches_voltage_signal(self):
        """signal_type='v' (lowercase) matches v(out) vector."""
        raw = _make_transient_rawdata()
        sig = NgspiceSignal.from_rawfile(raw, 'out', 'v')
        np.testing.assert_array_equal(sig.y, raw.vectors[1].data)

    def test_uppercase_I_matches_current_signal(self):
        """signal_type='I' (uppercase) matches i(drain) vector."""
        raw = _make_transient_rawdata()
        sig = NgspiceSignal.from_rawfile(raw, 'drain', 'I')
        np.testing.assert_array_equal(sig.y, raw.vectors[2].data)

    def test_lowercase_i_matches_current_signal(self):
        """signal_type='i' (lowercase) matches i(drain) vector."""
        raw = _make_transient_rawdata()
        sig = NgspiceSignal.from_rawfile(raw, 'drain', 'i')
        np.testing.assert_array_equal(sig.y, raw.vectors[2].data)

    def test_mixed_case_type_in_rawfile(self):
        """Vector with uppercase type prefix (V(out)) is matched case-insensitively."""
        t = np.linspace(0, 1e-6, 10)
        raw = make_rawdata([
            ('time',   t,           'time'),
            ('V(out)', np.ones(10), 'voltage'),   # uppercase prefix
        ])
        sig = NgspiceSignal.from_rawfile(raw, 'out', 'V')
        np.testing.assert_array_equal(sig.y, np.ones(10))


# ---------------------------------------------------------------------------
# Inheritance chain
# ---------------------------------------------------------------------------

class TestInheritanceChain:

    def test_pyspice_signal_is_subclass_of_pysignal(self):
        """NgspiceSignal is a subclass of Signal."""
        assert issubclass(NgspiceSignal, Signal)

    def test_instance_is_pyspice_signal_and_pysignal(self, sample_signal):
        """A NgspiceSignal instance satisfies isinstance checks for both classes."""
        assert isinstance(sample_signal, NgspiceSignal)
        assert isinstance(sample_signal, Signal)

    def test_arithmetic_result_is_pyspice_signal(self, sample_signal_pair):
        """Adding two NgspiceSignal instances yields a NgspiceSignal (not base Signal)."""
        sig1, sig2 = sample_signal_pair
        result = sig1 + sig2
        assert type(result) is NgspiceSignal

    def test_arithmetic_result_is_pysignal(self, sample_signal_pair):
        """Result of arithmetic on NgspiceSignal is also an instance of Signal."""
        sig1, sig2 = sample_signal_pair
        result = sig1 + sig2
        assert isinstance(result, Signal)


# ---------------------------------------------------------------------------
# NgspiceSignal.from_rawfile() — case-insensitive node matching (T7 / Xyce)
#
# Xyce upper-cases node names in its rawfile (e.g. 'V(VOUT)', 'FREQUENCY')
# whereas silk requests the lowercase bare name ('vout'). The loader must
# match case-insensitively so the same loader reads both ngspice and Xyce raw.
# ---------------------------------------------------------------------------

class TestFromRawfileCaseInsensitive:

    def _xyce_ac_rawdata(self):
        """Mock a Xyce AC rawfile: upper-cased FREQUENCY + V(VOUT)."""
        f = np.logspace(0, 9, 20)
        return make_rawdata([
            ('FREQUENCY', f,                'frequency'),
            ('V(VOUT)',   np.ones(20) * 2,  'voltage'),
        ])

    def test_uppercase_node_matched_by_lowercase_request(self):
        """A Xyce 'V(VOUT)' vector is found when silk asks for bare 'vout'."""
        raw = self._xyce_ac_rawdata()
        sig = NgspiceSignal.from_rawfile(raw, 'vout', 'V')
        np.testing.assert_array_equal(sig.y, raw.vectors[1].data)

    def test_uppercase_xaxis_still_detected(self):
        """Xyce's upper-cased 'FREQUENCY' is recognized as the x-axis."""
        raw = self._xyce_ac_rawdata()
        sig = NgspiceSignal.from_rawfile(raw, 'vout', 'V')
        np.testing.assert_array_equal(sig.x, raw.vectors[0].data)

    def test_uppercase_xaxis_label_is_folded_to_canonical(self):
        """The Signal's xlabel is the canonical lowercase axis name (TASK-87).

        Pre-fix ``xlabel`` was the verbatim vector-0 name ``'FREQUENCY'``,
        which every axis check in silk.metrics (exact ``== 'frequency'``)
        rejected — no AC or transient metric accepted a Xyce Signal. Only
        the label is folded; the rawfile vector keeps its verbatim name.
        """
        raw = self._xyce_ac_rawdata()
        sig = NgspiceSignal.from_rawfile(raw, 'vout', 'V')
        assert sig.xlabel == 'frequency'
        assert raw.vectors[0].name == 'FREQUENCY'

    def test_lowercase_xaxis_label_is_unchanged(self):
        """ngspice already writes lowercase names; the fold is the identity."""
        raw = _make_transient_rawdata()
        sig = NgspiceSignal.from_rawfile(raw, 'out', 'V')
        assert sig.xlabel == raw.vectors[0].name == 'time'

    def test_uppercase_request_matches_lowercase_vector(self):
        """Symmetry: an upper-case request matches a lower-case ngspice vector."""
        raw = _make_transient_rawdata()  # has lowercase 'v(out)'
        sig = NgspiceSignal.from_rawfile(raw, 'OUT', 'V')
        np.testing.assert_array_equal(sig.y, raw.vectors[1].data)

    def test_uppercase_bare_name_matched(self):
        """Bare (non-wrapped) names match case-insensitively too (noise spectra)."""
        raw = make_rawdata([
            ('FREQUENCY',       np.logspace(0, 6, 10), 'frequency'),
            ('ONOISE_SPECTRUM', np.ones(10) * 3,       'voltage'),
        ])
        sig = NgspiceSignal.from_rawfile(raw, 'onoise_spectrum', 'V')
        np.testing.assert_array_equal(sig.y, raw.vectors[1].data)


# ---------------------------------------------------------------------------
# TASK-214 repro: rms(vout * 0.5) on a real RC transient run.
#
# Before TASK-214, ``Signal.__mul__`` read ``other.name`` / ``.type`` / ``.y``
# unconditionally, so ``vout * 0.5`` raised ``AttributeError`` inside the
# operator. ``add_metric``'s extractor try/except (module_test.py) turned
# that into the string ``"Error: 'float' object has no attribute 'xlabel'"``
# recorded as the metric's *value* instead of a float -- silently corrupting
# the report rather than failing the run. This exercises the real
# add_metric -> run() -> SpecReport path end to end, per AC#5.
# ---------------------------------------------------------------------------

@needs_ngspice
def test_rms_of_signal_times_scalar_in_real_rc_report(tmp_path):
    """rms(vout * 0.5) on a real ngspice RC transient must be a float in
    report.metrics, not an ``"Error: ..."`` string."""
    from silk import Module, load_pdk, sources
    from silk.elements import elements
    from silk.metrics import rms
    from silk.simulation import Transient

    mod = Module('sky130', str(tmp_path), 'rc214.cir', load_pdk('sky130'),
                 module_name='rc214')
    mod.define_module_ports({'vin': 'in', 'vout': 'out', 'vss': 'ground'})
    mod.add(elements.resistor('r1', p='vin', n='vout', value=1e3))
    mod.add(elements.capacitor('c1', p='vout', n='vss', value=1e-9))
    tb = mod.create_test('step')
    tb.add_module_under_test({'vin': 'vin', 'vout': 'vout', 'vss': '0'})
    tb.add_source(sources.VoltageSource('vsrc', positive='vin', negative='0', dc=1.0))
    tb.add_sim_type(Transient(step=1e-8, stop=5e-6))
    tb.add_metric('rms_half', lambda c: rms(c.load_signal('vout') * 0.5))
    tb.add_metric('rms_ref', lambda c: rms(c.load_signal('vout')))

    report = tb.run(corner='tt')
    mod.close()

    assert isinstance(report.metrics['rms_half'], float)
    assert not str(report.metrics['rms_half']).startswith('Error')
    assert report.metrics['rms_half'] == pytest.approx(0.5 * report.metrics['rms_ref'])
