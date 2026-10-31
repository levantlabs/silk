"""``add_metric(signals={...})`` — binding N signal inputs by NAME (TASK-198).

Eight metrics in the catalog
take more than one ``Signal``, and before this the declarative form could pass
exactly one, so all eight were reachable only through the callable escape
hatch — the API half of TASK-183.

Two things are proved here:

**The binding is by name, and it is checked at DECLARATION time.** A missing
or misspelled slot raises out of ``add_metric`` itself, before any simulation,
with a message that names the slot — not a ``TypeError`` out of the deferred
extractor after a run has been paid for (§6.2). The fast class below asserts
the four §6.2 messages verbatim.

**The eight evaluate to correct numbers on a real testbench.** The two slow
legs run real ngspice against sky130: one transient bench (an inverter, a
buffer and a ramped buffer pair) covering the seven ``tran`` metrics, and one
AC bench (two 5T OTAs, one differentially driven and one common-mode driven)
covering ``cmrr_dB``. Each declared metric is cross-checked against the same
metric called directly on the same loaded signals, which is what makes a
swapped pair — the failure mode a positional binding would have — visible.
"""
from __future__ import annotations

import functools
import os
import shutil

import numpy as np
import pytest

from silk import sources
from silk.circuit.circuit_elements import PortDir
from silk.circuit.module_test import (
    _bind_signal_inputs, _metric_signal_slots, _slot_inventory,
)
from silk.circuit.modules import Module
from silk.pdk.basepdk import BasePDK
from silk.signals.base import Signal
from silk.signals.schema import get_metric, metric_schema
from silk.simulation import analysis

_PDK_YAML = os.path.join(os.path.dirname(__file__), '..', 'silk', 'pdk', 'data',
                         'skywater_130_device_details.yaml')
_SKY130_LIB = os.path.expanduser(
    '~/pdk/sky130A/libs.tech/ngspice/sky130.lib.spice')

#: The eight metrics TASK-183 is about (plus TASK-369's evm_percent and TASK-397's iip3_dbm), with a binding and the params each
#: needs. MEASURED against the schema by ``test_these_are_all_of_them``, so
#: this list cannot silently drift out of date.
MULTI_SIGNAL = {
    'propagation_delay':        (['sig_in', 'sig_out'], {}),
    'average_power':            (['sig_v', 'sig_i'], {}),
    'dynamic_energy_per_cycle': (['sig_v', 'sig_i'], {'period': 1e-9}),
    'error_norm':               (['sig', 'goal'], {}),
    'relative_error_norm':      (['sig', 'goal'], {}),
    'hysteresis':               (['sig_in_rising', 'sig_out_rising',
                                  'sig_in_falling', 'sig_out_falling'], {}),
    'cmrr_dB':                  (['vdiff', 'vcm'], {}),
    'evm_percent':              (['i_sig', 'q_sig'],
                                 {'symbol_period': 1e-9, 'modulation': 'qpsk'}),
    'iip3_dbm':                 (['out_sig', 'in_sig'],
                                 {'tclk': 1e-9, 'f1_hz': 1e6, 'f2_hz': 1.1e6}),
    'band_noise_figure':        (['noise_in', 'noise_out'],
                                 {'f1': 1.0, 'f2': 1e6, 'rs': 1e3,
                                  'temp_sim_k': 300.15}),
}


# ---------------------------------------------------------------------------
# Fast legs — no simulator, no PDK install
# ---------------------------------------------------------------------------

@pytest.fixture
def tb(pdk, tmp_dir):
    """A minimal testbench. Nothing here is ever simulated: every fast test
    stops at ``add_metric`` or drives the extractor it built by hand."""
    mod = Module('PDK_skywater130', tmp_dir, 'dut.cir', pdk_details=pdk,
                 module_name='dut')
    mod.define_module_ports({'a': PortDir.IN, 'b': PortDir.OUT,
                             'vdd': PortDir.POWER, 'vss': PortDir.GROUND})
    with mod:
        pdk.nmos1p8('mn', d='b', g='a', s='vss', b='vss', w=1.0, l=0.15)
    return mod.create_test('t')


class _FakeCkt:
    """Stands in for the simulated Module the extractor is handed. Each path
    loads to a distinct object, so which path landed in which slot is
    observable."""

    def __init__(self):
        self.loaded = []

    def load_signal(self, path):
        self.loaded.append(path)
        return f'<signal {path}>'


def _spy_for(name, monkeypatch):
    """Replace metric *name* in the registry with a recording stand-in that
    keeps the real signature (``functools.wraps`` carries ``__wrapped__``, so
    ``inspect.signature`` — and therefore the schema — is unchanged)."""
    import silk.signals.schema as schema_mod
    real = get_metric(name)
    calls = []

    @functools.wraps(real)
    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return 0.0

    monkeypatch.setitem(schema_mod._REGISTRY, name, spy)
    return calls


def test_these_are_all_of_them():
    """MULTI_SIGNAL is exactly the catalog's multi-signal set — if a ninth
    metric arrives, this file is told about it."""
    from silk.signals.schema import export_metric_schema
    schema = export_metric_schema(include_metadata=False)
    found = {n: [s[0] for s in _metric_signal_slots(get_metric(n))]
             for n, v in schema.items() if len(v['signal_inputs']) > 1}
    assert found == {n: slots for n, (slots, _) in MULTI_SIGNAL.items()}


@pytest.mark.parametrize('metric_name', sorted(MULTI_SIGNAL))
def test_every_multi_signal_metric_binds_each_slot_by_name(
        metric_name, tb, monkeypatch):
    """Each of the eight is declarable, and every slot receives the path it
    was bound to — by keyword, so no slot can be filled by position."""
    slots, params = MULTI_SIGNAL[metric_name]
    calls = _spy_for(metric_name, monkeypatch)
    binding = {slot: f'net_{slot}' for slot in slots}

    tb.add_metric('m', type=metric_name, signals=binding, **params)
    ckt = _FakeCkt()
    value = tb._metrics[-1]['extractor'](ckt)

    assert value == 0.0
    assert ckt.loaded == [f'net_{slot}' for slot in slots]
    (args, kwargs), = calls
    assert args == ()          # everything bound by keyword
    assert kwargs == {slot: f'<signal net_{slot}>' for slot in slots} | params


def test_signals_on_a_one_input_metric_is_accepted(tb, monkeypatch):
    """§6.2, case four: ``signals={'sig': path}`` is the explicit spelling of
    ``signal=`` and is NOT an error."""
    calls = _spy_for('dc_gain_dB', monkeypatch)
    tb.add_metric('g', type='dc_gain_dB', signals={'sig': 'xdut.vout'})
    tb._metrics[-1]['extractor'](_FakeCkt())
    assert calls == [((), {'sig': '<signal xdut.vout>'})]


def test_signal_on_a_one_input_metric_is_unchanged(tb, monkeypatch):
    """The one-input form every existing testbench uses is untouched, and
    reaches the metric identically to the ``signals=`` spelling above."""
    calls = _spy_for('dc_gain_dB', monkeypatch)
    tb.add_metric('g', type='dc_gain_dB', signal='xdut.vout')
    tb._metrics[-1]['extractor'](_FakeCkt())
    assert calls == [((), {'sig': '<signal xdut.vout>'})]


class TestTheFourCasesOfSection62:
    """The four cases of design §6.2, each message asserted verbatim."""

    def test_missing_slot_names_the_slot_all_slots_and_what_was_bound(self, tb):
        with pytest.raises(TypeError) as exc:
            tb.add_metric('tpd', type='propagation_delay',
                          signals={'sig_in': 'xdut.vin'})
        assert str(exc.value) == (
            "add_metric('tpd', type='propagation_delay'): signals= is missing "
            "'sig_out'. This metric takes 2 signal inputs: 'sig_in' (kind "
            "'transient'), 'sig_out' (kind 'transient'); bound: 'sig_in'."
        )

    def test_unknown_slot_names_the_key_and_the_valid_slots(self, tb):
        with pytest.raises(TypeError) as exc:
            tb.add_metric('tpd', type='propagation_delay',
                          signals={'sig_in': 'xdut.vin',
                                   'sig_output': 'xdut.vout'})
        assert str(exc.value) == (
            "add_metric('tpd', type='propagation_delay'): signals= has "
            "unknown key 'sig_output'; this metric's signal inputs are "
            "'sig_in', 'sig_out'."
        )

    def test_signal_on_a_multi_input_metric_points_at_signals(self, tb):
        with pytest.raises(TypeError) as exc:
            tb.add_metric('cmrr', type='cmrr_dB', signal='xdut.vout')
        assert str(exc.value) == (
            "add_metric('cmrr', type='cmrr_dB'): signal= binds a single "
            "input, but this metric takes 2: 'vdiff', 'vcm'. Use "
            "signals={'vdiff': ..., 'vcm': ...}."
        )

    def test_signals_on_a_one_input_metric_is_not_an_error(self, tb):
        tb.add_metric('g', type='dc_gain_dB', signals={'sig': 'xdut.vout'})
        assert tb._metrics[-1]['name'] == 'g'

    def test_hysteresis_missing_slots_are_all_named(self, tb):
        """The four-slot case the design calls out: 'needs 4 signals' would
        not tell the user which two they forgot."""
        with pytest.raises(TypeError) as exc:
            tb.add_metric('h', type='hysteresis',
                          signals={'sig_in_rising': 'a', 'sig_out_rising': 'b'})
        assert str(exc.value) == (
            "add_metric('h', type='hysteresis'): signals= is missing "
            "'sig_in_falling', 'sig_out_falling'. This metric takes 4 signal "
            "inputs: 'sig_in_rising' (kind 'transient'), 'sig_out_rising' "
            "(kind 'transient'), 'sig_in_falling' (kind 'transient'), "
            "'sig_out_falling' (kind 'transient'); bound: 'sig_in_rising', "
            "'sig_out_rising'."
        )


class TestDeclarationTimeArgumentMisuse:
    """The rest of the ``add_metric`` argument family, all at declaration
    time and all ``TypeError`` — the type ``add_metric`` already raises."""

    def test_nothing_is_appended_when_the_binding_is_refused(self, tb):
        """The refusal leaves no half-declared metric behind."""
        before = list(tb._metrics)
        with pytest.raises(TypeError):
            tb.add_metric('tpd', type='propagation_delay',
                          signals={'sig_in': 'xdut.vin'})
        assert tb._metrics == before

    def test_declaration_time_not_extraction_time(self, tb, monkeypatch):
        """The point of §6.2: the error arrives without a simulation. The
        metric function is never even called — the spy records nothing."""
        calls = _spy_for('propagation_delay', monkeypatch)
        with pytest.raises(TypeError):
            tb.add_metric('tpd', type='propagation_delay',
                          signals={'sig_in': 'xdut.vin'})
        assert calls == []

    def test_both_signal_and_signals_is_refused(self, tb):
        with pytest.raises(TypeError, match='not both'):
            tb.add_metric('g', type='dc_gain_dB', signal='a',
                          signals={'sig': 'a'})

    def test_neither_on_a_one_input_metric_keeps_its_message(self, tb):
        with pytest.raises(TypeError) as exc:
            tb.add_metric('g', type='dc_gain_dB')
        assert str(exc.value) == 'add_metric: declarative form requires signal='

    def test_neither_on_a_multi_input_metric_lists_the_slots(self, tb):
        with pytest.raises(TypeError) as exc:
            tb.add_metric('tpd', type='propagation_delay')
        assert str(exc.value) == (
            "add_metric('tpd', type='propagation_delay'): declarative form "
            "requires signals=, binding each of this metric's 2 signal "
            "inputs: 'sig_in' (kind 'transient'), 'sig_out' (kind "
            "'transient')."
        )

    def test_signals_must_be_a_dict(self, tb):
        with pytest.raises(TypeError, match='must be a dict'):
            tb.add_metric('tpd', type='propagation_delay',
                          signals=['xdut.vin', 'xdut.vout'])

    def test_a_slot_name_passed_as_a_scalar_param_is_refused(self, tb):
        """``**params`` and the binding are merged into one call; a name in
        both would otherwise be a duplicate-keyword TypeError from inside the
        deferred lambda, long after the mistake."""
        with pytest.raises(TypeError, match="names a signal input"):
            tb.add_metric('tpd', type='propagation_delay',
                          signals={'sig_in': 'a', 'sig_out': 'b'}, sig_in=3)


class TestTheTDSeam:
    """``_metric_signal_slots`` is the one place the ``signal_inputs`` SHAPE
    is read. T-D (TASK-196) landed: the live schema publishes ``{'name',
    'kind'}`` objects, and the seam held — the one ``isinstance`` branch was
    the whole change here, and every §6.2 message gained its kinds with no
    edit to the message code at all (only to the strings asserted above)."""

    def test_the_live_catalog_is_v2(self):
        assert _metric_signal_slots(get_metric('propagation_delay')) == [
            ('sig_in', 'transient'), ('sig_out', 'transient')]

    def test_the_v1_branch_still_reads_bare_strings(self, monkeypatch):
        """Kept, and kept tested. It costs one line, and it is what lets this
        accessor read a schema exported by an older silk (a pinned install, a
        cached export) instead of raising a TypeError on a str."""
        monkeypatch.setattr(
            'silk.signals.schema.metric_schema',
            lambda fn: {'signal_inputs': ['sig_in', 'sig_out']})
        slots = _metric_signal_slots(get_metric('propagation_delay'))
        assert slots == [('sig_in', None), ('sig_out', None)]
        assert _slot_inventory(slots) == "'sig_in', 'sig_out'"

    def test_v2_objects_with_kinds(self, monkeypatch):
        monkeypatch.setattr(
            'silk.signals.schema.metric_schema',
            lambda fn: {'signal_inputs': [{'name': 'sig_in', 'kind': 'transient'},
                                          {'name': 'sig_out', 'kind': 'transient'}]})
        slots = _metric_signal_slots(get_metric('propagation_delay'))
        assert slots == [('sig_in', 'transient'), ('sig_out', 'transient')]
        assert _slot_inventory(slots) == (
            "'sig_in' (kind 'transient'), 'sig_out' (kind 'transient')")

    def test_the_missing_slot_message_carries_the_kind_when_there_is_one(
            self, monkeypatch):
        """The §6.2 message as the design writes it, once T-D lands."""
        monkeypatch.setattr(
            'silk.signals.schema.metric_schema',
            lambda fn: {'signal_inputs': [{'name': 'sig_in', 'kind': 'transient'},
                                          {'name': 'sig_out', 'kind': 'transient'}]})
        with pytest.raises(TypeError) as exc:
            _bind_signal_inputs('tpd', 'propagation_delay',
                                get_metric('propagation_delay'),
                                None, {'sig_in': 'xdut.vin'}, {})
        assert str(exc.value) == (
            "add_metric('tpd', type='propagation_delay'): signals= is missing "
            "'sig_out'. This metric takes 2 signal inputs: 'sig_in' (kind "
            "'transient'), 'sig_out' (kind 'transient'); bound: 'sig_in'."
        )


# ---------------------------------------------------------------------------
# Slow legs — real ngspice, real sky130
# ---------------------------------------------------------------------------

slow_sim = pytest.mark.skipif(
    not os.path.exists(_SKY130_LIB) or shutil.which('ngspice') is None,
    reason='needs ngspice and the sky130 ngspice models',
)

_PERIOD = 10e-9
_STOP = 100e-9


def _inverter(pdk, tag, i, o):
    pdk.nmos1p8(f'mn_{tag}', d=o, g=i, s='vss', b='vss', w=1.0, l=0.15)
    pdk.pmos1p8(f'mp_{tag}', d=o, g=i, s='vdd', b='vdd', w=2.0, l=0.15)


def _tran_bench(pdk, tmp_dir):
    """Three stimuli in one deck: a pulse-driven inverter and buffer (delay,
    offset, error, power), and a buffer pair driven by a 100 ns ramp up and a
    100 ns ramp down (hysteresis's four slots, four distinct nodes)."""
    mod = Module('PDK_skywater130', tmp_dir, 'bench.cir', pdk_details=pdk,
                 module_name='bench')
    mod.define_module_ports({
        'vin': PortDir.IN, 'vout_inv': PortDir.OUT, 'vout_buf': PortDir.OUT,
        'vin_up': PortDir.IN, 'vout_up': PortDir.OUT,
        'vin_dn': PortDir.IN, 'vout_dn': PortDir.OUT,
        'vdd': PortDir.POWER, 'vss': PortDir.GROUND})
    for n in ('mid', 'mid_up', 'mid_dn'):
        mod.add_net(n)
    with mod:
        _inverter(pdk, 'i0', 'vin', 'vout_inv')
        _inverter(pdk, 'b0', 'vin', 'mid')
        _inverter(pdk, 'b1', 'mid', 'vout_buf')
        _inverter(pdk, 'u0', 'vin_up', 'mid_up')
        _inverter(pdk, 'u1', 'mid_up', 'vout_up')
        _inverter(pdk, 'd0', 'vin_dn', 'mid_dn')
        _inverter(pdk, 'd1', 'mid_dn', 'vout_dn')

    tb = mod.create_test('tran')
    tb.add_module_under_test({
        'vin': 'vin', 'vout_inv': 'vout_inv', 'vout_buf': 'vout_buf',
        'vin_up': 'vin_up', 'vout_up': 'vout_up',
        'vin_dn': 'vin_dn', 'vout_dn': 'vout_dn',
        'vdd': 'vdd', 'vss': '0'})
    tb.add(sources.vdc('vsup', positive='vdd', negative='0', dc=1.8))
    tb.add(sources.vpulse('vin', positive='vin', negative='0', v1=0.0, v2=1.8,
                          td=1e-9, tr=100e-12, tf=100e-12, pw=4e-9, per=_PERIOD))
    tb.add(sources.vpulse('vinup', positive='vin_up', negative='0', v1=0.0,
                          v2=1.8, td=0.0, tr=_STOP, tf=1e-12, pw=1e-12,
                          per=10 * _STOP))
    tb.add(sources.vpulse('vindn', positive='vin_dn', negative='0', v1=1.8,
                          v2=0.0, td=0.0, tr=_STOP, tf=1e-12, pw=1e-12,
                          per=10 * _STOP))
    tb.add_sim_type(analysis.Transient(step=10e-12, stop=_STOP))
    return tb


@pytest.mark.slow
@slow_sim
def test_slow_seven_transient_metrics_through_signals(tmp_dir):
    """The seven ``tran`` multi-signal metrics, declared through ``signals=``
    and measured on real ngspice output. Every value is cross-checked against
    the same metric called directly on the same loaded signals, which is what
    catches a slot bound to the wrong node."""
    pdk = BasePDK(_PDK_YAML, library_location=_SKY130_LIB)
    tb = _tran_bench(pdk, tmp_dir)

    tb.add_metric('tpd', type='propagation_delay',
                  signals={'sig_in': 'vin', 'sig_out': 'vout_inv'}, in_edge='any')
    # comparator_prop_delay's old binding (TASK-363 removed it): rising edges,
    # each side at its own mid-swing — propagation_delay's defaults.
    tb.add_metric('tpd_cmp', type='propagation_delay',
                  signals={'sig_in': 'vin', 'sig_out': 'vout_buf'})
    tb.add_metric('relerr', type='relative_error_norm',
                  signals={'sig': 'vout_buf', 'goal': 'vin'},
                  kind='rms', floor=0.1)
    tb.add_metric('hyst', type='hysteresis',
                  signals={'sig_in_rising': 'vin_up', 'sig_out_rising': 'vout_up',
                           'sig_in_falling': 'vin_dn', 'sig_out_falling': 'vout_dn'})
    tb.add_metric('err', type='error_norm',
                  signals={'sig': 'vout_buf', 'goal': 'vin'}, kind='rms')
    tb.add_metric('pavg', type='average_power',
                  signals={'sig_v': 'vdd', 'sig_i': 'i(vvsup)'})
    tb.add_metric('ecyc', type='dynamic_energy_per_cycle',
                  signals={'sig_v': 'vdd', 'sig_i': 'i(vvsup)'}, period=_PERIOD)

    report = tb.run('tt')
    got = report.metrics
    res = report.sim_results

    # Physically right, not merely finite (ngspice-42 / sky130A, corner tt).
    assert got['tpd'] == pytest.approx(12.8e-12, rel=0.5)      # inverter, 1x/2x
    assert got['tpd_cmp'] == pytest.approx(41.1e-12, rel=0.5)  # through a buffer
    # A relative rms error between the buffer output and its input: positive
    # and not nan (a nan would fail this comparison). The exact value is
    # checked against the direct call below.
    assert got['relerr'] > 0.0
    # A plain buffer has no hysteresis; what is left is the ramp lag, a few mV.
    assert 0.0 < got['hyst'] < 0.05
    assert got['err'] == pytest.approx(0.135, rel=0.5)
    # ngspice's source-branch current is negative while the supply delivers.
    assert -1e-3 < got['pavg'] < 0.0
    # The trace is exactly 10 whole periods, so energy/cycle IS P*T (TASK-61).
    assert got['ecyc'] == pytest.approx(got['pavg'] * _PERIOD, rel=1e-12)

    direct = {
        'tpd': get_metric('propagation_delay')(
            res.load_signal('vin'), res.load_signal('vout_inv'), in_edge='any'),
        'tpd_cmp': get_metric('propagation_delay')(
            res.load_signal('vin'), res.load_signal('vout_buf')),
        'relerr': get_metric('relative_error_norm')(
            res.load_signal('vout_buf'), res.load_signal('vin'),
            kind='rms', floor=0.1),
        'hyst': get_metric('hysteresis')(
            res.load_signal('vin_up'), res.load_signal('vout_up'),
            res.load_signal('vin_dn'), res.load_signal('vout_dn')),
        'err': get_metric('error_norm')(
            res.load_signal('vout_buf'), res.load_signal('vin'), kind='rms'),
        'pavg': get_metric('average_power')(
            res.load_signal('vdd'), res.load_signal('i(vvsup)')),
        'ecyc': get_metric('dynamic_energy_per_cycle')(
            res.load_signal('vdd'), res.load_signal('i(vvsup)'),
            period=_PERIOD),
    }
    assert got == direct


def _ota(pdk, tag, inp, inm, out):
    """A 5T OTA, the one from silk/examples/ota_5t_guide.py, built inline."""
    pdk.nmos1p8(f'm1_{tag}', d=f'vint_{tag}', g=inp, s=f'tail_{tag}', b='vss',
                w=4.0, l=0.5)
    pdk.nmos1p8(f'm2_{tag}', d=out, g=inm, s=f'tail_{tag}', b='vss',
                w=4.0, l=0.5)
    pdk.pmos1p8(f'm3_{tag}', d=f'vint_{tag}', g=f'vint_{tag}', s='vdd', b='vdd',
                w=8.0, l=0.5)
    pdk.pmos1p8(f'm4_{tag}', d=out, g=f'vint_{tag}', s='vdd', b='vdd',
                w=8.0, l=0.5)
    pdk.nmos1p8(f'm5_{tag}', d=f'tail_{tag}', g='ibias', s='vss', b='vss',
                w=6.0, l=1.0)


@pytest.mark.slow
@slow_sim
def test_slow_cmrr_through_signals(tmp_dir):
    """``cmrr_dB``, the metric TASK-183 names first, on a real AC run.

    CMRR needs a differential-mode gain AND a common-mode gain, i.e. two
    drives — so the deck carries two copies of the same 5T OTA, one driven
    single-endedly (its output is the differential-mode gain to within
    1/CMRR) and one with both inputs tied to the same source (its output is
    the common-mode gain). ``signals=`` binds one output to each slot."""
    pdk = BasePDK(_PDK_YAML, library_location=_SKY130_LIB)
    mod = Module('PDK_skywater130', tmp_dir, 'cmrr.cir', pdk_details=pdk,
                 module_name='cmrr_bench')
    mod.define_module_ports({
        'vinp_d': PortDir.IN, 'vinm_d': PortDir.IN, 'vin_c': PortDir.IN,
        'voutd': PortDir.OUT, 'voutc': PortDir.OUT, 'ibias': PortDir.IN,
        'vdd': PortDir.POWER, 'vss': PortDir.GROUND})
    for n in ('vint_d', 'tail_d', 'vint_c', 'tail_c'):
        mod.add_net(n)
    with mod:
        _ota(pdk, 'd', 'vinp_d', 'vinm_d', 'voutd')
        _ota(pdk, 'c', 'vin_c', 'vin_c', 'voutc')

    tb = mod.create_test('ac')
    tb.add_module_under_test({
        'vinp_d': 'vinp_d', 'vinm_d': 'vinm_d', 'vin_c': 'vin_c',
        'voutd': 'voutd', 'voutc': 'voutc', 'ibias': 'ibias',
        'vdd': 'vdd', 'vss': '0'})
    tb.add(sources.vdc('vsup', positive='vdd', negative='0', dc=1.8))
    tb.add(sources.vdc('vbias', positive='ibias', negative='0', dc=0.9))
    tb.add(sources.VoltageSource('vd', positive='vinp_d', negative='0',
                                 dc=0.9, ac=1.0))
    tb.add(sources.VoltageSource('vdm', positive='vinm_d', negative='0', dc=0.9))
    tb.add(sources.VoltageSource('vc', positive='vin_c', negative='0',
                                 dc=0.9, ac=1.0))
    tb.add_sim_type(analysis.AC(variation='dec', points=10,
                                fstart=1e2, fstop=1e7))
    tb.add_metric('cmrr', type='cmrr_dB',
                  signals={'vdiff': 'voutd', 'vcm': 'voutc'}, freq=1e3)

    report = tb.run('tt')
    res = report.sim_results
    vd = res.load_signal('voutd')
    vc = res.load_signal('voutc')
    i_1k = int(np.argmin(np.abs(vd.x - 1e3)))
    ad_dB = 20 * np.log10(np.abs(vd.y[i_1k]))
    acm_dB = 20 * np.log10(np.abs(vc.y[i_1k]))

    # 34.9 dB of differential gain against -8.1 dB of common-mode gain
    # (ngspice-42 / sky130A, tt) — a textbook 5T OTA on an NMOS tail.
    assert ad_dB == pytest.approx(34.9, abs=3.0)
    assert acm_dB == pytest.approx(-8.1, abs=3.0)
    assert report.metrics['cmrr'] == pytest.approx(ad_dB - acm_dB, abs=1e-9)
    assert report.metrics['cmrr'] == get_metric('cmrr_dB')(vd, vc, freq=1e3)
    # Swapping the two slots is the mistake a positional binding invites; it
    # is a different number, not an error, which is why the binding is by name.
    assert get_metric('cmrr_dB')(vc, vd, freq=1e3) == pytest.approx(
        -report.metrics['cmrr'], abs=1e-9)
