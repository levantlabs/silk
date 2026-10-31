"""Unit tests for net current annotation (spec 2026-07-03) — synthetic data,
no simulator. Probes P1/P1b/P2 (slow) live in test_current_probes.py.
"""
import numpy as np
import pytest

from silk.circuit.net import PortBinding
from tests._env import scratch_dir


# ---------------------------------------------------------------------------
# Trapezoidal window helper (§4.1 window semantics — time-weighted, NOT the
# silk.metrics uniform-resample convention)
# ---------------------------------------------------------------------------
class TestTrapezoidWindow:
    def test_constant_signal(self):
        from silk.circuit.modules_results import _trapezoid_window
        t = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.5, 0.5, 0.5, 0.5])
        avg, rms, peak = _trapezoid_window(t, y, None)
        assert avg == pytest.approx(0.5)
        assert rms == pytest.approx(0.5)
        assert peak == pytest.approx(0.5)

    def test_avg_is_signed(self):
        from silk.circuit.modules_results import _trapezoid_window
        t = np.array([0.0, 1.0, 2.0])
        y = np.array([-0.3, -0.3, -0.3])
        avg, rms, peak = _trapezoid_window(t, y, None)
        assert avg == pytest.approx(-0.3)   # signed average preserved
        assert rms == pytest.approx(0.3)    # RMS is magnitude
        assert peak == pytest.approx(0.3)   # peak = max|i|

    def test_nonuniform_timesteps_trapezoid_not_sample_mean(self):
        """On ngspice's non-uniform grid the sample mean is wrong; the
        time-weighted trapezoid is right (§4.1)."""
        from silk.circuit.modules_results import _trapezoid_window
        t = np.array([0.0, 1.0, 10.0])
        y = np.array([0.0, 0.0, 1.0])
        avg, _, _ = _trapezoid_window(t, y, None)
        assert avg == pytest.approx(0.45)          # trapezoid
        assert avg != pytest.approx(1.0 / 3.0)     # not the sample mean

    def test_peak_is_max_abs(self):
        from silk.circuit.modules_results import _trapezoid_window
        t = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.1, -0.9, 0.2, 0.3])
        _, _, peak = _trapezoid_window(t, y, None)
        assert peak == pytest.approx(0.9)

    def test_tstart_filters_window(self):
        from silk.circuit.modules_results import _trapezoid_window
        t = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([10.0, 1.0, 1.0, 1.0])       # big transient before tstart
        avg, _, peak = _trapezoid_window(t, y, tstart=1.0)
        assert avg == pytest.approx(1.0)
        assert peak == pytest.approx(1.0)          # 10.0 excluded

    def test_tstart_none_is_full_trace(self):
        from silk.circuit.modules_results import _trapezoid_window
        t = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([10.0, 1.0, 1.0, 1.0])
        _, _, peak = _trapezoid_window(t, y, None)
        assert peak == pytest.approx(10.0)

    def test_tstart_past_end_raises(self):
        from silk.circuit.modules_results import _trapezoid_window
        t = np.array([0.0, 1.0, 2.0])
        y = np.array([1.0, 1.0, 1.0])
        with pytest.raises(ValueError):
            _trapezoid_window(t, y, tstart=5.0)


class TestPortBindingFields:
    def test_new_current_fields_default_none(self):
        b = PortBinding(net='n', instance='m1', port='d')
        assert b.avg_current is None
        assert b.peak_current is None
        assert b.rms_current is None
        assert b.current_vector is None
        assert b._current_source is None

    def test_existing_fields_unchanged(self):
        b = PortBinding(net='n', instance='m1', port='d', kind='device')
        assert b.net == 'n' and b.instance == 'm1' and b.port == 'd'
        assert b.kind == 'device'


def test_not_supported_error_is_circuit_error():
    from silk.exceptions import NotSupportedError, CircuitError
    assert issubclass(NotSupportedError, CircuitError)


# ---------------------------------------------------------------------------
# Emitted-instance-name helper (§6.2 — single source of truth for leaf names)
# ---------------------------------------------------------------------------
class TestEmittedInstanceName:
    def test_pdk_device_prefix(self):
        from silk.circuit.emitted_names import emitted_instance_name
        from silk.pdk.basepdk import BasePDK
        import os
        pdk = BasePDK(
            os.path.join('silk', 'pdk', 'data', 'skywater_130_device_details.yaml'),
            library_location='/fake')
        dev = pdk.nmos1p8('mn', d='o', g='i', s='g', b='g', w=1, l=0.15)
        assert emitted_instance_name(dev, 'mn') == 'XMmn'

    def test_passive_prefix(self):
        from silk.circuit.emitted_names import emitted_instance_name
        from silk.elements import elements
        assert emitted_instance_name(elements.resistor('rl', p='a', n='b', value=1), 'rl') == 'Rrl'
        assert emitted_instance_name(elements.capacitor('cl', p='a', n='b', value=1), 'cl') == 'Ccl'
        assert emitted_instance_name(elements.inductor('ll', p='a', n='b', value=1), 'll') == 'Lll'

    def test_source_prefix(self):
        from silk.circuit.emitted_names import emitted_instance_name
        from silk import sources
        vs = sources.VoltageSource('vdd', positive='v', negative='0', dc=1)
        assert emitted_instance_name(vs, 'vdd') == 'Vvdd'

    def test_submodule_prefix(self):
        from silk.circuit.emitted_names import emitted_instance_name
        from silk.circuit.modules import Module
        sub = Module('p', '/tmp', 's.cir', pdk_details=None, module_name='sub')
        assert emitted_instance_name(sub, 'dut') == 'xdut'

    def test_pdk_device_via_add_element(self):
        """add_element(factory, name, nets=...) stores a PDKDevice (NOT an
        AnalogElement); the netlister emits it as 'XM'+name
        (basepdk.generate_netlist_entry). Verified against the OTA rawfile."""
        import os
        from silk.circuit.emitted_names import emitted_instance_name
        from silk.circuit.modules import Module
        from silk.circuit.circuit_elements import PortDir
        from silk.pdk.basepdk import PDKDevice
        from silk.pdk.basepdk import BasePDK
        pdk = BasePDK(
            os.path.join('silk', 'pdk', 'data', 'skywater_130_device_details.yaml'),
            library_location='/fake')
        m = Module('m', scratch_dir(), 'm.cir', pdk_details=pdk, module_name='sub')
        m.define_module_ports({'d': PortDir.OUT, 'g': PortDir.IN,
                               's': PortDir.GROUND, 'b': PortDir.GROUND})
        m.add_element(pdk.nmos1p8, 'M1', nets={'d': 'd', 'g': 'g', 's': 's', 'b': 'b'},
                      parameters={'w': 1.0, 'l': 0.15, 'm': 1})
        dev = m.elem_dict['M1']['device']
        assert isinstance(dev, PDKDevice)
        assert emitted_instance_name(dev, 'M1') == 'XMM1'


class TestCurrentRowPDKDevice:
    """§6.3 corrected against ngspice reality (see docs/backlog/2026-07-03-...):
    only the DRAIN [id] is reliably populated for sky130 subckt MOS, and pmos
    [id] carries the device-current sign (opposite into-net sign to nmos)."""

    def _mos(self, factory_name, nets=None):
        import os
        from silk.circuit.modules import Module
        from silk.circuit.circuit_elements import PortDir
        from silk.pdk.basepdk import BasePDK
        pdk = BasePDK(
            os.path.join('silk', 'pdk', 'data', 'skywater_130_device_details.yaml'),
            library_location='/fake')
        m = Module('m', scratch_dir(), 'm.cir', pdk_details=pdk, module_name='sub')
        m.define_module_ports({'d': PortDir.OUT, 'g': PortDir.IN,
                               's': PortDir.GROUND, 'b': PortDir.GROUND})
        m.add_element(getattr(pdk, factory_name), 'M1',
                      nets={'d': 'd', 'g': 'g', 's': 's', 'b': 'b'},
                      parameters={'w': 1.0, 'l': 0.15, 'm': 1})
        return m.elem_dict['M1']['device']

    def test_nmos_drain_only_negative_sign(self):
        from silk.circuit.modules_results import _current_row
        dev = self._mos('nmos1p8')
        # drain: nmos into-net = -raw
        assert _current_row(dev, 'd') == ('id', -1.0)
        # source/gate/bulk: NOT annotated (ngspice gives only [id]) → None
        assert _current_row(dev, 's') is None
        assert _current_row(dev, 'g') is None
        assert _current_row(dev, 'b') is None

    def test_pmos_drain_positive_sign(self):
        from silk.circuit.modules_results import _current_row
        dev = self._mos('pmos1p8')
        # drain: pmos into-net = +raw (device-current sign flip)
        assert _current_row(dev, 'd') == ('id', +1.0)
        assert _current_row(dev, 's') is None


# ---------------------------------------------------------------------------
# Backend seam #5 — device_current_vectors (§6.1/§6.2)
# ---------------------------------------------------------------------------
class _FakeVec:
    def __init__(self, name):
        self.name = name


class _FakeRaw:
    """Mimics the rawfile reader's result: .nvars + .vectors[i].name."""
    def __init__(self, names):
        self.vectors = [_FakeVec(n) for n in names]
        self.nvars = len(names)


def _raw(names):
    from silk.simulation.backend import RawResult
    return RawResult(native=_FakeRaw(names), source_path='x.raw')


class TestDeviceCurrentVectorsSeam:
    def test_base_raises_not_supported(self):
        from silk.simulation.backend import XyceBackend
        from silk.exceptions import NotSupportedError
        import pytest
        with pytest.raises(NotSupportedError):
            XyceBackend().device_current_vectors(_raw([]), 'xdut', 'XMmn')

    def test_ngspice_subckt_mos_all_terminals(self):
        from silk.simulation.backend import NgspiceBackend
        names = [
            'time', 'v(vout)',
            'i(@m.xdut.xmmn.msky130_fd_pr__nfet_01v8[id])',
            'i(@m.xdut.xmmn.msky130_fd_pr__nfet_01v8[is])',
            'i(@m.xdut.xmmn.msky130_fd_pr__nfet_01v8[ig])',
            'i(@m.xdut.xmmn.msky130_fd_pr__nfet_01v8[ib])',
        ]
        got = NgspiceBackend().device_current_vectors(_raw(names), 'xdut', 'XMmn')
        assert got['id'] == 'i(@m.xdut.xmmn.msky130_fd_pr__nfet_01v8[id])'
        assert set(got) == {'id', 'is', 'ig', 'ib'}

    def test_ngspice_top_level_resistor(self):
        from silk.simulation.backend import NgspiceBackend
        names = ['time', 'v(vout)', 'i(@rrload[i])']
        got = NgspiceBackend().device_current_vectors(_raw(names), '', 'Rrload')
        assert got['i'] == 'i(@rrload[i])'

    def test_ngspice_no_match_empty(self):
        from silk.simulation.backend import NgspiceBackend
        got = NgspiceBackend().device_current_vectors(
            _raw(['time', 'v(vout)']), 'xdut', 'XMmn')
        assert got == {}

    def test_ngspice_ambiguous_marked(self):
        """Two internal devices under the same path/leaf for one suffix →
        ambiguous (value None), not a silent pick (§6.2)."""
        from silk.simulation.backend import NgspiceBackend
        names = [
            'i(@m.xdut.xmmn.mA[id])',
            'i(@m.xdut.xmmn.mB[id])',
        ]
        got = NgspiceBackend().device_current_vectors(_raw(names), 'xdut', 'XMmn')
        assert got['id'] is None   # ambiguous sentinel

    def test_ngspice_voltage_source_plain_branch(self):
        """V-source current is emitted as a PLAIN ``i(v<name>)`` — no ``@``, no
        ``[suffix]`` (verified against ngspice-42 in P2a). The seam surfaces it
        under the 'branch' key so the core can sign-map it (§6.3 V row)."""
        from silk.simulation.backend import NgspiceBackend
        names = ['time', 'v(vdd)', 'i(vvdd)', 'i(@rr1[i])']
        got = NgspiceBackend().device_current_vectors(_raw(names), '', 'Vvdd')
        assert got['branch'] == 'i(vvdd)'

    def test_ngspice_plain_branch_does_not_match_node_voltage(self):
        """A ``v(vdd)`` node voltage must never be picked as a branch current."""
        from silk.simulation.backend import NgspiceBackend
        got = NgspiceBackend().device_current_vectors(
            _raw(['time', 'v(vdd)']), '', 'Vvdd')
        assert got == {}


class TestTerminalCurrentVectorSeam:
    """ammeter-wrapper mode: terminal current = branch current of the 0 V
    ammeter inside the device wrapper, i(v.<path>.xm<dev>.vamm_<port>)."""

    def test_ngspice_top_level_terminal(self):
        from silk.simulation.backend import NgspiceBackend
        names = ['time', 'v(drain)', 'i(v.xmm1.vamm_d)', 'i(v.xmm1.vamm_s)']
        got = NgspiceBackend().terminal_current_vector(_raw(names), '', 'XMM1', 's')
        assert got == 'i(v.xmm1.vamm_s)'

    def test_ngspice_in_module_terminal(self):
        from silk.simulation.backend import NgspiceBackend
        names = ['time', 'i(v.xdut.xmm1.vamm_d)', 'i(v.xdut.xmm1.vamm_s)']
        got = NgspiceBackend().terminal_current_vector(_raw(names), 'xdut', 'XMM1', 'd')
        assert got == 'i(v.xdut.xmm1.vamm_d)'

    def test_ngspice_missing_terminal_returns_none(self):
        from silk.simulation.backend import NgspiceBackend
        got = NgspiceBackend().terminal_current_vector(
            _raw(['time', 'i(v.xmm1.vamm_d)']), '', 'XMM1', 'g')
        assert got is None

    def test_base_raises_not_supported(self):
        from silk.simulation.backend import XyceBackend
        from silk.exceptions import NotSupportedError
        with pytest.raises(NotSupportedError):
            XyceBackend().terminal_current_vector(_raw([]), '', 'XMM1', 's')


# ---------------------------------------------------------------------------
# Flat extraction core — _annotate_level(module, backend, raw, path, ...)
# Synthetic resistive divider + fake backend (no simulator). Signs from §6.3,
# verified against real ngspice in P2a. Vsrc=1V -> R1=1k -> vmid -> R2=1k -> 0,
# I = 0.5 mA flowing vdd->vmid->0.
# ---------------------------------------------------------------------------
class _Sig:
    def __init__(self, t, y):
        self.x = np.asarray(t, float)
        self.y = np.asarray(y, float)
        self.xlabel = 'time'
        self.ylabel = 'i'


class _FakeBackend:
    """Returns canned suffix->vector maps and canned waveforms."""
    def __init__(self, vecs, sigs):
        self._vecs = vecs
        self._sigs = sigs

    def device_current_vectors(self, raw, instance_path, leaf_name):
        return dict(self._vecs.get((instance_path, leaf_name), {}))

    def load_signal(self, raw, name, type='V'):
        return _Sig(*self._sigs[name])


_T = [0.0, 1.0, 2.0]


def _divider_tb():
    from silk.circuit.modules import Module
    from silk import sources
    from silk.elements import elements
    tb = Module('d', scratch_dir(), 'd.cir', pdk_details=None, module_name=None)
    with tb:
        elements.resistor('r1', p='vdd', n='vmid', value=1e3)
        elements.resistor('r2', p='vmid', n='0', value=1e3)
        sources.VoltageSource('vsrc', positive='vdd', negative='0', dc=1.0)
    return tb


def _divider_backend():
    # raw branch currents (pos->neg), constant: +0.5 mA through both R's;
    # V-source branch i(vvsrc) = -0.5 mA (sourcing).
    vecs = {
        ('', 'Rr1'): {'i': 'i(@rr1[i])'},
        ('', 'Rr2'): {'i': 'i(@rr2[i])'},
        ('', 'Vvsrc'): {'branch': 'i(vvsrc)'},
    }
    sigs = {
        '@rr1[i]': (_T, [0.5e-3] * 3),
        '@rr2[i]': (_T, [0.5e-3] * 3),
        'vvsrc':   (_T, [-0.5e-3] * 3),
    }
    return _FakeBackend(vecs, sigs)


def _annotate_divider(tstart=None, force=False):
    from silk.circuit.modules_results import _annotate_level, CurrentAnnotationReport
    tb = _divider_tb()
    rpt = CurrentAnnotationReport()
    _annotate_level(tb, _divider_backend(), object(), '', tstart, force, rpt)
    return tb, rpt


def _pin(net, instance, port):
    return next(b for b in net.pins if b.instance == instance and b.port == port)


class TestFlatExtraction:
    def test_pin_signs_into_net(self):
        tb, _ = _annotate_divider()
        vmid = tb.net_dict['vmid']
        # R1 feeds vmid via its neg pin: +0.5 mA into net
        assert _pin(vmid, 'r1', 'n').avg_current == pytest.approx(0.5e-3)
        # R2 drains vmid via its pos pin: -0.5 mA into net
        assert _pin(vmid, 'r2', 'p').avg_current == pytest.approx(-0.5e-3)

    def test_vsource_positive_pin_sign(self):
        tb, _ = _annotate_divider()
        vdd = tb.net_dict['vdd']
        # V-source positive pin: -i(v) = -(-0.5m) = +0.5 mA into vdd
        assert _pin(vdd, 'vsrc', 'positive').avg_current == pytest.approx(0.5e-3)

    def test_pin_records_vector_and_source(self):
        tb, _ = _annotate_divider()
        b = _pin(tb.net_dict['vmid'], 'r1', 'n')
        assert b.current_vector == 'i(@rr1[i])'
        assert b._current_source == 'spice'
        assert b.peak_current == pytest.approx(0.5e-3)
        assert b.rms_current == pytest.approx(0.5e-3)

    def test_kcl_residual_near_zero(self):
        _, rpt = _annotate_divider()
        for net in ('vmid', 'vdd', '0'):
            assert abs(rpt.worst_kcl_residual[net]) < 1e-9

    def test_net_aggregate_two_pin_exact(self):
        tb, _ = _annotate_divider()
        vmid = tb.net_dict['vmid']
        # 1/2 * (|+0.5m| + |-0.5m|) = 0.5 mA (2-pin exact)
        assert vmid.avg_current == pytest.approx(0.5e-3)
        assert vmid.peak_current == pytest.approx(0.5e-3)
        assert vmid._current_source == 'spice'

    def test_unannotated_net_stays_none(self):
        # A net whose pins have no vectors: aggregates stay None, not 0.0
        from silk.circuit.modules_results import _annotate_level, CurrentAnnotationReport
        tb = _divider_tb()
        empty = _FakeBackend({}, {})
        rpt = CurrentAnnotationReport()
        _annotate_level(tb, empty, object(), '', None, False, rpt)
        assert tb.net_dict['vmid'].avg_current is None
        assert tb.net_dict['vmid']._current_source is None

    def test_user_override_pin_preserved(self):
        from silk.circuit.modules_results import _annotate_level, CurrentAnnotationReport
        tb = _divider_tb()
        b = _pin(tb.net_dict['vmid'], 'r1', 'n')
        b._current_source = 'user'
        b.avg_current = 42.0
        rpt = CurrentAnnotationReport()
        _annotate_level(tb, _divider_backend(), object(), '', None, False, rpt)
        assert b.avg_current == 42.0             # preserved
        assert ('vmid', 'r1.n', 'user-override') in rpt.skips

    def test_force_overrides_user_pin(self):
        from silk.circuit.modules_results import _annotate_level, CurrentAnnotationReport
        tb = _divider_tb()
        b = _pin(tb.net_dict['vmid'], 'r1', 'n')
        b._current_source = 'user'
        b.avg_current = 42.0
        rpt = CurrentAnnotationReport()
        _annotate_level(tb, _divider_backend(), object(), '', None, True, rpt)
        assert b.avg_current == pytest.approx(0.5e-3)   # overwritten
        assert b._current_source == 'spice'


# ---------------------------------------------------------------------------
# TASK-308 — ideal R/C/L family parity. Two independent ways to build the same
# ideal passive must get the same _current_row treatment: the hand-written
# silk.elements.passives classes (pos/neg pins) and the YAML-driven
# silk.elements.elements.resistor/capacitor/inductor (pinList [p, n], an
# AnalogElement). Before the fix, the YAML build fell into the MOS
# (PDKDevice/AnalogElement) branch and its p/n pins came back unsupported.
# ---------------------------------------------------------------------------
def _passives_rc_tb():
    from silk.circuit.modules import Module
    from silk.elements import passives
    tb = Module('d', scratch_dir(), 'd.cir', pdk_details=None, module_name=None)
    with tb:
        passives.Resistor('r1', pos='vdd', neg='vmid', resistance=1e3)
        passives.Capacitor('c1', pos='vmid', neg='0', capacitance=1e-9)
    return tb


def _yaml_rc_tb():
    from silk.circuit.modules import Module
    from silk.elements import elements
    tb = Module('d', scratch_dir(), 'd.cir', pdk_details=None, module_name=None)
    with tb:
        elements.resistor('r1', p='vdd', n='vmid', value=1e3)
        elements.capacitor('c1', p='vmid', n='0', value=1e-9)
    return tb


def _rc_backend():
    vecs = {
        ('', 'Rr1'): {'i': 'i(@rr1[i])'},
        ('', 'Cc1'): {'i': 'i(@cc1[i])'},
    }
    sigs = {
        '@rr1[i]': (_T, [0.5e-3] * 3),
        '@cc1[i]': (_T, [0.2e-3] * 3),
    }
    return _FakeBackend(vecs, sigs)


def _annotate_rc(tb):
    from silk.circuit.modules_results import _annotate_level, CurrentAnnotationReport
    rpt = CurrentAnnotationReport()
    _annotate_level(tb, _rc_backend(), object(), '', None, False, rpt)
    return tb, rpt


class TestIdealPassiveFamilyParity:
    def test_current_row_matches_across_families(self):
        from silk.circuit.modules_results import _current_row
        from silk.elements import passives, elements
        r_pass = passives.Resistor('rp', pos='a', neg='b', resistance=1e3)
        c_pass = passives.Capacitor('cp', pos='a', neg='b', capacitance=1e-9)
        r_yaml = elements.resistor('ry', p='a', n='b', value=1e3)
        c_yaml = elements.capacitor('cy', p='a', n='b', value=1e-9)

        assert _current_row(r_pass, 'pos') == _current_row(r_yaml, 'p')
        assert _current_row(r_pass, 'neg') == _current_row(r_yaml, 'n')
        assert _current_row(c_pass, 'pos') == _current_row(c_yaml, 'p')
        assert _current_row(c_pass, 'neg') == _current_row(c_yaml, 'n')

        # Before the fix, the YAML element fell into the MOS branch: only
        # port 'd' is non-None there, so p/n both came back None.
        assert _current_row(r_yaml, 'p') is not None
        assert _current_row(r_yaml, 'n') is not None
        assert _current_row(c_yaml, 'p') is not None
        assert _current_row(c_yaml, 'n') is not None

    def test_mos_branch_still_reached_only_by_mos(self):
        """The isinstance(element, (PDKDevice, AnalogElement)) branch must
        still catch an actual MOS/PDK device — the fix must not swallow it
        into the ideal-passive branch."""
        from silk.circuit.modules_results import _current_row
        dev = TestCurrentRowPDKDevice()._mos('nmos1p8')
        assert _current_row(dev, 'd') == ('id', -1.0)

    def test_full_annotation_identical_between_families(self):
        """Same circuit (R vdd->vmid, C vmid->0), built once with
        passives.*+add_passive and once with elements.*+add(): identical
        signed per-terminal current for every physical terminal."""
        tb_pass, rpt_pass = _annotate_rc(_passives_rc_tb())
        tb_yaml, rpt_yaml = _annotate_rc(_yaml_rc_tb())

        assert rpt_pass.skips == []
        assert rpt_yaml.skips == []   # before the fix: 2x unsupported-kind

        # (instance, hand-written port, YAML port) — same physical terminal.
        terminals = [
            ('r1', 'pos', 'p'), ('r1', 'neg', 'n'),
            ('c1', 'pos', 'p'), ('c1', 'neg', 'n'),
        ]
        for net_name in ('vdd', 'vmid', '0'):
            net_pass = tb_pass.net_dict[net_name]
            net_yaml = tb_yaml.net_dict[net_name]
            for instance, port_pass, port_yaml in terminals:
                b_pass = next((b for b in net_pass.pins
                               if b.instance == instance and b.port == port_pass), None)
                b_yaml = next((b for b in net_yaml.pins
                               if b.instance == instance and b.port == port_yaml), None)
                assert (b_pass is None) == (b_yaml is None)
                if b_pass is not None:
                    assert b_pass.avg_current == pytest.approx(b_yaml.avg_current)
                    assert b_pass._current_source == b_yaml._current_source == 'spice'


class TestAmmeterModeIdealPassiveCurrentRow:
    """TASK-312.7: an ideal R/C/L must never fall into the MOS-only ammeter
    branch of _annotate_level, even when method='ammeter_wrapper' — it always
    dispatches through _current_row/device_current_vectors, exactly as under
    the default 'savecurrents' method (current annotation for an ideal
    passive is method-independent, per the TASK-308 ruling)."""

    def test_ideal_passive_annotated_same_under_ammeter_wrapper_method(self):
        from silk.circuit.modules_results import _annotate_level, CurrentAnnotationReport
        tb = _yaml_rc_tb()
        rpt = CurrentAnnotationReport()
        _annotate_level(tb, _rc_backend(), object(), '', None, False, rpt,
                        method='ammeter_wrapper')
        assert rpt.skips == []      # before the fix: swallowed into the MOS
                                    # branch, whose ports (d/g/s/b) never match
                                    # 'p'/'n' -> unsupported-kind
        assert _pin(tb.net_dict['vmid'], 'r1', 'n').avg_current == pytest.approx(0.5e-3)
        assert _pin(tb.net_dict['vmid'], 'c1', 'p').avg_current == pytest.approx(-0.2e-3)

    def test_identical_to_add_passive_build_under_ammeter_wrapper_method(self):
        from silk.circuit.modules_results import _annotate_level, CurrentAnnotationReport
        tb_yaml = _yaml_rc_tb()
        tb_pass = _passives_rc_tb()
        rpt_yaml = CurrentAnnotationReport()
        rpt_pass = CurrentAnnotationReport()
        _annotate_level(tb_yaml, _rc_backend(), object(), '', None, False, rpt_yaml,
                        method='ammeter_wrapper')
        _annotate_level(tb_pass, _rc_backend(), object(), '', None, False, rpt_pass,
                        method='ammeter_wrapper')
        assert rpt_yaml.skips == rpt_pass.skips == []

        terminals = [('r1', 'n', 'neg'), ('c1', 'p', 'pos')]
        for instance, port_yaml, port_pass in terminals:
            b_yaml = _pin(tb_yaml.net_dict['vmid'], instance, port_yaml)
            b_pass = _pin(tb_pass.net_dict['vmid'], instance, port_pass)
            assert b_yaml.avg_current == pytest.approx(b_pass.avg_current)
            assert b_yaml._current_source == b_pass._current_source == 'spice'

    def test_mos_still_reaches_ammeter_branch_alongside_ideal_passive(self):
        """A regression guard for the fix itself: adding the ideal-passive
        exclusion must not also exclude an actual MOS/PDK device from the
        ammeter branch."""
        from silk.circuit.modules_results import _current_row, _is_ideal_passive
        dev = TestCurrentRowPDKDevice()._mos('nmos1p8')
        assert not _is_ideal_passive(dev)
        assert _current_row(dev, 'd') == ('id', -1.0)


class _AmmBackend:
    """Fake backend for ammeter mode: terminal_current_vector builds the
    i(v.<path>.<leaf>.vamm_<port>) name and load_signal serves canned waveforms
    keyed by the inner (i(...)-stripped) name."""
    def __init__(self, inner_sigs):
        self._sigs = inner_sigs

    def terminal_current_vector(self, raw, instance_path, leaf_name, port):
        path = instance_path.lower().strip('.')
        inner = 'v.' + (path + '.' if path else '') + leaf_name.lower() + '.vamm_' + port
        return f'i({inner})' if inner in self._sigs else None

    def load_signal(self, raw, name, type='V'):
        return _Sig(*self._sigs[name])


def _mos_tb():
    import os
    from silk.circuit.modules import Module
    from silk.pdk.basepdk import BasePDK
    pdk = BasePDK(
        os.path.join('silk', 'pdk', 'data', 'skywater_130_device_details.yaml'),
        library_location='/fake')
    tb = Module('PDK_skywater130', scratch_dir(), 'mos.cir',
                pdk_details=pdk, module_name=None)
    tb.add(pdk.nmos1p8('M1', d='vd', g='vg', s='0', b='0', w=1.0, l=0.15, m=1))
    return tb


class TestAmmeterReading:
    def test_all_four_terminals_read_uniform_sign(self):
        from silk.circuit.modules_results import _annotate_level, CurrentAnnotationReport
        tb = _mos_tb()
        sigs = {
            'v.xmm1.vamm_d': (_T, [1e-3] * 3),    # into device: +1 mA drain
            'v.xmm1.vamm_g': (_T, [0.0] * 3),
            'v.xmm1.vamm_s': (_T, [-1e-3] * 3),   # -1 mA source
            'v.xmm1.vamm_b': (_T, [0.0] * 3),
        }
        rpt = CurrentAnnotationReport()
        _annotate_level(tb, _AmmBackend(sigs), object(), '', None, False, rpt,
                        method='ammeter_wrapper')
        # every MOS terminal annotated (not drain-only), uniform into-net = -i(vamm)
        assert _pin(tb.net_dict['vd'], 'M1', 'd').avg_current == pytest.approx(-1e-3)
        assert _pin(tb.net_dict['0'], 'M1', 's').avg_current == pytest.approx(1e-3)
        g = _pin(tb.net_dict['vg'], 'M1', 'g')
        assert g.avg_current == pytest.approx(0.0)
        assert g._current_source == 'spice'          # gate annotated, not skipped
        assert g.current_vector == 'i(v.xmm1.vamm_g)'

    def test_missing_terminal_vector_is_no_vector_skip(self):
        from silk.circuit.modules_results import _annotate_level, CurrentAnnotationReport
        tb = _mos_tb()
        sigs = {'v.xmm1.vamm_d': (_T, [1e-3] * 3)}   # only drain present
        rpt = CurrentAnnotationReport()
        _annotate_level(tb, _AmmBackend(sigs), object(), '', None, False, rpt,
                        method='ammeter_wrapper')
        assert ('0', 'M1.s', 'no-vector') in rpt.skips


class TestAnnotateCurrentsPreconditions:
    def test_results_not_loaded_raises(self):
        from silk.exceptions import SimulationError
        tb = _divider_tb()
        tb.sim_results_loaded = False
        with pytest.raises(SimulationError, match='save_device_currents'):
            tb.annotate_currents()

    def test_behavioral_mode_raises(self):
        from silk.exceptions import SimulationError
        from silk.simulation.analysis import Transient
        tb = _divider_tb()
        tb.sim_results_loaded = True
        tb._behavioral_mode = True
        tb.sim_analysis_dict = {'transient': Transient(step=1e-9, stop=1e-6)}
        with pytest.raises(SimulationError, match='behavioral'):
            tb.annotate_currents()

    def test_non_transient_raises(self):
        from silk.exceptions import SimulationError
        from silk.simulation.analysis import AC
        tb = _divider_tb()
        tb.sim_results_loaded = True
        tb._behavioral_mode = False
        tb.sim_analysis_dict = {'ac': AC(variation='dec', points=10, fstart=1, fstop=1e6)}
        with pytest.raises(SimulationError, match='[Tt]ransient'):
            tb.annotate_currents()

    def test_backend_without_seam_becomes_simulation_error(self):
        """Xyce backend raises NotSupportedError at the seam → surfaced as a
        clear SimulationError (§4.1)."""
        from silk.exceptions import SimulationError
        from silk.simulation.analysis import Transient
        tb = _divider_tb()
        tb.simulator = 'xyce'
        tb.sim_results_loaded = True
        tb._behavioral_mode = False
        tb.sim_analysis_dict = {'transient': Transient(step=1e-9, stop=1e-6)}
        tb._raw = object()
        tb.results_file = 'x.raw'
        with pytest.raises(SimulationError, match='does not support'):
            tb.annotate_currents()

    def test_report_provenance(self):
        """Report carries results_file, analysis, window (§4.1)."""
        from silk.circuit.modules_results import CurrentAnnotationReport
        rpt = CurrentAnnotationReport(results_file='r.raw', analysis='transient',
                                      window=(1e-6,))
        assert rpt.results_file == 'r.raw'
        assert rpt.analysis == 'transient'
        assert rpt.window == (1e-6,)


# ---------------------------------------------------------------------------
# Recursion (§5) + D1 definition reduction — two instances of one subckt.
# ---------------------------------------------------------------------------
def _hier_two_instances():
    from silk.circuit.modules import Module
    from silk.circuit.circuit_elements import PortDir
    from silk.elements import elements
    tmp = scratch_dir()
    leaf = Module('leaf', tmp, 'leaf.cir', pdk_details=None, module_name='leafcell')
    leaf.define_module_ports({'a': PortDir.IN, 'b': PortDir.OUT})
    with leaf:
        elements.resistor('rint', p='a', n='b', value=1e3)
    tb = Module('tb', tmp, 'tb.cir', pdk_details=None, module_name=None)
    tb.add_module(leaf, 'dut1', nets={'a': 'na1', 'b': 'nb1'})
    tb.add_module(leaf, 'dut2', nets={'a': 'na2', 'b': 'nb2'})
    return tb


def _hier_backend():
    vecs = {
        ('xdut1', 'Rrint'): {'i': 'i(@rrint1[i])'},
        ('xdut2', 'Rrint'): {'i': 'i(@rrint2[i])'},
    }
    sigs = {
        '@rrint1[i]': (_T, [1e-3] * 3),   # dut1 branch = +1 mA
        '@rrint2[i]': (_T, [3e-3] * 3),   # dut2 branch = +3 mA
    }
    return _FakeBackend(vecs, sigs)


def _hier_internal():
    """Two instances of a cell with an INTERNAL net 'mid' (p -r1- mid -r2- g)
    plus port nets p/g — so D1 has both a complete internal defn net and a
    partial (port) defn net to check."""
    from silk.circuit.modules import Module
    from silk.circuit.circuit_elements import PortDir
    from silk.elements import elements
    tmp = scratch_dir()
    leaf = Module('leaf', tmp, 'leaf.cir', pdk_details=None, module_name='cell2')
    leaf.define_module_ports({'p': PortDir.IN, 'g': PortDir.GROUND})
    with leaf:
        elements.resistor('r1', p='p', n='mid', value=1e3)
        elements.resistor('r2', p='mid', n='g', value=1e3)
    tb = Module('tb', tmp, 'tb.cir', pdk_details=None, module_name=None)
    tb.add_module(leaf, 'X0', nets={'p': 'a0', 'g': '0'})
    tb.add_module(leaf, 'X1', nets={'p': 'a1', 'g': '0'})
    return tb


def _hier_internal_backend():
    vecs = {('xX0', 'Rr1'): {'i': 'i(@rr1_0[i])'}, ('xX0', 'Rr2'): {'i': 'i(@rr2_0[i])'},
            ('xX1', 'Rr1'): {'i': 'i(@rr1_1[i])'}, ('xX1', 'Rr2'): {'i': 'i(@rr2_1[i])'}}
    sigs = {'@rr1_0[i]': (_T, [1e-3] * 3), '@rr2_0[i]': (_T, [1e-3] * 3),   # X0 = 1 mA
            '@rr1_1[i]': (_T, [3e-3] * 3), '@rr2_1[i]': (_T, [3e-3] * 3)}   # X1 = 3 mA
    return _FakeBackend(vecs, sigs)


class TestRecursion:
    def test_submodule_instances_annotated(self):
        from silk.circuit.modules_results import (
            CurrentAnnotationReport, _annotate_submodules)
        tb = _hier_two_instances()
        rpt = CurrentAnnotationReport()
        _annotate_submodules(tb, _hier_backend(), object(), '', None, False, rpt)
        d1 = rpt.instance_modules['xdut1']
        d2 = rpt.instance_modules['xdut2']
        # rint.neg feeds net 'b': +1 mA (dut1), +3 mA (dut2)
        assert _pin(d1.net_dict['b'], 'rint', 'n').avg_current == pytest.approx(1e-3)
        assert _pin(d2.net_dict['b'], 'rint', 'n').avg_current == pytest.approx(3e-3)
        assert 'xdut1' in rpt.instance_modules
        assert 'xdut2' in rpt.instance_modules

    def test_instance_paths_prefixed(self):
        from silk.circuit.modules_results import (
            CurrentAnnotationReport, _annotate_submodules)
        tb = _hier_two_instances()
        rpt = CurrentAnnotationReport()
        _annotate_submodules(tb, _hier_backend(), object(), '', None, False, rpt)
        assert 'xdut1.b' in rpt.nets_annotated
        assert 'xdut2.a' in rpt.nets_annotated


class TestInterfaceNetKCLExempt:
    """A subckt's port nets have device pins but NO pin for the port current
    (ports are not PortBindings), so they can never close KCL. They must be
    treated as interface/partial (KCL-exempt), not falsely 'fully annotated'.
    Only genuinely-internal nets get the KCL self-check (§3.1, refined)."""

    def _leaf_with_ports(self):
        from silk.circuit.modules import Module
        from silk.circuit.circuit_elements import PortDir
        from silk.elements import elements
        leaf = Module('leaf', scratch_dir(), 'leaf.cir',
                      pdk_details=None, module_name='leafcell')
        leaf.define_module_ports({'a': PortDir.IN, 'b': PortDir.OUT})
        with leaf:
            elements.resistor('r1', p='a', n='mid', value=1e3)
            elements.resistor('r2', p='mid', n='b', value=1e3)
        return leaf

    def _backend(self):
        vecs = {('', 'Rr1'): {'i': 'i(@rr1[i])'},
                ('', 'Rr2'): {'i': 'i(@rr2[i])'}}
        sigs = {'@rr1[i]': (_T, [1e-3] * 3), '@rr2[i]': (_T, [1e-3] * 3)}
        return _FakeBackend(vecs, sigs)

    def test_internal_net_kcl_checked_interface_nets_exempt(self):
        from silk.circuit.modules_results import _annotate_level, CurrentAnnotationReport
        leaf = self._leaf_with_ports()
        rpt = CurrentAnnotationReport()
        _annotate_level(leaf, self._backend(), object(), '', None, False, rpt)
        # 'mid' is internal → KCL-checked and ≈ 0
        assert 'mid' in rpt.worst_kcl_residual
        assert abs(rpt.worst_kcl_residual['mid']) < 1e-9
        # 'a'/'b' are port nets → exempt from KCL, marked partial
        assert 'a' not in rpt.worst_kcl_residual
        assert 'b' not in rpt.worst_kcl_residual
        assert 'a' in rpt.partial_nets and 'b' in rpt.partial_nets

    def test_partial_net_keeps_none_aggregates(self):
        """The ½·Σ bound is KCL-derived and only conservative over a COMPLETE
        pin set; on a partial net it under-estimates (would EM-undersize the
        wire). Partial nets therefore keep None aggregates — like zero-pin nets
        — while per-pin currents and skips/partial_nets stay populated (§3.2)."""
        from silk.circuit.modules_results import _annotate_level, CurrentAnnotationReport
        leaf = self._leaf_with_ports()
        rpt = CurrentAnnotationReport()
        _annotate_level(leaf, self._backend(), object(), '', None, False, rpt)
        a = leaf.net_dict['a']                       # port net → partial
        assert a.avg_current is None
        assert a.peak_current is None and a.rms_current is None
        assert a._current_source is None
        # but the annotated pin still carries its current
        assert _pin(a, 'r1', 'p').avg_current is not None
        # 'mid' is complete (internal) → aggregate IS computed
        assert leaf.net_dict['mid'].avg_current is not None
        assert leaf.net_dict['mid']._current_source == 'spice'


class TestAggregateBoundProperty:
    """Property (§3.2): the ½·Σ|·| aggregate is ≥ the true worst-segment current
    for EVERY cut of the net, all three metrics — on random KCL-satisfying
    waveforms. The guarantee that layout EM sizing is never under-sized."""

    def _random_kcl_net(self, n_pins, n_t, rng):
        from silk.circuit.modules_results import _refresh_net_aggregates, _trapezoid_window
        t = np.sort(rng.uniform(0, 1e-6, n_t))
        t[0] = 0.0
        waves = rng.normal(0, 1e-3, (n_pins, n_t))
        waves[-1] = -waves[:-1].sum(axis=0)          # pointwise KCL: Σ_p i_p(t)=0
        pins = []
        for w in waves:
            b = PortBinding(net='n', instance='x', port='p')
            b.avg_current, b.rms_current, b.peak_current = _trapezoid_window(t, w, None)
            pins.append(b)
        net = type('N', (), {})()
        net.avg_current = net.peak_current = net.rms_current = net._current_source = None
        _refresh_net_aggregates(net, pins)
        return net, waves, t

    def test_bounds_exceed_worst_cut(self):
        import itertools
        from silk.circuit.modules_results import _trapezoid_window
        rng = np.random.default_rng(20260703)
        for _ in range(20):
            n = int(rng.integers(2, 6))
            net, waves, t = self._random_kcl_net(n, 40, rng)
            for r in range(1, n):
                for S in itertools.combinations(range(n), r):
                    seg = waves[list(S)].sum(axis=0)          # segment waveform
                    seg_avg, seg_rms, seg_peak = _trapezoid_window(t, seg, None)
                    assert abs(seg_avg) <= net.avg_current + 1e-18
                    assert seg_peak <= net.peak_current + 1e-18
                    assert seg_rms <= net.rms_current + 1e-15


class TestDefinitionReduction:
    def test_worst_case_across_instances(self):
        from silk.circuit.modules_results import (
            CurrentAnnotationReport, _annotate_submodules, _reduce_definitions)
        tb = _hier_two_instances()
        rpt = CurrentAnnotationReport()
        _annotate_submodules(tb, _hier_backend(), object(), '', None, False, rpt)
        _reduce_definitions(tb, rpt)
        defn = tb.module_dict['leafcell']['device']
        # worst |·| across instances: max(1m, 3m) = 3 mA, dut2's sign retained
        assert _pin(defn.net_dict['b'], 'rint', 'n').avg_current == pytest.approx(3e-3)
        assert _pin(defn.net_dict['a'], 'rint', 'p').avg_current == pytest.approx(-3e-3)
        assert 'leafcell' in rpt.definition_reduced

    def test_definition_complete_net_aggregate_recomputed_partial_none(self):
        """D1 aggregates obey the same conservativeness rule: a COMPLETE internal
        defn net gets ½·Σ from the worst-case pins; a partial (port) defn net
        keeps None (would under-size). Per-pin worst-case values set regardless."""
        from silk.circuit.modules_results import (
            CurrentAnnotationReport, _annotate_submodules, _reduce_definitions)
        tb = _hier_internal()
        rpt = CurrentAnnotationReport()
        _annotate_submodules(tb, _hier_internal_backend(), object(), '', None, False, rpt)
        _reduce_definitions(tb, rpt)
        defn = tb.module_dict['cell2']['device']
        # 'mid' is complete (r1.neg +3m, r2.pos -3m after worst-case): ½·(3m+3m)=3m
        assert defn.net_dict['mid'].avg_current == pytest.approx(3e-3)
        # 'p' is a port net → partial → None (not conservative on a partial set)
        assert defn.net_dict['p'].avg_current is None
        assert defn.net_dict['p']._current_source is None
        # per-pin worst-case still written on the port net's pin
        assert _pin(defn.net_dict['p'], 'r1', 'p').avg_current == pytest.approx(-3e-3)
