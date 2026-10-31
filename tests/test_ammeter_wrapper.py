"""Terminal-current ammeter-wrapper netlist emission (MOS-terminal backlog fix).

When save_device_currents is on AND the terminal-current method is
'ammeter_wrapper', each PDK device is wrapped in a subckt carrying 0 V series
ammeters on every terminal, so terminal currents are read from the ammeter
branch currents i(v.x<inst>.vamm_*) instead of the drain-only @m[id].

Netlist-string tests here (no simulator). Real-ngspice proof lives in
tests/test_current_probes.py (slow).
"""
import os

from silk.circuit.modules import Module
from silk.pdk.basepdk import BasePDK
from silk import sources
from silk.simulation import analysis
from tests._env import scratch_dir

_PDK_YAML = os.path.join(os.path.dirname(__file__), '..', 'silk', 'pdk', 'data',
                         'skywater_130_device_details.yaml')


def _pdk(lib='/fake/sky130.lib.spice'):
    return BasePDK(_PDK_YAML, library_location=lib)


def _single_nmos_tb():
    tb = Module('PDK_skywater130', scratch_dir(), 'amm_tb.cir',
                pdk_details=_pdk(), module_name=None)
    tb.add(_pdk().nmos1p8('M1', d='drain', g='gate', s='0', b='0', w=1.0, l=0.15, m=1))
    tb.add_source(sources.VoltageSource('vd', positive='drain', negative='0', dc=1.0))
    tb.add_source(sources.VoltageSource('vg', positive='gate', negative='0', dc=0.7))
    tb.add_sim_type(analysis.Transient(step=1e-9, stop=10e-9))
    return tb


def _single_nmos_factory_tb():
    """Same device via the factory-direct API (pdk.nmos1p8(...) -> AnalogElement,
    added with ckt.add) rather than add_element (-> PDKDevice)."""
    pdk = _pdk()
    tb = Module('PDK_skywater130', scratch_dir(), 'amm_fac.cir',
                pdk_details=pdk, module_name=None)
    tb.add(pdk.nmos1p8('M1', d='drain', g='gate', s='0', b='0', w=1.0, l=0.15, m=1))
    tb.add_source(sources.VoltageSource('vd', positive='drain', negative='0', dc=1.0))
    tb.add_source(sources.VoltageSource('vg', positive='gate', negative='0', dc=0.7))
    tb.add_sim_type(analysis.Transient(step=1e-9, stop=10e-9))
    return tb


def _gen(tb):
    deck_path = tb.generate_netlist(corner='tt')
    with open(deck_path) as fh:
        return fh.read()


class TestAmmeterAnalogElement:
    """The factory-direct AnalogElement path (to_spice()) must also get wrapped."""

    def test_factory_device_is_wrapped(self):
        tb = _single_nmos_factory_tb()
        tb.save_device_currents = True
        tb.terminal_current_method = 'ammeter_wrapper'
        deck = _gen(tb).lower()
        assert '.subckt amm_sky130_fd_pr__nfet_01v8' in deck
        for term in ('vamm_d', 'vamm_g', 'vamm_s', 'vamm_b'):
            assert term in deck
        # M1 instantiates the wrapper, not the raw model line
        assert 'xmm1 drain gate 0 0 amm_sky130_fd_pr__nfet_01v8' in deck

    def test_factory_flag_off_unchanged(self):
        tb = _single_nmos_factory_tb()
        deck = _gen(tb).lower()
        assert 'vamm' not in deck
        # raw model line preserved
        assert 'xmm1 drain gate 0 0 sky130_fd_pr__nfet_01v8' in deck


class TestAmmeterNetlist:
    def test_flag_off_is_unchanged(self):
        """No currents requested → deck must not contain any ammeter wrapper."""
        deck = _gen(_single_nmos_tb())
        assert 'vamm' not in deck.lower()
        assert '.subckt amm_' not in deck.lower()

    def test_ammeter_mode_emits_wrapper_subckt(self):
        tb = _single_nmos_tb()
        tb.save_device_currents = True
        tb.terminal_current_method = 'ammeter_wrapper'
        deck = _gen(tb).lower()
        # a wrapper subckt with 0V ammeters on all four terminals
        assert '.subckt amm_' in deck
        for term in ('vamm_d', 'vamm_g', 'vamm_s', 'vamm_b'):
            assert term in deck, f'missing ammeter {term}'

    def test_ammeter_mode_instantiates_through_wrapper(self):
        tb = _single_nmos_tb()
        tb.save_device_currents = True
        tb.terminal_current_method = 'ammeter_wrapper'
        deck = _gen(tb).lower()
        # the device is instantiated through the wrapper, not the raw model line
        assert 'amm_' in deck
        # M1's instance line references the wrapper subckt name
        assert 'xmm1' in deck and 'amm_sky130_fd_pr__nfet_01v8' in deck

    def test_savecurrents_method_unchanged(self):
        """Default method keeps the existing savecurrents directive, no wrapper."""
        tb = _single_nmos_tb()
        tb.save_device_currents = True  # default method = savecurrents
        deck = _gen(tb).lower()
        assert '.options savecurrents' in deck
        assert '.subckt amm_' not in deck


def _hier_tb(lib='/fake/sky130.lib.spice'):
    """A MOS inside a module subckt (leafcell), instantiated in a tb."""
    from silk.circuit.circuit_elements import PortDir
    pdk = _pdk(lib)
    tmp = scratch_dir()
    leaf = Module('PDK_skywater130', tmp, 'leaf.cir', pdk_details=pdk, module_name='leafcell')
    leaf.define_module_ports({'d': PortDir.OUT, 'g': PortDir.IN,
                              's': PortDir.GROUND, 'b': PortDir.GROUND})
    leaf.add(pdk.nmos1p8('M1', d='d', g='g', s='s', b='b', w=1.0, l=0.15, m=1))
    tb = Module('PDK_skywater130', tmp, 'hier_amm.cir', pdk_details=pdk, module_name=None)
    tb.add_module(leaf, 'DUT', nets={'d': 'vout', 'g': 'vin', 's': '0', 'b': '0'})
    tb.add_source(sources.VoltageSource('vg', positive='vin', negative='0', dc=0.7))
    tb.add_sim_type(analysis.Transient(step=1e-9, stop=10e-9))
    return tb


class TestAmmeterHierarchy:
    def test_in_module_device_is_wrapped(self):
        tb = _hier_tb()
        tb.save_device_currents = True
        tb.terminal_current_method = 'ammeter_wrapper'
        deck = _gen(tb).lower()
        assert '.subckt amm_' in deck                    # wrapper emitted
        # the device INSIDE leafcell instantiates the wrapper, not the raw model
        body = deck.split('.subckt leafcell')[1].split('.ends')[0]
        assert 'amm_sky130_fd_pr__nfet_01v8' in body

    def test_wrapper_def_is_top_level_not_nested(self):
        tb = _hier_tb()
        tb.save_device_currents = True
        tb.terminal_current_method = 'ammeter_wrapper'
        deck = _gen(tb).lower()
        # the wrapper .subckt def appears BEFORE the leafcell def (top level)
        assert deck.index('.subckt amm_') < deck.index('.subckt leafcell')
        # and NO .subckt definition appears between leafcell and its .ends
        body = deck.split('.subckt leafcell')[1].split('.ends')[0]
        assert '.subckt' not in body                     # no nested definition

    def test_wrapper_deduped_across_instances(self):
        """Two instances of the same device type share ONE wrapper def."""
        from silk.circuit.circuit_elements import PortDir
        pdk = _pdk()
        tmp = scratch_dir()
        tb = Module('PDK_skywater130', tmp, 'dedup.cir', pdk_details=pdk, module_name=None)
        for nm, dn in (('M1', 'na'), ('M2', 'nb')):
            tb.add(pdk.nmos1p8(nm, d=dn, g='vin', s='0', b='0', w=1.0, l=0.15, m=1))
        tb.add_source(sources.VoltageSource('vg', positive='vin', negative='0', dc=0.7))
        tb.add_sim_type(analysis.Transient(step=1e-9, stop=10e-9))
        tb.save_device_currents = True
        tb.terminal_current_method = 'ammeter_wrapper'
        deck = _gen(tb).lower()
        assert deck.count('.subckt amm_sky130_fd_pr__nfet_01v8') == 1  # deduped


# ---------------------------------------------------------------------------
# TASK-312.7 — an ideal R/C/L must never be ammeter-wrapped, in elem_dict
# (Module.add()) or passives_dict (Module.add_passive()), exactly like a
# generate_passives()-emitted passive never was. Only a real device (PDK/MOS)
# gets the synthetic 0V-ammeter subckt; TestAmmeterAnalogElement/
# TestAmmeterNetlist/TestAmmeterHierarchy above are the regression guard that
# PDK devices are still wrapped as before.
# ---------------------------------------------------------------------------

def _ideal_rcl_tb(build, name='ideal'):
    """*build*: 'add' registers in elem_dict, 'add_passive' in passives_dict."""
    from silk.elements import elements
    pdk = _pdk()
    tb = Module('PDK_skywater130', scratch_dir(), f'{name}_{build}.cir',
                pdk_details=pdk, module_name=None)
    r = elements.resistor('r1', p='a', n='b', value=1000)
    c = elements.capacitor('c1', p='b', n='gnd', value=1e-12)
    l = elements.inductor('l1', p='a', n='gnd', value=1e-9)
    if build == 'add':
        tb.add(r)
        tb.add(c)
        tb.add(l)
    else:
        tb.add_passive(r)
        tb.add_passive(c)
        tb.add_passive(l)
    tb.add_source(sources.VoltageSource('vsrc', positive='a', negative='gnd', dc=1.0))
    tb.add_sim_type(analysis.Transient(step=1e-9, stop=10e-9))
    return tb


class TestAmmeterIdealPassiveNotWrapped:
    """Top-level ideal R/C/L, built via Module.add(elements.*)."""

    def test_top_level_rcl_not_wrapped(self):
        tb = _ideal_rcl_tb('add')
        tb.save_device_currents = True
        tb.terminal_current_method = 'ammeter_wrapper'
        deck = _gen(tb)
        assert 'vamm' not in deck.lower()
        assert '.subckt amm_' not in deck.lower()
        assert 'Rr1 a b 1k' in deck
        assert 'Cc1 b gnd 1p' in deck
        assert 'Ll1 a gnd 1n' in deck

    def test_top_level_rcl_matches_add_passive_build(self):
        tb_add = _ideal_rcl_tb('add')
        tb_add.save_device_currents = True
        tb_add.terminal_current_method = 'ammeter_wrapper'
        tb_ap = _ideal_rcl_tb('add_passive')
        tb_ap.save_device_currents = True
        tb_ap.terminal_current_method = 'ammeter_wrapper'

        deck_add = _gen(tb_add)
        deck_ap = _gen(tb_ap)
        for line in ('Rr1 a b 1k', 'Cc1 b gnd 1p', 'Ll1 a gnd 1n'):
            assert line in deck_add
            assert line in deck_ap
        # neither build path ever emits an ammeter wrapper for an ideal passive
        assert 'vamm' not in deck_add.lower() and 'vamm' not in deck_ap.lower()


def _ideal_r_submodule_tb(build):
    """An ideal resistor inside a leaf subckt, instantiated in a tb."""
    from silk.circuit.circuit_elements import PortDir
    from silk.elements import elements
    pdk = _pdk()
    tmp = scratch_dir()
    leaf = Module('PDK_skywater130', tmp, 'leaf_rcl.cir', pdk_details=pdk,
                  module_name='rclleaf')
    leaf.define_module_ports({'a': PortDir.INOUT, 'vss': PortDir.GROUND})
    r = elements.resistor('r1', p='a', n='vss', value=1000)
    if build == 'add':
        leaf.add(r)
    else:
        leaf.add_passive(r)
    tb = Module('PDK_skywater130', tmp, f'hier_rcl_{build}.cir', pdk_details=pdk,
               module_name=None)
    tb.add_module(leaf, 'DUT', nets={'a': 'vin', 'vss': '0'})
    tb.add_source(sources.VoltageSource('vsrc', positive='vin', negative='0', dc=1.0))
    tb.add_sim_type(analysis.Transient(step=1e-9, stop=10e-9))
    return tb


class TestAmmeterIdealPassiveInSubmoduleNotWrapped:
    """Same guarantee, one level down a subckt hierarchy."""

    def test_submodule_r_not_wrapped(self):
        tb = _ideal_r_submodule_tb('add')
        tb.save_device_currents = True
        tb.terminal_current_method = 'ammeter_wrapper'
        deck = _gen(tb)
        assert 'vamm' not in deck.lower()
        assert '.subckt amm_' not in deck.lower()
        body = deck.split('.subckt rclleaf')[1].split('.ends')[0]
        assert 'Rr1 a vss 1k' in body

    def test_submodule_r_matches_add_passive_build(self):
        tb_add = _ideal_r_submodule_tb('add')
        tb_add.save_device_currents = True
        tb_add.terminal_current_method = 'ammeter_wrapper'
        tb_ap = _ideal_r_submodule_tb('add_passive')
        tb_ap.save_device_currents = True
        tb_ap.terminal_current_method = 'ammeter_wrapper'

        deck_add = _gen(tb_add)
        deck_ap = _gen(tb_ap)
        assert 'Rr1 a vss 1k' in deck_add
        assert 'Rr1 a vss 1k' in deck_ap


class TestAmmeterMixedPdkDeviceAndIdealPassive:
    """A real PDK device and an ideal passive in the same circuit: the device
    is still ammeter-wrapped, the passive never is."""

    def test_pdk_device_wrapped_ideal_passive_plain(self):
        from silk.elements import elements
        pdk = _pdk()
        tb = Module('PDK_skywater130', scratch_dir(), 'mixed.cir',
                    pdk_details=pdk, module_name=None)
        tb.add(pdk.nmos1p8('M1', d='drain', g='gate', s='0', b='0', w=1.0, l=0.15, m=1))
        tb.add(elements.resistor('rload', p='vdd', n='drain', value=1000))
        tb.add_source(sources.VoltageSource('vdd', positive='vdd', negative='0', dc=1.8))
        tb.add_source(sources.VoltageSource('vg', positive='gate', negative='0', dc=0.7))
        tb.add_sim_type(analysis.Transient(step=1e-9, stop=10e-9))
        tb.save_device_currents = True
        tb.terminal_current_method = 'ammeter_wrapper'
        deck = _gen(tb)

        assert '.subckt amm_sky130_fd_pr__nfet_01v8' in deck.lower()  # MOS wrapped
        assert 'Rrload vdd drain 1k' in deck                          # R plain
        # the resistor's own instance line never references an amm_ subckt
        rload_line = next(ln for ln in deck.splitlines() if ln.startswith('Rrload'))
        assert 'amm_' not in rload_line.lower()


# ---------------------------------------------------------------------------
# End-to-end proof against REAL ngspice: the ammeter wrapper recovers ALL four
# MOS terminal currents (which savecurrents/@m can't for BSIM4) and KCL closes.
# ---------------------------------------------------------------------------
import pytest  # noqa: E402

_SKY130_LIB = os.path.expanduser('~/pdk/sky130A/libs.tech/ngspice/sky130.lib.spice')
_have_pdk = os.path.exists(_SKY130_LIB)


@pytest.mark.slow
@pytest.mark.skipif(not _have_pdk, reason='sky130 ngspice models not installed')
def test_ammeter_recovers_all_terminals_and_kcl_closes():
    import numpy as np
    from silk.pdk.basepdk import BasePDK
    pdk = BasePDK(_PDK_YAML, library_location=_SKY130_LIB)
    tb = Module('PDK_skywater130', scratch_dir(), 'amm_e2e.cir',
                pdk_details=pdk, module_name=None)
    tb.add(pdk.nmos1p8('M1', d='drain', g='gate', s='0', b='0', w=1.0, l=0.15, m=1))
    tb.add_source(sources.VoltageSource('vd', positive='drain', negative='0', dc=1.0))
    tb.add_source(sources.VoltageSource('vg', positive='gate', negative='0', dc=0.7))
    tb.add_sim_type(analysis.Transient(step=1e-9, stop=10e-9))
    tb.save_device_currents = True
    tb.terminal_current_method = 'ammeter_wrapper'
    tb.simulate(corner='tt')

    def i(term):
        return float(np.asarray(
            tb.load_signal(f'v.xmm1.vamm_{term}', signalType='I').y)[-1])

    d, g, s, b = i('d'), i('g'), i('s'), i('b')
    assert abs(d) > 1e-9                       # drain conducts
    assert s == pytest.approx(-d, rel=1e-3)    # source = -drain (Ig,Ib ~ 0)
    assert abs(g) < 1e-12 and abs(b) < 1e-9    # gate/bulk ~ 0
    assert abs(d + g + s + b) < 1e-12          # KCL closes (all 4 terminals)


@pytest.mark.slow
@pytest.mark.skipif(not _have_pdk, reason='sky130 ngspice models not installed')
def test_ammeter_in_module_device_kcl_closes():
    """A device INSIDE a module subckt gets wrapped; its terminal currents read
    from the hierarchical path v.x<inst>.xm<dev>.vamm_* and KCL closes."""
    import numpy as np
    tb = _hier_tb(lib=_SKY130_LIB)
    tb.add_source(sources.VoltageSource('vd', positive='vout', negative='0', dc=1.0))
    tb.save_device_currents = True
    tb.terminal_current_method = 'ammeter_wrapper'
    tb.simulate(corner='tt')

    def i(term):
        return float(np.asarray(
            tb.load_signal(f'v.xdut.xmm1.vamm_{term}', signalType='I').y)[-1])

    d, g, s, b = i('d'), i('g'), i('s'), i('b')
    assert abs(d) > 1e-9
    assert s == pytest.approx(-d, rel=1e-3)
    assert abs(d + g + s + b) < 1e-12          # KCL closes for the in-module device


@pytest.mark.slow
@pytest.mark.skipif(not _have_pdk, reason='sky130 ngspice models not installed')
def test_ammeter_factory_direct_annotate_kcl():
    """Full factory-direct (AnalogElement) + ammeter + annotate_currents path:
    a CMOS stack built with pdk.nmos1p8(...)/pmos1p8(...) + tb.add; the internal
    all-drain 'mid' net annotates and closes KCL."""
    from silk.pdk.basepdk import BasePDK
    pdk = BasePDK(_PDK_YAML, library_location=_SKY130_LIB)
    tb = Module('PDK_skywater130', scratch_dir(), 'fac_stack.cir',
                pdk_details=pdk, module_name=None)
    tb.add(pdk.pmos1p8('MP', d='mid', g='0', s='vdd', b='vdd', w=2.0, l=0.15, m=1))
    tb.add(pdk.nmos1p8('MN', d='mid', g='vdd', s='0', b='0', w=1.0, l=0.15, m=1))
    tb.add_source(sources.VoltageSource('vdd', positive='vdd', negative='0', dc=1.8))
    tb.add_sim_type(analysis.Transient(step=1e-9, stop=20e-9))
    tb.save_device_currents = True
    tb.terminal_current_method = 'ammeter_wrapper'
    tb.simulate(corner='tt')

    rpt = tb.annotate_currents()
    mid = tb.net_dict['mid']
    i_mp = _pin_of(mid, 'MP', 'd').avg_current
    i_mn = _pin_of(mid, 'MN', 'd').avg_current
    assert i_mp is not None and i_mn is not None
    assert i_mp * i_mn < 0                      # opposite signs (source vs sink)
    assert abs(rpt.worst_kcl_residual['mid']) < 1e-7


def _pin_of(net, instance, port):
    return next(b for b in net.pins if b.instance == instance and b.port == port)
