"""Opt-in simulator save list — ``Module.set_save_list``.

The rawfile-size escape hatch (backlog 2026-08-12-simulator-save-list.md):
``save all`` stays the default (the explorer / annotate_currents /
no-re-simulate loop depend on complete rawfiles), and an explicit list — in
the same selector grammar as ``emit_view(signals=...)`` — restricts the
rawfile itself when a run is too big to store.

Design facts these tests pin, all verified by real runs 2026-08-22:

* ngspice-42 has no usable class filter (``save nointernals`` crashes into a
  constants-plot rawfile), so silk emits an **enumerated** ``save`` list built
  from its own net inventory — which is simultaneously the "design signals
  only" filter, since device parasitic nodes (``…#body``) are never silk nets.
* An unknown node in an ngspice ``save`` line is dropped **silently** (no
  warning, rc 0) — so silk's fail-loud selector validation at netlist time is
  the only typo defense.
* A subckt PORT net never gets its own hierarchical vector (the simulator
  aliases it to the parent net) — the inventory must list only the parent's
  name or the port entry would be silently dropped per the point above.
* Xyce spells hierarchy ``V(XDUT:VMID)`` where ngspice spells
  ``v(xdut.vmid)``; the save list is rendered per backend and readback
  normalizes the separator like it already normalizes case.
"""
import numpy as np
import os
from types import SimpleNamespace

import pytest

from silk.circuit.modules import Module
from silk.circuit.modules_netlist import _save_list_inventory
from silk.circuit.ports import PortDir
from silk.exceptions import NotSupportedError, CircuitError, SimulationError
from silk.simulation import analysis
from silk.elements import sources, elements


@pytest.fixture
def hier_tb(pdk, tmp_dir):
    """Inverter → buffer → tb: one subckt-internal net (``xdut.vmid``), two
    hierarchy levels (the inverter level has NO internal nets — every net is
    a port — which pins the port-aliasing rule at depth 2)."""
    inv = Module('PDK_skywater130', tmp_dir, 'inv.cir', pdk_details=pdk,
                 module_name='invx')
    inv.define_module_ports({'vin': PortDir.IN, 'vout': PortDir.OUT,
                             'vdd': PortDir.POWER, 'vss': PortDir.GROUND})
    inv.add(pdk.nmos1p8('mn1', d='vout', g='vin', s='vss', b='vss', w=1, l=0.15, m=1))
    inv.add(pdk.pmos1p8('mp1', d='vout', g='vin', s='vdd', b='vdd', w=2, l=0.15, m=1))

    buf = Module('PDK_skywater130', tmp_dir, 'buf.cir', pdk_details=pdk,
                 module_name='bufx')
    buf.define_module_ports({'vin': PortDir.IN, 'vout': PortDir.OUT,
                             'vdd': PortDir.POWER, 'vss': PortDir.GROUND})
    buf.add_module(inv, 'inv1', nets={'vin': 'vin', 'vout': 'vmid',
                                       'vdd': 'vdd', 'vss': 'vss'})
    buf.add_module(inv, 'inv2', nets={'vin': 'vmid', 'vout': 'vout',
                                       'vdd': 'vdd', 'vss': 'vss'})

    tb = Module('PDK_skywater130', tmp_dir, 'tb.cir', pdk_details=pdk,
                module_name=None)
    tb.add_module(buf, 'dut', nets={'vin': 'vin', 'vout': 'vout',
                                     'vdd': 'vdd', 'vss': '0'})
    with tb:
        sources.VoltageSource('vdd_src', positive='vdd', negative='0', dc=1.8)
        sources.VoltageSource('vin_src', positive='vin', negative='0',
                              pulse=sources.Pulse(0, 1.8, td=0, tr=1e-9,
                                                  tf=1e-9, pw=10e-9,
                                                  per=20e-9))
        elements.capacitor('cload', p='vout', n='0', value=10e-15)
    tb.add_sim_type(analysis.Transient(step=1e-9, stop=40e-9))
    return tb


def _deck(tb):
    tb_deck_path = tb.generate_netlist()
    with open(tb_deck_path) as f:
        return f.read()


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------

class TestInventory:

    def test_inventory_lists_parent_names_not_port_aliases(self, hier_tb):
        """Ports alias parent nets — only ``vmid`` (buffer-internal) earns a
        hierarchical path; the inverter level contributes nothing (every net
        there is a port) and ground is absent. The waveform source also
        contributes its private series node."""
        assert sorted(_save_list_inventory(hier_tb)) == [
            '__silk_src_Vvin_esrc', 'vdd', 'vin', 'vout', 'xdut.vmid']


# ---------------------------------------------------------------------------
# ngspice emission
# ---------------------------------------------------------------------------

class TestNgspiceEmission:

    def test_default_deck_keeps_save_all(self, hier_tb):
        deck = _deck(hier_tb)
        assert 'save all' in deck
        assert 'save list' not in deck

    def test_save_list_replaces_save_all(self, hier_tb):
        hier_tb.set_save_list(['vout', 'xdut.vmid'])
        deck = _deck(hier_tb)
        assert 'save all' not in deck
        assert 'save "v(vout)" "v(xdut.vmid)"\n' in deck
        # provenance: the deck itself says the rawfile is partial
        assert '*** save list active: rawfile holds 2 node voltages' in deck

    def test_selector_grammar_glob_exclude_max_depth(self, hier_tb):
        """Same words as emit_view: glob include, exclude subtracts,
        max_depth=0 keeps top-level nets only."""
        hier_tb.set_save_list(['*'], exclude=['vdd'], max_depth=0)
        deck = _deck(hier_tb)
        assert 'save "v(vin)" "v(vout)" "v(__silk_src_Vvin_esrc)"\n' in deck
        assert 'vmid' not in deck.split('.control')[1]

    def test_clearing_restores_save_all(self, hier_tb):
        hier_tb.set_save_list(['vout'])
        hier_tb.set_save_list(None)
        assert 'save all' in _deck(hier_tb)

    def test_long_lists_chunk_into_multiple_save_lines(self, pdk):
        """Multiple ``save`` commands accumulate (verified on ngspice-42);
        emission chunks so no line grows unboundedly with design size."""
        from silk.circuit.spice_netlister import PySpiceNetlister
        nl = PySpiceNetlister(pdk)
        block = nl.generate_control_block(
            '/out', save_list=[f'n{i}' for i in range(17)])
        saves = [l for l in block.splitlines() if l.startswith('save ')]
        assert len(saves) == 3
        assert saves[0].count('v(') == 8 and saves[2].count('v(') == 1

    def test_noise_deck_ignores_save_list(self, hier_tb, capsys):
        """Explicit no-op, not an error: the noise output path writes its own
        spectral vectors and a frequency sweep is small."""
        hier_tb.add_sim_type(analysis.Noise('vout', '0', 'vin_src', 'dec',
                                            10, 1, 1e6))
        hier_tb.set_save_list(['vout'])
        deck = _deck(hier_tb)
        assert 'save all' in deck
        assert 'ignored for a deck with a Noise analysis' in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Fail-loud
# ---------------------------------------------------------------------------

class TestFailLoud:

    def test_typo_dies_at_netlist_time_with_suggestion(self, hier_tb):
        """ngspice drops unknown save nodes SILENTLY (verified) — the typo
        must die here, before the simulation is paid for."""
        hier_tb.set_save_list(['vuot'])
        with pytest.raises(CircuitError, match=r"'vuot'.*[Cc]losest.*vout"):
            hier_tb.generate_netlist()

    def test_current_selectors_rejected_at_set_time(self, hier_tb):
        with pytest.raises(NotSupportedError, match='node voltages only'):
            hier_tb.set_save_list(['vout', 'i(vdd_src)'])
        with pytest.raises(NotSupportedError, match='node voltages only'):
            hier_tb.set_save_list(['@m.xdut.mn1[id]'])

    def test_conflict_with_save_device_currents(self, hier_tb):
        """Batch-write mode is built on a deck-level ``.save all`` — the
        combination contradicts itself and must refuse, not pick silently."""
        hier_tb.set_save_list(['vout'])
        hier_tb.save_device_currents = True
        with pytest.raises(SimulationError, match='contradict'):
            hier_tb.generate_netlist()


# ---------------------------------------------------------------------------
# Xyce
# ---------------------------------------------------------------------------

class TestXyce:

    def _output_block(self, save_list=None, output_vars=('vin', 'vout')):
        from silk.circuit.xyce_netlister import XyceNetlister
        nl = XyceNetlister.__new__(XyceNetlister)
        return nl.generate_output_block(
            '/out', {0: analysis.Transient(step=1e-9, stop=1e-6)},
            output_vars=list(output_vars), save_list=save_list)

    def test_default_print_unchanged(self):
        assert '.print tran format=raw file=/out/rawfile.raw ' \
               'v(vin) v(vout)\n' in self._output_block()

    def test_save_list_renders_colon_hierarchy(self):
        """Dot paths arrive in silk spelling; Xyce's separator is ':'
        (``V(X1:MID)`` verified on Xyce 7.10). The list REPLACES the default
        top-level print — and is additive in capability: subckt internals
        become printable on Xyce for the first time."""
        block = self._output_block(save_list=['vout', 'xdut.vmid'])
        assert 'v(vout) v(xdut:vmid)\n' in block
        assert 'v(vin)' not in block

    def test_readback_normalizes_hierarchy_separator(self):
        """load_signal('xdut.vmid') resolves Xyce's ``V(XDUT:VMID)`` — the
        separator is normalized exactly like case already is."""
        from silk.signals.spice import NgspiceSignal

        def _vec(name, data):
            return SimpleNamespace(name=name, data=np.asarray(data),
                                   type='voltage')
        raw = SimpleNamespace(nvars=2, vectors=[
            _vec('TIME', np.linspace(0, 1e-6, 5)),
            _vec('V(XDUT:VMID)', np.ones(5))])
        sig = NgspiceSignal.from_rawfile(raw, 'xdut.vmid', 'V')
        np.testing.assert_array_equal(sig.y, np.ones(5))


# ---------------------------------------------------------------------------
# Integration — real ngspice run (slow)
# ---------------------------------------------------------------------------

_SKY130_LIB = os.path.expanduser(
    '~/pdk/sky130A/libs.tech/ngspice/sky130.lib.spice')


@pytest.mark.slow
@pytest.mark.skipif(not os.path.exists(_SKY130_LIB),
                    reason='sky130 ngspice models not installed')
class TestNgspiceIntegration:

    def _tb(self, tmp_dir):
        from silk.pdk.basepdk import BasePDK
        yaml = os.path.join(os.path.dirname(__file__), '..', 'silk', 'pdk',
                            'data', 'skywater_130_device_details.yaml')
        pdk = BasePDK(yaml, library_location=_SKY130_LIB)
        # Reuse the emission fixture's topology with a real library.
        return hier_tb.__wrapped__(pdk, tmp_dir)

    def test_run_restricts_rawfile_and_reads_back(self, tmp_dir):
        from silk.signals.rawfile import read_rawfile
        tb = self._tb(tmp_dir)
        tb.set_save_list(['vout', 'xdut.vmid'])
        tb.simulate()
        rf = read_rawfile(tb.results_file)
        names = [v.name for p in rf.plots for v in p.vectors]
        assert names == ['time', 'v(vout)', 'v(xdut.vmid)']
        # silk's own readback still resolves what was kept…
        assert len(tb.load_signal('xdut.vmid').y) > 0
        # …and a dropped signal fails loud at readback (the documented cost)
        with pytest.raises(SimulationError, match="'vin'"):
            tb.load_signal('vin')
