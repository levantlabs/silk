"""Tests for the Xyce backend + netlister (second simulator backend).

Xyce shares silk's SPICE *body* (header, .lib, devices, sources, passives,
analysis lines) with ngspice via SpiceBackend, and diverges only in the
*wrapper tail*: it has no nutmeg .control block — it uses
``.options parser scale=<PDK length unit>`` (1e-6 for sky130, no line for a
metre-native PDK; TASK-22) + ``.print <analysis> format=raw file=...``
with an explicit output-var list (Xyce prints only the x-axis otherwise).

The seam is ``PySpiceNetlister.generate_output_block(dir_folder,
sim_analysis_dict, output_vars)``: ngspice keeps the nutmeg control block
(byte-identical, guarded by tests/test_golden_netlist.py); Xyce overrides it.
"""
import os
import re

import pytest

from silk.simulation.backend import SimulatorBackend, EmitContext
from silk.simulation.analysis import AC, Transient, Noise
from silk.circuit.spice_netlister import PySpiceNetlister
from silk.circuit.xyce_netlister import XyceNetlister


# A throwaway PDK stand-in: the netlister only needs pdk_details for header/
# .lib emission, which generate_output_block never touches.
class _DummyPDK:
    library_location = '/PDK/sky130.lib.spice'
    # The sky130 convention: the catalog's W/L are microns. A real PDK derives
    # this from its catalog (BasePDK.length_unit_in_metres); the stub states
    # it so the emission tests below keep pinning the sky130 deck.
    length_unit_in_metres = 1e-6


# ---------------------------------------------------------------------------
# Backend registration + shared rendering
# ---------------------------------------------------------------------------

class TestXyceBackendRegistration:

    def test_xyce_is_registered(self):
        """SimulatorBackend.get('xyce') returns an XyceBackend instance."""
        from silk.simulation.backend import XyceBackend
        assert isinstance(SimulatorBackend.get('xyce'), XyceBackend)

    def test_make_netlister_returns_xyce_netlister(self):
        """The xyce backend vends an XyceNetlister."""
        netlister = SimulatorBackend.get('xyce').make_netlister(_DummyPDK())
        assert isinstance(netlister, XyceNetlister)

    def test_render_analysis_shared_with_ngspice(self):
        """Xyce inherits SpiceBackend.render_analysis — AC line is identical."""
        xy = SimulatorBackend.get('xyce').render_analysis(AC('dec', 10, 1, 1e9),
                                                          EmitContext())
        ng = SimulatorBackend.get('ngspice').render_analysis(AC('dec', 10, 1, 1e9),
                                                             EmitContext())
        assert xy == ng == '.ac dec 10 1 1G\n'

    def test_load_signal_inherited_and_case_insensitive(self):
        """Xyce gets load_signal via SpiceBackend — no 'xyce' Signal registry
        entry needed (the C2b registry unification). Xyce upper-cases names;
        the load is case-insensitive so a lowercase request resolves."""
        import numpy as np
        from types import SimpleNamespace
        from silk.simulation.backend import RawResult

        def _vec(name, data):
            return SimpleNamespace(name=name, data=np.asarray(data),
                                   type='voltage')
        native = SimpleNamespace(
            nvars=2,
            vectors=[_vec('FREQUENCY', np.logspace(0, 3, 5)),
                     _vec('V(VOUT)', np.ones(5) * 2.0)],
        )
        raw = RawResult(native=native, source_path='/x/rawfile.raw')
        sig = SimulatorBackend.get('xyce').load_signal(raw, 'vout', type='V')
        np.testing.assert_array_equal(sig.y, native.vectors[1].data)

    def test_run_wired_to_xyce_runner(self, tmp_path, monkeypatch):
        """Xyce run() resolves XyceRunner; a missing binary raises a clear
        SimulationError (proves the run path is wired, no real Xyce needed)."""
        from silk.exceptions import SimulationError
        monkeypatch.setenv('SILK_XYCE_BIN', '/nonexistent/xyce-binary')
        netlist = str(tmp_path / 'x.cir')
        with open(netlist, 'w') as f:
            f.write('* x\n.end\n')
        with pytest.raises(SimulationError, match='Xyce executable not found'):
            SimulatorBackend.get('xyce').run(netlist, str(tmp_path))


# ---------------------------------------------------------------------------
# The output-block seam
# ---------------------------------------------------------------------------

class TestOutputBlockSeam:

    def test_ngspice_output_block_is_control_block(self):
        """Default generate_output_block == the nutmeg control block, so the
        refactor stays byte-identical for ngspice."""
        nl = PySpiceNetlister(_DummyPDK())
        analyses = {'ac': AC('dec', 10, 1, 1e9)}
        out = nl.generate_output_block('/run/dir', analyses, ['vout', 'vdd'])
        assert out == nl.generate_control_block('/run/dir', noise=False)

    def test_ngspice_output_block_noise_path(self):
        """Noise still triggers the setplot noise1 path via the seam."""
        nl = PySpiceNetlister(_DummyPDK())
        analyses = {'n': Noise('out', '0', 'vin', 'dec', 10, 1, 1e9)}
        out = nl.generate_output_block('/run/dir', analyses, ['out'])
        assert 'setplot noise1' in out


class TestXyceOutputBlock:

    def _block(self, analyses, output_vars):
        return XyceNetlister(_DummyPDK()).generate_output_block(
            '/run/dir', analyses, output_vars)

    def test_emits_scale_option(self):
        """Xyce needs the micron scale for sky130 geometry."""
        out = self._block({'ac': AC('dec', 10, 1, 1e9)}, ['vout'])
        assert '.options parser scale=1e-6' in out

    def test_emits_print_raw_with_analysis_keyword(self):
        """.print <analysis> format=raw file=... is emitted (here: ac)."""
        out = self._block({'ac': AC('dec', 10, 1, 1e9)}, ['vout'])
        assert '.print ac format=raw file=/run/dir/rawfile.raw' in out

    def test_print_lists_output_vars(self):
        """Output vars are listed as v(net) — Xyce prints only the x-axis
        without an explicit list."""
        out = self._block({'ac': AC('dec', 10, 1, 1e9)}, ['vout', 'vdd'])
        assert 'v(vout)' in out and 'v(vdd)' in out

    def test_ground_net_excluded(self):
        """Ground ('0') is not a meaningful print target."""
        out = self._block({'ac': AC('dec', 10, 1, 1e9)}, ['vout', '0'])
        assert 'v(0)' not in out

    def test_no_nutmeg_control_block(self):
        """Xyce has no .control/.endc; that is the whole point."""
        out = self._block({'ac': AC('dec', 10, 1, 1e9)}, ['vout'])
        assert '.control' not in out and '.endc' not in out

    def test_ends_with_dot_end(self):
        out = self._block({'ac': AC('dec', 10, 1, 1e9)}, ['vout'])
        assert out.rstrip().endswith('.end')

    def test_transient_keyword(self):
        out = self._block({'t': Transient(1e-9, 1e-6)}, ['vout'])
        assert '.print tran format=raw' in out


# ---------------------------------------------------------------------------
# TASK-22: the scale directive is the PDK's, not the netlister's.
#
# Xyce reads every W/L as metres. sky130's catalog is micron-native, so its
# deck needs `.options parser scale=1e-6`; GF180MCU and SG13G2 are
# metre-native (`pdk_unit: base`) and the same line would rescale their W/L a
# second time. The factor is derived from the loaded catalog
# (BasePDK.length_unit_in_metres) — never hardcoded, never guessed.
# ---------------------------------------------------------------------------

class _StubPDK:
    """A PDK object that states its length unit (or does not)."""
    library_location = '/PDK/x.lib.spice'

    def __init__(self, factor):
        if factor != 'absent':
            self.length_unit_in_metres = factor


def _output_block(pdk):
    return XyceNetlister(pdk).generate_output_block(
        '/run/dir', {'ac': AC('dec', 10, 1, 1e9)}, ['vout'])


class TestXyceScaleDerivedFromPDK:

    def test_micron_native_pdk_emits_the_sky130_line_verbatim(self):
        """1e-6 renders exactly as the original hardcoded line did."""
        out = _output_block(_StubPDK(1e-6))
        assert '.options parser scale=1e-6\n' in out

    def test_metre_native_pdk_emits_no_scale_line(self):
        """factor 1.0 → Xyce's default is already right; nothing to say."""
        out = _output_block(_StubPDK(1.0))
        assert 'scale' not in out
        assert out.startswith('.print ac ')

    def test_catalog_without_lengths_emits_no_scale_line(self):
        out = _output_block(_StubPDK(None))
        assert 'scale' not in out

    def test_pdk_free_deck_emits_no_scale_line(self):
        """An R/C/V testbench with no PDK has no W/L to rescale."""
        out = _output_block(None)
        assert 'scale' not in out

    def test_pdk_that_does_not_state_its_unit_is_an_error_not_a_micron(self):
        """No silent 1e-6 fallback: that guess is the bug this replaced."""
        with pytest.raises(TypeError, match='length_unit_in_metres'):
            _output_block(_StubPDK('absent'))

    def test_other_powers_of_ten_render_compactly(self):
        out = _output_block(_StubPDK(1e-9))
        assert '.options parser scale=1e-9\n' in out


class TestBasePDKLengthUnit:
    """BasePDK.length_unit_in_metres is derived from the catalog's tags."""

    def _pdk_with(self, tmp_path, params_by_device):
        import yaml
        from silk.pdk.basepdk import BasePDK
        catalog = {}
        for dev, params in params_by_device.items():
            catalog[dev] = {
                'PDKDevice': dev,
                'pinList': {'d': 'drain', 'g': 'gate', 's': 'source', 'b': 'bulk'},
                'paramList': {'m': {'minval': 1, 'maxval': -1}, **params},
            }
        path = tmp_path / 'cat.yaml'
        path.write_text(yaml.safe_dump(catalog))
        return BasePDK(str(path), library_location='/PDK/x.lib.spice')

    def test_micron_catalog(self, tmp_path):
        pdk = self._pdk_with(tmp_path, {'n': {
            'w': {'minval': 0.42, 'maxval': -1, 'pdk_unit': 'u'},
            'l': {'minval': 0.15, 'maxval': -1, 'pdk_unit': 'u'}}})
        assert pdk.length_unit_in_metres == 1e-6

    def test_base_unit_catalog(self, tmp_path):
        pdk = self._pdk_with(tmp_path, {'n': {
            'w': {'minval': -1, 'maxval': -1, 'default': 2.2e-7, 'pdk_unit': 'base'},
            'l': {'minval': -1, 'maxval': -1, 'default': 2.8e-7, 'pdk_unit': 'base'}}})
        assert pdk.length_unit_in_metres == 1.0

    def test_area_tags_do_not_participate(self, tmp_path):
        """sky130 tags diode area 'p' next to 'u' lengths — consistent, since
        SPICE applies scale squared to an area. Not a mixed catalog."""
        pdk = self._pdk_with(tmp_path, {
            'n': {'w': {'minval': 0.42, 'maxval': -1, 'pdk_unit': 'u'}},
            'd': {'area': {'minval': 0.09, 'maxval': -1, 'pdk_unit': 'p',
                           'quantity': 'area'}}})
        assert pdk.length_unit_in_metres == 1e-6

    def test_no_length_parameter_is_none(self, tmp_path):
        pdk = self._pdk_with(tmp_path, {'n': {}})
        assert pdk.length_unit_in_metres is None

    def test_mixed_length_prefixes_raise(self, tmp_path):
        """One deck-wide scale cannot serve two prefixes."""
        from silk.pdk.basepdk import ValidationError
        pdk = self._pdk_with(tmp_path, {
            'a': {'w': {'minval': 0.42, 'maxval': -1, 'pdk_unit': 'u'}},
            'b': {'w': {'minval': -1, 'maxval': -1, 'default': 1e-7,
                        'pdk_unit': 'base'}}})
        with pytest.raises(ValidationError, match='more than one prefix'):
            pdk.length_unit_in_metres


def _load_manifest_only(name):
    """Load a bundled PDK manifest without requiring installed libraries.

    These tests exercise manifests and emitted netlist TEXT, never a real
    simulation, so they must pass on a runner with no PDK installed (CI
    installs sky130 only — TASK-22.1). ``library_location=`` skips the
    loader's lib-existence check; the inert path only ever appears inside
    ``.lib`` include lines, which no assertion here reads.
    """
    from silk import load_pdk
    return load_pdk(name, library_location='/inert/never-opened.lib')


class TestXyceScaleOnShippedPDKs:
    """The three bundled catalogs, through the real load path (manifest-only,
    dependency-free — see :func:`_load_manifest_only`)."""

    @pytest.mark.parametrize('name, factor', [
        ('sky130', 1e-6), ('gf180mcu', 1.0), ('sg13g2', 1.0)])
    def test_catalog_length_unit(self, name, factor):
        assert _load_manifest_only(name).length_unit_in_metres == factor

    def test_sky130_output_block_unchanged(self):
        out = _output_block(_load_manifest_only('sky130'))
        assert out.startswith('.options parser scale=1e-6\n.print ac ')

    @pytest.mark.parametrize('name', ['gf180mcu', 'sg13g2'])
    def test_base_unit_pdks_emit_no_rescale(self, name):
        assert 'scale' not in _output_block(_load_manifest_only(name))

    @pytest.mark.parametrize('name, device', [
        ('sky130', 'nmos1p8'), ('gf180mcu', 'nmos3p3')])
    def test_wl_not_double_scaled_in_full_deck(self, tmp_path, name, device):
        """The regression: author writes w=1 (one micron, user_units 'u' from
        the manifest). Xyce reads (emitted W) × (emitted scale) as metres.
        That product must be 1e-6 m on every PDK. Before TASK-22 gf180mcu's
        deck carried w=1u AND scale=1e-6 → Xyce saw 1e-12 m. sg13g2 refuses
        Xyce earlier (OSDI models), so it is covered by the block test only."""
        from silk import sources
        from silk.circuit.modules import Module

        pdk = _load_manifest_only(name)
        mod = Module(name, str(tmp_path), 'dbl.cir', pdk_details=pdk)
        mod.simulator = 'xyce'
        mod.netlister = mod.backend.make_netlister(pdk)
        for n in ('vout', 'g', '0'):
            mod.add_net(n)
        mod.add(getattr(pdk, device)('M1', d='vout', g='g', s='0', b='0', w=1.0, l=0.5, m=1))
        mod.add_source(sources.VoltageSource(
            'vg', positive='g', negative='0', dc=0.9, ac=1.0))
        mod.add_sim_type(AC('dec', 10, 1, 1e3))
        deck_path = mod.generate_netlist()
        deck = open(deck_path).read()

        w_token = re.search(r'^XMM1 .*\bw=(\S+)', deck, re.M).group(1)
        scale_m = re.search(r'^\.options parser scale=(\S+)$', deck, re.M)
        scale = float(scale_m.group(1)) if scale_m else 1.0
        metres_xyce_sees = _si(w_token) * scale
        assert metres_xyce_sees == pytest.approx(1e-6, rel=1e-9), (
            f'{name}: w={w_token} × scale={scale} → {metres_xyce_sees} m')


def _si(token):
    """SPICE number with an optional SI suffix → float (u, n, p, m, k, meg)."""
    token = token.lower()
    for suffix, mult in (('meg', 1e6), ('k', 1e3), ('m', 1e-3), ('u', 1e-6),
                         ('n', 1e-9), ('p', 1e-12), ('f', 1e-15)):
        if token.endswith(suffix):
            return float(token[:-len(suffix)]) * mult
    return float(token)


# ---------------------------------------------------------------------------
# End-to-end: a simulator='xyce' Module emits a deck that real Xyce runs and
# silk's reader parses. Skips cleanly where Xyce / the sky130 PDK are absent.
# ---------------------------------------------------------------------------

_XYCE_BIN = os.path.expanduser('~/xyce/Xyce-7.10.0/bin/Xyce')
_SKY130_LIB = os.path.expanduser(
    '~/pdk/sky130A/libs.tech/ngspice/sky130.lib.spice')

_have_xyce = os.path.exists(_XYCE_BIN) and os.path.exists(_SKY130_LIB)


@pytest.mark.slow
@pytest.mark.requires_xyce
@pytest.mark.skipif(not _have_xyce, reason='Xyce binary or sky130 PDK not present')
class TestXyceEndToEnd:

    def test_xyce_module_runs_through_silk(self, tmp_path, monkeypatch):
        """Full silk path: simulator='xyce' Module → run_simulator() →
        backend.run() → XyceRunner → real Xyce → RawResult → silk reads it."""
        monkeypatch.setenv('SILK_XYCE_BIN', _XYCE_BIN)
        from silk.pdk.basepdk import BasePDK
        from silk.circuit.modules import Module
        from silk import sources

        pdk_yaml = os.path.join(
            os.path.dirname(__file__), '..', 'silk', 'pdk', 'data',
            'skywater_130_device_details.yaml')
        pdk = BasePDK(pdk_yaml, library_location=_SKY130_LIB)

        mod = Module('PDK_skywater130', str(tmp_path), 'xy_cs.cir',
                     pdk_details=pdk)
        mod.simulator = 'xyce'
        mod.netlister = mod.backend.make_netlister(pdk)   # eager netlister follows
        for n in ('vout', 'vdd', 'g', '0'):
            mod.add_net(n)
        mod.add(pdk.nmos1p8('M1', d='vout', g='g', s='0', b='0', w=4.0, l=0.5, m=1))
        mod.add_source(sources.VoltageSource('vdd', positive='vdd', negative='0', dc=1.8))
        mod.add_source(sources.VoltageSource('vg', positive='g', negative='0', dc=0.9, ac=1.0))
        mod.add_sim_type(AC('dec', 10, 1, 1e3))

        # The real silk run path — no hand-rolled subprocess.
        deck_path = mod.generate_netlist(corner='tt')
        assert '.print ac format=raw' in open(deck_path).read()
        mod.run_simulator()

        assert mod.sim_results_loaded is True
        assert mod.results_file.endswith('rawfile.raw')
        # C2b: load_signal now routes through backend.load_signal (no Signal
        # registry dispatch), so the Xyce result loads with a lowercase request.
        # The 'FREQUENCY' vector Xyce wrote comes back labelled with the
        # canonical 'frequency' (TASK-87) — exact, not .lower()-tolerant.
        sig = mod.load_signal('vout', 'V')
        assert len(sig.y) > 0
        assert sig.xlabel == 'frequency'


def _ground_rc(tmp_path, return_net='0', nested=False, simulator='xyce', fault_target=False):
    from silk import Module, load_pdk, sources
    from silk.elements import elements
    pdk = load_pdk('sky130')
    mod = Module('sky130', str(tmp_path), 'ground.cir', pdk)
    mod.simulator = simulator
    target = mod
    if nested:
        target = Module('sky130', str(tmp_path), 'block.cir', pdk,
                        module_name='block')
        target.define_module_ports({'inp': 'input', 'n': 'output'})
    target.add(elements.resistor('r', p='inp', n='n', value=1e3))
    target.add(elements.capacitor('c', p='n', n=return_net, value=1e-9))
    if fault_target:
        target.add(pdk.nmos1p8('mn', d='n', g='inp', s='0', b='0', w=1, l=.15))
    if nested:
        mod.add_module(target, 'dut', nets={'inp': 'inp', 'n': 'n'})
    mod.add_source(sources.VoltageSource(
        'drive', positive='inp', negative='0', dc=0,
        pulse=sources.Pulse(0, 1, 0, tr=1e-9, tf=1e-9, pw=5e-6, per=10e-6)))
    mod.add_sim_type(Transient(step=1e-8, stop=2e-6))
    return mod


@pytest.mark.parametrize('return_net', ['gnd', 'GND', 'Gnd'])
@pytest.mark.parametrize('nested', [False, True])
def test_xyce_rejects_ground_before_writing(tmp_path, return_net, nested):
    from silk.exceptions import CircuitError
    mod = _ground_rc(tmp_path, return_net, nested)
    mod.set_save_list(['n'])  # A filtered-out net must still be checked.
    with pytest.raises(CircuitError) as err:
        mod.generate_netlist()
    message = str(err.value)
    assert ('xdut.' if nested else '') + return_net in message
    assert 'Xyce' in message and '0' in message and 'vss' in message
    assert not (tmp_path / 'ground.cir').exists()


@pytest.mark.parametrize('kind', ['declared', 'source', 'passive', 'device', 'binding', 'custom'])
def test_xyce_ground_producers(tmp_path, kind):
    from silk import sources
    from silk.elements.passives import Resistor
    from silk.exceptions import CircuitError
    mod = _ground_rc(tmp_path, nested=(kind == 'binding'))
    if kind == 'declared':
        mod.add_net('gnd')
    elif kind == 'source':
        mod.add_source(sources.CurrentSource('bias', positive='n', negative='gnd', dc=1e-6))
    elif kind == 'passive':
        mod.add(Resistor('load', 'n', 'gnd', 1000))
    elif kind == 'device':
        mod.add(mod.pdk_details.nmos1p8('mn', d='n', g='inp', s='gnd', b='0', w=1, l=.15))
    elif kind == 'binding':
        mod.connect_module('dut', inp='gnd')
    else:
        mod.add_custom_line('Rhidden n gnd 1k')
    with pytest.raises(CircuitError, match='gnd'):
        mod.generate_netlist()
    assert not (tmp_path / 'ground.cir').exists()


@pytest.mark.parametrize('name', ['0', 'agnd', 'ground', 'vss'])
def test_xyce_other_ground_names_still_emit(tmp_path, name):
    mod = _ground_rc(tmp_path, name, nested=True)
    from pathlib import Path
    assert Path(mod.generate_netlist()).exists()


def test_ngspice_gnd_still_emits(tmp_path):
    mod = _ground_rc(tmp_path, 'gnd', nested=True, simulator='ngspice')
    assert 'gnd' in open(mod.generate_netlist()).read()


def test_xyce_ground_added_after_preparation_refuses_launch(tmp_path):
    from silk.exceptions import CircuitError
    mod = _ground_rc(tmp_path)
    mod.generate_netlist()
    mod.add_net('GND')
    with pytest.raises(CircuitError, match='GND'):
        mod.run_simulator()


@pytest.mark.slow
@pytest.mark.requires_xyce
@pytest.mark.parametrize('nested', [False, True], ids=['cap_zero', 'sub_zero'])
def test_xyce_zero_return_matches_ngspice(tmp_path, nested):
    import numpy as np
    from tests._env import HAVE_NGSPICE, HAVE_XYCE
    if not (HAVE_NGSPICE and HAVE_XYCE):
        pytest.skip('needs configured ngspice and Xyce')
    values = []
    for simulator in ('ngspice', 'xyce'):
        mod = _ground_rc(tmp_path / simulator, nested=nested, simulator=simulator)
        # The extra ideal series source changes the adaptive integration grid.
        # Request accuracy explicitly instead of pinning default-grid samples.
        mod.set_simulator_options(reltol=1e-5, abstol=1e-12)
        mod.generate_netlist()
        mod.run_simulator()
        sig = mod.load_signal('n', 'V')
        values.append(float(np.interp(1e-6, sig.x, sig.y)))
    # First-order RC driven by a 1 ns linear ramp; tau = 1 us, t = 1 us.
    expected = 1 - np.expm1(.001) / .001 * np.exp(-1)
    assert values[0] == pytest.approx(expected, abs=5e-6)
    assert values[1] == pytest.approx(expected, abs=5e-6)
    assert values[1] == pytest.approx(values[0], abs=5e-6)


@pytest.mark.parametrize('nested', [False, True])
def test_xyce_rejects_fault_added_ground(tmp_path, monkeypatch, nested):
    from silk.exceptions import CircuitError
    from silk.faults import FaultSet, short_ds
    from silk.faults import application
    mod = _ground_rc(tmp_path, nested=nested, fault_target=True)
    target = mod.module_dict['block']['device'] if nested else mod
    original = application._realize_fault

    def realize(*args, **kwargs):
        result = original(*args, **kwargs)
        result.added.append(application._AddedElement(
            kind='resistor', name='extra', nets={'p': 'n', 'n': 'gnd'}, value=1000))
        return result

    # Fault realizations add terminals without registering Module nets.
    monkeypatch.setattr(application, '_realize_fault', realize)
    with pytest.raises(CircuitError) as err:
        mod.generate_netlist(fault_set=FaultSet([short_ds('xdut.mn' if nested else 'mn')]))
    assert ('xdut.gnd' if nested else 'gnd') in str(err.value)
    assert not (tmp_path / 'ground.cir').exists()
    assert not hasattr(target.elem_dict['mn']['device'], '_silk_pending_fault')


def test_xyce_does_not_refuse_gnd_in_comments_or_values(tmp_path):
    mod = _ground_rc(tmp_path)
    mod.add_custom_line('* gnd is discussed here')
    mod.add_custom_line('.param gnd=1000')
    mod.add_custom_line('Rparam n 0 {gnd}')
    assert 'Rparam n 0 {gnd}' in open(mod.generate_netlist()).read()


@pytest.mark.parametrize('nested', [False, True], ids=['cap_gnd', 'sub_gnd'])
def test_xyce_gnd_simulate_refuses_before_launch(tmp_path, nested, monkeypatch):
    from silk.exceptions import CircuitError
    mod = _ground_rc(tmp_path, 'gnd', nested=nested)

    def unexpected_launch(*args, **kwargs):
        pytest.fail('a gnd deck reached the simulator')

    monkeypatch.setattr(type(mod), 'run_simulator', unexpected_launch)
    with pytest.raises(CircuitError, match='gnd'):
        mod.simulate()
    assert not (tmp_path / 'ground.cir').exists()


@pytest.mark.parametrize('line', [
    'Ehidden n 0 POLY(2) inp 0 GND 0 0 1 1',
    'Bhidden n 0 V={v(GND)}',
])
def test_xyce_custom_control_ground_refused(tmp_path, line):
    from silk.exceptions import CircuitError
    mod = _ground_rc(tmp_path)
    mod.add_custom_line(line)
    with pytest.raises(CircuitError, match='GND'):
        mod.generate_netlist()


def test_xyce_gnd_model_name_is_not_a_net(tmp_path):
    mod = _ground_rc(tmp_path)
    mod.add_custom_line('.model gnd NPN')
    mod.add_custom_line('Qextra n inp 0 gnd 1')
    assert 'Qextra n inp 0 gnd 1' in open(mod.generate_netlist()).read()


def test_xyce_ground_text_in_inline_comment_is_allowed(tmp_path):
    mod = _ground_rc(tmp_path)
    mod.add_custom_line('Rextra n 0 1k ; {v(gnd)} was the old return')
    mod.generate_netlist()


def test_xyce_bjt_ground_with_numeric_model_name_is_refused(tmp_path):
    from silk.exceptions import CircuitError
    mod = _ground_rc(tmp_path)
    mod.add_custom_line('.model 2N3904 NPN')
    mod.add_custom_line('Qextra n inp gnd 2N3904')
    with pytest.raises(CircuitError, match='gnd'):
        mod.generate_netlist()
