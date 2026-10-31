"""TASK-339: gf180mcu on Xyce.

sm141064.ngspice relies on ngspice-only `.LIB`/`.ENDL` looseness (a bare
`.ENDL` closing a named `.LIB`, and a section whose `.ENDL <name>` does not
match its `.LIB <name>`) that Xyce 7.10's stricter parser rejects outright
(see the committed evidence at
tests/fixtures/op_point/gf180mcu/xyce_probe_gf180mcu_{nmos3p3,pmos3p3}.log).
The PDK install ships a separate, Xyce-native model set under
libs.tech/xyce/ (verified to use the IDENTICAL `.LIB` section names as the
ngspice file), so the fix is silk pointing Xyce at that file instead of
translating anything.

The route (reviewer-approved 2026-09-28) is generic and manifest-driven: any
PDK's `library:` block may declare `xyce_lib_path` / `xyce_extra_includes`,
and any corner's `includes:` may declare a sibling `xyce_includes:` — see
silk/pdk/_loader.py and silk/circuit/xyce_netlister.py. gf180mcu.pdk.yaml is
simply the first manifest to use them; no code here checks the PDK name.

Four groups of tests:

* ``TestXyceModelSetSections`` -- the new `xyce_includes` sections actually
  exist in the PDK's Xyce-native model file (gf180mcu install only, no Xyce
  binary needed).
* ``TestDeckEmissionUnchanged`` -- condition (2) of the ruling: gf180mcu's
  ngspice deck and sky130's Xyce deck are byte-identical to before this
  change.
* ``TestGF180XyceRealRun`` -- condition (3): real Xyce 7.10 vs real ngspice,
  DC operating point plus a short transient, for nmos3p3 AND pmos3p3, plus
  the `ss` corner (not just `typical`) to prove the section mapping.
"""
import os
import re

import numpy as np
import pytest

from silk.circuit.spice_netlister import PySpiceNetlister
from silk.circuit.xyce_netlister import XyceNetlister
from silk.elements import elements
from silk.circuit.modules import Module
from silk import sources
from silk.simulation import analysis

from tests._env import needs_pdk_named, needs_ngspice, pdk_available
from tests._env import scratch_dir


def _load_manifest_only(name):
    """Load a bundled PDK manifest at a fixed, fake root -- deterministic
    output with no installed PDK required (mirrors test_xyce_netlister.py's
    helper of the same name)."""
    from silk import load_pdk
    return load_pdk(name, pdk_root='/PDK_ROOT')


# ---------------------------------------------------------------------------
# The xyce_includes sections silk would emit actually exist in the PDK's
# Xyce-native model file (mirrors test_gf180mcu_passives.py's
# test_every_corner_section_exists_in_the_model_file, for the xyce sibling).
# ---------------------------------------------------------------------------

@needs_pdk_named('gf180mcu')
class TestXyceModelSetSections:

    @staticmethod
    @pytest.fixture(scope='class')
    def pdk():
        from silk import load_pdk
        return load_pdk('gf180mcu')

    @staticmethod
    @pytest.fixture(scope='class')
    def xyce_model_text(pdk):
        return open(pdk.xyce_library_location, errors='ignore').read()

    def test_gf180_declares_xyce_fields(self, pdk):
        assert pdk.xyce_library_location is not None
        assert pdk.xyce_library_location.endswith('/xyce/sm141064.xyce')
        assert pdk.xyce_extra_includes == [
            os.path.join(os.path.dirname(pdk.xyce_library_location),
                         'design.xyce')]

    def test_every_corner_xyce_section_exists_in_the_xyce_model_file(
            self, pdk, xyce_model_text):
        defined = {m.group(1).lower() for m in
                   re.finditer(r'^\s*\.lib\s+([A-Za-z_][\w]*)\s*$',
                               xyce_model_text, re.I | re.M)}
        for corner, meta in pdk.corners.items():
            for inc in meta['xyce_includes']:
                assert inc['section'].lower() in defined, (corner, inc['section'])

    def test_xyce_sections_are_the_same_set_as_ngspice_sections(self, pdk):
        """The fix is a file swap, not a re-derivation of the section list:
        every corner's xyce_includes names the SAME sections as includes."""
        for corner, meta in pdk.corners.items():
            ng = [inc['section'] for inc in meta['includes']]
            xy = [inc['section'] for inc in meta['xyce_includes']]
            assert ng == xy, corner


# ---------------------------------------------------------------------------
# Condition (2): gf180 ngspice decks and all sky130 Xyce decks are
# byte-identical before and after this change.
# ---------------------------------------------------------------------------

class TestDeckEmissionUnchanged:

    def test_gf180_ngspice_deck_unchanged(self):
        """PySpiceNetlister (ngspice) still emits gf180's `includes:` list
        against sm141064.ngspice -- the new xyce_* fields are additive."""
        pdk = _load_manifest_only('gf180mcu')
        nl = PySpiceNetlister.__new__(PySpiceNetlister)
        nl.pdk_details = pdk
        out = nl.generate_pdk_include('typical')
        assert out == (
            '*** Including PDK library\n'
            '.lib "/PDK_ROOT/gf180mcuD/libs.tech/ngspice/sm141064.ngspice" typical\n'
            '.lib "/PDK_ROOT/gf180mcuD/libs.tech/ngspice/sm141064.ngspice" res_typical\n'
            '.lib "/PDK_ROOT/gf180mcuD/libs.tech/ngspice/sm141064.ngspice" moscap_typical\n'
            '.lib "/PDK_ROOT/gf180mcuD/libs.tech/ngspice/sm141064.ngspice" mimcap_typical\n'
            '.lib "/PDK_ROOT/gf180mcuD/libs.tech/ngspice/sm141064.ngspice" cap_mim\n'
            '.lib "/PDK_ROOT/gf180mcuD/libs.tech/ngspice/sm141064.ngspice" diode_typical\n'
            '.lib "/PDK_ROOT/gf180mcuD/libs.tech/ngspice/sm141064.ngspice" bjt_typical\n'
            '.include "/PDK_ROOT/gf180mcuD/libs.tech/ngspice/design.ngspice"\n'
        )

    def test_sky130_xyce_deck_unchanged(self):
        """XyceNetlister falls back to the ngspice-oriented fields when a
        manifest (sky130's) declares no xyce_* keys at all."""
        pdk = _load_manifest_only('sky130')
        nl = XyceNetlister.__new__(XyceNetlister)
        nl.pdk_details = pdk
        out = nl.generate_pdk_include('tt')
        assert out == (
            '*** Including PDK library\n'
            '.lib "/PDK_ROOT/sky130A/libs.tech/ngspice/sky130.lib.spice" tt\n'
            '.include "/PDK_ROOT/sky130A/libs.ref/sky130_fd_pr/spice/'
            'sky130_fd_pr__cap_vpp_11p5x11p7_m1m2_noshield.model.spice"\n'
        )

    def test_sky130_has_no_xyce_fields(self):
        """sky130's own ngspice file already parses under Xyce (no PDK
        Xyce-native twin exists for it) -- confirms the fallback above is
        exercising the "manifest declares nothing" path, not a coincidence."""
        pdk = _load_manifest_only('sky130')
        assert pdk.xyce_library_location is None
        assert pdk.xyce_extra_includes is None


# ---------------------------------------------------------------------------
# Condition (3): real Xyce 7.10 vs real ngspice, DC + a short transient, for
# nmos3p3 AND pmos3p3, plus the `ss` corner to prove the section mapping.
# ---------------------------------------------------------------------------

_XYCE_BIN = os.path.expanduser('~/xyce/Xyce-7.10.0/bin/Xyce')
_have_xyce_and_gf180 = os.path.exists(_XYCE_BIN) and pdk_available('gf180mcu')


def _build_nmos_dc(pdk, simulator, corner, tmp):
    mod = Module('gf180mcu', tmp, 'nmos_dc.cir', pdk_details=pdk)
    mod.simulator = simulator
    mod.netlister = mod.backend.make_netlister(pdk)
    for n in ('vdd', 'vout', 'vg', '0'):
        mod.add_net(n)
    with mod:
        pdk.nmos3p3('M1', d='vout', g='vg', s='0', b='0', w=10.0, l=0.28, m=1)
        elements.resistor('r1', p='vdd', n='vout', value=5000.0)
    mod.add_source(sources.vdc('vdd', positive='vdd', negative='0', dc=3.3))
    mod.add_source(sources.vdc('vg', positive='vg', negative='0', dc=1.65))
    mod.add_sim_type(analysis.Op())
    mod.generate_netlist(corner=corner)
    return mod


def _build_pmos_dc(pdk, simulator, corner, tmp):
    mod = Module('gf180mcu', tmp, 'pmos_dc.cir', pdk_details=pdk)
    mod.simulator = simulator
    mod.netlister = mod.backend.make_netlister(pdk)
    for n in ('vdd', 'vout', 'vg', '0'):
        mod.add_net(n)
    with mod:
        pdk.pmos3p3('M1', d='vout', g='vg', s='vdd', b='vdd', w=10.0, l=0.28, m=1)
        elements.resistor('r1', p='vout', n='0', value=5000.0)
    mod.add_source(sources.vdc('vdd', positive='vdd', negative='0', dc=3.3))
    mod.add_source(sources.vdc('vg', positive='vg', negative='0', dc=1.8))
    mod.add_sim_type(analysis.Op())
    mod.generate_netlist(corner=corner)
    return mod


def _build_nmos_tran(pdk, simulator, tmp):
    mod = Module('gf180mcu', tmp, 'nmos_tr.cir', pdk_details=pdk)
    mod.simulator = simulator
    mod.netlister = mod.backend.make_netlister(pdk)
    for n in ('vdd', 'vout', 'vg', '0'):
        mod.add_net(n)
    with mod:
        pdk.nmos3p3('M1', d='vout', g='vg', s='0', b='0', w=10.0, l=0.28, m=1)
        elements.resistor('r1', p='vdd', n='vout', value=5000.0)
    mod.add_source(sources.vdc('vdd', positive='vdd', negative='0', dc=3.3))
    mod.add_source(sources.vpulse('vg', positive='vg', negative='0',
                                  v1=1.0, v2=2.0, td=2e-9, tr=1e-9, tf=1e-9,
                                  pw=5e-9, per=20e-9))
    mod.add_sim_type(analysis.Transient(1e-10, 2e-8))
    mod.generate_netlist(corner='typical')
    return mod


def _build_pmos_tran(pdk, simulator, tmp):
    mod = Module('gf180mcu', tmp, 'pmos_tr.cir', pdk_details=pdk)
    mod.simulator = simulator
    mod.netlister = mod.backend.make_netlister(pdk)
    for n in ('vdd', 'vout', 'vg', '0'):
        mod.add_net(n)
    with mod:
        pdk.pmos3p3('M1', d='vout', g='vg', s='vdd', b='vdd', w=10.0, l=0.28, m=1)
        elements.resistor('r1', p='vout', n='0', value=5000.0)
    mod.add_source(sources.vdc('vdd', positive='vdd', negative='0', dc=3.3))
    mod.add_source(sources.vpulse('vg', positive='vg', negative='0',
                                  v1=2.3, v2=1.3, td=2e-9, tr=1e-9, tf=1e-9,
                                  pw=5e-9, per=20e-9))
    mod.add_sim_type(analysis.Transient(1e-10, 2e-8))
    mod.generate_netlist(corner='typical')
    return mod


@pytest.mark.slow
@pytest.mark.requires_xyce
@pytest.mark.requires_gf180
@needs_ngspice
@pytest.mark.skipif(not _have_xyce_and_gf180,
                    reason='needs the Xyce binary and an installed gf180mcu PDK')
class TestGF180XyceRealRun:
    """Same bench, run through the real silk path on both backends
    (Module -> generate_netlist -> run_simulator -> load_signal), compared
    directly against each other -- no hand-derived expected voltage. R-loaded
    common-source/source-follower stages so V(vout) reflects the device's
    actual I-V curve, not just the driving DC source."""

    @pytest.fixture(autouse=True)
    def _xyce_bin(self, monkeypatch):
        monkeypatch.setenv('SILK_XYCE_BIN', _XYCE_BIN)

    @staticmethod
    @pytest.fixture(scope='class')
    def pdk():
        from silk import load_pdk
        return load_pdk('gf180mcu')

    def _dc_both(self, pdk, builder, corner):
        results = {}
        for sim in ('ngspice', 'xyce'):
            tmp = scratch_dir()
            mod = builder(pdk, sim, corner, tmp)
            mod.run_simulator()
            results[sim] = float(mod.load_signal('vout', 'V').y[0])
        return results

    def _tran_both(self, pdk, builder):
        results = {}
        for sim in ('ngspice', 'xyce'):
            tmp = scratch_dir()
            mod = builder(pdk, sim, tmp)
            mod.run_simulator()
            sig = mod.load_signal('vout', 'V')
            results[sim] = (np.asarray(sig.x), np.asarray(sig.y))
        return results

    @pytest.mark.parametrize('builder, corner', [
        (_build_nmos_dc, 'typical'),
        (_build_pmos_dc, 'typical'),
        (_build_nmos_dc, 'ss'),   # proves the section mapping, not just 'typical'
    ])
    def test_dc_operating_point_matches_ngspice(self, pdk, builder, corner):
        r = self._dc_both(pdk, builder, corner)
        assert r['xyce'] == pytest.approx(r['ngspice'], abs=5e-3), (
            f"corner={corner}: ngspice={r['ngspice']!r} xyce={r['xyce']!r}")

    @pytest.mark.parametrize('builder', [_build_nmos_tran, _build_pmos_tran])
    def test_transient_matches_ngspice(self, pdk, builder):
        r = self._tran_both(pdk, builder)
        t_ng, y_ng = r['ngspice']
        t_xy, y_xy = r['xyce']
        # Xyce and ngspice choose different internal timesteps; compare Xyce
        # interpolated onto ngspice's grid, which is denser here.
        y_xy_on_ng = np.interp(t_ng, t_xy, y_xy)
        max_diff = np.max(np.abs(y_xy_on_ng - y_ng))
        assert max_diff < 0.05, f'max |V(vout) diff| = {max_diff:.4g} V'

    def test_working_deck_is_the_fixed_xyce_model_set(self, pdk, tmp_path):
        """The emitted deck actually names the PDK's xyce/ model file, not
        the ngspice one -- ties this test to the manifest fix, not an
        accidental pass."""
        mod = _build_nmos_dc(pdk, 'xyce', 'typical', str(tmp_path))
        deck = open(os.path.join(str(tmp_path), 'nmos_dc.cir')).read()
        assert '/xyce/sm141064.xyce' in deck
        assert '/ngspice/sm141064.ngspice' not in deck
