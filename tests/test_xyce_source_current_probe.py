"""``tb.probe('i(<source>)')`` on Xyce (TASK-346).

Xyce prints only what its ``.print`` line lists, so a source branch current
was never in the rawfile and the read raised "Signal 'Vvd' (type='I') not
found". The Xyce netlister now appends ``I(<SOURCE>)`` for every
``probe('i(...)')`` (measured on Xyce 7.10: also valid in DC sweeps, in
Transient, and as one complex column in AC). ngspice already saves every
branch current, so its deck is untouched, and a probe-free Xyce deck is
byte-identical to before.
"""
import os
import re

import numpy as np
import pytest

from silk import sources
from silk.circuit.circuit_elements import PortDir
from silk.circuit.modules import Module
from silk.exceptions import NotSupportedError
from silk.pdk import load_pdk
from silk.simulation import analysis

from tests._env import needs_ngspice, needs_pdk, needs_pdk_named
from tests._env import scratch_dir

_XYCE_BIN = os.path.expanduser('~/xyce/Xyce-7.10.0/bin/Xyce')
_have_xyce = os.path.exists(_XYCE_BIN)

_PDKS = {'sky130': ('nmos1p8', 0.15, 1.65, 1.0),
         'gf180mcu': ('nmos3p3', 0.5, 1.65, 1.2)}
_CORNER = {'sky130': 'tt', 'gf180mcu': 'typical'}

_ANALYSES = {
    'op': lambda: analysis.Op(),
    'tran': lambda: analysis.Transient(step=1e-9, stop=5e-9),
    'dc': lambda: analysis.DC(source='Vvg', start=0.0, stop=1.8, step=0.3),
}


def _bench(pdk_name, sim, an, current_probe='i(vd)'):
    devname, length, vd, vg = _PDKS[pdk_name]
    pdk = load_pdk(pdk_name)
    dut = Module(pdk_name, scratch_dir(), 'dut.cir', pdk_details=pdk,
                 module_name='dut')
    dut.define_module_ports({'d': PortDir.OUT, 'g': PortDir.IN,
                             's': PortDir.GROUND, 'b': PortDir.GROUND})
    dut.add_element(getattr(pdk, devname), 'M1',
                    nets={'d': 'd', 'g': 'g', 's': 's', 'b': 'b'},
                    parameters={'w': 2.0, 'l': length, 'm': 1})
    t = dut.create_test('ichk')
    t.add_module_under_test({'d': 'vd', 'g': 'vg', 's': '0', 'b': '0'})
    t.add_source(sources.VoltageSource('vd', positive='vd', negative='0', dc=vd))
    ac = 1 if an == 'ac' else 0
    t.add_source(sources.VoltageSource('vg', positive='vg', negative='0',
                                       dc=vg, ac=ac))
    dut.simulator = sim
    if an == 'ac':
        t.add_sim_type(analysis.AC('dec', 5, 1e3, 1e6))
    else:
        t.add_sim_type(_ANALYSES[an]())
    sig = t.probe(current_probe) if current_probe else None
    return dut, t, sig


def _deck(t):
    spec = t._prepare_run(corner='tt')
    spec_netlist_deck_path = spec.netlist.generate_netlist(corner=spec.corner_str)
    with open(spec_netlist_deck_path) as fh:
        return re.sub(r'/tmp/\S+', '<tmp>', fh.read())


@needs_pdk
class TestEmission:
    def test_xyce_print_line_lists_the_source_branch_current(self):
        _, t, _ = _bench('sky130', 'xyce', 'op')
        line = [l for l in _deck(t).splitlines() if l.startswith('.print')][0]
        assert line.endswith('v(vd) v(vg) I(VVD)')

    def test_two_probes_on_one_source_give_one_term(self):
        _, t, _ = _bench('sky130', 'xyce', 'op')      # i(vd)
        t.probe('i(vd)')
        t.probe('i(VD)')
        line = [l for l in _deck(t).splitlines() if l.startswith('.print')][0]
        assert line.upper().count('I(VVD)') == 1

    def test_probe_free_xyce_deck_is_unchanged(self):
        """No i() probe: no I() term, and a v() probe alone changes nothing."""
        _, t, _ = _bench('sky130', 'xyce', 'tran', current_probe=None)
        base = _deck(t)
        assert 'I(' not in base
        _, t2, _ = _bench('sky130', 'xyce', 'tran', current_probe=None)
        t2.probe('vd')
        assert _deck(t2) == base

    def test_ngspice_deck_is_identical_with_and_without_the_probe(self):
        _, t0, _ = _bench('sky130', 'ngspice', 'tran', current_probe=None)
        _, t1, _ = _bench('sky130', 'ngspice', 'tran')
        assert _deck(t0) == _deck(t1)

    def test_an_unknown_source_card_is_refused_before_the_run(self):
        _, t, _ = _bench('sky130', 'xyce', 'op', current_probe='i(v.xdut.vin)')
        with pytest.raises(NotSupportedError, match='not a source card'):
            _deck(t)


@pytest.mark.slow
@pytest.mark.skipif(not _have_xyce, reason='Xyce 7.10 not installed')
@needs_ngspice
@needs_pdk
class TestRealRuns:
    @pytest.mark.parametrize('pdk_name', [
        'sky130',
        pytest.param('gf180mcu', marks=[
            pytest.mark.requires_gf180, needs_pdk_named('gf180mcu'),
        ]),
    ])
    @pytest.mark.parametrize('an', ['op', 'tran', 'dc'])
    def test_xyce_matches_ngspice(self, monkeypatch, pdk_name, an):
        monkeypatch.setenv('SILK_XYCE_BIN', _XYCE_BIN)
        got = {}
        for sim in ('ngspice', 'xyce'):
            dut, t, sig = _bench(pdk_name, sim, an)
            dut.run_test('ichk', corner=_CORNER[pdk_name])
            got[sim] = np.asarray(sig.y, dtype=float)
        n, x = got['ngspice'], got['xyce']
        assert np.max(np.abs(n)) > 1e-6            # a real, non-zero current
        # Stated tolerance: sky130 agrees to 1e-4 relative; gf180mcu's two
        # simulators differ by ~0.1% in the model itself (measured 0.06% at
        # the operating point, 0.10% worst over the sweep), so 2e-3 there.
        rtol = 1e-4 if pdk_name == 'sky130' else 2e-3
        atol = 1e-9 + 1e-6 * np.max(np.abs(n))
        if an == 'tran':
            # The two simulators pick different time grids (59 vs 15
            # points); the bias is constant, so compare the settled value.
            np.testing.assert_allclose(x[-1], n[-1], rtol=rtol, atol=atol)
            np.testing.assert_allclose(np.mean(x), np.mean(n), rtol=rtol,
                                       atol=atol)
        else:
            assert x.shape == n.shape
            np.testing.assert_allclose(x, n, rtol=rtol, atol=atol)

    def test_duplicate_probes_read_the_same_value(self, monkeypatch):
        monkeypatch.setenv('SILK_XYCE_BIN', _XYCE_BIN)
        dut, t, a = _bench('sky130', 'xyce', 'op')
        b = t.probe('i(VD)')
        dut.run_test('ichk', corner='tt')
        assert float(a.y[0]) == float(b.y[0]) != 0.0

    @pytest.mark.parametrize('pdk_name', [
        'sky130',
        pytest.param('gf180mcu', marks=[
            pytest.mark.requires_gf180, needs_pdk_named('gf180mcu'),
        ]),
    ])
    def test_ac_reads_back_as_one_complex_column(self, monkeypatch, pdk_name):
        monkeypatch.setenv('SILK_XYCE_BIN', _XYCE_BIN)
        dut, t, sig = _bench(pdk_name, 'xyce', 'ac')
        dut.run_test('ichk', corner=_CORNER[pdk_name])
        y = np.asarray(sig.y)
        assert y.shape[0] == 16                    # 3 decades x 5 pts + 1
        assert np.iscomplexobj(y)
        assert np.all(np.isfinite(y)) and np.max(np.abs(y)) > 0
