"""Finite-bit NumPy/SPICE parity, including DC/AC and long explicit decks.

Voltage is read directly; a current source drives a one-ohm load so its
measured voltage is the same numeric current. No transistor models needed.
"""
import numpy as np
import pytest

from silk import PortDir, sources
from silk.circuit.modules import Module
from silk.elements import elements
from silk.pdk import load_pdk
from silk.simulation import AC, Op, Transient
from tests._env import HAVE_XYCE, needs_ngspice

SIMULATORS = [pytest.param('ngspice', marks=needs_ngspice),
              pytest.param('xyce', marks=[pytest.mark.requires_xyce,
                  pytest.mark.skipif(not HAVE_XYCE, reason='Xyce setting unavailable')])]
pytestmark = pytest.mark.slow


def bench(tmp_path, simulator, quantity, wave, analysis, dc=0, ac=0):
    dut = Module('sky130', str(tmp_path), 'load.cir', pdk_details=load_pdk('sky130'),
                 module_name='load')
    dut.define_module_ports({'out': PortDir.IN})
    dut.add(elements.resistor('r', p='out', n='0', value=1))
    dut.simulator = simulator
    tb = dut.create_test('bits')
    tb.add_module_under_test({'out': 'out'})
    # Xyce Op() sweeps the first voltage source at its declared dc. Give
    # that adapter a separate fixed source so it cannot overwrite the
    # waveform-under-test DC semantics (also needed for current-only decks).
    tb.add_source(sources.VoltageSource('op_anchor', 'anchor', '0', dc=1))
    cls = sources.VoltageSource if quantity == 'voltage' else sources.CurrentSource
    pins = ('out', '0') if quantity == 'voltage' else ('0', 'out')
    tb.add_source(cls('data', *pins, waveform=wave, dc=dc, ac=ac))
    tb.add_sim_type(analysis)
    signal = tb.probe('out')
    branch = tb.probe('i(data)') if quantity == 'voltage' else None
    tb.run()
    return tb, signal, branch


def waveform(case):
    params = dict(v1=.2, v2=1.2, ui=1e-6, td=.7e-6, tr=.12e-6, tf=.28e-6)
    if case == 'inverted':
        params.update(v1=1.2, v2=.2)
    if case == 'prbs':
        return sources.PRBS(7, 18, seed=1, **params)
    return sources.BitPattern({'constant0': '000', 'constant1': '111',
                               'repeat': '10', 'inverted': '10100',
                               'pattern': '11010'}[case],
                              repeat=3 if case == 'repeat' else 1, **params)


@pytest.mark.parametrize('simulator', SIMULATORS)
@pytest.mark.parametrize('quantity', ['voltage', 'current'])
@pytest.mark.parametrize('case', ['pattern', 'inverted', 'constant0', 'constant1', 'repeat', 'prbs'])
@pytest.mark.parametrize('dc', [0, .65])
def test_transient_numpy_parity(tmp_path, simulator, quantity, case, dc):
    wave = waveform(case)
    stop = wave.td + (len(wave.stream) + 1) * wave.ui
    _, sig, branch = bench(tmp_path, simulator, quantity, wave,
                           Transient(step=wave.ui / 100, stop=stop), dc=dc, ac=.3)
    times, measured = np.asarray(sig.x), np.asarray(sig.y)
    # Compare solver samples too, so interpolation cannot conceal a bad ramp.
    np.testing.assert_allclose(measured, dc + wave.to_numpy(times), atol=2e-6, rtol=2e-6)
    queries = [0, wave.td / 2, stop]
    for k, bit in enumerate(wave.stream):
        start = wave.td + k * wave.ui
        edge = wave.tr if bit else wave.tf
        queries += [start - wave.ui * .01, start, start + edge / 2,
                    start + edge, start + edge + wave.ui * .01, start + wave.ui / 2]
    # Adaptive grids can bracket a breakpoint: allow 0.2% of the 1-unit swing.
    np.testing.assert_allclose(np.interp(queries, times, measured),
                               dc + wave.to_numpy(queries), atol=.002, rtol=0)
    if branch is not None:
        np.testing.assert_allclose(branch.y, -measured, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize('simulator', SIMULATORS)
@pytest.mark.parametrize('quantity', ['voltage', 'current'])
@pytest.mark.parametrize('kind', ['pattern', 'prbs'])
@pytest.mark.parametrize('dc', [0, .65])
def test_operating_point_and_ac(tmp_path, simulator, quantity, kind, dc):
    wave = waveform(kind)
    _, sig, _ = bench(tmp_path / 'op', simulator, quantity, wave, Op(), dc=dc)
    expected = dc + wave.to_numpy([0])[0]
    np.testing.assert_allclose(sig.y, expected, atol=1e-9, rtol=1e-9)
    _, ac_sig, _ = bench(tmp_path / 'ac', simulator, quantity, wave,
                         AC('lin', 3, 1e3, 3e3), dc=dc, ac=.3)
    np.testing.assert_allclose(ac_sig.y, .3 + 0j, atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize('simulator', SIMULATORS)
@pytest.mark.parametrize('quantity', ['voltage', 'current'])
def test_full_prbs15_explicit_deck(tmp_path, simulator, quantity):
    wave = sources.PRBS(15, 32767, v1=.2, v2=1.2, ui=1e-6, tr=.1e-6, tf=.2e-6)
    stop = 32768e-6
    tb, sig, _ = bench(tmp_path, simulator, quantity, wave,
                       Transient(step=.1e-6, stop=stop))
    deck = wave.to_spice()
    assert deck.count('\n+ ') > 30000
    assert max(map(len, deck.splitlines())) < 100
    assert len(sig.x) > 32767
    # Every bit center is checked, including the last bit and final hold.
    centers = (np.arange(32767) + .5) * wave.ui
    expected = .2 + np.asarray(wave.stream)
    np.testing.assert_allclose(np.interp(centers, sig.x, sig.y), expected,
                               atol=2e-5, rtol=0)
    np.testing.assert_allclose(sig.y, wave.to_numpy(sig.x), atol=2e-5, rtol=2e-6)
    assert sig.y[-1] == pytest.approx(expected[-1], abs=2e-5)
