"""Canonical noise loading across backends (T7).

`load_signal('noise_out'|'noise_in')` returns the noise spectral density in
V/√Hz regardless of simulator. ngspice's rawfile already holds
`onoise_spectrum`/`inoise_spectrum` (V/√Hz); Xyce writes a CSV with
`ONOISE`/`INOISE` in V²/Hz, which the backend sqrt-converts. Legacy/terse
aliases (`onoise_spectrum`, `onoise`, …) keep resolving.
"""
import os
from types import SimpleNamespace

import numpy as np
import pytest

from silk.simulation.backend import SimulatorBackend, RawResult, _NOISE_PORT
from silk.simulation.analysis import Noise, AC
from silk.simulation.xyce_runner import read_xyce_noise_csv
from silk.circuit.xyce_netlister import XyceNetlister
from silk.exceptions import SimulationError


# ---------------------------------------------------------------------------
# ngspice: canonical names alias the native onoise/inoise vectors (no unit change)
# ---------------------------------------------------------------------------

def _ngspice_noise_native():
    f = np.logspace(0, 6, 7)
    onoise = np.linspace(1e-5, 1e-8, 7)   # V/√Hz already
    inoise = onoise / 50.0
    return SimpleNamespace(nvars=3, vectors=[
        SimpleNamespace(name='frequency', data=f, type='frequency'),
        SimpleNamespace(name='onoise_spectrum', data=onoise, type='voltage-density'),
        SimpleNamespace(name='inoise_spectrum', data=inoise, type='voltage-density'),
    ])


@pytest.mark.parametrize('name', ['noise_out', 'onoise_spectrum', 'onoise'])
def test_ngspice_noise_out_aliases(name):
    raw = RawResult(native=_ngspice_noise_native(), source_path='/x/rawfile.raw')
    sig = SimulatorBackend.get('ngspice').load_signal(raw, name)
    np.testing.assert_array_equal(sig.y, raw.native.vectors[1].data)  # V/√Hz, unchanged
    assert sig.name == 'noise_out'


@pytest.mark.parametrize('name', ['noise_in', 'inoise_spectrum', 'inoise'])
def test_ngspice_noise_in_aliases(name):
    raw = RawResult(native=_ngspice_noise_native(), source_path='/x/rawfile.raw')
    sig = SimulatorBackend.get('ngspice').load_signal(raw, name)
    np.testing.assert_array_equal(sig.y, raw.native.vectors[2].data)
    assert sig.name == 'noise_in'


def test_node_signal_still_routes_to_from_rawfile():
    """A non-noise name is unaffected by the noise routing."""
    native = SimpleNamespace(nvars=2, vectors=[
        SimpleNamespace(name='frequency', data=np.array([1.0, 2.0]), type='frequency'),
        SimpleNamespace(name='v(out)', data=np.array([0.5, 0.4]), type='voltage'),
    ])
    sig = SimulatorBackend.get('ngspice').load_signal(
        RawResult(native=native, source_path='/x/rawfile.raw'), 'out', type='V')
    np.testing.assert_array_equal(sig.y, native.vectors[1].data)


# ---------------------------------------------------------------------------
# Xyce: CSV reader + sqrt(V²/Hz) -> V/√Hz
# ---------------------------------------------------------------------------

def _write_xyce_noise_csv(tmp_path):
    csv = tmp_path / 'noise.csv'
    csv.write_text(
        '"FREQ","ONOISE","INOISE"\n'
        '1.00000000e+00,4.00000000e-10,1.00000000e-12\n'   # sqrt -> 2e-5, 1e-6
        '1.00000000e+01,1.00000000e-10,2.50000000e-13\n'   # sqrt -> 1e-5, 5e-7
    )
    return csv


def test_read_xyce_noise_csv(tmp_path):
    _write_xyce_noise_csv(tmp_path)
    cols = read_xyce_noise_csv(str(tmp_path / 'noise.csv'))
    assert set(cols) == {'FREQ', 'ONOISE', 'INOISE'}
    np.testing.assert_allclose(cols['FREQ'], [1.0, 10.0])
    np.testing.assert_allclose(cols['ONOISE'], [4e-10, 1e-10])


def test_xyce_load_noise_sqrt_converts(tmp_path):
    _write_xyce_noise_csv(tmp_path)
    raw = RawResult(native=None, source_path=str(tmp_path / 'rawfile.raw'))
    sig = SimulatorBackend.get('xyce').load_signal(raw, 'noise_out')
    assert sig.name == 'noise_out'
    np.testing.assert_allclose(sig.y, [2e-5, 1e-5])     # sqrt(ONOISE)
    np.testing.assert_allclose(sig.x, [1.0, 10.0])


def test_xyce_load_noise_in(tmp_path):
    _write_xyce_noise_csv(tmp_path)
    raw = RawResult(native=None, source_path=str(tmp_path / 'rawfile.raw'))
    sig = SimulatorBackend.get('xyce').load_signal(raw, 'noise_in')
    np.testing.assert_allclose(sig.y, [1e-6, 5e-7])     # sqrt(INOISE)


# ---------------------------------------------------------------------------
# Xyce emission: noise -> CSV print, not format=raw
# ---------------------------------------------------------------------------

class _DummyPDK:
    library_location = '/PDK/sky130.lib.spice'
    length_unit_in_metres = 1e-6     # sky130: W/L in microns (TASK-22)


def test_xyce_noise_emits_csv_print():
    nl = XyceNetlister(_DummyPDK())
    out = nl.generate_output_block('/run', {'noise': Noise('vout', '0', 'vin', 'dec', 10, 1, 1e6)}, ['vout'])
    assert '.print noise format=csv file=/run/noise.csv ONOISE INOISE' in out
    assert 'format=raw' not in out          # noise-only deck: no rawfile print


def test_xyce_mixed_ac_and_noise_is_refused():
    nl = XyceNetlister(_DummyPDK())
    with pytest.raises(SimulationError, match='Xyce.*AC, Noise'):
        nl.generate_output_block(
            '/run',
            {'ac': AC('dec', 10, 1, 1e6), 'noise': Noise('vout', '0', 'vin', 'dec', 10, 1, 1e6)},
            ['vout'])


def test_noise_port_table_covers_aliases():
    assert _NOISE_PORT['noise_out'] == 'out' and _NOISE_PORT['inoise_spectrum'] == 'in'


# ---------------------------------------------------------------------------
# End-to-end through silk: real noise runs, canonical load, cross-simulator
# agreement. Skips where the simulator / PDK is absent.
# ---------------------------------------------------------------------------

_XYCE_BIN = os.path.expanduser('~/xyce/Xyce-7.10.0/bin/Xyce')
_SKY130_LIB = os.path.expanduser('~/pdk/sky130A/libs.tech/ngspice/sky130.lib.spice')
_PDK_YAML = os.path.join(os.path.dirname(__file__), '..', 'silk', 'pdk', 'data',
                         'skywater_130_device_details.yaml')
_have_pdk = os.path.exists(_SKY130_LIB)
_have_xyce = os.path.exists(_XYCE_BIN) and _have_pdk

trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))


def _build_cs_noise(dir_path, simulator):
    os.makedirs(dir_path, exist_ok=True)
    from silk.pdk.basepdk import BasePDK
    from silk.circuit.modules import Module
    from silk import sources
    from silk.elements import elements
    pdk = BasePDK(_PDK_YAML, library_location=_SKY130_LIB)
    mod = Module('PDK_skywater130', str(dir_path), 'noise_cs.cir', pdk_details=pdk)
    mod.simulator = simulator
    for n in ('vout', 'g', 'vdd', '0'):
        mod.add_net(n)
    mod.add(pdk.nmos1p8('M1', d='vout', g='g', s='0', b='0', w=4.0, l=0.5, m=1))
    mod.add(elements.resistor('Rd', p='vout', n='vdd', value=10e3))
    mod.add_source(sources.VoltageSource('vdd', positive='vdd', negative='0', dc=1.8))
    mod.add_source(sources.VoltageSource('vg', positive='g', negative='0', dc=0.9, ac=1.0))
    mod.add_sim_type(Noise(output='vout', ref='0', source='vg',
                           variation='dec', points=10, fstart=1, fstop=1e6))
    return mod


def _integrated(sig):
    return float(np.sqrt(trapz(np.abs(sig.y) ** 2, np.abs(sig.x))))


@pytest.mark.slow
@pytest.mark.skipif(not _have_pdk, reason='sky130 PDK not present')
def test_ngspice_noise_loads_canonical(tmp_path):
    """A real ngspice noise run loads via the canonical name noise_out."""
    mod = _build_cs_noise(tmp_path, 'ngspice')
    mod.simulate(corner='tt')
    sig = mod.load_signal('noise_out')
    assert sig.name == 'noise_out'
    assert len(sig.y) > 0 and np.all(np.asarray(sig.y) > 0)


@pytest.mark.slow
@pytest.mark.requires_xyce
@pytest.mark.skipif(not _have_xyce, reason='Xyce or sky130 PDK not present')
def test_ngspice_xyce_noise_agree(tmp_path, monkeypatch):
    """The capstone: ngspice and Xyce noise, loaded via the same canonical
    name through silk, integrate to the same output noise."""
    monkeypatch.setenv('SILK_XYCE_BIN', _XYCE_BIN)
    ng = _build_cs_noise(tmp_path / 'ng', 'ngspice')
    xy = _build_cs_noise(tmp_path / 'xy', 'xyce')
    ng.simulate(corner='tt'); xy.simulate(corner='tt')
    n_ng = _integrated(ng.load_signal('noise_out'))
    n_xy = _integrated(xy.load_signal('noise_out'))
    assert n_ng > 0 and n_xy > 0
    assert abs(n_xy - n_ng) / n_ng < 0.02     # agree to ~2% (validation showed 1e-5)
