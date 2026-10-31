from types import SimpleNamespace

import pytest
from silk.simulation import Transient, AC, DC, Noise
from silk.simulation.backend import SimulatorBackend, EmitContext, SpiceBackend
from tests._env import needs_ngspice


def test_get_ngspice_returns_spice_backend():
    b = SimulatorBackend.get('ngspice')
    assert isinstance(b, SpiceBackend)


def test_get_unknown_raises():
    with pytest.raises(ValueError):
        SimulatorBackend.get('does-not-exist')


def test_render_transient():
    b = SimulatorBackend.get('ngspice')
    assert b.render_analysis(Transient(1e-9, 2e-6), EmitContext()) == '.tran 1n 2u\n'
    assert b.render_analysis(Transient(1e-9, 2e-6, maxstep=20e-12, uic=True),
                             EmitContext()) == '.tran 1n 2u 0 20p uic\n'


def test_render_ac():
    b = SimulatorBackend.get('ngspice')
    assert b.render_analysis(AC('dec', 20, 1, 1e9), EmitContext()) == '.ac dec 20 1 1G\n'


def test_render_dc():
    b = SimulatorBackend.get('ngspice')
    assert b.render_analysis(DC('vsweep', 0, 1.8, 0.01), EmitContext()) == '.dc vsweep 0 1.8 0.01\n'


def test_render_noise_resolves_source():
    b = SimulatorBackend.get('ngspice')
    ctx = EmitContext(resolve_source=lambda s: 'V' + s)
    assert b.render_analysis(Noise('vout', '0', 'vinp', 'dec', 10, 1, 1e9), ctx) \
        == '.noise v(vout,0) Vvinp dec 10 1 1G 1\n'


def test_render_noise_default_identity():
    b = SimulatorBackend.get('ngspice')
    assert b.render_analysis(Noise('vout', '0', 'vinp', 'dec', 10, 1, 1e9), EmitContext()) \
        == '.noise v(vout,0) vinp dec 10 1 1G 1\n'


@needs_ngspice


def test_ngspice_run_returns_rawresult(mock_ngspice, tmp_path):
    """ngspice run() runs + loads internally (NgspiceRunner is a private
    helper) and returns a RawResult with the rawfile path."""
    from silk.simulation.backend import RawResult
    b = SimulatorBackend.get('ngspice')
    netlist = str(tmp_path / 'x.cir')
    with open(netlist, 'w') as f:
        f.write('* x\n.end\n')
    raw = b.run(netlist_path=netlist, results_dir=str(tmp_path), debug=False)
    assert isinstance(raw, RawResult)
    assert raw.source_path == str(tmp_path) + '/rawfile.raw'
    assert mock_ngspice['called']


from silk.simulation.backend import RawResult


@SimulatorBackend.register('faketest')
class _FakeBackend(SpiceBackend):
    last = None
    # A synthetic native carrying one data row, so the data-presence gate that
    # SimulatorBackend.run applies to every backend (TASK-72) passes it.
    native = SimpleNamespace(npoints=1, vectors=[])
    def _launch(self, netlist_path, results_dir, *, debug=False, timeout=None):
        type(self).last = dict(netlist_path=netlist_path, results_dir=results_dir)
        return RawResult(native=type(self).native,
                         source_path=results_dir + '/rawfile.raw')
    def make_netlister(self, pdk_details):
        raise NotImplementedError


def test_backend_run_returns_rawresult():
    raw = SimulatorBackend.get('faketest').run(
        netlist_path='/tmp/y.cir', results_dir='/tmp')
    assert isinstance(raw, RawResult)
    assert raw.native is _FakeBackend.native
    assert _FakeBackend.last['netlist_path'] == '/tmp/y.cir'


def test_ngspice_make_netlister_returns_pyspice_netlister():
    from silk.circuit.spice_netlister import PySpiceNetlister
    b = SimulatorBackend.get('ngspice')
    nl = b.make_netlister(pdk_details=None)
    assert isinstance(nl, PySpiceNetlister)
    assert nl.pdk_details is None


@SimulatorBackend.register('fakenet')
class _FakeNetBackend(SpiceBackend):
    def _launch(self, netlist_path, results_dir, *, debug=False, timeout=None):
        raise NotImplementedError
    def make_netlister(self, pdk_details):
        return ('FAKE_NETLISTER', pdk_details)


def test_backend_selects_netlister_by_name():
    nl = SimulatorBackend.get('fakenet').make_netlister(pdk_details='PDK')
    assert nl == ('FAKE_NETLISTER', 'PDK')
