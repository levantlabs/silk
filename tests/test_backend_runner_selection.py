"""A Module with a non-ngspice simulator runs through that backend's run()
verb, proving a new backend plugs in without editing modules.py."""
from types import SimpleNamespace

from silk.simulation.backend import SimulatorBackend, SpiceBackend, RawResult
from silk.examples.ota_5t_guide import build_5t_ota
from tests._env import needs_pdk

# Every test here builds against the real installed PDK.
pytestmark = needs_pdk


@SimulatorBackend.register('spytest')
class _SpyBackend(SpiceBackend):
    last = None
    # A synthetic native carrying one data row: SimulatorBackend.run applies
    # the data-presence gate to every backend's result (TASK-72), so a spy
    # backend has to look like it produced something.
    native = SimpleNamespace(npoints=1, vectors=[])
    def _launch(self, netlist_path, results_dir, *, debug=False, timeout=None):
        type(self).last = dict(netlist_path=netlist_path, results_dir=results_dir)
        return RawResult(native=type(self).native,
                         source_path=results_dir + '/rawfile.raw')
    def make_netlister(self, pdk_details):
        raise NotImplementedError


def test_module_backend_property_resolves_simulator():
    mod = build_5t_ota()
    assert type(mod.backend).__name__ == 'NgspiceBackend'


def test_run_simulator_uses_backend_run():
    mod = build_5t_ota()
    mod.simulator = 'spytest'
    mod.netlistname = 'spy.cir'
    mod.dir_folder = '/tmp'
    mod.set_output_options(mode='overwrite')
    mod.run_simulator()
    assert _SpyBackend.last == {'netlist_path': '/tmp/spy.cir', 'results_dir': '/tmp'}
    # Module consumed the backend's RawResult — no vended runner object.
    assert mod.simResultsAll is _SpyBackend.native
    assert mod.sim_results_loaded is True
    assert mod.results_file == '/tmp/rawfile.raw'
