"""Module.__init__ sources its netlister from the selected backend, and a
non-ngspice backend can supply its own netlister without editing modules.py."""
from silk.simulation.backend import SimulatorBackend, SpiceBackend, NgspiceBackend
import silk.circuit.modules as m

# The ``pdk`` fixture from conftest, not the one an example builds: these tests
# never run a simulator, so they need a PDK *object*, not an installed PDK. The
# example import used to make this file — and so the whole suite — fail at
# COLLECTION on any machine without one.


class _SpyNetlister:
    def __init__(self, pdk_details):
        self.pdk_details = pdk_details


@SimulatorBackend.register('spynet')
class _SpyNetBackend(SpiceBackend):
    def _launch(self, netlist_path, results_dir, *, debug=False, timeout=None):
        raise NotImplementedError
    def make_netlister(self, pdk_details):
        return _SpyNetlister(pdk_details)


def test_init_sources_netlister_from_backend(monkeypatch, pdk):
    """Proof: if __init__ went through the backend, monkeypatching the ngspice
    backend's make_netlister changes the Module's netlister. (A hardcoded
    PySpiceNetlister in __init__ would make this fail.)"""
    sentinel = object()
    monkeypatch.setattr(NgspiceBackend, 'make_netlister',
                        lambda self, pdk_details: sentinel)
    mod = m.Module('PDK_skywater130', '/tmp', 'x.cir', pdk_details=pdk)
    assert mod.netlister is sentinel


def test_module_default_netlister_is_pyspice(pdk):
    """Regression guard: the ngspice default still yields a PySpiceNetlister."""
    from silk.circuit.spice_netlister import PySpiceNetlister
    mod = m.Module('PDK_skywater130', '/tmp', 'x.cir', pdk_details=pdk)
    assert isinstance(mod.netlister, PySpiceNetlister)


def test_changing_simulator_regenerates_netlister(pdk):
    """C2c: setting simulator post-construction rebuilds the netlister from the
    new backend automatically — no manual make_netlister call. Dissolves the
    eager-netlister / lazy-runner asymmetry."""
    from silk.circuit.spice_netlister import PySpiceNetlister
    mod = m.Module('PDK_skywater130', '/tmp', 'x.cir', pdk_details=pdk)
    assert isinstance(mod.netlister, PySpiceNetlister)   # ngspice default
    mod.simulator = 'spynet'
    assert isinstance(mod.netlister, _SpyNetlister)      # auto-regenerated


def test_unknown_simulator_rejected_at_assignment(pdk):
    """C2c: the validating setter fails fast on an unregistered simulator."""
    import pytest
    mod = m.Module('PDK_skywater130', '/tmp', 'x.cir', pdk_details=pdk)
    with pytest.raises(ValueError):
        mod.simulator = 'no_such_simulator_xyz'
