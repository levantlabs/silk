"""TASK-448.1: the backend seam and the 'behavioral' backend registration.

``mod.simulator = 'behavioral'`` selects silk's numpy engine through the same
setting that selects ngspice or Xyce. ``Module.simulate()`` dispatches on the
backend's ``runs_module`` capability to ``SimulatorBackend.run_module``, a
template method that applies the data-presence gate exactly as ``run`` does.
"""
import os

import numpy as np
import pytest

from silk.circuit.circuit_elements import PortDir
from silk.circuit.modules import Module
from silk.exceptions import NotSupportedError, SimulationAborted
from silk.simulation import Transient, AC
from silk.simulation.backend import RawResult, SimulatorBackend, SpiceBackend
from silk.sources.classes import VoltageSource


def _pdk():
    from silk.pdk.basepdk import BasePDK
    pdk_yaml = os.path.join(os.path.dirname(__file__), '..', 'silk', 'pdk',
                            'data', 'skywater_130_device_details.yaml')
    return BasePDK(pdk_yaml, library_location='/fake/lib')


def _gain_block(tmp_path, gain=2.0):
    blk = Module('PDK_skywater130', str(tmp_path / 'blk'), 'blk.cir',
                 pdk_details=_pdk())
    blk.module_name = 'gain'
    blk.define_module_ports({'vin': PortDir.IN, 'vout': PortDir.OUT})
    def model_1(ports, t, params):
        ports.vout.v = gain * ports.vin.v
    blk.set_behavioral_model(model_1, execution='vector')
    return blk


def _flat_top(tmp_path, *, analysis=None):
    """A flat behavioural top: one source driving one modelled block."""
    top = Module('PDK_skywater130', str(tmp_path / 'top'), 'top.cir',
                 pdk_details=_pdk())
    top.add_source(VoltageSource('vs', positive='vin', negative='gnd', dc=0.4))
    top.add_module(_gain_block(tmp_path), 'g1', nets={'vin': 'vin', 'vout': 'vout'})
    top.add_sim_type(analysis or Transient(step=1e-9, stop=10e-9))
    top.simulator = 'behavioral'
    return top


# AC #1 ----------------------------------------------------------------------

def test_behavioral_is_a_registered_backend(tmp_path):
    from silk.simulation.behavioral_backend import BehavioralBackend
    top = _flat_top(tmp_path)
    assert top.simulator == 'behavioral'
    assert isinstance(top.backend, BehavioralBackend)
    assert top.backend.runs_module is True
    assert SimulatorBackend.get('ngspice').runs_module is False


def test_simulate_on_flat_behavioral_top_produces_a_gated_rawresult(tmp_path, monkeypatch):
    import subprocess
    # numpy 1.26 imports numpy.testing lazily on first access, and that import
    # runs lscpu through subprocess.run; do it before the guard goes in.
    import numpy.testing  # noqa: F401

    def boom(*a, **k):
        raise AssertionError('a simulator process was launched')
    monkeypatch.setattr(subprocess, 'run', boom)
    monkeypatch.setattr(subprocess, 'Popen', boom)

    top = _flat_top(tmp_path)
    assert top.simulate() is top
    raw = top._raw
    assert isinstance(raw, RawResult)
    assert top.backend.has_data(raw.native)
    sig = top.load_signal('vout')
    np.testing.assert_allclose(sig.y, 0.8)
    assert len(sig.x) > 0
    # No deck was written for a behavioural run.
    assert not os.path.exists(os.path.join(top.dir_folder, top.netlistname))
    assert top.operating_conditions.corner_kind == 'none'


def test_empty_behavioral_result_raises_simulation_aborted(tmp_path, monkeypatch):
    from silk.simulation.behavioral import BehavioralEngine
    monkeypatch.setattr(BehavioralEngine, 'run',
                        lambda self, module, t_step, t_stop: ({}, np.array([])))
    top = _flat_top(tmp_path)
    with pytest.raises(SimulationAborted) as excinfo:
        top.simulate()
    assert excinfo.value.simulator == 'behavioral'
    # The aborted run published nothing.
    assert top.sim_results_loaded is False
    assert top._raw is None
    assert top.operating_conditions is None


def test_run_module_on_a_deck_backend_refuses(tmp_path):
    top = _flat_top(tmp_path)
    with pytest.raises(NotSupportedError, match="'ngspice' backend runs SPICE decks"):
        SimulatorBackend.get('ngspice').run_module(top)


def test_behavioral_run_refuses_a_non_transient_analysis(tmp_path):
    top = _flat_top(tmp_path, analysis=AC(variation='dec', points=10,
                                          fstart=1, fstop=1e6))
    with pytest.raises(NotSupportedError, match='declares: AC'):
        top.simulate()


# AC #2 ----------------------------------------------------------------------

def test_backend_overriding_run_module_is_refused_at_class_definition():
    with pytest.raises(TypeError, match='defines run_module()'):
        class _Rogue(SpiceBackend):  # noqa: F841
            def run_module(self, module, *, run_files=None):
                return None

            def _launch(self, *a, **k):
                return None

            def make_netlister(self, pdk_details):
                return None


def test_backend_overriding_run_is_still_refused():
    with pytest.raises(TypeError, match=r'defines run\(\)'):
        class _Rogue(SpiceBackend):  # noqa: F841
            def run(self, *a, **k):
                return None


# AC #3 ----------------------------------------------------------------------

def test_generate_netlist_on_behavioral_top_raises_naming_backend(tmp_path):
    top = _flat_top(tmp_path)
    with pytest.raises(NotSupportedError, match=r"generate_netlist\(\).*'behavioral' backend"):
        top.generate_netlist()
    assert not os.path.exists(os.path.join(top.dir_folder, top.netlistname))


def test_run_simulator_on_behavioral_top_raises_naming_backend(tmp_path):
    top = _flat_top(tmp_path)
    with pytest.raises(NotSupportedError, match=r"run_simulator\(\).*'behavioral' backend"):
        top.run_simulator()


def test_behavioral_selected_block_emits_its_subckt_but_a_spice_bench_refuses_it(tmp_path):
    blk = _gain_block(tmp_path)
    blk.simulator = 'behavioral'
    assert '.subckt gain' in blk.generate_netlist()
    bench = Module('PDK_skywater130', str(tmp_path / 'bench'), 'bench.cir',
                   pdk_details=_pdk())
    bench.add_source(VoltageSource('vs', positive='vin', negative='gnd', dc=0.4))
    bench.add_module(blk, 'g1', nets={'vin': 'vin', 'vout': 'vout'})
    bench.add_sim_type(Transient(step=1e-9, stop=10e-9))
    # The block is model-only, so an ngspice bench refuses it (TASK-448.8)
    # rather than emitting an empty .subckt.
    with pytest.raises(NotSupportedError, match=r"g1.*no SPICE content"):
        bench.generate_netlist()


# AC #4 ----------------------------------------------------------------------

def test_set_simulator_options_on_behavioral_raises_unknown_option(tmp_path):
    top = _flat_top(tmp_path)
    with pytest.raises(NotSupportedError, match=r"Unknown option\(s\) 'reltol'.*'behavioral'"):
        top.set_simulator_options(reltol=1e-6)
    assert 'reltol' not in top.simulator_options


def test_behavioral_portable_presets_are_empty_no_ops(tmp_path):
    top = _flat_top(tmp_path)
    for preset in ('default', 'tight', 'fast'):
        top.set_simulator_options(preset=preset)
    assert top.simulator_options == {}
    with pytest.raises(NotSupportedError, match="Unknown option"):
        top.set_simulator_options(preset='tight', reltol=1e-6)


def test_behavioral_config_is_a_shipped_yaml():
    from silk.simulators import load_simulator_config
    cfg = load_simulator_config('behavioral')
    assert cfg.name == 'behavioral'
    assert cfg.accepts_options is False
    assert cfg.presets == {'default': {}, 'tight': {}, 'fast': {}}
