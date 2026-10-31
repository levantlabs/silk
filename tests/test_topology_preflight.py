"""Connectivity admission precedes launches and preserves previous evidence."""
import os
from pathlib import Path

import numpy as np
import pytest

from silk import sources
from silk.elements import elements
from silk.circuit.modules_sim import _track_result_replacement, _track_attempt_stage
from silk.exceptions import NotSupportedError, SimulationError
from silk.simulation import Transient
from tests.test_topology import top


def prior_evidence(mod):
    sentinel = object()
    raw = Path(mod.dir_folder) / 'rawfile.raw'
    raw.write_text('previous waveform evidence')
    mod._raw = sentinel
    mod.simResultsAll = sentinel
    mod.sim_results_loaded = True
    mod.results_file = str(raw)
    return sentinel, raw


@pytest.mark.parametrize('backend', ['ngspice', 'xyce', 'behavioral'])
@pytest.mark.parametrize('mode', ['overwrite', 'new'])
def test_floating_refusal_precedes_replacement_cleanup_and_backend(tmp_path, monkeypatch, backend, mode):
    mod = top(tmp_path, backend)
    mod.set_output_options(mode=mode)
    mod.add_source(sources.VoltageSource('drive', positive='out', negative='return', dc=1))
    mod.add_sim_type(Transient(1e-9, 10e-9))
    if backend != 'behavioral':
        mod.generate_netlist()
    sentinel, raw = prior_evidence(mod)
    def forbidden(*args, **kwargs):
        pytest.fail('floating source must refuse before backend launch')
    monkeypatch.setattr(type(mod.backend), 'run_module' if backend == 'behavioral' else 'run', forbidden)
    with _track_result_replacement(mod) as replaced, _track_attempt_stage() as stage:
        with pytest.raises(SimulationError, match="drive.*no connection to ground") as error:
            if backend == 'behavioral':
                mod.simulate()
            else:
                mod.run_simulator()
    assert not replaced[0]
    assert stage[0] == 'preflight'
    assert mod._raw is sentinel and mod.simResultsAll is sentinel
    assert mod.sim_results_loaded
    assert mod.results_file == str(raw)
    assert raw.read_text() == 'previous waveform evidence'
    assert not mod._run_consumed
    assert error.value.topology.disconnected[0].source == 'drive'


@pytest.mark.parametrize('backend', ['ngspice', 'xyce', 'behavioral'])
def test_grounded_request_reaches_backend(tmp_path, monkeypatch, backend):
    mod = top(tmp_path, backend)
    mod.add_source(sources.VoltageSource('drive', positive='out', negative='0', dc=1))
    mod.add_sim_type(Transient(1e-9, 10e-9))
    class ReachedBackend(Exception):
        pass
    def launch(*args, **kwargs):
        raise ReachedBackend
    monkeypatch.setattr(type(mod.backend), 'run_module' if backend == 'behavioral' else 'run', launch)
    with _track_result_replacement(mod) as replaced:
        with pytest.raises(ReachedBackend):
            mod.simulate()
    assert replaced[0]


@pytest.mark.parametrize('backend', ['ngspice', 'xyce'])
def test_unrepresented_external_deck_is_not_judged_from_live_inventory(tmp_path, monkeypatch, backend):
    mod = top(tmp_path, backend)
    mod.add_source(sources.VoltageSource('live_floating', positive='out', negative='return', dc=1))
    mod.set_output_options(mode='overwrite')
    Path(os.path.join(mod.dir_folder, mod.netlistname)).write_text('* external deck\nVdrive out 0 1\n.end\n')
    assert mod._deck_preparation is None
    class ReachedBackend(Exception):
        pass
    monkeypatch.setattr(type(mod.backend), 'run', lambda *args, **kwargs: (_ for _ in ()).throw(ReachedBackend()))
    with pytest.raises(ReachedBackend):
        mod.run_simulator()


def test_behavioral_unloaded_current_stimulus_with_ground_runs(tmp_path):
    mod = top(tmp_path, 'behavioral')
    mod.add_source(sources.CurrentSource('drive', positive='input', negative='0', dc=2))
    mod.add_sim_type(Transient(1, 2))
    mod.simulate()
    np.testing.assert_array_equal(mod.load_signal('idrive', 'I').y, [2, 2, 2])
    assert 'v(input)' not in mod.get_signal_names()
    with pytest.raises(SimulationError, match="Signal 'input'.*not found"):
        mod.load_signal('input')


@pytest.mark.parametrize('kind', ['resistor', 'capacitor'])
def test_ground_path_admission_does_not_invent_a_behavioral_reference(tmp_path, kind):
    from silk.simulation.topology import inspect_topology

    mod = top(tmp_path, 'behavioral')
    mod.add_source(sources.VoltageSource('offset', positive='out', negative='return', dc=.5))
    mod.add(getattr(elements, kind)('return', p='return', n='0', value=1))
    mod.add_sim_type(Transient(1, 2))
    result = inspect_topology(mod)
    assert result.disconnected == ()
    assert result.unchecked == ()
    with pytest.raises(NotSupportedError, match='reference') as error:
        mod.simulate()
    assert 'offset' in str(error.value)
    assert 'return' in str(error.value)


def test_known_nonzero_reference_is_preserved_after_admission(tmp_path):
    from silk.simulation.topology import inspect_topology

    mod = top(tmp_path, 'behavioral')
    # Deliberately place the dependent source first; reference resolution
    # must use the producer's voltage, never manufacture zero after admission.
    mod.add_source(sources.VoltageSource('offset', positive='out', negative='reference', dc=.5))
    mod.add_source(sources.VoltageSource('bias', positive='reference', negative='0', dc=1))
    mod.add_sim_type(Transient(1, 2))
    assert inspect_topology(mod).disconnected == ()
    mod.simulate()
    np.testing.assert_allclose(mod.load_signal('out').y, [1.5, 1.5, 1.5])


def test_behavioral_shortcut_obeys_the_same_preflight(tmp_path):
    mod = top(tmp_path)
    mod.add_source(sources.VoltageSource('drive', positive='out', negative='return', dc=1))
    sentinel, raw = prior_evidence(mod)
    with pytest.raises(SimulationError, match='no connection to ground'):
        mod.simulate_behavioral(1, 2)
    assert mod._raw is sentinel
    assert raw.read_text() == 'previous waveform evidence'


def test_registered_bench_uses_the_same_connectivity_admission(tmp_path):
    mod = top(tmp_path, 'behavioral', 'dut')
    mod.define_module_ports({'p': 'in', 'n': 'in'})
    def model_1(ports, t, params):
        return
    mod.set_behavioral_model(model_1, execution='vector')
    tb = mod.create_test('floating')
    tb.add_module_under_test({'p': 'out', 'n': 'return'})
    tb.add_source(sources.VoltageSource('drive', positive='out', negative='return', dc=1))
    tb.add_sim_type(Transient(1, 2))
    with pytest.raises(SimulationError, match='no connection to ground'):
        tb.run()
