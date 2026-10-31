"""Module behavioral-model parameterization (Task 10) and behavioral engine
pass-through (Task 11)."""
import pytest
import numpy as np

from silk.circuit.modules import Module
from silk.pdk.basepdk import BasePDK
import os
from tests._env import scratch_dir


_PDK_YAML = os.path.join(
    os.path.dirname(__file__), '..', '..', 'silk', 'pdk', 'data',
    'skywater_130_device_details.yaml'
)


def _pdk():
    return BasePDK(_PDK_YAML, library_location='/fake/lib')


def _module(tmp_path, name='m'):
    return Module('PDK_skywater130', str(tmp_path), f'{name}.cir',
                  pdk_details=_pdk(), module_name=name)


def test_module_behavioral_params_defaults_to_none(tmp_path):
    m = _module(tmp_path)
    assert m._behavioral_params is None


def test_set_behavioral_model_without_params_unchanged(tmp_path):
    """Parameters may be omitted when attaching a ports model."""
    m = _module(tmp_path)
    def fn(ports, t, params):
        return
    m.set_behavioral_model(fn, execution='vector')
    assert m._behavioral_model is fn
    assert m._behavioral_params is None


def test_set_behavioral_model_with_params(tmp_path):
    m = _module(tmp_path)
    def fn(ports, t, params):
        return
    m.set_behavioral_model(fn, params={'NF': 2.0, 'gain': 15.0}, execution='vector')
    assert m._behavioral_model is fn
    assert m._behavioral_params == {'NF': 2.0, 'gain': 15.0}


def test_set_behavioral_model_params_dict_is_live(tmp_path):
    """The dict is stored by reference so calibration writes show up
    when the model is next invoked."""
    m = _module(tmp_path)
    params = {'NF': 2.0}
    def model_1(ports, t, p):
        return
    m.set_behavioral_model(model_1, params=params, execution='vector')
    params['NF'] = 2.3     # caller mutates
    assert m._behavioral_params['NF'] == 2.3


def test_set_behavioral_model_rejects_non_callable(tmp_path):
    m = _module(tmp_path)
    with pytest.raises(TypeError):
        m.set_behavioral_model('not a function', execution='vector')


def _wire_block_into_top(block, block_name='blk'):
    """Return a top-level Module with ``block`` wired in as a sub-element.

    The BehavioralEngine evaluates sub-blocks in ``elem_dict``; a module
    whose *own* model is set is never invoked unless it is added as an
    element inside a parent module. This helper performs that wiring so
    the tests can exercise the engine's call path.
    """
    from silk.circuit.modules import Module
    top = Module('PDK_skywater130', scratch_dir(), 'top.cir',
                 pdk_details=_pdk(), module_name='top')
    top.add_module(block, block_name, nets={'vout': 'vout'})
    return top


def test_behavioral_engine_passes_params_when_present(tmp_path):
    """A parameterized model receives params as its third argument."""
    # Observe parameters outside run-local callable copies.
    global seen_params
    m = _module(tmp_path)

    seen_params = {}

    def parameterized_model(ports, t, params):
        seen_params['copy'] = {key: params[key] for key in params}
        ports.vout.v = np.zeros_like(t)

    # Declare a port so the engine runs this module.
    from silk import PortDir
    m.define_module_ports({'vout': PortDir.OUT})
    m.set_behavioral_model(parameterized_model, params={'NF': 2.0, 'gain': 15.0}, execution='vector')

    top = _wire_block_into_top(m)
    top.simulate_behavioral(t_step=1e-9, t_stop=10e-9)
    assert seen_params['copy'] == {'NF': 2.0, 'gain': 15.0}


def test_behavioral_engine_passes_empty_params_view_when_params_is_none(tmp_path):
    """A ports model always receives its third argument."""
    # Observe parameters outside run-local callable copies.
    global called
    m = _module(tmp_path)
    called = {}

    def model(ports, t, params):
        called['yes'] = True
        assert len(params) == 0
        ports.vout.v = np.zeros_like(t)

    from silk import PortDir
    m.define_module_ports({'vout': PortDir.OUT})
    m.set_behavioral_model(model, execution='vector')  # no params

    top = _wire_block_into_top(m)
    top.simulate_behavioral(t_step=1e-9, t_stop=10e-9)
    assert called['yes'] is True


def test_task458_placed_params_remain_live(tmp_path):
    """TASK-458: public placement handles expose the live definition."""
    # Observe parameters outside run-local callable copies.
    global last
    m = _module(tmp_path)
    last = {}

    def model(ports, t, params):
        last['NF'] = params['NF']
        ports.vout.v = np.zeros_like(t)

    from silk import PortDir
    m.define_module_ports({'vout': PortDir.OUT})
    m.set_behavioral_model(model, params={'NF': 2.0}, execution='vector')

    top = _wire_block_into_top(m)
    top.simulate_behavioral(t_step=1e-9, t_stop=10e-9)
    assert last['NF'] == 2.0

    placed = next(top.iter_instances()).module
    assert placed is m
    placed.calibrate_behavioral('NF', 2.3)
    top.simulate_behavioral(t_step=1e-9, t_stop=10e-9)
    assert last['NF'] == 2.3


def test_task458_calibrate_after_placement_reaches_run(tmp_path):
    """TASK-458: calibration reaches an already placed definition."""
    # Observe parameters outside run-local callable copies.
    global last
    from silk import PortDir
    m = _module(tmp_path)
    last = {}

    def model(ports, t, params):
        last['NF'] = params['NF']
        ports.vout.v = np.zeros_like(t)

    m.define_module_ports({'vout': PortDir.OUT})
    m.set_behavioral_model(model, params={'NF': 2.0}, execution='vector')
    top = _wire_block_into_top(m)
    top.simulate_behavioral(t_step=1e-9, t_stop=10e-9)
    assert last['NF'] == 2.0
    m.calibrate_behavioral('NF', 2.3)
    top.simulate_behavioral(t_step=1e-9, t_stop=10e-9)
    assert last['NF'] == 2.3
