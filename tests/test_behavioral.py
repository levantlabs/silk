"""Tests for behavioral simulation — PortDir, define_module_ports, set_behavioral_model,
waveform to_numpy, BehavioralEngine, and simulate_behavioral."""

import pytest
from tests._env import scratch_dir


# ---------------------------------------------------------------------------
# Task 1 — PortDir enum
# ---------------------------------------------------------------------------

def test_portdir_values():
    from silk.circuit.circuit_elements import PortDir
    assert PortDir.IN == 'input'
    assert PortDir.OUT == 'output'
    assert PortDir.INOUT == 'inout'
    assert PortDir.POWER == 'power'
    assert PortDir.GROUND == 'ground'

def test_portdir_importable_from_silk():
    from silk import PortDir
    assert PortDir.IN == 'input'

def test_portdir_is_str_enum():
    from silk.circuit.circuit_elements import PortDir
    from enum import Enum
    assert issubclass(PortDir, str)
    assert issubclass(PortDir, Enum)


# ---------------------------------------------------------------------------
# Task 2 — define_module_ports() and set_behavioral_model()
# ---------------------------------------------------------------------------

def _make_module():
    """Minimal Module for testing — no real PDK needed."""
    import os
    from silk.pdk.basepdk import BasePDK
    from silk.circuit.modules import Module

    pdk_yaml = os.path.join(
        os.path.dirname(__file__), '..', 'silk', 'pdk', 'data',
        'skywater_130_device_details.yaml'
    )
    pdk = BasePDK(pdk_yaml, library_location='/fake/lib')
    tmp = scratch_dir()
    return Module('PDK_skywater130', tmp, 'test.cir', pdk_details=pdk)


def test_define_module_ports_sets_directions():
    from silk.circuit.circuit_elements import PortDir
    m = _make_module()
    m.define_module_ports({
        'vin':  PortDir.IN,
        'vout': PortDir.OUT,
        'vdd':  PortDir.POWER,
        'gnd':  PortDir.GROUND,
    })
    assert m.pin_list['vin'].direction == PortDir.IN
    assert m.pin_list['vout'].direction == PortDir.OUT
    assert m.pin_list['vdd'].direction == PortDir.POWER
    assert m.pin_list['gnd'].direction == PortDir.GROUND

def test_define_module_ports_none_defaults_to_inout():
    from silk.circuit.circuit_elements import PortDir
    m = _make_module()
    m.define_module_ports({'vp': PortDir.IN, 'vout': None})
    assert m.pin_list['vout'].direction == PortDir.INOUT

def test_define_module_nets_backward_compat():
    """define_module_nets still works and defaults to INOUT."""
    from silk.circuit.circuit_elements import PortDir
    m = _make_module()
    m.define_module_nets({'vp': None, 'vout': None})
    assert m.pin_list['vp'].direction == PortDir.INOUT
    assert m.pin_list['vout'].direction == PortDir.INOUT

def test_set_behavioral_model_stores_callable():
    m = _make_module()
    def model_1(ports, t, params):
        ports.vout.v = t * 0
    fn = model_1
    m.set_behavioral_model(fn, execution='vector')
    assert m._behavioral_model is fn

def test_set_behavioral_model_rejects_non_callable():
    m = _make_module()
    with pytest.raises(TypeError, match="callable"):
        m.set_behavioral_model('not_a_function', execution='vector')

def test_behavioral_attributes_initialized():
    m = _make_module()
    assert m._behavioral_model is None
    assert m._behavioral_mode is False
    assert all(n.voltage is None for n in m.net_dict.values())
    assert m._behavioral_t is None


# ---------------------------------------------------------------------------
# Task 3 — to_numpy(t) on waveform classes
# ---------------------------------------------------------------------------

import numpy as np

def test_pwl_to_numpy_interpolates():
    from silk.sources.waveforms import PWL
    pwl = PWL().add(0, 0.0).add(10e-9, 1.8).add(20e-9, 0.0)
    t = np.array([0.0, 5e-9, 10e-9, 15e-9, 20e-9])
    y = pwl.to_numpy(t)
    assert y[0] == pytest.approx(0.0)
    assert y[2] == pytest.approx(1.8)
    assert y[4] == pytest.approx(0.0)
    assert y[1] == pytest.approx(0.9)   # midpoint of rise

def test_pulse_to_numpy_high_during_pw():
    from silk.sources.waveforms import Pulse
    # tr=tf=1ns, pw=5ns, per=10ns → rise [0,1), high [1,6), fall [6,7), low [7,10)
    pulse = Pulse(v1=0, v2=1.8, td=0, tr=1e-9, tf=1e-9, pw=5e-9, per=10e-9)
    t = np.array([0e-9, 0.5e-9, 2.5e-9, 6.5e-9, 8e-9, 10e-9])
    y = pulse.to_numpy(t)
    assert y[0] == pytest.approx(0.0)   # t=0 is the *start* of the rising edge
    assert y[1] == pytest.approx(0.9)   # midpoint of rise
    assert y[2] == pytest.approx(1.8)   # inside pw
    assert y[3] == pytest.approx(0.9)   # midpoint of fall
    assert y[4] == pytest.approx(0.0)   # after fall, before next period
    assert y[5] == pytest.approx(0.0)   # second period restarts on the rise

def test_pulse_to_numpy_before_delay():
    from silk.sources.waveforms import Pulse
    pulse = Pulse(v1=0, v2=1.8, td=5e-9, tr=1e-9, tf=1e-9, pw=3e-9, per=10e-9)
    t = np.array([0e-9, 2e-9])
    y = pulse.to_numpy(t)
    assert y[0] == pytest.approx(0.0)
    assert y[1] == pytest.approx(0.0)

def test_sine_to_numpy_offset_and_peak():
    from silk.sources.waveforms import Sine
    sine = Sine(voffset=0.9, vampl=0.9, freq=1e6, td=0, theta=0, phase=0)
    t = np.array([0.0, 0.25e-6, 0.5e-6, 1.0e-6])
    y = sine.to_numpy(t)
    assert y[0] == pytest.approx(0.9)           # sin(0) = 0 → offset only
    assert y[1] == pytest.approx(0.9 + 0.9)     # sin(π/2) = 1 → peak
    assert y[2] == pytest.approx(0.9, abs=1e-9) # sin(π) ≈ 0 → offset
    assert y[3] == pytest.approx(0.9, abs=1e-9) # full cycle

def test_voltage_source_dc_to_numpy():
    from silk.sources.classes import VoltageSource
    vs = VoltageSource('vdd', positive='vdd', negative='gnd', dc=1.8)
    t = np.array([0.0, 1e-9, 2e-9])
    y = vs.to_numpy(t)
    assert np.all(y == pytest.approx(1.8))

def test_voltage_source_pulse_to_numpy():
    from silk.sources.waveforms import Pulse
    from silk.sources.classes import VoltageSource
    pulse = Pulse(v1=0, v2=1.8, td=0, tr=1e-9, tf=1e-9, pw=5e-9, per=10e-9)
    vs = VoltageSource('vin', positive='vin', negative='gnd', pulse=pulse)
    t = np.array([2.5e-9, 7.5e-9])
    y = vs.to_numpy(t)
    assert y[0] == pytest.approx(1.8)
    assert y[1] == pytest.approx(0.0)

def test_voltage_source_sine_to_numpy():
    from silk.sources.waveforms import Sine
    from silk.sources.classes import VoltageSource
    sine = Sine(voffset=0.9, vampl=0.9, freq=1e6, td=0, theta=0, phase=0)
    vs = VoltageSource('vin', positive='vin', negative='gnd', sine=sine)
    t = np.array([0.25e-6])
    y = vs.to_numpy(t)
    assert y[0] == pytest.approx(1.8)  # peak

def test_voltage_source_pwl_to_numpy():
    from silk.sources.waveforms import PWL
    from silk.sources.classes import VoltageSource
    pwl = PWL().add(0, 0.0).add(10e-9, 1.8)
    vs = VoltageSource('vin', positive='vin', negative='gnd', pwl=pwl)
    t = np.array([0.0, 10e-9])
    y = vs.to_numpy(t)
    assert y[0] == pytest.approx(0.0)
    assert y[1] == pytest.approx(1.8)

def test_current_source_dc_to_numpy():
    from silk.sources.classes import CurrentSource
    cs = CurrentSource('ibias', positive='vdd', negative='vnbias', dc=100e-6)
    t = np.array([0.0, 1e-9, 2e-9])
    y = cs.to_numpy(t)
    assert np.all(y == pytest.approx(100e-6))


# ---------------------------------------------------------------------------
# Task 4 — BehavioralEngine
# ---------------------------------------------------------------------------

def _make_comparator():
    """Return a configured comparator Module with behavioral model."""
    from silk.circuit.circuit_elements import PortDir
    comp = _make_module()
    comp.module_name = 'comparator'
    comp.define_module_ports({
        'vp':   PortDir.IN,
        'vm':   PortDir.IN,
        'vout': PortDir.OUT,
    })
    def model_2(ports, t, params):
        ports.vout.v = np.where(ports.vp.v > ports.vm.v, 1.8, 0.0)
    comp.set_behavioral_model(model_2, execution='vector')
    return comp

def _make_top_with_comparator():
    """Return top-level Module with two DC sources and one comparator subcircuit."""
    from silk.sources.classes import VoltageSource
    comp = _make_comparator()
    top = _make_module()
    top.add_source(VoltageSource('vp_src', positive='vp', negative='gnd', dc=1.0))
    top.add_source(VoltageSource('vm_src', positive='vm', negative='gnd', dc=0.5))
    top.add_module(comp, 'comp1', nets={'vp': 'vp', 'vm': 'vm', 'vout': 'vout'})
    return top

def test_behavioral_engine_returns_net_signals():
    from silk.simulation.behavioral import BehavioralEngine
    top = _make_top_with_comparator()
    engine = BehavioralEngine()
    results, t = engine.run(top, t_step=1e-9, t_stop=10e-9)
    assert 'vout' in results
    assert len(results['vout']) == len(t)
    # vp=1.0 > vm=0.5 → vout should be 1.8 everywhere
    assert np.all(results['vout'] == pytest.approx(1.8))

def test_behavioral_engine_sources_in_results():
    from silk.simulation.behavioral import BehavioralEngine
    top = _make_top_with_comparator()
    engine = BehavioralEngine()
    results, t = engine.run(top, t_step=1e-9, t_stop=10e-9)
    assert 'vp' in results
    assert 'vm' in results
    assert np.all(results['vp'] == pytest.approx(1.0))
    assert np.all(results['vm'] == pytest.approx(0.5))

def test_behavioral_engine_missing_model_raises():
    from silk.simulation.behavioral import BehavioralEngine
    from silk.exceptions import SimulationError
    from silk.circuit.circuit_elements import PortDir
    from silk.sources.classes import VoltageSource

    comp = _make_module()
    comp.module_name = 'no_model_comp'
    comp.define_module_ports({'vin': PortDir.IN, 'vout': PortDir.OUT})
    # No set_behavioral_model()

    top = _make_module()
    top.add_source(VoltageSource('vs', positive='vin', negative='gnd', dc=1.0))
    top.add_module(comp, 'comp1', nets={'vin': 'vin', 'vout': 'vout'})

    engine = BehavioralEngine()
    with pytest.raises(SimulationError, match="no behavioral model"):
        engine.run(top, t_step=1e-9, t_stop=10e-9)

def test_behavioral_engine_inout_warns():
    import warnings
    from silk.simulation.behavioral import BehavioralEngine
    from silk.circuit.circuit_elements import PortDir
    from silk.sources.classes import VoltageSource

    comp = _make_module()
    comp.module_name = 'inout_comp'
    comp.define_module_ports({'vin': PortDir.IN, 'vbias': PortDir.INOUT, 'vout': PortDir.OUT})
    def model_3(ports, t, params):
        ports.vout.v = ports.vin.v * 2
    comp.set_behavioral_model(model_3, execution='vector')

    top = _make_module()
    top.add_source(VoltageSource('vs', positive='vin', negative='gnd', dc=0.9))
    top.add_module(comp, 'comp1', nets={'vin': 'vin', 'vbias': 'vbias', 'vout': 'vout'})

    engine = BehavioralEngine()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        results, t = engine.run(top, t_step=1e-9, t_stop=5e-9)
    inout_warns = [x for x in w if 'INOUT' in str(x.message)]
    assert len(inout_warns) >= 1
    np.testing.assert_allclose(results['vout'], 1.8)

    # The former fixture also drove the INOUT slot with an ideal source.
    # That ambiguous ownership now refuses, even when this callable happens
    # not to return the optional INOUT output during one evaluation.
    from silk.exceptions import NotSupportedError
    top.add_source(VoltageSource('vb', positive='vbias', negative='gnd', dc=0.5))
    with pytest.raises(NotSupportedError, match='vb.*vbias.*ownership.*SPICE'):
        engine.run(top, t_step=1e-9, t_stop=5e-9)


# ---------------------------------------------------------------------------
# Task 5 — simulate_behavioral() + results integration
# ---------------------------------------------------------------------------

def test_simulate_behavioral_end_to_end():
    """Full pipeline: simulate_behavioral → get_signal_names → load_signal."""
    from silk.simulation.behavioral import BehavioralEngine  # noqa: F401 (confirm importable)
    top = _make_top_with_comparator()
    top.simulate_behavioral(t_step=1e-9, t_stop=10e-9)

    names = top.get_signal_names()
    assert 'v(vout)' in names

    sig = top.load_signal('vout')
    assert len(sig.y) == len(sig.x)
    assert np.all(sig.y == pytest.approx(1.8))

def test_simulate_behavioral_returns_self():
    """simulate_behavioral() returns self for method chaining."""
    top = _make_top_with_comparator()
    result = top.simulate_behavioral(t_step=1e-9, t_stop=5e-9)
    assert result is top

def test_simulate_behavioral_signal_x_is_time():
    """load_signal() returns signal with x = time vector."""
    top = _make_top_with_comparator()
    top.simulate_behavioral(t_step=1e-9, t_stop=10e-9)
    sig = top.load_signal('vout')
    assert sig.x[0] == pytest.approx(0.0)
    assert sig.x[-1] == pytest.approx(10e-9)  # the grid includes t_stop (TASK-448.4, D2)

def test_simulate_behavioral_load_unknown_signal_raises():
    """load_signal() with unknown name raises SimulationError in behavioral mode."""
    from silk.exceptions import SimulationError
    top = _make_top_with_comparator()
    top.simulate_behavioral(t_step=1e-9, t_stop=5e-9)
    with pytest.raises(SimulationError):
        top.load_signal('does_not_exist')

def test_simulate_behavioral_stateful_model():
    """Class-based model with state is called correctly."""
    from silk.circuit.circuit_elements import PortDir
    from silk.sources.classes import VoltageSource

    class Integrator:
        def __init__(self):
            self.call_count = 0
        def __call__(self, ports, t, params):
            self.call_count += 1
            dt = t[1] - t[0] if len(t) > 1 else 1e-09
            ports.vout.v = np.cumsum(ports.vin.v) * dt

    model = Integrator()
    integ = _make_module()
    integ.module_name = 'integrator'
    integ.define_module_ports({'vin': PortDir.IN, 'vout': PortDir.OUT})
    integ.set_behavioral_model(model, execution='vector')

    top = _make_module()
    top.add_source(VoltageSource('vs', positive='vin', negative='gnd', dc=1.0))
    top.add_module(integ, 'int1', nets={'vin': 'vin', 'vout': 'vout'})

    top.simulate_behavioral(t_step=1e-9, t_stop=10e-9)
    # Callable state is copied for each placement at run preparation.
    assert model.call_count == 0
    sig = top.load_signal('vout')
    assert sig.y[-1] > sig.y[0]  # integrated DC → increasing ramp


def test_simulate_behavioral_writes_net_voltage(tmp_path):
    """After simulate_behavioral, each declared net has its waveform
    available via Net.voltage (new Phase 2a path)."""
    import numpy as np
    from silk.circuit.modules import Module
    from silk.pdk.basepdk import BasePDK
    from silk import PortDir, PortRole
    import os

    pdk_yaml = os.path.join(
        os.path.dirname(__file__), '..', 'silk', 'pdk', 'data',
        'skywater_130_device_details.yaml',
    )
    pdk = BasePDK(pdk_yaml, library_location='/fake/lib')
    m = Module('PDK_skywater130', str(tmp_path), 'm.cir',
               pdk_details=pdk, module_name='m')
    m.define_module_ports({'vout': (PortDir.OUT, PortRole.ANALOG)})

    def model(ports, t, params):
        ports.vout.v = np.full_like(t, 1.23)

    m.set_behavioral_model(model, execution='vector')

    # Wire into a top so the engine evaluates m.
    top = Module('PDK_skywater130', str(tmp_path), 'top.cir',
                 pdk_details=pdk, module_name='top')
    top.add_module(m, 'm1', nets={'vout': 'vout'})
    top.simulate_behavioral(t_step=1e-9, t_stop=10e-9)

    # Phase 2a: Net.voltage is populated on the top module's nets.
    vout_net = top.net_dict.get('vout')
    assert vout_net is not None
    assert vout_net.voltage is not None
    assert np.all(vout_net.voltage == 1.23)


# ---------------------------------------------------------------------------
# TASK-408 — behavioural runs carry an operating-conditions snapshot
# ---------------------------------------------------------------------------

def test_behavioral_snapshot_corner_temperature_none_supply_unknown():
    """A nonzero DC source is not provably no-supply: supply stays unknown."""
    top = _make_top_with_comparator()
    assert top.operating_conditions is None
    top.simulate_behavioral(t_step=1e-9, t_stop=5e-9)
    c = top.operating_conditions
    assert (c.origin, c.backend) == ('silk', 'behavioral')
    assert (c.corner_kind, c.corner, c.corner_basis) == ('none', None, 'proven-behavioral')
    assert (c.temperature_kind, c.temperature_c, c.temperature_basis) == (
        'none', None, 'proven-behavioral')
    assert (c.supply_kind, c.supply_voltage_v, c.supply_basis) == ('unknown', None, 'unknown')


def test_behavioral_snapshot_supply_none_when_proven():
    """The snapshot's supply follows has_no_supply (a time-varying source only)."""
    from silk.circuit.circuit_elements import PortDir
    from silk.simulation import has_no_supply
    from silk.sources import VoltageSource
    from silk.sources.waveforms import Sine

    sub = _make_module()
    sub.module_name = 'follower'
    sub.define_module_ports({'vin': PortDir.IN, 'vout': PortDir.OUT})
    def model_4(ports, t, params):
        ports.vout.v = ports.vin.v
    sub.set_behavioral_model(model_4, execution='vector')
    top = _make_module()
    top.add_source(VoltageSource('vs', positive='vin', negative='gnd',
                                 sine=Sine(0.0, 1.0, 1e6)))
    top.add_module(sub, 'f1', nets={'vin': 'vin', 'vout': 'vout'})
    assert has_no_supply(top)
    top.simulate_behavioral(t_step=1e-9, t_stop=5e-9)
    c = top.operating_conditions
    assert (c.supply_kind, c.supply_voltage_v, c.supply_basis) == (
        'none', None, 'proven-no-supply')
    assert c.corner_kind == 'none' and c.temperature_kind == 'none'


def test_behavioral_run_leaves_caller_built_measurements_unchanged():
    from silk.specs import Measurement
    m = Measurement(source='behavioral', value=1.5, corner='tt', temperature=27.0, voltage=1.8)
    before = dict(vars(m))
    top = _make_top_with_comparator()
    top.simulate_behavioral(t_step=1e-9, t_stop=5e-9)
    assert vars(m) == before
    assert (m.corner, m.temperature, m.voltage) == ('tt', 27.0, 1.8)
