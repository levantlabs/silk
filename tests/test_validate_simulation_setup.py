"""TASK-590: ``validate_simulation_setup`` is the run's own static admission,
callable without running. Simulator-free: every refusal here precedes a launch."""
import os
from pathlib import Path

import numpy as np
import pytest

import silk.simulation as simulation
from silk import PortDir, load_pdk, sources
from silk.circuit.modules import Module
from silk.circuit.ports import Port
from silk.elements import elements
from silk.exceptions import (
    NotSupportedError, SimulationConfigurationError, SimulationError,
)
from silk.simulation import AC, DC, Transient, validate_simulation_setup
from silk.sources.classes import (
    BehavioralVoltage, CurrentControlledCurrentSource,
    VoltageControlledCurrentSource,
)
from tests.test_topology import top


def vsrc(name='drive', positive='out', negative='0', **kw):
    return sources.VoltageSource(name, positive=positive, negative=negative,
                                 **({'dc': 1} | kw))


def refusal(call):
    try:
        call()
    except Exception as exc:                      # noqa: BLE001 - compared by type
        return exc
    return None


def run_refusal(mod):
    """The error the real run raises (behavioral: simulate; SPICE: launch)."""
    if mod.simulator == 'behavioral':
        return refusal(mod.simulate)
    mod.generate_netlist()
    return refusal(mod.run_simulator)


def assert_parity(mod, match=None):
    """Public call and run raise the same class and message."""
    expected = run_refusal(mod)
    assert expected is not None, 'the run was expected to refuse'
    got = refusal(lambda: validate_simulation_setup(mod))
    assert got is not None, 'the public call returned None for a refused run'
    assert type(got) is type(expected)
    assert str(got) == str(expected)
    if match:
        assert match in str(got)


def behavioral(tmp_path, name=None):
    mod = top(tmp_path, 'behavioral', name)
    mod.add_sim_type(Transient(1e-9, 4e-9))
    return mod


@pytest.mark.parametrize('backend', ['ngspice', 'xyce', 'behavioral'])
@pytest.mark.parametrize('spelling', ['0', 'gnd', 'GND'])
@pytest.mark.parametrize('modeled', [False, True])
def test_ground_interface_setup_run_parity_is_read_only(tmp_path, backend, spelling, modeled):
    from silk.exceptions import CircuitError
    from tests.test_ground_port_guard import _legacy_interface, _assert_ground_diagnosis
    from tests.test_topology_preflight import prior_evidence
    calls = []
    child = _legacy_interface(tmp_path / 'child', spelling)
    if modeled:
        def model(ports, t, params):
            calls.append('model')
        child.set_behavioral_model(model, execution='vector')
    else:
        child.add_source(vsrc(positive='internal', negative='0'))
    mod = top(tmp_path, backend)
    mod.add_module(child, 'leaf', nets={spelling: 'mid'})
    mod.add_source(vsrc(positive='mid', negative='0'))
    def selector():
        calls.append('selector')
        pytest.fail('static interface refusal must precede source selection')
    mod.sig_sources_dict['drive'].selected_waveform = selector
    mod.add_sim_type(Transient(1, 2))
    sentinel, raw = prior_evidence(mod)
    placed = mod.elem_dict['leaf']['device']
    before = (repr(vars(mod._element_mgr)), repr(vars(placed._element_mgr)),
              repr(getattr(mod, '_prepared_device_param_requests', None)), mod._run_files,
              mod._deck_preparation, getattr(mod, '_behavioral_element_notes', None),
              tuple(sorted(str(p) for p in tmp_path.rglob('*'))))
    with pytest.raises(CircuitError) as setup:
        validate_simulation_setup(mod)
    _assert_ground_diagnosis(str(setup.value), spelling)
    assert before == (repr(vars(mod._element_mgr)), repr(vars(placed._element_mgr)),
                      repr(getattr(mod, '_prepared_device_param_requests', None)), mod._run_files,
                      mod._deck_preparation, getattr(mod, '_behavioral_element_notes', None),
                      tuple(sorted(str(p) for p in tmp_path.rglob('*'))))
    assert mod._raw is sentinel and mod.simResultsAll is sentinel
    assert raw.read_text() == 'previous waveform evidence'
    with pytest.raises(CircuitError) as run:
        mod.simulate()
    assert str(run.value) == str(setup.value)
    assert calls == []
    assert mod._raw is sentinel and mod.simResultsAll is sentinel
    assert raw.read_text() == 'previous waveform evidence'


def test_behavioral_model_replaces_hidden_invalid_interfaces(tmp_path):
    from silk.simulation.behavioral import BehavioralEngine
    from tests.test_ground_port_guard import _legacy_interface
    hidden = _legacy_interface(tmp_path / 'hidden')
    leaf = top(tmp_path / 'leaf', 'behavioral', 'leaf')
    leaf.define_module_ports({'a': PortDir.IN, 'z': PortDir.OUT})
    leaf.add_module(hidden, 'hidden', nets={'0': 'a'})
    def model(ports, t, params):
        ports.z.v = ports.a.v
    leaf.set_behavioral_model(model, execution='vector')
    mod = behavioral(tmp_path)
    mod.add_module(leaf, 'leaf', a='input', z='output')
    mod.add_source(vsrc(positive='input'))
    assert validate_simulation_setup(mod) is None
    blocks, _ = BehavioralEngine()._flatten(mod, report=False)
    assert 'leaf' in blocks
    assert not any(isinstance(key, str) and 'hidden' in key for key in blocks)
    mod.simulate()
    np.testing.assert_allclose(mod.load_signal('output').y, 1)


@pytest.mark.parametrize('spelling', ['0', 'gnd', 'GND'])
@pytest.mark.parametrize('modeled', [False, True])
def test_direct_behavioral_flatten_checks_interfaces(tmp_path, spelling, modeled):
    from silk.simulation.behavioral import BehavioralEngine
    from silk.exceptions import CircuitError
    from tests.test_ground_port_guard import _legacy_interface, _assert_ground_diagnosis
    child = _legacy_interface(tmp_path / 'child', spelling)
    if modeled:
        def model(ports, t, params):
            pytest.fail('static preparation must not call the model')
        child.set_behavioral_model(model, execution='vector')
    else:
        child.add_source(vsrc(positive='internal'))
    middle = top(tmp_path / 'middle', 'behavioral', 'middle')
    middle.add_module(child, 'leaf', nets={spelling: '0'})
    mod = behavioral(tmp_path)
    mod.add_module(middle, 'branch')
    with pytest.raises(CircuitError) as caught:
        BehavioralEngine()._flatten(mod, report=False)
    _assert_ground_diagnosis(str(caught.value), spelling)


# -- the contract ------------------------------------------------------------

def test_exported_from_private_module_without_public_submodule():
    assert 'validate_simulation_setup' in simulation.__all__
    assert simulation.validate_simulation_setup.__module__ == 'silk.simulation._setup_check'
    with pytest.raises(ImportError):
        __import__('silk.simulation.validate_simulation_setup')


def test_docstring_opens_with_the_promised_sentence():
    assert validate_simulation_setup.__doc__.startswith(
        'Validate the assembled simulation setup for its selected simulator '
        'without running it or changing it. Returns None if no static refusal '
        'is found; this does not guarantee convergence or successful execution.')


# -- parity: SPICE ------------------------------------------------------------

@pytest.mark.parametrize('backend', ['ngspice', 'xyce'])
def test_spice_disconnected_source_parity(tmp_path, backend):
    mod = top(tmp_path, backend)
    mod.add_source(vsrc(negative='return'))
    mod.add_sim_type(Transient(1e-9, 10e-9))
    assert_parity(mod, 'no connection to ground')


@pytest.mark.parametrize('backend', ['ngspice', 'xyce'])
def test_spice_nested_and_arrayed_source_parity(tmp_path, backend):
    child = top(tmp_path, backend, 'child')
    child.define_module_ports({})
    child.add_source(vsrc('drive[3]', positive='a', negative='b'))
    mod = top(tmp_path, backend)
    mod.add_module(child, 'u[2]')
    mod.add_sim_type(Transient(1e-9, 10e-9))
    assert_parity(mod, 'xu[2].drive[3]')


def test_spice_backend_ground_names(tmp_path):
    # ngspice folds GND onto node 0; vss never grounds.
    ok = top(tmp_path / 'a', 'ngspice')
    ok.add_source(vsrc(negative='GND'))
    assert validate_simulation_setup(ok) is None
    bad = top(tmp_path / 'b', 'ngspice')
    bad.add_source(vsrc(negative='vss'))
    bad.add_sim_type(Transient(1e-9, 10e-9))
    assert_parity(bad)
    # Xyce keeps gnd as an ordinary name.
    xyce = top(tmp_path / 'c', 'xyce')
    xyce.add_source(vsrc(negative='gnd'))
    with pytest.raises(SimulationError, match='no connection to ground'):
        validate_simulation_setup(xyce)


@pytest.mark.parametrize('kind', ['resistor', 'capacitor'])
def test_positive_controls_path_through_r_and_c(tmp_path, kind):
    for backend in ('ngspice', 'xyce'):
        mod = top(tmp_path / backend, backend)
        mod.add_source(vsrc(negative='return'))
        mod.add(getattr(elements, kind)('tie', p='return', n='0', value=1))
        mod.add_sim_type(Transient(1e-9, 4e-9))
        assert validate_simulation_setup(mod) is None
    # Behavioral: the ground check passes; the engine's own reference rule
    # (TASK-582) then refuses a reference reached only through R or C.
    mod = behavioral(tmp_path / 'behavioral')
    mod.add_source(vsrc(negative='return'))
    mod.add(getattr(elements, kind)('tie', p='return', n='0', value=1))
    err = refusal(lambda: validate_simulation_setup(mod))
    assert isinstance(err, NotSupportedError)
    assert 'no connection to ground' not in str(err)


def test_opaque_content_is_exempt(tmp_path):
    mod = top(tmp_path, 'ngspice')
    mod.add_source(vsrc(negative='return'))
    mod.add_custom_line('* opaque')
    assert validate_simulation_setup(mod) is None


def test_stale_deck_precedence_in_a_run(tmp_path):
    mod = top(tmp_path, 'ngspice')
    mod.add_source(vsrc(negative='0'))
    mod.add_sim_type(Transient(1e-9, 10e-9))
    mod.generate_netlist()
    mod.add_source(vsrc('floating', positive='x', negative='y'))
    # The run's freshness gate comes first; the public call is about the
    # live setup and says nothing of the deck.
    run = refusal(mod.run_simulator)
    assert isinstance(run, SimulationConfigurationError)
    assert 'no connection to ground' in str(refusal(
        lambda: validate_simulation_setup(mod)))


def test_unrelated_external_deck_is_not_judged_by_the_run(tmp_path):
    mod = top(tmp_path, 'ngspice')
    mod.add_source(vsrc('live_floating', negative='return'))
    mod.set_output_options(mode='overwrite')
    Path(os.path.join(mod.dir_folder, mod.netlistname)).write_text('* external deck\nVdrive out 0 1\n.end\n')
    assert mod._deck_preparation is None
    mod.add_sim_type(Transient(1e-9, 10e-9))
    # The run does not reach its ground check; the public call judges the
    # live module, as documented.
    assert 'live_floating' in str(refusal(lambda: validate_simulation_setup(mod)))


# -- parity: behavioral -------------------------------------------------------

def test_behavioral_disconnected_source_parity(tmp_path):
    mod = behavioral(tmp_path)
    mod.add_source(vsrc(negative='return'))
    assert_parity(mod, 'no connection to ground')


def test_behavioral_divider_parity(tmp_path):
    mod = behavioral(tmp_path)
    mod.add_source(vsrc('v', positive='in'))
    mod.add(elements.resistor('r1', p='in', n='mid', value=1e3))
    mod.add(elements.resistor('r2', p='mid', n='0', value=1e3))
    assert_parity(mod, 'divider')


def test_behavioral_series_capacitor_parity(tmp_path):
    mod = behavioral(tmp_path)
    mod.add_source(vsrc('v', positive='in'))
    mod.add(elements.capacitor('c1', p='in', n='out', value=1e-12))
    mod.add(elements.capacitor('c2', p='out', n='0', value=1e-12))
    assert_parity(mod, 'series C')


@pytest.mark.parametrize('kind', ['gm', 'cccs', 'behavioral'])
def test_behavioral_unsupported_sources_parity(tmp_path, kind):
    mod = behavioral(tmp_path)
    mod.add_source(vsrc('v', positive='in'))
    if kind == 'gm':
        mod.add_source(VoltageControlledCurrentSource(
            'g1', out_pos='o', out_neg='0', ctrl_pos='in', ctrl_neg='0', gm=1e-3))
    elif kind == 'cccs':
        mod.add_source(CurrentControlledCurrentSource(
            'f1', out_pos='o', out_neg='0', ctrl_vname='v', gain=2))
    else:
        mod.add_source(BehavioralVoltage('b1', positive='o', negative='0', expression='v(in)'))
    assert_parity(mod, 'electrical solve')


def test_behavioral_unsupported_analysis_parity(tmp_path):
    mod = top(tmp_path, 'behavioral')
    mod.add_source(vsrc())
    mod.add_sim_type(AC('dec', 5, 1, 1e6))
    assert_parity(mod, 'AC')


def test_behavioral_no_analysis_parity(tmp_path):
    mod = top(tmp_path, 'behavioral')
    mod.add_source(vsrc())
    assert_parity(mod, 'needs a Transient or DC analysis')


def test_behavioral_transistor_and_modeled_leaf_replacement(tmp_path):
    pdk = load_pdk('sky130')
    mod = Module('sky130', str(tmp_path / 'a'), 't.cir', pdk)
    mod.simulator = 'behavioral'
    mod.add_sim_type(Transient(1e-9, 4e-9))
    mod.add_source(vsrc('v', positive='in'))
    mod.add(pdk.nmos1p8('m', d='out', g='in', s='gnd', b='gnd', w=1, l=0.15))
    assert_parity(mod, 'transistor-level')

    # A model is a leaf: the transistor inside it is replaced, so it passes.
    leaf = Module('sky130', str(tmp_path / 'leaf'), 'l.cir', pdk, module_name='leaf')
    leaf.define_module_ports({'a': PortDir.IN, 'z': PortDir.OUT})
    leaf.add(pdk.nmos1p8('m', d='z', g='a', s='gnd', b='gnd', w=1, l=0.15))
    def model_1(ports, t, params):
        ports.z.v = ports.a.v
    leaf.set_behavioral_model(model_1, execution='vector')
    ok = Module('sky130', str(tmp_path / 'b'), 't.cir', pdk)
    ok.simulator = 'behavioral'
    ok.add_sim_type(Transient(1e-9, 4e-9))
    ok.add_source(vsrc('v', positive='in'))
    ok.add_module(leaf, 'leaf', a='in', z='out')
    assert validate_simulation_setup(ok) is None


def test_behavioral_empty_unmodelled_block_parity(tmp_path):
    child = top(tmp_path, 'behavioral', 'child')
    child.define_module_ports({'a': PortDir.IN})
    mod = behavioral(tmp_path / 'm')
    mod.add_source(vsrc('v', positive='in'))
    mod.add_module(child, 'u', a='in')
    assert_parity(mod, 'nothing inside it')


def test_behavioral_ambient_temperature_parity(tmp_path):
    mod = behavioral(tmp_path)
    mod.add_source(vsrc())
    mod.simulator_options['temp'] = 85
    assert_parity(mod, 'ambient TEMP')


@pytest.mark.parametrize('source', ['temp', 'nonexistent'])
def test_behavioral_dc_sweep_refusals_parity(tmp_path, source):
    mod = top(tmp_path, 'behavioral')
    mod.add_source(vsrc())
    mod.add_sim_type(DC(source=source, start=0, stop=1, step=0.5))
    assert_parity(mod)


def test_behavioral_custom_line_and_capture_switch_parity(tmp_path):
    mod = behavioral(tmp_path / 'a')
    mod.add_source(vsrc())
    mod.add_custom_line('* text')
    assert_parity(mod, 'custom SPICE text')
    mod = behavioral(tmp_path / 'b')
    mod.add_source(vsrc())
    mod.save_device_currents = True
    assert_parity(mod, 'save_device_currents')


def test_behavioral_reference_controls(tmp_path):
    # Reached only through R: refuses (TASK-582 control).
    mod = behavioral(tmp_path / 'a')
    mod.add_source(vsrc('ref', positive='ref', negative='x'))
    mod.add(elements.resistor('r', p='x', n='0', value=1e3))
    mod.add_source(vsrc('v', positive='out', negative='ref', dc=0.5))
    assert_parity(mod)
    # A 1 V reference plus a 0.5 V source gives 1.5 V.
    ok = behavioral(tmp_path / 'b')
    ok.add_source(vsrc('ref', positive='ref', negative='0', dc=1))
    ok.add_source(vsrc('v', positive='out', negative='ref', dc=0.5))
    assert validate_simulation_setup(ok) is None
    ok.simulate()
    assert float(ok.load_signal('out').y[-1]) == pytest.approx(1.5)


# -- no side effects ----------------------------------------------------------

def test_no_effects_and_no_model_calls_clean_and_refused(tmp_path):
    # Observe actual invocation outside run-local callable copies.
    global calls
    calls = []

    def model(ports, t, params):
        calls.append(1)
        ports.z.v = ports.a.v

    def waveform(t):
        calls.append(2)
        return t * 0

    leaf = top(tmp_path / 'leaf', 'behavioral', 'leaf')
    leaf.define_module_ports({'a': PortDir.IN, 'z': PortDir.OUT})
    leaf.set_behavioral_model(model, execution='vector')
    mod = behavioral(tmp_path / 'm')
    mod.add_source(vsrc('v', positive='in', dc=0.5))
    mod.add_module(leaf, 'leaf', a='in', z='out')
    mod.add(elements.resistor('load', p='out', n='0', value=1e3))
    mod.sig_sources_dict['v'].dc = 0.25
    before = (mod.sig_sources_dict['v'].dc, mod.results_file, getattr(mod, '_raw', None),
              mod._run_consumed, list(mod.sim_analysis_dict),
              getattr(mod, '_behavioral_element_notes', None),
              sorted(p.name for p in tmp_path.rglob('*')))
    for _ in range(2):
        assert validate_simulation_setup(mod) is None
    after = (mod.sig_sources_dict['v'].dc, mod.results_file, getattr(mod, '_raw', None),
             mod._run_consumed, list(mod.sim_analysis_dict),
             getattr(mod, '_behavioral_element_notes', None),
             sorted(p.name for p in tmp_path.rglob('*')))
    assert before == after
    assert calls == []

    # A later real run calls the model and emits its notice exactly once.
    with pytest.warns(UserWarning, match='load.*ignored') as caught:
        mod.simulate()
    assert len(caught) == 1
    assert calls

    # A refused call also leaves things unchanged.
    mod.add_source(vsrc('floating', positive='x', negative='y'))
    snapshot = (mod.results_file, getattr(mod, '_raw', None), sorted(p.name for p in tmp_path.rglob('*')))
    with pytest.raises(SimulationError):
        validate_simulation_setup(mod)
    assert snapshot == (mod.results_file, getattr(mod, '_raw', None),
                        sorted(p.name for p in tmp_path.rglob('*')))


def test_static_behavioral_refusal_keeps_earlier_results(tmp_path):
    mod = behavioral(tmp_path)
    mod.add_source(vsrc())
    mod.simulate()
    results, raw = mod.results_file, getattr(mod, '_raw', None)
    assert results is not None
    mod.add_source(vsrc('floating', positive='x', negative='y'))
    with pytest.raises(SimulationError):
        mod.simulate()
    assert mod.results_file == results and getattr(mod, '_raw', None) is raw
    assert Path(results).is_file()
    mod.add_sim_type(AC('dec', 5, 1, 1e6))
    mod.sig_sources_dict.pop('floating')
    with pytest.raises(NotSupportedError):
        mod.simulate()
    assert mod.results_file == results and getattr(mod, '_raw', None) is raw
    assert Path(results).is_file()


def test_transient_length_does_not_allocate(tmp_path):
    mod = top(tmp_path, 'behavioral')
    mod.add_source(vsrc())
    mod.add_sim_type(Transient(1e-12, 1.0))     # 1e12 points if it were allocated
    assert validate_simulation_setup(mod) is None


# -- the testbench method -----------------------------------------------------

def bench(tmp_path, simulator='behavioral'):
    dut = Module(None, str(tmp_path / 'dut'), 'dut.cir', None, module_name='dut')
    dut.define_module_ports({'vin': PortDir.IN, 'vout': PortDir.OUT})
    def model_2(ports, t, params):
        ports.vout.v = ports.vin.v
    dut.set_behavioral_model(model_2, execution='vector')
    dut.simulator = simulator
    tb = dut.create_test('t')
    tb.add_module_under_test({'vin': 'vin', 'vout': 'vout'})
    tb.add_source(vsrc('input', positive='vin', negative='0'))
    tb.add_sim_type(Transient(1e-9, 4e-9))
    return dut, tb


def test_bench_method_writes_and_consumes_nothing(tmp_path):
    dut, tb = bench(tmp_path)
    files = sorted(tmp_path.rglob('*'))
    consumed = tb.netlist._run_consumed
    assert tb.validate_simulation_setup() is None
    assert sorted(tmp_path.rglob('*')) == files
    assert tb.netlist._run_consumed == consumed
    assert tb.netlist._run_files is None


def test_bench_checks_inherited_settings_as_it_would_run(tmp_path):
    dut, tb = bench(tmp_path)
    tb.netlist.add_source(vsrc('floating', positive='x', negative='y'))
    dut.simulator = 'ngspice'
    # Inherited ngspice: the floating source refuses; the bench's authored
    # simulator selection stays unset.
    with pytest.raises(SimulationError, match='no connection to ground'):
        tb.validate_simulation_setup()
    assert not tb.netlist._simulator_explicit
    assert tb.netlist.simulator_options.get('temp') is None


def test_bench_inherited_temperature_refuses_like_the_run(tmp_path):
    dut, tb = bench(tmp_path)
    dut.set_temperature(85)
    options = dict(tb.netlist.simulator_options)
    got = refusal(tb.validate_simulation_setup)
    assert tb.netlist.simulator_options == options    # nothing inherited stuck
    expected = refusal(tb.run)
    assert isinstance(got, NotSupportedError)
    assert (type(got), str(got)) == (type(expected), str(expected))


def test_unwired_bench_raises_typed_error(tmp_path):
    dut = Module(None, str(tmp_path / 'dut'), 'dut.cir', None, module_name='dut')
    dut.define_module_ports({'a': PortDir.IN})
    tb = dut.create_test('empty')
    with pytest.raises(SimulationConfigurationError, match='add_module_under_test'):
        tb.validate_simulation_setup()


# -- codex review controls (P2-1..P2-4) ---------------------------------------

from silk import Variable                                       # noqa: E402
from silk.circuit import placement_reference                    # noqa: E402
from silk.sources.classes import CurrentControlledVoltageSource  # noqa: E402


def norm(error, root):
    """Message with the (twin-specific) output location removed."""
    return str(error).replace(str(root), '<root>')


def twin_bench(tmp_path, simulator='behavioral'):
    return bench(tmp_path, simulator)


def nested(tmp_path, simulator='behavioral'):
    """A parent owning a placed DUT whose test is selected by instance."""
    dut, tb = bench(tmp_path, simulator)
    parent = Module(None, str(tmp_path / 'parent'), 'parent.cir', None)
    parent.add_module(dut, 'u', vin='in', vout='out')
    return parent, dut, tb


def evidence_state(root, tb):
    index = root.__dict__.get('_test_evidence')
    records = {} if index is None else {
        key: (id(rec), repr(sorted(rec.values)), rec.report, rec.run_id,
              tuple(rec.corners)) for key, rec in index.records.items()}
    selectors = {} if index is None else {k: id(v) for k, v in index.selectors.items()}
    nl = tb.netlist
    return (records, selectors, id(vars(nl).get('_netlister')),
            vars(nl).get('_netlister_sim'),
            {n: s.dc for n, s in nl.sig_sources_dict.items() if hasattr(s, 'dc')},
            nl._supply_voltage, nl._supply_source, dict(nl.simulator_options),
            nl._simulator, nl._simulator_explicit, nl.save_device_currents,
            tb._module_added, nl._run_consumed, nl._run_files)


@pytest.mark.parametrize('stage,refuse', [
    ('fresh', False), ('fresh', True), ('after_run', False)])
def test_selected_nested_bench_validation_is_read_only(tmp_path, stage, refuse):
    parent, dut, tb = nested(tmp_path)
    if refuse:
        tb.netlist.add_source(vsrc('floating', positive='x', negative='y'))
        dut.simulator = 'ngspice'
    selected = parent.get_test('t', instance='u')
    if stage == 'after_run':
        assert not refuse
        selected.run()
    state = evidence_state(parent, tb)
    inventory = set(placement_reference._modules)
    if refuse:
        err = refusal(selected.validate_simulation_setup)
        assert isinstance(err, SimulationError)
        assert 'no connection to ground' in str(err)
        # The retained traceback keeps the private view alive; it must not
        # have joined the live-graph inventory.
        assert not (set(placement_reference._modules) - inventory)
    else:
        assert selected.validate_simulation_setup() is None
        assert not (set(placement_reference._modules) - inventory)
    assert evidence_state(parent, tb) == state


def test_selected_nested_bench_uses_inherited_conditions(tmp_path):
    parent, dut, tb = nested(tmp_path)
    parent.set_temperature(85)         # inherited by the nested placement
    selected = parent.get_test('t', instance='u')
    err = refusal(selected.validate_simulation_setup)
    assert isinstance(err, NotSupportedError)
    assert tb.netlist.simulator_options.get('temp') is None


def test_view_is_not_registered_or_serialized(tmp_path):
    dut, tb = bench(tmp_path)
    inventory = set(placement_reference._modules)
    tb.netlist.add_source(vsrc('floating', positive='x', negative='y'))
    dut.simulator = 'ngspice'
    err = refusal(tb.validate_simulation_setup)       # retains the traceback
    assert err is not None
    assert not (set(placement_reference._modules) - inventory)


def build_capture(tmp_path, owner_flag):
    dut, tb = bench(tmp_path)
    dut.save_device_currents = owner_flag
    return dut, tb


def test_capture_switch_inherited_from_owner_before_first_run(tmp_path):
    dut, tb = build_capture(tmp_path / 'a', True)
    got = refusal(tb.validate_simulation_setup)
    _, twin = build_capture(tmp_path / 'b', True)
    expected = refusal(twin.run)
    assert isinstance(got, NotSupportedError) and 'save_device_currents' in str(got)
    assert (type(got), norm(got, tmp_path / 'a')) == (type(expected), norm(expected, tmp_path / 'b'))


def test_stale_capture_switch_on_netlist_is_not_a_false_refusal(tmp_path):
    dut, tb = build_capture(tmp_path / 'a', True)
    assert refusal(tb.run) is not None          # run copies True onto the netlist
    assert tb.netlist.save_device_currents is True
    dut.save_device_currents = False            # owner cleared it
    assert tb.validate_simulation_setup() is None
    assert tb.netlist.save_device_currents is True   # view only; netlist untouched
    assert tb.run() is not None                 # and the run agrees


def test_configured_supply_resolution_parity(tmp_path):
    dut, tb = bench(tmp_path / 'a')
    dut.set_supply_voltage(1.8, source='absent')
    got = refusal(tb.validate_simulation_setup)
    _, twin = bench(tmp_path / 'b')
    twin.module.set_supply_voltage(1.8, source='absent')
    expected = refusal(twin.run)
    assert isinstance(got, SimulationConfigurationError)
    assert (type(got), str(got)) == (type(expected), str(expected))
    assert tb.netlist._supply_source is None     # nothing written back


@pytest.mark.parametrize('owner,explicit,outcome', [
    ('behavioral', None, NotSupportedError),     # inherited behavioral: AC refuses
    ('ngspice', None, None),                     # inherited ngspice: AC is fine
    ('behavioral', 'ngspice', None),             # explicit bench backend wins
    ('ngspice', 'behavioral', NotSupportedError),
])
def test_effective_backend_is_the_one_that_distinguishes(tmp_path, owner, explicit, outcome):
    dut, tb = bench(tmp_path)
    tb.netlist.sim_analysis_dict.clear()
    tb.netlist.add_sim_type(AC('dec', 5, 1, 1e6))
    dut.simulator = owner
    if explicit:
        tb.netlist.simulator = explicit
    err = refusal(tb.validate_simulation_setup)
    assert (type(err) if err else None) is outcome


def refusal_stack(tmp_path, kinds):
    """A behavioral top with several simultaneous refusals."""
    mod = behavioral(tmp_path)
    mod.add_source(vsrc('v', positive='in'))
    if 'temp' in kinds:
        mod.simulator_options['temp'] = 85
    if 'supply' in kinds:
        mod.set_supply_voltage(1.8, source='absent')
    if 'ground' in kinds:
        mod.add_source(vsrc('floating', positive='x', negative='y'))
    if 'ac' in kinds:
        mod.sim_analysis_dict.clear()
        mod.add_sim_type(AC('dec', 5, 1, 1e6))
    if 'divider' in kinds:
        mod.add(elements.resistor('r1', p='in', n='mid', value=1e3))
        mod.add(elements.resistor('r2', p='mid', n='0', value=1e3))
    if 'switch' in kinds:
        mod.save_device_currents = True
    return mod


@pytest.mark.parametrize('kinds,expected', [
    (('temp', 'supply', 'ground', 'ac', 'divider'), (NotSupportedError, 'ambient TEMP')),
    (('supply', 'ground', 'ac', 'divider'), (SimulationConfigurationError, 'supply')),
    (('ground', 'ac', 'divider'), (SimulationError, 'no connection to ground')),
    (('switch', 'ac', 'divider'), (NotSupportedError, 'save_device_currents')),
    (('ac', 'divider'), (NotSupportedError, 'AC')),
    (('divider',), (NotSupportedError, 'divider')),
])
def test_first_refusal_order_matches_the_run(tmp_path, kinds, expected):
    got = refusal(lambda: validate_simulation_setup(refusal_stack(tmp_path / 'a', kinds)))
    ran = refusal(refusal_stack(tmp_path / 'b', kinds).simulate)
    assert (type(got), expected[1] in str(got)) == (expected[0], True)
    assert (type(ran), norm(ran, tmp_path / 'b')) == (type(got), norm(got, tmp_path / 'a'))


def test_h_source_refuses_like_the_run(tmp_path):
    mod = behavioral(tmp_path)
    mod.add_source(vsrc('v', positive='in'))
    mod.add_source(CurrentControlledVoltageSource(
        'h1', out_pos='o', out_neg='0', transresistance=1e3, ctrl_vname='v'))
    assert_parity(mod, 'electrical solve')


def test_external_deck_is_not_judged_by_the_run_launch(tmp_path, monkeypatch):
    mod = top(tmp_path, 'ngspice')
    mod.add_source(vsrc('live_floating', negative='return'))
    mod.set_output_options(mode='overwrite')
    Path(os.path.join(mod.dir_folder, mod.netlistname)).write_text('* external deck\nVdrive out 0 1\n.end\n')
    mod.add_sim_type(Transient(1e-9, 10e-9))

    class Reached(Exception):
        pass

    def launch(*args, **kwargs):
        raise Reached
    monkeypatch.setattr(type(mod.backend), 'run', launch)
    with pytest.raises(Reached):                 # the run reaches the backend...
        mod.run_simulator()
    assert 'live_floating' in str(refusal(      # ...while the public call judges
        lambda: validate_simulation_setup(mod)))  # the live module, as documented


def test_spies_variable_and_provenance_untouched(tmp_path):
    calls = []

    class Spy(sources.Pulse):
        def to_numpy(self, t):
            calls.append('to_numpy')
            return super().to_numpy(t)

        def to_spice(self):
            calls.append('to_spice')
            return super().to_spice()

    gain = Variable(name='gain', value=3)
    mod = behavioral(tmp_path)
    src = sources.VoltageSource('v', positive='in', negative='0', dc=gain,
                                pulse=Spy(v1=0, v2=1, td=0, tr=1e-9, tf=1e-9, pw=2e-9, per=4e-9))
    src.selected_waveform = lambda: calls.append('selector') or src.pulse
    mod.add_source(src)
    mod.add(elements.resistor('load', p='in', n='0', value=1e3))
    files = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    state = (gain.value, mod.operating_conditions, mod.results_file,
             getattr(mod, '_behavioral_element_notes', None), mod._run_consumed,
             mod._supply_voltage, mod._supply_source, mod._deck_preparation,
             getattr(mod, '_prepared_deck_record', None), mod._run_files)
    assert validate_simulation_setup(mod) is None
    assert calls == []
    assert (gain.value, mod.operating_conditions, mod.results_file,
            getattr(mod, '_behavioral_element_notes', None), mod._run_consumed,
            mod._supply_voltage, mod._supply_source, mod._deck_preparation,
            getattr(mod, '_prepared_deck_record', None), mod._run_files) == state
    assert {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()} == files
    # A refused call changes nothing either.
    # The custom selector makes the ground check stand aside (opaque content),
    # so the engine's own reference refusal is the first one here.
    mod.add_source(vsrc('floating', positive='x', negative='y'))
    with pytest.raises(NotSupportedError):
        validate_simulation_setup(mod)
    assert calls == []
    assert {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()} == files


# -- codex re-review controls: live graph purity, sweep-source parity ---------

from silk.exceptions import SweepSourceError  # noqa: E402


def raw_graph(module, seen=None):
    """Raw manager state of a hierarchy, read without any sync accessor."""
    seen = {} if seen is None else seen
    if id(module) in seen:
        return seen
    mgr = module._element_mgr
    seen[id(module)] = (
        {n: (e['type'], e.get('module_name'), id(e['device']),
             {p: (pin.net, str(pin.direction), str(pin.role), pin.domain)
              for p, pin in e.get('pin_list', {}).items()})
         for n, e in mgr.elem_dict.items()},
        {n: [(b.instance, b.port, b.direction, b.role, b.domain) for b in net.pins]
         for n, net in mgr.net_dict.items()},
        {n: (id(e['device']), e.get('defined'), e.get('numberInstantiated'),
             frozenset(e.get('_source_ids', ())), id(e)) for n, e in mgr.module_dict.items()},
        dict(mgr._pin_to_net), id(mgr.elem_dict), id(mgr.module_dict),
        {n: id(e) for n, e in mgr.elem_dict.items()})
    for e in mgr.elem_dict.values():
        if e['type'].name == 'MODULE':
            raw_graph(e['device'], seen)
    return seen


def drifted_parent(tmp_path, simulator='behavioral'):
    """A placed child whose declaration changed after placement (stale pins)."""
    child = top(tmp_path / 'c', simulator, 'child')
    child.define_module_ports({'a': PortDir.IN, 'z': PortDir.OUT})
    def model_1(ports, t, params):
        ports.z.v = ports.a.v
    child.set_behavioral_model(model_1, execution='vector')
    mod = top(tmp_path / 'p', simulator)
    mod.add_sim_type(Transient(1e-9, 4e-9))
    mod.add_source(vsrc('v', positive='in'))
    mod.add_module(child, 'u', a='in', z='out')
    mod.add(elements.resistor('rl', p='out', n='0', value=1e3))
    child.pin_list['a'].direction = PortDir.OUT      # changed without a getter
    return mod, child


@pytest.mark.parametrize('refuse', [False, True])
def test_validation_leaves_the_live_graph_untouched(tmp_path, refuse):
    mod, child = drifted_parent(tmp_path)
    if refuse:
        mod.add_source(vsrc('floating', positive='x', negative='y'))
    before = raw_graph(mod)
    refusal(lambda: validate_simulation_setup(mod))
    assert raw_graph(mod) == before
    # The snapshot sees the drift that a sync would have rewritten.
    assert before[id(mod)][0]['u'][3]['a'][1] != str(child.pin_list['a'].direction)


@pytest.mark.parametrize('refuse', [False, True])
def test_selected_and_bench_validation_leave_the_live_graph_untouched(tmp_path, refuse):
    parent, dut, tb = nested(tmp_path)
    child = top(tmp_path / 'c', 'behavioral', 'child')
    child.define_module_ports({'a': PortDir.IN})
    child.add(elements.resistor('r', p='a', n='0', value=1))
    tb.netlist.add_module(child, 'extra', a='vin')
    child.pin_list['a'].direction = PortDir.OUT
    if refuse:
        tb.netlist.add_source(vsrc('floating', positive='x', negative='y'))
    selected = parent.get_test('t', instance='u')
    before = (raw_graph(parent), raw_graph(tb.netlist))
    refusal(selected.validate_simulation_setup)
    refusal(tb.validate_simulation_setup)
    assert (raw_graph(parent), raw_graph(tb.netlist)) == before


@pytest.mark.parametrize('kind', ['dc_missing', 'noise_missing', 'array_bare'])
def test_bench_sweep_source_parity_with_run(tmp_path, kind):
    def build(root):
        dut, tb = bench(root)
        tb.netlist.sim_analysis_dict.clear()
        if kind == 'dc_missing':
            tb.netlist.add_sim_type(DC(source='missing', start=0, stop=1, step=0.5))
        elif kind == 'noise_missing':
            from silk.simulation import Noise
            tb.netlist.add_sim_type(Noise(output='v(vout)', ref='0', source='missing',
                                          variation='dec', points=3, fstart=1, fstop=10))
        else:
            tb.netlist.add_source(vsrc('bias[0]', positive='b0'))
            tb.netlist.add_source(vsrc('bias[1]', positive='b1'))
            tb.netlist.add_sim_type(DC(source='bias', start=0, stop=1, step=0.5))
        return dut, tb

    _, tb = build(tmp_path / 'a')
    got = refusal(tb.validate_simulation_setup)
    _, twin = build(tmp_path / 'b')
    expected = refusal(twin.run)
    assert isinstance(got, SweepSourceError)
    assert (type(got), str(got)) == (type(expected), str(expected))
    # ...and it precedes a competing temperature refusal, as in the run.
    dut, tb2 = build(tmp_path / 'c')
    dut.set_temperature(85)
    assert isinstance(refusal(tb2.validate_simulation_setup), SweepSourceError)
    dut3, twin3 = build(tmp_path / 'd')
    dut3.set_temperature(85)
    assert isinstance(refusal(twin3.run), SweepSourceError)
