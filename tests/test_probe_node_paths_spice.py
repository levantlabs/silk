"""Real simulator agreement between designer and emitted node paths."""
import numpy as np
import pytest

from silk import Module, sources
from silk.elements import elements
from silk.simulation import Transient
from tests._env import HAVE_XYCE, needs_ngspice


@pytest.mark.parametrize('simulator', [
    pytest.param('ngspice', marks=needs_ngspice),
    pytest.param('xyce', marks=[pytest.mark.slow, pytest.mark.requires_xyce]),
])
def test_saved_designer_internal_and_port_read_same_vector(tmp_path, simulator):
    if simulator == 'xyce' and not HAVE_XYCE:
        pytest.skip('Xyce setting does not resolve to an executable')
    child = Module(None, str(tmp_path), 'child.cir', None, module_name='divider')
    child.define_module_ports({'out': 'output'})
    child.add(elements.resistor('upper', p='out', n='mid', value=1000))
    child.add(elements.resistor('lower', p='mid', n='0', value=1000))
    dut = Module(None, str(tmp_path), 'dut.cir', None, module_name='design')
    dut.define_module_ports({'out': 'output'})
    dut.add_module(child, 'x1[0]', {'out': 'out'})
    tb = dut.create_test('read')
    tb.add_module_under_test({'out': 'rail'})
    tb.add_source(sources.VoltageSource('rail', positive='rail', negative='0', dc=1.0))
    tb.add_sim_type(Transient(step=1e-9, stop=5e-9))
    tb.netlist.simulator = simulator
    internal = tb.probe('dut.x1[0].mid')
    port = tb.probe('dut.x1[0].out')
    tb.netlist.set_save_list(['dut.x1[0].mid', 'dut.out'])
    tb.run()
    np.testing.assert_allclose(internal.y, .5)
    np.testing.assert_allclose(port.y, 1.)
    np.testing.assert_array_equal(internal.y, tb.probe('xdut.xx1[0].mid').y)
    np.testing.assert_array_equal(port.y, tb.probe('rail').y)


@needs_ngspice
@pytest.mark.slow
@pytest.mark.requires_xyce
def test_declared_nodes_without_save_list_agree_between_simulators(tmp_path):
    if not HAVE_XYCE:
        pytest.skip('Xyce setting does not resolve to an executable')
    results = {}
    for simulator in ('ngspice', 'xyce'):
        child = Module(None, str(tmp_path / simulator), 'child.cir', None,
                       module_name='divider')
        child.define_module_ports({'hi': 'input', 'lo': 'output'})
        child.add(elements.resistor('upper', p='hi', n='g[1]', value=1000))
        child.add(elements.resistor('lower', p='g[1]', n='lo', value=1000))
        dut = Module(None, str(tmp_path / simulator), 'dut.cir', None,
                     module_name='design')
        dut.define_module_ports({'supply': 'input'})
        dut.add(elements.resistor('feed', p='supply', n='in[0]', value=1000))
        dut.add_module(child, 'X1[0]', {'hi': 'in[0]', 'lo': 'out[3]'})
        dut.add(elements.resistor('load', p='out[3]', n='0', value=2000))
        tb = dut.create_test('read')
        tb.add_module_under_test({'supply': 'vin'})
        tb.add_source(sources.VoltageSource('input', positive='vin', negative='0', dc=1.25))
        tb.add_sim_type(Transient(step=1e-9, stop=5e-9))
        tb.netlist.simulator = simulator
        names = ('vin', 'dut.in[0]', 'dut.X1[0].g[1]', 'dut.out[3]',
                 'dut.X1[0].hi', 'xdut.xx1[0].g[1]')
        probes = [tb.probe(name) for name in names]
        tb.run()
        results[simulator] = [probe.y[-1] for probe in probes]
        np.testing.assert_allclose(results[simulator], [1.25, 1., .75, .5, 1., .75])
    np.testing.assert_allclose(results['xyce'], results['ngspice'])


@needs_ngspice
@pytest.mark.slow
@pytest.mark.requires_xyce
def test_declarative_and_generated_nodes_agree_between_simulators(tmp_path):
    from tests.test_probe_node_paths import generated_node_metric_bench
    if not HAVE_XYCE:
        pytest.skip('Xyce setting does not resolve to an executable')
    results = {}
    for simulator in ('ngspice', 'xyce'):
        tb = generated_node_metric_bench(tmp_path / simulator, ['xdut.mid', 'xdut.y[2]'])
        tb.netlist.simulator = simulator
        tb.add_metric('designer', type='average', signal='dut.y[2]')
        tb.add_metric('delay', type='propagation_delay',
                      signals={'sig_in': 'dut.mid', 'sig_out': 'dut.y[2]'},
                      in_threshold=.9, out_threshold=.9)
        report = tb.run()
        results[simulator] = [report.metrics[name] for name in
                              ('level_0', 'level_1', 'designer', 'delay')]
        # Both voltage slots must load successfully. Constant levels have no
        # edge delay, so the metric returns NaN rather than a loading error.
        np.testing.assert_allclose(results[simulator], [.4, .2, .2, np.nan])
    np.testing.assert_allclose(results['xyce'], results['ngspice'])


@needs_ngspice
@pytest.mark.slow
@pytest.mark.requires_xyce
@pytest.mark.parametrize('path', ['0', 'xdut.unknown'])
def test_generated_invalid_declarative_nodes_never_produce_values(tmp_path, path):
    from tests.test_probe_node_paths import generated_node_metric_bench
    from silk.exceptions import NotSupportedError
    if not HAVE_XYCE:
        pytest.skip('Xyce setting does not resolve to an executable')
    for simulator in ('ngspice', 'xyce'):
        tb = generated_node_metric_bench(tmp_path / simulator, [path])
        tb.netlist.simulator = simulator
        if simulator == 'xyce':
            with pytest.raises(NotSupportedError, match='level_0'):
                tb.run()
        else:
            report = tb.run()
            assert report.metrics['level_0'].startswith('Error:')


@needs_ngspice
@pytest.mark.slow
def test_ngspice_declarative_readback_contract(tmp_path):
    """Keep emitted paths stable and resolve designer aliases to the same values."""
    import json
    import silk
    from silk.codegen.testbench_generator import generate_testbenches

    paths = ['rail', 'xdut.mid', 'xdut.y[2]', '0', 'xdut.unknown',
             'dut.mid', 'dut.y[2]', 'dut.unknown', 'v(xdut.mid)', 'v(dut.mid)']

    def build(pdk):
        dut = Module(None, str(tmp_path), 'design.cir', None, module_name='design')
        dut.define_module_ports({'out': 'output'})
        dut.add(elements.resistor('upper', p='out', n='mid', value=1000))
        dut.add(elements.resistor('lower', p='mid', n='0', value=1000))
        dut.add(elements.resistor('bus_upper', p='mid', n='y[2]', value=1000))
        dut.add(elements.resistor('bus_lower', p='y[2]', n='0', value=1000))
        return dut

    spec = {
        'circuit_name': 'design', 'circuit_builder': '_build_design',
        'tests': [{'name': 'read', 'dut_portmap': {'out': 'rail'},
                   'sources': [{'type': 'VoltageSource', 'instance': 'rail',
                                'positive': 'rail', 'negative': '0', 'dc': 1.0}],
                   'analyses': [{'kind': 'transient', 'step': 1e-9, 'stop': 5e-9}],
                   'metrics': [{'name': f'level_{i}', 'type': 'average', 'signal': path}
                               for i, path in enumerate(paths)]}]}
    ns = {'_build_design': build}
    exec(generate_testbenches(spec), ns)
    tb = ns['build_design_with_tests'](None)._tests['read']
    tb.netlist.simulator = 'ngspice'
    report = tb.run()
    actual = {path: report.metrics[f'level_{i}'] for i, path in enumerate(paths)}
    print('Imported silk:', silk.__file__)
    print('Ngspice declarative outcomes:', json.dumps(actual, sort_keys=True))
    expected_values = {'rail': 1.0, 'xdut.mid': .4, 'xdut.y[2]': .2, 'v(xdut.mid)': .4}
    designer_aliases = {'dut.mid': 'xdut.mid', 'dut.y[2]': 'xdut.y[2]',
                        'v(dut.mid)': 'v(xdut.mid)'}
    for designer, emitted in designer_aliases.items():
        assert actual[designer] == actual[emitted]
    expected = {path: expected_values.get(
        path, f"Error: Signal {path!r} (type='V') not found in rawfile.") for path in paths if path not in designer_aliases}
    differences = {}
    for path, wanted in expected.items():
        if actual[path] != (pytest.approx(wanted) if isinstance(wanted, float) else wanted):
            differences[path] = {'main': wanted, 'candidate': actual[path]}
    assert not differences, json.dumps(differences, indent=2, sort_keys=True)
