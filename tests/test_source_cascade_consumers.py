"""Logical sources address physical cascade cards consistently."""
import numpy as np
import pytest
from silk import sources
from silk.simulation import DC, Noise, Transient
from silk.circuit.spice_netlister import PySpiceNetlister
from silk.circuit.module_test import _current_probe_names
from tests.test_raw_current_paths import _bench, _both
from tests.test_behavioral_dc import _bench as behavioral_bench


def test_logical_current_sum_and_raw_operands(tmp_path):
    src = sources.CurrentSource('bias', 'a', '0', dc=.2, pwl=[(0, .5), (1, 1)])
    tb = _bench(tmp_path, ['@ibias__silk_pwl[current]', '@ibias__silk_dc[current]'], [src], data=[[.5, 1], [.2, .2]])
    for y in _both(tb, 'i(bias)'):
        np.testing.assert_allclose(y, [.7, 1.2])
    np.testing.assert_allclose(tb.probe('i(ibias__silk_pwl)').y, [.5, 1])
    assert set(_current_probe_names(tb._probe_registry)) == {'Ibias__silk_pwl', 'Ibias__silk_dc', 'ibias__silk_pwl'}


def test_logical_voltage_current_uses_series_branch(tmp_path):
    src = sources.VoltageSource('bias', 'a', '0', dc=.2, pwl=[(0, .5), (1, 1)])
    tb = _bench(tmp_path, ['vbias__silk_pwl'], [src], data=[[-.7, -1.2]])
    for y in _both(tb, 'i(bias)'):
        np.testing.assert_allclose(y, [-.7, -1.2])


@pytest.mark.parametrize('cls,prefix', [(sources.VoltageSource, 'V'), (sources.CurrentSource, 'I')])
def test_dc_and_noise_route_to_offset_card(cls, prefix):
    src = cls('stim', 'out', '0', sine=sources.Sine(.5, .3, 1e3))
    netlister = PySpiceNetlister(None)
    for spelling in ('stim', prefix + 'stim'):
        text = netlister.generate_analysis_commands({'dc': DC(spelling, 0, 1, .5)},
                                                    sig_sources_dict={'stim': src})
        assert f'.dc {prefix}stim__silk_dc 0 1 0.5' in text
    text = netlister.generate_analysis_commands(
        {'noise': Noise('out', '0', 'stim', 'dec', 10, 1, 1e3)}, sig_sources_dict={'stim': src})
    assert prefix + 'stim__silk_dc' in text


def test_controlled_current_uses_voltage_series_branch(tmp_path, pdk):
    from silk.circuit.modules import Module
    mod = Module('sky130', str(tmp_path), 'top.cir', pdk_details=pdk)
    mod.add_source(sources.VoltageSource('sense', 'a', '0', pwl=[(0, .5), (1, 1)]))
    ctrl = sources.CCCS('copy', 'b', '0', gain=2, ctrl_vname='sense')
    mod.add_source(ctrl)
    assert ' Vsense__silk_pwl ' in ctrl.to_spice()


def test_behavioral_dc_adds_initial_waveform_and_preserves_axis(tmp_path):
    src = sources.VoltageSource('drive', 'vin', '0', dc=9, sine=sources.Sine(.5, .3, 1e3))
    tb = behavioral_bench(tmp_path, source=src, analysis=DC('drive', 0, 1, .5))
    tb.add_source(sources.VoltageSource('bias', 'bias', '0', dc=.2, pwl=[(0, .5), (1, 1)]))
    tb.netlist.simulate()
    sig = tb.netlist.load_signal('vin')
    np.testing.assert_allclose(sig.x, [0, .5, 1])
    np.testing.assert_allclose(sig.y, [.5, 1, 1.5])
    np.testing.assert_allclose(tb.netlist.load_signal('bias').y, [.7, .7, .7])
    assert src.dc == 9


def test_ngspice_current_operand_saves_are_additive_to_selected_nodes(tmp_path):
    text = PySpiceNetlister(None).generate_output_block(
        str(tmp_path), {'tran': Transient(step=1e-9, stop=1e-6)}, save_list=['out'],
        current_probe_names=['Ibias__silk_pwl', 'Ibias__silk_dc'],
        source_instance_names=['Ibias__silk_pwl', 'Ibias__silk_dc'])
    assert '@ibias__silk_pwl[current]' in text.lower()
    assert '@ibias__silk_dc[current]' in text.lower()
    assert 'v(out)' in text


def test_nested_source_names_are_resolved_before_signal_readback(tmp_path, pdk):
    from silk.circuit.modules import Module
    from silk.circuit.module_test import _branch_vector_name
    child = Module('sky130', str(tmp_path), 'child.cir', pdk_details=pdk, module_name='child')
    child.define_module_ports({})
    child.add_source(sources.CurrentSource('stim[0]', 'out', '0', dc=.2, pwl=[0, .5, 1, 1]))
    top = Module('sky130', str(tmp_path), 'top.cir', pdk_details=pdk)
    top.add_module(child, 'leaf[0]', nets={})
    assert _branch_vector_name(top.sig_sources_dict, 'xleaf[0].stim[0]', top.elem_dict) == (
        ('xleaf<0>.Istim<0>__silk_pwl', 1), ('xleaf<0>.Istim<0>__silk_dc', 1))


@pytest.mark.parametrize('saved', ['@ibias__silk_pwl[current]', '@ibias__silk_dc[current]'])
def test_logical_current_does_not_replace_a_missing_operand_with_authored_values(tmp_path, saved):
    from silk.exceptions import SimulationError
    src = sources.CurrentSource('bias', 'a', '0', dc=.2, pwl=[0, .5, 1, 1])
    tb = _bench(tmp_path, [saved], [src], data=[[.2, .2]])
    with pytest.raises(SimulationError):
        tb.probe('i(bias)').y
    from silk.circuit.module_test import _load_metric_path
    with pytest.raises(SimulationError):
        _load_metric_path(tb.netlist, 'i(bias)')


@pytest.mark.parametrize('target', [
    sources.VCVS('sense', 'a', '0', 'ctrl', '0', gain=1),
    sources.CCVS('sense', 'a', '0', ctrl_vname='seed', transresistance=1),
    sources.BehavioralVoltage('sense', 'a', '0', expression='0'),
])
def test_existing_controlled_voltage_branch_targets_remain_valid(tmp_path, pdk, target):
    from silk.circuit.modules import Module
    mod = Module('sky130', str(tmp_path), 'top.cir', pdk_details=pdk)
    mod.add_source(target)
    ctrl = sources.CCCS('copy', 'b', '0', gain=2, ctrl_vname='sense')
    mod.add_source(ctrl)
    assert f' {target.instance_name} ' in ctrl.to_spice()


@pytest.mark.parametrize('spelling,card,expected', [
    ('@ibias__silk_dc[current]', 'Ibias__silk_dc', '@Ibias__silk_dc[current]'),
    ('@i.xleaf.Ibias__silk_dc[current]', 'xleaf.Ibias__silk_dc', '@i.xleaf.Ibias__silk_dc[current]'),
    ('i.xleaf.Ibias__silk_dc', 'xleaf.Ibias__silk_dc', '@i.xleaf.Ibias__silk_dc[current]'),
    ('ibias[0]__silk_dc', 'Ibias<0>__silk_dc', '@Ibias<0>__silk_dc[current]'),
])
def test_ngspice_raw_current_saves_use_canonical_source_card(tmp_path, spelling, card, expected):
    text = PySpiceNetlister(None).generate_output_block(
        str(tmp_path), {'tran': Transient(1e-9, 1e-6)},
        current_probe_names=[spelling], source_instance_names=[card])
    assert f'save "{expected}"\n' in text
    assert '@i.i.' not in text


@pytest.mark.parametrize('selection,refuses', [('tran', False), ('dc', False), ('ac', True), (None, True)])
def test_mixed_analysis_current_probe_preserves_selection(tmp_path, selection, refuses):
    from silk.simulation import AC
    from silk.exceptions import NotSupportedError
    tb = _bench(tmp_path, [], [sources.CurrentSource('stim', 'a', '0', dc=.2, pwl=[0, .5, 1, 1])])
    tb.add_sim_type(AC('lin', 3, 1e3, 2e3))
    tb.add_sim_type(DC('stim', 0, 1, .5))
    tb.probe('i(stim)', analysis=selection)
    tb._prepare_run()
    if refuses:
        with pytest.raises(NotSupportedError, match='ac=.*0 V VoltageSource'):
            tb.netlist.generate_netlist()
    else:
        tb.netlist.generate_netlist()


@pytest.mark.parametrize('derived', [False, True])
@pytest.mark.parametrize('shared_ac', [False, True])
def test_current_acquisition_uses_metric_inference_per_use(tmp_path, derived, shared_ac):
    from silk.simulation import AC
    from silk.metrics import rise_time, dc_gain_dB
    from silk.exceptions import NotSupportedError
    tb = _bench(tmp_path, [], [sources.CurrentSource('stim', 'a', '0', dc=.2, pwl=[0, .5, 1, 1])])
    tb.add_sim_type(AC('lin', 3, 1e3, 2e3))
    current = tb.probe('i(stim)')
    value = current * 2 if derived else current
    tb.add_metric('rise', rise_time, sig=value)
    if shared_ac:
        tb.add_metric('gain', dc_gain_dB, sig=value)
    tb._prepare_run()
    assert current._analysis is None
    selections = tb.netlist._pending_current_probe_analyses
    assert 'tran' in selections
    if shared_ac:
        assert 'ac' in selections
        with pytest.raises(NotSupportedError, match='ac=.*0 V VoltageSource'):
            tb.netlist.generate_netlist()
    else:
        assert set(selections) == {'tran'}
        tb.netlist.generate_netlist()


@pytest.mark.parametrize('name', ['stim', 'stim[0]'])
def test_generated_private_node_probe_uses_physical_inventory(tmp_path, name):
    from silk.spice_names import to_spice_name
    source = sources.VoltageSource(name, 'a', '0', dc=.2, pwl=[0, .5, 1, 1])
    tb = _bench(tmp_path, [], [source])
    node = source.physical_layout().internal_node
    for spelling in (node, to_spice_name(node, kind='node').lower()):
        assert tb.probe(spelling)._selector == 'V'
    with pytest.raises(ValueError, match='has no net'):
        tb.probe('__silk_src_missing')


@pytest.mark.parametrize('explicit_operands', [False, True])
def test_logical_current_default_plot_and_operand_acquisition(tmp_path, explicit_operands):
    import matplotlib.pyplot as plt
    src = sources.CurrentSource('stim', 'a', '0', dc=2, pwl=[0, .5, 1, 1])
    tb = _bench(tmp_path, ['@istim__silk_pwl[current]', '@istim__silk_dc[current]'],
                [src], data=[[.5, 1], [2, 2]])
    current = tb.probe('i(stim)', analysis='tran')
    expected = [[2.5, 3]]
    if explicit_operands:
        tb.probe('i(Istim__silk_pwl)', analysis='tran')
        tb.probe('i(Istim__silk_dc)', analysis='tran')
        expected += [[.5, 1], [2, 2]]
    fig = tb.plot(show=False)
    try:
        assert len(fig.axes) == len(expected)
        for ax, values in zip(fig.axes, expected):
            np.testing.assert_allclose(ax.lines[0].get_ydata(), values)
    finally:
        plt.close(fig)
    assert tb._probe_registry[0] is current
    tb._prepare_run()
    assert tb.netlist._pending_current_probes == ('Istim__silk_pwl', 'Istim__silk_dc')
    assert tb.netlist._pending_current_probe_analyses == ('tran', 'tran')
    output = PySpiceNetlister(None).generate_output_block(
        str(tmp_path), {'tran': Transient(1e-9, 1e-6)}, save_list=['a'],
        current_probe_names=tb.netlist._pending_current_probes,
        source_instance_names=src.physical_instance_names())
    for card in ('Istim__silk_pwl', 'Istim__silk_dc'):
        assert output.count(f'"@{card}[current]"') == 1


@pytest.mark.parametrize('transport', ['deepcopy', 'pickle'])
def test_logical_current_metric_copy_keeps_one_plot_and_rebinds_leaves(tmp_path, transport):
    import copy
    import pickle
    import matplotlib.pyplot as plt
    from silk.metrics import rise_time
    from silk.circuit.module_test import _unbound_leaves
    src = sources.CurrentSource('stim', 'a', '0', dc=2, pwl=[0, .5, 1, 1])
    tb = _bench(tmp_path, ['@istim__silk_pwl[current]', '@istim__silk_dc[current]'],
                [src], data=[[.5, 1], [2, 2]])
    tb._metrics = []
    current = tb.probe('i(stim)', analysis='tran')
    tb.add_metric('rise', rise_time, sig=current * 2)
    restored = copy.deepcopy(tb) if transport == 'deepcopy' else pickle.loads(pickle.dumps(tb))
    assert len(restored._probe_registry) == 1
    for leaf in _unbound_leaves(restored._probe_registry[0]):
        assert leaf._bench() is restored
        assert leaf._analysis == 'tran'
    fig = restored.plot(show=False)
    try:
        assert len(fig.axes) == 1
        np.testing.assert_allclose(fig.axes[0].lines[0].get_ydata(), [2.5, 3])
    finally:
        plt.close(fig)
    restored._prepare_run()
    assert restored.netlist._pending_current_probes == ('Istim__silk_pwl', 'Istim__silk_dc')


@pytest.mark.parametrize('simulator', ['ngspice', 'xyce'])
def test_logical_current_default_plot_selects_each_corners_operands(tmp_path, simulator):
    import matplotlib.pyplot as plt
    benches = []
    for label, offset in [('tt', 2), ('ff', 3)]:
        directory = tmp_path / label
        directory.mkdir()
        source = sources.CurrentSource('stim', 'a', '0', dc=offset, pwl=[0, .5, 1, 1])
        names = (['@istim__silk_pwl[current]', '@istim__silk_dc[current]'] if simulator == 'ngspice'
                 else ['Istim__silk_pwl', 'Istim__silk_dc'])
        bench = _bench(directory, names, [source], data=[[.5, 1], [offset, offset]])
        bench.netlist.simulator = simulator
        benches.append(bench)
    tb = benches[0]
    tb._sweep_table = {label: str(tmp_path / label / 's.raw') for label in ('tt', 'ff')}
    current = tb.probe('i(stim)', analysis='tran')
    np.testing.assert_allclose(current.corner('ff').y, [3.5, 4])
    fig = tb.plot(show=False)
    try:
        assert len(fig.axes) == 1
        assert len(fig.axes[0].lines) == 2
        for line, expected in zip(fig.axes[0].lines, ([2.5, 3], [3.5, 4])):
            np.testing.assert_allclose(line.get_ydata(), expected)
    finally:
        plt.close(fig)


@pytest.mark.parametrize('simulator', ['ngspice', 'xyce'])
def test_logical_current_corner_ac_keeps_typed_refusal(tmp_path, simulator):
    from silk.exceptions import NotSupportedError
    from silk.signals.rawfile import RawFile, RawPlot, RawVector, write_rawfile
    src = sources.CurrentSource('stim', 'a', '0', dc=2, ac=1, pwl=[0, .5, 1, 1])
    tb = _bench(tmp_path, [], [src])
    tb.netlist.simulator = simulator
    path = tmp_path / 'ac.raw'
    write_rawfile(path, RawFile([RawPlot({'plotname': 'AC Analysis', 'title': 'ac'},
        [RawVector('frequency', 'frequency', data=np.array([1., 2.]))],
        npoints=2, real=True, padded=True)]))
    tb._sweep_table = {'tt': str(path)}
    with pytest.raises(NotSupportedError, match='ac=.*0 V VoltageSource'):
        tb.probe('i(stim)', analysis='ac').corner('tt')
