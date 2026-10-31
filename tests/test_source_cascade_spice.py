"""Real simulator parity and current acquisition for source cascades."""
import numpy as np
import pytest
from silk import sources
from silk.circuit.modules import Module
from silk.elements import elements
from silk.simulation import Op, Transient, DC, AC
from silk.exceptions import NotSupportedError
from silk.circuit.module_test import _load_metric_path
from tests._env import needs_ngspice, xyce_executable

pytestmark = pytest.mark.slow
BACKENDS = [pytest.param('ngspice', marks=needs_ngspice),
            pytest.param('xyce', marks=[pytest.mark.requires_xyce,
                                       pytest.mark.skipif(not xyce_executable(), reason='Xyce unavailable')])]


def wave(slot):
    return {'pulse': lambda: sources.Pulse(.5, 1, td=1e-6, tr=1e-8, tf=1e-8, pw=1e-6, per=3e-6),
            'sine': lambda: sources.Sine(.5, .3, 1e3, td=1e-6, phase=30),
            'pwl': lambda: sources.PWL([(0, .5), (1e-6, 1)]),
            'waveform': lambda: sources.PWL([(0, .5), (1e-6, 1)]),
            'exponential': lambda: sources.Exponential(.5, 1, 1e-6, 1e-6, 2e-6, 1e-6),
            'sffm': lambda: sources.SFFM(.5, .3, 1e3, 2, 10)}[slot]()


def bench(tmp_path, pdk, backend, cls, slot, dc, analysis):
    mod = Module('sky130', str(tmp_path), 'source.cir', pdk_details=pdk)
    mod.simulator = backend
    mod.add(elements.resistor('load', p='out', n='0', value=1))
    mod.add_source(cls('stim', 'out', '0', dc=dc, ac=1, **{slot: wave(slot)}))
    mod.add_sim_type(analysis)
    mod._pending_current_probes = mod.sig_sources_dict['stim'].physical_instance_names()
    mod.set_save_list(['out'])
    return mod


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('cls,sign', [(sources.VoltageSource, 1), (sources.CurrentSource, -1)])
@pytest.mark.parametrize('slot,initial', [('pulse', .5), ('sine', .65), ('pwl', .5),
                                         ('waveform', .5), ('exponential', .5), ('sffm', .5)])
@pytest.mark.parametrize('dc', [0, .2])
def test_op_transient_and_measured_current(tmp_path, pdk, backend, cls, sign, slot, initial, dc):
    op = bench(tmp_path / 'op', pdk, backend, cls, slot, dc, Op())
    from silk.simulation._ngspice_version import ngspice_version
    if backend == 'ngspice' and slot == 'sffm' and ngspice_version() != '42':
        with pytest.raises(NotSupportedError, match='ngspice.*SFFM.*stim'):
            op.simulate()
        return
    op.simulate()
    assert op.load_signal('out').y[0] == pytest.approx(sign * (dc + initial), abs=1e-8)
    mod = bench(tmp_path / 'tran', pdk, backend, cls, slot, dc, Transient(1e-8, 2e-6))
    mod.simulate()
    voltage = mod.load_signal('out')
    assert voltage.y[0] == pytest.approx(sign * (dc + initial), abs=1e-8)
    current = _load_metric_path(mod, 'i(stim)')
    np.testing.assert_allclose(current.y, -voltage.y, atol=1e-8)
    if slot in ('pulse', 'sine', 'pwl', 'waveform'):
        np.testing.assert_allclose(voltage.y, sign * mod.sig_sources_dict['stim'].to_numpy(voltage.x), atol=1e-7)


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('cls,sign', [(sources.VoltageSource, 1), (sources.CurrentSource, -1)])
def test_dc_sweeps_offset_and_ac_is_applied_once(tmp_path, pdk, backend, cls, sign):
    mod = bench(tmp_path / 'dc', pdk, backend, cls, 'pwl', .2, DC('stim', 0, 1, .5))
    mod.simulate()
    sig = mod.load_signal('out')
    np.testing.assert_allclose(sig.y, sign * (sig.x + .5), atol=1e-8)
    mod = bench(tmp_path / 'ac', pdk, backend, cls, 'pwl', .2, AC('lin', 3, 1e3, 2e3))
    if cls is sources.CurrentSource:
        with pytest.raises(NotSupportedError, match='ac=.*0 V VoltageSource'):
            mod.generate_netlist()
        mod._pending_current_probes = ()
    mod.simulate()
    np.testing.assert_allclose(mod.load_signal('out').y, sign, atol=1e-8)
    if cls is sources.CurrentSource:
        for name in ('stim', 'Istim__silk_pwl', 'Istim__silk_dc', '@istim__silk_dc[current]'):
            with pytest.raises(NotSupportedError, match='ac=.*0 V VoltageSource'):
                _load_metric_path(mod, f'i({name})')
    else:
        np.testing.assert_allclose(_load_metric_path(mod, 'i(stim)').y, -sign, atol=1e-8)


@pytest.mark.parametrize('backend', BACKENDS)
def test_explicit_voltage_ammeter_measures_current_source_ac(tmp_path, pdk, backend):
    mod = bench(tmp_path, pdk, backend, sources.CurrentSource, 'pwl', .2, AC('lin', 3, 1e3, 3e3))
    mod.sig_sources_dict['stim'].positive = 'internal'
    mod.add_source(sources.VoltageSource('meter', 'out', 'internal', dc=0))
    mod._pending_current_probes = ('Vmeter',)
    mod.simulate()
    np.testing.assert_allclose(mod.load_signal('Vmeter', 'I').y, [1, 1, 1], atol=1e-8)


@pytest.mark.parametrize('backend', BACKENDS)
def test_nested_array_current_sum_and_ac_refusal(tmp_path, pdk, backend):
    from silk import PortDir
    from silk.circuit.module_test import _branch_vector_name
    child = Module('sky130', str(tmp_path), 'child.cir', pdk_details=pdk, module_name='child')
    child.define_module_ports({'out': PortDir.OUT})
    child.add_source(sources.CurrentSource('stim[0]', 'out', '0', dc=.2, pwl=[0, .5, 1e-6, 1]))
    dut = Module('sky130', str(tmp_path), 'dut.cir', pdk_details=pdk, module_name='dut')
    dut.define_module_ports({'out': PortDir.OUT})
    dut.add_module(child, 'leaf[0]', nets={'out': 'out'})
    tb = dut.create_test('nested')
    tb.add_module_under_test({'out': 'out'})
    tb.netlist.add(elements.resistor('load', p='out', n='0', value=1))
    tb.netlist.simulator = backend
    tb.add_sim_type(Transient(1e-8, 2e-6))
    current = tb.probe('i(xdut.xleaf[0].stim[0])')
    voltage = tb.probe('out')
    tb.run()
    np.testing.assert_allclose(current.y, -voltage.y, atol=1e-8)
    np.testing.assert_allclose(_load_metric_path(tb.netlist, 'i(xdut.xleaf[0].stim[0])').y,
                               current.y, atol=1e-8)
    raw_name = 'xdut:xleaf<0>:Istim<0>__silk_dc' if backend == 'xyce' else 'xdut.xleaf<0>.Istim<0>__silk_dc'
    np.testing.assert_allclose(tb.netlist.load_signal(raw_name, 'I').y, .2, atol=1e-8)
    tb.netlist.sim_analysis_dict = {'ac': AC('lin', 3, 1e3, 2e3)}
    with pytest.raises(NotSupportedError, match='ac=.*0 V VoltageSource'):
        tb.netlist.generate_netlist()


@pytest.mark.parametrize('backend', BACKENDS)
def test_public_current_annotation_sums_measured_source_cards(tmp_path, pdk, backend):
    mod = Module('sky130', str(tmp_path), 'annotation.cir', pdk_details=pdk)
    mod.simulator = backend
    mod.add_source(sources.CurrentSource('stim', 'out', '0', dc=2, pwl=[0, -1, 1e-6, -1]))
    mod.add_source(sources.VoltageSource('meter', 'out', '0', dc=0))
    mod.add_sim_type(Transient(1e-8, 1e-6))
    mod._pending_current_probes = ('Istim__silk_pwl', 'Istim__silk_dc', 'Vmeter')
    mod.simulate()
    report = mod.annotate_currents()
    binding = next(b for b in mod.net_dict['out'].pins if b.instance == 'stim')
    assert binding.current_vector is None
    assert binding.avg_current == pytest.approx(-1)
    assert binding.rms_current == pytest.approx(1)
    assert report.pins_annotated == 4


@pytest.mark.parametrize('cls,sign', [(sources.VoltageSource, 1), (sources.CurrentSource, -1)])
@pytest.mark.parametrize('slot,initial', [('am', 0), ('trnoise', 0), ('trrandom', .5)])
@pytest.mark.parametrize('dc', [0, .2])
@needs_ngspice
def test_ngspice_additive_stochastic_and_am_start(tmp_path, pdk, cls, sign, slot, initial, dc):
    def build():
        return {'am': sources.AM(.3, .5, 1e3, 1e6),
                'trnoise': sources.TransientNoise(.01, 1e-7),
                'trrandom': sources.RandomSource(1, 1e-7, param1=.1, param2=.5)}[slot]
    readings = []
    for label, analysis in [('op', Op()), ('tran', Transient(1e-8, 1e-6))]:
        mod = Module('sky130', str(tmp_path / label), 'start.cir', pdk_details=pdk)
        mod.add(elements.resistor('load', p='out', n='0', value=1))
        mod.add_source(cls('stim', 'out', '0', dc=dc, **{slot: build()}))
        mod.add_sim_type(analysis)
        mod.simulate()
        readings.append(mod.load_signal('out').y[0])
    np.testing.assert_allclose(readings, sign * (dc + initial), atol=1e-8)


@pytest.mark.parametrize('backend', BACKENDS)
def test_nested_voltage_private_nodes_keep_designer_array_spelling(tmp_path, pdk, backend):
    from silk import PortDir
    child = Module('sky130', str(tmp_path), 'cell.cir', pdk_details=pdk, module_name='cell')
    child.define_module_ports({'out': PortDir.OUT})
    child.add_source(sources.VoltageSource('bias[0]', 'out', '0', dc=.2, sine=sources.Sine(.5, .1, 1e3)))
    mod = Module('sky130', str(tmp_path), 'private.cir', pdk_details=pdk)
    mod.simulator = backend
    for i in (0, 1):
        mod.add_module(child, f'leaf[{i}]', nets={'out': f'out[{i}]'})
    mod.add_sim_type(Transient(1e-8, 1e-6))
    paths = [f'xleaf[{i}].__silk_src_Vbias[0]' for i in (0, 1)]
    mod.set_save_list([*paths, 'out[0]', 'out[1]'])
    mod.simulate()
    for path in paths:
        np.testing.assert_allclose(mod.load_signal(path).y, .2, atol=1e-8)
    assert all('<0>' not in name.rsplit('.', 1)[-1].rsplit(':', 1)[-1]
               for name in mod.get_signal_names(raw=True) if '__silk_src_' in name)


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('kind,ratio', [('CCCS', 2), ('CCVS', -2)])
def test_controlled_sources_sense_the_waveform_series_branch(tmp_path, pdk, backend, kind, ratio):
    mod = bench(tmp_path, pdk, backend, sources.VoltageSource, 'sine', .2, Transient(1e-8, 2e-6))
    kwargs = {'gain': 2} if kind == 'CCCS' else {'transresistance': 2}
    mod.add_source(getattr(sources, kind)('copy', out_pos='copyout', out_neg='0',
                                         ctrl_vname='stim', **kwargs))
    mod.add(elements.resistor('copyload', p='copyout', n='0', value=1))
    mod.set_save_list(['out', 'copyout'])
    mod.simulate()
    np.testing.assert_allclose(mod.load_signal('copyout').y, ratio * mod.load_signal('out').y, atol=1e-8)


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('spelling', ['istim[0]__silk_dc', '@istim[0]__silk_dc[current]'])
def test_public_raw_current_probe_acquires_array_card(tmp_path, pdk, backend, spelling):
    dut = Module('sky130', str(tmp_path), 'dut.cir', pdk_details=pdk, module_name='dut')
    from silk import PortDir
    dut.define_module_ports({'out': PortDir.OUT})
    dut.add(elements.resistor('load', p='out', n='0', value=1))
    tb = dut.create_test('raw')
    tb.add_module_under_test({'out': 'out'})
    tb.netlist.simulator = backend
    tb.add_source(sources.CurrentSource('stim[0]', 'out', '0', dc=.2, pwl=[0, .5, 1e-6, 1]))
    tb.add_sim_type(Transient(1e-8, 2e-6))
    current = tb.probe(f'i({spelling})', analysis='tran')
    tb.run()
    np.testing.assert_allclose(current.y, .2, atol=1e-8)


@pytest.mark.parametrize('inferred', [False, True])
def test_ngspice_mixed_analysis_transient_probe_remains_measurable(tmp_path, pdk, inferred):
    dut = Module('sky130', str(tmp_path), 'dut.cir', pdk_details=pdk, module_name='dut')
    from silk import PortDir
    dut.define_module_ports({'out': PortDir.OUT})
    dut.add(elements.resistor('load', p='out', n='0', value=1))
    tb = dut.create_test('mixed')
    tb.add_module_under_test({'out': 'out'})
    tb.add_source(sources.CurrentSource('stim', 'out', '0', dc=.2, ac=1, pwl=[0, .5, 1e-6, 1]))
    tb.add_sim_type(Transient(1e-8, 2e-6))
    tb.add_sim_type(AC('lin', 3, 1e3, 2e3))
    current = tb.probe('i(stim)', analysis=None if inferred else 'tran')
    if inferred:
        from silk.metrics import rise_time
        tb.add_metric('rise', rise_time, sig=current, initial=.7, final=1.2)
    voltage = tb.probe('out', analysis='tran')
    tb.run()
    np.testing.assert_allclose(current.resolve(tb.netlist, analysis='tran').y, -voltage.y, atol=1e-8)
    if inferred:
        assert tb._metric_values['rise'] == pytest.approx(.8e-6, rel=.02)


@pytest.mark.parametrize('cls,sign', [(sources.VoltageSource, 1), (sources.CurrentSource, -1)])
@pytest.mark.parametrize('slot,initial', [('am', 0), ('sffm', .7915166986870767)])
@needs_ngspice
def test_modulated_source_start_or_version_refusal(tmp_path, pdk, cls, sign, slot, initial):
    from silk.simulation._ngspice_version import ngspice_version
    readings = []
    for label, analysis in [('op', Op()), ('tran', Transient(1e-9, 1e-6))]:
        mod = Module('sky130', str(tmp_path / label), 'modulation.cir', pdk_details=pdk)
        mod.add(elements.resistor('load', p='out', n='0', value=1))
        waveform = (sources.AM(.3, .5, 10e3, 100e3) if slot == 'am'
                    else sources.SFFM(.5, .3, 100e3, 2, 10e3, 30, 40))
        mod.add_source(cls('stim', 'out', '0', dc=.2, waveform=waveform))
        mod.add_sim_type(analysis)
        if slot == 'sffm' and ngspice_version() != '42':
            with pytest.raises(NotSupportedError, match='ngspice.*stim'):
                mod.simulate()
        else:
            mod.simulate()
            readings.append(mod.load_signal('out').y[0])
    if slot == 'am' or ngspice_version() == '42':
        np.testing.assert_allclose(readings, sign * (.2 + initial), atol=1e-8)


@pytest.mark.parametrize('backend', BACKENDS)
def test_authored_dc_neighbor_has_distinct_cards_and_simulates(tmp_path, pdk, backend):
    mod = Module('sky130', str(tmp_path), 'neighbors.cir', pdk_details=pdk)
    mod.simulator = backend
    mod.add_source(sources.VoltageSource('in', 'out', '0', dc=1.2,
                                        sine=sources.Sine(0, .2, 1e6)))
    mod.add_source(sources.VoltageSource('in_dc', 'other', '0', dc=.5))
    mod.add(elements.resistor('load', p='out', n='0', value=1000))
    mod.add(elements.resistor('other_load', p='other', n='0', value=1000))
    mod.add_sim_type(Transient(1e-9, 1e-6))
    mod.set_save_list(['out', 'other'])
    mod._pending_current_probes = ('Vin__silk_wave', 'Vin__silk_dc', 'Vin_dc')
    mod_deck_path = mod.generate_netlist()
    from pathlib import Path
    deck = Path(mod_deck_path).read_text()
    assert 'Vin__silk_wave out __silk_src_Vin ' in deck
    assert 'Vin__silk_dc __silk_src_Vin 0 DC 1.2' in deck
    assert 'Vin_dc other 0 DC 0.5' in deck
    mod.simulate()
    out = mod.load_signal('out')
    np.testing.assert_allclose(out.y, 1.2 + .2 * np.sin(2 * np.pi * 1e6 * out.x), atol=1e-8)
    np.testing.assert_allclose(mod.load_signal('other').y, .5, atol=1e-8)
    np.testing.assert_allclose(_load_metric_path(mod, 'i(in)').y, -out.y / 1000, atol=1e-9)
    np.testing.assert_allclose(_load_metric_path(mod, 'i(in_dc)').y, -.0005, atol=1e-9)


@pytest.mark.parametrize('cls,sign', [(sources.VoltageSource, 1), (sources.CurrentSource, -1)])
@pytest.mark.parametrize('delay', [0, 200e-9])
@pytest.mark.parametrize('phase', [0, 30])
@pytest.mark.parametrize('parameters', ['numeric', 'si', 'expression'])
@needs_ngspice
def test_am_native_card_signal_parity(tmp_path, pdk, cls, sign, delay, phase, parameters):
    from silk.simulation._ngspice_compat import NgspiceCompatibility
    policy = NgspiceCompatibility()
    if phase and not policy.has('am_new_order'):
        pytest.skip('Legacy AM ignores phase; refusal is covered by emission tests')
    readings = []
    for label, analysis in [('op', Op()), ('tran', Transient(1e-9, 1e-6))]:
        mod = Module('sky130', str(tmp_path / label), 'am.cir', pdk_details=pdk)
        mod.add(elements.resistor('load', p='out', n='0', value=1))
        va, vo = {'numeric': (.3, .5), 'si': ('300m', '500m'),
                  'expression': ('{.1 + .2}', '{.4 + .1}')}[parameters]
        wave = sources.AM(va, vo, '10k', '100k', td=delay, phases=phase)
        mod.add_source(cls('stim', 'out', '0', dc=.2, waveform=wave))
        mod.add_sim_type(analysis)
        mod.simulate()
        sig = mod.load_signal('out')
        readings.append(sig.y[0])
        if label == 'tran':
            t = np.maximum(sig.x - delay, 0)
            # Independent analytic contract; includes the additive dc source.
            expected = sign * (.2 + .3 * (.5 + np.sin(2*np.pi*1e4*t + np.deg2rad(phase)))
                               * np.sin(2*np.pi*1e5*t))
            np.testing.assert_allclose(sig.y, expected, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(readings, sign * .2, rtol=1e-12, atol=1e-12)
