"""Finite bit sources: reference sequences, timing, validation and factories."""
import inspect

import numpy as np
import pytest

from silk import sources
from silk.circuit.modules import Module
from silk.circuit.variable import Variable
from silk.pdk import load_pdk


def pattern(bits='101', **kwargs):
    params = dict(v1=2, v2=4, ui=1, tr=.2, tf=.4, td=.5)
    params.update(kwargs)
    return sources.BitPattern(bits, **params)


def test_prbs_fixed_prefixes():
    # Post-shift output; the first feedback one traverses all n stages.
    expected = {
        7: '0000001111110111110011110101110000110111010011000101011000001011',
        9: '0000000011111000010000011101000110011011111011010110001001011100',
        15: '0000000000000011111111111111011111111111110011111111111101011111',
        23: '0000000000000000000000111111111111111111000001111111111111000000',
        31: '0000000000000000000000000000001111111111111111111111111111000111',
    }
    for order, bits in expected.items():
        assert sources.prbs_bits(order, len(bits)) == tuple(map(int, bits))


@pytest.mark.parametrize('order', [7, 9, 15])
def test_full_period_and_phase_relation(order):
    period = (1 << order) - 1
    bits = sources.prbs_bits(order, 2 * period)
    assert bits[:period] == bits[period:]
    assert sum(bits[:period]) == (1 << (order - 1)) - 1
    cyclic = bits[:period] + bits[:order - 1]
    assert len({cyclic[i:i + order] for i in range(period)}) == period
    # Independent stage-list XOR/pre-shift reference. XNOR is its complement
    # advanced one sample (left circular rotation), not delayed one sample.
    stages = [1] * order
    tap = {7: 6, 9: 5, 15: 14}[order]
    xor = []
    for _ in range(period):
        xor.append(stages[-1])
        stages = [stages[-1] ^ stages[tap - 1]] + stages[:-1]
    assert bits[:period] == tuple(1 - b for b in xor[1:] + xor[:1])


def test_nondefault_seed_and_required_length():
    assert sources.prbs_bits(7, 8, seed=1) == (0, 0, 0, 0, 0, 1, 1, 1)
    with pytest.raises(TypeError):
        sources.prbs_bits(31)


@pytest.mark.parametrize('field,value', [
    ('order', 8), ('order', True), ('order', 7.0), ('order', '7'),
    ('length', 0), ('length', -1), ('length', True), ('length', 1.5),
    ('seed', -1), ('seed', 128), ('seed', 127), ('seed', False),
    ('seed', 1.0), ('length', Variable(name='n', value=3)),
])
def test_invalid_prbs(field, value):
    args = dict(order=7, length=10, seed=0)
    args[field] = value
    with pytest.raises((TypeError, ValueError), match=field):
        sources.prbs_bits(**args)


@pytest.mark.parametrize('field,value', [
    ('order', 7), ('length', 10), ('seed', 1), ('repeat', 2),
])
def test_whole_number_constructor_settings_require_integers(field, value):
    def construct(setting):
        if field == 'repeat':
            return pattern(repeat=setting)
        args = dict(order=7, length=10, seed=0, v1=0, v2=1,
                    ui=1, tr=.1, tf=.2)
        args[field] = setting
        return sources.PRBS(**args)

    assert construct(value).to_spice()
    for invalid in (float(value), str(value)):
        with pytest.raises(TypeError, match=field):
            construct(invalid)


@pytest.mark.parametrize('bits', ['', '102', '01 ', [0, 1.0], [True], [2],
                                  iter([0, 1]), None, 10])
def test_literal_bits_only(bits):
    with pytest.raises((TypeError, ValueError), match='bits'):
        pattern(bits)


@pytest.mark.parametrize('repeat', [0, -1, True, 1.0, float('inf'), '2'])
def test_repeat_validation(repeat):
    with pytest.raises((TypeError, ValueError), match='repeat'):
        pattern(repeat=repeat)


@pytest.mark.parametrize('field,value', [
    ('ui', 0), ('ui', -1), ('ui', float('inf')), ('td', -1),
    ('v1', float('nan')), ('v2', float('inf')), ('tr', 0), ('tf', 0),
    ('tr', 1.1), ('tf', -1), ('tf', float('nan')),
])
def test_electrical_validation(field, value):
    with pytest.raises(ValueError, match=field):
        pattern(**{field: value})


def test_explicit_timing_and_repeat_join():
    wave = pattern('10', repeat=2)
    times = [-1, 0, .5, 1.49, 1.5, 1.7, 1.9, 2.5, 2.6, 2.7,
             3.5, 3.7, 3.9, 4.5, 100]
    expected = [4, 4, 4, 4, 4, 3, 2, 2, 3, 4, 4, 3, 2, 2, 2]
    np.testing.assert_allclose(wave.to_numpy(times), expected)
    inverted = pattern('10', repeat=2, v1=4, v2=2)
    np.testing.assert_allclose(wave.to_numpy(times) + inverted.to_numpy(times), 6)
    assert '\n+ ' in wave.to_spice()
    assert 'r=' not in wave.to_spice() and 'td=' not in wave.to_spice()


@pytest.mark.parametrize('bits,value', [('0', 2), ('111', 4), ('0000', 2)])
def test_constant_and_equal_levels(bits, value):
    np.testing.assert_equal(pattern(bits).to_numpy([-1, 0, 1, 100]), value)
    np.testing.assert_equal(pattern(bits, v1=3, v2=3).to_numpy([0, 1, 100]), 3)


def test_full_ui_edges_and_mandatory_edges():
    np.testing.assert_allclose(pattern('010', td=0, tr=1, tf=1).to_numpy(
        [0, 1, 1.5, 2, 2.5, 3, 4]), [2, 2, 3, 4, 3, 2, 2])
    for cls in (sources.BitPattern, sources.PRBS):
        assert inspect.signature(cls).parameters['tr'].default is inspect.Parameter.empty
        assert inspect.signature(cls).parameters['tf'].kind is inspect.Parameter.KEYWORD_ONLY


def test_stream_metadata_and_defensive_normalization():
    bits = [1, 0]
    wave = pattern(bits, repeat=2)
    bits[0] = 0
    assert wave.bits == '10'
    assert wave.stream == (1, 0, 1, 0)
    params = wave.as_parameters()
    assert params['kind'] == 'bitpattern' and params['bits'] == '10'
    assert params['repeat'] == 2
    params['bits'] = '0'
    assert wave.as_parameters()['bits'] == '10'
    prbs = sources.PRBS(9, 20, v1=0, v2=1, ui=1, tr=.1, tf=.2)
    assert prbs.stream == sources.prbs_bits(9, 20)
    metadata = prbs.as_parameters()
    assert metadata['kind'] == 'prbs' and metadata['convention'] == 'xnor-post-shift'
    assert metadata['order'] == 9 and metadata['seed'] == 0 and metadata['length'] == 20
    assert 'bits' not in metadata and 'stream' not in metadata


def test_variables_are_authored_and_resolved_afresh():
    amp = Variable(name='amp')
    edge = Variable(name='edge', value=.2)
    wave = pattern('01', v2=amp, tr=edge)
    assert wave.stream == (0, 1)  # no electrical resolution
    assert wave.as_parameters()['v2'] is amp
    with pytest.raises(ValueError, match='no value assigned'):
        wave.to_spice()
    amp.value = 4
    first = wave.to_spice()
    amp.value = 6
    assert wave.to_spice() != first
    np.testing.assert_allclose(wave.to_numpy([1.6, 2]), [4, 6])
    edge.value = 0
    for action in (wave.to_spice, lambda: wave.to_numpy([0])):
        with pytest.raises(ValueError, match='tr'):
            action()
    expression = pattern('01', v2=amp * 2, ui='1', tr='200m', tf='400m')
    assert expression.to_numpy([10])[0] == 12


@pytest.mark.parametrize('quantity,alias,cls', [
    ('voltage', 'v', sources.VoltageSource), ('current', 'i', sources.CurrentSource)])
@pytest.mark.parametrize('kind,params', [('bitpattern', {'bits': '10'}),
                                        ('prbs', {'order': 7, 'length': 12})])
def test_factories_and_registration(quantity, alias, cls, kind, params, tmp_path):
    factory = getattr(getattr(sources, quantity), kind)
    assert factory is getattr(sources, alias + kind)
    mod = Module('sky130', str(tmp_path), 'bits.cir', pdk_details=load_pdk('sky130'))
    for index, pins in enumerate((dict(p='out', n='0'),
                                  dict(positive='out', negative='0'))):
        with mod:
            src = factory(f'data{index}', **pins, **params,
                          v1=2, v2=4, ui=1, tr=.2, tf=.4, dc=3, ac=.1)
        assert isinstance(src, cls)
        np.testing.assert_allclose(src.to_numpy([0, 1, 3]), 3 + src.waveform.to_numpy([0, 1, 3]))
        assert 'DC 3' in src.to_spice() and 'AC 0.1' in src.to_spice()
    assert len(mod.sig_sources_dict) == 2


@pytest.mark.parametrize('cls', [sources.VoltageSource, sources.CurrentSource])
def test_generic_dispatch_and_multiple_waveform_refusal(cls):
    from silk.exceptions import CircuitError
    class Emitter:
        def to_spice(self):
            return 'PWL(0 1 1 1)'
    with pytest.raises(NotImplementedError):
        cls('s', 'out', '0', waveform=Emitter()).to_numpy([0])
    class Modeled(Emitter):
        def to_numpy(self, t):
            return np.full(len(t), 7.)
    generic = Modeled()
    assert cls('s', 'out', '0', waveform=generic).to_numpy([0])[0] == 7
    with pytest.raises(CircuitError, match='pwl.*waveform'):
        cls('s', 'out', '0', waveform=generic, pwl=sources.PWL([(0, 3), (1, 3)]))
    with pytest.raises(CircuitError, match='exponential.*waveform'):
        cls('s', 'out', '0', waveform=generic,
            exponential=sources.Exponential(0, 1)).to_numpy([0])


@pytest.mark.parametrize('kind', ['bitpattern', 'prbs'])
def test_behavioral_testbench_probes_and_sweep(kind, tmp_path):
    from silk import PortDir
    from silk.metrics import final_value
    from silk.simulation import Transient
    from silk.optimize import VariableSpace
    dut = Module('sky130', str(tmp_path), 'dut.cir', pdk_details=load_pdk('sky130'),
                 module_name='dut')
    dut.define_module_ports({'vin': PortDir.IN, 'vout': PortDir.OUT})
    def model_1(ports, t, params):
        ports.vout.v = 2 * ports.vin.v
    dut.set_behavioral_model(model_1, execution='vector')
    dut.simulator = 'behavioral'
    tb = dut.create_test('data')
    tb.add_module_under_test({'vin': 'in', 'vout': 'out'})
    amp = Variable(name='amplitude', value=1)
    kwargs = dict(bits='01') if kind == 'bitpattern' else dict(order=7, length=10)
    voltage = getattr(sources.voltage, kind)(
        'vin', p='in', n='0', **kwargs, v1=.2, v2=amp, ui=1e-6, tr=1e-7, tf=2e-7)
    current = getattr(sources.current, kind)(
        'iin', p='in', n='0', **kwargs, v1=.2, v2=amp, ui=1e-6, tr=1e-7, tf=2e-7)
    tb.add_source(voltage)
    tb.add_source(current)
    tb.add_sim_type(Transient(step=1e-7, stop=12e-6))
    output = tb.probe('out')
    supplied = tb.probe('i(iin)')
    tb.add_metric('final', final_value, sig=output)
    for value in (1., 2.):
        VariableSpace([amp]).apply({'amplitude': value})
        report = tb.run()
        np.testing.assert_allclose(output.y, 2 * voltage.waveform.to_numpy(output.x))
        np.testing.assert_allclose(supplied.y, current.waveform.to_numpy(supplied.x))
        assert report.metrics['final'] == pytest.approx(2 * value)



@pytest.mark.parametrize('field,initial,changed', [
    ('v1', 2., 1.), ('v2', 4., 5.), ('ui', 1., 2.),
    ('td', .5, .7), ('tr', .2, .3), ('tf', .4, .6),
])
@pytest.mark.parametrize('kind', ['bitpattern', 'prbs'])
def test_each_electrical_field_tracks_a_variable(field, initial, changed, kind):
    from silk.optimize import VariableSpace
    var = Variable(name='swept', value=initial)
    args = dict(v1=2., v2=4., ui=1., td=.5, tr=.2, tf=.4)
    args[field] = var
    wave = (sources.BitPattern('101', **args) if kind == 'bitpattern' else
            sources.PRBS(7, 20, **args))
    assert wave.as_parameters()[field] is var
    points = np.linspace(0, 42, 10001)
    old_deck, old_samples = wave.to_spice(), wave.to_numpy(points)
    VariableSpace([var]).apply({'swept': changed})
    assert old_deck != wave.to_spice()
    assert np.max(np.abs(old_samples - wave.to_numpy(points))) > .01
    with pytest.raises(AttributeError):
        wave.stream = (1, 0)


def test_si_timing_and_prbs_share_bitpattern_points():
    prbs = sources.PRBS(7, 20, v1='200m', v2='1200m', ui='1u', td='700n',
                        tr='120n', tf='280n')
    custom = sources.BitPattern(prbs.stream, v1=.2, v2=1.2, ui='1u', td='700n',
                                tr='120n', tf='280n')
    assert prbs.to_spice() == custom.to_spice()
    times = np.linspace(-1e-6, 22e-6, 1000)
    np.testing.assert_array_equal(prbs.to_numpy(times), custom.to_numpy(times))


@pytest.mark.parametrize('kind', ['bitpattern', 'prbs'])
def test_unassigned_expressions_stay_lazy(kind):
    amp = Variable(name='unset_amplitude')
    rate = Variable(name='unset_rate')
    level, interval = amp * 2, 1 / rate
    args = dict(v1=0, v2=level, ui=interval, tr=.1, tf=.2)
    wave = (sources.BitPattern('01', **args) if kind == 'bitpattern' else
            sources.PRBS(7, 10, **args))
    assert wave.stream[-1] == 1
    assert wave.as_parameters()['v2'] is level
    assert wave.as_parameters()['ui'] is interval
    with pytest.raises(ValueError, match='no value assigned'):
        wave.to_spice()
    amp.value, rate.value = 3, 1
    assert wave.to_numpy([100])[0] == 6
    assert '6.0' in wave.to_spice()


@pytest.mark.parametrize('kind', ['bitpattern', 'prbs'])
def test_si_edge_one_ulp_below_ui_stays_in_its_bit(kind):
    args = dict(v1=0, v2=1, ui=1e-5, tr='10u', tf='10u')
    wave = (sources.BitPattern('01' * 14, **args) if kind == 'bitpattern' else
            sources.PRBS(7, 100, **args))
    assert wave.to_spice().startswith('PWL(')
    boundaries = np.arange(1, len(wave.stream) + 1) * 1e-5
    np.testing.assert_allclose(wave.to_numpy(boundaries), wave.stream, atol=1e-13)


@pytest.mark.parametrize('kind', ['bitpattern', 'prbs'])
def test_module_discovers_bit_waveform_variables_without_resolution(kind, tmp_path):
    level, interval = Variable(name='level'), Variable(name='interval')
    args = dict(v1=level, v2=level * 2, ui=interval, tr=interval / 10,
                tf=interval / 5)
    wave = (sources.BitPattern('01', **args) if kind == 'bitpattern' else
            sources.PRBS(7, 10, **args))
    mod = Module('sky130', str(tmp_path), 'bits.cir', pdk_details=load_pdk('sky130'))
    mod.add_source(sources.VoltageSource('data', 'out', '0', waveform=wave))
    assert {id(v) for v in mod.variables()} == {id(level), id(interval)}
    assert wave.stream[-1] == 1


@pytest.mark.parametrize('kind', ['pattern', 'prbs'])
@pytest.mark.parametrize('first_bit', [0, 1])
def test_initial_value_preserves_resolved_level_token(kind, first_bit):
    low = Variable(name='initial_low', value='200m')
    high = Variable(name='initial_high', value='1.2')
    params = dict(v1=low, v2=high, ui=1, tr=.2, tf=.4, td=.5)
    wave = (sources.BitPattern(f'{first_bit}10', **params) if kind == 'pattern'
            else sources.PRBS(7, 8, seed=32 if first_bit else 0, **params))
    assert wave.stream[0] == first_bit
    assert wave.initial_value() == ('1.2' if first_bit else '200m')
    np.testing.assert_allclose(wave.to_numpy([0, .25]), 1.2 if first_bit else .2)
    selected = high if first_bit else low
    selected.value = '700m'
    assert wave.initial_value() == '700m'
    np.testing.assert_allclose(wave.to_numpy([0, .25]), .7)
