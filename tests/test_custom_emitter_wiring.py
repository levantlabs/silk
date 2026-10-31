"""Custom emitters retain their established single-owner wiring behavior."""
import copy
import pickle

import pytest
from silk import sources
from silk.circuit.modules import Module
from silk.elements import elements
from silk.exceptions import UndeclaredNetError


class CustomVoltage(sources.VoltageSource):
    emissions = 0

    def to_spice(self):
        type(self).emissions += 1
        return super().to_spice()


class CustomCurrent(sources.CurrentSource):
    emissions = 0

    def to_spice(self):
        type(self).emissions += 1
        return super().to_spice()


class CustomResistor(type(elements.resistor('template', p='x', n='0', value=1))):
    emissions = 0

    def to_spice(self):
        type(self).emissions += 1
        return super().to_spice()


class CustomF(sources.CurrentControlledCurrentSource):
    def to_spice(self):
        return super().to_spice()


class CustomH(sources.CurrentControlledVoltageSource):
    def to_spice(self):
        return super().to_spice()


def make_module(tmp_path):
    return Module(None, str(tmp_path), 'custom.cir', None, module_name='custom')


def graph(module):
    return ({name: [(p.instance, p.port, p.net, p.kind) for p in net.pins]
             for name, net in module.net_dict.items()}, dict(module._element_mgr._pin_to_net))


def move(module, source, route, destination):
    if route == 'assignment':
        source.positive = destination
    elif route == 'update':
        module.update_source('v', ['positive'], [destination])
    else:
        module.connect_source('v', ['positive'], [destination])


@pytest.mark.parametrize('route', ['assignment', 'update', 'connect'])
@pytest.mark.parametrize('strict', [False, True])
def test_custom_source_rewiring(tmp_path, route, strict):
    module = make_module(tmp_path)
    source = CustomVoltage('v', positive='x', negative='0', dc=1)
    module.add_source(source)
    module.add_net('y')
    CustomVoltage.emissions = 0
    if strict:
        module._element_mgr.strict_nets = True
        before = graph(module), source.positive
        with pytest.raises(UndeclaredNetError):
            move(module, source, route, 'unknown')
        assert (graph(module), source.positive) == before
    else:
        move(module, source, route, 'y')
        assert source.positive == 'y'
        assert module._element_mgr._pin_to_net['v', 'positive'] == 'y'
        assert [(p.instance, p.port) for p in module.net_dict['y'].pins] == [('v', 'positive')]
        assert not module.net_dict['x'].pins
    assert CustomVoltage.emissions == 0


@pytest.mark.parametrize('strict', [False, True])
def test_custom_factory_device_rewiring(tmp_path, strict):
    module = make_module(tmp_path)
    device = CustomResistor('r', p='x', n='0', value=1)
    module.add(device)
    module.add_net('y')
    CustomResistor.emissions = 0
    if strict:
        module._element_mgr.strict_nets = True
        before = graph(module), dict(device.nets)
        with pytest.raises(UndeclaredNetError):
            device.p = 'unknown'
        assert (graph(module), device.nets) == before
    else:
        device.p = 'y'
        assert device.nets['p'] == 'y'
        assert module._element_mgr._pin_to_net['r', 'p'] == 'y'
        assert [(p.instance, p.port) for p in module.net_dict['y'].pins] == [('r', 'p')]
        assert not module.net_dict['x'].pins
    assert CustomResistor.emissions == 0


def round_trip(module, route):
    return copy.deepcopy(module) if route == 'deepcopy' else pickle.loads(pickle.dumps(module))


@pytest.mark.parametrize('route', ['deepcopy', 'pickle'])
def test_custom_only_source_copy_is_pure_and_isolated(tmp_path, route):
    module = make_module(tmp_path)
    source = CustomVoltage('v', positive='x', negative='0', dc=1)
    module.add_source(source)
    keys = set(vars(module))
    before = graph(module)
    CustomVoltage.emissions = 0
    copied = round_trip(module, route)
    assert set(vars(module)) == keys
    copied.sig_sources_dict['v'].positive = 'y'
    assert copied._element_mgr._pin_to_net['v', 'positive'] == 'y'
    assert [(p.instance, p.port) for p in copied.net_dict['y'].pins] == [('v', 'positive')]
    assert not copied.net_dict['x'].pins
    assert graph(module) == before
    assert source.positive == 'x'
    assert CustomVoltage.emissions == 0


@pytest.mark.parametrize('route', ['deepcopy', 'pickle'])
@pytest.mark.parametrize('factory', [CustomF, CustomH])
def test_custom_control_context_round_trip(tmp_path, route, factory):
    module = make_module(tmp_path)
    module.add_source(sources.VoltageSource('sense', positive='x', negative='0', dc=0))
    params = {'gain': 2} if factory is CustomF else {'transresistance': 2}
    source = factory('controlled', out_pos='x', out_neg='0', ctrl_vname='sense', **params)
    module.add_source(source)
    expected = source.to_spice()
    assert expected.split()[3] == 'Vsense'
    keys = set(vars(module))
    copied = round_trip(module, route)
    assert copied.sig_sources_dict['controlled'].to_spice() == expected
    assert set(vars(module)) == keys


@pytest.mark.parametrize('factory', [CustomVoltage, CustomCurrent])
@pytest.mark.parametrize('refuse', [False, True])
def test_custom_mixed_batch(tmp_path, factory, refuse):
    module = make_module(tmp_path)
    source = factory('v', positive='x', negative='0', dc=1)
    module.add_source(source)
    module.add_net('y')
    module.add_net('z')
    module._element_mgr.strict_nets = True
    factory.emissions = 0
    before = source.dc, source.positive, source.negative, graph(module)
    if refuse:
        with pytest.raises(UndeclaredNetError):
            module.update_source('v', ['dc', 'positive', 'negative'], [2, 'y', 'missing'])
        assert (source.dc, source.positive, source.negative, graph(module)) == before
    else:
        module.update_source('v', ['dc', 'positive', 'negative'], [2, 'y', 'z'])
        assert (source.dc, source.positive, source.negative) == (2, 'y', 'z')
        assert module._element_mgr._pin_to_net == {('v', 'positive'): 'y', ('v', 'negative'): 'z'}
        assert [(p.instance, p.port) for p in module.net_dict['y'].pins] == [('v', 'positive')]
        assert [(p.instance, p.port) for p in module.net_dict['z'].pins] == [('v', 'negative')]
        assert not module.net_dict['x'].pins
        assert not module.net_dict['0'].pins
    assert factory.emissions == 0


@pytest.mark.parametrize('factory', [CustomVoltage, CustomCurrent])
def test_custom_unplaced_waveform_batch(factory):
    source = factory('v', positive='x', negative='0', dc=1,
                     sine=sources.Sine(voffset=0, vampl=1, freq=1))
    pulse = sources.Pulse(0, 1, tr='1n', tf='1n', pw='1u', per='2u')
    factory.emissions = 0
    source._apply_update({'dc': 2, 'pulse': pulse, 'sine': None})
    assert source.dc == 2
    assert source.pulse is pulse
    assert source.sine is None
    assert factory.emissions == 0
