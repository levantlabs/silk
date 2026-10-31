"""TASK-349: add_net on a declared port bus keeps the declared BusSpec."""

import pytest

from silk.circuit.modules import Module
from silk.circuit.ports import Port, PortDir
from silk.exceptions import CircuitError


def make(tmp_path, direction, **kw):
    child = Module('sky130', str(tmp_path), 'child.cir', None,
                   module_name='child')
    child.define_module_ports({'P': Port(dir=direction, bits=4, **kw)})
    return child


@pytest.mark.parametrize('direction', [PortDir.IN, PortDir.OUT, PortDir.INOUT])
def test_declared_spec_preserved(tmp_path, direction):
    child = make(tmp_path, direction)
    before = child._buses['P']
    handle = child.add_net('P', bits=4)
    assert child._buses['P'] == before
    assert child._buses['P'].direction is direction
    assert handle.width == 4
    assert all(f'P[{i}]' in child.net_dict for i in range(4))


def test_role_and_domain_preserved(tmp_path):
    from silk.circuit.ports import PortRole
    child = make(tmp_path, PortDir.IN, role=PortRole.DIGITAL)
    before = child._buses['P']
    child.add_net('P', bits=4)
    assert child._buses['P'] == before
    assert before.role is PortRole.DIGITAL


def test_in_port_keeps_scalar_broadcast(tmp_path):
    child = make(tmp_path, PortDir.IN)
    child.add_net('P', bits=4)
    parent = Module('sky130', str(tmp_path), 'parent.cir', None,
                    module_name='parent')
    s = parent.add_net('s')
    parent.add_module(child, 'u1', P=s)
    assert (parent.netlister.generate_device_instances(parent.elem_dict).splitlines(keepends=True)[-1]
            == 'xu1 s s s s child\n')


def test_out_port_still_refuses_as_output(tmp_path):
    child = make(tmp_path, PortDir.OUT)
    child.add_net('P', bits=4)
    parent = Module('sky130', str(tmp_path), 'parent.cir', None,
                    module_name='parent')
    s = parent.add_net('s')
    with pytest.raises(CircuitError) as caught:
        parent.add_module(child, 'u1', P=s)
    assert '(output)' in str(caught.value)
    assert '(inout)' not in str(caught.value)


def test_width_mismatch_raises(tmp_path):
    child = make(tmp_path, PortDir.IN)
    with pytest.raises(CircuitError, match='must include every port bit'):
        child.add_net('P', bits=3)
    with pytest.raises(CircuitError, match='must include every port bit'):
        child.add_net('P', bits=range(1, 9))
    assert child._buses['P'].direction is PortDir.IN


@pytest.mark.parametrize('naming', ['bracketed', 'underscore'])
@pytest.mark.parametrize('direction', list(PortDir))
def test_wider_internal_bus_preserves_formal_bits(tmp_path, naming, direction):
    from silk.circuit.ports import PortRole
    from silk.circuit.voltage_domain import VoltageDomain
    domain = VoltageDomain('AVDD', nominal=1.8)
    child = Module(None, str(tmp_path), 'child.cir', None, module_name='child')
    child.define_module_ports({'P': Port(dir=direction, bits=range(2, 6),
                                       naming=naming, role=PortRole.ANALOG,
                                       domain=domain)})
    formal = child._buses['P']
    for indices in (range(8), range(1, 9), range(2, 6)):
        handle = child.add_net('P', bits=indices)
        assert handle._index_vector == tuple(('P', i) for i in indices)
        names = [f'P[{i}]' if naming == 'bracketed' else f'P_{i}' for i in indices]
        assert [net.name for net in handle] == names
        assert child.nets.P is handle
        assert child.add_net('P', bits=indices) is handle
        assert child._buses['P'] is formal
        assert formal.indices == (2, 3, 4, 5)
        assert formal.direction is direction
        assert formal.role is PortRole.ANALOG
        assert formal.domain == 'AVDD'
        assert list(child.pin_list) == list(formal.expanded_names())
