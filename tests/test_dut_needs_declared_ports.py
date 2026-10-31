"""TASK-359 — a named Module with no declared ports is refused with a clear error."""
import pytest

from silk import Module, load_pdk
from silk.elements import elements
from silk.exceptions import CircuitError, NotASubcircuitError
from silk.simulation import Transient


def portless(tmp_path):
    mod = Module('sky130', str(tmp_path), 'blk.cir', load_pdk('sky130'),
                 module_name='blk')
    mod.add(elements.resistor('r1', p='a', n='0', value=1000))
    return mod


def test_add_module_under_test_refuses_portless_named_module(tmp_path):
    blk = portless(tmp_path)
    try:
        tb = blk.create_test('t')
        with pytest.raises(NotASubcircuitError) as exc:
            tb.add_module_under_test({})
        msg = str(exc.value)
        assert isinstance(exc.value, CircuitError)
        assert "'blk'" in msg and 'define_module_ports' in msg and '{}' in msg
        assert tb.netlist is None
    finally:
        blk.close()


@pytest.mark.parametrize('through_placement', [True, False])
def test_netlist_backstop_is_not_an_attribute_error(tmp_path, through_placement):
    """Live references still validate the current port declaration at emission."""
    blk = portless(tmp_path)
    blk.define_module_ports({})
    try:
        tb = blk.create_test('t')
        tb.add_module_under_test({})
        target = next(tb.netlist.iter_instances()).module if through_placement else blk
        target.pin_list = None
        with pytest.raises(NotASubcircuitError) as exc:
            tb.netlist.generate_netlist()
        assert 'define_module_ports' in str(exc.value)
    finally:
        blk.close()


def test_empty_declared_ports_still_work(tmp_path):
    blk = portless(tmp_path)
    blk.define_module_ports({})
    try:
        tb = blk.create_test('t')
        tb.add_module_under_test({})
        tb_netlist_deck_path = tb.netlist.generate_netlist()
        with open(tb_netlist_deck_path) as fh:
            assert 'xdut' in fh.read().lower()
    finally:
        blk.close()
