"""TASK-400 — a Variable is one shared knob, so copying a circuit keeps it.

``add_module`` and ``create_test``/``add_module_under_test`` place a
deep copy of the module. Before this fix the copy held copies of its
Variables too, so a value assigned to the caller's Variable after placement
never reached the placed child or the embedded DUT: the netlist kept the
placement-time number and nothing raised. The optimizer then varied DUT
variables that the simulated deck never saw.

``Variable.__deepcopy__``/``__copy__`` return the Variable itself. Pickling
does not use those hooks, so the parallel optimizer's pickle snapshot still
freezes each design point's values.
"""
import copy
import pickle
from pathlib import Path

from silk import Module, load_pdk
from silk.circuit.variable import Variable, _VarExpr
from silk.elements import elements


def _block(tmp_path, value, name='blk'):
    mod = Module('sky130', str(tmp_path), f'{name}.cir', load_pdk('sky130'),
                 module_name=name)
    mod.define_module_ports({'a': 'inout', 'b': 'inout'})
    mod.add(elements.resistor('r1', p='a', n='b', value=value))
    return mod


def _top(tmp_path):
    return Module('sky130', str(tmp_path), 'top.cir', load_pdk('sky130'))


def _deck(mod):
    mod.generate_netlist()
    return Path(mod._run_files.netlist_path).read_text()


# --- placed child -----------------------------------------------------------

def test_assignment_after_add_module_reaches_the_placed_child(tmp_path):
    w = Variable('w', value=1e3)
    child = _block(tmp_path, w)
    top = _top(tmp_path)
    top.add_module(child, 'x1', a='n1', b='0')
    w.value = 5e3
    deck = _deck(top)
    assert 'Rr1 a b 5k' in deck, deck
    assert 'Rr1 a b 1k' not in deck


def test_expression_assigned_after_add_module_reaches_the_placed_child(tmp_path):
    w = Variable('w', value=1e3)
    child = _block(tmp_path, 2 * w)
    top = _top(tmp_path)
    top.add_module(child, 'x1', a='n1', b='0')
    w.value = 5e3
    deck = _deck(top)
    assert 'Rr1 a b 10k' in deck, deck
    assert 'Rr1 a b 2k' not in deck


# --- embedded DUT -----------------------------------------------------------

def _embedded_deck(tmp_path, value, var):
    dut = _block(tmp_path, value, name='rc')
    tb = dut.create_test('t')
    tb.add_module_under_test({'a': 'inp', 'b': 'out'})
    var.value = 5e3
    return _deck(tb.netlist)


def test_assignment_after_embedding_reaches_the_dut(tmp_path):
    w = Variable('w', value=1e3)
    deck = _embedded_deck(tmp_path, w, w)
    assert 'Rr1 a b 5k' in deck, deck
    assert 'Rr1 a b 1k' not in deck


def test_expression_assigned_after_embedding_reaches_the_dut(tmp_path):
    w = Variable('w', value=1e3)
    deck = _embedded_deck(tmp_path, w * 2, w)
    assert 'Rr1 a b 10k' in deck, deck
    assert 'Rr1 a b 2k' not in deck


# --- copy identity ----------------------------------------------------------

def test_copy_and_deepcopy_return_the_same_variable():
    w = Variable('w', value=1.0)
    assert copy.copy(w) is w
    assert copy.deepcopy(w) is w


def test_deepcopied_expression_keeps_its_leaves_live():
    w = Variable('w', value=1.0)
    expr = copy.deepcopy(3 * w + 1)
    assert isinstance(expr, _VarExpr)
    w.value = 2.0
    assert expr.value == 7.0


def _resistor_value(mod, name='r1'):
    return mod._element_mgr.elem_dict[name]['device'].params['value']


def test_deepcopied_module_shares_its_variables(tmp_path):
    w = Variable('w', value=1e3)
    mod = _block(tmp_path, w)
    clone = copy.deepcopy(mod)
    assert clone is not mod
    assert _resistor_value(clone) is w


def test_deepcopied_testbench_shares_the_dut_variables(tmp_path):
    w = Variable('w', value=1e3)
    dut = _block(tmp_path, w, name='rc')
    tb = dut.create_test('t')
    tb.add_module_under_test({'a': 'inp', 'b': 'out'})
    clone = copy.deepcopy(tb)
    assert clone is not tb
    w.value = 5e3
    deck = _deck(clone.netlist)
    assert 'Rr1 a b 5k' in deck, deck


# --- pickling still snapshots (parallel optimizer path) ---------------------

def test_pickle_round_trip_copies_the_variable():
    w = Variable('w', value=1.0, bounds=(0.0, 10.0))
    back = pickle.loads(pickle.dumps(w))
    assert back is not w
    assert (back.name, back.value, back.bounds) == ('w', 1.0, (0.0, 10.0))
    w.value = 2.0
    assert back.value == 1.0


def test_pickle_round_trip_of_a_module_freezes_its_values(tmp_path):
    w = Variable('w', value=1e3)
    mod = _block(tmp_path, 2 * w)
    back = pickle.loads(pickle.dumps(mod))
    w.value = 5e3
    assert _resistor_value(back).value == 2e3
    assert _resistor_value(mod).value == 1e4
