"""TASK-597: add_custom_line on a child circuit is emitted inside its .subckt.

Simulator-free: only netlist text and a launch spy are used. Ideal elements
keep the decks PDK-model free.
"""
import re
import subprocess
from unittest.mock import Mock

import pytest

from silk import Module, load_pdk
from silk.elements import elements
from silk.exceptions import CircuitError, SimulationError

PORTS = {'a': 'in', 'b': 'out'}


def _child(tmp_path, name='blk', custom=(), *, variants=False, fname=None):
    m = Module('sky130', str(tmp_path), fname or f'{name}.cir', load_pdk('sky130'),
               module_name=name, variants=variants)
    m.define_module_ports(PORTS)
    m.add(elements.resistor('r1', p='a', n='b', value=2000))
    for line in custom:
        m.add_custom_line(line)
    return m


def _top(tmp_path, *children, name='top.cir'):
    top = Module('sky130', str(tmp_path), name, load_pdk('sky130'))
    for i, child in enumerate(children):
        top.add_module(child, f'x{i}', a='na', b=f'nb{i}')
    return top


def _deck(top):
    top.set_output_options(mode='overwrite')
    deck_path = top.generate_netlist()
    with open(deck_path) as f:
        return f.read()


def _body(deck, name):
    m = re.search(rf'(?is)^\.subckt {name}\b.*?^\.ends\b[^\n]*', deck, re.M)
    assert m, deck
    return m.group(0)


@pytest.mark.parametrize('simulator', ['ngspice', 'xyce'])
def test_child_custom_line_is_inside_owning_subckt(tmp_path, simulator):
    child = _child(tmp_path, custom=['Rcustom a b 1k'])
    top = _top(tmp_path, child)
    top.simulator = simulator
    body = _body(_deck(top), 'blk')
    lines = body.splitlines()
    assert any(l.lower().startswith('rr1 ') for l in lines)
    assert 'Rcustom a b 1k' in lines
    assert lines.index('Rcustom a b 1k') > max(
        i for i, l in enumerate(lines) if l.lower().startswith('rr1 '))
    assert lines[-1].lower().startswith('.ends')
    direct = child.generate_netlist()
    assert 'Rcustom a b 1k' in direct


def test_custom_only_child_and_authored_order_with_repeats(tmp_path):
    child = Module('sky130', str(tmp_path), 'c.cir', load_pdk('sky130'),
                   module_name='blk')
    child.define_module_ports(PORTS)
    authored = ['* first', 'Rx a b 1k', 'Rx2 a b 1k', '+ 1k', 'Rx a b 1k']
    for line in authored:
        child.add_custom_line(line)
    body = _body(_deck(_top(tmp_path, child)), 'blk').splitlines()
    got = [l for l in body if l in set(authored)]
    assert got == authored


def test_grandchild_mixed_levels_and_reuse_not_multiplied(tmp_path):
    grand = _child(tmp_path, 'grand', ['Rg a b 3k'])
    mid = Module('sky130', str(tmp_path), 'mid.cir', load_pdk('sky130'),
                 module_name='mid')
    mid.define_module_ports(PORTS)
    mid.add_module(grand, 'xg', a='a', b='b')
    mid.add_custom_line('Rm a b 5k')
    top = Module('sky130', str(tmp_path), 'top.cir', load_pdk('sky130'))
    top.add_module(mid, 'xm1', a='na', b='n1')
    top.add_module(mid, 'xm2', a='na', b='n2')
    top.add_custom_line('Rtop na 0 7k')
    deck = _deck(top)
    assert deck.count('Rg a b 3k') == 1
    assert deck.count('Rm a b 5k') == 1
    assert deck.count('Rtop na 0 7k') == 1
    assert 'Rg a b 3k' in _body(deck, 'grand')
    assert 'Rm a b 5k' in _body(deck, 'mid')
    assert 'Rtop' not in _body(deck, 'mid')
    assert _deck(top) == deck  # repeated emission is stable


def test_reserved_namespace_refused_in_child(tmp_path):
    child = _child(tmp_path, custom=['Rbad a __silk_x 1k'])
    with pytest.raises(CircuitError, match='__silk_'):
        _top(tmp_path, child).generate_netlist()


def test_same_name_different_custom_body_raises(tmp_path):
    a = _child(tmp_path, custom=['Rcustom a b 1k'], fname='a.cir')
    b = _child(tmp_path, custom=['Rcustom a b 9k'], fname='b.cir')
    with pytest.raises(CircuitError, match='blk'):
        _top(tmp_path, a, b).generate_netlist()


def test_identical_rebuilt_custom_definitions_dedup(tmp_path):
    a = _child(tmp_path, custom=['Rcustom a b 1k'], fname='a.cir')
    b = _child(tmp_path, custom=['Rcustom a b 1k'], fname='b.cir')
    deck = _deck(_top(tmp_path, a, b))
    assert deck.count('.subckt blk') == 1
    assert deck.count('Rcustom a b 1k') == 1


def test_variants_true_gives_distinct_names_for_distinct_custom(tmp_path):
    a = _child(tmp_path, custom=['Rcustom a b 1k'], variants=True, fname='a.cir')
    b = _child(tmp_path, custom=['Rcustom a b 9k'], variants=True, fname='b.cir')
    top = _top(tmp_path, a, b)
    names = list(top.module_dict)
    assert len(names) == 2 and all(re.fullmatch(r'blk_[0-9a-f]{8}', n) for n in names)
    deck = _deck(top)
    assert 'Rcustom a b 1k' in _body(deck, names[0])
    assert 'Rcustom a b 9k' in _body(deck, names[1])


def test_variants_true_without_custom_keeps_legacy_name(tmp_path):
    from silk.circuit.circuit_elements import _module_structure_hash
    a = _child(tmp_path, variants=True)
    top = _top(tmp_path, a)
    assert list(top.module_dict) == [f'blk_{_module_structure_hash(a)[:8]}']


def test_variants_true_identical_custom_dedups(tmp_path):
    a = _child(tmp_path, custom=['Rcustom a b 1k'], variants=True, fname='a.cir')
    b = _child(tmp_path, custom=['Rcustom a b 1k'], variants=True, fname='b.cir')
    assert len(_top(tmp_path, a, b).module_dict) == 1


def test_no_custom_child_and_top_level_emission_unchanged(tmp_path):
    child = _child(tmp_path)
    deck = _deck(_top(tmp_path, child))
    assert 'Custom circuit elements' not in _body(deck, 'blk')
    top = _top(tmp_path, _child(tmp_path, fname='c2.cir'), name='t2.cir')
    top.add_custom_line('Rtop na 0 1k')
    assert _deck(top).count('*** Custom circuit elements') == 1


def test_child_custom_change_after_generation_refuses_launch(
        tmp_path, monkeypatch, mock_ngspice_version):
    child = _child(tmp_path)
    top = _top(tmp_path, child)
    top.generate_netlist()
    launch = Mock(side_effect=AssertionError('a stale deck must not launch'))
    monkeypatch.setattr(subprocess, 'run', launch)
    child.add_custom_line('* added to the child after generation')
    with pytest.raises(SimulationError, match='Circuit configuration changed'):
        top.run_simulator()
    launch.assert_not_called()


# --- structural identity stays custom-blind (reviewer ruling) ----------------

def _struct_hash(module):
    from silk.circuit.circuit_elements import _module_structure_hash
    return _module_structure_hash(module)


def _variant_parent(tmp_path, tag, custom, *, wrapped=False):
    child = _child(tmp_path, custom=custom, variants=True, fname=f'{tag}_c.cir')
    top = Module('sky130', str(tmp_path), f'{tag}_p.cir', load_pdk('sky130'),
                 module_name='wrap' if wrapped else None, variants=wrapped)
    if wrapped:
        top.define_module_ports({'a': 'in', 'b': 'out'})
        top.add_module(child, 'x0', a='a', b='b')
    else:
        top.add_module(child, 'x0', a='na', b='nb')
    return child, top


def test_parent_digest_is_custom_blind_but_child_names_differ(tmp_path):
    c1, p1 = _variant_parent(tmp_path, 'k1', ['Rcustom a b 1k'])
    c9, p9 = _variant_parent(tmp_path, 'k9', ['Rcustom a b 9k'])
    assert list(p1.module_dict) != list(p9.module_dict)
    assert _struct_hash(c1) == _struct_hash(c9)
    assert _struct_hash(p1) == _struct_hash(p9)


def test_variant_wrapper_body_hash_is_custom_blind(tmp_path):
    _, w1 = _variant_parent(tmp_path, 'k1', ['Rcustom a b 1k'], wrapped=True)
    _, w9 = _variant_parent(tmp_path, 'k9', ['Rcustom a b 9k'], wrapped=True)
    g1 = Module('sky130', str(tmp_path), 'g1.cir', load_pdk('sky130'))
    g9 = Module('sky130', str(tmp_path), 'g9.cir', load_pdk('sky130'))
    g1.add_module(w1, 'xw', a='na', b='nb')
    g9.add_module(w9, 'xw', a='na', b='nb')
    (n1,), (n9,) = list(g1.module_dict), list(g9.module_dict)
    assert g1.module_dict[n1]['_body_hash'] == g9.module_dict[n9]['_body_hash']
    assert _struct_hash(g1) == _struct_hash(g9)
    assert _struct_hash(w1) == _struct_hash(w9)


def test_custom_bearing_digest_equals_custom_free_digest(tmp_path):
    """Custom text never contributes: same digest as the custom-free build."""
    _, with_custom = _variant_parent(tmp_path, 'a', ['Rcustom a b 1k'])
    _, without = _variant_parent(tmp_path, 'b', [])
    assert _struct_hash(with_custom) == _struct_hash(without)
