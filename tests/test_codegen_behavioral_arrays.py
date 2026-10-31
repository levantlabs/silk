"""TASK-502: arrays of behavioral (B) sources in ``sources[]``.

Same shape as the controlled-source arrays (TASK-466): ``array`` / ``naming``
and the ``$i`` / ``$(i±N)`` index templates in positive, negative and the
expression. silk does not parse expressions; only the templates are rewritten.
Emitted-name references (``i()``) are built with ``emitted_instance_name`` so
the tests hold whatever spelling the emitter uses.
"""
import ast
from pathlib import Path

import pytest

from silk import sources
from silk.circuit.emitted_names import emitted_instance_name
from silk.circuit.modules import Module
from silk.codegen.generator import CircuitSpec, Generator
from silk.codegen.testbench_generator import generate_testbenches
from silk.exceptions import DuplicateNameError
from tests._env import HAVE_XYCE, needs_ngspice

_TYPES = [('BehavioralVoltage', 'B'), ('BehavioralCurrent', 'B')]


def _entry(type_name='BehavioralVoltage', **kw):
    base = dict(type=type_name, instance='bx', positive='out[$i]',
                negative='ref[$(i+2)]', expression='v(in[$(i+1)]) + {gain}*$i')
    base.update(kw)
    return base


def _elements_from(code, ckt):
    """Execute the emitted ``add`` statements of generated circuit code."""
    tree = ast.parse(code)
    adds = [n for n in ast.walk(tree)
            if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Attribute)
            and n.value.func.attr == 'add']
    exec(compile(ast.Module(body=adds, type_ignores=[]), '<generated>', 'exec'),
         dict(ckt=ckt, mod=ckt, sources=sources))


def _cards(tmp_path, context, entries):
    """Return the emitted B cards for *entries* in the given context."""
    mod = Module('arrays', str(tmp_path), 'arrays.cir', pdk_details=None)
    if context in ('circuit', 'module'):
        spec = dict(circuit_name='arrays', sources=entries)
        if context == 'module':
            spec['ports'] = {'a': {'dir': 'input'}}
        _elements_from(Generator().generate(spec), mod)
        netlist = mod
    else:
        mod.module_name = 'dut'
        mod.define_module_ports({'a': 'input'})
        test = dict(name='t', dut_portmap={'a': 'a'})
        spec = dict(circuit_name='arrays', tests=[test])
        if context == 'rig':
            spec['rigs'] = [dict(name='r', sources=entries)]
            test['applied_rigs'] = ['r']
        else:
            test['sources'] = entries
        ns = {}
        exec(generate_testbenches(spec), ns)
        ns['_attach_t'](mod)
        netlist = mod.get_test('t').netlist
    deck_path = netlist.generate_netlist()
    deck = Path(deck_path).read_text()
    return [ln for ln in deck.splitlines() if ln.startswith('B')]


@pytest.mark.parametrize('type_name,_p', _TYPES)
@pytest.mark.parametrize('array,indices', [(2, [0, 1]), ([2, 4], [2, 3])])
@pytest.mark.parametrize('naming', ['bracketed', 'underscore'])
@pytest.mark.parametrize('context', ['circuit', 'module', 'test', 'rig'])
def test_behavioral_array_emits_per_index_cards(tmp_path, type_name, _p, array,
                                                indices, naming, context):
    cards = _cards(tmp_path, context,
                   [_entry(type_name, array=array, naming=naming)])
    kind = 'V' if type_name == 'BehavioralVoltage' else 'I'
    assert len(cards) == len(indices)
    for card, i in zip(cards, indices):
        name = f'bx[{i}]' if naming == 'bracketed' else f'bx_{i}'
        # the instance token comes from the one naming rule (TASK-520)
        element = getattr(sources, type_name)(name, positive='p', negative='n',
                                              expression='0')
        token = emitted_instance_name(element, name)
        assert card.startswith(f'{token} out[{i}] ref[{i+2}] {kind}=')
        # templates rewritten in the expression, braces survive as written
        assert f'v(in[{i+1}]) + {{gain}}*{i}' in card


@pytest.mark.parametrize('type_name,_p', _TYPES)
def test_expression_without_template_is_identical_per_element(tmp_path, type_name, _p):
    cards = _cards(tmp_path, 'circuit',
                   [_entry(type_name, expression='2*v(in)+{k}', array=3)])
    assert len(cards) == 3
    assert all(c.endswith('=2*v(in)+{k}') or '2*v(in)+{k}' in c for c in cards)


@pytest.mark.parametrize('type_name,_p', _TYPES)
def test_scalar_output_unchanged(type_name, _p):
    code = Generator().generate(dict(circuit_name='tb', sources=[dict(
        type=type_name, instance='bx', positive='o', negative='0',
        expression='v(in)*{k}')]))
    assert (f"ckt.add(sources.{type_name}('bx', positive='o', negative='0', "
            f"expression='v(in)*{{k}}'))") in code


def test_scalar_with_template_text_refuses_literal_pins(tmp_path):
    with pytest.raises(ValueError, match='ASCII'):
        _cards(tmp_path, 'circuit', [dict(
            type='BehavioralVoltage', instance='bx', positive='o$i', negative='0',
            expression='1')])


@pytest.mark.parametrize('bad', [0, -1, [3, 2]])
def test_invalid_array_refused_for_behavioral(bad):
    with pytest.raises(ValueError):
        CircuitSpec.from_dict(dict(sources=[_entry(array=bad)]))


def test_behavioral_array_collision_with_expanded_name_refused(tmp_path):
    # naming='underscore' expands bx to bx_0, bx_1; the scalar bx_1 is a
    # valid identifier, so it passes binding validation and reaches the
    # duplicate-name check.
    entries = [_entry(array=2, naming='underscore'),
               dict(type='BehavioralVoltage', instance='bx_1',
                    positive='z', negative='0', expression='1')]
    with pytest.raises(DuplicateNameError, match='bx_1'):
        _cards(tmp_path, 'circuit', entries)


def test_source_catalog_unchanged_by_behavioral_arrays():
    """The catalog has no array-support field (TASK-466): none is added."""
    from silk.sources.schema import export_source_schema
    for name in ('BehavioralVoltage', 'BehavioralCurrent'):
        entry = export_source_schema()[name]
        assert 'array' not in entry and 'naming' not in entry


# ---------------------------------------------------------------------------
# End to end: a bracketed B array with per-index v() and i() references.
# ---------------------------------------------------------------------------

def _i_ref_template():
    """i() takes the emitted, V-prefixed name; build it for a placeholder
    index and turn the placeholder back into the $i template."""
    probe = sources.VoltageSource('vx[7]', positive='m', negative='0', dc=0)
    return emitted_instance_name(probe, 'vx[7]').replace('7', '$i')


def _run_array(tmp_path, simulator):
    from silk import PortDir
    from silk.elements import elements
    from silk.pdk import load_pdk
    from silk.simulation import Op

    spec = dict(circuit_name='arr', sources=[
        dict(type='VoltageSource', instance='vin', positive='in[$i]',
             negative='0', dc=1.0, array=3),
        dict(type='VoltageSource', instance='vx', positive='m[$i]',
             negative='0', dc=0.0, array=3),
        dict(type='CurrentSource', instance='ib', positive='0',
             negative='m[$i]', dc=1e-3, array=3),
        dict(type='BehavioralVoltage', instance='bx', positive='o[$i]',
             negative='0', array=3,
             expression=f'2*v(in[$i]) + 1000*i({_i_ref_template()}) + $i'),
    ])
    code = Generator().generate(spec)
    items = []

    class Collector:
        def add(self, *a, **kw):
            for x in a:
                items.extend(x if isinstance(x, list) else [x])
    _elements_from(code, Collector())
    dut = Module('sky130', str(tmp_path), 'arr.cir',
                 pdk_details=load_pdk('sky130'), module_name='arr')
    dut.define_module_ports({'a': PortDir.IN})
    dut.add(elements.resistor('ra', p='a', n='0', value=1e3))
    dut.simulator = simulator
    tb = dut.create_test('t')
    tb.add_module_under_test({'a': 'a'})
    tb.add(*items)
    for i in range(3):
        tb.add(elements.resistor(f'rl{i}', p=f'o[{i}]', n='0', value=1e3))
    tb.add_sim_type(Op())
    sigs = [tb.probe(f'o[{i}]') for i in range(3)]
    # the B source's own branch current, probed by DESIGNER name
    currents = [tb.probe(f'i(bx[{i}])') for i in range(3)]
    tb.run()
    return ([float(s.y[0]) for s in sigs], [float(c.y[0]) for c in currents])


@pytest.mark.slow
@needs_ngspice
def test_behavioral_array_simulates_on_ngspice(tmp_path):
    volts, amps = _run_array(tmp_path, 'ngspice')
    assert volts == pytest.approx([3.0, 4.0, 5.0], abs=1e-6)
    # B V= sources into 1k: branch current flows out of + (negative sign)
    assert amps == pytest.approx([-3e-3, -4e-3, -5e-3], abs=1e-9)


@pytest.mark.slow
@pytest.mark.requires_xyce
@pytest.mark.skipif(not HAVE_XYCE, reason='Xyce setting unavailable')
def test_behavioral_array_simulates_on_xyce(tmp_path):
    volts, amps = _run_array(tmp_path, 'xyce')
    assert volts == pytest.approx([3.0, 4.0, 5.0], abs=1e-6)
    assert [abs(a) for a in amps] == pytest.approx([3e-3, 4e-3, 5e-3], abs=1e-9)


@pytest.mark.parametrize('naming', ['bracketed', 'underscore'])
def test_branch_vector_name_resolves_arrayed_b_source_by_designer_name(naming):
    """i(<designer name>) maps to the emitted instance through the one rule."""
    from silk.circuit.module_test import _branch_vector_name
    name = 'bx[1]' if naming == 'bracketed' else 'bx_1'
    src = sources.BehavioralVoltage(name, positive='o', negative='0', expression='1')
    other = sources.BehavioralVoltage('bx[0]', positive='p', negative='0', expression='1')
    got = _branch_vector_name({name: src, 'bx[0]': other}, name)
    assert got == emitted_instance_name(src, name)
    # raw emitted spelling still passes through verbatim
    assert _branch_vector_name({name: src}, 'raw_vec') == 'raw_vec'
