"""cml_cells builds must be distinguishable by every design knob (TASK-524).

Two builds that differ in any knob must be able to share one deck. Before
TASK-524 the cml builders named their subcircuit by ``imul`` alone (or not at
all), so two builds that differed only in ``rl`` raised ``CircuitError`` at
netlist generation. The builders now use ``variants=True``, which suffixes the
structure hash so each distinct body gets its own ``.subckt`` name.
"""
import pytest

from silk.circuit.modules import Module
from silk.examples import cml_cells as cml


def _ports(module):
    return list(module.pin_list)


def _instance(module, tag):
    """Wire every port of *module*; the supplies are shared, signals per tag."""
    nets = {}
    for port in _ports(module):
        if port == 'VDD':
            nets[port] = 'vdd'
        elif port == 'VSS':
            nets[port] = 'gnd'
        else:
            nets[port] = f'{tag}_{port.lower()}'
    return nets


def _deck_subckts(tmp_path, builds):
    """Place each (tag, module) in one deck, generate it, return its .subckt names."""
    top = Module("PDK_skywater130", str(tmp_path), "deck.cir", pdk_details=cml.pdk)
    top.define_module_ports({})
    for tag, module in builds:
        top.add_module(module, f'x{tag}', nets=_instance(module, tag))
    from pathlib import Path
    text = Path(top.generate_netlist()).read_text()
    return [line.split()[1] for line in text.splitlines()
            if line.lower().startswith('.subckt')]


@pytest.fixture
def out_dir(tmp_path, monkeypatch):
    monkeypatch.setenv('SILK_EXAMPLES_OUT', str(tmp_path))
    return tmp_path


# (label, builder, knob-A kwargs, knob-B kwargs). Each pair differs in exactly
# one knob that changes the body. build_cml2cmos takes rl but never uses it, so
# it has no rl pair: two rl values there are the same circuit and should share a name.
KNOB_PAIRS = [
    ('bias-imul', cml.build_cml_bias, {'imul': 1.0}, {'imul': 2.0}),
    ('buffer-rl', cml.build_cml_buffer, {'rl': 2000.0}, {'rl': 1000.0}),
    ('latch-rl', cml.build_cml_latch, {'rl': 2000.0}, {'rl': 1000.0}),
    ('latch-imul', cml.build_cml_latch, {'imul': 1.0}, {'imul': 4.0}),
    ('dff-rl', cml.build_cml_dff, {'rl': 2000.0}, {'rl': 1000.0}),
    ('dff-imul', cml.build_cml_dff, {'imul': 1.0}, {'imul': 4.0}),
    ('mux2-rl', cml.build_cml_mux2, {'rl': 2000.0}, {'rl': 1000.0}),
    ('mux2-imul', cml.build_cml_mux2, {'imul': 1.0}, {'imul': 4.0}),
    ('cmos2cml-rl', cml.build_cmos2cml, {'rl': 2000.0}, {'rl': 1000.0}),
    ('cml2cmos-imul', cml.build_cml2cmos, {'imul': 1.0}, {'imul': 4.0}),
    ('predriver-rl', cml.build_cml_predriver, {'rl': 875.0}, {'rl': 400.0}),
    ('predriver-imul', cml.build_cml_predriver, {'imul': 2.0}, {'imul': 4.0}),
    ('driver-rl', cml.build_cml_driver, {'rl': 50.0}, {'rl': 25.0}),
]


@pytest.mark.parametrize('label, builder, kw_a, kw_b', KNOB_PAIRS,
                         ids=[pair[0] for pair in KNOB_PAIRS])
def test_builds_differing_in_one_knob_share_a_deck(out_dir, label, builder, kw_a, kw_b):
    first, second = builder(**kw_a), builder(**kw_b)
    subckts = _deck_subckts(out_dir, [('a', first), ('b', second)])
    # Count only the builds' own definitions: a dff also emits its child latch.
    own = [name for name in subckts if name.startswith(first.module_name)]
    assert len(set(own)) == 2, f'{label}: both builds emitted {own}'


@pytest.mark.parametrize('builder', [cml.build_cml_latch, cml.build_cml_dff],
                         ids=['latch', 'dff'])
def test_identical_rebuilds_share_one_definition(out_dir, builder):
    """A legitimate identical rebuild must still collapse to one .subckt."""
    base = builder(rl=2000.0).module_name
    subckts = _deck_subckts(out_dir, [('a', builder(rl=2000.0)),
                                      ('b', builder(rl=2000.0))])
    assert sum(name.startswith(base) for name in subckts) == 1
