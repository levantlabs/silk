"""A passive's typed alias and its .value must never diverge.

`Resistor.resistance` / `Capacitor.capacitance` / `Inductor.inductance` used to
be a SECOND copy of the value: `to_spice()` read the alias while the structure
hash read `.value`, so mutating the alias changed the simulated circuit but not
the hash — silently defeating the net-current staleness guard (an edited
circuit would load currents measured on the old one). They are now one value.
"""
import pytest

from silk.elements.passives import Resistor, Capacitor, Inductor

_CASES = [
    (Resistor, 'resistance', 560.0, 1200.0, 'R'),
    (Capacitor, 'capacitance', 1e-12, 4e-12, 'C'),
    (Inductor, 'inductance', 1e-9, 5e-9, 'L'),
]


@pytest.mark.parametrize('cls,alias,initial,edited,prefix', _CASES)
def test_alias_and_value_agree_at_construction(cls, alias, initial, edited, prefix):
    p = cls('p1', pos='a', neg='b', **{alias: initial})
    assert getattr(p, alias) == initial
    assert p.value == initial


@pytest.mark.parametrize('cls,alias,initial,edited,prefix', _CASES)
def test_writing_alias_updates_value(cls, alias, initial, edited, prefix):
    """The hole: `r.resistance = X` left `.value` (what the hash reads) stale."""
    p = cls('p1', pos='a', neg='b', **{alias: initial})
    setattr(p, alias, edited)
    assert p.value == edited


@pytest.mark.parametrize('cls,alias,initial,edited,prefix', _CASES)
def test_writing_value_updates_alias(cls, alias, initial, edited, prefix):
    p = cls('p1', pos='a', neg='b', **{alias: initial})
    p.value = edited
    assert getattr(p, alias) == edited


@pytest.mark.parametrize('cls,alias,initial,edited,prefix', _CASES)
def test_netlist_follows_the_edit(cls, alias, initial, edited, prefix):
    p = cls('p1', pos='a', neg='b', **{alias: initial})
    setattr(p, alias, edited)
    assert p.to_spice().startswith(f'{prefix}p1 a b ')
    assert '560' not in p.to_spice()      # not the pre-edit value


def test_edited_passive_changes_the_structure_hash(pdk, tmp_dir):
    """End-to-end: editing a resistor in place must move the structure hash,
    so net-current persistence refuses to serve currents for the old circuit."""
    from silk.circuit.modules import Module
    from silk.circuit.circuit_elements import Port
    from silk.db.structure_tracker import CircuitStructureTracker
    from silk.db.circuit_db import Database

    def _hash(mod):
        t = CircuitStructureTracker(Database(':memory:'), None)
        return t.compute_structure_hash(t.extract_structure(mod))[0]

    def _blk():
        m = Module('PDK_skywater130', tmp_dir, 'blk.cir',
                   pdk_details=pdk, module_name='blk')
        m.define_module_ports({'A': Port.analog_input(), 'B': Port.analog_output()})
        m.add_passive(Resistor('r1', pos='A', neg='B', resistance=560.0))
        return m

    before = _blk()
    h_before = _hash(before)

    after = _blk()
    after.passives_dict['r1'].resistance = 1200.0     # edit in place
    assert _hash(after) != h_before
