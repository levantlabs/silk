"""TASK-175 — ``add_element`` refuses a child that cannot be a subcircuit, at
the call rather than at the parent's much-later ``generate_netlist()``.

TASK-167 made ``add_module_under_test`` refuse a module_name-less Module
where the mistake is made. ``Module.add_element`` — the general composition
API — was left accepting the same mistake, deferring the refusal to deck
emission (:class:`~silk.exceptions.NotASubcircuitError` naming the child,
since both emission guards already route through ``require_subcircuit``). A
second, worse shape was also reachable through ``add_element``: a child with
no declared ports at all (``pin_list`` is ``None``) used to reach
``connect_element``'s ``pinlist.keys()`` and raise a bare ``AttributeError``
naming neither the module nor the mistake, whenever port wiring was passed.

These tests pin the early refusal for both shapes on ``add_element``, confirm
ordinary named-subcircuit composition is unaffected, and document why the
same early refusal was NOT added to ``Module.add_submodule`` (AC #5) —  it
conflicts with an existing, deliberate use of that method as a purely
organizational registration for the recursive test-walking tree; see
``test_add_submodule_still_accepts_a_module_name_less_child`` below and the
TASK-175 task notes.
"""
import pytest

from silk import Module, load_pdk
from silk.elements.passives import Resistor
from silk.exceptions import NotASubcircuitError


def top_level(tmp_path, filename='owner.cir', ports=None):
    """A Module with no ``module_name`` — a top-level circuit by definition."""
    mod = Module('sky130', str(tmp_path), filename, load_pdk('sky130'))
    if ports:
        mod.define_module_ports(ports)
    return mod


def portless_subcircuit(tmp_path, name, filename):
    """A Module with a ``module_name`` but no declared ports at all —
    ``pin_list`` is still ``None`` (``define_module_ports`` was never
    called)."""
    return Module('sky130', str(tmp_path), filename, load_pdk('sky130'),
                   module_name=name)


def subcircuit(tmp_path, name='block', filename='block.cir'):
    """A properly composable child: named, with declared ports."""
    mod = Module('sky130', str(tmp_path), filename, load_pdk('sky130'),
                 module_name=name)
    mod.define_module_ports({'out': 'out', 'vss': 'ground'})
    mod.add_passive(Resistor('rload', pos='out', neg='vss', resistance=1000))
    return mod


# --- add_element: module_name-less child ------------------------------------

def test_add_element_refuses_a_module_name_less_child(tmp_path):
    parent = subcircuit(tmp_path / 'p', name='parent', filename='parent.cir')
    child = top_level(tmp_path / 'c', filename='child.cir',
                      ports={'out': 'out', 'vss': 'ground'})
    try:
        with pytest.raises(NotASubcircuitError) as excinfo:
            parent.add_element(child, 'u1', nets={'out': 'out', 'vss': 'vss'})
        message = str(excinfo.value)
        assert 'u1' in message               # names the instance
        assert 'child.cir' in message        # names the module
        assert 'module_name' in message      # names the mistake
        assert 'subcircuit' in message       # names the rule
        assert 'define_module_name' in message  # names the fix
    finally:
        parent.close()
        child.close()


def test_add_element_refuses_before_registering_anything(tmp_path):
    """A refused add leaves nothing behind: no dangling elem_dict entry, no
    stray module_dict registration (the dedup chokepoint keyed by
    ``module_name`` would otherwise see two ``None``-keyed children as one
    misdiagnosed collision)."""
    parent = subcircuit(tmp_path / 'p', name='parent', filename='parent.cir')
    child1 = top_level(tmp_path / 'c1', filename='c1.cir',
                       ports={'out': 'out', 'vss': 'ground'})
    child2 = top_level(tmp_path / 'c2', filename='c2.cir',
                       ports={'out': 'out', 'vss': 'ground'})
    try:
        with pytest.raises(NotASubcircuitError):
            parent.add_element(child1, 'u1', nets={'out': 'out', 'vss': 'vss'})
        # A second, structurally distinct module_name-less child used to
        # collide on the shared `None` key with a misleading "claimed by two
        # structurally different circuits" CircuitError. It must still get
        # the same clear early refusal, not that misdiagnosis.
        with pytest.raises(NotASubcircuitError) as excinfo:
            parent.add_element(child2, 'u2', nets={'out': 'out', 'vss': 'vss'})
        assert 'claimed by two structurally different' not in str(excinfo.value)
        assert 'u1' not in parent._element_mgr.elem_dict
        assert 'u2' not in parent._element_mgr.elem_dict
    finally:
        parent.close()
        child1.close()
        child2.close()


# --- add_element: no declared ports (pin_list is None) ----------------------

def test_add_element_refuses_a_child_with_no_declared_ports(tmp_path):
    """This used to be a bare ``AttributeError: 'NoneType' object has no
    attribute 'keys'`` from deep inside ``connect_element`` once wiring was
    passed. It must now be the same named, actionable refusal."""
    parent = subcircuit(tmp_path / 'p', name='parent', filename='parent.cir')
    child = portless_subcircuit(tmp_path / 'c', name='childmod',
                                filename='child.cir')
    try:
        with pytest.raises(NotASubcircuitError) as excinfo:
            parent.add_element(child, 'u1', out='out')
        message = str(excinfo.value)
        assert 'u1' in message
        assert 'childmod' in message
        assert 'declared ports' in message
        assert 'define_module_ports' in message
    finally:
        parent.close()
        child.close()


def test_add_element_accepts_a_named_portless_child_with_no_wiring(tmp_path):
    """A named module with no declared ports at all and nothing wired to it
    (e.g. a behavioral-model subcircuit with no pins) is a legitimate shape,
    not the bug: the old ``AttributeError`` only ever fired once wiring was
    passed, since nothing else touches ``pin_list``. Pinned alongside
    ``tests/specs/test_testbench_integration.py::test_composition_copies_protected_specs_and_runtime_relationships``,
    which relies on this via ``nets={}``."""
    parent = subcircuit(tmp_path / 'p', name='parent', filename='parent.cir')
    child = portless_subcircuit(tmp_path / 'c', name='childmod',
                                filename='child.cir')
    try:
        inst = parent.add_element(child, 'u1')
        assert inst is not None
    finally:
        parent.close()
        child.close()


def test_add_element_refuses_a_module_name_less_portless_child(tmp_path):
    """Both shapes at once: the module_name refusal fires (its message
    already tells the caller to declare ports too)."""
    parent = subcircuit(tmp_path / 'p', name='parent', filename='parent.cir')
    child = top_level(tmp_path / 'c', filename='child.cir')
    try:
        with pytest.raises(NotASubcircuitError) as excinfo:
            parent.add_element(child, 'u1')
        message = str(excinfo.value)
        assert 'module_name' in message
        assert 'define_module_ports' in message
    finally:
        parent.close()
        child.close()


# --- add_submodule: AC #5 blocked, current behaviour pinned -----------------

def test_add_submodule_still_accepts_a_module_name_less_child(tmp_path):
    """AC #5 asked for the same early refusal here. Implementing it broke a
    real, deliberate use documented in
    ``tests/test_temperature_inheritance.py::test_recursive_runs_apply_the_nearest_setting``:
    ``add_submodule`` also serves as a purely organizational parent/child
    registration for the recursive test-walking tree
    (``run_all_tests(recursive=True)``), where the registered module owns no
    ports and its parent's ``generate_netlist()`` is never actually called —
    so the late failure AC #5's premise describes never happens for this
    use. This is left unresolved for the owner (see TASK-175 notes); pinning
    today's unchanged behaviour so a future change here is deliberate."""
    parent = subcircuit(tmp_path / 'p', name='parent', filename='parent.cir')
    child = top_level(tmp_path / 'c', filename='child.cir',
                      ports={'out': 'out', 'vss': 'ground'})
    try:
        parent.add_submodule('sub', child)
        assert parent._element_mgr.module_dict['sub']['device'] is child
    finally:
        parent.close()
        child.close()


def test_add_submodule_still_accepts_a_named_portless_child(tmp_path):
    """Unlike ``add_element``, ``add_submodule`` never wires ports at add
    time, so a named module with no declared ports yet is not refused here —
    ``tests/test_pymodule.py::TestAddSubmodule`` already relies on exactly
    this to build a submodule before its ports are declared."""
    parent = subcircuit(tmp_path / 'p', name='parent', filename='parent.cir')
    child = portless_subcircuit(tmp_path / 'c', name='childmod',
                                filename='child.cir')
    try:
        parent.add_submodule('sub', child)
        assert parent._element_mgr.module_dict['sub']['device'] is child
    finally:
        parent.close()
        child.close()


def test_add_submodule_still_rejects_self_addition_first(tmp_path):
    """Cycle detection is unrelated to subcircuit-ness and must still fire
    for a self-add regardless of module_name/ports."""
    parent = subcircuit(tmp_path / 'p', name='parent', filename='parent.cir')
    try:
        with pytest.raises(ValueError, match='circular reference'):
            parent.add_submodule('self', parent)
    finally:
        parent.close()


# --- existing composition of named subcircuits is unaffected (AC #4) -------

def test_add_element_still_composes_a_properly_named_subcircuit(tmp_path):
    parent = subcircuit(tmp_path / 'p', name='parent', filename='parent.cir')
    child = subcircuit(tmp_path / 'c', name='child', filename='child.cir')
    try:
        inst = parent.add_element(child, 'u1', out='out', vss='vss')
        assert inst is not None
        subckt_text = parent.generate_netlist(subckt=True)
        assert '.subckt parent' in subckt_text
        assert 'xu1' in subckt_text
        assert 'child' in subckt_text
    finally:
        parent.close()
        child.close()


def test_add_submodule_still_registers_a_properly_named_subcircuit(tmp_path):
    parent = subcircuit(tmp_path / 'p', name='parent', filename='parent.cir')
    child = subcircuit(tmp_path / 'c', name='child', filename='child.cir')
    try:
        parent.add_submodule('sub', child)
        assert parent._element_mgr.module_dict['sub']['device'] is child
    finally:
        parent.close()
        child.close()
