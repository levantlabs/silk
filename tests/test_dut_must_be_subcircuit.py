"""TASK-167 — a Module with no ``module_name`` is refused, and says why.

``module_name`` is what makes a Module emittable as a subcircuit; a Module
without one is a top-level circuit and SPICE has nothing to instantiate. Using
one as a device under test was accepted silently by
``add_module_under_test`` and then failed at ``generate_netlist()`` with a bare
``TypeError`` — ``unsupported operand type(s) for +: 'NoneType' and 'str'``
from the netlister's ``module_name`` concatenation, or ``'NoneType' object is
not iterable`` when no ports were declared either. Neither names the module and
neither says what the mistake was, and both are raised far from the call that
made it.

Note what was NOT wrong: ``str + None`` raises in Python, it never yields the
text ``None``. No deck was ever written and no simulator ever saw a malformed
instance line, so there is no such emission to assert against here.

What these tests pin is the refusal: :class:`~silk.exceptions.NotASubcircuitError`
at ``add_module_under_test`` (where the mistake is made) and at the two direct
emission paths that could still be reached another way, plus the marker that
carries it out of ``run_all_tests`` instead of into the results dictionary.

TASK-175 closed the sibling gap on ``add_element``: a module_name-less or
portless child used to be accepted and the refusal deferred to the parent's
``generate_netlist()``. ``add_element`` now refuses at the call, matching
``add_module_under_test``'s wording — see
``tests/test_add_element_child_must_be_subcircuit.py``, which also documents
why the same early refusal was not extended to ``add_submodule`` (AC #5).
"""
import os
from pathlib import Path

import pytest

from silk import Module, load_pdk, sources
from silk.elements import elements
from silk.exceptions import (
    CircuitError, ConfigurationError, NotASubcircuitError,
)
from silk.simulation import Transient


def top_level(tmp_path, filename='owner.cir', ports=None):
    """A Module with no ``module_name`` — a top-level circuit by definition."""
    mod = Module('sky130', str(tmp_path), filename, load_pdk('sky130'))
    if ports:
        mod.define_module_ports(ports)
    mod.add(elements.resistor('rload', p='out', n='0', value=1000))
    return mod


def subcircuit(tmp_path, name='block', filename='block.cir'):
    """The same circuit made emittable: it has a ``module_name``."""
    mod = Module('sky130', str(tmp_path), filename, load_pdk('sky130'),
                 module_name=name)
    mod.define_module_ports({'out': 'out', 'vss': 'ground'})
    mod.add(elements.resistor('rload', p='out', n='vss', value=1000))
    return mod


# --- the refusal happens where the mistake is made -------------------------

@pytest.mark.parametrize('ports', [
    pytest.param(None, id='no-ports-declared'),
    pytest.param({'out': 'out', 'vss': 'ground'}, id='ports-declared'),
])
def test_top_level_module_is_refused_as_a_device_under_test(tmp_path, ports):
    """Both shapes used to be accepted here and fail much later, with two
    different builtin ``TypeError``s depending on whether ports existed."""
    owner = top_level(tmp_path, ports=ports)
    try:
        tb = owner.create_test('level')
        with pytest.raises(NotASubcircuitError) as excinfo:
            tb.add_module_under_test({'out': 'vout', 'vss': '0'})
        message = str(excinfo.value)
        assert 'owner.cir' in message, message          # names the module
        assert 'module_name' in message                  # names the mistake
        assert 'subcircuit' in message                   # names the rule
        assert 'define_module_name' in message           # names the fix
    finally:
        owner.close()


def test_a_refused_dut_leaves_the_testbench_untouched(tmp_path):
    """Refusing before ``_create_netlist`` means nothing half-built is left:
    the testbench can still be wired up correctly afterwards."""
    owner = top_level(tmp_path)
    try:
        tb = owner.create_test('level')
        with pytest.raises(NotASubcircuitError):
            tb.add_module_under_test({'out': 'vout'})
        assert tb._module_added is False
        assert tb.netlist is None
    finally:
        owner.close()


def test_the_refusal_is_a_configuration_error(tmp_path):
    """Classified at the exception's definition (TASK-163), so a wrong device
    under test propagates out of ``run_all_tests`` rather than being recorded
    as one test's error string. Still a ``CircuitError``, so callers that
    already catch silk's base type are unaffected."""
    assert issubclass(NotASubcircuitError, ConfigurationError)
    assert issubclass(NotASubcircuitError, CircuitError)


# --- direct emission ------------------------------------------------------

def test_instance_line_emission_refuses_and_names_the_module(tmp_path):
    """``generate_netlist_entry`` is the line that used to raise the bare
    ``TypeError``. Reached directly it now names both instance and module."""
    mod = top_level(tmp_path, filename='flat.cir',
                    ports={'out': 'out', 'vss': 'ground'})
    try:
        with pytest.raises(NotASubcircuitError) as excinfo:
            mod.generate_netlist_entry('u1')
        message = str(excinfo.value)
        assert "'u1'" in message
        assert 'flat.cir' in message
        assert 'module_name' in message
    finally:
        mod.close()


def test_subckt_definition_emission_refuses(tmp_path):
    """The other half of the same concatenation: ``.subckt`` + ``None``."""
    mod = top_level(tmp_path, filename='flat.cir',
                    ports={'out': 'out', 'vss': 'ground'})
    try:
        with pytest.raises(NotASubcircuitError) as excinfo:
            mod.generate_netlist(subckt=True)
        assert '.subckt' in str(excinfo.value)
        assert 'flat.cir' in str(excinfo.value)
    finally:
        mod.close()


def test_a_whole_deck_containing_one_refuses_by_name(tmp_path):
    """``add_element`` used to accept a top-level Module as a child and defer
    the refusal to the parent's ``generate_netlist()``. It now refuses at the
    ``add_element`` call itself, symmetric with ``add_module_under_test``
    (TASK-175) — the mistake is made here, so it is reported here."""
    parent = subcircuit(tmp_path / 'p', name='parent', filename='parent.cir')
    child = top_level(tmp_path / 'c', filename='child.cir',
                      ports={'out': 'out', 'vss': 'ground'})
    try:
        with pytest.raises(NotASubcircuitError) as excinfo:
            parent.add_module(child, 'u1', nets={'out': 'out', 'vss': 'vss'})
        assert 'child.cir' in str(excinfo.value)
        assert 'module_name' in str(excinfo.value)
        assert 'u1' in str(excinfo.value)
    finally:
        parent.close()
        child.close()


def test_add_element_refusal_means_run_all_tests_never_sees_a_broken_dut(
        tmp_path, monkeypatch):
    """This used to be a deck that could not be emitted, discovered only once
    ``run_all_tests`` reached ``generate_netlist()`` deep inside a run — the
    marker had to carry the error out of the runner rather than leave it
    folded into ``{'error': ...}``. It can no longer be built at all
    (TASK-175): ``add_element`` refuses the module_name-less child
    immediately, so the dut is never assembled and ``run_all_tests`` is never
    reached. Generic ``ConfigurationError`` propagation through
    ``run_all_tests`` remains covered by
    ``tests/test_run_all_tests_error_classification.py``.
    """
    dut = subcircuit(tmp_path / 'd', name='dut', filename='dut.cir')
    child = top_level(tmp_path / 'c', filename='child.cir',
                      ports={'out': 'out', 'vss': 'ground'})
    monkeypatch.setattr(dut.pdk_details, 'library_location', os.devnull)
    try:
        with pytest.raises(NotASubcircuitError):
            dut.add_module(child, 'u1', nets={'out': 'out', 'vss': 'vss'})
        tb = dut.create_test('level')
        assert tb._module_added is False
    finally:
        dut.close()
        child.close()


# --- subcircuits are untouched (AC #3) -------------------------------------

def test_a_named_subcircuit_still_emits_its_instance_line(tmp_path):
    mod = subcircuit(tmp_path)
    try:
        mod.pin_list['out'].net = 'a'
        mod.pin_list['vss'].net = '0'
        assert mod.generate_netlist_entry('u1') == 'xu1 a 0 block\n'
        assert mod.generate_subckt(0) == '.subckt block out vss \n'
    finally:
        mod.close()


def test_a_named_subcircuit_still_wires_up_as_a_device_under_test(tmp_path):
    """The ordinary path, unchanged: the DUT is instantiated and the emitted
    testbench deck carries its ``.subckt`` definition and its ``x`` line."""
    mod = subcircuit(tmp_path)
    try:
        tb = mod.create_test('level')
        tb.add_module_under_test({'out': 'vout', 'vss': '0'})
        tb.add_source(sources.VoltageSource('supply', positive='vout',
                                            negative='0', dc=1.8))
        tb.add_sim_type(Transient(step=1e-6, stop=2e-6))
        assert tb._module_added is True
        tb.netlist.generate_netlist()
        deck = Path(tb.netlist._run_files.netlist_path).read_text()
        assert '.subckt block out vss' in deck
        assert 'xdut vout 0 block' in deck
    finally:
        mod.close()
