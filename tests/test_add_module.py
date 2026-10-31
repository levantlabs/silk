"""TASK-298.1 — ``Module.add_module``, the hierarchy-placement verb shared
with ``ModuleTestbench.add_module`` (module_test.py), and the deprecation of
``Module.add_element``/``Module.add_submodule`` as its aliases.

``add_module(module, name, nets=None, **ports)`` accepts only Modules; a
device raises ``TypeError`` pointing to :meth:`Module.add`. ``add_element``
keeps today's exact catch-all dispatch (device classes, device instances,
legacy PDK devices, Modules) completely unchanged — only a
``DeprecationWarning`` naming the right replacement (``add_module`` for a
Module target, ``add()`` for a device target) is new. ``add_submodule``
(registration-only: it never places an instance) gets the same treatment,
pointing to ``add_module``.

Both public entry points share one underlying implementation
(``Module._add_element_impl``), so a Module built via ``add_module`` emits a
byte-identical netlist and structure hash to the same Module built via
``add_element`` — pinned below rather than assumed.
"""
import warnings

import pytest

from silk.circuit.modules import Module
from silk.db.circuit_db import Database
from silk.db.structure_tracker import CircuitStructureTracker
from silk.elements import elements
from silk.exceptions import NotASubcircuitError


def _hash(mod):
    tracker = CircuitStructureTracker(Database(':memory:'), None)
    structure = tracker.extract_structure(mod)
    structure_hash, _ = tracker.compute_structure_hash(structure)
    return structure_hash


def _subcircuit(pdk, tmp_path, subdir, filename, name='block', ports=None):
    """A named Module with declared ports and one ideal resistor — a
    properly composable child."""
    mod = Module('PDK_skywater130', str(tmp_path / subdir), filename,
                pdk_details=pdk, module_name=name)
    mod.define_module_ports(ports or {'out': 'out', 'vss': 'ground'})
    mod.add(elements.resistor('rload', p='out', n='vss', value=1000))
    return mod


def _top_level(pdk, tmp_path, subdir, filename, ports=None):
    """A Module with no ``module_name`` — a top-level circuit, not a
    composable child."""
    mod = Module('PDK_skywater130', str(tmp_path / subdir), filename,
                pdk_details=pdk)
    if ports:
        mod.define_module_ports(ports)
    return mod


def _no_deprecation_warning(caught):
    return not any(issubclass(w.category, DeprecationWarning) for w in caught)


# ---------------------------------------------------------------------------
# add_element: unchanged behaviour for every accepted kind, now warning
# ---------------------------------------------------------------------------

class TestAddElementDeprecatedAlias:

    def test_device_class(self, pdk, empty_netlist):
        """add_element(deviceClass, name, nets=..., parameters=...) — the
        new-style class-argument path — is unchanged; warns pointing to
        add()."""
        with pytest.warns(DeprecationWarning, match=r'add\(pdk\.nmos1p8\('):
            result = empty_netlist.add_element(
                pdk.nmos1p8, 'mn1',
                nets={'d': 'out', 'g': 'in', 's': 'gnd', 'b': 'gnd'},
                parameters={'w': 1, 'l': 0.15},
            )
        assert result is None
        assert 'mn1' in empty_netlist._element_mgr.elem_dict
        assert (empty_netlist._element_mgr.elem_dict['mn1']['typeName']
                == pdk.nmos1p8.device_name)

    def test_device_instance(self, pdk, empty_netlist):
        """add_element(instance, name) with an already-constructed
        AnalogElement — the new-style instance path — is unchanged; warns
        pointing to add()."""
        mn1 = pdk.nmos1p8('mn1', d='out', g='in', s='gnd', b='gnd',
                          w=1.0, l=0.15, m=1)
        with pytest.warns(DeprecationWarning, match=r'add\(device\)'):
            result = empty_netlist.add_element(mn1, 'mn1')
        assert result is None
        assert empty_netlist._element_mgr.elem_dict['mn1']['device'] is mn1

    def test_legacy_pdk_device(self, pdk, empty_netlist):
        """add_element(pdk._devices[...], name, nets=..., parameters=...) —
        the old-style PDKDevice path — is unchanged; warns pointing to
        add()."""
        with pytest.warns(DeprecationWarning, match=r'add\(pdk\.nmos1p8\('):
            result = empty_netlist.add_element(
                pdk._devices['nmos1p8'], 'mn1',
                nets={'d': 'out', 'g': 'in', 's': 'gnd', 'b': 'gnd'},
                parameters={'w': 1, 'l': 0.15},
            )
        assert result is None
        assert 'mn1' in empty_netlist._element_mgr.elem_dict

    def test_module(self, pdk, tmp_path):
        """add_element(module, name, **ports) with a Module target is
        unchanged (still returns a SubcircuitInstance); warns pointing to
        add_module()."""
        parent = _subcircuit(pdk, tmp_path, 'p', 'parent.cir', name='parent')
        child = _subcircuit(pdk, tmp_path, 'c', 'child.cir', name='child')
        try:
            with pytest.warns(DeprecationWarning, match=r'add_module\('):
                inst = parent.add_element(child, 'u1', out='out', vss='vss')
            assert inst is not None
            assert 'u1' in parent._element_mgr.elem_dict
        finally:
            parent.close()
            child.close()


# ---------------------------------------------------------------------------
# add_submodule: unchanged registration-only behaviour, now warning
# ---------------------------------------------------------------------------

class TestAddSubmoduleDeprecated:

    def test_warns_and_registers_unchanged(self, pdk, tmp_path):
        parent = _subcircuit(pdk, tmp_path, 'p', 'parent.cir', name='parent')
        child = _subcircuit(pdk, tmp_path, 'c', 'child.cir', name='child')
        try:
            with pytest.warns(DeprecationWarning, match=r'add_module\('):
                parent.add_submodule('sub', child)
            assert parent._element_mgr.module_dict['sub']['device'] is child
            # Registration-only: no instance, no .subckt line, no wiring.
            assert 'sub' not in parent._element_mgr.elem_dict
        finally:
            parent.close()
            child.close()

    def test_cycle_detection_still_raises(self, pdk, tmp_path):
        parent = _subcircuit(pdk, tmp_path, 'p', 'parent.cir', name='parent')
        try:
            with pytest.raises(ValueError, match='circular reference'):
                parent.add_submodule('self', parent)
        finally:
            parent.close()


# ---------------------------------------------------------------------------
# add_module: the new hierarchy-placement verb
# ---------------------------------------------------------------------------

class TestAddModule:

    def test_matches_add_element_netlist_and_hash_byte_for_byte(
            self, pdk, tmp_path):
        parent_e = _subcircuit(pdk, tmp_path, 'pe', 'parent.cir', name='parent')
        child_e = _subcircuit(pdk, tmp_path, 'ce', 'child.cir', name='child')
        parent_m = _subcircuit(pdk, tmp_path, 'pm', 'parent.cir', name='parent')
        child_m = _subcircuit(pdk, tmp_path, 'cm', 'child.cir', name='child')
        try:
            with pytest.warns(DeprecationWarning):
                parent_e.add_element(child_e, 'u1', out='out', vss='vss')
            netlist_e = parent_e.generate_netlist(subckt=True)
            hash_e = _hash(parent_e)

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                parent_m.add_module(child_m, 'u1', out='out', vss='vss')
            assert _no_deprecation_warning(caught)
            netlist_m = parent_m.generate_netlist(subckt=True)
            hash_m = _hash(parent_m)

            assert netlist_e == netlist_m
            assert hash_e == hash_m
        finally:
            parent_e.close()
            child_e.close()
            parent_m.close()
            child_m.close()

    def test_nets_dict_form(self, pdk, tmp_path):
        parent = _subcircuit(pdk, tmp_path, 'p', 'parent.cir', name='parent')
        child = _subcircuit(pdk, tmp_path, 'c', 'child.cir', name='child')
        try:
            inst = parent.add_module(child, 'u1',
                                     nets={'out': 'out', 'vss': 'vss'})
            assert inst is not None
            assert 'u1' in parent._element_mgr.elem_dict
        finally:
            parent.close()
            child.close()

    def test_ports_kwarg_form(self, pdk, tmp_path):
        parent = _subcircuit(pdk, tmp_path, 'p', 'parent.cir', name='parent')
        child = _subcircuit(pdk, tmp_path, 'c', 'child.cir', name='child')
        try:
            inst = parent.add_module(child, 'u1', out='out', vss='vss')
            assert inst is not None
            assert 'u1' in parent._element_mgr.elem_dict
        finally:
            parent.close()
            child.close()

    def test_rejects_a_device_instance(self, pdk, empty_netlist):
        mn1 = pdk.nmos1p8('mn1', d='out', g='in', s='gnd', b='gnd',
                          w=1.0, l=0.15, m=1)
        with pytest.raises(TypeError, match=r'add\(\)'):
            empty_netlist.add_module(mn1, 'u1')

    def test_rejects_a_device_class(self, pdk, empty_netlist):
        with pytest.raises(TypeError, match=r'add\(\)'):
            empty_netlist.add_module(pdk.nmos1p8, 'u1',
                                     d='out', g='in', s='gnd', b='gnd')

    def test_inherits_task_175_module_name_refusal(self, pdk, tmp_path):
        """add_module refuses a module_name-less child at the call, exactly
        like add_element — TASK-175's require_composable_child guard lives
        in the implementation both delegate to."""
        parent = _subcircuit(pdk, tmp_path, 'p', 'parent.cir', name='parent')
        child = _top_level(pdk, tmp_path, 'c', 'child.cir',
                           ports={'out': 'out', 'vss': 'ground'})
        try:
            with pytest.raises(NotASubcircuitError, match='module_name'):
                parent.add_module(child, 'u1', out='out', vss='vss')
        finally:
            parent.close()
            child.close()

    def test_no_deprecation_warning(self, pdk, tmp_path):
        parent = _subcircuit(pdk, tmp_path, 'p', 'parent.cir', name='parent')
        child = _subcircuit(pdk, tmp_path, 'c', 'child.cir', name='child')
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                parent.add_module(child, 'u1', out='out', vss='vss')
            assert _no_deprecation_warning(caught)
        finally:
            parent.close()
            child.close()


@pytest.mark.parametrize('legacy', [False, True])
def test_device_warning_contains_executable_replacement(pdk, tmp_path, legacy):
    old = _top_level(pdk, tmp_path, 'old', 'same.cir')
    new = _top_level(pdk, tmp_path, 'new', 'same.cir')
    factory = pdk._devices['nmos1p8'] if legacy else pdk.nmos1p8
    with pytest.warns(DeprecationWarning) as caught:
        old.add_element(factory, 'Mquoted',
                        nets={'d': 'out', 'g': 'in', 's': 'gnd', 'b': 'gnd'},
                        parameters={'w': 2.5, 'l': 0.5})
    replacement = "add(pdk.nmos1p8('Mquoted', d='out', g='in', s='gnd', b='gnd', w=2.5, l=0.5))"
    assert str(caught[0].message) == (
        f'Module.add_element() is deprecated; use {replacement} instead.')
    eval(replacement, {'add': new.add, 'pdk': pdk})
    old.set_output_options(mode='overwrite')
    new.set_output_options(mode='overwrite')
    with open(old.generate_netlist()) as f:
        old_deck = f.read().replace(old.dir_folder, '<TBDIR>')
    with open(new.generate_netlist()) as f:
        new_deck = f.read().replace(new.dir_folder, '<TBDIR>')
    assert old_deck == new_deck


def test_device_warning_resolves_net_objects_and_keyword_overrides(pdk, empty_netlist):
    gate = empty_netlist.add_net('gate')
    nets = {'d': 'out', 'g': 'overridden', 's': 'gnd', 'b': 'gnd'}
    params = {'w': 3, 'l': 0.5}
    with pytest.warns(DeprecationWarning) as caught:
        empty_netlist.add_element(pdk.nmos1p8, 'M1', nets=nets,
                                  parameters=params, g=gate)
    assert str(caught[0].message) == (
        "Module.add_element() is deprecated; use "
        "add(pdk.nmos1p8('M1', d='out', g='gate', s='gnd', b='gnd', w=3, l=0.5)) instead.")
    assert nets['g'] == 'overridden'
    assert params == {'w': 3, 'l': 0.5}


@pytest.mark.parametrize('parameter_form', ['pairs', 'positional', 'omitted'])
def test_device_warning_handles_legacy_argument_forms(pdk, empty_netlist, parameter_form):
    values = {'w': 3.0, 'l': 0.5, 'm': 2}
    if parameter_form == 'pairs':
        parameters = list(values.items())
    elif parameter_form == 'positional':
        parameters = [values[key] for key in pdk._devices['nmos1p8'].param_list]
    else:
        parameters = None
    with pytest.warns(DeprecationWarning) as caught:
        empty_netlist.add_element(pdk.nmos1p8, 'M1',
                                  nets=[('d', 'out'), ('g', 'in'), ('s', 'gnd'), ('b', 'gnd')],
                                  parameters=parameters)
    suffix = {'omitted': '', 'pairs': ', w=3.0, l=0.5, m=2',
              'positional': ', m=2, w=3.0, l=0.5'}[parameter_form]
    assert str(caught[0].message) == (
        "Module.add_element() is deprecated; use "
        f"add(pdk.nmos1p8('M1', d='out', g='in', s='gnd', b='gnd'{suffix})) instead.")


def test_device_warning_uses_target_catalog_positional_order(pdk, empty_netlist):
    class Reordered(pdk.nmos1p8):
        _param_names = ('l', 'w', 'm')

    values = {'m': 2, 'w': 3.0, 'l': 0.5}
    parameters = [values[key] for key in pdk._devices['nmos1p8'].param_list]
    with pytest.warns(DeprecationWarning) as caught:
        empty_netlist.add_element(Reordered, 'M1',
                                  nets={'d': 'out', 'g': 'in', 's': 'gnd', 'b': 'gnd'},
                                  parameters=parameters)
    assert str(caught[0].message) == (
        "Module.add_element() is deprecated; use "
        "add(pdk.nmos1p8('M1', d='out', g='in', s='gnd', b='gnd', m=2, w=3.0, l=0.5)) instead.")


def test_device_warning_without_pdk_preserves_legacy_positional_call(tmp_path):
    mod = Module(None, str(tmp_path), 'legacy.cir', pdk_details=None)
    device = elements.resistor('R1', p='out', n='gnd', value=1000)
    try:
        with pytest.warns(DeprecationWarning, match=r'use add\(device\) instead'):
            mod.add_element(device, 'R1', parameters=[1000])
        assert mod.elem_dict['R1']['device'] is device
        assert device.to_spice() == 'RR1 out gnd 1k\n'
    finally:
        mod.close()


@pytest.mark.parametrize('catalog', [None, object(), {'_devices': None}])
def test_device_warning_falls_back_without_catalog(pdk, catalog):
    from types import SimpleNamespace
    from silk.circuit.modules_structure import _device_add_replacement

    if isinstance(catalog, dict):
        catalog = SimpleNamespace(**catalog)
    assert _device_add_replacement(
        pdk.nmos1p8, 'M1', {'d': 'out'}, [2, 3.0, 0.5], {}, catalog
    ) == 'add(device)'
