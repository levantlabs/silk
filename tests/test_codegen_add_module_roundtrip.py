"""Committed round-trip test for TASK-298.2 (generator emits add_module).

Generates a child CircuitSpec with declared ports, and a parent that
places it scalar (x1, x2) and arrayed (array=2, both naming styles),
writes both to a temp package, imports the parent, and checks the
result builds cleanly with no ``NotASubcircuitError`` and no
``DeprecationWarning`` for the entries that carry no ``parameters``.

A second copy of the parent's emission, with every ``add_module(``
textually swapped for ``add_element(`` — the exact call shape the
generator emitted before this task, for an entry without parameters —
stands in for "the old add_element emission" and proves
``generate_netlist(subckt=True)`` is unaffected by which of the two
equivalent methods placed the instances.

A third section pins the *kept* ``add_element(..., parameters=...)``
form (scalar and arrayed) the generator still emits, and its current
run-time failure — TASK-292 owns fixing that; this only pins it.
"""
import importlib
import sys
import warnings

import pytest

from silk.codegen.generator import Generator
from silk.exceptions import CircuitError, NotASubcircuitError

CHILD_SPEC = {
    'circuit_name': 'leaf',
    'pdk': 'sky130',
    'ports': {
        'vin': {'dir': 'input'},
        'vout': {'dir': 'output'},
        'vdd': {'dir': 'inout', 'role': 'power'},
        'vss': {'dir': 'inout', 'role': 'ground'},
    },
    'devices': [
        {'type': 'nmos1p8', 'instance': 'mn',
         'nets': {'d': 'vout', 'g': 'vin', 's': 'vss', 'b': 'vss'},
         'params': {'w': 1, 'l': 0.15}},
        {'type': 'pmos1p8', 'instance': 'mp',
         'nets': {'d': 'vout', 'g': 'vin', 's': 'vdd', 'b': 'vdd'},
         'params': {'w': 2, 'l': 0.15}},
    ],
}

NO_PARAM_SUBMODULES = [
    {'instance_name': 'x1',
     'module_ref': {'import_path': '.leaf', 'module_var': 'mod'},
     'nets': {'vin': 'a', 'vout': 'm', 'vdd': 'vdd', 'vss': 'vss'}},
    {'instance_name': 'x2',
     'module_ref': {'import_path': '.leaf', 'module_var': 'mod'},
     'nets': {'vin': 'm', 'vout': 'y', 'vdd': 'vdd', 'vss': 'vss'}},
    {'instance_name': 'xa',
     'module_ref': {'import_path': '.leaf', 'module_var': 'mod'},
     'nets': {'vin': 'a', 'vout': 'y', 'vdd': 'vdd', 'vss': 'vss'},
     'array': 2, 'naming': 'bracketed'},
    {'instance_name': 'xb',
     'module_ref': {'import_path': '.leaf', 'module_var': 'mod'},
     'nets': {'vin': 'a', 'vout': 'y', 'vdd': 'vdd', 'vss': 'vss'},
     'array': 2, 'naming': 'underscore'},
]

PARAM_SCALAR_SUBMODULES = [
    {'instance_name': 'xp',
     'module_ref': {'import_path': '.leaf', 'module_var': 'mod'},
     'nets': {'vin': 'a', 'vout': 'y', 'vdd': 'vdd', 'vss': 'vss'},
     'parameters': {'drive_strength': 4}},
]

PARAM_ARRAY_SUBMODULES = [
    {'instance_name': 'xpa',
     'module_ref': {'import_path': '.leaf', 'module_var': 'mod'},
     'nets': {'vin': 'a', 'vout': 'y', 'vdd': 'vdd', 'vss': 'vss'},
     'parameters': {'drive_strength': 4},
     'array': 2, 'naming': 'bracketed'},
]


def _parent_spec(submodules, output_dir):
    return {
        'circuit_name': 'top',
        'pdk': 'sky130',
        'output_dir': output_dir,
        'ports': {
            'a': {'dir': 'input'},
            'y': {'dir': 'output'},
            'vdd': {'dir': 'inout', 'role': 'power'},
            'vss': {'dir': 'inout', 'role': 'ground'},
        },
        'submodules': submodules,
    }


def _write_package(tmp_path, pkg_name, leaf_code, top_code):
    root = tmp_path / pkg_name
    root.mkdir()
    (root / '__init__.py').write_text('')
    (root / 'leaf.py').write_text(leaf_code)
    (root / 'top.py').write_text(top_code)
    (root / 'sim').mkdir()
    return root


@pytest.fixture
def sys_path_cleanup(monkeypatch):
    imported = []

    def register(tmp_path, pkg_name):
        monkeypatch.syspath_prepend(str(tmp_path))
        imported.append(pkg_name)

    yield register
    for pkg_name in imported:
        for key in list(sys.modules):
            if key == pkg_name or key.startswith(pkg_name + '.'):
                del sys.modules[key]


class TestAddModuleRoundtrip:
    """No-parameters entries: add_module, no DeprecationWarning, byte-identical netlist."""

    def test_generated_code_emits_add_module_every_form(self):
        leaf_code = Generator().generate(dict(CHILD_SPEC, output_dir='./sim'))
        top_code = Generator().generate(
            _parent_spec(NO_PARAM_SUBMODULES, './sim')
        )
        assert "mod.add_module(mod_leaf, 'x1', nets=" in top_code
        assert "mod.add_module(mod_leaf, 'x2', nets=" in top_code
        assert "for i in range(2): mod.add_module(mod_leaf, f'xa[{i}]', nets=" in top_code
        assert "for i in range(2): mod.add_module(mod_leaf, f'xb_{i}', nets=" in top_code
        assert 'add_element' not in top_code
        assert 'add_element' not in leaf_code

    def test_builds_with_no_error_and_no_deprecation_warning(
        self, tmp_path, sys_path_cleanup
    ):
        pkg_name = 'add_module_roundtrip_new'
        leaf_code = Generator().generate(
            dict(CHILD_SPEC, output_dir=str(tmp_path / pkg_name / 'sim'))
        )
        top_code = Generator().generate(
            _parent_spec(NO_PARAM_SUBMODULES, str(tmp_path / pkg_name / 'sim'))
        )
        _write_package(tmp_path, pkg_name, leaf_code, top_code)
        sys_path_cleanup(tmp_path, pkg_name)
        importlib.invalidate_caches()

        with warnings.catch_warnings():
            warnings.simplefilter('error', DeprecationWarning)
            try:
                top = importlib.import_module(pkg_name + '.top')
            except NotASubcircuitError:
                pytest.fail(
                    'importing the generated parent raised '
                    'NotASubcircuitError'
                )

        netlist = top.mod.generate_netlist(subckt=True)
        for inst in ('x1', 'x2', 'xa[0]', 'xa[1]', 'xb_0', 'xb_1'):
            assert inst in netlist

    def test_netlist_byte_identical_to_old_add_element_emission(
        self, tmp_path, sys_path_cleanup
    ):
        new_pkg = 'add_module_roundtrip_addmodule'
        old_pkg = 'add_module_roundtrip_addelement'

        new_leaf = Generator().generate(
            dict(CHILD_SPEC, output_dir=str(tmp_path / new_pkg / 'sim'))
        )
        new_top = Generator().generate(
            _parent_spec(NO_PARAM_SUBMODULES, str(tmp_path / new_pkg / 'sim'))
        )
        _write_package(tmp_path, new_pkg, new_leaf, new_top)
        sys_path_cleanup(tmp_path, new_pkg)

        # The old_pkg copy replays exactly what the generator emitted
        # before TASK-298.2 for these (parameter-free) entries: the same
        # call, add_element instead of add_module. add_module has no
        # `parameters` argument, but none of these entries pass any, so
        # the substitution is the full old-vs-new diff.
        old_leaf = Generator().generate(
            dict(CHILD_SPEC, output_dir=str(tmp_path / old_pkg / 'sim'))
        )
        new_top_for_old = Generator().generate(
            _parent_spec(NO_PARAM_SUBMODULES, str(tmp_path / old_pkg / 'sim'))
        )
        old_top = new_top_for_old.replace('.add_module(', '.add_element(')
        assert '.add_element(' in old_top
        assert '.add_module(' not in old_top
        _write_package(tmp_path, old_pkg, old_leaf, old_top)
        sys_path_cleanup(tmp_path, old_pkg)

        importlib.invalidate_caches()
        with warnings.catch_warnings():
            warnings.simplefilter('error', DeprecationWarning)
            new_top_mod = importlib.import_module(new_pkg + '.top')

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            old_top_mod = importlib.import_module(old_pkg + '.top')
        assert any(
            issubclass(w.category, DeprecationWarning) for w in caught
        ), 'old-style add_element(...) placement should still warn'

        new_netlist = new_top_mod.mod.generate_netlist(subckt=True)
        old_netlist = old_top_mod.mod.generate_netlist(subckt=True)
        assert new_netlist == old_netlist
        for inst in ('x1', 'x2', 'xa[0]', 'xa[1]', 'xb_0', 'xb_1'):
            assert inst in new_netlist


class TestKeptAddElementParametersForm:
    """Entries with `parameters` still emit add_element (TASK-292 unresolved)."""

    def test_scalar_emitted_text(self):
        code = Generator().generate(
            _parent_spec(PARAM_SCALAR_SUBMODULES, './sim')
        )
        assert (
            "mod.add_element(mod_leaf, 'xp', "
            "nets={'vin': 'a', 'vout': 'y', 'vdd': 'vdd', 'vss': 'vss'}, "
            "parameters={'drive_strength': 4})"
        ) in code
        assert 'add_module' not in code

    def test_arrayed_emitted_text(self):
        code = Generator().generate(
            _parent_spec(PARAM_ARRAY_SUBMODULES, './sim')
        )
        assert (
            "for i in range(2): mod.add_element(mod_leaf, f'xpa[{i}]', "
            "nets={'vin': 'a', 'vout': 'y', 'vdd': 'vdd', 'vss': 'vss'}, "
            "parameters={'drive_strength': 4})"
        ) in code
        assert 'add_module' not in code

    def _run(self, tmp_path, sys_path_cleanup, pkg_name, submodules):
        leaf_code = Generator().generate(
            dict(CHILD_SPEC, output_dir=str(tmp_path / pkg_name / 'sim'))
        )
        top_code = Generator().generate(
            _parent_spec(submodules, str(tmp_path / pkg_name / 'sim'))
        )
        _write_package(tmp_path, pkg_name, leaf_code, top_code)
        sys_path_cleanup(tmp_path, pkg_name)
        importlib.invalidate_caches()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with pytest.raises(CircuitError, match='only supports DEVICE'):
                importlib.import_module(pkg_name + '.top')
        return caught

    def test_scalar_executes_deprecation_warning_then_raises(
        self, tmp_path, sys_path_cleanup
    ):
        caught = self._run(
            tmp_path, sys_path_cleanup,
            'add_module_roundtrip_param_scalar', PARAM_SCALAR_SUBMODULES,
        )
        assert any(
            issubclass(w.category, DeprecationWarning) for w in caught
        )

    def test_arrayed_executes_deprecation_warning_then_raises(
        self, tmp_path, sys_path_cleanup
    ):
        caught = self._run(
            tmp_path, sys_path_cleanup,
            'add_module_roundtrip_param_array', PARAM_ARRAY_SUBMODULES,
        )
        assert any(
            issubclass(w.category, DeprecationWarning) for w in caught
        )
