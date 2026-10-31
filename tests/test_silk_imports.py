"""Smoke tests — verify every silk.* import resolves without error.

These tests exist purely to catch broken re-exports. They do not test behavior.
"""


class TestTopLevelImports:
    def test_import_silk(self):
        import re
        import silk
        # Don't hardcode the release number here — pyproject agreement is
        # guarded by tests/test_packaging.py::test_version_matches_pyproject.
        assert re.fullmatch(r"\d+\.\d+\.\d+", silk.__version__)

    def test_module(self):
        from silk import Module
        assert Module is not None

    def test_module_testbench(self):
        from silk import ModuleTestbench
        assert ModuleTestbench is not None

    def test_module_variable(self):
        from silk import Variable
        assert Variable is not None

    def test_option_classes_removed(self):
        import silk
        for name in ('VarType', 'Distribution'):
            assert not hasattr(silk, name)

    def test_pdk_classes(self):
        from silk import BasePDK, UNIT_PDK, UNIT_SI
        assert UNIT_PDK is not None
        assert UNIT_SI is not None

    def test_simulation_types(self):
        from silk import Transient, AC, DC, Noise
        t = Transient(stop=1e-6, step=1e-9)
        assert t is not None

    def test_signal_types(self):
        from silk import Signal, NgspiceSignal
        assert Signal is not None
        assert NgspiceSignal is not None

    def test_optimize(self):
        from silk import VariableSpace, Variable, Optimizer
        assert VariableSpace is not None
        assert Variable is not None
        assert Optimizer is not None

    def test_db(self):
        from silk import Database, DesignStage
        assert Database is not None
        assert DesignStage is not None

    def test_codegen(self):
        from silk import Generator
        assert Generator is not None

    def test_exceptions(self):
        from silk import (
            CircuitError, DuplicateNameError, UndeclaredNetError,
            SimulationError, PDKError,
        )
        assert issubclass(CircuitError, Exception)
        assert issubclass(DuplicateNameError, CircuitError)
        assert issubclass(UndeclaredNetError, CircuitError)
        assert issubclass(SimulationError, CircuitError)
        assert issubclass(PDKError, CircuitError)


class TestSubPackageImports:
    def test_silk_circuit(self):
        from silk.circuit import Module, ModuleTestbench, Variable
        assert Module is not None

    def test_silk_pdk(self):
        from silk.pdk import BasePDK, UNIT_PDK, UNIT_SI
        assert BasePDK is not None

    def test_silk_simulation(self):
        from silk.simulation import Transient, AC, DC, Noise
        assert Transient is not None

    def test_silk_signals(self):
        from silk.signals import Signal, NgspiceSignal
        assert Signal is not None

    def test_silk_elements(self):
        from silk.elements import elements, sources, passives
        assert elements is not None

    def test_silk_optimize(self):
        from silk.optimize import VariableSpace, Optimizer
        assert VariableSpace is not None

    def test_silk_db(self):
        from silk.db import Database, DesignStage
        assert Database is not None

    def test_silk_codegen(self):
        from silk.codegen import Generator
        assert Generator is not None

    def test_silk_utils(self):
        from silk.utils import format_spice_value, parse_spice_value
        assert callable(format_spice_value)
        assert callable(parse_spice_value)

    def test_silk_exceptions(self):
        from silk.exceptions import (
            CircuitError, DuplicateNameError, UndeclaredNetError,
            SimulationError, PDKError,
        )
        assert issubclass(DuplicateNameError, CircuitError)
