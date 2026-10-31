"""Tests for CircuitSpec ports (dict form) and Generator subcircuit mode."""
from pathlib import Path

import pytest
from tests._env import needs_ngspice, needs_pdk
from silk.codegen import Generator, CircuitSpec, DeviceSpec
from silk.circuit.circuit_elements import PortDir, Port


class TestCircuitSpecPorts:
    def test_ports_field_defaults_none(self):
        spec = CircuitSpec(circuit_name='test')
        assert spec.ports is None

    def test_from_dict_parses_dict_form_minimal(self):
        d = {'circuit_name': 'ldo',
             'ports': {'vin': {'dir': 'input'},
                       'vout': {'dir': 'output'},
                       'gnd':  {'dir': 'inout'}}}
        spec = CircuitSpec.from_dict(d)
        assert isinstance(spec.ports['vin'], Port)
        assert spec.ports['vin'].dir is PortDir.IN
        assert spec.ports['vout'].dir is PortDir.OUT
        assert spec.ports['gnd'].dir is PortDir.INOUT

    def test_from_dict_no_ports_key(self):
        spec = CircuitSpec.from_dict({'circuit_name': 'x'})
        assert spec.ports is None

    def test_from_dict_empty_ports(self):
        spec = CircuitSpec.from_dict({'circuit_name': 'x', 'ports': {}})
        assert spec.ports == {} and spec.ports is not None

    def test_from_dict_rejects_string_form(self):
        with pytest.raises(TypeError):
            CircuitSpec.from_dict({'circuit_name': 'x',
                                   'ports': {'vin': 'input'}})

    def test_from_dict_rejects_missing_dir(self):
        with pytest.raises(KeyError):
            CircuitSpec.from_dict({'circuit_name': 'x',
                                   'ports': {'vin': {}}})


class TestGeneratorNoPortsEmitsNetlist:
    def test_no_ports_emits_module(self):
        spec = CircuitSpec(circuit_name='tb',
                           simulation={'type': 'transient',
                                       'step': '1n', 'stop': '1u'})
        code = Generator().generate(spec)
        assert 'Module' in code

    def test_no_ports_does_not_emit_subckt(self):
        spec = CircuitSpec(circuit_name='tb')
        code = Generator().generate(spec)
        assert 'module_name' not in code
        assert 'define_module_ports' not in code
        assert 'define_module_nets' not in code

    def test_no_ports_from_dict(self):
        code = Generator().generate({'circuit_name': 'sim'})
        assert 'Module' in code


class TestGeneratorWithPortsEmitsModule:
    def test_with_ports_emits_module(self):
        spec = CircuitSpec(
            circuit_name='ldo',
            ports={'vin': Port(dir=PortDir.IN),
                   'vout': Port(dir=PortDir.OUT)})
        code = Generator().generate(spec)
        assert 'Module' in code

    def test_with_ports_emits_module_name(self):
        spec = CircuitSpec(
            circuit_name='ldo',
            ports={'vin': Port(dir=PortDir.IN),
                   'vout': Port(dir=PortDir.OUT)})
        code = Generator().generate(spec)
        assert "module_name='ldo'" in code

    def test_with_ports_emits_define_module_ports(self):
        spec = CircuitSpec(
            circuit_name='buf',
            ports={'a': Port(dir=PortDir.IN),
                   'z': Port(dir=PortDir.OUT)})
        code = Generator().generate(spec)
        assert 'define_module_ports' in code
        assert "'a': PortDir.IN" in code
        assert "'z': PortDir.OUT" in code
        assert 'from silk.circuit.circuit_elements import' in code
        assert 'PortDir' in code

    def test_with_ports_minimal_emits_bare_PortDir(self):
        """Port with only dir set must emit bare PortDir, not Port(...)."""
        spec = CircuitSpec(
            circuit_name='buf',
            ports={'a': Port(dir=PortDir.IN)})
        code = Generator().generate(spec)
        assert "'a': PortDir.IN" in code
        assert "Port(" not in code

    def test_with_ports_does_not_emit_simulate(self):
        spec = CircuitSpec(
            circuit_name='buf',
            ports={'a': Port(dir=PortDir.IN),
                   'z': Port(dir=PortDir.OUT)})
        code = Generator().generate(spec)
        assert 'simulate(' not in code

    def test_with_ports_from_dict(self):
        d = {'circuit_name': 'buf',
             'ports': {'a': {'dir': 'input'},
                       'z': {'dir': 'output'}}}
        code = Generator().generate(d)
        assert 'Module' in code
        assert 'module_name' in code

    def test_with_ports_includes_pdk_setup(self):
        spec = CircuitSpec(
            circuit_name='ldo',
            ports={'vin': Port(dir=PortDir.IN),
                   'vout': Port(dir=PortDir.OUT)})
        code = Generator().generate(spec)
        assert "load_pdk('sky130')" in code
        assert 'pdk =' in code


class TestBitsLiteral:
    def test_bits_literal_rejects_stepped_range(self):
        from silk.codegen.generator import _bits_literal
        with pytest.raises(ValueError, match='step=1'):
            _bits_literal(range(0, 10, 2))

    def test_bits_literal_accepts_slice(self):
        from silk.codegen.generator import _bits_literal
        assert _bits_literal(slice(0, 4)) == 'slice(0, 4)'
        assert _bits_literal(slice(8, 12)) == 'slice(8, 12)'
        assert _bits_literal(slice(None, 4)) == 'slice(0, 4)'

    def test_bits_literal_rejects_stepped_slice(self):
        from silk.codegen.generator import _bits_literal
        with pytest.raises(ValueError, match='step'):
            _bits_literal(slice(0, 10, 2))


class TestExplicitEmptyPorts:
    """TASK-361: no 'ports' key -> standalone script; {} -> zero-pin module."""

    BASE = {'circuit_name': 'blk', 'pdk': 'sky130'}

    def test_no_ports_key_is_standalone(self):
        code = Generator().generate(dict(self.BASE))
        assert 'ckt = Module(' in code
        assert 'define_module_ports' not in code

    def test_empty_ports_is_zero_pin_module(self):
        code = Generator().generate({**self.BASE, 'ports': {}})
        assert 'mod = Module(' in code
        assert 'mod.define_module_ports({})' in code
        assert 'ckt = Module(' not in code

    def test_empty_ports_module_executes(self, tmp_path):
        code = Generator().generate(
            {**self.BASE, 'ports': {}, 'output_dir': str(tmp_path)})
        ns: dict = {}
        exec(code, ns)
        assert ns['mod'].module_name == 'blk'

    def test_voltage_domain_alias_reserved_for_zero_pin_module(self):
        spec = {
            **self.BASE, 'ports': {},
            'voltage_domains': {'mod_inv': {'nominal': 1.8}},
            'submodules': [{'instance_name': 'i1', 'nets': {},
                            'module_ref': {'import_path': '.inv',
                                           'module_var': 'mod'}}],
        }
        code = Generator().generate(spec)
        assert 'as mod_inv\n' not in code
        assert 'from .inv import mod as mod_inv_1' in code

    @needs_pdk
    @needs_ngspice
    def test_zero_pin_module_placed_as_dut_simulates(self, tmp_path):
        from silk import Module
        from silk.simulation import Transient
        spec = {
            **self.BASE, 'ports': {}, 'output_dir': str(tmp_path),
            'sources': [{'type': 'VoltageSource', 'instance': 'v1',
                         'positive': 'a', 'negative': 'gnd', 'dc': 1.0}],
            'ideal_elements': [
                {'type': 'resistor', 'instance': 'r1',
                 'nets': {'p': 'a', 'n': 'b'}, 'params': {'value': 1000.0}},
                {'type': 'resistor', 'instance': 'r2',
                 'nets': {'p': 'b', 'n': 'gnd'}, 'params': {'value': 1000.0}}],
        }
        ns: dict = {}
        exec(Generator().generate(spec), ns)
        top = Module('PDK_sky130', str(tmp_path), 'top.cir',
                     pdk_details=ns['pdk'])
        top.add_module(ns['mod'], 'xblk')
        top.add_sim_type(Transient(step=1e-9, stop=10e-9))
        deck_path = top.generate_netlist(corner='tt')
        top.run_simulator()
        deck = Path(deck_path).read_text()
        assert '.subckt blk \n' in deck and 'xxblk blk' in deck
