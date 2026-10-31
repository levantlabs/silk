"""Tests for voltage_domains, bus_nets, and rich-port codegen."""
import pytest
from silk.codegen import Generator, CircuitSpec, DeviceSpec, BusNetSpec
from silk.circuit.circuit_elements import PortDir, PortRole, Port
from silk.circuit.voltage_domain import VoltageDomain
from tests._env import needs_pdk
from tests._env import scratch_dir


class TestBusNetSpecParsing:
    def test_default_empty(self):
        spec = CircuitSpec(circuit_name='x')
        assert spec.bus_nets == []

    def test_from_dict_int_bits(self):
        d = {'circuit_name': 'x',
             'bus_nets': [{'name': 'count', 'bits': 4}]}
        spec = CircuitSpec.from_dict(d)
        assert len(spec.bus_nets) == 1
        bn = spec.bus_nets[0]
        assert isinstance(bn, BusNetSpec)
        assert bn.name == 'count'
        assert bn.bits == 4
        assert bn.naming == 'bracketed'   # default

    def test_from_dict_range_bits(self):
        d = {'circuit_name': 'x',
             'bus_nets': [{'name': 'addr', 'bits': [4, 12]}]}
        spec = CircuitSpec.from_dict(d)
        bn = spec.bus_nets[0]
        assert bn.bits == range(4, 12)

    def test_from_dict_naming_override(self):
        d = {'circuit_name': 'x',
             'bus_nets': [{'name': 'dbg', 'bits': 4,
                           'naming': 'underscore'}]}
        spec = CircuitSpec.from_dict(d)
        assert spec.bus_nets[0].naming == 'underscore'

    def test_from_dict_missing_name(self):
        with pytest.raises(KeyError):
            CircuitSpec.from_dict({'circuit_name': 'x',
                                   'bus_nets': [{'bits': 4}]})

    def test_from_dict_missing_bits(self):
        with pytest.raises(KeyError):
            CircuitSpec.from_dict({'circuit_name': 'x',
                                   'bus_nets': [{'name': 'count'}]})

    def test_bus_naming_null_treated_as_absent(self):
        d = {'circuit_name': 'x',
             'bus_nets': [{'name': 'count', 'bits': 4, 'naming': None}]}
        spec = CircuitSpec.from_dict(d)
        assert spec.bus_nets[0].naming == 'bracketed'


class TestVoltageDomainParsing:
    def test_default_empty(self):
        spec = CircuitSpec(circuit_name='x')
        assert spec.voltage_domains == {}

    def test_from_dict_nominal_only(self):
        d = {'circuit_name': 'x',
             'voltage_domains': {'AVDD': {'nominal': 1.8}}}
        spec = CircuitSpec.from_dict(d)
        assert 'AVDD' in spec.voltage_domains
        avdd = spec.voltage_domains['AVDD']
        assert isinstance(avdd, VoltageDomain)
        assert avdd.name == 'AVDD'
        assert avdd.nominal == 1.8
        assert avdd.tolerance is None
        assert avdd.min is None
        assert avdd.max is None

    def test_from_dict_with_tolerance(self):
        d = {'circuit_name': 'x',
             'voltage_domains': {'AVDD': {'nominal': 1.8, 'tolerance': 0.1}}}
        spec = CircuitSpec.from_dict(d)
        avdd = spec.voltage_domains['AVDD']
        assert avdd.tolerance == 0.1
        assert avdd.v_min() == pytest.approx(1.62)
        assert avdd.v_max() == pytest.approx(1.98)

    def test_from_dict_with_min_max(self):
        d = {'circuit_name': 'x',
             'voltage_domains': {'IO': {'nominal': 3.3,
                                        'min': 3.0, 'max': 3.6}}}
        spec = CircuitSpec.from_dict(d)
        io = spec.voltage_domains['IO']
        assert io.min == 3.0
        assert io.max == 3.6

    def test_from_dict_multiple_domains(self):
        d = {'circuit_name': 'x',
             'voltage_domains': {
                 'AVDD': {'nominal': 1.8},
                 'DVDD': {'nominal': 0.9},
             }}
        spec = CircuitSpec.from_dict(d)
        assert set(spec.voltage_domains.keys()) == {'AVDD', 'DVDD'}

    def test_from_dict_missing_nominal(self):
        with pytest.raises(KeyError):
            CircuitSpec.from_dict({'circuit_name': 'x',
                                   'voltage_domains': {'AVDD': {}}})

    def test_invalid_domain_name_raises(self):
        bad_names = ['1.8V', 'vdd-core', 'class', 'vdd.1', 'with space']
        for bad in bad_names:
            with pytest.raises(ValueError, match='valid Python identifier'):
                CircuitSpec.from_dict({
                    'circuit_name': 'x',
                    'voltage_domains': {bad: {'nominal': 1.8}},
                })

    def test_generator_local_domain_names_are_accepted(self):
        for name in ['pdk', 'mod', 'ckt', 'sources', 'elements']:
            spec = CircuitSpec.from_dict({
                'circuit_name': 'x',
                'voltage_domains': {name: {'nominal': 1.8}},
            })
            assert spec.voltage_domains[name].name == name


class TestRichPortParsing:
    def test_role_only(self):
        d = {'circuit_name': 'x',
             'ports': {'vh': {'dir': 'output', 'role': 'analog'}}}
        spec = CircuitSpec.from_dict(d)
        port = spec.ports['vh']
        assert port.dir is PortDir.OUT
        assert port.role is PortRole.ANALOG
        assert port.domain is None
        assert port.bits is None

    def test_domain_resolved_to_instance(self):
        d = {'circuit_name': 'x',
             'voltage_domains': {'AVDD': {'nominal': 1.8}},
             'ports': {'vh': {'dir': 'output', 'role': 'analog',
                              'domain': 'AVDD'}}}
        spec = CircuitSpec.from_dict(d)
        port = spec.ports['vh']
        # Identity check: must be the same VoltageDomain instance.
        assert port.domain is spec.voltage_domains['AVDD']

    def test_domain_unknown_raises(self):
        d = {'circuit_name': 'x',
             'voltage_domains': {'AVDD': {'nominal': 1.8}},
             'ports': {'vh': {'dir': 'output', 'domain': 'WRONG'}}}
        with pytest.raises(ValueError) as exc:
            CircuitSpec.from_dict(d)
        # Error must name both the port and the missing domain.
        assert 'vh' in str(exc.value)
        assert 'WRONG' in str(exc.value)

    def test_bits_int(self):
        d = {'circuit_name': 'x',
             'ports': {'code': {'dir': 'input', 'bits': 4}}}
        spec = CircuitSpec.from_dict(d)
        assert spec.ports['code'].bits == 4

    def test_bits_range(self):
        d = {'circuit_name': 'x',
             'ports': {'lanes': {'dir': 'input', 'bits': [8, 12]}}}
        spec = CircuitSpec.from_dict(d)
        assert spec.ports['lanes'].bits == range(8, 12)

    def test_naming_underscore(self):
        d = {'circuit_name': 'x',
             'ports': {'dbg': {'dir': 'output', 'bits': 4,
                               'naming': 'underscore'}}}
        spec = CircuitSpec.from_dict(d)
        assert spec.ports['dbg'].naming == 'underscore'

    def test_unknown_dir_raises(self):
        with pytest.raises(ValueError):
            CircuitSpec.from_dict({'circuit_name': 'x',
                                   'ports': {'x': {'dir': 'sideways'}}})

    def test_unknown_role_raises(self):
        with pytest.raises(ValueError):
            CircuitSpec.from_dict({'circuit_name': 'x',
                                   'ports': {'x': {'dir': 'input',
                                                   'role': 'pinkish'}}})

    def test_naming_null_treated_as_absent(self):
        d = {'circuit_name': 'x',
             'ports': {'a': {'dir': 'input', 'naming': None}}}
        spec = CircuitSpec.from_dict(d)
        # Should fall back to the Port dataclass default, not stay None.
        assert spec.ports['a'].naming == 'bracketed'


class TestVoltageDomainEmission:
    def test_emits_voltage_domain_line(self):
        spec = CircuitSpec(
            circuit_name='x',
            voltage_domains={'AVDD': VoltageDomain(name='AVDD', nominal=1.8)},
            ports={'vh': Port(dir=PortDir.OUT)})
        code = Generator().generate(spec)
        assert "AVDD = VoltageDomain('AVDD', nominal=1.8)" in code
        assert ('from silk.circuit.voltage_domain import VoltageDomain'
                in code)
        import hashlib
        assert hashlib.sha256(code.encode()).hexdigest() == (
            '056b92d8f05c3c8dd02ed0485a08f918265235a88428b2a4dc7c4e361f2ed703')

    def test_emits_tolerance_kwarg_when_set(self):
        spec = CircuitSpec(
            circuit_name='x',
            voltage_domains={'AVDD': VoltageDomain(name='AVDD',
                                                   nominal=1.8,
                                                   tolerance=0.1)},
            ports={'vh': Port(dir=PortDir.OUT)})
        code = Generator().generate(spec)
        assert "tolerance=0.1" in code

    def test_emits_min_max_when_set(self):
        spec = CircuitSpec(
            circuit_name='x',
            voltage_domains={'IO': VoltageDomain(name='IO',
                                                  nominal=3.3,
                                                  min=3.0, max=3.6)},
            ports={'vh': Port(dir=PortDir.OUT)})
        code = Generator().generate(spec)
        assert "min=3.0" in code
        assert "max=3.6" in code

    def test_omits_unset_optional_fields(self):
        spec = CircuitSpec(
            circuit_name='x',
            voltage_domains={'AVDD': VoltageDomain(name='AVDD', nominal=1.8)},
            ports={'vh': Port(dir=PortDir.OUT)})
        code = Generator().generate(spec)
        line = [l for l in code.splitlines()
                if l.startswith('AVDD = VoltageDomain')][0]
        assert 'tolerance=' not in line
        assert 'min=' not in line
        assert 'max=' not in line

    def test_no_domains_no_import(self):
        spec = CircuitSpec(
            circuit_name='x',
            ports={'vh': Port(dir=PortDir.OUT)})
        code = Generator().generate(spec)
        assert 'VoltageDomain' not in code

    def test_multiple_domains_emitted_in_order(self):
        spec = CircuitSpec(
            circuit_name='x',
            voltage_domains={
                'AVDD': VoltageDomain(name='AVDD', nominal=1.8),
                'DVDD': VoltageDomain(name='DVDD', nominal=0.9),
            },
            ports={'vh': Port(dir=PortDir.OUT)})
        code = Generator().generate(spec)
        avdd_pos = code.find('AVDD = VoltageDomain')
        dvdd_pos = code.find('DVDD = VoltageDomain')
        assert 0 <= avdd_pos < dvdd_pos


class TestPortEmissionForms:
    def test_minimal_emits_bare_portdir(self):
        spec = CircuitSpec(
            circuit_name='x',
            ports={'clk': Port(dir=PortDir.IN)})
        code = Generator().generate(spec)
        assert "'clk': PortDir.IN" in code
        assert "Port(" not in code
        assert "PortRole" not in code

    def test_dir_plus_role_emits_tuple(self):
        spec = CircuitSpec(
            circuit_name='x',
            ports={'vh': Port(dir=PortDir.OUT, role=PortRole.ANALOG)})
        code = Generator().generate(spec)
        assert "'vh': (PortDir.OUT, PortRole.ANALOG)" in code
        assert "Port(" not in code

    def test_with_domain_emits_full_port(self):
        avdd = VoltageDomain(name='AVDD', nominal=1.8)
        spec = CircuitSpec(
            circuit_name='x',
            voltage_domains={'AVDD': avdd},
            ports={'vh': Port(dir=PortDir.OUT, role=PortRole.ANALOG,
                              domain=avdd)})
        code = Generator().generate(spec)
        assert ("'vh': Port(dir=PortDir.OUT, role=PortRole.ANALOG, "
                "domain=AVDD)") in code

    def test_with_bits_int_emits_full_port(self):
        spec = CircuitSpec(
            circuit_name='x',
            ports={'code': Port(dir=PortDir.IN, role=PortRole.DIGITAL,
                                 bits=4)})
        code = Generator().generate(spec)
        assert "bits=4" in code
        assert "Port(" in code

    def test_with_bits_range_emits_range(self):
        spec = CircuitSpec(
            circuit_name='x',
            ports={'lanes': Port(dir=PortDir.IN, bits=range(8, 12))})
        code = Generator().generate(spec)
        assert "bits=range(8, 12)" in code

    def test_naming_underscore_in_full_port(self):
        spec = CircuitSpec(
            circuit_name='x',
            ports={'dbg': Port(dir=PortDir.OUT, bits=4,
                                naming='underscore')})
        code = Generator().generate(spec)
        assert "naming='underscore'" in code

    def test_naming_default_omitted(self):
        spec = CircuitSpec(
            circuit_name='x',
            ports={'dbg': Port(dir=PortDir.OUT, bits=4)})
        code = Generator().generate(spec)
        assert "naming=" not in code

    def test_imports_conditional(self):
        # No role anywhere -> no PortRole import.
        spec1 = CircuitSpec(circuit_name='x',
                            ports={'clk': Port(dir=PortDir.IN)})
        code1 = Generator().generate(spec1)
        assert 'PortRole' not in code1

        # Any role -> PortRole imported.
        spec2 = CircuitSpec(
            circuit_name='x',
            ports={'vh': Port(dir=PortDir.OUT, role=PortRole.ANALOG)})
        code2 = Generator().generate(spec2)
        assert 'PortRole' in code2

        # Any full-Port form -> Port imported on the codegen import line.
        spec3 = CircuitSpec(
            circuit_name='x',
            ports={'code': Port(dir=PortDir.IN, bits=4)})
        code3 = Generator().generate(spec3)
        import_line = [l for l in code3.splitlines()
                       if 'from silk.circuit.circuit_elements import' in l][0]
        assert 'Port' in import_line.split('import')[1]


class TestBusNetEmission:
    def test_emits_int_bits(self):
        spec = CircuitSpec(
            circuit_name='x',
            ports={'clk': Port(dir=PortDir.IN)},
            bus_nets=[BusNetSpec(name='count', bits=4)])
        code = Generator().generate(spec)
        assert "mod.add_net('count', bits=4)" in code

    def test_emits_range_bits(self):
        spec = CircuitSpec(
            circuit_name='x',
            ports={'clk': Port(dir=PortDir.IN)},
            bus_nets=[BusNetSpec(name='addr', bits=range(4, 12))])
        code = Generator().generate(spec)
        assert "mod.add_net('addr', bits=range(4, 12))" in code

    def test_emits_naming_when_underscore(self):
        spec = CircuitSpec(
            circuit_name='x',
            ports={'clk': Port(dir=PortDir.IN)},
            bus_nets=[BusNetSpec(name='dbg', bits=4, naming='underscore')])
        code = Generator().generate(spec)
        assert "mod.add_net('dbg', bits=4, naming='underscore')" in code

    def test_omits_naming_when_default(self):
        spec = CircuitSpec(
            circuit_name='x',
            ports={'clk': Port(dir=PortDir.IN)},
            bus_nets=[BusNetSpec(name='count', bits=4)])
        code = Generator().generate(spec)
        line = [l for l in code.splitlines() if 'add_net' in l][0]
        assert 'naming=' not in line

    def test_ordering_after_ports_before_devices(self):
        spec = CircuitSpec(
            circuit_name='x',
            ports={'clk': Port(dir=PortDir.IN)},
            bus_nets=[BusNetSpec(name='count', bits=4)],
            devices=[DeviceSpec(
                type='nmos1p8', instance='mn1',
                nets={'d': 'count[0]', 'g': 'clk',
                      's': 'gnd', 'b': 'gnd'},
                params={'w': 1.0, 'l': 0.15, 'm': 1})])
        code = Generator().generate(spec)
        pos_ports = code.find('define_module_ports')
        pos_addnet = code.find('add_net(')
        pos_device = code.find("mod.add(pdk.nmos1p8('mn1'")
        assert 0 <= pos_ports < pos_addnet < pos_device

    def test_no_bus_nets_no_block(self):
        spec = CircuitSpec(
            circuit_name='x',
            ports={'clk': Port(dir=PortDir.IN)})
        code = Generator().generate(spec)
        assert 'add_net' not in code


class TestEndToEnd:
    """Exec the generated .py and inspect the resulting Module."""

    def _exec(self, code):
        """Execute generated code in a fresh namespace; return locals."""
        import tempfile, os
        ns = {}
        exec(compile(code, '<generated>', 'exec'), ns)
        return ns

    @needs_pdk

    def test_th_block_compiles_and_execs(self):
        tmp_dir = scratch_dir()
        d = {
            'circuit_name': 'th_block',
            'output_dir': tmp_dir,
            'voltage_domains': {
                'AVDD': {'nominal': 1.8, 'tolerance': 0.1},
            },
            'ports': {
                'clk':  {'dir': 'input'},
                'vh':   {'dir': 'output', 'role': 'analog',
                         'domain': 'AVDD'},
                'vdd':  {'dir': 'input',  'role': 'power',
                         'domain': 'AVDD'},
            },
        }
        code = Generator().generate(d)
        # compile catches syntax; exec catches constructor-signature drift
        # (e.g., VoltageDomain(voltage=...) vs nominal=).
        ns = self._exec(code)
        mod = ns['mod']
        assert mod.module_name == 'th_block'
        assert set(mod.pin_list.keys()) == {'clk', 'vh', 'vdd'}

    @needs_pdk

    def test_bus_ports_round_trip(self):
        tmp_dir = scratch_dir()
        d = {
            'circuit_name': 'reg_file',
            'output_dir': tmp_dir,
            'ports': {
                'clk':  {'dir': 'input'},
                'code': {'dir': 'input',  'role': 'digital', 'bits': 4},
            },
            'bus_nets': [
                {'name': 'count', 'bits': 4},
            ],
        }
        code = Generator().generate(d)
        ns = self._exec(code)
        mod = ns['mod']

        # Port `code` must be bus-expanded into 4 scalar pins.
        for bit in range(4):
            assert f'code[{bit}]' in mod.pin_list

        # Internal bus net `count` must be registered in _buses.
        assert 'count' in mod._buses
        bus = mod._buses['count']
        assert bus.indices == (0, 1, 2, 3)

    @needs_pdk

    def test_bus_port_range_form_round_trip(self):
        tmp_dir = scratch_dir()
        d = {
            'circuit_name': 'lanes',
            'output_dir': tmp_dir,
            'ports': {
                'lanes': {'dir': 'input', 'role': 'analog',
                          'bits': [8, 12]},
            },
        }
        code = Generator().generate(d)
        ns = self._exec(code)
        mod = ns['mod']
        for bit in (8, 9, 10, 11):
            assert f'lanes[{bit}]' in mod.pin_list
        assert mod._buses['lanes'].indices == (8, 9, 10, 11)


class TestOutputDirAndPdkImport:
    def test_module_uses_spec_output_dir(self):
        spec = CircuitSpec(
            circuit_name='th_block', output_dir='./gen/out',
            ports={'clk': Port(dir=PortDir.IN)})
        code = Generator().generate(spec)
        assert "'./gen/out'" in code
        assert "'./subckt'" not in code

    def test_module_uses_default_output_dir(self):
        # When output_dir is the default './sim', that's what we expect.
        spec = CircuitSpec(
            circuit_name='th_block',
            ports={'clk': Port(dir=PortDir.IN)})
        code = Generator().generate(spec)
        assert "'./sim'" in code
        assert "'./subckt'" not in code

    def test_single_pdk_import_line(self):
        """The PDK setup must emit exactly one `load_pdk` call, keyed on
        spec.pdk via the registry (no per-PDK hardcoded import line)."""
        spec = CircuitSpec(
            circuit_name='x',
            ports={'clk': Port(dir=PortDir.IN)})
        code = Generator().generate(spec)
        pdk_lines = [l for l in code.splitlines()
                     if l == 'from silk import load_pdk']
        assert len(pdk_lines) == 1
        load_calls = [l for l in code.splitlines() if 'load_pdk(' in l]
        assert load_calls == [f"pdk = load_pdk({spec.pdk!r})"]


class TestVoltageDomainSwitchable:
    def test_from_dict_parses_switchable(self):
        d = {'circuit_name': 'x',
             'voltage_domains': {'DVDD': {'nominal': 0.9, 'switchable': True},
                                 'AVDD': {'nominal': 1.8}}}
        spec = CircuitSpec.from_dict(d)
        assert spec.voltage_domains['DVDD'].switchable is True
        assert spec.voltage_domains['AVDD'].switchable is False

    def test_emits_switchable_only_when_set(self):
        spec = CircuitSpec(
            circuit_name='x',
            voltage_domains={'DVDD': VoltageDomain(name='DVDD', nominal=0.9,
                                                   switchable=True),
                             'AVDD': VoltageDomain(name='AVDD', nominal=1.8)},
            ports={'vh': Port(dir=PortDir.OUT)})
        code = Generator().generate(spec)
        assert "DVDD = VoltageDomain('DVDD', nominal=0.9, switchable=True)" in code
        assert "AVDD = VoltageDomain('AVDD', nominal=1.8)" in code


@pytest.mark.parametrize('naming', ['bracketed', 'underscore'])
def test_generated_bus_ownership_and_scalar_alias_refusal(tmp_path, naming):
    from silk.exceptions import DuplicateNameError
    spec = CircuitSpec(circuit_name='generated_bus', output_dir=str(tmp_path),
                       ports={'d': Port(dir=PortDir.IN, bits=2, naming=naming)})
    code = Generator().generate(spec)
    namespace = {}
    exec(compile(code, '<generated>', 'exec'), namespace)
    mod = namespace['mod']
    assert list(mod.pin_list) == list(mod._buses['d'].expanded_names())
    mod.define_module_ports(spec.ports)
    with pytest.raises(DuplicateNameError):
        mod.define_module_ports({'d': PortDir.IN})
    with pytest.raises(DuplicateNameError):
        mod.add_net('d')


def test_generated_underscore_member_collision_refuses(tmp_path):
    from silk.exceptions import DuplicateNameError
    spec = CircuitSpec(circuit_name='generated_collision', output_dir=str(tmp_path),
                       ports={'d': Port(dir=PortDir.IN, bits=2, naming='underscore'),
                              'd_0': Port(dir=PortDir.IN)})
    code = Generator().generate(spec)
    with pytest.raises(DuplicateNameError, match='d_0'):
        exec(compile(code, '<generated>', 'exec'), {})


def test_generated_wider_internal_bus_keeps_four_formal_pins(tmp_path):
    code = Generator().generate(CircuitSpec(
        circuit_name='wide', output_dir=str(tmp_path),
        ports={'x': Port.analog_input(bits=4)},
        bus_nets=[BusNetSpec(name='x', bits=8)]))
    namespace = {}
    exec(compile(code, '<generated>', 'exec'), namespace)
    mod = namespace['mod']
    assert mod._buses['x'].indices == (0, 1, 2, 3)
    assert mod.nets.x._index_vector == tuple(('x', i) for i in range(8))
    header = mod.generate_subckt(0).splitlines()[0]
    assert header.split() == ['.subckt', 'wide', 'x[0]', 'x[1]', 'x[2]', 'x[3]']


@pytest.mark.parametrize('direct', [False, True])
@pytest.mark.parametrize('name', [
    'Module', 'Port', 'PortDir', 'PortRole', 'VoltageDomain', 'Variable',
    'range', 'slice', 'Net', 'load_pdk', 'i', 'pdk', 'mod', 'ckt', 'os',
    'analysis', 'sources', 'elements', '_pdk_root', '_lib_path',
])
def test_domain_binding_preserves_identity_and_shared_ports(tmp_path, direct, name):
    spec = {
        'circuit_name': 'safe_domain', 'output_dir': str(tmp_path),
        'voltage_domains': {name: {'nominal': 1.8},
                            'AVDD': {'nominal': 0.9}},
        'ports': {'a': {'dir': 'input', 'domain': name},
                  'b': {'dir': 'output', 'role': 'analog', 'domain': name},
                  'bus': {'dir': 'input', 'bits': [2, 4], 'domain': name}},
        'variables': [{'name': 'bias', 'value': 1.0}],
        'devices': [{'type': 'nmos1p8', 'instance': 'mn',
                     'nets': {'d': 'b', 'g': 'a', 's': 'gnd', 'b': 'gnd'},
                     'params': {'w': 1.0, 'l': 0.15}}],
        'ideal_elements': [{'type': 'resistor', 'instance': 'r',
                            'nets': {'p': 'b', 'n': 'gnd'},
                            'params': {'value': 1000}}],
        'sources': [{'type': 'VoltageSource', 'instance': 'vs',
                     'positive': 'a', 'negative': 'gnd',
                     'dc': {'variable': 'bias'}}],
    }
    if direct:
        from dataclasses import replace
        spec['voltage_domains'] = {'AVDD': {'nominal': 1.8}}
        for port in spec['ports'].values():
            port['domain'] = 'AVDD'
        spec = CircuitSpec.from_dict(spec)
        domain = VoltageDomain(name=name, nominal=1.8)
        spec.voltage_domains = {
            'different_mapping_key': domain,
            'AVDD': VoltageDomain(name='AVDD', nominal=0.9)}
        spec.ports = {key: replace(port, domain=domain)
                      for key, port in spec.ports.items()}
    code = Generator().generate(spec)
    assert code == Generator().generate(spec)
    ns = {}
    exec(compile(code, '<generated>', 'exec'), ns)
    ports = ns['mod'].pin_list
    domain = ports['a'].domain
    assert domain.name == name
    assert domain.nominal == 1.8
    assert ns['AVDD'].name == 'AVDD'
    assert ns['AVDD'].nominal == 0.9
    assert ns['AVDD'] is not domain
    assert ports['b'].domain is domain
    assert ports['bus[2]'].domain is domain
    assert ports['bus[3]'].domain is domain


def test_ordinary_domain_output_bytes():
    import hashlib
    code = Generator().generate({
        'circuit_name': 'x',
        'voltage_domains': {'AVDD': {'nominal': 1.8}},
        'ports': {'vdd': {'dir': 'input', 'domain': 'AVDD'}},
    })
    assert hashlib.sha256(code.encode()).hexdigest() == (
        '735a2d2ee5beb3ac13a4f0d18a452837a2ca824591f82b993beabb0069472cf5')


@pytest.mark.parametrize('direct', [False, True])
@pytest.mark.parametrize('name', ['class', 'vdd-core', '1.8V', 'with space'])
@pytest.mark.parametrize('exists', [False, True])
def test_invalid_domain_refuses_before_file_write(tmp_path, direct, name, exists):
    target = tmp_path / 'generated.py'
    if exists:
        target.write_text('keep me')
    spec = {'circuit_name': 'x', 'ports': {},
            'voltage_domains': {name: {'nominal': 1.8}}}
    if direct:
        spec = CircuitSpec('x', ports={}, voltage_domains={
            'valid_key': VoltageDomain(name=name, nominal=1.8)})
    with pytest.raises(ValueError, match='identifier|keyword'):
        Generator().generate_to_file(spec, str(target))
    assert target.read_text() == 'keep me' if exists else not target.exists()


@pytest.mark.parametrize('direct', [False, True])
def test_unicode_domain_bindings_preserve_distinct_authored_names(tmp_path, direct):
    names = ['𝑀odule', 'K', 'K', 'domain_Module', 'domain_𝑀odule']
    spec = {
        'circuit_name': 'unicode_domains', 'output_dir': str(tmp_path),
        'voltage_domains': {name: {'nominal': 1.0 + i}
                            for i, name in enumerate(names)},
        'ports': {f'p{i}': {'dir': 'input', 'domain': name}
                  for i, name in enumerate(names)},
    }
    if direct:
        spec = CircuitSpec.from_dict(spec)
        spec.voltage_domains = {f'key{i}': domain for i, domain
                                in enumerate(spec.voltage_domains.values())}
    code = Generator().generate(spec)
    assert code == Generator().generate(spec)
    ns = {}
    exec(compile(code, '<generated>', 'exec'), ns)
    for i, name in enumerate(names):
        domain = ns['mod'].pin_list[f'p{i}'].domain
        assert domain.name == name
        assert domain.nominal == 1.0 + i
