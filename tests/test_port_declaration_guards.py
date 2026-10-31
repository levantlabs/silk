"""TASK-37 — the three port-declaration guards on ``define_module_ports``.

The audit cells P5/P2/P4 in ``test_port_wiring_matrix.py`` record the
before/after verdicts; this module is the focused coverage for the guards
themselves:

#1 two port names differing only by case are ONE SPICE node (ngspice folds to
   lower, Xyce to upper), so the second parent connection is silently dropped
   — the declaration raises and names both spellings;
#2 a direction is normalized through ``Port._parse_direction`` or rejected —
   nothing is stored verbatim any more;
#3 redeclaring a port never silently overwrites it.
"""
import pytest

from silk.circuit.circuit_elements import Port, PortDir, PortRole
from silk.circuit.modules import Module
from silk.circuit.voltage_domain import VoltageDomain
from silk.exceptions import DuplicateNameError, CircuitError


def _module(pdk, tmp_path, name='dut'):
    return Module('PDK_skywater130', str(tmp_path / name), f'{name}.cir',
                  pdk_details=pdk, module_name=name)


# ---------------------------------------------------------------------------
# AC #1 — case-duplicate port names
# ---------------------------------------------------------------------------

class TestCaseDuplicatePorts:

    def test_within_one_call_raises_naming_both(self, pdk, tmp_path):
        m = _module(pdk, tmp_path)
        with pytest.raises(DuplicateNameError) as exc:
            m.define_module_ports({'Vin': PortDir.IN, 'vin': PortDir.IN})
        msg = str(exc.value)
        assert "'vin'" in msg and "'Vin'" in msg and "'dut'" in msg
        assert list(m.pin_list) == ['Vin']

    def test_across_calls_raises(self, pdk, tmp_path):
        m = _module(pdk, tmp_path)
        m.define_module_ports({'VDD': Port.supply()})
        with pytest.raises(DuplicateNameError, match="differs only by case"):
            m.define_module_ports({'vdd': Port.supply()})
        assert list(m.pin_list) == ['VDD']

    def test_mixed_case_variant_of_a_bus_bit_raises(self, pdk, tmp_path):
        m = _module(pdk, tmp_path)
        m.define_module_ports({'d': Port.digital_input(bits=2)})
        assert list(m.pin_list) == ['d[0]', 'd[1]']
        with pytest.raises(DuplicateNameError, match="differs only by case"):
            m.define_module_ports({'D[1]': PortDir.IN})

    def test_bus_expansion_onto_case_variant_pin_raises(self, pdk, tmp_path):
        m = _module(pdk, tmp_path)
        m.define_module_ports({'D[0]': PortDir.IN})
        with pytest.raises(CircuitError, match="expansion collides with existing pin"):
            m.define_module_ports({'d': Port.digital_input(bits=2)})

    def test_bus_expansion_onto_identical_pin_still_raises(self, pdk, tmp_path):
        m = _module(pdk, tmp_path)
        m.define_module_ports({'d[0]': PortDir.IN})
        with pytest.raises(CircuitError, match="expansion collides with existing pin"):
            m.define_module_ports({'d': Port.digital_input(bits=2)})

    def test_legacy_define_module_nets_is_guarded_too(self, pdk, tmp_path):
        m = _module(pdk, tmp_path)
        with pytest.raises(DuplicateNameError, match="differs only by case"):
            m.define_module_nets(['vin', 'vout', 'VIN'])

    def test_distinct_names_are_unaffected(self, pdk, tmp_path):
        m = _module(pdk, tmp_path)
        m.define_module_ports({'Vin': Port.analog_input(), 'vinp': Port.analog_input(),
                               'vout': Port.analog_output(), 'vdd': Port.supply(),
                               'gnd': Port.ground()})
        assert list(m.pin_list) == ['Vin', 'vinp', 'vout', 'vdd', 'gnd']


# ---------------------------------------------------------------------------
# AC #2 — direction validation
# ---------------------------------------------------------------------------

class TestDirectionNormalization:

    @pytest.mark.parametrize('given,expected', [
        ('in', PortDir.IN), ('input', PortDir.IN), ('INPUT', PortDir.IN),
        ('out', PortDir.OUT), ('output', PortDir.OUT), ('Out', PortDir.OUT),
        ('inout', PortDir.INOUT), ('INOUT', PortDir.INOUT),
        ('power', PortDir.POWER), ('ground', PortDir.GROUND),
        (PortDir.IN, PortDir.IN), (None, PortDir.INOUT),
    ])
    def test_accepted_spellings_normalize_to_portdir(self, pdk, tmp_path, given, expected):
        m = _module(pdk, tmp_path)
        m.define_module_ports({'a': given})
        assert m.pin_list['a'].direction is expected

    def test_string_direction_inside_tuple_and_port_descriptor(self, pdk, tmp_path):
        m = _module(pdk, tmp_path)
        m.define_module_ports({'a': ('out', PortRole.ANALOG),
                               'b': Port(dir='in', role=PortRole.DIGITAL)})
        assert m.pin_list['a'].direction is PortDir.OUT
        assert m.pin_list['b'].direction is PortDir.IN

    def test_unknown_string_raises_naming_port_and_spellings(self, pdk, tmp_path):
        m = _module(pdk, tmp_path)
        with pytest.raises(ValueError) as exc:
            m.define_module_ports({'b': 'inp'})
        msg = str(exc.value)
        assert "port 'b'" in msg and "'inp'" in msg
        for spelling in ('in', 'out', 'inout', 'input', 'output', 'power', 'ground'):
            assert f"'{spelling}'" in msg or f"{spelling}'" in msg
        assert 'b' not in m.pin_list

    def test_non_string_direction_raises_typeerror(self, pdk, tmp_path):
        m = _module(pdk, tmp_path)
        with pytest.raises(TypeError, match="port 'a'"):
            m.define_module_ports({'a': 3})
        assert 'a' not in m.pin_list

    def test_bus_direction_string_normalizes_on_every_bit(self, pdk, tmp_path):
        m = _module(pdk, tmp_path)
        m.define_module_ports({'d': Port(dir='out', bits=2)})
        assert [p.direction for p in m.pin_list.values()] == [PortDir.OUT, PortDir.OUT]
        assert m._buses['d'].direction is PortDir.OUT


# ---------------------------------------------------------------------------
# AC #3 — redeclaration
# ---------------------------------------------------------------------------

class TestRedeclare:

    def test_changed_direction_raises_showing_old_and_new(self, pdk, tmp_path):
        m = _module(pdk, tmp_path)
        m.define_module_ports({'a': PortDir.IN})
        with pytest.raises(DuplicateNameError) as exc:
            m.define_module_ports({'a': PortDir.OUT})
        msg = str(exc.value)
        assert 'input' in msg and 'output' in msg
        assert m.pin_list['a'].direction is PortDir.IN

    def test_changed_role_raises(self, pdk, tmp_path):
        m = _module(pdk, tmp_path)
        m.define_module_ports({'a': Port.analog_input()})
        with pytest.raises(DuplicateNameError, match="already declared"):
            m.define_module_ports({'a': Port.digital_input()})
        assert m.pin_list['a'].role is PortRole.ANALOG

    def test_changed_domain_raises(self, pdk, tmp_path):
        m = _module(pdk, tmp_path)
        avdd = VoltageDomain('avdd', 1.8)
        dvdd = VoltageDomain('dvdd', 1.2)
        m.define_module_ports({'a': Port.analog_input(avdd)})
        with pytest.raises(DuplicateNameError, match="already declared"):
            m.define_module_ports({'a': Port.analog_input(dvdd)})
        assert m.pin_list['a'].domain is avdd

    def test_identical_redeclare_is_a_noop_that_keeps_the_wiring(self, pdk, tmp_path):
        m = _module(pdk, tmp_path)
        avdd = VoltageDomain('avdd', 1.8)
        m.define_module_ports({'a': Port.analog_input(avdd)})
        pin = m.pin_list['a']
        pin.net = 'n1'
        m.define_module_ports({'a': Port.analog_input(avdd)})
        assert m.pin_list['a'] is pin and m.pin_list['a'].net == 'n1'

    def test_equivalent_spelling_of_the_same_direction_is_a_noop(self, pdk, tmp_path):
        m = _module(pdk, tmp_path)
        m.define_module_ports({'a': PortDir.IN})
        m.pin_list['a'].net = 'n1'
        m.define_module_ports({'a': 'input'})       # same direction, other spelling
        assert m.pin_list['a'].net == 'n1'


# ---------------------------------------------------------------------------
# The deck the audit measured can no longer be built
# ---------------------------------------------------------------------------

def test_casedup_subckt_can_no_longer_reach_the_netlist(pdk, tmp_path):
    """The 2026-09-02 repro emitted '.subckt casedup Vin vin vout vdd gnd' and
    'xu1 a b y vdd gnd casedup'; ngspice folded Vin/vin into one node and
    dropped the second connection (i(Vb)=0). The declaration now raises, so no
    such deck exists."""
    cd = _module(pdk, tmp_path, 'casedup')
    with pytest.raises(DuplicateNameError):
        cd.define_module_ports({'Vin': Port.analog_input(), 'vin': Port.analog_input(),
                                'vout': Port.analog_output(), 'vdd': Port.supply(),
                                'gnd': Port.ground()})


class TestBusOwnership:
    @pytest.mark.parametrize('naming,member', [('bracketed', 'd[0]'), ('underscore', 'd_0')])
    @pytest.mark.parametrize('reverse', [False, True])
    @pytest.mark.parametrize('scalar_alias,same_call,case', [
        (False, same_call, case) for same_call in (False, True) for case in (False, True)
    ] + [(True, False, False), (True, False, True), (True, True, True)])
    def test_separate_port_claim_refuses(self, tmp_path, naming, member,
                                        scalar_alias, reverse, same_call, case):
        m = Module(None, str(tmp_path), 'owner.cir', pdk_details=None, module_name='owner')
        scalar = 'd' if scalar_alias else member
        if case:
            scalar = scalar.upper()
        declarations = [('d', Port(dir=PortDir.IN, bits=2, naming=naming)),
                        (scalar, PortDir.IN)]
        if reverse:
            declarations.reverse()
        with pytest.raises(DuplicateNameError) as exc:
            if same_call:
                m.define_module_ports(dict(declarations))
            else:
                for key, value in declarations:
                    m.define_module_ports({key: value})
        assert 'owner' in str(exc.value)
        assert 'bus' in str(exc.value)
        assert scalar in str(exc.value)
        assert list(m.pin_list) == ([scalar] if reverse else
                                   (['d[0]', 'd[1]'] if naming == 'bracketed' else ['d_0', 'd_1']))

    def test_identical_bus_preserves_identity(self, tmp_path):
        m = Module(None, str(tmp_path), 'repeat.cir', pdk_details=None, module_name='repeat')
        declaration = {'d': Port(dir=PortDir.IN, bits=2)}
        m.define_module_ports(declaration)
        pins = dict(m.pin_list)
        bus = m.nets.d
        m.add_net('d[0]')
        m.define_module_ports(declaration)
        assert m.nets.d is bus
        assert all(m.pin_list[k] is v for k, v in pins.items())
        assert m.net_dict['d[0]'] is bus[0]

    @pytest.mark.parametrize('reverse', [False, True])
    @pytest.mark.parametrize('entry', ['module', 'graph', 'passive'])
    def test_internal_alias_refuses(self, tmp_path, reverse, entry):
        from silk.elements import ElementLibrary
        m = Module(None, str(tmp_path), 'internal.cir', pdk_details=None, module_name='internal')
        def scalar():
            if entry == 'module':
                m.add_net('D')
            elif entry == 'graph':
                m._element_mgr.add_net('D')
            else:
                m.add(ElementLibrary().resistor('r', p='D', n='0', value=1000))
        with pytest.raises(DuplicateNameError):
            if reverse:
                scalar()
                m.add_net('d', bits=2)
            else:
                m.add_net('d', bits=2)
                scalar()

    @pytest.mark.parametrize('reverse', [False, True])
    @pytest.mark.parametrize('equal', [False, True])
    def test_mapping_overlap_leaves_parent_unchanged(self, tmp_path, reverse, equal):
        child = Module(None, str(tmp_path), 'child.cir', pdk_details=None, module_name='child')
        child.define_module_ports({'d': Port(dir=PortDir.IN, bits=2)})
        parent = Module(None, str(tmp_path), 'parent.cir', pdk_details=None)
        parent.add_net('x')
        items = [('d', 'x'), ('d[0]', 'x' if equal else 'other')]
        if reverse:
            items.reverse()
        before = (dict(parent.net_dict), dict(parent.elem_dict))
        with pytest.raises(DuplicateNameError, match=r'd\[0\]'):
            parent.add_module(child, 'u', dict(items))
        assert dict(parent.net_dict) == before[0]
        assert dict(parent.elem_dict) == before[1]
        assert all(not n.pins for n in parent.net_dict.values())

    @pytest.mark.parametrize('change', [
        {'bits': 3}, {'bits': range(1, 3)}, {'naming': 'underscore'},
        {'dir': PortDir.OUT}, {'role': PortRole.DIGITAL},
        {'domain': VoltageDomain('vdd', nominal=1.8)},
    ])
    def test_changed_bus_preflights_without_mutation(self, tmp_path, change):
        m = Module(None, str(tmp_path), 'atomic.cir', None, module_name='atomic')
        m.define_module_ports({'d': Port(dir=PortDir.IN, bits=2)})
        m.add_net('d[0]')
        before = (dict(m.pin_list), dict(m.net_dict), dict(m._buses), dict(m._bus_handles))
        args = {'dir': PortDir.IN, 'bits': 2, **change}
        with pytest.raises(DuplicateNameError):
            m.define_module_ports({'earlier': PortDir.IN, 'd': Port(**args)})
        assert list(m.pin_list) == [*before[0], 'earlier']
        assert all(m.pin_list[k] is v for k, v in before[0].items())
        assert dict(m.net_dict) == before[1]
        assert m._buses == before[2]
        assert m._bus_handles == before[3]

    def test_domain_with_same_name_changed_contract_refuses(self, tmp_path):
        m = Module(None, str(tmp_path), 'domain.cir', None, module_name='domain')
        m.define_module_ports({'d': Port(dir=PortDir.IN, bits=2,
                                       domain=VoltageDomain('vdd', nominal=1.8))})
        with pytest.raises(DuplicateNameError):
            m.define_module_ports({'d': Port(dir=PortDir.IN, bits=2,
                                           domain=VoltageDomain('vdd', nominal=3.3))})

    @pytest.mark.parametrize('naming,member', [('bracketed', 'd[0]'), ('underscore', 'd_0')])
    def test_standalone_member_and_unowned_index_are_legal(self, tmp_path, naming, member):
        m = Module(None, str(tmp_path), 'scalar.cir', None, module_name='scalar')
        m.define_module_ports({member: PortDir.IN, 'd': Port(dir=PortDir.IN,
                                                           bits=range(1, 3), naming=naming)})
        assert member in m.pin_list

    @pytest.mark.parametrize('internal', [False, True])
    @pytest.mark.parametrize('scalar_port', [False, True])
    def test_bus_cannot_claim_existing_scalar_alias(self, tmp_path, internal, scalar_port):
        m = Module(None, str(tmp_path), 'alias.cir', None, module_name='alias')
        if scalar_port:
            m.define_module_ports({'D': PortDir.IN})
        else:
            m.add_net('D')
        before = dict(m.net_dict)
        with pytest.raises(DuplicateNameError):
            if internal:
                m.add_net('d', bits=2)
            else:
                m.define_module_ports({'d': Port(dir=PortDir.IN, bits=2)})
        assert not m._buses
        assert not m._bus_handles
        assert dict(m.net_dict) == before

    @pytest.mark.parametrize('internal', [False, True])
    def test_scalar_port_cannot_claim_bus_alias(self, tmp_path, internal):
        m = Module(None, str(tmp_path), 'alias.cir', None, module_name='alias')
        if internal:
            m.add_net('d', bits=2)
        else:
            m.define_module_ports({'d': Port(dir=PortDir.IN, bits=2)})
        before = dict(m.pin_list or {})
        with pytest.raises(DuplicateNameError):
            m.define_module_ports({'D': PortDir.IN})
        assert m.pin_list == before

    @pytest.mark.parametrize('entry', ['device', 'source', 'passive'])
    @pytest.mark.parametrize('rewire', [False, True])
    def test_implicit_alias_refusal_preserves_bindings(self, tmp_path, entry, rewire):
        from silk.elements import ElementLibrary
        from silk.elements.passives import Resistor
        from silk.sources import VoltageSource
        m = Module(None, str(tmp_path), 'rewire.cir', None, module_name='rewire')
        m.add_net('d', bits=2)
        if rewire:
            if entry == 'device':
                m.add(ElementLibrary().resistor('r', p='old', n='0', value=1000))
            elif entry == 'passive':
                m.add(Resistor('r', pos='old', neg='0', resistance=1000))
            else:
                m.add_source(VoltageSource('v', positive='old', negative='0', dc=1))
        before = {key: list(net.pins) for key, net in m.net_dict.items()}
        with pytest.raises(DuplicateNameError):
            if rewire:
                if entry == 'device':
                    m.elem_dict['r']['device'].p = 'd'
                elif entry == 'passive':
                    m.elem_dict['r']['device'].pos = 'd'
                else:
                    m.connect_source('v', ['positive'], ['d'])
            elif entry == 'device':
                m.add(ElementLibrary().resistor('r', p='d', n='0', value=1000))
            elif entry == 'passive':
                m.add_passive(Resistor('r', pos='d', neg='0', resistance=1000))
            else:
                m.add_source(VoltageSource('v', positive='d', negative='0', dc=1))
        assert {key: list(net.pins) for key, net in m.net_dict.items()} == before

    @pytest.mark.parametrize('internal', [False, True])
    @pytest.mark.parametrize('reverse', [False, True])
    def test_bus_alias_cannot_take_another_bus_member(self, tmp_path, internal, reverse):
        m = Module(None, str(tmp_path), 'buses.cir', None, module_name='buses')
        names = ['d', 'd_0']
        if reverse:
            names.reverse()
        def declare(name):
            if internal:
                m.add_net(name, bits=2, naming='underscore')
            else:
                m.define_module_ports({name: Port(dir=PortDir.IN, bits=2,
                                                 naming='underscore')})
        declare(names[0])
        before = (dict(m.pin_list or {}), dict(m.net_dict), dict(m._buses))
        with pytest.raises(DuplicateNameError):
            declare(names[1])
        assert (dict(m.pin_list or {}), dict(m.net_dict), dict(m._buses)) == before

    def test_identical_internal_bus_keeps_handle_and_bits(self, tmp_path):
        m = Module(None, str(tmp_path), 'nets.cir', None, module_name='nets')
        bus = m.add_net('d', bits=slice(4, 6), naming='underscore')
        assert m.add_net('d', bits=range(4, 6), naming='underscore') is bus
        assert list(m.net_dict) == ['d_4', 'd_5']
        assert m.add_net('D_4') is bus[4]

    @pytest.mark.parametrize('copy_method', ['deepcopy', 'pickle'])
    def test_ownership_survives_copy_without_new_state(self, tmp_path, copy_method):
        import copy
        import pickle
        m = Module(None, str(tmp_path), 'copy.cir', None, module_name='copy')
        m.define_module_ports({'d': Port(dir=PortDir.IN, bits=2)})
        m.add_net('d[0]')
        cloned = copy.deepcopy(m) if copy_method == 'deepcopy' else pickle.loads(pickle.dumps(m))
        handle = cloned.nets.d
        cloned.define_module_ports({'d': Port(dir=PortDir.IN, bits=2)})
        assert cloned.nets.d is handle
        assert cloned.add_net('d[0]') is handle[0]
        with pytest.raises(DuplicateNameError):
            cloned.define_module_ports({'d[0]': PortDir.IN})

    def test_source_rewire_keeps_undefined_non_string_diagnostic(self, tmp_path):
        from silk.sources import VoltageSource
        from silk.exceptions import UndeclaredNetError
        m = Module(None, str(tmp_path), 'source.cir', None, module_name='source')
        m.add_net('d', bits=2)
        m.add_source(VoltageSource('v', positive='old', negative='0', dc=1))
        with pytest.raises(UndeclaredNetError, match='undefined'):
            m.connect_source('v', ['positive'], [123])
        assert m.sig_sources_dict['v'].positive == 'old'
