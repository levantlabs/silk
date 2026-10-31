"""Tests for first-class bus port support (spec 2026-05-08-bus-ports-design.md)."""

import pytest


@pytest.mark.parametrize('naming', ['bracketed', 'underscore'])
def test_numeric_formal_bus_base_is_not_a_ground_net(tmp_path, naming):
    from silk.circuit.modules import Module
    from silk.circuit.ports import Port
    mod = Module(None, str(tmp_path), 'ports.cir', None, module_name='busblock')
    with pytest.raises(ValueError, match="block 'busblock'.*port named '0'.*global"):
        mod.define_module_ports({'0': Port.digital_input(bits=range(4, 6), naming=naming)})
    assert not mod.pin_list
    assert not mod._buses


@pytest.mark.parametrize('naming,expected', [
    ('bracketed', ['data[4]', 'data[5]']), ('underscore', ['data_4', 'data_5'])])
def test_safe_bus_adoption_preserves_declared_numbering(tmp_path, naming, expected):
    from silk.circuit.modules import Module
    from silk.circuit.ports import Port
    mod = Module(None, str(tmp_path), 'ports.cir', None, module_name='busblock')
    mod.add_net('data', bits=range(4, 6), naming=naming)
    mod.define_module_ports({'data': Port.digital_input(bits=range(4, 6), naming=naming)})
    assert list(mod.pin_list) == expected
    assert mod._buses['data'].indices == (4, 5)
    assert all(name in mod.net_dict for name in expected)


# ---------------------------------------------------------------------------
# §3.1 — BusSpec value-type contract
# ---------------------------------------------------------------------------

class TestBusSpec:

    def test_busspec_basic_construction(self):
        from silk.circuit.bus_spec import BusSpec
        from silk.circuit.circuit_elements import PortDir, PortRole
        spec = BusSpec(
            name='d',
            indices=(0, 1, 2, 3),
            direction=PortDir.OUT,
            role=PortRole.DIGITAL,
            domain='vdd_1p8',
            naming='bracketed',
        )
        assert spec.name == 'd'
        assert spec.indices == (0, 1, 2, 3)
        assert spec.naming == 'bracketed'

    def test_busspec_is_frozen(self):
        from silk.circuit.bus_spec import BusSpec
        from silk.circuit.circuit_elements import PortDir
        from dataclasses import FrozenInstanceError
        spec = BusSpec(name='d', indices=(0, 1), direction=PortDir.IN,
                      role=None, domain=None, naming='bracketed')
        with pytest.raises(FrozenInstanceError):
            spec.name = 'x'

    def test_busspec_value_typed_only(self):
        """Spec §3.1: every field must be one of str | int | tuple[int|str] | enum.

        This is the deepcopy-safety contract.
        """
        from silk.circuit.bus_spec import BusSpec
        from silk.circuit.circuit_elements import PortDir, PortRole
        from dataclasses import fields
        spec = BusSpec(name='d', indices=(8, 9, 10), direction=PortDir.OUT,
                       role=PortRole.DIGITAL, domain='avdd', naming='bracketed')
        for f in fields(spec):
            v = getattr(spec, f.name)
            allowed = (str, int, tuple, type(None))
            from enum import Enum
            assert v is None or isinstance(v, allowed) or isinstance(v, Enum), (
                f"BusSpec.{f.name} = {v!r} is not value-typed"
            )

    def test_busspec_width_property(self):
        from silk.circuit.bus_spec import BusSpec
        from silk.circuit.circuit_elements import PortDir
        spec = BusSpec(name='d', indices=(0, 1, 2, 3), direction=PortDir.OUT,
                      role=None, domain=None, naming='bracketed')
        assert spec.width == 4


# ---------------------------------------------------------------------------
# §2.1 — indices_from_bits normalization
# ---------------------------------------------------------------------------

class TestIndicesFromBits:

    def test_zero_raises(self):
        from silk.circuit.bus_spec import indices_from_bits
        with pytest.raises(ValueError, match='zero|negative'):
            indices_from_bits(0)

    def test_negative_int_raises(self):
        from silk.circuit.bus_spec import indices_from_bits
        with pytest.raises(ValueError):
            indices_from_bits(-1)

    def test_range_step_not_one_raises(self):
        from silk.circuit.bus_spec import indices_from_bits
        with pytest.raises(ValueError, match='step'):
            indices_from_bits(range(0, 10, 2))

    def test_empty_range_raises(self):
        from silk.circuit.bus_spec import indices_from_bits
        with pytest.raises(ValueError):
            indices_from_bits(range(5, 5))

    def test_reversed_slice_raises(self):
        from silk.circuit.bus_spec import indices_from_bits
        with pytest.raises(ValueError):
            indices_from_bits(slice(10, 5))

    def test_slice_without_stop_raises(self):
        from silk.circuit.bus_spec import indices_from_bits
        with pytest.raises(ValueError, match='stop'):
            indices_from_bits(slice(0, None))

    def test_non_int_range_slice_raises_typeerror(self):
        from silk.circuit.bus_spec import indices_from_bits
        with pytest.raises(TypeError):
            indices_from_bits('not a valid bits arg')

    def test_returns_tuple(self):
        from silk.circuit.bus_spec import indices_from_bits
        assert indices_from_bits(4) == (0, 1, 2, 3)
        assert indices_from_bits(range(8, 12)) == (8, 9, 10, 11)
        assert indices_from_bits(slice(8, 12)) == (8, 9, 10, 11)


# ---------------------------------------------------------------------------
# §2.1 — bit-name expansion
# ---------------------------------------------------------------------------

class TestExpandBits:

    def test_expand_int_default_naming(self):
        from silk.circuit.bus_spec import expand_bits
        names = expand_bits('d', 4, naming='bracketed')
        assert names == ('d[0]', 'd[1]', 'd[2]', 'd[3]')

    def test_expand_range(self):
        from silk.circuit.bus_spec import expand_bits
        names = expand_bits('addr', range(8, 16), naming='bracketed')
        assert names == tuple(f'addr[{i}]' for i in range(8, 16))

    def test_expand_slice(self):
        from silk.circuit.bus_spec import expand_bits
        names = expand_bits('addr', slice(8, 16), naming='bracketed')
        assert names == tuple(f'addr[{i}]' for i in range(8, 16))

    def test_expand_underscore_naming(self):
        from silk.circuit.bus_spec import expand_bits
        names = expand_bits('dbg', 4, naming='underscore')
        assert names == ('dbg_0', 'dbg_1', 'dbg_2', 'dbg_3')

    def test_expand_zero_width_raises(self):
        from silk.circuit.bus_spec import expand_bits
        with pytest.raises(ValueError, match='zero'):
            expand_bits('d', 0, naming='bracketed')

    def test_expand_negative_raises(self):
        from silk.circuit.bus_spec import expand_bits
        with pytest.raises(ValueError, match='negative|zero'):
            expand_bits('d', -1, naming='bracketed')

    def test_expand_slice_with_step_raises(self):
        from silk.circuit.bus_spec import expand_bits
        with pytest.raises(ValueError, match='step'):
            expand_bits('d', slice(0, 10, 2), naming='bracketed')

    def test_expand_unknown_naming_raises(self):
        from silk.circuit.bus_spec import expand_bits
        with pytest.raises(ValueError, match='naming'):
            expand_bits('d', 4, naming='something_else')


# ---------------------------------------------------------------------------
# §2.1 — Port factories accept `bits=`
# ---------------------------------------------------------------------------

class TestPortBitsField:

    def test_port_default_bits_is_none(self):
        from silk.circuit.circuit_elements import Port, PortDir
        p = Port(dir=PortDir.IN)
        assert p.bits is None
        assert p.naming == 'bracketed'

    def test_port_bits_accepts_int(self):
        from silk.circuit.circuit_elements import Port, PortDir
        p = Port(dir=PortDir.IN, bits=4)
        assert p.bits == 4

    def test_digital_input_bits(self):
        from silk.circuit.circuit_elements import Port, PortDir, PortRole
        p = Port.digital_input(bits=4)
        assert p.dir == PortDir.IN
        assert p.role == PortRole.DIGITAL
        assert p.bits == 4

    def test_digital_output_bits_with_range(self):
        from silk.circuit.circuit_elements import Port, PortDir
        p = Port.digital_output(bits=range(8, 16))
        assert p.bits == range(8, 16)
        assert p.dir == PortDir.OUT

    def test_digital_output_bits_with_slice(self):
        from silk.circuit.circuit_elements import Port
        p = Port.digital_output(bits=slice(8, 16))
        assert p.bits == slice(8, 16)

    def test_naming_underscore_override(self):
        from silk.circuit.circuit_elements import Port
        p = Port.digital_output(bits=4, naming='underscore')
        assert p.naming == 'underscore'

    def test_supply_rejects_bits(self):
        from silk.circuit.circuit_elements import Port
        with pytest.raises(TypeError, match='bits'):
            Port.supply(bits=4)

    def test_ground_rejects_bits(self):
        from silk.circuit.circuit_elements import Port
        with pytest.raises(TypeError, match='bits'):
            Port.ground(bits=4)

    def test_clock_input_bits(self):
        from silk.circuit.circuit_elements import Port, PortRole
        p = Port.clock_input(bits=4)
        assert p.bits == 4
        assert p.role == PortRole.CLOCK

    def test_analog_output_bits(self):
        from silk.circuit.circuit_elements import Port
        p = Port.analog_output(bits=2)
        assert p.bits == 2

    def test_analog_input_bits(self):
        from silk.circuit.circuit_elements import Port, PortRole
        p = Port.analog_input(bits=8)
        assert p.bits == 8
        assert p.role == PortRole.ANALOG

    def test_clock_output_bits(self):
        from silk.circuit.circuit_elements import Port, PortRole
        p = Port.clock_output(bits=4)
        assert p.bits == 4
        assert p.role == PortRole.CLOCK


# ---------------------------------------------------------------------------
# §3.2 — Net width awareness
# ---------------------------------------------------------------------------

class TestNetWidthAwareness:

    def test_scalar_net_width_is_one(self):
        from silk.circuit.net import Net
        n = Net('clk', module=None)
        assert n.width == 1
        assert n._index_vector is None
        assert n._underlying_pins is None

    def test_scalar_net_subscript_raises(self):
        from silk.circuit.net import Net
        n = Net('clk', module=None)
        with pytest.raises(TypeError):
            n[0]

    def test_bus_net_construction(self):
        """A bus-handle Net is constructed by Module.add_net(name, bits=4) —
        for now, test direct construction."""
        from silk.circuit.net import Net
        scalars = tuple(Net(f'd[{i}]', module=None) for i in range(4))
        bus = Net._make_bus_handle(name='d', underlying=scalars,
                                   index_vector=tuple(('d', i) for i in range(4)))
        assert bus.width == 4
        assert len(bus) == 4

    def test_bus_net_indexing_returns_underlying_scalar(self):
        from silk.circuit.net import Net
        scalars = tuple(Net(f'd[{i}]', module=None) for i in range(4))
        bus = Net._make_bus_handle(name='d', underlying=scalars,
                                   index_vector=tuple(('d', i) for i in range(4)))
        assert bus[0] is scalars[0]
        assert bus[3] is scalars[3]

    def test_bus_net_slice_returns_subhandle(self):
        from silk.circuit.net import Net
        scalars = tuple(Net(f'd[{i}]', module=None) for i in range(4))
        bus = Net._make_bus_handle(name='d', underlying=scalars,
                                   index_vector=tuple(('d', i) for i in range(4)))
        sub = bus[1:3]
        assert sub.width == 2
        assert sub[1] is scalars[1]
        assert sub[2] is scalars[2]
        assert sub._index_vector == (('d', 1), ('d', 2))

    def test_bus_net_slice_of_slice(self):
        from silk.circuit.net import Net
        scalars = tuple(Net(f'data[{i}]', module=None) for i in range(64, 128))
        bus = Net._make_bus_handle(
            name='data', underlying=scalars,
            index_vector=tuple(('data', i) for i in range(64, 128))
        )
        slice1 = bus[64:72]      # first 8 bits, indices 64..71
        slice2 = slice1[64:68]   # first 4 of those, indices 64..67
        assert slice2.width == 4
        assert slice2._index_vector == tuple(('data', i) for i in range(64, 68))

    def test_bus_net_reverse_slice(self):
        from silk.circuit.net import Net
        scalars = tuple(Net(f'd[{i}]', module=None) for i in range(4))
        bus = Net._make_bus_handle(name='d', underlying=scalars,
                                   index_vector=tuple(('d', i) for i in range(4)))
        rev = bus[::-1]
        assert rev.width == 4
        # Iteration yields reversed order
        assert list(rev) == [scalars[3], scalars[2], scalars[1], scalars[0]]

    def test_bus_net_iter(self):
        from silk.circuit.net import Net
        scalars = tuple(Net(f'd[{i}]', module=None) for i in range(4))
        bus = Net._make_bus_handle(name='d', underlying=scalars,
                                   index_vector=tuple(('d', i) for i in range(4)))
        assert list(bus) == list(scalars)

    def test_bus_net_pickle_roundtrip(self):
        """Bus handle pickle stability — frozen attribute names per §3.2."""
        import pickle
        from silk.circuit.net import Net
        scalars = tuple(Net(f'd[{i}]', module=None) for i in range(4))
        bus = Net._make_bus_handle(name='d', underlying=scalars,
                                   index_vector=tuple(('d', i) for i in range(4)))
        roundtripped = pickle.loads(pickle.dumps(bus))
        assert roundtripped.width == 4
        assert roundtripped._index_vector == bus._index_vector
        assert len(roundtripped._underlying_pins) == 4


# ---------------------------------------------------------------------------
# §2.1 — define_module_ports with bus declarations
# ---------------------------------------------------------------------------

class TestDefineModulePortsBus:

    def _make_module(self):
        from silk.circuit.modules import Module
        return Module(pdk_name='sky130', dir_folder='/tmp', netlistname='t.cir',
                      pdk_details=None, module_name='m')

    def test_bus_port_expands_pin_list(self):
        from silk.circuit.circuit_elements import Port
        m = self._make_module()
        m.define_module_ports({'d': Port.digital_output(bits=4)})
        assert list(m.pin_list.keys()) == ['d[0]', 'd[1]', 'd[2]', 'd[3]']

    def test_bus_port_records_busspec(self):
        from silk.circuit.circuit_elements import Port, PortDir, PortRole
        m = self._make_module()
        m.define_module_ports({'d': Port.digital_output(bits=4)})
        assert 'd' in m._buses
        spec = m._buses['d']
        assert spec.width == 4
        assert spec.indices == (0, 1, 2, 3)
        assert spec.direction == PortDir.OUT
        assert spec.role == PortRole.DIGITAL
        assert spec.naming == 'bracketed'

    def test_scalar_port_does_not_create_busspec(self):
        from silk.circuit.circuit_elements import Port
        m = self._make_module()
        m.define_module_ports({'clk': Port.digital_input()})
        assert 'clk' not in m._buses
        assert list(m.pin_list.keys()) == ['clk']

    def test_mixed_scalar_and_bus_ports(self):
        from silk.circuit.circuit_elements import Port
        m = self._make_module()
        m.define_module_ports({
            'clk': Port.clock_input(),
            'd':   Port.digital_output(bits=4),
            'rdy': Port.digital_output(),
        })
        # §10.1: insertion order interleaved, bits NOT clustered separately
        assert list(m.pin_list.keys()) == [
            'clk', 'd[0]', 'd[1]', 'd[2]', 'd[3]', 'rdy'
        ]

    def test_range_bits(self):
        from silk.circuit.circuit_elements import Port
        m = self._make_module()
        m.define_module_ports({'addr': Port.digital_input(bits=range(8, 12))})
        assert list(m.pin_list.keys()) == ['addr[8]', 'addr[9]', 'addr[10]', 'addr[11]']
        assert m._buses['addr'].indices == (8, 9, 10, 11)

    def test_underscore_naming(self):
        from silk.circuit.circuit_elements import Port
        m = self._make_module()
        m.define_module_ports({'dbg': Port.digital_output(bits=4, naming='underscore')})
        assert list(m.pin_list.keys()) == ['dbg_0', 'dbg_1', 'dbg_2', 'dbg_3']
        assert m._buses['dbg'].naming == 'underscore'

    def test_collision_with_existing_pin_raises(self):
        from silk.circuit.circuit_elements import Port
        from silk.exceptions import CircuitError
        m = self._make_module()
        m.define_module_ports({'d_0': Port.digital_input()})    # scalar named 'd_0'
        # Adding a bus 'd' with underscore naming would collide on 'd_0'
        with pytest.raises(CircuitError, match='collision|d_0'):
            m.define_module_ports({'d': Port.digital_output(bits=4, naming='underscore')})

    def test_zero_width_raises(self):
        from silk.circuit.circuit_elements import Port
        m = self._make_module()
        with pytest.raises(ValueError):
            m.define_module_ports({'d': Port.digital_output(bits=0)})

    def test_default_buses_attribute_is_empty_dict(self):
        m = self._make_module()
        assert m._buses == {}


# ---------------------------------------------------------------------------
# §2.2 — add_net with bits=
# ---------------------------------------------------------------------------

class TestAddNetBus:

    def _make_module(self):
        from silk.circuit.modules import Module
        return Module(pdk_name='sky130', dir_folder='/tmp', netlistname='t.cir',
                      pdk_details=None, module_name='m')

    def test_scalar_add_net_unchanged(self):
        m = self._make_module()
        n = m.add_net('clk')
        assert n.width == 1
        assert n.name == 'clk'

    def test_bus_add_net_returns_handle(self):
        m = self._make_module()
        bus = m.add_net('d', bits=4)
        assert bus.width == 4
        assert len(bus) == 4

    def test_bus_add_net_creates_scalar_entries(self):
        m = self._make_module()
        m.add_net('d', bits=4)
        # Each bit must exist as a real Net entry in net_dict
        for i in range(4):
            assert f'd[{i}]' in m.net_dict

    def test_bus_add_net_indexing(self):
        m = self._make_module()
        bus = m.add_net('d', bits=4)
        assert bus[0] is m.net_dict['d[0]']
        assert bus[3] is m.net_dict['d[3]']

    def test_bus_add_net_slice(self):
        m = self._make_module()
        bus = m.add_net('data', bits=192)
        sub = bus[64:128]
        assert sub.width == 64
        assert sub[64] is m.net_dict['data[64]']
        assert sub[127] is m.net_dict['data[127]']

    def test_bus_add_net_range(self):
        m = self._make_module()
        bus = m.add_net('addr', bits=range(8, 16))
        assert bus.width == 8
        for i in range(8, 16):
            assert f'addr[{i}]' in m.net_dict

    def test_bus_add_net_collision_raises(self):
        from silk.exceptions import CircuitError
        m = self._make_module()
        m.add_net('d_0')  # scalar
        with pytest.raises(CircuitError, match='collision|d_0'):
            m.add_net('d', bits=4, naming='underscore')


# ---------------------------------------------------------------------------
# §10.8 — pickle / deepcopy preserves _buses and bus-handle Nets
# ---------------------------------------------------------------------------

class TestPickleDeepcopy:

    def _make_module_with_bus(self):
        from silk.circuit.modules import Module
        from silk.circuit.circuit_elements import Port
        m = Module(pdk_name='sky130', dir_folder='/tmp', netlistname='t.cir',
                   pdk_details=None, module_name='m')
        m.define_module_ports({'d': Port.digital_output(bits=4)})
        m.add_net('data', bits=8)
        return m

    def test_deepcopy_preserves_buses(self):
        import copy
        m = self._make_module_with_bus()
        c = copy.deepcopy(m)
        # _buses must be reachable via the deepcopy walker, not silently
        # dropped by Module.__deepcopy__ fallback (modules.py:241).
        assert 'd' in c._buses
        assert c._buses['d'].width == 4
        # Bit pins also preserved
        assert 'd[0]' in c.pin_list
        assert 'd[3]' in c.pin_list
        # add_net creates scalar entries but doesn't populate _buses
        assert 'data[0]' in c.pin_list or 'data[0]' in c.net_dict
        assert 'data[7]' in c.pin_list or 'data[7]' in c.net_dict

    def test_pickle_preserves_buses(self):
        import pickle
        m = self._make_module_with_bus()
        c = pickle.loads(pickle.dumps(m))
        assert c._buses == m._buses
        assert list(c.pin_list.keys()) == list(m.pin_list.keys())

    def test_busspec_has_no_live_references(self):
        """§3.1 contract: BusSpec carries no references to live objects."""
        from dataclasses import fields
        from enum import Enum
        m = self._make_module_with_bus()
        spec = m._buses['d']
        for f in fields(spec):
            v = getattr(spec, f.name)
            assert v is None or isinstance(v, (str, int, tuple, Enum)), (
                f"BusSpec.{f.name} = {v!r} is not value-typed"
            )


# ---------------------------------------------------------------------------
# §2.4 — add_element with bus kwargs
# ---------------------------------------------------------------------------

class TestAddElementBus:

    def _make_modules(self):
        from silk.circuit.modules import Module
        from silk.circuit.circuit_elements import Port

        # Child with a bus port
        child = Module(pdk_name='sky130', dir_folder='/tmp', netlistname='c.cir',
                       pdk_details=None, module_name='child')
        child.define_module_ports({
            'clk': Port.clock_input(),
            'd':   Port.digital_output(bits=4),
        })

        # Parent
        parent = Module(pdk_name='sky130', dir_folder='/tmp', netlistname='p.cir',
                        pdk_details=None, module_name='parent')
        return child, parent

    def test_whole_bus_wiring(self):
        child, parent = self._make_modules()
        clk = parent.add_net('clk')
        out = parent.add_net('out', bits=4)
        parent.add_module(child, 'u1', clk=clk, d=out)
        # Per-bit pin.net assignments are populated correctly
        stored = parent.elem_dict['u1']['pin_list']
        for i in range(4):
            assert stored[f'd[{i}]'].net == f'out[{i}]'

    def test_width_mismatch_raises(self):
        from silk.exceptions import CircuitError
        child, parent = self._make_modules()
        clk = parent.add_net('clk')
        narrow = parent.add_net('narrow', bits=2)
        with pytest.raises(CircuitError, match='BUS_WIDTH_MISMATCH|width'):
            parent.add_module(child, 'u1', clk=clk, d=narrow)

    def test_slice_wiring(self):
        child, parent = self._make_modules()
        clk = parent.add_net('clk')
        wide = parent.add_net('wide', bits=8)
        parent.add_module(child, 'u1', clk=clk, d=wide[0:4])
        stored = parent.elem_dict['u1']['pin_list']
        for i in range(4):
            assert stored[f'd[{i}]'].net == f'wide[{i}]'

    def test_positional_binding_with_offset_range(self):
        """§7 case B — child declared bits=range(8,16); parent slice bits 0..7."""
        from silk.circuit.modules import Module
        from silk.circuit.circuit_elements import Port
        child = Module(pdk_name='sky130', dir_folder='/tmp', netlistname='c.cir',
                       pdk_details=None, module_name='child')
        child.define_module_ports({'d': Port.digital_input(bits=range(8, 16))})

        parent = Module(pdk_name='sky130', dir_folder='/tmp', netlistname='p.cir',
                        pdk_details=None, module_name='parent')
        pdat = parent.add_net('parent_d', bits=8)
        parent.add_module(child, 'u1', d=pdat[0:8])
        stored = parent.elem_dict['u1']['pin_list']
        # Positional matching: parent_d[0] ↔ d[8], parent_d[1] ↔ d[9], …
        for i, child_bit in enumerate(range(8, 16)):
            assert stored[f'd[{child_bit}]'].net == f'parent_d[{i}]'

    def test_multi_source_list_rejected(self):
        from silk.exceptions import CircuitError
        child, parent = self._make_modules()
        a = parent.add_net('a', bits=2)
        b = parent.add_net('b', bits=2)
        clk = parent.add_net('clk')
        # v1 forbids multi-source — concat is deferred to v1.1
        with pytest.raises((CircuitError, TypeError), match='concat|v1.1|multi'):
            parent.add_module(child, 'u1', clk=clk, d=[a, b])


# ---------------------------------------------------------------------------
# §2.5 — add_no_connect_port with bus port name
# ---------------------------------------------------------------------------

class TestNoConnectPortBus:

    def _setup(self):
        from silk.circuit.modules import Module, ModulePin
        from silk.circuit.circuit_elements import PortDir, PortRole
        from silk.circuit.bus_spec import BusSpec

        # Build a parent module with a submodule instance manually.
        # We'll skip add_element and just set up the elem_dict directly.
        parent = Module(pdk_name='sky130', dir_folder='/tmp', netlistname='p.cir',
                        pdk_details=None, module_name='parent')

        # Create a minimal child "submodule" object to act as the device
        child = Module(pdk_name='sky130', dir_folder='/tmp', netlistname='c.cir',
                       pdk_details=None, module_name='child')
        child.pin_list = {}  # Initialize pin_list

        # Give the child a 4-bit dbg port with both pin_list entries and _buses
        for i in range(4):
            child.pin_list[f'dbg[{i}]'] = ModulePin(
                name=f'dbg[{i}]', direction=PortDir.OUT, role=PortRole.DIGITAL
            )
        child._buses = {}
        child._buses['dbg'] = BusSpec(
            name='dbg', indices=(0, 1, 2, 3),
            direction=PortDir.OUT, role=PortRole.DIGITAL,
            domain=None, naming='bracketed'
        )

        # Manually register the instance in parent's elem_dict
        # (bypassing add_element since it requires pdk_details for submodules)
        parent._element_mgr.elem_dict['u1'] = {'device': child, 'type': 'module'}

        return parent

    def test_per_bit_no_connect_works_unchanged(self):
        parent = self._setup()
        marker = parent.add_no_connect_port('u1', 'dbg[2]', reason='unused')
        assert marker.reason == 'unused'

    def test_whole_bus_no_connect_fans_out(self):
        parent = self._setup()
        markers = parent.add_no_connect_port('u1', 'dbg', reason='wired in TB only')
        # Whole-bus form returns a list of NoConnectPort markers, one per bit
        assert isinstance(markers, list)
        assert len(markers) == 4
        for m in markers:
            assert m.reason == 'wired in TB only'

    def test_unknown_port_still_raises(self):
        from silk.exceptions import CircuitError
        parent = self._setup()
        with pytest.raises(CircuitError):
            parent.add_no_connect_port('u1', 'nonexistent', reason='x')


# ---------------------------------------------------------------------------
# §3.4 + §2.7 — SubcircuitInstance bus handle
# ---------------------------------------------------------------------------

class TestSubcircuitInstanceBus:

    def _setup(self):
        from silk.circuit.modules import Module
        from silk.circuit.circuit_elements import Port
        child = Module(pdk_name="sky130", dir_folder="/tmp", netlistname="c.cir",
                       pdk_details=None, module_name="child")
        child.define_module_ports({
            "clk": Port.clock_input(),
            "d":   Port.digital_output(bits=4),
        })
        parent = Module(pdk_name="sky130", dir_folder="/tmp", netlistname="p.cir",
                        pdk_details=None, module_name="parent")
        clk = parent.add_net("clk")
        out = parent.add_net("out", bits=4)
        inst = parent.add_module(child, "u1", clk=clk, d=out)
        return parent, inst

    def test_port_map_uses_portwiring(self):
        parent, inst = self._setup()
        from silk.circuit.port_binding import PortWiring
        # New shape: dict[str, PortWiring]
        for k, v in inst._port_map.items():
            assert isinstance(v, PortWiring), (
                f"_port_map[{k!r}] = {v!r} is not a PortWiring"
            )

    def test_port_map_records_bus_membership(self):
        parent, inst = self._setup()
        # Scalar port: bus is None
        assert inst._port_map["clk"].bus is None
        assert inst._port_map["clk"].parent_net == "clk"
        # Bus bits: bus = (bus_name, bit_index)
        assert inst._port_map["d[0]"].bus == ("d", 0)
        assert inst._port_map["d[3]"].bus == ("d", 3)
        # parent_net for bus bits points to the parent's expanded bit name
        assert inst._port_map["d[0]"].parent_net == "out[0]"

    def test_pickle_back_compat_old_shape(self):
        """A legacy port_map (dict[str, str]) unpickles via __setstate__ shim."""
        import warnings
        from silk.circuit.modules import SubcircuitInstance
        from silk.circuit.port_binding import PortWiring
        # Build a minimal SubcircuitInstance with old-shape port_map
        inst = SubcircuitInstance.__new__(SubcircuitInstance)
        old_state = {
            "_port_map": {"clk": "sysclk", "d": "top_d"},  # legacy shape
            "spice_name": "xu1",
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            inst.__setstate__(old_state)
        # After upgrade, port_map values should be PortWiring entries
        for k, v in inst._port_map.items():
            assert isinstance(v, PortWiring)
            assert v.bus is None  # legacy entries have no bus annotation
            assert v.parent_net in ("sysclk", "top_d")


# ---------------------------------------------------------------------------
# §2.7 — Signal width awareness + as_int
# ---------------------------------------------------------------------------

class TestSignalBus:

    def _make_bit_signal(self, name, yvalues):
        import numpy as np
        from silk.signals.base import Signal
        return Signal(name=name, type='voltage', xlabel='time', ylabel='V',
                      xvalues=np.linspace(0, 1, len(yvalues)),
                      yvalues=np.array(yvalues, dtype=float))

    def test_scalar_signal_width_one(self):
        sig = self._make_bit_signal('clk', [0, 1, 0, 1])
        assert sig.width == 1

    def test_bus_signal_construction(self):
        from silk.signals.base import Signal
        bits = [self._make_bit_signal(f'd[{i}]', [0, 0, 1, 1]) for i in range(4)]
        bus = Signal._make_bus_handle(name='d', underlying=tuple(bits))
        assert bus.width == 4
        assert len(bus) == 4

    def test_bus_signal_indexing(self):
        from silk.signals.base import Signal
        bits = [self._make_bit_signal(f'd[{i}]', [0, 0, 1, 1]) for i in range(4)]
        bus = Signal._make_bus_handle(name='d', underlying=tuple(bits))
        assert bus[2] is bits[2]

    def test_as_int_lsb_first(self):
        """as_int default: position 0 = LSB.
        bits = [d[0]=1, d[1]=1, d[2]=0, d[3]=0] at one timepoint
                ↓
        integer = 1*1 + 1*2 + 0*4 + 0*8 = 3
        """
        from silk.signals.base import Signal
        bits = [
            self._make_bit_signal('d[0]', [1.0]),
            self._make_bit_signal('d[1]', [1.0]),
            self._make_bit_signal('d[2]', [0.0]),
            self._make_bit_signal('d[3]', [0.0]),
        ]
        bus = Signal._make_bus_handle(name='d', underlying=tuple(bits))
        result = bus.as_int(threshold=0.5)
        assert int(result.y[0]) == 3

    def test_as_int_msb_first(self):
        from silk.signals.base import Signal
        bits = [
            self._make_bit_signal('d[0]', [1.0]),
            self._make_bit_signal('d[1]', [1.0]),
            self._make_bit_signal('d[2]', [0.0]),
            self._make_bit_signal('d[3]', [0.0]),
        ]
        bus = Signal._make_bus_handle(name='d', underlying=tuple(bits))
        result = bus.as_int(threshold=0.5, msb_first=True)
        # bits in order [d[0], d[1], d[2], d[3]] = [1,1,0,0]
        # msb_first: first iterated bit becomes MSB → integer = 1*8 + 1*4 + 0*2 + 0*1 = 12
        assert int(result.y[0]) == 12

    @pytest.mark.parametrize('msb_first', [False, True])
    def test_wide_bus_uses_positions_and_strict_threshold(self, msb_first):
        from silk.signals.base import Signal
        # The first bit is declared as d[8], but its weight is positional.
        bits = [self._make_bit_signal(f'd[{i + 8}]', [0.5, 0.75])
                if i == 64 else self._make_bit_signal(f'd[{i + 8}]', [0., 0.])
                for i in range(65)]
        if msb_first:
            bits.reverse()
        bus = Signal._make_bus_handle('d', tuple(bits), indices=tuple(range(8, 73)))
        assert bus[8] is bits[0]
        assert list(bus.as_int(threshold=0.5, msb_first=msb_first).y) == [0., float(1 << 64)]


# ---------------------------------------------------------------------------
# §2.6 — Behavioral model 2-D ndarray contract
# ---------------------------------------------------------------------------

def _evaluate_ports_model(module, input_signals, t):
    """Evaluate port views directly for bus shape and rendering unit tests."""
    from silk.simulation._behavioral_state import SharedState
    from silk.simulation._behavioral_views import PortsView, ParamsView
    state = SharedState(len(t), {(module.module_name, pin): pin for pin in module.pin_list},
                        dict(input_signals))
    module._behavioral_model(PortsView(state, module.module_name, module), t,
                             ParamsView(module._behavioral_params))
    return state.signals


class TestBehavioralModelBus:

    def _setup_module(self):
        import numpy as np
        from silk.circuit.modules import Module
        from silk.circuit.circuit_elements import Port
        m = Module(pdk_name='sky130', dir_folder='/tmp', netlistname='t.cir',
                   pdk_details=None, module_name='enc')
        m.define_module_ports({
            'clk':  Port.clock_input(),
            'vcmp': Port.digital_input(bits=15),
            'd':    Port.digital_output(bits=4),
        })
        return m, np

    def test_voltage_view_2d_for_bus_port(self):
        """When the model runs, ports.vcmp.v must be a (15, T) array,
        and ports.clk.v must be a (T,) array."""
        m, np = self._setup_module()
        seen = {}
        def model(ports, t, params):
            seen['vcmp_shape'] = ports.vcmp.v.shape
            seen['clk_shape'] = ports.clk.v.shape
            ports.d.v = np.zeros((4, len(t)))
        m.set_behavioral_model(model, execution='vector')
        T = 100
        _evaluate_ports_model(m,
            input_signals={
                **{f'vcmp[{i}]': np.zeros(T) for i in range(15)},
                'clk': np.zeros(T),
            },
            t=np.linspace(0, 1e-6, T),
        )
        assert seen['vcmp_shape'] == (15, T)
        assert seen['clk_shape'] == (T,)

    def test_assigned_bus_voltage_expands_to_bits(self):
        m, np = self._setup_module()
        T = 50
        def model(ports, t, params):
            ports.d.v = np.tile(np.array([[0], [1], [0], [1]], dtype=float), T)
        m.set_behavioral_model(model, execution='vector')
        result = _evaluate_ports_model(m,
            input_signals={
                **{f'vcmp[{i}]': np.zeros(T) for i in range(15)},
                'clk': np.zeros(T),
            },
            t=np.linspace(0, 1e-6, T),
        )
        # Per-bit signals must exist in the unpacked result
        assert 'd[0]' in result
        assert 'd[3]' in result
        np.testing.assert_array_equal(result['d[0]'], np.zeros(T))
        np.testing.assert_array_equal(result['d[1]'], np.ones(T))

    def test_one_bit_bus_stays_2d(self):
        """§2.6: a 1-bit bus (bits=1) presents as (1, T), NOT (T,)."""
        from silk.circuit.modules import Module
        from silk.circuit.circuit_elements import Port
        import numpy as np
        m = Module(pdk_name='sky130', dir_folder='/tmp', netlistname='t.cir',
                   pdk_details=None, module_name='enc')
        m.define_module_ports({
            'sel': Port.digital_input(bits=1),
            'd':   Port.digital_output(bits=1),
        })
        seen = {}
        def model(ports, t, params):
            seen['sel_shape'] = ports.sel.v.shape
            ports.d.v = np.zeros((1, len(t)))
        m.set_behavioral_model(model, execution='vector')
        T = 10
        _evaluate_ports_model(m,
            input_signals={'sel[0]': np.zeros(T)},
            t=np.linspace(0, 1e-6, T),
        )
        assert seen['sel_shape'] == (1, T)


# ---------------------------------------------------------------------------
# §3.5 + §4.1 — SPICE netlister double-quotes bracketed names in v(...)
# ---------------------------------------------------------------------------

class TestSpiceNetlisterBus:

    def test_quote_if_needed_helper(self):
        from silk.circuit.spice_netlister import _quote_if_needed
        assert _quote_if_needed('clk') == 'clk'
        assert _quote_if_needed('d[3]') == '"d[3]"'
        assert _quote_if_needed('addr[127]') == '"addr[127]"'
        # Underscore form: NOT quoted
        assert _quote_if_needed('d_3') == 'd_3'

    def test_subckt_line_emits_bare_brackets(self):
        """§4.1: .subckt port list uses bare brackets (no quoting)."""
        from silk.circuit.modules import Module
        from silk.circuit.circuit_elements import Port
        m = Module(pdk_name='sky130', dir_folder='/tmp', netlistname='t.cir',
                   pdk_details=None, module_name='enc')
        m.define_module_ports({
            'clk': Port.clock_input(),
            'd':   Port.digital_output(bits=4),
        })
        line = m.generate_subckt(0)
        # Bare bracketed names, no quotes:
        assert 'd[0]' in line
        assert '"d[0]"' not in line
        assert 'clk' in line


# ---------------------------------------------------------------------------
# §5 — BUS-002 BUS_PARTIAL_WIRING lint rule
# ---------------------------------------------------------------------------

class TestBusPartialWiring:

    def test_fully_wired_bus_passes(self):
        """All bits wired → no violation."""
        from silk.circuit.modules import Module
        from silk.circuit.circuit_elements import Port
        from silk.lint.rules.structural import BusPartialWiringRule

        child = Module(pdk_name='sky130', dir_folder='/tmp', netlistname='c.cir',
                       pdk_details=None, module_name='child')
        child.define_module_ports({
            'clk': Port.clock_input(),
            'd':   Port.digital_output(bits=4),
        })

        parent = Module(pdk_name='sky130', dir_folder='/tmp', netlistname='p.cir',
                        pdk_details=None, module_name='parent')
        clk = parent.add_net('clk')
        d = parent.add_net('d', bits=4)
        parent.add_module(child, 'u1', clk=clk, d=d)

        rule = BusPartialWiringRule()
        violations = list(rule.check(parent))
        assert violations == []

    def test_partial_wiring_via_no_connect_passes(self):
        """Unwired bits with per-bit NoConnectPort markers → no violation."""
        from silk.circuit.modules import Module, ModulePin
        from silk.circuit.circuit_elements import Port, PortDir, PortRole
        from silk.circuit.bus_spec import BusSpec
        from silk.lint.rules.structural import BusPartialWiringRule

        child = Module(pdk_name='sky130', dir_folder='/tmp', netlistname='c.cir',
                       pdk_details=None, module_name='child')
        child.define_module_ports({'d': Port.digital_output(bits=4)})

        parent = Module(pdk_name='sky130', dir_folder='/tmp', netlistname='p.cir',
                        pdk_details=None, module_name='parent')
        # Manually create a partially-wired stub instance: wire bits 0,1 only
        parent._element_mgr.elem_dict['u1'] = {'device': child, 'type': 'module'}
        n0 = parent.add_net('n0')
        n1 = parent.add_net('n1')
        child.pin_list['d[0]'].net = 'n0'
        child.pin_list['d[1]'].net = 'n1'
        # bits 2..3: net = None (dangling)
        # Mark them as no-connect
        parent.add_no_connect_port('u1', 'd[2]', reason='unused')
        parent.add_no_connect_port('u1', 'd[3]', reason='unused')

        rule = BusPartialWiringRule()
        violations = list(rule.check(parent))
        assert violations == []

    def test_partial_wiring_without_marker_warns(self):
        """Unwired bits without NoConnectPort markers → 1 violation listing the bits."""
        from silk.circuit.modules import Module
        from silk.circuit.circuit_elements import Port
        from silk.lint.rules.structural import BusPartialWiringRule

        child = Module(pdk_name='sky130', dir_folder='/tmp', netlistname='c.cir',
                       pdk_details=None, module_name='child')
        child.define_module_ports({'d': Port.digital_output(bits=4)})

        parent = Module(pdk_name='sky130', dir_folder='/tmp', netlistname='p.cir',
                        pdk_details=None, module_name='parent')
        # Build a sub-instance with bit 0 wired and bits 1, 2, 3 dangling.
        parent._element_mgr.elem_dict['u1'] = {'device': child, 'type': 'module'}
        n0 = parent.add_net('n0')
        child.pin_list['d[0]'].net = 'n0'
        # bits 1, 2, 3 dangle — no markers added

        rule = BusPartialWiringRule()
        violations = list(rule.check(parent))
        assert len(violations) == 1
        v = violations[0]
        assert v.rule_id == 'BUS-002'
        assert "'d'" in v.message or "d[" in v.message
        # The message should mention the count of unwired bits or the bit names
        assert any(s in v.message for s in ['3', 'd[1]', 'd[2]', 'd[3]'])


# ---------------------------------------------------------------------------
# §3.6 — structure_tracker hash stability
# ---------------------------------------------------------------------------

class TestStructureTrackerHashStability:

    def _make_pre_bus_module(self):
        """A module declaring 3 scalar ports — no buses involved."""
        from silk.circuit.modules import Module
        from silk.circuit.circuit_elements import Port
        m = Module(pdk_name='sky130', dir_folder='/tmp', netlistname='t.cir',
                   pdk_details=None, module_name='legacy')
        m.define_module_ports({
            'clk': Port.clock_input(),
            'a':   Port.digital_input(),
            'b':   Port.digital_output(),
        })
        return m

    def test_pre_bus_module_hash_stable(self):
        """A module that doesn't use buses must hash byte-identical regardless
        of whether _buses is present (it's empty for scalar-only modules)."""
        from silk.db.structure_tracker import CircuitStructureTracker
        tracker = CircuitStructureTracker(database=None, circuit_id=None)
        m1 = self._make_pre_bus_module()
        # Compute hash via the new canonical form
        struct1 = tracker._extract_module_structure(m1, 'legacy')
        h1 = tracker._compute_module_hash(struct1)

        m2 = self._make_pre_bus_module()
        # Force _buses to be missing entirely (simulating pre-bus state)
        if hasattr(m2, '_buses'):
            del m2._buses
        struct2 = tracker._extract_module_structure(m2, 'legacy')
        h2 = tracker._compute_module_hash(struct2)
        assert h1 == h2

    def test_bus_and_scalar_with_same_expanded_pins_match(self):
        """Module A: 4 scalar ports d_0..d_3.
        Module B: 1 bus port d, bits=4, naming='underscore' → expands to d_0..d_3.
        Same expanded pin names → same hash."""
        from silk.db.structure_tracker import CircuitStructureTracker
        from silk.circuit.modules import Module
        from silk.circuit.circuit_elements import Port

        tracker = CircuitStructureTracker(database=None, circuit_id=None)
        # Module A: 4 scalar ports with bracket-free names (since silk's scalar
        # API rejects brackets, use underscore-style names: d_0, d_1, d_2, d_3)
        m_a = Module(pdk_name='sky130', dir_folder='/tmp', netlistname='a.cir',
                     pdk_details=None, module_name='a')
        m_a.define_module_ports({
            'd_0': Port.digital_output(),
            'd_1': Port.digital_output(),
            'd_2': Port.digital_output(),
            'd_3': Port.digital_output(),
        })
        # Module B: same 4 pin names via a bus declaration
        m_b = Module(pdk_name='sky130', dir_folder='/tmp', netlistname='b.cir',
                     pdk_details=None, module_name='a')
        m_b.define_module_ports({
            'd': Port.digital_output(bits=4, naming='underscore'),
        })
        h_a = tracker._compute_module_hash(tracker._extract_module_structure(m_a, 'a'))
        h_b = tracker._compute_module_hash(tracker._extract_module_structure(m_b, 'a'))
        assert h_a == h_b


# ---------------------------------------------------------------------------
# §6 — Bus plot rendering
# ---------------------------------------------------------------------------

class TestBusPlotRendering:

    def _build_and_run(self):
        """Build a tiny module with a 4-bit bus output and a behavioral
        model that produces a known integer-decoded waveform."""
        import numpy as np
        from silk.circuit.modules import Module
        from silk.circuit.circuit_elements import Port
        m = Module(pdk_name='sky130', dir_folder='/tmp', netlistname='t.cir',
                   pdk_details=None, module_name='counter')
        m.define_module_ports({
            'clk': Port.clock_input(),
            'd':   Port.digital_output(bits=4),
        })

        T = 16
        def model(ports, t, params):
            counts = np.arange(T)
            bits = np.stack([counts >> b & 1 for b in range(4)]).astype(float)
            ports.d.v = bits
        m.set_behavioral_model(model, execution='vector')
        result = _evaluate_ports_model(m,
            input_signals={'clk': np.zeros(T)},
            t=np.linspace(0, 1e-6, T),
        )
        return m, result

    def test_resolve_signal_path_for_bus(self):
        m, result = self._build_and_run()
        from silk.circuit.report import resolve_signal_path
        handle = resolve_signal_path(m, 'd', behavioral_results=result)
        assert handle.width == 4

    def test_resolve_signal_path_for_single_bit(self):
        m, result = self._build_and_run()
        from silk.circuit.report import resolve_signal_path
        sig = resolve_signal_path(m, 'd[3]', behavioral_results=result)
        # Single-bit Signal — width 1
        assert sig.width == 1

    def test_render_bus_integer_trace_default(self):
        """render_bus_trace(...) renders a single integer-decoded line by default."""
        import matplotlib
        matplotlib.use('Agg')  # headless
        import matplotlib.pyplot as plt
        m, result = self._build_and_run()
        from silk.circuit.report import render_bus_trace
        fig = plt.figure()
        try:
            ax = fig.add_subplot(111)
            render_bus_trace(ax, m, 'd', behavioral_results=result, expand_bits=False)
            # Exactly one line on the axes
            assert len(ax.lines) == 1
            # Y values should reach 15 (max of 4-bit counter)
            assert ax.lines[0].get_ydata().max() == 15
        finally:
            plt.close(fig)

    def test_render_bus_expand_bits(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        m, result = self._build_and_run()
        from silk.circuit.report import render_bus_trace
        fig = plt.figure()
        try:
            ax = fig.add_subplot(111)
            render_bus_trace(ax, m, 'd', behavioral_results=result, expand_bits=True)
            # 4 lines for 4 bits
            assert len(ax.lines) == 4
        finally:
            plt.close(fig)


# ---------------------------------------------------------------------------
# §6 — HTML report bus rendering
# ---------------------------------------------------------------------------

class TestBusHtmlRendering:

    def test_bus_renders_one_row(self):
        """A 4-bit bus renders as ONE <tr>, not 4 separate rows."""
        import numpy as np
        from silk.circuit.modules import Module
        from silk.circuit.circuit_elements import Port
        from silk.circuit.report import render_bus_html_row

        m = Module(pdk_name='sky130', dir_folder='/tmp', netlistname='t.cir',
                   pdk_details=None, module_name='enc')
        m.define_module_ports({'d': Port.digital_output(bits=4)})

        T = 8
        results = {f'd[{i}]': np.zeros(T) for i in range(4)}
        # Make bit 0 toggle to produce some transitions
        results['d[0]'] = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=float)

        html = render_bus_html_row(m, 'd', behavioral_results=results)
        # Single <tr> element for the whole bus
        assert html.count('<tr') == 1
        # Bus name appears in the row
        assert 'd' in html

    def test_html_summary_includes_count(self):
        """Counter-bus rendering should mention transitions or min/max."""
        import numpy as np
        from silk.circuit.modules import Module
        from silk.circuit.circuit_elements import Port
        from silk.circuit.report import render_bus_html_row
        m = Module(pdk_name='sky130', dir_folder='/tmp', netlistname='t.cir',
                   pdk_details=None, module_name='enc')
        m.define_module_ports({'d': Port.digital_output(bits=4)})
        T = 16
        results = {}
        # 0,1,2,...,15 — a 4-bit counter
        counts = np.arange(T)
        for b in range(4):
            results[f'd[{b}]'] = ((counts >> b) & 1).astype(float)
        html = render_bus_html_row(m, 'd', behavioral_results=results)
        # Should include some kind of summary — transition count, min, max, or range
        assert any(s in html for s in ['transition', 'min', 'max', '15'])


# ---------------------------------------------------------------------------
# §10.5 / §11 — End-to-end ngspice round-trip
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestNgspiceEndToEnd:

    def test_4bit_bus_subckt_runs_in_ngspice(self, tmp_path):
        """Confirm the silk-emit pattern (.subckt / instance / .save v(\"d[3]\"))
        round-trips through ngspice 42 with bracketed names preserved."""
        import subprocess, shutil
        if not shutil.which('ngspice'):
            pytest.skip("ngspice not installed")

        netlist = '''* bus port end-to-end
.subckt buf4 d[3] d[2] d[1] d[0] vdd 0
r0 d[0] vdd 1k
r1 d[1] vdd 2k
r2 d[2] vdd 3k
r3 d[3] vdd 4k
.ends

vdd vdd 0 1.8
xb1 q[3] q[2] q[1] q[0] vdd 0 buf4
.tran 1n 10n
.save v("q[0]") v("q[1]") v("q[2]") v("q[3]")
.control
run
write {raw} v("q[0]") v("q[1]") v("q[2]") v("q[3]")
.endc
.end
'''.format(raw=str(tmp_path / 'out.raw'))
        cir = tmp_path / 'test.cir'
        cir.write_text(netlist)
        result = subprocess.run(
            ['ngspice', '-b', str(cir)], capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"ngspice failed: {result.stderr}"
        # Parse the rawfile and confirm variable names round-trip
        raw = (tmp_path / 'out.raw').read_bytes()
        # ASCII header lives in the first ~500 bytes
        header = raw[:500].decode('latin-1', errors='replace')
        # Bracketed names appear in the rawfile (no quotes — ngspice strips)
        assert 'q[0]' in header
        assert 'q[3]' in header


# ---------------------------------------------------------------------------
# §8 — Fault injection compatibility with bracketed names
# ---------------------------------------------------------------------------

class TestFaultInjectionBus:

    def test_setattr_getattr_bracketed_name(self):
        """Python's setattr/getattr accept bracketed strings — silk.faults
        uses string-keyed lookups, not attribute syntax, so this round-trip
        must work."""
        class Box:
            pass
        b = Box()
        setattr(b, 'd[3]', 'value')
        assert getattr(b, 'd[3]') == 'value'

    def test_faulted_bus_module_emits_valid_subckt(self):
        """Generate the .subckt for a module containing a 4-bit bus and
        confirm the .cir output is well-formed: bracketed names appear,
        and there are no malformed escape sequences."""
        from silk.circuit.modules import Module
        from silk.circuit.circuit_elements import Port

        m = Module(pdk_name='sky130', dir_folder='/tmp', netlistname='t.cir',
                   pdk_details=None, module_name='enc')
        m.define_module_ports({
            'clk': Port.clock_input(),
            'd':   Port.digital_output(bits=4),
        })
        line = m.generate_subckt(0)
        assert '.subckt enc' in line
        for i in range(4):
            assert f'd[{i}]' in line
        # No backslash-escape sequences (ngspice rejects v(d\[3\]))
        assert r'\[' not in line
        assert r'\]' not in line


# ---------------------------------------------------------------------------
# Post-merge regression: end-to-end paths that the per-task tests bypassed.
# §2.6 — simulate_behavioral() + bus ports must work
# §2.7 — inst.d must return a width-aware bus Signal
# ---------------------------------------------------------------------------

class TestBusEndToEndIntegration:

    def test_simulate_behavioral_with_bus_port(self):
        """A subcircuit with a bus port runs through simulate_behavioral()
        and produces correct per-bit waveforms. This exercises the
        BehavioralEngine._evaluate_blocks → electrical port views
        end-to-end (BUG-01 regression)."""
        import numpy as np
        from silk.circuit.modules import Module
        from silk.circuit.circuit_elements import Port

        # Child module: 4-bit counter. Receives 1-bit clk; outputs 4-bit count.
        counter = Module(pdk_name='sky130', dir_folder='/tmp', netlistname='c.cir',
                         pdk_details=None, module_name='counter')
        counter.define_module_ports({
            'clk': Port.clock_input(),
            'd':   Port.digital_output(bits=4),
        })

        def model(ports, t, params):
            T = len(t)
            counts = np.arange(T) & 15
            bits = np.stack([counts >> b & 1 for b in range(4)]).astype(float)
            ports.d.v = bits
        counter.set_behavioral_model(model, execution='vector')

        # Top: instantiate counter and provide a clock source
        tb = Module(pdk_name='sky130', dir_folder='/tmp', netlistname='t.cir',
                    pdk_details=None, module_name='tb')
        clk = tb.add_net('clk')
        d   = tb.add_net('d', bits=4)
        tb.add_module(counter, 'u1', clk=clk, d=d)

        # Run the behavioral engine end-to-end. We invoke _evaluate_blocks
        # (on the flattened blocks) directly with a hand-built order to avoid the source-generation /
        # topo-sort scaffolding (which would require declaring a clock source).
        from silk.simulation.behavioral import BehavioralEngine
        T = 8
        t = np.linspace(0, 1e-6, T)
        engine = BehavioralEngine()
        net_signals = {'clk': np.zeros(T)}
        for i in range(4):
            net_signals[f'd[{i}]'] = np.zeros(T)
        blocks, _ = engine._flatten(tb)
        engine._evaluate_blocks(blocks, t, ['u1'], net_signals)

        # bit 0 should toggle every time-step (LSB of 0,1,2,...,7)
        np.testing.assert_array_equal(
            net_signals['d[0]'], np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=float)
        )
        # bit 1 should toggle every two time-steps
        np.testing.assert_array_equal(
            net_signals['d[1]'], np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=float)
        )

    def test_inst_d_returns_bus_signal_handle(self, tmp_path):
        """A structural bus binds lazily to the direct Module result context."""
        import numpy as np
        from silk.circuit.modules import Module
        from silk.circuit.circuit_elements import Port
        from silk.signals.base import Signal

        child = Module(pdk_name='sky130', dir_folder='/tmp', netlistname='c.cir',
                       pdk_details=None, module_name='child')
        child.define_module_ports({
            'clk': Port.clock_input(),
            'd':   Port.digital_output(bits=4),
        })
        parent = Module(pdk_name='sky130', dir_folder='/tmp', netlistname='p.cir',
                        pdk_details=None, module_name='parent')
        clk_net = parent.add_net('clk')
        d_net   = parent.add_net('d', bits=4)
        inst = parent.add_module(child, 'u1', clk=clk_net, d=d_net)

        from silk.signals.rawfile import RawFile, RawPlot, RawVector, write_rawfile
        counts = np.arange(8)
        vectors = [RawVector('time', 'time', data=np.linspace(0, 1e-6, 8))]
        vectors.extend(RawVector(f'v(d[{bit}])', 'voltage',
                                 data=((counts >> bit) & 1).astype(float)) for bit in range(4))
        path = tmp_path / 'bus.raw'
        write_rawfile(path, RawFile([RawPlot({'plotname': 'Transient Analysis', 'title': 'bus'},
                                           vectors, npoints=8, real=True, padded=True)]))
        bus_ref = inst.d  # No result read here.
        assert bus_ref.width == 4
        assert inst.d[3].width == 1
        parent.load_file(str(path))
        parent.read_raw_file()
        np.testing.assert_array_equal(bus_ref.as_int().y.astype(int), counts)


@pytest.mark.parametrize('naming', ['bracketed', 'underscore'])
def test_wider_internal_placement_keeps_declared_order(tmp_path, naming):
    from silk import Module
    from silk.circuit.ports import Port
    child = Module(None, str(tmp_path), 'child.cir', None, module_name='child')
    child.define_module_ports({'x': Port.analog_input(bits=range(2, 6), naming=naming)})
    child.add_net('x', bits=range(1, 9))
    host = Module(None, str(tmp_path), 'host.cir', None, module_name='host')
    stage = host.add_module(child, 'stage', x=host.add_net('a', bits=4))
    ref = stage.x
    assert ref._indices == tuple(range(1, 9))
    assert [bit.name for bit in ref] == [
        f'x[{i}]' if naming == 'bracketed' else f'x_{i}' for i in range(1, 9)]


@pytest.mark.parametrize('direction', ['input', 'inout', 'output'])
def test_wider_internal_bus_keeps_formal_wiring_contract(tmp_path, direction):
    from silk import Module
    from silk.circuit.ports import Port, PortDir
    from silk.exceptions import CircuitError
    from silk.lint.rules.structural import BusPartialWiringRule
    child = Module(None, str(tmp_path), 'child.cir', None, module_name='child')
    child.define_module_ports({'x': Port(dir=PortDir(direction), bits=4)})
    schema = child.export_schema()
    child.add_net('x', bits=8)
    assert child.export_schema() == schema
    parent = Module(None, str(tmp_path), 'parent.cir', None, module_name='parent')
    parent.add_module(child, 'stage', x=parent.add_net('a', bits=4))
    pins = parent.elem_dict['stage']['pin_list']
    assert {name: pin.net for name, pin in pins.items()} == {
        f'x[{i}]': f'a[{i}]' for i in range(4)}
    assert list(BusPartialWiringRule().check(parent)) == []
    with pytest.raises(CircuitError, match='BUS_WIDTH_MISMATCH'):
        parent.add_module(child, 'wide', x=parent.add_net('b', bits=8))
    scalar = parent.add_net('tie')
    if direction == 'output':
        with pytest.raises(CircuitError, match='BUS_WIDTH_MISMATCH'):
            parent.add_module(child, 'tied', x=scalar)
    else:
        parent.add_module(child, 'tied', x=scalar)
        assert [pin.net for pin in parent.elem_dict['tied']['pin_list'].values()] == ['tie'] * 4


def test_wider_internal_bus_behavioral_view_keeps_formal_width(tmp_path):
    import numpy as np
    from silk import Module
    from silk.circuit.ports import Port
    from silk.simulation.behavioral import BehavioralEngine
    child = Module(None, str(tmp_path), 'child.cir', None, module_name='child')
    child.define_module_ports({'d': Port.digital_output(bits=4)})
    child.add_net('d', bits=8)
    def model(ports, t, params):
        assert ports.d.v.shape == (4, len(t))
        ports.d.v = np.arange(4.)[:, None] * np.ones((4, len(t)))
    child.set_behavioral_model(model, execution='vector')
    parent = Module(None, str(tmp_path), 'parent.cir', None, module_name='parent')
    parent.add_module(child, 'stage', d=parent.add_net('a', bits=4))
    engine = BehavioralEngine()
    blocks, _ = engine._flatten(parent)
    t = np.linspace(0, 1e-6, 3)
    signals = {f'a[{i}]': np.zeros(3) for i in range(4)}
    engine._evaluate_blocks(blocks, t, ['stage'], signals)
    for i in range(4):
        np.testing.assert_array_equal(signals[f'a[{i}]'], np.full(3, i))
