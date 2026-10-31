import pytest
from silk.circuit.circuit_elements import PortDir, PortRole


def test_port_role_values():
    assert PortRole.ANALOG.value  == 'analog'
    assert PortRole.DIGITAL.value == 'digital'
    assert PortRole.CLOCK.value   == 'clock'
    assert PortRole.POWER.value   == 'power'
    assert PortRole.GROUND.value  == 'ground'


def test_port_role_is_string_enum():
    assert isinstance(PortRole.DIGITAL, str)
    assert PortRole.ANALOG == 'analog'


def test_port_dir_backward_compat():
    """Existing POWER and GROUND on PortDir must still work."""
    assert PortDir.POWER  == 'power'
    assert PortDir.GROUND == 'ground'


from silk.circuit.modules import ModulePin
from unittest.mock import MagicMock


def _bare_module(tmp_path):
    from silk.circuit.modules import Module
    pdk = MagicMock()
    pdk.model_names = []
    pdk._devices = {}
    return Module('TEST', str(tmp_path), 'test.cir', pdk_details=pdk)


def test_define_ports_bare_portdir(tmp_path):
    """Passing a plain PortDir still works — role is None."""
    m = _bare_module(tmp_path)
    m.define_module_name('sub')
    m.define_module_ports({'vin': PortDir.IN, 'vout': PortDir.OUT})
    assert m.pin_list['vin'].direction == PortDir.IN
    assert m.pin_list['vin'].role is None
    assert m.pin_list['vout'].direction == PortDir.OUT
    assert m.pin_list['vout'].role is None


def test_define_ports_bare_supply_portdir_derives_role(tmp_path):
    """TASK-45 contract: PortDir.POWER / PortDir.GROUND imply the matching
    PortRole when the declaration gives none — bare enum, direction string,
    or a Port() with no role. Signal directions still imply nothing."""
    from silk.circuit.circuit_elements import Port
    m = _bare_module(tmp_path)
    m.define_module_name('sub')
    m.define_module_ports({
        'vdd':  PortDir.POWER,
        'gnd':  PortDir.GROUND,
        'vdda': 'power',
        'vssa': 'GROUND',
        'vddd': Port(dir=PortDir.POWER),
        'sig':  PortDir.INOUT,
    })
    assert m.pin_list['vdd'].role  == PortRole.POWER
    assert m.pin_list['gnd'].role  == PortRole.GROUND
    assert m.pin_list['vdda'].role == PortRole.POWER
    assert m.pin_list['vssa'].role == PortRole.GROUND
    assert m.pin_list['vddd'].role == PortRole.POWER
    assert m.pin_list['vdd'].direction == PortDir.POWER   # direction untouched
    assert m.pin_list['sig'].role is None


def test_define_ports_explicit_role_beats_implied(tmp_path):
    """An explicit role on a supply direction is stored as written."""
    m = _bare_module(tmp_path)
    m.define_module_name('sub')
    m.define_module_ports({'vref': (PortDir.POWER, PortRole.ANALOG)})
    assert m.pin_list['vref'].role == PortRole.ANALOG


def test_define_ports_bare_supply_bus_derives_role(tmp_path):
    """The derivation reaches bus ports: each bit and the BusSpec carry it."""
    from silk.circuit.circuit_elements import Port
    m = _bare_module(tmp_path)
    m.define_module_name('sub')
    m.define_module_ports({'vdd': Port(dir=PortDir.POWER, bits=2)})
    assert m._buses['vdd'].role == PortRole.POWER
    assert {p.role for n, p in m.pin_list.items() if n.startswith('vdd')} == {PortRole.POWER}


def test_port_dir_implied_role():
    assert PortDir.POWER.implied_role  == PortRole.POWER
    assert PortDir.GROUND.implied_role == PortRole.GROUND
    assert all(d.implied_role is None for d in (PortDir.IN, PortDir.OUT, PortDir.INOUT))


def test_define_ports_tuple_syntax(tmp_path):
    """Passing (PortDir, PortRole) tuple stores both."""
    m = _bare_module(tmp_path)
    m.define_module_name('sub')
    m.define_module_ports({
        'clk':  (PortDir.IN,  PortRole.CLOCK),
        'en':   (PortDir.IN,  PortRole.DIGITAL),
        'vout': (PortDir.OUT, PortRole.ANALOG),
        'vdd':  (PortDir.IN,  PortRole.POWER),
        'gnd':  (PortDir.IN,  PortRole.GROUND),
    })
    assert m.pin_list['clk'].direction  == PortDir.IN
    assert m.pin_list['clk'].role       == PortRole.CLOCK
    assert m.pin_list['en'].role        == PortRole.DIGITAL
    assert m.pin_list['vout'].role      == PortRole.ANALOG
    assert m.pin_list['vdd'].role       == PortRole.POWER
    assert m.pin_list['gnd'].role       == PortRole.GROUND


def test_define_ports_mixed_syntax(tmp_path):
    """Bare PortDir and tuple can be mixed in one call."""
    m = _bare_module(tmp_path)
    m.define_module_name('sub')
    m.define_module_ports({
        'vin': PortDir.IN,
        'clk': (PortDir.IN, PortRole.CLOCK),
    })
    assert m.pin_list['vin'].role is None
    assert m.pin_list['clk'].role == PortRole.CLOCK


def test_define_ports_none_defaults_to_inout(tmp_path):
    """None still defaults to INOUT with no role — backward compat."""
    m = _bare_module(tmp_path)
    m.define_module_name('sub')
    m.define_module_ports({'net': None})
    assert m.pin_list['net'].direction == PortDir.INOUT
    assert m.pin_list['net'].role is None


def test_module_pin_role_field():
    pin = ModulePin(name='test', direction=PortDir.IN, role=PortRole.DIGITAL)
    assert pin.role == PortRole.DIGITAL

    pin_no_role = ModulePin(name='test2', direction=PortDir.OUT)
    assert pin_no_role.role is None


def test_define_ports_malformed_tuple_raises(tmp_path):
    """Tuple with wrong element count raises a clear ValueError."""
    m = _bare_module(tmp_path)
    m.define_module_name('sub')
    with pytest.raises(ValueError, match="2-tuple"):
        m.define_module_ports({'vin': (PortDir.IN,)})
    with pytest.raises(ValueError, match="2-tuple"):
        m.define_module_ports({'vin': (PortDir.IN, PortRole.ANALOG, 'extra')})


def test_top_level_imports():
    """PortRole and check_interfaces importable from silk directly."""
    from silk import PortRole, check_interfaces
    assert PortRole.DIGITAL == 'digital'
    assert callable(check_interfaces)
