"""Tests for the Port classmethod factories (analog_input, supply, ground,
etc.) — convenience wrappers around the canonical Port(dir=, role=, domain=)
constructor that cut boilerplate in user code."""

import pytest
from silk import Port, PortDir, PortRole, VoltageDomain
from silk.circuit.modules import Module


AVDD = VoltageDomain('avdd', nominal=1.8, tolerance=0.1)


# ---------------------------------------------------------------------------
# Signal-family factories — (direction in name, domain as only arg)
# ---------------------------------------------------------------------------

class TestSignalFactories:

    def test_analog_input(self):
        p = Port.analog_input(AVDD)
        assert p.dir is PortDir.IN
        assert p.role is PortRole.ANALOG
        assert p.domain is AVDD

    def test_analog_output(self):
        p = Port.analog_output(AVDD)
        assert p.dir is PortDir.OUT
        assert p.role is PortRole.ANALOG
        assert p.domain is AVDD

    def test_digital_input(self):
        p = Port.digital_input(AVDD)
        assert p.dir is PortDir.IN
        assert p.role is PortRole.DIGITAL
        assert p.domain is AVDD

    def test_digital_output(self):
        p = Port.digital_output(AVDD)
        assert p.dir is PortDir.OUT
        assert p.role is PortRole.DIGITAL

    def test_clock_input(self):
        p = Port.clock_input()
        assert p.dir is PortDir.IN
        assert p.role is PortRole.CLOCK

    def test_clock_output(self):
        p = Port.clock_output(AVDD)
        assert p.dir is PortDir.OUT
        assert p.role is PortRole.CLOCK

    def test_domain_defaults_to_none(self):
        """All signal factories make domain optional."""
        assert Port.analog_input().domain is None
        assert Port.analog_output().domain is None
        assert Port.digital_input().domain is None
        assert Port.digital_output().domain is None
        assert Port.clock_input().domain is None
        assert Port.clock_output().domain is None


# ---------------------------------------------------------------------------
# Rail-family factories — supply / ground with direction override
# ---------------------------------------------------------------------------

class TestRailFactories:

    def test_supply_defaults_in(self):
        p = Port.supply(AVDD)
        assert p.dir is PortDir.IN
        assert p.role is PortRole.POWER
        assert p.domain is AVDD

    def test_ground_defaults_in_no_domain(self):
        p = Port.ground()
        assert p.dir is PortDir.IN
        assert p.role is PortRole.GROUND
        assert p.domain is None

    def test_supply_direction_override_enum(self):
        p = Port.supply(AVDD, direction=PortDir.OUT)
        assert p.dir is PortDir.OUT
        assert p.role is PortRole.POWER

    def test_supply_direction_override_string(self):
        """direction= accepts a string so user code doesn't need to
        import PortDir just to override the default."""
        p = Port.supply(AVDD, direction='out')
        assert p.dir is PortDir.OUT

    def test_ground_direction_override_string(self):
        p = Port.ground(direction='out')
        assert p.dir is PortDir.OUT
        assert p.role is PortRole.GROUND

    def test_supply_domain_optional(self):
        p = Port.supply()
        assert p.dir is PortDir.IN
        assert p.role is PortRole.POWER
        assert p.domain is None

    def test_invalid_direction_string_raises(self):
        with pytest.raises(ValueError):
            Port.supply(direction='sideways')


# ---------------------------------------------------------------------------
# End-to-end: factories flow through define_module_ports identically to
# manually-constructed Port objects.
# ---------------------------------------------------------------------------

class TestIntegrationWithDefineModulePorts:

    def test_factories_equivalent_to_manual_construction(self, pdk, tmp_dir):
        mod_a = Module('PDK_skywater130', tmp_dir, 'a.cir',
                       pdk_details=pdk, module_name='a')
        mod_b = Module('PDK_skywater130', tmp_dir, 'b.cir',
                       pdk_details=pdk, module_name='b')

        # Manual
        mod_a.define_module_ports({
            'vin':  Port(dir=PortDir.IN,  role=PortRole.ANALOG, domain=AVDD),
            'vout': Port(dir=PortDir.OUT, role=PortRole.ANALOG, domain=AVDD),
            'vdd':  Port(dir=PortDir.IN,  role=PortRole.POWER,  domain=AVDD),
            'gnd':  Port(dir=PortDir.IN,  role=PortRole.GROUND, domain=None),
        })

        # Factories
        mod_b.define_module_ports({
            'vin':  Port.analog_input(AVDD),
            'vout': Port.analog_output(AVDD),
            'vdd':  Port.supply(AVDD),
            'gnd':  Port.ground(),
        })

        # pin_list entries must match field-for-field.
        for name in ('vin', 'vout', 'vdd', 'gnd'):
            pa = mod_a.pin_list[name]
            pb = mod_b.pin_list[name]
            assert pa.direction == pb.direction
            assert pa.role == pb.role
            assert pa.domain is pb.domain

    def test_factories_compose_in_realistic_definition(self, pdk, tmp_dir):
        """The common case — four ports with the factory form — should
        work exactly like the canonical constructor."""
        mod = Module('PDK_skywater130', tmp_dir, 'c.cir',
                     pdk_details=pdk, module_name='c')
        mod.define_module_ports({
            'vinp':     Port.analog_input(AVDD),
            'vout':     Port.analog_output(AVDD),
            'avdd_1p8': Port.supply(AVDD),
            'gnd':      Port.ground(),
        })
        assert 'vinp' in mod.pin_list
        assert mod.pin_list['avdd_1p8'].role == PortRole.POWER
        assert mod.pin_list['avdd_1p8'].domain is AVDD
        assert mod.pin_list['gnd'].role == PortRole.GROUND
