import pytest
from silk.lint.locations import (
    ViolationLocation, NetLocation, PortLocation,
    InstanceLocation, TerminalLocation, ModuleLocation,
)


def test_net_location_holds_net_and_ports():
    loc = NetLocation(net_name="vout", port_names=["out1", "out2"])
    assert loc.net_name == "vout"
    assert loc.port_names == ["out1", "out2"]


def test_net_location_is_violation_location():
    loc = NetLocation(net_name="vout", port_names=[])
    assert isinstance(loc, ViolationLocation)


def test_port_location_holds_port_name():
    loc = PortLocation(port_name="vdd", module_name="opamp")
    assert loc.port_name == "vdd"
    assert loc.module_name == "opamp"


def test_instance_location_holds_instance_path():
    loc = InstanceLocation(instance_path="ldo.m_pass")
    assert loc.instance_path == "ldo.m_pass"


def test_terminal_location_holds_instance_and_terminal():
    loc = TerminalLocation(
        instance_path="ldo.m_pass", terminal="drain",
        terminal_pair=("gate", "drain"),
    )
    assert loc.terminal == "drain"
    assert loc.terminal_pair == ("gate", "drain")


def test_module_location_holds_module_name():
    loc = ModuleLocation(module_name="bandgap")
    assert loc.module_name == "bandgap"


def test_location_describe_returns_string():
    assert NetLocation(net_name="vout", port_names=[]).describe() == "net:vout"
    assert PortLocation(port_name="vdd", module_name="opamp").describe() == "port:opamp.vdd"
    assert InstanceLocation(instance_path="ldo.m_pass").describe() == "instance:ldo.m_pass"
    assert TerminalLocation(
        instance_path="ldo.m_pass", terminal="drain", terminal_pair=("gate", "drain"),
    ).describe() == "terminal:ldo.m_pass.drain"
    assert ModuleLocation(module_name="bandgap").describe() == "module:bandgap"
