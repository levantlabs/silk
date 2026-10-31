import pytest
from unittest.mock import MagicMock
from silk.lint.rules.structural import FloatingNetRule, SingleConnectionNetRule
from silk.lint.severity import Severity
from silk.lint.violations import InterfaceViolation
from silk.lint.locations import NetLocation
from silk.circuit.modules import Module


def _pdk():
    pdk = MagicMock()
    pdk.model_names = []
    pdk._devices = {}
    return pdk


def test_floating_net_rule_id():
    assert FloatingNetRule().rule_id == "STRUCT-003"


def test_floating_net_category():
    assert FloatingNetRule().category == "structural"


def test_floating_net_severity_default():
    assert FloatingNetRule().default_severity == Severity.WARNING


def test_floating_net_rationale_mentions_intentional():
    assert "intentional" in FloatingNetRule().rationale.lower()


def test_floating_net_applicable_to_module():
    assert Module in FloatingNetRule().applicable_to


def test_floating_net_fires_on_unconnected_net(tmp_path):
    m = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    m.add_net('unused_node')
    rule = FloatingNetRule()
    violations = list(rule.check(m))
    assert len(violations) == 1
    assert violations[0].rule_id == "STRUCT-003"
    assert violations[0].location.net_name == "unused_node"
    assert isinstance(violations[0], InterfaceViolation)
    assert isinstance(violations[0].location, NetLocation)


def test_floating_net_suppressed_by_noconnect(tmp_path):
    m = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    m.add_net('probe_out')
    m.add_no_connect('probe_out', reason="differential test tap")
    rule = FloatingNetRule()
    violations = list(rule.check(m))
    assert violations == []


def test_floating_net_does_not_fire_on_connected_net(tmp_path):
    """A net with real submodule pins is not floating."""
    from silk.circuit.circuit_elements import PortDir
    a = Module('TEST', str(tmp_path), 'a.cir', pdk_details=_pdk())
    a.define_module_name('a')
    a.define_module_ports({'out': PortDir.OUT})
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_module(a, 'a1', nets={'out': 'live_net'})
    rule = FloatingNetRule()
    violations = list(rule.check(top))
    # 'live_net' has a real binding, so no violation
    assert all(v.location.net_name != 'live_net' for v in violations)


# === Task 9: STRUCT-001, STRUCT-002, STRUCT-004 ===
from silk.lint.rules.structural import (
    DanglingPortRule, PowerPortUnconnectedRule, FloatingGateRule,
)
from silk.lint.locations import PortLocation, InstanceLocation
from silk.circuit.circuit_elements import PortDir, PortRole


def _sub_ppc(tmp_path, name, ports):
    s = Module('TEST', str(tmp_path), f'{name}.cir', pdk_details=_pdk())
    s.define_module_name(name)
    s.define_module_ports(ports)
    return s


def test_dangling_port_metadata():
    r = DanglingPortRule()
    assert r.rule_id == "STRUCT-001"
    assert r.category == "structural"
    assert r.default_severity == Severity.WARNING
    assert len(r.rationale) > 20


def test_dangling_port_fires_on_unconnected_submodule_port(tmp_path):
    a = _sub_ppc(tmp_path, 'a', {'in': PortDir.IN, 'out': PortDir.OUT})
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_module(a, 'a1', nets={'in': 'wire1'})  # 'out' left unbound
    rule = DanglingPortRule()
    violations = list(rule.check(top))
    rule_ids = {v.rule_id for v in violations}
    assert "STRUCT-001" in rule_ids
    # The 'out' port should be reported
    out_ports = [v for v in violations if v.rule_id == "STRUCT-001" and 'out' in v.message]
    assert len(out_ports) >= 1


def test_dangling_port_clean_when_all_connected(tmp_path):
    a = _sub_ppc(tmp_path, 'a', {'in': PortDir.IN, 'out': PortDir.OUT})
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_module(a, 'a1', nets={'in': 'wire1', 'out': 'wire2'})
    rule = DanglingPortRule()
    assert list(rule.check(top)) == []


def test_dangling_port_respects_ignore_patterns(tmp_path):
    a = _sub_ppc(tmp_path, 'a', {'in': PortDir.IN, 'debug_out': PortDir.OUT})
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_module(a, 'a1', nets={'in': 'wire1'})  # debug_out unbound
    rule = DanglingPortRule(config={"ignore_patterns": ["debug_*"]})
    violations = list(rule.check(top))
    assert violations == []


# --- NoConnectPort per-instance waivers (STRUCT-001) ---

def test_dangling_port_respects_no_connect_port_marker(tmp_path):
    """Module.add_no_connect_port() suppresses STRUCT-001 for that pin."""
    a = _sub_ppc(tmp_path, 'a',
                 {'in': PortDir.IN, 'test_enable': PortDir.IN})
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_module(a, 'a1', nets={'in': 'wire1'})  # test_enable unbound
    top.add_no_connect_port('a1', 'test_enable',
                            reason='tied off at upper level')

    violations = list(DanglingPortRule().check(top))
    assert violations == [], (
        f'Waived port should not fire STRUCT-001: {violations}')


def test_no_connect_port_only_waives_one_pin(tmp_path):
    """Waiving ONE port doesn't suppress STRUCT-001 for other unbound ports
    on the same instance."""
    a = _sub_ppc(tmp_path, 'a',
                 {'test_enable': PortDir.IN, 'scan_in': PortDir.IN})
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_module(a, 'a1', nets={})  # both unbound
    top.add_no_connect_port('a1', 'test_enable', reason='n/a here')

    violations = list(DanglingPortRule().check(top))
    rule_ids = [v.rule_id for v in violations]
    assert 'STRUCT-001' in rule_ids
    # Still exactly one violation — scan_in — not both.
    assert len(violations) == 1
    assert 'scan_in' in violations[0].message


def test_no_connect_port_scoped_to_instance(tmp_path):
    """Same port name on a different instance is NOT waived."""
    a = _sub_ppc(tmp_path, 'a', {'test_enable': PortDir.IN})
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_module(a, 'a1', nets={})
    top.add_module(a, 'a2', nets={})
    top.add_no_connect_port('a1', 'test_enable', reason='handled elsewhere')

    violations = list(DanglingPortRule().check(top))
    # a1.test_enable waived; a2.test_enable still fires.
    assert len(violations) == 1
    # Port location is (module_name, port_name) — module_name is the
    # instance name for this rule.
    loc = violations[0].location
    assert loc.module_name == 'a2'
    assert loc.port_name == 'test_enable'


def test_no_connect_port_requires_reason(tmp_path):
    """Empty reason must raise."""
    a = _sub_ppc(tmp_path, 'a', {'test_enable': PortDir.IN})
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_module(a, 'a1', nets={})

    with pytest.raises(ValueError, match='empty or blank'):
        top.add_no_connect_port('a1', 'test_enable', reason='')
    with pytest.raises(ValueError, match='empty or blank'):
        top.add_no_connect_port('a1', 'test_enable', reason='   ')


def test_no_connect_port_rejects_unknown_instance(tmp_path):
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    with pytest.raises(Exception, match="instance 'no_such' not found"):
        top.add_no_connect_port('no_such', 'x', reason='test')


def test_no_connect_port_rejects_unknown_port(tmp_path):
    a = _sub_ppc(tmp_path, 'a', {'only_port': PortDir.IN})
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_module(a, 'a1', nets={})
    with pytest.raises(Exception, match="port 'bogus' not declared"):
        top.add_no_connect_port('a1', 'bogus', reason='test')


def test_no_connect_port_combines_with_ignore_patterns(tmp_path):
    """Patterns and per-pin waivers both suppress; neither subsumes the
    other. Port matching EITHER is clean."""
    a = _sub_ppc(tmp_path, 'a',
                 {'debug_out': PortDir.OUT, 'test_enable': PortDir.IN})
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_module(a, 'a1', nets={})
    # debug_out matched by pattern; test_enable waived explicitly.
    top.add_no_connect_port('a1', 'test_enable', reason='manual')
    rule = DanglingPortRule(config={'ignore_patterns': ['debug_*']})
    assert list(rule.check(top)) == []


def test_no_connect_port_marker_import():
    """Marker is importable from silk.lint_markers."""
    from silk.lint_markers import NoConnectPort
    m = NoConnectPort(instance='a1', port='x', reason='because')
    assert m.is_lint_marker is True
    assert m.instance == 'a1'
    assert m.port == 'x'
    assert m.reason == 'because'


def test_power_port_unconnected_metadata():
    r = PowerPortUnconnectedRule()
    assert r.rule_id == "STRUCT-002"
    assert r.category == "structural"
    assert r.default_severity == Severity.ERROR


def test_power_port_unconnected_fires_on_unbound_power_pin(tmp_path):
    a = _sub_ppc(tmp_path, 'a', {
        'in': PortDir.IN,
        'vdd': (PortDir.IN, PortRole.POWER),
    })
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_module(a, 'a1', nets={'in': 'wire1'})  # vdd left unbound
    rule = PowerPortUnconnectedRule()
    violations = list(rule.check(top))
    assert any(v.rule_id == "STRUCT-002" for v in violations)


def test_power_port_unconnected_does_not_fire_on_signal_dangling(tmp_path):
    """Only POWER/GROUND ports trigger STRUCT-002; signal ports are STRUCT-001."""
    a = _sub_ppc(tmp_path, 'a', {
        'in': PortDir.IN,
        'out': PortDir.OUT,  # signal, not POWER/GROUND
    })
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_module(a, 'a1', nets={'in': 'wire1'})  # out unbound (but signal)
    rule = PowerPortUnconnectedRule()
    assert list(rule.check(top)) == []


def test_floating_gate_metadata():
    r = FloatingGateRule()
    assert r.rule_id == "STRUCT-004"
    assert r.category == "structural"
    assert r.default_severity == Severity.ERROR


def test_floating_gate_does_not_fire_on_module_with_no_devices(tmp_path):
    m = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    rule = FloatingGateRule()
    assert list(rule.check(m)) == []


# === Task 10: STRUCT-005..STRUCT-008 ===
from silk.lint.rules.structural import (
    BodyNotTiedRule, MissingParamRule,
    SignalOnSupplyNetRule, UnusedInstanceRule,
)


def test_body_not_tied_metadata():
    r = BodyNotTiedRule()
    assert r.rule_id == "STRUCT-005"
    assert r.category == "structural"
    assert r.default_severity == Severity.WARNING


def test_body_not_tied_does_not_fire_on_module_with_no_devices(tmp_path):
    m = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    rule = BodyNotTiedRule()
    assert list(rule.check(m)) == []


def test_missing_param_metadata():
    r = MissingParamRule()
    assert r.rule_id == "STRUCT-006"
    assert r.category == "structural"
    assert r.default_severity == Severity.ERROR


def test_missing_param_does_not_fire_on_empty_module(tmp_path):
    m = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    rule = MissingParamRule()
    assert list(rule.check(m)) == []


def test_missing_param_is_silent_on_a_device_that_declares_no_parameters(tmp_path):
    """A device with neither a factory parameter contract, a legacy
    ``param_list``, nor a passive value is not reported: the rule cannot
    tell an omission from an absent contract, and STRUCT-006 is an ERROR
    (TASK-44). Before the fix, any class whose name contained 'mos'/'fet'
    was asked for 'W' and 'L' whether or not it declared them."""
    from silk.circuit.circuit_elements import ElementType

    class nmos_no_contract:
        pass

    m = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    m._element_mgr.elem_dict['mn'] = {'type': ElementType.DEVICE,
                                      'device': nmos_no_contract(),
                                      'typeName': 'nmos_no_contract'}
    assert list(MissingParamRule().check(m)) == []


def test_signal_on_supply_net_metadata():
    r = SignalOnSupplyNetRule()
    assert r.rule_id == "STRUCT-007"
    assert r.category == "structural"
    assert r.default_severity == Severity.ERROR


def test_signal_on_supply_net_fires_when_signal_shares_net_with_power(tmp_path):
    """A net carrying both POWER and ANALOG roles fires STRUCT-007."""
    a = _sub_ppc(tmp_path, 'a', {
        'vdd_in': (PortDir.IN, PortRole.POWER),
    })
    b = _sub_ppc(tmp_path, 'b', {
        'sig': (PortDir.IN, PortRole.ANALOG),
    })
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_module(a, 'a1', nets={'vdd_in': 'shared'})
    top.add_module(b, 'b1', nets={'sig': 'shared'})
    rule = SignalOnSupplyNetRule()
    violations = list(rule.check(top))
    assert any(v.rule_id == "STRUCT-007" for v in violations)


def test_signal_on_supply_net_clean_when_no_supply(tmp_path):
    a = _sub_ppc(tmp_path, 'a', {
        'in': (PortDir.IN, PortRole.ANALOG),
        'out': (PortDir.OUT, PortRole.ANALOG),
    })
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_module(a, 'a1', nets={'in': 'wire1', 'out': 'wire2'})
    rule = SignalOnSupplyNetRule()
    assert list(rule.check(top)) == []


def test_unused_instance_metadata():
    r = UnusedInstanceRule()
    assert r.rule_id == "STRUCT-008"
    assert r.category == "structural"
    assert r.default_severity == Severity.WARNING


def test_unused_instance_fires_on_unconnected_submodule(tmp_path):
    """A submodule instance with all pins unbound fires STRUCT-008."""
    a = _sub_ppc(tmp_path, 'a', {
        'in': PortDir.IN,
        'out': PortDir.OUT,
    })
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_module(a, 'a1', nets={})  # nothing bound
    rule = UnusedInstanceRule()
    violations = list(rule.check(top))
    assert any(v.rule_id == "STRUCT-008" for v in violations)


def test_unused_instance_clean_when_at_least_one_pin_bound(tmp_path):
    a = _sub_ppc(tmp_path, 'a', {
        'in': PortDir.IN,
        'out': PortDir.OUT,
    })
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_module(a, 'a1', nets={'in': 'wire1'})
    rule = UnusedInstanceRule()
    violations = list(rule.check(top))
    assert all(v.rule_id != "STRUCT-008" for v in violations)


# === Task 7: STRUCT-009, STRUCT-010 — port declaration DRC ===
from silk.lint.rules.structural import (
    PowerGroundPortRequired, PortDirectionSanity,
)
from silk.lint.locations import ModuleLocation
from silk.circuit.circuit_elements import ElementType


def _inject_device(module):
    """Add a dummy primitive device entry so the module passes the
    'has transistors' gate in STRUCT-009."""
    mock_device = MagicMock()
    mock_device.pin_list = {}
    module._element_mgr.elem_dict['mn1'] = {
        'device': mock_device,
        'type': ElementType.DEVICE,
    }


# ---- STRUCT-009: PowerGroundPortRequired -----------------------------

def test_power_ground_port_required_metadata():
    r = PowerGroundPortRequired()
    assert r.rule_id == "STRUCT-009"
    assert r.category == "structural"
    assert r.default_severity == Severity.ERROR
    assert len(r.rationale) > 20


def test_power_ground_port_required_applicable_to_module():
    assert Module in PowerGroundPortRequired().applicable_to


def test_power_ground_port_required_fires_when_missing_power(tmp_path):
    """Subckt with devices declaring only a GROUND port fires missing POWER."""
    m = _sub_ppc(tmp_path, 'blk', {
        'gnd': (PortDir.IN, PortRole.GROUND),
        'sig': (PortDir.IN, PortRole.ANALOG),
    })
    _inject_device(m)
    rule = PowerGroundPortRequired()
    violations = list(rule.check(m))
    assert len(violations) == 1
    assert violations[0].rule_id == "STRUCT-009"
    assert "POWER" in violations[0].message
    assert isinstance(violations[0].location, ModuleLocation)
    assert violations[0].location.module_name == 'blk'


def test_power_ground_port_required_fires_when_missing_ground(tmp_path):
    """Subckt with devices declaring only a POWER port fires missing GROUND."""
    m = _sub_ppc(tmp_path, 'blk', {
        'vdd': (PortDir.IN, PortRole.POWER),
        'sig': (PortDir.IN, PortRole.ANALOG),
    })
    _inject_device(m)
    rule = PowerGroundPortRequired()
    violations = list(rule.check(m))
    assert len(violations) == 1
    assert violations[0].rule_id == "STRUCT-009"
    assert "GROUND" in violations[0].message


def test_power_ground_port_required_clean_when_both_present(tmp_path):
    m = _sub_ppc(tmp_path, 'blk', {
        'vdd': (PortDir.IN, PortRole.POWER),
        'gnd': (PortDir.IN, PortRole.GROUND),
        'sig': (PortDir.IN, PortRole.ANALOG),
    })
    _inject_device(m)
    rule = PowerGroundPortRequired()
    assert list(rule.check(m)) == []


def test_power_ground_port_required_skips_behavioral_module(tmp_path):
    """A subckt with NO devices (behavioral-only) is exempt from STRUCT-009."""
    m = _sub_ppc(tmp_path, 'beh', {
        'in': (PortDir.IN, PortRole.ANALOG),
        'out': (PortDir.OUT, PortRole.ANALOG),
    })
    # No devices injected.
    rule = PowerGroundPortRequired()
    assert list(rule.check(m)) == []


def test_power_ground_port_required_skips_top_level_testbench(tmp_path):
    """Top-level testbench (no module_name) is never a subckt."""
    m = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    _inject_device(m)
    rule = PowerGroundPortRequired()
    assert list(rule.check(m)) == []


def test_power_ground_port_required_fires_both_when_neither_declared(tmp_path):
    """Subckt with devices and no supply ports fires two violations."""
    m = _sub_ppc(tmp_path, 'blk', {
        'sig': (PortDir.IN, PortRole.ANALOG),
    })
    _inject_device(m)
    rule = PowerGroundPortRequired()
    violations = list(rule.check(m))
    assert len(violations) == 2
    messages = " ".join(v.message for v in violations)
    assert "POWER" in messages
    assert "GROUND" in messages


# ---- STRUCT-010: PortDirectionSanity --------------------------------

def test_port_direction_sanity_metadata():
    r = PortDirectionSanity()
    assert r.rule_id == "STRUCT-010"
    assert r.category == "structural"
    assert r.default_severity == Severity.WARNING
    assert len(r.rationale) > 20


def test_port_direction_sanity_applicable_to_module():
    assert Module in PortDirectionSanity().applicable_to


def test_port_direction_sanity_fires_on_power_out(tmp_path):
    """POWER port declared with OUT direction → warn."""
    m = _sub_ppc(tmp_path, 'blk', {
        'vdd': (PortDir.OUT, PortRole.POWER),
    })
    rule = PortDirectionSanity()
    violations = list(rule.check(m))
    assert len(violations) == 1
    assert violations[0].rule_id == "STRUCT-010"
    assert "POWER" in violations[0].message
    assert "OUT" in violations[0].message
    assert isinstance(violations[0].location, PortLocation)
    assert violations[0].location.port_name == 'vdd'
    assert violations[0].location.module_name == 'blk'


def test_port_direction_sanity_fires_on_ground_inout(tmp_path):
    """GROUND port declared with INOUT direction → warn."""
    m = _sub_ppc(tmp_path, 'blk', {
        'gnd': (PortDir.INOUT, PortRole.GROUND),
    })
    rule = PortDirectionSanity()
    violations = list(rule.check(m))
    assert len(violations) == 1
    assert violations[0].rule_id == "STRUCT-010"
    assert "GROUND" in violations[0].message
    assert "INOUT" in violations[0].message


def test_port_direction_sanity_clean_on_power_in(tmp_path):
    m = _sub_ppc(tmp_path, 'blk', {
        'vdd': (PortDir.IN, PortRole.POWER),
        'gnd': (PortDir.IN, PortRole.GROUND),
    })
    rule = PortDirectionSanity()
    assert list(rule.check(m)) == []


def test_port_direction_sanity_ignores_non_supply_ports(tmp_path):
    """ANALOG/DIGITAL ports with OUT direction are fine — not supply roles."""
    m = _sub_ppc(tmp_path, 'blk', {
        'sig_out': (PortDir.OUT, PortRole.ANALOG),
        'dig_out': (PortDir.OUT, PortRole.DIGITAL),
    })
    rule = PortDirectionSanity()
    assert list(rule.check(m)) == []


def test_port_direction_sanity_exempts_a_regulated_output(tmp_path):
    """TASK-94: a POWER OUT next to a POWER IN is a regulator's output,
    not a direction mix-up — silent."""
    m = _sub_ppc(tmp_path, 'ldo', {
        'vin': (PortDir.IN, PortRole.POWER),
        'vout': (PortDir.OUT, PortRole.POWER),
        'vss': (PortDir.IN, PortRole.GROUND),
    })
    assert list(PortDirectionSanity().check(m)) == []


def test_port_direction_sanity_exemption_requires_the_same_role(tmp_path):
    """A GROUND input does not license a POWER output; a GROUND input does
    license a gated GROUND output (footer switch)."""
    m = _sub_ppc(tmp_path, 'blk', {
        'vout': (PortDir.OUT, PortRole.POWER),
        'vss': (PortDir.IN, PortRole.GROUND),
        'vss_gated': (PortDir.OUT, PortRole.GROUND),
    })
    violations = list(PortDirectionSanity().check(m))
    assert [v.location.port_name for v in violations] == ['vout']
    assert "no POWER-role input" in violations[0].message


def test_port_direction_sanity_bare_supply_direction_counts_as_receiving(tmp_path):
    """The examples' idiom, ``'vdd': PortDir.POWER``, receives the rail
    (TASK-45 derives the role), so it licenses a POWER OUT."""
    m = _sub_ppc(tmp_path, 'hdr', {
        'vdd': PortDir.POWER,
        'vdd_gated': (PortDir.OUT, PortRole.POWER),
    })
    assert list(PortDirectionSanity().check(m)) == []


def test_port_direction_sanity_inout_does_not_count_as_receiving(tmp_path):
    """INOUT is reported itself and does not license a same-role OUT."""
    m = _sub_ppc(tmp_path, 'cp', {
        'vdd': (PortDir.INOUT, PortRole.POWER),
        'vpp': (PortDir.OUT, PortRole.POWER),
    })
    assert [v.location.port_name for v in PortDirectionSanity().check(m)] == ['vdd', 'vpp']


def test_port_direction_sanity_skips_top_level(tmp_path):
    """Top-level testbench (no module_name) is skipped."""
    m = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    # Even if we forcibly gave it ports, it's not a subckt.
    m.define_module_ports({'vdd': (PortDir.OUT, PortRole.POWER)})
    rule = PortDirectionSanity()
    assert list(rule.check(m)) == []


# ---------------------------------------------------------------------------
# STRUCT-011 SINGLE_CONNECTION_NET (TASK-35)
# ---------------------------------------------------------------------------

def _single(module):
    """(net_name, message) for each STRUCT-011 finding, in report order."""
    return [(v.location.net_name, v.message)
            for v in SingleConnectionNetRule().check(module)]


def test_single_connection_metadata():
    r = SingleConnectionNetRule()
    assert r.rule_id == "STRUCT-011"
    assert r.category == "structural"
    assert r.default_severity == Severity.WARNING
    assert r.requires_simulation is False
    assert Module in r.applicable_to
    assert "add_no_connect" in r.rationale


def test_single_connection_is_in_the_default_catalog():
    """AC #1: default-registered, not opt-in."""
    from silk.lint.rules import DEFAULT_CATALOG, OPTIONAL_CATALOG
    assert "STRUCT-011" in {r.rule_id for r in DEFAULT_CATALOG()}
    assert "STRUCT-011" not in {r.rule_id for r in OPTIONAL_CATALOG()}


def test_single_connection_fires_on_lone_submodule_port(tmp_path):
    a = Module('TEST', str(tmp_path), 'a.cir', pdk_details=_pdk())
    a.define_module_name('a')
    a.define_module_ports({'out': PortDir.OUT, 'in': PortDir.IN})
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_module(a, 'a1', nets={'out': 'y', 'in': 'x'})
    found = _single(top)
    assert [n for n, _ in found] == ['y', 'x']
    assert found[0][1] == (
        "Net 'y' has only one connection: a1.out(output) — "
        "nothing else is wired to it")
    assert isinstance(SingleConnectionNetRule().check(top).__next__(),
                      InterfaceViolation)


def test_single_connection_location_names_the_one_pin(tmp_path):
    a = Module('TEST', str(tmp_path), 'a.cir', pdk_details=_pdk())
    a.define_module_name('a')
    a.define_module_ports({'out': PortDir.OUT})
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_module(a, 'a1', nets={'out': 'y'})
    v = next(iter(SingleConnectionNetRule().check(top)))
    assert isinstance(v.location, NetLocation)
    assert v.location.net_name == 'y'
    assert v.location.port_names == ['a1.out']


def test_single_connection_quiet_when_two_pins_share_the_net(tmp_path):
    a = Module('TEST', str(tmp_path), 'a.cir', pdk_details=_pdk())
    a.define_module_name('a')
    a.define_module_ports({'out': PortDir.OUT})
    b = Module('TEST', str(tmp_path), 'b.cir', pdk_details=_pdk())
    b.define_module_name('b')
    b.define_module_ports({'in': PortDir.IN})
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_module(a, 'a1', nets={'out': 'y'})
    top.add_module(b, 'b1', nets={'in': 'y'})
    assert _single(top) == []


def test_single_connection_leaves_zero_pin_nets_to_struct_003(tmp_path):
    """The two rules partition the cases; neither double-reports."""
    m = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    m.add_net('orphan')
    assert _single(m) == []
    assert [v.location.net_name for v in FloatingNetRule().check(m)] == ['orphan']


def test_single_connection_suppressed_by_add_no_connect(tmp_path):
    """AC #1 suppression path: Module.add_no_connect, with a reason."""
    a = Module('TEST', str(tmp_path), 'a.cir', pdk_details=_pdk())
    a.define_module_name('a')
    a.define_module_ports({'out': PortDir.OUT})
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_module(a, 'a1', nets={'out': 'probe'})
    assert [n for n, _ in _single(top)] == ['probe']
    top.add_no_connect('probe', reason='scope probe point, wired in the bench')
    assert _single(top) == []


def test_single_connection_exempts_ports_of_the_module_under_check(tmp_path):
    """A subcircuit port net's second connection is made by the parent.

    Without the exemption every single-device port net in every library
    block would be reported — the rule would be unusable inside a .subckt.
    """
    blk = Module('TEST', str(tmp_path), 'blk.cir', pdk_details=_pdk())
    blk.define_module_name('blk')
    blk.define_module_ports({'out': PortDir.OUT, 'in': PortDir.IN})
    inner = Module('TEST', str(tmp_path), 'inner.cir', pdk_details=_pdk())
    inner.define_module_name('inner')
    inner.define_module_ports({'a': PortDir.IN, 'z': PortDir.OUT})
    blk.add_module(inner, 'x1', nets={'a': 'in', 'z': 'out'})
    assert _single(blk) == []
    # An internal net with one pin is still reported — only ports are exempt.
    blk.add_module(inner, 'x2', nets={'a': 'in', 'z': 'dead_end'})
    assert [n for n, _ in _single(blk)] == ['dead_end']


def test_single_connection_counts_source_bindings(tmp_path):
    from silk import sources
    m = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    m.add_source(sources.VoltageSource('v1', positive='hi', negative='lo', dc=1.8))
    assert [n for n, _ in _single(m)] == ['hi', 'lo']
    m.add_source(sources.VoltageSource('v2', positive='hi', negative='lo', dc=0.9))
    assert _single(m) == []


def test_single_connection_counts_passive_bindings(tmp_path):
    from silk.elements import elements
    m = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    m.add(elements.resistor('r1', p='a', n='b', value=1e3))
    assert [n for n, _ in _single(m)] == ['a', 'b']
    m.add(elements.resistor('r2', p='a', n='b', value=2e3))
    assert _single(m) == []


def test_single_connection_counts_device_bindings(pdk, tmp_path):
    """The lone-gate shape from TASK-35's repro, on a real PDK device."""
    m = Module('PDK_skywater130', str(tmp_path), 'top.cir', pdk_details=pdk)
    with m:
        pdk.nmos1p8('mn', d='y', g='nowhere', s='gnd', b='gnd', w=1, l=0.15)
    assert [n for n, _ in _single(m)] == ['y', 'nowhere']


def test_single_connection_does_not_double_count_the_marker_pin(tmp_path):
    """A NoConnect adds a real PortBinding; it must not turn a floating net
    into a 'single connection' one, nor a single-pin net into a two-pin one.
    """
    m = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    m.add_net('tap')
    m.add_no_connect('tap', reason='reserved')
    assert len(m.net_dict['tap'].pins) == 1      # the marker itself
    assert _single(m) == []
    assert list(FloatingNetRule().check(m)) == []


# ---------------------------------------------------------------------------
# STRUCT-011 vs the ground node '0' (TASK-88)
# ---------------------------------------------------------------------------

def test_single_connection_exempts_a_lone_ground_port_on_node_zero(tmp_path):
    """Pre-TASK-88 this was a FALSE POSITIVE: 'Net 0 has only one
    connection: u1.gnd(ground) — nothing else is wired to it'. Node 0 is
    never floating — the simulator is always its other party — so a lone
    GROUND-role binding on '0' must not be reported."""
    blk = Module('TEST', str(tmp_path), 'blk.cir', pdk_details=_pdk())
    blk.define_module_name('blk')
    blk.define_module_ports({'a': PortDir.IN, 'y': PortDir.OUT,
                              'gnd': PortDir.GROUND})
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_module(blk, 'u1', nets={'a': 'a', 'y': 'y', 'gnd': '0'})

    assert len(top.net_dict['0'].pins) == 1
    assert '0' not in [n for n, _ in _single(top)]


def test_single_connection_exempts_a_lone_power_port_on_node_zero(tmp_path):
    """Pre-TASK-88 a lone POWER port on '0' was double-reported: INT-002's
    'Net 0 is the simulator ground node — supply is grounded: u1.vdd(power)'
    AND STRUCT-011's 'Net 0 has only one connection: u1.vdd(power) —
    nothing else is wired to it' for the same mistake. STRUCT-011 now stays
    silent on '0' and INT-002 is the only finding."""
    from silk.lint.rules.interface import PowerGroundShortRule

    blk = Module('TEST', str(tmp_path), 'blk.cir', pdk_details=_pdk())
    blk.define_module_name('blk')
    blk.define_module_ports({'a': PortDir.IN, 'y': PortDir.OUT,
                              'vdd': PortDir.POWER, 'gnd': PortDir.GROUND})
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_module(blk, 'u1', nets={'a': 'a', 'y': 'y', 'vdd': '0', 'gnd': 'gnd'})

    assert len(top.net_dict['0'].pins) == 1
    assert '0' not in [n for n, _ in _single(top)]
    assert [v.message for v in PowerGroundShortRule().check(top)] == [
        "Net '0' is the simulator ground node — supply is grounded: "
        "u1.vdd(power)"]
