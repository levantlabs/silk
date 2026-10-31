import pytest
from unittest.mock import MagicMock
from silk.lint.rules.interface import (
    DriverConflictRule,
    PowerGroundShortRule,
    RoleMismatchRule,
    NoDriverRule,
    ClockOnAnalogRule,
)
from silk.lint.severity import Severity
from silk.lint.violations import InterfaceViolation
from silk.lint.locations import NetLocation
from silk.circuit.modules import Module
from silk.circuit.circuit_elements import PortDir, PortRole


def _pdk():
    pdk = MagicMock()
    pdk.model_names = []
    pdk._devices = {}
    return pdk


def _sub(tmp_path, name, ports):
    m = Module('TEST', str(tmp_path), f'{name}.cir', pdk_details=_pdk())
    m.define_module_name(name)
    m.define_module_ports(ports)
    return m


def test_interface_rule_ids():
    assert DriverConflictRule().rule_id == "INT-001"
    assert PowerGroundShortRule().rule_id == "INT-002"
    assert RoleMismatchRule().rule_id == "INT-003"
    assert NoDriverRule().rule_id == "INT-004"
    assert ClockOnAnalogRule().rule_id == "INT-005"


def test_interface_rule_category():
    for cls in (DriverConflictRule, PowerGroundShortRule, RoleMismatchRule,
                NoDriverRule, ClockOnAnalogRule):
        assert cls().category == "interface"


def test_interface_rule_severity_default_error():
    for cls in (DriverConflictRule, PowerGroundShortRule, RoleMismatchRule,
                NoDriverRule, ClockOnAnalogRule):
        assert cls().default_severity == Severity.ERROR


def test_interface_rule_rationale_non_empty():
    for cls in (DriverConflictRule, PowerGroundShortRule, RoleMismatchRule,
                NoDriverRule, ClockOnAnalogRule):
        r = cls()
        assert r.rationale
        assert len(r.rationale) > 20


def test_driver_conflict_detection_on_two_driver_module(tmp_path):
    a = _sub(tmp_path, 'a', {'out': PortDir.OUT})
    b = _sub(tmp_path, 'b', {'out': PortDir.OUT})
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_module(a, 'a1', nets={'out': 'shared'})
    top.add_module(b, 'b1', nets={'out': 'shared'})

    rule = DriverConflictRule()
    violations = list(rule.check(top))
    assert len(violations) >= 1
    assert all(isinstance(v, InterfaceViolation) for v in violations)
    assert all(v.rule_id == "INT-001" for v in violations)
    assert all(isinstance(v.location, NetLocation) for v in violations)
    assert violations[0].location.net_name == "shared"


def test_power_ground_short_detection(tmp_path):
    a = _sub(tmp_path, 'a', {'vdd': (PortDir.IN, PortRole.POWER)})
    b = _sub(tmp_path, 'b', {'gnd': (PortDir.IN, PortRole.GROUND)})
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_module(a, 'a1', nets={'vdd': 'shared_rail'})
    top.add_module(b, 'b1', nets={'gnd': 'shared_rail'})

    rule = PowerGroundShortRule()
    violations = list(rule.check(top))
    assert len(violations) >= 1
    assert all(v.rule_id == "INT-002" for v in violations)
    assert violations[0].location.net_name == "shared_rail"


def test_role_mismatch_detection(tmp_path):
    src = _sub(tmp_path, 'src', {'out': (PortDir.OUT, PortRole.DIGITAL)})
    dst = _sub(tmp_path, 'dst', {'inp': (PortDir.IN, PortRole.ANALOG)})
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_module(src, 'src1', nets={'out': 'mixed_net'})
    top.add_module(dst, 'dst1', nets={'inp': 'mixed_net'})

    rule = RoleMismatchRule()
    violations = list(rule.check(top))
    assert len(violations) >= 1
    assert all(v.rule_id == "INT-003" for v in violations)


def test_clean_module_no_violations(tmp_path):
    a = _sub(tmp_path, 'a', {'out': (PortDir.OUT, PortRole.DIGITAL)})
    b = _sub(tmp_path, 'b', {'inp': (PortDir.IN, PortRole.DIGITAL)})
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_module(a, 'a1', nets={'out': 'wire'})
    top.add_module(b, 'b1', nets={'inp': 'wire'})

    for cls in (DriverConflictRule, PowerGroundShortRule, RoleMismatchRule,
                NoDriverRule, ClockOnAnalogRule):
        violations = list(cls().check(top))
        assert violations == [], f"{cls.__name__} produced spurious violations: {violations}"


# ---------------------------------------------------------------------------
# BUG-001 — top-level sources must count as drivers for INT-004
# ---------------------------------------------------------------------------

def test_voltage_source_counts_as_driver(tmp_path):
    """Regression for BUG-001: a net driven by a top-level VoltageSource
    into multiple submodule inputs must NOT trigger INT-004 (no driver)."""
    from silk.sources.classes import VoltageSource

    sink_a = _sub(tmp_path, 'sink_a', {'vbias': PortDir.IN})
    sink_b = _sub(tmp_path, 'sink_b', {'vbias': PortDir.IN})
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_source(VoltageSource('vbias_src', positive='vbias',
                                 negative='gnd', dc=0.7))
    top.add_module(sink_a, 'a1', nets={'vbias': 'vbias'})
    top.add_module(sink_b, 'b1', nets={'vbias': 'vbias'})

    violations = list(NoDriverRule().check(top))
    # 'vbias' is driven by the source; no INT-004 should fire for it.
    assert not any(v.location.net_name == 'vbias' for v in violations), (
        f'VoltageSource on vbias should count as a driver; got: {violations}')


def test_current_source_counts_as_driver(tmp_path):
    """Same as above for a CurrentSource."""
    from silk.sources.classes import CurrentSource

    sink = _sub(tmp_path, 'sink', {'ibias': PortDir.IN})
    load = _sub(tmp_path, 'load', {'ibias': PortDir.IN})
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_source(CurrentSource('ibias_src', positive='ibias',
                                 negative='gnd', dc=10e-6))
    top.add_module(sink, 's1', nets={'ibias': 'ibias'})
    top.add_module(load, 'l1', nets={'ibias': 'ibias'})

    violations = list(NoDriverRule().check(top))
    assert not any(v.location.net_name == 'ibias' for v in violations)


def test_vcvs_output_drives_but_control_does_not(tmp_path):
    """VCVS: out_pos/out_neg drive; ctrl_pos/ctrl_neg only sense. A net
    connected ONLY to VCVS control terminals is still undriven."""
    from silk.sources.classes import VoltageControlledVoltageSource as VCVS

    # Two input sinks on a control net that has no other driver — control
    # pins alone should NOT be treated as a driver.
    sink_a = _sub(tmp_path, 'sa', {'vin': PortDir.IN})
    sink_b = _sub(tmp_path, 'sb', {'vin': PortDir.IN})
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    # VCVS: output goes to 'vout_net' (driven), sense from 'ctrl_net'.
    top.add_source(VCVS('eamp',
                        out_pos='vout_net', out_neg='gnd',
                        ctrl_pos='ctrl_net', ctrl_neg='gnd',
                        gain=10))
    top.add_module(sink_a, 'a1', nets={'vin': 'ctrl_net'})
    top.add_module(sink_b, 'b1', nets={'vin': 'ctrl_net'})

    violations = list(NoDriverRule().check(top))
    ctrl_viols = [v for v in violations if v.location.net_name == 'ctrl_net']
    assert len(ctrl_viols) == 1, (
        f'Control-only net must still report INT-004: {violations}')


def test_vcvs_output_terminal_counts_as_driver(tmp_path):
    """Counterpart: when VCVS drives a net through out_pos, that net IS driven."""
    from silk.sources.classes import VoltageControlledVoltageSource as VCVS

    sink_a = _sub(tmp_path, 'sa', {'vin': PortDir.IN})
    sink_b = _sub(tmp_path, 'sb', {'vin': PortDir.IN})
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_source(VCVS('eamp',
                        out_pos='vbuf', out_neg='gnd',
                        ctrl_pos='vin_raw', ctrl_neg='gnd',
                        gain=1.0))
    top.add_module(sink_a, 'a1', nets={'vin': 'vbuf'})
    top.add_module(sink_b, 'b1', nets={'vin': 'vbuf'})

    violations = list(NoDriverRule().check(top))
    assert not any(v.location.net_name == 'vbuf' for v in violations)


def test_two_voltage_sources_on_same_net_trigger_driver_conflict(tmp_path):
    """Two top-level VoltageSources driving the same net is a real bug —
    INT-001 (DriverConflict) must still catch it with the new fix."""
    from silk.sources.classes import VoltageSource

    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_source(VoltageSource('v1', positive='shared', negative='gnd', dc=1.8))
    top.add_source(VoltageSource('v2', positive='shared', negative='gnd', dc=1.2))

    violations = list(DriverConflictRule().check(top))
    assert any(v.location.net_name == 'shared' for v in violations), (
        f'Two sources on same net must trigger INT-001: {violations}')


def test_many_source_negatives_on_gnd_no_false_conflict(tmp_path):
    """Seven top-level sources all with negative='gnd' must NOT trigger
    INT-001: only `positive` terminals actively drive. Regression from
    BUG-001 fix that initially over-broadened 'driver' to every source
    terminal."""
    from silk.sources.classes import VoltageSource

    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    # Flash ADC-style: many sources, one gnd return.
    for i, (pos, volts) in enumerate([
        ('vdd', 1.8), ('vbias', 0.7), ('vref', 1.4), ('vrefl', 0.8),
        ('vclk', 1.8), ('vclkb', 1.8), ('vin', 1.1),
    ]):
        top.add_source(VoltageSource(f'v{i}', positive=pos, negative='gnd',
                                     dc=volts))

    violations = list(DriverConflictRule().check(top))
    gnd_viols = [v for v in violations if v.location.net_name == 'gnd']
    assert gnd_viols == [], (
        f'gnd should not be flagged as multi-driven just because many '
        f'sources declare negative=gnd: {gnd_viols}')


# ---------------------------------------------------------------------------
# INT-004 × STRUCT-011 hand-off (TASK-35 AC #2)
# ---------------------------------------------------------------------------

def test_no_driver_skips_single_binding_nets_and_struct_011_covers_them(tmp_path):
    """AC #2, second branch: INT-004 keeps its ``>= 2 bindings`` filter and
    STRUCT-011 provably covers the single-binding case.

    ``sink`` is one undriven input and nothing else — the shape a net-name
    typo or an unwired DUT input produces. INT-004 cannot see it (the
    helper it iterates drops nets with fewer than two submodule/source
    bindings), so the report must come from STRUCT-011, exactly once.
    """
    from silk.lint.registry import RuleRegistry
    from silk.lint.rules.structural import SingleConnectionNetRule

    sub = _sub(tmp_path, 'sink', {'a': (PortDir.IN, PortRole.ANALOG)})
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_module(sub, 'u1', nets={'a': 'sink'})

    assert len(top.net_dict['sink'].pins) == 1
    assert list(NoDriverRule().check(top)) == []
    assert [v.location.net_name
            for v in SingleConnectionNetRule().check(top)] == ['sink']

    # ... and through the default registry, with no double report.
    by_net = [(v.rule_id, v.location.net_name)
              for v in RuleRegistry.default().run(top).violations
              if getattr(v.location, 'net_name', None) == 'sink']
    assert by_net == [('STRUCT-011', 'sink')]


def test_no_driver_still_owns_the_multi_binding_case(tmp_path):
    """The other side of the hand-off: two undriven inputs on one net stay
    INT-004's, and STRUCT-011 must not chime in."""
    from silk.lint.rules.structural import SingleConnectionNetRule

    sub = _sub(tmp_path, 'sink', {'a': (PortDir.IN, PortRole.ANALOG)})
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_module(sub, 'u1', nets={'a': 'sink'})
    top.add_module(sub, 'u2', nets={'a': 'sink'})

    assert [v.location.net_name for v in NoDriverRule().check(top)] == ['sink']
    assert list(SingleConnectionNetRule().check(top)) == []


# ---------------------------------------------------------------------------
# '0' is the SPICE ground node (TASK-85; matrix cells S4b / D6)
# ---------------------------------------------------------------------------

def _inverter_block(tmp_path):
    return _sub(tmp_path, 'inv', {'a': PortDir.IN, 'y': PortDir.OUT,
                                  'vdd': PortDir.POWER, 'gnd': PortDir.GROUND})


def _top(tmp_path, name='top'):
    return Module('TEST', str(tmp_path), f'{name}.cir', pdk_details=_pdk())


_GROUNDED_SUPPLY = ("Net '0' is the simulator ground node — supply is grounded: "
                    "u1.vdd(power)")


def test_power_port_on_node_zero_is_a_grounded_supply(tmp_path):
    """S4b: a POWER port wired to '0' while the source drives 'vdd'. Both
    backends run an unpowered DUT (v(y)=0.0, audit 2026-09-02) and before
    TASK-85 INT-002 was silent — the net held one POWER binding and one
    role-less source return, and no rule knew '0' is ground. Now INT-002
    names the grounded supply."""
    from silk.sources.classes import VoltageSource

    top = _top(tmp_path)
    top.add_module(_inverter_block(tmp_path), 'u1',
                    nets={'a': 'a', 'y': 'y', 'vdd': '0', 'gnd': 'gnd'})
    top.add_source(VoltageSource('vdd', positive='vdd', negative='0', dc=1.8))

    assert [v.message for v in PowerGroundShortRule().check(top)] == [_GROUNDED_SUPPLY]
    assert [v.location.net_name for v in PowerGroundShortRule().check(top)] == ['0']


def test_lone_power_port_on_node_zero_not_double_reported(tmp_path):
    """A POWER port that is the ONLY binding on '0' (the source returns on
    some other net). ``_iter_module_net_bindings`` drops single-binding
    nets, which before TASK-85 hid this shape from INT-002 entirely; the
    simulator itself is always the other party on node 0, so the ground
    node is now let through with one binding and INT-002 fires. Before
    TASK-88, STRUCT-011 also reported the same net as a one-ended
    connection ("Net '0' has only one connection: u1.vdd(power)") — noise
    on the same mistake, since the simulator is always the net's other
    party. STRUCT-011 now exempts '0' and INT-002 is the only finding."""
    from silk.sources.classes import VoltageSource
    from silk.lint.rules.structural import SingleConnectionNetRule

    top = _top(tmp_path)
    top.add_module(_inverter_block(tmp_path), 'u1',
                    nets={'a': 'a', 'y': 'y', 'vdd': '0', 'gnd': 'gnd'})
    top.add_source(VoltageSource('vdd', positive='vdd', negative='gnd', dc=1.8))

    assert len(top.net_dict['0'].pins) == 1
    assert [v.message for v in PowerGroundShortRule().check(top)] == [_GROUNDED_SUPPLY]
    assert '0' not in [v.location.net_name for v in SingleConnectionNetRule().check(top)]


def test_power_and_ground_ports_both_on_node_zero(tmp_path):
    """Both rails on '0' is still one INT-002 finding, worded for the ground
    node, and it lists the GROUND-role port too so the location names every
    rail binding involved."""
    top = _top(tmp_path)
    top.add_module(_inverter_block(tmp_path), 'u1',
                    nets={'a': 'a', 'y': 'y', 'vdd': '0', 'gnd': '0'})

    viols = list(PowerGroundShortRule().check(top))
    assert [v.message for v in viols] == [
        "Net '0' is the simulator ground node — supply is grounded: "
        "u1.vdd(power), u1.gnd(ground)"]
    assert viols[0].location.port_names == ['u1.vdd', 'u1.gnd']


def test_no_driver_never_reports_node_zero(tmp_path):
    """D6: DUT gnd on 'gnd', both source returns on '0'. Before TASK-85
    INT-004 reported "Net '0' has no driver: vdd.negative(input),
    vin.negative(input)" — a false positive on the ground node itself (two
    role-less input-direction source returns, no output binding). The
    ground node is driven by definition; INT-004 skips it."""
    from silk.sources.classes import VoltageSource

    top = _top(tmp_path)
    top.add_module(_inverter_block(tmp_path), 'u1',
                    nets={'a': 'a', 'y': 'y', 'vdd': 'vdd', 'gnd': 'gnd'})
    top.add_source(VoltageSource('vdd', positive='vdd', negative='0', dc=1.8))
    top.add_source(VoltageSource('vin', positive='a', negative='0', dc=0.0))

    assert {b.instance for b in top.net_dict['0'].pins} == {'vdd', 'vin'}
    assert list(NoDriverRule().check(top)) == []


def test_no_driver_still_fires_on_an_undriven_signal_net_with_node_zero_present(tmp_path):
    """The skip is for the net NAMED '0' only: an undriven signal net in the
    same module is still reported, so the ground-node exemption cannot
    mask a real NO_DRIVER."""
    from silk.sources.classes import VoltageSource

    sink = _sub(tmp_path, 'sink', {'a': (PortDir.IN, PortRole.ANALOG),
                                   'gnd': PortDir.GROUND})
    top = _top(tmp_path)
    top.add_module(sink, 'u1', nets={'a': 'sense', 'gnd': '0'})
    top.add_module(sink, 'u2', nets={'a': 'sense', 'gnd': '0'})
    top.add_source(VoltageSource('vdd', positive='vdd', negative='0', dc=1.8))

    assert [v.location.net_name for v in NoDriverRule().check(top)] == ['sense']


@pytest.mark.parametrize('name', ['gnd', 'GND', 'vss', 'VSS', 'ground', '00', ' 0'])
def test_only_the_literal_zero_is_the_ground_node(tmp_path, name):
    """Reviewer ruling: '0' is a SPICE language constant; 'gnd' / 'vss' /
    'ground' are naming conventions that belong in a user RoleRuleset
    (ngspice folds 'gnd' to node 0, Xyce does not — TASK-36). None of them
    is recognised by spelling: a POWER port on such a net with a role-less
    source return gets no INT-002 (only a GROUND-role port would trigger
    it), and INT-004 does not skip the net."""
    from silk.lint.rules.interface import _is_ground_node
    from silk.sources.classes import VoltageSource

    assert _is_ground_node('0') and not _is_ground_node(name)

    if name.strip() != name or name == '00':
        # These spellings are neither ground nor legal authored net names.
        with pytest.raises(ValueError):
            VoltageSource('vdd', positive='vdd', negative=name, dc=1.8)
        return
    top = _top(tmp_path)
    top.add_module(_inverter_block(tmp_path), 'u1',
                    nets={'a': 'a', 'y': 'y', 'vdd': name, 'gnd': 'gnd0'})
    top.add_source(VoltageSource('vdd', positive='vdd', negative=name, dc=1.8))
    assert list(PowerGroundShortRule().check(top)) == []

    top = _top(tmp_path, 'top2')
    top.add_module(_inverter_block(tmp_path), 'u1',
                    nets={'a': 'a', 'y': 'y', 'vdd': 'vdd', 'gnd': 'gnd0'})
    top.add_source(VoltageSource('vdd', positive='vdd', negative=name, dc=1.8))
    top.add_source(VoltageSource('vin', positive='a', negative=name, dc=0.0))
    assert [v.location.net_name for v in NoDriverRule().check(top)] == [name]


@pytest.mark.parametrize('simulator', ['ngspice', 'xyce'])
def test_node_zero_knowledge_is_identical_on_every_backend(tmp_path, simulator):
    """AC #3: the '0'-is-ground knowledge is a language constant, not a
    backend condition. The S4b and D6 shapes produce byte-identical
    findings whichever simulator the module targets."""
    from silk.lint.registry import RuleRegistry
    from silk.sources.classes import VoltageSource

    def findings(top):
        return sorted((v.rule_id, v.message)
                      for v in RuleRegistry.default().run(top).violations
                      if v.rule_id.startswith('INT-'))

    s4b = _top(tmp_path, 's4b')
    s4b.simulator = simulator
    s4b.add_module(_inverter_block(tmp_path), 'u1',
                    nets={'a': 'a', 'y': 'y', 'vdd': '0', 'gnd': 'gnd'})
    s4b.add_source(VoltageSource('vdd', positive='vdd', negative='0', dc=1.8))
    assert s4b.simulator == simulator
    assert findings(s4b) == [('INT-002', _GROUNDED_SUPPLY)]

    d6 = _top(tmp_path, 'd6')
    d6.simulator = simulator
    d6.add_module(_inverter_block(tmp_path), 'u1',
                   nets={'a': 'a', 'y': 'y', 'vdd': 'vdd', 'gnd': 'gnd'})
    d6.add_source(VoltageSource('vdd', positive='vdd', negative='0', dc=1.8))
    d6.add_source(VoltageSource('vin', positive='a', negative='0', dc=0.0))
    assert findings(d6) == []


# ---------------------------------------------------------------------------
# An inferred supply role is a guess; a declared one is a statement.
# ---------------------------------------------------------------------------

def _sense_block(tmp_path, vdd_sense=PortDir.IN):
    """A monitor block: 'vdd_sense' is an IN port the '*vdd*' glob guesses
    POWER; 'vdd' / 'gnd' are IN ports the default ruleset fills too."""
    return _sub(tmp_path, 'sense', {'vdd_sense': vdd_sense, 'vout': PortDir.OUT,
                                    'vdd': PortDir.IN, 'gnd': PortDir.IN})


def _two_on_sense(tmp_path, block):
    """Two instances sharing the undriven 'sense' net, supply driven."""
    from silk.sources.classes import VoltageSource
    top = _top(tmp_path)
    top.add_module(block, 'u1', nets={'vdd_sense': 'sense', 'vout': 'y1',
                                       'vdd': 'vdd', 'gnd': 'gnd'})
    top.add_module(block, 'u2', nets={'vdd_sense': 'sense', 'vout': 'y2',
                                       'vdd': 'vdd', 'gnd': 'gnd'})
    top.add_source(VoltageSource('vdd', positive='vdd', negative='gnd', dc=1.8))
    return top


_SENSE_UNDRIVEN = "Net 'sense' has no driver: u1.vdd_sense(input), u2.vdd_sense(input)"


def test_inferred_power_role_does_not_silence_no_driver(tmp_path):
    """AC #1. Pre-fix (measured 2026-09-04 on 65a7a5c): with roles unset
    INT-004 reported ``sense`` AND ``gnd``; after
    ``infer_port_roles(block, RoleRuleset.default())`` filled
    vdd_sense->POWER, vdd->POWER, gnd->GROUND the rule returned ``[]`` —
    the supply skip could not tell a '*vdd*' guess from a statement and
    the real wiring error vanished along with the ground false positive.
    Now the guessed POWER role buys nothing and ``sense`` is reported
    exactly as it is without inference; ``gnd`` (inferred GROUND) stays
    silent."""
    from silk.circuit.port_roles import RoleRuleset, infer_port_roles

    plain = _two_on_sense(tmp_path, _sense_block(tmp_path))
    assert [v.message for v in NoDriverRule().check(plain)] == [
        _SENSE_UNDRIVEN,
        "Net 'gnd' has no driver: u1.gnd(input), u2.gnd(input), vdd.negative(input)"]

    block = _sense_block(tmp_path)
    report = infer_port_roles(block, RoleRuleset.default())
    assert dict(report.filled) == {'vdd_sense': PortRole.POWER,
                                   'vdd': PortRole.POWER, 'gnd': PortRole.GROUND}
    top = _two_on_sense(tmp_path, block)
    assert [(b.role, b.role_inferred) for b in top.net_dict['sense'].pins] == [
        ('power', True), ('power', True)]
    assert [v.message for v in NoDriverRule().check(top)] == [_SENSE_UNDRIVEN]


def test_declared_power_role_still_silences_no_driver(tmp_path):
    """The declared-role skip is untouched: ``Port.supply()`` on the same
    port is the user's statement that the net is fed from outside, and
    INT-004 stays silent on it — with the other ports inferred or not.
    A declared POWER next to an inferred one on the same net is still a
    statement about the net."""
    from silk.circuit.port_roles import RoleRuleset, infer_port_roles
    from silk.circuit.circuit_elements import Port

    declared = _sense_block(tmp_path, vdd_sense=Port.supply())
    infer_port_roles(declared, RoleRuleset.default())     # fills vdd / gnd only
    assert declared.pin_list['vdd_sense'].role_inferred is False
    assert declared.pin_list['vdd'].role_inferred is True
    top = _two_on_sense(tmp_path, declared)
    assert list(NoDriverRule().check(top)) == []

    # Mixed net: one declared POWER binding + one inferred POWER binding.
    guessed = _sense_block(tmp_path)
    infer_port_roles(guessed, RoleRuleset.default())
    from silk.sources.classes import VoltageSource
    mixed = _top(tmp_path, 'mixed')
    mixed.add_module(declared, 'u1', nets={'vdd_sense': 'sense', 'vout': 'y1',
                                            'vdd': 'vdd', 'gnd': 'gnd'})
    mixed.add_module(guessed, 'u2', nets={'vdd_sense': 'sense', 'vout': 'y2',
                                           'vdd': 'vdd', 'gnd': 'gnd'})
    mixed.add_source(VoltageSource('vdd', positive='vdd', negative='gnd', dc=1.8))
    assert sorted(b.role_inferred for b in mixed.net_dict['sense'].pins) == [False, True]
    assert list(NoDriverRule().check(mixed)) == []


def test_inferred_ground_role_still_silences_no_driver(tmp_path):
    """The asymmetry, pinned: an inferred GROUND role keeps the skip.
    Nothing ever drives a return rail in the OUT direction (source
    negatives bind as ``direction='input'``), so INT-004 has nothing to
    say about a ground net whatever the role's provenance — and revoking
    the skip would resurrect the TASK-45 false positive for every user who
    declares ``'gnd': PortDir.IN`` and infers. Measured pre-fix and
    post-fix: ``[]``."""
    from silk.circuit.port_roles import RoleRuleset, infer_port_roles
    from silk.sources.classes import VoltageSource

    leaf = _sub(tmp_path, 'leaf', {'a': PortDir.IN, 'y': PortDir.OUT,
                                   'vdd': PortDir.IN, 'gnd': PortDir.IN})
    infer_port_roles(leaf, RoleRuleset.default())
    tb = _top(tmp_path, 'tb')
    tb.add_module(leaf, 'u1', nets={'a': 'a', 'y': 'y', 'vdd': 'vdd', 'gnd': 'gnd'})
    tb.add_module(leaf, 'u2', nets={'a': 'y', 'y': 'z', 'vdd': 'vdd', 'gnd': 'gnd'})
    tb.add_source(VoltageSource('vdd', positive='vdd', negative='gnd', dc=1.8))
    tb.add_source(VoltageSource('vin', positive='a', negative='gnd', dc=0.9))
    assert [b.direction for b in tb.net_dict['gnd'].pins] == ['input'] * 4
    assert {b.role_inferred for b in tb.net_dict['gnd'].pins
            if b.kind == 'submodule'} == {True}
    assert list(NoDriverRule().check(tb)) == []


def test_inferred_provenance_survives_round_trip_for_int004(tmp_path):
    """TASK-97: an export/import round trip through
    ``silk.constraints.serialization`` must not change INT-004's
    verdict. Pre-fix, ``serialization.from_json`` set ``pin.role``
    directly with no notion of provenance, so the inferred POWER guess
    on 'vdd_sense' (``role_inferred=True`` before export, per
    ``test_inferred_power_role_does_not_silence_no_driver`` above) came
    back ``role_inferred=False`` after import — turning the guess into
    a declared statement and silently defeating the same guard: INT-004
    reported ``sense`` before the round trip and fell silent on it
    after. Now the provenance flag and the finding both survive."""
    import silk.constraints          # installs Module.add_constraint (idempotent)
    from silk.circuit.port_roles import RoleRuleset, infer_port_roles
    from silk.constraints import serialization

    block = _sense_block(tmp_path)
    infer_port_roles(block, RoleRuleset.default())
    assert block.pin_list['vdd_sense'].role_inferred is True

    before_msgs = [v.message for v in
                   NoDriverRule().check(_two_on_sense(tmp_path, block))]
    assert before_msgs == [_SENSE_UNDRIVEN]     # the guess buys no silence

    payload = serialization.to_json(block)
    fresh = _sense_block(tmp_path)
    serialization.from_json(fresh, payload)
    assert fresh.pin_list['vdd_sense'].role_inferred is True

    after_msgs = [v.message for v in
                  NoDriverRule().check(_two_on_sense(tmp_path, fresh))]
    assert after_msgs == before_msgs


def test_pass_through_inside_a_hierarchy_is_silent(tmp_path):
    """TASK-46 scope addition (the consequence TASK-42 pinned, now fixed).
    Inside a block whose own ports fan out to two children, the module's
    own port pin was not a binding, so INT-004 reported the signal net
    ``in`` (measured pre-fix: ``["Net 'in' has no driver: u1.a(input),
    u2.a(input)"]`` on the declared-supply block) and, once TASK-42 revoked
    the inferred-POWER skip, ``['in', 'vdd']`` on the inferred one — every
    pass-through net inside every hierarchy, none of them a wiring error:
    the parent drives them through the port. The module's own IN / POWER /
    GROUND port now counts as the other party on its net, so both blocks
    are silent."""
    from silk.circuit.port_roles import RoleRuleset, infer_port_roles

    def block(leaf, name):
        amp = _sub(tmp_path, name, {'in': PortDir.IN, 'o1': PortDir.OUT,
                                    'o2': PortDir.OUT, 'vdd': PortDir.IN,
                                    'gnd': PortDir.IN})
        amp.add_module(leaf, 'u1', nets={'a': 'in', 'y': 'o1', 'vdd': 'vdd', 'gnd': 'gnd'})
        amp.add_module(leaf, 'u2', nets={'a': 'in', 'y': 'o2', 'vdd': 'vdd', 'gnd': 'gnd'})
        return amp

    declared = _sub(tmp_path, 'dleaf', {'a': PortDir.IN, 'y': PortDir.OUT,
                                        'vdd': PortDir.POWER, 'gnd': PortDir.GROUND})
    assert list(NoDriverRule().check(block(declared, 'damp'))) == []

    inferred = _sub(tmp_path, 'ileaf', {'a': PortDir.IN, 'y': PortDir.OUT,
                                        'vdd': PortDir.IN, 'gnd': PortDir.IN})
    infer_port_roles(inferred, RoleRuleset.default())
    assert list(NoDriverRule().check(block(inferred, 'iamp'))) == []


def test_own_out_port_fanned_to_children_is_still_reported(tmp_path):
    """Fire-on-bad half of the own-port pair (TASK-46). The block's OUT
    port ``o`` is its promise to drive; wiring it only to two children's
    inputs leaves the output floating. An OUT port is deliberately NOT the
    other party, so the report is byte-identical to pre-fix."""
    leaf = _sub(tmp_path, 'leaf', {'a': PortDir.IN, 'y': PortDir.OUT})
    blk = _sub(tmp_path, 'blk', {'o': PortDir.OUT, 'x': PortDir.IN})
    blk.add_module(leaf, 'u1', nets={'a': 'o', 'y': 'y1'})
    blk.add_module(leaf, 'u2', nets={'a': 'o', 'y': 'y2'})
    assert [v.message for v in NoDriverRule().check(blk)] == [
        "Net 'o' has no driver: u1.a(input), u2.a(input)"]


def test_own_inout_port_is_the_other_party(tmp_path):
    """An INOUT port may be driven by the parent, so it is the other party
    like an IN port (TASK-46; measured pre-fix: ``Net 'b' has no driver:
    u1.a(input), u2.a(input)``). A port declared with no direction at all
    defaults to INOUT (define_module_ports), so the bare-port idiom is
    covered by the same cell."""
    leaf = _sub(tmp_path, 'leaf', {'a': PortDir.IN, 'y': PortDir.OUT})
    blk = _sub(tmp_path, 'blk', {'b': PortDir.INOUT})
    blk.add_module(leaf, 'u1', nets={'a': 'b', 'y': 'y1'})
    blk.add_module(leaf, 'u2', nets={'a': 'b', 'y': 'y2'})
    assert list(NoDriverRule().check(blk)) == []


def test_own_port_binding_never_drives_for_int001(tmp_path):
    """The synthetic own-port binding keeps the DECLARED direction, so the
    block's IN port is never an OUT-direction driver for INT-001: a child
    output wired onto the block's own input net is not reported as a
    driver conflict (unchanged from pre-fix — that shape is not this
    task's) and the pass-through stays silent for INT-004."""
    leaf = _sub(tmp_path, 'leaf', {'a': PortDir.IN, 'y': PortDir.OUT})
    blk = _sub(tmp_path, 'blk', {'in': PortDir.IN})
    blk.add_module(leaf, 'u1', nets={'a': 'in', 'y': 'in'})
    assert list(DriverConflictRule().check(blk)) == []
    assert list(NoDriverRule().check(blk)) == []


# ---------------------------------------------------------------------------
# The block's own port carries its role: INT-002 sees it, INT-003 / INT-005
# do not treat it as a peer of the children (TASK-104)
# ---------------------------------------------------------------------------

def _supply_leaves(tmp_path):
    """Two leaves: one exposes a GROUND port, one a POWER port."""
    gleaf = _sub(tmp_path, 'gleaf', {'gnd': PortDir.GROUND, 'y': PortDir.OUT})
    pleaf = _sub(tmp_path, 'pleaf', {'vdd': PortDir.POWER, 'y': PortDir.OUT})
    return gleaf, pleaf


def test_child_ground_port_on_own_power_port_is_reported(tmp_path):
    """AC #1, first half. A child's GROUND port wired onto the net that
    carries the block's own POWER port is a supply short expressed through
    the interface. Pre-fix (TASK-46 shipped the own-port binding with
    ``role=None``, pinned then by
    ``test_own_port_binding_carries_no_role_so_role_rules_are_unchanged``):
    INT-002 returned ``[]`` — the only POWER-role party on the net was the
    block's own port and it carried no role, so the rule saw one GROUND
    binding and nothing to short it against. Now the own port is named as
    the POWER party. INT-004 stays silent on the same net (the own port is
    the other party, TASK-46) — the role changes nothing there."""
    gleaf, _ = _supply_leaves(tmp_path)
    blk = _sub(tmp_path, 'blk', {'vdd': PortDir.POWER, 'vss': PortDir.GROUND})
    blk.add_module(gleaf, 'u1', nets={'gnd': 'vdd', 'y': 'y1'})
    assert [v.message for v in PowerGroundShortRule().check(blk)] == [
        "Net 'vdd' connects POWER and GROUND ports: u1.gnd(ground), blk.vdd(power)"]
    assert list(NoDriverRule().check(blk)) == []


def test_child_power_port_on_own_ground_port_is_reported(tmp_path):
    """AC #1, second half — the mirror shape. A child's POWER port on the
    block's own GROUND net; pre-fix ``[]`` for the same reason (the own
    GROUND port carried no role). The block's port is declared by bare
    ``PortDir.GROUND``, so the role it carries is TASK-45's direction-implied
    one — that counts as declared."""
    _, pleaf = _supply_leaves(tmp_path)
    blk = _sub(tmp_path, 'blk', {'vdd': PortDir.POWER, 'vss': PortDir.GROUND})
    blk.add_module(pleaf, 'u1', nets={'vdd': 'vss', 'y': 'y1'})
    assert [v.message for v in PowerGroundShortRule().check(blk)] == [
        "Net 'vss' connects POWER and GROUND ports: u1.vdd(power), blk.vss(ground)"]
    assert blk.pin_list['vss'].role_inferred is False


def test_own_supply_ports_wired_correctly_stay_silent(tmp_path):
    """Silent-on-good half of the INT-002 pair: the same leaves wired the
    right way round (child GROUND on own GROUND, child POWER on own POWER)
    produce nothing — a POWER own port beside POWER children is not a
    short, whatever the rule now sees on the own port."""
    gleaf, pleaf = _supply_leaves(tmp_path)
    blk = _sub(tmp_path, 'blk', {'vdd': PortDir.POWER, 'vss': PortDir.GROUND})
    blk.add_module(gleaf, 'u1', nets={'gnd': 'vss', 'y': 'y1'})
    blk.add_module(pleaf, 'u2', nets={'vdd': 'vdd', 'y': 'y2'})
    assert list(PowerGroundShortRule().check(blk)) == []
    assert list(NoDriverRule().check(blk)) == []


def test_own_port_binding_carries_role_and_provenance(tmp_path):
    """The synthetic own-port binding now carries the role exactly as the
    pin holds it (pre-fix: ``role=None`` always). Three provenances: an
    explicit ``PortRole``; TASK-45's direction-implied supply role
    (``role_inferred=False`` — a statement); a role ``infer_port_roles``
    filled from a name pattern (``role_inferred=True`` — a guess, TASK-42),
    so INT-002 reports through an inferred own-port role exactly as it
    already did through an inferred child role, and the provenance is there
    for any consumer that wants to weigh it."""
    from silk.lint.rules.interface import _own_port_binding
    from silk.circuit.port_roles import RoleRuleset, infer_port_roles
    gleaf, _ = _supply_leaves(tmp_path)

    explicit = _sub(tmp_path, 'e', {'c': (PortDir.IN, PortRole.CLOCK)})
    b = _own_port_binding(explicit, 'c')
    assert (b.kind, b.direction, b.role, b.role_inferred) == ('port', 'input', 'clock', False)

    implied = _sub(tmp_path, 'i', {'vdd': PortDir.POWER})
    b = _own_port_binding(implied, 'vdd')
    assert (b.kind, b.direction, b.role, b.role_inferred) == ('port', 'power', 'power', False)

    guessed = _sub(tmp_path, 'g', {'vdd': PortDir.IN, 'gnd': PortDir.IN})
    infer_port_roles(guessed, RoleRuleset.default())
    b = _own_port_binding(guessed, 'vdd')
    assert (b.kind, b.direction, b.role, b.role_inferred) == ('port', 'input', 'power', True)
    guessed.add_module(gleaf, 'u1', nets={'gnd': 'vdd', 'y': 'y1'})
    assert [v.message for v in PowerGroundShortRule().check(guessed)] == [
        "Net 'vdd' connects POWER and GROUND ports: u1.gnd(ground), g.vdd(power)"]

    plain = _sub(tmp_path, 'p', {'x': PortDir.IN})
    assert _own_port_binding(plain, 'x').role is None


def _analog_leaf(tmp_path):
    """The stdcell idiom: every pin ANALOG (silk/examples/stdcells.py)."""
    return _sub(tmp_path, 'aleaf', {'a': (PortDir.IN, PortRole.ANALOG),
                                    'y': (PortDir.OUT, PortRole.ANALOG)})


def test_own_digital_and_clock_ports_are_not_peers_for_int003_int005(tmp_path):
    """AC #2 pin. The shipped ``sar_adc_spice`` shape TASK-46 measured: a
    block whose own DIGITAL port (``comp``) and CLOCK port (``clk``) fan
    into ANALOG-declared stdcell pins. With the role carried verbatim and
    no peer filter, INT-003 reported ``comp`` and INT-005 ``clk`` — seven
    such findings on ``sar_fsm_8b`` / ``sar_logic_8b``. The block's own
    port is not a peer of its children: both rules stay silent, and INT-004
    stays silent too (TASK-46's own-port presence rule)."""
    aleaf = _analog_leaf(tmp_path)
    fsm = _sub(tmp_path, 'fsm', {'comp': (PortDir.IN, PortRole.DIGITAL),
                                 'clk': (PortDir.IN, PortRole.CLOCK)})
    fsm.add_module(aleaf, 'u1', nets={'a': 'comp', 'y': 'y1'})
    fsm.add_module(aleaf, 'u2', nets={'a': 'comp', 'y': 'y2'})
    fsm.add_module(aleaf, 'u3', nets={'a': 'clk', 'y': 'y3'})
    fsm.add_module(aleaf, 'u4', nets={'a': 'clk', 'y': 'y4'})
    assert list(RoleMismatchRule().check(fsm)) == []
    assert list(ClockOnAnalogRule().check(fsm)) == []
    assert list(NoDriverRule().check(fsm)) == []


def test_int003_int005_still_compare_children_beside_an_own_port(tmp_path):
    """Fire-on-bad half for the peer filter: dropping the own port from
    INT-003 / INT-005 must not drop the children. A DIGITAL child and an
    ANALOG child on the block's own (DIGITAL) net are still a mismatch,
    and the report names only the two children — the own port is neither
    a party nor a location (byte-identical to the pre-fix message)."""
    aleaf = _analog_leaf(tmp_path)
    dleaf = _sub(tmp_path, 'dleaf', {'a': (PortDir.IN, PortRole.DIGITAL),
                                     'y': (PortDir.OUT, PortRole.DIGITAL)})
    blk = _sub(tmp_path, 'blk', {'comp': (PortDir.IN, PortRole.DIGITAL),
                                 'clk': (PortDir.IN, PortRole.CLOCK)})
    blk.add_module(aleaf, 'u1', nets={'a': 'comp', 'y': 'y1'})
    blk.add_module(dleaf, 'u2', nets={'a': 'comp', 'y': 'y2'})
    cleaf = _sub(tmp_path, 'cleaf', {'a': (PortDir.IN, PortRole.CLOCK),
                                     'y': (PortDir.OUT, PortRole.CLOCK)})
    blk.add_module(aleaf, 'u3', nets={'a': 'clk', 'y': 'y3'})
    blk.add_module(cleaf, 'u4', nets={'a': 'clk', 'y': 'y4'})
    v3 = list(RoleMismatchRule().check(blk))
    assert [v.message for v in v3] == [
        "Net 'comp' mixes DIGITAL and ANALOG ports: u1.a(analog), u2.a(digital)"]
    assert v3[0].location.port_names == ['u1.a', 'u2.a']
    v5 = list(ClockOnAnalogRule().check(blk))
    assert [v.message for v in v5] == [
        "Net 'clk' connects CLOCK and ANALOG ports: u3.a(analog), u4.a(clock)"]
    assert v5[0].location.port_names == ['u3.a', 'u4.a']


def test_own_port_role_does_not_change_int004(tmp_path):
    """INT-004 is unchanged by the role on the own-port binding. Three
    shapes whose supply skip could in principle flip now that the own port
    is an annotated binding: a declared-POWER own port fanned to supply
    children plus a signal tie-off (the shipped ``pfd`` VDD net: ff.D on
    VDD); an inferred-POWER own port fanned to inferred-POWER children (the
    skip does not apply — the presence rule silences it, as before); and a
    DIGITAL own port beside role-less children. All ``[]`` pre-fix and
    post-fix; the own OUT port stays reported (TASK-46's fire-on-bad)."""
    from silk.circuit.port_roles import RoleRuleset, infer_port_roles
    leaf = _sub(tmp_path, 'leaf', {'d': PortDir.IN, 'vdd': PortDir.POWER,
                                   'q': PortDir.OUT})
    tie = _sub(tmp_path, 'tie', {'vdd': PortDir.POWER})
    tie.add_module(leaf, 'ff', nets={'d': 'vdd', 'vdd': 'vdd', 'q': 'q'})
    assert [b.role for b in tie.net_dict['vdd'].pins] == [None, 'power']
    assert list(NoDriverRule().check(tie)) == []

    ileaf = _sub(tmp_path, 'ileaf', {'vdd': PortDir.IN, 'q': PortDir.OUT})
    infer_port_roles(ileaf, RoleRuleset.default())
    iblk = _sub(tmp_path, 'iblk', {'vdd': PortDir.IN})
    infer_port_roles(iblk, RoleRuleset.default())
    iblk.add_module(ileaf, 'u1', nets={'vdd': 'vdd', 'q': 'q1'})
    iblk.add_module(ileaf, 'u2', nets={'vdd': 'vdd', 'q': 'q2'})
    assert iblk.pin_list['vdd'].role_inferred is True
    assert list(NoDriverRule().check(iblk)) == []

    plain = _sub(tmp_path, 'plain', {'a': PortDir.IN, 'y': PortDir.OUT})
    dblk = _sub(tmp_path, 'dblk', {'x': (PortDir.IN, PortRole.DIGITAL),
                                   'o': (PortDir.OUT, PortRole.DIGITAL)})
    dblk.add_module(plain, 'u1', nets={'a': 'x', 'y': 'y1'})
    dblk.add_module(plain, 'u2', nets={'a': 'x', 'y': 'y2'})
    dblk.add_module(plain, 'u3', nets={'a': 'o', 'y': 'y3'})
    dblk.add_module(plain, 'u4', nets={'a': 'o', 'y': 'y4'})
    assert [v.message for v in NoDriverRule().check(dblk)] == [
        "Net 'o' has no driver: u3.a(input), u4.a(input)"]
