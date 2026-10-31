"""TASK-38 / TASK-94 — STRUCT-009 / STRUCT-010 through the registry entry point.

``tests/lint/rules/test_structural_rules.py`` and
``tests/lint/test_rule_audit_real_circuits.py`` exercise both rules by
calling ``rule.check()`` directly. These cells go through
``RuleRegistry.default(...)`` on real sky130 ``Module`` fixtures — the path
``generate_report`` / ``interface_check`` take — so a promotion is a
one-entry flip in :data:`_PROMOTED` with every fire/silent assertion
already in place.

History of the flips (2026-09-04 promotion gate, TASK-38): a new default
rule must stay quiet on legitimate designs. Running the default catalog
plus both rules over every module built by every shipped
``silk/examples/*.py`` (SPICE stubbed, 68 examples, ~500 modules) found
these offenders, none of which is a wiring mistake:

* STRUCT-009 on supply-less-by-design blocks — a PMOS-only current bank
  (no GROUND), a BJT/resistor bandgap branch fed by injected current, a
  diode-connected NMOS bias reference, a resistor/cap reference ladder, a
  cap-DAC switched between references (no POWER), and three flat amps
  that declare no ports at all;
* STRUCT-010 on an LDO whose regulated ``vout`` is legitimately a
  ``PortRole.POWER`` port with direction OUT.

RULED 2026-09-07 (TASK-93): STRUCT-009 STAYS opt-in and was not weakened.
The *defect* it was written for — the hidden local supply in
:func:`test_struct009_fires_on_hidden_local_supply_net` below — moved to
STRUCT-013 HIDDEN_SUPPLY_NET, which keys on the WIRING rather than on the
port declaration and is silent on all nine supply-less blocks; that rule is
in the default catalog and has its own cells in
``tests/lint/test_struct013_hidden_supply.py``. What remains opt-in is
STRUCT-009's stricter demand (declare the rails whether or not you wire
them), which no shipped example meets.

RULED 2026-09-09 (TASK-94): STRUCT-010 is PROMOTED with a regulated-output
exemption. A supply-role port declared OUT is exempt when the block also
declares a port of the SAME role that receives (direction IN, or the bare
``PortDir.POWER`` / ``PortDir.GROUND``): a block that takes a rail and
emits one is a regulator, a converter, a header or a footer switch — the
LDO's shape. A supply-role OUT port in a block with no same-role input has
nothing to derive the rail from and is still reported; INOUT is unchanged.
The exemption is structural (role + direction only, no port-name
inference) and the TASK-38 gate, re-run with the shipped rule, reports
nothing on any module built by any shipped example. The STRUCT-010 cells
below therefore go through the plain ``RuleRegistry.default()``.
"""
import pytest

from silk.circuit.circuit_elements import Port, PortDir, PortRole
from silk.circuit.modules import Module
from silk.lint.registry import RuleRegistry
from silk.lint.rules import DEFAULT_CATALOG

# Per-rule promotion state, pinned by
# ``test_default_catalog_composition_matches_the_rulings``. STRUCT-009 stays
# False on purpose (TASK-93 ruling, module docstring); STRUCT-010 flipped to
# True in TASK-94. Every assertion below is written against the registry,
# not the rule objects, so a future flip of STRUCT-009 changes only this
# dict.
_PROMOTED = {'STRUCT-009': False, 'STRUCT-010': True}


def _registry():
    """The registry that dispatches BOTH rules: the default one plus the
    opt-in catalog while any rule is still unpromoted."""
    return RuleRegistry.default(include_optional=not all(_PROMOTED.values()))


def _lint(module, registry=None):
    """rule_id -> [messages] from the registry under test (both rules)."""
    out = {}
    for v in (registry or _registry()).run(module).violations:
        out.setdefault(v.rule_id, []).append(v.message)
    return out


def _default_lint(module):
    """rule_id -> [messages] from the plain default registry — the path a
    user gets with no flag at all. STRUCT-010 is asserted through this."""
    return _lint(module, RuleRegistry.default())


def _inverter(pdk, tmp_path, name, ports):
    """A real sky130 CMOS inverter subcircuit with the given port block.
    The body always wires the PMOS to ``vdd`` and the NMOS to ``gnd``, so a
    port block that omits one of them leaves a hidden local supply net."""
    m = Module('PDK_skywater130', str(tmp_path), f'{name}.cir',
               pdk_details=pdk, module_name=name)
    m.define_module_ports(ports)
    with m:
        pdk.nmos1p8('mn', d='vout', g='vin', s='gnd', b='gnd', w=1, l=0.15)
        pdk.pmos1p8('mp', d='vout', g='vin', s='vdd', b='vdd', w=2, l=0.15)
    return m


def _regulator(pdk, tmp_path, name, ports):
    """The LDO's shape, as a real sky130 subcircuit: a PMOS pass device from
    the supply the block receives (``vin``) to the rail it produces
    (``vout``), an NMOS gate driver referenced to ``vss``. The port block
    is the parameter so the cells can vary how the rails are declared."""
    m = Module('PDK_skywater130', str(tmp_path), f'{name}.cir',
               pdk_details=pdk, module_name=name)
    m.define_module_ports(ports)
    with m:
        pdk.pmos1p8('mp_pass', d='vout', g='vg', s='vin', b='vin', w=20, l=0.5)
        pdk.nmos1p8('mn_drv', d='vg', g='vref', s='vss', b='vss', w=2, l=0.5)
    return m


_SIGNAL_PORTS = {'vin': Port.analog_input(), 'vout': Port.analog_output()}
_SUPPLY_PORTS = {'vdd': Port.supply(), 'gnd': Port.ground()}

_OUT_NO_INPUT = ("Port '{port}' (role={role}) has direction OUT and the block "
                 "declares no {role}-role input; supply ports should be IN "
                 "unless the block derives the rail from a same-role supply "
                 "it receives")


# ---------------------------------------------------------------------------
# Catalog composition — the promotion state, pinned
# ---------------------------------------------------------------------------

def test_default_catalog_composition_matches_the_rulings():
    """The TASK-38 gate found legitimate shipped designs both rules flag;
    the TASK-93 ruling kept STRUCT-009 opt-in and the TASK-94 ruling
    promoted STRUCT-010 with the regulated-output exemption (module
    docstring). Pin the composition so a silent flip is visible; a future
    promotion of STRUCT-009 flips together with ``_PROMOTED``."""
    ids = {r.rule_id for r in DEFAULT_CATALOG()}
    for rule_id, promoted in _PROMOTED.items():
        assert (rule_id in ids) is promoted, rule_id


def test_registry_under_test_dispatches_both_rules(pdk, tmp_path):
    inv = _inverter(pdk, tmp_path, 'inv', {**_SIGNAL_PORTS, **_SUPPLY_PORTS})
    visited = {e.rule_id for e in _registry().run(inv).visit_log}
    assert {'STRUCT-009', 'STRUCT-010'} <= visited


def test_plain_default_registry_dispatches_struct010_but_not_struct009(pdk, tmp_path):
    inv = _inverter(pdk, tmp_path, 'inv', {**_SIGNAL_PORTS, **_SUPPLY_PORTS})
    visited = {e.rule_id for e in RuleRegistry.default().run(inv).visit_log}
    assert 'STRUCT-010' in visited
    assert 'STRUCT-009' not in visited


# ---------------------------------------------------------------------------
# STRUCT-009 POWER_GROUND_PORT_REQUIRED
# ---------------------------------------------------------------------------

def test_struct009_fires_on_hidden_local_supply_net(pdk, tmp_path):
    """Matrix cell S9's shape: the body wires the PMOS to ``vdd`` but the
    port block has no vdd, so ``.subckt leaky vin vout gnd`` carries an
    internal floating supply (ngspice-42: v(out)=0.0, no error). Before
    this file the registry path reported nothing; the rule fires once,
    naming the module and the missing role.

    This shape is the one both rules agree on, and since TASK-93 the DEFAULT
    registry reports it too — as STRUCT-013, naming the rail and the
    terminals on it rather than the missing role."""
    leaky = _inverter(pdk, tmp_path, 'leaky',
                      {**_SIGNAL_PORTS, 'gnd': Port.ground()})
    assert _lint(leaky)['STRUCT-009'] == [
        "Subcircuit 'leaky' contains devices but declares no port with "
        "role POWER (missing POWER supply port)"]


def test_struct009_fires_twice_on_a_device_block_with_no_supply_ports(pdk, tmp_path):
    """One finding per missing role, both naming the subcircuit."""
    bare = _inverter(pdk, tmp_path, 'bare', dict(_SIGNAL_PORTS))
    assert _lint(bare)['STRUCT-009'] == [
        "Subcircuit 'bare' contains devices but declares no port with "
        "role POWER (missing POWER supply port)",
        "Subcircuit 'bare' contains devices but declares no port with "
        "role GROUND (missing GROUND supply port)"]


@pytest.mark.parametrize('supply_ports', [
    pytest.param({'vdd': Port.supply(), 'gnd': Port.ground()}, id='Port.supply/ground'),
    pytest.param({'vdd': (PortDir.IN, PortRole.POWER), 'gnd': (PortDir.IN, PortRole.GROUND)},
                 id='explicit-role-tuples'),
    pytest.param({'vdd': PortDir.POWER, 'gnd': PortDir.GROUND}, id='bare-PortDir'),
    pytest.param({'vdd': 'power', 'gnd': 'ground'}, id='direction-strings'),
])
def test_struct009_silent_on_every_supply_declaration_idiom(pdk, tmp_path, supply_ports):
    """Every way the examples declare supplies carries the role since
    TASK-45 (the false positive that blocked promotion), so the rule stays
    quiet on a correctly ported inverter."""
    inv = _inverter(pdk, tmp_path, 'inv', {**_SIGNAL_PORTS, **supply_ports})
    assert 'STRUCT-009' not in _lint(inv)


def test_struct009_silent_on_a_device_free_wrapper(pdk, tmp_path):
    """A behavioral wrapper with no devices has nothing to power."""
    wrap = Module('PDK_skywater130', str(tmp_path), 'wrap.cir',
                  pdk_details=pdk, module_name='wrap')
    wrap.define_module_ports(dict(_SIGNAL_PORTS))
    assert 'STRUCT-009' not in _lint(wrap)


def test_struct009_silent_on_a_top_level_module(pdk, tmp_path):
    """A testbench (no ``module_name``) is not a subcircuit; its supplies
    are sources, not ports."""
    top = Module('PDK_skywater130', str(tmp_path), 'top.cir', pdk_details=pdk)
    with top:
        pdk.nmos1p8('mn', d='out', g='in', s='gnd', b='gnd', w=1, l=0.15)
    assert 'STRUCT-009' not in _lint(top)


# ---------------------------------------------------------------------------
# STRUCT-010 PORT_DIRECTION_SANITY — default registry (promoted, TASK-94)
# ---------------------------------------------------------------------------

def test_struct010_warns_on_out_supply_port(pdk, tmp_path):
    """Matrix cell P1's shape: ``Port.supply(direction='out')`` was accepted
    and never questioned on the registry path. The inverter declares a
    GROUND input but no POWER input, so its POWER OUT port has nothing to
    be derived from and the finding says so — the exemption keys on the
    SAME role, so a swapped-direction vdd is not excused by the gnd."""
    m = _inverter(pdk, tmp_path, 'outsup',
                  {**_SIGNAL_PORTS, 'vdd': Port.supply(direction='out'),
                   'gnd': Port.ground()})
    assert _default_lint(m)['STRUCT-010'] == [
        _OUT_NO_INPUT.format(port='vdd', role='POWER')]


def test_struct010_warns_on_inout_ground_port(pdk, tmp_path):
    """INOUT is unchanged by the TASK-94 ruling: still reported so the
    intent is documented (waive it when a rail genuinely sources and
    sinks)."""
    m = _inverter(pdk, tmp_path, 'inoutgnd',
                  {**_SIGNAL_PORTS, 'vdd': Port.supply(),
                   'gnd': Port(dir=PortDir.INOUT, role=PortRole.GROUND)})
    assert _default_lint(m)['STRUCT-010'] == [
        "Port 'gnd' (role=GROUND) has direction INOUT; supply ports should be IN"]


def test_struct010_silent_on_in_and_bare_supply_directions(pdk, tmp_path):
    """``PortDir.POWER`` / ``PortDir.GROUND`` are supply *directions*, not
    OUT/INOUT, and explicit IN is the documented shape — neither warns."""
    inv = _inverter(pdk, tmp_path, 'inv', {**_SIGNAL_PORTS, **_SUPPLY_PORTS})
    assert 'STRUCT-010' not in _default_lint(inv)
    explicit = _inverter(pdk, tmp_path, 'explicit',
                         {**_SIGNAL_PORTS,
                          'vdd': (PortDir.IN, PortRole.POWER),
                          'gnd': (PortDir.IN, PortRole.GROUND)})
    assert 'STRUCT-010' not in _default_lint(explicit)


def test_struct010_ignores_out_ports_without_a_supply_role(pdk, tmp_path):
    """An ordinary analog output is OUT by definition; only supply-role
    ports are checked."""
    inv = _inverter(pdk, tmp_path, 'inv', {**_SIGNAL_PORTS, **_SUPPLY_PORTS})
    assert inv.pin_list['vout'].direction is PortDir.OUT
    assert 'STRUCT-010' not in _default_lint(inv)


# --- the regulated-output exemption (TASK-94) ------------------------------

def test_struct010_silent_on_a_regulated_output(pdk, tmp_path):
    """The TASK-38 offender's shape: a POWER-role OUT ``vout`` in a block
    that also receives a POWER rail on ``vin``. That is a regulator, the
    canonical legitimate case, and the default registry says nothing."""
    ldo = _regulator(pdk, tmp_path, 'ldo', {
        'vin': Port.supply(), 'vref': Port.analog_input(),
        'vout': Port.supply(direction='out'), 'vss': Port.ground()})
    assert ldo.pin_list['vout'].direction is PortDir.OUT
    assert ldo.pin_list['vout'].role is PortRole.POWER
    assert 'STRUCT-010' not in _default_lint(ldo)


@pytest.mark.parametrize('vin', [
    pytest.param(Port.supply(), id='Port.supply'),
    pytest.param((PortDir.IN, PortRole.POWER), id='explicit-role-tuple'),
    pytest.param(PortDir.POWER, id='bare-PortDir.POWER'),
    pytest.param('power', id='direction-string'),
    pytest.param(Port(dir=PortDir.POWER), id='Port(dir=POWER)-role-derived'),
])
def test_struct010_every_receiving_idiom_licenses_the_regulated_output(pdk, tmp_path, vin):
    """The receiving port may be declared any way the examples declare
    supplies: explicit IN, or the bare supply direction, whose role is
    derived at declaration (TASK-45). All of them count as 'the block
    receives a POWER rail'."""
    ldo = _regulator(pdk, tmp_path, 'ldo', {
        'vin': vin, 'vref': Port.analog_input(),
        'vout': Port.supply(direction='out'), 'vss': Port.ground()})
    assert 'STRUCT-010' not in _default_lint(ldo)


def test_struct010_fires_on_a_supply_out_with_no_supply_input_at_all(pdk, tmp_path):
    """The fixture AC#1 asks for: a supply port declared OUT in a block
    with no supply input. Nothing feeds the rail, so it is reported, and
    the message names the missing role."""
    gen = _regulator(pdk, tmp_path, 'gen', {
        'vin': Port.analog_input(), 'vref': Port.analog_input(),
        'vout': Port.supply(direction='out'), 'vss': Port.analog_input()})
    assert _default_lint(gen)['STRUCT-010'] == [
        _OUT_NO_INPUT.format(port='vout', role='POWER')]


def test_struct010_exemption_needs_the_same_role(pdk, tmp_path):
    """A GROUND input does not license a POWER output (and vice versa):
    the swapped-direction inverter of the P1 cell has a gnd input and is
    still reported. Conversely a footer switch — GROUND in, gated GROUND
    out — is exempt, and a local-ground output in a block that only
    receives POWER is not."""
    footer = _regulator(pdk, tmp_path, 'footer', {
        'vin': Port.analog_input(), 'vref': Port.analog_input(),
        'vout': Port.analog_output(),
        'vss': Port.ground(), 'vss_gated': Port.ground(direction='out')})
    assert 'STRUCT-010' not in _default_lint(footer)

    mixed = _regulator(pdk, tmp_path, 'mixed', {
        'vin': Port.supply(), 'vref': Port.analog_input(),
        'vout': Port.analog_output(),
        'vss': Port.analog_input(), 'vss_gated': Port.ground(direction='out')})
    assert _default_lint(mixed)['STRUCT-010'] == [
        _OUT_NO_INPUT.format(port='vss_gated', role='GROUND')]


def test_struct010_one_input_licenses_several_outputs(pdk, tmp_path):
    """A header bank producing two gated rails from one supply: one POWER
    input, two POWER outputs, nothing reported."""
    bank = _regulator(pdk, tmp_path, 'bank', {
        'vin': Port.supply(), 'vref': Port.analog_input(),
        'vout': Port.supply(direction='out'),
        'vg': Port.supply(direction='out'), 'vss': Port.ground()})
    assert 'STRUCT-010' not in _default_lint(bank)


def test_struct010_an_inout_supply_does_not_count_as_receiving(pdk, tmp_path):
    """INOUT is itself reported and does not license an OUT of the same
    role — a bidirectional rail is not a declared input. Two findings, in
    port order."""
    cp = _regulator(pdk, tmp_path, 'cp', {
        'vin': Port.supply(direction='inout'), 'vref': Port.analog_input(),
        'vout': Port.supply(direction='out'), 'vss': Port.ground()})
    assert _default_lint(cp)['STRUCT-010'] == [
        "Port 'vin' (role=POWER) has direction INOUT; supply ports should be IN",
        _OUT_NO_INPUT.format(port='vout', role='POWER')]
