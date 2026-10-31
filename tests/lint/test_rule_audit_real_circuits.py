"""TASK-32 lint audit — every shipped rule against REAL circuit fixtures.

Each rule gets a fire-on-bad + silent-on-good pair built from the real
sky130 catalog (``BasePDK`` with a fake library path — no ngspice
is run) and real ``Module`` composition, never ``MagicMock`` stand-ins.
The message assert on every fire case checks that the violation names
the offending instance/net/port, because a finding a designer cannot
locate is not actionable.

Confirmed defects found by the audit are pinned as ``xfail(strict=True)``
tests carrying their follow-up task ID: they fail loudly (as XPASS) the
moment the follow-up lands, so the follow-up expectations stay executable.
"""
import types

import numpy as np
import pytest

from silk import sources
from silk.circuit.circuit_elements import Port, PortDir, PortRole
from silk.circuit.modules import Module
from silk.circuit.voltage_domain import VoltageDomain
from silk.elements import passives
from silk.elements import elements
from silk.lint.registry import RuleRegistry
from silk.lint.rules import DEFAULT_CATALOG
from silk.lint.rules.domain import (
    DomainCrossingRule, RailMismatchRule, SupplyDisabledStressRule,
)
from silk.lint.rules.interface import (
    ClockOnAnalogRule, DriverConflictRule, NoDriverRule,
    PowerGroundShortRule, RoleMismatchRule,
)
from silk.lint.rules.stress import JunctionOverrangeRule, _is_ground_node
from silk.lint.rules.structural import (
    BodyNotTiedRule, BusPartialWiringRule, DanglingPortRule, FloatingGateRule,
    FloatingNetRule, MissingParamRule, PortDirectionSanity,
    PowerGroundPortRequired, PowerPortUnconnectedRule, SignalOnSupplyNetRule,
    UnusedInstanceRule,
)
from silk.lint.severity import Severity
from silk.lint.violations import JunctionViolation

# Ratings come straight from the sky130 catalog (silk/pdk/data/…yaml):
# nmos1p8 Vgs_max = 1.95 V — the SkyWater-documented "operating voltage
# where SPICE models are valid" (TASK-48; it was an unsourced 1.98 = 1.1 x
# 1.8 before). The fixtures below drive 2.5 V (bad) / 1.5 V (good) so the
# margin either side of the rating is unambiguous.
_NMOS1P8_VGS_MAX_V = 1.95
_VGS_BAD_V = 2.5
_VGS_GOOD_V = 1.5


# ---------------------------------------------------------------------------
# Real-circuit builders
# ---------------------------------------------------------------------------

def _mod(pdk, tmp_path, fname, name=None):
    return Module('PDK_skywater130', str(tmp_path), fname,
                  pdk_details=pdk, module_name=name)


def _unbind(device, *pins):
    """Strip terminal bindings off an already-added factory device, in place.

    TASK-41 made the factory constructor and ``AnalogElement.__setattr__``
    reject ``''``/``None`` as a pin value, so no supported API produces an
    unbound factory-device terminal any more (it emitted a token-short line
    that ngspice mis-parsed, or died inside ``' '.join``). STRUCT-004/005/008
    keep their real inputs — the legacy ``add_element`` path leaves
    ``pin.net is None`` for every pin the caller omits, and submodule ports
    behave the same — so these cells reproduce the unbound state directly on
    the device's own dict, and clean up ``Net.pins`` so the module stays
    self-consistent.
    """
    module = getattr(device, '_module', None)
    for pin in pins:
        # Key kept, value cleared — that is what "unbound" looks like on the
        # legacy path too (``pin_list['g'].net is None``), and STR-001 keys
        # its MOSFET/diode/BJT shape detection off the pin-name SET, so
        # dropping the key would hide the device from the rule entirely.
        net_name = device.nets.get(pin)
        device.nets[pin] = None
        if module is None or net_name is None:
            continue
        module._element_mgr._pin_to_net.pop((device.name, pin), None)
        net = module.net_dict.get(net_name)
        if net is not None:
            net.pins = [b for b in net.pins
                        if not (b.instance == device.name and b.port == pin)]


_BODY_TIED_TO_GROUND = object()   # distinct from None, which means *unbound*


def _inverter(pdk, tmp_path, name='inv', *, gate='in',
              body_n=_BODY_TIED_TO_GROUND, supply_ports=None, ground='gnd'):
    """A real sky130 CMOS inverter subcircuit.

    ``supply_ports`` defaults to role-annotated POWER/GROUND ports; pass
    a dict to override the port declaration (used by the STRUCT-009/010
    cases).

    ``ground`` names the ground port *and* the net the nmos source/body sit
    on. It stays 'gnd' for the cells that only exercise one rule's
    ``check()``, and is set to 'vss' by the clean-baseline case: since
    TASK-36 a declared port named 'gnd' is itself a finding (STRUCT-012),
    so an inverter with one is no longer a clean circuit.

    A falsy ``gate``/``body_n`` means *that terminal is unbound*: the device
    is built wired and the binding is then stripped with :func:`_unbind`,
    because since TASK-41 the constructor refuses a falsy net outright.
    The default ``body_n`` ties the body to ``ground``; ``body_n=None``
    (or any other falsy value) leaves it unbound.
    """
    m = _mod(pdk, tmp_path, f'{name}.cir', name)
    ports = {
        'in': (PortDir.IN, PortRole.DIGITAL),
        'out': (PortDir.OUT, PortRole.DIGITAL),
    }
    ports.update(supply_ports if supply_ports is not None else {
        'vdd': (PortDir.POWER, PortRole.POWER),
        ground: (PortDir.GROUND, PortRole.GROUND),
    })
    m.define_module_ports(ports)
    with m:
        mn = pdk.nmos1p8('mn', d='out', g=gate or 'in', s=ground,
                         b=(ground if body_n is _BODY_TIED_TO_GROUND
                            else (body_n or ground)),
                         w=1.0, l=0.15)
        mp = pdk.pmos1p8('mp', d='out', g=gate or 'in', s='vdd', b='vdd',
                         w=2.0, l=0.15)
    if not gate:
        _unbind(mn, 'g')
        _unbind(mp, 'g')
    if body_n is not _BODY_TIED_TO_GROUND and not body_n:
        _unbind(mn, 'b')
    return m


def _block(pdk, tmp_path, name, ports):
    """A real subcircuit with the given port declaration and one nmos1p8
    hanging off the first two declared ports (so the block has devices)."""
    m = _mod(pdk, tmp_path, f'{name}.cir', name)
    m.define_module_ports(ports)
    pnames = list(ports)
    with m:
        pdk.nmos1p8('mn', d=pnames[0], g=pnames[1 % len(pnames)],
                    s='gnd_int', b='gnd_int', w=1.0, l=0.15)
    return m


def _rule_violations(rule, scope):
    return list(rule.check(scope))


def _only(rule_id, violations):
    return [v for v in violations if v.rule_id == rule_id]


# ---------------------------------------------------------------------------
# Catalog sanity — one place that pins what "every shipped rule" means
# ---------------------------------------------------------------------------

def test_default_catalog_is_22_unique_rules_with_fix_guidance():
    rules = DEFAULT_CATALOG()
    ids = [r.rule_id for r in rules]
    assert len(ids) == 22 and len(set(ids)) == 22
    # STRUCT-009 is shipped as OPT-IN (OPTIONAL_CATALOG, TASK-49 on ledger
    # F-06): the TASK-38 gate found legitimate shipped blocks it flags, and
    # the TASK-93 ruling kept it opt-in rather than weakening it.
    # STRUCT-013 HIDDEN_SUPPLY_NET carries the defect STRUCT-009 was written
    # for into the default catalog. STRUCT-010 was promoted in TASK-94 with
    # a regulated-output exemption. Pin all three so a silent change to the
    # catalog composition is visible.
    assert 'STRUCT-009' not in ids
    assert 'STRUCT-010' in ids
    assert 'STRUCT-013' in ids
    for r in rules:
        # Every rationale must tell the designer what to do about it.
        assert any(k in r.rationale.lower()
                   for k in ('fix', 'waive', 'add', 'connect', 'delete',
                             'mark', 'use', 'wire', 'suppress')), r.rule_id


def test_clean_real_inverter_is_silent_on_every_shipped_rule(pdk, tmp_path):
    """The silent-on-good baseline: a correctly wired sky130 inverter
    inside a correctly wired top-level testbench.

    Everything must be silent. STRUCT-006 (F-01, W/L case mismatch) used
    to be subtracted here and is not any more — TASK-44 made the rule read
    the device's own parameter names — so this test now guards every
    shipped rule against false positives on a clean circuit.

    The ground port is 'vss', not 'gnd': since TASK-36 a port named 'gnd'
    is itself a finding (STRUCT-012 — ngspice folds the port name onto
    global node 0 and silently discards the parent's wiring), so an
    inverter declaring one is not a clean circuit any more. The other
    cells in this file keep 'gnd' because they call one rule's ``check()``
    directly rather than the whole catalog.
    """
    inv = _inverter(pdk, tmp_path, ground='vss')
    top = _mod(pdk, tmp_path, 'top.cir')
    top.add_module(inv, 'u1', nets={'in': 'vin', 'out': 'vout',
                                     'vdd': 'vdd', 'vss': '0'})
    with top:
        sources.VoltageSource('vdd', positive='vdd', negative='0', dc=1.8)
        sources.VoltageSource('vin', positive='vin', negative='0', dc=0.9)
        elements.capacitor('cl', p='vout', n='0', value=1e-12)
    for scope in (inv, top):
        for rule in DEFAULT_CATALOG():
            hits = list(rule.check(scope))
            assert hits == [], f"{rule.rule_id} fired on a clean inverter: {hits}"


# ---------------------------------------------------------------------------
# INT-001..005 — interface rules
# ---------------------------------------------------------------------------

def test_int001_fires_on_two_real_outputs_shorted(pdk, tmp_path):
    inv = _inverter(pdk, tmp_path)
    top = _mod(pdk, tmp_path, 'top.cir')
    top.add_module(inv, 'u1', nets={'in': 'a', 'out': 'shared', 'vdd': 'vdd', 'gnd': '0'})
    top.add_module(inv, 'u2', nets={'in': 'b', 'out': 'shared', 'vdd': 'vdd', 'gnd': '0'})
    v = _only('INT-001', _rule_violations(DriverConflictRule(), top))
    assert len(v) == 1
    assert v[0].location.net_name == 'shared'
    assert 'u1.out' in v[0].message and 'u2.out' in v[0].message


def test_int001_silent_on_one_driver_many_sinks(pdk, tmp_path):
    inv = _inverter(pdk, tmp_path)
    top = _mod(pdk, tmp_path, 'top.cir')
    top.add_module(inv, 'u1', nets={'in': 'a', 'out': 'mid', 'vdd': 'vdd', 'gnd': '0'})
    top.add_module(inv, 'u2', nets={'in': 'mid', 'out': 'o2', 'vdd': 'vdd', 'gnd': '0'})
    top.add_module(inv, 'u3', nets={'in': 'mid', 'out': 'o3', 'vdd': 'vdd', 'gnd': '0'})
    assert _rule_violations(DriverConflictRule(), top) == []


def test_int002_fires_when_power_and_ground_ports_share_a_net(pdk, tmp_path):
    inv = _inverter(pdk, tmp_path)
    top = _mod(pdk, tmp_path, 'top.cir')
    top.add_module(inv, 'u1', nets={'in': 'a', 'out': 'y', 'vdd': 'rail', 'gnd': 'rail'})
    v = _only('INT-002', _rule_violations(PowerGroundShortRule(), top))
    assert len(v) == 1
    assert v[0].location.net_name == 'rail'
    assert 'u1.vdd(power)' in v[0].message and 'u1.gnd(ground)' in v[0].message


def test_int002_fires_on_bare_portdir_supply_ports_sharing_a_net(pdk, tmp_path):
    """TASK-45: INT-002 sees ports declared with bare PortDir.POWER/GROUND."""
    inv = _inverter(pdk, tmp_path, supply_ports={'vdd': PortDir.POWER, 'gnd': PortDir.GROUND})
    top = _mod(pdk, tmp_path, 'top.cir')
    top.add_module(inv, 'u1', nets={'in': 'a', 'out': 'y', 'vdd': 'rail', 'gnd': 'rail'})
    v = _only('INT-002', _rule_violations(PowerGroundShortRule(), top))
    assert [x.location.net_name for x in v] == ['rail']
    assert 'u1.vdd(power)' in v[0].message and 'u1.gnd(ground)' in v[0].message


def test_int002_silent_on_separate_rails(pdk, tmp_path):
    inv = _inverter(pdk, tmp_path)
    top = _mod(pdk, tmp_path, 'top.cir')
    top.add_module(inv, 'u1', nets={'in': 'a', 'out': 'y', 'vdd': 'vdd', 'gnd': '0'})
    top.add_module(inv, 'u2', nets={'in': 'y', 'out': 'z', 'vdd': 'vdd', 'gnd': '0'})
    assert _rule_violations(PowerGroundShortRule(), top) == []


def test_int003_fires_on_digital_output_into_analog_input(pdk, tmp_path):
    inv = _inverter(pdk, tmp_path)                      # digital out
    amp = _block(pdk, tmp_path, 'amp', {'out': (PortDir.OUT, PortRole.ANALOG),
                                        'in': (PortDir.IN, PortRole.ANALOG)})
    top = _mod(pdk, tmp_path, 'top.cir')
    top.add_module(inv, 'u1', nets={'in': 'a', 'out': 'x', 'vdd': 'vdd', 'gnd': '0'})
    top.add_module(amp, 'a1', nets={'in': 'x', 'out': 'y'})
    v = _only('INT-003', _rule_violations(RoleMismatchRule(), top))
    assert len(v) == 1
    assert 'u1.out(digital)' in v[0].message and 'a1.in(analog)' in v[0].message


def test_int003_silent_on_analog_to_analog(pdk, tmp_path):
    amp = _block(pdk, tmp_path, 'amp', {'out': (PortDir.OUT, PortRole.ANALOG),
                                        'in': (PortDir.IN, PortRole.ANALOG)})
    top = _mod(pdk, tmp_path, 'top.cir')
    top.add_module(amp, 'a1', nets={'in': 'x', 'out': 'y'})
    top.add_module(amp, 'a2', nets={'in': 'y', 'out': 'z'})
    assert _rule_violations(RoleMismatchRule(), top) == []


def test_int004_fires_on_two_inputs_with_no_driver(pdk, tmp_path):
    inv = _inverter(pdk, tmp_path)
    top = _mod(pdk, tmp_path, 'top.cir')
    top.add_module(inv, 'u1', nets={'in': 'orphan', 'out': 'o1', 'vdd': 'vdd', 'gnd': '0'})
    top.add_module(inv, 'u2', nets={'in': 'orphan', 'out': 'o2', 'vdd': 'vdd', 'gnd': '0'})
    v = [x for x in _rule_violations(NoDriverRule(), top)
         if x.location.net_name == 'orphan']
    assert len(v) == 1
    assert 'u1.in(input)' in v[0].message and 'u2.in(input)' in v[0].message


def test_int004_silent_when_a_submodule_output_drives(pdk, tmp_path):
    inv = _inverter(pdk, tmp_path)
    top = _mod(pdk, tmp_path, 'top.cir')
    top.add_module(inv, 'u0', nets={'in': 'a', 'out': 'drv', 'vdd': 'vdd', 'gnd': '0'})
    top.add_module(inv, 'u1', nets={'in': 'drv', 'out': 'o1', 'vdd': 'vdd', 'gnd': '0'})
    top.add_module(inv, 'u2', nets={'in': 'drv', 'out': 'o2', 'vdd': 'vdd', 'gnd': '0'})
    assert [x for x in _rule_violations(NoDriverRule(), top)
            if x.location.net_name == 'drv'] == []


def test_int004_silent_when_a_top_level_device_drives(pdk, tmp_path):
    """F-03 / TASK-46 (was xfail). The interface rules kept only
    'submodule'/'source' bindings, so a net driven by a same-level MOSFET
    drain (kind='device', direction=None) had no visible driver — measured
    pre-fix: ``Net 'x' has no driver: u1.in(input), u2.in(input)``. A
    device pin is now the other party on its net (presence, not a driver:
    a drain is bidirectional and pins carry no direction) and the net is
    silent. The fire-on-bad half is
    :func:`test_int004_fires_on_two_inputs_with_no_driver` — the same two
    inputs with nothing else on the net still report."""
    inv = _inverter(pdk, tmp_path)
    top = _mod(pdk, tmp_path, 'top.cir')
    top.add_module(inv, 'u1', nets={'in': 'x', 'out': 'o1', 'vdd': 'vdd', 'gnd': '0'})
    top.add_module(inv, 'u2', nets={'in': 'x', 'out': 'o2', 'vdd': 'vdd', 'gnd': '0'})
    with top:
        pdk.nmos1p8('mdrv', d='x', g='vin', s='0', b='0', w=2.0, l=0.15)
    assert [x for x in _rule_violations(NoDriverRule(), top)
            if x.location.net_name == 'x'] == []


def test_int004_silent_when_a_passive_divider_tap_drives(pdk, tmp_path):
    """TASK-46, the description's second shape: a resistor-divider tap
    (``passives.Resistor`` ends bind as kind='passive', direction=None)
    read by two submodule inputs — measured pre-fix: ``Net 'tap' has no
    driver: u1.in(input), u2.in(input)``. A passive end is the other party
    exactly like a device pin: nothing in the binding says which end is
    the source, so presence is all the rule can use."""
    inv = _inverter(pdk, tmp_path)
    top = _mod(pdk, tmp_path, 'top.cir')
    top.add_module(inv, 'u1', nets={'in': 'tap', 'out': 'o1', 'vdd': 'vdd', 'gnd': '0'})
    top.add_module(inv, 'u2', nets={'in': 'tap', 'out': 'o2', 'vdd': 'vdd', 'gnd': '0'})
    with top:
        elements.resistor('r1', p='vdd', n='tap', value=10e3)
        elements.resistor('r2', p='tap', n='0', value=10e3)
    assert [x for x in _rule_violations(NoDriverRule(), top)
            if x.location.net_name == 'tap'] == []


def test_int004_silent_on_a_single_input_plus_a_device(pdk, tmp_path):
    """A net with ONE submodule input and a device pin passes the helper's
    two-binding filter now that device pins count (pre-fix it was filtered
    out — silent by invisibility, not by judgment). It is still silent:
    the device is the other party. Pinned so the newly visible shape does
    not start reporting."""
    inv = _inverter(pdk, tmp_path)
    top = _mod(pdk, tmp_path, 'top.cir')
    top.add_module(inv, 'u1', nets={'in': 'x', 'out': 'o1', 'vdd': 'vdd', 'gnd': '0'})
    with top:
        pdk.nmos1p8('mdrv', d='x', g='vin', s='0', b='0', w=2.0, l=0.15)
    assert [x for x in _rule_violations(NoDriverRule(), top)
            if x.location.net_name == 'x'] == []


def test_int001_still_blind_to_a_device_against_a_submodule_output(pdk, tmp_path):
    """Documents the chosen device-binding semantics (TASK-46): a device pin
    is presence only, never an OUT-direction driver, so INT-001 does not
    report a MOSFET drain shorted against a submodule output. Deliberate
    — the alternative (drain/source = OUT) would flag every pass device
    and every output-stage transistor; a real direction model for catalog
    pins is not this task's. Unchanged from pre-fix (``[]``)."""
    inv = _inverter(pdk, tmp_path)
    top = _mod(pdk, tmp_path, 'top.cir')
    top.add_module(inv, 'u1', nets={'in': 'a', 'out': 'y', 'vdd': 'vdd', 'gnd': '0'})
    top.add_module(inv, 'u2', nets={'in': 'y', 'out': 'z', 'vdd': 'vdd', 'gnd': '0'})
    with top:
        pdk.nmos1p8('mfight', d='y', g='vin', s='0', b='0', w=2.0, l=0.15)
    assert _rule_violations(DriverConflictRule(), top) == []


def test_int004_silent_on_ground_net_of_a_bare_portdir_testbench(pdk, tmp_path):
    """F-02 / TASK-45 (was xfail). Bare ``PortDir.GROUND`` now carries
    ``PortRole.GROUND`` at declaration, so the ground net of an ordinary
    testbench (ground ports + source negatives) is a supply net to INT-004."""
    inv = _inverter(pdk, tmp_path, supply_ports={'vdd': PortDir.POWER,
                                                 'gnd': PortDir.GROUND})
    top = _mod(pdk, tmp_path, 'top.cir')
    top.add_module(inv, 'u1', nets={'in': 'a', 'out': 'y', 'vdd': 'vdd', 'gnd': '0'})
    top.add_module(inv, 'u2', nets={'in': 'y', 'out': 'z', 'vdd': 'vdd', 'gnd': '0'})
    with top:
        sources.VoltageSource('vdd', positive='vdd', negative='0', dc=1.8)
        sources.VoltageSource('vin', positive='a', negative='0', dc=0.9)
    assert [x for x in _rule_violations(NoDriverRule(), top)
            if x.location.net_name == '0'] == []


def test_int005_fires_on_clock_into_analog(pdk, tmp_path):
    clkgen = _block(pdk, tmp_path, 'clkgen', {'clk': (PortDir.OUT, PortRole.CLOCK),
                                              'en': (PortDir.IN, PortRole.DIGITAL)})
    amp = _block(pdk, tmp_path, 'amp', {'out': (PortDir.OUT, PortRole.ANALOG),
                                        'in': (PortDir.IN, PortRole.ANALOG)})
    top = _mod(pdk, tmp_path, 'top.cir')
    top.add_module(clkgen, 'c1', nets={'clk': 'ck', 'en': 'en'})
    top.add_module(amp, 'a1', nets={'in': 'ck', 'out': 'y'})
    v = _only('INT-005', _rule_violations(ClockOnAnalogRule(), top))
    assert len(v) == 1
    assert 'c1.clk(clock)' in v[0].message and 'a1.in(analog)' in v[0].message


def test_int005_silent_on_clock_into_digital(pdk, tmp_path):
    clkgen = _block(pdk, tmp_path, 'clkgen', {'clk': (PortDir.OUT, PortRole.CLOCK),
                                              'en': (PortDir.IN, PortRole.DIGITAL)})
    inv = _inverter(pdk, tmp_path)
    top = _mod(pdk, tmp_path, 'top.cir')
    top.add_module(clkgen, 'c1', nets={'clk': 'ck', 'en': 'en'})
    top.add_module(inv, 'u1', nets={'in': 'ck', 'out': 'y', 'vdd': 'vdd', 'gnd': '0'})
    assert _rule_violations(ClockOnAnalogRule(), top) == []


# ---------------------------------------------------------------------------
# DOM-001..003 — voltage-domain rules
# ---------------------------------------------------------------------------

def _domain_pair():
    return (VoltageDomain('avdd', nominal=1.8, tolerance=0.1),
            VoltageDomain('dvdd', nominal=0.9, tolerance=0.1))


def _domain_block(pdk, tmp_path, name, dom):
    return _block(pdk, tmp_path, name, {
        'out': Port(dir=PortDir.OUT, role=PortRole.ANALOG, domain=dom),
        'in': Port(dir=PortDir.IN, role=PortRole.ANALOG, domain=dom),
        'vdd': Port(dir=PortDir.POWER, role=PortRole.POWER, domain=dom),
    })


def test_dom001_fires_when_two_rails_meet_on_one_net(pdk, tmp_path):
    avdd, dvdd = _domain_pair()
    top = _mod(pdk, tmp_path, 'top.cir')
    top.add_module(_domain_block(pdk, tmp_path, 'ablk', avdd), 'a1',
                    nets={'in': 'ia', 'out': 'oa', 'vdd': 'shared_vdd'})
    top.add_module(_domain_block(pdk, tmp_path, 'dblk', dvdd), 'd1',
                    nets={'in': 'id', 'out': 'od', 'vdd': 'shared_vdd'})
    v = _only('DOM-001', _rule_violations(RailMismatchRule(), top))
    assert len(v) == 1
    assert v[0].location.net_name == 'shared_vdd'
    assert 'a1.vdd(avdd, 1.8V)' in v[0].message and 'd1.vdd(dvdd, 0.9V)' in v[0].message


def test_dom001_silent_when_rails_match(pdk, tmp_path):
    avdd, _ = _domain_pair()
    top = _mod(pdk, tmp_path, 'top.cir')
    blk = _domain_block(pdk, tmp_path, 'ablk', avdd)
    top.add_module(blk, 'a1', nets={'in': 'ia', 'out': 'x', 'vdd': 'vdd'})
    top.add_module(blk, 'a2', nets={'in': 'x', 'out': 'y', 'vdd': 'vdd'})
    assert _rule_violations(RailMismatchRule(), top) == []


def test_dom002_fires_on_unannotated_signal_crossing(pdk, tmp_path):
    avdd, dvdd = _domain_pair()
    top = _mod(pdk, tmp_path, 'top.cir')
    top.add_module(_domain_block(pdk, tmp_path, 'ablk', avdd), 'a1',
                    nets={'in': 'ia', 'out': 'cross', 'vdd': 'avdd'})
    top.add_module(_domain_block(pdk, tmp_path, 'dblk', dvdd), 'd1',
                    nets={'in': 'cross', 'out': 'od', 'vdd': 'dvdd'})
    v = _only('DOM-002', _rule_violations(DomainCrossingRule(), top))
    assert len(v) == 1
    assert v[0].location.net_name == 'cross'
    assert 'a1.out(avdd)' in v[0].message and 'd1.in(dvdd)' in v[0].message


def test_dom002_silent_within_one_domain(pdk, tmp_path):
    avdd, _ = _domain_pair()
    top = _mod(pdk, tmp_path, 'top.cir')
    blk = _domain_block(pdk, tmp_path, 'ablk', avdd)
    top.add_module(blk, 'a1', nets={'in': 'ia', 'out': 'x', 'vdd': 'vdd'})
    top.add_module(blk, 'a2', nets={'in': 'x', 'out': 'y', 'vdd': 'vdd'})
    assert _rule_violations(DomainCrossingRule(), top) == []


def test_dom003_silent_on_an_unannotated_crossing(pdk, tmp_path):
    """TASK-49 (ledger F-07): a crossing with no switchable rail is
    DOM-002's finding alone. DOM-003 no longer treats every domain as
    hypothetically the 0 V one."""
    avdd, dvdd = _domain_pair()
    top = _mod(pdk, tmp_path, 'top.cir')
    top.add_module(_domain_block(pdk, tmp_path, 'ablk', avdd), 'a1',
                    nets={'in': 'ia', 'out': 'cross', 'vdd': 'avdd'})
    top.add_module(_domain_block(pdk, tmp_path, 'dblk', dvdd), 'd1',
                    nets={'in': 'cross', 'out': 'od', 'vdd': 'dvdd'})
    assert len(_only('DOM-002', _rule_violations(DomainCrossingRule(), top))) == 1
    assert _rule_violations(SupplyDisabledStressRule(), top) == []


def test_dom003_fires_once_when_the_driven_rail_is_switchable(pdk, tmp_path):
    """Declaring dvdd switchable is the input DOM-003 keys on: one finding
    on the crossing, naming the switchable rail and the live rail that
    would forward-bias it — evidence DOM-002 does not carry."""
    avdd, _ = _domain_pair()
    dvdd = VoltageDomain('dvdd', nominal=0.9, tolerance=0.1, switchable=True)
    top = _mod(pdk, tmp_path, 'top.cir')
    top.add_module(_domain_block(pdk, tmp_path, 'ablk', avdd), 'a1',
                    nets={'in': 'ia', 'out': 'cross', 'vdd': 'avdd'})
    top.add_module(_domain_block(pdk, tmp_path, 'dblk', dvdd), 'd1',
                    nets={'in': 'cross', 'out': 'od', 'vdd': 'dvdd'})
    v = _only('DOM-003', _rule_violations(SupplyDisabledStressRule(), top))
    assert len(v) == 1
    assert v[0].location.net_name == 'cross'
    assert "stress domain 'dvdd' (switchable" in v[0].message
    assert "from domain 'avdd' (1.8V nominal)" in v[0].message
    # The always-on rail is never reported as the disabled side.
    assert "stress domain 'avdd'" not in v[0].message


def test_dom003_names_only_the_switchable_rail_as_the_disabled_side(pdk, tmp_path):
    """Flip the annotation: avdd switchable, dvdd always-on. The finding
    must name avdd as the rail at risk and never dvdd — the always-on rail
    is not a candidate for the 0 V side."""
    avdd = VoltageDomain('avdd', nominal=1.8, tolerance=0.1, switchable=True)
    _, dvdd = _domain_pair()
    top = _mod(pdk, tmp_path, 'top.cir')
    top.add_module(_domain_block(pdk, tmp_path, 'ablk', avdd), 'a1',
                    nets={'in': 'ia', 'out': 'cross', 'vdd': 'avdd'})
    top.add_module(_domain_block(pdk, tmp_path, 'dblk', dvdd), 'd1',
                    nets={'in': 'cross', 'out': 'od', 'vdd': 'dvdd'})
    v = _only('DOM-003', _rule_violations(SupplyDisabledStressRule(), top))
    # Both blocks expose in+out on 'cross', so the switchable avdd block has
    # an input pin there and is legitimately at risk from dvdd's output.
    assert len(v) == 1
    assert "stress domain 'avdd' (switchable" in v[0].message
    assert "stress domain 'dvdd'" not in v[0].message


def test_dom003_silent_within_one_domain(pdk, tmp_path):
    avdd, _ = _domain_pair()
    top = _mod(pdk, tmp_path, 'top.cir')
    blk = _domain_block(pdk, tmp_path, 'ablk', avdd)
    top.add_module(blk, 'a1', nets={'in': 'ia', 'out': 'x', 'vdd': 'vdd'})
    top.add_module(blk, 'a2', nets={'in': 'x', 'out': 'y', 'vdd': 'vdd'})
    assert _rule_violations(SupplyDisabledStressRule(), top) == []


# ---------------------------------------------------------------------------
# STR-001 — behavioral data path (real devices, Net.voltage waveforms)
# ---------------------------------------------------------------------------

def _behavioral(module, t, voltages):
    """Put a real module into silk's behavioral-results state."""
    module._behavioral_mode = True
    module._behavioral_t = t
    for net_name, wave in voltages.items():
        if net_name not in module.net_dict:
            module.add_net(net_name)
        module.net_dict[net_name].voltage = wave
    return module


def test_str001_behavioral_fires_naming_device_pair_value_and_time(pdk, tmp_path):
    inv = _inverter(pdk, tmp_path)
    n = 100
    t = np.linspace(0, 10e-9, n)
    vin = np.full(n, _VGS_GOOD_V)
    vin[40:] = _VGS_BAD_V                    # overstress begins at sample 40
    _behavioral(inv, t, {'in': vin, 'out': np.full(n, 0.9),
                         'vdd': np.full(n, 1.8), 'gnd': np.zeros(n)})
    v = _rule_violations(JunctionOverrangeRule(), inv)
    vgs = [x for x in v if x.location.instance_path == 'mn' and x.location.terminal == 'Vgs']
    assert len(vgs) == 1
    x = vgs[0]
    assert isinstance(x, JunctionViolation)
    assert x.severity == Severity.ERROR
    assert x.device_type == 'nmos1p8'
    assert x.measured_v == pytest.approx(_VGS_BAD_V)
    assert x.rated_max == pytest.approx(_NMOS1P8_VGS_MAX_V)
    assert x.worst_time == pytest.approx(t[40])
    assert x.location.terminal_pair == ('g', 's')
    assert x.message.startswith('mn (nmos1p8): Vgs=2.500V exceeds Vgs_max=1.95V at t=')
    assert x.message.endswith('ns')


def test_str001_behavioral_silent_within_ratings(pdk, tmp_path):
    inv = _inverter(pdk, tmp_path)
    n = 50
    _behavioral(inv, np.linspace(0, 1e-9, n),
                {'in': np.full(n, _VGS_GOOD_V), 'out': np.full(n, 0.9),
                 'vdd': np.full(n, 1.8), 'gnd': np.zeros(n)})
    assert _rule_violations(JunctionOverrangeRule(), inv) == []


def test_str001_behavioral_ground_net_without_waveform_is_zero(pdk, tmp_path):
    """s/b on 'gnd' with no stored waveform must read as 0 V, not skip."""
    inv = _inverter(pdk, tmp_path)
    n = 20
    _behavioral(inv, np.linspace(0, 1e-9, n),
                {'in': np.full(n, _VGS_BAD_V), 'out': np.full(n, 0.9),
                 'vdd': np.full(n, 1.8)})           # no 'gnd' waveform
    v = _rule_violations(JunctionOverrangeRule(), inv)
    assert any(x.location.instance_path == 'mn' and x.location.terminal == 'Vgs'
               for x in v)


def test_str001_behavioral_vss_net_without_waveform_is_zero(pdk, tmp_path):
    """AC#2 / TASK-50: pre-fix, ``_check_behavioral`` recognized only the
    inline ``('gnd', '0')`` pair as ground, so a body tied to 'vss' with no
    stored waveform made Vbs uncomputable (silently skipped) even though
    the SPICE path's ``_GROUND_HEURISTIC_NAMES`` already treated 'vss' as
    0 V. Both paths now share ``_is_ground_node``, so this fires the same
    way ``test_str001_spice_path_all_ground_device_and_missing_pdk_are_silent``
    exercises 'vss' on the SPICE path."""
    inv = _inverter(pdk, tmp_path, body_n='vss')
    n = 20
    _behavioral(inv, np.linspace(0, 1e-9, n),
                {'in': np.full(n, _VGS_GOOD_V), 'out': np.full(n, 0.9),
                 'vdd': np.full(n, 1.8),
                 'gnd': np.full(n, _VGS_BAD_V)})    # no 'vss' waveform
    v = _rule_violations(JunctionOverrangeRule(), inv)
    assert any(x.location.instance_path == 'mn' and x.location.terminal == 'Vbs'
               for x in v)


def test_str001_behavioral_device_with_missing_net_waveform_is_skipped(pdk, tmp_path):
    """A terminal whose net has no waveform (and is not ground) makes that
    pair uncomputable: the rule skips the pair rather than inventing 0 V."""
    inv = _inverter(pdk, tmp_path)
    n = 20
    _behavioral(inv, np.linspace(0, 1e-9, n),
                {'in': np.full(n, _VGS_BAD_V), 'gnd': np.zeros(n)})   # no 'out', no 'vdd'
    v = _rule_violations(JunctionOverrangeRule(), inv)
    # mn: Vgs computable (in - gnd) -> fires; mp: every pair needs 'vdd' or 'out' -> silent
    assert {x.location.instance_path for x in v} == {'mn'}


def test_str001_behavioral_unbound_terminal_makes_its_pairs_uncomputable(pdk, tmp_path):
    """g=None on a rated device: every pair involving the gate is skipped
    (STRUCT-004 owns that defect), the d-s/b-s pairs still score."""
    inv = _inverter(pdk, tmp_path, gate=None)
    n = 20
    _behavioral(inv, np.linspace(0, 1e-9, n),
                {'out': np.full(n, 2.5), 'vdd': np.full(n, 1.8), 'gnd': np.zeros(n)})
    v = _rule_violations(JunctionOverrangeRule(), inv)
    assert {(x.location.instance_path, x.location.terminal) for x in v} == {('mn', 'Vds'), ('mn', 'Vbd')}
    from silk.circuit.junction_check import junction_stress_coverage
    coverage = junction_stress_coverage(inv)
    assert coverage.summary() == {'total': 2, 'full': 0, 'partial': 2, 'none': 0, 'failed': 1}
    assert all(d.reasons == ('no_waveform',) for d in coverage.devices)


def test_str001_behavioral_no_time_axis_is_silent(pdk, tmp_path):
    inv = _inverter(pdk, tmp_path)
    inv._behavioral_mode = True
    inv._behavioral_t = None
    assert _rule_violations(JunctionOverrangeRule(), inv) == []


def test_str001_behavioral_catalog_device_without_ratings_is_skipped(pdk, tmp_path):
    """sky130 res_hpo declares no junction_ratings -> nothing to score."""
    m = _mod(pdk, tmp_path, 'r.cir', 'rblk')
    with m:
        pdk.res_hpo('r1', a='p', b='n', c='0', w=0.35, l=0.5)
        elements.capacitor('c1', p='p', n='n', value=1e-12)  # no catalog entry
    n = 10
    _behavioral(m, np.linspace(0, 1e-9, n), {'p': np.full(n, 30.0), 'n': np.zeros(n)})
    assert _rule_violations(JunctionOverrangeRule(), m) == []


def test_str001_behavioral_diode_and_bjt_pairs(pdk, tmp_path):
    """Non-MOS terminal pairs: Vak on a diode, Vbe/Vbc/Vce on a BJT."""
    m = _mod(pdk, tmp_path, 'db.cir', 'dblk')
    with m:
        pdk.diode_std('d1', a='an', k='ka')
        pdk.npn_std('q1', c='col', b='bas', e='emi', s='0')
    n = 10
    _behavioral(m, np.linspace(0, 1e-9, n), {
        'an': np.full(n, 6.0), 'ka': np.zeros(n),          # Vak 6.0 > 5.0
        'col': np.full(n, 6.0), 'bas': np.full(n, 0.5), 'emi': np.zeros(n),  # Vce 6.0 > 5.0
    })
    v = _rule_violations(JunctionOverrangeRule(), m)
    hits = {(x.location.instance_path, x.location.terminal) for x in v}
    assert ('d1', 'Vak') in hits
    assert ('q1', 'Vce') in hits
    assert ('q1', 'Vbe') not in hits          # 0.5 V < Vbe_max 5.0 (SkyWater |VBE| 0-5.0 V; F-05 / TASK-48)


# ---------------------------------------------------------------------------
# STR-001 — SPICE data path on a REAL two-level hierarchy
#
# ngspice is not run in the fast suite; the waveforms are served through
# Module.load_signal by name, exactly the way NgspiceSignal.from_rawfile
# would serve them. What is under test is the part that was wrong before
# the 2026-07-14 fix: resolving each device terminal to its rawfile node
# (port nets UPWARD to the parent, internal nets prefixed with x<inst>.,
# ground to 0 V) and scoring against the device's own rating. The real
# ngspice version of the same assertion is the @slow test in
# tests/lint/rules/test_stress_rules.py.
# ---------------------------------------------------------------------------

def _two_stage_leaf(pdk, tmp_path):
    leaf = _mod(pdk, tmp_path, 'leaf.cir', 'leaf2')
    leaf.define_module_ports({'in': PortDir.IN, 'out': PortDir.OUT,
                              'vdd': PortDir.POWER, 'gnd': PortDir.GROUND})
    with leaf:
        pdk.nmos1p8('mn1', d='mid', g='in', s='gnd', b='gnd', w=1.0, l=0.15)
        pdk.nmos1p8('mn2', d='out', g='mid', s='gnd', b='gnd', w=1.0, l=0.15)
        elements.resistor('r1', p='vdd', n='mid', value=10e3)
        elements.resistor('r2', p='vdd', n='out', value=10e3)
    return leaf


def _fake_rawfile(axis_name, real):
    """A rawfile stub shaped like the part STR-001 classifies off.

    Only ``current_plot.vectors[0].name`` and ``current_plot.real`` are read
    (see ``_spice_results_are_small_signal``). The values used by the tests
    are the ones real ngspice produces, verified against a live run under
    sky130: transient -> ('time', real), .ac -> ('frequency', complex),
    .dc -> ('v(v-sweep)', real).
    """
    plot = types.SimpleNamespace(
        vectors=[types.SimpleNamespace(name=axis_name)], real=real)
    return types.SimpleNamespace(current_plot=plot)


def _spice_top(pdk, tmp_path, waveforms, *, n=50, t_stop=10e-9,
               xaxis='time', real=True, x=None):
    """Build top -> x amp(leaf2) and serve *waveforms* {node: array} via
    load_signal. Returns (top, requested_nodes, x_axis).

    ``xaxis``/``real`` set the rawfile plot the module presents, so a test can
    stand the module up as a transient, AC or DC-sweep run (TASK-81); ``x``
    overrides the independent-axis values (defaults to a 0..t_stop ramp).
    """
    top = _mod(pdk, tmp_path, 'top.cir')
    top.add_module(_two_stage_leaf(pdk, tmp_path), 'amp',
                    nets={'in': 'vin', 'out': 'vout', 'vdd': 'vdd', 'gnd': '0'})
    with top:
        sources.VoltageSource('vdd', positive='vdd', negative='0', dc=1.8)
        sources.VoltageSource('vin', positive='vin', negative='0', dc=0.0)
    t = np.linspace(0, t_stop, n) if x is None else np.asarray(x, dtype=float)
    requested = []

    def load_signal(name, kind='V'):
        requested.append(name)
        if name not in waveforms:
            raise KeyError(name)
        y = np.asarray(waveforms[name])
        if not np.iscomplexobj(y):
            y = y.astype(float)
        return types.SimpleNamespace(x=t[:len(y)], y=y, xlabel=xaxis)

    top.sim_results_loaded = True
    top.simResultsAll = _fake_rawfile(xaxis, real)
    top.load_signal = load_signal
    return top, requested, t


def test_str001_spice_path_resolves_hierarchy_and_fires_inside_submodule(pdk, tmp_path):
    """Pre-TASK-50, this asserted ``instance_path == 'xamp.mn1'`` (the SPICE
    hierarchy prefix leaking into the reported path) — wrong because that
    form never matches a silk-form exclusion/waiver on ``amp`` (ledger F-08,
    see ``test_str001_spice_path_findings_match_subtree_exclusions``). The
    rawfile NODE name (what ``requested`` collects) still needs the SPICE
    ``x`` prefix to resolve against ngspice's own vector names, so only the
    reported ``instance_path``/message change, not node resolution."""
    n = 50
    top, requested, t = _spice_top(pdk, tmp_path, {
        'vin': np.full(n, _VGS_BAD_V),          # port net: resolved UPWARD, no prefix
        'vout': np.full(n, 1.7),
        'vdd': np.full(n, 1.8),
        'xamp.mid': np.full(n, 0.2),            # internal net: prefixed
    }, n=n)
    v = _rule_violations(JunctionOverrangeRule(), top)
    # Node resolution is the thing under test: exactly these names — the
    # port nets without prefix, the internal net prefixed, ground ('0')
    # never requested because it is synthesised as 0 V, and 'vdd' never
    # requested because no RATED device touches it (only the resistors do).
    assert set(requested) == {'vin', 'vout', 'xamp.mid'}
    hits = {(x.location.instance_path, x.location.terminal) for x in v}
    # instance_path is silk form ('amp.mn1'), not the SPICE node prefix
    # ('xamp.mn1') used above to resolve the waveform — TASK-50.
    assert ('amp.mn1', 'Vgs') in hits           # 2.5 V gate on a 1.95 V device
    assert ('amp.mn1', 'Vgd') in hits           # 2.5 - 0.2 = 2.3 V
    assert not any(p.endswith('mn2') for p, _ in hits)   # Vgs = 0.2 V: fine
    x = next(x for x in v if x.location.terminal == 'Vgs'
             and x.location.instance_path == 'amp.mn1')
    assert x.device_type == 'nmos1p8'
    assert x.message.startswith('amp.mn1 (nmos1p8): Vgs=2.500V exceeds Vgs_max=1.95V at t=')
    assert x.location.terminal_pair == ('g', 's')


def test_str001_spice_path_findings_match_subtree_exclusions(pdk, tmp_path):
    from silk.lint.exclusions import ScopeExclusion
    n = 50
    top, _, _ = _spice_top(pdk, tmp_path, {
        'vin': np.full(n, _VGS_BAD_V), 'vout': np.full(n, 1.7),
        'vdd': np.full(n, 1.8), 'xamp.mid': np.full(n, 0.2)}, n=n)
    reg = RuleRegistry(rules=[JunctionOverrangeRule()],
                       exclusions=[ScopeExclusion('instance', 'amp', reason='vendor block')])
    v = reg.run(top, has_simulation_data=True).violations
    assert v and all(reg.is_excluded(x.location.instance_path) for x in v)


def test_str001_spice_path_silent_within_ratings(pdk, tmp_path):
    n = 50
    top, _, _ = _spice_top(pdk, tmp_path, {
        'vin': np.full(n, _VGS_GOOD_V), 'vout': np.full(n, 1.7),
        'vdd': np.full(n, 1.8), 'xamp.mid': np.full(n, 0.2)}, n=n)
    assert _rule_violations(JunctionOverrangeRule(), top) == []


def test_str001_spice_path_unresolvable_node_skips_only_that_device(pdk, tmp_path):
    """No waveform for the internal net -> mn1 (d=mid) and mn2 (g=mid) are
    skipped, never crash; the rule still runs to completion."""
    n = 50
    top, requested, _ = _spice_top(pdk, tmp_path, {
        'vin': np.full(n, _VGS_BAD_V), 'vout': np.full(n, 1.7), 'vdd': np.full(n, 1.8)}, n=n)
    assert _rule_violations(JunctionOverrangeRule(), top) == []
    assert 'xamp.mid' in requested


def test_str001_spice_path_length_mismatch_skips_device(pdk, tmp_path):
    n = 50
    top, _, _ = _spice_top(pdk, tmp_path, {
        'vin': np.full(n, _VGS_BAD_V), 'vout': np.full(n, 1.7),
        'vdd': np.full(n, 1.8), 'xamp.mid': np.full(n // 2, 0.2)}, n=n)
    assert _rule_violations(JunctionOverrangeRule(), top) == []


def test_str001_spice_path_matches_node_case_variants(pdk, tmp_path):
    """Rawfile vector names may come back upper-cased; the resolver retries
    lower/upper before declaring the node unresolvable."""
    n = 50
    top, _, _ = _spice_top(pdk, tmp_path, {
        'VIN': np.full(n, _VGS_BAD_V), 'VOUT': np.full(n, 1.7),
        'VDD': np.full(n, 1.8), 'XAMP.MID': np.full(n, 0.2)}, n=n)
    v = _rule_violations(JunctionOverrangeRule(), top)
    assert ('amp.mn1', 'Vgs') in {(x.location.instance_path, x.location.terminal) for x in v}


def test_str001_spice_path_all_ground_device_and_missing_pdk_are_silent(pdk, tmp_path):
    n = 50
    top, _, _ = _spice_top(pdk, tmp_path, {
        'vin': np.full(n, _VGS_GOOD_V), 'vout': np.full(n, 1.7),
        'vdd': np.full(n, 1.8), 'xamp.mid': np.full(n, 0.2)}, n=n)
    with top:
        # every terminal on ground: nothing to measure, must not crash
        pdk.nmos1p8('mgnd', d='0', g='gnd', s='vss', b='0', w=1.0, l=0.15)
    assert _rule_violations(JunctionOverrangeRule(), top) == []
    top.pdk_details = None
    assert _rule_violations(JunctionOverrangeRule(), top) == []


def test_str001_ground_node_heuristic_is_documented_and_hierarchical():
    for name in ('0', 'gnd', 'GND', 'vss', 'xamp.gnd', 'xa.xb.0'):
        assert _is_ground_node(name), name
    for name in ('vssa', 'gnd_int', 'agnd', 'xamp.mid'):
        assert not _is_ground_node(name), name


def test_str001_registry_skips_with_no_simulation_data(pdk, tmp_path):
    inv = _inverter(pdk, tmp_path)
    reg = RuleRegistry(rules=[JunctionOverrangeRule()])
    res = reg.run(inv, has_simulation_data=False)
    assert res.violations == []
    assert res.visit_log[0].outcome.value == 'skipped_no_sim'


def test_str001_runs_through_generate_report_on_spice_data(pdk, tmp_path):
    # F-04 / TASK-47: generate_report used to gate STR-001 on _behavioral_mode
    # only, so a SPICE-simulated module never had its stress rule run through
    # the report path even though the rule itself reads the SPICE data.
    from silk.circuit.report import generate_report
    n = 50
    top, _, _ = _spice_top(pdk, tmp_path, {
        'vin': np.full(n, _VGS_BAD_V), 'vout': np.full(n, 1.7),
        'vdd': np.full(n, 1.8), 'xamp.mid': np.full(n, 0.2)}, n=n)
    rep = generate_report(top, auto_discover_config=False)
    assert any(v.rule_id == 'STR-001' for v in rep.lint.violations)


def test_str001_not_skipped_no_sim_in_report_visit_log_on_spice_data(pdk, tmp_path):
    """The scorecard must not claim STR-001 was skipped on a measured module."""
    from silk.circuit.report import generate_report
    n = 50
    # Both catalog transistors are checked; ideal resistor loads are out of scope.
    top, _, _ = _spice_top(pdk, tmp_path, {
        'vin': np.full(n, _VGS_GOOD_V), 'vout': np.full(n, 1.7),
        'vdd': np.full(n, 1.8), 'xamp.mid': np.full(n, 0.2)}, n=n)
    rep = generate_report(top, auto_discover_config=False)
    assert not any(v.rule_id == 'STR-001' for v in rep.lint.violations)
    assert rep.coverage.rule_coverage.root.local['STR-001'] == (2, 0, 0)
    assert {d['instance_path'] for d in rep.stress_coverage['devices']} == {'amp.mn1', 'amp.mn2'}


def test_str001_still_skipped_in_report_without_any_simulation(pdk, tmp_path):
    from silk.circuit.report import generate_report
    inv = _inverter(pdk, tmp_path)
    rep = generate_report(inv, auto_discover_config=False)
    assert rep.coverage.rule_coverage.root.local['STR-001'] == (0, 0, 2)


# ---------------------------------------------------------------------------
# TASK-81 — STR-001 must not score AC/spectral results as node voltages
# ---------------------------------------------------------------------------

def _str001_coverage(rep):
    """(checked, excluded, unchecked) for STR-001 in a CircuitReport."""
    return rep.coverage.rule_coverage.root.local.get('STR-001')


def test_str001_declines_ac_results_through_generate_report(pdk, tmp_path):
    """An AC plot's "voltages" are transfer ratios: measuring them fabricates
    ERROR findings on a circuit that is nowhere near its ratings (TASK-81)."""
    from silk.circuit.report import generate_report
    n = 50
    # Complex small-signal data whose magnitude sails past every rating.
    big = np.full(n, 4.6 + 1.2j)
    top, requested, _ = _spice_top(pdk, tmp_path, {
        'vin': big, 'vout': big, 'vdd': big, 'xamp.mid': big},
        n=n, xaxis='frequency', real=False)
    rep = generate_report(top, auto_discover_config=False)
    assert not any(v.rule_id == 'STR-001' for v in rep.lint.violations)
    # Recorded as unchecked, not as a clean pass — the scorecard must not
    # claim a stress check that never happened.
    assert _str001_coverage(rep) == (0, 0, 2)
    assert all('small_signal' in d['reasons'] for d in rep.stress_coverage['devices'])
    # Declined before reading anything, so no waveform was even requested.
    assert requested == []


def test_str001_declines_noise_results_through_generate_report(pdk, tmp_path):
    """A real-valued noise spectrum is still V/sqrt(Hz), not a node voltage."""
    from silk.circuit.report import generate_report
    n = 50
    top, requested, _ = _spice_top(pdk, tmp_path, {
        'vin': np.full(n, 9.9), 'vout': np.full(n, 9.9),
        'vdd': np.full(n, 9.9), 'xamp.mid': np.full(n, 9.9)},
        n=n, xaxis='frequency', real=True)
    rep = generate_report(top, auto_discover_config=False)
    assert not any(v.rule_id == 'STR-001' for v in rep.lint.violations)
    assert _str001_coverage(rep) == (0, 0, 2)
    assert all('small_signal' in d['reasons'] for d in rep.stress_coverage['devices'])
    assert requested == []


def test_str001_checks_the_loaded_plot_not_the_declared_analyses(pdk, tmp_path):
    """A testbench may declare several analyses while the rawfile holds one.

    Verified against real ngspice: declaring Transient + AC leaves a single
    transient plot, so the transient data must still be checked (TASK-81 AC#3).
    """
    from silk.circuit.report import generate_report
    from silk.simulation.analysis import AC, Transient
    n = 50
    top, _, _ = _spice_top(pdk, tmp_path, {
        'vin': np.full(n, _VGS_BAD_V), 'vout': np.full(n, 1.7),
        'vdd': np.full(n, 1.8), 'xamp.mid': np.full(n, 0.2)}, n=n)
    top.add_sim_type(Transient(step=1e-9, stop=20e-9))
    top.add_sim_type(AC(variation='dec', points=10, fstart=1, fstop=1e9))
    assert sorted(top.sim_analysis_dict) == ['ac', 'transient']
    rep = generate_report(top, auto_discover_config=False)
    assert any(v.rule_id == 'STR-001' for v in rep.lint.violations)
    assert _str001_coverage(rep) == (2, 0, 0)


def test_str001_dc_sweep_findings_label_the_sweep_axis_not_a_timestamp(pdk, tmp_path):
    """A DC sweep has no timestamp; printing its sweep value as nanoseconds
    ('t=3000000000.00ns' for 3.0 V) is not something a designer can act on."""
    n = 7
    sweep = np.linspace(0.0, 3.0, n)
    top, _, _ = _spice_top(pdk, tmp_path, {
        'vin': sweep, 'vout': np.full(n, 1.7),
        'vdd': np.full(n, 1.8), 'xamp.mid': np.full(n, 0.2)},
        n=n, xaxis='v(v-sweep)', real=True, x=sweep)
    v = _only('STR-001', _rule_violations(JunctionOverrangeRule(), top))
    worst = next(x for x in v if x.location.terminal == 'Vgs')
    assert worst.message.endswith('at v(v-sweep)=3.000'), worst.message
    assert 'ns' not in worst.message
    assert worst.worst_time == pytest.approx(3.0)


def test_str001_transient_findings_keep_the_nanosecond_timestamp(pdk, tmp_path):
    n = 50
    top, _, _ = _spice_top(pdk, tmp_path, {
        'vin': np.full(n, _VGS_BAD_V), 'vout': np.full(n, 1.7),
        'vdd': np.full(n, 1.8), 'xamp.mid': np.full(n, 0.2)}, n=n)
    v = _only('STR-001', _rule_violations(JunctionOverrangeRule(), top))
    assert all('at t=' in x.message and x.message.endswith('ns') for x in v)


def test_str001_skips_a_complex_vector_without_casting_it(pdk, tmp_path, recwarn):
    """Belt and braces behind the rule-level gate: a complex vector reaching
    the walk skips its device instead of being cast to real and measured."""
    n = 50
    # Plot classifies as transient (real, 'time'), but one node hands back a
    # complex vector — a mixed/unexpected rawfile.
    top, _, _ = _spice_top(pdk, tmp_path, {
        'vin': np.full(n, _VGS_BAD_V + 0.5j), 'vout': np.full(n, 1.7),
        'vdd': np.full(n, 1.8), 'xamp.mid': np.full(n, 0.2)}, n=n)
    assert _rule_violations(JunctionOverrangeRule(), top) == []
    assert not [w for w in recwarn if 'Complex' in type(w.message).__name__]


def test_str001_still_checks_ordinary_transient_spice_results(pdk, tmp_path):
    """The guard must not cost the rule its actual job."""
    n = 50
    top, _, _ = _spice_top(pdk, tmp_path, {
        'vin': np.full(n, _VGS_BAD_V), 'vout': np.full(n, 1.7),
        'vdd': np.full(n, 1.8), 'xamp.mid': np.full(n, 0.2)}, n=n)
    assert JunctionOverrangeRule().simulation_data_usable(top) is True
    assert _only('STR-001', _rule_violations(JunctionOverrangeRule(), top))


def test_str001_behavioral_results_are_always_usable(pdk, tmp_path):
    """Behavioral data is a time-domain walk by construction, and carries no
    rawfile for the classifier to read."""
    inv = _inverter(pdk, tmp_path)
    inv._behavioral_mode = True
    assert JunctionOverrangeRule().simulation_data_usable(inv) is True


def test_str001_unclassifiable_results_stay_checkable(pdk, tmp_path):
    """No rawfile object to classify (a hand-built module, a stubbed reader):
    behave exactly as before the guard rather than silently declining."""
    n = 50
    top, _, _ = _spice_top(pdk, tmp_path, {
        'vin': np.full(n, _VGS_BAD_V), 'vout': np.full(n, 1.7),
        'vdd': np.full(n, 1.8), 'xamp.mid': np.full(n, 0.2)}, n=n)
    del top.simResultsAll
    assert JunctionOverrangeRule().simulation_data_usable(top) is True
    assert _only('STR-001', _rule_violations(JunctionOverrangeRule(), top))


def test_registry_records_unusable_simulation_data_as_skipped(pdk, tmp_path):
    """The registry honours a rule's veto even when sim data is present."""
    class _Picky(JunctionOverrangeRule):
        def simulation_data_usable(self, scope):
            return False

    inv = _inverter(pdk, tmp_path)
    reg = RuleRegistry(rules=[_Picky()])
    res = reg.run(inv, has_simulation_data=True)
    assert res.violations == []
    assert res.visit_log[0].outcome.value == 'skipped_no_sim'


def test_rule_simulation_data_usable_defaults_to_true(pdk, tmp_path):
    """A rule with no opinion on the analysis type needs no override."""
    inv = _inverter(pdk, tmp_path)
    assert all(r.simulation_data_usable(inv) is True for r in DEFAULT_CATALOG())


@pytest.mark.parametrize('behavioral,spice,expected', [
    (False, False, False),
    (True, False, True),
    (False, True, True),
    (True, True, True),
])
def test_module_has_simulation_data_predicate(behavioral, spice, expected):
    """generate_report and check_junction_overrange share this one predicate."""
    from silk.lint.registry import module_has_simulation_data

    class _Stub:
        pass

    m = _Stub()
    m._behavioral_mode = behavioral
    m.sim_results_loaded = spice
    assert module_has_simulation_data(m) is expected
    # A module that has never been near a simulator has neither attribute.
    assert module_has_simulation_data(_Stub()) is False


# ---------------------------------------------------------------------------
# STRUCT-001..008 + BUS-002 — structural rules
# ---------------------------------------------------------------------------

def test_struct001_fires_naming_instance_and_port(pdk, tmp_path):
    inv = _inverter(pdk, tmp_path)
    top = _mod(pdk, tmp_path, 'top.cir')
    top.add_module(inv, 'u1', nets={'in': 'a', 'vdd': 'vdd', 'gnd': '0'})   # out unbound
    v = _only('STRUCT-001', _rule_violations(DanglingPortRule(), top))
    assert [(x.location.module_name, x.location.port_name) for x in v] == [('u1', 'out')]
    assert "instance 'u1' port 'out' is not connected" in v[0].message


def test_struct001_silent_when_fully_wired(pdk, tmp_path):
    inv = _inverter(pdk, tmp_path)
    top = _mod(pdk, tmp_path, 'top.cir')
    top.add_module(inv, 'u1', nets={'in': 'a', 'out': 'y', 'vdd': 'vdd', 'gnd': '0'})
    assert _rule_violations(DanglingPortRule(), top) == []


def test_struct002_fires_on_unbound_supply_port_only(pdk, tmp_path):
    inv = _inverter(pdk, tmp_path)
    top = _mod(pdk, tmp_path, 'top.cir')
    top.add_module(inv, 'u1', nets={'in': 'a', 'gnd': '0'})   # out AND vdd unbound
    v = _only('STRUCT-002', _rule_violations(PowerPortUnconnectedRule(), top))
    assert [(x.location.module_name, x.location.port_name) for x in v] == [('u1', 'vdd')]
    assert "port 'vdd' (role=power) is not connected" in v[0].message


def test_struct002_fires_on_unbound_bare_portdir_supply_port(pdk, tmp_path):
    """TASK-45: STRUCT-002 sees ports declared with bare PortDir.POWER/GROUND."""
    inv = _inverter(pdk, tmp_path, supply_ports={'vdd': PortDir.POWER, 'gnd': PortDir.GROUND})
    top = _mod(pdk, tmp_path, 'top.cir')
    top.add_module(inv, 'u1', nets={'in': 'a', 'out': 'y', 'gnd': '0'})   # vdd unbound
    v = _only('STRUCT-002', _rule_violations(PowerPortUnconnectedRule(), top))
    assert [(x.location.module_name, x.location.port_name) for x in v] == [('u1', 'vdd')]
    assert "port 'vdd' (role=power) is not connected" in v[0].message


def test_struct002_silent_when_supplies_wired(pdk, tmp_path):
    inv = _inverter(pdk, tmp_path)
    top = _mod(pdk, tmp_path, 'top.cir')
    top.add_module(inv, 'u1', nets={'in': 'a', 'vdd': 'vdd', 'gnd': '0'})  # only 'out' dangles
    assert _rule_violations(PowerPortUnconnectedRule(), top) == []


def test_struct003_fires_on_declared_but_unused_net(pdk, tmp_path):
    inv = _inverter(pdk, tmp_path)
    inv.add_net('spare')
    v = _only('STRUCT-003', _rule_violations(FloatingNetRule(), inv))
    assert [x.location.net_name for x in v] == ['spare']
    assert "Net 'spare' has no connections" == v[0].message


def test_struct003_silent_on_wired_inverter(pdk, tmp_path):
    assert _rule_violations(FloatingNetRule(), _inverter(pdk, tmp_path)) == []


def test_struct004_fires_on_floating_gate(pdk, tmp_path):
    # One case, not two: since TASK-41 both `g=None` and `g=''` are rejected
    # at construction (see tests/test_port_wiring_matrix.py cells D3/D3b), so
    # they no longer describe two ways to reach an unbound gate — there is
    # only the unbound state itself, which _inverter reproduces.
    inv = _inverter(pdk, tmp_path, gate=None)
    v = _only('STRUCT-004', _rule_violations(FloatingGateRule(), inv))
    assert sorted(x.location.instance_path for x in v) == ['mn', 'mp']
    assert "MOSFET 'mn' gate is not connected" in {x.message for x in v}


def test_struct004_silent_on_wired_inverter(pdk, tmp_path):
    assert _rule_violations(FloatingGateRule(), _inverter(pdk, tmp_path)) == []


def test_struct005_fires_on_untied_body(pdk, tmp_path):
    inv = _inverter(pdk, tmp_path, body_n=None)
    v = _only('STRUCT-005', _rule_violations(BodyNotTiedRule(), inv))
    assert [x.location.instance_path for x in v] == ['mn']
    assert v[0].message == "MOSFET 'mn' body/bulk terminal not bound"


def test_struct005_silent_on_wired_inverter(pdk, tmp_path):
    assert _rule_violations(BodyNotTiedRule(), _inverter(pdk, tmp_path)) == []


def test_struct006_fires_when_a_geometry_parameter_is_nulled(pdk, tmp_path):
    """The one reachable true positive: the factory refuses to build a device
    without its required parameters, so an unsized device can only be made by
    nulling one afterwards. The message names the instance and the parameter
    in the spelling the author writes it (``w``, not the legacy ``W``)."""
    inv = _inverter(pdk, tmp_path)
    inv.elem_dict['mn']['device'].w = None
    v = _only('STRUCT-006', _rule_violations(MissingParamRule(), inv))
    assert [x.location.instance_path for x in v] == ['mn']
    assert v[0].message == (
        "Instance 'mn' (type=nmos1p8) missing required parameter 'w'")


def test_struct006_silent_on_fully_parameterised_inverter(pdk, tmp_path):
    """F-01 / TASK-44: the rule reads the names the device declares
    (``w``/``l``/``m``), not the hardcoded legacy ``W``/``L`` spelling."""
    assert _rule_violations(MissingParamRule(), _inverter(pdk, tmp_path)) == []


def test_struct006_silent_on_valued_passives(pdk, tmp_path):
    m = _mod(pdk, tmp_path, 'p.cir', 'pas')
    with m:
        passives.Resistor('r1', pos='a', neg='b', resistance=1e3)
        passives.Capacitor('c1', pos='a', neg='b', capacitance=1e-12)
        passives.Inductor('l1', pos='a', neg='b', inductance=1e-9)
    assert _rule_violations(MissingParamRule(), m) == []


@pytest.mark.parametrize('kind, ctor, kwarg, param', [
    ('Resistor', passives.Resistor, 'resistance', 'resistance'),
    ('Capacitor', passives.Capacitor, 'capacitance', 'capacitance'),
    ('Inductor', passives.Inductor, 'inductance', 'inductance'),
])
def test_struct006_fires_on_a_nulled_passive_value(pdk, tmp_path, kind, ctor,
                                                   kwarg, param):
    """Native passives live in ``passives_dict``, not ``elem_dict`` — before
    TASK-44 the rule never saw them at all. The value is reported under its
    typed alias, the name the author passes to the constructor."""
    m = _mod(pdk, tmp_path, 'p.cir', 'pas')
    with m:
        ctor('p1', pos='a', neg='b', **{kwarg: 1.0})
    m.passives_dict['p1'].value = None
    v = _only('STRUCT-006', _rule_violations(MissingParamRule(), m))
    assert [x.location.instance_path for x in v] == ['p1']
    assert v[0].message == (
        f"Instance 'p1' (type={kind}) missing required parameter '{param}'")


def test_struct006_silent_on_the_legacy_add_element_path(pdk, tmp_path):
    """``add_element`` stores a ``PDKDevice`` whose ``param_list`` values are
    unset until the caller supplies them — the deck writer fills those from
    the catalog ``default``, so the rule must resolve them the same way
    instead of calling an unspecified-but-defaulted device unsized."""
    m = _mod(pdk, tmp_path, 'legacy.cir', 'legacy')
    m.add(pdk.nmos1p8('mn', d='out', g='in', s='gnd', b='gnd'))
    assert _rule_violations(MissingParamRule(), m) == []
    m.add(pdk.nmos1p8('mn2', d='out', g='in', s='gnd', b='gnd', w=1.0, l=0.15))
    assert _rule_violations(MissingParamRule(), m) == []


@pytest.mark.parametrize('catalog', [
    'skywater_130_device_details.yaml',
    'gf180mcu_device_details.yaml',
    'sg13g2_device_details.yaml',
])
def test_struct006_silent_on_every_device_of_every_shipped_catalog(tmp_path,
                                                                   catalog):
    """AC #1: one instance of EVERY device in each of the three shipped
    catalogs, built through the factory with catalog defaults, and the rule
    says nothing. Before TASK-44 every device whose class name contained
    ``mos``/``fet`` reported two ERRORs here."""
    import os
    from silk.pdk.basepdk import BasePDK
    data_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), 'silk', 'pdk', 'data')
    pdk = BasePDK(os.path.join(data_dir, catalog), '/fake/lib.spice',
                  pdk_name=catalog.split('_device')[0])
    m = Module(pdk.pdk_name, str(tmp_path), 'all.cir', pdk_details=pdk)
    with m:
        for i, dev in enumerate(pdk.model_names):
            cls = getattr(pdk, dev)
            cls(f'x{i}', **{pin: f'n{i}_{pin}' for pin in cls._pin_names})
    assert len(m.elem_dict) == len(pdk.model_names)
    assert _rule_violations(MissingParamRule(), m) == []


def test_struct007_fires_when_a_signal_port_sits_on_a_supply_net(pdk, tmp_path):
    inv = _inverter(pdk, tmp_path)
    top = _mod(pdk, tmp_path, 'top.cir')
    top.add_module(inv, 'u1', nets={'in': 'vdd', 'out': 'y', 'vdd': 'vdd', 'gnd': '0'})
    v = _only('STRUCT-007', _rule_violations(SignalOnSupplyNetRule(), top))
    assert [x.location.net_name for x in v] == ['vdd']
    assert v[0].message == "Signal port(s) on supply net 'vdd': u1.in"


def test_struct007_fires_when_a_signal_port_sits_on_a_bare_portdir_supply_net(pdk, tmp_path):
    """TASK-45: STRUCT-007 sees ports declared with bare PortDir.POWER/GROUND.

    The signal side must still carry a role (the inverter's ``in`` is
    DIGITAL); only the supply side is declared direction-only here."""
    inv = _inverter(pdk, tmp_path, supply_ports={'vdd': PortDir.POWER, 'gnd': PortDir.GROUND})
    top = _mod(pdk, tmp_path, 'top.cir')
    top.add_module(inv, 'u1', nets={'in': 'vdd', 'out': 'y', 'vdd': 'vdd', 'gnd': '0'})
    v = _only('STRUCT-007', _rule_violations(SignalOnSupplyNetRule(), top))
    assert [x.location.net_name for x in v] == ['vdd']
    assert v[0].message == "Signal port(s) on supply net 'vdd': u1.in"


def test_struct007_silent_when_signals_stay_off_the_rails(pdk, tmp_path):
    inv = _inverter(pdk, tmp_path)
    top = _mod(pdk, tmp_path, 'top.cir')
    top.add_module(inv, 'u1', nets={'in': 'a', 'out': 'y', 'vdd': 'vdd', 'gnd': '0'})
    top.add_module(inv, 'u2', nets={'in': 'y', 'out': 'z', 'vdd': 'vdd', 'gnd': '0'})
    assert _rule_violations(SignalOnSupplyNetRule(), top) == []


def test_struct008_fires_on_unwired_device_and_unwired_submodule(pdk, tmp_path):
    inv = _inverter(pdk, tmp_path)
    top = _mod(pdk, tmp_path, 'top.cir')
    top.add_module(inv, 'u1', nets={})
    with top:
        morph = pdk.nmos1p8('morph', d='m1', g='m2', s='m3', b='m4',
                            w=1.0, l=0.15)
    _unbind(morph, 'd', 'g', 's', 'b')          # every terminal unbound
    v = _only('STRUCT-008', _rule_violations(UnusedInstanceRule(), top))
    assert sorted(x.location.instance_path for x in v) == ['morph', 'u1']
    assert "Instance 'morph' has no terminal bindings" in {x.message for x in v}


def test_struct008_silent_when_any_terminal_is_bound(pdk, tmp_path):
    inv = _inverter(pdk, tmp_path, gate=None)          # gate floats but d/s/b bound
    top = _mod(pdk, tmp_path, 'top.cir')
    top.add_module(inv, 'u1', nets={'in': 'a'})        # one of four ports bound
    assert _rule_violations(UnusedInstanceRule(), top) == []
    assert _rule_violations(UnusedInstanceRule(), inv) == []


def test_bus002_fires_listing_the_unwired_bits(pdk, tmp_path):
    child = _mod(pdk, tmp_path, 'c.cir', 'child')
    child.define_module_ports({'clk': Port.clock_input(), 'd': Port.digital_output(bits=4)})
    parent = _mod(pdk, tmp_path, 'p.cir', 'parent')
    parent.add_module(child, 'u1', nets={'clk': 'clk', 'd[0]': 'n0', 'd[1]': 'n1'})
    v = _only('BUS-002', _rule_violations(BusPartialWiringRule(), parent))
    assert len(v) == 1
    assert v[0].location.instance_path == 'u1'
    assert "Bus 'd' on instance 'u1' has 2 unwired bit(s): d[2], d[3]" in v[0].message
    assert 'add_no_connect_port' in v[0].message


def test_bus002_silent_when_bus_fully_wired_or_marked(pdk, tmp_path):
    child = _mod(pdk, tmp_path, 'c.cir', 'child')
    child.define_module_ports({'clk': Port.clock_input(), 'd': Port.digital_output(bits=4)})
    parent = _mod(pdk, tmp_path, 'p.cir', 'parent')
    dbus = parent.add_net('d', bits=4)
    parent.add_module(child, 'u1', clk='clk', d=dbus)
    parent.add_module(child, 'u2', nets={'clk': 'clk', 'd[0]': 'x0', 'd[1]': 'x1'})
    parent.add_no_connect_port('u2', 'd[2]', reason='spare bit')
    parent.add_no_connect_port('u2', 'd[3]', reason='spare bit')
    assert _rule_violations(BusPartialWiringRule(), parent) == []


# ---------------------------------------------------------------------------
# STRUCT-009/010 — exist in structural.py, NOT in the default catalog
# ---------------------------------------------------------------------------

def test_struct009_fires_on_a_device_block_without_supply_ports(pdk, tmp_path):
    blk = _block(pdk, tmp_path, 'nosupply', {'out': (PortDir.OUT, PortRole.ANALOG),
                                             'in': (PortDir.IN, PortRole.ANALOG)})
    v = _only('STRUCT-009', _rule_violations(PowerGroundPortRequired(), blk))
    assert len(v) == 2
    msgs = ' '.join(x.message for x in v)
    assert "Subcircuit 'nosupply'" in msgs and 'POWER' in msgs and 'GROUND' in msgs


def test_struct009_silent_on_role_annotated_inverter(pdk, tmp_path):
    assert _rule_violations(PowerGroundPortRequired(), _inverter(pdk, tmp_path)) == []


def test_struct009_silent_on_bare_portdir_supply_ports(pdk, tmp_path):
    """F-02 / TASK-45 (was xfail). ``'vdd': PortDir.POWER`` / ``'gnd':
    PortDir.GROUND`` — the examples' idiom — carry the supply roles."""
    inv = _inverter(pdk, tmp_path, supply_ports={'vdd': PortDir.POWER,
                                                 'gnd': PortDir.GROUND})
    assert _rule_violations(PowerGroundPortRequired(), inv) == []


def test_struct010_fires_on_out_and_inout_supply_ports(pdk, tmp_path):
    inv = _inverter(pdk, tmp_path, supply_ports={'vdd': (PortDir.OUT, PortRole.POWER),
                                                 'gnd': (PortDir.INOUT, PortRole.GROUND)})
    v = _only('STRUCT-010', _rule_violations(PortDirectionSanity(), inv))
    got = {(x.location.port_name, x.message.split('direction ')[1].split()[0].rstrip(';'))
           for x in v}
    assert got == {('vdd', 'OUT'), ('gnd', 'INOUT')}


def test_struct010_silent_on_in_supply_ports(pdk, tmp_path):
    inv = _inverter(pdk, tmp_path, supply_ports={'vdd': (PortDir.IN, PortRole.POWER),
                                                 'gnd': (PortDir.IN, PortRole.GROUND)})
    assert _rule_violations(PortDirectionSanity(), inv) == []
    assert _rule_violations(PortDirectionSanity(), _inverter(pdk, tmp_path)) == []
