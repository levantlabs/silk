"""TASK-93 — STRUCT-013 HIDDEN_SUPPLY_NET through the default registry.

The owner ruling of 2026-09-07 split the STRUCT-009 promotion in two. The
*defect* STRUCT-009 was written for — a device-bearing subcircuit wiring
transistors to a supply net it never declares as a port, which simulates
dead with no diagnostic (ngspice-42 on the S9 inverter: ``v(vout) = 0.0``,
no error) — moved to STRUCT-013, which keys on the WIRING and is therefore
in ``DEFAULT_CATALOG``. STRUCT-009 keeps its published meaning ("declare a
POWER and a GROUND port") and stays opt-in, because nine legitimate shipped
blocks have no such port on purpose (TASK-38 gate; TASK-94 owns STRUCT-010).

Every cell here runs through ``RuleRegistry.default()`` — the path
``generate_report`` / ``interface_check`` take — on real sky130 fixtures.

The silent cells reproduce the SHAPE of each of the nine offenders the
TASK-38 gate found, built here rather than imported from ``silk/examples``
so the guarantee survives an example being rewritten:

===========================  =======================================
offender (TASK-38 gate)      shape reproduced here
===========================  =======================================
bandgap banba_branches       BJT + resistor-submodule branch, no VDD
bandgap pmos_ibank_simple    PMOS-only current bank, no GND
cml_cells cml_bias           diode-connected NMOS reference
flash ref_ladder             resistor/cap ladder, no MOSFET at all
sar cap_dac_4b/6b/8b/10b     cap plate + two NMOS switch sources
constraints external_demo    ports declared, devices on a local rail
===========================  =======================================

The last one is the exception, and it is deliberate: ``external_demo``'s two
NMOS put source AND bulk on an internal ``vdd_int`` that is exported
nowhere, so it is a true positive, not a supply-less-by-design block. The
owner ruled it (and ``constraints_current_mirror``) gets filed as an example
fix rather than excused by the rule; :func:`test_fires_on_the_external_demo
_shape_ports_declared_but_devices_on_a_local_rail` pins that reading.
"""
import pytest

from silk.circuit.circuit_elements import Port, PortDir, PortRole
from silk.circuit.modules import Module
from silk.elements import elements as el
from silk.lint.registry import RuleRegistry
from silk.lint.rules import DEFAULT_CATALOG, OPTIONAL_CATALOG

_SIGNAL_PORTS = {'vin': Port.analog_input(), 'vout': Port.analog_output()}
_SUPPLY_PORTS = {'vdd': Port.supply(), 'gnd': Port.ground()}


def _module(pdk, tmp_path, name, ports=None):
    m = Module('PDK_skywater130', str(tmp_path), f'{name}.cir',
               pdk_details=pdk, module_name=name)
    if ports is not None:
        m.define_module_ports(ports)
    return m


def _lint(module):
    """rule_id -> [messages] from the DEFAULT registry."""
    out = {}
    for v in RuleRegistry.default().run(module).violations:
        out.setdefault(v.rule_id, []).append(v.message)
    return out


def _inverter(pdk, tmp_path, name, ports):
    """A real sky130 CMOS inverter whose body always wires the PMOS to
    ``vdd`` and the NMOS to ``gnd``, so a port block omitting one of them
    leaves that rail as a hidden local supply."""
    m = _module(pdk, tmp_path, name, ports)
    with m:
        pdk.nmos1p8('mn', d='vout', g='vin', s='gnd', b='gnd', w=1, l=0.15)
        pdk.pmos1p8('mp', d='vout', g='vin', s='vdd', b='vdd', w=2, l=0.15)
    return m


# ---------------------------------------------------------------------------
# Catalog membership — STRUCT-013 is default, STRUCT-009/010 are not
# ---------------------------------------------------------------------------

def test_struct013_is_in_the_default_catalog():
    assert 'STRUCT-013' in {r.rule_id for r in DEFAULT_CATALOG()}
    assert 'STRUCT-013' not in {r.rule_id for r in OPTIONAL_CATALOG()}


def test_default_registry_dispatches_struct013(pdk, tmp_path):
    inv = _inverter(pdk, tmp_path, 'inv', {**_SIGNAL_PORTS, **_SUPPLY_PORTS})
    visited = {e.rule_id for e in RuleRegistry.default().run(inv).visit_log}
    assert 'STRUCT-013' in visited


# ---------------------------------------------------------------------------
# Fires — the defect
# ---------------------------------------------------------------------------

def test_fires_on_the_hidden_local_supply_fixture(pdk, tmp_path):
    """Matrix cell S9's shape: the body wires the PMOS to ``vdd`` but the
    port block has no vdd, so ``.subckt leaky vin vout gnd`` carries an
    internal floating supply (ngspice-42: v(vout)=0.0, no error). This is
    the one finding the whole promotion exists for."""
    leaky = _inverter(pdk, tmp_path, 'leaky',
                      {**_SIGNAL_PORTS, 'gnd': Port.ground()})
    assert _lint(leaky)['STRUCT-013'] == [
        "Subcircuit 'leaky' wires mp.s, mp.b to net 'vdd', which is not a "
        "declared port — hidden local supply"]


def test_fires_once_per_hidden_rail(pdk, tmp_path):
    """A block with no port declarations at all (the shape of the four
    example amps in TASK-95) hides BOTH rails; each is one finding."""
    bare = _inverter(pdk, tmp_path, 'bare', None)   # no port block at all
    assert sorted(_lint(bare)['STRUCT-013']) == [
        "Subcircuit 'bare' wires mn.s, mn.b to net 'gnd', which is not a "
        "declared port — hidden local supply",
        "Subcircuit 'bare' wires mp.s, mp.b to net 'vdd', which is not a "
        "declared port — hidden local supply"]


def test_fires_when_the_rail_is_shared_by_several_devices(pdk, tmp_path):
    """The diff-pair shape (layout_intelligence diff_amp/diff_amp_v2):
    two input devices' bulks plus the tail device's source and bulk on one
    undeclared ``vss``. The finding names every terminal on the rail."""
    amp = _module(pdk, tmp_path, 'diffamp', {
        'vin_p': Port.analog_input(), 'vin_n': Port.analog_input(),
        'vout_p': Port.analog_output(), 'vout_n': Port.analog_output()})
    with amp:
        pdk.nmos1p8('mn1', d='vout_p', g='vin_p', s='tail', b='vss', w=4, l=0.18)
        pdk.nmos1p8('mn2', d='vout_n', g='vin_n', s='tail', b='vss', w=4, l=0.18)
        pdk.nmos1p8('mn_tail', d='tail', g='tail', s='vss', b='vss', w=8, l=0.18)
    findings = _lint(amp)['STRUCT-013']
    assert findings == [
        "Subcircuit 'diffamp' wires mn1.b, mn2.b, mn_tail.s, mn_tail.b to "
        "net 'vss', which is not a declared port — hidden local supply"]
    # 'tail' carries a drain and a gate, so it is a node with a job, not a
    # rail — the rule must not report it.
    assert "net 'tail'" not in findings[0]


def test_fires_on_the_external_demo_shape_ports_declared_but_devices_on_a_local_rail(
        pdk, tmp_path):
    """The ninth TASK-38 offender, and the one the ruling did NOT excuse.

    ``constraints_external_load/external_demo`` declares VDD/VSS ports and
    then wires both NMOS source AND bulk to an internal ``vdd_int`` — the
    declared supplies reach no device and the rail floats. Old STRUCT-009
    went quiet here as soon as the demo's JSON assigned the port roles;
    STRUCT-013 reports it either way, because the wiring is the defect.
    Filed as an example fix, not waived here."""
    def _shell(name, vdd, vss):
        shell = _module(pdk, tmp_path, name, {
            'VDD': vdd, 'VSS': vss, 'CLK': PortDir.IN,
            'VINP': PortDir.IN, 'VOUT': PortDir.OUT})
        shell.add_net('vdd_int')
        for inst in ('M1', 'M2'):
            shell.add(pdk.nmos1p8(inst, d='VOUT', g='CLK', s='vdd_int', b='vdd_int', w=1.0, l=0.15))
        return shell

    bare = _shell('external_demo', PortDir.IN, PortDir.IN)
    assert _lint(bare)['STRUCT-013'] == [
        "Subcircuit 'external_demo' wires M1.s, M1.b, M2.s, M2.b to net "
        "'vdd_int', which is not a declared port — hidden local supply"]

    # Same block with the roles the demo's JSON overlay assigns: still a
    # hidden rail, still one finding — where old STRUCT-009 fell silent.
    roled = _shell('external_demo_roled', Port.supply(), Port.ground())
    assert len(_lint(roled)['STRUCT-013']) == 1


# ---------------------------------------------------------------------------
# Silent — the nine supply-less-by-design offenders, one cell per shape
# ---------------------------------------------------------------------------

def test_silent_on_a_pmos_only_current_bank(pdk, tmp_path):
    """bandgap_example/pmos_ibank_simple: three matched PMOS sources with a
    VDD port and no ground at all. Old STRUCT-009 reported 'no GROUND port';
    every terminal here sits on a declared port, so there is nothing hidden."""
    bank = _module(pdk, tmp_path, 'pmos_ibank_simple', {
        'VDD': Port.supply(), 'VG': Port.analog_input(),
        'OUT1': Port.analog_output(), 'OUT2': Port.analog_output(),
        'OUT3': Port.analog_output()})
    with bank:
        for name, out in (('MP1', 'OUT1'), ('MP2', 'OUT2'), ('MP3', 'OUT3')):
            pdk.pmos1p8(name, d=out, g='VG', s='VDD', b='VDD', w=5, l=1.0)
    assert 'STRUCT-013' not in _lint(bank)


def test_silent_on_a_bjt_resistor_branch_with_no_vdd(pdk, tmp_path):
    """bandgap_example/banba_branches: diode-connected PNPs and resistor
    submodules fed by an injected current. Internal net 'C' carries a BJT
    emitter and a resistor-submodule pin, so it is a signal node, not a
    rail; the block has a VSS port and no VDD by design."""
    res = _module(pdk, tmp_path, 'res_ptat',
                  {'P': Port.analog_input(), 'N': Port.analog_input()})
    with res:
        el.resistor('R', p='P', n='N', value=10e3)

    m = _module(pdk, tmp_path, 'banba_branches', {
        'VSS': Port.ground(), 'A': Port.analog_input(),
        'B': Port.analog_input(), 'VREF': Port.analog_output()})
    m.add_net('C')
    with m:
        pdk.pnp_small('Q1', e='A', b='VSS', c='VSS', m=1)
        pdk.pnp_small('Q2', e='C', b='VSS', c='VSS', m=8)
    m.add_module(res, 'R1', nets={'P': 'B', 'N': 'C'})
    m.add_module(res, 'R0A', nets={'P': 'A', 'N': 'VSS'})
    m.add_module(res, 'ROUT', nets={'P': 'VREF', 'N': 'VSS'})
    assert 'STRUCT-013' not in _lint(m)


def test_silent_on_a_diode_connected_bias_reference(pdk, tmp_path):
    """cml_cells/cml_bias: one diode-connected NMOS between an IBIAS port
    and a VSS port. No POWER port exists and none is needed."""
    m = _module(pdk, tmp_path, 'cml_bias',
                {'VSS': Port.ground(), 'IBIAS': Port.analog_input()})
    with m:
        pdk.nmos1p8('MBIAS', d='IBIAS', g='IBIAS', s='VSS', b='VSS',
                    w=4, l=0.5)
    assert 'STRUCT-013' not in _lint(m)


def test_silent_on_a_resistor_cap_reference_ladder(pdk, tmp_path):
    """flash_adc_example/ref_ladder: a resistor string with a decoupling cap
    per tap. No MOSFET at all, every node a declared port."""
    ports = {'vref': PortDir.IN, 'vrefl': PortDir.IN,
             'vss': Port(dir=PortDir.IN, role=PortRole.GROUND)}
    ports.update({f'vref{k}': PortDir.OUT for k in (1, 2, 3)})
    m = _module(pdk, tmp_path, 'ref_ladder', ports)
    nodes = ['vref', 'vref3', 'vref2', 'vref1', 'vrefl']
    for i in range(4):
        m.add(el.resistor(f'r{i + 1}', p=nodes[i], n=nodes[i + 1], value=10e3))
    for k in (1, 2, 3):
        m.add(el.capacitor(f'c{k}', p=f'vref{k}', n='vss', value=500e-15))
    assert 'STRUCT-013' not in _lint(m)


def test_silent_on_a_cap_dac_bottom_plate(pdk, tmp_path):
    """sar_adc_spice/cap_dac_*: the bottom plate of each binary-weighted cap
    carries a cap terminal plus the SOURCES of the two switches that steer
    it between vrefp and vrefn. That is a legitimate charge-redistribution
    node, and it is exactly why the rule does not tolerate a capacitor
    terminal on a candidate rail: widening the predicate that far would
    report this block."""
    m = _module(pdk, tmp_path, 'cap_dac_2b', {
        'vss': Port(dir=PortDir.IN, role=PortRole.GROUND),
        'vrefp': Port.analog_input(), 'vrefn': Port.analog_input(),
        'toutp': Port.analog_output(),
        'b0': Port(dir=PortDir.IN, role=PortRole.DIGITAL),
        'b0b': Port(dir=PortDir.IN, role=PortRole.DIGITAL)})
    m.add(el.capacitor('cp0', p='toutp', n='pbot0', value=1e-15))
    m.add(pdk.nmos1p8('swpp0', d='vrefp', g='b0', s='pbot0', b='vss',
                      w=1, l=0.15))
    m.add(pdk.nmos1p8('swpn0', d='vrefn', g='b0b', s='pbot0', b='vss',
                      w=1, l=0.15))
    assert 'STRUCT-013' not in _lint(m)


# ---------------------------------------------------------------------------
# Silent — correct designs and the rule's boundaries
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('supply_ports', [
    pytest.param({'vdd': Port.supply(), 'gnd': Port.ground()}, id='Port.supply/ground'),
    pytest.param({'vdd': (PortDir.IN, PortRole.POWER), 'gnd': (PortDir.IN, PortRole.GROUND)},
                 id='explicit-role-tuples'),
    pytest.param({'vdd': PortDir.POWER, 'gnd': PortDir.GROUND}, id='bare-PortDir'),
    pytest.param({'vdd': 'power', 'gnd': 'ground'}, id='direction-strings'),
    pytest.param({'vdd': PortDir.IN, 'gnd': PortDir.IN}, id='no-role-at-all'),
])
def test_silent_when_both_rails_are_declared_ports(pdk, tmp_path, supply_ports):
    """The rule reads the port LIST, never the role, so every declaration
    idiom works — including a bare ``PortDir.IN`` that carries no supply
    role at all. Role-keyed enforcement is STRUCT-009's job."""
    inv = _inverter(pdk, tmp_path, 'inv', {**_SIGNAL_PORTS, **supply_ports})
    assert 'STRUCT-013' not in _lint(inv)


def test_silent_when_the_rail_is_the_global_ground_node(pdk, tmp_path):
    """Net '0' is the simulator's own global ground and pierces subcircuit
    boundaries, so it needs no port to be driven."""
    m = _module(pdk, tmp_path, 'to_zero',
                {'vin': Port.analog_input(), 'vout': Port.analog_output()})
    with m:
        pdk.nmos1p8('mn', d='vout', g='vin', s='0', b='0', w=1, l=0.15)
    assert 'STRUCT-013' not in _lint(m)


def test_silent_on_a_locally_generated_rail_that_something_drives(pdk, tmp_path):
    """Power gating: a header PMOS's DRAIN makes ``vdd_local``, which the
    gated inverter then uses. The rail is internal but generated in-block,
    and the drain on it is the proof — not a hidden supply."""
    m = _module(pdk, tmp_path, 'gated', {**_SIGNAL_PORTS, **_SUPPLY_PORTS,
                                         'en_b': Port.analog_input()})
    with m:
        pdk.pmos1p8('m_head', d='vdd_local', g='en_b', s='vdd', b='vdd',
                    w=20, l=0.15)
        pdk.pmos1p8('mp', d='vout', g='vin', s='vdd_local', b='vdd_local',
                    w=2, l=0.15)
        pdk.nmos1p8('mn', d='vout', g='vin', s='gnd', b='gnd', w=1, l=0.15)
    assert 'STRUCT-013' not in _lint(m)


def test_silent_on_a_one_pin_rail_which_belongs_to_struct011(pdk, tmp_path):
    """A single source/bulk terminal on an undeclared net is already
    STRUCT-011 SINGLE_CONNECTION_NET's finding; STRUCT-013 does not repeat
    it, so the report carries one diagnosis, not two."""
    m = _module(pdk, tmp_path, 'lone',
                {'vin': Port.analog_input(), 'vout': Port.analog_output(),
                 'gnd': Port.ground()})
    with m:
        pdk.nmos1p8('mn', d='vout', g='vin', s='gnd', b='floater', w=1, l=0.15)
    report = _lint(m)
    assert 'STRUCT-013' not in report
    assert any("floater" in msg for msg in report['STRUCT-011'])


def test_a_no_connect_marker_takes_the_net_out_of_scope_but_reads_stale(pdk, tmp_path):
    """``Module.add_no_connect`` is NOT the suppression idiom for this rule,
    and the report says so on its own.

    The marker's binding is not a device terminal, so the rail falls out of
    STRUCT-013's scope as a side effect — but NoConnect means "intentionally
    UNCONNECTED", and a hidden rail has two or more real bindings, so
    ``find_stale_markers`` immediately reports the marker as stale. Waive a
    real hidden rail with a waiver on the rule id and a net matcher, which
    stays visible in the report's ``waivers_applied`` section."""
    from silk.lint_markers.stale import find_stale_markers

    leaky = _inverter(pdk, tmp_path, 'marked',
                      {**_SIGNAL_PORTS, 'gnd': Port.ground()})
    leaky.add_no_connect('vdd', reason='rail wired by the parent, honest')
    assert 'STRUCT-013' not in _lint(leaky)
    stale = find_stale_markers(leaky)
    assert [(m.net_name, m.wired_to) for m in stale] == [
        ('vdd', ('mp.s', 'mp.b'))]


def test_silent_on_a_device_free_wrapper(pdk, tmp_path):
    """A behavioral wrapper with no devices has no terminals to hide."""
    wrap = _module(pdk, tmp_path, 'wrap', dict(_SIGNAL_PORTS))
    assert 'STRUCT-013' not in _lint(wrap)


def test_silent_on_a_top_level_module(pdk, tmp_path):
    """A testbench (no ``module_name``) is not a subcircuit: its supplies
    are sources, and it has no port block to declare them in."""
    top = Module('PDK_skywater130', str(tmp_path), 'top.cir', pdk_details=pdk)
    with top:
        pdk.nmos1p8('mn', d='out', g='in', s='vss', b='vss', w=1, l=0.15)
    assert 'STRUCT-013' not in _lint(top)
