"""TASK-32 lint audit — marker + waiver ergonomics on real circuits.

The question each test answers: can a designer suppress EXACTLY ONE
finding (and no sibling), and does silk tell them when a suppression has
gone stale? The first half is proven here on real sky130 modules; the
second half checks the stale-marker reporting provided by ``LintReport.no_connect_ports`` and
``LintReport.stale_markers``.
"""
import numpy as np
import pytest

from silk.circuit.circuit_elements import PortDir, PortRole
from silk.circuit.modules import Module
from silk.circuit.report import generate_report
from silk.lint.registry import RuleRegistry
from silk.lint.rules.stress import JunctionOverrangeRule
from silk.lint.rules.structural import DanglingPortRule, FloatingNetRule
from silk.lint.waivers import Waiver, WaiverFile, WaiverMatcher, apply_waivers
from silk.lint_markers import NoConnect, NoConnectPort


def _mod(pdk, tmp_path, fname, name=None):
    return Module('PDK_skywater130', str(tmp_path), fname,
                  pdk_details=pdk, module_name=name)


def _inverter_with_enable(pdk, tmp_path):
    m = _mod(pdk, tmp_path, 'inv.cir', 'inv_en')
    m.define_module_ports({
        'in': (PortDir.IN, PortRole.DIGITAL),
        'en': (PortDir.IN, PortRole.DIGITAL),
        'out': (PortDir.OUT, PortRole.DIGITAL),
        'vdd': (PortDir.POWER, PortRole.POWER),
        'gnd': (PortDir.GROUND, PortRole.GROUND),
    })
    with m:
        pdk.nmos1p8('mn', d='out', g='in', s='gnd', b='gnd', w=1.0, l=0.15)
        pdk.nmos1p8('men', d='out', g='en', s='gnd', b='gnd', w=1.0, l=0.15)
        pdk.pmos1p8('mp', d='out', g='in', s='vdd', b='vdd', w=2.0, l=0.15)
    return m


def _top_with_two_dangling_ports(pdk, tmp_path):
    """u1.en and u1.out both unbound -> two STRUCT-001 findings on one instance."""
    top = _mod(pdk, tmp_path, 'top.cir')
    top.add_module(_inverter_with_enable(pdk, tmp_path), 'u1',
                    nets={'in': 'a', 'vdd': 'vdd', 'gnd': '0'})
    return top


# ---------------------------------------------------------------------------
# NoConnectPort — one pin, not the instance
# ---------------------------------------------------------------------------

def test_no_connect_port_suppresses_exactly_one_of_two_findings_on_one_instance(pdk, tmp_path):
    top = _top_with_two_dangling_ports(pdk, tmp_path)
    before = {v.location.port_name for v in DanglingPortRule().check(top)}
    assert before == {'en', 'out'}
    marker = top.add_no_connect_port('u1', 'en', reason='enable tied off at chip level')
    assert isinstance(marker, NoConnectPort)
    after = {v.location.port_name for v in DanglingPortRule().check(top)}
    assert after == {'out'}                       # sibling finding survives


def test_no_connect_port_is_scoped_to_its_instance(pdk, tmp_path):
    inv = _inverter_with_enable(pdk, tmp_path)
    top = _mod(pdk, tmp_path, 'top.cir')
    top.add_module(inv, 'u1', nets={'in': 'a', 'out': 'y', 'vdd': 'vdd', 'gnd': '0'})
    top.add_module(inv, 'u2', nets={'in': 'y', 'out': 'z', 'vdd': 'vdd', 'gnd': '0'})
    top.add_no_connect_port('u1', 'en', reason='u1 enable unused')
    left = {(v.location.module_name, v.location.port_name)
            for v in DanglingPortRule().check(top)}
    assert left == {('u2', 'en')}


def test_no_connect_port_rejects_typos_loudly(pdk, tmp_path):
    from silk.exceptions import CircuitError
    top = _top_with_two_dangling_ports(pdk, tmp_path)
    with pytest.raises(CircuitError, match="instance 'u9' not found"):
        top.add_no_connect_port('u9', 'en', reason='x')
    with pytest.raises(CircuitError, match="port 'enable' not declared"):
        top.add_no_connect_port('u1', 'enable', reason='x')
    with pytest.raises(ValueError, match='empty'):
        top.add_no_connect_port('u1', 'en', reason='   ')


def test_no_connect_port_constructor_validates_each_field():
    with pytest.raises(TypeError, match="'instance'"):
        NoConnectPort(instance='', port='en', reason='r')
    with pytest.raises(TypeError, match="'port'"):
        NoConnectPort(instance='u1', port='', reason='r')
    with pytest.raises(TypeError, match="'reason'"):
        NoConnectPort(instance='u1', port='en', reason=None)
    m = NoConnectPort(instance='u1', port='en', reason='tied off')
    assert m.is_lint_marker is True
    assert repr(m) == "NoConnectPort(instance='u1', port='en', reason='tied off')"


# ---------------------------------------------------------------------------
# NoConnect — one net, not the module
# ---------------------------------------------------------------------------

def test_no_connect_suppresses_exactly_one_of_two_floating_nets(pdk, tmp_path):
    inv = _inverter_with_enable(pdk, tmp_path)
    inv.add_net('probe_a')
    inv.add_net('probe_b')
    assert {v.location.net_name for v in FloatingNetRule().check(inv)} == {'probe_a', 'probe_b'}
    nc = inv.add_no_connect('probe_a', reason='test tap, probed at wafer level')
    assert isinstance(nc, NoConnect)
    assert {v.location.net_name for v in FloatingNetRule().check(inv)} == {'probe_b'}
    # The marker is a first-class binding on the net, never a hidden flag.
    binding = inv.net_dict['probe_a'].pins[0]
    assert (binding.kind, binding.port, binding.instance) == ('lint_marker', 'tie', '_nc_0')


def test_no_connect_survives_the_report_and_json_round_trip(pdk, tmp_path):
    inv = _inverter_with_enable(pdk, tmp_path)
    inv.add_net('probe_a')
    inv.add_no_connect('probe_a', reason='test tap')
    rep = generate_report(inv, auto_discover_config=False)
    assert [(i, n, nc.reason) for i, n, nc in rep.lint.no_connects] == \
        [('_nc_0', 'probe_a', 'test tap')]
    assert not any(v.rule_id == 'STRUCT-003' for v in rep.lint.violations)
    p = tmp_path / 'lint.json'
    rep.lint.save_json(p)
    back = rep.lint.load_json(p)
    assert back.no_connects[0][1:2] == ('probe_a',)
    assert back.no_connects[0][2].reason == 'test tap'
    html = tmp_path / 'lint.html'
    rep.lint.save_html(html)
    assert 'Intentionally Unconnected' in html.read_text() and 'probe_a' in html.read_text()


# ---------------------------------------------------------------------------
# Waivers — per-finding precision through the real matcher
# ---------------------------------------------------------------------------

def test_port_waiver_suppresses_one_finding_and_leaves_its_sibling(pdk, tmp_path):
    top = _top_with_two_dangling_ports(pdk, tmp_path)
    reg = RuleRegistry(rules=[DanglingPortRule()])
    w = Waiver(rule_id='STRUCT-001',
               matcher=WaiverMatcher('port', {'port_name': 'en', 'module_name': 'u1'}),
               reason='enable driven from the pad ring')
    unsup, sup = apply_waivers(reg.run(top).violations, [w])
    assert [v.location.port_name for v in sup] == ['en']
    assert [v.location.port_name for v in unsup] == ['out']


def test_waiver_from_yaml_is_as_precise_as_the_object_form(pdk, tmp_path):
    top = _top_with_two_dangling_ports(pdk, tmp_path)
    y = tmp_path / 'silk_waivers.yaml'
    y.write_text(
        "waivers:\n"
        "  - rule_id: STRUCT-001\n"
        "    matcher:\n"
        "      location_type: port\n"
        "      fields: {port_name: en, module_name: u1}\n"
        "    reason: enable driven from the pad ring\n"
        "    author: audit\n"
        "    expires: 2099-01-01\n"
    )
    reg = RuleRegistry(rules=[DanglingPortRule()])
    reg.load_waivers(y)
    rep = generate_report(top, registry=reg, auto_discover_config=False)
    assert [v.location.port_name for v in rep.lint.waivers_applied] == ['en']
    assert [v.location.port_name for v in rep.lint.violations] == ['out']
    assert rep.lint.summary()['waived'] == 1
    assert rep.lint.compliance_matrix() == {'STRUCT-001': {'violations': 1, 'waivers': 1}}


def test_waiver_on_the_wrong_instance_or_rule_does_not_match(pdk, tmp_path):
    top = _top_with_two_dangling_ports(pdk, tmp_path)
    reg = RuleRegistry(rules=[DanglingPortRule()])
    vs = reg.run(top).violations
    wrong_instance = Waiver('STRUCT-001', WaiverMatcher('port', {'port_name': 'en',
                                                                 'module_name': 'u2'}), 'x')
    wrong_rule = Waiver('STRUCT-002', WaiverMatcher('port', {'port_name': 'en',
                                                             'module_name': 'u1'}), 'x')
    wrong_type = Waiver('STRUCT-001', WaiverMatcher('instance', {'instance_path': 'u1'}), 'x')
    for w in (wrong_instance, wrong_rule, wrong_type):
        unsup, sup = apply_waivers(vs, [w])
        assert sup == [] and len(unsup) == 2


def test_terminal_waiver_targets_one_device_and_one_pair(pdk, tmp_path):
    """Two overstressed devices, four STR-001 findings; waive mn.Vgs only."""
    inv = _inverter_with_enable(pdk, tmp_path)
    n = 20
    inv._behavioral_mode = True
    inv._behavioral_t = np.linspace(0, 1e-9, n)
    for name, val in (('in', 2.5), ('en', 2.5), ('out', 0.0), ('vdd', 1.8), ('gnd', 0.0)):
        inv.net_dict[name].voltage = np.full(n, val)
    vs = list(JunctionOverrangeRule().check(inv))
    keys = {(v.location.instance_path, v.location.terminal) for v in vs}
    assert {('mn', 'Vgs'), ('mn', 'Vgd'), ('men', 'Vgs'), ('men', 'Vgd')} <= keys
    w = Waiver('STR-001', WaiverMatcher('terminal', {'instance_path': 'mn', 'terminal': 'Vgs'}),
               reason='gate is clamped by the ESD diode not modelled here')
    unsup, sup = apply_waivers(vs, [w])
    assert [(v.location.instance_path, v.location.terminal) for v in sup] == [('mn', 'Vgs')]
    assert ('men', 'Vgs') in {(v.location.instance_path, v.location.terminal) for v in unsup}


def test_expired_waiver_is_ignored_and_reason_is_mandatory(pdk, tmp_path):
    top = _top_with_two_dangling_ports(pdk, tmp_path)
    vs = list(DanglingPortRule().check(top))
    from datetime import date
    stale = Waiver('STRUCT-001', WaiverMatcher('port', {'port_name': 'en', 'module_name': 'u1'}),
                   reason='was temporary', expires=date(2020, 1, 1))
    unsup, sup = apply_waivers(vs, [stale])
    assert sup == [] and len(unsup) == 2
    with pytest.raises(ValueError, match='reason'):
        Waiver('STRUCT-001', WaiverMatcher('port', {}), reason='')
    y = tmp_path / 'w.yaml'
    y.write_text("waivers:\n  - rule_id: STRUCT-001\n    matcher: {location_type: port}\n")
    with pytest.raises(ValueError, match="missing required 'reason'"):
        WaiverFile.load(y)


# ---------------------------------------------------------------------------
# Marker lifecycle (F-09 / TASK-51): every marker reaches the report, and a
# marker whose target got wired after all is surfaced as stale
# ---------------------------------------------------------------------------

def test_no_connect_port_appears_in_the_lint_report(pdk, tmp_path):
    """Pre-fix: LintReport.to_dict()['no_connects'] listed NoConnect nets only,
    so a NoConnectPort suppression was invisible in the audit artifact even
    though its docstring promised it "surfaces in the lint report"."""
    top = _top_with_two_dangling_ports(pdk, tmp_path)
    top.add_no_connect_port('u1', 'en', reason='enable tied off at chip level')
    rep = generate_report(top, auto_discover_config=False)
    d = rep.lint.to_dict()
    assert any(e.get('instance_name') == 'u1' and e.get('port') == 'en'
               for e in d['no_connects'])
    assert [(m.instance, m.port, m.reason) for m in rep.lint.no_connect_ports] == \
        [('u1', 'en', 'enable tied off at chip level')]
    assert rep.lint.no_connects == []                  # the net list is untouched


def test_no_connect_port_survives_json_and_html(pdk, tmp_path):
    """Pre-fix: the port marker was dropped by to_dict, so load_json came back
    with zero markers and the HTML had no 'Intentionally Unconnected' table."""
    top = _top_with_two_dangling_ports(pdk, tmp_path)
    top.add_no_connect_port('u1', 'en', reason='enable tied off at chip level')
    rep = generate_report(top, auto_discover_config=False)
    p = tmp_path / 'lint.json'
    rep.lint.save_json(p)
    back = rep.lint.load_json(p)
    assert isinstance(back.no_connect_ports[0], NoConnectPort)
    assert (back.no_connect_ports[0].instance, back.no_connect_ports[0].port,
            back.no_connect_ports[0].reason) == ('u1', 'en', 'enable tied off at chip level')
    assert back.to_dict()['no_connects'] == rep.lint.to_dict()['no_connects']
    html = tmp_path / 'lint.html'
    rep.lint.save_html(html)
    text = html.read_text()
    assert 'Intentionally Unconnected' in text
    assert '<td>u1</td><td></td><td>en</td><td>enable tied off at chip level</td>' in text


def test_pre_task_51_json_without_kind_still_loads_as_net_markers(pdk, tmp_path):
    """Files written before the 'kind' key existed only ever held nets."""
    inv = _inverter_with_enable(pdk, tmp_path)
    inv.add_net('probe_a')
    inv.add_no_connect('probe_a', reason='test tap')
    d = generate_report(inv, auto_discover_config=False).lint.to_dict()
    for e in d['no_connects']:
        e.pop('kind')
    d.pop('stale_markers')
    from silk.lint import LintReport
    back = LintReport.from_dict(d)
    assert [(i, n, nc.reason) for i, n, nc in back.no_connects] == \
        [('_nc_0', 'probe_a', 'test tap')]
    assert back.no_connect_ports == [] and back.stale_markers == []


def test_stale_markers_surface_in_the_report(pdk, tmp_path):
    """Pre-fix: a NoConnect whose net later became a real node, or a
    NoConnectPort whose pin later got wired, was silently kept — nothing in
    the report surfaced the stale suppression (LintReport had no such field).

    Scenario adjusted from the audit's xfail: STRUCT-011 (TASK-35, merged
    after the audit) honours NoConnect on a ONE-pin net by design, so the
    tap has to become a real node (two real pins) before the marker is
    stale — see test_no_connect_on_a_one_pin_net_is_not_stale."""
    inv = _inverter_with_enable(pdk, tmp_path)
    inv.add_net('probe_a')
    inv.add_no_connect('probe_a', reason='test tap')
    with inv:                                   # the tap becomes a real node
        pdk.nmos1p8('mtap', d='probe_a', g='in', s='gnd', b='gnd', w=1.0, l=0.15)
        pdk.nmos1p8('mload', d='vdd', g='probe_a', s='gnd', b='gnd', w=1.0, l=0.15)
    top = _mod(pdk, tmp_path, 'top.cir')
    top.add_module(inv, 'u1', nets={'in': 'a', 'vdd': 'vdd', 'gnd': '0'})
    top.add_no_connect_port('u1', 'en', reason='enable unused')
    top.elem_dict['u1']['device'].pin_list['en'].net = 'en_wired'   # later wired
    stale_nets = generate_report(inv, auto_discover_config=False).lint.stale_markers
    stale_ports = generate_report(top, auto_discover_config=False).lint.stale_markers
    assert [m.net_name for m in stale_nets] == ['probe_a']
    assert stale_nets[0].wired_to == ('mtap.d', 'mload.g')
    assert [(m.instance, m.port) for m in stale_ports] == [('u1', 'en')]
    assert stale_ports[0].wired_to == ('en_wired',)
    assert [m.target() for m in stale_nets + stale_ports] == ['net probe_a', 'port u1.en']


def test_no_connect_on_a_one_pin_net_is_not_stale(pdk, tmp_path):
    """A NoConnect with exactly one real pin is the documented STRUCT-011
    suppression (a probe point / test tap), not a stale marker. Reporting it
    would flag every legitimate one-ended net the day the rule shipped."""
    from silk.lint.rules.structural import SingleConnectionNetRule
    inv = _inverter_with_enable(pdk, tmp_path)
    inv.add_net('probe_a')
    inv.add_no_connect('probe_a', reason='test tap')
    with inv:
        pdk.nmos1p8('mtap', d='probe_a', g='in', s='gnd', b='gnd', w=1.0, l=0.15)
    rep = generate_report(inv, auto_discover_config=False)
    assert rep.lint.stale_markers == []
    assert rep.lint.summary()['stale_markers'] == 0
    # ... and without the marker STRUCT-011 would indeed report the net.
    inv._no_connects.clear()
    inv.net_dict['probe_a'].pins[:] = [b for b in inv.net_dict['probe_a'].pins
                                       if b.kind != 'lint_marker']
    assert [v.location.net_name for v in SingleConnectionNetRule().check(inv)] == ['probe_a']


def test_stale_net_marker_is_reported_once_not_once_per_rule(pdk, tmp_path):
    """STRUCT-003 and STRUCT-011 both honour NoConnect; the stale row must
    come from one pass, not one per honouring rule."""
    inv = _inverter_with_enable(pdk, tmp_path)
    inv.add_net('probe_a')
    inv.add_no_connect('probe_a', reason='test tap')
    with inv:
        pdk.nmos1p8('mtap', d='probe_a', g='in', s='gnd', b='gnd', w=1.0, l=0.15)
        pdk.nmos1p8('mload', d='vdd', g='probe_a', s='gnd', b='gnd', w=1.0, l=0.15)
    reg = RuleRegistry.default()
    assert {'STRUCT-003', 'STRUCT-011'} <= {r.rule_id for r in reg.rules}
    rep = generate_report(inv, registry=reg, auto_discover_config=False)
    assert len(rep.lint.stale_markers) == 1
    assert rep.lint.summary()['stale_markers'] == 1
    # A stale marker is a hygiene row, not a violation: the wired net is fine.
    assert not any(v.location.describe().endswith('probe_a') for v in rep.lint.violations
                   if v.rule_id in ('STRUCT-003', 'STRUCT-011'))


def test_stale_markers_survive_json_and_render_as_warning_rows(pdk, tmp_path):
    top = _mod(pdk, tmp_path, 'top.cir')
    top.add_module(_inverter_with_enable(pdk, tmp_path), 'u1',
                    nets={'in': 'a', 'vdd': 'vdd', 'gnd': '0'})
    top.add_no_connect_port('u1', 'en', reason='enable unused')
    top.elem_dict['u1']['device'].pin_list['en'].net = 'en_wired'
    rep = generate_report(top, auto_discover_config=False)
    p = tmp_path / 'lint.json'
    rep.lint.save_json(p)
    back = rep.lint.load_json(p)
    assert back.stale_markers == rep.lint.stale_markers
    assert back.to_dict()['stale_markers'] == [{
        'kind': 'port', 'reason': 'enable unused', 'net_name': None,
        'instance': 'u1', 'port': 'en', 'wired_to': ['en_wired']}]
    html = tmp_path / 'lint.html'
    rep.lint.save_html(html)
    text = html.read_text()
    assert 'Stale Markers' in text
    assert '<tr class="warning"><td>port u1.en</td><td>en_wired</td><td>enable unused</td></tr>' in text
    assert '1 stale marker(s)' in text


# ---------------------------------------------------------------------------
# Marker registration dedupe (TASK-89): a second marker on the same target
# used to always append, doubling both the 'Intentionally Unconnected' row
# and — once TASK-51 landed — the stale-marker row for that target.
# ---------------------------------------------------------------------------

def test_no_connect_port_reregistered_with_same_reason_is_a_noop(pdk, tmp_path):
    """Pre-fix: two add_no_connect_port('u1', 'en', ...) calls with the same
    reason appended two NoConnectPort entries to Module._no_connect_ports
    (len == 2) instead of leaving the first marker alone."""
    top = _top_with_two_dangling_ports(pdk, tmp_path)
    first = top.add_no_connect_port('u1', 'en', reason='enable tied off at chip level')
    second = top.add_no_connect_port('u1', 'en', reason='enable tied off at chip level')
    assert second is first
    assert len(top._no_connect_ports) == 1


def test_no_connect_port_reregistered_with_different_reason_raises(pdk, tmp_path):
    """Pre-fix: a conflicting second reason on the same (instance, port) was
    accepted silently and both markers were kept, rather than raising so the
    caller notices the conflicting documentation."""
    from silk.exceptions import DuplicateNameError
    top = _top_with_two_dangling_ports(pdk, tmp_path)
    top.add_no_connect_port('u1', 'en', reason='enable tied off at chip level')
    with pytest.raises(DuplicateNameError,
                       match=r"enable tied off at chip level.*driven by testbench only"):
        top.add_no_connect_port('u1', 'en', reason='driven by testbench only')
    assert len(top._no_connect_ports) == 1     # the conflicting call left no trace


def test_no_connect_reregistered_with_same_reason_is_a_noop(pdk, tmp_path):
    """Pre-fix: two add_no_connect('probe_a', ...) calls with the same reason
    appended two entries to Module._no_connects (len == 2), each with its
    own '_nc_N' PortBinding on the net."""
    inv = _inverter_with_enable(pdk, tmp_path)
    inv.add_net('probe_a')
    first = inv.add_no_connect('probe_a', reason='test tap')
    second = inv.add_no_connect('probe_a', reason='test tap')
    assert second is first
    assert len(inv._no_connects) == 1
    markers = [b for b in inv.net_dict['probe_a'].pins if b.kind == 'lint_marker']
    assert len(markers) == 1


def test_no_connect_reregistered_with_different_reason_raises(pdk, tmp_path):
    """Pre-fix: a conflicting second reason on the same net was accepted
    silently and both markers were kept."""
    from silk.exceptions import DuplicateNameError
    inv = _inverter_with_enable(pdk, tmp_path)
    inv.add_net('probe_a')
    inv.add_no_connect('probe_a', reason='test tap')
    with pytest.raises(DuplicateNameError, match=r"test tap.*differential probe"):
        inv.add_no_connect('probe_a', reason='differential probe')
    assert len(inv._no_connects) == 1


def test_no_connect_port_whole_bus_reregistration_dedupes_per_bit(pdk, tmp_path):
    """Pre-fix: calling the whole-bus form twice fanned out 4 NEW markers
    each time (len == 8 after two calls) instead of recognizing each bit
    was already covered."""
    from silk.circuit.modules import Module, ModulePin
    from silk.circuit.circuit_elements import PortDir, PortRole
    from silk.circuit.bus_spec import BusSpec
    parent = _mod(pdk, tmp_path, 'p.cir', 'parent')
    child = Module(pdk_name='PDK_skywater130', dir_folder=str(tmp_path),
                   netlistname='c.cir', pdk_details=pdk, module_name='child')
    child.pin_list = {}
    for i in range(4):
        child.pin_list[f'dbg[{i}]'] = ModulePin(
            name=f'dbg[{i}]', direction=PortDir.OUT, role=PortRole.DIGITAL)
    child._buses = {'dbg': BusSpec(name='dbg', indices=(0, 1, 2, 3),
                                   direction=PortDir.OUT, role=PortRole.DIGITAL,
                                   domain=None, naming='bracketed')}
    parent._element_mgr.elem_dict['u1'] = {'device': child, 'type': 'module'}

    first = parent.add_no_connect_port('u1', 'dbg', reason='wired in TB only')
    second = parent.add_no_connect_port('u1', 'dbg', reason='wired in TB only')
    assert [m is n for m, n in zip(first, second)] == [True, True, True, True]
    assert len(parent._no_connect_ports) == 4

    from silk.exceptions import DuplicateNameError
    with pytest.raises(DuplicateNameError, match=r"wired in TB only.*spare bit"):
        parent.add_no_connect_port('u1', 'dbg[2]', reason='spare bit')
    assert len(parent._no_connect_ports) == 4  # conflicting bit-level call left no trace


def test_wired_target_with_reregistered_marker_yields_one_stale_row(pdk, tmp_path):
    """The TASK-89 fix must compose with TASK-51: a marker re-registered with
    the same reason (a no-op) still yields exactly ONE stale row once its
    target is wired, not two. Pre-fix, the second add_no_connect_port call
    would have appended a second NoConnectPort and find_stale_markers (which
    is deliberately per-marker) would have reported the same pin twice."""
    top = _mod(pdk, tmp_path, 'top.cir')
    top.add_module(_inverter_with_enable(pdk, tmp_path), 'u1',
                    nets={'in': 'a', 'vdd': 'vdd', 'gnd': '0'})
    top.add_no_connect_port('u1', 'en', reason='enable unused')
    top.add_no_connect_port('u1', 'en', reason='enable unused')   # re-registered, no-op
    top.elem_dict['u1']['device'].pin_list['en'].net = 'en_wired'
    stale = generate_report(top, auto_discover_config=False).lint.stale_markers
    assert [(m.instance, m.port) for m in stale] == [('u1', 'en')]
