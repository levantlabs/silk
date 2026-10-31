import pytest
from unittest.mock import MagicMock
from silk.circuit.circuit_elements import PortDir, PortRole
from silk.circuit.interface_check import (
    check_interfaces, InterfaceViolation,
    list_unannotated_ports, annotation_coverage, UnannotatedPort, AnnotationCoverage,
)
from silk.circuit.modules import Module
from silk.lint.severity import Severity


def _pdk():
    pdk = MagicMock()
    pdk.model_names = []
    pdk._devices = {}
    return pdk


def _sub(tmp_path, name, ports):
    """Build a bare subcircuit Module with given port definitions."""
    m = Module('TEST', str(tmp_path), f'{name}.cir', pdk_details=_pdk())
    m.define_module_name(name)
    m.define_module_ports(ports)
    return m


def test_clean_circuit_no_violations(tmp_path):
    comp = _sub(tmp_path, 'comp', {
        'vp':   PortDir.IN,
        'vm':   PortDir.IN,
        'vout': (PortDir.OUT, PortRole.DIGITAL),
    })
    enc = _sub(tmp_path, 'enc', {
        'vin':  (PortDir.IN, PortRole.DIGITAL),
        'dout': (PortDir.OUT, PortRole.DIGITAL),
    })
    comp.pin_list['vout'].net = 'vcmp'
    enc.pin_list['vin'].net   = 'vcmp'
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_submodule('comp1', comp)
    top.add_submodule('enc1', enc)

    violations = check_interfaces(top)
    assert violations == []


def test_driver_conflict(tmp_path):
    a = _sub(tmp_path, 'a', {'out': PortDir.OUT})
    b = _sub(tmp_path, 'b', {'out': PortDir.OUT})
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_module(a, 'a1', nets={'out': 'shared'})
    top.add_module(b, 'b1', nets={'out': 'shared'})

    violations = check_interfaces(top)
    assert len(violations) == 1
    assert violations[0].rule_id == 'INT-001'
    assert violations[0].severity == Severity.ERROR
    assert violations[0].net == 'shared'
    assert 'shared' in violations[0].message
    assert 'a1' in violations[0].message or 'b1' in violations[0].message


def test_power_ground_short(tmp_path):
    a = _sub(tmp_path, 'a', {'vdd': (PortDir.IN, PortRole.POWER)})
    b = _sub(tmp_path, 'b', {'gnd': (PortDir.IN, PortRole.GROUND)})
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_module(a, 'a1', nets={'vdd': 'shared_rail'})
    top.add_module(b, 'b1', nets={'gnd': 'shared_rail'})

    violations = check_interfaces(top)
    rule_ids = [v.rule_id for v in violations]
    assert 'INT-002' in rule_ids
    errors = [v for v in violations if v.severity == Severity.ERROR]
    assert len(errors) >= 1
    pg_violations = [v for v in violations if v.rule_id == 'INT-002']
    assert 'shared_rail' in pg_violations[0].message


def test_role_mismatch_warning(tmp_path):
    src = _sub(tmp_path, 'src', {'out': (PortDir.OUT, PortRole.DIGITAL)})
    dst = _sub(tmp_path, 'dst', {'inp': (PortDir.IN,  PortRole.ANALOG)})
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_module(src, 'src1', nets={'out': 'mixed_net'})
    top.add_module(dst, 'dst1', nets={'inp': 'mixed_net'})

    violations = check_interfaces(top)
    assert len(violations) == 1
    assert violations[0].rule_id == 'INT-003'
    assert violations[0].severity == Severity.ERROR
    assert 'mixed_net' in violations[0].message


def test_untyped_ports_no_violations(tmp_path):
    a = _sub(tmp_path, 'a', {'out': PortDir.OUT})
    b = _sub(tmp_path, 'b', {'inp': PortDir.IN})
    a.pin_list['out'].net = 'wire'
    b.pin_list['inp'].net = 'wire'
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_submodule('a1', a)
    top.add_submodule('b1', b)

    violations = check_interfaces(top)
    assert violations == []


def test_check_interfaces_returns_list(tmp_path):
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    result = check_interfaces(top)
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# list_unannotated_ports tests
# ---------------------------------------------------------------------------

def test_list_unannotated_ports_finds_bare_portdir(tmp_path):
    """Ports declared with bare PortDir (no role) appear in the unannotated list."""
    sub = _sub(tmp_path, 'comp', {
        'vp':   PortDir.IN,                        # no role
        'vm':   PortDir.IN,                        # no role
        'vout': (PortDir.OUT, PortRole.DIGITAL),   # has role — should NOT appear
    })
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_submodule('comp1', sub)

    result = list_unannotated_ports(top)
    ports = {u.port for u in result}
    assert 'vp'   in ports
    assert 'vm'   in ports
    assert 'vout' not in ports


def test_list_unannotated_ports_fields(tmp_path):
    """UnannotatedPort carries instance name, module name, port name, direction."""
    sub = _sub(tmp_path, 'myblock', {'vin': PortDir.IN})
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_submodule('blk1', sub)

    result = list_unannotated_ports(top)
    assert len(result) == 1
    u = result[0]
    assert isinstance(u, UnannotatedPort)
    assert u.instance    == 'blk1'
    assert u.module_name == 'myblock'
    assert u.port        == 'vin'
    assert u.direction   == 'input'


def test_list_unannotated_ports_empty_when_all_annotated(tmp_path):
    """Returns empty list when every port has a role."""
    sub = _sub(tmp_path, 'comp', {
        'vp':   (PortDir.IN,  PortRole.ANALOG),
        'vout': (PortDir.OUT, PortRole.DIGITAL),
    })
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_submodule('comp1', sub)

    assert list_unannotated_ports(top) == []


def test_list_unannotated_ports_empty_module(tmp_path):
    """Empty module_dict returns empty list."""
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    assert list_unannotated_ports(top) == []


# ---------------------------------------------------------------------------
# annotation_coverage tests
# ---------------------------------------------------------------------------

def test_annotation_coverage_counts(tmp_path):
    """Correctly counts annotated vs unannotated across multiple instances."""
    comp = _sub(tmp_path, 'comp', {
        'vp':   PortDir.IN,                        # unannotated
        'vm':   PortDir.IN,                        # unannotated
        'vout': (PortDir.OUT, PortRole.DIGITAL),   # annotated
    })
    enc = _sub(tmp_path, 'enc', {
        'vin':  (PortDir.IN,  PortRole.DIGITAL),   # annotated
        'dout': (PortDir.OUT, PortRole.DIGITAL),   # annotated
    })
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_submodule('comp1', comp)
    top.add_submodule('enc1', enc)

    cov = annotation_coverage(top)
    assert isinstance(cov, AnnotationCoverage)
    assert cov.total      == 5
    assert cov.annotated  == 3
    assert cov.unannotated == 2
    assert abs(cov.coverage_pct - 60.0) < 0.01


def test_annotation_coverage_str(tmp_path):
    """__str__ produces a readable summary string."""
    sub = _sub(tmp_path, 'blk', {'a': PortDir.IN, 'b': (PortDir.OUT, PortRole.ANALOG)})
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_submodule('blk1', sub)

    cov = annotation_coverage(top)
    s = str(cov)
    assert 'coverage' in s.lower()
    assert '50.0%' in s


def test_annotation_coverage_fully_annotated(tmp_path):
    """100% coverage when all ports have roles."""
    sub = _sub(tmp_path, 'blk', {'a': (PortDir.IN, PortRole.ANALOG)})
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_submodule('blk1', sub)

    cov = annotation_coverage(top)
    assert cov.coverage_pct == 100.0
    assert cov.unannotated_ports == []


def test_annotation_coverage_empty_module(tmp_path):
    """Empty module returns 100% coverage (nothing to annotate)."""
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    cov = annotation_coverage(top)
    assert cov.total == 0
    assert cov.coverage_pct == 100.0
    assert cov.unannotated_ports == []


def test_annotation_coverage_unannotated_ports_accessible(tmp_path):
    """unannotated_ports list is directly usable for programmatic inspection."""
    sub = _sub(tmp_path, 'comp', {'vp': PortDir.IN, 'vout': (PortDir.OUT, PortRole.DIGITAL)})
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_submodule('comp1', sub)

    cov = annotation_coverage(top)
    assert len(cov.unannotated_ports) == 1
    u = cov.unannotated_ports[0]
    assert u.port      == 'vp'
    assert u.instance  == 'comp1'
    assert u.direction == 'input'


# ---------------------------------------------------------------------------
# Port descriptor tests
# ---------------------------------------------------------------------------

from silk.circuit.voltage_domain import VoltageDomain
from silk.circuit.circuit_elements import Port


def test_port_descriptor_accepted_in_define_module_ports(tmp_path):
    avdd = VoltageDomain('avdd', nominal=1.8)
    sub = _sub(tmp_path, 'sub', {
        'vdd': Port(dir=PortDir.IN, role=PortRole.POWER, domain=avdd),
        'sig': Port(dir=PortDir.OUT, role=PortRole.ANALOG),
        'vin': PortDir.IN,   # bare PortDir still works
    })
    assert sub.pin_list['vdd'].domain is avdd
    assert sub.pin_list['sig'].domain is None
    assert sub.pin_list['vin'].domain is None


def test_two_tuple_still_works_after_port_descriptor_addition(tmp_path):
    sub = _sub(tmp_path, 'sub', {
        'vdd': (PortDir.IN, PortRole.POWER),
    })
    assert sub.pin_list['vdd'].role == 'power'
    assert sub.pin_list['vdd'].domain is None


def test_port_descriptor_no_role(tmp_path):
    avdd = VoltageDomain('avdd', nominal=1.8)
    sub = _sub(tmp_path, 'sub', {
        'vdd': Port(dir=PortDir.IN, domain=avdd),
    })
    assert sub.pin_list['vdd'].direction == 'input'
    assert sub.pin_list['vdd'].role is None
    assert sub.pin_list['vdd'].domain is avdd


# ---------------------------------------------------------------------------
# Module domain tracking tests (Task 3)
# ---------------------------------------------------------------------------

def test_assign_net_domain_stores_domain(tmp_path):
    avdd = VoltageDomain('avdd', nominal=1.8)
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.assign_net_domain('vdd', avdd)
    assert top.net_dict['vdd'].domain is avdd
    assert avdd in top._domains


def test_assign_net_domain_compatible_twice(tmp_path):
    a = VoltageDomain('avdd', nominal=1.8, min=1.62, max=1.98)
    b = VoltageDomain('avdd', nominal=1.8, min=1.60, max=2.00)
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.assign_net_domain('vdd', a)
    top.assign_net_domain('vdd', b)  # compatible — should not raise


def test_assign_net_domain_incompatible_raises(tmp_path):
    avdd = VoltageDomain('avdd', nominal=1.8, tolerance=0.1)
    dvdd = VoltageDomain('dvdd', nominal=3.3, tolerance=0.1)
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.assign_net_domain('vdd', avdd)
    with pytest.raises(ValueError, match="incompatible"):
        top.assign_net_domain('vdd', dvdd)


def test_no_driver_warning(tmp_path):
    a = _sub(tmp_path, 'a', {'sig': PortDir.IN})
    b = _sub(tmp_path, 'b', {'sig': PortDir.IN})
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_module(a, 'a1', nets={'sig': 'shared'})
    top.add_module(b, 'b1', nets={'sig': 'shared'})

    violations = check_interfaces(top)
    rule_ids = [v.rule_id for v in violations]
    assert 'INT-004' in rule_ids
    no_driver = next(v for v in violations if v.rule_id == 'INT-004')
    assert no_driver.severity == Severity.ERROR
    assert no_driver.net == 'shared'


def test_no_driver_not_fired_when_output_present(tmp_path):
    a = _sub(tmp_path, 'a', {'sig': PortDir.OUT})
    b = _sub(tmp_path, 'b', {'sig': PortDir.IN})
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_module(a, 'a1', nets={'sig': 'shared'})
    top.add_module(b, 'b1', nets={'sig': 'shared'})

    violations = check_interfaces(top)
    rule_ids = [v.rule_id for v in violations]
    assert 'INT-004' not in rule_ids


def test_clock_on_analog_warning(tmp_path):
    a = _sub(tmp_path, 'a', {'clk': (PortDir.IN, PortRole.CLOCK)})
    b = _sub(tmp_path, 'b', {'inp': (PortDir.IN, PortRole.ANALOG)})
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_module(a, 'a1', nets={'clk': 'shared'})
    top.add_module(b, 'b1', nets={'inp': 'shared'})

    violations = check_interfaces(top)
    rule_ids = [v.rule_id for v in violations]
    assert 'INT-005' in rule_ids
    coa = next(v for v in violations if v.rule_id == 'INT-005')
    assert coa.severity == Severity.ERROR
    assert coa.net == 'shared'


def test_clock_on_digital_no_violation(tmp_path):
    a = _sub(tmp_path, 'a', {'clk': (PortDir.OUT, PortRole.CLOCK)})
    b = _sub(tmp_path, 'b', {'clk': (PortDir.IN, PortRole.DIGITAL)})
    a.pin_list['clk'].net = 'clk_net'
    b.pin_list['clk'].net = 'clk_net'
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_submodule('a1', a)
    top.add_submodule('b1', b)

    violations = check_interfaces(top)
    rule_ids = [v.rule_id for v in violations]
    assert 'INT-005' not in rule_ids


# ---------------------------------------------------------------------------
# check_voltage_domains tests (Task 5)
# ---------------------------------------------------------------------------

from silk.circuit.interface_check import check_voltage_domains


def test_rail_mismatch_error(tmp_path):
    avdd = VoltageDomain('avdd', nominal=1.8)
    dvdd = VoltageDomain('dvdd', nominal=1.8)
    a = _sub(tmp_path, 'a', {'vdd': Port(dir=PortDir.IN, role=PortRole.POWER, domain=avdd)})
    b = _sub(tmp_path, 'b', {'vdd': Port(dir=PortDir.IN, role=PortRole.POWER, domain=dvdd)})
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_module(a, 'a1', nets={'vdd': 'vdd_net'})
    top.add_module(b, 'b1', nets={'vdd': 'vdd_net'})

    violations = check_voltage_domains(top)
    rule_ids = [v.rule_id for v in violations]
    assert 'DOM-001' in rule_ids
    rm = next(v for v in violations if v.rule_id == 'DOM-001')
    assert rm.severity == Severity.ERROR
    assert 'avdd' in rm.message
    assert 'dvdd' in rm.message


def test_rail_mismatch_same_domain_no_violation(tmp_path):
    avdd = VoltageDomain('avdd', nominal=1.8)
    a = _sub(tmp_path, 'a', {'vdd': Port(dir=PortDir.IN, role=PortRole.POWER, domain=avdd)})
    b = _sub(tmp_path, 'b', {'vdd': Port(dir=PortDir.IN, role=PortRole.POWER, domain=avdd)})
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_module(a, 'a1', nets={'vdd': 'vdd_net'})
    top.add_module(b, 'b1', nets={'vdd': 'vdd_net'})

    violations = check_voltage_domains(top)
    assert all(v.rule_id != 'DOM-001' for v in violations)


def test_domain_crossing_warning(tmp_path):
    avdd = VoltageDomain('avdd', nominal=1.8)
    dvdd = VoltageDomain('dvdd', nominal=1.8)
    a = _sub(tmp_path, 'a', {'sig': Port(dir=PortDir.OUT, role=PortRole.ANALOG, domain=avdd)})
    b = _sub(tmp_path, 'b', {'sig': Port(dir=PortDir.IN,  role=PortRole.DIGITAL, domain=dvdd)})
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_module(a, 'a1', nets={'sig': 'sig_net'})
    top.add_module(b, 'b1', nets={'sig': 'sig_net'})

    violations = check_voltage_domains(top)
    rule_ids = [v.rule_id for v in violations]
    assert 'DOM-002' in rule_ids
    dc = next(v for v in violations if v.rule_id == 'DOM-002')
    assert dc.severity == Severity.ERROR


def test_domain_crossing_no_violation_same_domain(tmp_path):
    avdd = VoltageDomain('avdd', nominal=1.8)
    a = _sub(tmp_path, 'a', {'sig': Port(dir=PortDir.OUT, role=PortRole.ANALOG, domain=avdd)})
    b = _sub(tmp_path, 'b', {'sig': Port(dir=PortDir.IN,  role=PortRole.ANALOG, domain=avdd)})
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_module(a, 'a1', nets={'sig': 'sig_net'})
    top.add_module(b, 'b1', nets={'sig': 'sig_net'})

    violations = check_voltage_domains(top)
    assert all(v.rule_id != 'DOM-002' for v in violations)


def test_supply_disabled_stress_warning(tmp_path):
    # DOM-003 keys on the rail that can be powered down (TASK-49).
    avdd  = VoltageDomain('avdd',  nominal=1.8, switchable=True)
    iovdd = VoltageDomain('iovdd', nominal=3.3)
    a = _sub(tmp_path, 'a', {'sig': Port(dir=PortDir.IN,  role=PortRole.ANALOG, domain=avdd)})
    b = _sub(tmp_path, 'b', {'sig': Port(dir=PortDir.OUT, role=PortRole.ANALOG, domain=iovdd)})
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_module(a, 'a1', nets={'sig': 'sig_net'})
    top.add_module(b, 'b1', nets={'sig': 'sig_net'})

    violations = check_voltage_domains(top)
    rule_ids = [v.rule_id for v in violations]
    assert 'DOM-003' in rule_ids
    sds = next(v for v in violations if v.rule_id == 'DOM-003')
    assert sds.severity == Severity.ERROR
    assert 'avdd' in sds.message
    assert 'iovdd' in sds.message


def test_supply_disabled_stress_silent_without_switchable_rail(tmp_path):
    """No rail declared switchable → DOM-002 only, no DOM-003 duplicate."""
    avdd  = VoltageDomain('avdd',  nominal=1.8)
    iovdd = VoltageDomain('iovdd', nominal=3.3)
    a = _sub(tmp_path, 'a', {'sig': Port(dir=PortDir.IN,  role=PortRole.ANALOG, domain=avdd)})
    b = _sub(tmp_path, 'b', {'sig': Port(dir=PortDir.OUT, role=PortRole.ANALOG, domain=iovdd)})
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_module(a, 'a1', nets={'sig': 'sig_net'})
    top.add_module(b, 'b1', nets={'sig': 'sig_net'})

    rule_ids = [v.rule_id for v in check_voltage_domains(top)]
    assert rule_ids.count('DOM-002') == 1
    assert 'DOM-003' not in rule_ids


def test_upward_domain_propagation(tmp_path):
    """Net domain inferred from child port when not explicitly assigned."""
    avdd = VoltageDomain('avdd', nominal=1.8)
    dvdd = VoltageDomain('dvdd', nominal=1.8)
    a = _sub(tmp_path, 'a', {'vdd': Port(dir=PortDir.IN, role=PortRole.POWER, domain=avdd)})
    b = _sub(tmp_path, 'b', {'vdd': Port(dir=PortDir.IN, role=PortRole.POWER, domain=dvdd)})
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_module(a, 'a1', nets={'vdd': 'vdd_net'})
    top.add_module(b, 'b1', nets={'vdd': 'vdd_net'})
    # No assign_net_domain call — domain must be inferred
    violations = check_voltage_domains(top)
    assert any(v.rule_id == 'DOM-001' for v in violations)


def test_explicit_net_domain_takes_precedence(tmp_path):
    avdd = VoltageDomain('avdd', nominal=1.8)
    a = _sub(tmp_path, 'a', {'vdd': Port(dir=PortDir.IN, role=PortRole.POWER, domain=avdd)})
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_module(a, 'a1', nets={'vdd': 'vdd_net'})
    top.assign_net_domain('vdd_net', avdd)  # explicit, same domain → no conflict
    violations = check_voltage_domains(top)
    assert all(v.rule_id != 'DOM-001' for v in violations)


# ---------------------------------------------------------------------------
# require_annotation_coverage tests (Task 6)
# ---------------------------------------------------------------------------

from silk.circuit.interface_check import require_annotation_coverage, AnnotationCoverageError


def test_require_annotation_coverage_passes(tmp_path):
    comp = _sub(tmp_path, 'comp', {
        'vdd': (PortDir.IN, PortRole.POWER),
        'gnd': (PortDir.IN, PortRole.GROUND),
        'vout': (PortDir.OUT, PortRole.ANALOG),
    })
    comp.pin_list['vdd'].net = 'vdd'
    comp.pin_list['gnd'].net = 'gnd'
    comp.pin_list['vout'].net = 'out'
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_submodule('comp1', comp)
    # 3/3 ports annotated = 100% coverage → should not raise
    require_annotation_coverage(top, min_pct=80.0)


def test_require_annotation_coverage_fails(tmp_path):
    comp = _sub(tmp_path, 'comp', {
        'vout': PortDir.OUT,   # no role
        'vin':  PortDir.IN,    # no role
    })
    comp.pin_list['vout'].net = 'out'
    comp.pin_list['vin'].net = 'inp'
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_submodule('comp1', comp)
    # 0% coverage → should raise
    with pytest.raises(AnnotationCoverageError) as exc_info:
        require_annotation_coverage(top, min_pct=80.0)
    err = exc_info.value
    assert err.min_pct == 80.0
    assert err.coverage.coverage_pct == 0.0
    assert len(err.coverage.unannotated_ports) == 2
    assert 'vout' in str(err)


def test_annotation_coverage_error_message_lists_ports(tmp_path):
    comp = _sub(tmp_path, 'comp', {'sig': PortDir.IN})
    comp.pin_list['sig'].net = 'net'
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_submodule('c1', comp)
    with pytest.raises(AnnotationCoverageError, match='sig'):
        require_annotation_coverage(top, min_pct=50.0)


def test_rail_mismatch_three_way_all_pairs_detected(tmp_path):
    avdd  = VoltageDomain('avdd',  nominal=1.8)
    dvdd  = VoltageDomain('dvdd',  nominal=3.3)
    hvdd  = VoltageDomain('hvdd',  nominal=5.0)
    a = _sub(tmp_path, 'a', {'vdd': Port(dir=PortDir.IN, role=PortRole.POWER, domain=avdd)})
    b = _sub(tmp_path, 'b', {'vdd': Port(dir=PortDir.IN, role=PortRole.POWER, domain=dvdd)})
    c = _sub(tmp_path, 'c', {'vdd': Port(dir=PortDir.IN, role=PortRole.POWER, domain=hvdd)})
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_module(a, 'a1', nets={'vdd': 'vdd_net'})
    top.add_module(b, 'b1', nets={'vdd': 'vdd_net'})
    top.add_module(c, 'c1', nets={'vdd': 'vdd_net'})
    violations = check_voltage_domains(top)
    rail_violations = [v for v in violations if v.rule_id == 'DOM-001']
    # Should detect avdd-dvdd, avdd-hvdd, dvdd-hvdd = 3 violations
    assert len(rail_violations) == 3


def test_no_driver_not_fired_on_power_rail(tmp_path):
    avdd = VoltageDomain('avdd', nominal=1.8)
    a = _sub(tmp_path, 'a', {'vdd': Port(dir=PortDir.IN, role=PortRole.POWER, domain=avdd)})
    b = _sub(tmp_path, 'b', {'vdd': Port(dir=PortDir.IN, role=PortRole.POWER, domain=avdd)})
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_module(a, 'a1', nets={'vdd': 'vdd_net'})
    top.add_module(b, 'b1', nets={'vdd': 'vdd_net'})
    violations = check_interfaces(top)
    assert all(v.rule_id != 'INT-004' for v in violations)


def test_downward_domain_propagation_enables_crossing_detection(tmp_path):
    avdd = VoltageDomain('avdd', nominal=1.8)
    dvdd = VoltageDomain('dvdd', nominal=3.3)
    # Port a has no explicit domain — it will inherit from assign_net_domain
    a = _sub(tmp_path, 'a', {'sig': Port(dir=PortDir.OUT, role=PortRole.ANALOG)})
    b = _sub(tmp_path, 'b', {'sig': Port(dir=PortDir.IN,  role=PortRole.DIGITAL, domain=dvdd)})
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_module(a, 'a1', nets={'sig': 'sig_net'})
    top.add_module(b, 'b1', nets={'sig': 'sig_net'})
    top.assign_net_domain('sig_net', avdd)  # explicit net domain → propagates down to port a
    violations = check_voltage_domains(top)
    rule_ids = [v.rule_id for v in violations]
    assert 'DOM-002' in rule_ids


# ---------------------------------------------------------------------------
# Phase 1 net refactor: PortBinding lives in silk.circuit.net
# ---------------------------------------------------------------------------

def test_portbinding_imported_from_net_module():
    """After Phase 1, PortBinding in interface_check is re-exported from net."""
    from silk.circuit.interface_check import PortBinding as PB_from_check
    from silk.circuit.net import PortBinding as PB_from_net
    assert PB_from_check is PB_from_net


def test_check_interfaces_reads_net_pins(tmp_path):
    """check_interfaces reads net.pins directly rather than rebuilding its map."""
    pdk = MagicMock()
    pdk.model_names = []
    pdk._devices = {}

    comp = Module('TEST', str(tmp_path), 'comp.cir', pdk_details=pdk)
    comp.define_module_name('comp')
    comp.define_module_ports({
        'out': (PortDir.OUT, PortRole.ANALOG),
    })

    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=pdk)
    top.module_name = 'top'
    top.add_module(comp, 'u1', nets={'out': 'vshared'})
    top.add_module(comp, 'u2', nets={'out': 'vshared'})

    # Two drivers on the same net -> DRIVER_CONFLICT (INT-001)
    violations = check_interfaces(top)
    rule_ids = {v.rule_id for v in violations}
    assert 'INT-001' in rule_ids
