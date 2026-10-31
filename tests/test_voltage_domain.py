# tests/test_voltage_domain.py
import pytest
from silk.circuit.voltage_domain import VoltageDomain


def test_v_min_v_max_from_tolerance():
    d = VoltageDomain('avdd', nominal=1.8, tolerance=0.1)
    assert d.v_min() == pytest.approx(1.62)
    assert d.v_max() == pytest.approx(1.98)


def test_v_min_v_max_explicit():
    d = VoltageDomain('avdd', nominal=1.8, min=1.6, max=2.0)
    assert d.v_min() == 1.6
    assert d.v_max() == 2.0


def test_explicit_overrides_tolerance():
    d = VoltageDomain('avdd', nominal=1.8, min=1.6, max=2.0, tolerance=0.5)
    assert d.v_min() == 1.6   # explicit wins
    assert d.v_max() == 2.0


def test_v_min_v_max_exact_nominal():
    d = VoltageDomain('avdd', nominal=1.8)
    assert d.v_min() == 1.8
    assert d.v_max() == 1.8


def test_v_min_v_max_partial_bounds():
    # Only max set — v_min falls back to nominal
    d = VoltageDomain('rail', nominal=1.8, max=2.0)
    assert d.v_min() == 1.8     # no min, no tolerance → nominal
    assert d.v_max() == 2.0     # explicit max

    # Only min set — v_max falls back to nominal
    d2 = VoltageDomain('rail', nominal=1.8, min=1.5)
    assert d2.v_min() == 1.5    # explicit min
    assert d2.v_max() == 1.8    # no max, no tolerance → nominal


def test_v_low_always_zero():
    d = VoltageDomain('avdd', nominal=1.8, tolerance=0.1)
    assert d.v_low() == 0.0


def test_would_stress_true():
    d = VoltageDomain('iovdd', nominal=3.3, tolerance=0.1)
    assert d.would_stress(1.98) is True   # v_max=3.63 > 1.98


def test_would_stress_false():
    d = VoltageDomain('avdd', nominal=1.8, tolerance=0.1)
    assert d.would_stress(1.98) is False  # v_max=1.98, not strictly greater


def test_contains_true():
    d = VoltageDomain('avdd', nominal=1.8, tolerance=0.1)
    assert d.contains(1.8) is True
    assert d.contains(1.62) is True
    assert d.contains(1.98) is True


def test_contains_false():
    d = VoltageDomain('avdd', nominal=1.8, tolerance=0.1)
    assert d.contains(2.0) is False
    assert d.contains(1.5) is False


def test_is_compatible_same_object():
    d = VoltageDomain('avdd', nominal=1.8, tolerance=0.1)
    assert d.is_compatible(d) is True


def test_is_compatible_same_name_overlapping():
    a = VoltageDomain('avdd', nominal=1.8, min=1.62, max=1.98)
    b = VoltageDomain('avdd', nominal=1.8, min=1.60, max=2.00)
    assert a.is_compatible(b) is True


def test_is_compatible_same_name_non_overlapping():
    a = VoltageDomain('avdd', nominal=1.8, min=1.62, max=1.98)
    b = VoltageDomain('avdd', nominal=3.3, min=3.0,  max=3.6)
    assert a.is_compatible(b) is False


def test_is_compatible_different_names():
    a = VoltageDomain('avdd', nominal=1.8)
    b = VoltageDomain('dvdd', nominal=1.8)
    assert a.is_compatible(b) is False


# ---------------------------------------------------------------------------
# Phase 1 net refactor: check_voltage_domains reads net.pins
# ---------------------------------------------------------------------------

def test_check_voltage_domains_reads_net_pins(tmp_path):
    """After the refactor, check_voltage_domains walks net_dict.values()."""
    from unittest.mock import MagicMock
    from silk.circuit.modules import Module
    from silk.circuit.circuit_elements import PortDir, PortRole, Port
    from silk.circuit.voltage_domain import VoltageDomain
    from silk.circuit.interface_check import check_voltage_domains

    pdk = MagicMock()
    pdk.model_names = []
    pdk._devices = {}

    avdd = VoltageDomain('avdd', nominal=1.8, tolerance=0.1)
    dvdd = VoltageDomain('dvdd', nominal=0.9, tolerance=0.1)

    blk_a = Module('TEST', str(tmp_path), 'a.cir', pdk_details=pdk)
    blk_a.define_module_name('blk_a')
    blk_a.define_module_ports({
        'vdd': Port(dir=PortDir.POWER, role=PortRole.POWER, domain=avdd),
    })

    blk_b = Module('TEST', str(tmp_path), 'b.cir', pdk_details=pdk)
    blk_b.define_module_name('blk_b')
    blk_b.define_module_ports({
        'vdd': Port(dir=PortDir.POWER, role=PortRole.POWER, domain=dvdd),
    })

    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=pdk)
    top.module_name = 'top'
    top.add_module(blk_a, 'a1', nets={'vdd': 'shared_vdd'})
    top.add_module(blk_b, 'b1', nets={'vdd': 'shared_vdd'})

    violations = check_voltage_domains(top)
    rule_ids = {v.rule_id for v in violations}
    assert 'DOM-001' in rule_ids


def test_switchable_defaults_false_and_is_a_plain_flag():
    assert VoltageDomain('avdd', nominal=1.8).switchable is False
    d = VoltageDomain('dvdd', nominal=0.9, tolerance=0.1, switchable=True)
    assert d.switchable is True
    assert d.v_low() == 0.0
    # Compatibility is about the physical rail, not its power-gating.
    assert d.is_compatible(VoltageDomain('dvdd', nominal=0.9, tolerance=0.1))


@pytest.mark.parametrize('port_count', [1, 2])
@pytest.mark.parametrize(
    'net_domain, port_domain, expected_count',
    [
        (VoltageDomain('io', nominal=3.3), VoltageDomain('core', nominal=1.8), 1),
        (VoltageDomain('io', nominal=1.8), VoltageDomain('core', nominal=1.8), 1),
        (VoltageDomain('rail', nominal=3.3), VoltageDomain('rail', nominal=1.8), 1),
        (VoltageDomain('rail', nominal=1.8, min=1.6, max=2.0),
         VoltageDomain('rail', nominal=1.9, min=1.7, max=2.1), 0),
        (None, VoltageDomain('core', nominal=1.8), 0),
        (VoltageDomain('io', nominal=3.3), None, 0),
    ],
)
def test_explicit_net_domain_vs_power_ports(
    tmp_path, port_count, net_domain, port_domain, expected_count,
):
    from silk.circuit.modules import Module
    from silk.circuit.circuit_elements import Port, PortDir, PortRole
    from silk.circuit.interface_check import check_voltage_domains
    from silk.lint.severity import Severity
    from silk.lint.violations import InterfaceViolation
    from silk.pdk import load_pdk

    pdk = load_pdk('sky130')
    top = Module('sky130', str(tmp_path), 'top', pdk_details=pdk)
    block = Module('sky130', str(tmp_path), 'block', pdk_details=pdk,
                   module_name='block')
    block.define_module_ports({
        'vdd': Port(dir=PortDir.POWER, role=PortRole.POWER, domain=port_domain),
    })
    for index in range(port_count):
        top.add_module(block, f'child{index}', nets={'vdd': 'supply'})
    if net_domain is not None:
        top.assign_net_domain('supply', net_domain)

    for _ in range(2):
        findings = [v for v in check_voltage_domains(top) if v.rule_id == 'DOM-001']
        assert len(findings) == expected_count * port_count
        for index, finding in enumerate(findings):
            assert isinstance(finding, InterfaceViolation)
            assert finding.severity == Severity.ERROR
            assert "Net 'supply'" in finding.message
            assert f'child{index}.vdd' in finding.message
            assert net_domain.name in finding.message
            assert port_domain.name in finding.message
            assert f'{net_domain.nominal}V' in finding.message
            assert f'{port_domain.nominal}V' in finding.message
        assert top.net_dict['supply'].domain is net_domain
        assert block.pin_list['vdd'].domain is port_domain


@pytest.mark.parametrize('role_name', ['POWER', 'ANALOG', 'GROUND'])
def test_explicit_net_domain_same_object_and_non_power_controls(tmp_path, role_name):
    from silk.circuit.modules import Module
    from silk.circuit.circuit_elements import Port, PortDir, PortRole
    from silk.circuit.interface_check import check_voltage_domains
    from silk.pdk import load_pdk

    pdk = load_pdk('sky130')
    net_domain = VoltageDomain('io', nominal=3.3)
    port_domain = (net_domain if role_name == 'POWER'
                   else VoltageDomain('core', nominal=1.8))
    top = Module('sky130', str(tmp_path), 'top', pdk_details=pdk)
    block = Module('sky130', str(tmp_path), 'block', pdk_details=pdk,
                   module_name='block')
    block.define_module_ports({
        'pin': Port(dir=PortDir.IN, role=getattr(PortRole, role_name),
                    domain=port_domain),
    })
    top.add_module(block, 'child', nets={'pin': 'shared'})
    top.assign_net_domain('shared', net_domain)
    assert check_voltage_domains(top) == []


@pytest.mark.parametrize('explicit_domain', [False, True])
def test_power_port_pair_conflict_preserved(tmp_path, explicit_domain):
    from silk.circuit.modules import Module
    from silk.circuit.circuit_elements import Port, PortDir, PortRole
    from silk.circuit.interface_check import check_voltage_domains
    from silk.pdk import load_pdk

    pdk = load_pdk('sky130')
    top = Module('sky130', str(tmp_path), 'top', pdk_details=pdk)
    domains = [VoltageDomain('io', nominal=3.3), VoltageDomain('core', nominal=1.8)]
    for index, domain in enumerate(domains):
        block = Module('sky130', str(tmp_path), f'block{index}', pdk_details=pdk,
                       module_name=f'block{index}')
        block.define_module_ports({
            'vdd': Port(dir=PortDir.POWER, role=PortRole.POWER, domain=domain),
        })
        top.add_module(block, f'child{index}', nets={'vdd': 'supply'})
    if explicit_domain:
        top.assign_net_domain('supply', domains[0])
    for _ in range(2):
        findings = [v for v in check_voltage_domains(top) if v.rule_id == 'DOM-001']
        assert len(findings) == (2 if explicit_domain else 1)
        assert findings[-1].message == (
            "Net 'supply' connects incompatible domains: "
            'child0.vdd(io, 3.3V) vs child1.vdd(core, 1.8V)'
        )


@pytest.mark.parametrize('strict_nets', [False, True])
def test_assign_net_domain_case_alias_preserves_net(tmp_path, strict_nets, capsys):
    from silk.circuit.modules import Module
    from silk.pdk import load_pdk

    top = Module('sky130', str(tmp_path), 'top', pdk_details=load_pdk('sky130'),
                 strict_nets=strict_nets)
    top.add_net('rail')
    net = top.net_dict['rail']
    inventory = dict(top.net_dict)
    domain = VoltageDomain('supply', nominal=1.8)
    capsys.readouterr()
    top.assign_net_domain('rail', domain)
    assert capsys.readouterr().err == ''
    top.assign_net_domain('RAIL', domain)
    assert capsys.readouterr().err == (
        "WARNING: net 'RAIL' differs only by case from existing net 'rail'. "
        'SPICE simulators fold node-name case (ngspice to lower, Xyce to upper), '
        'so these are ONE electrical node — silk merges them and uses '
        "'rail'. Rename one if you meant two nets.\n"
    )
    top.assign_net_domain('RAIL', domain)
    top.assign_net_domain('rail', domain)
    assert capsys.readouterr().err == ''
    assert top.net_dict == inventory
    assert top.net_dict['rail'] is net
    assert net.name == 'rail'
    assert net.domain is domain
    assert top._domains == [domain]


@pytest.mark.parametrize('strict_nets', [False, True])
def test_assign_net_domain_exact_name_bypasses_resolver(tmp_path, strict_nets, monkeypatch):
    from unittest.mock import Mock
    from silk.circuit.modules import Module
    from silk.pdk import load_pdk

    top = Module('sky130', str(tmp_path), 'top', pdk_details=load_pdk('sky130'),
                 strict_nets=strict_nets)
    top.add_net('rail')
    resolver = Mock(wraps=top._element_mgr._ensure_net)
    monkeypatch.setattr(top._element_mgr, '_ensure_net', resolver)
    domain = VoltageDomain('supply', nominal=1.8)
    top.assign_net_domain('rail', domain)
    resolver.assert_not_called()
    assert top.net_dict['rail'].domain is domain
    top.assign_net_domain('RAIL', domain)
    resolver.assert_called_once_with('RAIL')
    assert top.net_dict['rail'].domain is domain


@pytest.mark.parametrize('strict_nets', [False, True])
def test_assign_net_domain_missing_net_policy(tmp_path, strict_nets):
    from silk.circuit.modules import Module
    from silk.exceptions import UndeclaredNetError
    from silk.pdk import load_pdk

    top = Module('sky130', str(tmp_path), 'top', pdk_details=load_pdk('sky130'),
                 strict_nets=strict_nets)
    inventory = dict(top.net_dict)
    domain = VoltageDomain('supply', nominal=1.8)
    if strict_nets:
        with pytest.raises(UndeclaredNetError) as exc:
            top.assign_net_domain('rail', domain)
        assert str(exc.value) == (
            'ERROR: net "rail" is not declared. '
            'Call add_net("rail") before connecting it.'
        )
        assert top.net_dict == inventory
        assert top._domains == []
    else:
        top.assign_net_domain('rail', domain)
        assert set(top.net_dict) == set(inventory) | {'rail'}
        assert top.net_dict['rail'].domain is domain
        assert top._domains == [domain]


@pytest.mark.parametrize('strict_nets', [False, True])
@pytest.mark.parametrize('spelling', ['rail', 'RAIL'])
def test_assign_net_domain_compatibility_preserves_state(tmp_path, strict_nets, spelling):
    from silk.circuit.modules import Module
    from silk.pdk import load_pdk

    top = Module('sky130', str(tmp_path), 'top', pdk_details=load_pdk('sky130'),
                 strict_nets=strict_nets)
    top.add_net('rail')
    net = top.net_dict['rail']
    inventory = dict(top.net_dict)
    original = VoltageDomain('supply', nominal=1.8, min=1.6, max=2.0)
    compatible = VoltageDomain('supply', nominal=1.9, min=1.7, max=2.1)
    top.assign_net_domain('rail', original)
    top.assign_net_domain(spelling, compatible)
    top.assign_net_domain(spelling, compatible)
    assert net.domain is compatible
    assert top._domains == [original, compatible]
    incompatible = VoltageDomain('other', nominal=3.3)
    with pytest.raises(ValueError) as exc:
        top.assign_net_domain(spelling, incompatible)
    assert str(exc.value) == (
        f"Net '{spelling}': incompatible domain assignment. "
        "Existing: 'supply' (1.9V), New: 'other' (3.3V)"
    )
    assert net.domain is compatible
    assert top.net_dict == inventory
    assert top.net_dict['rail'] is net
    assert top._domains == [original, compatible]
