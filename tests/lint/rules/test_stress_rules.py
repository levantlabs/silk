import os

import pytest
from unittest.mock import MagicMock
from silk.lint.rules.stress import JunctionOverrangeRule
from tests._env import needs_ngspice, needs_pdk
from silk.lint.severity import Severity
from silk.lint.violations import JunctionViolation
from silk.lint.locations import TerminalLocation
from silk.circuit.modules import Module


def test_junction_rule_id():
    assert JunctionOverrangeRule().rule_id == "STR-001"


def test_junction_rule_category():
    assert JunctionOverrangeRule().category == "stress"


def test_junction_rule_requires_simulation():
    assert JunctionOverrangeRule().requires_simulation is True


def test_junction_rule_severity_default_error():
    assert JunctionOverrangeRule().default_severity == Severity.ERROR


def test_junction_rule_rationale_non_empty():
    assert len(JunctionOverrangeRule().rationale) > 50


def test_junction_rule_applicable_to_module():
    assert Module in JunctionOverrangeRule().applicable_to


def test_junction_rule_skips_module_without_simulation(tmp_path):
    """No _behavioral_mode flag means the rule yields nothing."""
    pdk = MagicMock()
    pdk.model_names = []
    pdk._devices = {}
    m = Module('TEST', str(tmp_path), 'top.cir', pdk_details=pdk)
    rule = JunctionOverrangeRule()
    assert list(rule.check(m)) == []


def test_junction_rule_yields_junction_violation_subclass(tmp_path):
    """When the rule fires, it produces JunctionViolation with TerminalLocation."""
    # This is a smoke test — proper fixtures with real simulation data live in
    # tests/test_junction_check.py. Here we just verify wiring: an empty module
    # in behavioral mode with no ratings produces no violations and no crashes.
    import numpy as np
    pdk = MagicMock()
    pdk.model_names = []
    pdk._devices = {}
    m = Module('TEST', str(tmp_path), 'top.cir', pdk_details=pdk)
    m._behavioral_mode = True
    m._behavioral_t = np.linspace(0, 1e-6, 100)
    rule = JunctionOverrangeRule()
    violations = list(rule.check(m))
    assert violations == []


# ---------------------------------------------------------------------------
# Regression: SPICE-path + hierarchy.
#
# Before the fix, STR-001 (a) bailed out on every SPICE-simulated module
# (`_behavioral_mode` gate) and (b) only iterated the top-level `elem_dict`,
# so an overstressed device living INSIDE a composed submodule was invisible
# — the exact "human won't look there" case the write-up called out.
#
# This builds a two-level hierarchy (a `leafcell` submodule instantiated as
# `amp` inside a top module), drives the leaf's internal nmos1p8 gate hard
# enough to blow its Vgs_max=1.95V rating, simulates the TOP module directly
# in real ngspice (no behavioral fabrication), and asserts the rule finds the
# violation with a hierarchical instance path naming the submodule instance.
# ---------------------------------------------------------------------------

_SKY130_LIB = os.path.expanduser(
    '~/pdk/sky130A/libs.tech/ngspice/sky130.lib.spice')
_PDK_YAML = os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'silk', 'pdk', 'data',
    'skywater_130_device_details.yaml')
_have_pdk = os.path.exists(_SKY130_LIB)

pytestmark_spice = [
    pytest.mark.slow,
    pytest.mark.skipif(not _have_pdk, reason='sky130 ngspice models not installed'),
]


def _composed_overstress_top(tmp_path):
    """top -> (submodule 'amp' = leafcell) -> nmos1p8 'mn', gate overdriven."""
    from silk.pdk.basepdk import BasePDK
    from silk.circuit.circuit_elements import PortDir
    from silk import sources
    from silk.elements import elements
    from silk.simulation import analysis

    pdk = BasePDK(_PDK_YAML, library_location=_SKY130_LIB)

    leaf = Module('PDK_skywater130', str(tmp_path), 'leaf.cir',
                  pdk_details=pdk, module_name='leafcell')
    leaf.define_module_ports({'in': PortDir.IN, 'out': PortDir.OUT,
                              'vdd': PortDir.POWER, 'vss': PortDir.GROUND})
    with leaf:
        pdk.nmos1p8('mn', d='out', g='in', s='vss', b='vss', w=1.0, l=0.15)

    top = Module('PDK_skywater130', str(tmp_path), 'top.cir',
                pdk_details=pdk, module_name=None)
    top.add_module(leaf, 'amp',
                    nets={'in': 'vin', 'out': 'vout', 'vdd': 'vdd', 'vss': '0'})
    with top:
        elements.resistor('rload', p='vout', n='0', value=10e3)
        sources.VoltageSource('vdd', positive='vdd', negative='0', dc=1.8)
        # 3.3V on the gate of a 1.8V-rated core device (Vgs_max=1.95V) —
        # deliberately over-rated, INSIDE the composed submodule.
        sources.VoltageSource('vin', positive='vin', negative='0', dc=3.3)
    top.add_sim_type(analysis.Transient(step=1e-9, stop=10e-9))
    return top


@pytest.mark.slow
@pytest.mark.skipif(not _have_pdk, reason='sky130 ngspice models not installed')
def test_junction_rule_finds_spice_violation_inside_submodule(tmp_path):
    top = _composed_overstress_top(tmp_path)
    top.simulate(corner='tt')
    assert top.sim_results_loaded
    assert not getattr(top, '_behavioral_mode', False)

    violations = list(JunctionOverrangeRule().check(top))

    assert violations, "STR-001 found nothing on a real SPICE run with a known overstress"
    hit = [v for v in violations if v.location.terminal == 'Vgs']
    assert hit, f"expected a Vgs violation; got {[v.location.terminal for v in violations]}"
    v = hit[0]
    # Hierarchical path: the device lives inside submodule instance 'amp'.
    assert 'amp' in v.location.instance_path, (
        f"violation instance_path {v.location.instance_path!r} does not name "
        "the submodule instance — the hierarchy walk regressed"
    )
    assert v.rule_id == 'STR-001'
    assert v.severity == Severity.ERROR
    assert v.device_type == 'nmos1p8'
    # 1.95 V is the SkyWater-documented model-valid VGS maximum for
    # nfet_01v8 (TASK-48; the catalog carried an unsourced 1.98 before).
    assert v.rated_max == pytest.approx(1.95)
    assert v.measured_v > 1.95


def _coverage_module(tmp_path, monkeypatch, *, behavioral=False):
    """Real catalog/device graph; synthetic results isolate data availability."""
    import numpy as np
    from types import SimpleNamespace
    from silk.pdk import load_pdk
    pdk = load_pdk('sky130')
    m = Module('sky130', str(tmp_path), 'coverage.cir', pdk_details=pdk)
    with m:
        pdk.nmos1p8('mn', d='out', g='inp', s='0', b='0', w=1., l=.15)
    waves = {'out': np.array([0., 0.]), 'inp': np.array([1., 1.])}
    t = np.array([0., 1e-9])
    if behavioral:
        m._behavioral_mode = True
        m._behavioral_t = t
        for node, arr in waves.items():
            m.net_dict[node].voltage = arr
    else:
        m.sim_results_loaded = True
        def load(node, quantity, **kwargs):
            if node not in waves:
                raise KeyError(node)
            return SimpleNamespace(x=t, y=waves[node], xlabel='time')
        monkeypatch.setattr(m, 'load_signal', load)
    return m, waves


@pytest.mark.parametrize('behavioral', [False, True])
@pytest.mark.parametrize('reason', ['no_ratings', 'no_waveform', 'small_signal',
                                    'length_mismatch', 'all_ground'])
def test_stress_coverage_reasons(tmp_path, monkeypatch, behavioral, reason):
    import numpy as np
    from silk.circuit.junction_check import junction_stress_coverage
    m, waves = _coverage_module(tmp_path, monkeypatch, behavioral=behavioral)
    if reason == 'no_ratings':
        catalog_device = m.pdk_details._devices['nmos1p8']
        catalog_device.junction_ranges.clear()
        catalog_device.junction_ratings.clear()
    elif reason == 'all_ground':
        m.elem_dict['mn']['device'].nets.update({p: '0' for p in ('g', 'd', 's', 'b')})
    else:
        value = {'no_waveform': None, 'small_signal': np.array([1j, 1j]),
                 'length_mismatch': np.array([1.])}[reason]
        if behavioral:
            m.net_dict['inp'].voltage = value
        elif value is None:
            del waves['inp']
        else:
            waves['inp'] = value
    coverage = junction_stress_coverage(m)
    assert coverage.total == 1
    device, = coverage.devices
    assert reason in device.reasons
    assert not device.failed
    if not behavioral or reason == 'no_ratings':
        assert device.coverage == 'none'
    elif reason != 'all_ground':
        assert device.coverage == 'partial'


def test_registry_coverage_uses_same_pass_and_never_plain_pass(tmp_path, monkeypatch):
    from silk.lint.registry import RuleRegistry, VisitOutcome
    m, waves = _coverage_module(tmp_path, monkeypatch)
    del waves['inp']
    result = RuleRegistry([JunctionOverrangeRule()]).run(m, has_simulation_data=True)
    event, = result.visit_log
    assert event.outcome != VisitOutcome.PASS
    assert event.context['stress_coverage']['summary'] == {
        'total': 1, 'full': 0, 'partial': 0, 'none': 1, 'failed': 0}


@pytest.mark.slow
@needs_ngspice
@needs_pdk
def test_9v_inverter_save_list_coverage(tmp_path):
    from silk.pdk import load_pdk
    from silk import sources
    from silk.simulation import Transient
    from silk.circuit.junction_check import check_junction_overrange, junction_stress_coverage
    pdk = load_pdk('sky130')
    m = Module('sky130', str(tmp_path), 'inverter.cir', pdk_details=pdk)
    with m:
        pdk.nmos1p8('mn', d='out', g='inp', s='0', b='0', w=1., l=.15)
        pdk.pmos1p8('mp', d='out', g='inp', s='vdd', b='vdd', w=2., l=.15)
        sources.VoltageSource('dd', positive='vdd', negative='0', dc=9.)
        sources.VoltageSource('in', positive='inp', negative='0',
                              waveform=sources.Pulse(0, 1.8, tr=1e-10, tf=1e-10,
                                                     pw=4e-9, per=8e-9))
    m.add_sim_type(Transient(step=1e-9, stop=10e-9))
    m.simulate(corner='tt')
    full = junction_stress_coverage(m)
    before = check_junction_overrange(m)
    assert full.summary() == {'total': 2, 'full': 2, 'partial': 0, 'none': 0, 'failed': 2}
    assert len(before) == 7
    m.set_save_list(['inp', 'out'])
    m.simulate(corner='tt')
    limited = junction_stress_coverage(m)
    after = check_junction_overrange(m)
    assert len(after) == 3
    assert after == [v for v in before if v.instance == 'mn']
    assert limited.summary() == {'total': 2, 'full': 1, 'partial': 0, 'none': 1, 'failed': 1}
    mp = next(d for d in limited.devices if d.instance_path == 'mp')
    assert mp.reasons == ('no_waveform',)
    assert set(mp.pairs.values()) == {'unavailable'}


def test_registry_stress_measures_once_and_does_not_reuse_previous_run(tmp_path, monkeypatch):
    from silk.lint.registry import RuleRegistry
    m, waves = _coverage_module(tmp_path, monkeypatch)
    calls = []
    load = m.load_signal
    def counted(*args, **kwargs):
        calls.append(args[0])
        return load(*args, **kwargs)
    monkeypatch.setattr(m, 'load_signal', counted)
    waves['inp'][:] = 3.
    reg = RuleRegistry([JunctionOverrangeRule()])
    result = reg.run(m, has_simulation_data=True)
    assert len(result.violations) == 2  # Vgs and Vgd, each 3 V
    assert result.visit_log[0].context['stress_coverage']['summary']['failed'] == 1
    assert calls == ['out', 'inp']
    waves['inp'][:] = 1.
    result = reg.run(m, has_simulation_data=True)
    assert result.violations == []
    assert result.visit_log[0].context['stress_coverage']['summary']['failed'] == 0
    assert calls == ['out', 'inp', 'out', 'inp']


def test_behavioral_nested_devices_are_unchecked_not_absent(tmp_path):
    import numpy as np
    from silk.circuit.junction_check import junction_stress_coverage
    top = _composed_overstress_top(tmp_path)
    top._behavioral_mode = True
    top._behavioral_t = np.array([0., 1e-9])
    coverage = junction_stress_coverage(top)
    device, = [d for d in coverage.devices if d.type_name == 'nmos1p8']
    assert device.instance_path == 'amp.mn'
    assert device.coverage == 'none'
    assert device.reasons == ('no_waveform',)
    assert not device.failed


@pytest.mark.parametrize('behavioral', [False, True])
def test_ideal_loads_do_not_prevent_catalog_device_pass(tmp_path, monkeypatch, capsys, behavioral):
    from silk.elements import elements
    from silk.circuit.junction_check import junction_stress_coverage
    from silk.circuit.report import generate_report
    from silk.lint.registry import RuleRegistry, VisitOutcome
    m, _ = _coverage_module(tmp_path, monkeypatch, behavioral=behavioral)
    with m:
        elements.resistor('RL', p='out', n='0', value=10e3)
        elements.capacitor('CL', p='out', n='0', value=1e-12)
        elements.inductor('LL', p='out', n='0', value=1e-9)
    coverage = junction_stress_coverage(m)
    assert coverage.summary() == {'total': 1, 'full': 1, 'partial': 0, 'none': 0, 'failed': 0}
    assert [d.instance_path for d in coverage.devices] == ['mn']
    reg = RuleRegistry([JunctionOverrangeRule()])
    event, = reg.run(m, has_simulation_data=True).visit_log
    assert event.outcome == VisitOutcome.PASS
    report = generate_report(m, registry=reg, auto_discover_config=False)
    report.print_summary()
    assert 'STR-001 PASS: 1/1 devices fully checked' in capsys.readouterr().out
    assert report.coverage.rule_coverage.root.local['STR-001'] == (1, 0, 0)
    path = tmp_path / 'lint.html'
    report.lint.save_html(path)
    assert 'STR-001 PASS: 1/1 devices fully checked' in path.read_text()


@pytest.mark.parametrize('behavioral', [False, True])
def test_non_catalog_type_is_out_of_stress_scope(tmp_path, monkeypatch, behavioral):
    from silk.circuit.junction_check import junction_stress_coverage
    m, _ = _coverage_module(tmp_path, monkeypatch, behavioral=behavioral)
    m.pdk_details._devices.clear()
    coverage = junction_stress_coverage(m)
    assert coverage.devices == []
    assert coverage.summary() == {'total': 0, 'full': 0, 'partial': 0, 'none': 0, 'failed': 0}
