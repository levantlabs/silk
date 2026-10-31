# tests/test_circuit_report.py
import os

import numpy as np
import pytest
from unittest.mock import MagicMock
from silk.circuit.report import TopologySummary, SimulationSummary, InstanceSummary, SignalStats
from silk.circuit.report import _build_topology, _build_simulation
from silk.circuit.modules import Module
from silk.circuit.circuit_elements import PortDir, PortRole
from tests._env import scratch_dir


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


def _top_with_instances(tmp_path):
    """Top-level module with two subcircuit instances and a net."""
    comp = _sub(tmp_path, 'comp', {
        'vp':   PortDir.IN,
        'vm':   PortDir.IN,
        'vout': (PortDir.OUT, PortRole.DIGITAL),
    })
    enc = _sub(tmp_path, 'enc', {
        'din':  (PortDir.IN,  PortRole.DIGITAL),
        'dout': (PortDir.OUT, PortRole.DIGITAL),
    })
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.module_name = 'top'
    top.add_net('vcmp')
    top.add_net('vdd')
    top.add_submodule('comp1', comp)
    top.add_submodule('enc1', enc)
    return top


# --- TopologySummary ---

def test_topology_instance_count(tmp_path):
    top = _top_with_instances(tmp_path)
    topo = _build_topology(top)
    assert isinstance(topo, TopologySummary)
    assert topo.total_instances == 2


def test_topology_net_count(tmp_path):
    top = _top_with_instances(tmp_path)
    topo = _build_topology(top)
    assert topo.total_nets == 2


def test_topology_instance_summary_fields(tmp_path):
    top = _top_with_instances(tmp_path)
    topo = _build_topology(top)
    names = {i.instance for i in topo.instances}
    assert 'comp1' in names
    assert 'enc1'  in names
    comp_entry = next(i for i in topo.instances if i.instance == 'comp1')
    assert isinstance(comp_entry, InstanceSummary)
    assert comp_entry.module_name == 'comp'
    assert comp_entry.port_count  == 3


def test_topology_module_name(tmp_path):
    top = _top_with_instances(tmp_path)
    topo = _build_topology(top)
    assert topo.module_name == 'top'


def test_topology_empty_module(tmp_path):
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    topo = _build_topology(top)
    assert topo.total_instances == 0
    assert topo.total_nets == 0


# --- SimulationSummary ---

def test_simulation_summary_none_when_no_results(tmp_path):
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    assert _build_simulation(top) is None


def test_simulation_summary_stats(tmp_path):
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top._behavioral_t = np.linspace(0, 1e-9, 100)
    # Phase 2a: populate Net.voltage directly.
    for name, arr in {
        'vin': np.linspace(0.0, 1.8, 100),
        'vout': np.ones(100) * 0.9,
    }.items():
        if name not in top.net_dict:
            top.add_net(name)
        top.net_dict[name].voltage = arr
    sim = _build_simulation(top)
    assert isinstance(sim, SimulationSummary)
    assert sim.n_points == 100
    assert abs(sim.t_stop - 1e-9) < 1e-12
    assert len(sim.signals) == 2
    vin_stat = next(s for s in sim.signals if s.name == 'vin')
    assert isinstance(vin_stat, SignalStats)
    assert abs(vin_stat.min  - 0.0) < 1e-6
    assert abs(vin_stat.max  - 1.8) < 1e-6
    assert abs(vin_stat.mean - 0.9) < 1e-3


# ---------------------------------------------------------------------------
# Task 2: CircuitReport and generate_report
# ---------------------------------------------------------------------------

from silk.circuit.report import CircuitReport, generate_report


def _top_with_simulation(tmp_path):
    """Top-level module with instances and completed behavioral simulation."""
    top = _top_with_instances(tmp_path)
    top._behavioral_t = np.linspace(0, 160e-9, 1600)
    # Phase 2a: populate Net.voltage directly.
    for name, arr in {
        'vin':  np.linspace(0.0, 1.8, 1600),
        'vout': np.ones(1600) * 0.9,
    }.items():
        if name not in top.net_dict:
            top.add_net(name)
        top.net_dict[name].voltage = arr
    return top


def test_generate_report_returns_circuit_report(tmp_path):
    top = _top_with_instances(tmp_path)
    report = generate_report(top)
    assert isinstance(report, CircuitReport)


def test_generate_report_topology(tmp_path):
    top = _top_with_instances(tmp_path)
    report = generate_report(top)
    assert report.topology.total_instances == 2
    assert report.topology.total_nets      == 2


def test_generate_report_annotation(tmp_path):
    top = _top_with_instances(tmp_path)
    report = generate_report(top)
    assert report.annotation.total == 5       # 3 comp ports + 2 enc ports
    assert report.annotation.annotated == 3   # vout, din, dout have roles


def test_generate_report_no_violations(tmp_path):
    top = _top_with_instances(tmp_path)
    report = generate_report(top)
    assert isinstance(report.violations, list)
    assert report.interface_check_passed is True


def test_generate_report_simulation_populated(tmp_path):
    top = _top_with_simulation(tmp_path)
    report = generate_report(top)
    assert report.simulation is not None
    assert report.simulation.n_points == 1600
    assert len(report.simulation.signals) == 2


def test_generate_report_no_simulation(tmp_path):
    top = _top_with_instances(tmp_path)
    report = generate_report(top)
    assert report.simulation is None


def test_generate_report_module_name(tmp_path):
    top = _top_with_instances(tmp_path)
    report = generate_report(top)
    assert report.module_name == 'top'


def test_print_summary_runs_without_error(tmp_path, capsys):
    top = _top_with_simulation(tmp_path)
    report = generate_report(top)
    report.print_summary()           # must not raise
    captured = capsys.readouterr()
    assert 'top' in captured.out
    assert 'coverage' in captured.out.lower()
    assert 'interface' in captured.out.lower()


def test_interface_check_passed_false_on_error(tmp_path):
    """interface_check_passed is False when violations contain an ERROR."""
    from silk.lint.violations import InterfaceViolation
    from silk.lint.severity import Severity
    from silk.lint.locations import NetLocation
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.module_name = 'top'
    report = generate_report(top)
    # Inject a synthetic ERROR violation using the new lint API
    report.violations.append(InterfaceViolation(
        rule_id='INT-001',
        severity=Severity.ERROR,
        message='test',
        location=NetLocation(net_name='x', port_names=[]),
    ))
    assert report.interface_check_passed is False


# ---------------------------------------------------------------------------
# Task 3: Markdown rendering
# ---------------------------------------------------------------------------


def test_save_creates_file(tmp_path):
    top = _top_with_simulation(tmp_path)
    report = generate_report(top)
    out = str(tmp_path / 'report.md')
    report.save(out)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0


def test_save_markdown_contains_module_name(tmp_path):
    top = _top_with_simulation(tmp_path)
    report = generate_report(top)
    out = str(tmp_path / 'report.md')
    report.save(out)
    text = open(out).read()
    assert 'top' in text


def test_save_markdown_has_topology_section(tmp_path):
    top = _top_with_instances(tmp_path)
    report = generate_report(top)
    out = str(tmp_path / 'report.md')
    report.save(out)
    text = open(out).read()
    assert 'Topology' in text
    assert 'comp1' in text
    assert 'enc1'  in text


def test_save_markdown_has_annotation_section(tmp_path):
    top = _top_with_instances(tmp_path)
    report = generate_report(top)
    out = str(tmp_path / 'report.md')
    report.save(out)
    text = open(out).read()
    assert 'Annotation' in text
    assert '%' in text   # coverage percentage present


def test_save_markdown_has_interface_section(tmp_path):
    top = _top_with_instances(tmp_path)
    report = generate_report(top)
    out = str(tmp_path / 'report.md')
    report.save(out)
    text = open(out).read()
    assert 'Interface' in text
    assert 'PASS' in text


def test_save_markdown_has_simulation_section(tmp_path):
    top = _top_with_simulation(tmp_path)
    report = generate_report(top)
    out = str(tmp_path / 'report.md')
    report.save(out)
    text = open(out).read()
    assert 'Simulation' in text
    assert 'vin' in text
    assert 'vout' in text


def test_save_markdown_no_simulation_section_when_not_run(tmp_path):
    top = _top_with_instances(tmp_path)
    report = generate_report(top)
    out = str(tmp_path / 'report.md')
    report.save(out)
    text = open(out).read()
    # Section heading should not appear when simulation was never run
    assert 'Behavioral Simulation' not in text


# ---------------------------------------------------------------------------
# Task 4: Top-level exports
# ---------------------------------------------------------------------------

def test_top_level_import():
    from silk import generate_report, CircuitReport
    assert callable(generate_report)


# ---------------------------------------------------------------------------
# Task 1: JSON persistence
# ---------------------------------------------------------------------------

import json as _json
from silk.circuit.report import CircuitReport, generate_report


def test_save_json_creates_file(tmp_path):
    top = _top_with_instances(tmp_path)
    report = generate_report(top)
    out = str(tmp_path / 'report.json')
    report.save_json(out)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0


def test_save_json_round_trip(tmp_path):
    top = _top_with_simulation(tmp_path)
    report = generate_report(top)
    out = str(tmp_path / 'report.json')
    report.save_json(out)
    loaded = CircuitReport.load_json(out)
    assert loaded.module_name == report.module_name
    assert loaded.topology.total_instances == report.topology.total_instances
    assert loaded.topology.total_nets == report.topology.total_nets
    assert loaded.simulation.n_points == report.simulation.n_points
    assert loaded.simulation.signals[0].name == report.simulation.signals[0].name
    assert abs(loaded.simulation.signals[0].mean - report.simulation.signals[0].mean) < 1e-9


def test_save_json_figures_excluded(tmp_path):
    top = _top_with_instances(tmp_path)
    report = generate_report(top)
    report.figures.append((MagicMock(), 'Test Figure'))
    out = str(tmp_path / 'report.json')
    report.save_json(out)   # must not raise even with a non-serializable figure
    d = _json.load(open(out))
    assert 'figures' not in d


def test_load_json_has_empty_figures(tmp_path):
    top = _top_with_instances(tmp_path)
    report = generate_report(top)
    out = str(tmp_path / 'report.json')
    report.save_json(out)
    loaded = CircuitReport.load_json(out)
    assert loaded.figures == []


def test_save_json_round_trip_no_simulation(tmp_path):
    top = _top_with_instances(tmp_path)
    report = generate_report(top)
    out = str(tmp_path / 'report.json')
    report.save_json(out)
    loaded = CircuitReport.load_json(out)
    assert loaded.simulation is None


def test_save_json_round_trip_annotation(tmp_path):
    top = _top_with_instances(tmp_path)
    report = generate_report(top)
    out = str(tmp_path / 'report.json')
    report.save_json(out)
    loaded = CircuitReport.load_json(out)
    assert loaded.annotation.total == report.annotation.total
    assert loaded.annotation.annotated == report.annotation.annotated
    assert len(loaded.annotation.unannotated_ports) == len(report.annotation.unannotated_ports)


# ---------------------------------------------------------------------------
# Task 2: HTML report
# ---------------------------------------------------------------------------


def test_save_html_creates_file(tmp_path):
    top = _top_with_instances(tmp_path)
    report = generate_report(top)
    out = str(tmp_path / 'report.html')
    report.save_html(out)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0


def test_save_html_contains_module_name(tmp_path):
    top = _top_with_instances(tmp_path)
    report = generate_report(top)
    out = str(tmp_path / 'report.html')
    report.save_html(out)
    text = open(out, encoding='utf-8').read()
    assert 'top' in text


def test_save_html_has_pass_badge(tmp_path):
    top = _top_with_instances(tmp_path)
    report = generate_report(top)
    out = str(tmp_path / 'report.html')
    report.save_html(out)
    text = open(out, encoding='utf-8').read()
    assert 'PASS' in text
    assert 'class="badge pass"' in text


def test_save_html_has_topology_section(tmp_path):
    top = _top_with_instances(tmp_path)
    report = generate_report(top)
    out = str(tmp_path / 'report.html')
    report.save_html(out)
    text = open(out, encoding='utf-8').read()
    assert 'Topology' in text
    assert 'comp1' in text
    assert 'enc1' in text


def test_save_html_has_simulation_section(tmp_path):
    top = _top_with_simulation(tmp_path)
    report = generate_report(top)
    out = str(tmp_path / 'report.html')
    report.save_html(out)
    text = open(out, encoding='utf-8').read()
    assert 'Behavioral Simulation' in text
    assert 'vin' in text


def test_save_html_no_simulation_when_not_run(tmp_path):
    top = _top_with_instances(tmp_path)
    report = generate_report(top)
    out = str(tmp_path / 'report.html')
    report.save_html(out)
    text = open(out, encoding='utf-8').read()
    assert 'Behavioral Simulation' not in text


def test_add_figure_embeds_in_html(tmp_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    top = _top_with_instances(tmp_path)
    report = generate_report(top)
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    report.add_figure(fig, title='Test Plot')
    plt.close(fig)
    out = str(tmp_path / 'report.html')
    report.save_html(out)
    text = open(out, encoding='utf-8').read()
    assert 'Test Plot' in text
    assert 'data:image/png;base64,' in text


def test_no_figure_section_when_no_figures(tmp_path):
    top = _top_with_instances(tmp_path)
    report = generate_report(top)
    out = str(tmp_path / 'report.html')
    report.save_html(out)
    text = open(out, encoding='utf-8').read()
    assert 'data:image/png;base64,' not in text


# ---------------------------------------------------------------------------
# Task 8: domain_violations and junction_violations in CircuitReport
# ---------------------------------------------------------------------------

from silk.circuit.junction_check import JunctionViolation
from silk.circuit.interface_check import check_voltage_domains
from silk.circuit.voltage_domain import VoltageDomain
from silk.circuit.circuit_elements import Port


def _make_top_with_domain_violation(tmp_path):
    """Build a minimal module with a RAIL_MISMATCH domain violation."""
    from unittest.mock import MagicMock
    pdk = MagicMock()
    pdk.model_names = []
    pdk._devices = {}
    avdd = VoltageDomain('avdd', nominal=1.8)
    dvdd = VoltageDomain('dvdd', nominal=1.8)
    from silk.circuit.modules import Module
    from silk.circuit.circuit_elements import PortDir, PortRole
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=pdk)
    sub_a = Module('TEST', str(tmp_path), 'a.cir', pdk_details=pdk)
    sub_a.define_module_name('a')
    sub_a.define_module_ports({'vdd': Port(dir=PortDir.IN, role=PortRole.POWER, domain=avdd)})
    sub_b = Module('TEST', str(tmp_path), 'b.cir', pdk_details=pdk)
    sub_b.define_module_name('b')
    sub_b.define_module_ports({'vdd': Port(dir=PortDir.IN, role=PortRole.POWER, domain=dvdd)})
    top.add_module(sub_a, 'a1', nets={'vdd': 'vdd_net'})
    top.add_module(sub_b, 'b1', nets={'vdd': 'vdd_net'})
    return top


def test_generate_report_includes_domain_violations(tmp_path):
    from silk.circuit.report import generate_report
    top = _make_top_with_domain_violation(tmp_path)
    report = generate_report(top)
    assert hasattr(report, 'domain_violations')
    assert any(v.rule_id == 'DOM-001' for v in report.domain_violations)


def test_generate_report_includes_junction_violations(tmp_path):
    import numpy as np
    from unittest.mock import MagicMock
    from silk.circuit.report import generate_report
    from silk.circuit.modules import Module
    pdk = MagicMock()
    pdk.model_names = []
    pdk._devices = {}
    m = Module('TEST', str(tmp_path), 'top.cir', pdk_details=pdk)
    m._behavioral_mode = True
    m._behavioral_t = np.linspace(0, 1e-9, 10)
    # No devices → junction_violations should be empty list
    report = generate_report(m)
    assert hasattr(report, 'junction_violations')
    assert report.junction_violations == []


def test_interface_check_passed_false_on_domain_error(tmp_path):
    from silk.circuit.report import generate_report
    top = _make_top_with_domain_violation(tmp_path)
    report = generate_report(top)
    assert report.interface_check_passed is False


def test_report_save_json_round_trip_with_new_fields(tmp_path):
    from silk.circuit.report import generate_report, CircuitReport
    top = _make_top_with_domain_violation(tmp_path)
    report = generate_report(top)
    path = str(tmp_path / 'report.json')
    report.save_json(path)
    loaded = CircuitReport.load_json(path)
    assert len(loaded.domain_violations) == len(report.domain_violations)
    assert loaded.junction_violations == []


# ---------------------------------------------------------------------------
# results-file review 2026-08-24: save_json/load_json round-trip completeness
# (metadata, spec_reconciliation, layout_estimate, schema_version,
# omitted_sections).
# ---------------------------------------------------------------------------

def test_save_json_round_trip_metadata(tmp_path):
    top = _top_with_instances(tmp_path)
    report = generate_report(top)
    out = str(tmp_path / 'report.json')
    report.save_json(out)
    loaded = CircuitReport.load_json(out)
    assert loaded.metadata is not None
    assert loaded.metadata.silk_version == report.metadata.silk_version
    assert loaded.metadata.pdk_name == report.metadata.pdk_name
    assert loaded.metadata.pdk_version == report.metadata.pdk_version
    assert loaded.metadata.module_name == report.metadata.module_name
    assert loaded.metadata.generated_at == report.metadata.generated_at
    assert loaded.metadata.context_hash == report.metadata.context_hash


def test_save_json_round_trip_metadata_none(tmp_path):
    """A directly-constructed report (no generate_report()) leaves metadata
    unset; that must round-trip as None, not crash."""
    top = _top_with_instances(tmp_path)
    report = generate_report(top)
    report.metadata = None
    out = str(tmp_path / 'report.json')
    report.save_json(out)
    loaded = CircuitReport.load_json(out)
    assert loaded.metadata is None


def test_save_json_writes_schema_version(tmp_path):
    top = _top_with_instances(tmp_path)
    report = generate_report(top)
    out = str(tmp_path / 'report.json')
    report.save_json(out)
    d = _json.load(open(out))
    assert d['schema_version'] == 5


def test_load_json_missing_schema_version_key_still_works(tmp_path):
    """Files written before schema versioning have no 'schema_version' key
    at all — load_json must treat that as version 1, not raise."""
    top = _top_with_instances(tmp_path)
    report = generate_report(top)
    out = str(tmp_path / 'report.json')
    report.save_json(out)
    d = _json.load(open(out))
    del d['schema_version']
    with open(out, 'w') as f:
        _json.dump(d, f)
    loaded = CircuitReport.load_json(out)
    assert loaded.module_name == report.module_name


def test_load_json_future_schema_version_raises(tmp_path):
    top = _top_with_instances(tmp_path)
    report = generate_report(top)
    out = str(tmp_path / 'report.json')
    report.save_json(out)
    d = _json.load(open(out))
    d['schema_version'] = 99
    with open(out, 'w') as f:
        _json.dump(d, f)
    with pytest.raises(ValueError, match='99'):
        CircuitReport.load_json(out)


def test_save_json_omitted_sections_key(tmp_path):
    top = _top_with_instances(tmp_path)
    report = generate_report(top)
    out = str(tmp_path / 'report.json')
    report.save_json(out)
    d = _json.load(open(out))
    assert d['omitted_sections'] == ['lint', 'coverage']


def test_load_json_lint_and_coverage_stay_none(tmp_path):
    top = _top_with_instances(tmp_path)
    report = generate_report(top)
    assert report.lint is not None       # generate_report populates these in-memory
    assert report.coverage is not None
    out = str(tmp_path / 'report.json')
    report.save_json(out)
    loaded = CircuitReport.load_json(out)
    assert loaded.lint is None
    assert loaded.coverage is None


def test_load_json_tolerates_missing_omitted_sections_key(tmp_path):
    top = _top_with_instances(tmp_path)
    report = generate_report(top)
    out = str(tmp_path / 'report.json')
    report.save_json(out)
    d = _json.load(open(out))
    del d['omitted_sections']
    with open(out, 'w') as f:
        _json.dump(d, f)
    loaded = CircuitReport.load_json(out)  # must not raise
    assert loaded.module_name == report.module_name


def test_save_json_round_trip_layout_estimate(tmp_path):
    from silk.layout.estimator import LayoutEstimate, NetLayout, DeviceLayout
    from silk.layout._types import LayoutWarning

    top = _top_with_instances(tmp_path)
    report = generate_report(top)
    report.layout_estimate = LayoutEstimate(
        computation_mode='estimated',
        routing_model_id='m1',
        timestamp=1.0,
        bounding_box=(10.0, 20.0),
        nets={'vcmp': NetLayout(length=5.0, width=0.5, r=100.0, c=1e-15,
                                 equivalence_class='cls_a')},
        devices={'comp1': DeviceLayout(area=12.5)},
        equivalence_classes={'cls_a': ('vcmp',)},
        warnings=(LayoutWarning(net_name='vcmp', reason='no_current_data',
                                 severity='warn', detail='no data'),),
    )
    out = str(tmp_path / 'report.json')
    report.save_json(out)
    loaded = CircuitReport.load_json(out)

    le = loaded.layout_estimate
    assert le is not None
    assert le.computation_mode == 'estimated'
    assert le.routing_model_id == 'm1'
    assert le.timestamp == 1.0
    assert le.bounding_box == (10.0, 20.0)
    assert le.nets['vcmp'].r == 100.0
    assert le.nets['vcmp'].c == 1e-15
    assert le.nets['vcmp'].length == 5.0
    assert le.nets['vcmp'].width == 0.5
    assert le.nets['vcmp'].equivalence_class == 'cls_a'
    assert le.devices['comp1'].area == 12.5
    assert len(le.warnings) == 1
    assert le.warnings[0].net_name == 'vcmp'
    assert le.warnings[0].reason == 'no_current_data'
    assert le.warnings[0].severity == 'warn'
    assert le.warnings[0].detail == 'no data'
    # equivalence_classes isn't in the JSON schema (no report/diff consumer
    # reads it — see _layout_from_dict's docstring), so it round-trips empty.
    assert le.equivalence_classes == {}


def test_save_json_round_trip_layout_estimate_none(tmp_path):
    top = _top_with_instances(tmp_path)
    report = generate_report(top)
    assert report.layout_estimate is None
    out = str(tmp_path / 'report.json')
    report.save_json(out)
    loaded = CircuitReport.load_json(out)
    assert loaded.layout_estimate is None


def test_save_json_round_trip_spec_reconciliation(tmp_path):
    from silk.specs import Spec, Measurement, SpecOutcome, SpecReconciliationReport

    top = _top_with_instances(tmp_path)
    report = generate_report(top)
    spec = Spec(name='gain', module=top, metric='gain_db', op='>=', target=20.0,
                units='dB', description='open-loop gain')
    measurement = Measurement(
        source='sim', value=25.0, corner='tt', timestamp=2.0,
        sim_run_id='run-1', notes='nominal', temperature=27.0,
        voltage=1.8, sim_type='ac', tool_name='ngspice-44',
        origin_version_id='version-7',
    )
    outcome = SpecOutcome(spec=spec, measurement=measurement, status='PASS',
                           margin=5.0, source_used='sim', calibrated=True,
                           notes='ok')
    report.spec_reconciliation = SpecReconciliationReport(
        module=top, outcomes=[outcome], calibrations_applied=['cal1'],
        timestamp=3.0,
    )
    out = str(tmp_path / 'report.json')
    report.save_json(out)
    loaded = CircuitReport.load_json(out)

    recon = loaded.spec_reconciliation
    assert recon is not None
    assert recon.module.module_name == 'top'
    assert recon.timestamp == 3.0
    assert recon.calibrations_applied == ['cal1']
    assert recon.pass_count == 1
    assert recon.fail_count == 0
    assert len(recon.outcomes) == 1

    o = recon.outcomes[0]
    assert o.spec.name == 'gain'
    assert o.spec.metric == 'gain_db'
    assert o.spec.op == '>='
    assert o.spec.target == 20.0
    assert o.spec.units == 'dB'
    assert o.spec.description == 'open-loop gain'
    assert o.status == 'PASS'
    assert o.margin == 5.0
    assert o.source_used == 'sim'
    assert o.calibrated is True
    assert o.notes == 'ok'
    assert o.measurement.value == 25.0
    assert o.measurement.source == 'sim'
    assert o.measurement.corner == 'tt'
    assert o.measurement.sim_run_id == 'run-1'
    assert o.measurement.notes == 'nominal'
    assert o.measurement.temperature == 27.0
    assert o.measurement.voltage == 1.8
    assert o.measurement.sim_type == 'ac'
    assert o.measurement.tool_name == 'ngspice-44'
    assert o.measurement.origin_version_id == 'version-7'


def test_save_json_round_trip_spec_reconciliation_when_none(tmp_path):
    """A report with no spec_reconciliation attached (direct construction,
    or never-called reconcile_specs()) still writes/loads without error.
    The reconstructed object has empty outcomes, which every consumer
    (report_diff, HTML/Markdown/terminal renderers) treats identically to
    None — see _spec_reconciliation_from_dict's docstring."""
    top = _top_with_instances(tmp_path)
    report = generate_report(top)
    report.spec_reconciliation = None
    out = str(tmp_path / 'report.json')
    report.save_json(out)
    loaded = CircuitReport.load_json(out)
    assert loaded.spec_reconciliation is not None
    assert loaded.spec_reconciliation.outcomes == []
    assert loaded.spec_reconciliation.calibrations_applied == []


def _nonfinite_report(tmp_path, op='<'):
    from silk.circuit.spec_report import SpecReport
    from silk.specs import Spec, Measurement, SpecOutcome, SpecReconciliationReport

    top = _top_with_instances(tmp_path)
    report = generate_report(top)
    values = [float('nan'), 0.0, 'Error: missing signal', float('inf'), -float('inf')]
    outcomes = []
    for name, value in zip(['unavailable', 'zero', 'error', 'positive', 'negative'], values):
        spec = Spec(name=name, module=top, metric=name, op=op, target=1.0)
        checked = SpecReport('test', 'top', {name: value}).check(name, name, op, 1.0)
        outcomes.append(SpecOutcome(
            spec=spec, measurement=Measurement(source='spice', value=value,
                                              passes=checked.passed),
            status='PASS' if checked.passed else 'FAIL',
        ))
    report.spec_reconciliation = SpecReconciliationReport(module=top, outcomes=outcomes)
    report.simulation = SimulationSummary(
        t_start=0.0, t_stop=1.0, n_points=2,
        signals=[SignalStats(name='out', min=float('nan'), max=float('nan'),
                             mean=float('nan'))],
    )
    return report


def _strict_json(raw):
    def reject(token):
        raise AssertionError(f'Non-standard JSON token: {token}')
    return _json.loads(raw, parse_constant=reject)


@pytest.mark.parametrize('op', ['<', '<=', '>', '>=', '=='])
def test_strict_report_round_trip_preserves_nan_and_spec_failures(tmp_path, op):
    import math
    from silk.circuit.spec_report import SpecReport

    report = _nonfinite_report(tmp_path, op)
    path = tmp_path / 'mixed.json'
    report.save_json(path)
    wire = _strict_json(path.read_text())
    unavailable = {'value': None, 'status': 'unavailable'}
    assert wire['spec_reconciliation']['outcomes'][0]['measurement']['value'] == unavailable
    assert wire['simulation']['signals'][0] == {
        'name': 'out', 'min': unavailable, 'max': unavailable, 'mean': unavailable,
    }
    assert wire['spec_reconciliation']['outcomes'][3]['measurement']['value'] == {
        'value': None, 'status': 'infinite', 'sign': '+',
    }
    assert wire['spec_reconciliation']['outcomes'][4]['measurement']['value'] == {
        'value': None, 'status': 'infinite', 'sign': '-',
    }
    restored = CircuitReport.load_json(path)
    outcomes = restored.spec_reconciliation.outcomes
    values = [o.measurement.value for o in outcomes]
    assert math.isnan(values[0])
    assert values[1:] == [0.0, 'Error: missing signal', math.inf, -math.inf]
    assert outcomes[0].status == 'FAIL'
    assert outcomes[0].measurement.passes is False
    check = SpecReport('test', 'top', {'unavailable': values[0]})
    assert check.check('still unavailable', 'unavailable', op, 1.0).passed is False
    assert [o.status for o in outcomes] == [o.status for o in report.spec_reconciliation.outcomes]
    stats = restored.simulation.signals[0]
    assert all(math.isnan(v) for v in (stats.min, stats.max, stats.mean))
    assert math.isnan(report.spec_reconciliation.outcomes[0].measurement.value)


def test_strict_report_parses_with_browser_json_parse(tmp_path):
    import shutil
    import subprocess

    node = shutil.which('node')
    if node is None:
        pytest.skip('Node.js is required to exercise browser-equivalent JSON.parse')
    path = tmp_path / 'browser.json'
    _nonfinite_report(tmp_path).save_json(path)
    result = subprocess.run([node, '-e', '''
        const assert = require('node:assert/strict');
        const report = JSON.parse(require('node:fs').readFileSync(process.argv[1], 'utf8'));
        const values = report.spec_reconciliation.outcomes.map(o => o.measurement.value);
        assert.deepEqual(values, [
            {value: null, status: 'unavailable'}, 0, 'Error: missing signal',
            {value: null, status: 'infinite', sign: '+'},
            {value: null, status: 'infinite', sign: '-'}
        ]);
        assert.deepEqual(report.simulation.signals[0].mean, values[0]);
    ''', str(path)], text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('version', [None, 1])
def test_load_legacy_report_with_nonstandard_float_tokens(tmp_path, version):
    import math
    from silk._json_codec import decode_json

    path = tmp_path / 'legacy.json'
    _nonfinite_report(tmp_path).save_json(path)
    legacy = decode_json(_json.loads(path.read_text()))
    if version is None:
        del legacy['schema_version']
    else:
        legacy['schema_version'] = version
    # Simulate the old Python-only wire format, which remains readable.
    path.write_text(_json.dumps(legacy))
    restored = CircuitReport.load_json(path)
    values = [o.measurement.value for o in restored.spec_reconciliation.outcomes]
    assert math.isnan(values[0])
    assert values[1:] == [0.0, 'Error: missing signal', math.inf, -math.inf]
    assert math.isnan(restored.simulation.signals[0].mean)


# ---------------------------------------------------------------------------
# Operating Point Table sections (TASK-330.6)
# ---------------------------------------------------------------------------

class _FrozenDatetime_330_6(__import__('datetime').datetime):
    """Frozen 'now' for the byte-identity checks below -- reviewer ruling:
    freeze the clock at the source, never normalize the rendered output."""
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 1, 1, 0, 0, 0)


class _FrozenDatetimeModule_330_6:
    datetime = _FrozenDatetime_330_6


def _frozen_clock_inverter(tmp_path, monkeypatch):
    """The exact sky130 inverter (pmos1p8+nmos1p8, w=1.0/0.5, l=0.18) used
    to capture tests/fixtures/report_html/no_op_capture_golden.html, built
    with the same frozen clock so generated_at matches the golden."""
    import silk.circuit.report as report_module
    monkeypatch.setattr(report_module, 'datetime', _FrozenDatetimeModule_330_6)
    from silk.pdk import load_pdk
    pdk = load_pdk('sky130')
    inv = Module('PDK_skywater130', str(tmp_path), 'inv.cir', pdk_details=pdk,
                 module_name='inv')
    inv.add(pdk.pmos1p8('mp', d='vout', g='vin', s='vdd', b='vdd', w=1.0, l=0.18))
    inv.add(pdk.nmos1p8('mn', d='vout', g='vin', s='vss', b='vss', w=0.5, l=0.18))
    return inv


def test_no_spec_reports_matches_pre_330_6_golden(tmp_path, monkeypatch):
    """Byte-identity (reviewer AC #4, first check): generate_report() with
    no spec_reports= at all must render EXACTLY the golden captured before
    op_tables support existed -- no regex, no normalizer, same frozen
    'Generated:' line both times. The baseline now also includes the explicit
    junction stress coverage section."""
    inv = _frozen_clock_inverter(tmp_path, monkeypatch)
    report = generate_report(inv, auto_discover_config=False)
    html = report._to_html()

    golden_path = os.path.join(os.path.dirname(__file__), 'fixtures',
                               'report_html', 'no_op_capture_golden.html')
    with open(golden_path, encoding='utf-8') as f:
        golden = f.read()
    assert html == golden


def test_no_capture_spec_report_matches_no_spec_reports_arg(tmp_path, monkeypatch):
    """Byte-identity (reviewer AC #4, second check): a SpecReport with
    op_devices=None (no save_device_params=True capture) must render
    IDENTICAL HTML to omitting spec_reports= entirely -- _build_op_tables
    skips it, so it contributes no section."""
    inv = _frozen_clock_inverter(tmp_path, monkeypatch)
    no_arg_html = generate_report(inv, auto_discover_config=False)._to_html()

    from silk.circuit.spec_report import SpecReport

    def no_op_table(self):
        raise AssertionError(
            'op_table() must never be called when op_devices is None')

    monkeypatch.setattr(SpecReport, 'op_table', no_op_table)
    with_spec_html = generate_report(
        inv, auto_discover_config=False,
        spec_reports=[SpecReport('unused', 'inv', {})],
    )._to_html()
    assert with_spec_html == no_arg_html


def test_build_op_tables_skips_no_capture_reports():
    from silk.circuit.report import _build_op_tables

    class _Captured:
        test_name = 'measured'
        corner = 'tt'
        op_devices = [{'path': 'xm1'}]

        def op_table(self):
            return 'SENTINEL_TABLE'

    class _NotCaptured:
        test_name = 'unmeasured'
        corner = None
        op_devices = None

        def op_table(self):
            raise AssertionError('must be skipped, never called')

    sections = _build_op_tables([_NotCaptured(), _Captured()])
    assert len(sections) == 1
    assert sections[0].test_name == 'measured'
    assert sections[0].corner == 'tt'
    assert sections[0].table == 'SENTINEL_TABLE'
    assert _build_op_tables(None) == []
    assert _build_op_tables([]) == []


def test_render_op_tables_html_classes(tmp_path):
    """Synthetic OpTable (PDK-free, same _record helper as
    tests/test_op_table.py's TestRegionRuleSynthetic): checks the HTML
    classes the reviewer's AC #5 + ft-marking ruling asked for, without
    needing a real simulator.
    """
    from silk.circuit.op_table import build_op_table
    from silk.circuit.report import OpTableSection
    from tests.test_op_table import _record
    from unittest.mock import MagicMock

    # Row 1: fully declared incl. cgg -> ft is a real derived_estimate;
    # vds - vdsat = 1mV -> weakly saturated -> op-flag on its <tr>.
    weak = _record(
        path='xm1', model='mA', polarity='n',
        declared=('id', 'vgs', 'vds', 'vbs', 'vth', 'vdsat', 'gm', 'gds',
                  'gmbs', 'cgg', 'cgs', 'cgd', 'cgso', 'cgdo', 'cgbo',
                  'capbd', 'capbs'),
        values={'vgs': 1.0, 'vth': 0.4, 'vds': 0.601, 'vdsat': 0.6,
                'id': 1e-4, 'gm': 2e-4, 'gds': 1e-5, 'cgg': 1e-15})
    # Row 2: Xyce-shaped -- only the 8-name XYCE_OP_CAPABILITY subset
    # declared, so gmbs/every capacitance (and the ft they'd derive) are
    # holes, not nan-from-a-real-reading.
    xyce_shaped = _record(
        path='xm2', model='mB', polarity='p',
        declared=('id', 'vgs', 'vds', 'vth', 'vdsat', 'gm', 'gds'),
        values={'vgs': 1.0, 'vth': 0.4, 'vds': 0.9, 'vdsat': 0.6,
                'id': 1e-4, 'gm': 2e-4, 'gds': 1e-5})
    table = build_op_table([weak, xyce_shaped], None, None)
    assert table[0].saturation_weak is True
    assert table[0].sources['ft'] == 'derived_estimate'
    assert table[1].sources['ft'] is None            # a hole, not an estimate

    pdk = MagicMock()
    pdk.model_names = []
    pdk._devices = {}
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=pdk, module_name='top')
    report = generate_report(top, auto_discover_config=False)
    report.op_tables = [OpTableSection(test_name='t1', corner='tt', table=table)]

    html = report._to_html()
    assert 'Operating Point Table — t1 (tt)' in html
    assert 'class="op-flag"' in html                  # row 1's weak saturation
    assert 'class="op-region-saturation"' in html
    assert 'class="op-hole"' in html                  # row 2's undeclared columns
    assert 'class="op-estimate"' in html              # row 1's real ft estimate
    assert 'title="Estimated:' in html
    assert 'op-note' in html                          # section note re: the estimate


def test_render_op_tables_html_cutoff_row_also_flagged(tmp_path):
    """TASK-330.6, reviewer ruling on AC #6: op-flag is NOT saturation_weak
    alone -- a cutoff row (vgs well below vth, real current) must carry it
    too. Synthetic, PDK-free (mirrors TestRegionRuleSynthetic's own cutoff
    fixtures in tests/test_op_table.py)."""
    from silk.circuit.op_table import build_op_table
    from silk.circuit.report import OpTableSection
    from tests.test_op_table import _record
    from unittest.mock import MagicMock

    cutoff = _record(
        path='xm3', name='m3', model='mC', polarity='n',
        declared=('id', 'vgs', 'vds', 'vbs', 'vth', 'vdsat', 'gm', 'gds'),
        values={'vgs': 0.1, 'vth': 0.4, 'vds': 0.9, 'vdsat': 0.05,
                'id': 1e-11, 'gm': 1e-8, 'gds': 1e-9})
    table = build_op_table([cutoff], None, None)
    assert table[0].region == 'cutoff'
    assert table[0].saturation_weak is False          # the case op-flag alone missed

    pdk = MagicMock()
    pdk.model_names = []
    pdk._devices = {}
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=pdk, module_name='top')
    report = generate_report(top, auto_discover_config=False)
    report.op_tables = [OpTableSection(test_name='t1', corner='tt', table=table)]

    html = report._to_html()
    # TASK-330.7: 'name' (the designer's own name) leads, 'path' next.
    assert '<tr class="op-flag">\n<td>m3</td>\n<td>xm3</td>' in html
    assert 'class="op-region-cutoff"' in html


def test_render_op_tables_html_escapes_caller_strings(tmp_path):
    """TASK-330.6 review fix: test_name/corner and every cell's text are
    caller/device data, not trusted literals -- must be HTML-escaped the
    same way every other section in report.py already escapes text."""
    from silk.circuit.op_table import build_op_table
    from silk.circuit.report import OpTableSection
    from tests.test_op_table import _record
    from unittest.mock import MagicMock

    rec = _record(
        path='<script>xm1</script>', model='m & co <b>',
        declared=('id', 'vgs', 'vds', 'vbs', 'vth', 'vdsat', 'gm', 'gds'),
        values={'vgs': 1.0, 'vth': 0.4, 'vds': 0.9, 'vdsat': 0.6,
                'id': 1e-4, 'gm': 2e-4, 'gds': 1e-5})
    table = build_op_table([rec], None, None)

    pdk = MagicMock()
    pdk.model_names = []
    pdk._devices = {}
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=pdk, module_name='top')
    report = generate_report(top, auto_discover_config=False)
    report.op_tables = [OpTableSection(
        test_name='<b>evil</b> & friends', corner='t&t', table=table)]

    html = report._to_html()
    # Nothing from caller data appears unescaped.
    assert '<script>' not in html
    assert '<b>evil</b>' not in html
    assert 'm & co <b>' not in html
    assert 't&t' not in html
    # The escaped forms are present instead.
    assert '&lt;b&gt;evil&lt;/b&gt; &amp; friends' in html
    assert 't&amp;t' in html
    assert '&lt;script&gt;xm1&lt;/script&gt;' in html
    assert 'm &amp; co &lt;b&gt;' in html


def test_render_op_tables_html_empty_table_message(tmp_path):
    """A captured SpecReport whose op_table() found zero MOS devices still
    renders its section heading, with the same '(no MOS devices)' message
    the console/Markdown renderers use (OpTable.__str__)."""
    from silk.circuit.op_table import build_op_table
    from silk.circuit.report import OpTableSection
    from unittest.mock import MagicMock

    table = build_op_table([], None, None)
    assert len(table) == 0

    pdk = MagicMock()
    pdk.model_names = []
    pdk._devices = {}
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=pdk, module_name='top')
    report = generate_report(top, auto_discover_config=False)
    report.op_tables = [OpTableSection(test_name='empty_test', corner=None, table=table)]

    html = report._to_html()
    assert 'Operating Point Table — empty_test' in html
    assert '(no MOS devices)' in html


def test_no_op_tables_no_section_in_html(tmp_path):
    """The default (empty) op_tables list renders no section at all --
    matches every pre-330.6 report."""
    top = _top_with_instances(tmp_path)
    report = generate_report(top)
    assert report.op_tables == []
    assert 'Operating Point Table' not in report._to_html()


from tests._env import needs_pdk, needs_ngspice


@needs_pdk
@needs_ngspice
def test_no_spec_reports_matches_with_sim_golden_real_ngspice(monkeypatch):
    """Byte-identity, second golden (team-lead request): the first golden
    (no_op_capture_golden.html) has no simulation, so it never exercises
    the insertion point right after Behavioral Simulation where the
    Operating Point Table section is spliced in. This golden is a REAL
    ngspice Op() run WITHOUT save_device_params -- report.simulation is
    populated, report.op_tables stays empty (no spec_reports= given) --
    captured from the pre-330.6 code at commit 7e8c2539 in a detached
    scratch worktree (same frozen clock, same bench: sky130 nmos1p8,
    tests/test_op_point_params.py's own _single_nmos_tb shape). Verified
    deterministic across two real runs of the capture script before being
    committed. The baseline now also includes the explicit junction stress
    coverage section.
    """
    _LIB = os.path.expanduser('~/pdk/sky130A/libs.tech/ngspice/sky130.lib.spice')

    import silk.circuit.report as report_module

    class _FrozenDatetime(__import__('datetime').datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 1, 1, 0, 0, 0)

    class _FrozenDatetimeModule:
        datetime = _FrozenDatetime

    monkeypatch.setattr(report_module, 'datetime', _FrozenDatetimeModule)

    from silk.pdk.basepdk import BasePDK
    from silk import sources
    from silk.simulation import analysis

    pdk_yaml = os.path.join(os.path.dirname(__file__), '..', 'silk', 'pdk',
                            'data', 'skywater_130_device_details.yaml')
    pdk = BasePDK(pdk_yaml, library_location=_LIB)

    dut = Module('PDK_skywater130', scratch_dir(), 'op_params_tb.cir',
                pdk_details=pdk, module_name=None)
    dut.add(pdk.nmos1p8('M1', d='vd', g='vg', s='0', b='0', w=1.0, l=0.15, m=1))
    dut.add_source(sources.VoltageSource('vd', positive='vd', negative='0', dc=1.0))
    dut.add_source(sources.VoltageSource('vg', positive='vg', negative='0', dc=1.0))
    dut.add_sim_type(analysis.Op())
    dut.simulate(corner='tt')

    circuit_report = generate_report(dut, auto_discover_config=False)
    assert circuit_report.simulation is not None
    html = circuit_report._to_html()

    golden_path = os.path.join(os.path.dirname(__file__), 'fixtures',
                               'report_html', 'no_op_capture_with_sim_golden.html')
    with open(golden_path, encoding='utf-8') as f:
        golden = f.read()
    assert html == golden


@pytest.mark.parametrize("roles, count, percentage", [
    ([], 0, None),
    ([PortRole.DIGITAL], 1, "100.0"),
    ([PortRole.DIGITAL, None], 2, "50.0"),
])
def test_annotation_report_empty_and_nonempty(tmp_path, capsys, roles, count, percentage):
    """Empty coverage is inapplicable; real ports retain coverage and findings."""
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    if roles:
        child = _sub(tmp_path, 'child', {
            f'p{i}': (PortDir.IN, role) if role else PortDir.IN
            for i, role in enumerate(roles)
        })
        top.add_submodule('child1', child)
    report = generate_report(top)
    html = report._to_html().split('<h2>Port Annotation Coverage</h2>')[1].split('<h2>')[0]
    md = report._to_markdown().split('## Port Annotation Coverage')[1].split('\n---\n')[0]
    report.print_summary()
    terminal = next(line for line in capsys.readouterr().out.splitlines()
                    if 'Annotation coverage:' in line)
    if count == 0:
        assert html.strip() == '<p><strong>No ports to annotate</strong></p>'
        assert md.strip() == '**No ports to annotate**'
        assert terminal == '  Annotation coverage: No ports to annotate'
        assert report.annotation.coverage_pct == 100.0
    else:
        assert f'1/{count} ports annotated ({percentage}%)' in html
        assert f'1/{count} ports annotated ({percentage}%)' in md
        assert terminal == (f'  Annotation coverage: 1/{count} ports ({percentage}%)'
                            f'  |  Unannotated: {count - 1}')
        if count == 1:
            assert 'All ports have a <code>PortRole</code> declared. ✅' in html
            assert 'All ports have a `PortRole` declared. ✅' in md
        else:
            assert 'child1' in html and 'p1' in html
            assert '| child1 | p1 | input | child |' in md
            assert '✅' not in html and '✅' not in md


# Location persistence uses the same typed wire contract as LintReport.
from pathlib import Path
from silk.lint.locations import (
    NetLocation, PortLocation, InstanceLocation, TerminalLocation, ModuleLocation,
)
from silk.lint.rule import Rule
from silk.lint.registry import RuleRegistry
from silk.lint.violations import InterfaceViolation, JunctionViolation
from silk.lint.severity import Severity


class _LocationRule(Rule):
    applicable_to = (object,)
    rationale = 'Exercise report persistence for registered custom rules.'

    def __init__(self, rule_id, location):
        self.rule_id = rule_id
        self.category = {'INT': 'interface', 'DOM': 'domain', 'STR': 'stress'}[rule_id[:3]]
        self.location = location

    def check(self, scope):
        kwargs = dict(rule_id=self.rule_id, severity=Severity.WARNING,
                      message=f'finding {self.rule_id}', location=self.location,
                      context={'ports': ['a', 'b'], 'limit': 1.8})
        if self.rule_id.startswith('STR'):
            yield JunctionViolation(**kwargs, device_type='nfet', measured_v=2.,
                                    rated_max=1.8, worst_time=1e-9)
        else:
            yield InterfaceViolation(**kwargs)


def test_custom_rule_location_roundtrip(tmp_path):
    locations = [
        PortLocation('vin', 'top'), InstanceLocation('u0.x1'), ModuleLocation('top'),
        NetLocation('vdd', ['vdd', 'u0.vdd']),
        TerminalLocation('u0', 'd', ('d', 's')),
        TerminalLocation('m1', 'g', ('g', 's')), InstanceLocation('m2'),
    ]
    ids = ['INT-901', 'INT-902', 'INT-903', 'DOM-901', 'DOM-902', 'STR-901', 'STR-902']
    report = generate_report(_top_with_instances(tmp_path),
                             registry=RuleRegistry(rules=[_LocationRule(r, loc)
                                 for r, loc in zip(ids, locations)]),
                             auto_discover_config=False)
    path = tmp_path / 'locations.json'
    report.save_json(path)
    restored = CircuitReport.load_json(path)
    for field in ['violations', 'domain_violations', 'junction_violations']:
        before, after = getattr(report, field), getattr(restored, field)
        assert after == before
        assert [type(v.location) for v in after] == [type(v.location) for v in before]
    wire = _json.loads(path.read_text())
    assert wire['schema_version'] == 5
    assert [v['location'] for field in ['violations', 'domain_violations', 'junction_violations']
            for v in wire[field]] == [
        {'__type': 'port', 'port_name': 'vin', 'module_name': 'top'},
        {'__type': 'instance', 'instance_path': 'u0.x1'},
        {'__type': 'module', 'module_name': 'top'},
        {'__type': 'net', 'net_name': 'vdd', 'port_names': ['vdd', 'u0.vdd']},
        {'__type': 'terminal', 'instance_path': 'u0', 'terminal': 'd', 'terminal_pair': ['d', 's']},
        {'__type': 'terminal', 'instance_path': 'm1', 'terminal': 'g', 'terminal_pair': ['g', 's']},
        {'__type': 'instance', 'instance_path': 'm2'},
    ]
    assert type(restored.domain_violations[1].location.terminal_pair) is tuple
    assert type(restored.junction_violations[0].location.terminal_pair) is tuple


def _legacy_location_wire():
    return _json.loads((Path(__file__).parent / 'fixtures/reports/circuit_report_v4_wire.json').read_text())


@pytest.mark.parametrize('field', ['violations', 'domain_violations', 'junction_violations'])
def test_legacy_structured_locations(tmp_path, field):
    wire = _legacy_location_wire()
    if field == 'junction_violations':
        location = {'instance_path': 'm1', 'terminal': 'Vgs', 'terminal_pair': ['g', 's']}
        expected = TerminalLocation('m1', 'Vgs', ('g', 's'))
    else:
        location = {'net_name': 'vdd', 'port_names': ['a', 'b']}
        expected = NetLocation('vdd', ['a', 'b'])
    wire[field] = [dict(rule_id='INT-901', severity='warning', message='legacy',
                        context={'key': 3}, location=location)]
    path = tmp_path / 'legacy.json'
    path.write_text(_json.dumps(wire))
    result = getattr(CircuitReport.load_json(path), field)[0]
    assert type(result.location) is type(expected)
    assert result.location == expected
    assert result.context == {'key': 3}


@pytest.mark.parametrize('field', ['violations', 'domain_violations', 'junction_violations'])
@pytest.mark.parametrize('location', [
    {'describe': 'port:top.vin'}, {}, None, [], 'net:vdd',
    {'__type': 'unknown'}, {'__type': []},
    {'__type': 'port', 'port_nam': 'vin'}, {'__type': 'terminal'},
    {'net_name': 'vdd'}, {'instance_path': 'm1', 'terminal': 'g'},
])
def test_invalid_location_read_has_context(tmp_path, field, location):
    wire = _legacy_location_wire()
    record = dict(rule_id='INT-901', severity='warning', message='bad', context={})
    if location is not None:
        record['location'] = location
    wire[field] = [record]
    path = tmp_path / 'invalid-location.json'
    path.write_text(_json.dumps(wire))
    with pytest.raises(ValueError) as exc:
        CircuitReport.load_json(path)
    assert str(path) in str(exc.value)
    assert f'{field}[0].location' in str(exc.value)
    if isinstance(location, dict) and 'describe' in location:
        assert 'regenerate' in str(exc.value)


@pytest.mark.parametrize('existing', [False, True])
@pytest.mark.parametrize('subclass', [False, True])
@pytest.mark.parametrize('field', ['violations', 'domain_violations', 'junction_violations'])
def test_unsupported_location_save_preserves_destination(tmp_path, existing, subclass, field):
    class UnsupportedNet(NetLocation):
        pass

    location = UnsupportedNet('vdd') if subclass else object()
    report = generate_report(_top_with_instances(tmp_path), auto_discover_config=False)
    violation_type = JunctionViolation if field == 'junction_violations' else InterfaceViolation
    setattr(report, field, [violation_type(rule_id='INT-901', severity=Severity.ERROR,
                                          message='unsupported', location=location)])
    path = tmp_path / 'unsupported.json'
    if existing:
        path.write_bytes(b'keep existing bytes')
    with pytest.raises(TypeError, match=type(location).__name__):
        report.save_json(path)
    assert path.read_bytes() == b'keep existing bytes' if existing else not path.exists()
