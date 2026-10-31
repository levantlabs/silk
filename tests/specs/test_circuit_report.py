"""CircuitReport integration with spec reconciliation."""
import os
import pytest

from silk.circuit.modules import Module
from silk.circuit.report import generate_report, CircuitReport
from silk.pdk.basepdk import BasePDK
from silk.specs import Measurement, SpecReconciliationReport


_PDK_YAML = os.path.join(
    os.path.dirname(__file__), '..', '..', 'silk', 'pdk', 'data',
    'skywater_130_device_details.yaml'
)


def _pdk():
    return BasePDK(_PDK_YAML, library_location='/fake/lib')


def _module(tmp_path, name='m'):
    return Module('PDK_skywater130', str(tmp_path), f'{name}.cir',
                  pdk_details=_pdk(), module_name=name)


def test_circuit_report_has_spec_reconciliation_field(tmp_path):
    m = _module(tmp_path)
    r = generate_report(m)
    assert hasattr(r, 'spec_reconciliation')
    # No specs declared → reconciliation is an empty report, not None.
    assert isinstance(r.spec_reconciliation, SpecReconciliationReport)
    assert r.spec_reconciliation.outcomes == []


def test_circuit_report_includes_reconciled_specs(tmp_path):
    m = _module(tmp_path)
    s = m.declare_spec(name='gain', metric='gain_dB', op='>', target=60.0)
    s.record_measurement(Measurement(source='spice', value=65.0, timestamp=100.0))
    r = generate_report(m)
    assert len(r.spec_reconciliation.outcomes) == 1
    assert r.spec_reconciliation.outcomes[0].status == 'PASS'


def test_circuit_report_reconciliation_excluded_when_no_specs_explicitly(tmp_path):
    """An empty-spec module still gets a (possibly empty) reconciliation report,
    not None — this is a consistency invariant for downstream consumers."""
    m = _module(tmp_path)
    r = generate_report(m)
    assert r.spec_reconciliation is not None


def test_report_save_markdown_includes_spec_section(tmp_path):
    m = _module(tmp_path)
    s = m.declare_spec(name='gain', metric='gain_dB', op='>', target=60.0,
                       units='dB', description='DC gain')
    s.record_measurement(Measurement(source='spice', value=65.0, timestamp=100.0))

    r = generate_report(m)
    md_path = tmp_path / 'report.md'
    r.save(str(md_path))
    text = md_path.read_text()

    assert 'Spec Reconciliation' in text
    assert 'gain' in text
    assert 'PASS' in text
    assert '65' in text            # measured value
    assert '60' in text            # target value
    assert 'dB' in text            # units


def test_report_save_markdown_omits_spec_section_when_empty(tmp_path):
    """When there are no specs, don't render an empty section header."""
    m = _module(tmp_path)
    r = generate_report(m)
    md_path = tmp_path / 'report.md'
    r.save(str(md_path))
    text = md_path.read_text()
    assert 'Spec Reconciliation' not in text


def test_report_markdown_shows_unmeasured_specs(tmp_path):
    m = _module(tmp_path)
    m.declare_spec(name='gain', metric='gain_dB', op='>', target=60.0, units='dB')
    r = generate_report(m)
    md_path = tmp_path / 'report.md'
    r.save(str(md_path))
    text = md_path.read_text()
    assert 'Spec Reconciliation' in text
    assert 'gain' in text
    assert 'UNMEASURED' in text


def test_report_markdown_shows_fail_and_margin(tmp_path):
    m = _module(tmp_path)
    s = m.declare_spec(name='gain', metric='gain_dB', op='>', target=60.0,
                       units='dB')
    s.record_measurement(Measurement(source='spice', value=55.0, timestamp=100.0))
    r = generate_report(m)
    md_path = tmp_path / 'report.md'
    r.save(str(md_path))
    text = md_path.read_text()
    assert 'FAIL' in text
    assert '55' in text


def test_report_save_html_includes_spec_section(tmp_path):
    m = _module(tmp_path)
    s = m.declare_spec(name='gain', metric='gain_dB', op='>', target=60.0,
                       units='dB')
    s.record_measurement(Measurement(source='spice', value=65.0, timestamp=100.0))

    r = generate_report(m)
    html_path = tmp_path / 'report.html'
    r.save_html(str(html_path))
    text = html_path.read_text()

    assert 'Spec Reconciliation' in text
    assert '<table' in text
    assert 'gain' in text
    assert 'PASS' in text
    assert 'pass' in text.lower()  # CSS class or status word


def test_report_html_highlights_fail_status(tmp_path):
    m = _module(tmp_path)
    s = m.declare_spec(name='gain', metric='gain_dB', op='>', target=60.0,
                       units='dB')
    s.record_measurement(Measurement(source='spice', value=55.0, timestamp=100.0))
    r = generate_report(m)
    html_path = tmp_path / 'report.html'
    r.save_html(str(html_path))
    text = html_path.read_text()
    assert 'FAIL' in text


def test_report_save_json_includes_spec_reconciliation(tmp_path):
    import json
    m = _module(tmp_path)
    s = m.declare_spec(name='gain', metric='gain_dB', op='>', target=60.0,
                       units='dB')
    s.record_measurement(Measurement(source='spice', value=65.0, timestamp=100.0))
    r = generate_report(m)
    json_path = tmp_path / 'report.json'
    r.save_json(str(json_path))
    with open(json_path) as f:
        data = json.load(f)
    assert 'spec_reconciliation' in data
    recon = data['spec_reconciliation']
    assert len(recon['outcomes']) == 1
    assert recon['outcomes'][0]['status'] == 'PASS'
    assert recon['outcomes'][0]['spec']['name'] == 'gain'
    assert recon['outcomes'][0]['measurement']['value'] == 65.0


def test_report_json_empty_reconciliation_serialized(tmp_path):
    """Empty reconciliation still appears in JSON as an empty outcomes list."""
    import json
    m = _module(tmp_path)
    r = generate_report(m)
    json_path = tmp_path / 'report.json'
    r.save_json(str(json_path))
    with open(json_path) as f:
        data = json.load(f)
    assert 'spec_reconciliation' in data
    assert data['spec_reconciliation']['outcomes'] == []


from silk.circuit.report_diff import diff_reports


def test_diff_reports_surfaces_pass_to_fail_transition(tmp_path):
    m = _module(tmp_path)
    s = m.declare_spec(name='gain', metric='gain_dB', op='>', target=60.0,
                       units='dB')
    s.record_measurement(Measurement(source='spice', value=65.0, timestamp=100.0))
    r_before = generate_report(m)

    # Simulate a later run where gain drops below target.
    s.record_measurement(Measurement(source='spice', value=55.0, timestamp=200.0))
    r_after = generate_report(m)

    diff = diff_reports(r_before, r_after)
    # Look for a spec transition entry describing PASS -> FAIL on 'gain'.
    assert hasattr(diff, 'spec_transitions') or hasattr(diff, 'specs_changed')
    # The actual attribute name depends on how the existing ReportDiff is shaped.
    # Use whatever the existing diff exposes and assert the gain transition
    # is visible in its string representation.
    text = str(diff) if hasattr(diff, '__str__') else repr(diff)
    assert 'gain' in text
    assert 'PASS' in text
    assert 'FAIL' in text


def test_diff_reports_surfaces_measurement_value_delta(tmp_path):
    m = _module(tmp_path)
    s = m.declare_spec(name='gain', metric='gain_dB', op='>', target=60.0,
                       units='dB')
    s.record_measurement(Measurement(source='spice', value=65.0, timestamp=100.0))
    r_before = generate_report(m)

    s.record_measurement(Measurement(source='spice', value=70.0, timestamp=200.0))
    r_after = generate_report(m)

    diff = diff_reports(r_before, r_after)
    text = str(diff) if hasattr(diff, '__str__') else repr(diff)
    assert 'gain' in text
    # Before and after values should both appear.
    assert '65' in text
    assert '70' in text


_LAST_RUN_MESSAGE = (
    'Results from the last run. Rerun the test after changing the circuit '
    'or stimulus to update these results.'
)


@pytest.mark.parametrize('values', [[], [None], [65.0, 55.0, None]])
def test_last_run_disclosure_preserves_recorded_results(tmp_path, capsys, values):
    m = _module(tmp_path)
    for i, value in enumerate(values):
        spec = m.declare_spec(name=f'gain{i}', metric='gain_dB', op='>',
                              target=60.0, units='dB')
        if value is not None:
            spec.record_measurement(Measurement(
                source='spice', value=value, timestamp=100.0))
    r = generate_report(m)
    json_path = tmp_path / 'before.json'
    r.save_json(str(json_path))
    before = json_path.read_text()
    outcomes_before = [(o.status, o.measurement) for o in r.spec_reconciliation.outcomes]

    md_path = tmp_path / 'report.md'
    html_path = tmp_path / 'report.html'
    r.save(str(md_path))
    r.save_html(str(html_path))
    r.print_summary()
    outputs = [capsys.readouterr().out, md_path.read_text(), html_path.read_text()]
    for text in outputs:
        assert text.count(_LAST_RUN_MESSAGE) == int(any(v is not None for v in values))
        if any(v is not None for v in values):
            assert text.index(_LAST_RUN_MESSAGE) < text.index('gain0')
            assert '65' in text and '55' in text
            assert 'PASS' in text and 'FAIL' in text and 'UNMEASURED' in text
    r.save_json(str(json_path))
    assert json_path.read_text() == before
    assert [(o.status, o.measurement) for o in r.spec_reconciliation.outcomes] == outcomes_before
    assert _LAST_RUN_MESSAGE not in before


def test_placement_labels_in_all_renderers_and_saved_identity(tmp_path):
    from silk.specs import SpecOutcome
    m = _module(tmp_path)
    spec = m.declare_spec(name='gain', metric='gain', op='>', target=1.)
    report = generate_report(m)
    paths = [None, '$', '$/u0', '$/u0/u1', '$/u1', '$/u%2F0/x%25']
    report.spec_reconciliation.outcomes = [
        SpecOutcome(spec, None, 'UNMEASURED', owner_path=path)
        for path in paths]
    for rendered in (report._render_spec_reconciliation_html(),
                     report._render_spec_reconciliation_md(),
                     report._render_metrics_terminal()):
        assert '$' not in rendered
        for label in ('gain (instance u0)', 'gain (instance u0/u1)',
                      'gain (instance u1)', 'gain (instance u%2F0/x%25)'):
            assert label in rendered
    summary = report.spec_reconciliation.summary()
    assert '$' not in summary
    assert all(label in summary for label in ('(instance u0)', '(instance u0/u1)', '(instance u1)'))
    path = tmp_path / 'placements.json'
    report.save_json(path)
    loaded = CircuitReport.load_json(path)
    assert [o.owner_path for o in loaded.spec_reconciliation.outcomes] == paths
    assert loaded._render_spec_reconciliation_md() == report._render_spec_reconciliation_md()
