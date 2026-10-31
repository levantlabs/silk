"""Failed latest evidence survives all report renderers and JSON restoration."""
import pytest
from silk.circuit.report import CircuitReport, generate_report
from silk.specs import Measurement
from tests.specs.test_testbench_integration import _module
from tests.specs.test_latest_attempt import failure


@pytest.mark.parametrize('previous', [None, 50., 70.])
def test_selected_attempts_roundtrip_and_render_without_reports(tmp_path, previous):
    mod = _module(tmp_path)
    spec = mod.declare_spec('Gain', 'gain', '>=', 60., units='dB')
    if previous is not None:
        original = Measurement('spice', previous, timestamp=100.)
        spec.record_measurement(original)
    failure(spec)
    recon = mod.reconcile_specs()
    report = generate_report(mod)
    path = tmp_path / 'report.json'
    report.save_json(path)
    restored = CircuitReport.load_json(path)
    if previous is not None:
        original.value = 999.
    for candidate in (report, restored):
        row = candidate.spec_reconciliation.outcomes[0]
        assert row.status == 'FAIL' and row.measurement is None
        for rendered in (candidate.spec_reconciliation.summary(),
                         candidate._render_metrics_terminal(),
                         candidate._render_spec_reconciliation_html(),
                         candidate._render_spec_reconciliation_md()):
            assert 'latest attempt failed' in rendered
            assert 'missing gain' in rendered
            assert '1970-' in rendered
            if previous is None:
                assert 'no successful attempt' in rendered
            else:
                assert 'last successful attempt' in rendered
                assert str(int(previous)) in rendered
        wire = row.as_dict()
        assert 'latest_attempt' in wire and 'last_successful_attempt' in wire
        assert 'attempt_history' not in wire


def test_failure_diagnostics_escape_and_diff_keeps_distinct_attempts(tmp_path):
    from silk.specs import SpecAttempt
    from silk.circuit.report_diff import diff_reports
    mod = _module(tmp_path)
    spec = mod.declare_spec('Gain', 'gain', '>=', 60.)
    reason = '<script>bad</script>|next\nrow\x1b'
    spec.record_attempt(SpecAttempt(source='spice', kind='extractor_error',
        reason=reason, timestamp=100., attempt_id='one'))
    first = generate_report(mod)
    html = first._render_spec_reconciliation_html()
    assert '<script>' not in html and '&lt;script&gt;' in html
    md = first._render_spec_reconciliation_md()
    assert '<script>' not in md and '\\|next' in md
    assert '\x1b' not in first._render_metrics_terminal()
    spec.record_attempt(SpecAttempt(source='spice', kind='extractor_error',
        reason=reason, timestamp=100., attempt_id='two'))
    later = generate_report(mod)
    transitions = diff_reports(first, later).spec_transitions
    assert len(transitions) == 1
    assert transitions[0]['before_status'] == transitions[0]['after_status'] == 'FAIL'
    assert spec.latest_attempt().attempt_id == 'two'
