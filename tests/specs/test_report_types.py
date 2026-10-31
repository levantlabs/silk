"""Dataclasses for spec reconciliation reporting."""
from silk.specs import (
    Measurement,
    Spec,
    SpecOutcome,
    SpecReconciliationReport,
)


def _spec(name='lna_NF', target=2.0, op='<='):
    return Spec(name=name, module=None, metric='noise_figure',
                op=op, target=target)


def test_spec_outcome_defaults():
    s = _spec()
    oc = SpecOutcome(spec=s, measurement=None, status='UNMEASURED')
    assert oc.spec is s
    assert oc.measurement is None
    assert oc.status == 'UNMEASURED'
    assert oc.margin is None
    assert oc.source_used is None
    assert oc.calibrated is False
    assert oc.notes == ''


def test_spec_outcome_pass_with_margin():
    s = _spec(target=2.0)
    m = Measurement(source='spice', value=1.8)
    oc = SpecOutcome(spec=s, measurement=m, status='PASS',
                     margin=0.2, source_used='spice')
    assert oc.status == 'PASS'
    assert oc.margin == 0.2
    assert oc.source_used == 'spice'


def test_reconciliation_report_empty():
    r = SpecReconciliationReport(module=None)
    assert r.outcomes == []
    assert r.calibrations_applied == []
    assert r.pass_count == 0
    assert r.fail_count == 0
    assert r.unmeasured == []


def test_reconciliation_report_counts():
    s1 = _spec('a', target=2.0)
    s2 = _spec('b', target=2.0)
    s3 = _spec('c', target=2.0)
    r = SpecReconciliationReport(
        module=None,
        outcomes=[
            SpecOutcome(spec=s1, measurement=Measurement('spice', 1.8),
                        status='PASS', margin=0.2, source_used='spice'),
            SpecOutcome(spec=s2, measurement=Measurement('spice', 2.5),
                        status='FAIL', margin=0.5, source_used='spice'),
            SpecOutcome(spec=s3, measurement=None, status='UNMEASURED'),
        ],
    )
    assert r.pass_count == 1
    assert r.fail_count == 1
    assert len(r.unmeasured) == 1
    assert r.unmeasured[0].spec is s3


def test_reconciliation_counts_are_properties_not_methods():
    """fail_count/pass_count/unmeasured read as nouns (no args, pure derivations),
    so callers write ``recon.fail_count == 0``. If they are METHODS, that compares
    a bound method to an int — always False — and silently reports FAIL for a run
    that passed, with nothing raised. Guard against re-introduction."""
    r = SpecReconciliationReport(
        module=None,
        outcomes=[
            SpecOutcome(spec=_spec('a', target=2.0),
                        measurement=Measurement('spice', 1.8),
                        status='PASS', margin=0.2, source_used='spice'),
        ],
    )
    assert r.fail_count == 0        # real int comparison, not a bound method
    assert r.pass_count == 1
    assert r.unmeasured == []
    assert r.passed is True


def test_reconciliation_passed_false_on_failure_or_unmeasured():
    """passed is False on any FAIL, and UNMEASURED does not count as a pass —
    silently treating an unmeasured spec as passing is the same class of bug."""
    fail = SpecOutcome(spec=_spec('f'), measurement=Measurement('spice', 3.0),
                       status='FAIL', margin=1.0, source_used='spice')
    unm = SpecOutcome(spec=_spec('u'), measurement=None, status='UNMEASURED')
    assert SpecReconciliationReport(module=None, outcomes=[fail]).passed is False
    assert SpecReconciliationReport(module=None, outcomes=[unm]).passed is False


def test_reconciliation_report_as_dict_roundtrip_shape():
    s1 = _spec('a', target=2.0)
    r = SpecReconciliationReport(
        module=None,
        outcomes=[
            SpecOutcome(spec=s1, measurement=Measurement('spice', 1.8),
                        status='PASS', margin=0.2, source_used='spice'),
        ],
        timestamp=1000.0,
    )
    d = r.as_dict()
    assert d['timestamp'] == 1000.0
    assert d['calibrations_applied'] == []
    assert len(d['outcomes']) == 1
    assert d['outcomes'][0]['spec']['name'] == 'a'
    assert d['outcomes'][0]['status'] == 'PASS'
    assert d['outcomes'][0]['measurement']['value'] == 1.8


def test_reconciliation_report_summary_is_human_readable():
    s1 = _spec('a')
    r = SpecReconciliationReport(
        module=None,
        outcomes=[
            SpecOutcome(spec=s1, measurement=Measurement('spice', 1.8),
                        status='PASS', margin=0.2, source_used='spice'),
        ],
    )
    text = r.summary()
    assert 'a' in text
    assert 'PASS' in text


def test_outcome_identity_serialization_and_summary():
    spec = Spec(name='display', module=None, metric='gain', op='>=', target=60,
                key='stable')
    row = SpecOutcome(spec, Measurement('user', 70), 'PASS',
                      owner_path='$/x1', report_key='$/x1:leaf.gain')
    wire = row.as_dict()
    assert wire['spec']['key'] == 'stable'
    assert wire['owner_path'] == '$/x1'
    assert wire['report_key'] == '$/x1:leaf.gain'
    report = SpecReconciliationReport(None, [row, SpecOutcome(
        spec, None, 'UNMEASURED', owner_path='$/x2')])
    summary = report.summary()
    assert '[PASS] display (instance x1): 70 >=60' in summary
    assert '[UNMEASURED] display (instance x2): (no measurement)' in summary


def test_placement_display_hides_internal_root_and_preserves_identity():
    spec = _spec('gain')
    paths = [None, '$', '$/u0', '$/u0/u1', '$/u1', '$/u%2F0/x%25']
    labels = ['', '', ' (instance u0)', ' (instance u0/u1)',
              ' (instance u1)', ' (instance u%2F0/x%25)']
    outcomes = []
    for path, label in zip(paths, labels):
        outcome = SpecOutcome(spec, None, 'UNMEASURED', owner_path=path)
        assert outcome.placement_label == label
        assert outcome.display_label == 'gain' + label
        assert outcome.owner_path == path
        assert outcome.as_dict()['owner_path'] == path
        outcomes.append(outcome)
    summary = SpecReconciliationReport(None, outcomes).summary()
    assert '$' not in summary
    assert '(instance u0)' in summary and '(instance u1)' in summary
    assert '(instance u0/u1)' in summary

    for label in labels:
        assert f'[UNMEASURED] gain{label}: (no measurement)' in summary
