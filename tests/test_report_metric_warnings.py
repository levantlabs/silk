"""TASK-421: SpecReport.metric_warnings reach the CircuitReport JSON, HTML and
text summary, beside the measurement they explain."""
import json
import math

from silk.circuit.report import CircuitReport, generate_report
from silk.circuit.spec_report import SpecReport, SpecResult
from silk.metrics.diagnostics import MetricWarningRecord
from silk.specs import Measurement, Spec, SpecOutcome, SpecReconciliationReport
from tests.test_circuit_report import _top_with_instances

HOSTILE = '</script><b>"x"&\'y\'</b>\nline2\x1b[31mred'


def _rec(**kw):
    base = dict(metric='gain_db', code='no_crossing', message='never crosses',
                details={'level': 0.5, 'n': math.nan}, suggestion='widen sweep',
                output='gain')
    base.update(kw)
    return MetricWarningRecord(**base)


def _sr(test, warnings, key='g', **kw):
    sr = SpecReport(test, 'top', {}, metric_warnings={'gain': warnings}, **kw)
    sr._results.append(SpecResult('g', 'gain', '>=', 20.0, math.nan, False,
                                  spec_key=key))
    return sr


def _build(tmp_path, value, warnings, reports=None):
    top = _top_with_instances(tmp_path)
    sr = reports if reports is not None else [_sr('ac', warnings)]
    rep = generate_report(top, spec_reports=sr)
    spec = Spec(name='g', module=top, metric='gain', op='>=', target=20.0)
    m = Measurement(source='sim', value=value)
    rep.spec_reconciliation = SpecReconciliationReport(
        module=top, timestamp=1.0, calibrations_applied=[],
        outcomes=[SpecOutcome(spec=spec, measurement=m,
                              status='FAIL' if value != value else 'PASS')])
    return rep


def test_round_trip_strict_json_and_order(tmp_path):
    recs = (_rec(code='a'), _rec(code='b'))
    rep = _build(tmp_path, math.nan, recs)
    p = tmp_path / 'r.json'
    rep.save_json(str(p))
    json.loads(p.read_text(), parse_constant=lambda c: (_ for _ in ()).throw(
        AssertionError(c)))
    back = CircuitReport.load_json(str(p))
    sec = back.metric_warnings[0]
    assert len(back.metric_warnings) == 1
    assert [w.to_dict()['code'] for w in sec.warnings] == ['a', 'b']
    assert sec.warnings[0].details['level'] == 0.5
    assert (sec.test_name, sec.output) == ('ac', 'gain')
    assert [w.code for w in sec.warnings] == ['a', 'b']
    assert math.isnan(sec.warnings[0].details['n'])


def test_no_warnings_is_unchanged(tmp_path):
    top = _top_with_instances(tmp_path)
    plain = generate_report(top)
    clean = generate_report(top, spec_reports=[SpecReport('ac', 'top', {})])
    assert clean.metric_warnings == []
    a, b = tmp_path / 'a.json', tmp_path / 'b.json'
    plain.save_json(str(a)); clean.save_json(str(b))
    ja, jb = json.loads(a.read_text()), json.loads(b.read_text())
    assert 'metric_warnings' not in jb
    assert ja.pop('test_results') is None
    evidence = jb.pop('test_results')
    assert evidence['kind'] == 'records'
    assert evidence['entries'][0]['test_name'] == 'ac'
    assert evidence['entries'][0]['metric_warnings'] == {}
    for j in (ja, jb):
        j.pop('generated_at'); j['spec_reconciliation'].pop('timestamp')
    assert ja == jb
    clean.generated_at = plain.generated_at
    clean.spec_reconciliation.timestamp = plain.spec_reconciliation.timestamp
    assert clean._to_html() == plain._to_html()
    old = CircuitReport.load_json(str(a))
    assert old.metric_warnings == []


def test_html_shows_reason_beside_nan_and_note_on_finite(tmp_path):
    nan_html = _build(tmp_path, math.nan, (_rec(),))._to_html()
    assert 'unavailable: [no_crossing' not in nan_html
    assert 'unavailable: no_crossing: never crosses' in nan_html
    fin = _build(tmp_path, 25.0, (_rec(),))
    html = fin._to_html()
    assert 'note: no_crossing' in html and 'unavailable:' not in html
    assert fin.spec_reconciliation.outcomes[0].status == 'PASS'


def test_same_metric_from_two_tests_names_each(tmp_path):
    reps = [_sr('ac1', (_rec(),), corner='tt'), _sr('ac1', (_rec(),), corner='ss'),
            _sr('ac2', (_rec(),))]
    html = _build(tmp_path, math.nan, None, reps)._to_html()
    assert '[ac1 $ tt] no_crossing' in html and '[ac1 $ ss] no_crossing' in html
    assert '[ac2 $] no_crossing' in html


def test_warning_of_another_block_does_not_show_on_this_row(tmp_path):
    reps = [_sr('ac', (_rec(),), key='other_gain')]
    rep = _build(tmp_path, math.nan, None, reps)
    assert 'no_crossing' not in rep._to_html()
    assert 'no_crossing' not in rep._to_markdown()


def test_markdown_notes_are_escaped(tmp_path):
    md = _build(tmp_path, math.nan, (_rec(message=HOSTILE),))._to_markdown()
    assert '</script><b>' not in md and '&lt;/script&gt;&lt;b&gt;' in md


def test_hostile_strings_escaped_in_html(tmp_path):
    rec = _rec(code=HOSTILE, message=HOSTILE, suggestion=HOSTILE,
               details={HOSTILE: HOSTILE})
    html = _build(tmp_path, math.nan, (rec,))._to_html()
    assert '</script><b>' not in html and '\x1b' not in html
    assert '&lt;/script&gt;&lt;b&gt;' in html
    assert '&quot;x&quot;&amp;' in html or '&quot;x&quot;&amp;' in html.replace('&#x27;', "'")


def test_spec_report_summary_shows_reason():
    sr = SpecReport('ac', 'top', {'gain': math.nan},
                    metric_warnings={'gain': (_rec(message=HOSTILE),)})
    sr.check('g', 'gain', '>=', 20.0)
    text = sr.summary()
    assert 'unavailable: no_crossing' in text and '\x1b' not in text
    assert all('line2' not in l or 'unavailable' in l for l in text.splitlines())
    sr2 = SpecReport('ac', 'top', {'gain': 30.0},
                     metric_warnings={'gain': (_rec(),)})
    sr2.check('g', 'gain', '>=', 20.0)
    assert 'note: no_crossing' in sr2.summary() and sr2.passed


def test_complex_refusal_keeps_warning_attribution_and_export(tmp_path):
    rec = _rec(metric='average', code='nan_waveform', output='mean')
    sr = SpecReport('ac', 'top', {'mean': complex(math.nan, 1)},
                    metric_warnings={'mean': (rec,)})
    sr.check('mean bound', 'mean', '<', 2, spec_key='mean-bound')
    assert sr.metric_warnings == {'mean': (rec,)}
    assert sr.metric_warnings['mean'][0] is rec
    assert 'complex' in sr.summary() and 'nan_waveform' in sr.summary()
    assert isinstance(sr.metrics['mean'], complex)
    report = generate_report(_top_with_instances(tmp_path), spec_reports=[sr])
    path = tmp_path / 'complex-report.json'
    report.save_json(str(path))
    restored = CircuitReport.load_json(str(path))
    section, = restored.metric_warnings
    assert section.output == 'mean'
    assert section.spec_keys == ('mean-bound',)
    assert section.warnings[0].metric == 'average'
    assert section.warnings[0].output == 'mean'
    assert section.warnings[0].code == 'nan_waveform'
    assert restored._to_html()


# TASK-514: shared declarations must not join warnings across placements.
import pytest
from pathlib import Path


@pytest.mark.parametrize('restored', [False, True])
@pytest.mark.parametrize('collection', [False, True])
def test_shared_placement_warning_identity(tmp_path, restored, collection):
    from silk.metrics.diagnostics import emit, MetricWarning
    from tests.circuit.test_selected_testbench import leaf, root
    child, tb = leaf(tmp_path)
    top = root(tmp_path)
    missing = True

    def gain(_):
        if missing:
            emit(metric='gain', code='audit_missing',
                 message='x1 cannot be measured', details={})
            return math.nan
        return 70.

    tb.add_metric('gain', gain)
    tb.add_spec('Gain limit', 'gain', '>=', 60, key='gain')
    raw = Path(__file__).parent / 'fixtures/rawfiles/e_ng_tran.raw'
    try:
        for name in ('x1', 'x2'):
            top.add_module(child, name, out=name, vss='0')
        with pytest.warns(MetricWarning):
            first = top.get_test('step', instance='x1').evaluate(rawfile=raw)
        missing = False
        second = top.get_test('step', instance='x2').evaluate(rawfile=raw)
        assert not first.passed and second.passed
        assert len(first.metric_warnings['gain']) == 1
        assert not second.metric_warnings
        reports = [first, second]
        if restored:
            reports = [SpecReport.from_dict(sr.to_dict()) for sr in reports]
        kwargs = ({'test_results': {sr.report_key: sr for sr in reports}}
                  if collection else {'spec_reports': reports})
        report = generate_report(top, **kwargs)
        for loaded in (False, True):
            if loaded:
                report.save_json(tmp_path / 'placement.json')
                report = CircuitReport.load_json(tmp_path / 'placement.json')
            # Test attribution before the new fields, so the old failure is visible.
            html = report._render_spec_reconciliation_html()
            fail_row = html.split('<tr class="spec-fail">')[1].split('</tr>')[0]
            pass_row = html.split('<tr class="spec-pass">')[1].split('</tr>')[0]
            assert 'audit_missing' in fail_row
            assert 'audit_missing' not in pass_row
            assert '(instance x1)' in fail_row and '(instance x2)' in pass_row
            md = report._render_spec_reconciliation_md()
            fail_md = next(line for line in md.splitlines() if '| FAIL |' in line)
            pass_md = next(line for line in md.splitlines() if '| PASS |' in line)
            assert '(instance x1)' in fail_md and 'audit_missing' in fail_md
            assert '(instance x2)' in pass_md and 'audit_missing' not in pass_md
            terminal = report._render_metrics_terminal()
            assert terminal.count('audit_missing') == 1
            assert '(instance x1)' in terminal and '(instance x2)' in terminal
            rows = report.spec_reconciliation.as_dict()['outcomes']
            assert [row['owner_path'] for row in rows] == ['$/x1', '$/x2']
            assert [row['report_key'] for row in rows] == [
                '$/x1:leaf.step', '$/x2:leaf.step']
            assert [row['spec']['key'] for row in rows] == ['gain', 'gain']
            # JSON warning association is structured rather than duplicated prose.
            for row, count in zip(rows, (1, 0)):
                sections = [s for s in report.metric_warnings
                            if s.owner_path == row['owner_path']
                            and s.output == row['spec']['metric']
                            and row['spec']['key'] in s.spec_keys]
                assert sum(len(s.warnings) for s in sections) == count
    finally:
        top.close()
        child.close()


@pytest.mark.parametrize('owner', [None, '$', '$/x', '$/x1/nested', '$/x1'])
def test_warning_owner_is_not_a_wildcard(tmp_path, owner):
    rep = _build(tmp_path, math.nan, None,
                 [_sr('ac', (_rec(),), owner_path='$/x1')])
    rep.spec_reconciliation.outcomes[0].owner_path = owner
    for text in (rep._to_html(), rep._to_markdown(), rep._render_metrics_terminal()):
        assert ('no_crossing' in text) == (owner == '$/x1')


def test_placement_labels_are_escaped(tmp_path):
    rep = _build(tmp_path, math.nan, ())
    outcome = rep.spec_reconciliation.outcomes[0]
    outcome.owner_path = '$/<b>x|y</b>\n\x1b[31m'
    for text in (rep._render_spec_reconciliation_html(),
                 rep._render_spec_reconciliation_md()):
        assert '<b>x' not in text and '&lt;b&gt;x' in text
        assert '\x1b' not in text
    assert 'x\\|y' in rep._render_spec_reconciliation_md()
    assert '\x1b' not in rep._render_metrics_terminal()


def test_evm_coverage_warning_is_attributed_and_round_trips(tmp_path):
    from pathlib import Path
    import numpy as np
    from silk.metrics import evm_percent, modulation_grid
    from silk.metrics.diagnostics import MetricWarning
    from silk.signals.base import Signal
    from tests.circuit.test_selected_testbench import leaf, root

    child, tb = leaf(tmp_path)
    top = root(tmp_path)
    symbols = modulation_grid('64qam')[:-1]
    t = np.arange(symbols.size * 4) / 4
    i = Signal('i', 'v', 'time', 'V', t, np.repeat(symbols.real, 4))
    q = Signal('q', 'v', 'time', 'V', t, np.repeat(symbols.imag, 4))
    tb.add_metric('evm', lambda _: evm_percent(i, q, symbol_period=1., modulation='64qam'))
    raw = Path(__file__).parent / 'fixtures/rawfiles/e_ng_tran.raw'
    try:
        top.add_module(child, 'stage', out='out', vss='0')
        with pytest.warns(MetricWarning, match='incomplete_constellation') as caught:
            report = top.get_test('step', instance='stage').evaluate(rawfile=raw)
        assert len(caught) == 1
        assert report.metrics['evm'] == pytest.approx(0., abs=1e-10)
        records = report.metric_warnings['evm']
        assert len(records) == 1
        assert (records[0].metric, records[0].code, records[0].output) == (
            'evm_constellation', 'incomplete_constellation', 'evm')
        assert records[0].details == dict(modulation='64qam', n_symbols=63,
                                         covered_points=63, total_points=64,
                                         proximity_fraction=.5)
        restored = SpecReport.from_dict(json.loads(json.dumps(report.to_dict())))
        assert restored.metric_warnings == report.metric_warnings
        assert restored.owner_path == '$/stage'
        assert restored.report_key == report.report_key
    finally:
        top.close()
        child.close()


@pytest.mark.parametrize('weighting', ['binary', 'thermometer'])
def test_unavailable_decoded_zero_cannot_pass_spec(tmp_path, weighting):
    import numpy as np
    from silk.signals import Signal, codes_from_bits
    from silk.metrics import value_at
    from silk.metrics.diagnostics import MetricWarning
    from tests.circuit.test_selected_testbench import leaf, root

    child, tb = leaf(tmp_path)
    top = root(tmp_path)
    bits = [Signal('d0', 'voltage', 'time', 'V', [0., 1.], [np.nan, 0.]),
            Signal('d1', 'voltage', 'time', 'V', [0., 1.], [0., 0.])]

    def decoded_zero(_):
        code = codes_from_bits(bits, weighting=weighting,
                               **({'msb_first': False} if weighting == 'binary' else {}))
        return value_at(code, 0.)

    tb.add_metric('zero', decoded_zero)
    tb.add_spec('Expected zero', 'zero', '==', 0., key='zero')
    top.add_module(child, 'stage', out='out', vss='0')
    raw = Path(__file__).parent / 'fixtures/rawfiles/e_ng_tran.raw'
    try:
        with pytest.warns(MetricWarning):
            report = top.get_test('step', instance='stage').evaluate(rawfile=raw)
        assert not report.passed
        assert np.isnan(report.metrics['zero'])
        records = report.metric_warnings['zero']
        decoder_records = [r for r in records if r.code == 'nan_bits']
        assert len(decoder_records) == 1
        assert decoder_records[0].details['affected_sample_count'] == 1
        assert decoder_records[0].output == 'zero'
        bits[0].y[0] = 0.
        clean = top.get_test('step', instance='stage').evaluate(rawfile=raw)
        assert clean.metrics['zero'] == 0. and clean.passed
        assert not clean.metric_warnings
    finally:
        top.close()
        child.close()
