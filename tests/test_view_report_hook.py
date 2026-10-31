"""Report hook tests for CircuitReport.add_waveform_view (design spec §8.3, §9.6).

The auto-populated `context=` tests below (TestAutoContext*) are for the
view-context-and-scopes design §2.2.
"""
import dataclasses
import json
import os
from pathlib import Path

import pytest

from silk.circuit.interface_check import AnnotationCoverage
from silk.circuit.report import CircuitReport, InstanceSummary, TopologySummary
from silk.specs.report import SpecOutcome, SpecReconciliationReport
from silk.specs.spec import Spec

FIXTURES = Path(__file__).parent / 'fixtures' / 'rawfiles'
A_TRAN = FIXTURES / 'a_tran_bin.raw'

# Golden output of _to_html() for a small fixture report with NO waveform
# views, captured before this feature's rendering hook was added. The §8.3
# guard (`if self.waveform_views:`) must keep this byte-identical forever.
_GOLDEN_HTML_NO_VIEWS = (
    '<!DOCTYPE html>\n<html>\n<head>\n<meta charset="utf-8">\n'
    '<title>Circuit Report: golden_mod</title>\n<style>\n'
    '  body { font-family: system-ui, sans-serif; max-width: 960px; margin: 40px auto; padding: 0 20px; color: #222; }\n'
    '  h1 { border-bottom: 2px solid #333; padding-bottom: 8px; }\n'
    '  h2 { border-bottom: 1px solid #ddd; margin-top: 2em; }\n'
    '  table { border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.9em; }\n'
    '  th, td { border: 1px solid #ddd; padding: 6px 12px; text-align: left; }\n'
    '  th { background: #f5f5f5; }\n'
    '  .badge { display: inline-block; padding: 4px 12px; border-radius: 4px; font-weight: bold; }\n'
    '  .pass { background: #d4edda; color: #155724; }\n'
    '  .fail { background: #f8d7da; color: #721c24; }\n'
    '  .warn { background: #fff3cd; color: #856404; }\n'
    '  tr.spec-pass td.status { background: #d4edda; color: #155724; font-weight: bold; }\n'
    '  tr.spec-fail td.status { background: #f8d7da; color: #721c24; font-weight: bold; }\n'
    '  tr.spec-unmeasured td.status { background: #e2e3e5; color: #383d41; font-weight: bold; }\n'
    '  details summary { cursor: pointer; color: #555; padding: 4px 0; }\n'
    '  img { max-width: 100%; margin: 1em 0; border: 1px solid #eee; }\n'
    '  .fig-title { font-weight: bold; margin-top: 1.5em; }\n'
    '</style>\n</head>\n<body>\n'
    '<h1>Circuit Report: golden_mod</h1>\n'
    '<p>Generated: 2026-01-01T00:00:00</p>\n'
    '<h2>Circuit Topology</h2>\n'
    '<p><strong>Module:</strong> golden_mod<br>\n'
    '<strong>Subcircuit instances:</strong> 1<br>\n'
    '<strong>Total nets:</strong> 3</p>\n'
    '<table><thead><tr><th>Instance</th><th>Subcircuit</th><th>Ports</th></tr></thead><tbody>\n'
    '<tr><td>m1</td><td>nmos</td><td>4</td></tr>\n'
    '</tbody></table>\n'
    '<h2>Port Annotation Coverage</h2>\n'
    '<p><strong>4/4 ports annotated (100.0%)</strong></p>\n'
    '<p>All ports have a <code>PortRole</code> declared. ✅</p>\n'
    '<h2>Interface Check</h2>\n'
    '<p><span class="badge pass">PASS</span></p>\n'
    '</body>\n</html>'
)


def _make_fixture_report():
    return CircuitReport(
        module_name='golden_mod',
        generated_at='2026-01-01T00:00:00',
        topology=TopologySummary(
            module_name='golden_mod',
            instances=[InstanceSummary(instance='m1', module_name='nmos', port_count=4)],
            total_nets=3,
        ),
        annotation=AnnotationCoverage(total=4, annotated=4, unannotated_ports=[]),
        violations=[],
        simulation=None,
    )


class StubTestbench:
    """Duck-typed stand-in for a simulated Module — only needs .results_file
    and .module_name, per spec §8.1/§8.3."""
    def __init__(self, results_file, module_name='stub_mod'):
        self.results_file = results_file
        self.module_name = module_name


class TestNoViewsByteIdentical:
    def test_to_html_unchanged_without_waveform_views(self):
        report = _make_fixture_report()
        assert report._to_html(base_dir='/tmp') == _GOLDEN_HTML_NO_VIEWS


class TestAddWaveformView:
    def test_with_path_source(self, tmp_path):
        report = _make_fixture_report()
        out_path = tmp_path / 'my_view.html'
        report.add_waveform_view(str(A_TRAN), out_path=str(out_path))
        assert len(report.waveform_views) == 1
        html_path, label = report.waveform_views[0]
        assert os.path.exists(html_path)
        assert label  # non-empty

    def test_with_results_file_stub(self, tmp_path):
        report = _make_fixture_report()
        stub = StubTestbench(str(A_TRAN), module_name='my_mod')
        out_path = tmp_path / 'my_mod_view.html'
        report.add_waveform_view(stub, out_path=str(out_path))
        html_path, label = report.waveform_views[0]
        assert os.path.exists(html_path)
        assert label == 'my_mod'

    def test_default_out_path_uses_module_name(self, tmp_path):
        report = _make_fixture_report()
        rawfile_copy = tmp_path / 'rawfile.raw'
        rawfile_copy.write_bytes(A_TRAN.read_bytes())
        stub = StubTestbench(str(rawfile_copy), module_name='auto_named')
        report.add_waveform_view(stub)
        html_path, label = report.waveform_views[0]
        assert os.path.basename(html_path) == 'auto_named_view.html'

    def test_repr_and_eq_unaffected_by_populated_waveform_views(self, tmp_path):
        report = _make_fixture_report()
        baseline_repr = repr(report)
        baseline_copy = dataclasses.replace(report)
        report.add_waveform_view(str(A_TRAN), out_path=str(tmp_path / 'v.html'))
        assert repr(report) == baseline_repr
        assert report == baseline_copy

    def test_save_json_omits_waveform_views(self, tmp_path):
        report = _make_fixture_report()
        report.add_waveform_view(str(A_TRAN), out_path=str(tmp_path / 'v.html'))
        json_path = tmp_path / 'report.json'
        report.save_json(str(json_path))
        with open(json_path) as f:
            d = json.load(f)
        assert 'waveform_views' not in d

    def test_save_html_relative_href_for_sibling_view(self, tmp_path):
        report = _make_fixture_report()
        report.add_waveform_view(str(A_TRAN), out_path=str(tmp_path / 'sibling_view.html'))
        report_path = tmp_path / 'report.html'
        report.save_html(str(report_path))
        text = report_path.read_text(encoding='utf-8')
        assert 'href="sibling_view.html"' in text
        assert str(tmp_path) not in text.split('href=')[1][:200]

    def test_save_html_absolute_href_for_non_sibling_view(self, tmp_path):
        report = _make_fixture_report()
        elsewhere = tmp_path / 'elsewhere'
        elsewhere.mkdir()
        view_path = elsewhere / 'far_view.html'
        report.add_waveform_view(str(A_TRAN), out_path=str(view_path))

        report_dir = tmp_path / 'reports'
        report_dir.mkdir()
        report_path = report_dir / 'report.html'
        report.save_html(str(report_path))
        text = report_path.read_text(encoding='utf-8')
        assert str(view_path.resolve()) in text or os.path.abspath(str(view_path)) in text

    def test_save_markdown_lists_waveform_views(self, tmp_path):
        report = _make_fixture_report()
        report.add_waveform_view(str(A_TRAN), out_path=str(tmp_path / 'v.html'), label='My View')
        md_path = tmp_path / 'report.md'
        report.save(str(md_path))
        text = md_path.read_text(encoding='utf-8')
        assert 'Waveform Explorer' in text
        assert 'My View' in text


class TestAddWaveformViewNoResultsFile:
    """2026-08-25 review, B1: an unsimulated ``ModuleTestbench`` (or any
    duck-typed stand-in) has ``.results_file is None`` — before the fix,
    ``add_waveform_view`` silently fell through to
    ``os.fspath(source)`` and raised a raw, unhelpful ``TypeError`` naming
    neither the missing rawfile nor the fix. It must now raise
    :class:`~silk.signals.view.ViewError` naming both."""

    def test_none_results_file_raises_view_error_not_fspath_typeerror(self, tmp_path):
        from silk.signals.view import ViewError
        report = _make_fixture_report()
        stub = StubTestbench(None, module_name='unrun_mod')
        with pytest.raises(ViewError, match='StubTestbench'):
            report.add_waveform_view(stub, out_path=str(tmp_path / 'v.html'))

    def test_object_without_results_file_attribute_raises_view_error(self, tmp_path):
        from silk.signals.view import ViewError

        class NotDuckTyped:
            pass

        report = _make_fixture_report()
        with pytest.raises(ViewError, match='NotDuckTyped'):
            report.add_waveform_view(NotDuckTyped(), out_path=str(tmp_path / 'v.html'))


# ---------------------------------------------------------------------------
# view-context-and-scopes design §2.2 — add_waveform_view's auto-populated
# `context`. Reads directly from view.data.js (context lives in the
# payload, not the SILK_VIEW_CONFIG sentinel — see test_view_cli.py's
# TestContextFlags for the same distinction on the CLI side).
# ---------------------------------------------------------------------------

def _data_js_of(html_path):
    stem = os.path.splitext(html_path)[0]
    return Path(f'{stem}.data.js').read_text(encoding='utf-8')


def _make_recon(pass_n=0, fail_n=0, unmeasured_n=0):
    outcomes = (
        [SpecOutcome(spec=None, measurement=None, status='PASS') for _ in range(pass_n)]
        + [SpecOutcome(spec=None, measurement=None, status='FAIL') for _ in range(fail_n)]
        + [SpecOutcome(spec=None, measurement=None, status='UNMEASURED') for _ in range(unmeasured_n)]
    )
    return SpecReconciliationReport(module=None, outcomes=outcomes)


class FullStubTestbench:
    """Every attribute add_waveform_view's auto-context (design spec §2.2)
    tries to read, all populated with real object types."""
    def __init__(self, results_file):
        from silk.simulation.analysis import Transient
        self.results_file = results_file
        self.module_name = 'full_mod'
        self.test_name = 'full_mod.tran'
        self.sim_analysis_dict = {'transient': Transient(step=1e-9, stop=1e-6)}
        self._last_sim_corner = 'tt'
        self.pdk_name = 'sky130'


class BareStubTestbench:
    """Only what add_waveform_view strictly needs (.results_file) — every
    other auto-context attribute is simply absent, the common case for a
    caller that just points at a rawfile path directly."""
    def __init__(self, results_file):
        self.results_file = results_file


class RaisingStubTestbench:
    """Every attribute *add_waveform_view's own new context logic* reads
    raises when accessed — the "never break a run" guarantee (design spec
    §2.2) must hold. ``module_name`` is deliberately a plain, well-behaved
    string here: it is read by add_waveform_view's pre-existing (unrelated,
    unchanged by this feature) out_path/label defaulting, not by the new
    auto-context code, so a raising ``module_name`` would be testing an old
    code path's robustness, not this one's."""
    def __init__(self, results_file):
        self.results_file = results_file
        self.module_name = 'raising_mod'

    @property
    def test_name(self):
        raise RuntimeError('boom: test_name')

    @property
    def sim_analysis_dict(self):
        raise RuntimeError('boom: sim_analysis_dict')

    @property
    def _last_sim_corner(self):
        raise RuntimeError('boom: corner')

    @property
    def pdk_name(self):
        raise RuntimeError('boom: pdk_name')


class TestAutoContextPopulation:
    def test_full_stub_populates_every_reachable_field(self, tmp_path):
        report = _make_fixture_report()
        stub = FullStubTestbench(str(A_TRAN))
        out = tmp_path / 'v.html'
        report.add_waveform_view(stub, out_path=str(out))
        text = _data_js_of(str(out))
        assert '"label": "module", "value": "full_mod"' in text
        assert '"label": "test", "value": "full_mod.tran"' in text
        assert '"label": "analysis"' in text and 'Transient(' in text
        assert '"label": "corner", "value": "tt"' in text
        assert '"label": "PDK", "value": "sky130"' in text
        assert '"label": "silk"' in text

    def test_bare_stub_falls_back_to_the_reports_own_module_name(self, tmp_path):
        report = _make_fixture_report()
        stub = BareStubTestbench(str(A_TRAN))
        out = tmp_path / 'v.html'
        report.add_waveform_view(stub, out_path=str(out))  # must not raise
        text = _data_js_of(str(out))
        assert '"context"' in text
        assert '"label": "silk"' in text
        # The source exposes no module_name, but the *report* always knows its
        # own module — and that is the single most useful provenance field, so
        # it falls back to it rather than omitting it (the common
        # top-level-testbench pattern and ModuleTestbench both expose nothing).
        assert '"label": "module", "value": "golden_mod"' in text
        # Fields genuinely unreachable from a bare source stay omitted.
        for label in ('test', 'analysis', 'corner', 'PDK'):
            assert f'"label": "{label}"' not in text

    def test_title_is_synthesized_from_provenance_when_none_given(self, tmp_path):
        """A report-emitted page must not be titled from the rawfile's own
        `Title:` line, which for a silk netlist is the boilerplate comment
        '*** automated netlist generator' — no information at all.
        """
        report = _make_fixture_report()
        stub = FullStubTestbench(str(A_TRAN))
        out = tmp_path / 'v.html'
        report.add_waveform_view(stub, out_path=str(out))
        text = _data_js_of(str(out))
        assert '"title": "full_mod \\u2014 Transient"' in text or \
               '"title": "full_mod — Transient"' in text
        assert 'automated netlist generator' not in text.split('"plots"')[0]

    def test_explicit_title_is_never_clobbered_by_synthesis(self, tmp_path):
        report = _make_fixture_report()
        stub = FullStubTestbench(str(A_TRAN))
        out = tmp_path / 'v.html'
        report.add_waveform_view(stub, out_path=str(out), title='MY TITLE')
        text = _data_js_of(str(out))
        assert '"title": "MY TITLE"' in text          # run.title
        assert '"title": null' in text                # context.title left unset

    def test_placeholder_module_name_is_omitted_not_displayed(self, tmp_path):
        """'unnamed' is the report builder's own placeholder for "no module".

        §2.2's rule is that a missed field is omitted, never filled with a
        placeholder, so the fallback must not surface it.
        """
        report = _make_fixture_report()
        report.module_name = 'unnamed'
        stub = BareStubTestbench(str(A_TRAN))
        out = tmp_path / 'v.html'
        report.add_waveform_view(stub, out_path=str(out))
        text = _data_js_of(str(out))
        assert '"label": "module"' not in text
        assert '"label": "silk"' in text  # the rest of the context still lands

    def test_raising_stub_never_raises_guarded_fields_omitted(self, tmp_path):
        report = _make_fixture_report()
        stub = RaisingStubTestbench(str(A_TRAN))
        out = tmp_path / 'v.html'
        report.add_waveform_view(stub, out_path=str(out))  # must not raise
        text = _data_js_of(str(out))
        assert '"label": "silk"' in text
        # module_name is a plain attribute here (not exercised by the
        # raising properties), so it legitimately resolves.
        assert '"label": "module", "value": "raising_mod"' in text
        # Every field this feature's own _safe()-guarded lookups touch
        # raises internally and is correctly omitted, not propagated.
        for label in ('test', 'analysis', 'corner', 'PDK'):
            assert f'"label": "{label}"' not in text

    def test_path_source_no_object_attrs_still_gets_silk_field(self, tmp_path):
        report = _make_fixture_report()
        out = tmp_path / 'v.html'
        report.add_waveform_view(str(A_TRAN), out_path=str(out))
        text = _data_js_of(str(out))
        assert '"label": "silk"' in text

    def test_spec_summary_field_added_when_outcomes_present(self, tmp_path):
        report = _make_fixture_report()
        report.spec_reconciliation = _make_recon(pass_n=3, fail_n=1)
        stub = BareStubTestbench(str(A_TRAN))
        out = tmp_path / 'v.html'
        report.add_waveform_view(stub, out_path=str(out))
        text = _data_js_of(str(out))
        assert '"label": "specs", "value": "3 PASS / 1 FAIL", "severity": "error"' in text

    def test_spec_summary_field_absent_when_no_outcomes(self, tmp_path):
        report = _make_fixture_report()
        report.spec_reconciliation = _make_recon()  # empty outcomes
        stub = BareStubTestbench(str(A_TRAN))
        out = tmp_path / 'v.html'
        report.add_waveform_view(stub, out_path=str(out))
        text = _data_js_of(str(out))
        assert '"label": "specs"' not in text

    def test_spec_summary_severity_warn_when_only_unmeasured(self, tmp_path):
        report = _make_fixture_report()
        report.spec_reconciliation = _make_recon(pass_n=2, unmeasured_n=1)
        stub = BareStubTestbench(str(A_TRAN))
        out = tmp_path / 'v.html'
        report.add_waveform_view(stub, out_path=str(out))
        text = _data_js_of(str(out))
        assert '"value": "2 PASS / 0 FAIL / 1 UNMEASURED", "severity": "warn"' in text

    def test_caller_context_field_wins_over_auto_by_label(self, tmp_path):
        report = _make_fixture_report()
        stub = FullStubTestbench(str(A_TRAN))
        out = tmp_path / 'v.html'
        report.add_waveform_view(stub, out_path=str(out), context={
            'fields': [{'label': 'corner', 'value': 'ff', 'severity': 'warn'}],
        })
        text = _data_js_of(str(out))
        assert '"label": "corner", "value": "ff", "severity": "warn"' in text
        assert '"label": "corner", "value": "tt"' not in text
        # Other auto fields are untouched.
        assert '"label": "module", "value": "full_mod"' in text

    def test_caller_context_field_appended_when_new_label(self, tmp_path):
        report = _make_fixture_report()
        stub = FullStubTestbench(str(A_TRAN))
        out = tmp_path / 'v.html'
        report.add_waveform_view(stub, out_path=str(out), context={
            'fields': [{'label': 'extra', 'value': 'hi'}],
        })
        text = _data_js_of(str(out))
        assert '"label": "extra", "value": "hi"' in text
        assert '"label": "module", "value": "full_mod"' in text  # not clobbered

    def test_title_param_still_controls_page_title_context_never_sets_it(self, tmp_path):
        # add_waveform_view's own `title=` keyword must keep working exactly
        # as before -- auto-population never sets context['title'], which
        # would otherwise silently outrank it (emit_view's own precedence:
        # context['title'] > title=).
        report = _make_fixture_report()
        stub = FullStubTestbench(str(A_TRAN))
        out = tmp_path / 'v.html'
        report.add_waveform_view(stub, out_path=str(out), title='Explicit Title')
        text = _data_js_of(str(out))
        assert '"title": "Explicit Title"' in text

    def test_caller_context_title_beats_title_param(self, tmp_path):
        report = _make_fixture_report()
        stub = FullStubTestbench(str(A_TRAN))
        out = tmp_path / 'v.html'
        report.add_waveform_view(
            stub, out_path=str(out), title='Explicit Title',
            context={'title': 'Context Title'},
        )
        text = _data_js_of(str(out))
        assert '"title": "Context Title"' in text

    def test_no_context_kwarg_still_gets_auto_context(self, tmp_path):
        # Not passing context= at all must not disable auto-population.
        report = _make_fixture_report()
        stub = FullStubTestbench(str(A_TRAN))
        out = tmp_path / 'v.html'
        report.add_waveform_view(stub, out_path=str(out))
        text = _data_js_of(str(out))
        assert '"label": "module", "value": "full_mod"' in text
