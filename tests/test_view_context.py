"""Tests for the provenance ``context=`` feature on ``emit_view``/
``emit_view_data`` (view-context-and-scopes design §2.1, §6 R1/R4).
"""
import html.parser
import json
from pathlib import Path

import numpy as np
import pytest

from silk.signals.rawfile import RawFile, RawPlot, RawVector
from silk.signals.view import emit_view, emit_view_data
from silk.signals.view.emit import MAX_CONTEXT_BYTES, MAX_CONTEXT_FIELDS, ViewError

FIXTURES = Path(__file__).parent / 'fixtures' / 'rawfiles'
A_TRAN = FIXTURES / 'a_tran_bin.raw'


def _titled_rawfile(title):
    header = {'plotname': 'Transient Analysis', 'flags': 'real',
             'no. points': '2', 'title': title}
    vectors = [
        RawVector('time', 'time', data=np.array([0.0, 1.0])),
        RawVector('v(x)', 'voltage', data=np.array([1.0, 2.0])),
    ]
    plot = RawPlot(header, vectors, 2, True, None)
    return RawFile([plot], source_path='titled.raw')


def test_fallback_title_changes_identity():
    first = emit_view_data(_titled_rawfile('First raw title'))
    second = emit_view_data(_titled_rawfile('Second raw title'))
    assert first['plots'] == second['plots']
    assert first['run']['id'] != second['run']['id']
    raw = _titled_rawfile('')
    first = emit_view_data(raw)
    raw.source_path = 'renamed.raw'
    second = emit_view_data(raw)
    assert first['run']['title'] == 'titled.raw'
    assert second['run']['title'] == 'renamed.raw'
    assert first['run']['id'] != second['run']['id']


def test_winning_context_title_masks_explicit_title_change():
    first = emit_view_data(A_TRAN, title='First', context={'title': 'Winner'})
    second = emit_view_data(A_TRAN, title='Second', context={'title': 'Winner'})
    assert first['run']['title'] == second['run']['title'] == 'Winner'
    assert first['run']['id'] == second['run']['id']


class TestAbsentContext:
    def test_default_none_no_context_key(self):
        payload = emit_view_data(A_TRAN)
        assert 'context' not in payload

    def test_explicit_empty_dict_no_context_key(self):
        # Documented decision (design spec §5 item 1, "pick one, document,
        # test it"): a falsy context (None OR {}) is "no context" — same
        # treatment annotations gives `not annotations`.
        payload = emit_view_data(A_TRAN, context={})
        assert 'context' not in payload

    def test_emit_view_html_has_no_context_key_when_absent(self, tmp_path):
        out = tmp_path / 'view.html'
        emit_view(A_TRAN, out)
        text = (tmp_path / 'view.data.js').read_text(encoding='utf-8')
        assert '"context"' not in text


class TestValidContextNormalizes:
    def test_full_shape_round_trips(self):
        payload = emit_view_data(A_TRAN, context={
            'title': 'My Run', 'subtitle': 'tt corner',
            'fields': [{'label': 'module', 'value': 'ota_5t'},
                      {'label': 'specs', 'value': '4 PASS', 'severity': 'warn'}],
        })
        ctx = payload['context']
        assert ctx['title'] == 'My Run'
        assert ctx['subtitle'] == 'tt corner'
        assert ctx['fields'][0] == {'label': 'module', 'value': 'ota_5t', 'severity': 'info'}
        assert ctx['fields'][1]['severity'] == 'warn'

    def test_defaults_filled_title_subtitle_none_fields_empty(self):
        payload = emit_view_data(A_TRAN, context={'fields': []})
        ctx = payload['context']
        assert ctx == {'title': None, 'subtitle': None, 'fields': []}

    def test_non_str_value_is_str_coerced(self):
        payload = emit_view_data(A_TRAN, context={'fields': [{'label': 'n', 'value': 42}]})
        assert payload['context']['fields'][0]['value'] == '42'

    def test_field_order_preserved(self):
        labels = ['z', 'a', 'm']
        payload = emit_view_data(A_TRAN, context={
            'fields': [{'label': l, 'value': l} for l in labels],
        })
        assert [f['label'] for f in payload['context']['fields']] == labels


class TestContextFailLoud:
    def test_unknown_top_level_key_raises(self):
        with pytest.raises(ViewError, match='unknown context key'):
            emit_view_data(A_TRAN, context={'bogus': 1})

    def test_not_a_dict_raises(self):
        with pytest.raises(ViewError, match='must be a dict'):
            emit_view_data(A_TRAN, context=['not', 'a', 'dict'])

    def test_title_not_str_raises(self):
        with pytest.raises(ViewError, match="'title'"):
            emit_view_data(A_TRAN, context={'title': 123})

    def test_subtitle_not_str_raises(self):
        with pytest.raises(ViewError, match="'subtitle'"):
            emit_view_data(A_TRAN, context={'subtitle': 123})

    def test_fields_not_a_list_raises(self):
        with pytest.raises(ViewError, match="'fields'"):
            emit_view_data(A_TRAN, context={'fields': 'nope'})

    def test_field_not_a_dict_raises(self):
        with pytest.raises(ViewError, match='must be a dict'):
            emit_view_data(A_TRAN, context={'fields': ['nope']})

    def test_field_missing_label_raises(self):
        with pytest.raises(ViewError, match="'label'"):
            emit_view_data(A_TRAN, context={'fields': [{'value': 'x'}]})

    def test_field_missing_value_raises(self):
        with pytest.raises(ViewError, match="'value'"):
            emit_view_data(A_TRAN, context={'fields': [{'label': 'x'}]})

    def test_field_label_not_str_raises(self):
        with pytest.raises(ViewError, match="'label'"):
            emit_view_data(A_TRAN, context={'fields': [{'label': 1, 'value': 'x'}]})

    def test_field_bad_severity_raises(self):
        with pytest.raises(ViewError, match='severity'):
            emit_view_data(A_TRAN, context={
                'fields': [{'label': 'x', 'value': 'y', 'severity': 'critical'}],
            })

    def test_field_unknown_key_raises(self):
        with pytest.raises(ViewError, match='unknown key'):
            emit_view_data(A_TRAN, context={
                'fields': [{'label': 'x', 'value': 'y', 'bogus': 1}],
            })


class TestContextCap:
    def test_over_field_count_cap_raises(self):
        fields = [{'label': f'f{i}', 'value': str(i)} for i in range(MAX_CONTEXT_FIELDS + 1)]
        with pytest.raises(ViewError, match=r'field'):
            emit_view_data(A_TRAN, context={'fields': fields})

    def test_at_field_count_cap_succeeds(self):
        fields = [{'label': f'f{i}', 'value': str(i)} for i in range(MAX_CONTEXT_FIELDS)]
        payload = emit_view_data(A_TRAN, context={'fields': fields})
        assert len(payload['context']['fields']) == MAX_CONTEXT_FIELDS

    def test_over_byte_cap_raises(self):
        big_value = 'x' * (MAX_CONTEXT_BYTES + 1000)
        with pytest.raises(ViewError, match='bytes'):
            emit_view_data(A_TRAN, context={'fields': [{'label': 'huge', 'value': big_value}]})


class TestRunTitlePrecedence:
    """context['title'] -> title= -> rawfile title -> basename (spec §2.1)."""

    def test_context_title_wins_over_everything(self):
        raw = _titled_rawfile('rawfile title')
        payload = emit_view_data(raw, title='explicit title',
                                 context={'title': 'context title'})
        assert payload['run']['title'] == 'context title'

    def test_title_kwarg_wins_over_rawfile_title_when_no_context(self):
        raw = _titled_rawfile('rawfile title')
        payload = emit_view_data(raw, title='explicit title')
        assert payload['run']['title'] == 'explicit title'

    def test_rawfile_title_wins_over_basename_when_no_context_or_title(self):
        raw = _titled_rawfile('rawfile title')
        payload = emit_view_data(raw)
        assert payload['run']['title'] == 'rawfile title'

    def test_basename_fallback_when_nothing_else(self):
        raw = _titled_rawfile('')  # empty rawfile title -> falls through
        payload = emit_view_data(raw)
        assert payload['run']['title'] == 'titled.raw'

    def test_empty_context_title_does_not_win(self):
        # context={'title': ''} normalizes to '' which is falsy -> title=
        # still applies (an empty string is not "a title", mirrors the
        # existing `if title:` falsy-check convention used throughout).
        raw = _titled_rawfile('rawfile title')
        payload = emit_view_data(raw, title='explicit title', context={'title': ''})
        assert payload['run']['title'] == 'explicit title'


class _ImgDetector(html.parser.HTMLParser):
    """Mirrors test_view_template.py's TestEscaping helper: proves an
    injected ``<img>`` never becomes a real DOM element, as opposed to
    inert text sitting (harmlessly) inside a ``<script>`` element's raw-text
    body — the literal substring ``<img`` surviving is expected and fine;
    it becoming a parsed element is not."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.saw_img = False

    def handle_starttag(self, tag, attrs):
        if tag == 'img':
            self.saw_img = True


class TestEscaping:
    """Spec §6 R1: context is untrusted text, escaped through the same
    script-close escaper as the rest of the payload. The actual risk is
    ``</script>`` breaking OUT of the script element early — a literal
    ``<img onerror=...>`` sitting inert *inside* a script body is harmless
    (the HTML parser's raw-text mode for <script> never turns it into a DOM
    node), so the proof is "no real <img> element", not a bare substring
    check for the tag text."""

    _EVIL = 'Evil</script><img onerror=alert(1)>' + chr(0xE9)

    def test_evil_field_value_escaped_sidecar(self, tmp_path):
        out = tmp_path / 'view.html'
        emit_view(A_TRAN, out, context={'fields': [{'label': 'x', 'value': self._EVIL}]})
        data_text = (tmp_path / 'view.data.js').read_text(encoding='utf-8')
        assert '</script><img' not in data_text
        html_text = out.read_text(encoding='utf-8')
        assert '</script><img' not in html_text
        detector = _ImgDetector()
        detector.feed(html_text)
        assert not detector.saw_img, 'the injected <img> became a real DOM element'

    def test_evil_title_escaped_inline(self, tmp_path):
        out = tmp_path / 'view.html'
        emit_view(A_TRAN, out, inline=True, context={'title': self._EVIL})
        text = out.read_text(encoding='utf-8')
        assert '</script><img' not in text
        detector = _ImgDetector()
        detector.feed(text)
        assert not detector.saw_img, 'the injected <img> became a real DOM element'

    def test_evil_value_survives_json_round_trip(self):
        # json.dumps with the default ensure_ascii=True already neutralizes
        # the high byte (\\uXXXX escape) and </script> gets defused by the
        # shared escaper applied at serialization time; the *validated*
        # payload dict itself carries the value verbatim (escaping is a
        # serialization-time concern, not a validation-time one).
        payload = emit_view_data(A_TRAN, context={'fields': [{'label': 'x', 'value': self._EVIL}]})
        assert payload['context']['fields'][0]['value'] == self._EVIL
        # And it must still be representable as JSON (no crash / mangling).
        json.dumps(payload, allow_nan=False)


class TestRunIdIncludesContext:
    def test_different_context_changes_run_id(self):
        p1 = emit_view_data(A_TRAN, context={'fields': [{'label': 'x', 'value': '1'}]})
        p2 = emit_view_data(A_TRAN, context={'fields': [{'label': 'x', 'value': '2'}]})
        assert p1['run']['id'] != p2['run']['id']

    def test_same_context_same_run_id(self):
        ctx = {'fields': [{'label': 'x', 'value': '1'}]}
        p1 = emit_view_data(A_TRAN, context=dict(ctx))
        p2 = emit_view_data(A_TRAN, context=dict(ctx))
        assert p1['run']['id'] == p2['run']['id']

    def test_no_context_vs_context_different_run_id(self):
        p1 = emit_view_data(A_TRAN)
        p2 = emit_view_data(A_TRAN, context={'fields': [{'label': 'x', 'value': '1'}]})
        assert p1['run']['id'] != p2['run']['id']
