"""HTML/JS sanity tests without node (design spec §9.5).

Node is not available and is not being introduced; these are cheap checks
that catch the realistic failure modes (a truncated asset, a mis-substituted
sentinel, an unescaped attacker-adjacent string).
"""
import base64
import html.parser
import json
from pathlib import Path

import numpy as np
import pytest

from silk.signals.rawfile import RawFile, RawPlot, RawVector
from silk.signals.view import emit_view
from silk.signals.view import template as template_mod
from silk.signals.view.template import (
    ASSET_NAMES, ViewError, load_asset_bytes, load_asset_text, render_view_html,
)

FIXTURES = Path(__file__).parent / 'fixtures' / 'rawfiles'
A_TRAN = FIXTURES / 'a_tran_bin.raw'


# ---------------------------------------------------------------------------
# A small, comment/string/template-literal/regex-aware bracket-balance
# scanner. Not a JS parser — a smoke test for a truncated or mis-substituted
# asset (spec §9.5). Regex-vs-division is resolved with a standard heuristic
# (a `/` starts a regex unless the previous significant token looks like the
# end of an expression); template-literal bodies are treated as fully opaque
# (their `${...}` interior is not balance-checked) — a documented
# simplification, not a bug.
# ---------------------------------------------------------------------------

_REGEX_PRECEDERS = set('(,=:;!&|?{[\n') | {''}


def js_delimiters_balanced(text):
    depth = {'(': 0, '{': 0, '[': 0}
    closes = {')': '(', '}': '{', ']': '['}
    state = 'normal'
    prev = ''
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if state == 'normal':
            if c == '/' and i + 1 < n and text[i + 1] == '/':
                state = 'line_comment'; i += 2; continue
            if c == '/' and i + 1 < n and text[i + 1] == '*':
                state = 'block_comment'; i += 2; continue
            if c == "'":
                state = 'sq'; i += 1; continue
            if c == '"':
                state = 'dq'; i += 1; continue
            if c == '`':
                state = 'template'; i += 1; continue
            if c == '/' and prev in _REGEX_PRECEDERS:
                state = 'regex'; i += 1; continue
            if c in '({[':
                depth[c] += 1
                prev = c; i += 1; continue
            if c in ')}]':
                depth[closes[c]] -= 1
                if depth[closes[c]] < 0:
                    return False
                prev = c; i += 1; continue
            if not c.isspace():
                prev = c
            i += 1
        elif state == 'line_comment':
            if c == '\n':
                state = 'normal'
            i += 1
        elif state == 'block_comment':
            if c == '*' and i + 1 < n and text[i + 1] == '/':
                state = 'normal'; i += 2; continue
            i += 1
        elif state in ('sq', 'dq'):
            quote = "'" if state == 'sq' else '"'
            if c == '\\':
                i += 2; continue
            if c == quote:
                state = 'normal'
            i += 1
        elif state == 'template':
            if c == '\\':
                i += 2; continue
            if c == '`':
                state = 'normal'
            i += 1
        elif state == 'regex':
            if c == '\\':
                i += 2; continue
            if c == '[':
                j = i + 1
                while j < n and text[j] != ']':
                    j += 2 if text[j] == '\\' else 1
                i = j + 1; continue
            if c == '/':
                state = 'normal'
            i += 1
    return state == 'normal' and all(v == 0 for v in depth.values())


class TestBalanceScanner:
    def test_known_good_sample(self):
        assert js_delimiters_balanced("function f(a,b){ if(a){return [1,2,3];} } var s='a{b';")

    def test_known_truncated_sample(self):
        assert not js_delimiters_balanced("function f(a,b){ if(a){return [1,2,3]; } ")

    def test_regex_with_brace_like_chars_not_miscounted(self):
        assert js_delimiters_balanced("var re = /\\{[a-z]+\\}/; function g(){return re;}")

    def test_string_with_unbalanced_brace_not_miscounted(self):
        assert js_delimiters_balanced("var s = '{{{'; function h(){}")


# ---------------------------------------------------------------------------
# Asset packaging
# ---------------------------------------------------------------------------

class TestAssetsPackaged:
    @pytest.mark.parametrize('name', ASSET_NAMES)
    def test_asset_resolves_and_nonempty(self, name):
        data = load_asset_bytes(name)
        assert len(data) > 0

    def test_uplot_banner_byte_identical_in_asset(self):
        text = load_asset_text('uplot.iife.min.js')
        assert text.startswith('/*! https://github.com/leeoniya/uPlot (v1.6.32) */')

    def test_license_has_verbatim_mit_copyright_line(self):
        text = load_asset_text('LICENSE.uPlot')
        assert 'Leon Sorokin' in text
        assert 'MIT' in text


# ---------------------------------------------------------------------------
# Emitted HTML sanity
# ---------------------------------------------------------------------------

class TestEmittedHtml:
    @pytest.fixture
    def html_text(self, tmp_path):
        out = tmp_path / 'view.html'
        emit_view(A_TRAN, out)
        return out.read_text(encoding='utf-8')

    def test_no_unresolved_sentinels(self, html_text):
        assert '__SILK_' not in html_text

    def test_uplot_attribution_survives(self, html_text):
        assert 'https://github.com/leeoniya/uPlot (v1.6.32)' in html_text

    def test_parseable_and_script_style_counts(self, html_text):
        class Counter(html.parser.HTMLParser):
            def __init__(self):
                super().__init__(convert_charrefs=True)
                self.scripts = 0
                self.styles = 0

            def handle_starttag(self, tag, attrs):
                if tag == 'script':
                    self.scripts += 1
                elif tag == 'style':
                    self.styles += 1

        counter = Counter()
        counter.feed(html_text)  # must not raise
        # uplot JS, data-script(src), config, shared module bootstrap
        assert counter.scripts == 4
        # All CSS is inlined; how many <style> blocks the template uses is
        # its own business (the placeholder used 2, the real template 1).
        assert counter.styles >= 1

    def test_inlined_scripts_balanced(self, html_text):
        class ScriptExtractor(html.parser.HTMLParser):
            def __init__(self):
                super().__init__(convert_charrefs=False)
                self.bodies = []
                self._capturing = False

            def handle_starttag(self, tag, attrs):
                attr_dict = dict(attrs)
                if tag == 'script' and 'src' not in attr_dict:
                    self._capturing = True

            def handle_data(self, data):
                if self._capturing:
                    self.bodies.append(data)

            def handle_endtag(self, tag):
                if tag == 'script':
                    self._capturing = False

        extractor = ScriptExtractor()
        extractor.feed(html_text)
        assert extractor.bodies, 'expected at least one inline <script> body'
        for body in extractor.bodies:
            assert js_delimiters_balanced(body), 'an inline <script> is unbalanced'


class TestDataJsShape:
    def test_starts_and_ends_correctly_and_json_parses(self, tmp_path):
        out = tmp_path / 'view.html'
        emit_view(A_TRAN, out)
        text = (tmp_path / 'view.data.js').read_text(encoding='utf-8')
        assert text.startswith('(function(){var p={')
        assert text.rstrip().endswith('})();')
        marker = ';\nif(typeof window.__silkViewData'
        json_slice = text[len('(function(){var p='):text.index(marker)]
        payload = json.loads(json_slice)
        assert payload['format'] == 2


# ---------------------------------------------------------------------------
# v1.3 — the three globals riding the existing /*__SILK_CONFIG__*/ sentinel
# (design spec §3.3, §5, §7.1)
# ---------------------------------------------------------------------------

class TestThreeGlobalsPresent:
    @pytest.mark.parametrize('inline', [False, True])
    def test_all_three_globals_present_no_state(self, tmp_path, inline):
        out = tmp_path / 'view.html'
        emit_view(A_TRAN, out, inline=inline)
        text = out.read_text(encoding='utf-8')
        assert 'window.SILK_VIEW_CONFIG = {' in text
        assert 'window.SILK_VIEW_STATE = null;' in text
        assert 'window.SILK_VIEW_COMMANDS = [];' in text

    @pytest.mark.parametrize('inline', [False, True])
    def test_all_three_globals_present_with_state_and_commands(self, tmp_path, inline):
        out = tmp_path / 'view.html'
        emit_view(
            A_TRAN, out, inline=inline,
            view_state={'v': 1, 'cols': 2},
            commands=[{'cmd': 'resetZoom'}],
        )
        text = out.read_text(encoding='utf-8')
        assert 'window.SILK_VIEW_CONFIG = {' in text
        assert 'window.SILK_VIEW_STATE = {' in text
        assert '"cols": 2' in text
        assert 'window.SILK_VIEW_COMMANDS = [{' in text
        assert '"cmd": "resetZoom"' in text

    def test_globals_appear_in_order(self, tmp_path):
        out = tmp_path / 'view.html'
        emit_view(A_TRAN, out)
        text = out.read_text(encoding='utf-8')
        i_config = text.index('window.SILK_VIEW_CONFIG')
        i_state = text.index('window.SILK_VIEW_STATE')
        i_commands = text.index('window.SILK_VIEW_COMMANDS')
        i_scopes = text.index('window.SILK_VIEW_SCOPES')
        assert i_config < i_state < i_commands < i_scopes

    def test_emitted_payload_declares_current_format(self, tmp_path):
        # The format-2 producer keeps the existing data.js wrapper.
        out = tmp_path / 'view.html'
        emit_view(A_TRAN, out)
        payload = json.loads((tmp_path / 'view.data.js').read_text(encoding='utf-8')
                             .split('var p=', 1)[1].split(';\n', 1)[0])
        assert payload['format'] == 2


# ---------------------------------------------------------------------------
# view-context-and-scopes design §3.2/§5 — SILK_VIEW_SCOPES, the 4th global
# riding the same /*__SILK_CONFIG__*/ sentinel, always present.
# ---------------------------------------------------------------------------

class TestScopesGlobalAlwaysPresent:
    @pytest.mark.parametrize('inline', [False, True])
    def test_default_empty_shape_present(self, tmp_path, inline):
        out = tmp_path / 'view.html'
        emit_view(A_TRAN, out, inline=inline)
        text = out.read_text(encoding='utf-8')
        assert 'window.SILK_VIEW_SCOPES = {"current": null, "scopes": []};' in text

    def test_render_view_html_scopes_kwarg_embeds_manifest(self):
        from silk.signals.view.template import render_view_html
        manifest = {'current': 'a', 'scopes': [
            {'key': 'a', 'label': 'a', 'html': 'a.html', 'data_src': 'a.data.js',
             'stamp_src': 'a.stamp.js', 'n_signals': 3},
        ]}
        doc = render_view_html(
            title='t', data_script_html='<script></script>',
            poll_ms=2000, inline=False, scopes=manifest,
        )
        assert '"current": "a"' in doc
        assert '"key": "a"' in doc

    def test_render_view_html_none_scopes_defaults(self):
        from silk.signals.view.template import render_view_html
        doc = render_view_html(
            title='t', data_script_html='<script></script>',
            poll_ms=2000, inline=False,
        )
        assert 'window.SILK_VIEW_SCOPES = {"current": null, "scopes": []};' in doc


# ---------------------------------------------------------------------------
# TASK-319 — the standalone page's half of the groupSignalsByKind option,
# riding the same window.SILK_VIEW_CONFIG channel as poll_ms/inline. Display
# only: the payload itself (data.js / the inline payload script) must be
# byte-for-byte the same regardless of this setting.
# ---------------------------------------------------------------------------

class TestGroupSignalsByKindConfig:
    def test_default_is_true_in_config(self, tmp_path):
        out = tmp_path / 'view.html'
        emit_view(A_TRAN, out)
        text = out.read_text(encoding='utf-8')
        assert '"group_by_kind": true' in text

    def test_explicit_false_in_config(self, tmp_path):
        out = tmp_path / 'view.html'
        emit_view(A_TRAN, out, group_signals_by_kind=False)
        text = out.read_text(encoding='utf-8')
        assert '"group_by_kind": false' in text

    def test_render_view_html_default_true(self):
        doc = render_view_html(
            title='t', data_script_html='<script></script>',
            poll_ms=2000, inline=False,
        )
        assert '"group_by_kind": true' in doc

    def test_render_view_html_explicit_false(self):
        doc = render_view_html(
            title='t', data_script_html='<script></script>',
            poll_ms=2000, inline=False, group_signals_by_kind=False,
        )
        assert '"group_by_kind": false' in doc

    @staticmethod
    def _payload_dict(text):
        # Same slice shape TestDataJsShape uses, whether *text* is a bare
        # data.js file or the larger inline HTML page containing it.
        marker = ';\nif(typeof window.__silkViewData'
        json_slice = text[text.index('var p=') + len('var p='):text.index(marker)]
        return json.loads(json_slice)

    @pytest.mark.parametrize('inline', [False, True])
    def test_payload_is_byte_identical_regardless_of_setting(self, tmp_path, inline):
        # The one thing this option must never touch (spec: "display only").
        # 'state_key' (derived from the out_path, which differs between the
        # two directories here) and 'emitted_at' (wall-clock) are the only
        # legitimately path/time-varying fields; everything else, including
        # 'run.id' (a structure hash with no path/time component), must be
        # identical.
        on_dir, off_dir = tmp_path / 'on', tmp_path / 'off'
        on_dir.mkdir(); off_dir.mkdir()
        emit_view(A_TRAN, on_dir / 'view.html', inline=inline, group_signals_by_kind=True)
        emit_view(A_TRAN, off_dir / 'view.html', inline=inline, group_signals_by_kind=False)
        if inline:
            on_text = (on_dir / 'view.html').read_text(encoding='utf-8')
            off_text = (off_dir / 'view.html').read_text(encoding='utf-8')
        else:
            on_text = (on_dir / 'view.data.js').read_text(encoding='utf-8')
            off_text = (off_dir / 'view.data.js').read_text(encoding='utf-8')
        on_payload = self._payload_dict(on_text)
        off_payload = self._payload_dict(off_text)
        for run in (on_payload['run'], off_payload['run']):
            del run['state_key'], run['emitted_at']
        assert on_payload == off_payload


# ---------------------------------------------------------------------------
# view-context-and-scopes design §2.1 — the payload's optional 'context' key
# (a data.js concern, not a template-sentinel one, but the shape test lives
# alongside TestDataJsShape's payload-parsing pattern).
# ---------------------------------------------------------------------------

class TestPayloadContextKey:
    def test_context_key_present_and_parses_when_given(self, tmp_path):
        out = tmp_path / 'view.html'
        emit_view(A_TRAN, out, context={'title': 't', 'fields': [
            {'label': 'module', 'value': 'foo'},
        ]})
        text = (tmp_path / 'view.data.js').read_text(encoding='utf-8')
        marker = ';\nif(typeof window.__silkViewData'
        json_slice = text[len('(function(){var p='):text.index(marker)]
        payload = json.loads(json_slice)
        assert payload['context']['title'] == 't'
        assert payload['context']['fields'][0] == {
            'label': 'module', 'value': 'foo', 'severity': 'info',
        }

    def test_context_key_absent_when_not_given(self, tmp_path):
        out = tmp_path / 'view.html'
        emit_view(A_TRAN, out)
        text = (tmp_path / 'view.data.js').read_text(encoding='utf-8')
        marker = ';\nif(typeof window.__silkViewData'
        json_slice = text[len('(function(){var p='):text.index(marker)]
        payload = json.loads(json_slice)
        assert 'context' not in payload


class TestMissingSentinelGuard:
    def test_missing_sentinel_raises_naming_it(self, monkeypatch):
        real_tmpl = load_asset_text('view.html.tmpl')
        stripped = real_tmpl.replace('/*__SILK_CONFIG__*/', '')

        def fake_load_asset_text(name):
            if name == 'view.html.tmpl':
                return stripped
            return load_asset_text(name)

        monkeypatch.setattr(template_mod, 'load_asset_text', fake_load_asset_text)
        with pytest.raises(ViewError, match=r'__SILK_CONFIG__'):
            render_view_html(
                title='t', data_script_html='<script></script>',
                poll_ms=2000, inline=False,
            )

    def test_untouched_tmpl_does_not_raise(self):
        # Sanity check that the guard is not vacuously true.
        render_view_html(
            title='t', data_script_html='<script></script>',
            poll_ms=2000, inline=False,
        )


class TestEscaping:
    def _malicious_rawfile(self):
        evil_title = 'Evil</script><img onerror=alert(1)>' + chr(0xE9)
        header = {'plotname': 'Transient Analysis', 'flags': 'real',
                 'no. points': '2', 'title': evil_title}
        vectors = [
            RawVector('time', 'time', data=np.array([0.0, 1.0])),
            RawVector('v(x)', 'voltage', data=np.array([1.0, 2.0])),
        ]
        plot = RawPlot(header, vectors, 2, True, None)
        return RawFile([plot], source_path='evil.raw')

    def test_malicious_title_never_appears_as_live_markup_sidecar(self, tmp_path):
        raw = self._malicious_rawfile()
        out = tmp_path / 'view.html'
        emit_view(raw, out)
        text = out.read_text(encoding='utf-8')
        assert '</script><img' not in text
        assert '<img onerror=' not in text
        assert '&lt;img onerror=' in text or '&lt;script&gt;' in text or 'Evil&lt;' in text

    def test_malicious_title_never_appears_as_live_markup_inline(self, tmp_path):
        # In inline mode the title also lands inside the payload JSON, which
        # is embedded in an inline <script>. "<img onerror=...>" as inert
        # text *inside* a script element is harmless (the HTML parser's raw
        # text mode never turns it into a DOM node) — the actual risk is
        # "</script>" breaking OUT of the script early, which is what must
        # never happen unescaped.
        raw = self._malicious_rawfile()
        out = tmp_path / 'view.html'
        emit_view(raw, out, inline=True)
        text = out.read_text(encoding='utf-8')
        assert '</script><img' not in text

        class ImgDetector(html.parser.HTMLParser):
            def __init__(self):
                super().__init__(convert_charrefs=True)
                self.saw_img = False

            def handle_starttag(self, tag, attrs):
                if tag == 'img':
                    self.saw_img = True

        detector = ImgDetector()
        detector.feed(text)
        assert not detector.saw_img, 'the injected <img> became a real DOM element'
