"""View-state sidecar envelope tests (v1.3 design spec §3.2/§3.6).

Python validates the §3.2 **envelope** only — ``silkview``, the top-level
key set, and ``view``'s presence/shape. ``view``'s interior is embedded
verbatim, unvalidated (the frontend owns that tolerance). These tests cover
every fail-loud path plus the round trip through the hand-written fixture.
"""
import json
from pathlib import Path

import pytest

from silk.signals.view.emit import ViewError, _validate_view_state
from silk.signals.view.template import ViewError as TemplateViewError

FIXTURES = Path(__file__).parent / 'fixtures'
BASIC_STATE = FIXTURES / 'viewstate' / 'basic.silkview.json'


class TestSameViewError:
    def test_emit_and_template_view_error_are_the_same_class(self):
        # emit.ViewError is a re-export of template.ViewError (spec §5, item
        # 7's guard and §3.6's validation must raise the SAME type).
        assert ViewError is TemplateViewError


class TestEnvelopeValidation:
    def test_non_dict_raises(self):
        with pytest.raises(ViewError, match='must be a dict'):
            _validate_view_state([1, 2, 3])

    def test_silkview_missing_is_treated_as_bare_view_block(self):
        # No 'silkview' key at all -> the whole object IS the bare view
        # block, wrapped and accepted (this is the documented dict-input
        # shorthand, and applies uniformly to loaded JSON too).
        view = _validate_view_state({'v': 1, 'cols': 2})
        assert view == {'v': 1, 'cols': 2}

    def test_silkview_wrong_version_raises(self):
        with pytest.raises(ViewError, match='silkview'):
            _validate_view_state({'silkview': 2, 'view': {}})

    def test_unknown_top_level_key_raises_listing_accepted_set(self):
        with pytest.raises(ViewError) as excinfo:
            _validate_view_state({'silkview': 1, 'view': {}, 'bogus': True})
        msg = str(excinfo.value)
        assert 'bogus' in msg
        assert 'silkview' in msg and 'generator' in msg and 'view' in msg

    def test_view_missing_raises(self):
        with pytest.raises(ViewError, match="'view'"):
            _validate_view_state({'silkview': 1})

    def test_view_not_a_dict_raises(self):
        with pytest.raises(ViewError, match="'view'"):
            _validate_view_state({'silkview': 1, 'view': [1, 2]})

    def test_view_v_field_wrong_version_raises(self):
        with pytest.raises(ViewError, match="view\\['v'\\]|'v'"):
            _validate_view_state({'silkview': 1, 'view': {'v': 2}})

    def test_view_v_field_correct_version_accepted(self):
        view = _validate_view_state({'silkview': 1, 'view': {'v': 1, 'cols': 1}})
        assert view == {'v': 1, 'cols': 1}

    def test_bare_view_block_with_silkview_key_inside_view_still_wraps(self):
        # A bare block is detected by the ABSENCE of a top-level 'silkview'
        # key. A full envelope (has 'silkview') is validated as given.
        envelope = {'silkview': 1, 'generator': 'x', 'view': {'v': 1}}
        assert _validate_view_state(envelope) == {'v': 1}


class TestRoundTrip:
    def test_fixture_reaches_emitted_html_unmodified(self, tmp_path):
        from silk.signals.view import emit_view

        FIXTURES_RAW = FIXTURES / 'rawfiles'
        out = tmp_path / 'view.html'
        result = emit_view(
            FIXTURES_RAW / 'a_tran_bin.raw', out,
            view_state=str(BASIC_STATE),
        )
        assert result.view_state_path == str(BASIC_STATE.resolve())

        expected_view = json.loads(BASIC_STATE.read_text(encoding='utf-8'))['view']
        text = out.read_text(encoding='utf-8')
        marker = 'window.SILK_VIEW_STATE = '
        start = text.index(marker) + len(marker)
        end = text.index(';\n', start)
        embedded_view = json.loads(text[start:end])
        assert embedded_view == expected_view
        # Every §3.2 field the fixture carries survives, including the
        # per-plot extras (expanded/markers/treeMode).
        assert embedded_view['treeMode'] is True
        assert embedded_view['plots']['1']['expanded'] == ['m', 'm.xdut', 'm.xdut.xmmout']
        assert embedded_view['plots']['1']['markers'][0]['name'] == 't_settle'


class TestOperatingPointFields:
    """TASK-591: the additive panel ``sort`` and marker ``unit``/``ref`` fields
    pass through Python emission verbatim (no second UI schema)."""

    def test_sort_unit_and_ref_reach_the_emitted_html_unmodified(self, tmp_path):
        from silk.signals.view import emit_view

        raw = tmp_path / 'op.raw'
        raw.write_text('Title: t\nPlotname: Operating Point\nFlags: real\nNo. Variables: 2\n'
                       'No. Points: 1\nVariables:\n0\tv(a)\tvoltage\n1\ti(v1)\tcurrent\nValues:\n0\t1\n\t0.5\n')
        view = {'v': 1, 'plots': {'0': {
            'panels': [{'signals': ['v(a)'], 'sort': 'name'}, {'signals': []}],
            'markers': [{'name': 'M1', 'x': 0.5, 'color': '#e0575a', 'unit': 'V', 'ref': True},
                        {'name': 'M2', 'x': 0, 'color': '#5ad1a0', 'unit': ''}]}}}
        state = tmp_path / 'op.silkview.json'
        state.write_text(json.dumps({'silkview': 1, 'view': view}), encoding='utf-8')
        out = tmp_path / 'view.html'
        emit_view(raw, out, view_state=str(state))
        text = out.read_text(encoding='utf-8')
        marker = 'window.SILK_VIEW_STATE = '
        start = text.index(marker) + len(marker)
        embedded = json.loads(text[start:text.index(';\n', start)])
        assert embedded == view
        assert embedded['plots']['0']['markers'][1]['unit'] == ''
