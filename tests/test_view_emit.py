"""Emitter behaviour tests for silk.signals.view.emit (design spec §9.4)."""
import json
import os
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from silk.signals.rawfile import RawFile, RawPlot, RawVector, read_rawfile
from silk.signals.view import emit_view, emit_view_data
from silk.signals.view.emit import ViewError
from silk.signals.view import encode as encode_mod

FIXTURES = Path(__file__).parent / 'fixtures' / 'rawfiles'
A_TRAN = FIXTURES / 'a_tran_bin.raw'
A_AC = FIXTURES / 'a_ac_bin.raw'


def _make_case_variant_signals_rawfile():
    """Two DISTINCT signals differing only by case, plus an unrelated third
    -- case-insensitive resolution is genuinely ambiguous for 'v(out)' /
    'V(OUT)' here (D1), but unambiguous for 'v(in)'."""
    header = {'plotname': 'Transient Analysis', 'flags': 'real', 'no. points': '2'}
    vectors = [
        RawVector('time', 'time', data=np.array([0.0, 1.0])),
        RawVector('v(out)', 'voltage', data=np.array([1.0, 2.0])),
        RawVector('V(OUT)', 'voltage', data=np.array([3.0, 4.0])),
        RawVector('v(in)', 'voltage', data=np.array([5.0, 6.0])),
    ]
    return RawFile([RawPlot(header, vectors, 2, True, None)], source_path='casevariant.raw')


def _make_dup_name_plot():
    header = {'plotname': 'Transient Analysis', 'flags': 'real', 'no. points': '3'}
    vectors = [
        RawVector('time', 'time', data=np.array([0.0, 1.0, 2.0])),
        RawVector('v(out)', 'voltage', data=np.array([1.0, 2.0, 3.0])),
        RawVector('v(out)', 'voltage', data=np.array([4.0, 5.0, 6.0])),
        RawVector('v(out)', 'voltage', data=np.array([7.0, 8.0, 9.0])),
    ]
    plot = RawPlot(header, vectors, 3, True, None)
    return RawFile([plot], source_path='synthetic.raw')


class TestSourceTypes:
    def test_accepts_str_path(self, tmp_path):
        out = tmp_path / 'view.html'
        result = emit_view(str(A_TRAN), out)
        assert os.path.exists(result.html_path)

    def test_accepts_pathlib_path(self, tmp_path):
        out = tmp_path / 'view.html'
        result = emit_view(A_TRAN, out)
        assert os.path.exists(result.html_path)

    def test_accepts_rawfile_instance(self, tmp_path):
        raw = read_rawfile(str(A_TRAN))
        out = tmp_path / 'view.html'
        result = emit_view(raw, out)
        assert os.path.exists(result.html_path)

    def test_none_source_raises_actionable_view_error(self, tmp_path):
        # The documented habit is emit_view(tb.results_file, ...) /
        # emit_view(module.results_file, ...); results_file is None by
        # contract before the testbench/module has been simulated. Without
        # the guard, os.fspath(None) raises a raw, unhelpful TypeError
        # instead of a fail-loud ViewError naming the actual cause.
        with pytest.raises(ViewError, match='has not been simulated yet'):
            emit_view(None, tmp_path / 'view.html')

    def test_none_source_raises_from_emit_view_data_too(self):
        with pytest.raises(ViewError, match='has not been simulated yet'):
            emit_view_data(None)


class TestIdentity:
    def test_title_only_reemission_changes_identity_not_waveforms(self, tmp_path):
        out = tmp_path / 'view.html'
        payloads = []
        results = []
        for title in ('Original title', 'Updated title', 'Updated title'):
            result = emit_view(A_TRAN, out, title=title)
            text = Path(result.data_path).read_text(encoding='utf-8')
            payload = json.JSONDecoder().raw_decode(text.split('var p=', 1)[1])[0]
            assert payload['run']['id'] == result.run_id
            assert result.run_id in Path(result.stamp_path).read_text(encoding='utf-8')
            assert emit_view_data(A_TRAN, title=title)['run']['id'] == result.run_id
            payloads.append(payload)
            results.append(result)
        assert results[0].run_id != results[1].run_id
        assert results[1].run_id == results[2].run_id
        assert len({result.state_key for result in results}) == 1
        assert set(payloads[0]) == {'format', 'endian', 'generator', 'generator_version',
                                    'run', 'plots', 'default_plot', 'annotations'}
        for payload in payloads[1:]:
            assert set(payload) == set(payloads[0])
            assert set(payload['run']) == set(payloads[0]['run'])
            for key in ('plots', 'annotations', 'default_plot'):
                assert payload[key] == payloads[0][key]

    def test_emission_time_does_not_change_identity(self, tmp_path):
        out = tmp_path / 'view.html'
        with mock.patch('silk.signals.view.emit._now_iso', return_value='2026-01-01T00:00:00Z'):
            first = emit_view(A_TRAN, out, title='Stable title')
        with mock.patch('silk.signals.view.emit._now_iso', return_value='2026-01-02T00:00:00Z'):
            second = emit_view(A_TRAN, out, title='Stable title')
        assert first.run_id == second.run_id
        assert first.state_key == second.state_key

    def test_same_rawfile_different_out_path_same_run_id_diff_state_key(self, tmp_path):
        r1 = emit_view(A_TRAN, tmp_path / 'a' / 'view.html')
        r2 = emit_view(A_TRAN, tmp_path / 'b' / 'view.html')
        assert r1.run_id == r2.run_id
        assert r1.state_key != r2.state_key

    def test_same_out_path_same_state_key(self, tmp_path):
        out = tmp_path / 'view.html'
        r1 = emit_view(A_TRAN, out)
        r2 = emit_view(A_TRAN, out)
        assert r1.state_key == r2.state_key
        assert r1.run_id == r2.run_id


class TestInlineMode:
    def test_writes_exactly_one_file(self, tmp_path):
        out = tmp_path / 'view.html'
        result = emit_view(A_TRAN, out, inline=True)
        assert result.data_path is None
        assert result.stamp_path is None
        assert sorted(os.listdir(tmp_path)) == ['view.html']

    def test_contains_payload_no_script_src(self, tmp_path):
        out = tmp_path / 'view.html'
        emit_view(A_TRAN, out, inline=True)
        text = out.read_text(encoding='utf-8')
        assert '__silkViewData' in text
        assert '<script src=' not in text

    def test_reemitting_inline_over_sidecars_removes_the_orphans(self, tmp_path):
        # An inline page never reads a sidecar, so one left by a previous
        # non-inline emit to the same path is an orphan holding stale data
        # while looking current.
        out = tmp_path / 'view.html'
        emit_view(A_TRAN, out)
        assert (tmp_path / 'view.data.js').exists()
        emit_view(A_TRAN, out, inline=True)
        assert sorted(os.listdir(tmp_path)) == ['view.html']

    def test_only_this_pages_sidecars_are_removed(self, tmp_path):
        emit_view(A_TRAN, tmp_path / 'other.html')
        emit_view(A_TRAN, tmp_path / 'view.html')
        emit_view(A_TRAN, tmp_path / 'view.html', inline=True)
        assert sorted(os.listdir(tmp_path)) == [
            'other.data.js', 'other.html', 'other.stamp.js', 'view.html',
        ]

    def test_inline_then_inline_is_unaffected(self, tmp_path):
        out = tmp_path / 'view.html'
        emit_view(A_TRAN, out, inline=True)
        emit_view(A_TRAN, out, inline=True)
        assert sorted(os.listdir(tmp_path)) == ['view.html']


class TestSidecarMode:
    def test_writes_three_files(self, tmp_path):
        # Sidecar filenames are derived from the HTML stem (v1.4): 'view.html'
        # -> 'view.data.js' / 'view.stamp.js', not the old hardcoded names —
        # so two pages emitted into one directory don't clobber each other.
        out = tmp_path / 'view.html'
        emit_view(A_TRAN, out)
        assert sorted(os.listdir(tmp_path)) == ['view.data.js', 'view.html', 'view.stamp.js']

    def test_stamp_carries_same_run_id_as_data(self, tmp_path):
        out = tmp_path / 'view.html'
        result = emit_view(A_TRAN, out)
        stamp_text = (tmp_path / 'view.stamp.js').read_text(encoding='utf-8')
        assert result.run_id in stamp_text

    def test_sidecar_names_derived_from_stem(self, tmp_path):
        out = tmp_path / 'my_scope.html'
        result = emit_view(A_TRAN, out)
        assert result.data_path == str((tmp_path / 'my_scope.data.js').resolve())
        assert result.stamp_path == str((tmp_path / 'my_scope.stamp.js').resolve())
        assert sorted(os.listdir(tmp_path)) == [
            'my_scope.data.js', 'my_scope.html', 'my_scope.stamp.js',
        ]

    def test_two_pages_same_directory_do_not_clobber(self, tmp_path):
        a = emit_view(A_TRAN, tmp_path / 'a.html', title='Page A')
        b = emit_view(A_TRAN, tmp_path / 'b.html', title='Page B')
        assert sorted(os.listdir(tmp_path)) == [
            'a.data.js', 'a.html', 'a.stamp.js', 'b.data.js', 'b.html', 'b.stamp.js',
        ]
        a_text = (tmp_path / 'a.html').read_text(encoding='utf-8')
        b_text = (tmp_path / 'b.html').read_text(encoding='utf-8')
        assert '"data_src": "a.data.js"' in a_text
        assert '"data_src": "b.data.js"' in b_text
        assert 'Page A' in a_text and 'Page B' not in a_text
        assert 'Page B' in b_text and 'Page A' not in b_text

    def test_sidecar_filename_is_html_escaped_in_the_script_tag(self, tmp_path):
        # The stem comes from a caller-chosen out_path, so it can carry '"' or
        # '&' and break out of the src attribute it is written into — the page
        # title has always been escaped, this attribute was not.
        out = tmp_path / 'he"llo&co.html'
        result = emit_view(A_TRAN, out)
        text = out.read_text(encoding='utf-8')
        assert '<script src="he&quot;llo&amp;co.data.js"></script>' in text
        assert '<script src="he"llo' not in text
        # The escaped attribute must still resolve to the file on disk.
        assert (tmp_path / 'he"llo&co.data.js').exists()
        assert result.data_path.endswith('he"llo&co.data.js')

    def test_html_not_rewritten_when_only_data_changes(self, tmp_path):
        out = tmp_path / 'view.html'
        emit_view(A_TRAN, out, title='same title')
        mtime1 = os.path.getmtime(out)
        import time
        time.sleep(0.02)
        emit_view(A_TRAN, out, title='same title')
        mtime2 = os.path.getmtime(out)
        assert mtime1 == mtime2

    def test_atomicity_uses_os_replace_and_leaves_no_tmp(self, tmp_path):
        out = tmp_path / 'view.html'
        with mock.patch('os.replace', wraps=os.replace) as mocked:
            emit_view(A_TRAN, out)
            assert mocked.called
        assert not any(str(p).endswith('.tmp') for p in tmp_path.iterdir())


class TestAnnotations:
    def test_valid_payload_round_trips_normalized(self, tmp_path):
        payload = emit_view_data(A_TRAN, annotations={
            'markers': [{'axis': 'x', 'value': 2.5e-9, 'label': 't_settle'}],
            'bands': [{'axis': 'y', 'lo': 0.1, 'hi': 0.9, 'signal': 'v(out)'}],
            'notes': [{'signal': 'v(out)', 'text': 'note'}],
        })
        ann = payload['annotations']
        assert ann['markers'][0]['signal'] is None
        assert ann['markers'][0]['color'] is None
        assert ann['bands'][0]['label'] == ''
        assert ann['notes'][0]['severity'] == 'info'

    def test_empty_annotations_normalize_to_empty_lists(self):
        payload = emit_view_data(A_TRAN, annotations=None)
        assert payload['annotations'] == {'markers': [], 'bands': [], 'notes': []}

    def test_unknown_top_level_key_raises(self):
        with pytest.raises(ViewError, match='unknown annotations key'):
            emit_view_data(A_TRAN, annotations={'bogus': []})

    def test_marker_missing_value_raises(self):
        with pytest.raises(ViewError):
            emit_view_data(A_TRAN, annotations={'markers': [{'axis': 'x'}]})

    def test_band_lo_ge_hi_raises(self):
        with pytest.raises(ViewError, match='lo.*hi|hi.*lo'):
            emit_view_data(A_TRAN, annotations={
                'bands': [{'axis': 'y', 'lo': 5.0, 'hi': 1.0}],
            })

    def test_non_finite_value_raises(self):
        with pytest.raises(ViewError):
            emit_view_data(A_TRAN, annotations={
                'markers': [{'axis': 'x', 'value': float('nan')}],
            })
        with pytest.raises(ViewError):
            emit_view_data(A_TRAN, annotations={
                'markers': [{'axis': 'x', 'value': float('inf')}],
            })

    def test_bad_severity_raises(self):
        with pytest.raises(ViewError):
            emit_view_data(A_TRAN, annotations={
                'notes': [{'signal': 'v(out)', 'text': 'x', 'severity': 'critical'}],
            })

    def test_unknown_signal_raises_with_suggestions(self):
        with pytest.raises(ViewError, match='does not match any signal'):
            emit_view_data(A_TRAN, annotations={
                'notes': [{'signal': 'v(nonexistent)', 'text': 'x'}],
            })

    def test_annotation_matching_uses_post_expansion_names(self):
        # Bare vector name must NOT resolve on a complex (AC) file — only the
        # mag/phase-expanded name does.
        with pytest.raises(ViewError, match=r'\(mag\)|\(phase\)'):
            emit_view_data(A_AC, annotations={
                'notes': [{'signal': 'v(out)', 'text': 'x'}],
            })
        payload = emit_view_data(A_AC, annotations={
            'notes': [{'signal': 'v(out) (mag)', 'text': 'x'}],
        })
        assert payload['annotations']['notes'][0]['signal'] == 'v(out) (mag)'

    def test_case_insensitive_signal_resolves_to_payload_spelling(self):
        # select.py's matches_selector is case-insensitive by design (G16
        # net-identity policy); annotations must be too. 'V(OUT)' must
        # resolve against the fixture's actual 'v(out)' spelling instead of
        # raising -- and the normalized annotation must carry the payload's
        # spelling, since that is what the frontend matches on.
        payload = emit_view_data(A_TRAN, annotations={
            'markers': [{'axis': 'x', 'value': 1e-9, 'signal': 'V(OUT)'}],
        })
        assert payload['annotations']['markers'][0]['signal'] == 'v(out)'

    def test_unknown_signal_hint_present_even_with_no_close_match(self):
        # A name with nothing similar in the payload used to raise with an
        # empty hint (''); the hint is now always present, falling back to
        # a few candidate names -- and labeled honestly as NOT a close match
        # (D5), since these are just alphabetically-first names, not hits.
        with pytest.raises(ViewError, match='No close match; available signals include'):
            emit_view_data(A_TRAN, annotations={
                'notes': [{'signal': 'totally_bogus_zzz', 'text': 'x'}],
            })

    def test_case_insensitive_resolution_is_deterministic_when_unambiguous(self):
        # valid_names is a set (hash-ordered); resolution must not depend on
        # that order. Repeated calls against a rawfile with several
        # same-casefold-bucket-adjacent names must always land on the one
        # actual unambiguous match.
        for _ in range(5):
            payload = emit_view_data(_make_case_variant_signals_rawfile(), annotations={
                'markers': [{'axis': 'x', 'value': 0.0, 'signal': 'V(IN)'}],
            })
            assert payload['annotations']['markers'][0]['signal'] == 'v(in)'

    def test_case_insensitive_ambiguous_match_raises_not_guesses(self):
        # 'v(out)' and 'V(OUT)' are two DISTINCT signals in this rawfile.
        # Silently picking one (previously hash-seed-dependent: verified
        # PYTHONHASHSEED 0..7 gave 'v(out)' x6, 'V(OUT)' x2) is a
        # silent-wrong-data path -- fail loud and name both candidates.
        with pytest.raises(ViewError, match=r"matches more than one signal.*Spell it exactly"):
            emit_view_data(_make_case_variant_signals_rawfile(), annotations={
                'markers': [{'axis': 'x', 'value': 0.0, 'signal': 'v(Out)'}],
            })
        # Exact spelling of either still resolves fine -- not banned outright.
        payload = emit_view_data(_make_case_variant_signals_rawfile(), annotations={
            'markers': [{'axis': 'x', 'value': 0.0, 'signal': 'V(OUT)'}],
        })
        assert payload['annotations']['markers'][0]['signal'] == 'V(OUT)'


class TestDestinationErrors:
    def test_unwritable_parent_raises_view_error(self, tmp_path):
        blocker = tmp_path / 'blocker'
        blocker.write_text('not a directory')
        bad_out = blocker / 'sub' / 'view.html'
        with pytest.raises(ViewError):
            emit_view(A_TRAN, bad_out)

    def test_no_plots_raises_view_error(self):
        empty = RawFile([], source_path='empty.raw')
        with pytest.raises(ViewError, match='no plots'):
            emit_view_data(empty)


class TestSizeGuard:
    def test_oversized_payload_raises_before_encoding(self, tmp_path):
        out = tmp_path / 'view.html'
        with mock.patch.object(encode_mod, 'b64_f4') as m4, \
             mock.patch.object(encode_mod, 'b64_f8') as m8:
            with pytest.raises(ViewError) as excinfo:
                emit_view(A_TRAN, out, max_bytes=10)
        assert 'max_bytes' in str(excinfo.value)
        m4.assert_not_called()
        m8.assert_not_called()

    def test_generous_max_bytes_succeeds(self, tmp_path):
        out = tmp_path / 'view.html'
        result = emit_view(A_TRAN, out, max_bytes=512_000_000)
        assert os.path.exists(result.html_path)


def _make_plot_with_names(names):
    header = {'plotname': 'Transient Analysis', 'flags': 'real', 'no. points': '3'}
    vectors = [RawVector('time', 'time', data=np.array([0.0, 1.0, 2.0]))]
    for i, name in enumerate(names):
        vectors.append(
            RawVector(name, 'voltage', data=np.array([1.0, 2.0, 3.0]) + i)
        )
    plot = RawPlot(header, vectors, 3, True, None)
    return RawFile([plot], source_path='synthetic.raw')


class TestDuplicateNames:
    def test_duplicate_vector_names_get_suffixes_and_renamed_flag(self):
        raw = _make_dup_name_plot()
        payload = emit_view_data(raw)
        names = [s['name'] for s in payload['plots'][0]['signals']]
        renamed = [s.get('renamed', False) for s in payload['plots'][0]['signals']]
        assert names == ['v(out)', 'v(out)#2', 'v(out)#3']
        assert renamed == [False, True, True]

    def test_generated_suffix_skips_a_name_the_file_already_uses(self):
        # ngspice really emits '#'-bearing names (v(m.xm1.…#dbody)), so the
        # generated 'v(a)#2' can collide with a vector literally called
        # 'v(a)#2'. Bumping past it is what keeps every payload name unique;
        # a survivor would silently mis-address annotations and addSignal.
        raw = _make_plot_with_names(['v(a)', 'v(a)#2', 'v(a)'])
        payload = emit_view_data(raw)
        names = [s['name'] for s in payload['plots'][0]['signals']]
        assert names == ['v(a)', 'v(a)#2', 'v(a)#3']
        assert len(set(names)) == 3

    def test_unique_names_are_left_alone(self):
        names_in = ['v(a)', 'v(b)', 'v(a)#2']
        payload = emit_view_data(_make_plot_with_names(names_in))
        signals = payload['plots'][0]['signals']
        assert [s['name'] for s in signals] == names_in
        assert not any(s.get('renamed', False) for s in signals)

    def test_repeated_collisions_all_stay_unique(self):
        raw = _make_plot_with_names(['v(a)', 'v(a)#2', 'v(a)#3', 'v(a)', 'v(a)'])
        payload = emit_view_data(raw)
        names = [s['name'] for s in payload['plots'][0]['signals']]
        assert len(set(names)) == len(names)
        assert names[:3] == ['v(a)', 'v(a)#2', 'v(a)#3']


class TestCurrentVectorNamePreserved:
    """TASK-317: a current vector's payload ``name`` must stay byte-identical
    to the rawfile's own vector name. Dropping the ``i(...)``/``I(...)``
    wrapping in the waveform explorer is a DISPLAY-layer choice
    (silk/signals/view/assets/explorer.mjs); this payload must never do it,
    since selectors, annotations and results all key off this exact string."""

    def test_real_ngspice_fixture_current_name_untouched(self):
        # a_tran_ascii.raw is a real ngspice rawfile with an i(v1) current
        # vector alongside v(in)/v(out) voltages.
        payload = emit_view_data(FIXTURES / 'a_tran_ascii.raw')
        signals = payload['plots'][0]['signals']
        current = next(s for s in signals if s['type'] == 'current')
        assert current['name'] == 'i(v1)'
        assert current['unit'] == 'A'

    @pytest.mark.parametrize('name', ['i(vvclk)', 'I(VVCLK)'])
    def test_synthetic_current_name_untouched(self, name):
        # Lower-case 'i(...)' is ngspice's convention, upper-case 'I(...)' is
        # Xyce's (it upper-cases everything) -- both must survive verbatim.
        header = {'plotname': 'Transient Analysis', 'flags': 'real', 'no. points': '2'}
        vectors = [
            RawVector('time', 'time', data=np.array([0.0, 1.0])),
            RawVector(name, 'current', data=np.array([0.001, 0.002])),
        ]
        raw = RawFile([RawPlot(header, vectors, 2, True, None)], source_path='synthetic.raw')
        payload = emit_view_data(raw)
        signal = payload['plots'][0]['signals'][0]
        assert signal['name'] == name
        assert signal['type'] == 'current'
        assert signal['unit'] == 'A'


# ---------------------------------------------------------------------------
# v1.3 — view-state sidecar (design spec §3.6, §7.1)
# ---------------------------------------------------------------------------

_BARE_VIEW = {'v': 1, 'cols': 2, 'activePlot': 0}
_ENVELOPE = {'silkview': 1, 'generator': 'test', 'view': _BARE_VIEW}


class TestViewStateEmbedding:
    def test_dict_bare_view_block_embeds(self, tmp_path):
        out = tmp_path / 'view.html'
        emit_view(A_TRAN, out, view_state=_BARE_VIEW)
        text = out.read_text(encoding='utf-8')
        assert 'window.SILK_VIEW_STATE = {' in text
        assert '"cols": 2' in text

    def test_dict_full_envelope_embeds_view_block_only(self, tmp_path):
        out = tmp_path / 'view.html'
        emit_view(A_TRAN, out, view_state=_ENVELOPE)
        text = out.read_text(encoding='utf-8')
        assert 'window.SILK_VIEW_STATE = {' in text
        assert '"cols": 2' in text
        # The envelope's 'generator' key is NOT part of the embedded block.
        marker = 'window.SILK_VIEW_STATE = '
        start = text.index(marker) + len(marker)
        end = text.index(';\n', start)
        assert json.loads(text[start:end]) == _BARE_VIEW

    def test_str_path_embeds(self, tmp_path):
        state_path = tmp_path / 'state.json'
        state_path.write_text(json.dumps(_ENVELOPE), encoding='utf-8')
        out = tmp_path / 'view.html'
        result = emit_view(A_TRAN, out, view_state=str(state_path))
        text = out.read_text(encoding='utf-8')
        assert 'window.SILK_VIEW_STATE = {' in text
        assert result.view_state_path == str(state_path.resolve())

    def test_pathlib_path_embeds(self, tmp_path):
        state_path = tmp_path / 'state.json'
        state_path.write_text(json.dumps(_ENVELOPE), encoding='utf-8')
        out = tmp_path / 'view.html'
        result = emit_view(A_TRAN, out, view_state=state_path)
        text = out.read_text(encoding='utf-8')
        assert 'window.SILK_VIEW_STATE = {' in text
        assert result.view_state_path == str(state_path.resolve())

    def test_missing_explicit_path_raises_naming_path(self, tmp_path):
        out = tmp_path / 'view.html'
        missing = tmp_path / 'nope.silkview.json'
        with pytest.raises(ViewError, match=r'nope\.silkview\.json'):
            emit_view(A_TRAN, out, view_state=str(missing))

    def test_autoload_default_picks_up_adjacent_sidecar(self, tmp_path):
        out = tmp_path / 'view.html'
        (tmp_path / 'view.silkview.json').write_text(json.dumps(_ENVELOPE), encoding='utf-8')
        result = emit_view(A_TRAN, out)
        text = out.read_text(encoding='utf-8')
        assert 'window.SILK_VIEW_STATE = {' in text
        assert result.view_state_path == str((tmp_path / 'view.silkview.json').resolve())

    def test_autoload_false_ignores_adjacent_sidecar(self, tmp_path):
        out = tmp_path / 'view.html'
        (tmp_path / 'view.silkview.json').write_text(json.dumps(_ENVELOPE), encoding='utf-8')
        result = emit_view(A_TRAN, out, autoload_view_state=False)
        text = out.read_text(encoding='utf-8')
        assert 'window.SILK_VIEW_STATE = null;' in text
        assert result.view_state_path is None

    def test_explicit_view_state_wins_over_adjacent_sidecar(self, tmp_path):
        out = tmp_path / 'view.html'
        (tmp_path / 'view.silkview.json').write_text(
            json.dumps({'silkview': 1, 'view': {'v': 1, 'cols': 99}}), encoding='utf-8',
        )
        result = emit_view(A_TRAN, out, view_state=_BARE_VIEW)
        text = out.read_text(encoding='utf-8')
        assert '"cols": 2' in text
        assert '"cols": 99' not in text
        assert result.view_state_path is None  # dict input has no path

    def test_no_state_anywhere_embeds_null(self, tmp_path):
        out = tmp_path / 'view.html'
        result = emit_view(A_TRAN, out)
        text = out.read_text(encoding='utf-8')
        assert 'window.SILK_VIEW_STATE = null;' in text
        assert result.view_state_path is None

    def test_dict_input_view_state_path_is_none(self, tmp_path):
        out = tmp_path / 'view.html'
        result = emit_view(A_TRAN, out, view_state=_BARE_VIEW)
        assert result.view_state_path is None


class TestViewStateEscaping:
    def test_malicious_state_never_appears_as_live_markup(self, tmp_path):
        evil = {
            'silkview': 1,
            'run': {'title': 'Evil</script><img onerror=alert(1)>' + chr(0xE9)},
            'view': {'v': 1, 'filter': '</script>bad' + chr(0xE9)},
        }
        state_path = tmp_path / 'view.silkview.json'
        state_path.write_text(json.dumps(evil), encoding='utf-8')
        out = tmp_path / 'view.html'
        emit_view(A_TRAN, out)
        text = out.read_text(encoding='utf-8')
        assert '</script>bad' not in text
        assert '<\\/script>bad' in text or '<\\/scriptbad' in text or '<\\/script' in text
        assert '</script><img' not in text


class TestViewStateRunIdUnchanged:
    def test_run_id_unchanged_by_view_state_and_commands(self, tmp_path):
        out1 = tmp_path / 'a' / 'view.html'
        out2 = tmp_path / 'b' / 'view.html'
        r1 = emit_view(A_TRAN, out1)
        r2 = emit_view(
            A_TRAN, out2, view_state=_BARE_VIEW,
            commands=[{'cmd': 'resetZoom'}],
        )
        assert r1.run_id == r2.run_id

    def test_emit_view_data_unaffected_by_view_state_kwargs_not_existing(self):
        # emit_view_data has no view_state=/commands= kwargs at all (design
        # spec §3.6: "view state is not payload") — passing them is a
        # TypeError, not a silently-ignored no-op.
        with pytest.raises(TypeError):
            emit_view_data(A_TRAN, view_state=_BARE_VIEW)


# ---------------------------------------------------------------------------
# v1.3 — command bus, Python-side structural validation (design spec §4.5)
# ---------------------------------------------------------------------------

class TestCommandsEmbedding:
    def test_commands_list_embeds(self, tmp_path):
        out = tmp_path / 'view.html'
        emit_view(A_TRAN, out, commands=[{'cmd': 'resetZoom'}])
        text = out.read_text(encoding='utf-8')
        assert 'window.SILK_VIEW_COMMANDS = [' in text
        assert '"cmd": "resetZoom"' in text

    def test_no_commands_embeds_empty_array(self, tmp_path):
        out = tmp_path / 'view.html'
        emit_view(A_TRAN, out)
        text = out.read_text(encoding='utf-8')
        assert 'window.SILK_VIEW_COMMANDS = [];' in text

    def test_bare_string_command_embedded_as_is(self, tmp_path):
        out = tmp_path / 'view.html'
        emit_view(A_TRAN, out, commands=['resetZoom'])
        text = out.read_text(encoding='utf-8')
        assert 'window.SILK_VIEW_COMMANDS = ["resetZoom"];' in text


class TestCommandsStructuralRejections:
    def test_not_a_list_raises(self, tmp_path):
        out = tmp_path / 'view.html'
        with pytest.raises(ViewError, match='must be a list'):
            emit_view(A_TRAN, out, commands={'cmd': 'resetZoom'})

    def test_element_not_str_or_dict_raises_naming_index(self, tmp_path):
        out = tmp_path / 'view.html'
        with pytest.raises(ViewError, match=r'commands\[0\]'):
            emit_view(A_TRAN, out, commands=[42])

    def test_dict_missing_cmd_raises_naming_index(self, tmp_path):
        out = tmp_path / 'view.html'
        with pytest.raises(ViewError, match=r"commands\[1\].*'cmd'"):
            emit_view(A_TRAN, out, commands=['resetZoom', {}])

    def test_cmd_not_a_str_raises_naming_index(self, tmp_path):
        out = tmp_path / 'view.html'
        with pytest.raises(ViewError, match=r'commands\[0\]'):
            emit_view(A_TRAN, out, commands=[{'cmd': 42}])

    def test_args_not_a_dict_raises_naming_index(self, tmp_path):
        out = tmp_path / 'view.html'
        with pytest.raises(ViewError, match=r'commands\[0\]'):
            emit_view(A_TRAN, out, commands=[{'cmd': 'addPanel', 'args': [1, 2]}])

    def test_unknown_key_raises_naming_index(self, tmp_path):
        out = tmp_path / 'view.html'
        with pytest.raises(ViewError, match=r'commands\[0\]'):
            emit_view(A_TRAN, out, commands=[{'cmd': 'resetZoom', 'bogus': 1}])


# ---------------------------------------------------------------------------
# view-context-and-scopes design §3.2 — the SILK_VIEW_SCOPES manifest on a
# hand-built emit_view(..., scopes=[...]) call. emit_views_by_block's own
# (auto-built) manifest is covered in test_view_select.py.
# ---------------------------------------------------------------------------

class TestScopesManifest:
    def _two_scope_list(self, current_html='view.html'):
        return [
            {'key': 'top', 'label': 'top-level', 'html': 'top.html',
             'data_src': 'top.data.js', 'stamp_src': 'top.stamp.js', 'n_signals': 5},
            {'key': 'xdut', 'html': current_html, 'data_src': 'view.data.js',
             'stamp_src': 'view.stamp.js', 'n_signals': 12},
        ]

    def test_no_scopes_embeds_default_empty_shape(self, tmp_path):
        out = tmp_path / 'view.html'
        emit_view(A_TRAN, out)
        text = out.read_text(encoding='utf-8')
        assert 'window.SILK_VIEW_SCOPES = {"current": null, "scopes": []};' in text

    def test_scopes_embedded_and_current_resolved_by_basename(self, tmp_path):
        out = tmp_path / 'view.html'
        emit_view(A_TRAN, out, scopes=self._two_scope_list('view.html'))
        text = out.read_text(encoding='utf-8')
        assert '"current": "xdut"' in text
        assert '"key": "top"' in text and '"key": "xdut"' in text

    def test_current_null_when_no_entry_matches_out_path(self, tmp_path):
        out = tmp_path / 'view.html'
        emit_view(A_TRAN, out, scopes=self._two_scope_list('other.html'))
        text = out.read_text(encoding='utf-8')
        assert '"current": null' in text

    def test_label_defaults_to_key_when_omitted(self, tmp_path):
        out = tmp_path / 'view.html'
        emit_view(A_TRAN, out, scopes=self._two_scope_list('view.html'))
        text = out.read_text(encoding='utf-8')
        assert '"key": "xdut", "label": "xdut"' in text

    def test_scopes_present_in_both_inline_and_sidecar_modes(self, tmp_path):
        # A single-entry manifest (no inline+scopes rejection since it is
        # empty by omission is a different case — here scopes IS given but
        # inline=False, sidecar mode) — the "always present" rule holds in
        # sidecar mode; inline+non-empty scopes is rejected below.
        out = tmp_path / 'view.html'
        emit_view(A_TRAN, out, scopes=self._two_scope_list('view.html'), inline=False)
        assert 'window.SILK_VIEW_SCOPES' in out.read_text(encoding='utf-8')

    def test_inline_with_scopes_raises(self, tmp_path):
        out = tmp_path / 'view.html'
        with pytest.raises(ViewError, match='inline'):
            emit_view(A_TRAN, out, inline=True, scopes=self._two_scope_list('view.html'))

    def test_inline_without_scopes_still_gets_default_shape(self, tmp_path):
        out = tmp_path / 'view.html'
        emit_view(A_TRAN, out, inline=True)
        text = out.read_text(encoding='utf-8')
        assert 'window.SILK_VIEW_SCOPES = {"current": null, "scopes": []};' in text

    def test_not_a_list_raises(self, tmp_path):
        out = tmp_path / 'view.html'
        with pytest.raises(ViewError, match='must be a list'):
            emit_view(A_TRAN, out, scopes={'key': 'a'})

    def test_missing_required_key_raises(self, tmp_path):
        out = tmp_path / 'view.html'
        with pytest.raises(ViewError, match=r'scopes\[0\].*missing required'):
            emit_view(A_TRAN, out, scopes=[{'key': 'a', 'html': 'a.html'}])

    def test_unknown_key_raises(self, tmp_path):
        out = tmp_path / 'view.html'
        entry = dict(self._two_scope_list()[0], bogus=1)
        with pytest.raises(ViewError, match=r'scopes\[0\].*unknown key'):
            emit_view(A_TRAN, out, scopes=[entry])

    def test_absolute_path_in_html_field_raises(self, tmp_path):
        out = tmp_path / 'view.html'
        entry = dict(self._two_scope_list()[0])
        entry['html'] = '/abs/path.html'
        with pytest.raises(ViewError, match='relative basename'):
            emit_view(A_TRAN, out, scopes=[entry])

    def test_path_separator_in_data_src_raises(self, tmp_path):
        out = tmp_path / 'view.html'
        entry = dict(self._two_scope_list()[0])
        entry['data_src'] = 'sub/dir.data.js'
        with pytest.raises(ViewError, match='relative basename'):
            emit_view(A_TRAN, out, scopes=[entry])

    def test_negative_n_signals_raises(self, tmp_path):
        out = tmp_path / 'view.html'
        entry = dict(self._two_scope_list()[0])
        entry['n_signals'] = -1
        with pytest.raises(ViewError, match='n_signals'):
            emit_view(A_TRAN, out, scopes=[entry])

    def test_duplicate_key_raises(self, tmp_path):
        out = tmp_path / 'view.html'
        entries = self._two_scope_list()
        entries[1]['key'] = entries[0]['key']
        with pytest.raises(ViewError, match='duplicate key'):
            emit_view(A_TRAN, out, scopes=entries)

    def test_scopes_do_not_affect_run_id(self, tmp_path):
        out1 = tmp_path / 'a' / 'view.html'
        out2 = tmp_path / 'b' / 'view.html'
        r1 = emit_view(A_TRAN, out1)
        r2 = emit_view(A_TRAN, out2, scopes=[{
            'key': 'x', 'html': 'view.html', 'data_src': 'view.data.js',
            'stamp_src': 'view.stamp.js', 'n_signals': 1,
        }])
        assert r1.run_id == r2.run_id


# ---------------------------------------------------------------------------
# v1.4 — signal selection on emit_view (large-rawfile-scaling backlog Tier 1
# "payload filter at emit"). Grammar/matcher unit tests live in
# tests/test_view_select.py; this is the emit_view()-level contract:
# fail-loud paths raised through the real API, ViewOutput.n_signals_total,
# and the sidecar-name coexistence that feature 1 exists to support.
# ---------------------------------------------------------------------------

class TestSignalSelectionOnEmitView:
    def test_default_signals_none_unchanged_behaviour(self, tmp_path):
        out = tmp_path / 'view.html'
        result = emit_view(A_TRAN, out)
        assert result.n_signals_total == result.n_signals

    def test_signals_filters_and_reports_total(self, tmp_path):
        out = tmp_path / 'view.html'
        result = emit_view(A_TRAN, out, signals=['v(out)'])
        assert result.n_signals < result.n_signals_total
        assert result.n_signals == 1

    def test_exclude_only_also_filters(self, tmp_path):
        out = tmp_path / 'view.html'
        full = emit_view(A_TRAN, tmp_path / 'full.html')
        result = emit_view(A_TRAN, out, exclude=['v(out)'])
        assert result.n_signals == full.n_signals - 1
        assert result.n_signals_total == full.n_signals_total

    def test_max_depth_only_filters(self, tmp_path):
        out = tmp_path / 'view.html'
        # a_tran_bin.raw has no dotted names, so every signal has depth 0 --
        # max_depth=0 must be a no-op here (all names survive).
        result = emit_view(A_TRAN, out, max_depth=0)
        assert result.n_signals == result.n_signals_total

    def test_unmatched_signals_selector_raises_via_emit_view(self, tmp_path):
        out = tmp_path / 'view.html'
        with pytest.raises(ViewError, match='matched no signal'):
            emit_view(A_TRAN, out, signals=['totally_bogus_name'])
        # Fail loud BEFORE writing anything.
        assert not out.exists()

    def test_unmatched_exclude_selector_raises_via_emit_view(self, tmp_path):
        out = tmp_path / 'view.html'
        with pytest.raises(ViewError, match='matched no signal'):
            emit_view(A_TRAN, out, exclude=['totally_bogus_name'])

    def test_empty_final_selection_raises_via_emit_view(self, tmp_path):
        out = tmp_path / 'view.html'
        with pytest.raises(ViewError, match='matched zero signals'):
            emit_view(A_TRAN, out, max_depth=0, exclude=['*'])

    def test_selection_applies_to_every_plot_multiplot_file(self):
        multi = FIXTURES / 'd_multiplot_ac_tran.raw'
        from silk.signals.view import emit_view_data
        payload = emit_view_data(multi, max_depth=0)
        # Every plot's signal count must be <= its unfiltered count; the
        # file must not crash even if one plot ends up with zero signals.
        unfiltered = emit_view_data(multi)
        for filt_plot, full_plot in zip(payload['plots'], unfiltered['plots']):
            assert len(filt_plot['signals']) <= len(full_plot['signals'])

    def test_x_axis_never_filtered(self, tmp_path):
        from silk.signals.view import emit_view_data
        payload = emit_view_data(A_TRAN, signals=['v(out)'])
        assert payload['plots'][0]['x']['levels'][0]['n'] > 0
        assert payload['plots'][0]['x']['name'] == 'time'


# ---------------------------------------------------------------------------
# AC phase unwrap: one non-converged frequency point must NaN exactly itself.
# np.unwrap works on differences, so an unguarded unwrap propagated that NaN
# through every later sample (12-point vector, one bad point -> 7 NaN phase
# samples) and the page's nonfinite badge reported the damage as if the
# simulator had produced it.
# ---------------------------------------------------------------------------

def _make_complex_plot(z):
    header = {'plotname': 'AC Analysis', 'flags': 'complex',
              'no. points': str(len(z))}
    vectors = [
        RawVector('frequency', 'frequency',
                  data=np.linspace(1.0, float(len(z)), len(z))),
        RawVector('v(out)', 'voltage', data=np.asarray(z)),
    ]
    return RawFile([RawPlot(header, vectors, len(z), False, None)],
                   source_path='synthetic_ac.raw')


def _decoded_y(signal):
    import base64
    level = signal['levels'][0]
    dtype = np.float32 if level['dt'] == 'f32' else np.float64
    return np.frombuffer(base64.b64decode(level['y']), dtype=dtype)


class TestComplexPhaseUnwrapNaN:
    def _payload_signals(self, z):
        payload = emit_view_data(_make_complex_plot(z))
        return {s['part']: s for s in payload['plots'][0]['signals']}

    def test_one_bad_sample_nans_only_itself(self):
        z = np.exp(1j * np.linspace(0.0, 6.0, 12))
        z[5] = np.nan + 1j * np.nan
        signals = self._payload_signals(z)
        assert signals['phase']['nonfinite'] == 1
        assert signals['mag']['nonfinite'] == 1
        phase = _decoded_y(signals['phase'])
        assert np.isnan(phase[5])
        assert np.isfinite(np.delete(phase, 5)).all()

    def test_finite_samples_match_unwrap_of_the_finite_subsequence(self):
        z = np.exp(1j * np.linspace(0.0, 6.0, 12))
        z[5] = np.nan + 1j * np.nan
        phase = _decoded_y(self._payload_signals(z)['phase'])
        angle = np.angle(z)
        finite = np.isfinite(angle)
        expected = np.degrees(np.unwrap(angle[finite]))
        # Continuity is preserved ACROSS the gap: the surviving samples are
        # one unwrapped run, not two independently-wrapped halves.
        np.testing.assert_allclose(phase[finite], expected, rtol=1e-6)

    def test_no_nan_path_unchanged(self):
        z = np.exp(1j * np.linspace(0.0, 20.0, 64))  # several 2pi wraps
        signals = self._payload_signals(z)
        assert signals['phase']['nonfinite'] == 0
        phase = _decoded_y(signals['phase'])
        expected = np.degrees(np.unwrap(np.angle(z)))
        np.testing.assert_allclose(phase, expected, rtol=1e-6)

    def test_all_samples_nonfinite_does_not_crash(self):
        z = np.full(8, np.nan + 1j * np.nan)
        signals = self._payload_signals(z)
        assert signals['phase']['nonfinite'] == 8


# ---------------------------------------------------------------------------
# Default-plot choice under a selection: a valid selection must never open a
# blank page. The emptiness check in select_for_payload is global across
# plots, so a selection living entirely in an earlier plot used to leave the
# last non-constants plot (the default) with zero signals -- exit 0, page
# opens showing nothing.
# ---------------------------------------------------------------------------

def _make_two_plot_rawfile():
    ac = RawPlot(
        {'plotname': 'AC Analysis', 'flags': 'real', 'no. points': '2'},
        [RawVector('frequency', 'frequency', data=np.array([1.0, 10.0])),
         RawVector('v(xdut.a)', 'voltage', data=np.array([1.0, 2.0])),
         RawVector('v(xdut.b)', 'voltage', data=np.array([3.0, 4.0]))],
        2, True, None)
    tran = RawPlot(
        {'plotname': 'Transient Analysis', 'flags': 'real', 'no. points': '2'},
        [RawVector('time', 'time', data=np.array([0.0, 1.0])),
         RawVector('v(top1)', 'voltage', data=np.array([1.0, 2.0])),
         RawVector('v(top2)', 'voltage', data=np.array([3.0, 4.0]))],
        2, True, None)
    return RawFile([ac, tran], source_path='two_plot.raw')


class TestDefaultPlotNeverEmpty:
    def test_selection_in_earlier_plot_makes_it_the_default(self):
        payload = emit_view_data(_make_two_plot_rawfile(), signals=['xdut'])
        default = payload['plots'][payload['default_plot']]
        assert default['name'] == 'AC Analysis'
        assert len(default['signals']) == 2

    def test_every_plot_with_signals_is_a_valid_default(self):
        # Whatever the selection, the default plot has signals whenever ANY
        # plot does -- the blank-open case is unconstructable.
        raw = _make_two_plot_rawfile()
        for selection in (['xdut'], ['v(top1)'], ['xdut', 'v(top2)'], None):
            payload = emit_view_data(raw, signals=selection)
            default = payload['plots'][payload['default_plot']]
            assert len(default['signals']) > 0, selection

    def test_unfiltered_default_is_still_the_last_non_constants_plot(self):
        payload = emit_view_data(_make_two_plot_rawfile())
        assert payload['plots'][payload['default_plot']]['name'] == 'Transient Analysis'

    def test_n_signals_of_emitted_page_is_nonzero(self, tmp_path):
        result = emit_view(_make_two_plot_rawfile(), tmp_path / 'view.html',
                           signals=['xdut'])
        assert result.n_signals == 2

    def test_split_by_block_pages_are_never_blank(self, tmp_path):
        from silk.signals.view.select import emit_views_by_block
        results = emit_views_by_block(_make_two_plot_rawfile(), str(tmp_path),
                                      depth=1)
        assert results
        for r in results:
            assert r.n_signals > 0, os.path.basename(r.html_path)


def _legacy_default_plot(plots_plan):
    """Index of the plot the explorer opens showing.

    Preference order, first hit wins, each scanned last-first:

    1. a non-constants plot that still has at least one signal;
    2. any plot that still has at least one signal (only ``constants``
       survived the filter);
    3. a non-constants plot (nothing anywhere has signals);
    4. the last plot.

    Steps 1-2 are what keeps a *selection* from opening a blank page: a
    ``signals=``/``exclude=``/``max_depth=`` filter is applied per plot, so
    on a multi-plot file the last non-constants plot can legitimately end up
    with zero signals while an earlier plot holds the whole selection
    (:func:`_plan_plot` runs the mask). Picking it anyway would have emitted
    a valid, exit-0 page that opens showing nothing.
    """
    def _last_matching(pred):
        for i in range(len(plots_plan) - 1, -1, -1):
            if pred(plots_plan[i]):
                return i
        return None

    for pred in (lambda p: p['kind'] != 'constants' and p['signals'],
                 lambda p: bool(p['signals']),
                 lambda p: p['kind'] != 'constants'):
        idx = _last_matching(pred)
        if idx is not None:
            return idx
    return len(plots_plan) - 1 if plots_plan else 0


# Captured from emit_view_data at fb6d9c17, before the public extraction.
_DEFAULT_PLOTS_BEFORE = json.loads(
    (Path(__file__).parent / 'fixtures' / 'view_default_plots.json').read_text())


class TestPublicDefaultPlot:
    @pytest.mark.parametrize("path", sorted(FIXTURES.glob("*.raw")), ids=lambda p: p.name)
    def test_existing_fixture_default_and_order_are_unchanged(self, path):
        from silk.signals.rawfile import RawfileError
        from silk.signals.view import pick_default_plot
        try:
            raw = read_rawfile(path)
        except RawfileError:
            pytest.skip("malformed reader fixture, not accepted by the viewer")
        if not raw.plots:
            assert pick_default_plot(raw) == 0
            return
        payload = emit_view_data(raw)
        before = _DEFAULT_PLOTS_BEFORE[path.name]
        assert payload['default_plot'] == before['default_plot']
        assert [p['name'] for p in payload['plots']] == before['plot_names']
        assert [p['name'] for p in payload['plots']] == [p.plotname for p in raw.plots]
        assert pick_default_plot(raw) == payload['default_plot']
        assert payload['default_plot'] == _legacy_default_plot(payload['plots'])

    @pytest.mark.parametrize("names", [
        ['Operating Point', 'Transient Analysis'],
        ['Transient Analysis', 'Operating Point'],
        ['constants', 'constants'],
        ['empty', 'constants'],
        ['empty', 'empty'],
        ['Transient Analysis', 'empty', 'constants'],
        [],
    ])
    def test_multiplot_default_and_order_are_unchanged(self, names):
        from silk.signals.view import pick_default_plot
        plots = []
        for name in names:
            n = 1 if name == 'Operating Point' else 2
            vectors = [] if name == 'empty' else [RawVector('v(out)', 'voltage', data=np.ones(n))]
            if name not in ('Operating Point', 'empty'):
                vectors.insert(0, RawVector('time', 'time', data=np.arange(n)))
            plots.append(RawPlot({'plotname': name, 'flags': 'real'}, vectors, n, True, None))
        raw = RawFile(plots)
        if not plots:
            assert pick_default_plot(raw) == pick_default_plot([]) == 0
            return
        payload = emit_view_data(raw)
        assert [p['name'] for p in payload['plots']] == names
        assert payload['default_plot'] == _legacy_default_plot(payload['plots'])
        assert pick_default_plot(payload['plots']) == payload['default_plot']
        assert pick_default_plot(raw) == payload['default_plot']

    def test_filter_can_move_default_to_an_earlier_plot(self):
        from silk.signals.view import pick_default_plot
        raw = _make_two_plot_rawfile()
        payload = emit_view_data(raw, signals=['xdut'])
        assert payload['default_plot'] == 0
        assert pick_default_plot(payload['plots']) == _legacy_default_plot(payload['plots'])
        assert payload['default_plot'] == pick_default_plot(payload['plots'])
