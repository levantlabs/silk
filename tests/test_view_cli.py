"""CLI tests for `python -m silk.signals.view` (design spec §9.6, v1.3 §7.1)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from silk.signals.view.__main__ import main

FIXTURES = Path(__file__).parent / 'fixtures' / 'rawfiles'
A_TRAN = FIXTURES / 'a_tran_bin.raw'
_ENVELOPE = {'silkview': 1, 'view': {'v': 1, 'cols': 2}}


class TestMainInProcess:
    def test_default_output_path(self, tmp_path, monkeypatch):
        # Copy the fixture into tmp_path so the default <rawfile dir>/view.html
        # lands somewhere disposable.
        src = tmp_path / 'a_tran_bin.raw'
        src.write_bytes(A_TRAN.read_bytes())
        rc = main([str(src)])
        assert rc == 0
        assert (tmp_path / 'view.html').exists()

    def test_dash_o_output(self, tmp_path):
        out = tmp_path / 'custom.html'
        rc = main([str(A_TRAN), '-o', str(out)])
        assert rc == 0
        assert out.exists()

    def test_title_flag(self, tmp_path):
        out = tmp_path / 'view.html'
        rc = main([str(A_TRAN), '-o', str(out), '--title', 'My Custom Title'])
        assert rc == 0
        text = out.read_text(encoding='utf-8')
        assert 'My Custom Title' in text

    def test_inline_flag_writes_single_file(self, tmp_path):
        out = tmp_path / 'view.html'
        rc = main([str(A_TRAN), '-o', str(out), '--inline'])
        assert rc == 0
        assert sorted(os.listdir(tmp_path)) == ['view.html']

    def test_poll_ms_zero(self, tmp_path):
        out = tmp_path / 'view.html'
        rc = main([str(A_TRAN), '-o', str(out), '--poll-ms', '0'])
        assert rc == 0
        text = out.read_text(encoding='utf-8')
        assert '"poll_ms": 0' in text or '"poll_ms":0' in text

    def test_exit_1_on_non_rawfile_clean_message_no_traceback(self, tmp_path, capsys):
        bogus = tmp_path / 'not_a_rawfile.raw'
        bogus.write_text('this is not a nutmeg rawfile')
        out = tmp_path / 'view.html'
        rc = main([str(bogus), '-o', str(out)])
        assert rc == 1
        captured = capsys.readouterr()
        assert 'Traceback' not in captured.err
        assert captured.err.strip() != ''

    def test_exit_1_on_missing_rawfile_clean_message_no_traceback(self, tmp_path, capsys):
        # read_rawfile raises FileNotFoundError, not (ViewError, RawfileError)
        # -- the __main__ except clause used to miss it entirely and let a
        # 15-line traceback through, contradicting cli.md's exit-code
        # contract (the typo'd path is the most common CLI error).
        missing = tmp_path / 'does_not_exist.raw'
        out = tmp_path / 'view.html'
        rc = main([str(missing), '-o', str(out)])
        assert rc == 1
        captured = capsys.readouterr()
        assert 'Traceback' not in captured.err
        assert captured.err.strip() != ''

    def test_exit_2_on_missing_positional(self, capsys):
        rc = main([])
        assert rc == 2

    def test_inline_and_serve_rejected(self, capsys):
        rc = main([str(A_TRAN), '--inline', '--serve'])
        assert rc == 2
        captured = capsys.readouterr()
        assert 'not allowed' in captured.err or 'inline' in captured.err.lower()

    def test_success_message_format(self, tmp_path, capsys):
        out = tmp_path / 'view.html'
        rc = main([str(A_TRAN), '-o', str(out)])
        assert rc == 0
        captured = capsys.readouterr()
        assert 'wrote view.html' in captured.out
        assert 'plots' in captured.out
        assert 'signals' in captured.out
        assert 'pts' in captured.out
        assert str(out) in captured.out


class TestViewStateFlags:
    def test_view_state_path_flag(self, tmp_path):
        state_path = tmp_path / 'state.json'
        state_path.write_text(json.dumps(_ENVELOPE), encoding='utf-8')
        out = tmp_path / 'view.html'
        rc = main([str(A_TRAN), '-o', str(out), '--view-state', str(state_path)])
        assert rc == 0
        text = out.read_text(encoding='utf-8')
        assert 'window.SILK_VIEW_STATE = {' in text
        assert '"cols": 2' in text

    def test_no_view_state_flag_ignores_adjacent_sidecar(self, tmp_path):
        out = tmp_path / 'view.html'
        (tmp_path / 'view.silkview.json').write_text(json.dumps(_ENVELOPE), encoding='utf-8')
        rc = main([str(A_TRAN), '-o', str(out), '--no-view-state'])
        assert rc == 0
        text = out.read_text(encoding='utf-8')
        assert 'window.SILK_VIEW_STATE = null;' in text

    def test_view_state_and_no_view_state_together_exit_2(self, capsys, tmp_path):
        out = tmp_path / 'view.html'
        rc = main([
            str(A_TRAN), '-o', str(out),
            '--view-state', 'x.json', '--no-view-state',
        ])
        assert rc == 2
        captured = capsys.readouterr()
        assert 'not allowed' in captured.err

    def test_missing_view_state_file_exit_1(self, tmp_path, capsys):
        out = tmp_path / 'view.html'
        rc = main([str(A_TRAN), '-o', str(out), '--view-state', str(tmp_path / 'nope.json')])
        assert rc == 1
        captured = capsys.readouterr()
        assert 'Traceback' not in captured.err
        assert 'nope.json' in captured.err

    def test_success_line_gains_view_state_suffix(self, tmp_path, capsys):
        state_path = tmp_path / 'state.json'
        state_path.write_text(json.dumps(_ENVELOPE), encoding='utf-8')
        out = tmp_path / 'view.html'
        rc = main([str(A_TRAN), '-o', str(out), '--view-state', str(state_path)])
        assert rc == 0
        captured = capsys.readouterr()
        assert '[+view-state]' in captured.out

    def test_success_line_no_suffix_without_view_state(self, tmp_path, capsys):
        out = tmp_path / 'view.html'
        rc = main([str(A_TRAN), '-o', str(out)])
        assert rc == 0
        captured = capsys.readouterr()
        assert '[+view-state]' not in captured.out


class TestCommandsFlag:
    def test_commands_json_flag(self, tmp_path):
        out = tmp_path / 'view.html'
        rc = main([
            str(A_TRAN), '-o', str(out),
            '--commands', json.dumps([{'cmd': 'resetZoom'}]),
        ])
        assert rc == 0
        text = out.read_text(encoding='utf-8')
        assert 'window.SILK_VIEW_COMMANDS = [{' in text
        assert '"cmd": "resetZoom"' in text

    def test_invalid_commands_json_exit_1_clean_stderr(self, tmp_path, capsys):
        out = tmp_path / 'view.html'
        rc = main([str(A_TRAN), '-o', str(out), '--commands', 'not json'])
        assert rc == 1
        captured = capsys.readouterr()
        assert 'Traceback' not in captured.err
        assert captured.err.strip() != ''
        assert not out.exists()


class TestContextFlags:
    # Context (unlike commands/view-state) lives in the payload, not the
    # SILK_VIEW_CONFIG sentinel (design spec §2.1) — so it is written to the
    # data.js sidecar, not view.html itself.
    def _data_text(self, tmp_path):
        return (tmp_path / 'view.data.js').read_text(encoding='utf-8')

    def test_context_title_flag(self, tmp_path):
        out = tmp_path / 'view.html'
        rc = main([str(A_TRAN), '-o', str(out), '--context-title', 'My Run'])
        assert rc == 0
        assert '"title": "My Run"' in self._data_text(tmp_path)

    def test_context_subtitle_flag(self, tmp_path):
        out = tmp_path / 'view.html'
        rc = main([str(A_TRAN), '-o', str(out), '--context-subtitle', 'tt corner'])
        assert rc == 0
        assert '"subtitle": "tt corner"' in self._data_text(tmp_path)

    def test_context_field_flag_repeatable(self, tmp_path):
        out = tmp_path / 'view.html'
        rc = main([
            str(A_TRAN), '-o', str(out),
            '--context-field', 'corner=tt', '--context-field', 'PDK=sky130A',
        ])
        assert rc == 0
        text = self._data_text(tmp_path)
        assert '{"label": "corner", "value": "tt"' in text
        assert '{"label": "PDK", "value": "sky130A"' in text

    def test_context_field_value_may_contain_equals(self, tmp_path):
        out = tmp_path / 'view.html'
        rc = main([str(A_TRAN), '-o', str(out), '--context-field', 'expr=a=b'])
        assert rc == 0
        assert '{"label": "expr", "value": "a=b"' in self._data_text(tmp_path)

    def test_context_field_missing_equals_exit_2_clean_stderr(self, tmp_path, capsys):
        out = tmp_path / 'view.html'
        rc = main([str(A_TRAN), '-o', str(out), '--context-field', 'noequals'])
        assert rc == 2
        captured = capsys.readouterr()
        assert 'Traceback' not in captured.err
        assert 'LABEL=VALUE' in captured.err
        assert not out.exists()

    def test_no_context_flags_no_context_key(self, tmp_path):
        out = tmp_path / 'view.html'
        rc = main([str(A_TRAN), '-o', str(out)])
        assert rc == 0
        data_text = (tmp_path / 'view.data.js').read_text(encoding='utf-8')
        assert '"context"' not in data_text

    def test_split_by_block_scopes_manifest_present(self, tmp_path, capsys):
        out = tmp_path / 'blocks' / 'view.html'
        rc = main([str(A_TRAN), '-o', str(out), '--split-by-block'])
        assert rc == 0
        for html_file in (tmp_path / 'blocks').glob('*.html'):
            text = html_file.read_text(encoding='utf-8')
            assert 'window.SILK_VIEW_SCOPES = {' in text
            assert '"current": null, "scopes": []' not in text


class TestSignalSelectionFlags:
    def test_signals_flag_filters(self, tmp_path):
        out = tmp_path / 'view.html'
        rc = main([str(A_TRAN), '-o', str(out), '--signals', 'v(out)'])
        assert rc == 0
        data_text = (tmp_path / 'view.data.js').read_text(encoding='utf-8')
        assert '"v(out)"' in data_text
        assert '"v(in)"' not in data_text

    def test_signals_flag_repeatable(self, tmp_path, capsys):
        out = tmp_path / 'view.html'
        rc = main([
            str(A_TRAN), '-o', str(out),
            '--signals', 'v(out)', '--signals', 'v(in)',
        ])
        assert rc == 0
        captured = capsys.readouterr()
        assert '2 of 3 signals' in captured.out

    def test_success_line_shows_reduction_when_filtered(self, tmp_path, capsys):
        out = tmp_path / 'view.html'
        rc = main([str(A_TRAN), '-o', str(out), '--signals', 'v(out)'])
        assert rc == 0
        captured = capsys.readouterr()
        assert ' of ' in captured.out
        assert 'signals' in captured.out

    def test_success_line_no_of_when_unfiltered(self, tmp_path, capsys):
        out = tmp_path / 'view.html'
        rc = main([str(A_TRAN), '-o', str(out)])
        assert rc == 0
        captured = capsys.readouterr()
        assert ' of ' not in captured.out

    def test_exclude_flag(self, tmp_path):
        out_full = tmp_path / 'full.html'
        out = tmp_path / 'view.html'
        main([str(A_TRAN), '-o', str(out_full)])
        rc = main([str(A_TRAN), '-o', str(out), '--exclude', 'v(out)'])
        assert rc == 0

    def test_max_depth_flag(self, tmp_path):
        out = tmp_path / 'view.html'
        rc = main([str(A_TRAN), '-o', str(out), '--max-depth', '0'])
        assert rc == 0

    def test_unmatched_selector_exit_1_clean_stderr(self, tmp_path, capsys):
        out = tmp_path / 'view.html'
        rc = main([str(A_TRAN), '-o', str(out), '--signals', 'totally_bogus'])
        assert rc == 1
        captured = capsys.readouterr()
        assert 'Traceback' not in captured.err
        assert 'totally_bogus' in captured.err
        assert not out.exists()


class TestListBlocksFlag:
    def test_prints_table_and_exits_0(self, capsys):
        rc = main([str(A_TRAN), '--list-blocks'])
        assert rc == 0
        captured = capsys.readouterr()
        assert 'hierarchy blocks at depth 1' in captured.out
        assert 'top-level' in captured.out

    def test_depth_argument(self, capsys):
        rc = main([str(A_TRAN), '--list-blocks', '2'])
        assert rc == 0
        captured = capsys.readouterr()
        assert 'hierarchy blocks at depth 2' in captured.out

    def test_emits_nothing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        rc = main([str(A_TRAN), '--list-blocks'])
        assert rc == 0
        assert os.listdir(tmp_path) == []

    def test_conflicts_with_output_flag_exit_2(self, tmp_path, capsys):
        rc = main([str(A_TRAN), '--list-blocks', '-o', str(tmp_path / 'x.html')])
        assert rc == 2
        captured = capsys.readouterr()
        assert 'not allowed' in captured.err
        assert '--output' in captured.err or '-o' in captured.err

    def test_conflicts_with_inline_flag_exit_2(self, capsys):
        rc = main([str(A_TRAN), '--list-blocks', '--inline'])
        assert rc == 2
        captured = capsys.readouterr()
        assert 'not allowed' in captured.err

    def test_conflicts_with_signals_flag_exit_2(self, capsys):
        rc = main([str(A_TRAN), '--list-blocks', '--signals', 'v(out)'])
        assert rc == 2
        captured = capsys.readouterr()
        assert 'not allowed' in captured.err

    def test_exit_1_on_missing_rawfile_clean_message_no_traceback(self, tmp_path, capsys):
        rc = main([str(tmp_path / 'nope.raw'), '--list-blocks'])
        assert rc == 1
        captured = capsys.readouterr()
        assert 'Traceback' not in captured.err
        assert captured.err.strip() != ''


class TestSplitByBlockFlag:
    def test_emits_one_page_per_block_plus_top(self, tmp_path, capsys):
        out = tmp_path / 'view.html'
        rc = main([str(A_TRAN), '-o', str(out), '--split-by-block'])
        assert rc == 0
        listing = sorted(os.listdir(tmp_path))
        assert 'top.html' in listing
        assert 'top.data.js' in listing

    def test_success_line_per_page(self, tmp_path, capsys):
        out = tmp_path / 'view.html'
        rc = main([str(A_TRAN), '-o', str(out), '--split-by-block'])
        assert rc == 0
        captured = capsys.readouterr()
        assert 'wrote top.html' in captured.out

    def test_depth_argument(self, tmp_path):
        out = tmp_path / 'view.html'
        rc = main([str(A_TRAN), '-o', str(out), '--split-by-block', '2'])
        assert rc == 0

    def test_conflicts_with_serve_exit_2(self, tmp_path, capsys):
        out = tmp_path / 'view.html'
        rc = main([str(A_TRAN), '-o', str(out), '--split-by-block', '--serve'])
        assert rc == 2
        captured = capsys.readouterr()
        assert 'not allowed' in captured.err

    def test_exit_1_on_missing_rawfile_clean_message_no_traceback(self, tmp_path, capsys):
        out = tmp_path / 'view.html'
        rc = main([str(tmp_path / 'nope.raw'), '-o', str(out), '--split-by-block'])
        assert rc == 1
        captured = capsys.readouterr()
        assert 'Traceback' not in captured.err
        assert captured.err.strip() != ''

    def test_zero_data_vectors_prints_message_and_exits_1(self, tmp_path, capsys):
        # Before the fix: emit_views_by_block returns [], the per-result loop
        # prints nothing, and main() fell through to `return 0` -- a silent
        # no-op in an otherwise fail-loud package. The single-page path
        # (emit_view) honestly emits a 0-signal page instead; split has no
        # such fallback since there is nothing to split.
        raw_path = tmp_path / 'no_signals.raw'
        raw_path.write_text(_NO_DATA_VECTORS_ASCII_RAWFILE, encoding='utf-8')
        rc = main([str(raw_path), '-o', str(tmp_path / 'view.html'), '--split-by-block'])
        assert rc == 1
        captured = capsys.readouterr()
        assert captured.out == ''
        assert 'wrote no pages' in captured.err
        assert not any(f.endswith('.html') for f in os.listdir(tmp_path))

    def test_no_blocks_at_depth_and_no_top_level_names_the_actual_cause(
            self, tmp_path, capsys):
        # results == [] is not ONLY the zero-data-vectors case above: a
        # rawfile can have real data vectors and still produce no pages,
        # when every signal is hierarchical and --split-by-block 0 asks for
        # top-level-only. The old hardcoded "no data vectors" message would
        # have been false here -- the message must name the actual cause.
        raw_path = tmp_path / 'hier.raw'
        raw_path.write_text(_ALL_HIER_ASCII_RAWFILE, encoding='utf-8')
        rc = main([str(raw_path), '-o', str(tmp_path / 'view.html'),
                   '--split-by-block', '0'])
        assert rc == 1
        captured = capsys.readouterr()
        assert captured.out == ''
        assert ('wrote no pages: no hierarchy blocks at this depth and no '
                'top-level signals') in captured.err
        assert not any(f.endswith('.html') for f in os.listdir(tmp_path))


class TestMainSubprocess:
    def test_subprocess_smoke(self, tmp_path):
        out = tmp_path / 'view.html'
        repo_root = Path(__file__).parent.parent
        result = subprocess.run(
            [sys.executable, '-m', 'silk.signals.view', str(A_TRAN), '-o', str(out)],
            cwd=str(repo_root), capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, result.stderr
        assert out.exists()


# ---------------------------------------------------------------------------
# Selection flags under --serve and --split-by-block. Both used to discard
# --signals/--exclude/--max-depth silently: the serve watcher re-emitted
# unfiltered on the first rawfile change, and --split-by-block accepted (and
# ignored) even a typo'd selector that is a hard error on every other path.
# ---------------------------------------------------------------------------

class TestServeCarriesSelection:
    def _captured_emit_kwargs(self, monkeypatch, argv):
        import silk.signals.view.__main__ as main_mod
        captured = {}

        def fake_run_server(out_path, rawfile_path, **kwargs):
            captured.update(kwargs.get('emit_kwargs') or {})

        monkeypatch.setattr(main_mod, 'run_server', fake_run_server)
        assert main_mod.main(argv) == 0
        return captured

    def test_signals_reach_the_watcher(self, tmp_path, monkeypatch, capsys):
        out = tmp_path / 'view.html'
        kwargs = self._captured_emit_kwargs(
            monkeypatch,
            [str(A_TRAN), '-o', str(out), '--serve', '--signals', 'v(out)'],
        )
        assert kwargs['signals'] == ['v(out)']

    def test_exclude_and_max_depth_reach_the_watcher(self, tmp_path, monkeypatch, capsys):
        out = tmp_path / 'view.html'
        kwargs = self._captured_emit_kwargs(
            monkeypatch,
            [str(A_TRAN), '-o', str(out), '--serve',
             '--exclude', 'v(out)', '--max-depth', '0'],
        )
        assert kwargs['exclude'] == ['v(out)']
        assert kwargs['max_depth'] == 0

    def test_re_emit_stays_filtered(self, tmp_path, monkeypatch, capsys):
        # The kwargs are not just carried, they still emit a filtered page:
        # feed them back through emit_view exactly as the watcher does.
        from silk.signals.view.emit import emit_view
        out = tmp_path / 'view.html'
        kwargs = self._captured_emit_kwargs(
            monkeypatch,
            [str(A_TRAN), '-o', str(out), '--serve', '--signals', 'v(out)'],
        )
        result = emit_view(str(A_TRAN), str(out), **kwargs)
        assert result.n_signals == 1
        assert result.n_signals < result.n_signals_total

    def test_unfiltered_serve_carries_none(self, tmp_path, monkeypatch, capsys):
        out = tmp_path / 'view.html'
        kwargs = self._captured_emit_kwargs(
            monkeypatch, [str(A_TRAN), '-o', str(out), '--serve'],
        )
        assert kwargs['signals'] is None
        assert kwargs['exclude'] is None
        assert kwargs['max_depth'] is None


class TestSplitByBlockSelectionConflicts:
    def test_bogus_selector_is_no_longer_silently_ignored(self, tmp_path, capsys):
        out = tmp_path / 'view.html'
        rc = main([str(A_TRAN), '-o', str(out), '--split-by-block',
                   '--signals', 'nope-does-not-exist'])
        assert rc == 2
        captured = capsys.readouterr()
        assert 'not allowed' in captured.err
        assert '--signals' in captured.err
        assert os.listdir(tmp_path) == []

    def test_valid_signals_selector_also_rejected(self, tmp_path, capsys):
        out = tmp_path / 'view.html'
        rc = main([str(A_TRAN), '-o', str(out), '--split-by-block',
                   '--signals', 'v(out)'])
        assert rc == 2
        assert os.listdir(tmp_path) == []

    def test_exclude_rejected(self, tmp_path, capsys):
        out = tmp_path / 'view.html'
        rc = main([str(A_TRAN), '-o', str(out), '--split-by-block',
                   '--exclude', 'v(out)'])
        assert rc == 2
        assert '--exclude' in capsys.readouterr().err

    def test_max_depth_zero_rejected(self, tmp_path, capsys):
        # max_depth=0 is a real selection, not an absent flag -- it must not
        # slip through a truthiness check.
        out = tmp_path / 'view.html'
        rc = main([str(A_TRAN), '-o', str(out), '--split-by-block',
                   '--max-depth', '0'])
        assert rc == 2
        assert '--max-depth' in capsys.readouterr().err

    def test_split_by_block_without_selection_still_works(self, tmp_path, capsys):
        out = tmp_path / 'view.html'
        rc = main([str(A_TRAN), '-o', str(out), '--split-by-block'])
        assert rc == 0
        assert 'top.html' in os.listdir(tmp_path)


class TestSplitByBlockOpen:
    def test_open_opens_top_html(self, tmp_path, monkeypatch, capsys):
        import silk.signals.view.__main__ as main_mod
        opened = []
        monkeypatch.setattr(main_mod.webbrowser, 'open', opened.append)
        out = tmp_path / 'view.html'
        rc = main_mod.main([str(A_TRAN), '-o', str(out), '--split-by-block', '--open'])
        assert rc == 0
        assert len(opened) == 1
        assert opened[0].endswith('top.html')

    def test_open_falls_back_to_the_first_page_when_no_top_level(
            self, tmp_path, monkeypatch, capsys):
        import silk.signals.view.__main__ as main_mod
        # Every signal hierarchical -> no top.html. The set is scope-linked
        # (each page carries the SILK_VIEW_SCOPES manifest), so the first
        # page is a valid entry point rather than an error.
        raw_path = tmp_path / 'hier.raw'
        raw_path.write_text(_ALL_HIER_ASCII_RAWFILE, encoding='utf-8')
        opened = []
        monkeypatch.setattr(main_mod.webbrowser, 'open', opened.append)
        rc = main_mod.main([str(raw_path), '-o', str(tmp_path / 'view.html'),
                            '--split-by-block', '--open'])
        assert rc == 0
        assert len(opened) == 1
        assert opened[0].endswith('xdut.html')

    def test_open_without_split_unchanged(self, tmp_path, monkeypatch, capsys):
        import silk.signals.view.__main__ as main_mod
        opened = []
        monkeypatch.setattr(main_mod.webbrowser, 'open', opened.append)
        out = tmp_path / 'view.html'
        rc = main_mod.main([str(A_TRAN), '-o', str(out), '--open'])
        assert rc == 0
        assert opened == ['file://' + str(out)]


# A two-point ASCII transient whose only data vector is hierarchical, so
# emit_views_by_block writes xdut.html and no top.html.
_ALL_HIER_ASCII_RAWFILE = (
    'Title: synthetic all-hierarchical transient\n'
    'Date: Mon Aug 11 00:00:00  2026\n'
    'Plotname: Transient Analysis\n'
    'Flags: real\n'
    'No. Variables: 2\n'
    'No. Points: 2\n'
    'Variables:\n'
    '\t0\ttime\ttime\n'
    '\t1\tv(xdut.out)\tvoltage\n'
    'Values:\n'
    ' 0\t0.000000000000000e+00\n'
    '\t1.000000000000000e+00\n'
    '\n'
    ' 1\t1.000000000000000e-09\n'
    '\t2.000000000000000e+00\n'
)

# A two-point ASCII transient with only the independent-axis vector -- zero
# data vectors, so emit_views_by_block finds no blocks and no top-level
# signals and returns [].
_NO_DATA_VECTORS_ASCII_RAWFILE = (
    'Title: synthetic no-data-vectors transient\n'
    'Date: Mon Aug 25 00:00:00  2026\n'
    'Plotname: Transient Analysis\n'
    'Flags: real\n'
    'No. Variables: 1\n'
    'No. Points: 2\n'
    'Variables:\n'
    '\t0\ttime\ttime\n'
    'Values:\n'
    ' 0\t0.000000000000000e+00\n'
    ' 1\t1.000000000000000e-09\n'
)
