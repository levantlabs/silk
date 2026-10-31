"""Tests for silk.signals.view.select — hierPical selection & block discovery.

Covers: the hierParse grammar-drift fixture (tests/fixtures/hierparse_cases.json,
pinned against the v1.3 §2.2 worked table and §2.1 corpus — this is the pin
that keeps this module's parser and the frontend's JS ``hierParse`` from
drifting), the three selector match forms, ``select_for_payload``'s
signals=/exclude=/max_depth= semantics and fail-loud paths, and block
discovery/splitting.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from silk.signals.rawfile import RawFile, RawPlot, RawVector
from silk.signals.view import emit_view
from silk.signals.view.emit import ViewError
from silk.signals.view.select import (
    derive_stems, emit_views_by_block, hier_parse, list_blocks,
    matches_selector, select_for_payload, signal_path_segments,
)

FIXTURES = Path(__file__).parent / 'fixtures' / 'rawfiles'
A_TRAN = FIXTURES / 'a_tran_bin.raw'
HIERPARSE_CASES = json.loads(
    (Path(__file__).parent / 'fixtures' / 'hierparse_cases.json').read_text(encoding='utf-8')
)


# ---------------------------------------------------------------------------
# hierParse grammar-drift guard
# ---------------------------------------------------------------------------

class TestHierParseFixture:
    def test_fixture_is_nonempty(self):
        assert len(HIERPARSE_CASES) >= 90

    @pytest.mark.parametrize('case', HIERPARSE_CASES, ids=lambda c: c['name'])
    def test_matches_expected(self, case):
        result = hier_parse(case['name'], part=case['part'])
        assert result['wrapper'] == case['wrapper']
        assert result['segments'] == case['segments']
        assert result['leaf'] == case['leaf']


class TestHierParseDirect:
    """A handful of hand-checked cases inline, independent of the fixture
    file, so a corrupted fixture can't hide a broken parser."""

    def test_bare_name(self):
        assert hier_parse('inoise_spectrum') == {
            'wrapper': None, 'segments': [], 'leaf': 'inoise_spectrum',
        }

    def test_wrapped_no_dots(self):
        assert hier_parse('v(in)') == {'wrapper': 'v', 'segments': [], 'leaf': 'in'}

    def test_dotted_hierarchy(self):
        assert hier_parse('v(xdut.out)') == {
            'wrapper': 'v', 'segments': ['xdut'], 'leaf': 'out',
        }

    def test_interior_node_is_also_a_signal(self):
        r = hier_parse('onoise.m.xdut.xmm1.msky130_fd_pr__nfet_01v8')
        assert r['segments'] == ['onoise', 'm', 'xdut', 'xmm1']
        assert r['leaf'] == 'msky130_fd_pr__nfet_01v8'

    def test_bracket_suffix_not_a_separator(self):
        r = hier_parse('i(@r.xdut.rrload[i])')
        assert r == {'wrapper': 'i', 'segments': ['@r', 'xdut'], 'leaf': 'rrload[i]'}

    def test_dot_inside_brackets_does_not_split(self):
        # Synthetic: a bracketed suffix containing a literal '.' must not be
        # treated as a path separator (spec §2.1 rule 3).
        r = hier_parse('i(@m.xdut.xmdev[a.b])')
        assert r == {'wrapper': 'i', 'segments': ['@m', 'xdut'], 'leaf': 'xmdev[a.b]'}

    def test_hash_suffix_stays_on_leaf(self):
        r = hier_parse('v(m.xdut.xmmout.xmdev.msky130_fd_pr__nfet_01v8#dbody)')
        assert r['leaf'] == 'msky130_fd_pr__nfet_01v8#dbody'
        assert r['segments'] == ['m', 'xdut', 'xmmout', 'xmdev']

    def test_mag_suffix_stripped_only_when_part_is_mag(self):
        assert hier_parse('v(out) (mag)', part='mag') == {
            'wrapper': 'v', 'segments': [], 'leaf': 'out',
        }
        # Without part='mag', the " (mag)" text is NOT stripped -- it is
        # just more of the (non-wrapper-matching) name.
        r = hier_parse('v(out) (mag)')
        assert r['leaf'] != 'out'

    def test_phase_suffix_stripped(self):
        assert hier_parse('v(out) (phase)', part='phase') == {
            'wrapper': 'v', 'segments': [], 'leaf': 'out',
        }

    def test_dedup_hash_suffix_not_stripped(self):
        # A "#2" dedup suffix (v1 §4.6) is a real, distinguishing part of the
        # name and must NOT be treated like the " (mag)"/" (phase)" suffix.
        r = hier_parse('v(out)#2')
        assert r['leaf'] == 'out)#2' or r['wrapper'] is None
        # Either way, the raw text is preserved verbatim somewhere in the
        # parse (no silent truncation of "#2").
        assert '#2' in (r['leaf'] or '')

    def test_xyce_uppercase_no_dots(self):
        assert hier_parse('V(OUT)') == {'wrapper': 'V', 'segments': [], 'leaf': 'OUT'}
        assert hier_parse('FREQUENCY') == {'wrapper': None, 'segments': [], 'leaf': 'FREQUENCY'}


class TestSignalPathSegments:
    def test_excludes_wrapper_includes_leaf(self):
        assert signal_path_segments('v(xdut.out)') == ['xdut', 'out']

    def test_bare_name_is_single_segment(self):
        assert signal_path_segments('v(vdd)') == ['vdd']


# ---------------------------------------------------------------------------
# Selector matching — the three forms, case-insensitivity, anchoring
# ---------------------------------------------------------------------------

class TestMatchesSelectorForms:
    def test_exact_full_name_match(self):
        assert matches_selector('v(out)', 'v(out)')
        assert not matches_selector('v(out)', 'v(outx)')

    def test_case_insensitive_exact_match(self):
        # ngspice lowercases, Xyce upper-cases -- a user must not have to care.
        assert matches_selector('V(OUT)', 'v(out)')
        assert matches_selector('v(out)', 'V(OUT)')

    def test_glob_match(self):
        assert matches_selector('v(xdut.out)', 'v(xdut.*)')
        assert matches_selector('onoise.m.xdut.xmm3.msky130_fd_pr__pfet_01v8', '*xmm3*')
        assert not matches_selector('v(xdut.out)', 'i(xdut.*)')

    def test_glob_case_insensitive(self):
        assert matches_selector('V(XDUT.OUT)', 'v(xdut.*)')

    def test_hierarchy_single_segment_reaches_any_depth(self):
        # The motivating case: "xdut" must select signals rooted differently
        # (v vs i vs the bare device-class prefix) because ngspice's
        # device-class prefix breaks a literal shared root (spec §2.1 rule 5).
        assert matches_selector('v(xdut.out)', 'xdut')
        assert matches_selector('i(v.xdut.xmmout.vamm_d)', 'xdut')
        assert matches_selector('onoise.m.xdut.xmm1.msky130_fd_pr__nfet_01v8', 'xdut')
        assert matches_selector(
            'onoise.m.xdut.xmm1.msky130_fd_pr__nfet_01v8.rd', 'xdut',
        )
        assert not matches_selector('v(vdd)', 'xdut')

    def test_hierarchy_multi_segment_is_a_tighter_anchor(self):
        assert matches_selector(
            'onoise.m.xdut.xmm3.msky130_fd_pr__pfet_01v8', 'm.xdut.xmm3',
        )
        assert not matches_selector(
            'onoise.m.xdut.xmm1.msky130_fd_pr__nfet_01v8', 'm.xdut.xmm3',
        )

    def test_hierarchy_run_must_be_contiguous_and_ordered(self):
        # segments = ['m', 'xdut', 'xmmout', 'xmdev'] -- "xdut.xmdev" skips
        # "xmmout" and must NOT match (not contiguous).
        assert not matches_selector(
            'v(m.xdut.xmmout.xmdev.msky130_fd_pr__nfet_01v8#dbody)', 'xdut.xmdev',
        )
        # Reversed order must not match either.
        assert not matches_selector(
            'v(m.xdut.xmmout.xmdev.msky130_fd_pr__nfet_01v8#dbody)', 'xmdev.xdut',
        )

    def test_hierarchy_matches_against_leaf_too(self):
        # signal_path_segments includes the leaf, so a selector can target
        # the leaf-only token of an interior-node-is-also-a-signal name.
        assert matches_selector(
            'onoise.m.xdut.xmm1.msky130_fd_pr__nfet_01v8', 'msky130_fd_pr__nfet_01v8',
        )

    def test_wrapper_is_not_matched_by_hierarchy_form(self):
        # 'v' as a hierarchy selector must not match every v(...) signal --
        # the wrapper is excluded from signal_path_segments (spec §2.2: "the
        # wrapper is not a tree level").
        assert not matches_selector('v(xdut.out)', 'v')


# ---------------------------------------------------------------------------
# select_for_payload — masks, fail-loud paths
# ---------------------------------------------------------------------------

class TestSelectForPayload:
    def test_signals_none_keeps_everything(self):
        names = [['v(a)', 'v(b)']]
        masks = select_for_payload(names, signals=None, exclude=None, max_depth=None)
        assert masks == [[True, True]]

    def test_signals_filters_by_any_match(self):
        names = [['v(xdut.out)', 'v(vdd)', 'i(v.xdut.xmmout.vamm_d)']]
        masks = select_for_payload(names, signals=['xdut'])
        assert masks == [[True, False, True]]

    def test_exclude_applied_after_signals_and_wins(self):
        names = [['v(xdut.out)', 'v(xdut.vbias)', 'v(vdd)']]
        masks = select_for_payload(names, signals=['xdut'], exclude=['*vbias*'])
        assert masks == [[True, False, False]]

    def test_max_depth_zero_keeps_only_bare_names(self):
        names = [['v(vdd)', 'v(xdut.out)', 'inoise_spectrum']]
        masks = select_for_payload(names, max_depth=0)
        assert masks == [[True, False, True]]

    def test_max_depth_anded_with_signals(self):
        names = [['v(xdut.out)', 'onoise.m.xdut.xmm1.msky130_fd_pr__nfet_01v8']]
        # Both match 'xdut' by hierarchy; only the first has segments <= 1.
        masks = select_for_payload(names, signals=['xdut'], max_depth=1)
        assert masks == [[True, False]]

    def test_applies_per_plot_independently(self):
        names = [['v(xdut.out)', 'v(vdd)'], ['v(yother.a)', 'v(vdd)']]
        masks = select_for_payload(names, signals=['xdut'])
        assert masks == [[True, False], [False, False]]

    def test_unmatched_signals_selector_raises_naming_it(self):
        names = [['v(xdut.out)']]
        with pytest.raises(ViewError, match=r"signals= selector 'nope' matched no signal"):
            select_for_payload(names, signals=['nope'])

    def test_unmatched_selector_gives_near_miss_suggestion(self):
        names = [['v(xdut.out)']]
        with pytest.raises(ViewError, match=r'xdut'):
            select_for_payload(names, signals=['xdu'])

    def test_unmatched_exclude_selector_raises_naming_it(self):
        names = [['v(xdut.out)']]
        with pytest.raises(ViewError, match=r"exclude= selector 'nope' matched no signal"):
            select_for_payload(names, exclude=['nope'])

    def test_a_selector_matching_in_one_plot_but_not_another_is_fine(self):
        # "matches ZERO vectors ACROSS THE WHOLE FILE" -- matching in any one
        # plot is enough; no error, even though plot 1 alone would look
        # unmatched for that selector.
        names = [['v(xdut.out)'], ['v(vdd)']]
        masks = select_for_payload(names, signals=['xdut'])
        assert masks == [[True], [False]]

    def test_empty_final_selection_raises(self):
        names = [['v(vdd)', 'v(vp)']]
        with pytest.raises(ViewError, match='matched zero signals'):
            select_for_payload(names, max_depth=0, exclude=['v(vdd)', 'v(vp)'])

    def test_empty_plot_list_is_not_itself_an_error(self):
        # A plot with zero data vectors is legal (renders an empty panel) as
        # long as some OTHER plot in the file has a nonempty selection.
        names = [[], ['v(vdd)']]
        masks = select_for_payload(names, max_depth=0)
        assert masks == [[], [True]]


# ---------------------------------------------------------------------------
# select_for_payload / signals= integrated through emit_view — the
# pre-complex-expansion + n_signals_total contract.
# ---------------------------------------------------------------------------

A_AC = FIXTURES / 'a_ac_bin.raw'


class TestSelectionIntegratedInEmit:
    def test_signals_selects_pre_expansion_keeps_both_mag_and_phase(self):
        from silk.signals.view import emit_view_data
        full = emit_view_data(A_AC)
        vec_name = full['plots'][0]['signals'][0]['group'] or full['plots'][0]['signals'][0]['name']
        filtered = emit_view_data(A_AC, signals=[vec_name])
        parts = {s['part'] for s in filtered['plots'][0]['signals']}
        assert parts == {'mag', 'phase'}

    def test_n_signals_total_unfiltered_equals_n_signals(self, tmp_path):
        result = emit_view(A_TRAN, tmp_path / 'view.html')
        assert result.n_signals_total == result.n_signals

    def test_n_signals_total_reflects_prefilter_count(self, tmp_path):
        result = emit_view(A_TRAN, tmp_path / 'view.html', max_depth=0)
        unfiltered = emit_view(A_TRAN, tmp_path / 'full.html')
        assert result.n_signals_total == unfiltered.n_signals_total
        assert result.n_signals <= result.n_signals_total

    def test_annotation_on_filtered_out_signal_raises(self, tmp_path):
        # A filtered-out signal is not addressable by annotations= either --
        # it falls out naturally from all_names being computed post-filter.
        with pytest.raises(ViewError, match='does not match any signal'):
            emit_view(
                A_TRAN, tmp_path / 'view.html', signals=['v(out)'],
                annotations={'notes': [{'signal': 'time', 'text': 'x'}]},
            )


# ---------------------------------------------------------------------------
# Block discovery
# ---------------------------------------------------------------------------

def _vec(name, type_, data):
    return RawVector(name, type_, data=np.asarray(data, dtype=np.float64))


def _make_hier_rawfile():
    """A small synthetic RawFile exercising the same device-class-prefix
    split as the real corpus: 'xdut' and 'v' do not share a root."""
    header0 = {'plotname': 'Transient Analysis', 'flags': 'real', 'no. points': '2'}
    vectors0 = [
        _vec('time', 'time', [0.0, 1.0]),
        _vec('v(xdut.out)', 'voltage', [1.0, 2.0]),
        _vec('v(xdut.vbias)', 'voltage', [0.1, 0.2]),
        _vec('i(v.xdut.xmmout.vamm_d)', 'current', [1e-3, 2e-3]),
        _vec('v(vdd)', 'voltage', [1.8, 1.8]),
        _vec('v(vp)', 'voltage', [0.9, 0.9]),
    ]
    plot0 = RawPlot(header0, vectors0, 2, True, None)

    header1 = {'plotname': 'AC Analysis', 'flags': 'real', 'no. points': '2'}
    vectors1 = [
        _vec('frequency', 'frequency', [1.0, 10.0]),
        _vec('v(xdut.out2)', 'voltage', [1.0, 2.0]),
    ]
    plot1 = RawPlot(header1, vectors1, 2, True, None)
    return RawFile([plot0, plot1], source_path='hier.raw')


def _make_all_bare_rawfile():
    header = {'plotname': 'Transient Analysis', 'flags': 'real', 'no. points': '2'}
    vectors = [
        _vec('time', 'time', [0.0, 1.0]),
        _vec('v(vdd)', 'voltage', [1.8, 1.8]),
        _vec('v(vp)', 'voltage', [0.9, 0.9]),
    ]
    return RawFile([RawPlot(header, vectors, 2, True, None)], source_path='bare.raw')


def _make_all_hier_rawfile():
    header = {'plotname': 'Transient Analysis', 'flags': 'real', 'no. points': '2'}
    vectors = [
        _vec('time', 'time', [0.0, 1.0]),
        _vec('v(xdut.out)', 'voltage', [1.0, 2.0]),
        _vec('v(xdut.vbias)', 'voltage', [0.1, 0.2]),
    ]
    return RawFile([RawPlot(header, vectors, 2, True, None)], source_path='allhier.raw')


def _make_mixed_case_hier_rawfile():
    """Same block, two spellings -- each simulator folds case uniformly, but
    the rawfile itself does not enforce it, so a mixed file is possible."""
    header = {'plotname': 'Transient Analysis', 'flags': 'real', 'no. points': '2'}
    vectors = [
        _vec('time', 'time', [0.0, 1.0]),
        _vec('v(xdut.a)', 'voltage', [1.0, 2.0]),
        _vec('v(XDUT.b)', 'voltage', [0.1, 0.2]),
    ]
    return RawFile([RawPlot(header, vectors, 2, True, None)], source_path='mixedcase.raw')


def _make_empty_signals_rawfile():
    # A plot with only the independent-axis vector -- zero data signals.
    header = {'plotname': 'Transient Analysis', 'flags': 'real', 'no. points': '2'}
    vectors = [_vec('time', 'time', [0.0, 1.0])]
    return RawFile([RawPlot(header, vectors, 2, True, None)], source_path='empty.raw')


class TestListBlocks:
    def test_none_source_raises_actionable_view_error(self, tmp_path):
        # tb.results_file is None until the testbench is simulated; both
        # select-side readers must match emit._resolve_source's contract
        # instead of leaking os.fspath's raw TypeError.
        with pytest.raises(ViewError, match='has not been simulated'):
            list_blocks(None)
        with pytest.raises(ViewError, match='has not been simulated'):
            emit_views_by_block(None, str(tmp_path))

    def test_depth1_buckets_by_first_segment_across_device_class_prefixes(self):
        blocks, n_top = list_blocks(_make_hier_rawfile(), depth=1)
        # Keys are discovered from leading segments (first-appearance order:
        # 'xdut' from plot 0, then 'v' from the ammeter name), but COUNTED with
        # the selector semantics emit_views_by_block uses. So 'xdut' is 4, not
        # 3: v(xdut.out), v(xdut.vbias), plot 1's v(xdut.out2), AND
        # i(v.xdut.xmmout.vamm_d) — which lives on the xdut page because the
        # hierarchy match spans ngspice's device-class prefix. Blocks
        # therefore overlap ('v' counts that ammeter too) and do not sum to
        # the signal total; that is the documented contract.
        assert blocks == [('xdut', 4), ('v', 1)]
        assert n_top == 2

    def test_block_count_equals_the_page_that_block_would_emit(self, tmp_path):
        """The consistency guarantee: a reported count is what the page holds.

        Before this was pinned, `--list-blocks` said `xdut 6` on the real
        72-signal transient while `--signals xdut` emitted 45 — a table that
        contradicted its own pages.
        """
        from silk.signals.view import emit_view_data
        raw = _make_hier_rawfile()
        blocks, _ = list_blocks(raw, depth=1)
        assert blocks, 'fixture must have at least one block'
        for path, n in blocks:
            payload = emit_view_data(raw, signals=[path])
            got = sum(len(p['signals']) for p in payload['plots'])
            assert got == n, f'block {path!r}: table says {n}, page has {got}'

    def test_depth2_uses_two_segments(self):
        blocks, n_top = list_blocks(_make_hier_rawfile(), depth=2)
        assert ('v.xdut', 1) in blocks
        assert n_top == 2

    def test_shallower_than_requested_depth_keeps_available_segments(self):
        # v(xdut.out) has only 1 segment; at depth=3 it is still bucketed
        # under 'xdut' rather than dropped (documented resolved ambiguity).
        blocks, n_top = list_blocks(_make_hier_rawfile(), depth=3)
        paths = dict(blocks)
        assert 'xdut' in paths

    def test_depth_zero_groups_nothing(self):
        blocks, n_top = list_blocks(_make_hier_rawfile(), depth=0)
        assert blocks == []
        assert n_top == 2

    def test_all_bare_signals_all_top_level(self):
        blocks, n_top = list_blocks(_make_all_bare_rawfile())
        assert blocks == []
        assert n_top == 2

    def test_all_hierarchical_no_top_level(self):
        blocks, n_top = list_blocks(_make_all_hier_rawfile())
        assert n_top == 0
        assert blocks == [('xdut', 2)]

    def test_accepts_path_source(self, tmp_path):
        # list_blocks must also accept a path/RawFile-agnostic source, like
        # emit_view does.
        blocks, n_top = list_blocks(str(A_TRAN))
        assert blocks == []  # a_tran_bin.raw has no dotted hierarchy
        assert n_top >= 1

    def test_mixed_case_spellings_of_the_same_block_dedupe(self):
        # matches_selector (block MATCHING) already folds case; discovery
        # must too, or xdut.* and XDUT.* list as two blocks with identical
        # membership. First-appearance spelling wins (G16 net-identity
        # policy), and the count covers both spellings' vectors.
        blocks, n_top = list_blocks(_make_mixed_case_hier_rawfile(), depth=1)
        assert blocks == [('xdut', 2)]
        assert n_top == 0


class TestEmitViewsByBlock:
    def test_one_page_per_block_plus_top(self, tmp_path):
        raw = _make_hier_rawfile()
        results = emit_views_by_block(raw, str(tmp_path))
        names = sorted(os.path.basename(r.html_path) for r in results)
        assert names == ['top.html', 'v.html', 'xdut.html']

    def test_sidecars_coexist_without_clobbering(self, tmp_path):
        raw = _make_hier_rawfile()
        emit_views_by_block(raw, str(tmp_path))
        listing = sorted(os.listdir(tmp_path))
        # Every page's stem-derived data/stamp sidecars must all be present
        # (feature 1 is what makes multi-page-one-directory safe).
        for stem in ('top', 'v', 'xdut'):
            assert f'{stem}.html' in listing
            assert f'{stem}.data.js' in listing
            assert f'{stem}.stamp.js' in listing

    def test_block_page_selects_all_descendants_via_hierarchy_match(self, tmp_path):
        raw = _make_hier_rawfile()
        results = emit_views_by_block(raw, str(tmp_path))
        xdut_result = next(r for r in results if os.path.basename(r.html_path) == 'xdut.html')
        # v(xdut.out), v(xdut.vbias) in plot 0 -> default plot is plot 1
        # (AC), which has v(xdut.out2) -- 1 signal selected there.
        assert xdut_result.n_signals >= 1

    def test_top_html_skipped_when_no_bare_signals(self, tmp_path):
        raw = _make_all_hier_rawfile()
        results = emit_views_by_block(raw, str(tmp_path))
        names = sorted(os.path.basename(r.html_path) for r in results)
        assert 'top.html' not in names
        assert names == ['xdut.html']

    def test_no_blocks_no_bare_signals_returns_empty_list(self, tmp_path):
        raw = _make_empty_signals_rawfile()
        results = emit_views_by_block(raw, str(tmp_path))
        assert results == []

    def test_title_prefix_applied(self, tmp_path):
        raw = _make_hier_rawfile()
        results = emit_views_by_block(raw, str(tmp_path), title_prefix='My Run')
        xdut_result = next(r for r in results if os.path.basename(r.html_path) == 'xdut.html')
        text = Path(xdut_result.html_path).read_text(encoding='utf-8')
        assert 'My Run' in text
        assert 'xdut' in text

    def test_sanitizes_dotted_block_paths_into_filenames(self, tmp_path):
        raw = _make_hier_rawfile()
        results = emit_views_by_block(raw, str(tmp_path), depth=2)
        stems = sorted(os.path.splitext(os.path.basename(r.html_path))[0] for r in results)
        assert 'v_xdut' in stems  # 'v.xdut' sanitized to 'v_xdut'
        assert not any('.' in s for s in stems)

    def test_emit_kwargs_forwarded(self, tmp_path):
        raw = _make_hier_rawfile()
        results = emit_views_by_block(raw, str(tmp_path), poll_ms=0)
        for r in results:
            text = Path(r.html_path).read_text(encoding='utf-8')
            assert '"poll_ms": 0' in text

    def test_mixed_case_block_spellings_write_one_page_not_two(self, tmp_path):
        # Before the fix, list_blocks discovered 'xdut' and 'XDUT' as two
        # blocks, so this wrote xdut.html AND XDUT.html -- on a
        # case-insensitive filesystem (macOS/Windows) the second clobbers
        # the first while the scope manifest still claims both exist.
        raw = _make_mixed_case_hier_rawfile()
        results = emit_views_by_block(raw, str(tmp_path))
        names = sorted(os.path.basename(r.html_path) for r in results)
        assert names == ['xdut.html']
        assert sorted(f for f in os.listdir(tmp_path) if f.endswith('.html')) == ['xdut.html']


# ---------------------------------------------------------------------------
# Stem collisions. `_sanitize` maps every run of non-[A-Za-z0-9_-] to one '_',
# so it is many-to-one: 'x.y_z' and 'x_y.z' both reduce to 'x_y_z'. Two blocks
# sharing one stem used to kill the split emit with a misleading
# `ViewError: scopes[1] has duplicate key 'x_y_z'`, and under inline=True
# (which builds no manifest, so nothing caught it) silently wrote the second
# page over the first while printing two "wrote" lines.
# ---------------------------------------------------------------------------

def _make_stem_collision_rawfile():
    """Two distinct depth-2 block paths that sanitize to the same string."""
    header = {'plotname': 'Transient Analysis', 'flags': 'real', 'no. points': '2'}
    vectors = [
        _vec('time', 'time', [0.0, 1.0]),
        _vec('v(x.y_z.out)', 'voltage', [1.0, 2.0]),    # block path 'x.y_z'
        _vec('v(x_y.z.out)', 'voltage', [0.1, 0.2]),    # block path 'x_y.z'
    ]
    return RawFile([RawPlot(header, vectors, 2, True, None)], source_path='collide.raw')


class TestDeriveStems:
    def test_distinct_paths_that_sanitize_alike_get_distinct_stems(self):
        stems = derive_stems(['x.y_z', 'x_y.z'])
        assert stems == {'x.y_z': 'x_y_z', 'x_y.z': 'x_y_z-2'}

    def test_assignment_is_path_sorted_not_input_ordered(self):
        # Same mapping whichever order the rawfile happened to reveal them in,
        # so a stem does not move between re-runs of the same design.
        assert derive_stems(['x_y.z', 'x.y_z']) == derive_stems(['x.y_z', 'x_y.z'])

    def test_non_colliding_paths_keep_their_bare_sanitized_stem(self):
        assert derive_stems(['xdut', 'v.xdut', 'm']) == {
            'xdut': 'xdut', 'v.xdut': 'v_xdut', 'm': 'm',
        }

    def test_bumps_past_a_stem_another_block_already_owns(self):
        stems = derive_stems(['x.y_z', 'x_y.z', 'x_y_z-2'])
        assert sorted(stems.values()) == ['x_y_z', 'x_y_z-2', 'x_y_z-2-2']
        assert len(set(stems.values())) == 3

    def test_top_is_reserved_for_the_top_level_page(self):
        # 'top.html' belongs to the depth-0 page; a block that sanitizes to
        # 'top' must not overwrite it.
        assert derive_stems(['top'])['top'] != 'top'

    def test_three_way_collision(self):
        stems = derive_stems(['a.b_c', 'a_b.c', 'a b c'])
        assert len(set(stems.values())) == 3
        assert all(s.startswith('a_b_c') for s in stems.values())

    def test_case_differing_paths_get_distinct_stems_by_case_insensitive_count(self):
        # 'Xdut' and 'XDUT' sanitize to two DIFFERENT Python strings (case is
        # letters, untouched by _sanitize), but are the same file on a
        # case-insensitive filesystem (macOS/Windows) -- the second (by
        # sorted-group order) must be bumped rather than silently clobbering
        # the first on disk.
        stems = derive_stems(['Xdut', 'XDUT'])
        assert stems['XDUT'] == 'XDUT'
        assert stems['Xdut'].casefold() != stems['XDUT'].casefold()


class TestEmitViewsByBlockStemCollisions:
    def test_sidecar_mode_writes_a_page_per_block(self, tmp_path):
        results = emit_views_by_block(_make_stem_collision_rawfile(),
                                      str(tmp_path), depth=2)
        assert len(results) == 2
        paths = {r.html_path for r in results}
        assert len(paths) == 2
        for path in paths:
            assert os.path.exists(path)

    def test_sidecar_mode_manifest_is_valid_and_names_both_pages(self, tmp_path):
        import json as _json
        import re as _re
        results = emit_views_by_block(_make_stem_collision_rawfile(),
                                      str(tmp_path), depth=2)
        text = Path(results[0].html_path).read_text(encoding='utf-8')
        manifest = _json.loads(
            _re.search(r'window\.SILK_VIEW_SCOPES = (.*?);', text).group(1)
        )
        keys = [s['key'] for s in manifest['scopes']]
        assert len(keys) == len(set(keys)) == 2
        assert {s['label'] for s in manifest['scopes']} == {'x.y_z', 'x_y.z'}
        assert {s['html'] for s in manifest['scopes']} == {
            os.path.basename(r.html_path) for r in results
        }

    def test_inline_mode_does_not_overwrite_the_first_page(self, tmp_path):
        results = emit_views_by_block(_make_stem_collision_rawfile(),
                                      str(tmp_path), depth=2, inline=True)
        assert len(results) == 2
        assert len(os.listdir(tmp_path)) == 2  # two HTML files, no sidecars
        # Both blocks survive: each page on disk carries its own, and the
        # second did not land on top of the first.
        rendered = [Path(r.html_path).read_text(encoding='utf-8') for r in results]
        assert sum('v(x.y_z.out)' in text for text in rendered) == 1
        assert sum('v(x_y.z.out)' in text for text in rendered) == 1

    def test_stems_are_stable_across_re_runs(self, tmp_path):
        raw = _make_stem_collision_rawfile()
        first = sorted(os.path.basename(r.html_path)
                       for r in emit_views_by_block(raw, str(tmp_path / 'a'), depth=2))
        second = sorted(os.path.basename(r.html_path)
                        for r in emit_views_by_block(raw, str(tmp_path / 'b'), depth=2))
        assert first == second == ['x_y_z-2.html', 'x_y_z.html']


# ---------------------------------------------------------------------------
# view-context-and-scopes design §3.2 — the SILK_VIEW_SCOPES manifest that
# emit_views_by_block builds automatically for the whole page set. A
# single-plot rawfile is used deliberately: with only one plot, list_blocks'
# cross-plot-summed counts (the manifest's n_signals source, see
# emit_views_by_block's docstring) exactly equal what each page renders, so
# these assertions can be exact rather than a >= bound.
# ---------------------------------------------------------------------------

def _make_single_plot_hier_rawfile():
    header = {'plotname': 'Transient Analysis', 'flags': 'real', 'no. points': '2'}
    vectors = [
        _vec('time', 'time', [0.0, 1.0]),
        _vec('v(xdut.out)', 'voltage', [1.0, 2.0]),
        _vec('v(xdut.vbias)', 'voltage', [0.1, 0.2]),
        _vec('v(xdut.in)', 'voltage', [0.3, 0.4]),
        _vec('v(m.n1)', 'voltage', [0.5, 0.6]),
        _vec('v(vdd)', 'voltage', [1.8, 1.8]),
        _vec('v(vp)', 'voltage', [0.9, 0.9]),
    ]
    return RawFile([RawPlot(header, vectors, 2, True, None)], source_path='sp_hier.raw')


class TestEmitViewsByBlockScopeManifest:
    def _scopes_from(self, html_path):
        import json as _json
        import re as _re
        text = Path(html_path).read_text(encoding='utf-8')
        m = _re.search(r'window\.SILK_VIEW_SCOPES = (.*?);', text)
        return _json.loads(m.group(1))

    def test_manifest_present_on_every_page(self, tmp_path):
        raw = _make_single_plot_hier_rawfile()
        results = emit_views_by_block(raw, str(tmp_path))
        for r in results:
            manifest = self._scopes_from(r.html_path)
            assert manifest['current'] is not None
            assert len(manifest['scopes']) == len(results)

    def test_manifest_keys_labels_basenames_match_pages(self, tmp_path):
        raw = _make_single_plot_hier_rawfile()
        results = emit_views_by_block(raw, str(tmp_path))
        html_basenames = {os.path.basename(r.html_path) for r in results}
        manifest = self._scopes_from(results[0].html_path)
        manifest_htmls = {s['html'] for s in manifest['scopes']}
        assert manifest_htmls == html_basenames
        for entry in manifest['scopes']:
            stem = os.path.splitext(entry['html'])[0]
            assert entry['key'] == stem
            assert entry['data_src'] == f'{stem}.data.js'
            assert entry['stamp_src'] == f'{stem}.stamp.js'
            assert os.path.exists(os.path.join(str(tmp_path), entry['data_src']))
            assert os.path.exists(os.path.join(str(tmp_path), entry['stamp_src']))

    def test_manifest_labels_match_block_paths_and_top_level(self, tmp_path):
        raw = _make_single_plot_hier_rawfile()
        results = emit_views_by_block(raw, str(tmp_path))
        manifest = self._scopes_from(results[0].html_path)
        labels_by_key = {s['key']: s['label'] for s in manifest['scopes']}
        assert labels_by_key['xdut'] == 'xdut'
        assert labels_by_key['m'] == 'm'
        assert labels_by_key['top'] == 'top-level'

    def test_manifest_n_signals_matches_actual_page_n_signals(self, tmp_path):
        # Single-plot rawfile: list_blocks' count == the page's own
        # ViewOutput.n_signals (no AC-expansion or multi-plot divergence).
        raw = _make_single_plot_hier_rawfile()
        results = emit_views_by_block(raw, str(tmp_path))
        manifest = self._scopes_from(results[0].html_path)
        n_signals_by_key = {s['key']: s['n_signals'] for s in manifest['scopes']}
        for r in results:
            key = os.path.splitext(os.path.basename(r.html_path))[0]
            assert n_signals_by_key[key] == r.n_signals

    def test_current_differs_per_page_rest_of_manifest_identical(self, tmp_path):
        raw = _make_single_plot_hier_rawfile()
        results = emit_views_by_block(raw, str(tmp_path))
        manifests = [self._scopes_from(r.html_path) for r in results]
        currents = {m['current'] for m in manifests}
        assert len(currents) == len(results)  # each page names itself
        scopes_lists = [m['scopes'] for m in manifests]
        assert all(s == scopes_lists[0] for s in scopes_lists)

    def test_no_manifest_under_inline(self, tmp_path):
        raw = _make_single_plot_hier_rawfile()
        results = emit_views_by_block(raw, str(tmp_path), inline=True)
        for r in results:
            manifest = self._scopes_from(r.html_path)
            assert manifest == {'current': None, 'scopes': []}

    def test_single_block_no_top_still_gets_one_entry_manifest(self, tmp_path):
        raw = _make_all_hier_rawfile()  # no bare/top-level signals
        results = emit_views_by_block(raw, str(tmp_path))
        assert len(results) == 1
        manifest = self._scopes_from(results[0].html_path)
        assert len(manifest['scopes']) == 1
        assert manifest['current'] == manifest['scopes'][0]['key']


# ---------------------------------------------------------------------------
# Overlapping selectors. The "did this selector match anything?" bookkeeping
# must credit EVERY selector that matches a signal, not just the first one
# tried: a selector whose match set is a subset of an earlier selector's
# (['*', 'v(out)'], or ['xdut', 'xdut.m1']) otherwise looks unmatched and
# trips the fail-loud check -- an error whose truth depended on argument
# order.
# ---------------------------------------------------------------------------

class TestOverlappingSelectors:
    NAMES = [['v(xdut.out)', 'v(xdut.m1.id)', 'v(vdd)']]

    def test_star_plus_specific_selector(self):
        masks = select_for_payload([['v(out)', 'v(in)']],
                                   signals=['*', 'v(out)'])
        assert masks == [[True, True]]

    def test_nested_hierarchy_selectors_both_orders(self):
        expected = [[True, True, False]]
        assert select_for_payload(self.NAMES, signals=['xdut', 'xdut.m1']) == expected
        assert select_for_payload(self.NAMES, signals=['xdut.m1', 'xdut']) == expected

    def test_exclude_nested_selectors_both_orders(self):
        expected = [[False, False, True]]
        assert select_for_payload(self.NAMES, exclude=['xdut', 'xdut.m1']) == expected
        assert select_for_payload(self.NAMES, exclude=['xdut.m1', 'xdut']) == expected

    def test_exclude_star_plus_specific(self):
        with pytest.raises(ViewError, match='matched zero signals'):
            # Both selectors match; the error must be the honest "selection
            # is empty" one, never "exclude= selector 'v(out)' matched no
            # signal".
            select_for_payload([['v(out)', 'v(in)']], exclude=['*', 'v(out)'])

    def test_exclude_credited_for_a_signal_already_dropped_by_signals(self):
        # 'v(vdd)' is in the file, so excluding it is not a typo -- even
        # though signals= had already dropped it.
        masks = select_for_payload(self.NAMES, signals=['xdut'], exclude=['v(vdd)'])
        assert masks == [[True, True, False]]

    def test_genuinely_bogus_selector_still_raises_with_suggestion(self):
        with pytest.raises(ViewError, match=r"selector 'xdu' matched no signal"):
            select_for_payload(self.NAMES, signals=['*', 'xdu'])
        with pytest.raises(ViewError, match=r'xdut'):  # near-miss suggestion
            select_for_payload(self.NAMES, signals=['*', 'xdu'])

    def test_bogus_exclude_selector_still_raises(self):
        with pytest.raises(ViewError, match=r"exclude= selector 'nope' matched no"):
            select_for_payload(self.NAMES, exclude=['xdut', 'nope'])

    def test_overlapping_selectors_through_the_cli_path(self, tmp_path):
        # The reviewer's repro: this exited 1 on a file that contains v(out).
        result = emit_view(A_TRAN, tmp_path / 'view.html',
                           signals=['*', 'v(out)'])
        assert result.n_signals == result.n_signals_total


def test_javascript_hierparse_matches_shared_fixture():
    import subprocess
    from tests.test_view_js_differential import _require_node

    root = Path(__file__).resolve().parents[1]
    script = r"""
import fs from 'node:fs';
const {hierParse: parse} = await import('./silk/signals/view/assets/selectors.mjs');
const cases = JSON.parse(fs.readFileSync('tests/fixtures/hierparse_cases.json', 'utf8'));
process.stdout.write(JSON.stringify(cases.map(c => parse(c))));
"""
    result = subprocess.run([_require_node(), '--input-type=module', '-e', script], cwd=root,
                            capture_output=True, text=True, check=True)
    expected = [{k: c[k] for k in ('wrapper', 'segments', 'leaf')}
                for c in HIERPARSE_CASES]
    assert json.loads(result.stdout) == expected
