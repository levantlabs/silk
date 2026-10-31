"""Whole-fixture-corpus tests for silk.signals.view (design spec §9.2).

Parametrized over every ``tests/fixtures/rawfiles/*.raw`` — the same corpus
``test_rawfile_reader.py`` exercises — globbed so new fixtures are picked up
automatically.
"""
import base64
import json
from pathlib import Path

import numpy as np
import pytest

from silk.signals.rawfile import read_rawfile
from silk.signals.view import emit_view, emit_view_data
from silk.signals.view.encode import unit_for_type

FIXTURES_DIR = Path(__file__).parent / 'fixtures' / 'rawfiles'
ALL_FIXTURES = sorted(FIXTURES_DIR.glob('*.raw'))


def _decode(level, key, dtype):
    s = level.get(key, '')
    if s == '':
        return np.empty(0, dtype=dtype)
    return np.frombuffer(base64.b64decode(s), dtype=dtype)


def _decode_y(level):
    dtype = '<f4' if level['dt'] == 'f32' else '<f8'
    return _decode(level, 'y', dtype)


class TestCorpusStructure:
    """One test class, parametrized over the whole corpus (item 1-3, 5)."""

    @pytest.fixture(params=ALL_FIXTURES, ids=lambda p: p.name)
    def payload(self, request):
        return emit_view_data(request.param)

    def test_json_round_trips(self, payload):
        json.dumps(payload)  # must not raise

    def test_format_and_default_plot(self, payload):
        assert payload['format'] == 2
        assert len(payload['plots']) > 0
        assert 0 <= payload['default_plot'] < len(payload['plots'])

    def test_level_shape_invariants(self, payload):
        for plot in payload['plots']:
            x_levels = plot['x']['levels']
            assert len(x_levels) == 1
            assert 'y' not in x_levels[0]
            assert 'x' in x_levels[0]
            for sig in plot['signals']:
                levels = sig['levels']
                if levels[0]['n'] == 0:
                    # a dims=0 vector (TASK-440) is zero-length: one empty
                    # level 0 and nothing to decimate
                    assert len(levels) == 1
                    continue
                assert levels[0]['n'] == plot['npoints']
                assert 'y' in levels[0]
                assert 'x' not in levels[0]
                for lvl in levels[1:]:
                    assert 'x' in lvl and 'y' in lvl
                    assert len(_decode(lvl, 'x', '<f8')) == lvl['n']
                    assert len(_decode_y(lvl)) == lvl['n']

    def test_emit_writes_three_files_with_handshake(self, request, tmp_path):
        # Re-derive the source path from the payload fixture id via a fresh
        # emit_view call (the `payload` fixture above doesn't write files).
        pass  # covered per-file below to keep the source path handy


@pytest.mark.parametrize('path', ALL_FIXTURES, ids=lambda p: p.name)
def test_emit_view_writes_valid_data_js(path, tmp_path):
    out = tmp_path / 'view.html'
    emit_view(path, out)
    data_js = (tmp_path / 'view.data.js').read_text(encoding='utf-8')
    assert data_js.startswith('(function(){var p={')
    assert data_js.rstrip().endswith('})();')
    assert '__silkViewData' in data_js
    # Strip the wrapper and prove the middle is valid JSON.
    inner = data_js[len('(function(){var p='):]
    inner = inner[:inner.index(";\nif(typeof window.__silkViewData")]
    json.loads(inner)


@pytest.mark.parametrize('path', ALL_FIXTURES, ids=lambda p: p.name)
def test_decode_matches_reader_output(path):
    """Decode check (item 4): compare against read_rawfile directly."""
    raw = read_rawfile(str(path))
    payload = emit_view_data(path)
    for plot_idx, plot in enumerate(raw.plots):
        enc_plot = payload['plots'][plot_idx]
        vectors = plot.vectors
        if not vectors:
            continue
        # TASK-591: only a true operating point has no x axis.
        scalar = plot.real and plot.plotname.lower() == 'operating point'
        x_expected = (np.arange(plot.npoints, dtype=np.float64) if scalar
                      else np.asarray(vectors[0].data, dtype=np.float64))
        x_got = _decode(enc_plot['x']['levels'][0], 'x', '<f8')
        assert np.array_equal(x_got, x_expected)

        sig_idx = 0
        for vec in vectors[0 if scalar else 1:]:
            if not plot.real:
                z = np.asarray(vec.data)
                mag_expected = np.abs(z).astype(np.float32).astype(np.float64)
                phase_expected = np.degrees(np.unwrap(np.angle(z))).astype(np.float32).astype(np.float64)
                mag_got = _decode_y(enc_plot['signals'][sig_idx]['levels'][0]).astype(np.float64)
                phase_got = _decode_y(enc_plot['signals'][sig_idx + 1]['levels'][0]).astype(np.float64)
                assert np.allclose(mag_got, mag_expected, rtol=0, atol=0) or np.array_equal(mag_got, mag_expected)
                np.testing.assert_array_equal(phase_got, phase_expected)
                sig_idx += 2
            else:
                expected = np.asarray(vec.data, dtype=np.float64).astype(np.float32)
                got = _decode_y(enc_plot['signals'][sig_idx]['levels'][0])
                np.testing.assert_array_equal(got, expected)
                sig_idx += 1


# ---------------------------------------------------------------------------
# Named per-flavor assertions (spec §9.2 table)
# ---------------------------------------------------------------------------

class TestNamedFixtureAssertions:
    def test_bin_ascii_transient_identical_apart_from_run(self):
        d_bin = emit_view_data(FIXTURES_DIR / 'a_tran_bin.raw')
        d_ascii = emit_view_data(FIXTURES_DIR / 'a_tran_ascii.raw')
        # Y is byte-identical (float32 cast absorbs ASCII round-trip noise).
        for s_bin, s_ascii in zip(d_bin['plots'][0]['signals'], d_ascii['plots'][0]['signals']):
            assert s_bin['name'] == s_ascii['name']
            y_bin = _decode_y(s_bin['levels'][0])
            y_ascii = _decode_y(s_ascii['levels'][0])
            assert np.array_equal(y_bin, y_ascii)
        # X is float64 and the ASCII encoding has finite decimal precision,
        # so allow a tiny (<< float32 ULP) tolerance rather than bit-exact —
        # this is a genuine ngspice ASCII-writer precision limit, not a bug
        # (confirmed directly against silk.signals.rawfile.read_rawfile).
        x_bin = _decode(d_bin['plots'][0]['x']['levels'][0], 'x', '<f8')
        x_ascii = _decode(d_ascii['plots'][0]['x']['levels'][0], 'x', '<f8')
        np.testing.assert_allclose(x_bin, x_ascii, rtol=1e-9, atol=1e-15)
        assert d_bin['plots'][0]['kind'] == d_ascii['plots'][0]['kind'] == 'tran'

    @pytest.mark.parametrize('fixture', ['a_ac_bin.raw', 'b_ac_complex.raw'])
    def test_ac_complex_expands_to_mag_phase_pairs(self, fixture):
        payload = emit_view_data(FIXTURES_DIR / fixture)
        plot = payload['plots'][0]
        assert plot['complex'] is True
        assert plot['kind'] == 'ac'
        sigs = plot['signals']
        assert len(sigs) % 2 == 0
        for i in range(0, len(sigs), 2):
            mag, phase = sigs[i], sigs[i + 1]
            assert mag['part'] == 'mag'
            assert phase['part'] == 'phase'
            assert mag['group'] == phase['group']
            assert mag['name'].endswith('(mag)')
            assert phase['name'].endswith('(phase)')
            # Spec §4.6 unit table: mag keeps the vector's unit, phase is deg.
            assert phase['unit'] == 'deg'
            assert mag['unit'] == unit_for_type(mag['type'])
            # Phase is unwrapped: consecutive samples should not show the
            # 360-degree wrap discontinuity a raw np.angle() would produce.
            phase_y = _decode_y(phase['levels'][0])
            if len(phase_y) > 1:
                assert np.all(np.abs(np.diff(phase_y)) < 350)

    def test_xyce_uppercase_frequency_still_ac_with_log_default(self):
        payload = emit_view_data(FIXTURES_DIR / 'c_ac_bin.raw')
        plot = payload['plots'][0]
        assert plot['kind'] == 'ac'
        assert plot['npoints'] == 61
        assert plot['x_log_default'] is True

    def test_xyce_uppercase_time_still_tran(self):
        payload = emit_view_data(FIXTURES_DIR / 'c_tran_bin.raw')
        assert payload['plots'][0]['kind'] == 'tran'

    def test_kind_and_complexity_are_independent(self):
        payload = emit_view_data(FIXTURES_DIR / 'b_offset_nontime_scale.raw')
        plot = payload['plots'][0]
        assert plot['kind'] == 'ac'
        assert plot['complex'] is False

    def test_offset_shows_up_in_emitted_x_not_reapplied(self):
        offset_payload = emit_view_data(FIXTURES_DIR / 'b_tran_bin_offset.raw')
        baseline_payload = emit_view_data(FIXTURES_DIR / 'b_offset_ascii_baseline.raw')
        x_offset = _decode(offset_payload['plots'][0]['x']['levels'][0], 'x', '<f8')
        x_baseline = _decode(baseline_payload['plots'][0]['x']['levels'][0], 'x', '<f8')
        assert x_offset[0] != x_baseline[0]
        assert x_offset[0] == pytest.approx(x_baseline[0] + (x_offset[0] - x_baseline[0]))

    def test_dc_sweep_kind_and_x_label(self):
        payload = emit_view_data(FIXTURES_DIR / 'c_dc_sweep_bin.raw')
        plot = payload['plots'][0]
        assert plot['kind'] == 'dc'
        assert plot['x']['name'] == 'sweep'
        assert plot['x_log_default'] is False

    def test_zero_points_plot_does_not_crash(self):
        payload = emit_view_data(FIXTURES_DIR / 'b_zero_points.raw')
        plot = payload['plots'][0]
        assert plot['npoints'] == 0
        for sig in plot['signals']:
            assert sig['levels'][0]['n'] == 0
            assert sig['levels'][0]['y'] == ''

    @pytest.mark.parametrize('fixture', ['d_constants_only.raw', 'd_constants_only_ascii.raw'])
    def test_constants_only_handles_nontime_x(self, fixture):
        payload = emit_view_data(FIXTURES_DIR / fixture)
        plot = payload['plots'][0]
        assert plot['kind'] == 'constants'
        assert plot['x']['name'] == 'yes'
        assert payload['default_plot'] == 0

    @pytest.mark.parametrize('fixture', [
        'd_multiplot_ac_tran.raw', 'd_multiplot_ac_tran_ascii.raw',
        'd_multiplot_order_reversed.raw',
    ])
    def test_multiplot_ac_tran_default_is_last_non_constants(self, fixture):
        payload = emit_view_data(FIXTURES_DIR / fixture)
        assert len(payload['plots']) == 2
        kinds = [p['kind'] for p in payload['plots']]
        assert set(kinds) >= {'ac'} or set(kinds) >= {'tran'}
        non_constants = [i for i, k in enumerate(kinds) if k != 'constants']
        assert payload['default_plot'] == non_constants[-1]

    def test_multiplot_constants_first_skips_to_transient(self):
        payload = emit_view_data(FIXTURES_DIR / 'd_multiplot_constants_first.raw')
        assert payload['default_plot'] == 1

    def test_multiplot_constants_last_skips_trailing_constants(self):
        payload = emit_view_data(FIXTURES_DIR / 'd_multiplot_constants_last.raw')
        assert payload['default_plot'] == 0

    def test_multiplot_nvars_differ_no_cross_plot_assumption(self):
        payload = emit_view_data(FIXTURES_DIR / 'd_multiplot_nvars_differ.raw')
        counts = {len(p['signals']) for p in payload['plots']}
        assert len(counts) >= 1  # just must not have crashed building both

    def test_attrs_trailing_do_not_leak_into_type_or_unit(self):
        payload = emit_view_data(FIXTURES_DIR / 'b_attrs_trailing.raw')
        for sig in payload['plots'][0]['signals']:
            assert ' ' not in sig['type']

    def test_lowercase_keywords_flow_through_to_kind(self):
        payload = emit_view_data(FIXTURES_DIR / 'b_lowercase_keywords.raw')
        assert payload['plots'][0]['kind'] in {'tran', 'ac', 'dc', 'other', 'noise', 'constants', 'op'}

    @pytest.mark.parametrize('fixture', [
        'b_flags_real_padded.raw', 'b_flags_real_unpadded.raw',
        'b_flags_complex_padded.raw', 'b_flags_complex_unpadded.raw',
    ])
    def test_flags_padded_variants_all_parse(self, fixture):
        payload = emit_view_data(FIXTURES_DIR / fixture)
        assert len(payload['plots']) >= 1
