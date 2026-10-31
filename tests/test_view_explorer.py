"""A vendored consumer exercises the mountable explorer in a real browser."""
import json
import math
import os
import selectors
import signal
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest

from silk.signals.view import emit_view, emit_view_data
from tests.test_view_e2e import _find_chrome, _reason_to_skip

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'silk/signals/view/assets'
CONSUMER_ASSETS = ('selectors.mjs', 'explorer.mjs', 'view.css', 'uplot.iife.min.js',
                   'waveform-core.mjs', 'LICENSE.uPlot')


def stage_consumer(directory, source=ASSETS):
    directory.mkdir(parents=True, exist_ok=True)
    for name in CONSUMER_ASSETS:
        shutil.copyfile(source / name, directory / name)
    raw = ROOT / 'tests/fixtures/rawfiles/c_tran_ascii_cli.raw'
    shutil.copyfile(raw, directory / 'fixture.raw')
    (directory / 'payload.json').write_text(json.dumps(emit_view_data(raw)), encoding='utf-8')


def run_consumer(directory, source=ASSETS):
    stage_consumer(directory, source)
    result = subprocess.run(
        ['node', str(ROOT / 'tests/browser/explorer_e2e.mjs'), str(directory), _find_chrome()],
        cwd=ROOT / 'tests/browser', capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(_reason_to_skip() is not None, reason=_reason_to_skip() or '')
def test_vendored_explorer_two_instances(tmp_path):
    run_consumer(tmp_path / 'host')


@pytest.mark.skipif(_reason_to_skip() is not None, reason=_reason_to_skip() or '')
def test_toolbar_wraps_at_embedded_panel_widths(tmp_path):
    """TASK-241: a host embeds this explorer far narrower than a browser
    window, with no visible scrollbar to discover a horizontally scrolling
    toolbar. Every control must stay reachable, at a consistent size, and
    the plot area must yield the height wrapped rows need."""
    directory = tmp_path / 'host'
    stage_consumer(directory)
    result = subprocess.run(
        ['node', str(ROOT / 'tests/browser/explorer_toolbar_wrap_e2e.mjs'), str(directory), _find_chrome()],
        cwd=ROOT / 'tests/browser', capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(_reason_to_skip() is not None, reason=_reason_to_skip() or '')
def test_explorer_css_does_not_restyle_host():
    result = subprocess.run(
        ['node', str(ROOT / 'tests/browser/explorer_css_e2e.mjs'), str(ASSETS / 'view.css'), _find_chrome()],
        cwd=ROOT / 'tests/browser', capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(_reason_to_skip() is not None, reason=_reason_to_skip() or '')
def test_standalone_explorer_transport(tmp_path):
    raw = ROOT / 'tests/fixtures/rawfiles/c_tran_ascii_cli.raw'
    scopes = {'current': 'main', 'scopes': [
        {'key': key, 'label': key, 'html': key + '.html', 'data_src': key + '.data.js',
         'stamp_src': key + '.stamp.js', 'n_signals': 1} for key in ('main', 'sibling')
    ]}
    emit_view(raw, tmp_path / 'sibling.html', context={'title': 'Sibling context'}, poll_ms=50)
    options = dict(scopes=scopes['scopes'], poll_ms=50,
                   annotations={'notes': [{'signal': 'IN', 'text': 'Preserved annotation'}]},
                   view_state={'v': 1, 'cols': 2},
                   commands=[{'cmd': 'setCols', 'args': {'n': 1}}])
    emit_view(raw, tmp_path / 'main.html', title='Initial title', **options)
    emit_view(raw, tmp_path / 'paused.html', title='Initial title', poll_ms=0)
    emit_view(raw, tmp_path / 'inline.html', title='Initial title', inline=True, poll_ms=50)
    proc = subprocess.Popen(
        ['node', str(ROOT / 'tests/browser/explorer_standalone_e2e.mjs'), str(tmp_path), _find_chrome()],
        cwd=ROOT / 'tests/browser', stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, start_new_session=True,
    )
    try:
        with selectors.DefaultSelector() as ready:
            ready.register(proc.stdout, selectors.EVENT_READ)
            assert ready.select(timeout=45), 'Browser did not signal readiness within 45 seconds'
            line = proc.stdout.readline()
        if line != 'READY\n':
            stdout, stderr = proc.communicate(timeout=15)
            pytest.fail(line + stdout + stderr)
        emit_view(raw, tmp_path / 'main.html', title='Polled refresh', **options)
        emit_view(raw, tmp_path / 'paused.html', title='Polled refresh', poll_ms=0)
        emit_view(raw, tmp_path / 'inline.html', title='Polled refresh', inline=True, poll_ms=50)
        stdout, stderr = proc.communicate(input='REEMITTED\n', timeout=60)
        assert proc.returncode == 0, stdout + stderr
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.communicate(timeout=10)


@pytest.mark.skipif(_reason_to_skip() is not None, reason=_reason_to_skip() or '')
def test_explorer_row_stability_on_selection(tmp_path):
    """TASK-338: a plain click that selects a leaf row must not move any
    picker row under the pointer, in a full-size standalone mount and in a
    short, narrow mount matching how a host embeds this explorer -- and a
    double-click from an empty selection must plot exactly the clicked leaf
    in both."""
    directory = tmp_path / 'host'
    stage_consumer(directory)
    result = subprocess.run(
        ['node', str(ROOT / 'tests/browser/explorer_row_stability_e2e.mjs'), str(directory), _find_chrome()],
        cwd=ROOT / 'tests/browser', capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(_reason_to_skip() is not None, reason=_reason_to_skip() or '')
def test_explorer_picker_compact_sweep(tmp_path):
    """TASK-341: below a container-query-derived sidebar-height breakpoint,
    the picker's action rows collapse into one anchored "more actions" menu
    and the count moves inline into the filter row, so a short embedded
    mount still shows a usable list. Sweeps the sidebar's own rendered
    height from 150px to 420px in 10px steps (crossing the breakpoint) and
    at every step asserts at least 5 fully visible rows plus TASK-338's
    no-layout-shift guarantee (a plain click leaves every row's position
    unchanged, a double-click plots exactly the clicked leaf) -- checked
    with the compact menu both closed and, below the breakpoint, open."""
    directory = tmp_path / 'host'
    stage_consumer(directory)
    # A 5-signal fixture (not the 3-signal default): "at least 5 fully
    # visible rows" needs at least 5 rows to show in the first place.
    raw = ROOT / 'tests/fixtures/rawfiles/e_ng_tran.raw'
    (directory / 'sweep_payload.json').write_text(json.dumps(emit_view_data(raw)), encoding='utf-8')
    result = subprocess.run(
        ['node', str(ROOT / 'tests/browser/explorer_picker_compact_sweep_e2e.mjs'), str(directory), _find_chrome()],
        cwd=ROOT / 'tests/browser', capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(_reason_to_skip() is not None, reason=_reason_to_skip() or '')
def test_explorer_panel_units(tmp_path):
    raw = ROOT / 'tests/fixtures/rawfiles/a_ac_bin.raw'
    html = tmp_path / 'ac.html'
    emit_view(raw, html, inline=True, annotations={
        'markers': [
            {'axis': 'y', 'signal': 'v(out) (mag)', 'value': 0.9997, 'label': 'test-mag'},
            {'axis': 'y', 'signal': 'v(out) (phase)', 'value': -1, 'label': 'test-phase'},
        ],
        'bands': [{'axis': 'y', 'signal': 'v(out) (phase)', 'lo': -1.5, 'hi': -0.5}],
    })
    unitless = tmp_path / 'units.raw'
    unitless.write_text('''Title: Literal units
Plotname: Transient Analysis
Flags: real
No. Variables: 6
No. Points: 3
Variables:
0 time time
1 v(out) voltage
2 i(v1) current
3 unknown notype
4 also_unknown unrecognized
5 p(out) power
Values:
0 0 1 10 100 1000 10000
1 1 2 20 200 2000 20000
2 2 3 30 300 3000 30000
''', encoding='utf-8')
    unitless_html = tmp_path / 'units.html'
    emit_view(unitless, unitless_html, inline=True)
    result = subprocess.run(
        ['node', str(ROOT / 'tests/browser/explorer_units_e2e.mjs'), str(html), _find_chrome(), str(unitless_html)],
        cwd=ROOT / 'tests/browser', capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.slow
@pytest.mark.skipif(_reason_to_skip() is not None, reason=_reason_to_skip() or '')
def test_wheel_explorer_is_byte_identical_and_mounts(tmp_path):
    pytest.importorskip('build')
    result = subprocess.run(
        [sys.executable, '-m', 'build', '--wheel', '-o', str(tmp_path / 'dist')],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    wheel, = (tmp_path / 'dist').glob('*.whl')
    extracted = tmp_path / 'extracted'
    extracted.mkdir()
    with zipfile.ZipFile(wheel) as archive:
        for name in CONSUMER_ASSETS:
            data = archive.read('silk/signals/view/assets/' + name)
            assert data == (ASSETS / name).read_bytes(), name
            (extracted / name).write_bytes(data)
    run_consumer(tmp_path / 'consumer', source=extracted)


@pytest.mark.skipif(_reason_to_skip() is not None, reason=_reason_to_skip() or '')
def test_explorer_derived_signals(tmp_path):
    for name in CONSUMER_ASSETS:
        shutil.copyfile(ASSETS / name, tmp_path / name)
    (tmp_path / 'ac.json').write_text(json.dumps(emit_view_data(
        ROOT / 'tests/fixtures/rawfiles/e_ng_ac.raw')), encoding='utf-8')
    html = tmp_path / 'derived.html'
    emit_view(ROOT / 'tests/fixtures/rawfiles/c_tran_ascii_cli.raw', html, inline=True)
    result = subprocess.run(
        ['node', str(ROOT / 'tests/browser/explorer_derived_e2e.mjs'), str(html), _find_chrome()],
        cwd=ROOT / 'tests/browser', capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(_reason_to_skip() is not None, reason=_reason_to_skip() or '')
def test_explorer_derived_state(tmp_path):
    for name in CONSUMER_ASSETS:
        shutil.copyfile(ASSETS / name, tmp_path / name)
    raw = ROOT / 'tests/fixtures/rawfiles/c_tran_ascii_cli.raw'
    html = tmp_path / 'derived-state.html'
    emit_view(raw, html, inline=True)
    result = subprocess.run(
        ['node', str(ROOT / 'tests/browser/explorer_derived_state_e2e.mjs'),
         str(html), str(raw), _find_chrome()],
        cwd=ROOT / 'tests/browser', capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(_reason_to_skip() is not None, reason=_reason_to_skip() or '')
def test_explorer_current_signal_shows_full_name(tmp_path):
    """TASK-317: a current signal's picker row reads "i(v1)", not "v1";
    nested under a group, it reads "i(vsup)", not the full "i(xdut.vsup)".

    group_signals_by_kind=False here: this test is about the label TEXT
    (TASK-317/318), not about TASK-319's sections -- with grouping at its
    new default (on), i(v1)/i(vvclk) would sit behind a closed Currents
    section this test never opens. TASK-319 has its own coverage for the
    label unaffected by section boundaries.
    """
    html = tmp_path / 'current-label.html'
    emit_view(ROOT / 'tests/fixtures/rawfiles/a_tran_ascii.raw', html, inline=True,
              group_signals_by_kind=False)

    nested = tmp_path / 'nested.raw'
    nested.write_text('''Title: Nested current
Plotname: Transient Analysis
Flags: real
No. Variables: 4
No. Points: 2
Variables:
0 time time
1 v(xdut.vsup) voltage
2 i(vvclk) current
3 i(xdut.vsup) current
Values:
0 0 0 0 0
1 1 1 1 2
''', encoding='utf-8')
    nested_html = tmp_path / 'nested.html'
    emit_view(nested, nested_html, inline=True, group_signals_by_kind=False)

    result = subprocess.run(
        ['node', str(ROOT / 'tests/browser/explorer_current_label_e2e.mjs'),
         str(html), _find_chrome(), str(nested_html)],
        cwd=ROOT / 'tests/browser', capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(_reason_to_skip() is not None, reason=_reason_to_skip() or '')
def test_explorer_first_click_on_twisty_registers(tmp_path):
    """The first, unprimed click on a twisty toggles it (TASK-322)."""
    raw = tmp_path / 'nested.raw'
    raw.write_text('''Title: Nested current
Plotname: Transient Analysis
Flags: real
No. Variables: 4
No. Points: 2
Variables:
0 time time
1 v(xdut.vsup) voltage
2 i(vvclk) current
3 i(xdut.vsup) current
Values:
0 0 0 0 0
1 1 1 1 2
''', encoding='utf-8')
    html = tmp_path / 'first_click.html'
    emit_view(raw, html, inline=True, group_signals_by_kind=False)

    result = subprocess.run(
        ['node', str(ROOT / 'tests/browser/explorer_first_click_e2e.mjs'), str(html), _find_chrome()],
        cwd=ROOT / 'tests/browser', capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


_KIND_SECTIONS_RAW = '''Title: Kind sections test
Plotname: Transient Analysis
Flags: real
No. Variables: 6
No. Points: 2
Variables:
0 time time
1 v(vdd) voltage
2 v(xdut.vsup) voltage
3 i(vvclk) current
4 i(xdut.vsup) current
5 p(out) power
Values:
0 0 0 0 0 0 0
1 1 1 1 1 2 3
'''


@pytest.mark.skipif(_reason_to_skip() is not None, reason=_reason_to_skip() or '')
def test_explorer_kind_sections(tmp_path):
    """TASK-319: Voltages/Currents get their own collapsible section, in
    table order, nested groups behave like today inside a section, an
    unsectioned kind (power) stays in the ordinary tree, and collapsing a
    section never touches what is plotted."""
    raw = tmp_path / 'kind_sections.raw'
    raw.write_text(_KIND_SECTIONS_RAW, encoding='utf-8')
    html = tmp_path / 'kind_sections.html'
    emit_view(raw, html, inline=True)

    result = subprocess.run(
        ['node', str(ROOT / 'tests/browser/explorer_kind_sections_e2e.mjs'), str(html), _find_chrome()],
        cwd=ROOT / 'tests/browser', capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(_reason_to_skip() is not None, reason=_reason_to_skip() or '')
def test_explorer_kind_sections_off_matches_pre_task_319(tmp_path):
    """TASK-319 AC#3: group_signals_by_kind=False reproduces the exact
    (pre-TASK-319) picker rows on the same fixture used by
    test_explorer_current_signal_shows_full_name -- see the baseline capture
    note in explorer_kind_sections_off_e2e.mjs."""
    raw = tmp_path / 'nested.raw'
    raw.write_text('''Title: Nested current
Plotname: Transient Analysis
Flags: real
No. Variables: 4
No. Points: 2
Variables:
0 time time
1 v(xdut.vsup) voltage
2 i(vvclk) current
3 i(xdut.vsup) current
Values:
0 0 0 0 0
1 1 1 1 2
''', encoding='utf-8')
    html = tmp_path / 'nested_off.html'
    emit_view(raw, html, inline=True, group_signals_by_kind=False)

    result = subprocess.run(
        ['node', str(ROOT / 'tests/browser/explorer_kind_sections_off_e2e.mjs'), str(html), _find_chrome()],
        cwd=ROOT / 'tests/browser', capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


_PRE_FEATURE_LAYOUT_RAW = '''Title: Pre-feature layout test
Plotname: Transient Analysis
Flags: real
No. Variables: 4
No. Points: 2
Variables:
0 time time
1 v(vdd) voltage
2 i(vvclk) current
3 p(xdut.out) power
Values:
0 0 0 0 0
1 1 1 1 2
'''


@pytest.mark.skipif(_reason_to_skip() is not None, reason=_reason_to_skip() or '')
def test_explorer_kind_sections_pre_feature_layout_restore(tmp_path):
    """TASK-319, reviewer decision (c): a saved layout from before this
    feature has an 'expanded' array but no 'sections' field at all. On
    restore that absence means "apply the KIND_SECTIONS defaults" (Voltages
    open, Currents closed) -- never "the user closed both", which is what
    an explicit empty 'sections' array would mean. Restoring must also
    leave that layout's ordinary tree-group expansion (the ``xdut`` power
    group, an unsectioned "any other kind" the sections feature never
    touches) exactly as saved, proving 'expanded' and 'sections' restore
    independently of each other."""
    raw = tmp_path / 'pre_feature.raw'
    raw.write_text(_PRE_FEATURE_LAYOUT_RAW, encoding='utf-8')
    html = tmp_path / 'pre_feature.html'
    emit_view(raw, html, inline=True)

    result = subprocess.run(
        ['node', str(ROOT / 'tests/browser/explorer_pre_feature_layout_e2e.mjs'), str(html), _find_chrome()],
        cwd=ROOT / 'tests/browser', capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _sample_marks_raw(path, n=4000):
    lines = ['Title: Sample marks', 'Plotname: Transient Analysis', 'Flags: real',
             'No. Variables: 2', f'No. Points: {n}', 'Variables:', '0 time time', '1 v(out) voltage',
             'Values:']
    for i in range(n):
        lines.append(f'{i} {i * 1e-9:.6e} {math.sin(i * 0.05):.6e}')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path


@pytest.mark.skipif(_reason_to_skip() is not None, reason=_reason_to_skip() or '')
def test_sample_marks_toggle_standalone(tmp_path):
    """TASK-293: the Samples toggle (button and `s` key) marks every simulated
    sample; off by default; survives zoom/pan; dense views are hidden with a
    note; no layout shift."""
    raw = _sample_marks_raw(tmp_path / 'dense.raw')
    html = tmp_path / 'marks.html'
    emit_view(raw, html, inline=True)
    result = subprocess.run(
        ['node', str(ROOT / 'tests/browser/explorer_sample_marks_e2e.mjs'), 'standalone',
         str(html), _find_chrome()],
        cwd=ROOT / 'tests/browser', capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(_reason_to_skip() is not None, reason=_reason_to_skip() or '')
def test_sample_marks_toggle_embedded(tmp_path):
    """TASK-293: the same contract in a createExplorer mount."""
    directory = tmp_path / 'host'
    stage_consumer(directory)
    raw = _sample_marks_raw(tmp_path / 'dense.raw')
    (directory / 'payload.json').write_text(json.dumps(emit_view_data(raw)), encoding='utf-8')
    result = subprocess.run(
        ['node', str(ROOT / 'tests/browser/explorer_sample_marks_e2e.mjs'), 'embedded',
         str(directory), _find_chrome()],
        cwd=ROOT / 'tests/browser', capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(_reason_to_skip() is not None, reason=_reason_to_skip() or '')
def test_no_stray_pixel_at_plot_corner_standalone(tmp_path):
    """TASK-347: uPlot's empty selection box (0x0 at the plot origin) must
    paint nothing; a border on it left a 2x2 accent dot at the top-left."""
    raw = _sample_marks_raw(tmp_path / 'dense.raw')
    html = tmp_path / 'stray.html'
    emit_view(raw, html, inline=True)
    result = subprocess.run(
        ['node', str(ROOT / 'tests/browser/explorer_stray_select_e2e.mjs'), 'standalone',
         str(html), _find_chrome()],
        cwd=ROOT / 'tests/browser', capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(_reason_to_skip() is not None, reason=_reason_to_skip() or '')
def test_no_stray_pixel_at_plot_corner_embedded(tmp_path):
    """TASK-347: the same contract in a createExplorer mount."""
    directory = tmp_path / 'host'
    stage_consumer(directory)
    raw = _sample_marks_raw(tmp_path / 'dense.raw')
    (directory / 'payload.json').write_text(json.dumps(emit_view_data(raw)), encoding='utf-8')
    result = subprocess.run(
        ['node', str(ROOT / 'tests/browser/explorer_stray_select_e2e.mjs'), 'embedded',
         str(directory), _find_chrome()],
        cwd=ROOT / 'tests/browser', capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _expand_all_raw(path):
    names = [('v(x1.a)', 'voltage'), ('v(x1.b)', 'voltage'), ('v(x2.c)', 'voltage'),
             ('i(v.x1.d)', 'current'), ('i(v.x1.e)', 'current'), ('i(v.x2.f)', 'current')]
    lines = ['Title: Expand all', 'Plotname: Transient Analysis', 'Flags: real',
             f'No. Variables: {len(names) + 1}', 'No. Points: 3', 'Variables:', '0 time time']
    lines += [f'{i + 1} {n} {t}' for i, (n, t) in enumerate(names)]
    lines.append('Values:')
    for i in range(3):
        lines.append(f'{i} {i * 1e-9:.6e} ' + ' '.join(f'{0.1 * (k + 1) * (i + 1):.6e}' for k in range(len(names))))
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path


@pytest.mark.skipif(_reason_to_skip() is not None, reason=_reason_to_skip() or '')
def test_expand_all_opens_kind_sections_standalone(tmp_path):
    """TASK-360: Expand all opens every kind section and the groups inside."""
    raw = _expand_all_raw(tmp_path / 'ea.raw')
    html = tmp_path / 'ea.html'
    emit_view(raw, html, inline=True)
    result = subprocess.run(
        ['node', str(ROOT / 'tests/browser/explorer_expand_all_e2e.mjs'), 'standalone',
         str(html), _find_chrome()],
        cwd=ROOT / 'tests/browser', capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(_reason_to_skip() is not None, reason=_reason_to_skip() or '')
def test_expand_all_opens_kind_sections_embedded(tmp_path):
    """TASK-360: the same contract in a createExplorer mount."""
    directory = tmp_path / 'host'
    stage_consumer(directory)
    raw = _expand_all_raw(tmp_path / 'ea.raw')
    (directory / 'payload.json').write_text(json.dumps(emit_view_data(raw)), encoding='utf-8')
    result = subprocess.run(
        ['node', str(ROOT / 'tests/browser/explorer_expand_all_e2e.mjs'), 'embedded',
         str(directory), _find_chrome()],
        cwd=ROOT / 'tests/browser', capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(_reason_to_skip() is not None, reason=_reason_to_skip() or '')
def test_explorer_shows_label_as_primary_text(tmp_path):
    """TASK-527/530: labels= is the primary text (raw secondary); the raw name
    stays the row identity; the filter searches both."""
    html = tmp_path / 'labels.html'
    emit_view(ROOT / 'tests/fixtures/rawfiles/a_tran_ascii.raw', html, inline=True,
              group_signals_by_kind=False, labels={'v(in)': '<b>in[0]</b>'})
    result = subprocess.run(
        ['node', str(ROOT / 'tests/browser/explorer_labels_e2e.mjs'), str(html), _find_chrome()],
        cwd=ROOT / 'tests/browser', capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(_reason_to_skip() is not None, reason=_reason_to_skip() or '')
def test_label_only_reemit_restores_state_and_annotations_and_group_collisions(tmp_path):
    """TASK-530: save non-default state, re-emit with other labels (run.id
    changes, state_key does not): state and raw-key annotations restore, raw
    identities hold, and collision groups keep their raw segments."""
    import numpy as np
    from silk.signals.rawfile import RawFile, RawPlot, RawVector, write_rawfile
    n = 6
    names = ['v(xx1<0>.mid)', 'v(xx1[0].mid)', 'v(other<1>.a)']
    vecs = [RawVector('time', 'time', data=np.arange(1, n + 1, dtype=float))] + [
        RawVector(nm, 'voltage', data=np.arange(n, dtype=float) + i) for i, nm in enumerate(names)]
    raw = tmp_path / 'g.raw'
    write_rawfile(raw, RawFile([RawPlot({'plotname': 'Transient Analysis', 'title': 't'}, vecs,
                                        npoints=n, real=True, padded=True)]), binary=False)
    html = tmp_path / 'g.html'
    notes = {'markers': [{'signal': 'v(other<1>.a)', 'axis': 'x', 'value': 2.0}],
             'notes': [{'signal': 'v(other<1>.a)', 'text': 'check me', 'severity': 'warn'}]}
    emit_view(raw, html, inline=True, labels='designer', annotations=notes, group_signals_by_kind=False)
    code = ("from silk.signals.view import emit_view; emit_view(%r, %r, inline=True, "
            "labels={'v(other<1>.a)': 'OTHER'}, annotations=%r, group_signals_by_kind=False)"
            % (str(raw), str(html), notes))
    script = json.dumps([sys.executable, '-c', code])
    result = subprocess.run(
        ['node', str(ROOT / 'tests/browser/explorer_labels_reemit_e2e.mjs'), str(html), _find_chrome(), script],
        cwd=ROOT / 'tests/browser', capture_output=True, text=True, timeout=120,
        env={**__import__('os').environ, 'PYTHONPATH': str(ROOT)})
    assert result.returncode == 0, result.stdout + result.stderr


def test_named_derived_restore_and_replace_in_configured_browser(tmp_path):
    """Named math follows the host rerun lifecycle using the configured browser."""
    from tests.test_view_op_e2e import _run_browser

    _run_browser(tmp_path, 'explorer_named_derived.mjs')
