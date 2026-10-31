"""TASK-533: scalar and zero-span results must show values, not invalid axes."""
import json
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import os
import signal
import threading
from pathlib import Path
import shutil
import subprocess

import pytest

from silk.signals.view import emit_view, emit_view_data
from tests._env import browser_executable

ROOT = Path(__file__).resolve().parents[1]


def raw_plot(name, vectors, rows, complex=False):
    values = lambda row: '\n\t'.join(str(v) + (',0' if complex else '') for v in row)
    return (f'Title: scalar regression\nDate: fixture\nPlotname: {name}\n'
            f'Flags: {"complex" if complex else "real"}\nNo. Variables: {len(vectors)}\n'
            f'No. Points: {len(rows)}\nVariables:\n'
            + ''.join(f'{i}\t{n}\t{t}\n' for i, (n, t) in enumerate(vectors))
            + 'Values:\n' + ''.join(f'{i}\t{values(row)}\n' for i, row in enumerate(rows)))


def op_raw():
    return raw_plot('Operating Point', [('v(in)', 'voltage'), ('v(out)', 'voltage'),
                                       ('i(v1)', 'current')], [[1, 2, -0.003]])


def ac_raw():
    return raw_plot('AC Analysis', [('frequency', 'frequency'), ('v(out)', 'voltage')],
                    [[10**(i / 10), 2] for i in range(71)], complex=True)


@pytest.mark.parametrize('combined', [False, True], ids=['op', 'ac-op'])
def test_op_browser_shows_every_value(tmp_path, combined):
    raw = tmp_path / 'op.raw'
    raw.write_text((ac_raw() if combined else '') + op_raw())
    emit_view(raw, tmp_path / 'view.html')
    # Capture errors from the emitted page itself, before its startup script.
    view = tmp_path / 'view.html'
    view.write_text('<script>window.addEventListener("error",e=>parent.postMessage({error:e.message},"*"));</script>' + view.read_text())
    _run_browser(tmp_path, 'explorer_op_standalone.mjs')


def _run_browser(tmp_path, module):
    """Same direct-browser/report protocol as the core's Firefox smoke.

    A new process group belongs to this test and is always reaped, including
    when an unfixed axis-split loop makes the browser unresponsive.
    """
    browser = browser_executable()
    if not browser:
        pytest.skip('requires SILK_FIREFOX or SILK_CHROME')
    assert Path(browser).is_file(), browser
    assets = ROOT / 'silk/signals/view/assets'
    for name in ('selectors.mjs', 'explorer.mjs', 'waveform-core.mjs', 'view.css', 'uplot.min.css', 'uplot.iife.min.js'):
        shutil.copyfile(assets / name, tmp_path / name)
    shutil.copyfile(ROOT / 'tests/browser' / module, tmp_path / module)
    (tmp_path / 'index.html').write_text('<!doctype html><link rel="stylesheet" href="view.css">'
        '<link rel="stylesheet" href="uplot.min.css"><script src="uplot.iife.min.js"></script>'
        f'<script type="module" src="{module}"></script><body></body>')
    results, complete = [], threading.Event()
    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            results.append(self.rfile.read(int(self.headers['Content-Length'])).decode())
            self.send_response(200); self.end_headers(); complete.set()
    server = ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Handler, directory=str(tmp_path)))
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    profile = tmp_path / 'profile'; profile.mkdir()
    firefox = 'firefox' in Path(browser).name.lower()
    args = ['--headless', '--no-remote', '--profile', str(profile)] if firefox else [
        '--headless', '--no-sandbox', '--disable-gpu', f'--user-data-dir={profile}']
    env = dict(os.environ)
    if firefox:
        env['MOZ_DISABLE_CONTENT_SANDBOX'] = '1'
    try:
        with (tmp_path / 'browser.log').open('w') as log:
            proc = subprocess.Popen([browser, *args, f'http://127.0.0.1:{server.server_port}/index.html'],
                                    stdout=log, stderr=log, env=env, start_new_session=True)
            try:
                complete.wait(35)
            finally:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL); proc.wait()
    finally:
        server.shutdown(); thread.join(); server.server_close()
    assert results == ['PASS'], (results, (tmp_path / 'browser.log').read_text())


def test_scalar_mount_reload_actions_and_restoration(tmp_path):
    cases = {'op': op_raw(), 'ac-op': ac_raw() + op_raw(),
        'dc': raw_plot('DC transfer characteristic', [('sweep','voltage'), ('v(out)','voltage'), ('v(sweep)','voltage')], [[0,30,4]]),
        'dc-wrapped': raw_plot('DC transfer characteristic', [('v(v-sweep)','voltage'), ('v(out)','voltage'), ('v(sweep)','voltage')], [[0,30,4]]),
        'tran': raw_plot('Transient Analysis', [('time','time'), ('v(out)','voltage'), ('v(time)','voltage')], [[1,40,5]]),
        'equal': raw_plot('DC transfer characteristic', [('sweep','voltage'), ('v(out)','voltage')], [[1,2],[1,3]]),
        'loop': raw_plot('Transient Analysis', [('time','time'), ('v(out)','voltage')], [[0,2],[1,3],[0,4]]),
        'tiny': raw_plot('Transient Analysis', [('time','time'), ('v(out)','voltage')], [[1,2],[1.0000000000000002,3]]),
        'many': raw_plot('DC transfer characteristic', [('sweep','voltage'), ('v(out)','voltage')], [[1,i] for i in range(205)]),
        'ac-one': raw_plot('AC Analysis', [('frequency','frequency'), ('v(out)','voltage')], [[10,2]], complex=True)}
    payloads = {}
    for name, content in cases.items():
        raw = tmp_path / (name+'.raw'); raw.write_text(content)
        payloads[name] = emit_view_data(raw)
    (tmp_path / 'cases.json').write_text(json.dumps(payloads))
    _run_browser(tmp_path, 'explorer_op_contract.mjs')


def test_op_layout_persists_through_saved_state(tmp_path):
    _run_browser(tmp_path, 'explorer_op_state.mjs')


def test_op_view_is_dot_plot_strips_in_ordinary_panels(tmp_path):
    """TASK-591: panels, strips, axes, picker, sort, hover, colours, persistence, plot switching."""
    cases = {
        'op': raw_plot('Operating Point', [('v(vdd)', 'voltage'), ('v(mid)', 'voltage'),
                                           ('v(gnd2)', 'voltage'), ('v(neg)', 'voltage'),
                                           ('i(v1)', 'current'), ('i(v2)', 'current')],
                       [[1.8, 0.9, 0, -0.4, -3e-5, 0]]),
        'ac-op': ac_raw() + op_raw(),
        'prec': raw_plot('Operating Point', [('v(p)', 'voltage'), ('v(tiny)', 'voltage')], [[1.234567890123, 1.234567890123e-40]]),
        'many': raw_plot('Operating Point', [(f'v(n{i})', 'voltage') for i in range(60)],
                         [[i / 40 for i in range(60)]]),
    }
    payloads = {}
    for name, content in cases.items():
        raw = tmp_path / (name + '.raw'); raw.write_text(content)
        payloads[name] = emit_view_data(raw)
    (tmp_path / 'cases.json').write_text(json.dumps(payloads))
    _run_browser(tmp_path, 'explorer_op_view.mjs')


def test_op_markers_are_reference_levels_and_export(tmp_path):
    """TASK-591: reference markers (add, drag, edit, delete, restore), the table, CSV and PNG."""
    cases = {
        'op': raw_plot('Operating Point', [('v(vdd)', 'voltage'), ('v(mid)', 'voltage'),
                                           ('v(gnd2)', 'voltage'), ('v(neg)', 'voltage'),
                                           ('i(v1)', 'current'), ('i(v2)', 'current')],
                       [[1.8, 0.9, 0, -0.4, -3e-5, 0]]),
        'ac-op': ac_raw() + op_raw(),
        # exactly representable in float32, so the CSV can be pinned byte for byte
        'csv': raw_plot('Operating Point', [('v(vdd)', 'voltage'), ('v(vout)', 'voltage'),
                                            ('i(ibias)', 'current'), ('i(idd)', 'current')],
                        [[1.5, 1.25, 0.125, -0.25]]),
    }
    payloads = {}
    for name, content in cases.items():
        raw = tmp_path / (name + '.raw'); raw.write_text(content)
        payloads[name] = emit_view_data(raw)
    (tmp_path / 'cases.json').write_text(json.dumps(payloads))
    _run_browser(tmp_path, 'explorer_op_markers.mjs')
