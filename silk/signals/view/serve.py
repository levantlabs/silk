"""silk.signals.view.serve — ``--serve``: stdlib HTTP server + mtime watch.

Sidesteps the ``file://`` reload-mechanism uncertainty (spec §7.2) entirely:
over ``http://`` with ``Cache-Control: no-store`` the poll loop is plain and
reliable, which is the whole point of this mode.
"""
import functools
import http.server
import os
import threading
import time
import webbrowser

from silk.signals.view.emit import emit_view

__all__ = ['run_server']


class _NoStoreHandler(http.server.SimpleHTTPRequestHandler):
    """Serves *directory* with caching disabled and quiet 200 logging."""

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def log_request(self, code='-', size='-'):
        # "silencing the per-request log line to one line per non-200"
        if str(code) != '200':
            super().log_request(code, size)


def _stat(path):
    try:
        st = os.stat(path)
        return (st.st_mtime, st.st_size)
    except OSError:
        return None


def _watch_and_reemit(rawfile_path, out_path, emit_kwargs, stop_event, interval):
    """Daemon-thread body: re-emit *out_path* whenever *rawfile_path*'s mtime
    or size changes and then holds steady for two consecutive stats — so a
    simulator still writing the file is never read mid-write."""
    last_stable = _stat(rawfile_path)
    pending = None
    while not stop_event.is_set():
        stop_event.wait(interval)
        if stop_event.is_set():
            return
        current = _stat(rawfile_path)
        if current is None or current == last_stable:
            pending = None
            continue
        if pending == current:
            try:
                emit_view(rawfile_path, out_path, **emit_kwargs)
            except Exception as exc:
                print(f'silk.signals.view --serve: re-emit failed: {exc}')
            last_stable = current
            pending = None
        else:
            pending = current


def run_server(out_path, rawfile_path, *, host='127.0.0.1', port=0,
               stat_interval=0.5, emit_kwargs=None, open_browser=False):
    """Serve the directory containing *out_path* over HTTP and watch
    *rawfile_path*, re-emitting the payload whenever it settles on a change.

    Blocks until interrupted (``Ctrl-C``). Binds ``127.0.0.1`` only — this
    serves a directory of local files, never ``0.0.0.0``.
    """
    emit_kwargs = dict(emit_kwargs or {})
    out_dir = os.path.dirname(os.path.abspath(out_path)) or '.'
    basename = os.path.basename(out_path)

    handler_cls = functools.partial(_NoStoreHandler, directory=out_dir)
    httpd = http.server.ThreadingHTTPServer((host, port), handler_cls)
    actual_port = httpd.server_address[1]

    stop_event = threading.Event()
    watcher = threading.Thread(
        target=_watch_and_reemit,
        args=(rawfile_path, out_path, emit_kwargs, stop_event, stat_interval),
        daemon=True,
    )
    watcher.start()

    url = f'http://{host}:{actual_port}/{basename}'
    print(f'serving {url} — watching {rawfile_path} (Ctrl-C to stop)')
    if open_browser:
        webbrowser.open(url)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        httpd.shutdown()
        httpd.server_close()
