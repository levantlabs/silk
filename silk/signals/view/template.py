"""silk.signals.view.template — asset loading and sentinel substitution.

Builds the final ``view.html`` text from ``assets/view.html.tmpl`` plus the
vendored/first-party CSS and JS, via plain ``str.replace`` on unique sentinel
comments (spec §7.1). Not ``str.format``/``string.Template``: the JS payload
contains ``{``, ``}`` and ``${...}`` template literals that collide with both.
"""
import html as _html
import base64
import importlib.resources

from silk.signals.view.serialize import _escape_script_close, to_json

__all__ = [
    'ASSET_NAMES', 'asset_path', 'load_asset_text', 'load_asset_bytes',
    'render_view_html', 'ViewError', 'SelectionError', 'SelectorNotFoundError',
    'AmbiguousSelectorError', 'EmptySelectionError',
]


class ViewError(Exception):
    """Raised by the waveform-explorer package for a caller-facing failure.

    Deliberately **not** a subclass of silk's usual ``CircuitError`` base —
    the import boundary that keeps this package extractable to a standalone
    ``rawview`` forbids importing silk's exception hierarchy at all (spec
    §3.2). It still obeys silk's fail-loud convention: a typed error with an
    actionable message, never a silent fallback.

    Defined here (not in ``silk.signals.view.emit``) so
    ``render_view_html`` can raise it too without an emit->template->emit
    import cycle; ``silk.signals.view.emit`` re-exports the same class
    object, so ``from silk.signals.view.emit import ViewError`` still works.
    """


class SelectionError(ViewError, ValueError):
    """Base for the waveform selection failures of ``signals=`` / ``exclude=``.

    A subclass of :class:`ViewError`, so ``except ViewError`` keeps catching
    every selection failure, and of ``ValueError``, the convention silk uses
    for a bad caller argument (as ``UnknownCornerError`` does). Catch the
    subclass to tell the three cases apart without parsing the message.
    """


class SelectorNotFoundError(SelectionError):
    """A ``signals=`` or ``exclude=`` selector matches no signal in the rawfile.

    ``selector`` is the failing selector string and ``kwarg`` is the keyword
    it was passed to, ``'signals'`` or ``'exclude'``.
    """

    def __init__(self, message, *, selector, kwarg):
        super().__init__(message)
        self.selector = selector
        self.kwarg = kwarg


class AmbiguousSelectorError(SelectionError):
    """A designer-path selector names two different physical nodes.

    ``selector`` is the failing selector string and ``kwarg`` is the keyword
    it was passed to, ``'signals'`` or ``'exclude'``.
    """

    def __init__(self, message, *, selector, kwarg):
        super().__init__(message)
        self.selector = selector
        self.kwarg = kwarg


class EmptySelectionError(SelectionError):
    """The final selection, after ``signals=``, ``exclude=`` and ``max_depth=``, is empty."""


# Runtime assets; tests iterate this list to prove every asset is packaged.
ASSET_NAMES = (
    'uplot.iife.min.js',
    'uplot.min.css',
    'LICENSE.uPlot',
    'view.css',
    'view.js',
    'explorer.mjs',
    'waveform-core.mjs',
    'selectors.mjs',
    'view.html.tmpl',
)

_PACKAGE = 'silk.signals.view'


def asset_path(name):
    """``importlib.resources`` handle for ``assets/<name>``.

    ``importlib.resources.files`` (not ``__file__`` arithmetic) is required so
    asset loading works identically from a source checkout and from an
    installed wheel — see AGENTS.md's package-data gotcha.
    """
    return importlib.resources.files(_PACKAGE) / 'assets' / name


def load_asset_text(name):
    """Read an asset as UTF-8 text."""
    return asset_path(name).read_text(encoding='utf-8')


def load_asset_bytes(name):
    """Read an asset as raw bytes."""
    return asset_path(name).read_bytes()


_DEFAULT_SCOPES = {'current': None, 'scopes': []}


def render_view_html(*, title, data_script_html, poll_ms, inline,
                      view_state=None, commands=None, scopes=None,
                      data_src=None, stamp_src=None, group_signals_by_kind=True):
    """Render the complete ``view.html`` text.

    Parameters
    ----------
    title : str
        Page title; HTML-escaped before substitution.
    data_script_html : str
        The markup that replaces ``<!--__SILK_DATA_SCRIPT__-->`` — either
        ``<script src="...data.js"></script>`` (sidecar mode) or an inlined
        ``<script>...</script>`` carrying the payload (``inline=True``).
        Built by the caller (:mod:`silk.signals.view.emit`), which knows the
        mode and owns the payload.
    poll_ms : int
        Baked into ``window.SILK_VIEW_CONFIG`` for the reload loop.
    inline : bool
        Baked into ``window.SILK_VIEW_CONFIG``; the app disables auto-reload
        and says so when true.
    view_state : dict, optional
        The already-validated ``view`` block (v1.3 spec §3.2/§3.6), or
        ``None``. Embedded as ``window.SILK_VIEW_STATE``, ``null`` when
        absent. The caller (:mod:`silk.signals.view.emit`) owns validation;
        this function only serializes.
    commands : list, optional
        Already-validated command list (v1.3 spec §4.5), or ``None``.
        Embedded as ``window.SILK_VIEW_COMMANDS``, ``[]`` when absent.
    scopes : dict, optional
        The already-validated/normalized scope manifest (design spec §3.2),
        shape ``{'current': str|None, 'scopes': [...]}``, or ``None``.
        Embedded as ``window.SILK_VIEW_SCOPES``, always present — defaults
        to ``{"current": null, "scopes": []}`` when *scopes* is ``None`` so
        the frontend never branches on ``typeof undefined``.
    data_src, stamp_src : str, optional
        The sidecar filenames (derived from the HTML stem by the caller, so
        multiple pages can share one output directory without clobbering
        each other's data). Baked into ``window.SILK_VIEW_CONFIG`` as
        ``data_src``/``stamp_src``; the Reloader falls back to the legacy
        ``'data.js'``/``'stamp.js'`` when omitted (``None``, e.g.
        ``inline=True``, where there is no sidecar at all).
    group_signals_by_kind : bool, default True
        Baked into ``window.SILK_VIEW_CONFIG`` as ``group_by_kind``; ``view.js``
        forwards it as the explorer's ``groupSignalsByKind`` option.
        ``False`` restores the single, unsectioned signal list.

    Raises
    ------
    ViewError
        If any sentinel this function substitutes is missing from
        ``view.html.tmpl`` (v1.3 spec §5, item 7) — a removed sentinel used
        to silently produce a page with a missing global.
    """
    doc = load_asset_text('view.html.tmpl')
    config = {'poll_ms': poll_ms, 'inline': inline, 'group_by_kind': group_signals_by_kind}
    if data_src is not None:
        config['data_src'] = data_src
    if stamp_src is not None:
        config['stamp_src'] = stamp_src
    config_js = (
        'window.SILK_VIEW_CONFIG = '
        + _escape_script_close(to_json(config)) + ';\n'
        'window.SILK_VIEW_STATE = '
        + _escape_script_close(to_json(view_state)) + ';\n'
        'window.SILK_VIEW_COMMANDS = '
        + _escape_script_close(to_json(commands if commands is not None else [])) + ';\n'
        'window.SILK_VIEW_SCOPES = '
        + _escape_script_close(to_json(scopes if scopes is not None else _DEFAULT_SCOPES)) + ';'
    )
    def module_url(text):
        return 'data:text/javascript;base64,' + base64.b64encode(text.encode('utf-8')).decode('ascii')

    core_url = module_url(load_asset_text('waveform-core.mjs'))
    selector_source = load_asset_text('selectors.mjs').replace(
        "'./waveform-core.mjs'", repr(core_url))
    selector_url = module_url(selector_source)
    explorer_source = load_asset_text('explorer.mjs').replace(
        "'./selectors.mjs'", repr(selector_url))
    substitutions = (
        ('<!--__SILK_TITLE__-->', _html.escape(title)),
        ('/*__SILK_VIEW_CSS__*/', load_asset_text('view.css')),
        ('/*__SILK_UPLOT_JS__*/', load_asset_text('uplot.iife.min.js')),
        ('/*__SILK_VIEW_JS__*/', load_asset_text('view.js')),
        ('/*__SILK_EXPLORER_JS__*/', _escape_script_close(explorer_source)),
        ('/*__SILK_WAVEFORM_CORE__*/', "import * as waveformCore from " + repr(core_url) + ';'),
        ('<!--__SILK_DATA_SCRIPT__-->', data_script_html),
        ('/*__SILK_CONFIG__*/', config_js),
    )
    for sentinel, value in substitutions:
        if sentinel not in doc:
            raise ViewError(
                f'view.html.tmpl is missing the required sentinel {sentinel!r} '
                f'— render_view_html cannot substitute it'
            )
        doc = doc.replace(sentinel, value)
    return doc
