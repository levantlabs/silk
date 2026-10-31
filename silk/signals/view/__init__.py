"""silk.signals.view — the waveform explorer.

Turns a nutmeg rawfile into a self-contained, browser-openable HTML page
(uPlot-backed) with **all** signals from the file and no pre-selection. See
:mod:`silk.signals.view.emit` for ``emit_view``'s docstring.

Import boundary: every module in this package may import the
standard library, numpy, :mod:`silk.signals.rawfile`, the shared stdlib-only
:mod:`silk._json_codec`, and sibling view modules. A standalone
``rawview`` needs only the viewer, rawfile reader and JSON codec.
"""
from silk.signals.view.emit import (
    DATA_FORMAT_VERSION, ViewError, ViewOutput, emit_view, emit_view_data,
    pick_default_plot,
)
from silk.signals.view.select import emit_views_by_block, list_blocks
from silk.signals.view.template import (
    AmbiguousSelectorError, EmptySelectionError, SelectionError, SelectorNotFoundError,
)

__all__ = [
    'emit_view', 'emit_view_data', 'pick_default_plot', 'ViewOutput', 'ViewError',
    'DATA_FORMAT_VERSION', 'list_blocks', 'emit_views_by_block',
    'SelectionError', 'SelectorNotFoundError', 'AmbiguousSelectorError', 'EmptySelectionError',
]

from silk.signals.view.node_context import ViewNodeContext
__all__.append("ViewNodeContext")
