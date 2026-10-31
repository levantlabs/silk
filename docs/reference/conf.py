"""Sphinx configuration for the generated API reference.

Build (from the repo root, in an environment with ``pip install -e ".[docs]"``)::

    python -m sphinx -W -n -b html docs/reference OUT
    python scripts/api_reference.py OUT

``-W`` turns warnings into errors and ``-n`` is nitpicky mode, so a broken
cross-reference fails the build. There is no ``nitpick_ignore`` and no
``suppress_warnings``, and silk is imported for real (no ``autodoc_mock_imports``).

External dependency: ``sphinx.ext.intersphinx`` downloads the Python, numpy and
matplotlib ``objects.inv`` inventories at build time (30 s timeout each), so a
build needs network access; they are not vendored. If a host is unreachable the
build fails (``-W``) rather than silently dropping those links.
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
os.environ.setdefault('MPLBACKEND', 'Agg')

from scripts import api_reference  # noqa: E402

project = 'circuitsilk'
author = 'silk contributors'
copyright = 'silk contributors, MIT licence'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'myst_parser',
]
source_suffix = {'.md': 'markdown', '.rst': 'restructuredtext'}
exclude_patterns = ['_build']
root_doc = 'index'

myst_enable_extensions = ['colon_fence']

# numpy-style docstrings (owner ruling, TASK-305).
napoleon_numpy_docstring = True
# Transitional: 32 docstrings are still Google style (the audit's convention
# column); the numpy ruling converts them in the per-package docstring tasks,
# after which this can go back to False.
napoleon_google_docstring = True
napoleon_preprocess_types = True
# napoleon would otherwise merge napoleon_type_aliases into autodoc_type_aliases
# while looking up Attributes annotations, rewriting every annotation.
napoleon_attr_annotations = False
# Short type names the docstrings write, mapped to what they mean.
_aliases = api_reference.short_type_names()
napoleon_type_aliases = {
    **_aliases,
    'Signal': 'silk.signals.Signal',
    'Module': 'silk.circuit.Module',
    'Variable': 'silk.circuit.Variable',
    'Spec': 'silk.specs.Spec',
    'array-like': ':term:`array_like`',
    'keyword-only': '*keyword-only*',
    'sequence': ':term:`sequence`',
    'iterable': ':term:`iterable`',
    'callable': ':term:`callable`',
    'np.ndarray': 'numpy.ndarray',
    'datetime': 'datetime.datetime',
}

autodoc_member_order = 'bysource'
# A constant's value is documented by its #: comment, never by the docstring
# of its type (a dict instance would otherwise show dict's own help text).
autodoc_inherit_docstrings = False
autodoc_typehints = 'signature'

# Metric signatures are annotated with ``Annotated[Signal, <kind>]`` aliases.
# Keep the alias name instead of expanding it to the kind's repr; the aliases
# are documented (silk.signals.schema) so the name resolves.
from silk.signals import schema as _schema  # noqa: E402

autodoc_type_aliases = {
    name: f'silk.signals.schema.{name}'
    for name, obj in vars(_schema).items()
    if getattr(obj, '__metadata__', None) and name[:1].isupper()
    and obj.__origin__.__name__ == 'Signal'
}
autodoc_type_aliases['datetime'] = 'datetime.datetime'

# Bounded so an unreachable inventory host fails the build in seconds, not
# minutes.
intersphinx_timeout = 30

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}

html_theme = 'alabaster'


def _generate(app, config):
    api_reference.write_pages(Path(app.srcdir) / 'api')


def setup(app):
    app.connect('config-inited', _generate)
