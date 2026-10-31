"""Generate and check the Sphinx API reference from the surface manifest.

TASK-305.1 (A1). ``scripts/api_surface.yaml`` is the one list of what silk's
public API is; ``scripts/docstring_audit.py`` scores it and this module
renders it. ``docs/reference/conf.py`` calls :func:`write_pages` before each
build, so a manifest row cannot be forgotten by hand-written pages.

    python -m sphinx -W -n -b html docs/reference OUT
    python scripts/api_reference.py OUT       # every row is in the build

The check reads ``OUT/objects.inv``, the inventory Sphinx writes for the
objects it documented.

Rows that are not rendered, by design:

* ``factory`` rows: the device and element classes generated at run time from
  PDK catalogs. ``docs/reference/factories.md`` documents the mechanism
  (``ElementLibrary``, ``load_pdk``) and points at the catalogs.
* ``alias`` rows: a second name for an object that is already a row. They are
  listed on the page of the package that exports the alias.

Instance attributes assigned in ``__init__`` are rows like any other: each is
declared with ``py:attribute`` under its class (text from the audit's member
documentation) unless the class's ``Attributes`` section already lists it.
"""
import collections
import inspect
import re
import sys
import textwrap
import typing
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import docstring_audit as audit  # noqa: E402

NOT_RENDERED = {
    'factory': 'catalog-generated class; see docs/reference/factories.md',
    'alias': 'second name of a row that is already rendered',
}
_MEMBER_DIRECTIVE = {'method': 'automethod', 'property': 'autoproperty',
                     'field': 'autoattribute'}
_PAGE_ORDER_TITLE = 'API reference'


def _slug(package):
    return package.replace('.', '_')


def _qualname(entry):
    return entry.name


def build_rows(manifest=None):
    """Return ``(rendered, skipped)`` Entry lists for the manifest."""
    entries = audit.collect(manifest or audit.load_manifest())
    bad = [e.name for e in entries if e.kind == 'unresolved']
    if bad:
        raise ValueError(f'manifest rows that do not resolve: {bad}')
    rendered = [e for e in entries if e.kind not in NOT_RENDERED]
    skipped = [e for e in entries if e.kind in NOT_RENDERED]
    return rendered, skipped


def expected_objects(manifest=None):
    """Names the build must contain, as ``(name, entry kind)`` pairs.

    A member is expected under its class's rendered name, so a base-class
    member appears under the class that carries it in the manifest.
    """
    rendered, _ = build_rows(manifest)
    classes = {id(e.obj): e.name for e in rendered
               if e.kind in ('class', 'exception', 'enum')}
    out = []
    for e in rendered:
        if e.cls is None:
            out.append((e.name, e.kind))
        else:
            out.append((f'{classes[id(e.cls)]}.{e.attr}', e.kind))
    return out


def row_counts(manifest=None):
    """Rendered and excluded row counts, the exclusions by kind."""
    rendered, skipped = build_rows(manifest)
    excluded = collections.Counter(e.kind for e in skipped)
    return {'rendered': len(rendered), 'excluded': dict(excluded)}


def short_type_names(manifest=None):
    """Map each unambiguous short class name to its documented dotted name.

    Docstrings write ``Signal`` or ``RunProvenance`` where the cross-reference
    needs the full path; this is the napoleon alias table that bridges them.
    """
    rendered, _ = build_rows(manifest)
    found = {}
    for e in rendered:
        if e.kind in ('class', 'exception', 'enum') and e.cls is None:
            found.setdefault(e.name.rsplit('.', 1)[-1], set()).add(e.name)
    return {short: next(iter(full)) for short, full in found.items()
            if len(full) == 1}


def _page(package, rendered, skipped):
    classes = {id(e.obj): e for e in rendered
               if e.kind in ('class', 'exception', 'enum')}
    members = {}
    for e in rendered:
        if e.cls is not None:
            members.setdefault(id(e.cls), []).append(e)
    lines = [f'# `{package}`', '']
    for e in rendered:
        if e.group != package or e.cls is not None:
            continue
        q = _qualname(e)
        mod, _, name = q.rpartition(".")
        if e.kind == 'module':
            lines += [f'```{{eval-rst}}\n.. automodule:: {q}\n   :no-members:\n```', '']
        elif e.kind == 'function':
            lines += [f'```{{eval-rst}}\n.. autofunction:: {q}\n```', '']
        elif e.kind == 'constant' and (hasattr(e.obj, '__metadata__')
                                       or typing.get_origin(e.obj) is typing.Literal):
            # An ``Annotated[Signal, kind]`` or ``Literal[...]`` alias. Signatures cross-reference
            # it as a class, and Sphinx resolves a class reference only to
            # class-like objects, so it is declared as one.
            doc, _ = audit._constant_doc(e)
            text = textwrap.indent(re.sub(r'^#: ?', '', doc or '', flags=re.M), '   ')
            lines += [f'```{{eval-rst}}\n.. py:currentmodule:: {mod}\n\n.. py:class:: {name}\n\n{text}\n```', '']
        elif e.kind == 'constant':
            # Documented by its ``#:`` comment, never by the docstring of its
            # type (a dict or namespace instance would show its type's help).
            doc, _ = audit._constant_doc(e)
            text = textwrap.indent(re.sub(r'^#: ?', '', doc or '', flags=re.M), '   ')
            lines += [f'```{{eval-rst}}\n.. py:currentmodule:: {mod}\n\n.. py:data:: {name}\n\n{text}\n```', '']
        else:
            body = [f'.. autoclass:: {q}']
            # A name in the class docstring's Attributes section is rendered
            # by napoleon; a second directive would register it twice.
            listed = set(audit._section_names(inspect.getdoc(e.obj) or '',
                                              ['Attributes']))
            for m in sorted(members.get(id(e.obj), ()), key=lambda m: m.attr):
                if m.attr in listed:
                    continue
                if m.kind == 'attribute':
                    doc, _ = audit._member_doc(m)
                    text = re.sub(r'(?m)^#: ?', '', doc or '')
                    body += ['', f'   .. py:attribute:: {m.attr}', '']
                    body += [f'      {ln}' if ln else '' for ln in text.splitlines()]
                    continue
                body += ['', f'   .. {_MEMBER_DIRECTIVE[m.kind]}:: {m.attr}']
            lines += ['```{eval-rst}', *body, '```', '']
    aliases = [e for e in skipped if e.kind == 'alias' and e.group == package]
    if aliases:
        lines += ['## Aliases', '']
        lines += [f'- `{e.name}`: {e.note}' for e in aliases] + ['']
    return '\n'.join(lines)


def write_pages(outdir, manifest=None):
    """Write one Markdown page per manifest package plus the generated index."""
    manifest = manifest or audit.load_manifest()
    rendered, skipped = build_rows(manifest)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pages = {}
    for package in manifest['packages']:
        if any(e.group == package for e in rendered):
            pages[f'{_slug(package)}.md'] = _page(package, rendered, skipped)
    names = [name[:-3] for name in pages]
    index = ['# Packages', '', '```{toctree}', ':maxdepth: 1', '',
             *names, '```', '']
    pages['index.md'] = '\n'.join(index)
    # Rewrite only what changed, so concurrent builds never see a missing page.
    for old in outdir.glob('*.md'):
        if old.name not in pages:
            old.unlink()
    for name, text in pages.items():
        target = outdir / name
        if not target.exists() or target.read_text(encoding='utf-8') != text:
            target.write_text(text, encoding='utf-8')
    return names


def read_inventory(path):
    """Return the set of object names in a Sphinx ``objects.inv`` (v2)."""
    raw = Path(path).read_bytes()
    parts = raw.split(b'\n', 4)
    if parts[0] != b'# Sphinx inventory version 2':
        raise ValueError(f'{path}: not a Sphinx v2 inventory')
    names = set()
    for line in zlib.decompress(parts[4]).decode('utf-8').splitlines():
        m = re.match(r'(\S+)\s+py:', line)
        if m:
            names.add(m.group(1))
    return names


def missing_from_build(outdir, manifest=None):
    """Return the manifest rows absent from the build in *outdir*."""
    built = read_inventory(Path(outdir) / 'objects.inv')
    return sorted(n for n, _ in expected_objects(manifest) if n not in built)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__.split('\n\n')[0], file=sys.stderr)
        return 2
    missing = missing_from_build(argv[0])
    counts = row_counts()
    total = len(expected_objects())
    print(f"rendered rows: {counts['rendered']}; excluded rows: "
          + ', '.join(f'{k} {v}' for k, v in sorted(counts['excluded'].items())))
    if missing:
        print(f'{len(missing)} of {total} manifest rows are missing from the build:')
        print('\n'.join(f'  {n}' for n in missing))
        return 1
    print(f'all {total} manifest rows are in the build')
    return 0


if __name__ == '__main__':
    sys.exit(main())
