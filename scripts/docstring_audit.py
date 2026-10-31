"""Score the docstrings of silk's public surface (TASK-305, Phase 1).

Before an API reference is generated from docstrings, the docstrings have to
be good enough to generate from. This tool measures that, re-runnably, so the
answer is a table rather than an impression::

    python scripts/docstring_audit.py                  # static audit, markdown
    python scripts/docstring_audit.py --worst 50       # longer offender list
    python scripts/docstring_audit.py --json out.json  # every row, for diffing
    python scripts/docstring_audit.py --rows --group silk.metrics
    python scripts/docstring_audit.py --run-examples   # also execute >>> examples

**The surface** is read from ``scripts/api_surface.yaml``, the manifest the
generated reference will use too: packages (their ``__all__``, recursing into
exported silk modules), opt-in imports, the members, data fields and listed
instance attributes of every silk class on it, run-time factories, and
exclusions. Its header states the collection rules.

**What is not audited:** ordinary ``#`` comments, private helpers and their
docstrings, and anything outside the manifest.

**The checks are triage, not verification.** Each is ``True`` (pass),
``False`` (fail), ``None`` (not applicable) or ``'unknown'`` (the audit cannot
decide, for example a generated function with no source). A pass means the
docstring *mentions* the thing, not that the statement is right; whether a
docstring matches the implementation (shapes, units, warnings, mutation,
side effects) needs a reading against the code and tests.

* ``present`` — the symbol has its own docstring. Auto-generated text (a
  dataclass's ``Name(field, ...)``, Enum's default) or a docstring inherited
  from a base class counts as missing. Data fields, listed attributes and
  constants pass with a ``#:`` comment or a string right after the
  assignment (what Sphinx autodoc reads) or an entry in the class's
  ``Attributes``/``Parameters`` section.
* ``params`` — every named parameter of the signature (not ``self``/``cls``,
  ``*args``/``**kwargs``) appears as a word in the docstring. Classes are
  scored on ``inspect.signature(cls)``.
* ``returns`` — functions and methods that return a value (a non-``None``
  annotation, or, unannotated, a ``return <value>`` other than ``return None``
  in their own body, nested defs excluded): the docstring says
  ``return``/``yield``. ``returns_section`` counts a structured ``Returns``
  section. Properties are ``unknown``: their docstring describes the value as
  an attribute, which no keyword test can judge.
* ``units`` — ``silk.metrics`` callables, and callables, properties and
  fields with a physical-looking name (``PHYSICAL``): a unit token appears.
* ``failure`` — every ``silk.metrics`` function (decision-1), and any
  callable or property whose own body contains ``raise``: the docstring
  mentions a failure outcome (``Raises``, an exception name, ``nan``, a floor,
  "cannot"). A mention does not prove decision-1 or ``MetricWarning``
  compliance.
* ``example`` — top-level functions and classes: an ``Examples`` section or
  an example block of any kind.
* ``stale`` — no documented parameter (numpy ``Parameters``/``Other
  Parameters``, Google ``Args``, Sphinx ``:param x:``/``:type x:``) that is
  not in the signature, zero-argument callables included; no import from or
  attribute use of the removed ``silk.signals.metrics`` import path.

**Cross-references** (``:func:``, ``:meth:``, ... roles) are counted but not
judged: resolving them is the Sphinx ``-n`` build's job. The report tallies
them and lists, as triage only, fully-qualified ``silk.`` targets that do not
import-resolve; relative targets are left to the build.

**Example blocks** are classified by language: ``>>>`` sessions (wherever
they appear, including ``pycon`` code-blocks and fences) are doctests, compiled
statement by statement; ``code-block`` directives and fenced blocks with a
Python language are Python programs; ``::`` literal
blocks carry no language and are reported as such. Python blocks go through
``compile()`` (so a top-level ``return`` or ``await`` fails) and their silk
imports are resolved. Nothing is executed by default.

**``--run-examples``** executes each docstring's ``>>>`` session (module
docstrings included) with doctest
in a fresh interpreter, in a temporary working directory, under a timeout,
and reports passed, failed (with the failure text), all-skipped and not-run
(timeout, nonzero worker exit, or no result; with the worker's stderr)
separately.

The static audit imports silk, the manifest's opt-in modules and builds its
factory objects (a catalog-only ``load_pdk``); it executes no docstring code
and writes nothing except ``--json``.
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import doctest
import enum
import functools
import importlib
import importlib.util
import inspect
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / 'scripts' / 'api_surface.yaml'
CHECKS = ('present', 'params', 'returns', 'returns_section', 'units',
          'failure', 'example', 'stale')
UNKNOWN = 'unknown'

PHYSICAL = re.compile(
    r'^(w|l|f|fs|fc|f[0-9]|fmin|fmax|freq\w*|\w+_freq|f_\w+|\w+_hz|'
    r't|t[0-9]|t_\w+|\w+_time|time\w*|tstep|step|stop|start|tstart|tstop|'
    r'period|delay|tau|temp\w*|vdd|vss|vref|vfs|vcm|vin|vout|v_\w+|\w+_v|'
    r'ibias|i_\w+|\w+_a|cap\w*|c_\w+|cload|r_\w+|rload|load\w*|'
    r'width|length|bw|bandwidth|\w+_db|\w+_um|threshold|level|swing|'
    r'tolerance|tol|rtol|atol|settle\w*|band|f_lo|f_hi)$')
UNIT = re.compile(
    r'(\b(Hz|kHz|MHz|GHz|dB|dBc|dBFS|dBm|dBV|V|mV|uV|A|mA|uA|nA|s|ms|us|ns|ps|'
    r'fs|F|pF|fF|nF|uF|ohms?|um|nm|mm|m|K|LSB|LSBs|ppm|rad|deg|degrees?|'
    r'seconds?|volts?|amps?|amperes?|farads?|hertz|radians?|metres?|meters?|'
    r'microns?|kelvin|celsius|percent|unitless|dimensionless|SI|bits?|'
    r'decades?|octaves?|V/s|V/V|A/V|W)\b)|µ|Ω|°|%|\bin units of\b|\bunits?\b')
FAILURE = re.compile(
    r'\b(Raises|raises?|raised|raising|\w+Error|\w+Exception|nan|NaN|inf|'
    r'infinite|floor|cannot|can ?not|unmeasurable|refuses?|fails?)\b')
RETURNS_WORD = re.compile(r'\b(returns?|returned|returning|yields?)\b', re.I)
ROLE = re.compile(r':(func|meth|class|attr|mod|data|exc|obj|const|py:\w+):`~?!?'
                  r'(?:[^`<]*<)?([^`>]+)>?`')

_NUMPY_HDR = re.compile(r'^\s*(Parameters|Returns|Yields|Raises|Examples?|'
                        r'Notes|See Also|Attributes|Warns|Other Parameters)'
                        r'\s*\n\s*-{3,}\s*$', re.M)
_GOOGLE_HDR = re.compile(r'^\s*(Args|Arguments|Parameters|Returns|Yields|'
                         r'Raises|Examples?|Attributes|Note|Notes):\s*$', re.M)
_SPHINX = re.compile(r':(param|returns?|raises?|rtype|type)\b[^:]*:')
_AUTO_DOC = re.compile(r'^\w+\(.*\)$', re.S)
_PY_LANGS = {'python', 'py', 'python3', 'default'}
# Console sessions are doctests: the whole-docstring doctest parse already
# collects their >>> lines, so they are not collected again as code.
_CONSOLE_LANGS = {'pycon', 'python-console', 'doctest'}


@dataclass
class Row:
    group: str
    name: str
    kind: str      # function class exception enum method property field
    #               attribute constant module factory alias unresolved
    source: str = '?'
    results: dict = field(default_factory=dict)
    convention: str = 'none'
    notes: list = field(default_factory=list)
    removed: list = field(default_factory=list)
    examples: list = field(default_factory=list)   # dicts: kind lang status
    xrefs: dict = field(default_factory=dict)
    target: tuple = ()                             # (module, qualname) for doctest

    @property
    def fails(self):
        return [c for c, v in self.results.items()
                if v is False and c != 'returns_section']


@dataclass
class Entry:
    group: str
    name: str
    kind: str
    obj: object = None
    owner: type = None       # class defining a member
    attr: str = None
    cls: type = None         # public class a member/field belongs to
    note: str = ''


# ---------------------------------------------------------------- collection

def load_manifest(path=MANIFEST):
    with open(path, encoding='utf-8') as fh:
        return yaml.safe_load(fh)


def _is_silk(obj):
    mod = getattr(obj, '__module__', '') or ''
    return mod == 'silk' or mod.startswith('silk.')


def _is_silk_module(mod):
    return mod.__name__ == 'silk' or mod.__name__.startswith('silk.')


def _resolve(path):
    """Import-resolve a dotted path; raise LookupError when it does not."""
    parts = path.split('.')
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module('.'.join(parts[:i]))
        except ImportError:
            continue
        for attr in parts[i:]:
            if not hasattr(obj, attr):
                raise LookupError(path)
            obj = getattr(obj, attr)
        return obj
    raise LookupError(path)


def _class_kind(obj):
    if issubclass(obj, BaseException):
        return 'exception'
    if issubclass(obj, enum.Enum):
        return 'enum'
    return 'class'


def _data_fields(cls):
    """Names of the declared data fields that *cls* itself defines."""
    own = set(cls.__dict__.get('__annotations__', {}))
    if dataclasses.is_dataclass(cls):
        return [f.name for f in dataclasses.fields(cls)
                if f.name in own and not f.name.startswith('_')]
    pyd = cls.__dict__.get('__pydantic_fields__')
    if isinstance(pyd, dict):
        return [n for n in pyd if n in own and not n.startswith('_')]
    if issubclass(cls, tuple) and '_fields' in cls.__dict__:
        return [n for n in cls._fields if not n.startswith('_')]
    return []


def collect(manifest):
    """Return the Entry list for *manifest* (see api_surface.yaml's header)."""
    for name in manifest.get('imports') or ():
        importlib.import_module(name)
    excluded = set(manifest.get('exclude') or {})
    member_groups = set(manifest.get('member_groups') or ())
    attributes = manifest.get('attributes') or {}
    entries, seen_ids, seen_consts, classes = [], {}, set(), []
    queue = list(manifest.get('packages') or ())
    done = set()

    while queue:
        modname = queue.pop(0)
        if modname in done:
            continue
        done.add(modname)
        mod = importlib.import_module(modname)
        names = getattr(mod, '__all__', None)
        if names is None:
            raise ValueError(f'{modname}: a manifest package must define __all__')
        if id(mod) not in seen_ids:                 # the package's own docstring
            seen_ids[id(mod)] = modname
            entries.append(Entry(modname, modname, 'module', mod))
        for name in names:
            qual = f'{modname}.{name}'
            if qual in excluded:
                continue
            if not hasattr(mod, name):
                entries.append(Entry(modname, qual, 'unresolved'))
                continue
            obj = getattr(mod, name)
            if inspect.ismodule(obj):
                if obj.__name__ in excluded or id(obj) in seen_ids:
                    continue
                seen_ids[id(obj)] = obj.__name__
                entries.append(Entry(modname, obj.__name__, 'module', obj))
                if _is_silk_module(obj) and hasattr(obj, '__all__'):
                    queue.append(obj.__name__)     # after the current package
                continue
            if not (inspect.isclass(obj) or inspect.isroutine(obj)):
                # Constants, type aliases (Literal[...]) and exported instances,
                # documented where they are assigned. Values such as 'u' may be
                # interned and shared, so these are keyed by name, not identity.
                key = (name, repr(obj))
                if key in seen_consts:
                    continue
                seen_consts.add(key)
                entries.append(Entry(modname, qual, 'constant', obj, attr=name))
                continue
            if id(obj) in seen_ids:
                first = seen_ids[id(obj)]
                if first.rsplit('.', 1)[-1] != name:
                    entries.append(Entry(modname, qual, 'alias', obj,
                                         note=f'alias of {first}'))
                continue
            seen_ids[id(obj)] = qual
            if inspect.isclass(obj):
                entries.append(Entry(modname, qual, _class_kind(obj), obj))
                if _is_silk(obj):
                    classes.append((modname, obj))
            else:
                entries.append(Entry(modname, qual, 'function', obj))

    # Symbols the hand-written reference documents that no package exports in
    # its __all__ (so adding a row does not change the public API).
    for dotted in manifest.get('include') or ():
        obj = _resolve(dotted)
        group = next((p for p in sorted(manifest.get('packages') or (), key=len,
                                        reverse=True)
                      if dotted.startswith(p + '.')), 'silk')
        if id(obj) in seen_ids:
            continue
        seen_ids[id(obj)] = dotted
        if inspect.ismodule(obj):
            entries.append(Entry(group, dotted, 'module', obj))
        elif inspect.isclass(obj):
            entries.append(Entry(group, dotted, _class_kind(obj), obj))
            if _is_silk(obj):
                classes.append((group, obj))
        elif inspect.isroutine(obj):
            entries.append(Entry(group, dotted, 'function', obj))
        else:
            entries.append(Entry(group, dotted, 'constant', obj,
                                 attr=dotted.rsplit('.', 1)[-1]))

    public = {id(c) for _, c in classes}
    seen_members = set()
    for group, cls in classes:
        if issubclass(cls, enum.Enum):
            continue
        mgroup = (f'{cls.__name__} members' if cls.__name__ in member_groups
                  else f'{group} (members)')
        for attr in sorted(dir(cls)):
            if attr.startswith('_'):
                continue
            owner = next((k for k in cls.__mro__ if attr in k.__dict__), None)
            if owner is None or not _is_silk(owner):
                continue
            raw = owner.__dict__[attr]
            if isinstance(raw, (property, functools.cached_property)):
                kind = 'property'
            elif isinstance(raw, (staticmethod, classmethod)) or inspect.isfunction(raw):
                kind = 'method'
            else:
                continue                 # class constants; data fields below
            key = (id(owner), attr)
            if key in seen_members:
                continue
            seen_members.add(key)
            # A member of a public base is named after the base; a mixin's
            # member after the public class that carries it.
            label = owner.__name__ if id(owner) in public else cls.__name__
            entries.append(Entry(mgroup, f'{label}.{attr}', kind,
                                 raw, owner, attr, cls))
        for name in _data_fields(cls):
            if (id(cls), name) not in seen_members:
                seen_members.add((id(cls), name))
                entries.append(Entry(mgroup, f'{cls.__name__}.{name}', 'field',
                                     None, cls, name, cls))
        for name in attributes.get(cls.__name__, ()):
            if (id(cls), name) not in seen_members:
                seen_members.add((id(cls), name))
                entries.append(Entry(mgroup, f'{cls.__name__}.{name}', 'attribute',
                                     None, cls, name, cls))

    for fac in manifest.get('factories') or ():
        if 'call' in fac:
            holder = _resolve(fac['call'])(*fac.get('args', ()), **fac.get('kwargs', {}))
        else:
            holder = _resolve(fac['object'])
        for name in sorted(dir(holder)):
            if name.startswith('_'):
                continue
            obj = getattr(holder, name, None)
            if inspect.isclass(obj):
                entries.append(Entry(fac['group'], f"{fac.get('label', '')}.{name}",
                                     'factory', obj,
                                     note=f'generated by {obj.__module__}'))
    return entries


# ---------------------------------------------------------------- source helpers

def _func_of(obj):
    if isinstance(obj, property):
        return obj.fget
    if isinstance(obj, functools.cached_property):
        return obj.func
    if isinstance(obj, (staticmethod, classmethod)):
        return obj.__func__
    return getattr(obj, '__func__', obj)


def _source_loc(obj):
    target = _func_of(obj) if obj is not None else None
    try:
        path = Path(inspect.getsourcefile(target)).resolve()
        line = inspect.getsourcelines(target)[1]
    except (TypeError, OSError):
        return '?'
    try:
        path = path.relative_to(REPO)
    except ValueError:
        pass
    return f'{path}:{line}'


def _ast_of(fn):
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    except (TypeError, OSError, SyntaxError):
        return None
    node = tree.body[0] if tree.body else None
    return node if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else None


_NESTED = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


def _own_nodes(fn_node):
    """Nodes in *fn_node*'s own body; nested defs, classes and lambdas excluded."""
    stack = [n for n in fn_node.body if not isinstance(n, _NESTED)]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(c for c in ast.iter_child_nodes(node)
                     if not isinstance(c, _NESTED))


def _raises(fn):
    """True / False, or UNKNOWN when the source is unavailable."""
    node = _ast_of(fn)
    if node is None:
        return UNKNOWN
    return any(isinstance(n, ast.Raise) for n in _own_nodes(node))


def _returns_value(fn):
    """True / False, or UNKNOWN when neither annotation nor source decides."""
    try:
        ann = inspect.signature(fn).return_annotation
    except (TypeError, ValueError):
        ann = inspect.Signature.empty
    if ann is None or ann == 'None':
        return False
    if ann is not inspect.Signature.empty:
        return True
    node = _ast_of(fn)
    if node is None:
        return UNKNOWN
    for n in _own_nodes(node):
        if isinstance(n, (ast.Yield, ast.YieldFrom)):
            return True
        if isinstance(n, ast.Return) and n.value is not None and not (
                isinstance(n.value, ast.Constant) and n.value.value is None):
            return True
    return False


def _signature_params(obj, kind, varargs=False):
    """Parameter names, or None when the signature cannot be read."""
    target = obj if kind in ('class', 'exception', 'factory') else _func_of(obj)
    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError):
        return None
    return [p.name for p in sig.parameters.values()
            if p.name not in ('self', 'cls')
            and (varargs or p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD))]


def _silk_constructor(obj, kind):
    """False for a class whose constructor is inherited from outside silk
    (an exception on BaseException's, a pydantic model's): no parameters of
    its own to document."""
    if kind not in ('class', 'exception', 'factory'):
        return True
    owner = next((k for k in obj.__mro__
                  if '__init__' in k.__dict__ or '__new__' in k.__dict__), None)
    return owner is not None and _is_silk(owner)


# ---------------------------------------------------------------- docstrings

def _own_doc(entry):
    obj, kind = entry.obj, entry.kind
    if kind in ('class', 'exception', 'enum', 'factory'):
        doc = obj.__dict__.get('__doc__')
        if not doc or doc.strip() in ('An enumeration.', ''):
            return ''
        if _AUTO_DOC.match(doc.strip()):
            return ''
        return inspect.cleandoc(doc)
    if kind == 'module':
        return inspect.cleandoc(obj.__doc__ or '')
    doc = getattr(_func_of(obj), '__doc__', None)
    return inspect.cleandoc(doc) if doc else ''


def _hash_colon(lines, lineno):
    """``#:`` comment above line *lineno* (1-based) or trailing on it."""
    line = lines[lineno - 1]
    if '#:' in line:
        return line.split('#:', 1)[1].strip()
    out, j = [], lineno - 2
    while j >= 0 and lines[j].strip().startswith('#:'):
        out.insert(0, lines[j].strip()[2:].strip())
        j -= 1
    return ' '.join(out)


def _attr_doc_in(body, name, lines):
    """Doc of assignment *name* in AST *body*: text, '' if undocumented,
    None if there is no such assignment."""
    for i, stmt in enumerate(body):
        if isinstance(stmt, ast.Assign):
            targets = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            targets = [stmt.target.id]
        else:
            continue
        if name not in targets:
            continue
        nxt = body[i + 1] if i + 1 < len(body) else None
        if (isinstance(nxt, ast.Expr) and isinstance(nxt.value, ast.Constant)
                and isinstance(nxt.value.value, str)):
            return nxt.value.value
        return _hash_colon(lines, stmt.lineno)
    return None


def _section_names(doc, headers):
    """Entry names in the numpy sections *headers* and Google equivalents."""
    names = []
    hdr = '|'.join(headers)
    entry = re.compile(r'^(\*{0,2}[A-Za-z_]\w*(?:\s*,\s*\*{0,2}[A-Za-z_]\w*)*)'
                       r'\s*(?:\s:.*)?$')
    for m in re.finditer(rf'^\s*({hdr})\s*\n\s*-{{3,}}\s*\n(.*?)'
                         r'(?=^\s*\w[\w ]*\n\s*-{3,}\s*$|\Z)', doc, re.M | re.S):
        for line in m.group(2).splitlines():
            mm = entry.match(line)
            if mm:
                names += [n.strip().lstrip('*') for n in mm.group(1).split(',')]
    google = {'Parameters': 'Args|Arguments', 'Other Parameters': 'Other Parameters',
              'Attributes': 'Attributes'}
    ghdr = '|'.join(google[h] for h in headers if h in google)
    for m in re.finditer(rf'^(\s*)({ghdr}):\s*\n(.*?)(?=^\1\S|\Z)', doc, re.M | re.S):
        for line in textwrap.dedent(m.group(3)).splitlines():
            mm = re.match(r'^(\*{0,2}[A-Za-z_]\w*)\s*(\(.*?\))?\s*:', line)
            if mm:
                names.append(mm.group(1).lstrip('*'))
    return names


def documented_params(doc):
    """Parameter names a docstring documents, in any of the three styles."""
    names = _section_names(doc, ('Parameters', 'Other Parameters'))
    names += re.findall(r':(?:param|type)\s+(?:[^:]*\s)?\*{0,2}([A-Za-z_]\w*)\s*:', doc)
    return names


def _member_doc(entry):
    """(text, how) for a field or listed attribute; how is UNKNOWN when the
    class source cannot be read."""
    cls, name = entry.owner, entry.attr
    try:
        src = textwrap.dedent(inspect.getsource(cls))
        cdef = ast.parse(src).body[0]
    except (TypeError, OSError, SyntaxError, IndexError):
        return None, UNKNOWN
    lines = src.splitlines()
    doc = _attr_doc_in(cdef.body, name, lines)
    if doc:
        return doc, 'attribute docstring'
    for node in ast.walk(cdef):                  # self.<name> = ... in methods
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            tgts = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Attribute) and t.attr == name
                   and isinstance(t.value, ast.Name) and t.value.id == 'self'
                   for t in tgts):
                text = _hash_colon(lines, node.lineno)
                if text:
                    return text, '#: comment'
    cdoc = inspect.cleandoc(cls.__dict__.get('__doc__') or '')
    if name in _section_names(cdoc, ('Attributes', 'Parameters')):
        m = re.search(rf'^{re.escape(name)}\b.*\n((?:[ \t]+.*\n?)*)', cdoc, re.M)
        return (m.group(0) if m else name), 'class docstring section'
    return '', ''


def _constant_doc(entry):
    """(doc, where) for a module-level constant: the ``#:`` comment or string
    after its assignment in a loaded silk module that binds the same object.
    doc is None when no assignment is found."""
    for modname, mod in sorted(sys.modules.items()):
        if mod is None or not (modname == 'silk' or modname.startswith('silk.')):
            continue
        if getattr(mod, entry.attr, object()) is not entry.obj:
            continue
        try:
            src = inspect.getsource(mod)
        except (TypeError, OSError):
            continue
        doc = _attr_doc_in(ast.parse(src).body, entry.attr, src.splitlines())
        if doc is not None:
            path = Path(mod.__file__).resolve()
            try:
                path = path.relative_to(REPO)
            except ValueError:
                pass
            return doc, str(path)
    return None, '?'


def _convention(doc):
    if _NUMPY_HDR.search(doc):
        return 'numpy'
    if _GOOGLE_HDR.search(doc):
        return 'google'
    if _SPHINX.search(doc):
        return 'sphinx'
    return 'prose'


# ---------------------------------------------------------------- examples

def _is_submodule(mod, name):
    """True when *name* is an importable submodule of package *mod*; a plain
    module (no ``__path__``) has none."""
    if not hasattr(mod, '__path__'):
        return False
    try:
        return importlib.util.find_spec(f'{mod.__name__}.{name}') is not None
    except (ImportError, ValueError):
        return False


def _import_problems(tree):
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or '').split('.')[0] == 'silk':
            try:
                mod = importlib.import_module(node.module)
            except ImportError:
                bad.append(node.module)
                continue
            bad += [f'{node.module}.{a.name}' for a in node.names
                    if a.name != '*' and not hasattr(mod, a.name)
                    and not _is_submodule(mod, a.name)]
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split('.')[0] == 'silk':
                    try:
                        importlib.import_module(a.name)
                    except ImportError:
                        bad.append(a.name)
    return bad


def _indented_block(lines, i):
    """The block indented under line *i*, dedented."""
    indent = len(lines[i]) - len(lines[i].lstrip())
    body, j = [], i + 1
    while j < len(lines) and not lines[j].strip():
        j += 1
    while j < len(lines) and (not lines[j].strip()
                              or len(lines[j]) - len(lines[j].lstrip()) > indent):
        body.append(lines[j])
        j += 1
    return textwrap.dedent('\n'.join(body)).strip()


def example_blocks(doc):
    """Classify and statically check every example block in *doc*.

    Returns dicts ``{kind, lang, status}``. kind: doctest, code-block, fence
    or literal. lang: 'python', another declared language, or 'none' (a
    ``::`` literal block declares none). status: for Python 'ok',
    'compile: <msg>' or 'import: <names>'; for another language
    'not checked'; for an undeclared literal block 'compiles as python',
    'import: <names>' or 'not python'.
    """
    blocks = []
    examples = doctest.DocTestParser().get_examples(doc)
    if examples:
        blocks.append(('doctest', 'python', [e.source for e in examples]))
    lines = doc.splitlines()
    for i, line in enumerate(lines):
        s = line.strip()
        m = re.match(r'\.\. (?:code-block|code|sourcecode)::\s*(\S*)', s)
        if m:
            lang = (m.group(1) or 'none').lower()
            body = '\n'.join(l for l in _indented_block(lines, i).splitlines()
                             if not re.match(r'\s*:\w[\w-]*:', l)).strip()
            if lang in _CONSOLE_LANGS or body.startswith('>>>'):
                continue
            blocks.append(('code-block', 'python' if lang in _PY_LANGS else lang, [body]))
        elif s.endswith('::') and not s.startswith('..') and '>>>' not in s:
            body = _indented_block(lines, i)
            if body and not body.startswith('>>>'):
                blocks.append(('literal', 'none', [body]))
    for m in re.finditer(r'^[ \t]*```[ \t]*(\w*)[^\n]*\n(.*?)^[ \t]*```', doc, re.M | re.S):
        lang = (m.group(1) or 'none').lower()
        body = textwrap.dedent(m.group(2))
        if lang in _CONSOLE_LANGS or body.lstrip().startswith('>>>'):
            continue
        blocks.append(('fence', 'python' if lang in _PY_LANGS else lang, [body]))
    out = []
    for kind, lang, sources in blocks:
        if lang not in ('python', 'none'):
            out.append({'kind': kind, 'lang': lang, 'status': 'not checked'})
            continue
        mode = 'single' if kind == 'doctest' else 'exec'
        err, trees = None, []
        for src in sources:
            try:
                compile(src, '<docstring>', mode, dont_inherit=True)
                trees.append(ast.parse(src))
            except SyntaxError as exc:
                err = f'{exc.msg} (line {exc.lineno})'
                break
        if err:
            status = 'not python' if lang == 'none' else f'compile: {err}'
        else:
            bad = [b for t in trees for b in _import_problems(t)]
            status = ('import: ' + ', '.join(bad) if bad
                      else 'compiles as python' if lang == 'none' else 'ok')
        out.append({'kind': kind, 'lang': lang, 'status': status})
    return out


def xrefs(doc):
    """Tally reST cross-reference roles; resolving them is the Sphinx build's job."""
    total, qualified, unresolved = 0, 0, []
    for m in ROLE.finditer(doc):
        total += 1
        target = re.sub(r'\s+', '', m.group(2)).lstrip('.~').rstrip('()')
        if target.startswith('silk.'):
            qualified += 1
            try:
                _resolve(target)
            except LookupError:
                unresolved.append(target)
    return {'roles': total, 'qualified': qualified,
            'unresolved': list(dict.fromkeys(unresolved))}


def removed_names(changelog=None):
    """Removed public names, using development history when available."""
    if changelog is None:
        development = REPO / 'CHANGELOG-dev.md'
        changelog = development if development.exists() else REPO / 'CHANGELOG.md'
    text = Path(changelog).read_text(encoding='utf-8')
    names = set()
    for para in re.split(r'\n(?=- )', text):
        if 'BREAKING' not in para:
            continue
        for m in re.finditer(r'(.{0,400}?)\b(?:is|are) removed\b', para, re.S):
            for tok in re.findall(r'`([A-Za-z_][\w.]*)(?:\([^`]*\))?`', m.group(1)):
                names.add(tok.split('.')[-1])
    import silk
    import silk.metrics
    live = set(dir(silk)) | set(dir(silk.metrics))
    for cls in (silk.Module, silk.ModuleTestbench, silk.Signal, silk.Spec):
        live |= set(dir(cls))
    # Short or underscore-free names would match ordinary prose.
    return sorted(n for n in names if n not in live and len(n) > 4 and '_' in n)


# ---------------------------------------------------------------- scoring

def score(entry, removed=()):
    row = Row(entry.group, entry.name, entry.kind)
    r = row.results = {c: None for c in CHECKS}
    if entry.note:
        row.notes.append(entry.note)
    if entry.kind == 'unresolved':
        r['present'] = False
        row.notes.append('name in __all__ does not resolve')
        return row
    if entry.kind == 'alias':
        return row
    if entry.kind in ('field', 'attribute'):
        doc, how = _member_doc(entry)
        row.source = _source_loc(entry.owner)
        if how == UNKNOWN:
            r['present'] = UNKNOWN
            return row
        r['present'] = bool(doc)
        if how:
            row.notes.append(f'documented by {how}')
        if PHYSICAL.match(entry.attr):
            r['units'] = bool(UNIT.search(doc or ''))
        return row
    if entry.kind == 'constant':
        doc, row.source = _constant_doc(entry)
        r['present'] = UNKNOWN if doc is None else bool(doc)
        return row

    obj, kind = entry.obj, entry.kind
    row.source = _source_loc(obj)
    doc = _own_doc(entry)
    r['present'] = bool(doc)
    row.convention = _convention(doc) if doc else 'none'
    if kind == 'module':
        row.target = (obj.__name__, '')
        row.examples, row.xrefs = example_blocks(doc), xrefs(doc)
        row.removed = [n for n in removed if re.search(rf'\b{re.escape(n)}\b', doc)]
        return row
    is_class = kind in ('class', 'exception', 'enum', 'factory')
    fn = None if is_class else _func_of(obj)
    if fn is not None:
        row.target = (fn.__module__, fn.__qualname__)
    elif kind != 'factory':
        row.target = (obj.__module__, obj.__qualname__)
    is_metric = (kind == 'function'
                 and (getattr(obj, '__module__', '') or '').startswith('silk.metrics'))

    params = None
    if kind in ('function', 'method', 'class', 'exception', 'factory') \
            and _silk_constructor(obj, kind):
        params = _signature_params(obj, kind)
        if params is None:
            r['params'] = UNKNOWN
        elif params:
            missing = [p for p in params if not re.search(rf'\b{re.escape(p)}\b', doc)]
            r['params'] = not missing
            if missing and doc:
                row.notes.append('undocumented params: ' + ', '.join(missing))
    if kind in ('function', 'method'):
        rv = _returns_value(fn)
        if rv == UNKNOWN:
            r['returns'] = r['returns_section'] = UNKNOWN
        elif rv:
            r['returns'] = bool(RETURNS_WORD.search(doc))
            r['returns_section'] = bool(re.search(
                r'^\s*(Returns|Yields)\s*(\n\s*-{3,}|:)\s*$', doc, re.M))
    elif kind == 'property':
        r['returns'] = UNKNOWN
    physical = (bool(PHYSICAL.match(entry.attr)) if kind == 'property'
                else any(PHYSICAL.match(p) for p in (params or ())))
    if kind in ('function', 'method', 'class', 'property', 'factory') and (is_metric or physical):
        r['units'] = bool(UNIT.search(doc))
    if kind in ('function', 'method', 'property'):
        raises = _raises(fn)
        if is_metric or raises is True:
            r['failure'] = bool(FAILURE.search(doc))
        elif raises == UNKNOWN:
            r['failure'] = UNKNOWN
    elif kind == 'class' and '__init__' in obj.__dict__:
        if _raises(obj.__dict__['__init__']) is True:
            r['failure'] = bool(FAILURE.search(doc))
    if doc:
        row.examples = example_blocks(doc)
        row.xrefs = xrefs(doc)
    if kind in ('function', 'class'):
        r['example'] = bool(row.examples) or bool(re.search(
            r'^\s*Examples?\s*(\n\s*-{3,}|:)\s*$', doc, re.M))
    if doc:
        stale = []
        allowed = (_signature_params(obj, kind, varargs=True)
                   if kind in ('function', 'method', 'class', 'exception', 'factory')
                   else None)
        if allowed is not None:
            stale += [f'param {n} not in signature' for n in documented_params(doc)
                      if n not in allowed]
        if re.search(r'from\s+silk\.signals\.metrics\s+import|silk\.signals\.metrics\.\w', doc):
            stale.append('uses the removed silk.signals.metrics import path')
        r['stale'] = not stale
        row.notes.extend(sorted(set(stale)))
        row.removed = [n for n in removed if re.search(rf'\b{re.escape(n)}\b', doc)]
    return row


# ---------------------------------------------------------------- execution

def _doctest_worker(module, qualname):
    """Run one docstring's ``>>>`` session; print a JSON result line."""
    obj = importlib.import_module(module)
    for part in filter(None, qualname.split('.')):    # '' = the module itself
        obj = getattr(obj, part)
    target = obj if inspect.isclass(obj) or inspect.ismodule(obj) else _func_of(obj)
    tests = doctest.DocTestFinder(recurse=False).find(
        target, qualname or module, module=False,
        globs=dict(vars(sys.modules[module])))
    flags = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
    attempted = failed = skipped = 0
    out = []
    for t in tests:
        skipped += sum(1 for e in t.examples if e.options.get(doctest.SKIP))
        res = doctest.DocTestRunner(optionflags=flags).run(t, out=out.append)
        # Python 3.13 counts skipped examples as attempted and exposes a
        # separate skipped count; earlier versions exclude them already.
        attempted += res.attempted - getattr(res, 'skipped', 0)
        failed += res.failed
    print(json.dumps({'attempted': attempted, 'failed': failed,
                      'skipped': skipped, 'text': ''.join(out)[-4000:]}))


def classify_run(res):
    """Outcome of one worker result dict. The worker's ``attempted`` count
    excludes skipped examples, so a session is all-skipped only when nothing
    was attempted and something was skipped."""
    if res['failed']:
        return f"failed {res['failed']}/{res['attempted']}"
    if res['attempted'] == 0:
        return 'skipped' if res['skipped'] else 'not run'
    return 'passed'


def run_examples(rows, timeout=60):
    """Execute each row's doctest session in a fresh interpreter, temporary cwd."""
    results = []
    env = dict(os.environ, MPLBACKEND='Agg',
               PYTHONPATH=os.pathsep.join(p for p in (str(REPO), os.environ.get('PYTHONPATH')) if p))
    for row in rows:
        if not row.target or not any(b['kind'] == 'doctest' for b in row.examples):
            continue
        with tempfile.TemporaryDirectory() as cwd:
            try:
                proc = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()),
                     '--doctest-worker', *row.target],
                    cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
            except subprocess.TimeoutExpired:
                results.append((row.name, 'not run', f'timeout after {timeout}s'))
                continue
        if proc.returncode != 0:
            results.append((row.name, 'not run',
                            f'worker exited {proc.returncode}: {proc.stderr[-2000:]}'))
            continue
        try:
            res = json.loads(proc.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            results.append((row.name, 'not run',
                            f'no result from worker: {proc.stderr[-2000:]}'))
            continue
        results.append((row.name, classify_run(res), res['text']))
    return results


# ---------------------------------------------------------------- report

def _cell(rows, c):
    app = [r for r in rows if r.results.get(c) is not None]
    if not app:
        return 'n/a'
    known = [r for r in app if r.results[c] != UNKNOWN]
    unk = len(app) - len(known)
    passed = sum(1 for r in known if r.results[c] is True)
    cell = f'{passed}/{len(known)}' + (f' ({100 * passed // len(known)}%)' if known else '')
    return cell + (f' +{unk}?' if unk else '')


def report(rows, base, worst=30, executed=None):
    groups = list(dict.fromkeys(r.group for r in rows))
    scored = [r for r in rows if r.kind != 'alias']
    lines = [f'# Docstring audit (base {base})', '',
             f'{len(rows)} rows, {len(rows) - len(scored)} of them aliases (not scored). '
             'Cell: passes / decidable applicable rows; `+N?` = applicable but '
             'unknown. `returns §` = structured Returns section. Keyword triage, '
             'not verification.', '']
    hdr = ['group', 'rows'] + ['returns §' if c == 'returns_section' else c
                               for c in CHECKS]
    lines += ['| ' + ' | '.join(hdr) + ' |', '|' + '---|' * len(hdr)]
    for g in groups + ['**all**']:
        rs = scored if g == '**all**' else [r for r in scored if r.group == g]
        lines.append('| ' + ' | '.join([g, str(len(rs))] + [_cell(rs, c) for c in CHECKS]) + ' |')
    kinds = {}
    for r in rows:
        kinds[r.kind] = kinds.get(r.kind, 0) + 1
    lines += ['', 'Rows by kind: ' + ', '.join(f'{k} {v}' for k, v in sorted(kinds.items()))]

    lines += ['', '## Conventions in use', '',
              '| group | numpy | google | sphinx | prose only | no docstring |',
              '|---|---|---|---|---|---|']
    docs = [r for r in scored if r.kind not in ('field', 'attribute', 'constant')]
    for g in groups + ['**all**']:
        rs = docs if g == '**all**' else [r for r in docs if r.group == g]
        cnt = [sum(1 for r in rs if r.convention == k)
               for k in ('numpy', 'google', 'sphinx', 'prose', 'none')]
        lines.append(f'| {g} | ' + ' | '.join(map(str, cnt)) + ' |')

    stale = [r for r in rows if r.results.get('stale') is False]
    lines += ['', f'## Stale (ghost parameters, removed imports): {len(stale)} rows', '']
    for r in stale:
        lines.append(f'- `{r.name}` ({r.source}): ' + '; '.join(
            n for n in r.notes if 'signature' in n or 'removed' in n))

    roles = sum(r.xrefs.get('roles', 0) for r in rows)
    qual = sum(r.xrefs.get('qualified', 0) for r in rows)
    unres = [(r, t) for r in rows for t in r.xrefs.get('unresolved', ())]
    lines += ['', '## Cross-reference roles (resolution is the Sphinx -n build\'s job)', '',
              f'{roles} roles; {qual} name a fully-qualified `silk.` target, of which '
              f'{len(unres)} do not import-resolve (triage list below; the build is '
              f'authoritative); the {roles - qual} relative targets are not checked here.', '']
    for r, t in unres:
        lines.append(f'- `{r.name}` ({r.source}): {t}')

    removed = [r for r in rows if r.removed]
    lines += ['', f'## Mentions of removed names ({len(removed)} rows)', '',
              'Names the CHANGELOG records as removed; most are migration notes. '
              'A generated reference would publish them.', '']
    for r in removed:
        lines.append(f'- `{r.name}` ({r.source}): ' + ', '.join(r.removed))

    blocks = [(r, b) for r in rows for b in r.examples]
    lines += ['', '## Example blocks (static; nothing executed)', '',
              '| kind | language | blocks | ok / compiles | compile fails | '
              'silk import fails | not python / not checked |',
              '|---|---|---|---|---|---|---|']
    for kind, lang in sorted({(b['kind'], b['lang']) for _, b in blocks}):
        bs = [b for _, b in blocks if b['kind'] == kind and b['lang'] == lang]
        cnt = [sum(b['status'] in ('ok', 'compiles as python') for b in bs),
               sum(b['status'].startswith('compile:') for b in bs),
               sum(b['status'].startswith('import') for b in bs),
               sum(b['status'] in ('not python', 'not checked') for b in bs)]
        lines.append(f'| {kind} | {lang} | {len(bs)} | ' + ' | '.join(map(str, cnt)) + ' |')
    lines.append('')
    for r, b in blocks:
        if b['status'].startswith(('compile:', 'import')):
            lines.append(f'- `{r.name}` ({r.source}): {b["kind"]} block, {b["status"]}')

    def weight(r):
        return (r.results['present'] is False, len(r.fails),
                r.kind in ('function', 'class'))
    ranked = sorted((r for r in scored if r.fails), key=weight, reverse=True)
    lines += ['', f'## Worst offenders (top {worst} of {len(ranked)} rows with any failure)', '',
              '| symbol | kind | fails | where |', '|---|---|---|---|']
    for r in ranked[:worst]:
        lines.append(f'| `{r.name}` | {r.kind} | {", ".join(r.fails)} | {r.source} |')

    n = sum(1 for r in rows if any(b['kind'] == 'doctest' for b in r.examples))
    if executed is None:
        lines += ['', f'## Doctest execution: not run ({n} docstrings with `>>>` '
                  'sessions; pass --run-examples)']
    else:
        tally = {}
        for _, st, _ in executed:
            tally[st.split(' ')[0]] = tally.get(st.split(' ')[0], 0) + 1
        lines += ['', f'## Doctest execution ({n} sessions; fresh interpreter, '
                  'temporary cwd)', '',
                  ', '.join(f'{k} {v}' for k, v in sorted(tally.items())), '']
        for name, st, text in executed:
            lines.append(f'- `{name}`: {st}')
            if st != 'passed' and text:
                lines += ['', '  ```text', textwrap.indent(text.strip(), '  '), '  ```', '']
    return '\n'.join(lines) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--manifest', default=str(MANIFEST))
    ap.add_argument('--worst', type=int, default=30)
    ap.add_argument('--json', metavar='PATH')
    ap.add_argument('--group', help='restrict --rows to one group')
    ap.add_argument('--rows', action='store_true',
                    help='print one line per row instead of the report')
    ap.add_argument('--run-examples', action='store_true',
                    help='execute >>> sessions, each in a fresh interpreter')
    ap.add_argument('--timeout', type=int, default=60,
                    help='seconds per docstring for --run-examples')
    ap.add_argument('--doctest-worker', nargs=2, metavar=('MODULE', 'QUALNAME'),
                    help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    if args.doctest_worker:
        _doctest_worker(*args.doctest_worker)
        return 0

    entries = collect(load_manifest(args.manifest))
    removed = removed_names()
    rows = [score(e, removed) for e in entries]
    base = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=REPO,
                          capture_output=True, text=True).stdout.strip() or '?'
    executed = run_examples(rows, args.timeout) if args.run_examples else None

    if args.json:
        Path(args.json).write_text(json.dumps(
            {'base': base, 'removed_names': removed, 'executed': executed,
             'rows': [dict(r.__dict__) for r in rows]}, indent=1, default=str))
    if args.rows:
        for r in rows:
            if args.group and r.group != args.group:
                continue
            res = ' '.join(f'{c}={"-" if v is None else "?" if v == UNKNOWN else int(v)}'
                           for c, v in r.results.items())
            print(f'{r.group}\t{r.name}\t{r.kind}\t{r.convention}\t{res}\t'
                  + '; '.join(r.notes))
        return 0
    sys.stdout.write(report(rows, base, args.worst, executed))
    return 0


if __name__ == '__main__':
    sys.exit(main())
