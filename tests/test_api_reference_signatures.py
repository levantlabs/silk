"""Signature currency guard on ``docs/api-reference.md`` (TASK-249).

``docs/api-reference.md`` is the page a user reaches for once past the
quickstart, and a signature that raises when copied is worse than a missing
one: the reader assumes their own mistake. TASK-249 found ten wrong signatures
on the page — seven whose parameter *names* did not exist (``TypeError:
unexpected keyword argument``), three properties documented under a
``**Methods:**`` header (``TypeError: 'int' object is not callable``) — plus
``Module.run_simulator``/``Module.simulate``, whose doc signatures omitted the
*leading* ``auto_track_structure`` parameter, so ``run_simulator(30)`` bound 30
to it (truthy → structure tracking on, database writes) while ``timeout``
stayed ``None``: no error at all, two wrong things silently.

The page mixes real signatures with deliberate abbreviations, so this guard is
built to stay quiet about the latter. The rules below are exactly the
properties a reader can be misled by:

* **Positional names must be an exact prefix of the code's.** A doc may stop
  early (``load_file(dataFile)`` for ``(dataFile, simulatorType='ngspice')``)
  because everything it does show still binds correctly. It may not rename
  (``add_net(name)`` for ``(netname)``) or skip a leading parameter
  (``run_simulator(timeout=None)`` for ``(auto_track_structure=False,
  timeout=None)``) — both misbind.
* **Keyword-only names must be a subset of the code's.** Order is irrelevant
  to a caller, and omitting some is fine; inventing one is not.
* **A row written ``name()`` must not be a property.** That is the
  ``pass_count()`` class of defect.
* **Rows containing a literal ``...`` are elisions**, and are skipped.
* **Rows whose arguments are values rather than parameter names are usage
  examples**, not signatures (``tb.set_temperature(85)``), and are skipped —
  as are chained expressions (``RoleRuleset().add(pattern, role)``).

Rows whose owner cannot be resolved are skipped rather than failed, so prose
rows like ``len(rs)`` cost nothing — but ``test_resolution_did_not_collapse``
pins a floor on how many rows *do* resolve, so a refactor cannot quietly turn
this file into a no-op.

Fast suite: no PDK, no simulator.
"""
import inspect
import re
import types
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_DOC = _REPO / 'docs' / 'api-reference.md'

#: Modules searched, in order, when a bare name is not found on the row's owner.
_SEARCH_MODULES = (
    'silk', 'silk.circuit', 'silk.specs', 'silk.lint', 'silk.simulation',
    'silk.pdk', 'silk.faults', 'silk.metrics', 'silk.db',
    'silk.signals', 'silk.sources',
)

#: Doc rows that abbreviate on purpose in a way the rules above cannot see.
#: Keyed by the exact first-cell token, valued by the reason it is allowed.
_DELIBERATE = {
    # `preset` is real, but it is consumed out of **options
    # (silk/circuit/modules_sim.py: `preset = options.pop('preset', None)`),
    # so inspect.signature cannot see it. Documenting it is the honest choice.
    "set_simulator_options(preset=None, **options)":
        'preset is popped from **options',
    # The code's `*extra_positional_signals` exists only so that passing a
    # signal positionally can raise TypeError naming the slots. Showing it in
    # the signature would advertise a calling convention that is refused.
    "add_metric(name, extractor_func=None, *, type=None, signal=None, "
    "signals=None, analysis=None, **params)":
        'the var-positional slot exists only to refuse positional signals',
}

_HEADING = re.compile(r'^#{2,4}\s+(.*)$')
_TICKED = re.compile(r'`([^`]+)`')
#: A bold lead-in such as ``**`RuleCoverageNode` methods:**`` also sets context.
_BOLD_CTX = re.compile(
    r'^\*\*`([A-Za-z_][\w.]*)`[^*]*\b(?:methods|fields|properties|members|attributes)\b',
    re.IGNORECASE)
_CALL = re.compile(r'^([A-Za-z_][\w.]*)\((.*)\)$', re.DOTALL)


def _resolve(dotted):
    """Resolve a dotted name against the search modules; None if not found."""
    import importlib
    candidates = [dotted] if '.' in dotted else []
    candidates += [f'{m}.{dotted}' for m in _SEARCH_MODULES]
    for cand in candidates:
        parts = cand.split('.')
        for i in range(len(parts), 0, -1):
            try:
                obj = importlib.import_module('.'.join(parts[:i]))
            except Exception:
                continue
            for part in parts[i:]:
                obj = getattr(obj, part, _MISSING)
                if obj is _MISSING:
                    break
            else:
                return obj
    return None


class _Missing:
    pass


_MISSING = _Missing()


def _static_attr(owner, name):
    """``getattr`` that does not fire descriptors, so properties stay visible."""
    if inspect.isclass(owner):
        for klass in owner.__mro__:
            if name in klass.__dict__:
                return klass.__dict__[name]
        return _MISSING
    return getattr(owner, name, _MISSING)


def _is_balanced(argstr):
    """Reject a chained expression such as ``RoleRuleset().add(pattern, role)``,
    where the greedy call regex swallows ``).add(`` into the argument text."""
    depth = 0
    for ch in argstr:
        if ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _split_top_level(argstr):
    out, depth, cur = [], 0, ''
    for ch in argstr:
        if ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth -= 1
        if ch == ',' and depth == 0:
            out.append(cur.strip())
            cur = ''
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def _doc_param_groups(argstr):
    """(positional, keyword-only, var-keyword) — or None if this is not a signature.

    Returns ``None`` when any argument is a *value* rather than a parameter
    name, which is how a usage example (``tb.set_temperature(85)``,
    ``add_metric('gain', type='dc_gain_dB')``) is told apart from a signature.
    """
    positional, keyword_only, var_kw = [], [], False
    after_star = False
    for raw in _split_top_level(argstr):
        if raw == '/':
            continue
        if raw == '*':
            after_star = True
            continue
        name = raw.split('=')[0].split(':')[0].strip()
        if name.startswith('**'):
            var_kw = True
            name = name[2:]
        elif name.startswith('*'):
            after_star = True
            name = name[1:]
            if not name:
                continue
        else:
            (keyword_only if after_star else positional).append(name)
        if not name.isidentifier():
            return None
    return positional, keyword_only, var_kw


def _code_param_groups(func):
    sig = inspect.signature(func)
    params = list(sig.parameters.values())
    if params and params[0].name in ('self', 'cls'):
        params = params[1:]
    P = inspect.Parameter
    positional = [p.name for p in params
                  if p.kind in (P.POSITIONAL_ONLY, P.POSITIONAL_OR_KEYWORD)]
    keyword_only = [p.name for p in params if p.kind is P.KEYWORD_ONLY]
    var_kw = any(p.kind is P.VAR_KEYWORD for p in params)
    return positional, keyword_only, var_kw, sig


def _rows():
    """Yield (lineno, owner_label, owner_obj, func_name, argstr, token)."""
    text = _DOC.read_text()
    owners, owner_label = [], None
    for lineno, line in enumerate(text.split('\n'), 1):
        heading = _HEADING.match(line)
        if heading:
            found = []
            for tick in _TICKED.findall(heading.group(1)):
                obj = _resolve(tick.split('(')[0].strip())
                if inspect.isclass(obj) or isinstance(obj, types.ModuleType):
                    found.append((tick.split('(')[0].strip(), obj))
            if found:
                owners, owner_label = found, found[0][0]
            elif heading.group(0).startswith('## '):
                owners, owner_label = [], None
            continue
        bold = _BOLD_CTX.match(line)
        if bold:
            obj = _resolve(bold.group(1))
            if inspect.isclass(obj) or isinstance(obj, types.ModuleType):
                owners, owner_label = [(bold.group(1), obj)], bold.group(1)
            continue
        if not line.startswith('|') or line.count('|') < 2:
            continue
        cell = line.split('|')[1]
        for token in _TICKED.findall(cell):
            call = _CALL.match(token.strip())
            if not call:
                continue
            yield lineno, owner_label, owners, call.group(1), call.group(2), token.strip()


def _checked_rows():
    """Every row that resolves to something with a signature, plus its owner."""
    for lineno, label, owners, name, argstr, token in _rows():
        if '...' in token or not _is_balanced(argstr):
            continue
        leaf = name.split('.')[-1]
        target = None
        for owner_name, owner in owners:
            attr = _static_attr(owner, leaf)
            if attr is not _MISSING:
                target = (f'{owner_name}.{leaf}', attr)
                break
        if target is None:
            obj = _resolve(name)
            if obj is not None:
                target = (name, obj)
        if target is None:
            continue
        yield lineno, target[0], target[1], argstr, token


def test_no_property_is_documented_as_a_method():
    """``pass_count()`` when the code says ``@property`` raises on every copy."""
    offenders = []
    for lineno, label, obj, argstr, token in _checked_rows():
        if isinstance(obj, property) and argstr.strip() == '':
            offenders.append(f'  docs/api-reference.md:{lineno}  `{token}` — '
                             f'{label} is a property; calling it raises TypeError. '
                             f'Document it as `{label.split(".")[-1]}` (property).')
    assert not offenders, (
        'docs/api-reference.md documents a property as a method:\n'
        + '\n'.join(offenders))


def test_documented_signatures_bind_the_way_the_code_does():
    offenders = []
    for lineno, label, obj, argstr, token in _checked_rows():
        if token in _DELIBERATE:
            continue
        if isinstance(obj, property) or not callable(obj):
            continue
        try:
            code_pos, code_kw, code_varkw, sig = _code_param_groups(obj)
        except (TypeError, ValueError):
            continue
        groups = _doc_param_groups(argstr)
        if groups is None:
            continue          # a usage example, not a signature
        doc_pos, doc_kw, doc_varkw = groups

        def fail(why):
            offenders.append(
                f'  docs/api-reference.md:{lineno}  {label}\n'
                f'      doc : {token}\n'
                f'      code: {label.split(".")[-1]}{sig}\n'
                f'      why : {why}')

        if doc_pos != code_pos[:len(doc_pos)]:
            fail('the documented positional parameters are not a prefix of the '
                 "code's — a reader's positional call misbinds")
        elif set(doc_kw) - set(code_kw) and not code_varkw:
            fail(f'documented keyword-only name(s) {sorted(set(doc_kw) - set(code_kw))} '
                 'do not exist — a keyword call raises TypeError')
        elif doc_varkw and not code_varkw:
            fail('the doc shows **kwargs; the code takes none')
    assert not offenders, (
        f'{len(offenders)} signature(s) in docs/api-reference.md disagree with the '
        'code (TASK-249). Fix the doc, or add a reason to _DELIBERATE if the\n'
        'abbreviation is intentional:\n' + '\n'.join(offenders))


def test_resolution_did_not_collapse():
    """Unresolvable rows are skipped, so pin how many actually get checked."""
    checked = list(_checked_rows())
    assert len(checked) >= 125, (
        f'only {len(checked)} doc signatures resolved to code objects; the guard '
        'is close to a no-op. A renamed module or a restructured heading most '
        'likely broke _resolve() or the heading-context tracking.')


@pytest.mark.parametrize('claim, measured', [
    ('registered simulator backends', 'backends'),
    ('silk.metrics.__all__ breakdown', 'metrics_all'),
    ('@metric catalog size', 'catalog'),
])
def test_counts_quoted_on_the_page_are_still_true(claim, measured):
    """The page quotes four measured counts; pin each so they cannot rot."""
    text = _DOC.read_text()
    if measured == 'backends':
        from silk.simulation.backend import SimulatorBackend
        # Other tests register fake backends at import time, so filter to the
        # ones silk itself ships.
        registered = sorted(
            name for name, klass in SimulatorBackend._registry.items()
            if getattr(klass, '__module__', '').startswith('silk.'))
        assert registered == ['behavioral', 'ngspice', 'xyce'], registered
        sentence = re.search(r'^Built-in backends are.*?\n\n', text,
                             re.MULTILINE | re.DOTALL)
        assert sentence, ('docs/api-reference.md no longer opens its backend '
                          'section with "Built-in backends are ..."')
        for name in registered:
            assert f"`'{name}'`" in sentence.group(0), (
                f"backend '{name}' is registered on SimulatorBackend but the "
                "\"Built-in backends are ...\" sentence in docs/api-reference.md "
                'does not name it')
    elif measured == 'metrics_all':
        import silk.metrics as M
        names = list(M.__all__)
        functions = [n for n in names if isinstance(getattr(M, n), types.FunctionType)]
        classes = [n for n in names if inspect.isclass(getattr(M, n))]
        modules = [n for n in names if isinstance(getattr(M, n), types.ModuleType)]
        assert len(functions) + len(classes) + len(modules) == len(names)
        expected = (f'{len(functions)} functions, {len(classes)} result classes '
                    f'and the {len(modules)}\nsubmodule namespaces — {len(names)} '
                    'public names in `silk.metrics.__all__`')
        assert expected in text, (
            'docs/api-reference.md no longer states the measured '
            f'silk.metrics.__all__ breakdown; it should read:\n{expected}')
    else:
        from silk.signals.schema import export_metric_schema
        size = len({k for k in export_metric_schema() if k != '_meta'})
        assert f'the {size} entries in the catalog' in text, (
            f'the @metric catalog holds {size} entries; docs/api-reference.md '
            'quotes a different number')
