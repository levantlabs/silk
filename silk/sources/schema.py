"""Machine-readable schema export for testbench sources.

The parallel to :mod:`silk.elements.schema` (which exports *device* schemas):
schema consumers reflect this instead of hardcoding source params, so a new source
param or waveform in silk shows up downstream automatically.

Truth split (see ``source_metadata.yaml``):

- **params + defaults are reflected** from each class's ``__init__`` signature —
  they cannot drift from the Python.
- **authored metadata** (symbol, unit, pin roles, the waveform-kind → class
  map) lives in ``source_metadata.yaml`` because it is not derivable by
  reflection.

Public API: :func:`export_source_schema` (and :func:`source_schema` for one
class). ``python -m silk.sources.schema`` prints the JSON.
"""
from __future__ import annotations

import enum
import inspect
import json
import os

import yaml

from silk._json_codec import encode_json
from silk.exceptions import CircuitError
from silk.utils import require_python_identifier
from silk.sources import waveforms as _waveforms
import silk.sources as _sources

_META_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'source_metadata.yaml')


def _load_metadata() -> dict:
    with open(_META_PATH) as fh:
        meta = yaml.safe_load(fh)
    try:
        for name, spec in meta.get('sources', {}).items():
            context = f"{_META_PATH}: source {name!r}"
            require_python_identifier(name, f'{context}, source key')
            for field in ('pins', 'scalar_params', 'references', 'expressions'):
                for key in spec.get(field, ()):
                    require_python_identifier(key, f'{context}, {field} name')
            for field in ('quantity', 'alias'):
                if field in spec:
                    require_python_identifier(spec[field], f'{context}, {field}')
        for kind, cls in meta.get('waveforms', {}).items():
            context = f"{_META_PATH}: waveform {kind!r}"
            require_python_identifier(kind, f'{context}, waveform key')
            require_python_identifier(cls, f'{context}, waveform class')
        # The flat factories publish alias + scalar/kind as a Python name.
        # Individually legal components can still form a keyword (a + nd).
        for name, spec in meta.get('sources', {}).items():
            alias = spec.get('alias')
            if spec.get('quantity') is None or alias is None:
                continue
            for kind in (*spec.get('scalar_params', ()), *meta.get('waveforms', {})):
                require_python_identifier(
                    alias + kind, f'{_META_PATH}: source {name!r}, factory name')
        for field in ('param_units', 'param_kinds'):
            for key in meta.get(field, {}):
                require_python_identifier(key, f'{_META_PATH}: {field} key')
    except ValueError as exc:
        raise CircuitError(str(exc)) from exc
    return meta


def _default_of(param: inspect.Parameter):
    return None if param.default is inspect.Parameter.empty else param.default


def _unit_of(param_name: str, base_unit: str, param_units: dict):
    """Resolve a param's unit; the sentinel ``'source'`` → the source's base
    unit (V/A). Params not in the map export ``None`` (unknown, not a guess)."""
    unit = param_units.get(param_name)
    if unit == 'source':
        return base_unit
    return unit


def _choices_of(param: inspect.Parameter | None):
    """``[{'value': int, 'label': name}, ...]`` when *param*'s annotation is
    an ``IntEnum`` subclass, else ``None``.

    Reflected from the annotation and from nothing else — the same
    discipline as the metric schema's ``choices`` (``silk.signals.schema``), so the mapping has exactly one home: the enum
    itself (e.g. :class:`silk.sources.waveforms.RandomType`). Unlike that
    metric-schema field, the caller omits the ``'choices'`` key entirely
    when this returns ``None`` — a source-schema parameter with no
    permitted set carries no ``choices`` key at all, rather than a ``null``
    value (see docs/codegen-contract.md's RandomSource subsection).
    """
    if param is None:
        return None
    ann = param.annotation
    if ann is inspect.Parameter.empty:
        return None
    if isinstance(ann, type) and issubclass(ann, enum.IntEnum):
        return [{'value': int(m.value), 'label': m.name} for m in ann]
    return None


def _reflect_params(cls, names, base_unit, param_units, param_kinds):
    """Return [{name, unit, default, required, [choices], [kind]}] for *names*
    reflected off ``cls.__init__``. Skips var-positional/var-keyword params.
    ``choices`` is present only for a parameter whose annotation is an
    ``IntEnum`` (see :func:`_choices_of`) — absent everywhere else."""
    sig = inspect.signature(cls.__init__)
    out = []
    for name in names:
        p = sig.parameters.get(name)
        if p is not None and p.kind in (inspect.Parameter.VAR_POSITIONAL,
                                        inspect.Parameter.VAR_KEYWORD):
            continue
        required = p is not None and (p.default is inspect.Parameter.empty or p.default is _waveforms.REQUIRED)
        entry = {
            'name': name,
            'unit': _unit_of(name, base_unit, param_units),
            'default': None if required else _default_of(p) if p is not None else None,
            'required': required,
        }
        if name in param_kinds:
            entry['kind'] = param_kinds[name]
        choices = _choices_of(p)
        if choices is not None:
            entry['choices'] = choices
        out.append(entry)
    return out


def _reflect_references(cls, names):
    """Return ``[{name, kind, required, default}]`` for constructor params
    that name another source's INSTANCE rather than carrying a pin or a
    physical quantity — e.g. ``ctrl_vname`` on CCVS/CCCS, which names the
    controlling voltage source whose branch current the element reads
    (:class:`~silk.sources.classes.CurrentControlledVoltageSource`,
    :class:`~silk.sources.classes.CurrentControlledCurrentSource`).

    Reflected for required-ness/default exactly like :func:`_reflect_params`,
    but carries no ``unit`` (it is not a physical quantity) and is returned
    under its own ``references`` key rather than folded into ``parameters`` — so a consumer can tell a source-name reference
    apart from a scalar param or a pin.
    """
    if not names:
        return []
    sig = inspect.signature(cls.__init__)
    out = []
    for name in names:
        p = sig.parameters.get(name)
        required = p is not None and (p.default is inspect.Parameter.empty or p.default is _waveforms.REQUIRED)
        out.append({
            'name': name,
            'kind': 'source',   # what the referenced name must resolve to
            'required': required,
            'default': None if required else _default_of(p) if p is not None else None,
        })
    return out


def _reflect_expressions(cls, names):
    """Return ``[{name, kind, required, default}]`` for constructor params
    that carry a free-form SPICE behavioral expression string — e.g.
    ``expression`` on :class:`~silk.sources.classes.BehavioralVoltage` /
    :class:`~silk.sources.classes.BehavioralCurrent` — passed through to
    SPICE verbatim (``B`` element ``V=``/``I=``).

    Reflected for required-ness/default exactly like :func:`_reflect_params`
    and :func:`_reflect_references`, but carries no ``unit`` (not a physical
    quantity) and no source-name resolution (not a ``references`` entry —
    nothing looks this string up against other ``sources[]`` entries).
    Returned under its own ``expressions`` key so
    a consumer can tell a raw SPICE expression apart from a scalar param or
    a source-instance reference.
    """
    if not names:
        return []
    sig = inspect.signature(cls.__init__)
    out = []
    for name in names:
        p = sig.parameters.get(name)
        required = p is not None and (p.default is inspect.Parameter.empty or p.default is _waveforms.REQUIRED)
        out.append({
            'name': name,
            'kind': 'expression',   # what the string is: a free-form SPICE expr
            'required': required,
            'default': None if required else _default_of(p) if p is not None else None,
        })
    return out


def source_schema(class_name: str, meta: dict | None = None) -> dict:
    """Return the schema dict for a single source class.

    Parameters
    ----------
    class_name : str
        A key under ``sources:`` in ``source_metadata.yaml``
        (e.g. ``'VoltageSource'``).
    meta : dict, optional
        Pre-loaded metadata (avoids re-reading the YAML in a loop). An empty
        or ``None`` value reloads it.

    Returns
    -------
    dict
        A JSON-serializable schema with the keys ``name``, ``category``,
        ``symbol``, ``unit`` (the source's base unit, such as ``'V'`` or
        ``'A'``), ``pins`` (``[{'name': ...}]`` in terminal order),
        ``parameters`` (the scalar parameters, each ``{name, unit, default,
        required}`` and, when declared, ``kind`` or ``choices``),
        ``waveforms`` (``{kind: {class, parameters, kwarg, code_template}}``
        for each waveform the source accepts; empty for sources with no
        waveform arguments), ``waveform_exclusive`` (always ``True``: at most
        one waveform may be set) and ``code_template``. Sources that declare
        them also carry ``references`` (names of other sources, such as
        ``ctrl_vname``) and ``expressions`` (free-form SPICE expressions).
        Parameter names, defaults and required-ness are reflected from the
        constructor signatures; units come from the metadata file and are
        ``None`` when none is declared.

    Raises
    ------
    KeyError
        If *class_name* is not declared in the metadata.
    silk.exceptions.CircuitError
        If the metadata file holds a name that is not a valid Python
        identifier.

    Examples
    --------
    >>> from silk.sources import source_schema
    >>> schema = source_schema('VoltageSource')
    >>> schema['unit'], [p['name'] for p in schema['parameters']]
    ('V', ['dc', 'ac'])
    """
    meta = meta or _load_metadata()
    src_meta = meta['sources'][class_name]
    param_units = meta.get('param_units', {})
    param_kinds = meta.get('param_kinds', {})
    wf_map = meta.get('waveforms', {})

    cls = getattr(_sources, class_name)
    base_unit = src_meta['unit']
    sig = inspect.signature(cls.__init__)

    # Scalar params (dc, ac, …) — reflected defaults + authored units.
    parameters = _reflect_params(
        cls, src_meta.get('scalar_params', []), base_unit, param_units, param_kinds)

    # Waveform kinds: those the source accepts via a dedicated kwarg, OR any
    # kind if the source has the generic `waveform=` slot — so a NEW kind added
    # by one metadata line (no dedicated kwarg) still surfaces in the schema
    # (and thus the catalog). No-op for the existing 8 kinds (they have kwargs).
    has_generic_slot = 'waveform' in sig.parameters
    waveforms = {}
    for kwarg, wf_class_name in wf_map.items():
        if kwarg not in sig.parameters and not has_generic_slot:
            continue
        wf_cls = getattr(_waveforms, wf_class_name)
        wf_sig = inspect.signature(wf_cls.__init__)
        names = [n for n in wf_sig.parameters if n != 'self']
        wf_params = _reflect_params(wf_cls, names, base_unit, param_units, param_kinds)
        # Composition template (spec §5.4): how the consumer builds this waveform
        # and which source kwarg it fills — reflected, not hardcoded downstream.
        wf_arg_str = ', '.join(f"{c['name']}={{param_{c['name']}}}" for c in wf_params)
        waveforms[kwarg] = {
            'class': wf_class_name,
            'parameters': wf_params,
            'kwarg': kwarg if kwarg in sig.parameters else 'waveform',
            'code_template': f"sources.{wf_class_name}({wf_arg_str})",
        }

    pins = [{'name': p} for p in src_meta.get('pins', [])]

    # References (TASK-253): constructor params naming another source's
    # INSTANCE (e.g. ctrl_vname on CCVS/CCCS) rather than a pin or a scalar.
    references = _reflect_references(cls, src_meta.get('references', []))

    # Expressions (TASK-282.5): constructor params carrying a free-form
    # SPICE behavioral expression string (e.g. `expression` on
    # BehavioralVoltage/BehavioralCurrent) rather than a pin, a scalar, or
    # a source-instance reference.
    expressions = _reflect_expressions(cls, src_meta.get('expressions', []))

    # Code template — the exact Python constructor line, pins + scalar params
    # (waveform kwargs are composed by the consumer from ``waveforms``).
    pin_args = ', '.join(f"{p['name']}={{net_{p['name']}}}" for p in pins)
    param_args = ', '.join(f"{c['name']}={{param_{c['name']}}}"
                           for c in parameters)
    ref_args = ', '.join(f"{r['name']}={{ref_{r['name']}}}" for r in references)
    expr_args = ', '.join(f"{e['name']}={{expr_{e['name']}}}" for e in expressions)
    all_args = ', '.join(filter(
        None, ["'{instance_name}'", pin_args, param_args, ref_args, expr_args]))
    code_template = f"sources.{class_name}({all_args})"

    result = {
        'name': class_name,
        'category': src_meta.get('category', 'source'),
        'symbol': src_meta.get('symbol'),
        'unit': base_unit,
        'pins': pins,
        'parameters': parameters,
        'waveforms': waveforms,
        # At most one waveform kind may be set on a source (structural: the
        # generic slot / first-set-wins). Reflected so consumers enforce it (§5.4).
        'waveform_exclusive': True,
        'code_template': code_template,
    }
    # Present only when the class declares one (mirrors the 'choices' field's
    # absent-not-null convention, TASK-252) — VoltageSource/CurrentSource's
    # schema is unchanged by this key existing at all (TASK-253 AC #3).
    if references:
        result['references'] = references
    # Same absent-not-null convention (TASK-282.5) — only BehavioralVoltage/
    # BehavioralCurrent declare one; the six pre-existing entries are
    # unaffected by this key existing at all (AC #3).
    if expressions:
        result['expressions'] = expressions
    return result


def export_source_schema(include_metadata: bool = True) -> dict:
    """Export schemas for every source declared in ``source_metadata.yaml``.

    Parameters
    ----------
    include_metadata : bool, optional
        Add the ``_meta`` entry described below. Default ``True``.

    Returns
    -------
    dict
        A mapping from source class name to its schema (see
        :func:`silk.sources.source_schema`). With *include_metadata* it also
        has a ``_meta`` key holding ``source_type`` (``'sources'``),
        ``source_count``, ``unschematized_sources`` (sorted names of direct
        ``SourceBase`` subclasses that have no schema entry; empty when every
        source is covered) and ``format_version``.

    Raises
    ------
    silk.exceptions.CircuitError
        If the metadata file holds a name that is not a valid Python
        identifier.

    Examples
    --------
    >>> from silk import export_source_schema
    >>> schema = export_source_schema(include_metadata=False)
    >>> 'VoltageSource' in schema
    True
    """
    meta = _load_metadata()
    result = {name: source_schema(name, meta) for name in meta['sources']}
    if include_metadata:
        # Surface SourceBase subclasses that have no schema entry yet, so a
        # newly-added source class is visible (in output + testable) rather
        # than silently dropped. All SourceBase subclasses are schematized as
        # of TASK-282.5 (independent stimulus, dependent/controlled, and
        # behavioral sources); this key stays and is empty until a new
        # subclass is added without a metadata entry.
        all_sources = {c.__name__ for c in _sources.SourceBase.__subclasses__()}
        unschematized = sorted(all_sources - set(meta['sources']))
        result['_meta'] = {
            'source_type': 'sources',
            'source_count': len(meta['sources']),
            'unschematized_sources': unschematized,
            'format_version': '1.0',
        }
    return result


def export_source_schema_json(indent: int = 2, **kwargs) -> str:
    # No default= — the export must be strictly JSON-serializable (matches
    # elements.schema); a bad value should raise, not be silently stringified.
    return json.dumps(encode_json(export_source_schema(**kwargs)), allow_nan=False, indent=indent)


if __name__ == '__main__':
    print(export_source_schema_json())
