"""Machine-readable schema export for signal metrics.

Parallel to :mod:`silk.elements.schema` (devices) and :mod:`silk.sources.schema`
(sources): schema consumers reflect this instead of hardcoding a name-only metric
catalog, so a new metric — or a new param on one — shows up downstream with its
required analysis domain, unit, and inspector fields automatically.

Truth split:

- **params (names, defaults, types) and signal inputs (names and KINDS) are
  reflected** from each metric's signature (an ``Annotated[Signal, KIND]``
  annotation marks a waveform input vs a scalar param, and names the kind of
  Signal the slot takes) — they cannot drift from the Python. Each scalar param's
  ``type``/``optional`` come from its annotation and from nothing else, and a ``Literal``-annotated param's permitted values reflect the
  same way into ``choices``: there is deliberately no authored
  ``param_types=`` and no authored ``param_choices=``, because — unlike a
  unit — the type and the permitted set ARE in the Python already.
- **authored metadata** (analysis ``domain``, result ``unit``, scalar-param
  units) is declared at the function via the :func:`metric` decorator, because
  it is not derivable by reflection (it lives only in prose docstrings today).
  One further authored fact lives in :mod:`silk.signals.kinds` and is
  published once per export as ``_meta.signal_kinds``: which silk analyses
  yield a Signal of each kind. It is authored because the fact that matters is
  an ABSENCE — no analysis produces a power sweep or a code-transition
  table — and an absence cannot be reflected from code that does not exist.

Only scalar-returning metrics are exported. Result-object functions may register
with an explicit ``outputs={field: unit}`` mapping; testbench evaluation keeps
the result and publishes dotted scalar entries, but export omits these functions
until consumers support output ports. Undeclared non-scalar returns and missing
return annotations retain the existing ``NonScalarMetricError`` refusals.

Public API: :func:`metric` (decorator, applied in :mod:`silk.metrics`), the
signal-input annotation aliases it is used with (:data:`Transient`,
:data:`FrequencyResponse`, :data:`NoiseDensity`,
:data:`CodeTransitions`, :data:`PowerSweep`, :data:`ReferenceWaveform`,
:data:`AnySignal`), :func:`export_metric_schema`, and the four decoration-time
errors :class:`DuplicateMetricError`, :class:`NonScalarMetricError`,
:class:`UnsupportedParamTypeError` and :class:`UnsupportedSignalKindError`.
``python -m silk.signals.schema`` prints the JSON.

The decorator lives here, in ``silk.signals``, but the catalog it feeds is
:mod:`silk.metrics`, the package that populates the registry.
"""
from __future__ import annotations

import inspect
from collections.abc import Mapping
import json
import types
import typing

import numpy as np

from silk._json_codec import encode_json
from silk.signals.base import Signal
from silk.signals.kinds import (
    ANY, CODE_TRANSITIONS, FREQUENCY_RESPONSE, NOISE_DENSITY,
    POWER_SWEEP, REFERENCE_WAVEFORM, SignalKind,
    TRANSIENT, signal_kinds_meta,
)


# ---------------------------------------------------------------------------
# The annotation vocabulary a metric declares its signal inputs with
# ---------------------------------------------------------------------------
#
# One alias per
# :class:`~silk.signals.kinds.SignalKind`, so a metric says WHICH kind of
# Signal a slot takes in the one place the fact already lived — the
# annotation — and ``signal_inputs[].kind`` is REFLECTED from it rather than
# authored a second time beside ``param_units``.
#
# ``Annotated`` is erased at runtime: a caller still passes a plain Signal,
# ``isinstance`` still sees a Signal, and nothing about the Python API
# changes. The kind object travels in ``__metadata__``, which is where
# :func:`_signal_slot` reads it and where a runtime guard (T-E) can read the
# very same object rather than a matching string.
#
# THEY LIVE HERE, not in ``silk.signals.kinds``, and that is deliberate:
# ``kinds`` is a LEAF module (stdlib + numpy, nothing from silk) because
# ``silk.signals`` and ``silk.metrics`` import into each other and a module
# either may need mid-cycle must not reach back into either. Building
# ``typing.Annotated[Signal, KIND]`` needs ``Signal``. This module already imports
# it, and every ``silk.metrics`` submodule already imports ``metric`` from
# here, so the aliases add ZERO new import edges — the option
# ``silk/signals/kinds.py``'s own docstring names first.

#: Any 1-D Signal, whatever its axis. For the primitives
#: :mod:`silk.metrics.waveform` and :mod:`silk.metrics.statistics`
#: document as deliberately axis-agnostic —
#: the ones whose unit is ``'[y]'`` or ``'[x]'`` because it is the INPUT's
#: unit. Its ``produced_by`` names every analysis, so it refuses nobody:
#: ``tests/test_metric_schema.py`` pins the exact set of inputs that may use
#: it, so reaching for it has to be a deliberate edit of that list.
AnySignal = typing.Annotated[Signal, ANY]

#: A trace over time (``xlabel='time'``) — a ``.tran`` probe.
#:
#: NOTE the name collision, which is why these aliases are deliberately NOT
#: re-exported from ``silk.signals``: :class:`silk.simulation.analysis.Transient`
#: is the ANALYSIS a testbench runs, and it is the far more commonly written
#: of the two. These live beside :func:`silk.signals.metric`, are imported only where a
#: metric is DEFINED, and are spelled as the approved design (§4.2) spells
#: them. A module that needs both should alias one at the import.
Transient = typing.Annotated[Signal, TRANSIENT]

#: A small-signal transfer over frequency — a ``.ac`` probe.
FrequencyResponse = typing.Annotated[Signal, FREQUENCY_RESPONSE]

#: A noise density over frequency (V/√Hz, A/√Hz) — a ``.noise`` probe.
NoiseDensity = typing.Annotated[Signal, NOISE_DENSITY]

#: An ADC transition-level table (``xlabel='code'``), as
#: :func:`silk.metrics.extract_code_transitions` derives it. No ANALYSIS
#: yields one.
CodeTransitions = typing.Annotated[Signal, CODE_TRANSITIONS]

#: x = input power in dBm, y = output power in dBm. No analysis yields one.
PowerSweep = typing.Annotated[Signal, POWER_SWEEP]

#: A target waveform the USER authored, to measure against. No analysis
#: yields one — ``error_norm``'s ``goal`` slot is the only user of this.
ReferenceWaveform = typing.Annotated[Signal, REFERENCE_WAVEFORM]


#: Array-like complex constellation/reference points. This annotation describes
#: accepted Python inputs; it does not coerce them or change runtime validation.
ComplexPoints = typing.Annotated[
    typing.Union[typing.Sequence[complex], np.ndarray], 'complex_points']


class DuplicateMetricError(ValueError):
    """Two metrics tried to register under the same name.

    A ``ValueError`` subclass so existing ``except ValueError`` handlers
    still catch it.
    """


class NonScalarMetricError(TypeError):
    """A decorated function does not return a declarable scalar.

    ``@metric`` declares *the unit of the returned scalar*. A function that
    returns a :class:`~silk.signals.base.Signal`, an array, a complex number,
    a bool or an undeclared typed result object has no single scalar unit, and a
    :class:`~silk.specs.Spec` cannot reconcile a pass/fail against one — so
    the refusal is correct rather than merely convenient.

    A ``TypeError`` subclass: this is decoration-time misuse of the API, the
    same family as :func:`metric_schema <silk.signals.metric_schema>` rejecting an undecorated function.

    Raised in two distinguishable situations, see ``_SCALAR_RETURNS``:

    - the return annotation is present but is not a scalar type, or
    - there is no return annotation at all, so the return cannot be
      classified either way (``ac.gain_margins`` is the live example).
      Refusing is the safe default: the decorator must not guess a unit for
      a return type nobody declared.
    """


class UnsupportedParamTypeError(TypeError):
    """A scalar param's annotation cannot be normalised for the schema.

    The schema publishes ``parameters[].type`` from a CLOSED vocabulary
    (``_PARAM_TYPES``) so that a consumer which cannot evaluate a Python
    type expression still gets a string
    it can switch on. An annotation outside that vocabulary has no honest
    normalisation, so the decorator refuses it at DECORATION time, naming the
    metric, the parameter and the annotation, rather than exporting a type
    nobody downstream can read (or, worse, guessing one).

    A ``TypeError`` subclass, the same family as :class:`NonScalarMetricError`:
    this is decoration-time misuse of the API.

    Raised in three distinguishable situations:

    - the parameter has no annotation at all, so there is nothing to reflect
      (the same reasoning as the missing-return-annotation refusal: the
      decorator will not guess);
    - the annotations cannot be resolved by :func:`typing.get_type_hints`
      (a forward reference to a name the module never defines);
    - the annotation resolves but is outside the vocabulary — for example,
      a ``Union`` of two real types with no single type name to publish.
      ``bool`` has its own entry and is never folded into ``int``.
    """


class UnsupportedSignalKindError(TypeError):
    """A Signal input does not declare which KIND of Signal it takes.

    Schema v2 publishes ``signal_inputs[]`` as ``{'name', 'kind'}`` objects,
    and ``kind`` is what lets a consumer say which probe fits which slot —
    and, through an empty
    :attr:`~silk.signals.kinds.SignalKind.produced_by`, refuse to offer a
    metric whose slot NOTHING silk can simulate will ever satisfy. A bare
    ``sig: Signal`` carries no such fact, so the decorator refuses it by
    parameter name rather than publishing a slot a consumer would have to
    guess at.

    A ``TypeError`` subclass, the same family as
    :class:`NonScalarMetricError` and :class:`UnsupportedParamTypeError`:
    decoration-time misuse of the API.

    **Why it must be loud, and at decoration time.** Under v1
    ``_is_signal_param`` matched the annotation TEXT, so an annotation the
    schema could not resolve degraded silently into a scalar param. v2
    resolves hints instead, and the
    new failure mode the contract in docs/schema-export.md warns about is the mirror image: a metric
    whose hints do not resolve, or whose Signal input names no kind, would
    export with FEWER signal inputs than it has — in the limit, with zero —
    and a consumer would wire it with none. Raising here means the refusal
    lands when :mod:`silk.metrics` is imported, naming the metric and the
    parameter, instead of appearing downstream as a metric that looks
    parameter-only.
    """


#: Return annotations ``@metric`` accepts as "a scalar with one unit".
#: Compared by NAME and by identity, never with ``issubclass``: ``bool`` is a
#: subclass of ``int`` in Python, and ``dc.is_monotonic() -> bool`` is exactly
#: one of the functions this guard has to refuse.
# Mixed real/complex metrics remain declarable; spec limits require real values.
_SCALAR_RETURNS = frozenset({'float', 'int', 'float | complex'})


def _return_annotation_name(ann) -> str:
    """Render a return annotation as a comparable/printable type name.

    Every module in :mod:`silk.metrics` uses ``from __future__ import
    annotations`` (PEP 563), so annotations arrive as STRINGS
    (``'float'``, ``'np.ndarray'``). Real type objects are handled too, so
    the guard does not depend on how a future module chooses to spell its
    annotations. A dotted or generic form keeps its full text for the error
    message but is compared on its trailing bare name (``'np.ndarray'`` →
    ``'ndarray'``), so no qualification can sneak past the check.
    """
    if isinstance(ann, str):
        return ann.strip()
    return getattr(ann, '__name__', None) or str(ann)


def _check_scalar_return(fn) -> None:
    """Refuse *fn* unless its RETURN ANNOTATION says it yields a scalar.

    This is a return-annotation check, deliberately NOT a list of excluded
    function names : the
    classification lives at the definition, so a new non-scalar metric is
    refused the day it is written and nothing has to be kept in sync
    elsewhere. When the result-shape question is settled the rule
    is relaxed in this one place.
    """
    ann = inspect.signature(fn).return_annotation
    where = f"{getattr(fn, '__module__', '?')}.{fn.__qualname__}"
    if ann is inspect.Signature.empty:
        raise NonScalarMetricError(
            f"@metric cannot decorate {where}: it has no return annotation, "
            f"so it cannot be classified as scalar or not. @metric declares "
            f"the unit of the RETURNED SCALAR, and the decorator will not "
            f"guess a unit for a return type nobody declared. Annotate the "
            f"return type — {' or '.join(sorted(_SCALAR_RETURNS))} to be "
            f"schema-exportable, anything else to be refused explicitly."
        )
    name = _return_annotation_name(ann)
    bare = name.split('[')[0].split('.')[-1].strip()
    if bare not in _SCALAR_RETURNS:
        raise NonScalarMetricError(
            f"@metric cannot decorate {where}: it returns {name!r}, which is "
            f"not a scalar. @metric declares the unit of the returned scalar, "
            f"and a Signal / array / complex / bool / result-object return "
            f"has no single unit for a Spec to reconcile a pass/fail against. "
            f"Accepted return annotations: "
            f"{', '.join(sorted(_SCALAR_RETURNS))}."
        )


# name → decorated function. Populated as silk.metrics is imported.
_REGISTRY: dict = {}


def _check_outputs(fn, outputs):
    """Validate explicit scalar fields without admitting undeclared results."""
    if not isinstance(outputs, Mapping) or not outputs:
        raise ValueError('metric outputs must be a nonempty field -> unit mapping')
    if inspect.signature(fn).return_annotation is inspect.Signature.empty:
        _check_scalar_return(fn)  # Preserve the existing missing-annotation error.
    result_type = _resolve_hints(fn, fn.__qualname__).get('return')
    if not inspect.isclass(result_type) or not getattr(result_type, '__annotations__', None):
        raise NonScalarMetricError(
            f"@metric cannot decorate {fn.__qualname__}: outputs require an "
            f"annotated result-object return, got {result_type!r}")
    fields = typing.get_type_hints(result_type)
    for field, unit in outputs.items():
        if (not isinstance(field, str) or not field.isidentifier()
                or field.startswith('_') or fields.get(field) is not float):
            raise ValueError(
                f"@metric {fn.__qualname__}: output {field!r} must name a "
                f"public float field of {result_type.__name__}")
        if not isinstance(unit, str):
            raise TypeError(f"@metric {fn.__qualname__}: unit for {field!r} must be str")


def metric(*, domain: str, unit: str = '', param_units: dict | None = None,
           outputs: Mapping[str, str] | None = None):
    """Attach schema metadata to a metric function and register it.

    Parameters
    ----------
    domain : str
        Analysis domain the metric requires — ``'ac'``, ``'tran'``,
        ``'noise'``, ``'dc'``. Lets a consumer auto-pair the metric with the right
        analysis.
    unit : str
        Unit of the returned scalar (e.g. ``'dB'``, ``'Hz'``, ``'deg'``,
        ``'V'``, ``'V/V'``). Four conventions beyond plain SI / log units, used where the maths does not fix a concrete unit:

        - ``''`` — genuinely dimensionless (a ratio or a fraction). Matches
          the ``''`` default of :attr:`silk.specs.Spec.units`, which formats
          as ``f'{value:g}{units}'``, so a ratio prints bare.
        - ``'[y]'`` — the unit of the INPUT Signal's y-axis (V on a voltage
          trace, A on a current trace), for a metric whose result IS an
          input y-value (or a function of only them) and so has no single
          concrete unit of its own. Most of these are the axis-agnostic
          primitives in :mod:`silk.metrics.waveform` and
          :mod:`silk.metrics.statistics` (``average``, ``rms``, ``peak``, …).
          ``noise.noise_density_at_freq`` and ``statistics.error_norm`` declare it for
          the same reason on a NON-``any`` Signal kind — each returns
          exactly one (or a norm of) the input's own samples, so a
          hard-coded concrete unit (``'V/sqrt(Hz)'``, ``'V'``) would be
          wrong on an A/sqrt(Hz) or a current-trace input.
        - ``'[x]'`` — likewise the unit of the input Signal's x-axis (s in a
          transient, Hz in a sweep), for the axis-agnostic primitives that
          return an x position (``argmax_x``, ``argmin_x``, ``crossing``).
        - ``'[y]*sqrt([x])'`` — the input's y-axis unit times the square
          root of its x-axis unit, for a metric whose result is
          ``sqrt(integral of y^2 dx)`` over a density
          (``noise.integrated_noise``): ``sig.y`` is a density in
          ``[y]``/sqrt(Hz) — V/sqrt(Hz) for a voltage-noise probe,
          A/sqrt(Hz) for a current-noise one — so integrating its square
          over a frequency band and taking the square root gives
          ``[y]*sqrt(Hz)``, i.e. ``[y]*sqrt([x])`` in general. A hard-coded
          ``'V'`` mislabels the result on a current-noise probe, the same
          defect Silk fixed for ``error_norm``.

    param_units : dict, optional
        ``{param_name: unit}`` for the scalar (non-Signal) params. Declared
        for params that carry a physical unit; a count, a string selector or
        a dimensionless fraction is left out and reflects as ``None``.

    outputs : dict, optional
        Public float fields of an annotated result object, mapped to units.
        Evaluation stores both the bare result and ``<metric>.<field>`` values.
        A failed extraction gives every declared field the same error string.
        Such functions are registered for Python binding but omitted from
        :func:`export_metric_schema`. Without this mapping, the scalar return
        guard is unchanged. Empty mappings and unknown/non-float fields fail.

    Notes
    -----

    Returns the function unchanged (no wrapping) — only metadata + registration.

    Raises
    ------
    NonScalarMetricError
        If *fn*'s return annotation is not a scalar and no outputs are declared,
        or if it has no return annotation at all. ``@metric`` declares the unit of the
        returned scalar. A float | complex return is allowed, with complex
        values preserved in reports but refused by spec comparisons. A Signal-, array-, complex-only, bool- or
        result-object-returning function cannot carry one and an unannotated
        one cannot be classified. See ``_check_scalar_return`` — this is
        a return-annotation check at the definition, NOT a hand-maintained
        list of excluded names.
    UnsupportedParamTypeError
        If a scalar param has no annotation, or one the schema's closed type
        vocabulary cannot normalise. ``type`` and ``optional`` are
        REFLECTED from the annotation — there is no authored ``param_types=``
        and no authored ``param_choices=`` — so an annotation
        nobody downstream could read fails loudly at decoration time instead
        of exporting as a guess. See ``_reflect_param_types``.
    DuplicateMetricError
        If *name* is already registered by a different function. The registry
        is keyed on ``fn.__name__`` alone, so two metrics sharing a name used
        to silently overwrite each other: whichever module imported last won,
        the other vanished from the exported schema, and ``metric_count``
        quietly dropped by one. That is exactly the failure the move
        between packages could have caused, so it fails loudly instead. Re-registering the SAME function object (a module
        re-executed under a second name, e.g. ``python -m``) is not a
        collision and is allowed.
    """
    def deco(fn):
        if outputs is None:
            _check_scalar_return(fn)
        else:
            _check_outputs(fn, outputs)
        # Reflect (and therefore validate) every scalar param's type and
        # every signal input's kind before anything is recorded, so a metric
        # the schema could not describe is never registered and never carries
        # metadata. The kind pass is TASK-196's: without it a Signal input
        # with no kind would export as a slot a consumer cannot wire.
        _reflect_param_types(fn)
        _reflect_signal_inputs(fn)
        fn._silk_metric = {
            'domain': domain,
            'unit': unit,
            'param_units': dict(param_units or {}),
        }
        if outputs is not None:
            fn._silk_metric['outputs'] = dict(outputs)
        name = fn.__name__
        existing = _REGISTRY.get(name)
        if existing is not None and existing is not fn:
            raise DuplicateMetricError(
                f"metric {name!r} is already registered by "
                f"{getattr(existing, '__module__', '?')}.{existing.__qualname__}; "
                f"{getattr(fn, '__module__', '?')}.{fn.__qualname__} would "
                f"silently replace it. Metric names are the registry key and "
                f"must be unique across silk.metrics — rename one of them."
            )
        _REGISTRY[name] = fn
        return fn
    return deco


def get_metric(name: str):
    """Return the registered @metric function for `name` (imports the catalog first).

    Parameters
    ----------
    name : str
        The name of the metric function to retrieve.

    Returns
    -------
    callable
        The registered metric function.

    Raises
    ------
    KeyError
        If the metric name is not found. Includes the list of known metrics.
    """
    import silk.metrics  # noqa: F401 — populate the registry
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown metric type {name!r}; known: {sorted(_REGISTRY)}") from None


def _signal_slot(hint):
    """``(is_signal_input, kind_or_None)`` for one RESOLVED annotation.

    The v2 replacement for v1's annotation-TEXT match. v1 asked whether the raw annotation's trailing name
    spelled ``'Signal'``, which was wrong twice over: a ``Signal`` SUBCLASS
    spelled in a module with ``from __future__ import annotations`` read as
    ``'NgspiceSignal'`` and became a scalar param. Here the hint is already
    resolved by :func:`typing.get_type_hints`, so the question is asked of the
    object.

    ``Optional[Signal]`` is deliberately NOT a signal input: it is neither a
    class nor an ``Annotated`` Signal, so it falls through to the scalar path
    and :func:`_reflect_param_types` refuses it by name. A slot that may be
    absent is a shape the schema has no way to publish, and refusing it beats
    either guess.

    ``Annotated[Signal, KIND]`` carries the kind in ``__metadata__``; the
    first :class:`~silk.signals.kinds.SignalKind` found there is the slot's
    kind. A bare ``Signal`` is still recognised as a signal INPUT — it has
    to be, or it would silently become a scalar param, which is the failure
    this function exists to prevent — but with ``None`` for its kind, and
    :func:`_reflect_signal_inputs` is what turns that into a refusal.
    """
    metadata = getattr(hint, '__metadata__', None)
    base = typing.get_args(hint)[0] if metadata else hint
    if not (inspect.isclass(base) and issubclass(base, Signal)):
        return (False, None)
    for item in (metadata or ()):
        if isinstance(item, SignalKind):
            return (True, item)
    return (True, None)


def _is_signal_param(hint) -> bool:
    """True when a RESOLVED annotation declares a waveform input.

    The one place the signal/scalar split is decided. Takes the resolved
    hint, not the :class:`inspect.Parameter`: see :func:`_signal_slot`.
    """
    return _signal_slot(hint)[0]


def _resolve_hints(fn, where: str) -> dict:
    """``typing.get_type_hints(fn, include_extras=True)``, or refuse loudly.

    ``include_extras`` is load-bearing: without it ``Annotated[Signal,
    TRANSIENT]`` resolves to a bare ``Signal`` and every kind in the catalog
    would vanish silently.
    """
    try:
        return typing.get_type_hints(fn, include_extras=True)
    except Exception as exc:
        raise UnsupportedParamTypeError(
            f"@metric cannot decorate {where}: its annotations could not be "
            f"resolved ({type(exc).__name__}: {exc}). The schema publishes "
            f"each scalar param's type and each signal input's kind, both "
            f"reflected from the annotation, so every annotation on a metric "
            f"must resolve at decoration time."
        ) from exc


def _kind_vocabulary_text() -> str:
    """The annotation aliases, as prose, for an error message."""
    return ', '.join(sorted(
        n for n, v in globals().items()
        if getattr(v, '__metadata__', None)
        and any(isinstance(m, SignalKind) for m in v.__metadata__)
    ))


def _reflect_signal_inputs(fn) -> dict:
    """``{param_name: SignalKind}`` for *fn*'s waveform inputs, in order.

    REFLECTED from ``Annotated[Signal, KIND]`` and from nothing else — there
    is deliberately no authored ``signal_kinds={}`` beside ``param_units``,
    because the fact "this slot takes a sideband" is already in the Python
    (as the guard in the body), and a second copy is the one that rots
    (the contract in docs/schema-export.md).

    Raises
    ------
    UnsupportedSignalKindError
        For a Signal input with no kind, naming the metric and the
        parameter. See that class for why the refusal is loud and why it is
        at decoration time.
    """
    where = f"{getattr(fn, '__module__', '?')}.{fn.__qualname__}"
    hints = _resolve_hints(fn, where)
    out = {}
    for name, p in inspect.signature(fn).parameters.items():
        if p.kind in (inspect.Parameter.VAR_POSITIONAL,
                      inspect.Parameter.VAR_KEYWORD):
            continue
        is_signal, kind = _signal_slot(hints.get(name))
        if not is_signal:
            continue
        if kind is None:
            raise UnsupportedSignalKindError(
                f"@metric cannot decorate {where}: signal input {name!r} is "
                f"annotated with a bare Signal, which does not say WHICH "
                f"kind of Signal it takes. schema v2 publishes "
                f"signal_inputs[].kind so a consumer can tell which probe "
                f"fits the slot — and refuse the metric when nothing it can "
                f"simulate fits. Annotate it with one of: "
                f"{_kind_vocabulary_text()} (from silk.signals.schema)."
            )
        out[name] = kind
    return out


#: Python types ``parameters[].type`` may name — a CLOSED vocabulary
#: See docs/schema-export.md for the type vocabulary. Normalised rather
#: than published as annotation text because ``"Optional[float]"`` forces every
#: consumer to write a parser and a JS consumer cannot evaluate a Python type
#: expression at all.
#:
#: Types are matched by identity, never through subclasses. ``bool`` has its
#: own entry rather than inheriting ``int``. ``ComplexPoints`` marks array-like
#: complex constellation/reference values without admitting arbitrary arrays.
_PARAM_TYPES = {float: 'float', int: 'int', str: 'str', bool: 'bool',
                ComplexPoints: 'complex_points'}


def _vocabulary_text() -> str:
    """The vocabulary as prose, for an error message."""
    return ', '.join(sorted(set(_PARAM_TYPES.values())))


def _annotation_text(ann) -> str:
    """Render an annotation for an error message.

    Every module in :mod:`silk.metrics` carries ``from __future__ import
    annotations`` (PEP 563), so the raw annotation is already the source text
    the author wrote (``'Optional[float]'``) and that is the most useful thing
    to quote back. A real object keeps its full generic form via ``str`` —
    ``__name__`` on a ``typing`` alias is the unhelpful bare ``'Optional'``.
    """
    if isinstance(ann, str):
        return ann.strip()
    if typing.get_origin(ann) is not None:
        return str(ann)
    return getattr(ann, '__name__', None) or str(ann)


def _normalise_param_type(ann):
    """Normalise a RESOLVED annotation to ``(type_name, optional, choices)``.

    ``Optional[X]`` (equivalently ``X | None``) contributes ``X``'s name plus
    ``optional=True``. ``optional`` and ``required`` answer different questions
    and the schema publishes both: ``required`` = no default, ``optional`` =
    ``None`` is an accepted value. ``spectral_power(window:
    Optional[Literal['hann', ...]] = 'hann')`` is optional and not required;
    ``settling_time.target`` defaults to ``None`` and is optional and not required.

    ``Literal['a', 'b']`` normalises to the type of its VALUES plus
    ``choices=['a', 'b']`` — the permitted set is reflected from
    the annotation and from nothing else, so it cannot be a second copy of
    the tuple the function body already guards on. ``choices`` is ``None``
    for a plain annotation. ``Optional[Literal[...]]`` is both: the strings
    are the choices and ``None`` is additionally accepted, which is exactly
    ``spectral_power.window``.

    Returns ``None`` — never a guess — for anything outside
    :data:`_PARAM_TYPES`, including a ``Literal`` whose values are not all of
    one supported type (``Literal['a', 1]`` has no single ``type`` to publish).
    Bool literals normalise to ``bool``, never to ``int``.
    """
    optional = False
    if typing.get_origin(ann) in (typing.Union, types.UnionType):
        args = typing.get_args(ann)
        inner = [a for a in args if a is not type(None)]
        optional = len(inner) != len(args)
        if len(inner) != 1:
            return None
        ann = inner[0]
    choices = None
    if typing.get_origin(ann) is typing.Literal:
        values = typing.get_args(ann)
        value_types = {type(v) for v in values}
        if len(value_types) != 1:
            return None            # no single type to publish
        choices = list(values)
        ann = value_types.pop()
    try:
        name = _PARAM_TYPES.get(ann)
    except TypeError:              # an unhashable annotation is not a type
        return None
    return None if name is None else (name, optional, choices)


def _reflect_param_types(fn) -> dict:
    """``{param_name: (type_name, optional, choices)}`` for *fn*'s SCALAR params.

    REFLECTED from the annotations and from nothing else (for the
    type and permitted values — there is no authored
    ``param_types=`` and no authored ``param_choices=``). Signal
    inputs are skipped — :func:`_is_signal_param` owns that split, and what a
    signal input must additionally declare belongs to :func:`_is_signal_param`.

    Annotations are resolved with :func:`typing.get_type_hints` because the
    metric modules stringize theirs (PEP 563); the same call is what a consumer
    had to reach around the schema and make for itself before this existed.

    Raises
    ------
    UnsupportedParamTypeError
        For a param with no annotation, an unresolvable one, or one outside
        :data:`_PARAM_TYPES`. Called from the decorator, so the refusal lands
        when :mod:`silk.metrics` is imported rather than on the consumer
        reading the export.
    """
    where = f"{getattr(fn, '__module__', '?')}.{fn.__qualname__}"
    hints = _resolve_hints(fn, where)

    out = {}
    for name, p in inspect.signature(fn).parameters.items():
        if p.kind in (inspect.Parameter.VAR_POSITIONAL,
                      inspect.Parameter.VAR_KEYWORD):
            continue
        if _is_signal_param(hints.get(name)):
            continue
        if name not in hints:
            raise UnsupportedParamTypeError(
                f"@metric cannot decorate {where}: parameter {name!r} has no "
                f"annotation, so the schema cannot publish its type and the "
                f"decorator will not guess one. Annotate it with "
                f"{_vocabulary_text()} (or Optional[...] of one)."
            )
        normalised = _normalise_param_type(hints[name])
        if normalised is None:
            raise UnsupportedParamTypeError(
                f"@metric cannot decorate {where}: parameter {name!r} is "
                f"annotated {_annotation_text(p.annotation)!r}, which the "
                f"schema cannot normalise. parameters[].type is a closed "
                f"vocabulary a consumer can switch on: {_vocabulary_text()}, "
                f"each optionally wrapped in Optional[...] and/or declared as "
                f"a Literal[...] of values that all share one of those types."
            )
        out[name] = normalised
    return out


def metric_schema(fn) -> dict:
    """Return the schema dict for one decorated metric function.

    ``signal_inputs[]`` is a list of ``{'name', 'kind'}`` OBJECTS
    (``format_version`` ``'2.0'``). ``kind`` is the name of a
    ``silk.signals.kinds.SignalKind``, reflected from the parameter's
    ``Annotated[Signal, KIND]`` annotation, and the table it indexes into is
    published once per export under ``_meta.signal_kinds``.

    ``parameters[]`` carries ``type``, ``optional`` and ``choices`` alongside
    the name, unit, default and required flag — all three reflected from the
    annotation by ``_reflect_param_types`` on every call, so none can be
    authored and none can go stale. ``choices`` is the permitted set of a
    ``Literal``-annotated param and ``None`` for every other one; the function
    body's own ``if x not in (...)`` guard is what ENFORCES it (an annotation
    does not), and a conformance test asserts the two agree.
    """
    meta = getattr(fn, '_silk_metric', None)
    if meta is None:
        raise TypeError(f"{fn.__name__} is not a @metric-decorated function")
    sig = inspect.signature(fn)
    param_types = _reflect_param_types(fn)
    signal_kinds = _reflect_signal_inputs(fn)

    signal_inputs = []
    parameters = []
    for name, p in sig.parameters.items():
        if p.kind in (inspect.Parameter.VAR_POSITIONAL,
                      inspect.Parameter.VAR_KEYWORD):
            continue
        if name in signal_kinds:
            signal_inputs.append({'name': name,
                                  'kind': signal_kinds[name].name})
        else:
            required = p.default is inspect.Parameter.empty
            type_name, optional, choices = param_types[name]
            parameters.append({
                'name': name,
                'unit': meta['param_units'].get(name),
                'default': None if required else p.default,
                'required': required,
                'type': type_name,
                'optional': optional,
                'choices': choices,
            })

    return {
        'name': fn.__name__,
        'domain': meta['domain'],
        'unit': meta['unit'],
        'signal_inputs': signal_inputs,
        'parameters': parameters,
    }


def export_metric_schema(include_metadata: bool = True) -> dict:
    """Export schemas for scalar ``@metric`` functions, omitting result objects.

    Importing this module does not by itself populate the registry — import
    the catalog first (``import silk.metrics``). Done here so a bare
    ``export_metric_schema()`` works. The registry is populated from
    ``silk.metrics``.

    Examples
    --------
    >>> from silk import export_metric_schema
    >>> schema = export_metric_schema(include_metadata=False)
    >>> 'average' in schema
    True
    """
    import silk.metrics  # noqa: F401 — ensure the catalog is registered
    result = {name: metric_schema(fn) for name, fn in _REGISTRY.items()
              if not fn._silk_metric.get('outputs')}
    if include_metadata:
        result['_meta'] = {
            'source_type': 'metrics',
            'metric_count': len(result),
            # '2.0' because signal_inputs[] changed SHAPE, from a list of
            # names to a list of {name, kind} objects (TASK-196, design
            # §8.1). Everything else v2 added is additive. Nothing in silk
            # branches on this number and, as of writing, nothing downstream
            # did either (design §8.3) — a tolerant schema reader is what
            # makes the bump mean something, by checking the MAJOR at import.
            'format_version': '2.0',
            # The one AUTHORED field of v2 (design §1): which silk analyses
            # yield a Signal of each kind. Published ONCE here rather than
            # repeated per entry, so a consumer evaluates the refusal rule —
            # offer a metric iff EVERY signal_inputs[].kind has a produced_by
            # that intersects the analyses it can author — against one table.
            # An EMPTY produced_by is the fact that matters: it says nothing
            # silk simulates will ever satisfy that slot.
            'signal_kinds': signal_kinds_meta(),
        }
    return result


def export_metric_schema_json(indent: int = 2, **kwargs) -> str:
    # No default= — the export must be strictly JSON-serializable; a bad value
    # should fail loudly, not be silently stringified (matches elements.schema).
    return json.dumps(encode_json(export_metric_schema(**kwargs)), allow_nan=False, indent=indent)


if __name__ == '__main__':
    # Re-import under the canonical module name: `python -m` runs this file as
    # __main__, but silk.metrics registers into the *silk.signals.schema*
    # instance's registry, so call that instance's function.
    from silk.signals.schema import export_metric_schema_json as _emj
    print(_emj())
