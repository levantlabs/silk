"""Dotted-path metric resolution.

A Spec's ``metric`` field can be either a plain metric name (looked
up in a ModuleTestbench's metric dict at reconcile time) or a dotted
path of the form ``'net.<name>.<attr>'`` or ``'device.<name>.<attr>'``.
The dotted form lets a spec point directly at per-Net or per-Device
state — for example, ``'net.vout.peak_current'`` or
``'device.mn1.vgs'``. This is the hook point for per-object derived
annotations.

The resolver is a pure function: it takes a Module and a metric
string and returns a ``ResolveResult`` describing whether the path
resolved to a numeric value. The caller (the reconciler) uses the
result to either record a measurement or mark the spec UNMEASURED.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ResolveResult:
    """Outcome of attempting to resolve a dotted-path metric.

    Parameters
    ----------
    resolved : bool
        True when the path produced a numeric value.
    value : float, optional
        The value, converted with ``float``; ``None`` unless *resolved*.
    reason : str, optional
        Why resolution did not produce a value; ``None`` when it did. One of
        ``'not_a_dotted_path'``, ``'malformed_path'``, ``'unknown_prefix'``,
        ``'net_not_found'``, ``'device_not_found'``, ``'attr_not_found'``,
        ``'value_is_none'`` or ``'value_not_numeric'``.
    """
    #: True when the path produced a numeric value.
    resolved: bool
    #: The resolved value as a float; ``None`` unless ``resolved``.
    value: Optional[float] = None
    #: Machine-readable reason resolution failed; ``None`` on success.
    reason: Optional[str] = None


def resolve_metric(module: Any, metric: str, *, reader=None) -> ResolveResult:
    """Resolve a dotted-path metric against a module's state.

    Supported forms, each of exactly three dot-separated parts:

    - ``net.<name>.<attr>`` reads ``module.net_dict[name].<attr>``;
    - ``device.<name>.<attr>`` reads ``module.elem_dict[name].<attr>``
      (the ``'device'`` entry when the element is stored as a dict).

    A metric without a ``'.'`` is a plain metric name: the result has
    ``resolved=False, reason='not_a_dotted_path'`` so the caller knows to look
    it up in a testbench's metric dict instead. Failures are returned in the
    result, never raised.

    Parameters
    ----------
    module : Module
        The module whose ``net_dict`` / ``elem_dict`` is read.
    metric : str
        The metric name or dotted path.
    reader : callable, optional
        Keyword-only. Called first with *metric*; a non-``None`` return value
        is returned unchanged and the module is not read.

    Returns
    -------
    ResolveResult
        ``resolved=True`` with the attribute converted by ``float``, or
        ``resolved=False`` with a ``reason``. A ``None`` attribute gives
        ``'value_is_none'`` and a non-numeric one ``'value_not_numeric'``.

    Examples
    --------
    >>> from types import SimpleNamespace
    >>> from silk.specs.resolver import resolve_metric
    >>> module = SimpleNamespace(net_dict={'vout': SimpleNamespace(peak_current=1.5e-3)})
    >>> resolve_metric(module, 'net.vout.peak_current')
    ResolveResult(resolved=True, value=0.0015, reason=None)
    >>> resolve_metric(module, 'gain_dB').reason
    'not_a_dotted_path'
    """
    if reader is not None:
        selected = reader(metric)
        if selected is not None:
            return selected
    if '.' not in metric:
        return ResolveResult(resolved=False, reason='not_a_dotted_path')

    parts = metric.split('.')
    if len(parts) != 3:
        return ResolveResult(resolved=False, reason='malformed_path')

    prefix, name, attr = parts

    if prefix == 'net':
        container = getattr(module, 'net_dict', None) or {}
    elif prefix == 'device':
        container = getattr(module, 'elem_dict', None) or {}
    else:
        return ResolveResult(resolved=False, reason='unknown_prefix')

    obj = container.get(name)
    if prefix == 'device' and isinstance(obj, dict):
        obj = obj.get('device')
    if obj is None:
        return ResolveResult(
            resolved=False,
            reason='net_not_found' if prefix == 'net' else 'device_not_found',
        )

    if not hasattr(obj, attr):
        return ResolveResult(resolved=False, reason='attr_not_found')

    value = getattr(obj, attr)
    if value is None:
        return ResolveResult(resolved=False, reason='value_is_none')

    try:
        return ResolveResult(resolved=True, value=float(value))
    except (TypeError, ValueError):
        return ResolveResult(resolved=False, reason='value_not_numeric')
