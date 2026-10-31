"""Ergonomic constructor functions for the fault types.

Each helper has two call forms:

* ``stuck_open(handle_or_path)`` (keyword options allowed) returns a
  :class:`silk.faults.Fault` for the named device.
* ``stuck_open()`` with no device returns a :class:`silk.faults.FaultFactory`,
  a callable that makes a ``Fault`` from a device path. It is what
  :meth:`silk.faults.FaultSet.sprinkle` takes to apply one fault template to
  many devices.
"""

from __future__ import annotations

from typing import Any, Union

from silk.faults.types import Fault


def _resolve_target(handle_or_path: Any) -> str:
    """Accept either a device handle (with ``_silk_fault_target_path`` attr)
    or a path string. Returns the hierarchical path."""
    if isinstance(handle_or_path, str):
        return handle_or_path
    path = getattr(handle_or_path, '_silk_fault_target_path', None)
    if path is not None:
        return path
    raise TypeError(
        f'Expected a device handle or path string, got '
        f'{type(handle_or_path).__name__}. Device handles must expose a '
        f'_silk_fault_target_path attribute (SubcircuitInstance does).'
    )


class FaultFactory:
    """A fault template: calling it with a device makes a concrete fault.

    Returned by the fault helpers (:func:`silk.faults.stuck_open`, ...) when
    called without a device. Calling the factory with a device path string or
    a device handle returns a :class:`silk.faults.Fault` with a copy of the
    template's parameters. :meth:`silk.faults.FaultSet.sprinkle` calls it for
    each matched device.

    Parameters
    ----------
    fault_type : str
        Fault-type name the produced faults carry.
    params : dict
        Parameters copied into each produced fault.

    Attributes
    ----------
    fault_type : str
        Fault-type name the produced faults carry.
    params : dict
        Copy of the parameters given at construction.

    Examples
    --------
    >>> from silk.faults import w_variation
    >>> template = w_variation(sigma=0.02)
    >>> template('xcomp.mn1').fingerprint()
    'w_variation@xcomp.mn1:sigma=0.02'
    """
    def __init__(self, fault_type: str, params: dict):
        self.fault_type = fault_type
        self.params = dict(params)

    def __call__(self, handle_or_path: Any) -> Fault:
        """Make a :class:`silk.faults.Fault` for one device.

        Parameters
        ----------
        handle_or_path : str or object
            Hierarchical device path, or a device handle exposing a
            ``_silk_fault_target_path`` attribute (a ``SubcircuitInstance``
            does).

        Returns
        -------
        Fault
            A fault with this factory's type and a copy of its parameters.

        Raises
        ------
        TypeError
            The argument is neither a string nor a handle with a
            ``_silk_fault_target_path``.
        """
        return Fault(
            target=_resolve_target(handle_or_path),
            fault_type=self.fault_type,
            params=dict(self.params),
        )


def _sugar_hard(fault_type: str, handle_or_path: Any = None,
                *, per_finger: bool = True) -> Union[Fault, FaultFactory]:
    """Shared body for stuck_open / short_ds / short_gs / short_gd."""
    params = {'per_finger': per_finger}
    if handle_or_path is None:
        return FaultFactory(fault_type, params)
    return Fault(
        target=_resolve_target(handle_or_path),
        fault_type=fault_type,
        params=params,
    )


def stuck_open(handle_or_path: Any = None, *, per_finger: bool = True):
    """Open-circuit a transistor, or one finger of a multi-finger device.

    Realized as deleting the device when ``per_finger`` is false or the
    device has multiplier ``m == 1``; otherwise the device is kept with
    ``m`` reduced by one.

    Parameters
    ----------
    handle_or_path : str or object, optional
        Device path (e.g. ``'xcomp.mn_tail'``) or device handle. Omit it to
        get a :class:`silk.faults.FaultFactory` for
        :meth:`silk.faults.FaultSet.sprinkle`.
    per_finger : bool, default True
        Fault one finger only (``m`` to ``m - 1``) rather than the whole
        device. Keyword-only.

    Returns
    -------
    Fault or FaultFactory
        A :class:`silk.faults.Fault` when a device is given, else a factory.

    Raises
    ------
    TypeError
        ``handle_or_path`` is neither a string nor a handle.

    Examples
    --------
    >>> from silk.faults import stuck_open
    >>> stuck_open('xcomp.mn_tail').fingerprint()
    'stuck_open@xcomp.mn_tail:per_finger=True'
    """
    return _sugar_hard('stuck_open', handle_or_path, per_finger=per_finger)


def short_ds(handle_or_path: Any = None, *, per_finger: bool = True):
    """Short a transistor's drain to its source.

    The shunt is an ideal 1 micro-ohm resistor between the two terminals.
    With ``per_finger`` true and ``m > 1`` the device is split into ``m - 1``
    good fingers plus one shorted finger; otherwise the whole device is
    shunted.

    Parameters
    ----------
    handle_or_path : str or object, optional
        Device path or device handle. Omit it to get a
        :class:`silk.faults.FaultFactory` for
        :meth:`silk.faults.FaultSet.sprinkle`.
    per_finger : bool, default True
        Fault one finger only, rather than the whole device. Keyword-only.

    Returns
    -------
    Fault or FaultFactory
        A :class:`silk.faults.Fault` when a device is given, else a factory.

    Raises
    ------
    TypeError
        ``handle_or_path`` is neither a string nor a handle.

    Examples
    --------
    >>> from silk.faults import short_ds
    >>> short_ds('xcomp.mn1', per_finger=False).params
    {'per_finger': False}
    """
    return _sugar_hard('short_ds', handle_or_path, per_finger=per_finger)


def short_gs(handle_or_path: Any = None, *, per_finger: bool = True):
    """Short a transistor's gate to its source.

    The shunt is an ideal 1 micro-ohm resistor between the two terminals.
    With ``per_finger`` true and ``m > 1`` the device is split into ``m - 1``
    good fingers plus one shorted finger; otherwise the whole device is
    shunted.

    Parameters
    ----------
    handle_or_path : str or object, optional
        Device path or device handle. Omit it to get a
        :class:`silk.faults.FaultFactory` for
        :meth:`silk.faults.FaultSet.sprinkle`.
    per_finger : bool, default True
        Fault one finger only, rather than the whole device. Keyword-only.

    Returns
    -------
    Fault or FaultFactory
        A :class:`silk.faults.Fault` when a device is given, else a factory.

    Raises
    ------
    TypeError
        ``handle_or_path`` is neither a string nor a handle.

    Examples
    --------
    >>> from silk.faults import short_gs
    >>> short_gs('xcomp.mn1', per_finger=False).params
    {'per_finger': False}
    """
    return _sugar_hard('short_gs', handle_or_path, per_finger=per_finger)


def short_gd(handle_or_path: Any = None, *, per_finger: bool = True):
    """Short a transistor's gate to its drain.

    The shunt is an ideal 1 micro-ohm resistor between the two terminals.
    With ``per_finger`` true and ``m > 1`` the device is split into ``m - 1``
    good fingers plus one shorted finger; otherwise the whole device is
    shunted.

    Parameters
    ----------
    handle_or_path : str or object, optional
        Device path or device handle. Omit it to get a
        :class:`silk.faults.FaultFactory` for
        :meth:`silk.faults.FaultSet.sprinkle`.
    per_finger : bool, default True
        Fault one finger only, rather than the whole device. Keyword-only.

    Returns
    -------
    Fault or FaultFactory
        A :class:`silk.faults.Fault` when a device is given, else a factory.

    Raises
    ------
    TypeError
        ``handle_or_path`` is neither a string nor a handle.

    Examples
    --------
    >>> from silk.faults import short_gd
    >>> short_gd('xcomp.mn1', per_finger=False).params
    {'per_finger': False}
    """
    return _sugar_hard('short_gd', handle_or_path, per_finger=per_finger)


def w_variation(handle_or_path: Any = None, *, sigma: float):
    """Perturb a transistor's width by a Gaussian random factor.

    The realized width is ``W * (1 + sigma * N(0, 1))``, clamped, quantized
    or snapped according to the device's ``fault_params`` in the loaded PDK.
    If no schema is available, a ``UserWarning`` is issued and the
    perturbation is applied unclamped.

    Parameters
    ----------
    handle_or_path : str or object, optional
        Device path or device handle. Omit it to get a
        :class:`silk.faults.FaultFactory` for
        :meth:`silk.faults.FaultSet.sprinkle`.
    sigma : float
        Standard deviation as a fraction of the nominal width (``0.02`` is
        2 %). Required, keyword-only.

    Returns
    -------
    Fault or FaultFactory
        A :class:`silk.faults.Fault` when a device is given, else a factory.

    Raises
    ------
    TypeError
        ``sigma`` is missing, or ``handle_or_path`` is neither a string nor
        a handle.

    Examples
    --------
    >>> from silk.faults import w_variation
    >>> w_variation('xcomp.mn1', sigma=0.02).params
    {'sigma': 0.02}
    """
    params = {'sigma': float(sigma)}
    if handle_or_path is None:
        return FaultFactory('w_variation', params)
    return Fault(
        target=_resolve_target(handle_or_path),
        fault_type='w_variation',
        params=params,
    )


def l_variation(handle_or_path: Any = None, *, sigma: float):
    """Perturb a transistor's length by a Gaussian random factor.

    The realized length is ``L * (1 + sigma * N(0, 1))``, clamped, quantized
    or snapped according to the device's ``fault_params`` in the loaded PDK.
    If no schema is available, a ``UserWarning`` is issued and the
    perturbation is applied unclamped.

    Parameters
    ----------
    handle_or_path : str or object, optional
        Device path or device handle. Omit it to get a
        :class:`silk.faults.FaultFactory` for
        :meth:`silk.faults.FaultSet.sprinkle`.
    sigma : float
        Standard deviation as a fraction of the nominal length. Required,
        keyword-only.

    Returns
    -------
    Fault or FaultFactory
        A :class:`silk.faults.Fault` when a device is given, else a factory.

    Raises
    ------
    TypeError
        ``sigma`` is missing, or ``handle_or_path`` is neither a string nor
        a handle.

    Examples
    --------
    >>> from silk.faults import l_variation
    >>> l_variation('xcomp.mn1', sigma=0.03).params
    {'sigma': 0.03}
    """
    params = {'sigma': float(sigma)}
    if handle_or_path is None:
        return FaultFactory('l_variation', params)
    return Fault(
        target=_resolve_target(handle_or_path),
        fault_type='l_variation',
        params=params,
    )


def vth_shift(handle_or_path: Any = None, *, delta: float, sigma: float = 0.0):
    """Shift a transistor's gate voltage by a fixed offset.

    Realized as an ideal voltage offset in series with the gate: the gate
    sees ``V(gate) + delta`` volts (plus jitter when ``sigma`` is non-zero).

    Parameters
    ----------
    handle_or_path : str or object, optional
        Device path or device handle. Omit it to get a
        :class:`silk.faults.FaultFactory` for
        :meth:`silk.faults.FaultSet.sprinkle`.
    delta : float
        Gate-voltage offset in volts. Required, keyword-only.
    sigma : float, default 0.0
        Standard deviation, in volts, of Gaussian jitter added to ``delta``
        for each realization. Keyword-only.

    Returns
    -------
    Fault or FaultFactory
        A :class:`silk.faults.Fault` when a device is given, else a factory.

    Raises
    ------
    TypeError
        ``delta`` is missing, or ``handle_or_path`` is neither a string nor
        a handle.

    Examples
    --------
    >>> from silk.faults import vth_shift
    >>> vth_shift('xcomp.mn1', delta=0.03).params
    {'delta': 0.03, 'sigma': 0.0}
    """
    params = {'delta': float(delta), 'sigma': float(sigma)}
    if handle_or_path is None:
        return FaultFactory('vth_shift', params)
    return Fault(
        target=_resolve_target(handle_or_path),
        fault_type='vth_shift',
        params=params,
    )
