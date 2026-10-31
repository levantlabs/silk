"""PDK-agnostic W/L perturbation helpers for fault injection.

``perturb_w()`` / ``perturb_l()`` apply a random multiplicative perturbation
to a device's nominal W or L, then clamp/quantize/snap it according to that
device's ``fault_params:`` schema. The schema itself is parsed by
``BasePDK``/``PDKDevice.from_dict`` from the loaded PDK's own device catalog
(``silk.pdk.basepdk.PDKDevice.fault_params``) — this module never opens a
catalog file itself and hardcodes no PDK name (carried over from a
sky130-only catalog hard-code AUDIT-02 found).

``load_device_wl_schema(pdk, device_name)`` reads that schema off the *given*
PDK object's own device list. Because the schema always comes from that one
PDK instance's own devices, a same-named device on a *different* PDK can
never contribute its bounds — there is no global cache keyed by name alone
to collide on.

Perturbation kinds (the schema's ``.w.kind`` / ``.l.kind``):

  continuous -> new = nominal * (1 + sigma*N(0,1)), clamped to [min, max]
  quantized  -> same perturbation, then round to step * integer and clamp
  discrete   -> same perturbation, then snap to nearest allowed value

For a device with no resolvable ``fault_params`` schema (no PDK given, the
device is not one of that PDK's own, or the device declares no
``fault_params:``), ``perturb_w``/``perturb_l`` emit a ``UserWarning`` and
apply continuous perturbation with no clamping — the honest "this PDK
declares no fault bounds for this device" signal, never a silent no-op and
never another device's or another PDK's bounds.

Unit contract: every bound on the schema is in the PDK's own unit, exactly
as the catalog declares it (see ``silk.pdk.basepdk.FaultParams``) — this
module performs no unit conversion.
"""

from __future__ import annotations

import warnings

import numpy as np


def load_device_wl_schema(pdk, device_name: str):
    """Return the ``fault_params`` schema for *device_name* on *pdk*.

    *pdk* is a loaded ``BasePDK`` instance, or ``None``. The returned object
    (or ``None``) is ``silk.pdk.basepdk.PDKDevice.fault_params`` — read
    straight off ``pdk._devices[device_name]``, never opened from a catalog
    file by this function.

    Returns ``None`` when *pdk* is ``None``, *device_name* names none of
    *pdk*'s own devices, or that device declares no ``fault_params:`` in its
    catalog entry. Never falls back to another device's or another PDK's
    schema: a name that happens to collide across two different PDK catalogs
    resolves independently on each, because both catalogs are read through
    their own ``pdk._devices`` mapping and nothing else.
    """
    if pdk is None:
        return None
    device = getattr(pdk, '_devices', {}).get(device_name)
    if device is None:
        return None
    return getattr(device, 'fault_params', None)


def _apply_kind(raw: float, dim) -> tuple[float, bool]:
    if dim.kind == 'continuous':
        lo = dim.min if dim.min is not None else -float('inf')
        hi = dim.max if dim.max is not None else float('inf')
        clamped = raw < lo or raw > hi
        return max(lo, min(hi, raw)), clamped
    if dim.kind == 'quantized':
        step = dim.step
        n = int(round(raw / step))
        lo = dim.min_count if dim.min_count is not None else 1
        hi = dim.max_count if dim.max_count is not None else 10_000_000
        clamped = n < lo or n > hi
        n = max(lo, min(hi, n))
        return n * step, clamped
    if dim.kind == 'discrete':
        allowed = dim.allowed
        nearest = min(allowed, key=lambda a: abs(a - raw))
        span_lo, span_hi = min(allowed), max(allowed)
        clamped = raw < span_lo or raw > span_hi
        return nearest, clamped
    raise ValueError(f'Unknown schema kind: {dim.kind!r}')


def _perturb_dim(schema, which: str,
                 nominal: float, sigma: float,
                 rng: np.random.Generator) -> dict:
    multiplier = 1.0 + sigma * rng.standard_normal()
    raw = nominal * multiplier
    if schema is None:
        warnings.warn(
            f'No W/L schema available; applying continuous {which}-perturbation '
            f'without clamping. This PDK declares no fault_params for this '
            f'device (or none was given) — add fault_params to the PDK '
            f'catalog to get technology-aware behaviour.',
            UserWarning,
            stacklevel=2,
        )
        return {
            f'{which}_nominal': nominal,
            f'{which}_multiplier': float(multiplier),
            f'{which}_new': float(raw),
            'clamped': False,
        }
    dim = schema.w if which == 'w' else schema.l
    new, clamped = _apply_kind(raw, dim)
    return {
        f'{which}_nominal': nominal,
        f'{which}_multiplier': float(multiplier),
        f'{which}_new': float(new),
        'clamped': clamped,
    }


def perturb_w(schema, *, nominal: float, sigma: float,
              rng: np.random.Generator) -> dict:
    return _perturb_dim(schema, 'w', nominal, sigma, rng)


def perturb_l(schema, *, nominal: float, sigma: float,
              rng: np.random.Generator) -> dict:
    return _perturb_dim(schema, 'l', nominal, sigma, rng)
