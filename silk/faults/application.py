"""Fault realization: what one fault changes about one device.

A fault realizes to a change in DEVICES, never to SPICE text:

* the target device's instances to emit — none when the fault removes it,
  otherwise each with its own optional name, parameter overrides (in the PDK's
  own unit) and pin rewires;
* silk elements to add next to it (an ideal resistor for a short, a
  voltage-controlled voltage source plus a DC source for a threshold shift),
  described here as data and built as silk objects by the netlister;
* the comment lines that mark the fault in the deck, and the
  :class:`AppliedFault` record for the manifest.

The netlister (``silk/circuit/spice_netlister.py::_emit_faulted_device``)
turns the realization into silk objects and emits every one of them through
the element's own writer, so unit conversion, the shared parameter renderer
and the ammeter wrapper apply exactly as they do in an unfaulted deck. This
module formats no SPICE and imports nothing from silk outside its own
submodules: the device is read duck-typed (``name``, ``nets``, ``params``,
``_caller_supplied``, ``_param_scales``, ``_device_type_name``) — a
factory-built ``AnalogElement`` as it is, a stored ``PDKDevice`` through the
netlister's ``_PDKDeviceView``.

Units. A device's caller-supplied parameters are in the user's units;
emission scales them by the device's ``_param_scales`` into the PDK's own
unit, and a catalog default is already in the PDK's unit. Every override
here is in the PDK's unit, the unit ``fault_params`` bounds are declared in,
so ``w_variation``/``l_variation`` perturb and clamp in one unit system.
The manifest's realized ``w_nominal``/``w_new`` (``l_*``) are reported in
the user's units, the units the circuit was written in.

No side effects. Determinism is the responsibility of the caller via the
provided rng.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from silk.faults.pdk_metadata import (
    load_device_wl_schema,
    perturb_l,
    perturb_w,
)
from silk.faults.types import AppliedFault, Fault

# Short-fault shunt resistance, in ohms. Emitted through the ideal resistor's
# own writer, which renders it as ``1u``.
_SHORT_SHUNT_OHMS = 1e-6


@dataclass(frozen=True)
class _DeviceInstance:
    """One instance of the faulted device to emit.

    ``params`` overrides parameter values in the PDK's own unit; ``nets``
    rewires pins. Everything not named is emitted exactly as in the clean
    deck.
    """
    params: dict = field(default_factory=dict)
    nets: dict = field(default_factory=dict)
    name: str | None = None


@dataclass(frozen=True)
class _AddedElement:
    """A silk element the fault adds next to the target.

    ``kind`` is ``'resistor'`` (nets ``p``/``n``, value in ohms),
    ``'vcvs'`` (nets ``out_pos``/``out_neg``/``ctrl_pos``/``ctrl_neg``,
    value = gain) or ``'vdc'`` (nets ``positive``/``negative``, value =
    DC volts). The netlister builds the matching silk object.
    """
    kind: str
    name: str
    nets: dict
    value: float


@dataclass
class _FaultRealization:
    """What one fault changes about one device. ``instances == []`` means
    the device is removed from the deck."""
    instances: List[_DeviceInstance] = field(default_factory=list)
    added: List[_AddedElement] = field(default_factory=list)
    comments: List[str] = field(default_factory=list)
    applied: List[AppliedFault] = field(default_factory=list)


def _helper_name(role: str, target: str) -> str:
    """An identifier with the complete target encoded without punctuation loss."""
    return f'__silk_fault_{role}_{target.encode("utf-8").hex()}'


def _raw(value):
    """A parameter's number: a ``Variable`` resolves to its value."""
    return getattr(value, 'value', value)


def _scale(device, param: str) -> float:
    """The factor that takes *param* from the user's units to the PDK's
    unit, as emission applies it: caller-supplied values scale, a catalog
    default is already in the PDK's unit."""
    if param not in getattr(device, '_caller_supplied', ()):
        return 1.0
    return getattr(device, '_param_scales', {}).get(param, 1.0)


def _pdk_value(device, param: str) -> float:
    """*param*'s value in the PDK's own unit."""
    return float(_raw(device.params[param])) * _scale(device, param)


def _user_value(device, param: str, pdk_value: float) -> float:
    """*pdk_value* expressed in the units *param* was written in."""
    scale = getattr(device, '_param_scales', {}).get(param, 1.0)
    return pdk_value / scale


def _multiplicity(device) -> int:
    m = device.params.get('m', 1)
    return int(_raw(m) or 1)


def _stuck_open(device, fault: Fault) -> _FaultRealization:
    per_finger = fault.params.get('per_finger', True)
    m = _multiplicity(device)
    target = fault.target
    if not per_finger or m == 1:
        comment = f'*** FAULT: stuck_open on {target} (was m={m}, deleted)'
        return _FaultRealization(
            instances=[],
            comments=[comment],
            applied=[AppliedFault(
                target=target,
                fault_type='stuck_open',
                declared_params=dict(fault.params),
                realized_params={'m_before': m, 'm_after': 0, 'deleted': True},
                netlist_comment=comment,
            )],
        )
    comment = f'*** FAULT: stuck_open on {target} (m {m} -> {m-1})'
    return _FaultRealization(
        instances=[_DeviceInstance(params={'m': m - 1})],
        comments=[comment],
        applied=[AppliedFault(
            target=target,
            fault_type='stuck_open',
            declared_params=dict(fault.params),
            realized_params={'m_before': m, 'm_after': m - 1, 'deleted': False},
            netlist_comment=comment,
        )],
    )


def _short(device, fault: Fault, terminal_a: str,
           terminal_b: str, short_name: str) -> _FaultRealization:
    per_finger = fault.params.get('per_finger', True)
    m = _multiplicity(device)
    net_a = device.nets[terminal_a]
    net_b = device.nets[terminal_b]
    target = fault.target
    shunt = _AddedElement(
        kind='resistor', name=_helper_name('short', target),
        nets={'p': net_a, 'n': net_b}, value=_SHORT_SHUNT_OHMS)
    if not per_finger or m == 1:
        comment = (f'*** FAULT: {short_name} on {target} '
                   f'(whole device, shunt {net_a}-{net_b})')
        return _FaultRealization(
            instances=[_DeviceInstance()],
            added=[shunt],
            comments=[comment],
            applied=[AppliedFault(
                target=target,
                fault_type=short_name,
                declared_params=dict(fault.params),
                realized_params={'m_before': m, 'm_good': m, 'm_bad': 0,
                                 'shunt_ohms': _SHORT_SHUNT_OHMS},
                netlist_comment=comment,
            )],
        )
    comment = (f'*** FAULT: {short_name} on {target} '
               f'(1 of {m} fingers, shunt {net_a}-{net_b})')
    return _FaultRealization(
        instances=[_DeviceInstance(params={'m': m - 1}),
                   _DeviceInstance(name=_helper_name('bad', target), params={'m': 1})],
        added=[shunt],
        comments=[comment],
        applied=[AppliedFault(
            target=target,
            fault_type=short_name,
            declared_params=dict(fault.params),
            realized_params={'m_before': m, 'm_good': m - 1, 'm_bad': 1,
                             'shunt_ohms': _SHORT_SHUNT_OHMS},
            netlist_comment=comment,
        )],
    )


def _short_ds(device, fault: Fault) -> _FaultRealization:
    return _short(device, fault, 'd', 's', 'short_ds')


def _short_gs(device, fault: Fault) -> _FaultRealization:
    return _short(device, fault, 'g', 's', 'short_gs')


def _short_gd(device, fault: Fault) -> _FaultRealization:
    return _short(device, fault, 'g', 'd', 'short_gd')


def _w_or_l_variation(device, fault: Fault, which: str,
                      rng: np.random.Generator, pdk) -> _FaultRealization:
    # Perturb and clamp in the PDK's own unit: the fault_params bounds are
    # declared in it (silk.pdk.basepdk.FaultParams). The schema resolves off
    # THIS pdk's own device list (TASK-24), never a hard-coded catalog.
    nominal_pdk = _pdk_value(device, which)
    sigma = fault.params['sigma']
    schema = load_device_wl_schema(
        pdk, getattr(device, '_device_type_name', ''))
    realized = (perturb_w if which == 'w' else perturb_l)(
        schema, nominal=nominal_pdk, sigma=sigma, rng=rng)
    new_pdk = realized[f'{which}_new']
    # The manifest reports W/L in the user's units, the units the circuit
    # was written in. A caller-supplied nominal is reported as written, not
    # round-tripped through the scale.
    if which in getattr(device, '_caller_supplied', ()):
        realized[f'{which}_nominal'] = float(_raw(device.params[which]))
    else:
        realized[f'{which}_nominal'] = _user_value(device, which, nominal_pdk)
    realized[f'{which}_new'] = _user_value(device, which, new_pdk)
    target = fault.target
    comment = (f'*** FAULT: {which}_variation on {target} '
               f'(declared sigma={sigma*100:.2f}%, '
               f'realized x{realized[f"{which}_multiplier"]:.4f})')
    return _FaultRealization(
        instances=[_DeviceInstance(params={which: new_pdk})],
        comments=[comment],
        applied=[AppliedFault(
            target=target,
            fault_type=f'{which}_variation',
            declared_params=dict(fault.params),
            realized_params=realized,
            netlist_comment=comment,
        )],
    )


def _vth_shift(device, fault: Fault,
               rng: np.random.Generator) -> _FaultRealization:
    orig_gate = device.nets['g']
    target = fault.target
    delta = fault.params['delta']
    sigma = fault.params.get('sigma', 0.0)
    jitter = float(sigma * rng.standard_normal()) if sigma > 0 else 0.0
    delta_total = delta + jitter
    shifted = _helper_name('shifted', target)
    offset = _helper_name('offset', target)
    comment = (f'*** FAULT: vth_shift on {target} '
               f'(declared delta={delta:.4f} V, jitter={jitter:.4f} V)')
    # V(shifted) = V(gate) + delta_total: a unity-gain VCVS senses the gate
    # against ground and sits on a DC source of delta_total, so the shifted
    # gate does not load the original gate net.
    return _FaultRealization(
        instances=[_DeviceInstance(nets={'g': shifted})],
        added=[
            _AddedElement(kind='vcvs', name=_helper_name('vt', target),
                          nets={'out_pos': shifted, 'out_neg': offset,
                                'ctrl_pos': orig_gate, 'ctrl_neg': '0'},
                          value=1),
            _AddedElement(kind='vdc', name=_helper_name('vt', target),
                          nets={'positive': offset, 'negative': '0'},
                          value=delta_total),
        ],
        comments=[comment],
        applied=[AppliedFault(
            target=target,
            fault_type='vth_shift',
            declared_params=dict(fault.params),
            realized_params={
                'delta_declared': delta,
                'delta_jitter': jitter,
                'delta_total': delta_total,
            },
            netlist_comment=comment,
        )],
    )


_DISPATCH = {
    'stuck_open':  lambda d, f, rng, pdk: _stuck_open(d, f),
    'short_ds':    lambda d, f, rng, pdk: _short_ds(d, f),
    'short_gs':    lambda d, f, rng, pdk: _short_gs(d, f),
    'short_gd':    lambda d, f, rng, pdk: _short_gd(d, f),
    'w_variation': lambda d, f, rng, pdk: _w_or_l_variation(d, f, 'w', rng, pdk),
    'l_variation': lambda d, f, rng, pdk: _w_or_l_variation(d, f, 'l', rng, pdk),
    'vth_shift':   lambda d, f, rng, pdk: _vth_shift(d, f, rng),
}


def _realize_fault(device, fault: Fault, rng: np.random.Generator,
                   pdk=None) -> _FaultRealization:
    """Realize *fault* against *device* (an ``AnalogElement`` or a stored
    ``PDKDevice`` view, read duck-typed; never mutated).

    *pdk* — the loaded ``BasePDK`` the emitting module carries, or ``None``
    — is read by ``w_variation`` / ``l_variation`` to resolve
    ``fault_params`` bounds through that PDK's own device list
    (``silk.faults.pdk_metadata.load_device_wl_schema``); every other fault
    type ignores it.
    """
    handler = _DISPATCH.get(fault.fault_type)
    if handler is None:
        raise ValueError(f'Unknown fault_type: {fault.fault_type!r}')
    return handler(device, fault, rng, pdk)
