"""silk.layout.area_model — device footprint and module bounding-box estimation.

Phase 1: statistical model from device W/L/M parameters and existing child
footprints (no placement, no PDK geometry lookup). A future PDK YAML
``geometry:`` section will replace the packing_factor constant with real
device-specific data.

Units — the contract
--------------------
Every length in this module is in **micrometres** and every area in µm².
This module owns NO unit knowledge of its own: device geometry is read
through ONE accessor, :func:`silk.pdk.basepdk.device_geometry_um`, which the
PDK layer provides and which returns ``(W_um, L_um, M)`` already converted.
The conversion is the same two steps netlist emission trusts — the author's
unit → the PDK's own unit (``user_units`` vs the catalog's per-parameter
``pdk_unit:``, applied only to caller-supplied values), then the PDK's unit
→ µm. So the same circuit source gives the same footprint on a µm-native PDK
(sky130, ``pdk_unit: u``) and a metre-native one (GF180MCU / SG13G2,
``pdk_unit: base``), whether the device was built factory-style
(``pdk.nmos1p8(...)`` inside ``with mod:``) or via ``add_element``.

The accessor is deliberately loud: a W/L whose catalog entry carries no
recognised ``pdk_unit:`` raises rather than being assumed to be µm, and a
device object of an unknown shape raises rather than being read as SI
metres. (Both of those guesses used to live here; the second produced a
1e12 area error for factory-built devices.)

``packing_factor`` and ``routing_overhead`` are **adjustable heuristics**,
not layout truth — see :func:`compute_device_area`.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Set
from numbers import Real
from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from silk.circuit.modules import Module

#: Default physical packing factor for MOSFETs: accounts for source/drain
#: diffusion, contacts, and well spacing around the channel. A Phase-1
#: heuristic (footprint ≈ 2.5 × W·L·M), NOT a PDK design rule — override it
#: per call (``estimate_layout(..., packing_factor=...)``) when you have a
#: better number for your process or device style.
DEFAULT_PACKING_FACTOR = 2.5
#: Default multiplier over summed device and child-block area for routing
#: channels, isolation, and hierarchy gaps. Same status: a tunable heuristic.
DEFAULT_ROUTING_OVERHEAD = 2.0
# Backward-compatible private aliases.
_DEFAULT_PACKING_FACTOR = DEFAULT_PACKING_FACTOR
# Minimum module area floor to avoid zero-area edge cases.
_MIN_MODULE_AREA_UM2 = 1.0
#: Placeholder for a child module without an estimate or a device without
#: channel geometry. Recursive estimation is opt-in through estimate_layout.
_PLACEHOLDER_AREA_UM2 = 1.0


def _check_positive_finite(name: str, value) -> float:
    """Validate a heuristic multiplier: a real, finite, strictly positive number."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise ValueError(f'{name} must be a number, got {value!r}') from None
    if isinstance(value, bool) or not math.isfinite(f) or f <= 0.0:
        raise ValueError(f'{name} must be a finite positive number, got {value!r}')
    return f


def _get_wlm_um(device, pdk=None):
    """Return (W_um, L_um, M) for a device, or (None, None, 1) if it has none.

    A thin adapter over :func:`silk.pdk.basepdk.device_geometry_um` — the
    single unit-aware reader of device geometry. See the module docstring
    for the contract. Passives, ideal elements and devices whose W/L is a
    SPICE literal or an unassigned variable have no footprint here.
    """
    from silk.pdk.basepdk import device_geometry_um
    geometry = device_geometry_um(device, pdk)
    if geometry is None:
        return None, None, 1
    return geometry


def compute_device_area(
    device,
    packing_factor: float = DEFAULT_PACKING_FACTOR,
    *,
    pdk=None,
) -> float:
    """Estimate the physical footprint of one device instance.

    Parameters
    ----------
    device : AnalogElement or PDKDevice
        A device instance from ``module.elem_dict[name]['device']``.
    packing_factor : float
        Multiplier over W×L×M that accounts for diffusion, contacts, and
        isolation spacing. Default 2.5 — an adjustable heuristic, not a
        design rule. Must be finite and positive.
    pdk : BasePDK, optional
        The PDK the device was added from (``module.pdk_details``). Required
        for an ``add_element``-style ``PDKDevice``, whose author→PDK unit
        factor lives on the PDK instance; a factory-built element carries its
        own and ignores this.

    Returns
    -------
    float
        Estimated area in μm². Returns 1.0 for devices without W/L/M params.

    Raises
    ------
    silk.pdk.basepdk.ValidationError
        The device declares a W or L with no recognised ``pdk_unit:``.
    TypeError
        The device is of a shape silk cannot read geometry from, or a
        ``PDKDevice`` was given without its *pdk*.
    """
    packing_factor = _check_positive_finite('packing_factor', packing_factor)
    W_um, L_um, M_val = _get_wlm_um(device, pdk)
    if W_um is None or L_um is None:
        return _PLACEHOLDER_AREA_UM2   # passives / no geometry params
    return W_um * L_um * M_val * packing_factor


def _validate_bounding_box(bbox):
    """Return normalized dimensions and area, using the child-box contract."""
    try:
        if (isinstance(bbox, (str, bytes, bytearray, Mapping, Set))
                or len(bbox) != 2):
            raise ValueError
        width, height = bbox
        if any(isinstance(d, bool) or not isinstance(d, Real)
               for d in (width, height)):
            raise ValueError
        width, height = float(width), float(height)
        area = width * height
        if any(not math.isfinite(v) or v <= 0
               for v in (width, height, area)):
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            f'Child module bounding_box must contain exactly two finite '
            f'positive numeric dimensions in micrometres with a finite '
            f'positive area; got {bbox!r}. Estimate the child before its '
            f'parent or supply a valid (width_um, height_um).') from None
    return (width, height), area


def set_footprint(module, box):
    """Fix a module footprint in µm; ``None`` clears the box and fixed mark.

    Dimensions must be finite positive real numbers with finite positive area.
    Validation completes before either field is changed. The private mark
    survives module insertion copies without changing the Module API.
    """
    if box is not None:
        box, _ = _validate_bounding_box(box)
    module.bounding_box = box
    module._layout_footprint_fixed = box is not None


def _bounding_box_from_areas(areas, routing_overhead):
    total = max(sum(areas) * routing_overhead, _MIN_MODULE_AREA_UM2)
    length = math.sqrt(total)
    box = (length, length)
    _validate_bounding_box(box)
    return box


def compute_entry_area(entry, packing_factor: float = DEFAULT_PACKING_FACTOR,
                       *, pdk=None) -> float:
    """Footprint of one ``module.elem_dict`` entry (µm²).

    A child module contributes its existing ``bounding_box`` width × height,
    already in micrometres, without device packing or PDK unit conversion.
    Only a missing or ``None`` box gets the 1 µm² placeholder. Estimate
    children explicitly before their parent; this function never recomputes
    child state. Insertion copies modules, so estimate before insertion or
    estimate the actual child in the parent's ``elem_dict``.

    Other entries go through :func:`compute_device_area` and its PDK accessor.

    Raises
    ------
    ValueError
        A supplied child box does not contain exactly two finite positive
        real numeric dimensions, or their area is not finite and positive.
    """
    from silk.circuit.ports import ElementType
    if entry.get('type') == ElementType.MODULE:
        bbox = getattr(entry['device'], 'bounding_box', None)
        if bbox is None:
            return _PLACEHOLDER_AREA_UM2
        return _validate_bounding_box(bbox)[1]
    return compute_device_area(entry['device'], packing_factor, pdk=pdk)


def compute_module_bounding_box(
    module,
    packing_factor: float = DEFAULT_PACKING_FACTOR,
    routing_overhead: float = DEFAULT_ROUTING_OVERHEAD,
    *,
    pdk=None,
) -> Tuple[float, float]:
    """Estimate a module's bounding box from local devices and child footprints.

    Iterates ``module.elem_dict``, calls :func:`compute_entry_area` for each
    entry, sums the results, applies routing overhead, then takes the square
    root to get the characteristic length of a square bounding box.

    Writes the result back to ``module.bounding_box`` as a side-effect.

    Parameters
    ----------
    module : Module
        The module to estimate. ``elem_dict`` must be populated.
    packing_factor : float
        Passed to :func:`compute_device_area`. Default 2.5 (heuristic).
    routing_overhead : float
        Multiplier over summed local device and child-block footprints for
        routing channels, isolation, and hierarchy gaps. Applied once here,
        in addition to any overhead already baked into child footprints.
        Default 2.0 (heuristic). Must be finite and positive.
    pdk : BasePDK, optional
        Passed to :func:`compute_device_area`. Defaults to
        ``module.pdk_details`` when the module has one.

    Returns
    -------
    (width_um, height_um) : tuple of float
        Square approximation — width == height == sqrt(total_with_overhead).
        Both values are >= 1.0.
    """
    packing_factor = _check_positive_finite('packing_factor', packing_factor)
    routing_overhead = _check_positive_finite('routing_overhead', routing_overhead)
    if pdk is None:
        pdk = getattr(module, 'pdk_details', None)
    if getattr(module, '_layout_footprint_fixed', False):
        _validate_bounding_box(module.bounding_box)
        return module.bounding_box
    areas = [compute_entry_area(entry, packing_factor, pdk=pdk)
             for entry in module.elem_dict.values()
             if entry.get('device') is not None]
    bbox = _bounding_box_from_areas(areas, routing_overhead)
    module.bounding_box = bbox
    return bbox
