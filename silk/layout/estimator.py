"""silk.layout.estimator — LayoutEstimate dataclasses, compute_scalars, estimate_layout.

Public API (consumed by silk/layout/__init__.py):
    LayoutEstimate, NetLayout, DeviceLayout, LayoutWarning (re-exported from _types)
    estimate_layout, capture_layout
    compute_scalars (also used by tests directly)
    default_escalation_rule (overridable layer-escalation default)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, Optional, Tuple, TYPE_CHECKING

from silk.layout._types import LayoutWarning  # re-export
from silk.layout.area_model import DEFAULT_PACKING_FACTOR, DEFAULT_ROUTING_OVERHEAD
from silk.layout.routing_model import RoutingModel

if TYPE_CHECKING:
    from silk.circuit.modules import Module
    from silk.layout.routing_model import RoutingModelOrStack
    from silk.layout.matching import _CanonicalInputs


# ---------------------------------------------------------------------------
# Sub-dataclasses
# ---------------------------------------------------------------------------

# ``NetLayout.c`` (and ``RoutingModel.cap_per_um2``) are femtofarads;
# ``silk.elements.passives.Capacitor`` takes farads. The one fF->F conversion
# site in silk (AUDIT-02 row 27) — exposed as ``NetLayout.c_farads`` so
# ``silk/layout/parasitics.py`` no longer converts fF itself (TASK-25).
_FF_TO_F = 1e-15


@dataclass(frozen=True)
class NetLayout:
    """Immutable per-net layout scalars for one estimation run.

    Fields
    ------
    length : float
        Statistical wire length in μm.
    width : float
        Single-wire width in μm from ``peak_current / j_max``, clamped to
        ``[min_width, max_width]`` of the selected layer. When the
        current-based requirement exceeds ``max_width`` this is the capped
        value, not a width guaranteed sufficient for the current — see the
        run's ``'required_width_exceeds_max'`` warning.
    r : float
        Estimated total wire resistance in Ω (max pin-to-pin, directionless).
    c : float
        Estimated total wire capacitance in fF.
    equivalence_class : str or None
        Synthetic class_id when this net is in a size-≥2 matched class;
        None for singleton (unmatched) nets.
    """
    length: float
    width:  float
    r:      float
    c:      float
    equivalence_class: Optional[str]

    @property
    def c_farads(self) -> float:
        """``c`` (femtofarads) converted to farads — the unit
        ``silk.elements.passives.Capacitor`` takes. The sole fF->F accessor;
        callers should read this instead of multiplying ``c`` themselves."""
        return self.c * _FF_TO_F


@dataclass(frozen=True)
class DeviceLayout:
    """Immutable per-device layout scalars for one estimation run."""
    area: float   # μm²


@dataclass(frozen=True)
class LayoutEstimate:
    """Immutable snapshot of a layout estimation run.

    Returned by both ``estimate_layout`` (after computation) and
    ``capture_layout`` (reading current Net/device/Module state without
    recomputing). Holds the complete per-object numbers plus run metadata
    and warnings.

    Fields
    ------
    computation_mode : str
        ``'statistical'`` (Phase 1) | ``'single_placement'`` | ``'best_of_n'`` | ``'captured'``.
    routing_model_id : str
        Stable id of the RoutingModel used (from RoutingModel.id).
    timestamp : float
        Seconds since epoch when this estimate was produced.
    nets : dict
        ``{net_name: NetLayout}`` for every net in the module.
    devices : dict
        ``{instance_name: DeviceLayout}`` for every device in elem_dict.
    bounding_box : (float, float)
        Module bounding box (width_μm, height_μm).
    equivalence_classes : dict
        ``{class_id: (net_name, ...)}`` — only size-≥2 classes (matched pairs).
    warnings : tuple of LayoutWarning
        Run-scoped warnings. Empty tuple when there are no issues.
    packing_factor : float or None
        The local device packing heuristic this run used (default 2.5):
        devices with channel geometry use ``W·L·M × packing_factor``.
        Existing child footprints and geometry-less placeholders are not
        scaled. Provenance for reports; ``None`` for a captured estimate.
    child_footprints : dict
        Direct child instance names mapped to ``(area_um2, source)``; source
        is ``estimated``, ``fixed``, ``reused`` or ``placeholder``. In-memory
        provenance only; captured estimates default to an empty dictionary.
    routing_overhead : float or None
        The module-level routing multiplier this run used (default 2.0);
        ``None`` for a captured estimate.
    """
    computation_mode: str
    routing_model_id: str
    timestamp: float
    nets: Dict[str, NetLayout]
    devices: Dict[str, DeviceLayout]
    bounding_box: Tuple[float, float]
    equivalence_classes: Dict[str, Tuple[str, ...]]
    warnings: Tuple[LayoutWarning, ...]
    packing_factor: Optional[float] = None
    routing_overhead: Optional[float] = None
    child_footprints: Dict[str, Tuple[float, str]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# layer-escalation stack (TASK-25) — normalization, default rule, ids
# ---------------------------------------------------------------------------

def _stack_layers(routing_model: 'RoutingModelOrStack') -> Tuple[RoutingModel, ...]:
    """Normalize the ``routing_model`` argument to an ordered tuple of
    ``RoutingModel``, lowest layer first.

    Accepts a single ``RoutingModel`` (Phase-1 shape, unchanged), a
    ``dict[str, RoutingModel]`` (e.g. from ``RoutingModel.stack_from_pdk``,
    insertion order = escalation order), or a plain sequence of
    ``RoutingModel`` in escalation order.
    """
    if isinstance(routing_model, RoutingModel):
        return (routing_model,)
    if isinstance(routing_model, dict):
        layers = tuple(routing_model.values())
    else:
        layers = tuple(routing_model)
    if not layers:
        raise ValueError('routing_model: empty stack — need at least one layer')
    for layer in layers:
        if not isinstance(layer, RoutingModel):
            raise TypeError(
                f'routing_model: every stack entry must be a RoutingModel, '
                f'got {type(layer).__name__}')
    return layers


def _routing_model_id(routing_model: 'RoutingModelOrStack') -> str:
    """Stable id for provenance on ``LayoutEstimate.routing_model_id`` —
    a single layer's own ``.id`` unchanged, or each stack layer's id joined
    with ``;`` in escalation order for a multi-layer stack."""
    if isinstance(routing_model, RoutingModel):
        return routing_model.id
    return ';'.join(layer.id for layer in _stack_layers(routing_model))


def default_escalation_rule(
    peak_current: Optional[float],
    stack: Tuple[RoutingModel, ...],
) -> RoutingModel:
    """Default layer-escalation rule: skip a layer only when even its own
    minimum-width wire would violate its own EM limit.

    A layer is skipped in favour of the next one when ``peak_current``
    exceeds what that layer's ``min_width`` wire can carry at ``j_max``
    (``min_width * j_max``). The last layer in *stack*
    is always the eventual fallback (its own ``max_width`` clamp in
    ``compute_scalars`` behaves exactly as Phase 1 did for a single-layer
    stack, escalated or not).

    ``peak_current is None`` (a partial net) returns the lowest (first)
    layer, unescalated — matching ``compute_scalars``'s own min-width
    fallback for a net with no current annotation.
    """
    if peak_current is None or len(stack) == 1:
        return stack[0]
    current = abs(peak_current)
    for layer in stack[:-1]:
        if current <= layer.min_width * layer.j_max:
            return layer
    return stack[-1]


# ---------------------------------------------------------------------------
# compute_scalars
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _SizingResult:
    """Internal: everything ``compute_scalars`` computes, plus the chosen
    layer and pre-clamp required width — needed by ``estimate_layout`` to
    build a width-cap warning without re-running ``escalation_rule`` or
    re-implementing ``W = I / j_max``."""
    width: float
    r: float
    c: float
    layer: RoutingModel
    required_width: Optional[float]   # pre-clamp W=I/j_max; None if no current data


def _size_class(
    canon: '_CanonicalInputs',
    routing_model: 'RoutingModelOrStack',
    escalation_rule,
) -> _SizingResult:
    """Shared arithmetic behind ``compute_scalars`` and ``estimate_layout``'s
    width-cap warning — the single place layer selection and W=I/j_max live."""
    if isinstance(routing_model, RoutingModel):
        layer = routing_model
    else:
        stack = _stack_layers(routing_model)
        layer = escalation_rule(canon.peak_current, stack)

    if canon.peak_current is not None:
        required_width = canon.peak_current / layer.j_max
        width = max(layer.min_width, min(layer.max_width, required_width))
    else:
        required_width = None
        width = layer.min_width

    r = canon.length * layer.sheet_resistance / width
    c = canon.length * width * layer.cap_per_um2

    return _SizingResult(width=width, r=r, c=c, layer=layer, required_width=required_width)


def compute_scalars(
    canon: '_CanonicalInputs',
    routing_model: 'RoutingModelOrStack',
    *,
    escalation_rule=default_escalation_rule,
) -> Tuple[float, float, float]:
    """Compute (width, r, c) from canonical inputs and a RoutingModel.

    Parameters
    ----------
    canon : _CanonicalInputs
        Aggregated inputs for one equivalence class (from matching.py).
    routing_model : RoutingModel or sequence of RoutingModel
        Metal-stack constants for one layer (Phase-1 shape, unchanged), or
        an ordered stack (dict[name, RoutingModel] or a plain sequence,
        lowest layer first) to escalate across — see ``escalation_rule``.
    escalation_rule : callable, optional
        ``(peak_current, stack_of_RoutingModel) -> RoutingModel``. Only
        called when *routing_model* is a multi-layer stack; a single
        ``RoutingModel`` always uses that layer directly (identical
        arithmetic to Previously, — no behaviour change for existing
        callers). Default: ``default_escalation_rule``.

    Returns
    -------
    (width_um, r_ohm, c_fF)
        Priced on whichever layer was chosen (the escalated layer, for a
        net that triggered escalation). When the current-based requirement
        exceeds the chosen layer's ``max_width``, ``width`` is the capped
        single-wire value — see ``estimate_layout``'s
        ``'required_width_exceeds_max'`` warning for that condition.
    """
    result = _size_class(canon, routing_model, escalation_rule)
    return result.width, result.r, result.c


# ---------------------------------------------------------------------------
# estimate_layout orchestrator
# ---------------------------------------------------------------------------

def _stage_footprints(module, packing_factor, routing_overhead, pdk, children,
                      warnings):
    """Plan footprints by object identity without modifying the circuit graph."""
    from silk.circuit.ports import ElementType
    from silk.exceptions import CircuitError
    from silk.layout.area_model import (
        _bounding_box_from_areas, _validate_bounding_box, _get_wlm_um,
        _check_positive_finite,
        compute_entry_area)

    active = set()
    completed = {}
    writes = []

    def visit(mod, path, local_pdk, *, root=False):
        key = id(mod)
        if key in active:
            raise CircuitError(f'Layout child module cycle at {path}')
        if key in completed:
            return completed[key]
        active.add(key)
        fixed = getattr(mod, '_layout_footprint_fixed', False)
        if fixed or (not root and getattr(mod, 'bounding_box', None) is not None):
            _validate_bounding_box(mod.bounding_box)
        # A fixed descendant is opaque. A fixed root still sizes its own nets
        # and records the areas of its direct entries.
        if fixed and not root:
            result = (mod.bounding_box, {}, {}, True)
        else:
            areas = {}
            provenance = {}
            has_geometry = False
            for name, entry in mod.elem_dict.items():
                dev = entry['device']
                if entry.get('type') == ElementType.MODULE:
                    child_path = f'{path}.{name}'
                    child_fixed = getattr(dev, '_layout_footprint_fixed', False)
                    if children == 'estimate':
                        box, _, _, geometry = visit(
                            dev, child_path, getattr(dev, 'pdk_details', None))
                        area = _validate_bounding_box(box)[1]
                        source = 'fixed' if child_fixed else 'estimated'
                        if not geometry:
                            warnings.append(LayoutWarning(
                                net_name=None, reason='child_has_no_geometry',
                                severity='warn', detail=f"Child '{child_path}' has no device geometry."))
                        has_geometry |= geometry
                    else:
                        area = compute_entry_area(entry, packing_factor, pdk=local_pdk)
                        source = ('fixed' if child_fixed else 'placeholder'
                                  if getattr(dev, 'bounding_box', None) is None else 'reused')
                        if source == 'placeholder':
                            warnings.append(LayoutWarning(
                                net_name=None, reason='child_footprint_missing',
                                severity='warn', detail=f"Child '{child_path}' has no footprint; using 1 µm² placeholder."))
                    provenance[name] = (area, source)
                else:
                    area = compute_entry_area(entry, packing_factor, pdk=local_pdk)
                    if children == 'estimate':
                        has_geometry |= _get_wlm_um(dev, local_pdk)[0] is not None
                areas[name] = _check_positive_finite(f'Entry {path}.{name} area', area)
            box = (mod.bounding_box if fixed else
                   _bounding_box_from_areas(areas.values(), routing_overhead))
            result = (box, areas, provenance, has_geometry or fixed)
        active.remove(key)
        completed[key] = result
        if not root:
            writes.append((mod, result[0]))
        return result

    result = visit(module, getattr(module, 'module_name', None) or '<root>', pdk, root=True)
    return result, writes


def estimate_layout(
    module: 'Module',
    routing_model: 'RoutingModelOrStack',
    pdk=None,
    *,
    packing_factor: float = DEFAULT_PACKING_FACTOR,
    routing_overhead: float = DEFAULT_ROUTING_OVERHEAD,
    escalation_rule=default_escalation_rule,
    children: str = 'reuse',
) -> LayoutEstimate:
    """Populate per-net / per-device / per-module layout estimates.

    Mutates ``module`` (writes to Net, AnalogElement, and Module fields) and
    returns an immutable ``LayoutEstimate`` describing the run.

    Phase 1 runs in 'statistical' mode: no placement input, statistical
    wirelength heuristics, lumped-π parasitic model.

    Child modules reuse stored boxes by default (``children='reuse'``), with
    a warning for each missing 1 µm² placeholder. ``children='estimate'``
    computes descendant footprints bottom-up, once per module object, using
    each child's PDK and this call's heuristics. Child nets are untouched.
    Insertion copies children; traversal operates on the actual entries in
    ``module.elem_dict``, not on the original templates. Fixed footprints
    set by ``set_footprint`` are preserved; fixed children are opaque leaves.
    Cycles raise CircuitError with the instance path. All computation and
    validation finish before any estimates are written.

    Parameters
    ----------
    module : Module
        The module to estimate. Must have elem_dict and net_dict populated.
    routing_model : RoutingModel or sequence of RoutingModel
        Metal-stack constants for wire sizing: either one layer (Phase-1
        shape — construct directly, or ``RoutingModel.from_pdk(pdk)``) or an
        ordered multi-layer stack to escalate across (``dict[name,
        RoutingModel]`` — e.g. ``RoutingModel.stack_from_pdk(pdk)`` — or a
        plain sequence, lowest layer first). See ``escalation_rule``.
    pdk : BasePDK, optional
        The PDK whose unit declarations resolve device geometry. Defaults to
        ``module.pdk_details``. Device W/L are read in canonical micrometres
        through :func:`silk.pdk.basepdk.device_geometry_um <silk.pdk.device_geometry_um>` — the PDK layer
        is the only place units are interpreted.
    packing_factor : float, optional
        Per-device footprint heuristic: ``area = W·L·M × packing_factor``.
        Default 2.5 (``silk.layout.area_model.DEFAULT_PACKING_FACTOR`` — also
        the value ``inspect.signature`` reports for this parameter). An
        adjustable Phase-1 heuristic, NOT a PDK design rule. Must be finite
        and positive. Passing ``None`` explicitly is equivalent to omitting
        the argument (both resolve to the default) — kept for callers that
        forward an optional override without a branch. The same value feeds
        local device areas and the module bounding box, and is recorded on
        the returned estimate. Existing child footprints are not rescaled.
    routing_overhead : float, optional
        Module-level multiplier over summed local device and child-block
        footprints for routing channels and isolation, additional to overhead
        already included in child boxes. Default 2.0 (also
        ``inspect.signature``'s reported default). Same validation,
        ``None``-means-default, and provenance rules as ``packing_factor``.
    children : {'reuse', 'estimate'}
        Reuse stored child footprints (default), or estimate descendant
        footprints without touching child nets. See the hierarchy rules above.
    escalation_rule : callable, optional
        ``(peak_current, stack_of_RoutingModel) -> RoutingModel``. Only
        consulted when *routing_model* is a multi-layer stack; a
        single ``RoutingModel`` always prices every net on that one layer,
        identically to before this parameter existed. Default:
        ``default_escalation_rule`` — escalate past a layer only when
        ``peak_current`` would violate EM even at that layer's own
        ``min_width``.

    Returns
    -------
    LayoutEstimate
        Immutable snapshot of the run. Warnings live here, not on Net.

    Examples
    --------
    >>> from tempfile import TemporaryDirectory
    >>> from silk import Module, load_pdk
    >>> folder = TemporaryDirectory()
    >>> pdk = load_pdk('sky130')
    >>> dut = Module('sky130', folder.name, 'demo', pdk)
    >>> from silk import RoutingModel, estimate_layout
    >>> from silk.elements import elements
    >>> dut.add(elements.resistor('r1', p='out', n='gnd', value=1000))
    >>> metal = RoutingModel(sheet_resistance=0.1, cap_per_um2=0.2, j_max=0.01, min_width=0.2, max_width=10)
    >>> estimate = estimate_layout(dut, metal)
    >>> estimate is not None
    True
    """
    from silk.layout.area_model import (
        _check_positive_finite)
    from silk.layout.matching import build_equivalence_classes, compute_canonical_inputs

    if children not in ('reuse', 'estimate'):
        raise ValueError("children must be 'reuse' or 'estimate'")
    packing_factor = _check_positive_finite(
        'packing_factor',
        DEFAULT_PACKING_FACTOR if packing_factor is None else packing_factor)
    routing_overhead = _check_positive_finite(
        'routing_overhead',
        DEFAULT_ROUTING_OVERHEAD if routing_overhead is None else routing_overhead)
    if pdk is None:
        pdk = getattr(module, 'pdk_details', None)

    warnings_list: list = []

    (bbox, areas, child_footprints, _), footprint_writes = _stage_footprints(
        module, packing_factor, routing_overhead, pdk, children, warnings_list)
    devices = {name: DeviceLayout(area=area) for name, area in areas.items()}
    # Net length computation needs only the planned box, not a live mutation.
    sizing_module = SimpleNamespace(bounding_box=bbox)

    # 2. Net-equivalence classes (total over net_dict)
    classes = build_equivalence_classes(module, warnings_list)

    # 3. Per-class computation and write-back
    nets: Dict[str, NetLayout] = {}
    for class_id, members in classes.items():
        net_objects = [module.net_dict[n] for n in members if n in module.net_dict]
        if not net_objects:
            continue

        canon = compute_canonical_inputs(net_objects, sizing_module, warnings_list)
        sizing = _size_class(canon, routing_model, escalation_rule)
        width, r, c = sizing.width, sizing.r, sizing.c

        if sizing.required_width is not None and sizing.required_width > sizing.layer.max_width:
            class_note = (
                f' Class-derived maximum current across the '
                f'{len(net_objects)} matched members of this equivalence '
                f'class — not every member was independently measured at '
                f'this level.' if len(net_objects) > 1 else ''
            )
            for net in net_objects:
                warnings_list.append(LayoutWarning(
                    net_name=net.name,
                    reason='required_width_exceeds_max',
                    severity='warn',
                    detail=(
                        f"Net '{net.name}': at {canon.peak_current:.4g} A, the "
                        f"selected layer's current rule (j_max="
                        f"{sizing.layer.j_max:.4g} A/um) requires "
                        f"{sizing.required_width:.4g} um width; the single-wire "
                        f"estimate is capped at {width:.4g} um (max_width). "
                        f"The width/R/C on this estimate still describe only "
                        f"that capped single-wire approximation. Parallel "
                        f"wires or another routing arrangement may satisfy the "
                        f"requirement, but this estimate does not model or "
                        f"verify that solution.{class_note}"
                    ),
                ))

        for net in net_objects:
            nets[net.name] = NetLayout(
                length=canon.length,
                width=width,
                r=r,
                c=c,
                equivalence_class=class_id if len(members) > 1 else None,
            )

    # Build equivalence_classes dict: only size->=2 classes, sorted member tuples
    eq_classes: Dict[str, Tuple[str, ...]] = {
        class_id: tuple(sorted(members))
        for class_id, members in classes.items()
        if len(members) >= 2
    }

    estimate = LayoutEstimate(
        computation_mode='statistical',
        routing_model_id=_routing_model_id(routing_model),
        timestamp=time.time(),
        nets=nets,
        devices=devices,
        bounding_box=bbox,
        equivalence_classes=eq_classes,
        warnings=tuple(warnings_list),
        packing_factor=packing_factor,
        routing_overhead=routing_overhead,
        child_footprints=child_footprints,
    )

    # Commit only after footprint, matching, sizing and metadata validation.
    for child, box in footprint_writes:
        child.bounding_box = box
        child.estimated_area = box[0] * box[1]
    for name, area in areas.items():
        module.elem_dict[name]['device'].estimated_area = area
    module.bounding_box = bbox
    for name, layout in nets.items():
        net = module.net_dict[name]
        net.estimated_length = layout.length
        net.estimated_width = layout.width
        net.estimated_r = layout.r
        net.estimated_c = layout.c
    return estimate


# ---------------------------------------------------------------------------
# capture_layout
# ---------------------------------------------------------------------------

def capture_layout(module: 'Module') -> LayoutEstimate:
    """Snapshot the current per-net / per-device / per-module layout state.

    Reads the fields that ``estimate_layout`` populates — without running
    any computation — and builds an immutable ``LayoutEstimate``. Use this
    to diff two runs, persist state, or preserve a baseline before a second
    estimation pass.

    Returns an estimate with ``computation_mode='captured'`` and an empty
    warnings tuple — warnings belong to the run that populated the state,
    not to a read of current state.

    Parameters
    ----------
    module : Module
        The module whose current state to snapshot.

    Returns
    -------
    LayoutEstimate
        Immutable snapshot of what is currently on the Net / device / Module objects.
    """
    nets: Dict[str, NetLayout] = {}
    for net_name, net in module.net_dict.items():
        if net_name.startswith('__par_'):
            continue   # skip internal parasitic midpoint nets
        if net.estimated_length is not None:
            nets[net_name] = NetLayout(
                length=net.estimated_length,
                width=net.estimated_width or 0.0,
                r=net.estimated_r or 0.0,
                c=net.estimated_c or 0.0,
                equivalence_class=None,   # not reconstructed from state in Phase 1
            )

    devices: Dict[str, DeviceLayout] = {}
    for inst_name, entry in module.elem_dict.items():
        dev = entry['device']
        if dev.estimated_area is not None:
            devices[inst_name] = DeviceLayout(area=dev.estimated_area)

    bbox = module.bounding_box or (0.0, 0.0)

    return LayoutEstimate(
        computation_mode='captured',
        routing_model_id='',
        timestamp=time.time(),
        nets=nets,
        devices=devices,
        bounding_box=bbox,
        equivalence_classes={},
        warnings=(),
    )
