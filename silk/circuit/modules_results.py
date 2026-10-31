"""Result-loading, signal-access and plotting methods for :class:`~silk.circuit.modules.Module`.

Split out of ``modules.py`` as a per-subsystem **part**. ``_ResultsMethods`` is a pure mixin: no state, no separate identity. Backend/signal/plot libraries are imported lazily inside the methods that need them.
"""

from silk.circuit.emitted_names import emitted_submodule_name
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import os

from silk.exceptions import SimulationError, NotSupportedError

if TYPE_CHECKING:  # pragma: no cover
    from silk.circuit._module_protocol import _ModuleAttrs
    _Base = _ModuleAttrs
else:
    _Base = object


def _source_current_probe_name(module, raw, signalName, signalType, analysis):
    """Translate source cards and enforce AC limits for live and corner reads."""
    from silk.circuit.node_paths import NodeVoltageRequest
    if (isinstance(signalName, NodeVoltageRequest)
            or getattr(signalName, '_literal_voltage', False)):
        return signalName
    if str(signalType).upper() == 'I' or str(signalName).lower().startswith('i('):
        from silk.sources._layout import current_source_for_request, refuse_ac_current
        source = current_source_for_request(module.sig_sources_dict, signalName, module.elem_dict)
        if source is not None:
            from silk.simulation.backend import select_plot
            from silk.signals.analysis_kind import plot_silk_analysis
            plot = select_plot(raw.native, analysis)
            if plot_silk_analysis(plot if plot is not None else raw.native.current_plot) == 'ac':
                refuse_ac_current(source)
    if str(signalType).upper() == 'I' and module.simulator in ('ngspice', 'xyce'):
        from silk.sources._layout import current_source_card
        _, card = current_source_card(module.sig_sources_dict, signalName, module.elem_dict)
        if card is not None:
            if module.simulator == 'ngspice':
                # ngspice's hierarchical independent-current device spelling.
                if '.' in card:
                    card = 'i.' + card
                signalName = f'@{card}[current]'
            else:
                signalName = card.replace('.', ':')
        elif module.simulator == 'ngspice':
            from silk.sources._layout import source_instances, physical_source_card
            from silk.sources.classes import VoltageSource
            for _, source, prefix in source_instances(module.sig_sources_dict, module.elem_dict):
                if not prefix or not isinstance(source, VoltageSource):
                    continue
                voltage_card = physical_source_card(
                    signalName, [prefix + name for name in source.physical_instance_names()])
                if voltage_card is not None:
                    signalName = 'v.' + voltage_card
                    break
    return signalName


def _trapezoid_window(t, y, tstart):
    """Time-weighted (trapezoidal) window statistics for a current waveform.

    Returns ``(avg, rms, peak)`` where ``avg`` is the **signed** time-weighted
    mean over the window, ``rms`` is the time-weighted RMS magnitude, and
    ``peak`` is ``max |i(t)|``. Deliberately *not* the ``silk.metrics`` uniform
    resample convention — ngspice's timesteps are non-uniform, so a sample mean
    would be wrong; trapezoidal integration weights each sample by its interval.

    ``tstart=None`` uses the full trace; otherwise all timepoints with
    ``t >= tstart`` are kept. ``tstart`` at or past the trace end raises
    ``ValueError`` (mirrors ``spectral.py``'s error style).
    """
    import numpy as np
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if tstart is not None:
        if tstart >= t[-1]:
            raise ValueError(
                f"tstart={tstart} is at or past the trace end t={t[-1]}; "
                f"no timepoints remain in the window.")
        mask = t >= tstart
        t = t[mask]
        y = y[mask]
    if len(y) == 0:
        return 0.0, 0.0, 0.0
    peak = float(np.max(np.abs(y)))
    span = float(t[-1] - t[0])
    if span <= 0.0:                      # single-point window
        return float(y[-1]), float(abs(y[-1])), peak
    from silk.utils._compat import trapezoid as trap   # NumPy 2.0 rename shim
    avg = float(trap(y, t) / span)
    rms = float(np.sqrt(trap(y * y, t) / span))
    return avg, rms, peak


@dataclass
class CurrentAnnotationReport:
    """Outcome of :meth:`Module.annotate_currents <silk.circuit.Module.annotate_currents>`.

    Attributes
    ----------
    nets_annotated : list of str
        Instance-qualified net names that received at least one annotated pin.
        Net paths here, in ``skips``, ``partial_nets`` and the keys of
        ``worst_kcl_residual`` use the ``[k]`` spelling of array instances
        (``xx1[0].xxy[1].g[0]``); ``instance_modules`` keys stay the emitted
        SPICE paths (lookup keys) and ``PinBinding.current_vector`` the raw
        rawfile vector name.
    pins_annotated : int
        Total pins populated with SPICE currents.
    skips : list of tuple of str
        Per-pin skips, each ``(net, pin_id, reason)``; ``reason`` is one of ``'no-vector'``, ``'ambiguous-match'``,
        ``'unsupported-kind'``, ``'user-override'``.
    worst_kcl_residual : dict of str to float
        Keyed by net name. For each **fully-annotated** net, the signed ``Σ avg_current`` residual
        (should be ≈ 0). Partially-annotated nets are absent (KCL-exempt).
    partial_nets : list of str
        Nets with at least one skipped non-marker pin (KCL-exempt).
    instance_modules : dict of str to Module
        Keyed by ``instance_path``. The annotated per-instance :class:`Module <silk.circuit.Module>` copies.
    definition_reduced : list of str
        ``module_name`` s whose ``module_dict`` live definition got the
        worst-case-across-instances reduction; KCL-exempt.
    results_file : str
        Provenance: rawfile path.
    analysis : str
        Provenance: analysis type (``'transient'``).
    window : tuple
        Provenance: the ``(tstart,)`` window.
    """
    nets_annotated: List[str] = field(default_factory=list)
    pins_annotated: int = 0
    skips: List[Tuple[str, str, str]] = field(default_factory=list)
    worst_kcl_residual: Dict[str, float] = field(default_factory=dict)
    partial_nets: List[str] = field(default_factory=list)
    instance_modules: Dict[str, object] = field(default_factory=dict)
    definition_reduced: List[str] = field(default_factory=list)
    results_file: str = ''
    analysis: str = ''
    window: Tuple = ()


def _resolve_element(module, binding):
    """Return the concrete element object for a pin binding. Elements live in
    three collections: ``elem_dict`` (devices +
    submodules), ``passives_dict`` (R/C/L), ``sig_sources_dict`` (V/I sources).
    ``binding.kind`` selects the collection."""
    kind = binding.kind
    if kind == 'passive':
        return module.passives_dict.get(binding.instance)
    if kind == 'source':
        return module.sig_sources_dict.get(binding.instance)
    if kind in ('device', 'submodule'):
        entry = module.elem_dict.get(binding.instance)
        return entry.get('device') if isinstance(entry, dict) else entry
    return None


# port -> (suffix, into-net sign) per element kind (R/V/I signs verified
# against ngspice). Two-terminal + sources. Ideal R/C/L come from two
# families with different pin names for the same physical terminals: the
# hand-written silk.elements.passives classes (pos/neg) and the YAML-built
# silk.elements.elements.resistor/capacitor/inductor (pinList [p, n],
# analog_devices.yaml) — both map to the same into-net signs (TASK-308).
_TWO_TERMINAL = {'pos': ('i', -1.0), 'neg': ('i', +1.0),
                 'p': ('i', -1.0), 'n': ('i', +1.0)}
_SOURCE_BRANCH = {'positive': -1.0, 'negative': +1.0}
_MOS_TERMINALS = frozenset({'d', 'g', 's', 'b'})
# Ideal-passive recognition (TASK-308): what the element EMITS, not which
# class built it. R/C/L are the only SPICE prefixes analog_devices.yaml (or
# silk.elements.passives) assigns to a two-terminal ideal passive; MOS/PDK
# devices use 'XM' (or a PDK-specific prefix), never these three.
_IDEAL_PASSIVE_PREFIXES = frozenset({'R', 'C', 'L'})


def _is_ideal_passive(element) -> bool:
    """True for an ideal two-terminal R/C/L, whichever class built it: the
    hand-written ``silk.elements.passives.Resistor/Capacitor/Inductor`` or the
    YAML-driven ``silk.elements.elements.resistor/capacitor/inductor``
    (an ``AnalogElement``). Both carry a ``_spice_prefix`` of ``'R'``/``'C'``/
    ``'L'`` — the former as a class attribute mirroring what ``to_spice()``
    hard-codes, the latter from ``analog_devices.yaml``."""
    return getattr(element, '_spice_prefix', None) in _IDEAL_PASSIVE_PREFIXES


def _entered_instances(module, prefix=''):
    """Yield ``(prefix, instance_module)`` for every submodule instance under
    *module*, depth first. *prefix* is the instance's net prefix in SPICE
    spelling (``'xdut.'``, ``'xdut.xcomp1.'``).

    Behavioural results live on these instances' ``Net.voltage``
    for the nets the engine names hierarchically.
    """
    from silk.circuit.circuit_elements import ElementType
    for name, entry in (module.elem_dict or {}).items():
        if entry.get('type') == ElementType.MODULE:
            here = f'{prefix}{emitted_submodule_name(name)}.'
            yield here, entry['device']
            yield from _entered_instances(entry['device'], here)


def _instance_net(module, path):
    """The :class:`~silk.circuit.net.Net` a hierarchical behavioural name
    (``'xdut.vcomp'``) denotes, or None.

    Each ``x<inst>.`` segment enters that submodule instance; the remainder is
    a net of the innermost one. A port net is not a vector of its own (the
    parent's net is), so a port name resolves to None, as in a SPICE rawfile.
    """
    from silk.circuit.circuit_elements import ElementType
    head, dot, rest = path.partition('.')
    if not dot or not head.startswith('x'):
        return None
    entry = next((entry for name, entry in (module.elem_dict or {}).items()
                  if emitted_submodule_name(name) == head), None)
    if entry is None or entry.get('type') != ElementType.MODULE:
        return None
    sub = entry['device']
    if rest not in (sub.pin_list or {}):
        net = sub.net_dict.get(rest)
        if net is not None:
            return net
    return _instance_net(sub, rest)


def _write_behavioral_voltages(module, arrays):
    """Write one behavioural run's net arrays onto ``Net.voltage``.

    A plain name is a net of *module*; a hierarchical one (``'xdut.vcomp'``, a net inside an entered instance) goes onto that
    instance's Net. Nets are cleared first because live definitions can retain
    earlier waveforms, and a modelled leaf's internal nets must never
    read back a value this run did not compute. Names that are not nets (a
    model's intermediate signal) are skipped. Shared by
    ``simulate_behavioral()`` and the ``'behavioral'`` backend's
    ``simulate()``.
    """
    for _, sub in _entered_instances(module):
        for net in sub.net_dict.values():
            net.voltage = None
    for name, arr in arrays.items():
        net = module.net_dict.get(name)
        if net is None and '.' in name:
            net = _instance_net(module, name)
        if net is not None:
            net.voltage = arr


def _mos_is_pmos(element) -> bool:
    """True for a p-channel MOS. Polarity comes from the device model / factory
    name (``pfet``/``pmos``/``pch``). Needed for the drain into-net sign
    (finding B: ngspice ``@m[id]`` is the device drain current, positive in
    normal conduction for both n and p; pmos node-into = −id)."""
    for attr in ('pdk_device', 'device_name', '_device_type_name', 'model_name'):
        s = getattr(element, attr, None)
        if isinstance(s, str):
            s = s.lower()
            if 'pfet' in s or 'pmos' in s or 'pch' in s:
                return True
            if 'nfet' in s or 'nmos' in s or 'nch' in s:
                return False
    return False


def _current_row(element, port):
    """Map (element, port) → (suffix, sign) for the seam lookup, or ``None`` for
    a terminal/kind this v1 cannot annotate (skip with reason
    ``unsupported-kind``).

    MOS is **drain-only**: ngspice ``savecurrents`` on a sky130 subckt-wrapped
    BSIM device populates only ``@m[id]``; ``[is]/[ig]/[ib]`` come back 0,
    so s/g/b return ``None`` and their nets go partial/KCL-exempt rather than
    carrying wrong zeros. The drain sign is polarity-dependent."""
    from silk.elements.base import AnalogElement
    from silk.sources.classes import VoltageSource, CurrentSource
    from silk.pdk.basepdk import PDKDevice
    if _is_ideal_passive(element):
        return _TWO_TERMINAL.get(port)
    if isinstance(element, VoltageSource):
        sign = _SOURCE_BRANCH.get(port)
        return ('branch', sign) if sign is not None else None
    if isinstance(element, CurrentSource):
        sign = _SOURCE_BRANCH.get(port)
        return ('current', sign) if sign is not None else None
    if isinstance(element, (PDKDevice, AnalogElement)):   # MOS (subckt-wrapped)
        if port != 'd':                                   # drain-only (finding A)
            return None
        return ('id', +1.0 if _mos_is_pmos(element) else -1.0)   # finding B
    return None                                    # B/E/F/G/H, D, Q, submodule …


def _annotate_level(module, backend, raw, instance_path, tstart, force, report,
                    method='savecurrents'):
    """Annotate one module's own nets/pins in place from a transient rawfile.

    ``instance_path`` is the ``x``-prefixed SPICE path of this module ('' for the top level); ``backend`` provides the seam + ``load_signal``.
    ``method`` selects how MOS terminal currents are read: ``'savecurrents'``
    (drain-only, polarity-signed) or ``'ammeter_wrapper'`` (all four terminals
    from the 0 V ammeter branches, uniform sign — no drain-only limitation).
    Recursion into submodules is the caller's job."""
    import numpy as np
    from silk.circuit.emitted_names import emitted_instance_name
    from silk.elements.base import AnalogElement
    from silk.pdk.basepdk import PDKDevice

    # Subckt port nets carry device pins but NO pin for the boundary current
    # (ports are not PortBindings), so they can never close KCL — treat them as
    # interface/partial, never falsely "fully annotated".
    port_names = set(getattr(module, 'pin_list', None) or {})

    for net in module.net_dict.values():
        if net.width != 1:                         # bus handles have no own pins
            continue
        # Emitted spelling while annotating (it is the lookup spelling and
        # unique per net); finalize_report_names() maps the whole report to
        # the [k] display spelling at once, collision-safe (TASK-530).
        qualified = f'{instance_path}.{net.name}' if instance_path else net.name
        annotated = []
        had_skip = False
        for b in net.pins:
            if b.kind == 'lint_marker':
                continue                            # excluded from annotation + KCL
            pin_id = f'{b.instance}.{b.port}'
            if b._current_source == 'user' and not force:
                report.skips.append((qualified, pin_id, 'user-override'))
                had_skip = True
                continue
            element = _resolve_element(module, b)
            if element is None:
                report.skips.append((qualified, pin_id, 'unsupported-kind'))
                had_skip = True
                continue

            measured_terms = None
            if (method == 'ammeter_wrapper' and isinstance(element, (PDKDevice, AnalogElement))
                    and not _is_ideal_passive(element)):
                # All 4 MOS terminals via the ammeter branch; uniform into-net
                # sign (−i(vamm)); no drain-only/polarity limitation. An ideal
                # R/C/L is never ammeter-wrapped by the netlister (TASK-312.7),
                # so it always falls through to the plain _current_row branch
                # below, exactly as under 'savecurrents' (TASK-308).
                if b.port not in _MOS_TERMINALS:
                    report.skips.append((qualified, pin_id, 'unsupported-kind'))
                    had_skip = True
                    continue
                leaf = emitted_instance_name(element, b.instance)
                vec = backend.terminal_current_vector(raw, instance_path, leaf, b.port)
                if vec is None:
                    report.skips.append((qualified, pin_id, 'no-vector'))
                    had_skip = True
                    continue
                sign = -1.0
            else:
                row = _current_row(element, b.port)
                if row is None:
                    report.skips.append((qualified, pin_id, 'unsupported-kind'))
                    had_skip = True
                    continue
                suffix, sign = row
                if hasattr(element, 'physical_layout'):
                    measured_terms = []
                    for card, polarity in element.physical_layout().current_terms:
                        if getattr(backend, 'name', None) == 'xyce':
                            from silk.vector_match import collect_vector_candidates
                            path = ':'.join(filter(None, (instance_path.replace('.', ':'), card)))
                            rows = collect_vector_candidates(raw.native, path, 'I')
                            vector = raw.native.vectors[rows[0]].name if len(rows) == 1 else None
                        else:
                            vecs = backend.device_current_vectors(raw, instance_path, card)
                            vector = vecs.get(suffix)
                        if vector is None:
                            measured_terms = []
                            break
                        measured_terms.append((vector, polarity))
                    if not measured_terms:
                        report.skips.append((qualified, pin_id, 'no-vector'))
                        had_skip = True
                        continue
                    vec = measured_terms[0][0] if len(measured_terms) == 1 else None
                else:
                    leaf = emitted_instance_name(element, b.instance)
                    vecs = backend.device_current_vectors(raw, instance_path, leaf)
                    if suffix not in vecs or vecs[suffix] is None:
                        report.skips.append((qualified, pin_id, 'no-vector'))
                        had_skip = True
                        continue
                    vec = vecs[suffix]
            y = None
            for vector, polarity in measured_terms or [(vec, 1)]:
                inner = vector[2:-1] if vector.lower().startswith('i(') and vector.endswith(')') else vector
                sig = backend.load_signal(raw, inner, type='I')
                contribution = sign * polarity * np.asarray(sig.y, dtype=float)
                y = contribution if y is None else y + contribution
            avg, rms, peak = _trapezoid_window(sig.x, y, tstart)
            b.avg_current = avg
            b.rms_current = rms
            b.peak_current = peak
            b.current_vector = vec
            b._current_source = 'spice'
            annotated.append(b)
            report.pins_annotated += 1

        if not annotated:
            continue                                # zero-pin nets stay None
        report.nets_annotated.append(qualified)
        if had_skip or net.name in port_names:
            # Partial net: the ½·Σ bound is only conservative over a COMPLETE
            # pin set (its derivation uses KCL), so a partial aggregate would
            # UNDER-size the wire. Keep None aggregates — like zero-pin nets —
            # and mark partial/KCL-exempt. Per-pin currents stay.
            report.partial_nets.append(qualified)
        else:
            _refresh_net_aggregates(net, annotated)
            report.worst_kcl_residual[qualified] = sum(b.avg_current for b in annotated)


def plan_report_net_names(raw_names):
    """``{emitted net path: display path}`` for a whole report.

    Array instance parts are shown as ``[k]``, but a path whose bracket
    spelling would coincide with another path of the report (a literal net
    ``xx1[0].mid`` beside ``xx1<0>.mid``) keeps its raw spelling together with
    its whole collision group, so no row overwrites another. The result does
    not depend on traversal order."""
    from silk.spice_names import designer_alias_map
    return designer_alias_map(raw_names)


def finalize_report_names(report):
    """Rewrite every net path of *report* (``skips``' net column,
    ``nets_annotated``, ``partial_nets``, ``worst_kcl_residual`` keys) through
    ONE collision-safe mapping. Rows are all kept; values and the
    ``instance_modules`` / ``PinBinding.current_vector`` keys are untouched."""
    names = ([net for net, _pin, _why in report.skips] + list(report.nets_annotated)
             + list(report.partial_nets) + list(report.worst_kcl_residual))
    mapping = plan_report_net_names(names)
    report.skips = [(mapping[n], pin, why) for n, pin, why in report.skips]
    report.nets_annotated = [mapping[n] for n in report.nets_annotated]
    report.partial_nets = [mapping[n] for n in report.partial_nets]
    report.worst_kcl_residual = {mapping[n]: v for n, v in report.worst_kcl_residual.items()}


def _refresh_net_aggregates(net, annotated):
    """Conservative worst-segment bounds from annotated pins: each is
    ``½·Σ|·|`` over the pins (avg uses |signed avg|)."""
    net.avg_current = 0.5 * sum(abs(b.avg_current) for b in annotated)
    net.peak_current = 0.5 * sum(b.peak_current for b in annotated)
    net.rms_current = 0.5 * sum(b.rms_current for b in annotated)
    net._current_source = 'spice'


def _annotate_submodules(module, backend, raw, parent_path, tstart, force, report,
                         method='savecurrents'):
    """Recurse over ``module``'s ``elem_dict`` MODULE entries, mirroring the
    fault system's traversal: prefix ``x<name>`` (lowercased — submodule
    emission uses ``emitted_submodule_name``) and annotate each per-run instance ``Module`` snapshot.
    Module-level (not a method) to keep ``dir(Module)`` frozen to the golden."""
    from silk.circuit.circuit_elements import ElementType
    for name, entry in module.elem_dict.items():
        if not isinstance(entry, dict) or entry.get('type') != ElementType.MODULE:
            continue
        child = entry.get('device')
        if child is None:
            continue
        # Annotation is a per-run result, never shared circuit state. Keep a
        # snapshot in the instance record and expose it through the report.
        import copy
        from silk.simulation._worker_pickle import copy_simulation_netlist
        child = copy_simulation_netlist(child)
        child.module_name = entry.get('module_name', child.module_name)
        child.pin_list = copy.deepcopy(entry.get('pin_list', child.pin_list))
        entry['annotation'] = child
        path = f'{parent_path}.{emitted_submodule_name(name)}' if parent_path else emitted_submodule_name(name)
        _annotate_level(child, backend, raw, path, tstart, force, report, method)
        report.instance_modules[path] = child
        _annotate_submodules(child, backend, raw, path, tstart, force, report, method)


def _device_pin_names(element) -> set:
    """Terminal names for a device, whichever family built it: ``PDKDevice``
    (``pin_list``, a dict) or ``AnalogElement`` (``_pin_names``, a tuple)."""
    pin_list = getattr(element, 'pin_list', None)
    if pin_list is not None:
        return set(pin_list)
    return set(getattr(element, '_pin_names', ()))


def _iter_all_device_paths(elem_dict, instance_path='', designer_path=''):
    """EMISSION-time walk: every DEVICE-type instance reachable from
    ``elem_dict`` (whatever its shape — MOS, ideal passive, ...), as
    ``(instance_path, designer_path, name, device)`` 4-tuples. Unfiltered
    counterpart of :func:`_iter_mos_device_paths`: a
    device-parameter probe needs to name what it found — "not a MOS-shaped
    device" is a different, more useful refusal than "no such device" —
    which needs a walk that does not already drop non-MOS matches on the
    floor. See :func:`_iter_mos_device_paths` for the path/hierarchy
    convention, unchanged here.
    """
    from silk.circuit.circuit_elements import ElementType
    for name, entry in (elem_dict or {}).items():
        if not isinstance(entry, dict):
            continue
        if entry.get('type') == ElementType.MODULE:
            child = entry.get('device')
            if child is None:
                continue
            path = f'{instance_path}.{emitted_submodule_name(name)}' if instance_path else emitted_submodule_name(name)
            dpath = f'{designer_path}.{name}' if designer_path else name
            yield from _iter_all_device_paths(child.elem_dict, path, dpath)
            continue
        if entry.get('type') != ElementType.DEVICE:
            continue
        device = entry.get('device')
        if device is None:
            continue
        yield (instance_path, designer_path, name, device)


def _iter_mos_device_paths(elem_dict, instance_path='', designer_path=''):
    """EMISSION-time walk: every MOS-shaped device instance
    reachable from ``elem_dict``, as ``(instance_path, designer_path, name,
    device)`` 4-tuples — mirrors :func:`generate_device_instances`'s own
    convention (a bare ``elem_dict``, not a ``Module``) and
    :func:`_annotate_submodules`'s ``x<name>`` hierarchy-path construction,
    but over device INSTANCES (available at emission time) rather than
    :func:`_annotate_level`'s net/pin walk (which only reaches a device a
    Net binding already points at, and only post-run). "MOS-shaped" mirrors
    :data:`_MOS_TERMINALS`: a device whose own pin names are a superset of
    ``{d, g, s, b}`` — an ideal R/C/L is never a match
    (:func:`_is_ideal_passive` guards even a pathological pin-name
    collision), and a two-terminal source or passive falls out on pin count
    alone.

    ``instance_path`` accumulates the EMITTED (SPICE, ``emitted_submodule_name(name)`` per
    submodule level) container path, exactly as before.
    ``designer_path`` accumulates the same hierarchy from the raw names
    given to ``add_module``/``add_element`` themselves — no ``'x'`` prefix,
    no per-device SPICE-prefix translation at the leaf (that only happens
    at the leaf via :func:`~silk.circuit.emitted_names.emitted_instance_name`,
    applied by each caller, same as ``instance_path``'s own leaf) — ``OpRow.name`` (silk's own designer-facing spelling; ``OpRow.path`` keeps
    reading ``instance_path``, unchanged).

    Feeds :func:`device_op_save_lines`, which builds the actual
    ``save @m.<path>.<leaf>.m<model>[<param>]`` text — never by string
    pattern over the ``Module``: the leaf comes from
    :func:`~silk.circuit.emitted_names.emitted_instance_name` itself.

    A thin filter over :func:`_iter_all_device_paths` (same walk, order and yielded tuples).
    """
    for instance_path, designer_path, name, device in _iter_all_device_paths(
            elem_dict, instance_path, designer_path):
        if _is_ideal_passive(device):
            continue
        if not (_device_pin_names(device) >= _MOS_TERMINALS):
            continue
        yield (instance_path, designer_path, name, device)


def _device_model_name(device) -> str:
    """The PDK model name a MOS device instantiates on its own SPICE line
    (``PDKDevice.pdk_device`` / ``AnalogElement._pdk_device``, e.g.
    ``'sky130_fd_pr__nfet_01v8'``) — the exact token
    :meth:`~silk.pdk.basepdk.BasePDK.generate_netlist_entry` already appends
    to the instantiation line. Used only for diagnostics/messages now — the
    actual ``'@'`` address's internal suffix is :func:`_device_op_internal_name`,
    a SEPARATE per-device declaration (see its docstring for why the two
    are not the same string)."""
    return getattr(device, 'pdk_device', None) or getattr(device, '_pdk_device', '') or ''


def _device_op_params(device) -> Optional[Tuple[str, ...]]:
    """This device's verified ``'@'`` op-point parameter names
    (``PDKDevice.op_params`` / ``AnalogElement._op_params``), or ``None``
    when its model declares none — see :attr:`~silk.pdk.basepdk.PDKDevice.op_params`."""
    params = getattr(device, 'op_params', None)
    if params:
        return params
    params = getattr(device, '_op_params', None)
    return params if params else None


def _device_op_internal_name(device) -> Optional[str]:
    """The literal internal element-instance name (e.g. ``'m0'``,
    ``'msky130_fd_pr__nfet_01v8'``) this device's model needs right before
    ``[<param>]`` in its ``'@'`` address (``PDKDevice.op_internal_name`` /
    ``AnalogElement._op_internal_name``), or ``None`` when undeclared — see
    :attr:`~silk.pdk.basepdk.PDKDevice.op_internal_name` for why this is its
    own declaration and not derived from :func:`_device_model_name`."""
    name = getattr(device, 'op_internal_name', None)
    if name:
        return name
    name = getattr(device, '_op_internal_name', None)
    return name if name else None


def _device_op_prefix(instance_path: str, leaf: str) -> str:
    """``'<path>.<leaf>'`` (or bare ``leaf`` at the top level) — the
    container-path segment shared by both backends' addressing schemes."""
    return f'{instance_path}.{leaf}' if instance_path else leaf


def ngspice_device_op_line(instance_path: str, leaf: str, internal: str, param: str) -> str:
    """``'@m.<path>.<leaf>.<internal>[<param>]'`` — the ngspice pre/post-run
    ``save`` spelling for ONE (device, param) pair. Factored out of
    :func:`device_op_save_lines`'s loop so it and the
    device-parameter-probe resolver (:func:`resolve_probe_device_lines`)
    build the identical address by construction — a probe and its
    ``op_table()`` cell can never disagree about what string they ask
    ngspice to save."""
    return f'@m.{_device_op_prefix(instance_path, leaf)}.{internal}[{param}]'


def xyce_device_op_term(instance_path: str, leaf: str, internal: str, param: str) -> str:
    """``'N(<hierarchy>:<internal>:PARAM)'`` / ``'ID(<hierarchy>:<internal>)'``
    — the Xyce ``.print`` spelling for ONE (device, param) pair. Factored
    out of :func:`xyce_op_print_terms`'s loop, same reason as
    :func:`ngspice_device_op_line`."""
    # *internal* is itself split on '.' (not just joined verbatim) so an
    # ammeter-wrapped address's extra 'XMdev.<internal>' segment becomes
    # two colon-separated Xyce hierarchy tokens, not one token with a stray
    # dot in it -- a no-op split for the ordinary, unwrapped case.
    segments = ((instance_path.split('.') if instance_path else [])
               + [leaf] + internal.split('.'))
    qualified = ':'.join(segments).upper()
    if param == 'id':
        return f'ID({qualified})'
    return f'N({qualified}:{param.upper()})'


def xyce_draw_name_map(elem_dict, record):
    """Which Xyce columns of a Monte Carlo ``.res`` belong to which designer
    device (``record_draws``).

    *record* is the PDK's ``statistical.record`` (device type -> drawn
    parameters). Returns ``(name_map, params, skipped_types)``: the Xyce
    instance prefix (``'XOFF:XM1'``) -> designer name (``'off.m1'``), the
    designer name -> its recorded parameter names, and the device types with
    no ``record`` entry (not recorded). Same walk as :func:`xyce_op_print_terms`.
    """
    from silk.circuit.emitted_names import emitted_instance_name
    name_map, params, skipped = {}, {}, []
    for instance_path, designer_path, name, device in _iter_mos_device_paths(elem_dict):
        kind = getattr(device, 'device_name', None)
        if kind not in record:
            if kind not in skipped:
                skipped.append(kind)
            continue
        leaf = emitted_instance_name(device, name)
        prefix = ':'.join((instance_path.split('.') if instance_path else [])
                          + [leaf]).upper()
        designer = f'{designer_path}.{name}' if designer_path else name
        name_map[prefix] = designer
        params[designer] = tuple(record[kind])
    return name_map, params, tuple(skipped)


def device_op_save_lines(elem_dict) -> List[str]:
    """The ``'@m.<path>.<leaf>.<internal>[<param>]'`` vector names
    ``save_device_params`` saves — one per (MOS device
    instance, declared op-param name) pair.

    A device whose model declares NO op-param set (:func:`_device_op_params`
    returns ``None``) is SKIPPED entirely: no saves are attempted for it, and
    it is never guessed at. This
    is not a cosmetic choice: a device whose model does not actually expose a
    given ``'@'`` name gets a zero-length vector, and referencing even ONE
    such name in a ``save``/``write`` sequence silently aborts the WHOLE
    ``.control`` ``write`` with no rawfile at all for the ENTIRE test — the
    same failure mode already documented on ``generate_output_block``'s
    ``savecurrents`` incompatibility, not a new one. Measured directly: a
    nested two-level gf180mcu bench with the global (undeclared-per-model)
    17-name list produced no rawfile at all, on ``nfet_03v3``/``pfet_03v3``
    alike; declaring exactly the 14 names that model actually exposes (see
    ``tests/fixtures/op_point/gf180mcu/``) fixes it.

    The internal address suffix (:func:`_device_op_internal_name`) is a
    SEPARATE per-device declaration from the param names, and genuinely
    cannot be derived from the model name by any general rule: sky130's own
    model-instance line happens to be named ``'m' + model name``
    (``'msky130_fd_pr__nfet_01v8'``), which is a fact about how sky130's
    models were authored, not an ngspice convention — gf180mcu's
    ``nfet_03v3``/``pfet_03v3`` name their internal instance ``'m0'``
    instead, unrelated to the model name entirely. Guessing the sky130
    spelling universally is exactly what the gate finding's gf180mcu/sg13g2
    failures traced back to: EVERY declared param failed identically (not
    only the one ngspice happened to name in its abort message), because the
    whole address was wrong, not just one parameter.
    """
    from silk.circuit.emitted_names import emitted_instance_name
    lines = []
    for instance_path, _designer_path, name, device in _iter_mos_device_paths(elem_dict):
        params = _device_op_params(device)
        internal = _device_op_internal_name(device)
        if not params or not internal:
            continue
        leaf = emitted_instance_name(device, name)
        for param in params:
            lines.append(ngspice_device_op_line(instance_path, leaf, internal, param))
    return lines


def _device_op_backends(device) -> Optional[Tuple[str, ...]]:
    """This device's verified op-point simulator backends
    (``PDKDevice.op_backends`` / ``AnalogElement._op_backends``), or
    ``None`` when undeclared -- see
    :attr:`~silk.pdk.basepdk.PDKDevice.op_backends`. Same
    dual-family accessor shape as :func:`_device_op_params`."""
    backends = getattr(device, 'op_backends', None)
    if backends:
        return backends
    backends = getattr(device, '_op_backends', None)
    return backends if backends else None


def xyce_op_print_terms(elem_dict) -> List[str]:
    """The ``N(<path>:PARAM)``/``ID(<path>)`` ``.print`` terms
    ``save_device_params`` adds to a Xyce Op()-only deck — one
    per (MOS device instance, capability-narrowed op-param name) pair.
    Xyce's own emission-side counterpart of :func:`device_op_save_lines`,
    same walk (:func:`_iter_mos_device_paths`) and the same "guess nothing,
    skip a device entirely rather than risk the whole run" discipline —
    Xyce aborts the WHOLE run on one unresolvable ``.print`` symbol, a strictly harsher failure mode than ngspice's
    per-vector zero-length.

    A device is skipped entirely (no terms at all) unless it declares an
    op-param set AND an internal name (the pair) AND lists
    ``'xyce'`` in its own :func:`_device_op_backends`. What's left is then narrowed to
    :data:`~silk.simulation.backend.XYCE_OP_CAPABILITY` (``gmbs`` and the
    capacitances are confirmed unavailable on Xyce regardless of what the
    device declares for ngspice) — this is the "may only try" / "can
    actually read" split :meth:`~silk.simulation.backend.SimulatorBackend.op_param_capability`
    formalizes on the read side; this function applies the identical
    narrowing on the emission side so the two can never disagree about
    which columns a Xyce deck asks for.

    Qualified-name shape (measured directly, spike):
    ``N(<hierarchy>:<internal>:PARAM)`` / ``ID(<hierarchy>:<internal>)``,
    colon-separated, uppercased, the FULL path down to the internal
    element instance -- the same three-part structure
    :func:`device_op_save_lines` builds for ngspice's dotted ``'@'``
    address, just joined with ``':'`` instead of ``'.'`` and with no
    literal ``'m'`` prefix segment (Xyce has none).
    """
    from silk.circuit.emitted_names import emitted_instance_name
    from silk.simulation.backend import XYCE_OP_CAPABILITY

    terms = []
    for instance_path, _designer_path, name, device in _iter_mos_device_paths(elem_dict):
        params = _device_op_params(device)
        internal = _device_op_internal_name(device)
        op_backends = _device_op_backends(device)
        if not params or not internal or not op_backends or 'xyce' not in op_backends:
            continue
        capability = tuple(p for p in params if p in XYCE_OP_CAPABILITY)
        if not capability:
            continue
        leaf = emitted_instance_name(device, name)
        for param in capability:
            terms.append(xyce_device_op_term(instance_path, leaf, internal, param))
    return terms


def _device_polarity(device) -> Optional[str]:
    """This device's declared channel polarity (``'n'``/``'p'``), or
    ``None`` when its model declares none — see
    :attr:`~silk.pdk.basepdk.PDKDevice.polarity`. Same
    dual-family accessor shape as :func:`_device_op_params`."""
    polarity = getattr(device, 'polarity', None)
    if polarity:
        return polarity
    polarity = getattr(device, '_polarity', None)
    return polarity if polarity else None


#: The DUT's own fixed instance name inside a testbench netlist
#: (``module_test.py``'s ``add_module_under_test`` -- the single call site,
#: always ``self.netlist.add_module(self.module, 'dut', nets=port_mapping)``)
#: -- the one and only string this constant may ever need to match.
DUT_ELEMENT_NAME = 'dut'


def dut_module_from_elem_dict(elem_dict):
    """The DUT's own per-instance ``Module`` inside a testbench's top-level
    ``elem_dict`` (``elem_dict['dut']['device']``), or ``None`` when there
    is no DUT entry (``add_module_under_test`` was never called) —
    read/emission-side counterpart of :data:`DUT_ELEMENT_NAME`.
    """
    entry = (elem_dict or {}).get(DUT_ELEMENT_NAME)
    return entry.get('device') if isinstance(entry, dict) else None


# ---------------------------------------------------------------------
# Device-parameter probes (TASK-330.8) — tb.probe('<designer-name>:<param>')
#
# One spelling, resolved the same way at emission time (building the
# pre-run save/print line) and at read time (finding the value): both
# walk ``dut_module.elem_dict`` by TASK-330.7's designer-name convention
# and build the address through :func:`ngspice_device_op_line` /
# :func:`xyce_device_op_term`, the same builders ``save_device_params``
# uses — a probe and its ``op_table()`` cell can never disagree about
# what string addresses a given (device, param) pair.
#
# Reviewer ruling on the AC #2 proposal (2026-09-28), CHANGE 1: a probe is
# an explicit request and is NOT limited to a device's declared
# ``op_params`` (that list only picks ``op_table()``'s own default
# columns). Refuse before the run only when the device cannot be
# ADDRESSED at all — not MOS-shaped, or no declared internal address — and
# (Xyce only) when the device/param is outside what Xyce can even try to
# print, since one bad Xyce ``.print`` symbol aborts the whole run.
#
# OWNER DECISION (2026-09-28) superseding BOTH the original
# ``'<name>[<param>]'`` spelling and an interim gate-fix that tried to
# rescue it: brackets collide with a legitimate net name ending the same
# way (a bus bit designed ``'bus[en]'`` was wrongly refused as a bad
# device parameter — a regression on a492303d). The
# spelling is ``'<designer-name>:<param>'`` instead: ``':'`` was already
# reserved for the pre-330.8 terminal-current spelling
# (``'xdut.m1:d'``, see ``ModuleTestbench._ammeter_vector``) and is not
# used in a hierarchy path, so it does not collide with any net or path
# name the way a trailing bracket could. ``ModuleTestbench.probe()``
# dispatches on the SUFFIX after the LAST ``':'`` (case-insensitive,
# normalised to lowercase): one of the 4 terminals
# (``d``/``g``/``s``/``b``) is the unchanged pre-330.8 terminal-current
# probe; one of the 17 :data:`~silk.simulation.backend.DEVICE_OP_PARAMS`
# names is this device-parameter probe, with the prefix as the DESIGNER
# name (never the emitted path the terminal spelling uses); anything else
# is refused before the run, naming both closed sets. The two sets are
# disjoint by construction (``tests/test_device_param_probe.py`` asserts
# it), so the dispatch is never ambiguous.
# ---------------------------------------------------------------------

#: op_table() DERIVED columns (op_table.py's own ``DERIVED_COLUMNS``) plus
#: the two categorical region fields: none of
#: these are probe-able device parameters. Each backs from ordinary Signal
#: arithmetic on RAW probes instead (see :func:`_derived_column_hint`).
_DERIVED_ARITHMETIC_HINTS = {
    'vov':        "tb.probe({name!r} + ':vgs') - tb.probe({name!r} + ':vth')",
    'gm_id':      "tb.probe({name!r} + ':gm') / tb.probe({name!r} + ':id')",
    'gm_gds':     "tb.probe({name!r} + ':gm') / tb.probe({name!r} + ':gds')",
    'ro':         "1 / tb.probe({name!r} + ':gds')",
    'ft':         "tb.probe({name!r} + ':gm') / (2 * math.pi * "
                  "tb.probe({name!r} + ':cgg'))",
    'vds_margin': "tb.probe({name!r} + ':vds') - tb.probe({name!r} + ':vdsat')",
}
#: Categorical op_table() columns with no arithmetic equivalent at all.
_DERIVED_CATEGORICAL = ('region', 'saturation_weak')
NON_PROBEABLE_OP_COLUMNS = tuple(_DERIVED_ARITHMETIC_HINTS) + _DERIVED_CATEGORICAL


def device_probe_spelling(designer_name: str, param: str) -> str:
    """``probe('M1:cgso')`` — a device-parameter probe rendered exactly
    as the caller would type it, for every diagnostic that names one: a single
    quoted ``'<name>:<param>'`` string, never two separately-repr'd
    pieces juxtaposed."""
    return repr(f'{designer_name}:{param}')


def derived_column_probe_error(raw: str, designer_name: str, param: str) -> ValueError:
    """The refusal for ``tb.probe('<name>:<derived-column>')`` —
    ``param`` is one of :data:`NON_PROBEABLE_OP_COLUMNS`. Named the
    arithmetic to use instead when there is one."""
    hint = _DERIVED_ARITHMETIC_HINTS.get(param)
    if hint is not None:
        return ValueError(
            f"probe({raw!r}): {param!r} is a DERIVED op_table() column, "
            f"not a probe-able device parameter — no simulator vector "
            f"backs it directly. Build it from ordinary Signal arithmetic "
            f"on raw device-parameter probes instead: "
            f"{hint.format(name=designer_name)}."
        )
    table_key = f'{DUT_ELEMENT_NAME}.{designer_name}'
    return ValueError(
        f"probe({raw!r}): {param!r} is a categorical op_table() column "
        f"(region classification), not a probe-able device parameter. "
        f"Read report.op_table()[{table_key!r}].{param} after "
        f"evaluate() instead, or compare raw device-parameter probes "
        f"directly (e.g. tb.probe({designer_name!r} + ':vgs') vs "
        f"tb.probe({designer_name!r} + ':vth'))."
    )


def unrecognized_device_param_error(designer_name: str, param: str) -> ValueError:
    """Refuse an unrecognised device parameter before simulation.

    *param* is checked
    against :data:`~silk.simulation.backend.DEVICE_OP_PARAMS` right here,
    in ``probe()`` itself, BEFORE the run — not deferred to a read-time
    raise. An unrecognised ``'@'``
    name can abort the WHOLE ngspice write (losing every other probe of
    that run too), or, alongside enough other saved vectors, come back as
    a real, populated, all-zero reading with no warning at all — neither
    failure mode is something a read-time check alone can always catch or
    contain. Refusing here, before any deck is even generated, is strictly
    safer and catches the mistake at the earliest possible point.

    Used both here (``ModuleTestbench.probe()``, the primary check) and by
    :meth:`~silk.simulation.backend.NgspiceBackend.load_device_probe` (a
    defense-in-depth check at read time, for any other construction path
    that might reach it) — the exact same message either way.
    """
    from silk.simulation.backend import DEVICE_OP_PARAMS
    import difflib
    close = difflib.get_close_matches(param, DEVICE_OP_PARAMS, n=3, cutoff=0.5)
    return ValueError(
        f"probe({device_probe_spelling(designer_name, param)}): {param!r} "
        f"is not one of the {len(DEVICE_OP_PARAMS)} recognised device-parameter names "
        f"— {sorted(DEVICE_OP_PARAMS)}"
        + (f". Did you mean {close!r}?" if close else ".") +
        f" Refused before the run: an unrecognised name can abort the "
        f"whole write on ngspice, or come back as a real, populated, "
        f"all-zero reading with no warning at all — silk never risks "
        f"either by accepting a name outside this set."
    )


@dataclass(frozen=True)
class DeviceProbeAddress:
    """One device-parameter probe's ADDRESS — everything needed to build
    the ngspice/Xyce vector spelling for a given param, resolved once
    against a live DUT tree and shared by emission and read time."""
    instance_path: str   # ABSOLUTE (xdut-prefixed) emitted container path
    leaf: str
    internal: str
    device: object
    designer_name: str
    model: Optional[str]
    polarity: Optional[str]


@dataclass(frozen=True)
class PreflightResult:
    """The answer ``PySpiceNetlister._ngspice_param_preflight`` gives for one (binary, model, param)
    combination: decided by what the resolved ngspice binary actually does
    with a minimal single-device deck, never by a hardcoded version
    table. ``version`` is this binary's own version string, read from the
    preflight's own log the same way ``silk.simulation.conditions.read_ngspice_version`` reads a main run's log — the SAME string a main
    run against this binary would carry as ``RawResult.simulator_version``.
    ``reason`` is empty when ``available`` is True.

    Parameters
    ----------
    available : bool
        Whether the binary reports the device parameter.
    version : str, optional
        The binary's own version string; ``None`` if it could not be read.
    log_path : str, optional
        Path of the preflight run's log, for diagnosis.
    reason : str
        Why the parameter is unavailable; empty when ``available`` is true.
    """
    #: Whether the binary reports the device parameter.
    available: bool
    #: The binary's own version string, or ``None`` if unread.
    version: Optional[str]
    #: Path of the preflight run's log.
    log_path: Optional[str]
    #: Why the parameter is unavailable; empty when ``available`` is true.
    reason: str


def _known_designer_names(dut_module) -> List[str]:
    return [
        (f'{dpath}.{name}' if dpath else name)
        for _ip, dpath, name, _dev in _iter_all_device_paths(
            getattr(dut_module, 'elem_dict', None) or {})
    ]


#: The FIXED internal instance name every ammeter-wrapper subckt gives the
#: real model instance it wraps (``spice_netlister.py``'s
#: ``_build_ammeter_wrapper`` — literally ``'XMdev'``, the same for every
#: model). A device-parameter address built while
#: ``terminal_current_method == 'ammeter_wrapper'`` needs this as an EXTRA
#: hierarchy segment between the wrapper's own instance and the model's op
#: internal name — discovered empirically during the build (a
#: probe naming a wrapped device's parameter otherwise silently reads a
#: phantom all-zero vector, since ngspice accepts an address it has never
#: seen without complaint): ``save_device_params`` never had to know this,
#: because it is forbidden to combine with ``save_device_currents`` (any
#: ``terminal_current_method``) — a probe is not.
AMMETER_WRAPPER_INTERNAL_NAME = 'XMdev'


def find_designer_device(dut_module, designer_name: str):
    """``(instance_path, name, device)`` for the device whose
    designer name (dot-joined raw instance names, exact case) is
    *designer_name*, or ``None``. ``instance_path`` is the emitted container
    path RELATIVE to the DUT. The one lookup behind both ``'<name>:<param>'``
    device-parameter probes and ``'<name>:<terminal>'`` terminal-current
    probes, so the two can never resolve a name differently."""
    if dut_module is None:
        return None
    for instance_path, dpath, name, device in _iter_all_device_paths(
            getattr(dut_module, 'elem_dict', None) or {}):
        candidate = f'{dpath}.{name}' if dpath else name
        if candidate == designer_name:
            return (instance_path, name, device)
    return None


def resolve_device_probe_address(dut_module, designer_name: str, param: str, *,
                                 ammeter_wrapped: bool = False) -> DeviceProbeAddress:
    """Refuse before the
    run only when *designer_name* cannot be ADDRESSED — unknown name, not
    MOS-shaped, or its model declares no internal address
    (:func:`_device_op_internal_name`). *param* itself is never checked
    against the device's declared ``op_params`` here (see the module
    docstring above); an unsupported/mistyped one surfaces at read time.

    Raises ``ValueError`` naming *designer_name*/*param* and, for an
    unknown name, the closest known ones (``difflib``, mirroring
    ``OpTable.__getitem__``).
    """
    if dut_module is None:
        raise ValueError(
            f"probe({device_probe_spelling(designer_name, param)}): this "
            f"testbench has no DUT yet — call add_module_under_test() "
            f"before probing a device parameter."
        )
    found = find_designer_device(dut_module, designer_name)
    if found is None:
        import difflib
        names = _known_designer_names(dut_module)
        close = difflib.get_close_matches(designer_name, names, n=3, cutoff=0.6)
        candidates = close or sorted(names)
        label = 'close matches' if close else 'known device names'
        raise ValueError(
            f"probe({device_probe_spelling(designer_name, param)}): no "
            f"device named {designer_name!r} under this DUT ({label}: "
            f"{', '.join(candidates) or '(none)'})."
        )
    instance_path, name, device = found
    if _is_ideal_passive(device) or not (_device_pin_names(device) >= _MOS_TERMINALS):
        raise ValueError(
            f"probe({device_probe_spelling(designer_name, param)}): "
            f"{designer_name!r} is not a MOS-shaped device — device-"
            f"parameter probes (gm, vth, id, cgg, ...) only address a "
            f"4-terminal MOS instance."
        )
    internal = _device_op_internal_name(device)
    if not internal:
        model = _device_model_name(device) or 'its model'
        raise ValueError(
            f"probe({device_probe_spelling(designer_name, param)}): "
            f"{designer_name!r} ({model}) declares no operating-point "
            f"internal address (PDKDevice.op_internal_name); device-"
            f"parameter probes are unsupported for this device."
        )
    from silk.circuit.emitted_names import emitted_instance_name
    leaf = emitted_instance_name(device, name)
    dut_prefix = emitted_submodule_name(DUT_ELEMENT_NAME)
    absolute_path = f'{dut_prefix}.{instance_path}' if instance_path else dut_prefix
    if ammeter_wrapped:
        internal = f'{AMMETER_WRAPPER_INTERNAL_NAME}.{internal}'
    return DeviceProbeAddress(
        instance_path=absolute_path, leaf=leaf, internal=internal,
        device=device, designer_name=designer_name,
        model=_device_model_name(device) or None,
        polarity=_device_polarity(device),
    )


def require_xyce_probe_capability(address: DeviceProbeAddress, param: str) -> None:
    """Xyce-only pre-run refusal: a device never verified on the
    'xyce' backend, or a parameter outside
    :data:`~silk.simulation.backend.XYCE_OP_CAPABILITY` (``gmbs``, every
    capacitance — confirmed unavailable regardless of what the device
    declares). One bad Xyce ``.print`` symbol aborts the WHOLE run, so
    this is checked before the run rather than surfaced at read time."""
    op_backends = _device_op_backends(address.device)
    if not op_backends or 'xyce' not in op_backends:
        raise ValueError(
            f"probe({device_probe_spelling(address.designer_name, param)}) "
            f"on Xyce: {address.designer_name!r}'s model has not been "
            f"verified on the 'xyce' backend (PDKDevice.op_backends); "
            f"refused before the run rather than risk aborting the whole "
            f"Xyce deck on an unresolvable .print symbol."
        )
    from silk.simulation.backend import XYCE_OP_CAPABILITY
    if param not in XYCE_OP_CAPABILITY:
        raise ValueError(
            f"probe({device_probe_spelling(address.designer_name, param)}) "
            f"on Xyce: {param!r} is outside Xyce's own N()/ID() capability "
            f"({sorted(XYCE_OP_CAPABILITY)}) — gmbs and every capacitance "
            f"are confirmed unavailable on Xyce regardless of what the "
            f"device declares. A bad .print symbol aborts the whole Xyce "
            f"run, so this is refused before the run."
        )


def resolve_probe_device_lines(dut_module, probe_requests, simulator: str, *,
                               ammeter_wrapped: bool = False) -> List[str]:
    """Validate and address every ``(designer_name, param)`` device-probe
    request, returning the ngspice ``'@'`` save lines (or, for
    ``simulator='xyce'``, the ``N()``/``ID()`` print terms) to emit —
    deduped, in request order. Raises ``ValueError`` before the run for
    anything statically wrong.

    *ammeter_wrapped*: this deck's ``terminal_current_method`` is
    ``'ammeter_wrapper'`` — every MOS-shaped device is emitted one
    hierarchy level deeper (inside its 0 V-ammeter wrapper subckt), so the
    address needs :data:`AMMETER_WRAPPER_INTERNAL_NAME` threaded in."""
    lines: List[str] = []
    seen = set()
    for designer_name, param in probe_requests:
        address = resolve_device_probe_address(
            dut_module, designer_name, param, ammeter_wrapped=ammeter_wrapped)
        if simulator == 'xyce':
            require_xyce_probe_capability(address, param)
            line = xyce_device_op_term(address.instance_path, address.leaf,
                                       address.internal, param)
        else:
            line = ngspice_device_op_line(address.instance_path, address.leaf,
                                          address.internal, param)
        if line not in seen:
            seen.add(line)
            lines.append(line)
    return lines


def capture_op_device_records(bench_module, backend, raw, pdk=None) -> List[dict]:
    """Capture every MOS-shaped device in the testbench at evaluate time.

    Walk each placement, including DUT and nested/repeated helper modules,
    in depth-first insertion order. Records are plain JSON/pickle-safe dicts
    with bench-rooted emitted path and designer name (for example
    'xdut.xbias1.XMMN' / 'dut.bias1.MN', versus a sibling helper's
    'xbias1.XMMN' / 'bias1.MN'). Backend reads use the complete emitted
    container path independently of displayed labels.

    Each record stores path, name, model, polarity, w_um, l_um, m,
    declared_params and op_values. Geometry comes from device_geometry_um.
    Declared parameters are narrowed by verified backend and capability;
    unavailable readings are nan. No additional saves are requested here.

    Capture is a snapshot: reports need neither a live Module nor the
    rawfile later. A backend's NotSupportedError propagates so the caller
    can record that no inventory is available.
    """
    from silk.circuit.emitted_names import emitted_instance_name
    from silk.pdk.basepdk import device_geometry_um
    from silk.simulation.backend import DEVICE_OP_PARAMS

    nan_values = {p: float('nan') for p in DEVICE_OP_PARAMS}
    records = []
    for container_path, designer_path, name, device in _iter_mos_device_paths(bench_module.elem_dict):
        leaf = emitted_instance_name(device, name)
        emitted_path = f'{container_path}.{leaf}' if container_path else leaf
        designer_name = f'{designer_path}.{name}' if designer_path else name
        device_polarity = _device_polarity(device)
        values = backend.device_op_vectors(raw, container_path, leaf,
                                           polarity=device_polarity)
        if not values:
            values = dict(nan_values)

        geometry = device_geometry_um(device, pdk)
        w_um, l_um, m = geometry if geometry is not None else (None, None, None)

        # TASK-330.5: a column is only ever "declared" for THIS report when
        # (a) the model declares it at all, (b) this device has been
        # verified on THIS backend (op_backends), and (c) this backend can
        # actually read it back (op_param_capability — Xyce's fixed 8-name
        # set). All three narrow the same tuple; a Xyce hole (gmbs, the
        # capacitances) or an unverified device therefore never reads
        # source == 'sim' downstream (op_table.py), regardless of what the
        # model's own op_params says.
        model_params = _device_op_params(device) or ()
        op_backends = _device_op_backends(device) or ()
        backend_params = model_params if backend.name in op_backends else ()
        declared_params = backend.op_param_capability(backend_params)

        records.append({
            'path': emitted_path,
            'name': designer_name,
            'model': _device_model_name(device) or None,
            'polarity': device_polarity,
            'w_um': w_um,
            'l_um': l_um,
            'm': m,
            'declared_params': tuple(declared_params),
            'op_values': dict(values),
        })
    return records


def _reduce_definitions(top, report):
    """After per-instance annotation, write the **worst case
    across instances** onto each ``module_dict`` live definition — per
    corresponding pin, the max-|avg| instance's value (its sign and vector
    retained). Layout is drawn once per definition, so it is sized for its
    worst-duty instance. The shared definition mixes instances, so it is
    **exempt** from the KCL predicate (not added to ``worst_kcl_residual``);
    per-instance copies keep the KCL self-check.
    """
    by_name = {}
    for mod in report.instance_modules.values():
        mn = getattr(mod, 'module_name', None)
        if mn is None:                              # the top-level tb
            continue
        by_name.setdefault(mn, []).append(mod)

    for mn, instances in by_name.items():
        entry = top.module_dict.get(mn)
        if not isinstance(entry, dict):
            continue                                # nested-only defn (v1 skips)
        defn = entry.get('device')
        if defn is None:
            continue
        port_names = set(getattr(defn, 'pin_list', None) or {})
        reduced_any = False
        for net_name, net in defn.net_dict.items():
            if net.width != 1:
                continue
            annotated = []
            had_skip = False
            for b in net.pins:
                if b.kind == 'lint_marker':
                    continue
                best = None
                for inst in instances:
                    inet = inst.net_dict.get(net_name)
                    if inet is None:
                        continue
                    ib = next((x for x in inet.pins
                               if x.instance == b.instance and x.port == b.port), None)
                    if ib is None or ib.avg_current is None:
                        continue
                    if best is None or abs(ib.avg_current) > abs(best.avg_current):
                        best = ib
                if best is None:
                    had_skip = True                 # this pin annotated in no instance
                    continue
                b.avg_current = best.avg_current
                b.rms_current = best.rms_current
                b.peak_current = best.peak_current
                b.current_vector = best.current_vector
                b._current_source = 'spice'
                annotated.append(b)
            if annotated:
                reduced_any = True
                # Same conservativeness rule as per-instance: only a COMPLETE,
                # non-interface net gets a ½·Σ aggregate; partial defn nets keep
                # None. Per-pin worst-case values are set regardless.
                if not (had_skip or net_name in port_names):
                    _refresh_net_aggregates(net, annotated)
        if reduced_any:
            report.definition_reduced.append(mn)


def _plot_loaded_signals(module, signals, *, overlay, show):
    """Draw already-loaded ``(label, Signal)`` pairs for *module*.

    The auto-scaled multi-panel / single-axis drawing factored out of
    :meth:`_ResultsMethods.plot`, as a plain module-level
    function rather than a method: :meth:`~silk.circuit.module_test.
    ModuleTestbench.plot` reuses it for probes and derived Signals it has
    already resolved (no re-loading by name), and a free function keeps
    ``dir(Module)`` — the frozen surface ``tests/refactor/test_module_golden.
    py`` checks — untouched by an implementation detail two callers share.
    *module* supplies ``_auto_scale`` and ``netlistname`` (the title source);
    *signals* is a list of ``(label, Signal)`` pairs, each Signal already
    carrying data (``x``/``y``/``xlabel``/``ylabel``) — no loading or
    resolving happens here.

    Signals do **not** have to share an x-axis. Previously, they always
    did, because a run held one plot; a run that declares several analyses
    holds several, and ``tb.plot(vout_t, vout_f)`` is one trace off each. So
    each panel is scaled and labelled from its OWN x, and ``sharex`` is
    turned on only when every x really is the same array — which keeps a
    single-analysis figure identical, since identical axes scale identically.
    Overlay is the one case that cannot be reconciled: a single axis cannot
    be seconds and hertz at once, so it refuses naming both.

    A **complex** Signal (an AC probe) is not auto-scaled here at all: its
    panel is drawn by handing its own ``ax`` to :meth:`~silk.signals.base.
    Signal.plot`, which reads magnitude in dB on a log-frequency axis
    instead of silently truncating to the real part (matplotlib's own
    ``ComplexWarning``, twice, is what casting a complex array to real would
    otherwise raise). That panel's axis/label treatment is then identical to
    calling ``sig.plot()`` directly — the one copy of the mode/axis logic — and this function's own SI auto-scaling is skipped for it,
    real signals sharing a figure with it are unaffected.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if not signals:
        raise ValueError(
            "No signals to plot. Use ckt.get_signal_names() to list available signals."
        )

    # Auto-scale each signal's own x-axis; share the axis only if they agree.
    scaled_x = [module._auto_scale(sig.x, sig.xlabel or 'time')
                for _, sig in signals]
    first_sig = signals[0][1]
    x_scaled, x_label = scaled_x[0]
    shared_x = all(
        getattr(sig.x, 'shape', None) == getattr(first_sig.x, 'shape', None)
        and np.array_equal(sig.x, first_sig.x)
        for _, sig in signals)

    # Derive circuit title
    title = os.path.basename(module.netlistname).replace('.cir', '')

    if overlay:
        if not shared_x:
            axes_named = sorted({(sig.xlabel or 'time') for _, sig in signals})
            raise ValueError(
                f"overlay=True needs every signal on one x-axis, and these "
                f"are not: {', '.join(repr(a) for a in axes_named)}. A run "
                f"that declares several analyses writes one plot per "
                f"analysis; plot them as separate panels (overlay=False), or "
                f"pass the signals of one analysis at a time.")
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.suptitle(title, fontsize=12)
        any_complex = False
        y_label = None
        for label, sig in signals:
            if np.iscomplexobj(np.asarray(sig.y)):
                sig.plot(ax=ax, label=label)
                any_complex = True
            else:
                y_scaled, y_label = module._auto_scale(sig.y, sig.ylabel or 'v')
                ax.plot(x_scaled, y_scaled, label=label)
        if not any_complex:
            # Unchanged from before TASK-230: every signal here is real, so
            # the auto-scaled x/y labels this function computed are correct.
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label if len(signals) == 1 else 'Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        n = len(signals)
        fig, axes = plt.subplots(n, 1, figsize=(10, 2.5 * n + 0.5),
                                 sharex=shared_x)
        fig.suptitle(title, fontsize=12)
        if n == 1:
            axes = [axes]
        for ax, (label, sig), (panel_x, panel_x_label) in zip(
                axes, signals, scaled_x):
            if np.iscomplexobj(np.asarray(sig.y)):
                # Delegate the whole panel to Signal.plot: dB magnitude on a
                # log-frequency axis, its own (unscaled) xlabel/ylabel — no
                # SI auto-scaling and no shared bottom-axis override below.
                sig.plot(ax=ax)
                ax.grid(True, alpha=0.3)
                continue
            y_scaled, y_label = module._auto_scale(sig.y, sig.ylabel or 'v')
            ax.plot(panel_x, y_scaled)
            ax.set_ylabel(f'{label}\n{y_label}')
            ax.grid(True, alpha=0.3)
            if not shared_x:
                # Each panel carries its own axis label; with a shared axis
                # only the bottom one does, exactly as before.
                ax.set_xlabel(panel_x_label)
        last_is_complex = np.iscomplexobj(np.asarray(signals[-1][1].y))
        if shared_x and not last_is_complex:
            axes[-1].set_xlabel(x_label)

    fig.tight_layout()
    if show:
        plt.show()
    return fig


def _plot_corner_groups(module, groups, *, overlay, show):
    """Draw one panel per signal, one labelled trace per corner.

    The corner-aware sibling of :func:`_plot_loaded_signals`. *groups* is a
    list of ``(label, [(corner, Signal), ...])``: one entry per signal the
    caller asked for, each carrying that signal as measured in each selected
    corner, already resolved (every Signal carries data — nothing is loaded
    or resolved here).

    Two things differ from the single-run drawing, and both follow from what
    a corner sweep is:

    * **Corners share an axis without sharing an x array.** Two runs of the
      same transient land on different adaptive timesteps, so array equality
      is the wrong question; the right one is whether the axes mean the same
      thing, which is what ``xlabel`` says. Overlay therefore refuses on
      mixed axes (seconds and hertz cannot share one x) but not on mixed
      sample points, and panels are scaled once per group across every
      corner's samples so the traces in a panel are directly comparable.
    * **Every panel gets a legend**, because a panel now holds several
      traces that are only told apart by their corner.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    flat = [(label, corner, sig)
            for label, traces in groups for corner, sig in traces]
    if not flat:
        raise ValueError(
            "No signals to plot. Pass probes (tb.plot(vp, vm)) and corners "
            "the bench retained (tb.sweep_corners).")

    axes_named = sorted({(sig.xlabel or 'time') for _, _, sig in flat})
    shared_x = len(axes_named) == 1
    title = os.path.basename(module.netlistname).replace('.cir', '')

    def _joined(arrays):
        """Every trace's samples end to end — what one shared prefix is
        chosen over, so the traces in a panel divide by the same number."""
        parts = [np.asarray(a).ravel() for a in arrays if a is not None]
        parts = [a for a in parts if a.size]
        return np.concatenate(parts) if parts else np.zeros(1)

    if overlay:
        if not shared_x:
            raise ValueError(
                f"overlay=True needs every signal on one x-axis, and these "
                f"are not: {', '.join(repr(a) for a in axes_named)}. A run "
                f"that declares several analyses writes one plot per "
                f"analysis; plot them as separate panels (overlay=False), or "
                f"pass the signals of one analysis at a time.")
        first = flat[0][2]
        x_scale, x_label = _si_scale(_joined([s.x for _, _, s in flat]),
                                     first.xlabel or 'time')
        y_scale, y_label = _si_scale(_joined([s.y for _, _, s in flat]),
                                     first.ylabel or 'v')
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.suptitle(title, fontsize=12)
        for label, corner, sig in flat:
            trace = corner if len(groups) == 1 else f'{label} ({corner})'
            ax.plot(np.asarray(sig.x) / x_scale,
                    np.asarray(sig.y) / y_scale, label=trace)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label if len(groups) == 1 else 'Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        n = len(groups)
        fig, axes = plt.subplots(n, 1, figsize=(10, 2.5 * n + 0.5),
                                 sharex=shared_x)
        fig.suptitle(title, fontsize=12)
        if n == 1:
            axes = [axes]
        x_label = ''
        for ax, (label, traces) in zip(axes, groups):
            first = traces[0][1]
            x_scale, x_label = _si_scale(_joined([s.x for _, s in traces]),
                                         first.xlabel or 'time')
            y_scale, y_label = _si_scale(_joined([s.y for _, s in traces]),
                                         first.ylabel or 'v')
            for corner, sig in traces:
                ax.plot(np.asarray(sig.x) / x_scale,
                        np.asarray(sig.y) / y_scale, label=corner)
            ax.set_ylabel(f'{label}\n{y_label}')
            ax.grid(True, alpha=0.3)
            ax.legend()
            if not shared_x:
                ax.set_xlabel(x_label)
        if shared_x:
            axes[-1].set_xlabel(x_label)

    fig.tight_layout()
    if show:
        plt.show()
    return fig


def _si_scale(values, base_unit):
    """Pick an SI prefix for *values*: returns ``(divisor, axis label)``.

    The engine under :meth:`_ResultsMethods._auto_scale`; separate so a
    caller with several arrays to put on one axis can choose the prefix once
    (over all of them) and divide each by the same number.
    """
    import numpy as np
    # Display-only, SI for humans: 'M' is mega here, and 'm' milli. The
    # SPICE emitter writes 'Meg' and the SPICE parser reads any 'M' as
    # milli — never feed an axis label back into silk.utils._parse_si
    # (its docstring is the contract; TASK-26).
    _PREFIXES = [
        (1e-15, 'f'), (1e-12, 'p'), (1e-9,  'n'), (1e-6, 'u'),
        (1e-3,  'm'), (1,     ''),  (1e3,   'k'), (1e6,  'M'),
        (1e9,   'G'), (1e12,  'T'),
    ]
    _UNIT_LABELS = {
        'time': 'Time', 'frequency': 'Frequency',
        'v': 'Voltage', 'i': 'Current',
    }
    _UNIT_SYMBOLS = {
        'time': 's', 'frequency': 'Hz',
        'v': 'V', 'i': 'A',
    }
    magnitude = float(np.max(np.abs(values))) if len(values) else 1.0
    if magnitude == 0:
        magnitude = 1.0

    # Find best prefix
    scale = 1.0
    prefix = ''
    for threshold, pfx in _PREFIXES:
        if magnitude >= threshold * 0.999:
            scale = threshold
            prefix = pfx

    label_name = _UNIT_LABELS.get(base_unit.lower(), base_unit)
    unit_sym = _UNIT_SYMBOLS.get(base_unit.lower(), '')
    label = f'{label_name} ({prefix}{unit_sym})'

    return scale, label


class _ResultsMethods(_Base):
    """Result-loading / signal / plotting methods mixed into ``Module``. Holds no state."""

    def get_output_signal_list(self):
        """Print a header for the available output signals.

        Raises
        ------
        SimulationError
            If no results file has been loaded. Use :meth:`load_file` first.

        Notes
        -----
        This currently prints only the line ``List of signals`` and returns
        ``None``; use :meth:`get_signal_names` to get the names themselves.
        """
        if not self.output_file_available:
            raise SimulationError('File not loaded. Run load_file first.')
        print('List of signals')

    def load_file(self, dataFile, simulatorType='ngspice'):
        """Point ``read_raw_file()`` at a rawfile.

        A relative path that doesn't exist from the CWD resolves against
        this module's ``dir_folder``. This lets ``load_file('rawfile.raw')``
        find the module's output from another working directory. If the
        named file differs from the in-memory result of the last run, the
        cached result is dropped so ``read_raw_file()`` actually opens the
        file — it used to silently reuse the cache, making this call a
        no-op in-session.

        ``simulatorType`` names the simulator that produced the file. It is a
        hint only — nothing in the rawfile records it — and is used to
        attribute the abort when the file turns out to carry no analysis data: ``read_raw_file()`` hands it to ``backend.load_raw``.

        Parameters
        ----------
        dataFile : str
            Path of the rawfile. A relative path that does not exist from the
            working directory is looked up under ``dir_folder``.
        simulatorType : str, default 'ngspice'
            Name of the simulator that wrote the file; a hint used only to
            attribute an abort when the file carries no analysis data.

        Notes
        -----
        Nothing is read here; the file is opened by the next result access.
        If the path differs from the last run's rawfile, the cached results
        and operating conditions are dropped.
        """
        import os
        path = dataFile
        if not os.path.isabs(path) and not os.path.exists(path):
            candidate = os.path.join(self.dir_folder, path)
            if os.path.exists(candidate):
                path = candidate
        self.dataFile = path
        self._raw_simulator_hint = simulatorType
        raw = getattr(self, '_raw', None)
        if raw is not None:
            current = getattr(raw, 'source_path', None)
            if current is None or os.path.normpath(current) != os.path.normpath(path):
                self._raw = None
                self.operating_conditions = None
                self.characterization_context = None
                self.sim_results_loaded = False
                self.simResultsAll = None
                self.results_file = None

    def get_signal_names(self, *, raw=False):
        """Names of every vector in the loaded results, independent axis first.

        By default array instances are shown in silk's ``[k]`` spelling
        (``i(vbias[0])``, ``v(xx1[0].xxy[1].g[0])``), but only where that is
        collision-safe: names whose bracket spelling would coincide (a net
        literally named ``xx1[0].mid`` beside the arrayed copy's
        ``xx1<0>.mid``) all keep their raw spelling, so no new alias collision
        is introduced (duplicate rows already in the rawfile stay duplicates). Order and count are preserved and the axis is
        untouched. ``raw=True`` returns the rawfile's exact names.

        There is no promise that every returned name reads back uniquely: a
        collision group's names are raw because their bracket spellings are
        ambiguous (:meth:`load_signal` refuses the ambiguous alias).
        """
        if not self.sim_results_loaded:
            raise SimulationError('Simulation results not loaded.')
        names = [self.simResultsAll.vectors[i].name for i in range(self.simResultsAll.nvars)]
        if raw:
            return names
        from silk.spice_names import designer_signal_names
        return designer_signal_names(names)

    def load_signal(self, signalName, signalType='V', *, analysis=None):
        """Load one :class:`~silk.signals.base.Signal` from this module's
        results.

        THE one function every metric value in silk is computed from — the
        callable and declarative ``add_metric`` forms, ``inst.<net>`` attribute
        access, post-run scripts and the STR-001 lint rule all arrive here.

        Parameters
        ----------
        signalName : str
            Rawfile signal name (``'vout'``, ``'xdut.vmid'``).
        signalType : str
            ``'V'`` or ``'I'``.
        analysis : {'tran', 'ac', 'dc', 'noise', 'op'}, optional
            WHICH analysis's plot to read, for a rawfile that holds more than
            one. A file with a single plot needs no argument and is
            unchanged. A file with several and no *analysis* RAISES, naming the
            plots it holds, rather than reading the last one — see
            ``silk.simulation.backend.select_plot``. Behavioral results use
            the same selector and refuse absent analyses. If several plots
            match the named kind, selection is not currently supported by silk
            and is a future extension. Save the intended result in a separate rawfile.

            With ``save_device_params`` on, a bench that declares
            exactly one NON-op analysis gets an extra ``op1`` plot it never
            asked to select between — the rawfile becomes two-plot even
            though the test itself is single-analysis. An omitted *analysis*
            then defaults to that one declared analysis's own word (never to
            ``'op'`` itself — the implicit op-point solve is not a declared
            analysis, and is only ever reachable by naming it explicitly),
            so every existing ``probe()``/``load_signal()`` call written
            before the switch was ever turned on keeps resolving exactly as
            it did with it off.

        Returns
        -------
        Signal
            The waveform. The units of ``x`` and ``y`` depend on the analysis
            and the quantity read: ``x`` is time (s) for transient, frequency
            (Hz) for AC and noise, and the swept quantity for DC (a point
            index placeholder for an operating point); ``y`` is a voltage (V)
            or current (A) for node and branch signals, a density for noise
            vectors, and the device parameter's own unit (for example
            siemens for ``gm``) for device-parameter probes.

        Raises
        ------
        SimulationError
            If no results are loaded, the name is not in the results, or a
            rawfile holding several plots is read without ``analysis``, or
            more than one plot matches the requested analysis kind.
        """
        if not self.sim_results_loaded:
            raise SimulationError('ERROR:  Simulation Results are not Loaded')
        if analysis is None and getattr(self, 'save_device_params', False):
            from silk.circuit.module_test import _declared_analyses
            declared = [w for w in _declared_analyses(self) if w != 'op']
            if len(declared) == 1:
                analysis = declared[0]
        raw = getattr(self, '_raw', None)
        if raw is None:
            from silk.simulation.backend import RawResult
            raw = RawResult(native=self.simResultsAll,
                            source_path=getattr(self, 'results_file', ''))
        from silk.circuit.node_paths import NodeVoltageRequest, resolve_node_request
        signalName = resolve_node_request(self, raw, signalName, analysis)
        from silk.simulation.behavioral_backend import (
            BEHAVIORAL_COMMAND, refuse_behavioral_probe)
        if any(plot.command == BEHAVIORAL_COMMAND
               for plot in getattr(raw.native, 'plots', ())):
            refuse_behavioral_probe(self, signalName, signalType)
        if (':' in signalName and not getattr(signalName, '_literal_voltage', False)
                and not isinstance(signalName, NodeVoltageRequest)):
            prefix, _, suffix = signalName.rpartition(':')
            from silk.simulation.backend import DEVICE_OP_PARAMS
            if suffix in DEVICE_OP_PARAMS:
                return self._load_device_probe(raw, prefix, suffix,
                                               analysis=analysis)
        signalName = _source_current_probe_name(self, raw, signalName,
                                                signalType, analysis)
        return self.backend.load_signal(raw, signalName, type=signalType,
                                        analysis=analysis)

    def _load_device_probe(self, raw, designer_name, param, *, analysis=None):
        """Read-time counterpart of the ``'<name>:<param>'`` probe spelling. Re-resolves the device address against THIS module's
        *current* DUT tree (never cached on the ``Signal`` — a probe can
        never show a stale run, same discipline as every other probe) and,
        for Xyce, re-checks the same capability the before-run
        refusal already applied, so read and emission can never disagree.

        Distinguishes two read-time failure classes: a probe that was never part of THIS run's
        emission (created after the results now loaded were produced) is
        refused HERE, with the lifecycle message; one that WAS requested but
        the backend/version came back missing or zero-length is the
        backend's device-parameter availability message
        (:meth:`~silk.simulation.backend.SimulatorBackend.load_device_probe`).
        """
        if param in NON_PROBEABLE_OP_COLUMNS:
            raise derived_column_probe_error(
                f'{designer_name}:{param}', designer_name, param)
        dut_module = dut_module_from_elem_dict(self.elem_dict)
        from silk.circuit.modules_netlist import _resolve_terminal_current_method
        ammeter_wrapped = (
            bool(getattr(self, 'save_device_currents', False))
            and _resolve_terminal_current_method(self) == 'ammeter_wrapper')
        address = resolve_device_probe_address(
            dut_module, designer_name, param, ammeter_wrapped=ammeter_wrapped)
        if getattr(self.backend, 'name', None) == 'xyce':
            require_xyce_probe_capability(address, param)
        pending = getattr(self, '_pending_device_probes', None) or ()
        if (designer_name, param) not in pending:
            raise SimulationError(
                f"probe({device_probe_spelling(designer_name, param)}): "
                f"this probe was created AFTER the results now loaded were "
                f"produced, so its save/print line was never emitted — "
                f"device-parameter probes must exist BEFORE run() (silk "
                f"requests each one's vector before the simulator runs, "
                f"per its declared analysis). Create tb.probe({designer_name!r} + "
                f"':{param}') again before the next run() and re-run this "
                f"bench so its results include it."
            )
        # ngspice preflight (TASK-330.8, reviewer ruling 2026-09-28, fix B):
        # a param the preflight already found unavailable was never saved
        # in this run at all -- raised here directly, never a rawfile read.
        preflight_unavailable = getattr(
            getattr(self, 'netlister', None), '_preflight_unavailable_params',
            None) or {}
        preflight_result = preflight_unavailable.get((designer_name, param))
        if preflight_result is not None:
            raise self.backend.preflight_unavailable_error(
                raw, address, param, analysis, preflight_result)
        return self.backend.load_device_probe(raw, address, param, analysis=analysis)

    def annotate_currents(self, tstart=None, recursive=True, force=False):
        """Populate per-pin currents on this module's nets from a transient run.

        Signed ``avg``/``rms``/``peak`` land on each
        :class:`~silk.circuit.net.PortBinding` (``+`` = device → net); Net-level
        conservative aggregates (``½·Σ|·|``) are refreshed from the pins.

        Parameters
        ----------
        tstart : float, optional
            Window start (seconds). ``None`` (default) = full trace. A value at
            or past the trace end raises ``ValueError``.
        recursive : bool
            Also annotate submodule instances (their own per-instance ``Net``
            copies), walking ``elem_dict`` MODULE entries with ``x``-prefixing. A worst-case-across-instances reduction is then written onto
            each ``module_dict`` live definition.
        force : bool
            Overwrite pins/nets previously set by the user
            (``_current_source == 'user'``); by default those are skipped.

        Returns
        -------
        CurrentAnnotationReport
            Annotated nets/pins, per-pin skips with reasons, worst KCL residual
            per fully-annotated net, the per-instance annotated ``Module`` map,
            and provenance.

        Notes
        -----

        Read per-instance annotated snapshots from ``report.instance_modules``.
        Shared definitions, including the original module objects, receive the
        worst-case reduction across their placements.

        Raises
        ------
        SimulationError
            Results not loaded (message names ``save_device_currents``);
            behavioral results (no electrical solve); non-transient results; or a
            backend without the current-annotation seam.
        """
        if not self.sim_results_loaded:
            raise SimulationError(
                "No simulation results loaded. Run simulate()/run_simulator() "
                "with `save_device_currents = True` set before netlist "
                "generation, then call annotate_currents().")
        if self._behavioral_mode:
            raise SimulationError(
                "annotate_currents() cannot annotate results from the "
                "'behavioral' backend: device-terminal currents require an "
                "electrical solve, which behavioral models do not perform. "
                "Run a SPICE transient (simulator 'ngspice' or 'xyce') with "
                "save_device_currents = True.")
        from silk.signals.analysis_kind import plot_silk_analysis
        native = getattr(getattr(self, '_raw', None), 'native', getattr(self, 'simResultsAll', None))
        plots = getattr(native, 'plots', ())
        loaded_ac_only = bool(plots) and all(plot_silk_analysis(p) == 'ac' for p in plots)
        if loaded_ac_only or ('ac' in self.sim_analysis_dict and 'transient' not in self.sim_analysis_dict):
            from silk.sources.classes import CurrentSource
            from silk.sources._layout import refuse_ac_current, source_instances
            for _, source, _ in source_instances(self.sig_sources_dict, self.elem_dict):
                if isinstance(source, CurrentSource):
                    refuse_ac_current(source)
        if 'transient' not in self.sim_analysis_dict:
            have = sorted(self.sim_analysis_dict) or ['(none)']
            raise SimulationError(
                "annotate_currents() only supports Transient results; this "
                f"module's analyses are {have}. AC/DC/Noise avg/peak/rms "
                "semantics are undefined (spec §2 non-goals).")

        raw = getattr(self, '_raw', None)
        if raw is None:
            from silk.simulation.backend import RawResult
            raw = RawResult(native=self.simResultsAll,
                            source_path=getattr(self, 'results_file', ''))

        from silk.circuit.modules_netlist import _resolve_terminal_current_method
        method = _resolve_terminal_current_method(self)

        report = CurrentAnnotationReport(
            results_file=getattr(self, 'results_file', ''),
            analysis='transient',
            window=(tstart,),
        )
        try:
            _annotate_level(self, self.backend, raw, '', tstart, force, report, method)
            report.instance_modules[''] = self
            if recursive:
                _annotate_submodules(self, self.backend, raw, '', tstart, force,
                                     report, method)
                _reduce_definitions(self, report)
        except NotSupportedError as e:
            raise SimulationError(
                f"The '{self.simulator}' backend does not support current "
                f"annotation: {e}") from e
        finalize_report_names(report)
        from uuid import uuid4
        import time
        report._observation_id = uuid4().hex
        report._observation_time = time.time()
        from silk.specs.evidence import retain_annotations
        retain_annotations(self, report)
        return report

    @staticmethod
    def _auto_scale(values, base_unit):
        """Pick an SI prefix and return (scaled_values, label).

        E.g. for x-values in seconds ranging 0..100e-9, returns
        (values * 1e9, 'Time (ns)').

        The choice itself lives in :func:`_si_scale`, which returns the
        divisor rather than the divided array — several traces that must
        share one axis (corner overlays) have to be scaled by ONE
        factor chosen across all of them, which this signature cannot
        express. It is frozen (``tests/refactor/test_module_golden.py``), so
        it stays exactly as it is and delegates.
        """
        scale, label = _si_scale(values, base_unit)
        return values / scale, label

    def plot(self, *signal_names, overlay=False, show=True):
        """Quick multi-panel plot of simulation signals.

        Auto-scales axes to human-readable units (ns, mV, uA, etc.)
        and adds proper labels.

        Parameters
        ----------
        *signal_names : str
            Signal names to plot (e.g. ``'vout'``, ``'vin'``, ``'vsupply:I'``).
            Append ``:I`` to plot current instead of voltage.
        overlay : bool
            If True, plot all signals on a single axis. If False (default),
            each signal gets its own subplot panel with a shared x-axis.
        show : bool
            Call ``plt.show()`` after plotting (default True). Set to False
            to customize the figure before showing.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object, for further customization.

        Raises
        ------
        ValueError
            If no signal names are given.
        SimulationError
            If results are not loaded or a signal is not in them.

        Examples
        --------
        ::

            ckt.simulate(corner='tt')
            ckt.plot('vin', 'vout')                # two panels
            ckt.plot('vin', 'vout', overlay=True)   # one panel
            ckt.plot('vout', 'vsupply:I')           # voltage + current
        """
        if not signal_names:
            raise ValueError(
                "No signal names provided. Use ckt.get_signal_names() to list available signals."
            )

        # Load signals
        signals = []
        for name in signal_names:
            if ':I' in name:
                sig_name = name.replace(':I', '')
                signals.append((name, self.load_signal(sig_name, signalType='I')))
            else:
                signals.append((name, self.load_signal(name)))

        return _plot_loaded_signals(self, signals, overlay=overlay, show=show)
