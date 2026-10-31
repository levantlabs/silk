"""Stress rules: STR-001 JUNCTION_OVERRANGE.

This module is the single source of the junction-overrange detection logic;
``silk/circuit/junction_check.py`` is a thin backward-compat wrapper that
dispatches here through the registry (no detection logic of its own).

The hierarchical + SPICE walk (``_spice_nodes`` / ``_spice_stress_violations``)
is lifted from the working reference implementation in
``silk/examples/ldo_regulator_example.py`` (``_ratings_for`` / ``_spice_nodes`` /
``_device_stress``), which solved the two fiddly parts: resolving a submodule
port net UPWARD to its parent's node (ngspice does not re-emit a port net under
the child's ``x<inst>.`` prefix — only genuinely internal nets get prefixed),
and scoring each terminal against *its own* PDK rating by ratio rather than
comparing raw volts across terminals with different ratings.
"""
from silk.circuit.emitted_names import emitted_submodule_name
from typing import Iterator, Optional
import numpy as np
from silk.lint.stress_coverage import DeviceStressCoverage, StressCoverage
from silk.lint.rule import Rule
from silk.lint.severity import Severity
from silk.lint.violations import JunctionViolation
from silk.lint.locations import TerminalLocation
from silk.circuit.modules import Module
from silk.circuit.circuit_elements import ElementType
from silk.elements.base import AnalogElement
from silk.lint.rules.interface import _is_ground_node as _is_spice_ground_node
from silk.pdk.basepdk import parse_junction_rating
# The analysis-kind classifier (complex plot or a 'freq'-named axis =>
# small-signal) lives in silk.signals.analysis_kind so the report layer
# shares it (TASK-82); imported under its original name for callers and
# tests that reference it here.
from silk.signals.analysis_kind import _spice_results_are_small_signal


# Display-only seconds -> ns for violation messages (AUDIT-02).
_S_TO_NS = 1e9

# Thresholds (TASK-32 audit): STR-001 carries NO numeric threshold of its
# own. Every rated limit comes from the PDK catalog
# (``PDKDevice.junction_ranges``, volts, keyed ``<pair>_max``, normalised to
# a signed ``(lo, hi)`` operating range); the rule compares the SIGNED pair
# voltage against that range with no margin of its own. Each catalog number
# is a foundry-stated operating limit with its source (or derivation rule)
# recorded beside it in ``junction_ratings_provenance`` and enforced by
# ``silk.pdk.validate`` — the convention is written once in
# docs/adding-a-pdk.md ("Junction ratings"); audit F-05 / TASK-48.
#
# Sign (TASK-91): the pair named ``Vxy`` is ``V(x) - V(y)`` — the first
# pin minus the second in the tables below, which is also the order
# ``_compute_junction_pairs`` subtracts in and the convention a signed
# catalog range (``Vbs_max: {min: -1.95, max: 0.3}``) is written in
# (``silk.pdk.basepdk.JUNCTION_PAIR_SIGN_CONVENTION``). Before TASK-91 the
# rule compared ``|V| > m``, so a rating was one magnitude per pair and an
# asymmetric published range (sky130 NMOS VBS +0.3 V forward / -1.95 V
# reverse) could only store its larger side: a body forward-biased by 1 V
# was silently in range.

_MOSFET_PAIRS = [('g', 's', 'Vgs'), ('d', 's', 'Vds'), ('g', 'd', 'Vgd'),
                 ('b', 's', 'Vbs'), ('b', 'd', 'Vbd')]
_DIODE_PAIRS  = [('a', 'k', 'Vak')]
_BJT_PAIRS    = [('b', 'e', 'Vbe'), ('b', 'c', 'Vbc'), ('c', 'e', 'Vce')]

# HEURISTIC (name-based, no electrical check): a terminal on one of these
# net names is taken as 0 V when no waveform exists for it. ``gnd``/``vss``
# are the two names the examples and the SPICE-path fixtures use for the
# simulator's ground node. A ground net under any other name that has no
# saved waveform makes its device UNRESOLVABLE on the SPICE path (skipped,
# never scored) or its terminal pair uncomputable on the behavioral path
# (that pair skipped, other pairs on the same device still score) — see
# ``_is_ground_node``.
#
# TASK-85's ruling keeps this HEURISTIC (name-based) set out of
# ``silk/lint/rules/interface.py``: only the literal SPICE ground node
# ``'0'`` is backend fact, sourced from there as ``_is_spice_ground_node``
# (``interface.SPICE_GROUND_NODE``) rather than re-spelled here; ``gnd``/
# ``vss`` stay this module's own naming convention.
#
# TASK-50: this is now the ONE ground-name predicate for STR-001, used by
# both the SPICE-path walk (``_spice_stress_violations``) and the
# behavioral-path walk (``_check_behavioral``); before TASK-50 the
# behavioral path had its own narrower ``('gnd', '0')`` check inline,
# so a body tied to ``vss`` with no stored waveform was silently
# uncomputable there while the SPICE path already treated it as 0 V.
_GROUND_HEURISTIC_NAMES = {'gnd', 'vss'}


def _compute_junction_pairs(pin_nets: dict, get_v) -> dict:
    """Return {pair_name: voltage_array} for all computable terminal pairs."""
    result: dict = {}
    pins = set(pin_nets.keys())

    if {'d', 'g', 's', 'b'}.issubset(pins):
        pairs = _MOSFET_PAIRS
    elif {'a', 'k'}.issubset(pins):
        pairs = _DIODE_PAIRS
    elif {'c', 'b', 'e'}.issubset(pins):
        pairs = _BJT_PAIRS
    else:
        return result

    for pin_a, pin_b, name in pairs:
        v_a = get_v(pin_a)
        v_b = get_v(pin_b)
        if v_a is not None and v_b is not None:
            result[name] = v_a - v_b

    return result


def _worst_point_label(xlabel: Optional[str], value: float) -> str:
    """Render the worst-case point on whatever axis the data is swept over.

    A transient trace gets the familiar ``t=<n>ns``; a DC sweep gets its own
    axis and value (``v(v-sweep)=3.000``) instead of a sweep voltage printed
    as nanoseconds. No unit is invented for a non-time axis — the axis name
    already carries it, and a `.dc` sweep can be over temperature.
    """
    if 'time' in (xlabel or 'time').lower():
        return f't={value * _S_TO_NS:.2f}ns'
    return f'{xlabel}={value:.3f}'


def _is_ground_node(node: str) -> bool:
    """True if *node* names a ground/supply-return pseudo-node.

    Checked both as the whole (resolved, possibly hierarchy-prefixed) node
    name and as its last dotted segment, since an internal ground net inside
    a submodule would be prefixed (``xamp.gnd``) while a port-resolved one
    generally is not (it resolves straight to the parent's ``0``/``gnd``).
    The literal SPICE ground node ``'0'`` is delegated to
    ``interface._is_ground_node`` (``SPICE_GROUND_NODE``) rather than
    re-spelled here;
    ``gnd``/``vss`` are this module's own naming heuristic.

    Shared by both STR-001 data paths: the SPICE path calls this
    on a resolved, possibly ``x``-prefixed rawfile node name, the behavioral
    path calls it on a plain (never hierarchy-prefixed) net name — the
    dotted-segment split is then a no-op, so one predicate covers both.
    """
    node = node.strip()
    tail = node.rsplit('.', 1)[-1]
    if _is_spice_ground_node(node) or _is_spice_ground_node(tail):
        return True
    return node.lower() in _GROUND_HEURISTIC_NAMES or tail.lower() in _GROUND_HEURISTIC_NAMES


def _ranges_for(dev) -> Optional[dict]:
    """``{'<pair>_max': (lo, hi)}`` for a catalog device, or ``None`` if unrated.

    Reads ``PDKDevice.junction_ranges`` (the loader's normalised form). A
    device object that carries only ``junction_ratings`` — the legacy
    shape, still what hand-built fixtures and mocks provide — has its
    magnitudes expanded through the ONE shorthand parser
    (``parse_junction_rating``: ``m`` -> ``(-m, m)``), so both shapes score
    identically and the shorthand rule is never re-spelled here.
    """
    if dev is None:
        return None
    ranges = getattr(dev, 'junction_ranges', None)
    if isinstance(ranges, dict) and ranges:
        return ranges
    ratings = getattr(dev, 'junction_ratings', None)
    if not ratings:
        return None
    return {k: parse_junction_rating(k, m) for k, m in ratings.items()}


def _ratings_for(pdk, type_name: str) -> Optional[dict]:
    """Per-terminal rated ranges for a PDK device type, straight from the PDK."""
    return _ranges_for(getattr(pdk, '_devices', {}).get(type_name))


def _score_pair(v_array, lo: float, hi: float):
    """Worst excursion of a signed pair voltage outside ``[lo, hi]``.

    Returns ``(worst_idx, measured_v, rated_bound, verb)`` or ``None`` when
    every sample is in range. The excess at a sample is
    ``max(v - hi, lo - v)``; the worst sample is its argmax and the pair is
    over-range when that excess is > 0 (no margin of the rule's own).

    Reporting keeps the two catalog forms distinguishable in the finding:

    * a SYMMETRIC range ``(-m, m)`` — the magnitude form — reports exactly
      what STR-001 reported before ``measured_v = |v|`` at the
      worst sample (``argmax(|v| - m)`` is ``argmax(|v|)``, same sample,
      same first-of-ties) and ``rated_bound = m``, verb ``exceeds``;
    * an ASYMMETRIC range reports the SIGNED ``v`` and the bound it crossed
      (``hi`` with ``exceeds`` when ``v > hi``; ``lo`` — negative — with
      ``below`` when ``v < lo``), because there ``|v|`` would hide which
      limit was broken: +1.0 V on a sky130 NMOS body is 0.7 V past its
      0.3 V forward limit, not 0.95 V inside the 1.95 V reverse one.
    """
    if lo == -hi:
        abs_v = np.abs(v_array)
        worst_idx = int(np.argmax(abs_v))
        worst_v = float(abs_v[worst_idx])
        if worst_v > hi:
            return worst_idx, worst_v, float(hi), 'exceeds'
        return None
    excess = np.maximum(v_array - hi, lo - v_array)
    worst_idx = int(np.argmax(excess))
    if not float(excess[worst_idx]) > 0.0:
        return None
    worst_v = float(v_array[worst_idx])
    if worst_v > hi:
        return worst_idx, worst_v, float(hi), 'exceeds'
    return worst_idx, worst_v, float(lo), 'below'


def _bound_label(pair_name: str, verb: str) -> str:
    """``Vbs_max`` for a crossed upper bound, ``Vbs_min`` for a lower one."""
    return f'{pair_name}_max' if verb == 'exceeds' else f'{pair_name}_min'


def _spice_nodes(mod, prefix: str = "", port_map: Optional[dict] = None,
                  silk_prefix: str = ""):
    """Walk the module tree; yield ``(instance_path, device, {pin: spice_node}, type_name)``.

    This is the *hierarchical* part, lifted from
    ``ldo_regulator_example.py::_spice_nodes``. A stress check needs each
    device's TERMINAL NODES resolved to rawfile names, and that resolution
    differs by net kind:

      * an **internal** net of a submodule is prefixed  (``n2`` -> ``xamp.n2``);
      * a **port** net is electrically the parent's net, and ngspice does not
        re-emit it under the child's prefix — so it must be resolved UPWARD.

    ``Net.pins`` carries ``PortBinding(instance, port, kind='submodule')``,
    which is exactly the child-port -> parent-net map needed to do that.: the walk needs TWO independent hierarchy prefixes,
    which is why ``prefix`` and ``silk_prefix`` are threaded separately
    rather than derived from one another:

      * ``prefix`` is the SPICE-form prefix (``xamp``, with ngspice's ``x``
        instance-name prefix) — used ONLY inside ``node_of`` to build rawfile
        node names, because that is the literal string ngspice put in the
        rawfile.
      * ``silk_prefix`` is the silk-form prefix (``amp``, no ``x``) — used
        for the yielded ``instance_path``, because that is the path form
        ``ScopeExclusion``/waivers/``generate_report`` exclusions and the
        behavioral path both key on. Reporting ``prefix`` here (the pre-fix
        behavior) made a subtree exclusion on ``amp`` never match a finding
        the walk reported as ``xamp.mn1``.

    The two never need to match: node resolution (rawfile lookup) and
    instance identification (exclusions/waivers/diff) are different
    concerns that happen to share the same walk.
    """
    port_map = port_map or {}

    def node_of(local_net):
        if local_net in port_map:            # a port: use the parent's resolved node
            return port_map[local_net]
        return f"{prefix}.{local_net}" if prefix else local_net

    for name, entry in mod.elem_dict.items():
        etype = entry.get("type")
        dev = entry.get("device")
        if dev is None:
            continue
        if etype == ElementType.DEVICE:
            nets = getattr(dev, "nets", None)
            if not nets:
                continue
            instance_path = f"{silk_prefix}.{name}" if silk_prefix else name
            yield (instance_path, dev, {pin: node_of(net) for pin, net in nets.items()},
                   entry.get("typeName", ""))
        elif etype == ElementType.MODULE:
            # Build the child's port map: child port -> node in THIS module.
            child_map = {}
            for netname, net in mod.net_dict.items():
                for pb in net.pins:
                    if getattr(pb, "kind", None) == "submodule" and pb.instance == name:
                        child_map[pb.port] = node_of(netname)
            child_prefix = f"{prefix}.{emitted_submodule_name(name)}" if prefix else emitted_submodule_name(name)
            child_silk_prefix = f"{silk_prefix}.{name}" if silk_prefix else name
            yield from _spice_nodes(dev, child_prefix, child_map, child_silk_prefix)


def _resolve_waveform(module, node: str):
    """Load a node's voltage waveform via ``module.load_signal``.

    Tries the resolved node as-is first, then case variants — ``load_signal``
    (via ``NgspiceSignal.from_rawfile``) already matches node names
    case-insensitively, so this is mostly defensive: it also means a
    truly-unresolvable node (bad hierarchy resolution, a stripped/renamed
    internal net, etc.) is reported as "no waveform" rather than raising, so
    the caller can skip just that device instead of crashing the whole rule.

    Each candidate is asked for TWICE since bare first (a
    single-plot rawfile, the unchanged path), then ``analysis='tran'`` for a
    multi-analysis rawfile, where a bare load refuses rather than guessing a
    plot. STR-001 scores node voltages against junction ratings, so the
    transient plot is the only one it can read — and naming it here is what
    keeps the rule working on a run that also declared an .ac, where the bare
    load would otherwise be swallowed as "no waveform" and the whole device
    silently skipped.
    """
    for cand in dict.fromkeys((node, node.lower(), node.upper())):
        for select in ({}, {'analysis': 'tran'}):
            try:
                return module.load_signal(cand, 'V', **select)
            except Exception:
                continue
    return None


def _behavioral_devices(module, prefix=""):
    """Enumerate the hierarchy; only top-level behavioral terminals are readable.

    A modeled leaf replaces its internals, so its transistors must remain
    visible as unchecked rather than borrowing unrelated top-level voltages.
    """
    for name, entry in module.elem_dict.items():
        device = entry.get('device')
        if device is None:
            continue
        path = f'{prefix}.{name}' if prefix else name
        if entry.get('type') == ElementType.MODULE:
            yield from _behavioral_devices(device, path)
            continue
        if entry.get('type') not in (ElementType.DEVICE, 'device'):
            continue
        nodes = (device.nets if isinstance(device, AnalogElement) else
                 {p: pin.net for p, pin in getattr(device, 'pin_list', {}).items()})
        yield path, device, nodes, entry.get('typeName', ''), not prefix


def _device_stress_outcomes(module, rule_id, severity, *, skipped=False):
    """Yield (coverage, violations) from ONE resolver/scorer pass per device.

    SPICE retains its whole-device availability gate; behavioral retains its
    pair-by-pair gate. Coverage records those exact decisions, not a second
    measurement with different availability semantics.
    """
    behavioral = bool(getattr(module, '_behavioral_mode', False))
    small_signal = not behavioral and _spice_results_are_small_signal(module)
    available = not skipped and (behavioral or getattr(module, 'sim_results_loaded', False))
    devices = (_behavioral_devices(module) if behavioral else
               ((*record, True) for record in _spice_nodes(module)))
    pdk = getattr(module, 'pdk_details', None)
    for path, dev, nodes, type_name, readable in devices:
        if type_name not in getattr(pdk, '_devices', {}):
            continue
        device_available = available and readable
        ratings = _ratings_for(pdk, type_name) or {}
        # Enumerate even unmeasurable pairs, using the same pin vocabulary.
        pair_names = _compute_junction_pairs(nodes, lambda pin: np.zeros(1))
        pairs = {name: 'unavailable' if f'{name}_max' in ratings else 'unrated'
                 for name in pair_names}
        reasons = []
        if not ratings or any(value == 'unrated' for value in pairs.values()):
            reasons.append('no_ratings')
        waves = {}
        t = getattr(module, '_behavioral_t', None) if behavioral else None
        xlabel = 'time' if behavioral else None
        if small_signal:
            reasons.append('small_signal')
        elif not device_available or (behavioral and t is None):
            reasons.append('no_waveform')
        elif ratings:
            for pin, node in nodes.items():
                if node is None:
                    reasons.append('no_waveform')
                    continue
                if behavioral:
                    net = module.net_dict.get(node)
                    y = getattr(net, 'voltage', None) if net is not None else None
                    if y is None and _is_ground_node(node):
                        waves[pin] = None
                        continue
                    sig = None
                else:
                    if _is_ground_node(node):
                        waves[pin] = None
                        continue
                    sig = _resolve_waveform(module, node)
                    y = sig.y if sig is not None else None
                if y is None:
                    reasons.append('no_waveform')
                    continue
                arr = np.asarray(y)
                if np.iscomplexobj(arr):
                    reasons.append('small_signal')
                    continue
                if t is None:
                    t = np.asarray(sig.x, dtype=float)
                    xlabel = getattr(sig, 'xlabel', None)
                if arr.ndim != 1 or len(arr) != len(t):
                    reasons.append('length_mismatch')
                    continue
                if not len(arr):
                    reasons.append('no_waveform')
                    continue
                waves[pin] = arr.astype(float, copy=False)
            if nodes and all(node is not None and _is_ground_node(node)
                             for node in nodes.values()):
                reasons.append('all_ground')
            if t is not None:
                waves = {pin: np.zeros_like(t) if arr is None else arr
                         for pin, arr in waves.items()}
        blocked = (small_signal or not device_available or t is None or not ratings or
                   (not behavioral and any(r != 'no_ratings' for r in reasons)))
        violations = []
        if not blocked:
            for name, v_array in _compute_junction_pairs(nodes, waves.get).items():
                if f'{name}_max' not in ratings:
                    continue
                lo, hi = ratings[f'{name}_max']
                hit = _score_pair(v_array, lo, hi)
                pairs[name] = 'fail' if hit is not None else 'pass'
                if hit is None:
                    continue
                worst_idx, worst_v, rated_bound, verb = hit
                violations.append(JunctionViolation(
                    rule_id=rule_id, severity=severity,
                    message=(f"{path} ({type_name}): {name}={worst_v:.3f}V "
                             f"{verb} {_bound_label(name, verb)}={rated_bound}V "
                             f"at {_worst_point_label(xlabel, float(t[worst_idx]))}"),
                    location=TerminalLocation(instance_path=path, terminal=name,
                                              terminal_pair=(name[1].lower(), name[2].lower())),
                    device_type=type_name, measured_v=worst_v,
                    rated_max=rated_bound, worst_time=float(t[worst_idx]),
                ))
        checked = sum(value in ('pass', 'fail') for value in pairs.values())
        coverage = 'none' if not checked else 'full' if checked == len(pairs) else 'partial'
        yield DeviceStressCoverage(path, type_name, bool(violations), coverage,
                                   tuple(dict.fromkeys(reasons)), pairs), violations


def _evaluate_stress(module, rule_id='STR-001', severity=Severity.ERROR, *, skipped=False):
    coverage = StressCoverage()
    violations = []
    for device, findings in _device_stress_outcomes(module, rule_id, severity, skipped=skipped):
        coverage.devices.append(device)
        violations.extend(findings)
    return coverage, violations


def _spice_stress_violations(module, rule_id, severity):
    yield from _evaluate_stress(module, rule_id, severity)[1]


class JunctionOverrangeRule(Rule):
    rule_id = "STR-001"
    category = "stress"
    default_severity = Severity.ERROR
    applicable_to = (Module,)
    requires_simulation = True
    rationale = (
        "A device terminal voltage (Vgs, Vds, Vgd, Vbs, or Vbd) exceeds the "
        "PDK-rated maximum at some point during the simulated waveform. "
        "Gate-oxide reliability is governed by time-dependent dielectric "
        "breakdown (TDDB), whose MTTF decreases exponentially with applied "
        "voltage above Vmax. PDK ratings already include a margin for "
        "sustained operation, so exceeding them — even briefly — materially "
        "reduces device lifetime. The violation reports the worst-case "
        "magnitude and the timestamp at which it occurred so the designer "
        "can correlate to the testbench event. Fix by adding clamps, "
        "re-sizing, or cascoding."
    )

    def simulation_data_usable(self, module) -> bool:
        """Decline AC/spectral SPICE results.

        Behavioral results are always a time-domain walk, so they are usable
        by construction; only the SPICE path can hand this rule a plot whose
        "voltages" are small-signal transfer ratios or noise densities.
        """
        if getattr(module, '_behavioral_mode', False):
            return True
        return not _spice_results_are_small_signal(module)

    def check(self, module: Module) -> Iterator[JunctionViolation]:
        yield from _evaluate_stress(module, self.rule_id, self.severity)[1]

    def check_with_context(
        self, module: Module, *, skipped: bool = False,
    ) -> tuple[list[JunctionViolation], dict]:
        """Registry hook: findings and device coverage from the same pass."""
        coverage, violations = _evaluate_stress(
            module, self.rule_id, self.severity, skipped=skipped)
        return violations, {'stress_coverage': coverage.to_dict()}
