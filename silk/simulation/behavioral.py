"""Behavioral simulation engine for silk.

Evaluates a Module circuit entirely in Python/numpy — no ngspice required.
Each subcircuit with a behavioral model attached via Module.set_behavioral_model()
is evaluated in driver order determined by PortDir port declarations, with
feedback members in placement order.

The engine walks the hierarchy: a model is a leaf. An instance whose module
has a behavioural model is one block, and its model stands in for everything inside it. An instance without
one is entered: its ports are its parent's nets, and its other nets take the
name SPICE gives them (``xdut.vcomp``).
"""

from silk.circuit.emitted_names import emitted_submodule_name
import warnings
from dataclasses import dataclass

import numpy as np

from silk.simulation._analysis_context import analysis_scope, analysis
from silk.simulation._behavioral_feedback import schedule

from silk.circuit.circuit_elements import ElementType, PortDir, PortRole
from silk.circuit.modules_results import _is_ideal_passive
from silk.exceptions import NotSupportedError, SimulationError, LoopError
from silk.sources.classes import VoltageSource, CurrentSource, VoltageControlledVoltageSource
from silk.simulation._behavioral_state import SharedState
from silk.simulation._behavioral_views import PortsView
from silk.simulation._behavioral_elements import (
    AlgebraicModel, classify_passives, orient_voltage_sources, refuse, vcvs_block,
    validate_voltage_references, source_refuse,
)

# Ground nets are global: an entered instance's ``gnd`` is the bench's.
_GLOBAL_NETS = ('gnd', '0')


@dataclass
class _Block:
    """One modelled instance or algebraic element of the flattened hierarchy.

    ``path`` is the instance's name as the engine reports it (``'comp1'``,
    ``'xdut.comp1'``). ``pins`` maps each port to ``(net, direction)``, with
    the net already resolved to its bench-level name.
    """
    path: str
    device: object
    pins: dict
    segments: tuple = ()
    mismatch: bool = True


def _time_grid(t_step, t_stop):
    """Uniform grid from 0 to ``t_stop`` inclusive.

    The last point is exactly ``t_stop``; when ``t_stop`` is not a multiple of
    ``t_step`` the final interval is shorter.
    """
    n = int(np.ceil(t_stop / t_step - 1e-9))
    t = np.arange(n + 1) * t_step
    t[-1] = t_stop
    return t


class BehavioralEngine:
    """Evaluates a Module circuit in Python using numpy.

    Each independent ``run()`` or ``run_dc()`` call starts a fresh logical
    run, using the current models and parameters and fresh per-placement
    callable state. Constructor ``scope`` and ``sample`` are preserved.
    Analyses within one backend launch share their attachment snapshot,
    callable state and resolved variation. Declare analyses in one testbench
    run when they should share that state. Callable closure cells and
    captured Python objects are copied: writes affect silk's copy. Module-level
    globals are not reset.

    Usage::

        engine = BehavioralEngine()
        results, t = engine.run(top_module, t_step=1e-9, t_stop=100e-9)
        # results: dict[net_name, np.ndarray]
        # t: np.ndarray of time points
    """

    def __init__(self, *, scope=None, sample=None):
        self.scope = scope
        self.sample = sample
        self.resolution = None
        self.source_currents = {}
        self.state = None
        self._attachments = None
        self._port_metadata = {}
        self._param_views = {}
        self._writers = ({}, {})
        # One backend launch shares setup and notices across its analyses.
        self._warned_notes = None
        self._feedback_warned = False
        self._shared_run = False

    def _begin_run(self, module):
        """Start independent calls with fresh run state; preserve shared launches."""
        if not self._shared_run:
            self.resolution = None
            self._attachments = None
            self._port_metadata = {}
            self._param_views = {}
            self._writers = ({}, {})
            self._warned_notes = None
            self._feedback_warned = False
            self.state = None
            self.source_currents = {}
        module._behavioral_element_notes = ()

    @analysis_scope('transient')
    def run(self, module, t_step, t_stop):
        """Simulate module behaviorally.

        Parameters
        ----------
        module : Module
            Top-level module with sources and behavioral subcircuits.
        t_step : float
            Simulation time step in seconds.
        t_stop : float
            Simulation stop time in seconds.

        Returns
        -------
        results : dict
            Net name → signal array. Nets inside an entered instance are
            keyed by their hierarchical name (``'xdut.vcomp'``).
        t : np.ndarray
            Time vector.
        """
        self._begin_run(module)
        # Walk first: every static refusal is raised before any model runs.
        blocks, sources = self._flatten(module)
        if self.resolution is None:
            from silk.simulation._behavioral_variation import resolve_run
            self.resolution = resolve_run(blocks, scope=self.scope, sample=self.sample)

        if self._attachments is None:
            from silk.simulation._behavioral_models import attachments
            self._attachments = attachments(blocks, self.resolution)
        self._prepare_views(blocks)
        t = _time_grid(t_step, t_stop)

        net_signals = self._generate_source_signals(sources, t, blocks=blocks)
        graph = self._build_graph(blocks, net_signals, t)
        order = schedule(graph, self._attachments or {}, dc=False)
        self._evaluate_blocks(blocks, t, order, net_signals)

        return net_signals, t

    @analysis_scope('dc')
    def run_dc(self, module, analysis):
        """Evaluate models once per source value, in sweep order.

        Vector models receive one-sample arrays; sample models
        receive scalar values. `t` is the increasing point index, not physical
        time. Callable state persists between points; electrical state is fresh
        at each point. No settling or electrical solve is implied. Unswept
        independent sources hold their DC values.
        """
        self._begin_run(module)
        from silk.sources.classes import CurrentSource, VoltageSource
        from silk.utils._utils import _parse_si

        blocks, sources = self._flatten(module)
        if self.resolution is None:
            from silk.simulation._behavioral_variation import resolve_run
            self.resolution = resolve_run(blocks, scope=self.scope, sample=self.sample)
        if self._attachments is None:
            from silk.simulation._behavioral_models import attachments
            self._attachments = attachments(blocks, self.resolution)
        self._prepare_views(blocks)
        target, ambiguous = self._dc_target(sources, analysis)
        if ambiguous:
            warnings.warn(ambiguous, UserWarning, stacklevel=2)

        start, stop, step = (_parse_si(value) for value in
                             (analysis.start, analysis.stop, analysis.step))
        intervals = (stop - start) / step
        # Unlike transient, DC never adds a shorter final interval.
        count = int(np.floor(intervals + 1e-9)) + 1
        sweep = start + np.arange(count) * step
        if np.isclose(intervals, count - 1, rtol=0, atol=1e-9):
            sweep[-1] = stop
        t = np.array([0.0])
        initial = {path: src.initial_value() for path, src, _, _ in sources
                   if path != target[0]}
        waveform_start = _parse_si(target[1].waveform_initial_value())
        values = {**initial, target[0]: start + waveform_start}
        base = self._generate_source_signals(sources, t, blocks=blocks, values=values)
        order = schedule(self._build_graph(blocks, base, t),
                         self._attachments or {}, dc=True)
        results = {}
        for index, value in enumerate(sweep):
            values[target[0]] = value + waveform_start
            signals = self._generate_source_signals(sources, t, blocks=blocks, values=values)
            # Preserve unrelated graph defaults in results, not in the set of
            # actual voltage producers used by algebraic source operations.
            for net, arr in base.items():
                signals.setdefault(net, arr.copy())
            self._evaluate_blocks(blocks, np.array([float(index)]), order, signals)
            if index and signals.keys() != results.keys():
                raise SimulationError(
                    "behavioural DC model changed its output nets between "
                    "sweep points; assign every output at every point.")
            for net, arr in signals.items():
                try:
                    sample = np.broadcast_to(np.asarray(arr, dtype=float), (1,))[0]
                except ValueError:
                    raise SimulationError(
                        f"behavioural DC result for net {net!r} has shape "
                        f"{np.shape(arr)}; assign one value per sweep point "
                        f"(or a scalar).") from None
                if index == 0:
                    results[net] = np.empty(count)
                results[net][index] = sample
        axis_type = 'current' if isinstance(target[1], CurrentSource) else 'voltage'
        return results, sweep, axis_type

    @staticmethod
    def _dc_target(sources, analysis):
        """Resolve a DC sweep's source, raising its static refusals.

        Returns ``(entry, ambiguity)``; *ambiguity* is the notice text when
        the emitted and designer names pick different sources, else ``None``.
        Pure: the caller decides whether to warn, so the read-only setup
        check (:func:`silk.simulation.validate_simulation_setup`) shares the
        run's refusals without emitting its notices.
        """
        name = str(analysis.source)
        if name.lower() == 'temp':
            raise NotSupportedError(
                "a behavioural DC sweep cannot sweep ambient temperature; "
                "select an independent source or use a SPICE backend.")
        independent = [entry for entry in sources
                       if isinstance(entry[1], (VoltageSource, CurrentSource))]
        # Match the DC netlister: emitted names first, then designer names.
        emitted = {path.removesuffix(src.name) + src.instance_name: entry
                   for entry in independent for path, src, _, _ in [entry]}
        target = next((entry for key, entry in emitted.items()
                       if key.lower() == name.lower()), None)
        designer = next((entry for entry in independent if entry[0] == name), None)
        if designer is None:
            designer = next((entry for entry in independent
                             if entry[0].lower() == name.lower()), None)
        ambiguity = None
        if target is not None and designer is not None and target != designer:
            ambiguity = (f"DC source {name!r} is ambiguous; resolved to source "
                         f"{target[0]!r} (emitted names win for DC).")
        target = target if target is not None else designer
        if target is None:
            raise SimulationError(
                f"behavioural DC sweep has no independent source {name!r}; "
                f"available sources: {[entry[0] for entry in independent]}.")
        return target, ambiguity

    def _flatten(self, module, *, report=True):
        """Walk the hierarchy under *module* ("a model is a leaf").

        ``report=False`` discovers declarations during preflight without
        emitting approximation notices or changing the run's evidence.

        Returns ``(blocks, sources)``: ``blocks`` maps each modelled
        instance's scheduler key to its :class:`_Block`, in hierarchy order
        (algebraic keys are registry-qualified to avoid name collisions), and
        ``sources`` lists independent sources as ``(name, source, positive_net, negative_net)``
        with the nets resolved to bench-level names.

        Raises
        ------
        NotSupportedError
            An unsupported element or topology with no model above it.
        SimulationError
            An entered instance with nothing inside it to evaluate.
        CircuitError
            An entered or modeled-leaf interface declares formal ``0`` or
            case-insensitive ``gnd``. Rename the formal to ``vss`` and wire
            it to parent ground; actual net handling is unchanged.
        """
        from silk.circuit.ground_port import _require_entered_ground_interfaces
        blocks = {}
        sources = []
        passives = []
        controls = []
        rails = set(_GLOBAL_NETS)

        def passive(path, device, resolve):
            passives.append((path, device._spice_prefix,
                             resolve(device.nets['p']), resolve(device.nets['n'])))

        def visit(mod, prefix, resolve, inst_path, segments=(), mismatch=True):
            _require_entered_ground_interfaces(mod)
            for name, net in mod.net_dict.items():
                if net.classification in (PortRole.POWER, PortRole.GROUND):
                    rails.add(resolve(name))
            for port, pin in (mod.pin_list or {}).items():
                if pin.direction in (PortDir.POWER, PortDir.GROUND):
                    rails.add(resolve(port))
            supply_source = getattr(mod, '_supply_source', None)
            if supply_source is None and getattr(mod, '_supply_voltage', None) is not None:
                from silk.simulation.conditions import prepare_supply
                supply_source = prepare_supply(mod, apply=False).source
            for src_name, src in mod.sig_sources_dict.items():
                path = prefix + src_name
                if isinstance(src, VoltageControlledVoltageSource):
                    device, pins = vcvs_block(src, resolve)
                    controls.append((path, device, pins))
                elif isinstance(src, (VoltageSource, CurrentSource)):
                    positive, negative = resolve(src.positive), resolve(src.negative)
                    sources.append((path, src, positive, negative))
                    if src_name == supply_source:
                        rails.add(positive)
                else:
                    refuse(path, f'{type(src).__name__} needs a SPICE electrical solve')
            for name, device in mod.passives_dict.items():
                passive(prefix + name, device, resolve)

            for inst_name, entry in mod._element_mgr.elem_dict.items():
                device = entry['device']
                path = prefix + inst_name
                if entry['type'] == ElementType.DEVICE:
                    if _is_ideal_passive(device):
                        passive(path, device, resolve)
                        continue
                    raise NotSupportedError(_transistor_message(path, inst_path))

                pins = {
                    port: (resolve(pin.net), pin.direction)
                    for port, pin in (entry.get('pin_list', device.pin_list) or {}).items()
                }
                rails.update(net for net, direction in pins.values()
                             if direction in (PortDir.POWER, PortDir.GROUND))
                if device._behavioral_model is not None:
                    # A model is a leaf: it stands in for everything inside
                    # the block, transistors included (the twin case).
                    blocks[path] = _Block(path, device, pins, segments + (inst_name,),
                                          mismatch and device.mismatch_enabled)
                    continue
                if not (device._element_mgr.elem_dict or device.sig_sources_dict
                        or device.passives_dict):
                    raise SimulationError(
                        f"Block '{path}' (module '{device.module_name}') has no "
                        f"behavioral model and nothing inside it to evaluate. "
                        f"Call set_behavioral_model() on the subcircuit before "
                        f"simulate_behavioral().")
                child_prefix = f'{prefix}{emitted_submodule_name(inst_name)}.'
                ports = {port: net for port, (net, _) in pins.items()}
                visit(device, child_prefix, _resolver(child_prefix, ports),
                      child_prefix[:-1], segments + (inst_name,),
                      mismatch and device.mismatch_enabled)

        visit(module, '', lambda net: net, None, mismatch=module.mismatch_enabled)
        for path, device, pins in orient_voltage_sources(
                controls, blocks, sources, rails, passives):
            blocks[('source', path)] = _Block(path, device, pins)
        try:
            algebraic, notes = classify_passives(passives, blocks, sources, rails)
        except NotSupportedError as exc:
            # Keep the passive diagnosis and identify affected source equations.
            equations = [b.device for b in blocks.values()
                         if isinstance(b.device, AlgebraicModel) and b.device.terminals]
            if equations:
                detail = '; '.join(f"{d.path}: {d.terminals!r}" for d in equations)
                raise NotSupportedError(f"{exc} Source terminals: {detail}") from None
            raise
        for key, path, device, pins in algebraic:
            blocks[('wire', key)] = _Block(path, device, pins)
        validate_voltage_references(blocks)
        if report:
            module._behavioral_element_notes = notes
        if report and notes and notes != self._warned_notes:
            self._warned_notes = notes
            warnings.warn('Behavioral ideal-element approximations: ' + '; '.join(notes),
                          UserWarning, stacklevel=3)
        return blocks, sources

    def _generate_source_signals(self, sources, t, *, blocks, values=None):
        """Evaluate stimuli once; record branch currents and stage voltage differences."""
        from silk.sources import CurrentSource
        from silk.utils._utils import _parse_si, _resolve_value
        self.source_currents = {}
        net_signals = {
            'gnd': np.zeros_like(t),
            '0':   np.zeros_like(t),
        }

        pins = {(block.path, port): net for block in blocks.values()
                if not isinstance(block.device, AlgebraicModel)
                for port, (net, _) in block.pins.items()}
        for path, source, positive, negative in sources:
            if isinstance(source, CurrentSource):
                pins[(('current-source', path), 'positive')] = positive
                pins[(('current-source', path), 'negative')] = negative
        self.state = SharedState(len(t), pins, net_signals, writers=self._writers)

        voltage_models = {block.device.path: block.device for block in blocks.values()
                          if isinstance(block.device, AlgebraicModel)
                          and block.device.independent}
        for src_name, src, positive, negative in sources:
            waveform = src.to_numpy(t) if values is None else np.full(len(t), values[src_name])
            if isinstance(src, CurrentSource):
                prefix = src_name.removesuffix(src.name)
                name = prefix + src.instance_name
                # Own the evaluated source waveform: later model writes to
                # its net must never change the branch-current result.
                self.source_currents[name] = np.array(waveform, copy=True)
                self.state.write_current(('current-source', src_name), 'positive', waveform)
                self.state.write_current(('current-source', src_name), 'negative', -waveform)
                layout = src.physical_layout()
                if len(layout.names) == 2:
                    dc = _parse_si(_resolve_value(src.dc, f'Source {src.name!r} dc'))
                    self.source_currents[prefix + layout.names[0]] = np.array(waveform - dc, copy=True)
                    self.source_currents[prefix + layout.dc_name] = np.full(len(t), dc)
            else:
                voltage_models[src_name].waveform = waveform

        return net_signals

    def _build_graph(self, blocks, net_signals, t):
        """Build the dependency graph of the modelled blocks. Returns dict of
        path→info."""
        graph = {}
        outputs = {net for block in blocks.values()
                   for net, direction in block.pins.values() if direction == PortDir.OUT}

        for path, block in blocks.items():
            reads = set()
            writes = set()
            inout_reads = set()
            # Note: INOUT nets are not added to reads or writes — they are
            # accessible through electrical views, but do not
            # participate in dependency ordering. This is intentional: the
            # warning below informs users that evaluation order is assumed.

            for port_name, (net, direction) in block.pins.items():
                if direction == PortDir.IN:
                    reads.add(net)
                elif direction == PortDir.OUT:
                    writes.add(net)
                elif direction == PortDir.POWER:
                    reads.add(net)
                    if net not in net_signals and net not in outputs:
                        warnings.warn(
                            f"POWER port '{port_name}' on block '{path}' "
                            f"(net '{net}') has no connected source — defaulting to 0V.",
                            UserWarning, stacklevel=4,
                        )
                        net_signals[net] = np.zeros_like(t)
                elif direction == PortDir.GROUND:
                    reads.add(net)
                    if net not in net_signals and net not in outputs:
                        net_signals[net] = np.zeros_like(t)
                elif direction == PortDir.INOUT:
                    inout_reads.add(net)
                    warnings.warn(
                        f"Port '{port_name}' on block '{path}' is INOUT — "
                        f"evaluation order assumed. Verify results if signals are unexpected.",
                        UserWarning, stacklevel=4,
                    )

            if isinstance(block.device, AlgebraicModel) and reads & writes:
                raise SimulationError(
                    f"Circular dependency in element '{block.path}' involving "
                    f"nets: {sorted(reads & writes)}.")
            graph[path] = {
                'reads':  reads,
                'writes': writes,
                'inout_reads': inout_reads,
                'device': block.device,
            }

        return graph

    def _prepare_views(self, blocks):
        """Prepare placement metadata once, independently of DC electrical state."""
        from silk.simulation._behavioral_views import _prepare_port_metadata, ParamsView
        from silk.simulation._behavioral_state import _register_writer
        for path, attached in self._attachments.items():
            if path not in self._port_metadata:
                self._port_metadata[path] = _prepare_port_metadata(blocks[path].device)
                self._param_views[path] = ParamsView(attached[3])
                # Metadata and writer maps reset together; preparation precedes
                # source publication, so a new placement has no writer yet.
                _register_writer(self._writers, path)

    def _evaluate_blocks(self, blocks, t, order, net_signals):
        """Evaluate models in declared dependency order and update net signals.

        Models read and assign through SharedState-backed electrical views.
        Consecutive sample models run
        in order at each point; vector calls finish over the whole grid.
        `silk.analysis()` reports the dispatched analysis during evaluation.
        """
        if self.resolution is None:
            from silk.simulation._behavioral_variation import resolve_run
            self.resolution = resolve_run(blocks, scope=self.scope, sample=self.sample)
        if self.state is None or self.state.signals is not net_signals:
            pins = {(block.path, port): net for block in blocks.values()
                    if not isinstance(block.device, AlgebraicModel)
                    for port, (net, _) in block.pins.items()}
            self.state = SharedState(len(t), pins, net_signals, writers=self._writers)
        if self._attachments is None:
            from silk.simulation._behavioral_models import attachments
            self._attachments = attachments(blocks, self.resolution)
        self._prepare_views(blocks)
        produced = {'0', 'gnd'}

        def evaluate(path, index=slice(None)):
            block = blocks[path]
            device = block.device
            if isinstance(device, AlgebraicModel) and device.terminals:
                missing = [net for net, direction in block.pins.values()
                           if direction == PortDir.IN and net not in produced]
                if missing:
                    source_refuse(device.path, *device.terminals,
                                  f"required voltage producer did not publish {missing!r}")
            if isinstance(device, AlgebraicModel):
                self.state.read_events.update(
                    (path, port, 'v', 0) for port, (_, direction) in block.pins.items()
                    if direction in (PortDir.IN, PortDir.POWER, PortDir.GROUND, PortDir.INOUT))
                inputs = {port: net_signals.get(net, np.zeros_like(t))
                          for port, (net, direction) in block.pins.items()
                          if direction in (PortDir.IN, PortDir.POWER, PortDir.GROUND, PortDir.INOUT)}
                outputs = device._invoke_behavioral_model(inputs, t)
                for port, (net, direction) in block.pins.items():
                    if direction in (PortDir.OUT, PortDir.INOUT) and port in outputs:
                        self.state.publish_voltage(net, path, outputs[port])
                        if device.terminals or block.pins['input'][0] in produced:
                            produced.add(net)
                return

            fn, _, execution, params = self._attachments[path]
            ports = PortsView(self.state, block.path, device, index,
                              metadata=self._port_metadata[path])
            model_t = t if isinstance(index, slice) else t[index]
            model_params = self._param_views[path]
            try:
                result = fn(ports, model_t, model_params)
            except TypeError as exc:
                # Diagnose invalid call binding without masking errors inside a model.
                from inspect import signature
                try:
                    contract = signature(fn)
                except (TypeError, ValueError):
                    contract = None
                if contract is not None:
                    try:
                        contract.bind(ports, model_t, model_params)
                    except TypeError:
                        raise SimulationError(
                            f"Model {block.path!r}: use fn(ports, t, params); read "
                            "ports.VIN.v, assign ports.VOUT.v or ports.VOUT.i, "
                            "and return None instead of an output dictionary.") from exc
                raise
            if result is not None:
                raise SimulationError(
                    f"Model {block.path!r}: a ports-form model must return None; "
                    "publish values by assigning ports.X.v or ports.X.i.")
            for net, direction in block.pins.values():
                if direction in (PortDir.OUT, PortDir.INOUT) and net in net_signals:
                    produced.add(net)

        # Keep the existing FIFO graph walk. Consecutive sample models advance
        # in that order at each point; each vector invocation finishes once.
        position = 0
        while position < len(order):
            path = order[position]
            attached = self._attachments.get(path)
            if attached is not None and attached[1:3] == ('ports', 'sample'):
                end = position + 1
                while end < len(order):
                    other = self._attachments.get(order[end])
                    if other is None or other[1:3] != ('ports', 'sample'):
                        break
                    end += 1
                for index in range(len(t)):
                    for sample_path in order[position:end]:
                        evaluate(sample_path, index)
                position = end
            else:
                evaluate(path)
                position += 1

        from silk.simulation._behavioral_feedback import feedback_reads
        cycles, reads = feedback_reads(
            self.state, blocks, self._attachments,
            check_mixed=(any(attached[2] == 'sample'
                             for attached in self._attachments.values())
                         and analysis() == 'transient'))
        if cycles:
            if analysis() == 'dc':
                raise LoopError(
                    f"DC feedback at point {t[0]:g}: models {cycles}; reads {reads}. "
                    "Use a transient simulation or put the loop inside one model.")
            if not self._feedback_warned:
                warnings.warn(
                    f"Feedback without a modelled delay: models {cycles}; reads {reads}. "
                    "Use previous_value for an explicit transient delay.",
                    UserWarning, stacklevel=3)
                self._feedback_warned = True


def _resolver(prefix, ports):
    """Name resolution inside an entered instance.

    A port net is the parent's net it is bound to (SPICE aliases it the same
    way), ground is global, and every other net is spelled as SPICE spells
    it: the instance prefix plus the local name (``'xdut.vcomp'``).
    """
    def resolve(net):
        if net in _GLOBAL_NETS:
            return net
        if net in ports:
            return ports[net]
        return prefix + net
    return resolve


def _transistor_message(path, inst_path):
    """The refusal for a transistor-level device no model stands in for.

    Paths are shown with instance parts as ``[k]`` (display only)."""
    from silk.spice_names import designer_component
    path = designer_component(path)
    inst_path = None if inst_path is None else designer_component(inst_path)
    remedy = ("a behavioural run cannot evaluate it. Give that block (or one "
              "above it) a behavioural model with set_behavioral_model(), or "
              "set simulator to a SPICE backend ('ngspice', 'xyce').")
    if inst_path is None:
        return (f"Element '{path}' is a transistor-level device on the top "
                f"level, with no behavioural model above it: {remedy}")
    return (f"Instance '{inst_path}' has no behavioural model and contains "
            f"transistor-level device '{path}': {remedy}")
