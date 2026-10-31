"""silk.simulation.behavioral_backend — the ``'behavioral'`` simulator backend.

Selected by the existing ``simulator`` setting::

    mod.simulator = 'behavioral'
    mod.simulate()

The backend runs silk's numpy engine (:class:`~silk.simulation.behavioral.
BehavioralEngine`) on the :class:`~silk.circuit.modules.Module` itself. There is
no SPICE deck, so it declares :attr:`~silk.simulation.backend.SimulatorBackend.
runs_module` and implements ``_launch_module``; ``SimulatorBackend.run_module``
applies the same data-presence gate every SPICE run passes through.

The result is nutmeg-shaped (design §5.1): the engine's arrays are
packaged by :func:`build_behavioral_rawfile` into the same
:class:`~silk.signals.rawfile.RawFile` a SPICE run produces, and written to the
run's rawfile path. Every reader of SPICE results -- ``load_signal``,
``tb.probe``, ``results_file``, ``view()``, ``emit_view``,
``evaluate(rawfile=...)`` -- then reads a behavioural run the same way.
"""
from silk.circuit.emitted_names import emitted_submodule_name
from silk.spice_names import designer_component
import time
import json

import numpy as np

from silk._json_codec import encode_json
from silk.simulation._analysis_context import analysis_scope
from silk.exceptions import NotSupportedError, SimulationError
from silk.signals.rawfile import RawFile, RawPlot, RawVector, write_rawfile
from silk.simulation.analysis import DC, Transient
from silk.utils._utils import _parse_si
from silk.simulation.backend import RawResult, SimulatorBackend, SpiceBackend

#: The ``Command:`` header of a behavioural rawfile, so a file read back later
#: says which engine wrote it.
BEHAVIORAL_COMMAND = 'silk behavioral engine'


@SimulatorBackend.register('behavioral')
class BehavioralBackend(SpiceBackend):
    """Runs a module through silk's behavioural (numpy) engine.

    It subclasses :class:`SpiceBackend` so results are read the way SPICE
    results are. It renders no deck and launches no simulator: ``_launch``
    refuses, and ``Module.simulate()`` calls ``run_module`` instead.
    """

    runs_module = True

    def make_netlister(self, pdk_details):
        """The ngspice netlister.

        A block whose own ``simulator`` is ``'behavioral'`` can still be
        instantiated in a SPICE bench, and the bench emits its ``.subckt``
        through the block's netlister. A top-level deck is never emitted for
        a behavioural run: ``generate_netlist()`` refuses first.
        """
        from silk.circuit.spice_netlister import PySpiceNetlister
        return PySpiceNetlister(pdk_details)

    def _launch(self, netlist_path, results_dir, *, debug=False, timeout=None):
        raise NotSupportedError(
            "the 'behavioral' backend has no SPICE deck to run: it evaluates "
            "the Module itself. Call Module.simulate(), or set simulator to a "
            "SPICE backend ('ngspice', 'xyce') to run a netlist.")

    def _launch_module(self, module, *, run_files=None):
        """Run the engine, package its arrays as a RawFile, and write it.

        With *run_files* (``Module.simulate()`` always passes them) the
        rawfile is written to ``run_files.rawfile_path`` and that path is the
        result's ``source_path``. Without them the result stays in memory.
        """
        from silk.simulation.behavioral import BehavioralEngine
        _refuse_inapplicable_switches(module)
        analyses = _declared_analyses(module)
        plots = []
        # One engine for every analysis, so the D4 notices warn once per run.
        engine = BehavioralEngine(scope=getattr(module, '_variation_scope', None),
                                  sample=getattr(module, '_mc_sample', None))
        # All analyses share one logical run's attachments, setup and notices.
        engine._shared_run = True
        for analysis in analyses:
            with analysis_scope('dc' if isinstance(analysis, DC) else 'transient'):
                if isinstance(analysis, DC):
                    signals, axis, axis_type = engine.run_dc(module, analysis)
                    axis_name = 'i(i-sweep)' if axis_type == 'current' else 'v(v-sweep)'
                    result = build_behavioral_rawfile(
                        module, signals, axis, plotname='DC transfer characteristic',
                        axis_name=axis_name, axis_type=axis_type)
                else:
                    step = _parse_si(analysis.step)
                    if analysis.maxstep is not None:
                        step = min(step, _parse_si(analysis.maxstep))
                    signals, t = engine.run(
                        module, t_step=step, t_stop=_parse_si(analysis.stop))
                    currents = engine.source_currents
                    start = _parse_si(analysis.start)
                    if start > 0:
                        # SPICE does not save samples before TSTART; the engine ran
                        # from 0 so stateful models still see the lead-in.
                        keep = t >= start - step * 1e-9
                        t = t[keep]
                        signals = {name: np.asarray(arr)[..., keep]
                                   for name, arr in signals.items()}
                        currents = {name: arr[..., keep]
                                    for name, arr in currents.items()}
                    result = build_behavioral_rawfile(module, signals, t,
                                                      currents=currents)
                plots.extend(result.plots)
        # Leave an empty engine result to run_module's SimulationAborted gate.
        module._behavioral_draws = (engine.resolution.draws
                                    if engine.resolution is not None else {})
        rawfile = RawFile(plots)
        path = getattr(run_files, 'rawfile_path', None)
        if path is not None:
            write_rawfile(path, rawfile)
            rawfile.source_path = path
        return RawResult(
            native=rawfile,
            source_path=path or f'<behavioral run of {_module_label(module)}>',
            run_files=run_files,
        )


def _rawfile_net_names(module):
    """The net names a behavioural rawfile can carry, in order: *module*'s own
    nets, then every instance's non-port nets by hierarchical name."""
    from silk.circuit.modules_results import _entered_instances
    names = list(getattr(module, 'net_dict', {}) or {})
    for prefix, sub in _entered_instances(module):
        ports = sub.pin_list or {}
        names.extend(prefix + name for name in sub.net_dict or {}
                     if name not in ports)
    return names


def build_behavioral_rawfile(module, signals, t, *, currents=None,
                             plotname='Transient Analysis',
                             axis_name='time', axis_type='time') -> RawFile:
    """Package one behavioural run as a nutmeg ``RawFile`` (design §5.1).

    **The one place a behavioural result is typed and named.** One plot,
    ``Transient Analysis``, spelled as ngspice spells it: vector 0 is
    ``time`` (type ``time``), then ``v(<net>)`` (type ``voltage``) for every
    net of *module* the engine produced an array for, in the module's net
    order, then each entered instance's internal nets by SPICE's
    hierarchical name (``v(xdut.vcomp)``; a port is its parent's
    net and has no vector of its own). An engine array whose name is not a
    declared net (a model's intermediate signal) is not a net of the circuit
    and is left out, as ``simulate_behavioral()`` leaves it off
    ``Net.voltage``.

    DC uses the same builder with a ``DC transfer characteristic`` plot and
    a swept-source axis (``v(v-sweep)`` or ``i(i-sweep)``).

    Every net array is typed ``voltage`` and labelled by its vector name,
    as before; changing that is a change to this
    function alone. Independent CurrentSource waveforms captured by the engine
    are appended as current-typed i(<emitted source name>) vectors; voltage
    source branch currents are never synthesized.
    """
    t = np.asarray(t, dtype=np.float64)
    vectors = [RawVector(axis_name, axis_type, data=t)]
    for name in _rawfile_net_names(module):
        arr = signals.get(name)
        if arr is None:
            continue
        try:
            data = np.broadcast_to(np.asarray(arr, dtype=np.float64), t.shape)
        except ValueError:
            raise SimulationError(
                f"behavioural result for net {name!r} has shape "
                f"{np.shape(arr)}, but the analysis axis has {len(t)} points: a "
                f"model must return one value per analysis point (or a scalar)."
            ) from None
        vectors.append(RawVector(f'v({name})', 'voltage',
                                 data=np.array(data, dtype=np.float64)))
    for name, data in (currents or {}).items():
        vectors.append(RawVector(f'i({name.lower()})', 'current', data=data))
    header = {
        'title': _module_label(module),
        'date': time.strftime('%a %b %d %H:%M:%S %Y'),
        'plotname': plotname,
        'flags': 'real',
        'command': BEHAVIORAL_COMMAND,
    }
    notes = getattr(module, '_behavioral_element_notes', ())
    if notes:
        header['silk behavioral elements'] = json.dumps(encode_json(notes), allow_nan=False)
    plot = RawPlot(header, vectors, npoints=len(t), real=True, padded=True)
    return RawFile([plot])


def behavioral_net_arrays(rawfile) -> dict:
    """The inverse of :func:`build_behavioral_rawfile`: net name -> array.

    Used to write ``Net.voltage`` from exactly what the run's rawfile holds.
    """
    out = {}
    for vec in rawfile.vectors[1:]:
        name = vec.name
        if name.startswith('v(') and name.endswith(')'):
            out[name[2:-1]] = vec.data
    return out


def _module_label(module) -> str:
    return (getattr(module, 'circuit_name', None)
            or getattr(module, 'netlistname', None) or 'module')


def _refuse_inapplicable_switches(module):
    """Refuse, before the engine starts, what has no behavioural meaning."""
    label = _module_label(module)
    if getattr(module, 'custom_lines', None):
        raise NotSupportedError(
            f"{label!r} has custom SPICE text (add_custom_line), and a "
            f"behavioural run cannot evaluate SPICE text. Remove it, or set "
            f"simulator to a SPICE backend.")
    if getattr(module, 'omit_device_internal_nodes', False):
        raise NotSupportedError(
            'omit_device_internal_nodes=True requires ngspice; behavioral '
            'simulation has no device-internal nodes. Keep '
            'omit_device_internal_nodes=False.')
    for flag in ('save_device_currents', 'save_device_params'):
        if getattr(module, flag, False):
            raise NotSupportedError(
                f"{flag} is set on {label!r}, and a behavioural run has no "
                f"device currents or device parameters. Clear it, or set "
                f"simulator to a SPICE backend.")


def _declared_analyses(module):
    """Validate every analysis before evaluating any models."""
    analyses = list((getattr(module, 'sim_analysis_dict', None) or {}).values())
    others = [a for a in analyses if not isinstance(a, (Transient, DC))]
    if others:
        names = ', '.join(type(a).__name__ for a in others)
        raise NotSupportedError(
            f"the behavioural engine evaluates transient waveforms and DC "
            f"source sweeps only; {_module_label(module)!r} declares: {names}. "
            f"Remove it, or set simulator to a SPICE backend.")
    if not analyses:
        raise NotSupportedError(
            f"a behavioural run needs a Transient or DC analysis, and "
            f"{_module_label(module)!r} declares none: call "
            f"add_sim_type(Transient(step=..., stop=...)) or "
            f"add_sim_type(DC(source=..., start=..., stop=..., step=...)).")
    return analyses


def refuse_behavioral_probe(module, name, selector='V'):
    """Explain quantities a behavioural model cannot supply (design §5.2).

    Called before SPICE's device-address/ammeter validation, both when a
    bench declares a probe and when Module.load_signal resolves a result.
    Ordinary missing names still reach the shared rawfile reader.
    """
    from silk.circuit.circuit_elements import ElementType
    from silk.sources import VoltageSource

    spelling = name.strip().lower()
    from silk.circuit.node_paths import NodeVoltageRequest
    literal = (isinstance(name, NodeVoltageRequest)
               or getattr(name, '_literal_voltage', False))
    wrapped = not literal and spelling.startswith(('v(', 'i(')) and spelling.endswith(')')
    if wrapped:
        selector, spelling = spelling[0].upper(), spelling[2:-1]
    from silk.simulation.backend import DEVICE_OP_PARAMS
    device_suffix = spelling.rpartition(':')[2]
    if ((not literal and not wrapped and ':' in spelling
         and device_suffix in {*DEVICE_OP_PARAMS, 'd', 'g', 's', 'b'})
            or (not literal and spelling.startswith('@'))
            or (selector.upper() == 'I' and '.vamm_' in spelling)):
        raise NotSupportedError(
            f"Probe {name!r}: a behavioral run has no device currents or "
            "device parameters; these require an electrical solve. "
            "Use a SPICE backend.")

    def visit(mod, prefix):
        if selector.upper() == 'I':
            for src in mod.sig_sources_dict.values():
                if isinstance(src, VoltageSource) and spelling in {
                        (prefix + card).lower() for card in
                        (src.instance_name, *src.physical_instance_names())}:
                    raise NotSupportedError(
                        f"Probe {name!r}: a voltage source's branch current "
                        "needs an electrical solve; behavioral models do not "
                        "compute it. Use a SPICE backend.")
        for inst, entry in (mod.elem_dict or {}).items():
            if entry['type'] != ElementType.MODULE:
                continue
            path = f'{prefix}{emitted_submodule_name(inst)}.'
            if not spelling.startswith(path.lower()):
                continue
            child = entry['device']
            if child._behavioral_model is not None:
                raise NotSupportedError(
                    f"Probe {name!r}: {designer_component(path[:-1])} is evaluated by its "
                    "behavioral model; its internal nets do not exist in "
                    "a behavioral run. Probe the parent net connected to "
                    "an output port, or use a SPICE backend.")
            visit(child, path)

    if module is not None:
        visit(module, '')


def behavioral_element_notes(module):
    """Read notices from current evidence, never from a prior run's topology."""
    raw = getattr(module, '_raw', None)
    if raw is not None:
        plot = getattr(raw.native, 'current_plot', None)
        header = getattr(plot, 'header', {})
        if header.get('command') != BEHAVIORAL_COMMAND:
            return ()
        value = json.loads(header.get('silk behavioral elements', '[]'))
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise SimulationError('Invalid behavioral element notices in rawfile')
        return tuple(value)
    if getattr(module, '_behavioral_mode', False):
        return tuple(getattr(module, '_behavioral_element_notes', ()))
    return ()
