"""Completion evidence for generation-bound, built-in SPICE requests.

External/custom decks retain the data-presence gate. Only generation records
bound to the exact launched bytes establish which analyses actually ran.
"""
from dataclasses import dataclass
from hashlib import sha256
import math
import re
from pathlib import Path

from silk.exceptions import SimulationAborted
from silk.simulation._frequency_grid import _whole_steps, logarithmic_steps
from silk.signals.analysis_kind import plot_silk_analysis
from silk.utils._utils import _parse_si

ENDPOINT_RTOL = 1e-7


@dataclass(frozen=True)
class CompletionRequest:
    kind: str
    start: float
    stop: float
    end: float
    count: int | None = None
    final_interval: float | None = None
    variation: str | None = None
    endpoint_unverified_reason: str | None = None


@dataclass(frozen=True)
class CompletionPlan:
    simulator: str
    digest: bytes
    requests: tuple[CompletionRequest, ...]
    #: The deck the digest binds when it is not the run's own deck path: a
    #: multi-sample Monte Carlo process runs one chunk deck for many samples.
    deck_path: str | None = None


def _requests(text, simulator):
    """Parse only already-certified emitted cards, within the effective deck."""
    requests = []
    depth = 0
    control = False
    controls = runs = 0
    for line in text.splitlines()[1:]:
        fields = line.lower().split()
        if not fields or fields[0].startswith('*'):
            continue
        card = fields[0]
        if card == '.end' and depth == 0:
            break
        if card == '.subckt':
            depth += 1
        elif card == '.ends':
            depth -= 1
        elif card == '.control':
            control = True
            controls += 1
            if controls > 1:
                return None
        elif card == '.endc':
            control = False
        if card == '+' or card == '.step':
            return None
        if control and not card.startswith('.'):
            # Recognise only commands used by silk's built-in output block.
            # This is an emission check, not a control-language interpreter.
            if card not in ('save', 'run', 'op', 'set', 'setplot', 'option',
                            'write', 'showmod', 'pre_osdi', 'exit', 'quit'):
                return None
            if card == 'run':
                runs += 1
                if runs > 1:
                    return None
        if depth or control or card not in ('.tran', '.dc', '.ac', '.noise'):
            continue
        kind = card[1:]
        if any(r.kind == kind for r in requests):
            return None
        try:
            count = interval = variation = unverified = None
            if kind == 'tran':
                start, stop = 0.0, _parse_si(fields[2])
                end = stop
            elif kind == 'dc':
                if len(fields) != 5:  # nested DC is not a silk analysis
                    return None
                start, stop, step = map(_parse_si, fields[2:5])
                uncertainty = 8 * math.ulp(max(abs(start), abs(stop))) / abs(step)
                steps = _whole_steps((stop - start) / step, uncertainty)
                count, end = steps + 1, start + steps * step
                interval = abs(step) if steps else None
            else:
                offset = 3 if kind == 'noise' else 1
                variation = fields[offset]
                points = int(fields[offset + 1])
                start, stop = map(_parse_si, fields[offset + 2:offset + 4])
                if variation == 'lin':
                    count = points
                    end = stop if points > 1 else start
                    interval = abs(stop - start) / (points - 1) if points > 1 else None
                elif variation in ('dec', 'oct'):
                    base = 10 if variation == 'dec' else 2
                    steps = logarithmic_steps(points, start, stop, base)
                    count = steps + 1
                    stretched = kind == 'ac' and simulator == 'ngspice' and variation == 'dec'
                    if stretched and 1 < stop / start < 10:
                        from silk.simulation._ngspice_compat import NgspiceCompatibility
                        compatibility = NgspiceCompatibility()
                        if compatibility.release is None:
                            unverified = 'ngspice version is unknown; sub-decade DEC AC grid is unverified'
                        else:
                            stretched = not compatibility.has('ac_dec_subdecade_plain_steps')
                    if stretched:
                        end = stop if steps else start
                        previous = start * (stop / start) ** ((steps - 1) / steps) if steps else start
                    else:
                        end = start * base ** (steps / points)
                        previous = start * base ** ((steps - 1) / points)
                    interval = end - previous if steps else None
                else:
                    return None
            if not all(math.isfinite(x) for x in (start, stop, end)):
                return None
            if count is not None and (count < 1 or (count > 1 and
                    (interval is None or interval <= 8 * math.ulp(max(abs(start), abs(stop)))))):
                return None
        except (ValueError, IndexError, ZeroDivisionError, OverflowError):
            return None
        requests.append(CompletionRequest(kind, start, stop, end, count, interval,
                                          variation, unverified))
    return tuple(requests)


def _opaque_emission_reason(module):
    """Describe custom emission inputs, or return None for built-in emitters."""
    from silk.circuit.spice_netlister import PySpiceNetlister
    from silk.circuit.xyce_netlister import XyceNetlister
    from silk.simulation.analysis import Transient, DC, AC, Noise, Op
    from silk.circuit.circuit_elements import ElementType
    from silk.sources.classes import _IndependentSource
    from inspect import getattr_static
    seen = set()

    def opaque(mod):
        if id(mod) in seen:
            return None
        seen.add(id(mod))
        if mod.custom_lines:
            return 'Module.add_custom_line()/custom_lines'
        for name in ('generate_netlist', 'generate_subckt'):
            renderer = getattr(mod, name)
            if not getattr(renderer, '__module__', '').startswith('silk.'):
                return f'custom Module.{name}() emitter'
        renderer = getattr(mod.pdk_details, 'generate_netlist_entry', None)
        if renderer is not None and not getattr(renderer, '__module__', '').startswith('silk.'):
            return 'custom pdk_details.generate_netlist_entry() emitter'
        if type(mod.netlister) not in (PySpiceNetlister, XyceNetlister):
            return 'replacement netlister'
        for analysis in mod.sim_analysis_dict.values():
            if type(analysis) not in (Transient, DC, AC, Noise, Op):
                return 'custom analysis subclass'
        for obj in (list(mod.sig_sources_dict.values()) + list(mod.passives_dict.values())
                    + [entry.get('device') for entry in mod.elem_dict.values()
                       if entry.get('type') != ElementType.MODULE]):
            renderer = getattr(obj, 'to_spice', None)
            if renderer is not None and not getattr(renderer, '__module__', '').startswith('silk.'):
                return 'Module.add()/add_source()/add_passive() custom to_spice emitter'
            selector = getattr_static(obj, 'selected_waveform', None)
            if isinstance(obj, _IndependentSource) or selector is not None:
                if (not isinstance(obj, _IndependentSource)
                        or selector is not _IndependentSource.selected_waveform
                        or 'selected_waveform' in vars(obj)):
                    return 'VoltageSource/CurrentSource custom waveform selector'
                wave = _IndependentSource.selected_waveform(obj)
            else:
                wave = None
            renderer = getattr(wave, 'to_spice', None)
            if renderer is not None and not getattr(renderer, '__module__', '').startswith('silk.'):
                return 'VoltageSource/CurrentSource custom waveform emitter'
        children = [entry.get('device') for entry in mod.module_dict.values()]
        children += [entry.get('device') for entry in mod.elem_dict.values()
                     if entry.get('type') == ElementType.MODULE]
        for child in children:
            if child is not None:
                reason = opaque(child)
                if reason:
                    return reason
        return None

    return opaque(module)


#: ``certify_generated``'s reason for a deck with no transient/DC/AC/noise card
#: (an op-only bench): no plan is expected.
NO_REQUEST_REASON = 'no transient/DC/AC/noise request'
#: The one missing-certificate case that needs none: a deck that declares only
#: Op analyses and emits an ``.op`` card (declared and emitted agree).
OP_ONLY_REASON = 'op-only deck: nothing to certify'


def uncertifiable_reason(analyses):
    """Why a deck for the declared *analyses* would get no completion
    certificate on ngspice, or ``None``. Applies the same card rules as
    :func:`certify_generated` to the analyses' own cards, without emitting a
    deck or touching the filesystem, so a caller can decide before writing.
    An op-only declaration needs no certificate and returns ``None`` too."""
    from silk.simulation.analysis import Transient, DC, AC, Noise, Op
    kinds = {Transient: 'tran', DC: 'dc', AC: 'ac', Noise: 'noise'}
    cards, expected = [], []
    for analysis in analyses:
        if type(analysis) is Op:
            continue
        if type(analysis) not in kinds:
            return 'a custom analysis'
        cards.append(analysis.to_spice().strip())
        expected.append(kinds[type(analysis)])
    if not cards:
        return None
    requests = _requests('*\n' + '\n'.join(cards) + '\n.end\n', 'ngspice')
    if requests is None:
        return 'unsupported, repeated or numerically unresolvable analysis cards'
    if sorted(expected) != sorted(r.kind for r in requests):
        return 'emitted analysis cards do not match silk analysis records'
    return None


def certify_generated(module, text):
    """Return a plan/reason for a built-in Module emission; never a bare deck.

    Opaque API inputs are add_custom_line/custom_lines at any hierarchy level,
    custom analysis, source, waveform, waveform selectors, PDK or Module
    emitters, and replacement
    netlisters. Built-in
    source waveform data (including PWL), PDK includes and options are emitted
    by silk and do not themselves remove the guarantee.
    """
    from silk.simulation.analysis import Transient, DC, AC, Noise, Op
    simulator = module.simulator
    if simulator not in ('ngspice', 'xyce'):
        return None, 'not a supported SPICE backend'
    reason = _opaque_emission_reason(module)
    if reason:
        return None, reason
    requests = _requests(text, simulator)
    if requests is None:
        return None, 'unsupported, repeated or numerically unresolvable analysis cards'
    if not requests:
        declared_types = [type(a) for a in module.sim_analysis_dict.values()]
        if (declared_types and all(t is Op for t in declared_types)
                and re.search(r'^\.op\b', text, flags=re.M | re.I)):
            return None, OP_ONLY_REASON
        return None, NO_REQUEST_REASON
    declared = {Transient: 'tran', DC: 'dc', AC: 'ac', Noise: 'noise'}
    expected_kinds = [declared[type(a)] for a in module.sim_analysis_dict.values()
                      if type(a) in declared]
    if sorted(expected_kinds) != sorted(r.kind for r in requests):
        return None, 'emitted analysis cards do not match silk analysis records'
    return CompletionPlan(simulator, sha256(text.encode()).digest(), requests), None


def require_complete(raw, deck_path, *, simulator, log_path=None, reason=None):
    """Check only certified requests with an unambiguous result association."""
    files = raw.run_files
    plan = getattr(files, 'completion', None)
    if not isinstance(plan, CompletionPlan) or plan.simulator != simulator:
        return
    try:
        if sha256(Path(plan.deck_path or deck_path).read_bytes()).digest() != plan.digest:
            return
    except OSError:
        return
    plots = getattr(raw.native, 'plots', None)
    if not isinstance(plots, (list, tuple)) and raw.native is not None:
        return
    plots = plots or []
    associated = []
    for request in plan.requests:
        if getattr(request, 'endpoint_unverified_reason', None):
            continue  # Other requests in this run still receive completion checks.
        matching = [p for p in plots if plot_silk_analysis(p) == request.kind]
        if request.kind == 'noise':
            matching = [p for p in matching if p.vectors
                        and p.vectors[0].name.lower().startswith('freq')]
        if len(matching) > 1:
            return  # No provable request-to-plot identity: no endpoint verdict.
        associated.append((request, matching[0] if matching else None))
    for request, plot in associated:
        if request.kind == 'noise' and simulator == 'xyce' and files.noise_path:
            from silk.simulation.xyce_runner import read_xyce_noise_csv
            axis = (read_xyce_noise_csv(files.noise_path)['FREQ']
                    if Path(files.noise_path).is_file() else ())
        else:
            axis = plot.vectors[0].data if plot is not None and plot.vectors else ()
        tolerance = ENDPOINT_RTOL * max(abs(request.start), abs(request.stop),
                                       abs(request.stop - request.start))
        if request.final_interval is not None:
            tolerance = min(tolerance, request.final_interval / 4)
        direction = -1 if request.stop < request.start else 1
        last = float(axis[-1].real) if len(axis) else None
        if (last is not None and math.isfinite(last)
                and (request.count is None or len(axis) >= request.count)
                and direction * (request.end - last) <= tolerance):
            continue
        if reason is None and log_path is not None:
            from silk.simulation.runner import abort_reason_from_log
            reason = abort_reason_from_log(log_path)
        axis_name = {'tran': 'time', 'dc': 'swept value',
                     'ac': 'frequency', 'noise': 'frequency'}[request.kind]
        message = (
            f'{simulator} {request.kind} analysis aborted: requested stop {request.stop:g}, '
            f'expected final grid point {request.end:g}, last {axis_name} reached '
            f'{last!r} (tolerance {tolerance:g}, required points {request.count}). '
            f'Rawfile: {raw.source_path}' + (f'. Reason: {reason}' if reason else ''))
        if (simulator == 'ngspice' and request.kind == 'ac'
                and getattr(request, 'variation', None) == 'lin'
                and request.count == 2 and len(axis) == 1
                and last is not None and abs(last - request.start) <= tolerance):
            message += (
                '. ngspice wrote 1 of 2 requested LIN AC points; ngspice writes '
                'only the start frequency for a 2-point LIN AC sweep; '
                'use 3 or more points or Xyce.')
        raise SimulationAborted(message, simulator=simulator, reason=reason,
                                data_rows=len(axis), log_path=log_path)
