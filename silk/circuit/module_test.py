"""Testbench construction, simulation runs, metrics and spec checks for modules."""

# SpecReport is imported lazily in run() to avoid circular imports
# Python based module test class
# Maintained by the silk contributors
# Provides testbench capabilities for Module testing
# Each Module can have multiple tests with different testbenches and metrics

import copy
import hashlib
from numbers import Real
from typing import Optional
import inspect
import os
import re
import warnings
import weakref
from dataclasses import dataclass, field, replace
from contextlib import contextmanager, nullcontext
import time
from uuid import uuid4
from urllib.parse import quote

from silk.circuit.variable import Variable
from silk.circuit.emitted_names import emitted_submodule_name
from silk.circuit.test_placement import ResultAttribute, current_placement

from silk.circuit.metric_handle import MetricHandle, _evaluate_expression
from silk.circuit.modules_results import (
    DUT_ELEMENT_NAME, NON_PROBEABLE_OP_COLUMNS,
    capture_op_device_records, derived_column_probe_error,
    find_designer_device,
)
from silk.db.context import build_characterization_context, freeze_parameters
from silk.exceptions import (
    DuplicateNameError,
    AmbiguousBranchCurrentError,
    MetricBindingError,
    NotASubcircuitError,
    NotSupportedError,
    SimulationAborted,
    ConvergenceError,
    SimulationError,
    UnboundSignalError,
)


# ---------------------------------------------------------------------------
# DUT-embedding scope (TASK-205)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Declarative metric binding. Module-level rather than methods because these
# are pure functions of the
# metric and the caller's arguments — nothing about them needs a testbench,
# and T-G (testbench_generator._validate) will want the same slot accessor.
# ---------------------------------------------------------------------------

def _metric_signal_slots(fn):
    """This metric's signal-input slots as ``[(slot_name, kind_or_None)]``.

    **Schema format.** ``metric_schema(fn)['signal_inputs']`` is a
    list of ``{'name', 'kind'}`` objects (schema
    ``format_version`` ``'2.0'``); it was a list of bare name strings through
    ``'1.0'``. This function is the only thing in the declarative binding that
    reads that shape. :func:`_slot_inventory` uses the declared kinds in
    binding error messages. The ``str`` branch is kept so
    this accessor can still read a schema exported by an older silk rather
    than raising on a ``str``.
    """
    from silk.signals.schema import metric_schema
    slots = []
    for entry in metric_schema(fn)['signal_inputs']:
        if isinstance(entry, str):          # schema v1 — a bare name
            slots.append((entry, None))
        else:                               # schema v2 — {'name', 'kind'}
            slots.append((entry['name'], entry.get('kind')))
    return slots


def _type_name(value):
    return value.__class__.__name__


def _quoted(names):
    """``'a', 'b'`` — the repr of each name, comma-joined."""
    return ', '.join(repr(n) for n in names)


def _slot_inventory(slots):
    """``'sig_in' (kind 'transient'), 'sig_out' (kind 'transient')``.

    The kind is written only when the schema carries one. Slots without
    kinds are listed as bare names, such as ``'sig_in', 'sig_out'``.
    """
    return ', '.join(f"{n!r} (kind {k!r})" if k else repr(n) for n, k in slots)


def _bind_signal_inputs(metric_name, type_name, fn, signal, signals, params):
    """Resolve ``signal=``/``signals=`` into ``{slot_name: signal_path}``.

    Validation happens HERE — at declaration time, inside ``add_metric`` —
    and not inside the deferred extractor lambda, because a missing slot must
    not cost the user a simulation run to discover (the contract in docs/probes.md). Every error
    is a ``TypeError``, the same family ``add_metric`` already raises for its
    own argument misuse, and every message names the slot(s) involved.
    """
    where = f"add_metric({metric_name!r}, type={type_name!r})"
    slots = _metric_signal_slots(fn)
    names = [n for n, _ in slots]
    plural = 's' if len(slots) != 1 else ''

    if signal is not None and signals is not None:
        raise TypeError(
            f"{where}: pass signal= or signals=, not both. signals= is the "
            f"general form; for this metric's {len(slots)} signal input{plural} "
            f"({_quoted(names)}) it is the only one that can bind them all."
        )

    if signal is None and signals is None:
        if len(slots) == 1:
            # Unchanged message: the one-input declarative form is untouched.
            raise TypeError("add_metric: declarative form requires signal=")
        raise TypeError(
            f"{where}: declarative form requires signals=, binding each of "
            f"this metric's {len(slots)} signal inputs: {_slot_inventory(slots)}."
        )

    if signals is None:
        if len(slots) != 1:
            raise TypeError(
                f"{where}: signal= binds a single input, but this metric "
                f"takes {len(slots)}: {_quoted(names)}. Use "
                f"signals={{{', '.join(f'{n!r}: ...' for n in names)}}}."
            )
        bound = {names[0]: signal}
    else:
        if not isinstance(signals, dict):
            raise TypeError(
                f"{where}: signals= must be a dict keyed by this metric's "
                f"signal-input name{plural} ({_quoted(names)}); got "
                f"{type(signals).__name__}."
            )
        # Unknown keys first: on a typo ('sig_output') that is the useful half
        # of the diagnosis, and the missing-slot message would otherwise report
        # the same mistake from the far side.
        unknown = [k for k in signals if k not in names]
        if unknown:
            raise TypeError(
                f"{where}: signals= has unknown "
                f"key{'s' if len(unknown) != 1 else ''} {_quoted(unknown)}; "
                f"this metric's signal inputs are {_quoted(names)}."
            )
        missing = [n for n in names if n not in signals]
        if missing:
            raise TypeError(
                f"{where}: signals= is missing {_quoted(missing)}. This metric "
                f"takes {len(slots)} signal input{plural}: "
                f"{_slot_inventory(slots)}; "
                f"bound: {_quoted([n for n in names if n in signals]) or 'nothing'}."
            )
        bound = {n: signals[n] for n in names}

    clash = [n for n in names if n in params]
    if clash:
        raise TypeError(
            f"{where}: {_quoted(clash)} names a signal input of this metric "
            f"and so cannot also be passed as a scalar parameter; bind it "
            f"through signals=."
        )
    return bound


# ---------------------------------------------------------------------------
# Which analysis a declarative metric reads (TASK-212). silk NEVER GUESSES:
# one producer among the declared analyses needs no field, more than one
# requires ``analysis=``. ``domain`` is a label and is never consulted here.
# ---------------------------------------------------------------------------

#: The analysis word ``add_sim_type`` files each analysis under, translated
#: into silk's analysis vocabulary (``SILK_ANALYSES`` in
#: silk/signals/kinds.py).
#: ``sim_analysis_dict`` is keyed by ``type(analysis).__name__.lower()``, and
#: ``'transient'`` is the one word the two vocabularies spell differently —
#: the same single disagreement ``testbench_generator._DOMAIN_ANALYSES``
#: records for the spec side. ``'op'`` needs no translation:
#: ``Op``.lower() already is silk's own word for it.
#:
#: the ``'op'`` entry has no ``@metric(domain=...)``, and this table
#: is internal to ``_declared_analyses``/analysis validation, never
#: exported to the metric schema — it exists only so a test that
#: declares ``add_sim_type(analysis.Op())`` is recognised as having
#: declared the ``'op'`` analysis, letting a plain-def metric bind
#: ``analysis='op'`` against an Op() run's signals. ``SILK_ANALYSES``
#: itself stays ``('tran', 'ac', 'dc', 'noise')`` on purpose.
_SIM_KEY_ANALYSIS = {
    'transient': 'tran',
    'ac': 'ac',
    'dc': 'dc',
    'noise': 'noise',
    'op': 'op',
}


def _declared_analyses(module) -> list:
    """The analyses a test declares, in ``add_sim_type`` order, as silk words.

    Read off the LIVE ``sim_analysis_dict`` rather than off anything recorded
    when the metric was added, because a hand-built test may add sim types
    after its metrics — which is exactly why this question cannot be settled
    once and for all inside ``add_metric``.
    """
    d = getattr(module, 'sim_analysis_dict', None) or {}
    out = []
    for key in d:
        word = _SIM_KEY_ANALYSIS.get(str(key).lower())
        if word is not None and word not in out:
            out.append(word)
    return out


def _analysis_candidates(slots, declared) -> list:
    """The declared analyses that could feed EVERY one of this metric's signal
    inputs, in declared order.

    An analysis is a candidate when it appears in ``produced_by`` for every
    slot's kind — the published ``_meta.signal_kinds`` table used for analysis selection.
    A slot whose kind is unregistered, or which carries no kind at all (a
    schema exported by a pre-2.0 silk), constrains nothing: an absence of
    information is not a statement that an analysis cannot feed it.
    """
    from silk.signals.schema import signal_kinds_meta

    kinds = signal_kinds_meta()
    candidates = list(declared)
    for _slot, kind in slots:
        entry = kinds.get(kind)
        if entry is None or not entry['produced_by']:
            continue
        produced = set(entry['produced_by'])
        candidates = [a for a in candidates if a in produced]
    return candidates


def _resolve_metric_analysis(where, slots, declared, analysis, signals=None,
                             fix=None):
    """The analysis a declarative metric reads, or ``None`` for "the only plot".

    THE rule, stated once and applied at all three layers (generate time in
    :func:`silk.codegen.testbench_generator._validate`, run time in
    :meth:`ModuleTestbench.evaluate`, and inside the generated extractor):

    * fewer than two declared analyses — ``None``. One analysis is one plot;
      nothing to select and nothing to ask the user for. Every test silk has
      ever run takes this branch.
    * *analysis* given — it must be one of the declared analyses, and one that
      could actually feed this metric's inputs. Otherwise refuse.
    * exactly one candidate — read it, with no field. An AC-domain metric in a
      tran+ac test is not ambiguous, so asking would be noise.
    * more than one candidate — REFUSE, naming them. This is the ``kind='any'``
      family (``average`` in a tran+ac test): both plots could feed it and
      only the user knows which one they meant.

    *where* is the caller's half of every message (``"test 'rc' metric 'vavg'
    (type 'average')"``) and *signals* the bound signal paths, named in the
    ambiguity refusal so the user sees WHICH probe is ambiguous rather than
    only which metric. *fix*, when given, is called with the candidate
    analyses and returns the closing sentence of the ambiguity refusal, and
    only that sentence: the declarative form's remedy is ``analysis=`` on the
    metric, the function form's is ``analysis=`` on the probe, and
    the diagnosis they share must not fork with them.

    Raises ``ValueError``. ``TestbenchSpecError`` is a
    ``ValueError``, so the generate-time layer raises its own subclass through
    the same call and the wording stays identical.
    """
    if analysis is not None:
        if analysis not in _SIM_KEY_ANALYSIS.values():
            raise ValueError(
                f"{where}: analysis={analysis!r} is not a silk analysis; "
                f"expected one of {sorted(set(_SIM_KEY_ANALYSIS.values()))}.")
        # Checked BEFORE the one-analysis short-circuit below: naming an
        # analysis the test does not declare is a caller error whether or not
        # the choice would have mattered. An EMPTY declared list is not that
        # — it means nothing was declared to compare against (an external
        # rawfile through evaluate(rawfile=…)), so the loader answers.
        if declared and analysis not in declared:
            raise ValueError(
                f"{where}: analysis={analysis!r}, but this test declares "
                f"{_quoted(declared)}. Add {analysis!r} to the test with "
                f"add_sim_type, or name one of the analyses it has.")
    if len(declared) < 2:
        return None
    candidates = sorted(_analysis_candidates(slots, declared))
    if analysis is not None:
        if analysis not in candidates:
            raise ValueError(
                f"{where}: analysis={analysis!r} cannot feed this metric's "
                f"input{'s' if len(slots) != 1 else ''} "
                f"({_slot_inventory(slots)}). Analyses in this test that can: "
                f"{_quoted(candidates) or 'none'}.")
        return analysis
    if len(candidates) > 1:
        what = _quoted(signals) if signals else 'its signal'
        remedy = fix(candidates) if fix is not None else (
            f"Add {' or '.join(f'analysis={a!r}' for a in candidates)} to the "
            f"metric.")
        raise ValueError(
            f"{where}: {what} could come from more than one analysis in this "
            f"test: {_quoted(candidates)}. {remedy}")
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# Probe spellings; see docs/probes.md for authored and raw paths.
# ---------------------------------------------------------------------------

#: The device terminals silk's 0 V ammeter wrappers instrument. Mirrors
#: ``silk.circuit.modules_results._MOS_TERMINALS``; kept local so this module
#: does not import the results mixin at module scope.
_TERMINAL_PORTS = ('d', 'g', 's', 'b')
#: The DUT's emitted instance name at the testbench top level.
DUT_ELEMENT_NAME_PREFIX = emitted_submodule_name(DUT_ELEMENT_NAME)


def _unrecognized_probe_colon_suffix_error(raw, prefix, suffix):
    """Refuse an unknown colon suffix before building a deck.

    ``'<prefix>:<suffix>'`` dispatches on
    *suffix* alone (case-insensitive, already lowercased by the caller) —
    one of the 4 terminals or one of the 17
    :data:`~silk.simulation.backend.DEVICE_OP_PARAMS` names. Neither is
    refused HERE, before any deck is built, naming both closed sets plus a
    difflib hint — silk never guesses which the caller meant."""
    import difflib
    from silk.simulation.backend import DEVICE_OP_PARAMS
    candidates = _TERMINAL_PORTS + DEVICE_OP_PARAMS
    close = difflib.get_close_matches(suffix, candidates, n=3, cutoff=0.5)
    return ValueError(
        f"probe({raw!r}): {suffix!r} is neither a terminal-current suffix "
        f"({_quoted(_TERMINAL_PORTS)}) nor one of the "
        f"{len(DEVICE_OP_PARAMS)} recognised device-parameter names "
        f"({sorted(DEVICE_OP_PARAMS)})"
        + (f". Did you mean {close!r}?" if close else ".") +
        f" A device-parameter probe is refused here, before any deck is "
        f"built, rather than deferred: an unrecognised name can abort "
        f"ngspice's whole write, or come back a real, populated, all-zero "
        f"reading with no warning at all, depending on what else that run "
        f"happens to save."
    )


def _split_spice_selector(text):
    """``('i', 'vs')`` for ``'i(vs)'``, ``('v', 'out')`` for ``'V(out)'``.

    Returns ``(None, text.strip())`` for anything that is not a
    single-letter SPICE selector wrapper, so a raw vector name such as
    ``'v.xdut.xmm1.vamm_d'`` (second character ``'.'``, not ``'('``) passes
    through untouched.
    """
    stripped = text.strip()
    if len(stripped) > 3 and stripped[1] == '(' and stripped.endswith(')'):
        head = stripped[0].lower()
        if head in ('v', 'i'):
            return head, stripped[2:-1].strip()
    return None, stripped


def _branch_vector_name(sources, inner, elements=None):
    """THE ``i(x)`` lookup shared by ``tb.probe`` and a metric's string path.

    *sources* maps silk source names to source objects. A source NAME ``x``
    reads that source's branch current under its emitted instance (``vs`` ->
    ``vvs``). Otherwise ``x`` is a raw vector name and passes through
    verbatim. When ``x`` is both a source name and the emitted instance of a
    DIFFERENT source, ``i(x)`` has two readings and raises
    :class:`~silk.exceptions.AmbiguousBranchCurrentError` naming both -- never a silent pick.
    """
    from silk.circuit.emitted_names import emitted_instance_name
    from silk.exceptions import AmbiguousBranchCurrentError
    from silk.spice_names import to_spice_name
    from silk.sources._layout import source_instances
    instances = {name: (source, prefix) for name, source, prefix
                 in source_instances(sources, elements)}
    sources = {name: source for name, (source, _) in instances.items()}
    want = inner.replace(':', '.').lower()
    want_emitted = to_spice_name(inner, kind='branch').replace(':', '.').lower()
    emitted = {name: prefix + emitted_instance_name(src, str(src.name))
               for name, (src, prefix) in instances.items()}
    physical = {name: [(prefix + p).lower() for p in src.physical_instance_names()]
                for name, (src, prefix) in instances.items()}
    by_name = next((n for n in emitted if n.lower() == want), None)
    by_instance = [n for n, cards in physical.items()
                   if n != by_name and (want in cards or want_emitted in cards)]
    if by_name is not None and by_instance:
        other = by_instance[0]
        card = next(c for c in physical[other] if c in (want, want_emitted))
        # A one-card source is reached by its designer name; a multi-card one
        # (a dedicated PWL) has no default card, so the remedy is the matched
        # physical card itself, in its emitted spelling.
        remedy = other if len(physical[other]) == 1 else next(
            instances[other][1] + c for c in sources[other].physical_instance_names()
            if (instances[other][1] + c).lower() == card)
        raise AmbiguousBranchCurrentError(
            f"i({inner}) is ambiguous: it names source {by_name!r} (emitted "
            f"as {emitted[by_name]!r}, vector i({emitted[by_name].lower()})) "
            f"and it is also the emitted instance of source {other!r} "
            f"(vector i({card})). Write i({remedy}) for "
            f"source {other!r}'s current, or rename a source."
        )
    if by_name is not None:
        source = sources[by_name]
        layout = getattr(source, 'physical_layout', None)
        if layout is not None:
            prefix = instances[by_name][1]
            terms = tuple((prefix + card, sign) for card, sign in layout().current_terms)
            return terms[0][0] if len(terms) == 1 else terms
        return emitted[by_name]
    # No designer source selected: a raw vector name passes through with its
    # ORIGINAL spelling; the one collector considers it and its simulator
    # spelling together (never a retry or a precedence here).
    return inner


def _metric_node_request(path):
    """Classify declarative voltage paths without changing special vector routes."""
    from silk.circuit.node_paths import NodeVoltageRequest
    from silk.simulation.backend import _NOISE_PORT
    head, inner = _split_spice_selector(path)
    if head is None and inner.lower() in _NOISE_PORT:
        return None
    if head == 'i' or '@' in inner or inner.lower().endswith('#branch'):
        return None
    if head == 'v':
        return NodeVoltageRequest(inner)
    if ':' in inner:
        return None
    return NodeVoltageRequest(inner)


def _metric_node_requests(metrics, declared):
    """Snapshot declarative node requests with their diagnostic provenance."""
    for metric in metrics:
        if 'signals' not in metric:
            continue
        # Noise inputs are spectral vectors, not node-voltage requests.
        if (metric['analysis'] == 'noise' or declared == ['noise']
                or _analysis_candidates(metric['slots'], declared) == ['noise']):
            continue
        for path in metric['signals']:
            request = _metric_node_request(path)
            if request is not None:
                yield request, f"Xyce metric {metric['name']!r} signal {path!r}"


def _load_metric_path(ckt, path, **select):
    """Read a metric's string signal path from *ckt*'s results.

    A ``i(x)`` path goes through :func:`_branch_vector_name`, so ``i(vs)``
    and the raw ``i(vvs)`` both work and mean one thing. Voltage paths use
    the same graph-backed resolution as node probes; device parameters and
    colon-qualified vectors keep their existing loader route. A current that
    is not found is re-raised with the accepted spellings.
    """
    from silk.exceptions import AmbiguousSignalError, SimulationError
    head, inner = _split_spice_selector(path)
    if head != 'i':
        request = _metric_node_request(path)
        return ckt.load_signal(path if request is None else request, **select)
    sources = getattr(ckt, 'sig_sources_dict', None) or {}
    vector = _branch_vector_name(sources, inner, getattr(ckt, 'elem_dict', None))
    try:
        if isinstance(vector, tuple):
            signals = [ckt.load_signal(name, 'I', **select) * sign for name, sign in vector]
            result = signals[0]
            for signal in signals[1:]:
                result = result + signal
            return result
        return ckt.load_signal(vector, 'I', **select)
    except AmbiguousSignalError:
        raise                  # an ambiguity is not a missing current
    except SimulationError as exc:
        names = ', '.join(f'i({n})' for n in sorted(str(n) for n in sources)) or 'none'
        raise SimulationError(
            f"{exc} Accepted spellings for a branch current: i(<source "
            f"name>) (this bench's sources: {names}) or the raw vector "
            f"name, e.g. i(vvs) for source 'vs'."
        ) from exc


def _is_plain_net_name(name):
    """True for a bare net name — no hierarchy, no wrapper, no terminal.

    A plain name is the only spelling ``probe()`` validates: a dot means a
    SPICE path into a subcircuit, whose existence depends on what the deck
    saved and is therefore a run-time fact; parentheses and a colon mean a
    current spelling, which names an element rather than a net.
    """
    return not any(ch in name for ch in '.():')


def _attribute_abort_to_device_probe(exc, netlist):
    """Re-raise a :class:`SimulationAborted` with a device-probe-attributed
    supplement when its reason names a currently-pending probe's parameter
    (discovered empirically during the build, real ngspice-42
    run: ``save all`` + one zero-length ``'@...[cgso]'`` in the save list
    can abort the WHOLE ``.control`` ``write`` with no rawfile at all — the
    SAME known ngspice-42 quirk ``device_op_save_lines``'s own docstring
    already documents for ``save_device_params``, also relevant to explicit probes
    whose parameters are outside a model's declared overview set. This is not silently absorbed into a per-probe read-time
    raise: the WHOLE run produced no data, which ``SimulationAborted`` is
    the correct, honest exception for — this only makes the existing
    message ACTIONABLE by naming which probe is implicated, never changes
    what is raised or hides the original ngspice diagnostic.

    Best-effort text correlation only (never a second simulator run): if
    the reason names none of the pending probes' parameters, *exc* is
    returned unchanged.
    """
    pending = getattr(netlist, '_pending_device_probes', None) or ()
    if not pending:
        return exc
    text = str(exc)
    # ngspice's OWN abort text always addresses a device-parameter vector
    # as '<name>[<param>]' (its '@' vector-naming convention), regardless
    # of the ':' spelling probe() itself uses -- so the search key below
    # must stay bracket-shaped to ever match. The name shown back to the
    # caller is rendered in the spelling they actually typed (':').
    named = sorted({f'{name}:{param}' for name, param in pending
                    if f'[{param}]' in text})
    if not named:
        return exc
    cls, extra = SimulationAborted, {}
    if isinstance(exc, ConvergenceError):
        # Keep the convergence classification and its matched lines (TASK-316).
        cls, extra = ConvergenceError, {'matched_lines': exc.matched_lines}
    return cls(
        str(exc) + " This run had pending device-parameter probe(s) "
        f"{named} — one of their requested '@'/N()/ID() vectors is the "
        f"likely cause (a zero-length vector referenced in the save/print "
        f"list can abort the whole write, on some simulator versions). "
        f"Try a different parameter, a newer simulator binary, or drop "
        f"the probe.",
        simulator=exc.simulator, reason=exc.reason,
        data_rows=exc.data_rows, log_path=exc.log_path, **extra,
    )


def _device_probe_requests(probe_registry):
    """``[(designer_name, param), ...]`` for every device-parameter probe
    currently in *probe_registry* — a plain snapshot, taken fresh at every ``run()``,
    of what THIS run will request. Order-preserving, not deduped
    (:func:`~silk.circuit.modules_results.resolve_probe_device_lines`
    dedupes when it builds the actual save/print lines).

    Voltage probes retain their kind in a NodeVoltageRequest or literal
    placement address, even when a node name contains a device-like suffix.
    Skip those carriers before examining punctuation. Genuine device probes
    retain the lowercase ``'<designer-name>:<param>'`` spelling classified
    by ``probe()``; terminal currents already name their resolved ammeter.
    """
    from silk.circuit.node_paths import NodeVoltageRequest
    from silk.simulation.backend import DEVICE_OP_PARAMS
    out = []
    for sig in probe_registry:
        raw = getattr(sig, 'name', '') or ''
        if (isinstance(raw, NodeVoltageRequest)
                or getattr(raw, '_literal_voltage', False) or ':' not in raw):
            continue
        prefix, _, suffix = raw.rpartition(':')
        if suffix in DEVICE_OP_PARAMS:
            out.append((prefix, suffix))
    return tuple(out)


def _current_probe_requests(probe_registry, metrics=(), declared=()):
    """Emitted source-instance names and analysis selectors for current probes,
    in registry order. Ammeter-vector probes
    (``'xdut.m1:d'``, an ngspice-only construction) are not source branches
    and are skipped. Both backends acquire the named physical operands."""
    # Infer per metric use, never on the shared leaf: one expression may
    # feed both a transient slot and an AC slot, which must still refuse.
    selections = {}
    for metric in metrics:
        if not _is_function_form_metric(metric):
            continue
        leaves = {slot: list(_unbound_leaves(value))
                  for slot, value in metric['slots'].items()}
        if not any(getattr(leaf, '_selector', None) == 'I'
                   for group in leaves.values() for leaf in group):
            continue
        analyses = _function_form_slot_analyses(metric, declared)
        for slot, group in leaves.items():
            for leaf in group:
                analysis = getattr(leaf, '_analysis', None) or analyses[slot]
                selections.setdefault(id(leaf), []).append(analysis)
    out = []
    for sig in (leaf for probe in probe_registry for leaf in _unbound_leaves(probe)):
        if getattr(sig, '_selector', None) != 'I':
            continue
        raw = getattr(sig, 'name', '') or ''
        if '.vamm_' in raw:
            continue
        for analysis in selections.get(id(sig), [getattr(sig, '_analysis', None)]):
            request = (raw, analysis)
            if request not in out:
                out.append(request)
    return tuple(out)


def _current_probe_names(probe_registry):
    return tuple(name for name, _ in _current_probe_requests(probe_registry))


def _element_entry(module, name):
    """``(key, elem_dict record)`` for *name* in *module*, case-insensitively.

    SPICE is case-insensitive and the rawfile is lower-cased, so a probe
    spelled ``'xdut.m1:d'`` must find an element registered as ``'M1'``.
    ``None`` when the module has no such element.
    """
    elements = getattr(module, 'elem_dict', None) or {}
    wanted = name.lower()
    for key, record in elements.items():
        if str(key).lower() == wanted:
            return str(key), record
    return None


def _instance_names(module, element_type):
    """Sorted element keys of one kind — for a refusal that lists the options."""
    elements = getattr(module, 'elem_dict', None) or {}
    return sorted(k for k, r in elements.items()
                  if r.get('type') == element_type)


# ---------------------------------------------------------------------------
# Function-form metric evaluation.
#
# Resolution of a metric's Signal slots happens HERE and nowhere else: the
# extractor stored in the metric dict is a thin closure over
# :func:`_evaluate_function_form_metric`, and the evaluation loop in
# ``_simulate``'s tail calls that extractor. One resolution point means one
# place where a binding failure is recognised and re-raised as
# :class:`~silk.exceptions.MetricBindingError`.
# ---------------------------------------------------------------------------

def _unbound_leaves(value, *, logical_probes=False):
    """Yield every unbound leaf :class:`Signal` reachable through *value*.

    A leaf is an unbound Signal with no recipe — a probe. A recipe node is
    walked into: ``(op, *operands)`` for operators and
    ``(method_name, receiver, args, kwargs)`` for transforms,
    where the receiver and any Signal-valued argument are themselves
    expressions. A bound Signal carries its own data and belongs to no
    bench, so it is not walked; neither is a scalar. With logical_probes,
    stop at a composite returned by probe() for registry reconstruction;
    acquisition and rebinding always walk through to its physical leaves.
    """
    from silk.signals.base import Signal
    if not isinstance(value, Signal) or not value._is_unbound():
        return
    for leaf in value._walk_leaves(logical_probes=logical_probes):
        if leaf._is_unbound():
            yield leaf


def _bench_label(bench):
    """Name a bench and its selected owner in a cross-bench refusal."""
    test_name = getattr(bench, 'test_name', None) or '<unnamed>'
    module = getattr(bench, 'module', None)
    module_name = getattr(module, 'module_name', None)
    label = (f'{test_name!r} of module {module_name!r}' if module_name
             else f'{test_name!r}')
    from silk.circuit.test_placement import SelectedTestbench
    if isinstance(bench, SelectedTestbench):
        entry = bench._entry()
    else:
        selection = current_placement(bench)
        entry = selection.entry if selection is not None else None
    if entry is not None:
        label += f' at {entry.owner_path!r} (root {id(entry._root_ref()):#x})'
    return label


def _describe_slot_value(value):
    """How a slot value is named in a binding-failure message."""
    from silk.signals.base import Signal
    if isinstance(value, str):
        return f'path {value!r}'
    if isinstance(value, Signal):
        if not value._is_unbound():
            return f'signal {value.name!r}'
        if value.__dict__.get('_recipe') is None:
            return f'probe {value.name!r}'
        return f'expression {value.name!r}'
    return repr(value)


def _probe_analysis_fix(candidates):
    """The function form's remedy sentence for an ambiguous slot.

    The declarative form's answer is ``analysis=`` on the metric; a probe's
    is ``analysis=`` on the probe. Only this sentence differs between the
    two refusals — the diagnosis above it is one string in one place.
    """
    spellings = ' or '.join(
        f"tb.probe(..., analysis={a!r})" for a in candidates)
    return f"Name the analysis on the probe: {spellings}."


def _leaf_analyses(value):
    """``(the analyses this slot's probes pin, whether any probe pins none)``.

    A probe pins an analysis when it was created as
    ``tb.probe(name, analysis='ac')`` (the contract in docs/probes.md). An expression
    is walked to its leaves, so ``vp + vm`` reports what its two probes pin
    between them.
    """
    pinned, unpinned = [], False
    for leaf in _unbound_leaves(value):
        word = leaf.__dict__.get('_analysis')
        if word is None:
            unpinned = True
        elif word not in pinned:
            pinned.append(word)
    return pinned, unpinned


def _function_form_slot_analyses(metric, declared):
    """``{slot name: the analysis that slot reads, or None}`` for a
    function-form metric — rule applied to probes.

    THE point where the two features meet. The rule is the one
    :func:`_resolve_metric_analysis` states for the declarative form, asked
    once per slot because each slot has its own kind and its own probe:

    * a probe created with ``analysis=`` names the plot, and that word is
      validated against the declared analyses and against the slot's kind
      exactly as the declarative ``analysis=`` is;
    * a probe without one is settled by the slot's kind — ``dc_gain_dB``'s
      ``frequency_response`` input in a tran+ac test reads the AC plot with
      nothing said, because ``.ac`` is the only declared analysis that
      produces that kind;
    * a slot whose kind more than one declared analysis could feed
      (``average``'s ``any`` in a tran+ac test) is REFUSED, naming the
      candidates and the probe spelling that settles it.

    A slot bound to a ``Signal`` that already carries data selects nothing:
    the samples are in hand, and no plot is read. Neither does a slot bound
    to anything that is not a path or a Signal at all. An expression whose probes
    pin *different* analyses (``vout_t`` and ``vout_f`` in one recipe) is
    left to its leaves — each already names its own plot, so there is
    nothing left to choose — and every word is still validated.

    Raises :class:`~silk.exceptions.MetricBindingError`, which propagates through ``run_all_tests``, carrying
    :func:`_resolve_metric_analysis`'s own wording so the two forms refuse
    the same ambiguity in the same sentence.
    """
    from silk.exceptions import MetricBindingError
    from silk.signals.base import Signal

    name = metric['name']
    fn = metric['fn']
    fn_name = getattr(fn, '__name__', repr(fn))
    where = f"add_metric({name!r}, {fn_name})"
    # A plain-function metric (TASK-323) carries its kinds on the dict — it
    # has no schema to read them from; a registered one reads its schema.
    kinds = (dict(metric['slot_kinds']) if 'slot_kinds' in metric
             else dict(_metric_signal_slots(fn)))

    def settle(slots, explicit, signals):
        try:
            return _resolve_metric_analysis(
                where, slots, declared, explicit, signals=signals,
                fix=_probe_analysis_fix)
        except ValueError as exc:
            raise MetricBindingError(str(exc)) from exc

    out = {}
    for slot, value in metric['slots'].items():
        # Only the two slot values that READ A PLOT get one chosen: a path
        # string and an unbound Signal. A Signal that already carries data
        # has its samples in hand, and anything else (a bare array a caller
        # handed straight to the metric) never reaches the loader at all —
        # neither can be ambiguous, so neither may be refused as such.
        if not isinstance(value, (str, Signal)) or (
                isinstance(value, Signal) and not value._is_unbound()):
            out[slot] = None
            continue
        pinned, unpinned = _leaf_analyses(value)
        if len(pinned) > 1 and not unpinned:
            for word in pinned:         # validate each; choose nothing
                settle([], word, None)
            out[slot] = None
            continue
        label = value if isinstance(value, str) else getattr(value, 'name', slot)
        out[slot] = settle(
            [(slot, kinds.get(slot))], pinned[0] if pinned else None, [label])
    return out


def _resolve_slot_value(ckt, value, analysis=None):
    """Bind one slot value against the results loaded on *ckt*.

    A ``str`` is a signal path, resolved through ``ckt.load_signal`` exactly
    as the declarative form does. A ``Signal`` that already carries data is
    used as is; an unbound one (a probe, or an expression over probes) is
    resolved through ``Signal.resolve()``, so each leaf reads its own bench's
    current results and nothing is stored back. Anything else passes through
    unchanged.

    *analysis* is the plot :func:`_function_form_slot_analyses` settled for
    this slot, and it is a DEFAULT, not an override: a probe that
    names its own analysis keeps it, and a leaf that names none reads this
    one. It is passed on only when there is something to select, so a
    single-analysis test's ``load_signal`` call is the one it always made,
    down to its argument list.

    The test is ``_is_unbound()`` — does this Signal carry samples — and
    NOT the ``bound`` property, which asks the different question "would
    reading ``.x``/``.y`` return data right now". After a run the latter is
    ``True`` for an unresolved probe, so testing it handed the probe itself
    to the metric and let the metric body resolve it lazily through
    ``__getattr__``: a loader failure then surfaced from inside the body and
    was buried as a report string. Resolving the slot here lets the binding
    failure propagate.
    """
    from silk.signals.base import Signal
    if isinstance(value, str):
        if analysis is None:
            return _load_metric_path(ckt, value)
        return _load_metric_path(ckt, value, analysis=analysis)
    if isinstance(value, Signal):
        if not value._is_unbound():
            return value
        return value.resolve(analysis=analysis)
    return value


def _evaluate_function_form_metric(ckt, metric):
    """Resolve a function-form metric's slots, then call its function.

    THE single resolution point for the function form. Two failures are the
    caller's, not the circuit's, and both raise
    :class:`~silk.exceptions.MetricBindingError` (a ``ConfigurationError``,
    so ``Module.run_all_tests`` propagates it instead of burying it as this
    test's ``{'error': str}``):

    * **Resolution.** A name absent from the results (``SimulationError``
      from the loader), an operator or transform refusing at resolution
      (``ValueError``), or a probe whose bench has not run, has no netlist
      or has gone away (``UnboundSignalError``).
    * **The kind guard.** ``SignalKindError`` is raised inside the metric
      body — an AC metric handed a transient trace — but it is a statement
      about the binding, so it is promoted here too (the contract in docs/probes.md,
      the metric failure convention). Every other error inside the metric body keeps today's
      report-string convention and is left to the evaluation loop.

    Which plot each slot reads is settled first, by
    :func:`_function_form_slot_analyses`, against the analyses
    this test declares *now* — a hand-built test may add sim types after its
    metrics, which is why the question cannot be answered in ``add_metric``.
    That resolution refuses an ambiguous slot with the same
    :class:`~silk.exceptions.MetricBindingError`.
    """
    from silk.exceptions import (AmbiguousBranchCurrentError,
                                 MetricBindingError, SimulationError,
                                 UnboundSignalError)
    from silk.signals.kinds import SignalKindError

    name = metric['name']
    fn = metric['fn']
    fn_name = getattr(fn, '__name__', repr(fn))
    plots = _function_form_slot_analyses(metric, _declared_analyses(ckt))
    resolved = {}
    for slot, value in metric['slots'].items():
        try:
            resolved[slot] = _resolve_slot_value(ckt, value, plots.get(slot))
        except AmbiguousBranchCurrentError:
            raise
        except (SimulationError, UnboundSignalError, ValueError) as exc:
            raise MetricBindingError(
                f"metric {name!r} ({fn_name}): cannot resolve {slot}="
                f"{_describe_slot_value(value)} against this test's results: "
                f"{exc}"
            ) from exc
    try:
        return fn(**resolved, **metric['params'])
    except SignalKindError as exc:
        raise MetricBindingError(
            f"metric {name!r} ({fn_name}): the signal bound to this metric is "
            f"the wrong kind — {exc}"
        ) from exc


def _is_function_form_metric(metric):
    """True for a metric dict built by ``add_metric``'s function form.

    The discriminator is ``'fn'``: only that form stores the registered
    metric function alongside its raw ``slots`` and ``params``. The
    declarative (``type=``) and plain-callable forms store exactly
    ``{'name', 'extractor'}``.
    """
    return isinstance(metric, dict) and 'fn' in metric and 'slots' in metric


def _function_form_extractor(metric):
    """Build the extractor a function-form metric dict stores.

    A thin closure over the dict, so all resolution and all
    :class:`~silk.exceptions.MetricBindingError` wrapping stay in
    :func:`_evaluate_function_form_metric`. Factored out of ``add_metric``
    because :meth:`ModuleTestbench.__setstate__` has to build the same thing
    again: a closure cannot be pickled, so it is dropped on the way out and
    rebuilt on the way in.
    """
    return lambda ckt, _metric=metric: _evaluate_function_form_metric(ckt, _metric)


def _bind_plain_function(name, fn, extra_positional, kwargs):
    """``(slots, params, slot_kinds)`` for a plain function given probes.

    The plain-function form: an UNDECORATED ``fn`` handed keyword arguments.
    Nothing is registered and nothing is read from the metric catalog, so
    everything here comes from ``fn``'s own signature and from the values
    bound to it:

    * the keyword NAMES are checked against the signature, now, so a
      misspelled or missing one raises before any simulation;
    * a keyword whose VALUE is a :class:`~silk.signals.base.Signal` (a probe,
      an expression over probes, or a Signal that already carries data) is a
      signal slot; every other value is a scalar param passed through as is.
      A ``str`` is always a scalar here, never a signal path — a plain
      function has nothing that tells ``edge='rising'`` from a net name;
    * a slot's kind is read from an ``Annotated[Signal, KIND]`` annotation
      when the author wrote one, and is ``None`` otherwise. ``None`` narrows
      no analysis, so an unpinned probe in a multi-analysis test is refused
      by :func:`_resolve_metric_analysis` rather than guessed.

    Raises ``TypeError`` for every binding mistake, naming the metric.
    """
    import typing
    from silk.signals.base import Signal
    from silk.signals.schema import _signal_slot

    fn_name = getattr(fn, '__name__', repr(fn))
    where = f"add_metric({name!r}, {fn_name})"
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{where}: cannot read {fn_name}'s signature ({exc}), so its "
            f"keyword arguments cannot be bound by name.") from exc
    named = [p for p in signature.parameters.values()
             if p.kind not in (inspect.Parameter.VAR_POSITIONAL,
                               inspect.Parameter.VAR_KEYWORD)]
    keywordable = [p.name for p in named
                   if p.kind is not inspect.Parameter.POSITIONAL_ONLY]
    takes_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD
                       for p in signature.parameters.values())
    listing = _quoted(keywordable) or 'none'

    if extra_positional:
        n_extra = len(extra_positional)
        raise TypeError(
            f"{where}: signal slots are bound by keyword name, not position "
            f"({n_extra} extra positional argument{'s' if n_extra != 1 else ''} "
            f"given). {fn_name}'s parameters are {listing}; bind each probe "
            f"by its parameter name, e.g. add_metric({name!r}, {fn_name}, "
            f"<param>=tb.probe(...)).")

    unknown = [] if takes_var_kw else [k for k in kwargs if k not in keywordable]
    if unknown:
        raise TypeError(
            f"{where}: unknown keyword argument{'s' if len(unknown) != 1 else ''} "
            f"{_quoted(unknown)}; {fn_name}'s parameters are {listing}.")

    missing = [p.name for p in named
               if p.default is inspect.Parameter.empty and p.name not in kwargs]
    if missing:
        raise TypeError(
            f"{where}: missing parameter{'s' if len(missing) != 1 else ''} "
            f"{_quoted(missing)} (no default). {fn_name}'s parameters are "
            f"{listing}; bind each signal input to a probe by keyword, e.g. "
            f"{missing[0]}=tb.probe(...).")

    try:
        hints = typing.get_type_hints(fn, include_extras=True)
    except Exception:           # unresolvable annotations: no kinds, no guard
        hints = {}

    slots, params, slot_kinds = {}, {}, {}
    for key, value in kwargs.items():
        is_signal_param, kind = _signal_slot(hints.get(key))
        if isinstance(value, Signal):
            slots[key] = value
            slot_kinds[key] = kind.name if kind is not None else None
        elif is_signal_param:
            raise TypeError(
                f"{where}: {key} is annotated as a Signal but was given "
                f"{value!r}. {fn_name} is not a @metric, so a string is never "
                f"read as a signal path; bind a probe: "
                f"{key}=tb.probe({value!r}).")
        else:
            params[key] = value

    if not slots:
        raise TypeError(
            f"{where}: none of the keyword arguments is a Signal. {fn_name} "
            f"is not a @metric, so its signal slots are the keywords bound to "
            f"a probe or a Signal (e.g. sig=tb.probe('vout')); a string is a "
            f"scalar, never a signal path. To use {fn_name} as an "
            f"extractor(ckt) instead, pass no keyword arguments.")
    return slots, params, slot_kinds


class _PlainMetricFunction:
    """Pickle stand-in for a plain-function metric's ``fn``.

    A lambda or a locally defined function cannot be pickled, and the bare
    ``PicklingError`` Python raises names the function object and nothing
    else — not the metric, not the bench. :meth:`ModuleTestbench.__getstate__`
    puts one of these in place of each plain-function metric's ``fn``; its
    ``__reduce__`` tries the function first and re-raises naming the metric
    and the fix. A picklable function round-trips to itself, so an unpickled
    metric dict holds the function, never this wrapper. ``copy.copy``, which
    routes through the same ``__getstate__``/``__setstate__`` pair but never
    reduces the state's contents, gets it unwrapped in ``__setstate__``.
    """

    __slots__ = ('metric_name', 'fn')

    def __init__(self, metric_name, fn):
        self.metric_name = metric_name
        self.fn = fn

    def __reduce__(self):
        import pickle
        try:
            pickle.dumps(self.fn)
        except Exception as exc:
            fn_name = getattr(self.fn, '__name__', repr(self.fn))
            raise pickle.PicklingError(
                f"metric {self.metric_name!r}: its function {fn_name} cannot "
                f"be pickled ({exc}). A bench is pickled when it is sent to "
                f"another process (a FaultCampaign, for one); define the "
                f"metric function at module level instead of as a lambda or "
                f"a nested def.") from exc
        return (_return_as_is, (self.fn,))


def _return_as_is(value):
    """Unpickle helper for :class:`_PlainMetricFunction`: the function itself."""
    return value


def _metric_slot_leaves(metrics, *, logical_probes=False):
    """Yield every unbound leaf Signal reachable from *metrics*' slot values.

    Only the named slots are walked: a probe captured inside a user
    lambda is not reachable from any data structure the bench owns, so it
    cannot be rebound, and that is a stated limit of keeping lambdas legal.
    """
    for metric in metrics or ():
        if not _is_function_form_metric(metric):
            continue
        slots = metric.get('slots')
        if not isinstance(slots, dict):
            continue
        for value in slots.values():
            yield from _unbound_leaves(value, logical_probes=logical_probes)


def _registry_from_metric_leaves(metrics):
    """Registry entries (strong references) for every distinct
    probe reachable from *metrics*' slot values.

    Used to REBUILD :attr:`ModuleTestbench._probe_registry` after a copy or
    an unpickle: a naive copy of the registry list would either keep
    pointing at the ORIGINAL bench's probes (deep-copying a leaf whose
    ``_bench`` weak reference is not yet rebound) or duplicate work already
    done by the metric-slot rebinding walk, so the registry is instead
    rebuilt from the (already rebound) metric slots — giving ``tb.plot()``
    with no arguments a correct list on the copy or the restored bench, at
    the same reach the design's rebinding already has: only probes reachable
    from a metric slot come along. A probe minted only for plotting, never
    passed to ``add_metric``, does not survive a copy or a pickle round
    trip — the same stated limit as a probe captured in a user lambda
    (see docs/known_issues.md); strong references prevent premature collection on the SAME bench,
    not this copy-time reach.
    """
    seen = set()
    registry = []
    for leaf in _metric_slot_leaves(metrics, logical_probes=True):
        if id(leaf) not in seen:
            seen.add(id(leaf))
            registry.append(leaf)
    return registry


def _rebind_probes(bench, metrics, selects):
    """Point every probe in *metrics* that *selects* accepts at *bench*.

    *selects* is called with the leaf's current ``_bench`` entry and decides
    whether this bench owns it. Two callers, two different questions:
    :meth:`ModuleTestbench.__deepcopy__` rebinds the leaves that still point
    at the bench being copied, and :meth:`ModuleTestbench.__setstate__` the
    leaves that pickling detached. Anything else — a probe of a third bench,
    a probe that never had one — is left exactly as it is.

    Returns the number of leaves rebound.
    """
    ref = None
    rebound = 0
    for leaf in _metric_slot_leaves(metrics):
        if not selects(leaf.__dict__.get('_bench')):
            continue
        if ref is None:
            ref = weakref.ref(bench)
        leaf.__dict__['_bench'] = ref
        rebound += 1
    return rebound


def _retained_sweep(reports):
    """The ``{corner: rawfile path}`` table a completed sweep actually retains.

    *reports* maps corner name to :class:`~silk.circuit.spec_report.
    SpecReport`, in the order the corners ran. A corner with no recorded rawfile
    path contributes nothing.

    The one real rule is **one path, one corner**: if two corners ever did
    land on the same file, an earlier one would be dropped in favour of the
    later one rather than this table lying about which corner a shared file
    actually holds. ``run_corners()`` always gives every
    corner its own ``<testbench dir>/<corner>/`` folder before calling this,
    so two corners landing on the same path can no longer happen in
    practice — this stays purely defensive (originally collapse
    rule for the old shared-unlabeled-file overwrite-mode behaviour).
    """
    table = {}
    for corner, report in reports.items():
        path = getattr(report, 'rawfile_path', None)
        if path is None:
            continue
        key = os.path.normpath(path)
        for earlier, earlier_path in list(table.items()):
            if os.path.normpath(earlier_path) == key:
                del table[earlier]
        table[corner] = path
    return table


@dataclass
class _RunSpec:
    """A picklable ``run()`` invocation, ready for :meth:`ModuleTestbench.
    _simulate_run`.

    Produced by :meth:`ModuleTestbench._prepare_run` once the netlist is
    deck-ready (directory made, folder/file name/output mode/
    device-current capture/declared supply all set exactly as ``run()`` sets
    them today). Holds no reference to the owning :class:`ModuleTestbench` —
    only the netlist :class:`~silk.circuit.modules.Module` (already proven
    picklable by ``FaultCampaign``) and plain values — so it can cross a
    process boundary on its own, ready for the executor.
    """
    netlist: object
    corner_str: str
    tb_dir: str
    invocation: tuple
    test_name: str = None
    cache_key: object = None
    location: object = None
    declared_supply: object = None
    execution_prepared: bool = False
    sharing_identity: object = None
    variable_values: tuple = ()
    attempt_id: str = field(default_factory=lambda: uuid4().hex)
    attempt_context: object = field(init=False, repr=False)

    def __post_init__(self):
        self.attempt_context = _build_attempt_context(self)


    def __getstate__(self):
        from silk.simulation._worker_pickle import dump_worker_netlist
        from silk.circuit.test_placement import prepared_execution
        with prepared_execution(self):
            return dict(self.__dict__, netlist=dump_worker_netlist(self.netlist))

    def __setstate__(self, state):
        import pickle
        self.__dict__.update(state, netlist=pickle.loads(state['netlist']))

    def __deepcopy__(self, memo):
        new = type(self).__new__(type(self))
        memo[id(self)] = new
        new.__dict__.update(copy.deepcopy(self.__dict__, memo))
        return new


@dataclass
class _RunSimResult:
    """What :meth:`ModuleTestbench._simulate_run` returns: enough for
    :meth:`ModuleTestbench._evaluate_run` to re-attach the (possibly
    worker-produced) netlist to the live bench and run the existing
    ``evaluate()`` tail with the silk-origin conditions it recorded.
    """
    netlist: object
    corner_str: str
    conditions: object
    invocation: tuple
    cache_key: object = None
    run_id: str = None
    run: object = None      # TASK-499 RunRecord of the launch that produced this


    def __getstate__(self):
        raw = getattr(self.netlist, '_raw', None)
        if getattr(raw, '_generation', None) is None:
            return dict(self.__dict__)
        from silk.simulation._worker_pickle import dump_worker_netlist
        from silk.simulation.generation import transfer_result
        transfer_result(raw)
        from types import SimpleNamespace
        netlist = self.netlist
        if isinstance(netlist, SimpleNamespace):
            netlist = SimpleNamespace(**dict(netlist.__dict__, _raw=None, simResultsAll=None))
        return dict(self.__dict__, netlist=dump_worker_netlist(netlist),
                    _raw_descriptor=dict(raw.__dict__, native=None, _needs_generation_read=raw.native is not None))

    def __setstate__(self, state):
        descriptor = state.pop('_raw_descriptor', None)
        if descriptor is None:
            self.__dict__.update(state)
            return
        import pickle
        from silk.simulation.backend import RawResult
        netlist = pickle.loads(state['netlist'])
        raw = RawResult(None, descriptor['source_path'])
        raw.__dict__.update(descriptor)
        # Pool result-handler threads must never read or validate artifacts:
        # a failed unpickle would kill that thread and strand ar.get().
        netlist._raw, netlist.simResultsAll = raw, None
        self.__dict__.update(state, netlist=netlist)


@dataclass
class _RunOutcome:
    """One attempt's result or error and whether it replaced current evidence.

    ``None`` means transport failed without delivering the attempt's fact.
    It must not be interpreted as either a launch or a pre-launch refusal.
    """
    result: object = None
    error: object = None
    replacement_started: object = None
    error_text: str = None

    error_type: str = None
    attempt_context: object = None
    failure_stage: str = None
    completed_at: float = field(default_factory=time.time)
    # TASK-499: the original error's identity and the attempt's files, captured
    # before the pool packing fallback can replace the class.
    error_class: str = None
    details: dict = None
    run: object = None
    error_fallback: bool = False   # the carried error is a stand-in (see __getstate__)

    def __post_init__(self):
        if self.error is not None:
            from silk.circuit.test_failure import failure_facts
            error_type, error_class, details = failure_facts(self.error)
            if self.error_type is None:
                self.error_type = error_type
            if self.error_class is None:
                self.error_class = error_class
            if self.details is None:
                self.details = details

    def __getstate__(self):
        # Built-in errors can carry constructor state outside .args (e.g.
        # OSError.filename). Silk errors additionally carry keyword-only
        # metadata that cannot be reconstructed by normal exception pickle.
        # Ordinary failures used to cross workers as text: a local error class
        # or an unpicklable attribute falls back to the nearest reconstructible
        # ancestor with the message, without losing the launch fact. The
        # fallback is flagged so a stand-in never passes for the original.
        from silk.circuit.test_failure import pack_exception
        packed, fallback = (None, False)
        if self.error is not None:
            packed, fallback = pack_exception(self.error)
        return dict(self.__dict__, error=packed,
                    error_fallback=self.error_fallback or fallback)

    def __setstate__(self, state):
        from silk.circuit.test_failure import unpack_exception
        self.__dict__.update(state, error=unpack_exception(state['error']))


def _attempt_context(spec):
    return spec.attempt_context


def _build_attempt_context(spec):
    """Prepared request facts survive even when no worker result returns."""
    from silk.simulation.conditions import encode_condition
    request = spec.invocation[0] if spec.invocation else None
    sample = getattr(spec.netlist, '_mc_sample', None)
    return dict(attempt_id=spec.attempt_id,
                source='behavioral' if spec.netlist.simulator == 'behavioral' else 'spice',
                test_name=spec.test_name, suppressed=sample is not None,
                sample=(dict(index=getattr(sample, 'index', None), seed=getattr(sample, 'seed', None))
                        if sample is not None else None),
                requested_conditions=dict(corner=encode_condition(spec.corner_str),
                    temperature=encode_condition(getattr(request, 'temperature', None)),
                    voltage=encode_condition(spec.declared_supply)))


def _simulate_run_outcome(spec, simulate=None):
    """Capture simulation failures before transporting an outcome to its owner."""
    import traceback
    from silk.circuit.modules_sim import _track_result_replacement, _track_attempt_stage
    context = _attempt_context(spec)
    from silk.simulation.generation import attempt_scope
    from silk.simulation.launch_fact import track_launch
    from silk.circuit.test_placement import prepared_execution
    with prepared_execution(spec), attempt_scope(spec.attempt_id), _track_result_replacement(spec.netlist) as state, _track_attempt_stage() as stage, track_launch() as launch:
        try:
            result = (ModuleTestbench._simulate_run if simulate is None else simulate)(spec)
        except Exception as error:
            failure_stage = getattr(error, '_silk_attempt_stage', stage[0])
            run = getattr(error, '_silk_run_record', None)
            if run is None and launch['launched']:
                # The real launch boundary was crossed (SPICE: the process
                # start; behavioral: engine entry), not merely the broader
                # simulation stage, which also covers refusals before launch.
                from silk.circuit.test_failure import capture_run_record
                run = capture_run_record(spec.netlist, files=launch['files'],
                                         log_path=getattr(error, 'log_path', None))
                try:
                    error._silk_run_record = run
                except Exception:
                    pass
            return _RunOutcome(error=error, replacement_started=state[0],
                               attempt_context=context, run=run,
                               failure_stage=failure_stage,
                               error_text=f'{type(error).__name__}: {error}\n'
                                          f'{traceback.format_exc()}')
        return _RunOutcome(result=result, replacement_started=state[0],
                           attempt_context=context)


def _spec_attempt_scope(test):
    from contextlib import nullcontext
    from silk.specs.evidence import _current, owner_scope
    from silk.circuit.test_placement import evidence_index
    if _current.get() is not None:
        return nullcontext()
    return owner_scope(evidence_index(test.module), '$')


#: Exactly what ``Module.run_simulator``/``read_raw_file`` set on a
#: successful simulate (``_evaluate_run`` copies these,
#: and only these, from a corner's deep-copied netlist onto the LIVE one —
#: see ``_evaluate_run``'s docstring for why not the whole ``__dict__``).
_SIMULATED_STATE_ATTRS = (
    '_raw', 'simResultsAll', 'sim_results_loaded', 'results_file',
    'dataFile', '_last_sim_corner', '_last_sim_label', '_behavioral_mode',
    'operating_conditions',
    # TASK-306 provenance facts, recorded wherever the simulate step ran.
    '_launch_record', '_prepared_deck_record',
    # The Monte Carlo sample the results are from (TASK-427), or None.
    '_mc_sample',
)

_UNSET = object()


def _corner_simulate_task(run_dir, spec):
    """:class:`~silk.simulation.executor.RunTask` ``fn`` for one corner of a
    :meth:`ModuleTestbench.run_corners` sweep.

    Module-level (not a closure or bound method) so it stays picklable for
    the executor's pool path; ignores *run_dir* (the executor creates it,
    but ``_prepare_run`` already made *spec*'s own ``tb_dir`` before this
    runs) and just drives :meth:`ModuleTestbench._simulate_run` — proven
    picklable across a process boundary.
    """
    return _simulate_run_outcome(spec)


@dataclass(frozen=True)
class _PlannedRun:
    """One run of a test's declared plan.

    :meth:`ModuleTestbench._plan_runs` expands a test's declared settings
    into these; ``run()`` and both ``run_all_tests`` walks prepare, simulate
    and evaluate every one of them. *key* is the run's key in the
    :class:`~silk.circuit.spec_report.CornerResults` the plan returns (the
    corner name for a declared corner list, or full corner/temperature/supply
    tuple for a grid); *corner_subdir* is the folder
    under the testbench directory its deck goes to, so every planned run has
    its own folder whatever the output mode (a tuple nests one folder per
    component). *sample* is a Monte Carlo sample's ``_SampleSpec`` and
    *monte_carlo* the ``MonteCarloSettings`` its plan was built from; both are ``None`` for a corner run.
    """
    key: object
    corner: str
    corner_subdir: object
    sample: object = None
    monte_carlo: object = None
    temperature: object = None
    supply: object = None
    supply_source: object = None
    grid: bool = False
    temperature_basis: str = 'configured-and-observed'
    temperature_option: object = None
    variable_values: tuple = ()

    @property
    def point(self):
        return self.key[0] if self.sample is not None else self.key



@contextmanager
def _planned_conditions(test, run):
    """Apply one scalar point privately; restore authored state on every exit."""
    if run is None or not run.grid:
        yield
        return
    netlist = test.netlist
    names = ('_temperature_explicit', '_temperature_basis', '_temperature_list',
             '_supply_explicit', '_supply_voltage', '_supply_source', '_supply_list',
             '_simulator', '_inherited_simulator_option_keys')
    saved = {name: getattr(netlist, name, _UNSET) for name in names}
    variables = [(var, var._value) for var, value in run.variable_values]
    options = dict(netlist.simulator_options)
    sources = [(source, source.dc) for source in netlist.sig_sources_dict.values()
               if hasattr(source, 'dc')]
    try:
        for var, value in run.variable_values:
            var.value = value
        if run.variable_values:
            from silk.utils._utils import _actual_values
            for source, value in sources:
                source.dc = _actual_values(value)
        netlist._temperature_list = netlist._supply_list = (None, None)
        netlist._temperature_explicit = netlist._supply_explicit = True
        netlist._temperature_basis = run.temperature_basis
        # Preserve native spelling and raw TEMP evidence: a known raw .temp
        # is a coordinate, not permission to inject a second native option.
        if run.temperature_option is None:
            netlist.simulator_options.pop('temp', None)
        else:
            netlist.simulator_options['temp'] = run.temperature_option
        netlist._supply_voltage, netlist._supply_source = run.supply, run.supply_source
        if run.variable_values and run.supply is None and run.supply_source is not None:
            from silk.circuit.variable import _leaf_variables
            from silk.simulation.conditions import prepare_supply
            source = netlist.sig_sources_dict[run.supply_source]
            binding = next(value for saved_source, value in sources if saved_source is source)
            if any(leaf is var for leaf in _leaf_variables(binding) for var, _ in run.variable_values):
                # A Variable-driven rail is this point's scalar condition; the
                # normal launch path must verify it just as it verifies a list.
                netlist._supply_voltage = prepare_supply(netlist, apply=False).voltage
        yield
    finally:
        netlist.simulator_options.clear()
        netlist.simulator_options.update(options)
        for name, value in saved.items():
            if value is _UNSET:
                netlist.__dict__.pop(name, None)
            else:
                setattr(netlist, name, value)
        for source, value in sources:
            source.dc = value
        for var, value in variables:
            var._value = value

def _planned_simulate_task(run_dir, spec):
    """:class:`~silk.simulation.executor.RunTask` ``fn`` for one planned run, used alike by ``run()`` and the parallel ``run_all_tests``
    walk so a run fails with the same text either way.

    Returns the shared outcome with its replacement fact and typed error.
    Configuration errors propagate in the parent; ordinary failures retain
    the traceback used by the corner collection.
    """
    return _simulate_run_outcome(spec)


def _chunk_simulate_task(run_dir, specs, chunk_dir):
    """:class:`~silk.simulation.executor.RunTask` ``fn`` for one chunk of
    Monte Carlo samples that share one ngspice process.

    Each member runs the ordinary per-sample pipeline
    (:func:`_simulate_run_outcome`); only the process launch is taken from the
    chunk (``silk.simulation.mc_chunk``). A member the process did not finish,
    or that could not join one (a deck found uncertifiable once emitted), is
    rerun once, alone, after the others. Returns ``(outcomes, rerun)``: one
    shared outcome per spec and ``(position, reason)`` for each rerun.
    """
    from silk.simulation import mc_chunk
    members = [mc_chunk.ChunkMember(spec.netlist._mc_sample.index,
                                    spec.netlist._mc_sample.seed)
               for spec in specs]
    chunk = mc_chunk.ChunkRun(chunk_dir, members)
    outcomes, rerun = [], []
    for position, (spec, member) in enumerate(zip(specs, members)):
        with mc_chunk.active(chunk, member.index):
            outcome = _simulate_run_outcome(spec)
        if isinstance(outcome.error, mc_chunk.ChunkUnfinished):
            rerun.append((position, str(outcome.error)))
        outcomes.append(outcome)
    for position, _ in rerun:
        outcomes[position] = _simulate_run_outcome(specs[position])
    return outcomes, tuple(rerun)


def _plan_tasks(planned, prepared):
    """``(keys, tasks)``: one :class:`~silk.simulation.executor.RunTask` per
    planned run that prepared to a :class:`_RunSpec`, in plan order."""
    from silk.simulation.executor import RunTask
    keys = [run.key for run in planned
            if isinstance(prepared.get(run.key), _RunSpec)]
    tasks = [RunTask(fn=_planned_simulate_task, run_dir=prepared[k].tb_dir,
                     args=(prepared[k],), simulator=prepared[k].netlist.simulator)
             for k in keys]
    return keys, tasks


def _sample_label(key):
    """Legacy corner/sample label, or the exact structured key for a grid.

    Structured identities are never recovered by parsing display strings.
    """
    corner, index = key
    return key if isinstance(corner, tuple) else f'{corner}/mc_{index:04d}'


def _write_wrapper(pdk, path):
    """Write the both-variation wrapper library for *pdk* at *path*."""
    from silk.circuit.monte_carlo import write_both_wrapper
    return write_both_wrapper(path, pdk.statistical)


def _stamp_sample_provenance(report, sample):
    """Record a Monte Carlo sample on its report's provenance: the seed, and
    the variation, library section and index in ``seedinfo`` (spec 3.3.3).
    The report's corner stays the declared corner."""
    provenance = getattr(report, 'provenance', None)
    if provenance is None:
        return
    # The wrapper's file is in pdk_libraries (path-scrubbed); name only its
    # section here, so no absolute path enters the record.
    section = sample.WRAPPER_SECTION if sample.wrapper_path else sample.section
    report.provenance = replace(
        provenance, seed=sample.seed,
        seedinfo=(f'monte-carlo variation={sample.variation} '
                  f'section={section} sample={sample.index}'))


def _read_sample_draws(netlist):
    """``{(designer name, param): value}``: the drawn parameters a
    ``record_draws`` sample printed into its log. *netlist* is
    the netlist that ran the sample (the simulate step's, not the bench's
    live one, which a batch has since re-prepared for another sample).

    The ``showmod`` blocks are matched to the devices by the order the
    netlister emitted them in (:func:`~silk.circuit.monte_carlo.draw_devices`
    over the same deck), never by the model name ngspice truncates.
    """
    from silk.circuit.monte_carlo import draw_devices, showmod_requests
    from silk.circuit.monte_carlo_draws import parse_showmod
    from silk.exceptions import NotSupportedError
    simulator = getattr(netlist, 'simulator', None)
    if simulator != 'ngspice':
        raise NotSupportedError(
            f"_read_sample_draws reads an ngspice log; this sample ran on "
            f"{simulator!r}.")
    devices, _ = draw_devices(netlist.elem_dict,
                              netlist.pdk_details.statistical)
    _, requested = showmod_requests(devices)
    # The deck's own run files: a worker's netlist comes back without its
    # loaded results (_raw is not pickled), but with these.
    from silk.simulation.generation import artifact_read
    raw = getattr(netlist, '_raw', None)
    with artifact_read(raw, 'logfile_path') as log_path:
        if log_path is None and getattr(raw, '_generation', None) is None:
            log_path = getattr(getattr(netlist, '_run_files', None), 'logfile_path', None)
        if log_path is None or not os.path.exists(log_path):
            raise SimulationError(
                f"record_draws: the sample's ngspice log ({log_path}) is "
                f"missing, so its drawn parameters cannot be read.")
        with open(log_path, errors='replace') as f:
            return parse_showmod(f.read(), requested)


def _attach_xyce_draws(report, netlist, sample):
    """With ``record_draws``, read a Xyce sample's drawn parameters from the
    ``.res`` file its run wrote and keep them on *report* as ``_mc_draws``
    (``{(designer device, param): normalised draw}``). A no-op on any other
    simulator (ngspice reads its draws from the log)."""
    if not sample.record_draws or netlist.simulator != 'xyce':
        return
    from silk.circuit.modules_results import xyce_draw_name_map
    from silk.circuit.monte_carlo_draws import parse_xyce_res
    from silk.exceptions import MonteCarloError
    name_map, params, _skipped = xyce_draw_name_map(
        netlist.elem_dict, netlist.pdk_details.statistical.record)
    files = netlist._run_files
    from silk.simulation.generation import artifact_read
    raw = getattr(netlist, '_raw', None)
    with artifact_read(raw, 'draws_path') as res:
        if res is None and getattr(raw, '_generation', None) is None:
            res = os.path.join(files.directory, os.path.basename(files.netlist_path) + '.res')
        if res is None or not os.path.isfile(res):
            raise MonteCarloError(
                f'record_draws: Xyce wrote no {res}; the sample deck has no '
                f'drawn-parameter table to read.')
        report._mc_draws = {key: value
                            for key, value in parse_xyce_res(res, name_map).items()
                            if key[1] in params[key[0]]}


def _attach_sample_draws(report, netlist, sample):
    """Read a ``record_draws`` sample's drawn parameters by the one path
    its simulator has, onto *report*: ``_mc_draws`` and ``_mc_draws_raw``.

    Behavioral samples expose their resolved parameter values directly.
    ngspice prints actual parameter values (``showmod`` in the log,
    ``_mc_draws_raw = True``); Xyce writes normalised draws to its ``.res``
    (``False``). Any other simulator raises
    :class:`~silk.exceptions.NotSupportedError`.
    """
    from silk.exceptions import NotSupportedError
    simulator = getattr(netlist, 'simulator', None)
    if simulator == 'ngspice':
        report._mc_draws = _read_sample_draws(netlist)
        report._mc_draws_raw = True
    elif simulator == 'behavioral':
        report._mc_draws = dict(sorted(netlist._behavioral_draws.items()))
        report._mc_draws_raw = True
    elif simulator == 'xyce':
        _attach_xyce_draws(report, netlist, sample)
        report._mc_draws_raw = False
    else:
        raise NotSupportedError(
            f"record_draws is read on ngspice and Xyce; this sample ran on "
            f"{simulator!r}.")


def _reconcile_run_outcome(test, outcome):
    """Apply proven replacement facts without evaluating or raising an outcome.

    Executed batches reconcile every consumer before fail-fast evaluation.
    False or unknown replacement state leaves retained evidence alone.
    """
    if outcome.replacement_started is True:
        from silk.circuit.test_placement import begin_execution
        begin_execution(test)
        if outcome.error is not None:
            from silk.circuit.modules_sim import _invalidate_current_results
            _invalidate_current_results(test.netlist)


def _planned_outcome(outcome, prepared=None):
    """A :class:`~silk.simulation.executor.RunOutcome` of
    :func:`_planned_simulate_task` as a shared run outcome. The coordinator's
    replacement fact survives even when the worker lost its begin reply."""
    from silk.simulation.generation import replacement_for
    started = getattr(outcome, '_replacement_started', None)
    if prepared is not None:
        parent_started = replacement_for(prepared.attempt_id)
        if parent_started is True or started is None:
            started = parent_started
    if outcome.error is None:
        changes = {}
        if started is True:
            changes['replacement_started'] = True
        if prepared is not None:
            changes['attempt_context'] = _attempt_context(prepared)
        return replace(outcome.value, **changes) if changes else outcome.value
    return _RunOutcome(error=RuntimeError(outcome.error),
                       error_text=outcome.error, replacement_started=started,
                       failure_stage='transport',
                       attempt_context=_attempt_context(prepared) if prepared is not None else None)


class _CornerRawfile:
    """One corner's rawfile, read as a stand-in for a bench's netlist.

    ``Signal.resolve()`` asks whatever it is given for
    ``load_signal(name, selector, analysis=...)``; that is the whole contract,
    and this class implements exactly it for a file on disk. It is what makes
    ``vcm.corner('ff')`` possible after ``run_corners()`` has moved on: the
    reports' ``sim_results`` all reference the ONE live netlist Module, whose
    results are the last corner's, but each corner's file is still there and
    its path is in the bench's sweep table.

    The live Module is never touched — no ``load_file``, no ``read_raw_file``,
    no write to ``_raw``/``simResultsAll``/``results_file``. The file is read
    through ``backend.load_raw`` into a private
    :class:`~silk.simulation.backend.RawResult` and signals come off it
    through ``backend.load_signal``, the same two calls the live path makes.

    Nothing is cached beyond one instance. A derived expression resolves each
    of its leaves against the same reader, so ``(vp + vm).corner('ff')`` opens
    ff's file once; the reader is then dropped, and the next ``.corner('ff')``
    reads the file again. Corner waveforms are deliberately not kept in
    memory — a sweep over many corners on a large rawfile would otherwise
    accumulate every one of them.
    """

    __slots__ = ('corner', 'path', '_backend', '_simulator', '_raw', '_source_context', '_generation')

    def __init__(self, corner, path, backend, simulator=None, source_context=None, generation=None):
        self.corner = corner
        self.path = path
        self._backend = backend
        self._simulator = simulator
        self._source_context = source_context
        self._generation = generation
        self._raw = None

    def __repr__(self):
        return f'<corner {self.corner!r} results: {self.path}>'

    def _raw_result(self):
        """Read the file on first use; reuse it for the rest of this call."""
        if self._raw is None:
            if self._generation is None:
                self._raw = self._backend.load_raw(self.path, simulator=self._simulator)
            else:
                from silk.simulation.generation import attach
                with self._generation.read():
                    raw = self._backend.load_raw(self._generation.resolve('rawfile_path'),
                                                 simulator=self._simulator)
                    self._raw = attach(raw, self._generation)
        return self._raw

    def load_signal(self, signalName, signalType='V', *, analysis=None):
        """The one method ``Signal.resolve()`` calls — same shape as
        :meth:`~silk.circuit.modules.Module.load_signal`, reading this
        corner's file instead of the live results.

        Device-parameter probes (``'<name>:<param>'``) are not supported
        here: this reader uses source identity only to translate current
        cards; device metadata still requires the live DUT result context.
        Raises naming the
        gap rather than mis-reading the colon spelling as a raw vector
        name (never found, so it would otherwise surface as a confusing
        "vector not found"). The suffix-membership check is the same
        closed-set test ``probe()`` itself applies (never a DUT-tree
        lookup) — a terminal-
        current probe's stored name never reaches here with a ``':'`` at
        all (already resolved to its ammeter vector at registration time),
        so this can only ever match a genuine device-parameter probe.
        """
        from silk.circuit.node_paths import NodeVoltageRequest, resolve_node_request
        if (':' in signalName and not isinstance(signalName, NodeVoltageRequest)
                and not getattr(signalName, '_literal_voltage', False)):
            from silk.simulation.backend import DEVICE_OP_PARAMS
            _prefix, _, _suffix = signalName.rpartition(':')
            device_probe = _suffix in DEVICE_OP_PARAMS
        else:
            device_probe = False
        if device_probe:
            raise SimulationError(
                f"probe({signalName!r}).corner({self.corner!r}): "
                f"device-parameter probes are not supported for a retained "
                f"corner — only the live testbench's current results can "
                f"resolve a device by its designer name. Read this probe "
                f"without .corner(), or read {signalName!r}'s op_table() "
                f"cell instead."
            )
        raw = self._raw_result()
        signalName = resolve_node_request(self._source_context, raw, signalName, analysis)
        if self._source_context is not None:
            from silk.circuit.modules_results import _source_current_probe_name
            signalName = _source_current_probe_name(
                self._source_context, raw, signalName, signalType, analysis)
        return self._backend.load_signal(raw, signalName,
                                         type=signalType, analysis=analysis)


def _metric_output_declarations(entries):
    """Resolve declared output names/types without evaluating any metrics."""
    output_declarations = []
    metric_types = {}
    reserved = {entry['name'] for entry in entries}
    for entry in entries:
        if 'expression' in entry:
            # A derived metric (handle arithmetic) has no extractor and
            # declares no outputs.
            output_declarations.append({})
            metric_types[entry['name']] = None
            continue
        fn = entry.get('fn', entry['extractor'])
        if 'type' in entry:
            from silk.signals.schema import get_metric
            fn = get_metric(entry['type'])
        metadata = inspect.getattr_static(fn, '_silk_metric', {})
        outputs = metadata.get('outputs', {})
        metric_type = entry.get('type')
        if metric_type is None and metadata:
            from silk.signals.schema import metric_schema
            metric_type = metric_schema(fn)['name']
        metric_types[entry['name']] = metric_type
        for field in outputs:
            dotted = f"{entry['name']}.{field}"
            if dotted in reserved:
                raise ValueError(f"metric {dotted!r}: output name collision; "
                                 "rename the metric or its explicit dotted entry")
            reserved.add(dotted)
            metric_types[dotted] = metric_type
        output_declarations.append(outputs)
    return output_declarations, metric_types


def _probe_preflight_reason(test, simulator):
    """Return a deliberate auxiliary-simulation exclusion, not config errors."""
    # Undeclared ngspice parameters require an auxiliary simulation even
    # during deck emission. Refuse without consulting its process cache,
    # so preparation has the same no-simulation contract in every session.
    requests = _device_probe_requests(test._probe_registry)
    if simulator == 'ngspice' and requests:
        from silk.circuit.modules_results import (
            _device_op_params, dut_module_from_elem_dict,
            resolve_device_probe_address,
        )
        dut = dut_module_from_elem_dict(test.netlist.elem_dict)
        for name, param in requests:
            address = resolve_device_probe_address(
                dut, name, param, ammeter_wrapped=False)
            if param not in (_device_op_params(address.device) or ()):
                return (
                    f'prepare() cannot prepare probe {name}:{param}: '
                    'it requires a separate ngspice preflight simulation '
                    '(multi-deck). Use declared device parameters or run().')

def _require_no_probe_preflight(test, simulator):
    """Refuse auxiliary device-probe simulations for disk-free consumers."""
    reason = _probe_preflight_reason(test, simulator)
    if reason is not None:
        raise NotSupportedError(reason)


def _configure_conditions(test, netlist, *, temperature=None, voltage=None):
    """Resolve the same current inherited settings for launch and comparison."""
    from silk.simulation._replay import apply_effective_settings
    from silk.simulation.conditions import (
        prepare_temperature, prepare_supply, assert_temperature, assert_supply,
        _classify_supply_sweep,
    )
    apply_effective_settings(test, netlist)
    request = prepare_temperature(netlist)
    # Preserve preparation's refusal order: a bad TEMP assertion must not
    # apply a configured supply to the authored source first.
    assert_temperature(request, temperature)
    supply = prepare_supply(netlist)
    if netlist.simulator == 'behavioral':
        supply = _classify_supply_sweep(netlist, supply)
    assert_supply(supply, voltage)
    return request, supply


def _configure_capture(test, netlist):
    """Refresh one bench's authored capture requests without running it."""
    # Propagate current capture to the netlist that emits/reads it.
    netlist.save_device_currents = test.module.save_device_currents
    netlist.terminal_current_method = test.module.terminal_current_method
    # Same propagation for op-point device-parameter capture (TASK-330.3).
    netlist.save_device_params = test.effective_save_device_params()
    netlist.omit_device_internal_nodes = test.module.omit_device_internal_nodes
    # Node capture requests are refreshed for every invocation too.
    from silk.circuit.node_paths import NodeVoltageRequest
    netlist._pending_node_probes = tuple(
        leaf.name for probe in test._probe_registry
        for leaf in _unbound_leaves(probe)
        if isinstance(leaf.name, NodeVoltageRequest)
        or getattr(leaf.name, '_literal_voltage', False))
    netlist._pending_metric_nodes = tuple(_metric_node_requests(
        test._metrics, _declared_analyses(netlist)))
    # Snapshot of every device-parameter probe registered on THIS bench
    # right now (TASK-330.8) — what emission requests (building the
    # save/print lines) and what read time compares against to tell "the
    # backend/version could not supply this" apart from "this probe did
    # not exist when the loaded results were produced" (reviewer's
    # MISSING 2 — lifecycle).
    netlist._pending_device_probes = _device_probe_requests(
        test._probe_registry)
    # Save physical operands while retaining each probe's analysis selection.
    current_requests = _current_probe_requests(
        test._probe_registry, test._metrics, _declared_analyses(netlist))
    netlist._pending_current_probes = tuple(
        name for name, _ in current_requests)
    netlist._pending_current_probe_analyses = tuple(
        analysis for _, analysis in current_requests)


class ModuleTestbench:
    """A registered test with its own executing Module and metrics.

    ``netlist`` emits, runs and loads the simulation. At each ``run()``, it
    inherits the live owning ``module.simulator`` unless the caller explicitly
    assigned ``netlist.simulator``. Ambient temperature, primary supply and
    simulator options all inherit from the live owner the same
    way -- resolved fresh from the owner at every run, never copied, unless
    this test set its own -- via ``set_temperature``, ``set_supply_voltage``
    and ``set_simulator_options``. Each copied child test follows its own
    owner.

    Direct construction and :meth:`Module.create_test` share the same
    registration path: a successfully constructed test is immediately
    visible through ``module.list_tests()`` and ``module.run_all_tests()``.

    Parameters
    ----------
    module : Module
        The module to test.
    name : str, optional
        Name for this test; unique within ``module``. Default
        ``'default_test'``.
    cache : bool, optional
        Disabled for 0.2.0. Passing True emits one
        ``UserWarning`` at construction and the testbench then behaves
        exactly like ``cache=False``: every ``run()`` simulates. The
        parameter and ``clear_cache()`` are kept so existing code does not
        break: the fingerprint (device names, specs and simulator only)
        cannot yet detect a changed device value, metric or analysis, so a
        cache hit could return a stale SpecReport. A correct cache is
        future work.
    parameters : dict, optional
        Declared test-parameter map (defaults materialized by the caller).
        Part of this test's characterization identity; never inspected for
        netlisting. Omitted or ``None`` means ``{}``: no declared
        parameters, not "unknown".
    save_device_params : bool, optional
        per-test operating-point device-parameter capture. ``None``
        (the default) inherits the device under test's ``save_device_params``;
        ``True``/``False`` overrides it for this test only. Also settable as
        ``tb.save_device_params`` later.
    strict_nets : bool, optional
        Keyword-only. Whether the testbench netlist rejects undeclared nets
        the way ``module`` does. ``None`` (default) inherits
        ``module.strict_nets``, so a strict DUT gets a strict testbench and
        a misspelled net in ``add_module_under_test``'s ``port_mapping``
        raises instead of silently wiring a fresh net. Pass True or False
        to override the inherited value in either direction.

    Raises
    ------
    silk.exceptions.DuplicateNameError
        If ``module`` already owns a test named ``name``.

    Examples
    --------
    >>> from tempfile import TemporaryDirectory
    >>> from silk import Module, load_pdk
    >>> folder = TemporaryDirectory()
    >>> pdk = load_pdk('sky130')
    >>> dut = Module('sky130', folder.name, 'demo', pdk)
    >>> from silk import ModuleTestbench
    >>> tb = ModuleTestbench(dut, name='gain')
    >>> tb.test_name
    'gain'
    """

    _sweep_table = ResultAttribute()
    _sweep_generations = ResultAttribute()
    _mc_samples = ResultAttribute()
    _metric_values = ResultAttribute()
    _record_measurements = ResultAttribute()
    _cache = ResultAttribute()

    def __init__(self, module, name='default_test', cache=False, parameters=None,
                 *, strict_nets=None, save_device_params=None):
        """Create and register a test; see the class docstring."""
        if name in module._tests:
            raise DuplicateNameError(
                f"Test {name!r} already exists on module "
                f"{module.module_name!r}"
            )

        self.module = module
        self.save_device_params = save_device_params
        self.test_name = name
        self._metrics = []
        self._specs = []
        self._module_added = False
        self.netlist = None
        # Every probe this bench has minted, by STRONG reference, in creation
        # order (TASK-215.9; strong since TASK-230). ``probe()`` appends;
        # ``tb.plot()`` with no arguments walks this to draw every probe of
        # the bench. A probe minted only to be plotted -- ``tb.plot(tb.probe
        # ('x'))`` with no variable holding it -- used to be collected before
        # the call ever ran, because the registry held only a ``weakref.ref``
        # (TASK-230 finding 2); it is held strongly now. This is safe and
        # cheap: a probe references its bench only WEAKLY (see ``Signal.
        # _unbound``), so this is a one-way strong edge, no cycle, and a
        # probe carries no data (name + selector + a weak bench ref), so
        # TASK-205's linear-growth guarantee is unaffected -- see
        # ``_live_probes`` and the ``__deepcopy__``/``__setstate__`` notes
        # below for how this registry still behaves across a copy (that
        # limit -- a probe minted only for plotting does not survive a copy
        # or a pickle round trip -- is unchanged; only premature collection
        # on the SAME bench, before ``tb.plot()`` runs, is fixed).
        self._probe_registry = []
        # The bench's LATEST sweep: {corner name: rawfile path}, in the order
        # the corners ran (TASK-218). Rewritten to one entry by every
        # evaluate() (so run() and evaluate(rawfile=) leave exactly one), and
        # to the whole sweep by run_corners() once its corners have all
        # completed. Read through the `sweep_corners` property.
        self._sweep_table = {}
        self._sweep_generations = {}
        # The report cache is disabled for 0.2.0 (TASK-284): _cache_enabled
        # is forced False regardless of what the caller passed, so every
        # run() simulates. cache=True only earns a warning explaining why
        # (the fingerprint below cannot see a changed device value, metric
        # or analysis) and pointing at TASK-130, the redesign that will
        # re-enable this. The cache-hit code later in this file is left in
        # place, dead, for that redesign to build on.
        self._cache_enabled = False
        if cache:
            import warnings
            warnings.warn(
                "ModuleTestbench(cache=True) is disabled in this release: "
                "its circuit fingerprint hashes device NAMES, specs and the "
                "simulator only, so it cannot detect a changed device "
                "value, metric or analysis and could return a stale "
                "SpecReport. Caching is forced off -- every run() "
                "simulates, exactly as cache=False. A correct cache is "
                "future work (TASK-130).",
                UserWarning, stacklevel=2,
            )
        self._cache = {}   # {(fingerprint, corner_str): SpecReport}; unused while disabled
        # This test's own process corner (TASK-371), set by set_corner():
        # the corner run()/evaluate(rawfile=) use when no corner is passed.
        # None = the PDK's default_corner.
        self._corner = None
        # This test's declared corner list (TASK-426), set by set_corners():
        # run() and both run_all_tests walks run every corner in it. None =
        # no declaration, today's single run.
        self._corners = None
        self._variable_lists = []
        self._groups = ()
        # Whether the testbench netlist honors strict_nets — inherited from
        # the DUT unless the caller overrides it (see the strict_nets Arg
        # above). Resolved once here so _create_netlist has a plain bool to
        # pass through, and so the effective value is inspectable as
        # tb.strict_nets before any netlist exists.
        self.strict_nets = (
            bool(module.strict_nets) if strict_nets is None else strict_nets
        )
        # Mirrors the DUT module's PDK so run()/run_corners() can derive
        # their corner defaults from the manifest (silk.pdk.registry is the
        # authoritative source; see silk/simulation/corner.py).
        self.pdk_details = getattr(module, 'pdk_details', None)
        # Declared parameter map — part of characterization identity, never
        # read by netlisting. A fresh, deep-copied snapshot so later mutation
        # of a caller-supplied dict isn't observable here.
        self.parameters = freeze_parameters(
            parameters if parameters is not None else {}
        )
        module._tests[name] = self

    @property
    def save_device_params(self):
        """``None`` (inherit the DUT's ``save_device_params``) or a ``bool``
        that overrides it for this test only. Emission and the
        report's operating-point inventory both read
        :meth:`effective_save_device_params`."""
        return getattr(self, '_save_device_params', None)

    @save_device_params.setter
    def save_device_params(self, value):
        if value is not None and not isinstance(value, bool):
            raise TypeError(
                f"save_device_params must be None (inherit the device under "
                f"test's value), True or False, not {value!r}.")
        self._save_device_params = value

    def effective_save_device_params(self):
        """The value this test runs with: its own setting, else the DUT's."""
        own = self.save_device_params
        return bool(self.module.save_device_params) if own is None else own

    @property
    def owner_path(self):
        """Canonical root/instance-path of the live module that owns this
        registered test (the ``owner_path`` half of characterization
        identity — see ``silk/db/context.py``)."""
        entry = current_placement(self)
        return entry.owner_path if entry is not None else getattr(self.module, '_spec_owner_path', '$')

    @property
    def report_key(self):
        """The bound entry's runner key, or this unbound test's local key."""
        entry = current_placement(self)
        return entry.report_key if entry is not None else f'{self.module.module_name}.{self.test_name}'

    @property
    def _placement_module_name(self):
        entry = current_placement(self)
        return entry.module_name if entry is not None else self.module.module_name

    def _build_context(self, corner_str, temperature, voltage):
        """Build this run's :class:`~silk.db.context.CharacterizationContext`,
        or ``None`` when the caller supplied neither ``temperature`` nor
        ``voltage`` (the default, explicitly-unscoped path). Supplying only
        one of the two is a partial descriptor and raises
        ``CharacterizationContextError`` — see the contract in docs/probes.md.
        """
        if temperature is None and voltage is None:
            return None
        return build_characterization_context(
            owner_path=self.owner_path,
            test_name=self.test_name,
            parameters=self.parameters,
            corner=corner_str,
            temperature=temperature,
            voltage=voltage,
        )

    def _result_netlist(self):
        selection = current_placement(self)
        if selection is not None:
            if selection.reader is not None:
                return selection.reader
            if not selection.executing:
                return selection.evidence.results
        else:
            from silk.circuit.test_placement import latest_record
            tracked, record = latest_record(self)
            if tracked:
                return record.results if record is not None else None
        return self.netlist

    @property
    def results_file(self):
        """Source path from the most recent ``run()`` or successful
        ``evaluate(rawfile=...)``. SPICE runs identify a rawfile or Xyce noise
        CSV; an in-memory backend may identify its input deck instead.
        ``None`` before results are available, or when a simulation or explicit
        file load fails after result replacement starts. Requests refused before
        that boundary, including invalid declarations before explicit loading,
        preserve the prior path. An unknown replacement state alone does not
        clear it. Save this path immediately after a successful run to read
        those files later. New mode preserves each
        run in its own persistent child. In default overwrite mode, another
        label preserves that directory; the same label replaces its outputs.
        Selected hierarchy reports retain their own result reader; direct
        unselected reports keep the authored Module reader.

        Reads the selected evidence (or authored netlist), populated by simulation or
        a successful explicit rawfile evaluation. Returns ``None``,
        never raises, when the netlist hasn't been created yet either
        (before the first ``add_module_under_test``/``add_source``/etc.
        call) — callers that need a hard failure should call ``run()`` or
        ``evaluate(rawfile=...)`` first and check for ``None``. ``report.add_waveform_view()`` treats
        a ``None`` here as "not simulated yet" and raises a clear
        :class:`silk.signals.view.ViewError <silk.signals.view.template.ViewError>` naming the fix, rather than
        the raw ``os.fspath`` ``TypeError`` a bare ``None`` would otherwise
        produce; passing ``tb.results_file`` straight to ``emit_view()``
        still requires an available results file.
        """
        netlist = self._result_netlist()
        return None if netlist is None else netlist.results_file

    @property
    def sweep_corners(self):
        """This bench's LATEST sweep — ``{corner name: rawfile path}``.

        A copy, in the order the corners ran. Three things write it:

        * ``run()`` and ``evaluate()`` (including ``evaluate(rawfile=...)``)
          leave exactly ONE entry — the corner they just evaluated, mapped to
          the file they read;
        * ``run_corners()`` replaces it with every corner it ran, once they
          have all completed;
        * a bench that has never produced results has an empty table.

        **Every corner of a ``run_corners()`` sweep is always retained**: each corner writes to its own
        ``<testbench dir>/<corner>/`` folder unconditionally, so the output
        mode no longer determines how many corners survive a sweep the way
        it did before this task — overwrite mode (the default) and
        ``module.set_output_options(mode='new')`` behave identically here.
        A corner is missing from this table only when ``run_corners()``
        itself was given fewer corners, or when that specific corner's run
        failed (see :class:`silk.circuit.spec_report.CornerRunError`).

        The paths are what ``Signal.corner()`` reads and what
        ``SpecReport.rawfile_path`` records per corner.

        The report cache (``cache=True`` on the constructor) is disabled in
        this release: a cache hit can no longer occur, so this
        table is always the corner(s) actually just simulated. See docs/known_issues.md for the report-cache limitation.
        """
        return dict(self._sweep_table)

    def _corner_netlist(self, corner, readers=None):
        """A read-only stand-in for this bench's netlist serving *corner*'s file.

        The resolution seam behind :meth:`~silk.signals.base.Signal.corner`.
        *readers* is an optional dict shared across the leaves of ONE
        expression so a derived Signal opens each corner's file once; it is
        not a cache that outlives the call (see :class:`_CornerRawfile`).

        Raises
        ------
        UnboundSignalError
            This bench has retained no sweep at all, or does not have
            *corner* — the message lists the corners it does have.
        """
        table = getattr(self, '_sweep_table', {}) or {}
        name = corner if isinstance(corner, tuple) else str(corner)
        if isinstance(name, str) and any(isinstance(key, tuple) for key in table):
            raise UnboundSignalError(
                f'testbench {self.test_name!r}: select a full grid point with .point(...) '
                f'or an MC sample with .sample(...); available keys: {list(table)!r}')
        if not table:
            raise UnboundSignalError(
                f"testbench {self.test_name!r} has no results to select a "
                f"corner from; call tb.run() or tb.run_corners([...]) first")
        samples = getattr(self, '_mc_samples', None)
        if name not in table and samples:
            example = next(iter(samples))
            raise UnboundSignalError(
                f"testbench {self.test_name!r}: the latest run was Monte "
                f"Carlo, so corner {name!r} names no single run (its samples "
                f"share it); read one sample with .sample(), e.g. "
                f".sample({example!r})")
        if name not in table:
            raise UnboundSignalError(
                f"testbench {self.test_name!r} did not retain corner "
                f"{name!r}; it retained {', '.join(repr(c) for c in table)}. "
                f"(Only the corners of the LATEST sweep are retained, and a "
                f"corner run_corners() was given but that failed is not — "
                f"see tb.sweep_corners.)")
        selection = current_placement(self)
        cache_key = (id(self), id(selection.evidence) if selection else None, name)
        if readers is not None and cache_key in readers:
            return readers[cache_key]
        if selection is not None and not selection.executing:
            retained = selection.evidence.corners.get(name)
            if retained is not None:
                if readers is not None:
                    readers[cache_key] = retained
                return retained
        netlist = self.netlist
        reader = _CornerRawfile(
            name, table[name],
            backend=netlist.backend,
            simulator=getattr(netlist, 'simulator', None),
            source_context=netlist,
            generation=(getattr(self, '_sweep_generations', {}) or {}).get(name),
        )
        if readers is not None:
            readers[cache_key] = reader
        return reader

    def _point_corner(self, key):
        """Select an exact retained deterministic point for Signal.point()."""
        table = getattr(self, '_sweep_table', {}) or {}
        if getattr(self, '_mc_samples', None):
            raise UnboundSignalError('This grid has Monte Carlo samples; use .sample((point, index)).')
        if not isinstance(key, tuple) or len(key) not in (3, 4) or key not in table:
            raise UnboundSignalError(f'No retained point {key!r}; available points: {list(table)!r}')
        return key

    def _sample_corner(self, key):
        """The sweep-table name of Monte Carlo sample *key* of the latest
        run: the seam behind :meth:`~silk.signals.base.Signal.sample`.

        Raises
        ------
        UnboundSignalError
            The latest run was not Monte Carlo, or *key* is not one of its
            retained samples (a failed sample retains no file).
        """
        samples = getattr(self, '_mc_samples', None)
        if not samples:
            raise UnboundSignalError(
                f"testbench {self.test_name!r} has no Monte Carlo samples to "
                f"select from; .sample() reads one sample of the latest "
                f"Monte Carlo run (tb.set_monte_carlo(...) then tb.run(), or "
                f"tb.run_monte_carlo(...)). Use .corner() for a corner sweep.")
        if not (isinstance(key, tuple) and len(key) == 2):
            raise TypeError(
                f".sample() takes a (corner, index) key such as "
                f"{next(iter(samples))!r}, got {key!r}")
        if key not in samples:
            raise UnboundSignalError(
                f"testbench {self.test_name!r}: the latest Monte Carlo run "
                f"retained no sample {key!r}; it has {len(samples)} samples "
                f"from {next(iter(samples))!r} to {list(samples)[-1]!r} "
                f"(a failed sample retains no file).")
        return _sample_label(key)

    def _corners_to_draw(self, corners):
        """Which corners a plot of this bench should overlay, or ``None``.

        ``None`` means "no corner overlay": draw the live results, which is
        the default for a bench that has results, and what a bench that ran once
        still does. A list means one labelled trace per named corner.

        *corners* ``None`` asks for the default — every corner of the latest
        sweep when more than one was retained, and the live results
        otherwise. An explicit list is validated against the table here, so
        an unknown name is refused before anything is drawn.
        """
        table = getattr(self, '_sweep_table', {}) or {}
        if corners is None:
            return list(table) if len(table) > 1 else None
        if isinstance(corners, str):
            corners = [corners]
        names = [str(c) for c in corners]
        if not names:
            raise ValueError(
                f"corners=[] selects nothing to plot; pass corner names, or "
                f"omit corners= to draw {'every retained corner' if len(table) > 1 else 'the current results'}.")
        unknown = [c for c in names if c not in table]
        if unknown:
            raise UnboundSignalError(
                f"testbench {self.test_name!r} did not retain "
                f"{', '.join(repr(c) for c in unknown)}; it retained "
                f"{', '.join(repr(c) for c in table) or '(no corners)'}. "
                f"(Only the corners of the LATEST sweep are retained, and a "
                f"corner run_corners() was given but that failed is not — "
                f"see tb.sweep_corners.)")
        return names

    def _output_paths(self, run_label=None, *, corner_subdir=None):
        """Select safe paths from the live owner, never a prior run's directory.

        The folder is ``test_<module>~<test>``. Underscores stay
        literal: percent and tilde are always escaped in names, so the first
        ``~`` after ``test_<module>`` is the module/test separator and
        (a_b, c) / (a, b_c) cannot alias. The literal ~owner~/~run~
        delimiters can likewise never be confused with user text. The canonical owner
        path already distinguishes nested instances from slashes in names.

        ``corner_subdir``, keyword-only and private to :meth:`run_corners`: when given, the returned directory is nested one level
        further, ``<testbench dir>/<encoded corner_subdir>/``, so every
        corner of a sweep always gets its own folder. Percent-encoded the
        same way as every other name here (so a corner string containing
        ``..`` or ``/`` cannot escape the testbench directory) and checked
        against the output root exactly like the rest of this path. A tuple
        nests one folder per component, each encoded on its own (a Monte
        Carlo sample's ``(corner, 'mc_NNNN')``).

        Every refusal here raises
        :class:`~silk.exceptions.OutputPathError`, a ``ValueError`` that also
        carries the :class:`~silk.exceptions.ConfigurationError` marker: the
        request cannot name an output location, which is a fact about the
        call and not about the circuit, so ``run_all_tests`` stops rather
        than recording the same string once per test.
        """
        from silk.exceptions import OutputPathError

        if run_label is not None and (
            not isinstance(run_label, str)
            or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}', run_label) is None
        ):
            raise OutputPathError(
                "run_label must be None or 1–64 ASCII characters, starting "
                "with a letter or digit and containing only letters, digits, "
                "underscore, dot or hyphen (no paths)."
            )

        def encode(name):
            if not isinstance(name, str):
                raise OutputPathError(f"Test/module output names must be strings; got {name!r}")
            return quote(name, safe='').replace('~', '%7E')

        # %N is outside the percent-encoded byte alphabet; an unnamed Module
        # cannot alias one explicitly named "None" (or "%N").
        module_name = self._placement_module_name
        module_token = '%N' if module_name is None else encode(module_name)
        test_token = encode(self.test_name)
        basename = f"test_{module_token}~{test_token}"
        directory = basename
        selection = current_placement(self)
        if self.owner_path != '$' and (selection is None or selection.separate_output):
            directory += f"~owner~{encode(self.owner_path)}"
        if run_label is not None:
            directory += f"~run~{run_label}"

        def bounded(component):
            # All encoded characters are ASCII. Keep room for the deck's
            # extension under the usual 255-byte filename limit. Hash the
            # complete unambiguous encoding, never a truncated owner/name.
            # Ordinary names always start test_, so test~ is disjoint.
            if len(component) > 240:
                return 'test~' + hashlib.sha256(component.encode('ascii')).hexdigest()
            return component

        directory = bounded(directory)
        basename = bounded(basename)
        root = self.module.dir_folder
        tb_dir = os.path.join(root, directory)
        if corner_subdir is not None:
            # A tuple nests one folder per component (a Monte Carlo sample:
            # ('ss', 'mc_0017') -> <testbench dir>/ss/mc_0017/); each is
            # encoded on its own, so no component can hold a separator.
            parts = (corner_subdir if isinstance(corner_subdir, tuple)
                     else (corner_subdir,))
            # Encoding keeps a separator inside the folder name, but '.' and
            # '..' encode to themselves (TASK-593).
            from silk.utils.deck_text import validate_corner_name
            for part in parts:
                validate_corner_name(part)
            for part in parts:
                tb_dir = os.path.join(tb_dir, bounded(encode(str(part))))
        # Check before mkdir or deck generation, including an existing
        # output-directory symlink that would redirect writes outside root.
        resolved_root = os.path.realpath(root)
        if os.path.commonpath([resolved_root, os.path.realpath(tb_dir)]) != resolved_root:
            raise OutputPathError(f"Testbench directory {tb_dir!r} escapes output root {root!r}")
        return tb_dir, basename + '.cir'

    def _create_netlist(self):
        """Build a testbench in its own directory under the DUT output root.

        ``run()`` selects the directory again using the live owner path:
        composition may have copied an already-built testbench since here.
        Deck, rawfile and simulator log use that same selected directory.
        The testbench Module inherits ``self.strict_nets``.
        """
        from silk.circuit.modules import Module
        tb_dir, deck_name = self._output_paths()
        self.netlist = Module(
            self.module.pdk_name,
            tb_dir,
            deck_name,
            self.module.pdk_details,
            strict_nets=self.strict_nets,
        )
        self.netlist.__dict__['_probe_execution_deck'] = True

    def __deepcopy__(self, memo):
        """Copy an explicitly requested bench or circuit, preserving its netlist.

        Placement itself holds references, so test registration
        creates no recursive copies and needs no DUT-embedding copy scope.

        **Probes are rebound to the copy** (the contract in docs/probes.md). A probe
        holds a *weak* reference to its bench, and copying a weak reference is
        atomic — the copy points at the ORIGINAL bench — so a composed block
        whose metric uses ``tb.probe('out')`` would otherwise resolve against
        the template bench's results and measure its parent's circuit. After
        the attribute loop, every unbound leaf Signal reachable from the
        copied metrics' slot values whose referent is still ``self`` is
        pointed at ``new``. Leaves belonging to some other bench, or to no
        bench, are untouched. The orphan-replica path needs nothing further:
        its probes follow it to a netlist-less bench that is never run, where
        ``bound`` answers ``False`` without raising.

        Only slot values are rebound. A probe captured inside a user lambda
        is reachable from no structure this bench owns and cannot be rebound;
        that is the stated limit of keeping lambdas legal.

        ``_probe_registry`` (the list ``tb.plot()`` with no
        arguments walks; strong references) is REBUILT rather
        than left as whatever the generic loop above deep-copied: a probe
        reachable ONLY through the registry (never passed to ``add_metric``)
        would deep-copy there with its ``_bench`` weak reference still
        pointing at the ORIGINAL bench, unrebound — the walk below only
        rebinds leaves reachable from a metric slot. Rebuilding from those
        same (already rebound) metric slots is simpler than teaching the
        rebind walk a second reach, and keeps the existing, documented
        limit: a probe minted only for plotting on the template, never
        passed to ``add_metric``, is not carried into the copy. It also
        keeps this deep copy linear: a registry-only probe is not
        yet in ``memo`` when the generic loop reaches it and does get deep-
        copied there, but that copy is discarded by the rebuild below rather
        than kept, and a probe ALSO reachable from a metric slot is already
        in ``memo`` by then, so the registry's copy of it is the same object
        the metric-slot copy produced, not a second one.

        The function form's extractor is rebuilt over the copied metric dict
        for the same reason, and it has to be: a function is *atomic* to
        :mod:`copy`, so the copied dict came back holding the ORIGINAL
        closure, which reads the original dict's slots. Rebinding the copied
        slot values would then have changed nothing the run ever looked at —
        measured on a composed block, whose metric resolved against the
        template bench's results with every copied probe correctly pointing
        at the copy.
        """
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        for key, value in self.__dict__.items():
            new.__dict__[key] = copy.deepcopy(value, memo)
        new.__dict__.pop('_metric_values', None)
        metrics = new.__dict__.get('_metrics')
        for metric in metrics or ():
            if _is_function_form_metric(metric):
                metric['extractor'] = _function_form_extractor(metric)
        _rebind_probes(
            new, metrics, lambda ref: ref is not None and ref() is self,
        )
        new._probe_registry = _registry_from_metric_leaves(metrics)
        # The copy has run nothing, so it has retained no sweep (TASK-218).
        # Copying the table would point the replica's ``Signal.corner()`` at
        # the TEMPLATE bench's files — the same mistake the probe rebinding
        # above exists to prevent, one layer down. A replica that is run
        # fills its own table at that run. (``__setstate__`` deliberately
        # differs: an unpickled bench is the same bench, and the files its
        # table names are still on disk.)
        new._sweep_table = {}
        new._mc_samples = None
        return new

    # ------------------------------------------------------------------
    # Pickling (TASK-215.8)
    #
    # ``FaultCampaign`` pickles the whole Module, testbenches included
    # (``silk/faults/campaign.py`` -> ``silk/faults/runner.py``), so a probe
    # must not be the thing that stops it. Two problems, one hook each:
    #
    # * A ``weakref`` cannot be pickled. ``Signal.__getstate__`` (TASK-215.2)
    #   already drops it and leaves a detached marker, so the probe survives
    #   the round trip explaining itself. ``__setstate__`` below re-attaches
    #   those probes to the restored bench with the same walk
    #   ``__deepcopy__`` uses, which is what makes a restored bench's metrics
    #   work at all.
    # * A closure cannot be pickled. The function form's extractor is one, so
    #   it is dropped on the way out and rebuilt on the way in from the
    #   ``fn``/``slots``/``params`` the metric dict already stores raw — the
    #   registered metric function pickles by reference like any other
    #   module-level function.
    # * ``_probe_registry`` (TASK-215.9) holds strong references (TASK-230),
    #   which pickle fine on their own, but a probe reachable only from the
    #   registry would restore with its ``_bench`` weak reference still
    #   detached (only metric-slot leaves are re-attached below). Dropped
    #   here to ``[]`` and rebuilt in ``__setstate__`` from the metric slots
    #   the same rebinding walk just fixed — see
    #   :func:`_registry_from_metric_leaves` — the same documented limit as
    #   ``__deepcopy__``'s.
    #
    # Deliberately NOT changed: the declarative (``type=``) form's extractor
    # is also a closure, over a signal binding this dict does not record, and
    # a bench carrying one still refuses to pickle exactly as it does today.
    # ------------------------------------------------------------------

    def __getstate__(self):
        """Pickle state: everything, minus the function form's closures."""
        state = dict(self.__dict__)
        metrics = state.get('_metrics')
        if isinstance(metrics, list) and any(
            _is_function_form_metric(m) for m in metrics
        ):
            state['_metrics'] = [
                dict(m, extractor=None) if _is_function_form_metric(m) else m
                for m in metrics
            ]
            # A plain-function metric's fn may be a lambda or a nested def
            # (TASK-323); the stand-in makes an unpicklable one fail naming
            # the metric instead of with Python's bare PicklingError.
            for m in state['_metrics']:
                if _is_function_form_metric(m) and 'slot_kinds' in m:
                    m['fn'] = _PlainMetricFunction(m['name'], m['fn'])
        state['_probe_registry'] = []
        return state

    def __setstate__(self, state):
        """Restore, rebuild dropped extractors, re-attach detached probes.

        The rebinding selects exactly the leaves pickling detached — the ones
        carrying :data:`~silk.signals.base._DETACHED_BENCH`. A detached selected
        leaf remains detached until its owning Module graph restores its exact
        root/owner/authored-test binding; it never falls back to this bench.
        A leaf still
        holding a live weak reference is left alone, which is what keeps
        ``copy.copy`` (which also routes through this pair) from silently
        re-pointing probes that a shallow copy shares with the original.

        ``_probe_registry`` is rebuilt from the (now-attached) metric slots
        either way — ``__getstate__`` always empties it, since a probe
        reachable only from the registry would otherwise restore detached
        (see the pickling block comment above). On a genuine unpickle this
        recovers exactly the probes :meth:`ModuleTestbench.plot`'s no-argument form
        needs; on a ``copy.copy`` (metric slots shared, not copied) it
        recovers the same shared probes, correctly still pointing at
        ``self``. Either way, a probe minted only for plotting and never
        passed to ``add_metric`` does not survive — the same stated limit
        ``__deepcopy__`` documents.
        """
        from silk.signals.base import _DETACHED_BENCH
        self.__dict__.update(state)
        metrics = self.__dict__.get('_metrics')
        for metric in metrics or ():
            if not _is_function_form_metric(metric):
                continue
            # copy.copy never reduces the state, so the TASK-323 pickle
            # stand-in is still here on that path; a real unpickle has
            # already turned it back into the function.
            if isinstance(metric.get('fn'), _PlainMetricFunction):
                metric['fn'] = metric['fn'].fn
            if metric.get('extractor') is None:
                metric['extractor'] = _function_form_extractor(metric)
        _rebind_probes(self, metrics, lambda ref: ref is _DETACHED_BENCH)
        self._probe_registry = _registry_from_metric_leaves(metrics)

    def add_module_under_test(self, port_mapping):
        """Add the module being tested to the netlist.

        The fixed placement is named ``dut`` (``xdut`` in the netlist).
        Designer node paths into it start with ``dut.``; for example,
        ``probe('dut.mid')`` or ``probe('dut.x1[0].mid')``. A port path
        such as ``dut.out`` follows this placement's wiring to the bench net.

        Parameters
        ----------
        port_mapping : dict
            Maps module ports to testbench nets, for example
            ``{'vdd': 'vdd', 'gnd': '0', 'in': 'vin', 'out': 'vout'}``.

        Returns
        -------
        ModuleTestbench
            This testbench, for method chaining.

        Raises
        ------
        silk.exceptions.UndeclaredNetError
            If ``self.strict_nets`` is True (inherited from the DUT by
            default; see the class parameters) and ``port_mapping`` names a
            net the testbench netlist has not declared, for example a
            misspelled net name. Same error type and message shape a strict
            standalone ``Module`` raises.
        silk.exceptions.NotASubcircuitError
            If the module under test has no ``module_name``. A testbench
            instantiates its DUT as an ``x`` line, so the DUT must be a
            subcircuit; a Module without a ``module_name`` is a top-level
            circuit and has no ``.subckt`` to reference. Refused here, where
            the mistake is made, rather than at ``generate_netlist()`` much
            later, and as a ``ConfigurationError``, so
            ``run_all_tests`` propagates it instead of recording it as one
            test's error string.
        ValueError
            If a module under test was already added to this test.
        """
        if self._module_added:
            raise ValueError(f"Module already added to test {self.test_name}")

        # A top-level circuit cannot be a device under test: the testbench has
        # no .subckt to write an instance line against. Refuse at the call that
        # made the mistake -- this used to be accepted silently and surface as a
        # bare TypeError from the netlister's None concatenation (TASK-167).
        if self.module.module_name is None:
            from silk.circuit.modules_netlist import describe_module
            raise NotASubcircuitError(
                f"Device under test {describe_module(self.module)!r} has no "
                f"module_name, so it is a top-level circuit: a testbench "
                f"instantiates its DUT as an 'x' line and there is no .subckt "
                f"to reference. A device under test must be a subcircuit — "
                f"call define_module_name('<name>') on it, and declare its "
                f"ports with define_module_ports({{...}}), before "
                f"add_module_under_test(...).")

        from silk.circuit.modules_netlist import require_declared_ports
        require_declared_ports(self.module, "Device under test")

        # Create netlist if needed
        if self.netlist is None:
            self._create_netlist()

        # The bench keeps the live DUT. Placement wiring belongs to its own
        # instance record, so later edits and calibration reach every run.
        self.netlist.add_module(self.module, DUT_ELEMENT_NAME, nets=port_mapping)
        self._module_added = True
        return self

    def add_module(self, module: 'Module', name: str, nets: dict = None,
                   **ports) -> 'SubcircuitInstance':
        """Place a helper subcircuit beside the DUT and return its instance.

        ``module`` must be a Module with a ``module_name``. Its ports are
        wired via ``nets={'pin': net, ...}`` and/or ``**ports`` keyword
        form — the same two equivalent forms
        :meth:`add_module <silk.circuit.Module.add_module>` accepts;
        the positional third argument keeps working for existing callers
        that pass a plain port-mapping dict. The executing netlist is
        created lazily, and ``Module.add_module`` handles reference placement,
        connections, name collisions and strict nets. Missing required
        ports are rejected when generating the netlist.

        Helpers do not change the owning DUT, metrics or specs. Call
        ``add_module_under_test`` once before running, even when helpers
        have already been added.

        Parameters
        ----------
        module : Module
            The helper subcircuit; it must have a ``module_name``.
        name : str
            Instance name for the placement.
        nets : dict, optional
            Port-to-net mapping ``{'pin': net, ...}``.
        **ports
            The same mapping in keyword form; merged over ``nets``.

        Returns
        -------
        SubcircuitInstance
            The same instance handle returned by ``Module.add_module``.

        Raises
        ------
        TypeError
            If ``module`` is not a :class:`~silk.circuit.Module`.
        silk.exceptions.NotASubcircuitError
            If ``module`` has no ``module_name``.
        """
        # Local import: Module imports ModuleTestbench during initialization.
        from silk.circuit.modules import Module

        if not isinstance(module, Module):
            raise TypeError(
                f"Helper {name!r}: module must be a Module, "
                f"got {type(module).__name__}")
        if not module.module_name:
            from silk.circuit.modules_netlist import describe_module
            raise NotASubcircuitError(
                f"Helper {name!r} ({describe_module(module)!r}) "
                f"has no module_name. A helper must be a named subcircuit; "
                f"call define_module_name('<name>') and declare its ports "
                f"with define_module_ports({{...}}) before add_module(...).")
        if self.netlist is None:
            self._create_netlist()
        wiring = dict(nets) if nets else {}
        wiring.update(ports)
        return self.netlist.add_module(module, name, nets=wiring)

    def connect_module(self, name, nets=None, **ports):
        """Rewire a helper or the ``dut`` placement through the bench netlist.

        Accepts the same partial wiring as :meth:`Module.connect_module`.
        Raises CircuitError before a netlist exists; place a block with
        add_module_under_test or add_module first.
        """
        from silk.exceptions import CircuitError
        if self.netlist is None:
            raise CircuitError(
                "Call add_module_under_test or add_module before connect_module; "
                "this testbench has no netlist.")
        return self.netlist.connect_module(name, nets=nets, **ports)

    # Delegate testbench setup methods to netlist

    def add(self, *items):
        """Unified front door. A source object goes to ``add_source``; every
        other item is forwarded to the testbench's own top-level circuit
        (``self.netlist``, an ordinary nameless ``Module``) through
        ``Module.add()``, with the same validation and the same errors. A
        test holds anything a circuit can: PDK devices, ideal R/C/L and
        other elements. Use add_sim_type / add_metric / add_spec for the
        rest, and add_module for helper subcircuits.

        Parameters
        ----------
        *items
            Source objects (SourceBase) and/or any element
            ``Module.add()`` accepts.

        Returns
        -------
        ModuleTestbench
            self (for method chaining)

        Raises
        ------
        TypeError
            If ``Module.add()`` refuses an item.

        Examples
        --------
        ::

            from silk import sources
            from silk.elements import elements
            tb.add(sources.vdc('vdd', positive='vdd', negative='0', dc=3.3))
            tb.add(elements.resistor('rload', p='vout', n='0', value=1e3))
            tb.add(pdk.nmos1p8('Mload', d='vout', g='vin', s='0', b='0',
                               w=1.0, l=0.15))
        """
        from silk.sources import SourceBase
        for item in items:
            if isinstance(item, SourceBase):
                self.add_source(item)
            else:
                if self.netlist is None:
                    self._create_netlist()
                self.netlist.add(item)
        return self

    def add_source(self, source):
        """Add a source to the testbench.

        Parameters
        ----------
        source : SourceBase
            A source object from :mod:`silk.sources` (``VoltageSource``,
            ``CurrentSource``, ``VCVS`` and so on).

        Returns
        -------
        ModuleTestbench
            This testbench, for method chaining.

        Examples
        --------
        ::

            from silk import sources
            tb.add(sources.vdc('vdd', positive='vdd', negative='0', dc=3.3))
        """
        if self.netlist is None:
            self._create_netlist()

        self.netlist.add_source(source)
        return self

    def add_passive(self, passive):
        """Add a passive element (Resistor, Capacitor, or Inductor) to the testbench.

        Parameters
        ----------
        passive : object
            An instance from :mod:`silk.elements.passives` (``Resistor``,
            ``Capacitor`` or ``Inductor``).

        Returns
        -------
        ModuleTestbench
            This testbench, for method chaining.

        Examples
        --------
        ::

            from silk.elements import passives
            test.add_passive(passives.Capacitor('cload', pos='vout', neg='0', capacitance=100e-12))
        """
        if self.netlist is None:
            self._create_netlist()

        self.netlist.add_passive(passive)
        return self

    def add_custom_line(self, line):
        """Add a raw SPICE line to the testbench.

        Parameters
        ----------
        line : str
            SPICE line to add, for example ``'.param vbias=0.9'``.

        Returns
        -------
        ModuleTestbench
            This testbench, for method chaining.
        """
        if self.netlist is None:
            self._create_netlist()

        self.netlist.add_custom_line(line)
        return self

    def set_temperature(self, temperature_c: float | None) -> 'ModuleTestbench':
        """Configure ambient TEMP on this test's executing netlist; return self.

        See ``Module.set_temperature``. This never changes the DUT's options.
        A setting made here is this test's own and always wins over the one
        it would otherwise inherit from its live owner, in either direction:
        a hot test under a nominal circuit and a nominal test under a hot one
        both keep what they were told. ``set_temperature(None)`` is equally
        deliberate — it opts this test out of the owner's temperature rather
        than resuming inheritance.

        Parameters
        ----------
        temperature_c : float or None
            Ambient temperature in degrees Celsius, finite and above
            -273.15; ``None`` clears the setting.

        Returns
        -------
        ModuleTestbench
            This testbench, for method chaining.

        Raises
        ------
        TypeError
            If ``temperature_c`` is neither a real number nor ``None``
            (booleans are refused).
        ValueError
            If it is NaN, infinite, or at or below absolute zero.
        """
        if self.netlist is None:
            self._create_netlist()
        self.netlist.set_temperature(temperature_c)
        return self

    def set_corner(self, corner: str | None) -> 'ModuleTestbench':
        """Set the process corner this test runs at by default; return self.

        The corner :meth:`run` and :meth:`evaluate` (with ``rawfile=``) use
        when they are called without one. Precedence, highest first:

        1. a corner passed to the call — ``tb.run(corner='ss')``, or
           ``run_all_tests(corner='ss')``, which forwards it to every test;
        2. this setting;
        3. the PDK's ``default_corner``.

        :meth:`run_corners` is an explicit sweep and ignores this setting:
        called with no list it still runs the PDK's process corners.

        The name is checked here, against the PDK's declared corners
        (``load_pdk(name).corners``), so a misspelled corner fails at the
        call rather than after a simulation. A PDK that declares no corners
        accepts any non-empty name. ``set_corner(None)`` clears the setting.
        A :class:`~silk.simulation.corner.Corner` member is accepted and
        stored as its name.

        Parameters
        ----------
        corner : str or Corner or None
            Corner name or :class:`~silk.simulation.corner.Corner` member;
            ``None`` clears the setting.

        Returns
        -------
        ModuleTestbench
            This testbench, for method chaining.

        Raises
        ------
        TypeError
            *corner* is not a string, ``Corner`` or ``None``.
        silk.exceptions.UnknownCornerError
            *corner* is empty, or not one of the PDK's declared corners. The
            message lists the valid ones. A :class:`ValueError` and a
            :class:`~silk.exceptions.ConfigurationError`.
        """
        if corner is None:
            self._corner = None
            return self
        if not isinstance(corner, str):
            raise TypeError(
                f"test {self.test_name!r}: set_corner() takes a corner name "
                f"string or None, got {type(corner).__name__}")
        self._corner = self._checked_corner_name(corner)
        return self

    def _checked_corner_name(self, corner):
        """*corner* as a name this PDK declares, else
        :class:`~silk.exceptions.UnknownCornerError` listing the valid ones."""
        from silk.exceptions import UnknownCornerError
        from silk.utils.deck_text import validate_corner_name
        name = validate_corner_name(
            corner, who=f'test {self.test_name!r}: corner')
        corners = getattr(self.pdk_details, 'corners', {}) or {}
        if not name or (corners and name not in corners):
            valid = ', '.join(repr(c) for c in sorted(corners)) or '(any name)'
            raise UnknownCornerError(
                f"test {self.test_name!r}: unknown corner {name!r}; this "
                f"PDK's corners are: {valid}")
        return name

    def set_corners(self, corners: list[str] | None) -> 'ModuleTestbench':
        """Declare the corners this test runs at; return self.

        With a list declared, :meth:`run` and both ``run_all_tests`` walks
        (``n_workers`` ``None``, 1 or more) run this test once per corner,
        each in its own ``<testbench dir>/<corner>/`` folder, and return one
        :class:`~silk.circuit.spec_report.CornerResults` keyed by corner in
        the declared order, instead of one
        :class:`~silk.circuit.spec_report.SpecReport`.

        A corner that fails is kept out of the results rather than stopping
        the others: it is listed in ``.failed`` with its
        error, ``passed`` is ``False``, ``summary()`` counts it, and ``stat``
        and ``worst`` skip it with one counted ``MetricWarning``. Only when
        every corner fails does the run raise
        :class:`~silk.exceptions.SimulationError`, carrying the first error
        (``run_all_tests`` records it as that test's ``{'error': ...}``). A
        :class:`~silk.exceptions.ConfigurationError` still stops the run.

        A corner passed to the call — ``tb.run(corner='ss')`` or
        ``run_all_tests(corner='ss')`` — runs that one corner of the plan
        (still a ``CornerResults``). :meth:`run_corners` is an explicit sweep
        and ignores this setting; its behaviour is unchanged. The setting is
        this test's own and is not inherited from the owner Module.

        Grouped corner selections retain all matching complete rows and
        refuse an absent corner. Each name is checked as :meth:`set_corner`
        checks its one name.
        ``set_corners(None)`` clears the declaration. A
        :class:`~silk.simulation.corner.Corner` member is stored as its name.

        Parameters
        ----------
        corners : sequence of str or Corner or None
            The corners to sweep; ``None`` clears the declaration.

        Returns
        -------
        ModuleTestbench
            This testbench, for method chaining.

        Raises
        ------
        TypeError
            *corners* is a single string, not a list, or holds a non-string.
        silk.exceptions.UnknownCornerError
            A name is empty or not one of the PDK's corners, the list is
            empty. Repeated ungrouped coordinates and complete runs refuse
            during planning, before emission. A :class:`ValueError` and a
            :class:`~silk.exceptions.ConfigurationError`. Nothing is stored
            when it raises.
        """
        from silk.exceptions import UnknownCornerError
        if corners is None:
            self._corners = None
            return self
        if isinstance(corners, str) or not isinstance(corners, (list, tuple)):
            raise TypeError(
                f"test {self.test_name!r}: set_corners() takes a list of "
                f"corner names or None, got {type(corners).__name__}"
                + (" (use set_corner() for one corner, or wrap it in a list)"
                   if isinstance(corners, str) else ''))
        for corner in corners:
            if not isinstance(corner, str):
                raise TypeError(
                    f"test {self.test_name!r}: set_corners() takes corner "
                    f"name strings, got {type(corner).__name__} {corner!r}")
        if not corners:
            raise UnknownCornerError(
                f"test {self.test_name!r}: set_corners() needs at least one "
                "corner; pass None to clear the declaration")
        names = tuple(self._checked_corner_name(c) for c in corners)
        self._corners = names
        return self

    @property
    def corners(self):
        """This test's declared corner list from :meth:`set_corners`, as a
        tuple, or ``None`` when it has none. Assigning to it is
        :meth:`set_corners`."""
        return getattr(self, '_corners', None)

    @corners.setter
    def corners(self, corners):
        # Kept assignable: subclasses that predate set_corners() store their
        # own corner list here (the PVTTestbench example). Such a subclass
        # overrides run(), so the plan never applies to it (_plan_runs).
        self.set_corners(corners)

    def set_monte_carlo(self, n, *, variation='mismatch', seed=0,
                        record_draws=False, correlate=None,
                        min_yield=None, samples_per_process=1,
                        output_mode='overwrite') -> 'ModuleTestbench':
        """Declare a Monte Carlo run for this test; return self.

        A campaign overwrites by default even when the module writes new folders.
        Use ``output_mode='new'`` to keep each campaign in its own folder per corner.

        With a setting declared, :meth:`run` and both ``run_all_tests`` walks
        run this test *n* times per corner — at every corner of
        :meth:`set_corners` when a list is declared, else at the test's
        corner — and return one
        :class:`~silk.circuit.monte_carlo.MonteCarloResults` keyed
        ``(corner, sample index)``. Each sample is a statistical corner run:
        its deck loads the PDK's statistical library section for *variation*
        and carries its own seed, derived from *seed* and the index, so the
        same *seed* gives the same samples. Sample *k* of corner ``ss`` runs
        in ``<testbench dir>/ss/mc_<kkkk>/`` with ``output_mode='overwrite'``;
        ``output_mode='new'`` inserts a fresh ``run-.../`` before the sample
        folders on every campaign. silk never deletes
        sample output; removing it is the user's job.

        Parameters
        ----------
        n : int or None
            Samples per corner, a positive int. ``None`` clears the
            setting (the other arguments are then ignored).
        variation : str, default 'mismatch'
            ``'mismatch'`` (per-device draws, the default),
            ``'process'`` (one draw per sample, shared by every device;
            only at the corners the PDK lists for it) or ``'both'``.
        seed : int, default 0
            Base seed, an int (default 0).
        record_draws : bool, default False
            Record each device's drawn model parameters
            (consumed by ``MonteCarloResults.sensitivity``).
        correlate : sequence, optional
            Probe Signals to record per sample.
        min_yield : float, optional
            With a value in (0, 1], ``passed`` means every spec's
            yield is at least this.
        output_mode : {'overwrite', 'new'}, default 'overwrite'
            Campaign placement, independent of the module's output mode.
            ``'new'`` keeps samples, draws and statistical wrappers together
            in a fresh folder per corner on every campaign. Invalid values
            raise ``ValueError``.
        samples_per_process : int, default 1
            ngspice only. At most this many consecutive samples of one
            corner run in one ngspice process, which loads the model
            library once instead of once per sample; the results are the
            same as one process per sample. 1 runs one process per sample.
            Chunks run in parallel under ``n_workers``, so ``n`` samples per
            corner at ``samples_per_process=s`` make ``ceil(n / s)``
            processes per corner. Larger than 1 refuses on other
            simulators (:class:`~silk.exceptions.NotSupportedError`) and
            with ``save_device_currents``
            (:class:`~silk.exceptions.MonteCarloError`).

        Notes
        -----
        The behavioral backend draws Normal/Uniform model parameters once
        per sample. It requires an active statistical declaration, and needs
        neither statistical PDK models nor an external simulator.

        Before the first sample the run refuses, without writing a deck: a
        PDK with no statistical block
        (:class:`~silk.exceptions.NotSupportedError`); a deck with no PDK
        device, two specs of one name, process variation off the PDK's
        process corners (:class:`~silk.exceptions.MonteCarloError`); a model
        library lacking a needed section
        (:class:`~silk.exceptions.MissingLibrarySectionError`).

        Samples record no design-stage measurement on the first-class Specs. The setting is this test's own and is not
        inherited from the owner Module.

        Returns
        -------
        ModuleTestbench
            This testbench, for method chaining.

        Raises
        ------
        MonteCarloError
            Invalid *n*, *variation*, *seed*,
            *record_draws* or *min_yield*; nothing is stored.
        ValueError
            Unsupported *output_mode*; nothing is stored.
        """
        from silk.circuit.monte_carlo import MonteCarloSettings
        if n is None:
            self._monte_carlo = None
            return self
        self._monte_carlo = MonteCarloSettings(
            n=n, variation=variation, seed=seed, record_draws=record_draws,
            correlate=tuple(correlate or ()), min_yield=min_yield,
            samples_per_process=samples_per_process, output_mode=output_mode)
        return self

    @property
    def monte_carlo(self):
        """This test's declared
        :class:`silk.circuit.monte_carlo.MonteCarloSettings` from
        :meth:`set_monte_carlo`, or ``None`` when it has none."""
        return getattr(self, '_monte_carlo', None)

    def run_monte_carlo(self, n, *, n_workers=None, variation='mismatch',
                        seed=0, record_draws=False, correlate=None,
                        min_yield=None, samples_per_process=1,
                        output_mode='overwrite'):
        """Run this test as a Monte Carlo campaign once; return its
        :class:`~silk.circuit.monte_carlo.MonteCarloResults`.

        Campaigns overwrite by default even when the module writes new folders;
        use ``output_mode='new'`` to keep each campaign.

        The one-off form of :meth:`set_monte_carlo`, taking the same
        arguments: it runs with them and leaves the declared setting as it
        was. The samples run at every corner of :meth:`set_corners` when a
        list is declared, else at the test's corner. *n_workers* means what
        it means for :meth:`run_corners`: samples are prepared and simulated
        in batches of that many.

        Parameters
        ----------
        n : int
            Samples per corner, a positive int.
        n_workers : int, optional
            Samples prepared and simulated per batch; as for
            :meth:`run_corners`.
        variation : str, optional
            ``'mismatch'`` (default), ``'process'`` or ``'both'``; see
            :meth:`set_monte_carlo`.
        seed : int, optional
            Base seed from which each sample's seed is derived. Default 0.
        record_draws : bool, optional
            Record each device's drawn model parameters. Default False.
        correlate : sequence, optional
            Probe Signals to record per sample.
        min_yield : float, optional
            Yield in (0, 1] at which ``passed`` is true; see
            :meth:`set_monte_carlo`.
        output_mode : {'overwrite', 'new'}, default 'overwrite'
            Overwrite campaign folders even when the module uses new folders;
            use ``'new'`` to keep each campaign. See :meth:`set_monte_carlo`.

        Returns
        -------
        MonteCarloResults
            Results keyed by ``(corner, sample index)``.

        Raises
        ------
        silk.exceptions.MonteCarloError
            Invalid *n*, *variation*, *seed*, *record_draws* or *min_yield*,
            or a deck the sampler cannot vary.
        ValueError
            Unsupported *output_mode*.
        silk.exceptions.NotSupportedError
            The backend or PDK has no statistical models.
        silk.exceptions.SimulationError
            Every planned run failed.
        """
        from silk.circuit.monte_carlo import MonteCarloSettings
        settings = MonteCarloSettings(
            n=n, variation=variation, seed=seed, record_draws=record_draws,
            correlate=tuple(correlate or ()), min_yield=min_yield,
            samples_per_process=samples_per_process, output_mode=output_mode)
        from silk.circuit.modules_testbench import _check_process_sources
        _check_process_sources((), standalone=self)
        planned = self._plan_runs(_monte_carlo=settings)
        return self._run_plan(planned, n_workers=n_workers)

    @property
    def corner(self):
        """This test's own default corner from :meth:`set_corner`, or
        ``None`` when it has none (runs then use the PDK's default)."""
        return getattr(self, '_corner', None)

    def _default_corner(self):
        """The corner a run uses when none is passed: this test's own
        (:meth:`set_corner`), else the PDK's ``default_corner``."""
        if self.corner is not None:
            return self.corner
        return getattr(self.pdk_details, 'default_corner', 'tt')

    def set_supply_voltage(self, volts: float | None, *,
                           source: str | None = None) -> 'ModuleTestbench':
        """Configure the primary supply on this test's netlist; return self.

        See ``Module.set_supply_voltage``. This never changes the DUT's own
        sources. A setting made here is this test's own and always wins over
        the one it would otherwise inherit from its live owner, in either
        direction. ``set_supply_voltage(None)`` is equally deliberate — it
        opts this test out of the owner's supply rather than resuming
        inheritance.

        Parameters
        ----------
        volts : float or None
            Finite real volts, or ``None`` to stop configuring a supply.
        source : str, optional
            Name of the voltage source to set; resolved from the circuit
            when omitted (see ``Module.set_supply_voltage``).

        Returns
        -------
        ModuleTestbench
            This testbench, for method chaining.

        Raises
        ------
        ValueError
            If ``source`` is not a non-empty string, or is given together
            with ``volts=None``.
        silk.exceptions.CharacterizationContextError
            If ``volts`` is not a finite real number.
        """
        if self.netlist is None:
            self._create_netlist()
        self.netlist.set_supply_voltage(volts, source=source)
        return self

    def set_simulator_options(self, **options):
        """Configure simulator options on this test's executing netlist; return self.

        A thin delegate to ``self.netlist.set_simulator_options`` -- see
        ``Module.set_simulator_options`` for preset expansion, ``temp``
        handling and the unknown-key warning; this method holds none of that
        logic itself. A key set here is this test's own and
        always wins, key by key, over the same key inherited from the live
        owner Module's own ``set_simulator_options``; an owner key this test
        never set reaches the emitted deck as a default, re-resolved at
        every run so a later change on the owner is followed.

        Parameters
        ----------
        **options
            Simulator option names and values, as for
            ``Module.set_simulator_options``.

        Returns
        -------
        ModuleTestbench
            This testbench, for method chaining.
        """
        if self.netlist is None:
            self._create_netlist()
        self.netlist.set_simulator_options(**options)
        return self

    def _inherited_supply(self):
        """The primary-supply setting in force for this run, or ``None``.

        Resolved exactly as :meth:`_inherited_temperature` is: a recursive
        ``run_all_tests`` supplies the nearest setting found up the live tree
        it is walking, and otherwise the only candidate is this test's own
        live owner. The named source, if any, is re-resolved against this
        test's own netlist, which is where the rail actually lives.
        """
        from silk.simulation.conditions import own_supply_setting
        entry = current_placement(self)
        if entry is not None:
            return entry.inherited_supply
        return own_supply_setting(self.module)

    def _inherited_temperature(self):
        """The ambient temperature setting in force for this run, or ``None``.

        A recursive ``run_all_tests`` supplies the nearest setting found up
        the live tree it is walking; otherwise the only candidate is this
        test's own live owner. The two agree whenever both apply, because a
        walk that reaches this test has already folded its direct owner's
        setting into what it propagates.
        """
        from silk.simulation.conditions import own_temperature_setting
        entry = current_placement(self)
        if entry is not None:
            return entry.inherited_temperature
        return own_temperature_setting(self.module)

    @staticmethod
    def _check_declaration(fact, value):
        """A declaration is a value, ``None`` (not declared) or ``NONE``.

        ``ConditionState.NONE`` declares that the condition does not apply.
        The other members are not declarations: unknown is what ``None``
        already records, and FIXED / SWEEP carry no value.
        """
        from silk.exceptions import CharacterizationContextError
        from silk.simulation.conditions import ConditionState
        if isinstance(value, ConditionState) and value is not ConditionState.NONE:
            raise CharacterizationContextError(
                f'{fact}={value} is not a declaration: pass a value, '
                'ConditionState.NONE (the condition does not apply), or omit '
                'it to record unknown.')
        return value

    def _external_evidence(self, corner, temperature, voltage, sweep,
                           *, previous=None, backend=None):
        """Context and conditions for caller-declared external evidence.

        Recording conditions is not building a profile: each declared fact is
        recorded as given. Each of corner, temperature and voltage is a value,
        ``ConditionState.NONE`` (declared not to apply) or ``None`` (not
        declared, recorded unknown). Only a declared set where every fact is a
        value or none yields a scoped ``CharacterizationContext``; anything
        else stays unscoped. ``previous`` supplies facts the caller omitted
        when redeclaring already-loaded external results.
        """
        from silk.exceptions import CharacterizationContextError
        from silk.simulation.conditions import (
            ConditionState, OperatingConditions, validate_supply,
            validate_temperature,
        )
        NONE = ConditionState.NONE
        for fact, value in (('corner', corner), ('temperature', temperature),
                            ('voltage', voltage)):
            self._check_declaration(fact, value)
        if sweep is not None and temperature is not None:
            raise CharacterizationContextError(
                'Loaded temperature sweep cannot have scalar temperature '
                'declarations (including ConditionState.NONE); evaluate the '
                'file without temperature.')
        if previous is not None:
            if corner is None:
                corner = previous.scalar('corner')
            if temperature is None and previous.temperature_kind in ('fixed', 'none'):
                temperature = previous.scalar('temperature')
            if voltage is None and previous.supply_kind in ('fixed', 'none'):
                voltage = previous.scalar('supply')
        if sweep is not None:
            temperature = None
        if (corner is not None and temperature is not None
                and voltage is not None):
            context = self._build_context(corner, temperature, voltage)
            temperature, voltage = context.temperature, context.voltage
        else:
            context = None
            if temperature is not None and temperature is not NONE:
                temperature = validate_temperature(temperature)
            if voltage is not NONE:
                voltage = validate_supply(voltage)

        def kind(value):
            return 'unknown' if value is None else (
                'none' if value is NONE else 'fixed')

        def basis(value):
            return 'unknown' if value is None else 'caller-declared'

        def held(value):
            return None if value is NONE else value

        conditions = OperatingConditions(
            'external',
            (backend if backend is not None
             else getattr(self.netlist, '_raw_simulator_hint', None)),
            held(corner),
            'sweep' if sweep else kind(temperature),
            held(temperature),
            'unknown' if sweep else basis(temperature), sweep,
            held(voltage),
            basis(voltage),
            corner_kind=kind(corner), corner_basis=basis(corner),
            supply_kind=kind(voltage),
        )
        return context, conditions

    def _context_from_conditions(self, conditions):
        """The scoped context of silk-origin *conditions*, or ``None``.

        A profile needs each of corner, temperature and supply to be a value
        or none (design 2026-10-01 section 2.2); unknown and sweep never
        scope. ``ConditionState.NONE`` passes through as the scalar.
        """
        from silk.simulation.conditions import ConditionState
        profile_states = (ConditionState.FIXED, ConditionState.NONE)
        if any(conditions.state(fact) not in profile_states
               for fact in ('corner', 'temperature', 'supply')):
            return None
        return self._build_context(conditions.scalar('corner'),
                                   conditions.scalar('temperature'),
                                   conditions.scalar('supply'))

    def add_sim_type(self, simType):
        """Set the simulation analysis.

        Parameters
        ----------
        simType : Transient or AC or DC or Noise
            A typed analysis object from ``silk.simulation.analysis``. The
            legacy string form was removed and raises ``TypeError``.

        Returns
        -------
        ModuleTestbench
            This testbench, for method chaining.

        Examples
        --------
        ::

            from silk.simulation import analysis
            test.add_sim_type(analysis.Transient(step=10e-9, stop=50e-6))
            test.add_sim_type(analysis.AC(variation='dec', points=10, fstart=1, fstop=1e9))
        """
        if self.netlist is None:
            self._create_netlist()

        self.netlist.add_sim_type(simType)
        return self

    # -----------------------------------------------------------------
    # Probes — name a signal now, read it after the run (TASK-215.5)
    # -----------------------------------------------------------------

    def probe(self, name, type='V', analysis=None):
        """Name a signal of this testbench and return it as an unbound Signal.

        A probe carries no data. It holds the rawfile name, the loader
        selector and a **weak** reference to this testbench; reading ``.x``,
        ``.y``, ``.xlabel`` or ``.ylabel`` resolves it against the bench's
        *current* results at that moment and stores nothing back, so a probe
        can never show a stale run. Before the bench has run, a read raises
        :class:`~silk.exceptions.UnboundSignalError` naming the probe and the
        bench, and ``probe.bound`` answers the same question
        without raising.

        The reference is weak, so a bench's probes never keep the bench (and
        through it the whole design) alive, and a metric list carrying probes
        drags no bench graph into a deep copy.

        Parameters
        ----------
        name : str or PlacementReference
            A placement net/port/bus reference, such as ``stage.vbias`` or
            ``stage.bus('d')``, binds its voltage to this bench after checking
            placement ownership. A bare definition ``Net`` refuses because
            it has no placement identity. Strings use these spellings:

            ``'out'`` / ``'v(out)'``
                The voltage on net ``out``. The two are the same probe. A
                *plain* name (no dot, no parentheses, no colon) is checked
                against this testbench's nets right here — see *Raises*.
            ``'dut.mid'`` / ``'dut.x1[0].mid'`` / ``'stage.mid'``
                A scalar voltage by designer placement path, rooted at this
                bench. The DUT placement is ``dut``; helpers use their given
                names. Connected ports follow placement wiring. Candidates
                resolve against the selected reader's graph and raw plot on
                every read; the probe retains the original request.
            ``'xdut.mid'``
                The existing emitted-path spelling also works, including
                opaque model internals. An emitted reading wins over a
                different designer target with a warning. Multiple designer
                targets refuse with a placement-reference remedy; ambiguous
                raw rows still raise AmbiguousSignalError. A known node need
                not have been saved; missing vectors still raise on read.
            ``'i(vs)'``
                The branch current of source ``vs``. The emitted-instance
                mapping is applied for you — a ``VoltageSource('vs', …)`` is
                emitted as ``Vvs`` and its vector is ``i(vvs)``, so this is
                the same probe as ``probe('vvs', 'I')``. A name inside
                ``i(…)`` that matches no source of this bench is passed
                through verbatim, which is what makes the raw spelling work.
            ``'xdut.m1:d'`` / ``'st1.MN:gm'``
                A colon splits on its LAST occurrence; the suffix, case-
                insensitive, decides which of two probes this is. A bracket suffix would collide with a net name ending the
                same way, so only the colon spelling dispatches device probes. The two suffix
                sets are disjoint, so the dispatch is never ambiguous:

                - one of ``d``/``g``/``s``/``b`` — device ``m1``'s
                  TERMINAL current, read through the 0 V ammeter wrappers
                  (using the emitted device path): everything before
                  the last colon is the emitted instance path, the last
                  path component is the device's silk name. Resolves to
                  the vector ``v.<path>.<emitted device>.vamm_<port>``,
                  the same string
                  :meth:`silk.simulation.backend.SpiceBackend.terminal_current_vector`
                  (overridden by ``NgspiceBackend``) builds. Needs the ammeter opt-in — see *Raises*.
                  The prefix may instead be the DESIGNER name
                  (``'st1.MN:d'``) — the same dot-joined
                  instance names ``'st1.MN:gm'`` takes — and yields the
                  same vector. The emitted-path reading is tried first
                  and wins; a prefix that reaches DIFFERENT devices under
                  both readings emits a ``UserWarning`` naming the
                  ``'xdut.<path>.<name>:d'`` spelling that selects the
                  shadowed DUT device.
                - one of the 17 :data:`silk.simulation.backend.DEVICE_OP_PARAMS`
                  names (``id``, ``vgs``, ``vth``, ``gm``, ``gds``, ``cgg``,
                  … — not just the smaller set ``op_table()``/
                  ``save_device_params`` default to) — device ``st1.MN``'s
                  raw operating-point PARAMETER, by
                  dot-joined designer instance name as the
                  prefix. One spelling for every analysis: ``Op()`` gives
                  one value, a ``DC`` sweep and a ``Transient`` give the
                  parameter against the swept source or time, and ``AC``
                  gives the constant bias-point value (ngspice writes it
                  as a zero-imaginary complex vector; silk returns it
                  real). Returns the same
                  :class:`~silk.signals.base.Signal` type as every other
                  probe spelling — it composes with arithmetic,
                  ``add_metric`` and :class:`~silk.specs.spec.Spec` the
                  same way, and an ``op_table()`` cell and its probe can
                  never disagree because both read the same saved vector.
                  A *derived* op_table column (``gm_id``, ``ro``, ``vov``,
                  ``vds_margin``, ``ft``, ``region``) is refused — the
                  error names the plain-Signal arithmetic to use instead
                  (e.g. ``probe('st1.MN:gm') / probe('st1.MN:id')`` for
                  ``gm_id``). Not limited to the PDK's declared
                  ``op_params`` set (that default only narrows the
                  all-devices ``op_table()`` overview): a probe may name
                  any backend-supported parameter, validated before the
                  run — see *Raises*.
                - anything else is refused before the run, naming both
                  closed sets and the closest match — see *Raises*. The
                  raw net spelling ``'v(M1:gm)'`` always means the NET (a
                  net literally named that way, however unlikely) —
                  ``'v(...)'`` never reaches this dispatch at all.
        type : {'V', 'I'}, optional
            The low-level loader selector, what ``Module.load_signal`` takes.
            It is the escape hatch for a raw vector name
            (``probe('v.xdut.xmm1.vamm_d', 'I')``) and is not the taught
            form. When *name* carries its own SPICE selector (``i(…)``,
            ``v(…)``, ``…:port``) the spelling wins.
        analysis : {'tran', 'ac', 'dc', 'noise', 'op'}, optional
            WHICH analysis's plot this probe reads, in a test that declares
            more than one (the contract in docs/probes.md). Omit it when the test
            declares one analysis, and when the metric this probe feeds
            could only have come from one of the declared ones — silk reads
            that plot and says nothing. Name it when the choice is open: a
            kind-``any`` metric such as ``average`` in a tran+ac test is
            REFUSED without it, naming both candidates, because silk never
            guesses a plot::

                tb.add_sim_type(Transient(...)); tb.add_sim_type(AC(...))
                vout_t = tb.probe('vout', analysis='tran')   # required here
                vout_f = tb.probe('vout', analysis='ac')
                tb.add_metric('vavg', average,    sig=vout_t)
                tb.add_metric('gain', dc_gain_dB, sig=tb.probe('vout'))

            The last line needs no analysis: ``dc_gain_dB`` takes a
            frequency response, and ``.ac`` is the only declared analysis
            that produces one. Reading ``.x``/``.y`` on the probe itself
            reads the same plot.

        Returns
        -------
        Signal
            An unbound Signal (``bound`` is ``False`` until the bench has
            results). Arithmetic and transforms on it defer; ``add_metric(name, fn, sig=probe)`` binds it.

        Raises
        ------
        TypeError
            If *name* is not a non-empty string.
        ValueError
            If *analysis* is not one of ``'tran'``/``'ac'``/``'dc'``/
            ``'noise'``/``'op'`` (whether the test declares it is settled at run
            time, since a hand-built test may add its sim types after its
            probes); if *name* is a plain net name this testbench's netlist
            does not have (the message names the net and the available nets); if the
            selector is not ``'V'``/``'I'``; if a ``…:d``/``…:g``/``…:s``/
            ``…:b`` terminal spelling names a device this testbench does
            not contain, or is used without the ammeter opt-in —
            ``save_device_currents = True`` on the DUT,
            ``terminal_current_method = 'ammeter_wrapper'``, and a
            Transient analysis, the three conditions that put the vector
            in the rawfile at all.

            A ``'name:param'`` device-parameter probe raises IMMEDIATELY,
            right here in ``probe()``, before any deck is even built,
            when *param* is a derived ``op_table()`` column, not
            a raw parameter; or the suffix after the last ``':'`` is
            neither a terminal (``d``/``g``/``s``/``b``) nor one of
            :data:`silk.simulation.backend.DEVICE_OP_PARAMS`' 17
            recognised names (a difflib hint names the closest match
            against both closed sets) — an
            unrecognised ``'@'`` name can abort ngspice's whole write, or
            come back a real, populated, all-zero reading with no warning
            at all, depending on what else that run happens to save, so
            neither failure is safe to defer. ``run()`` itself raises when
            *name* does not resolve to a MOS device of this testbench (the
            message lists the nearest known names), or, on Xyce only, the
            device or *param* is outside what Xyce's ``N()``/``ID()``
            forms can print (a bad Xyce ``.print`` symbol aborts the whole
            run, so this is checked in advance). ``run()`` also raises if
            this probe combines with
            ``save_device_currents`` / ``terminal_current_method =
            'ammeter_wrapper'`` (unsupported together — see
            ``silk.circuit.spice_netlister.PySpiceNetlister._resolve_device_probe_lines``),
            or if the vector a validated probe asked for came back missing or
            zero-length at read time — naming the device, the parameter, the
            analysis and the simulator version, and what that run did
            provide. Reading a device-parameter probe created *after* the
            bench last ran raises naming the probe, saying to re-run —
            silk never returns 0 for one it never asked the simulator for.

        Notes
        -----

        The taught program is probe -> ``add_metric(fn, sig=...)`` -> run ->
        plot (``docs/probes.md``).

        Examples
        --------
        ::

            vp   = tb.probe('vp')
            vm   = tb.probe('vm')
            iout = tb.probe('i(vs)')
            vcm  = (vp + vm) / 2
            tb.add_metric('vcm_avg', average, sig=vcm)
            tb.run()
            vcm.plot()
            print(vp.y.mean())
        """
        from silk.signals.base import Signal
        from silk.circuit.placement_reference import PlacementReference, LiteralVoltageName
        from silk.circuit.net import Net
        from silk.exceptions import CircuitError
        if isinstance(name, PlacementReference):
            return name._bind(self, type=type, analysis=analysis)
        if isinstance(name, Net):
            raise CircuitError('A definition-level Net has no placement; use tb.probe(stage.net(name)).')
        if isinstance(name, LiteralVoltageName):
            if type is not None and str(type).upper() != 'V':
                raise CircuitError('A literal net reference is voltage; use a source or terminal-current probe.')
            if analysis is not None and analysis not in _SIM_KEY_ANALYSIS.values():
                raise ValueError(f'Unknown analysis {analysis!r}.')
            if self._effective_simulator() == 'behavioral' or getattr(self.netlist, '_behavioral_mode', False):
                from silk.simulation.behavioral_backend import refuse_behavioral_probe
                refuse_behavioral_probe(self.netlist, name, 'V')
            if _is_plain_net_name(name):
                self._check_probe_net(name)
            return self._register_probe(Signal._unbound(name, 'V', self, analysis=analysis))

        if analysis is not None and analysis not in _SIM_KEY_ANALYSIS.values():
            raise ValueError(
                f"probe({name!r}, analysis={analysis!r}): not a silk "
                f"analysis; expected one of "
                f"{sorted(set(_SIM_KEY_ANALYSIS.values()))}.")
        if not isinstance(name, str) or not name.strip():
            raise TypeError(
                f"probe() needs a non-empty signal name as a string; got "
                f"{name!r}. Spellings: 'out', 'v(out)', '{DUT_ELEMENT_NAME_PREFIX}.mid', "
                f"'i(vs)', '{DUT_ELEMENT_NAME_PREFIX}.m1:d'."
            )

        selector = 'V' if type is None else str(type).upper()
        if selector not in ('V', 'I'):
            raise ValueError(
                f"probe() selector must be 'V' (voltage) or 'I' (current), "
                f"got {type!r}"
            )
        raw = name.strip()

        # A 'v(...)'/'i(...)' wrapper ALWAYS means the raw net/branch
        # spelling inside it — checked FIRST, before the colon dispatch
        # below, so 'v(M1:gm)' bypasses the device branch entirely and
        # always means the net (owner decision 2026-09-28, point 3): the
        # colon dispatch must never see a name still wrapped in
        # 'v(...)'/'i(...)'.
        head, inner = _split_spice_selector(raw)

        # The loaded result decides too: simulate_behavioral() restores the
        # caller's simulator selection after its run (TASK-448.10 review).
        if (head is None and ':' in raw
                and (self._effective_simulator() == 'behavioral'
                     or getattr(self.netlist, '_behavioral_mode', False))):
            from silk.simulation.behavioral_backend import refuse_behavioral_probe
            refuse_behavioral_probe(self.netlist, raw, selector)

        # ':' is the terminal-current spelling ('xdut.m1:d', pre-330.8) AND
        # the device-parameter spelling ('st1.MN:gm', TASK-330.8) — owner
        # decision 2026-09-28, superseding BOTH the original '[param]'
        # bracket spelling and an interim gate-fix that tried to rescue it:
        # a bracket collided with a legitimate net name ending the same way
        # (a bus bit named 'bus[en]' was wrongly refused as a bad device
        # parameter, a regression). ':' does not collide:
        # it was already reserved for the terminal spelling before 330.8,
        # is never used in a hierarchy path, and a net/path name containing
        # it was already unsupported (raised below) before this feature
        # existed at all. Split on the LAST ':' (a device name carries no
        # character validation, TASK-330.7 finding (b), so nothing rules
        # out an instance literally containing ':'); the suffix, lower-
        # cased, decides which of the two this is — a terminal
        # (d/g/s/b, unchanged) or one of the 17 DEVICE_OP_PARAMS names
        # (this device's designer name as the prefix, never the emitted
        # path the terminal spelling uses). The two sets are disjoint by
        # construction (asserted in tests/test_device_param_probe.py).
        # Anything else is refused HERE, before any deck is even built —
        # not deferred to a read-time raise (reviewer gate finding on
        # 85cd1657, 2026-09-28): an unrecognised name can abort ngspice's
        # whole write, or come back a real, populated, all-zero reading
        # with no warning at all, depending on what else that run happens
        # to save. `type=` is accepted but ignored for a device-parameter
        # probe: which of ngspice's i(@...)/v(@...)/bare @... wrappers
        # applies is a fact about the PARAMETER, resolved when reading it
        # back, never something the caller states.
        if head is None and ':' in raw:
            prefix, _, suffix = raw.rpartition(':')
            prefix = prefix.strip()
            suffix_norm = suffix.strip().lower()
            if suffix_norm in _TERMINAL_PORTS:
                return self._register_probe(
                    Signal._unbound(self._ammeter_vector(raw), 'I', self,
                                    analysis=analysis)
                )
            if suffix_norm in NON_PROBEABLE_OP_COLUMNS:
                raise derived_column_probe_error(
                    f'{prefix}:{suffix_norm}', prefix, suffix_norm)
            from silk.simulation.backend import DEVICE_OP_PARAMS
            if suffix_norm in DEVICE_OP_PARAMS:
                designer_name, param = prefix, suffix_norm
                return self._register_probe(
                    Signal._unbound(f'{designer_name}:{param}', 'V', self,
                                    analysis=analysis)
                )
            raise _unrecognized_probe_colon_suffix_error(raw, prefix, suffix_norm)

        if head == 'i':
            selector = 'I'
            resolved = self._emitted_branch_name(inner)
            if isinstance(resolved, tuple):
                operands = [Signal._unbound(name, 'I', self, analysis=analysis) * sign
                            for name, sign in resolved]
                result = operands[0]
                for operand in operands[1:]:
                    result = result + operand
                result._logical_probe = True
                return self._register_probe(result)
        elif head == 'v':
            if selector == 'I':
                raise ValueError(
                    f"probe({raw!r}, 'I') contradicts itself: the 'v(…)' "
                    f"spelling names a voltage and the selector asks for a "
                    f"current. Write probe({inner!r}) for the voltage, or "
                    f"probe('i({inner})') for the branch current."
                )
            resolved = inner
        else:
            resolved = raw

        # Only a plain VOLTAGE name is checked. A current names an element,
        # not a net, and a path is a run-time fact (see the docstring).
        if selector == 'V':
            from silk.circuit.node_paths import node_request
            if _is_plain_net_name(resolved):
                self._check_probe_net(resolved)
            resolved = node_request(self.netlist, resolved)

        return self._register_probe(
            Signal._unbound(resolved, selector, self, analysis=analysis))

    def bus(self, name):
        """Return a reference to a bus net of this test bench itself.

        The bench-level counterpart of ``stage.bus('d')``: *name* is a bus
        declared on this bench's own netlist (``add_net('d', bits=range(4,8))``).
        Bits are addressed by their declared numbers, so ``ref[7]`` is bit 7
        and ``ref[5:7]`` bits 5 and 6 (slices follow the usual declared-number
        rules: ascending step, stop excluded; ``ref[7:4:-1]`` is 7, 6, 5). Bind it
        with :meth:`probe`; the result is a bus Signal whose arithmetic and
        transforms act element-wise. ``+ - * /`` combine two buses of equal
        length by position (a length mismatch raises) and broadcast a number
        or scalar Signal on either side; ``** number`` applies the exponent to
        every bit. Every maths result is a new bus numbered 0..N-1 by
        position, while ``d`` itself keeps its declared numbers::

            d = tb.probe(tb.bus('d'))             # declared 4..7
            err = d - tb.probe(tb.bus('ref'))     # err[0] = d[4] - ref[first bit]

        Raises
        ------
        CircuitError
            If this bench has no bus *name* (the message lists the buses it
            has).
        """
        from silk.circuit.placement_reference import BenchBusReference
        from silk.exceptions import CircuitError
        buses = getattr(self.netlist, '_buses', None) or {}
        spec = buses.get(name)
        if spec is None:
            raise CircuitError(
                f"testbench {self.test_name!r} has no bus {name!r}. "
                f"Buses declared on it: {_quoted(sorted(buses))}.")
        return BenchBusReference._from_names(self, name, spec.expanded_names(), spec.indices)

    def probe_bus(self, names, type='V', analysis=None):
        """Name several signals of this testbench and return them as one bus Signal.

        Element *i* of the result is ``self.probe(names[i], type=type,
        analysis=analysis)``, so every spelling and every error of
        :meth:`probe` applies per element, unchanged. The result is a **new
        ordered aggregate numbered 0..N-1** in list order; each leaf keeps its
        own probe path (``'d[10]'`` stays the probe of net ``d[10]``). It does
        not rename or redeclare any circuit bus. Its arithmetic and
        transforms act per element and pair equal-length buses by position
        (see docs/probes.md, "Bus maths")::

            d = tb.probe_bus(['d[3]', 'd[2]', 'd[1]', 'd[0]'])
            mags = d.mag()                 # a bus of four magnitudes
            d[0].name                      # 'd[3]': the first listed

        Parameters
        ----------
        names : list or tuple of str
            At least one signal path, in the bit order wanted. Repeats are
            allowed here.
        type, analysis
            As for :meth:`probe`, applied to every element.

        Raises
        ------
        TypeError
            If *names* is a bare string (use :meth:`probe` for one signal), is
            not a list or tuple, or has an element that is not a string.
        ConfigurationError
            If *names* is empty (``[]`` and ``()`` alike).
        """
        from silk.exceptions import ConfigurationError
        from silk.signals.base import Signal
        if isinstance(names, str):
            raise TypeError(
                f"probe_bus() takes a list or tuple of names, not the string "
                f"{names!r}; use probe({names!r}) for one signal")
        if not isinstance(names, (list, tuple)):
            raise TypeError(
                f"probe_bus() takes a list or tuple of names, got "
                f"{_type_name(names)}")
        if not names:
            raise ConfigurationError('probe_bus() needs at least one name')
        for index, name in enumerate(names):
            if not isinstance(name, str):
                raise TypeError(
                    f"probe_bus() names[{index}] must be a string, got "
                    f"{_type_name(name)}")
        leaves = tuple(self.probe(name, type=type, analysis=analysis) for name in names)
        return Signal._make_bus_handle(f"bus({','.join(names)})", leaves)

    def _append_placement_declaration(self, name, value):
        self.__dict__.setdefault(name, []).append(value)

    def _register_probe(self, sig):
        """Append *sig* (a requested probe or composite) to :attr:`_probe_registry` and
        return it. The single funnel every ``probe()`` return path uses, so
        ``tb.plot()`` with no arguments has a complete list to walk.

        The reference is STRONG: *sig* carries no data (a name, a
        selector, a weak reference back to this bench) and is never itself
        the target of a strong reference from this bench elsewhere, so
        holding it here costs nothing and keeps a probe minted only for
        plotting alive until ``tb.plot()`` actually runs.
        """
        self._append_placement_declaration('_probe_registry', sig)
        return sig

    # -- probe name resolution -----------------------------------------

    def _check_probe_net(self, net_name):
        """Refuse a plain net name this testbench's netlist does not have.

        Silent when there is nothing to check against: no netlist yet (the
        netlist is created lazily by the first ``add_*`` call), or a netlist
        with no nets wired yet. Matching is case-insensitive because SPICE
        is, and the rawfile is lower-cased.
        """
        netlist = self.netlist
        if netlist is None:
            return
        nets = getattr(netlist, 'net_dict', None) or {}
        if not nets:
            return
        if net_name.lower() in {str(n).lower() for n in nets}:
            return
        if net_name.lower().startswith('__silk_'):
            # Generated nodes are absent from authored net_dict, but the
            # rawfile viewer must be able to copy a probe for their vectors.
            from silk.circuit.modules_netlist import _save_list_inventory
            from silk.spice_names import to_spice_name
            physical = {to_spice_name(n, kind='node').lower()
                        for n in _save_list_inventory(netlist)}
            if to_spice_name(net_name, kind='node').lower() in physical:
                return
        from silk.circuit.node_paths import dut_hint, node_candidates
        hint = dut_hint(node_candidates(netlist), net_name)
        raise ValueError(
            f"probe({net_name!r}): testbench {self.test_name!r} has no net "
            f"{net_name!r}. Available nets: {_quoted(sorted(nets))}. "
            f"(A hierarchical path such as '{DUT_ELEMENT_NAME_PREFIX}.{net_name}' is not checked "
            f"here — pass the full SPICE path for a node inside the DUT.){hint}"
        )

    def _emitted_branch_name(self, inner):
        """Map the name inside ``i(…)`` to a rawfile vector name.

        Delegates to :func:`_branch_vector_name`, the lookup a metric's
        string path shares: a source NAME maps to its emitted instance, any
        other name is a raw vector verbatim, and a name that is both raises
        :class:`~silk.exceptions.AmbiguousBranchCurrentError`.
        """
        netlist = self.netlist
        sources = {} if netlist is None else (
            getattr(netlist, 'sig_sources_dict', None) or {})
        return _branch_vector_name(sources, inner, getattr(netlist, 'elem_dict', None))

    def _ammeter_vector(self, spelling):
        """``'xdut.m1:d'`` -> ``'v.xdut.xmm1.vamm_d'``.

        Builds the same string
        :meth:`silk.simulation.backend.NgspiceBackend.terminal_current_vector`
        looks for, minus its rawfile-presence check (there is no rawfile at
        declaration time). The three conditions that put that vector in the
        file are checked here instead, so the refusal lands at the call that
        made the mistake rather than as a missing vector after a simulation.
        """
        path, _, port = spelling.rpartition(':')
        port = port.strip().lower()
        path = path.strip()
        if not path or port not in _TERMINAL_PORTS:
            raise ValueError(
                f"probe({spelling!r}): the terminal-current spelling is "
                f"'<instance path>.<device>:<terminal>', e.g. '{DUT_ELEMENT_NAME_PREFIX}.m1:d'; "
                f"the terminal must be one of "
                f"{_quoted(_TERMINAL_PORTS)}."
            )

        self._require_ammeter_currents(spelling)

        instance_path, _, device_name = path.rpartition('.')
        try:
            leaf, emitted_device, instance_path = self._emitted_device_entry(
                spelling, instance_path, device_name)
            emitted_error = None
        except ValueError as exc:
            emitted_error = exc

        # TASK-342: the prefix may also be a DESIGNER name ('st1.MN', the
        # spelling ``'st1.MN:gm'`` uses), resolved against the DUT by the
        # same lookup. The emitted-path reading is tried FIRST and wins, so
        # every string that meant something before this task means the same
        # thing now.
        designer = find_designer_device(self.module, path)
        if emitted_error is None:
            shadow_instance = None
            if designer is not None:
                shadow_instance = DUT_ELEMENT_NAME_PREFIX + ('.' + designer[0] if designer[0] else '')
            if (designer is not None and emitted_device is not None
                    and (designer[2] is not emitted_device
                         or shadow_instance.casefold() != instance_path.casefold())):
                shadow_path, shadow_name, _dev = designer
                shadowed = (f'{DUT_ELEMENT_NAME_PREFIX}.{shadow_path}.{shadow_name}'
                            if shadow_path
                            else f'{DUT_ELEMENT_NAME_PREFIX}.{shadow_name}')
                warnings.warn(
                    f"probe({spelling!r}): {path!r} names two different "
                    f"devices — the testbench's own emitted-path device "
                    f"{path!r} and the DUT's designer-named device "
                    f"{path!r}. The emitted-path device is used; probe "
                    f"{shadowed + ':' + port!r} to select the DUT device.",
                    UserWarning, stacklevel=3)
            prefix = f'{instance_path.lower()}.' if instance_path else ''
            return f'v.{prefix}{leaf.lower()}.vamm_{port}'

        if designer is None:
            from silk.circuit.modules_results import _known_designer_names
            import difflib
            names = _known_designer_names(self.module)
            close = difflib.get_close_matches(path, names, n=3, cutoff=0.6)
            raise ValueError(
                f"{emitted_error} Read as a designer name, {path!r} matches "
                f"no device of the DUT either "
                f"({'close matches' if close else 'known device names'}: "
                f"{', '.join(close or sorted(names)) or '(none)'}).")
        from silk.circuit.emitted_names import emitted_instance_name
        from silk.circuit.modules_results import (
            _device_pin_names, _is_ideal_passive, _MOS_TERMINALS)
        d_path, d_name, d_device = designer
        if _is_ideal_passive(d_device) or not (
                _device_pin_names(d_device) >= _MOS_TERMINALS):
            raise ValueError(
                f"probe({spelling!r}): {path!r} is not a MOS-shaped device "
                f"— terminal currents through the 0 V ammeter wrappers "
                f"exist only for 4-terminal MOS instances.")
        d_leaf = emitted_instance_name(d_device, d_name)
        d_prefix = (f'{DUT_ELEMENT_NAME_PREFIX}.{d_path}' if d_path
                    else DUT_ELEMENT_NAME_PREFIX)
        return f'v.{d_prefix.lower()}.{d_leaf.lower()}.vamm_{port}'

    def _require_ammeter_currents(self, spelling):
        """Refuse a terminal-current probe when the wrappers will not be emitted.

        Device terminal currents exist only in ammeter-wrapper mode, which is
        three opt-ins: ``save_device_currents`` on the device under test (the
        flag ``run()`` copies onto the executing netlist),
        ``terminal_current_method = 'ammeter_wrapper'``, and a Transient
        analysis — ``save_device_currents`` refuses anything else at netlist
        generation. Each missing piece is named, with the whole recipe, so
        the fix is one edit away.
        """
        dut = self.module
        missing = []
        if not getattr(dut, 'save_device_currents', False):
            missing.append(
                f"save_device_currents is "
                f"{getattr(dut, 'save_device_currents', False)!r} on "
                f"{dut.module_name!r}")
        method = getattr(dut, 'terminal_current_method', None)
        if method != 'ammeter_wrapper':
            missing.append(
                f"terminal_current_method is {method!r} (the default "
                f"'savecurrents' mode reads the drain only, through a "
                f"different vector)")
        analyses = {} if self.netlist is None else (
            getattr(self.netlist, 'sim_analysis_dict', None) or {})
        if 'transient' not in analyses:
            missing.append(
                f"this test declares {_quoted(sorted(analyses)) or 'no analysis'}, "
                f"not a Transient")
        if not missing:
            return
        raise ValueError(
            f"probe({spelling!r}) reads a device terminal current. The 0 V "
            f"ammeter wrappers that carry it are emitted only when all three "
            f"of these hold: save_device_currents = True on the device under "
            f"test, terminal_current_method = 'ammeter_wrapper' on it, and a "
            f"Transient analysis on this test (save_device_currents supports "
            f"exactly one). Missing here: " + '; '.join(missing) + '.'
        )

    def _emitted_device_leaf(self, spelling, instance_path, device_name):
        """The emitted SPICE instance name of the device a terminal probe names."""
        return self._emitted_device_entry(spelling, instance_path,
                                          device_name)[0]

    def _emitted_device_entry(self, spelling, instance_path, device_name):
        """``(emitted leaf, device object, emitted instance path)`` for the
        emitted-path reading of a terminal probe's prefix (device is ``None``
        before any netlist exists). The instance path is rebuilt from the
        names the walk matched, in the netlist's own ``x``-prefixed spelling.

        ``'m1'`` inside instance path ``'xdut'`` is emitted as ``'XMm1'``, so
        the vector is ``v.xdut.xmm1.vamm_d``. The name comes from
        :func:`~silk.circuit.emitted_names.emitted_instance_name`, the single
        source of truth the netlister and current annotation already share —
        never re-derived here.

        The walk is authoritative when there is a netlist to walk: an
        instance path component is the emitted ``'x'``-prefixed name of a
        submodule, and the last component is the device's silk name, matched
        case-insensitively (SPICE is). A device this testbench does not
        contain is refused — unlike a net path, a device instance is part of
        silk's own structure, and probe() would otherwise have to invent a
        vector name. Before any netlist exists there is nothing to walk, so
        the PDK's default device prefix is used and the run reports the miss.
        """
        from silk.circuit.emitted_names import emitted_instance_name
        from silk.circuit.ports import ElementType
        from silk.pdk.basepdk import DEFAULT_PDK_SPICE_PREFIX

        module = self.netlist
        if module is None:
            return DEFAULT_PDK_SPICE_PREFIX + device_name, None, instance_path

        walked = ''
        emitted_parts = []
        for component in [c for c in instance_path.split('.') if c]:
            inst = component[1:] if component[:1].lower() == 'x' else component
            entry = _element_entry(module, inst)
            if entry is None or entry[1].get('type') != ElementType.MODULE:
                raise ValueError(
                    f"probe({spelling!r}): {component!r} is not a submodule "
                    f"instance of {walked or 'the testbench'}. Instances "
                    f"there: {_quoted(_instance_names(module, ElementType.MODULE))}."
                )
            module = entry[1]['device']
            # the netlist's OWN emitted spelling of what matched, not the
            # typed component: 'st1' / 'XST1' / 'xst1' all read 'xst1'
            emitted_parts.append(emitted_instance_name(module, entry[0]))
            walked = f'{walked}.{component}' if walked else component

        entry = _element_entry(module, device_name)
        if entry is None:
            raise ValueError(
                f"probe({spelling!r}): {device_name!r} is not a device of "
                f"{walked or 'the testbench'}. Devices there: "
                f"{_quoted(_instance_names(module, ElementType.DEVICE))}."
            )
        key, record = entry
        return (emitted_instance_name(record['device'], key), record['device'],
                '.'.join(emitted_parts))

    # -- plotting ---------------------------------------------------------

    def _live_probes(self):
        """Every probe this bench has minted, in creation order.

        :attr:`_probe_registry` holds a STRONG reference to each one, so every entry is always "live" — a probe minted only
        to be plotted, with no variable of its own, is no longer collected
        before this call sees it. The name is kept (as is the copy made
        here, matching the old contract) for the callers below that don't
        want to observe later ``probe()`` calls through a shared list.
        """
        return list(self._probe_registry)

    def plot(self, *signals, overlay=False, show=True, corners=None):
        """Plot probes and derived Signals of this bench, after it has run.

        Draws bound data by delegating to
        :meth:`plot <silk.circuit.Module.plot>`'s auto-scaled multi-panel
        (or ``overlay=True`` single-axis) layout — the same drawing this
        testbench's netlist already does for plain signal names, reused here
        for Signal objects instead. Each argument is a ``Signal``: a probe
        from :meth:`probe`, or an expression built from one by arithmetic or
        a transform (``vp + vm``, ``vd.derivative()``, ...). Every Signal is
        resolved **exactly once** — :meth:`resolve <silk.signals.Signal.resolve>`
        is called on it directly, and the returned bound Signal's ``x``/
        ``y``/``xlabel``/``ylabel`` are read from that result, never by
        re-reading the original attribute (which would resolve again).

        With no arguments, plots every probe :meth:`probe` has created on
        this bench (see ``_probe_registry``) — a Signal built purely by
        arithmetic/transforms on a probe, with no ``tb.probe(...)`` call of
        its own, is not itself in the registry and is not drawn by the
        no-argument form; pass it explicitly.

        Parameters
        ----------
        *signals : Signal
            Probes or derived Signals of this bench. Omit to plot every
            probe created by :meth:`probe` so far.
        overlay : bool, optional
            One shared axis instead of one panel per signal. Forwarded to
            :meth:`Module.plot`.
        show : bool, optional
            Call ``plt.show()`` after plotting (default ``True``). Forwarded
            to :meth:`Module.plot`.
        corners : list of str, optional
            Which retained corners to draw. Omitted is the useful
            default: after ``run_corners([...])`` in an output mode that kept
            a file per run, each signal's panel carries one labelled trace
            per retained corner with a legend, and after a plain ``run()``
            the drawing is exactly what it was before. Name a subset to draw
            only those; an unknown name raises, listing what was retained
            (see :attr:`sweep_corners`). Default overlays also retain full
            grid point and Monte Carlo sample keys as trace labels. Each
            trace resolves separately, leaving the bench's live results
            untouched.

        Returns
        -------
        matplotlib.figure.Figure

        Raises
        ------
        UnboundSignalError
            If this bench has not run yet (no netlist, or no results
            loaded) — raised before anything is drawn, whether or not
            *signals* is empty.
        ValueError
            Called with no arguments on a bench that has never called
            :meth:`probe`.
        TypeError
            An argument is not a ``Signal``.

        Examples
        --------
        ::

            vp, vm = tb.probe('vp'), tb.probe('vm')
            vcm = (vp + vm) / 2
            tb.run()
            tb.plot(vp, vm, vcm)   # three panels
            tb.plot()              # every probe of this bench: vp, vm

            tb.run_corners(['tt', 'ff', 'ss'])
            tb.plot(vp, vm)                    # 2 panels x 3 corner traces
            tb.plot(vp, corners=['tt', 'ss'])  # 1 panel, 2 corner traces
        """
        from silk.signals.base import Signal

        netlist = self.netlist
        if netlist is None or not getattr(netlist, 'sim_results_loaded', False):
            raise UnboundSignalError(
                f"tb.plot(): testbench {self.test_name!r} has not run yet; "
                f"call tb.run() before tb.plot()"
            )

        if not signals:
            signals = self._live_probes()
            if not signals:
                raise ValueError(
                    f"tb.plot(): testbench {self.test_name!r} has no probes "
                    f"to plot; call tb.probe(...) first, or pass Signals "
                    f"explicitly, e.g. tb.plot(vp, vm)."
                )
        else:
            for sig in signals:
                if not isinstance(sig, Signal):
                    raise TypeError(
                        f"tb.plot() takes Signal objects returned by "
                        f"tb.probe() or built from one by arithmetic or a "
                        f"transform; got {sig!r}"
                    )

        from silk.circuit.test_placement import placement_plot_traces
        traces = placement_plot_traces(self, signals, corners)
        if traces is not None:
            from silk.circuit.modules_results import _plot_corner_groups
            return _plot_corner_groups(netlist,
                                       [(sig.name, runs) for sig, runs in zip(signals, traces)],
                                       overlay=overlay, show=show)

        draw_corners = self._corners_to_draw(corners)
        if draw_corners is not None:
            for sig in signals:
                if not sig._is_unbound():
                    raise ValueError(
                        f"Signal {sig.name!r} already carries data, so there "
                        "is no retained run to select; pass a deferred probe "
                        "or plot the bound Signal directly."
                    )
            # One panel per signal, one trace per corner (TASK-218). Each
            # Signal is resolved once per corner, against that corner's file;
            # the live results are not read at all.
            groups = [
                (sig.name, [(str(c), sig.resolve(corner=c, _readers={}))
                            for c in draw_corners])
                for sig in signals
            ]
            from silk.circuit.modules_results import _plot_corner_groups
            return _plot_corner_groups(netlist, groups, overlay=overlay,
                                       show=show)

        named = [(sig.name, sig.resolve()) for sig in signals]
        from silk.circuit.modules_results import _plot_loaded_signals
        return _plot_loaded_signals(netlist, named, overlay=overlay, show=show)

    def add_metric(self, name, extractor_func=None, *extra_positional_signals,
                   type=None, signal=None, signals=None, analysis=None,
                   **params):
        """Add a metric to extract after simulation. Four forms:

        ::

          derived:     add_metric('ratio', numerator / denominator)
          function:    add_metric('g', dc_gain_dB, sig=vout_probe, **params)
          declarative: add_metric('g', type='dc_gain_dB', signal='xdut.vout', **params)
          callable:    add_metric('g', lambda ckt: dc_gain_dB(ckt.load_signal('xdut.vout')))

        The **function form** is the taught path
        (``docs/probes.md``): ``probe`` → ``add_metric(fn,
        sig=...)`` → ``run`` → ``vcm.plot()`` — name what you want to
        measure with :meth:`probe`, build a derived Signal from it with
        ordinary arithmetic or a transform if the metric is over something
        other than a raw probe, hand the ``@metric``-registered function and
        that Signal to ``add_metric``, run, and plot::

            vp, vm = tb.probe('vp'), tb.probe('vm')
            vcm = (vp + vm) / 2
            tb.add_metric('vcm_avg', average, sig=vcm)
            tb.run()
            vcm.plot()

        A derived-signal metric — ``sig=`` bound to an expression built from
        probes rather than a single one — is a Python-only capability:
        codegen still emits the declarative ``type=``/``signal=`` form,
        which is unchanged. Underneath every form, the primitive that
        actually reads a signal off the run is
        :meth:`load_signal <silk.circuit.Module.load_signal>`; ``probe``/
        ``add_metric``/``resolve()`` are all built on it, and it stays public
        for scripts that want data directly after a run — as
        ``tb.netlist.load_signal(name)`` while the bench's test object is in
        scope, or ``report.sim_results.load_signal(name)`` from a
        :class:`~silk.circuit.spec_report.SpecReport` returned by :meth:`run`.

        The declarative form binds a metric's signal inputs by NAME. One input
        is ``signal='xdut.vout'``; N inputs are ``signals={slot: path, ...}``,
        keyed by the metric's own signal-input parameter names as published in
        ``export_metric_schema()[type]['signal_inputs']``::

            tb.add_metric('cmrr', type='cmrr_dB',
                          signals={'vdiff': 'xdut.voutd', 'vcm': 'xdut.voutc'},
                          freq=1e3)
            tb.add_metric('tpd', type='propagation_delay',
                          signals={'sig_in': 'xdut.vin', 'sig_out': 'xdut.vout'},
                          in_edge='rising')

        A multi-input function-form metric binds every signal slot by its own
        keyword name the same way (``docs/probes.md``)::

            tb.add_metric('p_avg', average_power, sig_v=tb.probe('vout'),
                          sig_i=tb.probe('i(vs)'))

        A slot value may be a signal path (a ``str``, resolved through
        ``ckt.load_signal`` exactly as the declarative form does) or a
        ``Signal`` — bound or unbound (a probe, or an expression built on
        probes); an unbound one is resolved at evaluation time. ``fn`` is
        recognised as a registered metric only by the ``@metric`` decoration
        marker, never by its name, so a plain function called ``average`` is
        not mistaken for the registry one.

        A **plain function** takes probes too, with no ``@metric`` and no
        registration — a designer's one-off metric::

            def duty(sig, level=0.5):
                return float(np.mean(sig.y > level * np.max(sig.y)))

            tb.add_metric('duty_up', duty, sig=tb.probe('up'), level=0.5)
            tb.add_metric('gm_over_id', lambda gm, i: float(gm.y[0] / i.y[0]),
                          gm=tb.probe('st1.MN:gm'), i=tb.probe('st1.MN:id'))

        The keyword names are checked against the function's signature here,
        when the metric is declared. A keyword whose value is a ``Signal`` (a
        probe, an expression over probes, or a Signal with data) is a signal
        slot; every other value is a scalar passed through as is — a ``str``
        included, which in this form is never a signal path. Nothing reaches
        the metric catalog or ``export_metric_schema()``. With no metadata,
        a slot has no kind unless the function annotates it
        ``Annotated[Signal, KIND]``, so in a test with more than one analysis
        each probe must name its own (``tb.probe('vout', analysis='tran')``)
        or the metric is refused at run time — silk never guesses a plot. An
        undecorated function given NO arguments is still the callable form
        below. A bench carrying one is pickled (a ``FaultCampaign``) only if
        the function is defined at module level: a lambda or a nested def
        raises ``PicklingError`` naming the metric.

        By name and never by position, in both the declarative and function
        forms, because ``hysteresis``'s four slots would otherwise be a
        guessing game whose failure mode is a plausible wrong number rather
        than an error. ``signals={'sig': path}`` on a one-input metric is
        simply the explicit spelling of ``signal=``.

        The binding is checked HERE, when the metric is declared — a missing
        or misspelled slot raises immediately instead of costing a simulation
        run.

        Parameters
        ----------
        name : str
            Name of the metric
        extractor_func : callable or MetricHandle, optional
            Either a plain function that takes sim_results
            (Module) and returns the metric value — e.g.
            ``lambda nr: np.mean(nr.load_signal('vout').y)`` —
            or a function — ``@metric``-registered or plain —
            bound via the keyword slots described above. Mutually
            exclusive with type=.
        type : str, optional
            Name of a metric registered via silk.signals.schema.get_metric
            (e.g. 'dc_gain_dB'). Mutually exclusive with extractor_func.
        signal : str, optional
            Signal path passed to ckt.load_signal(), for a metric with
            exactly one signal input. Mutually exclusive with signals=.
        signals : dict, optional
            ``{slot_name: signal path}`` binding every signal input of
            the metric. Required for the multi-input metrics; accepted
            for a one-input metric as the explicit form of signal=.
        analysis : str, optional
            ``'tran'``, ``'ac'``, ``'dc'`` or ``'noise'`` — WHICH
            analysis's plot this metric reads, in a test that declares
            more than one. Declarative form only. Omit it
            whenever the test declares one analysis, or when only one of
            the declared analyses can produce this metric's inputs. The
            function form says the same thing on the probe,
            ``tb.probe('vout', analysis='tran')``.
        **params
            For the declarative (type=) form, extra scalar params
            forwarded to the resolved metric function. For the function
            form, a mix of signal-input slots (by the metric's own slot
            names) and scalar params, forwarded the same way.

        Returns
        -------
        MetricHandle
            Deferred result with arithmetic and .value; public
            testbench methods are forwarded for builder chaining.

        Notes
        -----
        **Choosing the analysis (a test with more than one)**

        A test that declares two analyses writes two plots, and **silk never
        guesses which one a metric meant**. The rule, the same in
        every form:

        * one declared analysis — nothing to say, exactly as before;
        * only one of the declared analyses can produce this metric's inputs
          — read it, no field. ``dc_gain_dB`` in a tran+ac test reads the AC
          plot because a ``frequency_response`` comes from ``.ac`` and from
          nothing else in the test;
        * more than one could — the analysis is REQUIRED, and its absence
          raises naming the candidates. ``average`` takes a Signal of kind
          ``any``, so in a tran+ac test both plots could feed it.

        Each form says it in its own place — the metric, the probe, the
        loader::

            tb.add_metric('vavg', type='average', signal='vout',
                          analysis='tran')                       # declarative
            tb.add_metric('vavg', average,
                          sig=tb.probe('vout', analysis='tran')) # function
            tb.add_metric('vavg', lambda ckt: average(
                ckt.load_signal('vout', analysis='tran')))       # callable

        Raises
        ------
        TypeError
            If neither or both of extractor_func/type are given; if
            type= is given without signal=/signals=; if the
            binding is incomplete, names an unknown slot, or uses
            signal= for a metric that takes more than one input; if
            a signal slot is passed positionally to the function
            form instead of by keyword; if a plain (undecorated)
            ``extractor_func`` is given a keyword its signature
            does not have, is missing a parameter with no default,
            or is given no Signal at all; or if an ``@metric``
            function passed as ``extractor_func`` has no
            signal-input slots to bind, or is missing one; or if
            analysis= is passed with the function or callable form
            (name it on the probe, or on ``ckt.load_signal``,
            there instead).
        ValueError
            Immediately if this testbench already has a metric with the
            exact same name, even for an identical recipe. Use distinct
            names for distinct measurements; independent testbenches may
            reuse names.
            At run time, if analysis= is absent where the test's
            declared analyses make the choice ambiguous, or names
            an analysis this test does not declare or that cannot
            feed the metric's inputs. Checked before extraction, so
            it raises out of ``run()``/``evaluate()`` rather than
            landing in the report as a string. The function form
            raises ``MetricBindingError`` for the same three cases,
            with the same wording.
        """
        if any(metric['name'] == name for metric in self._metrics):
            raise ValueError(
                f"Testbench {self.test_name!r} already has metric {name!r}; "
                "use a distinct name for each measurement.")
        if isinstance(extractor_func, MetricHandle):
            if (type is not None or signal is not None or signals is not None
                    or analysis is not None or extra_positional_signals or params):
                raise TypeError('add_metric: a derived expression takes no signal or metric arguments.')
            if not extractor_func._belongs_to(self):
                raise ValueError('add_metric: derived metrics must belong to this testbench.')
            self._append_placement_declaration('_metrics', {'name': name, 'expression': extractor_func._expression})
            return MetricHandle(self, ('metric', name))

        if (extractor_func is None) == (type is None):
            raise TypeError("add_metric: pass exactly one of extractor_func or type=")
        if type is None and analysis is not None:
            raise TypeError(
                f"add_metric({name!r}): analysis= applies to the declarative "
                f"form (type=...). The function form names the analysis on "
                f"the probe: tb.probe(<net>, analysis={analysis!r}); the "
                f"callable form names it where it loads the signal: "
                f"ckt.load_signal(<path>, analysis={analysis!r}).")
        if type is not None:
            from silk.signals.schema import get_metric
            fn = get_metric(type)
            bound = _bind_signal_inputs(name, type, fn, signal, signals, params)
            slots = _metric_signal_slots(fn)
            where = f"add_metric({name!r}, type={type!r})"
            if analysis is not None and analysis not in _SIM_KEY_ANALYSIS.values():
                raise ValueError(
                    f"{where}: analysis={analysis!r} is not a silk analysis; "
                    f"expected one of "
                    f"{sorted(set(_SIM_KEY_ANALYSIS.values()))}.")

            # Resolved per evaluation, not here: a hand-built test may call
            # add_sim_type AFTER its metrics, so the declared analyses are not
            # final at declaration time. evaluate() runs the same resolution
            # once, before extraction, so a refusal raises instead of being
            # caught into an 'Error: ...' string by the extraction loop.
            def extractor_func(ckt, _fn=fn, _bound=bound, _params=params,
                               _slots=slots, _where=where, _analysis=analysis,
                               _name=name, _type=type):
                from silk.exceptions import MetricBindingError
                from silk.signals.kinds import SignalKindError
                plot = _resolve_metric_analysis(
                    _where, _slots, _declared_analyses(ckt), _analysis,
                    signals=list(_bound.values()))
                # The keyword is passed only when there IS a selection to make
                # — a single-analysis test calls load_signal exactly as it
                # always did, down to the argument list.
                select = {} if plot is None else {'analysis': plot}
                # Bound by keyword — safe because no metric in the catalog has
                # a positional-only parameter, and the only ordering a caller
                # could get wrong this way is one they never have to state.
                loaded = {slot: _load_metric_path(ckt, path, **select)
                          for slot, path in _bound.items()}
                try:
                    return _fn(**loaded, **_params)
                except SignalKindError as exc:
                    # Same promotion as the function form (TASK-215.7):
                    # The two forms bind the same way, so an
                    # AC metric handed a transient trace must raise from
                    # run() in both instead of landing as a report string.
                    raise MetricBindingError(
                        f"metric {_name!r} (type={_type!r}): the signal "
                        f"bound to this metric is the wrong kind — {exc}"
                    ) from exc

            self._append_placement_declaration('_metrics', {
                'name': name,
                'extractor': extractor_func,
                'type': type,
                'slots': slots,
                'analysis': analysis,
                'signals': list(bound.values()),
                'where': where,
            })
            return MetricHandle(self, ('metric', name))

        fn = extractor_func
        fn_name = getattr(fn, '__name__', repr(fn))
        # inspect.getattr_static, not getattr: a plain getattr() on a
        # test-double callable (e.g. unittest.mock.Mock, used throughout the
        # suite as an extractor(ckt) stand-in) auto-vivifies any
        # non-dunder attribute, including '_silk_metric', which would
        # misdetect an ordinary Mock as a registered metric. getattr_static
        # reads the real attribute (or class attribute) without triggering
        # __getattr__/descriptors, so only an actually-decorated function
        # (whose @metric call did fn._silk_metric = {...}) matches.
        marker = inspect.getattr_static(fn, '_silk_metric', None)

        if marker is None:
            if not (extra_positional_signals or params):
                # Undecorated, no arguments: today's extractor(ckt) form — a
                # plain function (any name, including one that happens to be
                # called 'average') called with the single sim-results
                # argument.
                self._append_placement_declaration('_metrics', {'name': name, 'extractor': fn})
                return MetricHandle(self, ('metric', name))
            # Undecorated WITH arguments: the plain-function form (TASK-323).
            # A designer's one-off function takes probes without being
            # registered; slots come from its signature and the values bound
            # to it, and nothing reaches the exported metric schema.
            slots, fn_params, slot_kinds = _bind_plain_function(
                name, fn, extra_positional_signals, params)
            self._refuse_cross_bench_probes(name, fn_name, slots)
            metric = {
                'name': name,
                'extractor': None,
                'fn': fn,
                'slots': slots,
                'params': fn_params,
                'slot_kinds': slot_kinds,
            }
            metric['extractor'] = _function_form_extractor(metric)
            self._append_placement_declaration('_metrics', metric)
            return MetricHandle(self, ('metric', name))

        # Decorated: the function form; see docs/probes.md for registration usage.
        slots = _metric_signal_slots(fn)
        slot_names = [n for n, _ in slots]

        if extra_positional_signals:
            n_extra = len(extra_positional_signals)
            if slot_names:
                call = ', '.join(f'{n}=...' for n in slot_names)
                detail = (
                    f"{fn_name}'s signal-input slots are {_quoted(slot_names)}; "
                    f"bind them as add_metric({name!r}, {fn_name}, {call})."
                )
            else:
                detail = f"{fn_name} has no signal-input slots to bind."
            raise TypeError(
                f"add_metric({name!r}, {fn_name}): signal slots are bound by "
                f"keyword name, not position ({n_extra} extra positional "
                f"argument{'s' if n_extra != 1 else ''} given). {detail}"
            )

        if not slot_names:
            raise TypeError(
                f"add_metric({name!r}, {fn_name}): {fn_name} is a "
                f"@metric-decorated function with no signal-input slots "
                f"(signal_inputs=[]); the function form binds Signal "
                f"arguments by keyword name, so there is nothing here to "
                f"bind."
            )

        from silk.signals.schema import metric_schema
        param_names = {p['name'] for p in metric_schema(fn)['parameters']}

        slot_kwargs = {k: v for k, v in params.items() if k in slot_names}
        fn_params = {k: v for k, v in params.items() if k not in slot_names}

        unknown = [k for k in fn_params if k not in param_names]
        if unknown:
            raise TypeError(
                f"add_metric({name!r}, {fn_name}): unknown keyword "
                f"argument{'s' if len(unknown) != 1 else ''} {_quoted(unknown)}; "
                f"{fn_name}'s signal-input slots are {_quoted(slot_names)} "
                f"and its scalar parameters are "
                f"{_quoted(sorted(param_names)) if param_names else 'none'}."
            )

        missing = [n for n in slot_names if n not in slot_kwargs]
        if missing:
            raise TypeError(
                f"add_metric({name!r}, {fn_name}): missing signal-input "
                f"slot{'s' if len(missing) != 1 else ''} {_quoted(missing)}. "
                f"{fn_name}'s signal-input slots are {_slot_inventory(slots)}; "
                f"bound: {_quoted([n for n in slot_names if n in slot_kwargs]) or 'nothing'}."
            )

        self._refuse_cross_bench_probes(name, fn_name, slot_kwargs)

        metric = {
            'name': name,
            'extractor': None,      # assigned just below, over this dict
            'fn': fn,
            # Raw, unresolved slot values — inspectable, not a closure over.
            'slots': dict(slot_kwargs),
            'params': dict(fn_params),
        }
        # The extractor is a thin closure over the metric dict: all the
        # resolution, and the MetricBindingError wrapping around it, live in
        # the module-level _evaluate_function_form_metric (TASK-215.7), so
        # the evaluation loop and a direct extractor call take exactly the
        # same path. Nothing about the dict's shape changed.
        metric['extractor'] = _function_form_extractor(metric)
        self._append_placement_declaration('_metrics', metric)
        return MetricHandle(self, ('metric', name))

    def _refuse_cross_bench_probes(self, name, fn_name, slots):
        """One bench per recipe (the contract in docs/probes.md), for both function forms.

        Every unbound leaf reachable through a slot value must be a probe of
        THIS bench: the recursive test runner walks benches in insertion
        order, never dependency order, so a metric here over another bench's
        probe would raise mid-run and abort the whole tree. Comparing two
        benches' outputs is a post-run script over the bound Signals.
        """
        for slot, value in slots.items():
            for leaf in _unbound_leaves(value):
                other = leaf._bench_or_none()
                if other is None or other is self:
                    continue
                from silk.circuit.test_placement import SelectedTestbench
                selection = current_placement(self)
                if (isinstance(other, SelectedTestbench) and selection is not None
                        and selection.entry is not None):
                    owner = other._entry()
                    if (other._authored is self and owner._root_ref() is selection.entry._root_ref()
                            and owner.owner_path == selection.owner_path):
                        continue
                raise TypeError(
                    f"add_metric({name!r}, {fn_name}): {slot}="
                    f"{_describe_slot_value(value)} reaches probe "
                    f"{leaf.name!r}, which belongs to bench "
                    f"{_bench_label(other)}, but this metric is declared on "
                    f"bench {_bench_label(self)}. Every probe in a metric's "
                    f"recipe must come from the bench that runs it. Use an "
                    f"authored/local probe or a declarative signal name to "
                    f"measure each placement's own node; to "
                    f"compare two benches, run both and do the arithmetic "
                    f"on the bound Signals afterwards."
                )

    def add_spec(
        self, name: str, metric_name, op: str, target: float, *, key=None,
    ):
        """Declare a pass/fail specification evaluated after run().

        Creates a first-class :class:`~silk.specs.Spec` on the underlying
        :class:`~silk.circuit.modules.Module` via ``declare_spec``, and
        also tracks it in this testbench's legacy ``_specs`` list so
        :class:`~silk.circuit.spec_report.SpecReport` construction
        continues unchanged.

        Parameters
        ----------
        name : str
            Human-readable spec label, e.g. 'DC gain > 60 dB'.
        metric_name : str or MetricHandle
            A metric name added via add_metric(), or the handle it returned.
        op : str
            Comparison operator: '>', '>=', '<', '<=', or '=='.
        target : float
            Target value.
        key : str, optional
            Stable local identity. When omitted, the exact ``name`` is used.

        Returns
        -------
        Spec
            The newly-created Spec object. The legacy chaining pattern
            ``tb.add_spec(...).add_spec(...)`` is preserved by a
            forwarder on Spec that delegates back to this testbench.

        Examples
        --------
        ::

            tb.add_metric('gain_dB', lambda ckt: dc_gain_dB(ckt.load_signal('vout')))
            spec = tb.add_spec('DC gain > 60 dB', 'gain_dB', '>', 60.0)
            report = tb.run()
            print(spec.latest().value)
        """
        if isinstance(metric_name, MetricHandle):
            if not metric_name._belongs_to(self):
                raise ValueError(
                    f'add_spec({name!r}): the metric handle belongs to a '
                    f'different testbench.')
            metric_name = metric_name.name
        elif not isinstance(metric_name, str):
            raise TypeError(
                f'add_spec({name!r}): metric must be a metric name (str) or '
                f'a MetricHandle from add_metric, got '
                f'{type(metric_name).__name__}.')
        spec = self.module.declare_spec(
            name=name, metric=metric_name, op=op, target=target, key=key,
        )
        # Back-reference so Spec.add_spec can forward to this testbench.
        spec._owning_testbench = self
        # Keep the legacy dict in self._specs so SpecReport construction
        # continues to work; add a 'spec' key for new consumers.
        self._specs.append({
            'name': name, 'metric': metric_name, 'op': op, 'target': target,
            'spec': spec,
        })
        return spec

    def _fingerprint(self) -> str:
        """Return a short fingerprint of current circuit + spec state.

        Includes the executing netlist's simulator, resolved by ``run()``
        before cache lookup, so cached evidence cannot cross backend selections.
        Device additions/removals and spec changes also change the fingerprint.
        """
        import hashlib
        import json
        from silk._json_codec import encode_json
        parts = sorted(self.module.elem_dict.keys())
        spec_parts = [(s['name'], s['metric'], s['op'], str(s['target']))
                      for s in self._specs]
        payload = json.dumps(
            encode_json({'devices': parts, 'specs': spec_parts,
             'simulator': self.netlist.simulator}), allow_nan=False,
            sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    @staticmethod
    def _simulate_run(spec):
        """Run the simulator for a prepared *spec* and return a result.

        The middle third of ``run()``: self-free, touching only
        ``spec`` and the plain values on it, so it can run in a worker from a
        pickled copy of ``spec.netlist`` for dispatch by the executor. Reattaching that result to a live bench and extracting
        metrics is :meth:`ModuleTestbench._evaluate_run`'s job, not this
        one's — it needs the owning :class:`ModuleTestbench` (for
        ``_context_from_conditions``) that this method never receives.

        Conditions are attached only after successful simulation. The supply
        is established by the run itself and confirmed against the launched
        deck, never pasted on here from a caller declaration.
        """
        from silk.circuit.test_execution import current_invocation, equivalence, RunEvidence
        invocation = current_invocation()
        if invocation is not None:
            spec = invocation.prepare(spec)
        key = equivalence(spec) if invocation is not None else None
        if key is not None and key in invocation.failures:
            raise invocation.failures[key]
        if key is not None and key in invocation.runs:
            return invocation.runs[key].result(spec)
        print(f"  Simulating {spec.test_name} (corner={spec.corner_str})...")
        from silk.simulation.conditions import (
            OperatingConditions, _launched_corner, has_no_pdk_device,
        )
        netlist = spec.netlist
        try:
            from silk.circuit.test_placement import prepared_execution
            with prepared_execution(spec):
                netlist.simulate(corner=spec.corner_str)
        except Exception as exc:
            error = (_attribute_abort_to_device_probe(exc, netlist)
                     if isinstance(exc, SimulationAborted) else exc)
            from silk.circuit.modules_sim import _attempt_stage
            stage = _attempt_stage.get()
            if stage is not None:
                error._silk_attempt_stage = stage[0]
            if key is not None:
                invocation.failures[key] = error
            if error is exc:
                raise
            raise error from exc
        conditions = netlist.operating_conditions
        if conditions is None:
            # The corner kind comes from the same proof the launch path uses,
            # never inferred 'fixed' from the requested label.
            conditions = OperatingConditions(
                'silk', netlist.simulator,
                **_launched_corner(spec.corner_str, has_no_pdk_device(netlist)))
        netlist.operating_conditions = conditions
        from silk.circuit.test_failure import capture_run_record
        from silk.simulation.launch_fact import _tracker
        tracked = _tracker.get()
        run = capture_run_record(netlist, files=tracked['files'] if tracked else None)
        result = _RunSimResult(netlist=netlist, corner_str=spec.corner_str,
                               conditions=conditions, invocation=spec.invocation,
                               cache_key=spec.cache_key, run_id=run.launch_id, run=run)
        if key is not None:
            invocation.runs[key] = RunEvidence.capture(result)
            return invocation.runs[key].result(spec)
        return result

    def _consume_run_outcome(self, outcome):
        """Apply an attempt's evidence transition, then evaluate or raise.

        A refusal before replacement (or unknown transport state) leaves the
        previous completed evidence alone. Never restore a worker's old copy.
        """
        if outcome.error is not None:
            _reconcile_run_outcome(self, outcome)
            from silk.exceptions import ConfigurationError
            context = outcome.attempt_context
            if (context is not None and not context['suppressed']
                    and outcome.failure_stage != 'preflight'
                    and not isinstance(outcome.error, ConfigurationError)):
                from silk.specs import SpecAttempt
                kind = 'transport_error' if outcome.failure_stage == 'transport' else 'simulation_error'
                reason = f'{kind}: {outcome.error_type or type(outcome.error).__name__}: {outcome.error}'
                with _spec_attempt_scope(self):
                    for entry in self._specs:
                        spec = entry.get('spec')
                        if spec is not None:
                            spec.record_attempt(SpecAttempt(
                                attempt_id=context['attempt_id'] + ':' + spec.key,
                                source=context['source'], kind=kind, reason=reason,
                                timestamp=outcome.completed_at, test_name=self.test_name,
                                report_key=self.report_key, failure_stage=outcome.failure_stage,
                                requested_conditions=context['requested_conditions']))
            from silk.circuit.test_failure import attach_failure_info
            attach_failure_info(outcome.error, outcome)
            raise outcome.error
        from silk.specs.attempt import _evaluation_attempt
        token = _evaluation_attempt.set((outcome.attempt_context, outcome.completed_at))
        try:
            try:
                return self._evaluate_run(outcome.result)
            except Exception as error:
                from silk.circuit.test_failure import attach_failure_info
                attach_failure_info(error, run=getattr(outcome.result, 'run', None))
                raise
        finally:
            _evaluation_attempt.reset(token)

    def _evaluate_run(self, result):
        """Evaluate a launch outcome in this consumer's execution context.

        Selected placements read retained run facts without replacing the
        authored bench or copying results onto another placement's netlist.
        Direct/root execution preserves the authored netlist identity and
        main's result-state update. A reader selects the recorded backend
        when preparation has already restored a different authored backend.
        Each consumer derives its own characterization context and report.
        """
        from silk.circuit.test_placement import begin_execution
        begin_execution(self)
        from silk.simulation.generation import accept
        raw = getattr(result.netlist, '_raw', None)
        if raw is not None:
            try:
                accept(raw)
                result.netlist.simResultsAll = raw.native
            except Exception:
                from silk.circuit.modules_sim import _invalidate_current_results
                _invalidate_current_results(self.netlist)
                raise
        selection = current_placement(self)
        if selection is not None:
            from uuid import uuid4
            selection.evidence.run_id = result.run_id or uuid4().hex
        reader = None
        if selection is not None and (selection.path or
                result.conditions.backend != self.netlist.simulator):
            from silk.circuit.test_placement import EvidenceResults
            reader = EvidenceResults(result.netlist, authored=self.netlist)
            selection.reader = reader
            netlist = reader
        else:
            netlist = self.netlist
            if result.netlist is not netlist:
                for attr in _SIMULATED_STATE_ATTRS:
                    value = getattr(result.netlist, attr, _UNSET)
                    if value is not _UNSET:
                        setattr(netlist, attr, value)
        from silk.simulation.generation import register_owner
        from silk.circuit.modules import Module
        if isinstance(netlist, Module) and type(netlist).__name__ != 'EvidenceResults':
            register_owner(netlist, getattr(netlist, '_raw', None))
        netlist.operating_conditions = result.conditions
        netlist._last_sim_label = result.corner_str
        netlist.characterization_context = self._context_from_conditions(result.conditions)
        sample = getattr(result.netlist, '_mc_sample', None)
        # A Monte Carlo sample records no design-stage measurement (owner
        # ruling D3): one slot per corner would hold whichever random sample
        # ran last.
        self._record_measurements = sample is None
        try:
            report = self.evaluate(result.corner_str)
        finally:
            self._record_measurements = True
            if selection is not None:
                selection.reader = None
        if sample is not None:
            _stamp_sample_provenance(report, sample)
            if sample.record_draws:
                _attach_sample_draws(report, result.netlist, sample)
        report._operating_invocation = result.invocation
        report.run = result.run
        if result.cache_key is not None:
            self._cache[result.cache_key] = report
        return report

    def _require_unambiguous_analyses(self):
        """Refuse, before any extraction, incompatible probe ownership or a
        metric whose plot this test's declared analyses cannot settle.

        Layer 2 of the three that share one wording — the generate-time layer
        (:func:`silk.codegen.testbench_generator._validate`) catches a spec
        before it becomes Python, this one catches a hand-built test, which
        may add sim types after its metrics, and the loader
        (:func:`silk.simulation.backend.select_plot`) catches whatever reaches
        a multi-plot rawfile without naming an analysis.

        Both bound forms are checked here. The declarative form asks
        :func:`_resolve_metric_analysis` about its ``analysis=``; the function
        form asks :func:`_function_form_slot_analyses` about each slot's
        probe, which is the same rule per slot and refuses with
        :class:`~silk.exceptions.MetricBindingError` rather than
        ``ValueError`` so that a bad binding propagates out of
        ``run_all_tests`` like other binding failures.

        Callable-form metrics are not checked: silk does not know what they
        load. They meet the rule at the loader instead.
        """
        declared = _declared_analyses(self.netlist)
        for metric in self._metrics:
            if _is_function_form_metric(metric):
                try:
                    self._refuse_cross_bench_probes(
                        metric['name'],
                        getattr(metric['fn'], '__name__', repr(metric['fn'])),
                        metric['slots'])
                except TypeError as exc:
                    raise MetricBindingError(str(exc)) from exc
                _function_form_slot_analyses(metric, declared)
            elif 'where' in metric:
                _resolve_metric_analysis(
                    metric['where'], metric['slots'], declared,
                    metric['analysis'], signals=metric['signals'])

    def evaluate(self, corner=None, *, rawfile=None, temperature=None,
                 voltage=None, log=None, reason=None):
        """Evaluate loaded SPICE evidence without launching a simulator.

        Parameters
        ----------
        corner : str or ConditionState, optional
            Caller-declared process for external files (default: this
            test's :meth:`set_corner` corner, else **unknown**: the PDK
            default is not assumed). ``ConditionState.NONE`` declares that no
            process corner applies (a deck with no PDK device).
            For Silk-origin memory, omit to reuse its recorded process or
            supply a matching assertion; relabeling raises.
        rawfile : str or os.PathLike, optional
            Read this file afresh on every explicit call, including unchanged
            paths. Relative paths also resolve against the test directory.
            Omit to reuse memory without reading disk.
        temperature, voltage : float or ConditionState, optional
            External-file declarations are recorded as given: each may be
            declared alone, and the other is then recorded as unknown. A
            scoped characterization context needs corner, temperature and
            voltage each a value or ``ConditionState.NONE``; anything else
            leaves the report unscoped. They never configure a
            simulator or supply, and need not match the local setup.
            Recognized temperature axes reject scalar temperatures.
            For Silk-origin memory these arguments may only confirm the
            existing snapshot, including its unknown fields. External memory
            permits redeclaration; omitted arguments reuse its context.
        log, reason : optional
            Abort diagnostics for an explicit file, not provenance evidence.

        Returns
        -------
        SpecReport
            Metrics, spec verdicts and an immutable ``operating_conditions``
            snapshot. Explicit files have external origin and caller-declared
            PVT (or unknown/swept temperature). Selected hierarchy reports
            retain their result reader; direct unselected reports reference
            the authored netlist.

        Raises
        ------
        CharacterizationContextError
            A scalar temperature on a loaded sweep, or
            attempted relabeling of Silk-origin evidence.
        SimulationError
            No loaded results and no file. Use ``generate_netlist()`` plus
            external simulation and ``evaluate(rawfile=...)``, or ``run()``.
        SimulationAborted
            The file has no analysis data. No metrics or Measurements run.
        OSError, RawfileError
            Missing, unreadable or malformed explicit file. Failed loads
            clear live results/context/conditions; prior scalar reports and
            committed DB evidence remain. Invalid declarations are checked
            before clearing valid live results.
        """
        from silk.exceptions import CharacterizationContextError, SimulationError
        from silk.simulation.conditions import (
            ConditionState, OperatingConditions, confirm_conditions, external_sweep,
        )
        from silk.circuit.test_placement import campaign_run
        run = campaign_run(self)
        if run is not None:
            netlist = run.reader
        else:
            netlist = self.netlist if rawfile is not None else self._result_netlist()
        if not self._module_added:
            raise ValueError(f"Must call add_module_under_test() before evaluating test {self.test_name}")
        if netlist is None:
            if self.netlist is not None:
                raise SimulationError(f'No retained results for selected test {self.test_name!r}; run it first.')
            raise ValueError(f"Netlist not created for test {self.test_name}")
        previous = netlist.operating_conditions
        if (rawfile is None and previous is not None
                and previous.backend == 'behavioral'
                and netlist.sim_analysis_dict):
            from silk.simulation.behavioral_backend import _declared_analyses
            _declared_analyses(netlist)
        # TASK-406 + TASK-407 (merged by the reviewer). Label versus evidence
        # (design 2026-10-01 section 3.1): a silk run keeps the corner it was
        # requested at as its label (SpecReport.corner, the sweep table) even
        # when its evidence records corner none. An external file's corner
        # comes from the caller or set_corner(); with neither it is unknown,
        # never the PDK default (the label then only names the run).
        from silk.simulation.conditions import ConditionState
        self._check_declaration('corner', corner)
        declared_corner = corner
        label = None
        if rawfile is None and previous is not None and previous.origin == 'silk':
            label = getattr(netlist, '_last_sim_label', None)
        if corner is ConditionState.NONE:
            corner_decl = corner
            corner_str = (label if (rawfile is None and label is not None)
                          else self._default_corner())
        elif corner is not None:
            corner_decl, corner_str = corner, str(corner)
        elif rawfile is None and previous is not None:
            # Memory: a silk snapshot confirms its own corner; an external
            # one is redeclared from what it already recorded.
            corner_decl = None
            if label is not None:
                corner_str = label
            elif (previous.corner is not None
                  and not isinstance(previous.corner, ConditionState)):
                corner_str = str(previous.corner)
            else:
                corner_str = self._default_corner()
        else:
            corner_decl = self.corner
            corner_str = self._default_corner()
        if rawfile is not None:
            # Validate declarations before replacing live evidence (the
            # scoped context is built once the sweep is known). Every
            # explicit file is a fresh load, including an unchanged path.
            self._external_evidence(corner_decl, temperature, voltage, None,
                                    backend=getattr(run, 'backend', None))
            from silk.circuit.test_placement import begin_execution
            begin_execution(self)
            netlist._raw = None
            netlist.simResultsAll = None
            netlist.sim_results_loaded = False
            netlist.results_file = None
            netlist.characterization_context = None
            netlist.operating_conditions = None
            if run is not None:
                netlist.load_file(rawfile, simulatorType=run.backend)
            else:
                netlist.load_file(rawfile)
            netlist.read_raw_file(log=log, reason=reason)
            sweep = external_sweep(netlist._raw)
            try:
                context, conditions = self._external_evidence(
                    corner_decl, temperature, voltage, sweep,
                    backend=getattr(run, 'backend', None))
            except CharacterizationContextError:
                from silk.circuit.modules_sim import _invalidate_current_results
                _invalidate_current_results(netlist)
                raise
            netlist.results_file = netlist.dataFile
        else:
            if not netlist.sim_results_loaded:
                raise SimulationError(
                    f"evaluate(): no simulation results are loaded for test "
                    f"{self.test_name!r} and no rawfile was given. Either pass "
                    "rawfile=<path> after generate_netlist() and external simulation, "
                    "use load_file() and read_raw_file(), or call run().")
            if previous is not None and previous.origin == 'silk':
                # The run's own label is not a claim about the evidence; any
                # other declared corner (or ConditionState.NONE) must confirm
                # what the run recorded.
                confirm_conditions(
                    previous,
                    corner=(None if (declared_corner is None
                                     or (label is not None
                                         and not isinstance(declared_corner, ConditionState)
                                         and str(declared_corner) == label))
                            else declared_corner if declared_corner is ConditionState.NONE
                            else str(declared_corner)),
                    temperature=temperature, voltage=voltage)
                conditions = previous
                context = self._context_from_conditions(conditions)
                if context == netlist.characterization_context:
                    context = netlist.characterization_context
            else:
                context, conditions = self._external_evidence(
                    corner_decl, temperature, voltage,
                    external_sweep(netlist._raw), previous=previous)
                if context is not None and context == netlist.characterization_context:
                    context = netlist.characterization_context
        netlist.characterization_context = context
        netlist.operating_conditions = conditions

        # --- Shared extraction/spec tail (formerly the private
        # _evaluate_loaded, retired by TASK-115 so this is the only copy):
        # extract every registered metric from the results loaded on
        # netlist, build the SpecReport, check each spec, and record an
        # evidence-derived Measurement on each attached first-class Spec.
        # corner_str and context are stamped on as metadata — nothing below
        # reads the simulator.
        from silk.circuit.spec_report import SpecReport
        # This evaluation is now the bench's whole sweep: ONE corner, read
        # from ONE file (TASK-218). Rewriting the table here, before the
        # extraction loop, is what keeps a metric's probes resolvable while
        # run_corners() is mid-sweep -- a one-entry table is never ambiguous,
        # so the loop below resolves against the live results exactly as it
        # did before this task. run_corners() installs the full table only
        # once every corner has completed.
        if run is not None:
            # The label the worker requested, independent of the corner the
            # evidence proves (a PDK-free deck proves none).
            corner_str = run.label
        rawfile_path = getattr(netlist, 'results_file', None)
        self._sweep_table = ({} if rawfile_path is None
                             else {corner_str: rawfile_path})
        generation = getattr(getattr(netlist, '_raw', None), '_generation', None)
        self._sweep_generations = {corner_str: generation} if generation is not None else {}
        self._mc_samples = None
        # Plot selection is settled BEFORE the loop below, against the
        # finally-declared analyses (TASK-212). It has to be: the loop catches
        # every exception into an 'Error: ...' string, which the spec flow
        # does not raise on — exactly how a metric reading the wrong analysis
        # used to pass unnoticed.
        self._require_unambiguous_analyses()
        print(f"  Extracting metrics...")
        from silk.metrics.diagnostics import MetricWarning, MetricWarningRecord
        # Reserve derived names before invoking any extractor. A caller's
        # explicit dotted name must never overwrite (or be overwritten by)
        # a declared result field, regardless of registration order.
        output_declarations, metric_types = _metric_output_declarations(self._metrics)
        metrics = {}
        metric_warnings = {}
        metric_errors = {}
        inventory_loaded = False
        available_signals = None
        for metric, outputs in zip(self._metrics, output_declarations):
            name = metric['name']
            try:
                # Recording is scoped to this one extractor call so a
                # warning is attributed to the output whose extractor
                # raised it (TASK-337). The inner filter always shows
                # (and thus records) a MetricWarning regardless of the
                # caller's own filters on that category -- the swallow
                # below must not depend on caller state -- while every
                # other category keeps whatever action the ambient
                # filters already gave it (an 'ignore'd warning is never
                # captured here either, exactly as before this loop
                # existed).
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always', MetricWarning)
                    value = (_evaluate_expression(metric['expression'], metrics)
                             if 'expression' in metric
                             else metric['extractor'](netlist))
                # The 'with' block above has already exited (restoring the
                # caller's filters and showwarning), so every re-emission
                # below is dispatched exactly as it would be from ordinary
                # code -- not re-captured by the now-closed recorder, and
                # exactly one MetricWarning surfaces per occurrence.
                own_warnings = []
                reemit = []
                for w in caught:
                    if isinstance(w.message, MetricWarning):
                        original = w.message
                        attributed = MetricWarning(
                            metric=original.metric, code=original.code,
                            message=original.message, details=original.details,
                            suggestion=original.suggestion, output=name,
                        )
                        own_warnings.append(MetricWarningRecord.from_warning(attributed))
                        reemit.append(attributed)
                    else:
                        reemit.append(w)
                if own_warnings:
                    # Recorded BEFORE any re-emission below, which may raise
                    # under the caller's own filterwarnings('error',
                    # MetricWarning) policy -- the ruling requires the
                    # record to survive on the report either way, and a
                    # raise from inside the loop must not skip this.
                    metric_warnings[name] = tuple(own_warnings)
                for item in reemit:
                    if isinstance(item, MetricWarning):
                        # Re-emitted under the CALLER's ordinary filters: a
                        # filterwarnings('error', MetricWarning) raises here
                        # exactly as it would have from inside the extractor
                        # (falling to the 'Error: ...' branch below), and
                        # 'ignore' silences it -- the record above is kept
                        # on the report either way.
                        warnings.warn(item, stacklevel=2)
                    else:
                        warnings.warn_explicit(
                            item.message, item.category, item.filename, item.lineno)
                # Build all fields before publishing any, so a bad result
                # cannot leave a mixture of stale values and errors.
                expanded = {}
                for field in outputs:
                    scalar = getattr(value, field)
                    if isinstance(scalar, bool) or not isinstance(scalar, Real):
                        raise TypeError(
                            f"metric {name!r}: output {field!r} must be a real scalar")
                    expanded[f'{name}.{field}'] = float(scalar)
                metrics[name] = value
                metrics.update(expanded)
            except (MetricBindingError, AmbiguousBranchCurrentError):
                # TASK-449: an ambiguous i(x) is equally the caller's error.
                # TASK-215.7: a Signal slot that could not be bound, or was
                # bound to the wrong kind of signal, is the caller's error
                # (decision-1). It raises from run()/evaluate() and, being a
                # ConfigurationError, propagates through run_all_tests()
                # instead of becoming this test's {'error': str}.
                raise
            except Exception as e:
                print(f"    Warning: error extracting metric '{name}': {e}")
                metrics[name] = f"Error: {e}"
                if not inventory_loaded:
                    inventory_loaded = True
                    try:
                        available_signals = list(netlist.get_signal_names())
                    except Exception:
                        # Diagnostic enrichment must never mask extraction failure.
                        available_signals = None
                for output in (name, *(f'{name}.{field}' for field in outputs)):
                    metrics[output] = metrics[name]
                    metric_errors[output] = {
                        'message': str(e), 'exception_type': type(e).__name__,
                        'available_signals': (None if available_signals is None
                                              else list(available_signals)),
                    }

        # TASK-330.4 (op_table), reviewer ruling (A): capture a plain-data
        # operating-point device inventory HERE, once, rather than let
        # op_table() hold a live Module reference or re-read the rawfile
        # later (a report is pickled across the parallel test executor,
        # TASK-301, and stored -- the rawfile may be long gone by the time
        # someone calls op_table()). Read the switch off self.module (the
        # DUT template the user actually set it on) -- not netlist,
        # whose own copy of the flag is written only inside run()
        # (_build_run_spec's propagation line below) and would read stale
        # False on an evaluate(rawfile=...)-only flow, same reasoning
        # _require_ammeter_currents already applies to save_device_currents.
        op_devices = None
        op_unavailable_reason = None
        if self.effective_save_device_params():
            dut_entry = (netlist.elem_dict.get(DUT_ELEMENT_NAME)
                        if netlist is not None else None)
            dut_copy = dut_entry.get('device') if isinstance(dut_entry, dict) else None
            if dut_copy is None:
                op_unavailable_reason = (
                    "save_device_params is True, but this test has no DUT "
                    "definition to walk (add_module_under_test() has not run); no "
                    "operating-point data was captured.")
            else:
                raw = getattr(netlist, '_raw', None)
                if raw is None:
                    from silk.simulation.backend import RawResult
                    raw = RawResult(native=netlist.simResultsAll,
                                    source_path=getattr(netlist, 'results_file', ''))
                try:
                    op_devices = capture_op_device_records(
                        netlist, netlist.backend, raw, pdk=self.pdk_details)
                except NotSupportedError as e:
                    op_unavailable_reason = (
                        f"save_device_params is True, but the "
                        f"{getattr(netlist, 'simulator', '?')!r} backend "
                        f"does not support operating-point device capture yet "
                        f"({e}); no data was captured for this report.")
        else:
            op_unavailable_reason = (
                "save_device_params is False for this test; no "
                "operating-point data was captured. Set "
                "save_device_params = True on the device under test, or on "
                "this test alone, before run()/evaluate() to use op_table().")

        # TASK-306: provenance. 'silk-launch' only when silk ran the
        # simulator for these results (silk-origin conditions); anything
        # handed in is 'external-rawfile' and claims no binary.
        from silk.simulation.provenance import from_module as _provenance_from_module
        run_provenance = _provenance_from_module(
            netlist, conditions, pdk_details=self.pdk_details)
        if run_provenance.run_origin == 'silk-launch':
            run_provenance = replace(run_provenance, producing_test={
                'owner_path': self.owner_path, 'test_name': self.test_name})

        from silk.simulation.behavioral_backend import behavioral_element_notes
        report = SpecReport(
            test_name=self.test_name,
            module_name=self._placement_module_name,
            metrics=metrics,
            owner_path=self.owner_path,
            parameters=self.parameters,
            characterization_context=context,
            operating_conditions=conditions,
            # The corner this evaluation was requested at: a silk run's
            # LABEL (kept even when the evidence records corner none), an
            # external file's recorded corner (None when unknown).
            # store_report() files an unscoped report under it instead of the
            # tracker's current corner.
            corner=(corner_str if (conditions.origin == 'silk' or run is not None)
                    else conditions.corner),
            # The file these metrics came from (TASK-218): after
            # run_corners() has moved on, this is how a script finds the
            # waveforms behind a per-corner number.
            rawfile_path=rawfile_path,
            metric_warnings=metric_warnings,
            report_key=self.report_key,
            metric_types=metric_types,
            metric_errors=metric_errors,
            op_devices=op_devices,
            op_unavailable_reason=op_unavailable_reason,
            provenance=run_provenance,
            behavioral_elements=behavioral_element_notes(netlist),
        )
        # Record a Measurement on each attached first-class Spec, in
        # addition to the legacy SpecReport.check() path. The recorded
        # backend determines the source, even if the live selection changed
        # after simulation.
        for s in self._specs:
            spec_obj = s.get('spec')
            report.check(
                s['name'], s['metric'], s['op'], s['target'],
                spec_key=(None if spec_obj is None else spec_obj.key),
                owner_path=(
                    None
                    if spec_obj is None
                    else self.owner_path
                ),
            )
            result = report.results[-1]
            if (spec_obj is None or run is not None
                    or not getattr(self, '_record_measurements', True)):
                continue
            metric_value = metrics.get(s['metric'])
            from silk.specs.attempt import publish_metric
            with _spec_attempt_scope(self):
                publish_metric(spec_obj, metric_value, conditions=conditions,
                               report=report, metric=s['metric'])
        # Preserve legacy .sim_results attribute for signal loading
        report._raw_generation = generation
        report.sim_results = netlist
        self._metric_values = dict(metrics)
        if run is not None:
            # A campaign run publishes nothing: no evidence index, no retained
            # record. The caller reads the report it is handed.
            return report
        from silk.circuit.test_placement import publish_report
        publish_report(self, report, netlist)
        return report

    def prepare(self, corner=None, *, temperature=None, voltage=None,
                run_label=None, ngspice_version=None):
        """Write one simulation deck and return its concrete path as a string.

        Resolves the same live owner settings and output paths as :meth:`run`,
        using its preparation and netlist writer. A selected test handle uses
        its hierarchy owner's inherited conditions, just like ``run()``.
        Arguments have the same
        meaning as on ``run()``; temperature and voltage are assertions, not
        overrides. No simulator is launched and no metrics, reports, cache
        entries or database records are read or changed.

        A declared plan must resolve to one deterministic point (pass
        ``corner`` to select one corner); its point subdirectory is retained.
        One-element temperature and supply lists are supported. Behavioral benches,
        Monte Carlo and multi-deck plans raise ``NotSupportedError`` before
        writing. This includes ngspice parameter probes outside the device's
        declared operating-point parameters: emission needs an auxiliary
        preflight simulation, which this method never runs or caches.
        New output mode allocates a fresh directory on each call;
        the returned path includes it. Existing simulation outputs survive.
        Use :meth:`evaluate` to score results produced by an external runner.

        Parameters
        ----------
        corner : str or Corner, optional
            Process corner, as for :meth:`run`.
        temperature : float, optional
            Deprecated assertion of the configured TEMP, as for :meth:`run`.
        voltage : float, optional
            Deprecated assertion of the primary supply, as for :meth:`run`.
        run_label : str, optional
            Output-directory label, as for :meth:`run`.

        ngspice_version : str, optional
            Target release for ``generate_netlist()``, including selected tests.
            Requires the effective ngspice backend. ``None`` uses the installed
            release; an explicit target permits external execution without a
            local binary. silk refuses local execution if releases differ or
            the installed release is unknown.

        Returns
        -------
        str
            Path of the written simulation deck.

        Raises
        ------
        silk.exceptions.NotSupportedError
            For a behavioral bench, a Monte Carlo setting, a multi-deck
            corner plan without ``corner``, or a probe that needs an
            auxiliary preflight simulation.
        silk.exceptions.OutputPathError
            Invalid ``run_label``, as for :meth:`run`.
        """
        from silk.exceptions import NotSupportedError
        simulator = self._effective_simulator()
        if ngspice_version is not None and simulator != 'ngspice':
            raise ValueError('ngspice_version requires the ngspice simulator')
        if simulator == 'behavioral':
            raise NotSupportedError(
                'prepare() cannot write a behavioral deck: behavioral benches '
                'have no SPICE deck. Use run().')
        if self.monte_carlo is not None:
            raise NotSupportedError(
                'prepare() does not support Monte Carlo runs. Use run().')
        if corner is None and self.corners is not None and len(self.corners) > 1:
            raise NotSupportedError(
                'prepare() does not support a multi-deck corner run. '
                'Pass corner=... to prepare one corner.')
        _require_no_probe_preflight(self, simulator)
        planned = self._plan_runs(corner=corner)
        corner_subdir = None
        if planned is not None:
            if len(planned) != 1:
                raise NotSupportedError(
                    'prepare() requires one deck; multi-deck runs are unsupported.')
            corner = planned[0].corner
            corner_subdir = planned[0].corner_subdir
        with _planned_conditions(self, planned[0] if planned else None):
            spec = self._prepare_run(
                corner, temperature=temperature, voltage=voltage,
                run_label=run_label, corner_subdir=corner_subdir, use_cache=False)
            if ngspice_version is None:
                spec.netlist.generate_netlist(corner=spec.corner_str)
            else:
                spec.netlist.generate_netlist(corner=spec.corner_str,
                                              ngspice_version=ngspice_version)
            return spec.netlist._run_files.netlist_path

    def validate_simulation_setup(self):
        """Validate this bench's simulation setup as ``run()`` would run it, without running it or changing it.

        Resolves the bench's effective simulator, settings and supply on a
        private view of its netlist (the live owner's inherited simulator,
        ambient TEMP, supply and options, as ``run()`` would apply them), then
        calls :func:`silk.simulation.validate_simulation_setup` on that view.
        Returns ``None`` if no static refusal is found, otherwise raises the
        first refusal the run would raise; see that function for the checks
        covered and not covered. Beyond them, the bench first refuses a DC or
        Noise analysis whose source it lacks (``SweepSourceError``), resolves
        its effective ambient TEMP as a run does (see the API reference for the
        refusals), and takes the owner's device-capture switches.
        Nothing is prepared, written, allocated or
        consumed: no deck, output folder, attempt, placement evidence or
        prepared run is touched, and no authored setting changes.

        Raises
        ------
        SimulationConfigurationError
            The bench has no module under test (it was never wired), rather
            than passing as an empty circuit. This is an input error of this
            method, outside the same-error-as-the-run guarantee: ``run()``
            raises a ``ValueError`` there.
        """
        return self._validate_setup(self._inherited_temperature(),
                                    self._inherited_supply())

    def _validate_setup(self, temperature, supply):
        """The read-only body of :meth:`validate_simulation_setup`.

        *temperature* and *supply* are the inherited settings in force. A
        selected nested bench passes its entry's, so no placement scope (which
        records evidence and touches the netlister) is entered.
        """
        from silk.exceptions import SimulationConfigurationError
        from silk.simulation._replay import apply_effective_settings
        from silk.simulation._setup_check import check_view, private_view
        from silk.simulation.conditions import prepare_temperature
        if not self._module_added or self.netlist is None:
            raise SimulationConfigurationError(
                f"Must call add_module_under_test() before validating test "
                f"{self.test_name}: an unwired testbench has no circuit to check.")
        # run() refuses a DC/Noise source this bench lacks before anything else.
        self._check_sweep_sources()
        # A private graph: no pickling, no live-graph registration, and no
        # placement refresh or pin sync written to the live hierarchy.
        view = private_view(self.netlist)
        apply_effective_settings(self, view, temperature=temperature, supply=supply)
        # The capture switches a run inherits from the live owner
        # (``_configure_capture``); the behavioral refusals read them.
        for field in ('save_device_currents', 'save_device_params',
                      'omit_device_internal_nodes'):
            setattr(view, field, getattr(self.module, field))
        view.save_device_params = self.effective_save_device_params()
        # run() resolves ambient temperature (read-only) before it reaches
        # the module's own admission, and its refusals come first.
        prepare_temperature(view)
        return check_view(view)

    def run(self, corner=None, *, temperature=None, voltage=None,
            run_label=None):
        """Run the test: simulate and extract metrics.

        Equivalent to ``self.netlist.simulate(corner)`` followed by
        :meth:`evaluate` on the results it loaded; the extraction
        tail is shared, not duplicated. To evaluate a rawfile produced
        elsewhere, see :meth:`evaluate`.

        Before cache lookup or deck preparation, ``self.netlist`` selects the
        live owner's current simulator unless ``self.netlist.simulator`` was
        explicitly assigned. Internal inheritance never pins that selection.
        The selected backend owns emission, execution and result loading.

        Ambient TEMP resolves the same way and at the same point: unless this
        test set its own temperature, it takes the setting in force from its
        live owner — or, inside a recursive ``run_all_tests``, from the
        nearest owner up that tree which set one. The value is re-read at
        every run, so a change or a clear on the owner is followed, and an
        inherited value never becomes this test's own. The primary supply
        inherits by exactly the same rule, and the source it names is
        re-resolved against this test's own netlist at every run. Simulator
        options other than ``temp`` inherit key by key: an option
        this test never set of its own reaches the emitted deck as the live
        owner's current value; one it set itself always wins.

        Parameters
        ----------
        corner : str or Corner, optional
            Process corner: a plain string (``'tt'``) or a
            :class:`~silk.simulation.corner.Corner` member. When omitted:
            this test's own corner from :meth:`set_corner`, else the
            testbench PDK's ``default_corner`` (from the manifest), falling
            back to ``'tt'`` if the PDK has none.
        temperature : float, optional
            Deprecated assertion of independently configured fixed TEMP;
            never sets it. A mismatch or absent setup raises before cache
            lookup or launch. Prefer ``set_temperature(...)`` followed by
            ``run()``.
        voltage : float, optional
            Deprecated assertion of the primary supply; never sets one. The
            declaration must match the supply this deck actually runs at, and
            a supply silk cannot establish refuses rather than being reported. Prefer ``set_supply_voltage(...)`` followed by
            ``run()``. With verified fixed temperature an established supply
            produces a complete scoped context; otherwise the report has
            partial, unscoped conditions.
        run_label : str, optional
            Output-directory label, keyword-only. Use 1-64 ASCII characters,
            starting with a letter or digit, followed by letters, digits,
            underscores, dots or hyphens. Each live instance/test is isolated
            automatically, even without a label. Distinct labels retain
            earlier files; the same label reuses its base directory. The live
            owning circuit's ``set_output_options`` setting chooses a
            persistent unique child per actual run (default), or explicit
            overwrite. Monte Carlo uses its independent ``output_mode``
            setting, which defaults to overwrite. Recursive
            ``run_all_tests`` uses its invoking root's mode for that
            invocation without changing child settings. Omitting the label
            selects the unlabeled directory again. Labels organize files
            only; test/report/profile identity and database storage policy
            are unchanged. The report cache is disabled in this release, so every ``run()`` simulates and writes
            labeled output.

        Returns
        -------
        SpecReport or CornerResults or MonteCarloResults
            A :class:`~silk.circuit.spec_report.SpecReport`; a
            :class:`~silk.circuit.spec_report.CornerResults` when
            :meth:`set_corners` declared a list (every declared corner, or
            only *corner* when one is passed), simulated one at a time; a
            :class:`~silk.circuit.monte_carlo.MonteCarloResults` when
            :meth:`set_monte_carlo` declared a sampling plan.

        Raises
        ------
        silk.exceptions.OutputPathError
            Invalid ``run_label`` or an output directory that resolves
            outside the DUT output root, before writing files. A
            ``ValueError`` subclass, so the historical contract is
            unchanged; also a :class:`~silk.exceptions.ConfigurationError`,
            so ``run_all_tests`` propagates it.
        silk.exceptions.SimulationAborted
            The run produced no data rows (aborted analysis); raised by the
            backend's ``run()``, before the shared evaluation tail, so no
            Measurement is recorded.
        """
        if self.monte_carlo is not None:
            from silk.circuit.modules_testbench import _check_process_sources
            _check_process_sources((), standalone=self, corner=corner)
        planned = self._plan_runs(corner=corner, temperature=temperature, voltage=voltage, run_label=run_label)
        if planned is not None:
            return self._run_plan(planned, temperature=temperature,
                                  voltage=voltage, run_label=run_label)
        prepared = self._prepare_run(corner, temperature=temperature,
                                     voltage=voltage, run_label=run_label)
        if not isinstance(prepared, _RunSpec):
            # A cache hit: _prepare_run() already returned the SpecReport,
            # exactly as run() itself used to.
            return prepared
        return self._consume_run_outcome(
            _simulate_run_outcome(prepared, self._simulate_run))

    # --- the run plan (TASK-426) -------------------------------------------
    #
    # A declared corner list (and, later, Monte Carlo samples) turns one test
    # into many runs. The expansion is a planning step BELOW run(), because
    # the parallel run_all_tests walk never calls run() for a plain
    # ModuleTestbench: both walks call _plan_runs, prepare the plan here in
    # the parent, simulate through the shared executor, then evaluate in plan
    # order and collect with _collect_runs.

    def set_temperatures(self, temperatures: list[float] | tuple[float, ...] | None) -> 'ModuleTestbench':
        """Declare ambient Celsius points, or explicitly clear with None."""
        if self.netlist is None:
            self._create_netlist()
        self.netlist.set_temperatures(temperatures)
        return self

    def set_supply_voltages(self, voltages: list[float] | tuple[float, ...] | None,
                            *, source: str | None = None) -> 'ModuleTestbench':
        """Declare supply points using the executing bench's source rules."""
        if self.netlist is None:
            self._create_netlist()
        self.netlist.set_supply_voltages(voltages, source=source)
        return self

    def set_variable_values(self, var: Variable, values: Optional[list | tuple]) -> 'ModuleTestbench':
        """Declare the ordered values of one shared Variable, or clear with None.

        Values use the Variable's bounds and resolution. Lists cross unless
        joined by :meth:`set_groups`. Replacing a list preserves its position.

        Parameters
        ----------
        var : Variable
            Shared design Variable whose references follow each planned value.
        values : list or tuple or None
            Nonempty finite scalar values, or ``None`` to clear this list.
            Numeric Variables accept real numbers excluding booleans;
            categorical Variables accept their declared JSON scalar values.

        Returns
        -------
        ModuleTestbench
            This testbench, for method chaining.

        Raises
        ------
        TypeError
            ``var`` is not a Variable or ``values`` has the wrong container type.
        ValueError
            The list is empty or a value is invalid for this Variable.
        silk.exceptions.SimulationConfigurationError
            On planning, names collide, group relationships are invalid or
            effective coordinates repeat a complete run.

        Notes
        -----
        Authored Variable values are restored after preparation, simulation
        and evaluation, including failures. Clearing and readding a list
        appends it after the remaining Variable declarations.
        """
        if not isinstance(var, Variable):
            raise TypeError('set_variable_values expects a Variable')
        if values is not None:
            if not isinstance(values, (list, tuple)):
                raise TypeError(f'Variable {var.name!r}: expected a list or tuple, or None')
            if not values:
                raise ValueError(f'Variable {var.name!r}: expected a nonempty list or tuple')
            values = tuple(self._checked_variable_value(var, value) for value in values)
        declarations = list(self._variable_lists)
        index = next((i for i, (other, _) in enumerate(declarations) if other is var), None)
        if index is not None:
            if values is None:
                declarations.pop(index)
            else:
                declarations[index] = (var, values)
        elif values is not None:
            declarations.append((var, values))
        self._variable_lists = declarations
        return self

    @staticmethod
    def _checked_variable_value(var, value):
        import math
        import numpy as np
        if isinstance(value, np.generic):
            value = value.item()
        valid = isinstance(value, (str, bool, int, float)) and value is not None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            valid = valid and math.isfinite(value)
        if var.var_type != 'categorical':
            valid = valid and isinstance(value, Real) and not isinstance(value, bool)
        elif valid:
            valid = var.validate(value)
        if not valid:
            raise ValueError(f'Variable {var.name!r}: unusable sweep value {value!r}')
        saved = var._value
        try:
            var.value = value
            effective = var.value
            if isinstance(effective, Real) and not math.isfinite(effective):
                raise ValueError(f'Variable {var.name!r}: nonfinite effective sweep value {effective!r}')
            return effective
        finally:
            var._value = saved

    def set_groups(self, groups: Optional[list | tuple]) -> 'ModuleTestbench':
        """Step each group's lists together; cross groups and remaining lists.

        Members are Variables or ``'corner'``, ``'temperature'``, ``'supply'``.
        None clears grouping. All group relationships are checked on planning.

        Parameters
        ----------
        groups : list or tuple or None
            Ordered nonempty groups of declared lists, or ``None`` to restore
            plain crossing. Lists in a group must have equal lengths and
            each list may belong to at most one group.

        Returns
        -------
        ModuleTestbench
            This testbench, for method chaining.

        Raises
        ------
        TypeError
            ``groups`` is not a list, tuple or ``None``.
        ValueError
            A group is empty or contains an unknown member.
        silk.exceptions.SimulationConfigurationError
            On planning, a member lacks a declared list, lengths differ,
            membership repeats or complete run coordinates repeat.

        Notes
        -----
        A group occupies its earliest member's axis position: corner,
        temperature, supply, then Variable declaration order. Group rows
        retain their list order. Selecting a grouped corner retains every
        complete matching row and refuses an absent corner.
        """
        if groups is None:
            self._groups = ()
            return self
        if not isinstance(groups, (list, tuple)):
            raise TypeError('set_groups expects a list or tuple of groups, or None')
        if not groups:
            raise ValueError('set_groups needs at least one group; use None to clear')
        checked = []
        for group in groups:
            if not isinstance(group, (list, tuple)) or not group:
                raise ValueError('Each group must be a nonempty list or tuple')
            for member in group:
                if not isinstance(member, Variable) and not (
                        isinstance(member, str) and member in ('corner', 'temperature', 'supply')):
                    raise ValueError(f'Unknown group member {member!r}')
            checked.append(tuple(group))
        self._groups = tuple(checked)
        return self

    def _point_assignments(self, key):
        if not isinstance(key, tuple) or len(key) != 4:
            return ()
        coordinates = dict(key[3])
        return tuple((var, coordinates[var.name]) for var, _ in self._variable_lists)

    def _grouped_coordinates(self, names, temps, supplies, axes):
        from itertools import product
        from silk.exceptions import SimulationConfigurationError
        variables = [var for var, _ in self._variable_lists]
        reserved = {'corner', 'temperature', 'supply', 'supply_source', 'status', 'error', 'sample', 'seed'}
        var_names = [var.name for var in variables]
        if any(not isinstance(name, str) or not name for name in var_names):
            raise SimulationConfigurationError('Swept Variables require nonempty string names')
        if len(set(var_names)) != len(var_names) or reserved.intersection(var_names):
            raise SimulationConfigurationError('Swept Variable names must be unique and not collide with run columns')
        if variables and 'silk_run_variables' in self.parameters:
            raise SimulationConfigurationError('parameters.silk_run_variables is reserved for swept values')
        labels = ['corner', 'temperature', 'supply'] + var_names
        columns = [names, temps, supplies] + [tuple(self._checked_variable_value(var, value)
                   for value in values) for var, values in self._variable_lists]
        declared = [self.corners is not None,
                    axes['temperature'] is not None and axes['temperature'][0] is not None,
                    axes['supply'] is not None and axes['supply'][0] is not None] + [True] * len(variables)
        groups, used = [], set()
        for group in self._groups:
            indices = []
            for member in group:
                if isinstance(member, str):
                    index = labels.index(member)
                else:
                    index = next((i + 3 for i, var in enumerate(variables) if var is member), None)
                if index is None or not declared[index]:
                    raise SimulationConfigurationError(f'Group member {member!r} has no declared list')
                if index in used:
                    raise SimulationConfigurationError(f'List {labels[index]!r} belongs to two groups')
                used.add(index)
                indices.append(index)
            lengths = [len(columns[i]) for i in indices]
            if len(set(lengths)) != 1:
                raise SimulationConfigurationError(f'Unequal group lengths for {[labels[i] for i in indices]}: {lengths}')
            groups.append(indices)
        for i, column in enumerate(columns):
            if i not in used and len(set(column)) != len(column):
                raise SimulationConfigurationError(f'repeated coordinates in ungrouped list {labels[i]!r}: {column!r}')
        groups.extend([i] for i in range(len(columns)) if i not in used)
        groups.sort(key=min)
        points = []
        for rows in product(*(list(zip(*(columns[i] for i in group))) for group in groups)):
            values = [None] * len(columns)
            for group, row in zip(groups, rows):
                for i, value in zip(group, row):
                    values[i] = value
            points.append(values)
        seen = {}
        for values in points:
            key = tuple(values[:3]) + ((tuple(zip(var_names, values[3:])),) if variables else ())
            if key in seen:
                raise SimulationConfigurationError(f'Repeated whole run {key!r}; values {seen[key]!r} and {values!r} compare equal')
            seen[key] = values
        return points

    def _condition_axes(self):
        from silk.circuit.modules_sim import _axis_declaration
        selection = current_placement(self)
        inherited = (getattr(selection, 'inherited_axes', None) if selection else None)
        if inherited is None:
            inherited = {axis: _axis_declaration(self.module, axis)
                         for axis in ('temperature', 'supply')}
        return {axis: _axis_declaration(self.netlist, axis, inherited.get(axis))
                for axis in ('temperature', 'supply')}

    def _has_condition_grid(self):
        return bool(self._variable_lists or self._groups) or any(value is not None and value[0] is not None
                   for value in self._condition_axes().values())

    def _run_declarations(self, monte_carlo=None):
        if monte_carlo is None and type(self).run is not ModuleTestbench.run:
            return None, None
        return self.corners, self.monte_carlo if monte_carlo is None else monte_carlo

    def _require_single_run(self):
        """Refuse collection declarations for a scalar report consumer."""
        from silk.exceptions import NotSupportedError
        corners, monte_carlo = self._run_declarations()
        declarations = []
        if monte_carlo is not None:
            declarations.append('set_monte_carlo(...)')
        if corners is not None:
            declarations.append('set_corners(...)')
        if self._has_condition_grid():
            declarations.append('Variable lists (set_variable_values)' if self._variable_lists else 'condition lists')
        if declarations:
            raise NotSupportedError(
                f"Optimizer requires one report per point; test {self.test_name!r} "
                f"declares {' and '.join(declarations)}, which returns a collection. "
                "Clear those declarations with None before optimizing.")

    def _prepare_single_run(self, *, corner=None, run_label=None):
        """Prepare a scalar consumer through the ordinary testbench planner."""
        from silk.exceptions import NotSupportedError
        self._require_single_run()
        if self._plan_runs(corner=corner, run_label=run_label) is not None:
            raise NotSupportedError('Optimizer cannot consume a declared run collection.')
        return self._prepare_run(corner=corner, run_label=run_label)

    def _plan_runs(self, *, corner=None, temperature=None, voltage=None,
                   run_label=None, _monte_carlo=None, _deterministic=False):
        """This test's declared runs as a list of :class:`_PlannedRun`, or
        ``None`` when it declares nothing (today's single-run path).

        Takes ``run()``'s keyword arguments. An explicit *corner* restricts
        the plan to that one corner (spec section 3.3.1). The *run_label*
        selects the campaign output base; temperature and voltage are
        handled at preparation. Two or more corners on a deck with no PDK device
        warn, as :meth:`run_corners` does: the runs are identical.

        With Monte Carlo declared (:meth:`set_monte_carlo`, or
        *_monte_carlo* from :meth:`run_monte_carlo`) every corner expands
        into its samples, keyed ``(corner, index)``, each with its
        ``_SampleSpec`` and its own sample folder below the corner (inside
        the campaign folder in new mode). Every Monte Carlo refusal (:meth:`_check_monte_carlo`) is raised here,
        before any deck is written.

        A subclass that overrides ``run()`` gets ``None`` too: its override
        decides what a run is (the parallel walk already runs it in-process
        through that override), so its own calls to ``ModuleTestbench.run``
        keep returning one ``SpecReport`` each.
        """
        declared, settings = self._run_declarations(_monte_carlo)
        if _deterministic:
            settings = None
        axes = self._condition_axes()
        grid = self._has_condition_grid()
        if declared is None and settings is None and not grid:
            return None
        grouped_corner = any('corner' in [m for m in group if isinstance(m, str)] for group in self._groups)
        names = ([str(corner)] if corner is not None and not grouped_corner else list(declared)
                 if declared is not None else [str(self._default_corner())])
        from silk.utils.deck_text import validate_corner_name
        for name in names:
            validate_corner_name(name)
        if grid:
            self._check_sweep_sources()
            from silk.exceptions import UnknownCornerError
            registered = getattr(self.pdk_details, 'corners', {}) or {}
            for name in names:
                if registered and name not in registered:
                    raise UnknownCornerError(f'Unknown corner {name!r}; available: {list(registered)}')
        if settings is not None and not grid:
            self._check_monte_carlo(settings, names)
        if self.netlist is None and not grid:
            points = [(name, None, None, None, name, name, 'unknown', None) for name in names]
        else:
            # Legacy declared runs validate deprecated assertions during
            # preparation, where failures retain the existing run evidence.
            points = self._grid_points(names, axes,
                                       temperature if grid else None,
                                       voltage if grid else None)
        if grouped_corner and corner is not None:
            points = [point for point in points if point[0] == str(corner)]
            if not points:
                from silk.exceptions import SimulationConfigurationError
                raise SimulationConfigurationError(f'Grouped corner {corner!r} is absent from the declared rows')
        if not grid:
            # Retain scalar requests without changing legacy keys or folders.
            points = [(name, temp, supply, source, name, name, basis, option)
                      for name, temp, supply, source, _key, _folder, basis, option in points]
        if len({point[4] for point in points}) != len(points):
            from silk.exceptions import SimulationConfigurationError
            raise SimulationConfigurationError(f'Repeated whole run in {names!r}')
        if settings is None:
            if len(names) >= 2:
                self._warn_identical_corners(names, stacklevel=4)
            return [_PlannedRun(key=key, corner=name, corner_subdir=folder,
                                temperature=temp, supply=supply,
                                supply_source=source, grid=grid, temperature_basis=basis, temperature_option=option,
                                variable_values=self._point_assignments(key))
                    for name, temp, supply, source, key, folder, basis, option in points]
        point_records = {}
        if grid:
            for name, temp, supply, source, key, folder, basis, option in points:
                run = _PlannedRun(key, name, folder, temperature=temp, supply=supply,
                                  supply_source=source, grid=True, temperature_basis=basis,
                                  temperature_option=option, variable_values=self._point_assignments(key))
                with _planned_conditions(self, run):
                    self._check_monte_carlo(settings, [name])
                    point_records[key] = getattr(self, '_behavioral_mc_records', ())
        count = (f"{len(points)} point{'s' if len(points) != 1 else ''}" if grid else
                 f"{len(names)} corner{'s' if len(names) != 1 else ''}")
        print(f'Monte Carlo: {count} × {settings.n} samples = '
              f'{len(points) * settings.n} simulations')
        planned = []
        for name, temp, supply, source, point, folder, basis, option in points:
            campaign_subdir = folder if isinstance(folder, tuple) else (folder,)
            tb_dir, deck_name = self._output_paths(run_label, corner_subdir=folder)
            if settings.output_mode == 'new':
                from silk.simulation.run_files import RunFiles
                files = RunFiles.resolve(os.path.join(tb_dir, deck_name), tb_dir).prepare_new()
                tb_dir = files.directory
                campaign_subdir += (os.path.basename(tb_dir),)
            wrapper = None
            if settings.variation == 'both' and self._effective_simulator() != 'behavioral':
                wrapper = os.path.join(tb_dir, 'mc_both.lib.spice')
            if point in point_records:
                self._behavioral_mc_records = point_records[point]
            for index in range(settings.n):
                planned.append(_PlannedRun(
                    key=(point, index), corner=name,
                    corner_subdir=campaign_subdir + (f'mc_{index:04d}',),
                    sample=self._sample_spec(settings, name, index, wrapper),
                    monte_carlo=settings, temperature=temp, supply=supply,
                    supply_source=source, grid=grid, temperature_basis=basis, temperature_option=option,
                    variable_values=self._point_assignments(point)))
        return planned

    def _grid_points(self, names, axes, temperature, voltage):
        from silk.simulation._replay import apply_effective_settings
        from silk.simulation.conditions import (
            prepare_temperature, prepare_supply, _classify_supply_sweep,
        )
        from silk.exceptions import SimulationConfigurationError
        if self.netlist is None:
            raise SimulationConfigurationError(
                'Call add_module_under_test() before running condition lists.')
        # A private view validates the complete request before allocating output.
        view = copy.copy(self.netlist)
        view.simulator_options = dict(self.netlist.simulator_options)
        apply_effective_settings(self, view)
        temps = axes['temperature'][0] if axes['temperature'] is not None else None
        supplies = axes['supply'][0] if axes['supply'] is not None else None
        source = axes['supply'][1] if supplies is not None else view._supply_source
        if temps is not None:
            view.simulator_options['temp'] = temps[0]
        request = prepare_temperature(view)
        basis = 'configured-and-observed' if temps is not None else request.basis
        if temps is not None and request.sweep is not None:
            raise SimulationConfigurationError('Temperature list conflicts with DC temperature sweep.')
        if supplies is not None:
            view._supply_voltage, view._supply_source = supplies[0], source
        supply_request = prepare_supply(view, apply=False)
        from silk.circuit.variable import _leaf_variables
        if view._supply_voltage is not None and self._variable_lists:
            source_name = supply_request.source
            source_value = view.sig_sources_dict[source_name].dc
            leaves = tuple(_leaf_variables(source_value))
            conflicts = [var.name for var, _ in self._variable_lists
                         if any(leaf is var for leaf in leaves)]
            if conflicts:
                raise SimulationConfigurationError(f'Supply override conflicts with swept Variables {conflicts!r}')
        if supplies is not None and view.simulator == 'behavioral':
            from silk.simulation._setup_check import private_view
            view = private_view(view)
            view.sig_sources_dict = {name: copy.copy(source)
                                     for name, source in view.sig_sources_dict.items()}
            supply_request = _classify_supply_sweep(view, prepare_supply(view))
        if supplies is not None and supply_request.sweep is not None:
            raise SimulationConfigurationError('Supply list conflicts with DC sweep of the selected supply source.')
        temps = temps if temps is not None else (request.temperature,)
        supplies = supplies if supplies is not None else (view._supply_voltage,)
        temps = tuple(0.0 if value == 0 else value for value in temps)
        supplies = tuple(0.0 if value == 0 else value for value in supplies)
        points = []
        from silk.simulation.conditions import assert_temperature, assert_supply
        coordinates = self._grouped_coordinates(names, temps, supplies, axes)
        for row in coordinates:
            name, temp, supply = row[:3]
            named = tuple((var.name, value) for (var, _), value in zip(self._variable_lists, row[3:]))
            key = (name, temp, supply) + ((named,) if self._variable_lists else ())
            token = lambda value: 'unset' if value is None else repr(float(value))
            folder = (name, 'temperature-' + token(temp), 'supply-' + token(supply))
            for label, value in named:
                kind = 'b' if isinstance(value, bool) else 'i' if isinstance(value, int) else 'f' if isinstance(value, float) else 's'
                text = str(value).lower() if isinstance(value, bool) else str(value)
                folder += (f'variable-{label}-{kind}-{text}',)
            option = (temp if axes['temperature'] is not None and axes['temperature'][0] is not None
                      else view.simulator_options.get('temp'))
            run = _PlannedRun(key, name, folder, temperature=temp, supply=supply,
                              supply_source=supply_request.source, grid=True,
                              temperature_basis=basis, temperature_option=option,
                              variable_values=self._point_assignments(key))
            with _planned_conditions(self, run) if self._variable_lists else nullcontext():
                candidate = copy.copy(view)
                candidate.simulator_options = dict(view.simulator_options)
                if axes['temperature'] is not None and axes['temperature'][0] is not None:
                    candidate.simulator_options['temp'] = temp
                candidate._supply_voltage = supply
                assert_temperature(prepare_temperature(candidate), temperature)
                assert_supply(prepare_supply(candidate, apply=False), voltage)
            points.append((name, temp, supply, supply_request.source, key, folder, basis, option))
        return points

    def _sample_spec(self, settings, corner, index, wrapper_path):
        """The ``_SampleSpec`` of sample *index* at *corner*."""
        from silk.circuit.monte_carlo import _SampleSpec, derive_sample_seed
        if self._effective_simulator() == 'behavioral':
            return _SampleSpec(index=index,
                               seed=derive_sample_seed(settings.seed, index),
                               base_seed=settings.seed, variation=settings.variation,
                               section=None, wrapper_path=None,
                               record_draws=settings.record_draws,
                               behavioral_records=self._behavioral_mc_records)
        st = self.pdk_details.statistical
        section = None
        if settings.variation == 'mismatch':
            section = st.mismatch_sections[corner]
        elif settings.variation == 'process':
            section = st.process_section
        return _SampleSpec(index=index,
                           seed=derive_sample_seed(settings.seed, index),
                           variation=settings.variation, section=section,
                           wrapper_path=wrapper_path,
                           record_draws=settings.record_draws)

    def _effective_simulator(self):
        """The simulator a run would use, resolved as ``_prepare_run`` does."""
        if self.netlist is not None and self.netlist._simulator_explicit:
            return self.netlist.simulator
        return self.module.simulator

    def _check_behavioral_monte_carlo(self, settings, corners):
        from silk.exceptions import MonteCarloError
        from silk.simulation._behavioral_variation import active, resolve_test
        resolution = resolve_test(self)
        if not any(active(r, settings.variation) for r in resolution.records):
            raise MonteCarloError(
                f'test {self.test_name!r}: no active statistical declaration '
                f'under variation={settings.variation!r} (check mismatch enablement '
                'and nonzero spread), so every sample would be identical')
        if settings.record_draws:
            columns = {}
            for record in resolution.records:
                if not active(record, settings.variation):
                    continue
                process = record.resolved.variation == 'process'
                column = (('process', record.resolved.name) if process else
                          (record.display, record.param))
                if column in columns and columns[column] != process:
                    raise MonteCarloError(
                        f'draw column {column!r} collides between mismatch and process')
                columns[column] = process
        names = [s['name'] for s in self._specs]
        repeated = sorted({n for n in names if names.count(n) > 1})
        if repeated:
            raise MonteCarloError(
                f'test {self.test_name!r}: Monte Carlo yield is keyed by spec name, '
                f'{repeated!r} names more than one spec; give each spec its own name.')
        self._behavioral_mc_records = resolution.records
        if len(corners) >= 2:
            self._warn_identical_corners(corners, stacklevel=4)

    def _check_monte_carlo(self, settings, corners):
        """Refuse a Monte Carlo request before any deck is written.

        In order: a PDK with no ``statistical:``
        block (:class:`~silk.exceptions.NotSupportedError`, naming the PDK);
        a deck with no PDK device, two specs of one name, process variation
        off the PDK's process corners, a corner with no mismatch section
        (:class:`~silk.exceptions.MonteCarloError`); then the model
        library's section check, before the first sample
        (:func:`silk.pdk.statistical.require_library_sections`).
        """
        from silk.exceptions import MonteCarloError, NotSupportedError
        from silk.pdk import statistical as _statistical
        from silk.simulation.conditions import has_no_pdk_device
        if settings.samples_per_process > 1:
            self._check_samples_per_process(settings)
        if self._effective_simulator() == 'behavioral':
            return self._check_behavioral_monte_carlo(settings, corners)
        test = f'test {self.test_name!r}'
        simulator = self._effective_simulator()
        pdk = self.pdk_details
        pdk_name = (getattr(pdk, 'pdk_name', None) or getattr(pdk, 'name', None)
                    or self.module.pdk_name)
        st = getattr(pdk, 'statistical', None)
        if st is None:
            raise NotSupportedError(
                f"{test}: PDK {pdk_name!r} declares no statistical models "
                "(no library.statistical block in its manifest), so it "
                "cannot run Monte Carlo.")
        if self.netlist is None or has_no_pdk_device(self.netlist):
            raise MonteCarloError(
                f"{test}: Monte Carlo on a deck with no PDK device: the deck "
                "loads no model library, so every sample would be identical.")
        names = [s['name'] for s in self._specs]
        repeated = sorted({n for n in names if names.count(n) > 1})
        if repeated:
            raise MonteCarloError(
                f"{test}: Monte Carlo yield is keyed by spec name, and "
                f"{', '.join(map(repr, repeated))} names more than one spec; "
                "give each spec its own name.")
        if settings.record_draws:
            self._check_record_draws(st, test)
        if settings.variation in ('process', 'both'):
            off = [c for c in corners if c not in st.process_corners]
            if off:
                raise MonteCarloError(
                    f"{test}: variation={settings.variation!r} is valid only "
                    f"at the PDK's process corners "
                    f"({', '.join(map(repr, st.process_corners))}); got "
                    f"{', '.join(map(repr, off))}.")
        if settings.variation == 'mismatch':
            missing = [c for c in corners if c not in st.mismatch_sections]
            if missing:
                raise MonteCarloError(
                    f"{test}: PDK {pdk_name!r} declares no mismatch section "
                    f"for corner {', '.join(map(repr, missing))}; it has "
                    f"{', '.join(map(repr, st.mismatch_sections))}.")
            sections = [st.mismatch_sections[c] for c in corners]
        else:
            # 'both' loads silk's wrapper, which includes the process files;
            # the process section is the install's statistical marker.
            sections = [st.process_section]
        _statistical.require_library_sections(
            pdk, list(dict.fromkeys(sections)), simulator=simulator)

    def _chunk_admission_reason(self):
        """Why this test's decks cannot share a process, or ``None``. Decided
        before anything is written, from the declared analyses and the same
        card rules completion certification applies (no deck is emitted): a
        multi-sample process accepts a sample only when its analyses can be
        certified for completion, and the op-only case is checked against the
        emitted deck when each sample's deck is made."""
        from silk.simulation.analysis import Op
        from silk.simulation.completion import uncertifiable_reason
        analyses = list(self.netlist.sim_analysis_dict.values())
        if not analyses:
            return 'it declares no analysis to check'
        if all(type(a) is Op for a in analyses):
            return None
        reason = uncertifiable_reason(analyses)
        return None if reason is None else f'its analyses cannot be certified for completion: {reason}'

    def _check_samples_per_process(self, settings):
        """Refuse ``samples_per_process > 1`` where several samples cannot
        share one ngspice process, before any deck is written: any simulator
        but ngspice (:class:`~silk.exceptions.NotSupportedError`), content
        the multi-sample deck cannot reproduce or certify
        (:class:`~silk.exceptions.NotSupportedError`) and
        ``save_device_currents`` (:class:`~silk.exceptions.MonteCarloError`).
        Nothing falls back to one process per sample."""
        from silk.exceptions import MonteCarloError, NotSupportedError
        test = f'test {self.test_name!r}'
        simulator = self._effective_simulator()
        n = settings.samples_per_process
        if simulator != 'ngspice':
            raise NotSupportedError(
                f"{test}: samples_per_process={n} needs ngspice, the only "
                f"simulator that can run several Monte Carlo samples in one "
                f"process; this test runs on {simulator!r}. Use "
                f"samples_per_process=1.")
        if self.module.save_device_currents:
            raise MonteCarloError(
                f"{test}: samples_per_process={n} and save_device_currents "
                "contradict each other: save_device_currents forces "
                "batch-write mode, which emits no .control block for the "
                "multi-sample loop. Clear one of them.")
        if self.netlist is None:
            return
        from silk.simulation.completion import _opaque_emission_reason
        reason = _opaque_emission_reason(self.netlist)
        # The module's flag is what preparation copies onto the netlist.
        if reason is None and self.module.save_device_params:
            reason = 'save_device_params'
        if reason is None and getattr(self.netlist.pdk_details, 'osdi_files', None):
            reason = 'a PDK with OSDI models'
        if reason is not None:
            raise NotSupportedError(
                f"{test}: samples_per_process={n} cannot run this test: "
                f"{reason} is not supported in a multi-sample process, "
                f"because silk cannot prove the multi-sample deck runs the "
                f"same analyses as the single-sample deck. Use "
                f"samples_per_process=1.")

    def _check_record_draws(self, statistical, test):
        """Refuse ``record_draws=True`` where it cannot record:
        on ngspice with ``save_device_currents`` (no control block for the
        ``showmod`` lines), and on any simulator a deck with no device type
        the manifest records. Xyce reads its ``.res`` file and needs neither
        a control block nor the ngspice addresses."""
        from silk.exceptions import MonteCarloError
        if self._effective_simulator() == 'xyce':
            from silk.circuit.modules_results import xyce_draw_name_map
            name_map, _, skipped = xyce_draw_name_map(
                self.netlist.elem_dict, statistical.record)
            recorded = bool(name_map)
        else:
            from silk.circuit.monte_carlo import draw_devices
            if self.module.save_device_currents:
                raise MonteCarloError(
                    f"{test}: record_draws=True and save_device_currents "
                    "contradict each other: save_device_currents forces "
                    "batch-write mode, which emits no .control block, and "
                    "the drawn parameters are read by showmod lines in that "
                    "block. Clear one of them.")
            devices, skipped = draw_devices(self.netlist.elem_dict,
                                            statistical)
            recorded = bool(devices)
        if not recorded:
            raise MonteCarloError(
                f"{test}: record_draws=True, but no device in this deck has a "
                f"statistical record entry in the PDK manifest (device types "
                f"here: {', '.join(map(str, skipped)) or 'none'}), so there "
                f"is nothing to record.")

    def _warn_identical_corners(self, distinct, *, stacklevel):
        """Warn when process corners do not affect the selected simulation."""
        if self.netlist is None:
            return
        from silk.simulation.conditions import has_no_pdk_device
        behavioral = self._effective_simulator() == 'behavioral'
        if behavioral or has_no_pdk_device(self.netlist):
            reason = ('behavioral models do not use process corners' if behavioral
                      else 'this deck instantiates no PDK device')
            # Not refused (design 2026-10-01 section 3.1, revised after
            # review): each corner keeps its label and records corner none.
            warnings.warn(
                f'test {getattr(self, "test_name", None)!r}: {reason}, '
                f'so the {len(distinct)} corner runs '
                f'({", ".join(distinct)}) are identical; each records '
                'corner none and keeps its corner only as a label.',
                UserWarning, stacklevel=stacklevel)

    def _prepare_plan(self, planned, *, temperature=None, voltage=None,
                      run_label=None):
        """Prepare every planned run in this process, in plan order.

        Each prepared netlist is deep-copied before the next run's
        ``_prepare_run`` mutates the live one again, as :meth:`run_corners`
        does. Returns ``(prepared, errors)``: key -> ``_RunSpec`` (or a
        cached ``SpecReport``), and key -> formatted error for a run whose
        preparation failed. A :class:`~silk.exceptions.ConfigurationError`
        propagates: the request is wrong for every run.
        """
        import traceback
        from silk.exceptions import ConfigurationError
        prepared, errors = {}, {}
        for run in planned:
            try:
                with _planned_conditions(self, run):
                    spec = self._prepare_run(
                        run.corner, temperature=temperature, voltage=voltage,
                        run_label=run_label, corner_subdir=run.corner_subdir,
                        sample=run.sample)
                    if isinstance(spec, _RunSpec):
                        spec = replace(spec, variable_values=run.variable_values)
                        from silk.circuit.test_execution import current_invocation
                        coordinator = current_invocation()
                        if coordinator is not None:
                            spec = coordinator.prepare(spec)
                        from silk.simulation._worker_pickle import copy_simulation_netlist
                        spec = replace(spec, netlist=copy_simulation_netlist(spec.netlist))
            except ConfigurationError:
                raise
            except Exception as exc:
                errors[run.key] = (f'{type(exc).__name__}: {exc}\n'
                                   f'{traceback.format_exc()}')
                continue
            prepared[run.key] = spec
        return prepared, errors

    def _evaluate_plan(self, planned, prepared, outcomes, errors):
        """Evaluate every simulated planned run in this process, in plan
        order; return key -> ``SpecReport``.

        *outcomes* maps keys to shared simulation outcomes. Configuration
        errors propagate before evaluation. Ordinary errors are collected;
        the shared consumer applies their actual replacement transition.
        """
        import traceback
        from silk.exceptions import ConfigurationError
        for outcome in outcomes.values():
            _reconcile_run_outcome(self, outcome)
        for run in planned:
            outcome = outcomes.get(run.key)
            if outcome is not None and isinstance(outcome.error, ConfigurationError):
                self._consume_run_outcome(outcome)
        reports = {}
        for run in planned:
            if run.key not in prepared:
                continue
            spec = prepared[run.key]
            if not isinstance(spec, _RunSpec):
                reports[run.key] = spec    # cache hit; already evaluated
                self._metric_values = dict(spec.metrics)
                continue
            outcome = outcomes[run.key]
            try:
                with _planned_conditions(self, run):
                    if run.grid:
                        _configure_conditions(self, self.netlist)
                    reports[run.key] = self._consume_run_outcome(outcome)
                    if run.variable_values:
                        reports[run.key].parameters['silk_run_variables'] = {
                            var.name: value for var, value in spec.variable_values}
            except ConfigurationError:
                raise
            except Exception as exc:
                errors[run.key] = outcome.error_text or (
                    f'{type(exc).__name__}: {exc}\n{traceback.format_exc()}')
        return reports

    def _collect_runs(self, planned, reports, errors, *, rerun_config=None,
                      replay_signature=None, campaign_source=None, draw_scheme=None,
                      rerun_samples=()):
        """The :class:`~silk.circuit.spec_report.CornerResults` of a plan.

        *reports*: key -> ``SpecReport``; *errors*: key -> formatted error.
        Sets the bench's sweep table to the successful runs. A failed run is
        kept out of the reports and listed in ``.failed``;
        the results skip failures in ``stat``/``worst`` with a counted
        warning.

        A Monte Carlo plan returns a
        :class:`~silk.circuit.monte_carlo.MonteCarloResults` instead: its
        ``failed`` maps key -> ``SampleFailure`` (error and seed), it records
        the circuit's structure hash, and its ``rerun`` replays one sample
        with the ``run()`` keywords in *rerun_config*. The sweep table then
        holds one ``'<corner>/mc_NNNN'`` entry per sample, which is what
        ``Signal.sample`` reads.

        Raises:
            SimulationError: every planned run failed; the message carries
                the first failure and the number of others. Its ``failed``
                attribute retains the full ordered failure map.
        """
        from silk.circuit.spec_report import CornerResults
        from silk.circuit.monte_carlo import SampleFailure
        order = [run.key for run in planned]
        failed = {key: errors[key] for key in order if key in errors}
        samples = [run for run in planned if run.sample is not None]
        inventory = [dict(key=run.key, corner=run.corner, temperature=run.temperature,
                          supply=run.supply, supply_source=run.supply_source)
                     for run in planned]
        if self._variable_lists or self._groups:
            groups = [[member if isinstance(member, str) else member.name for member in group]
                      for group in self._groups]
            for record, run in zip(inventory, planned):
                record.update(variables=[[var.name, value] for var, value in run.variable_values], groups=groups)
        checks = [dict(key=item['spec'].key if item.get('spec') else item['name'],
                       name=item['name'], metric=item['metric'], op=item['op'],
                       target=item['target'], owner_path=self.owner_path) for item in self._specs]
        metadata = dict(run_inventory=inventory,
                        metric_names=_metric_output_declarations(self._metrics)[1],
                        check_declarations=checks)

        if not reports:
            if current_placement(self) is None:
                self._sweep_table = {}
            first = next(iter(failed))
            raise SimulationError(
                f"test {self.test_name!r}: every declared run failed "
                f"({len(failed)} of {len(order)}); the first, {first!r}: "
                f"{failed[first]}; {len(failed) - 1} other failure(s)",
                failed=(
                    {run.key: SampleFailure(failed[run.key], run.sample.seed)
                     for run in samples}
                    if samples else failed))
        from silk.circuit.test_placement import retain_plan
        if not samples:
            self._sweep_table = _retained_sweep(reports)
            self._sweep_generations = {key: getattr(report, '_raw_generation', None)
                                       for key, report in reports.items()}
            retain_plan(self, reports)
            return CornerResults(reports, failed=failed, skip_unavailable=True, **metadata)
        from silk.circuit.monte_carlo import MonteCarloResults, SampleFailure
        seeds = {run.key: run.sample.seed for run in samples}
        retained = {key: reports[key].rawfile_path for key in order
                    if key in reports
                    and getattr(reports[key], 'rawfile_path', None) is not None}
        self._sweep_generations = {_sample_label(key): getattr(reports[key], '_raw_generation', None)
                                   for key in retained}
        self._sweep_table = {_sample_label(key): path
                             for key, path in retained.items()}
        self._mc_samples = retained
        retain_plan(self, reports, samples=True)
        settings = samples[0].monte_carlo
        draws = None
        if settings.record_draws:
            draws = self._draw_table(samples, reports, seeds,
                                     (rerun_config or {}).get('run_label'))
        return MonteCarloResults(
            {key: reports[key] for key in order if key in reports},
            settings=settings,
            seeds=seeds,
            failed={key: SampleFailure(failed[key], seeds[key])
                    for key in failed},
            structure_hash=self._structure_hash(),
            rerun_hook=self._rerun_hook(samples, dict(rerun_config or {}),
                                        replay_signature),
            draws=draws, campaign_source=campaign_source, draw_scheme=draw_scheme,
            rerun_samples=rerun_samples,
            spec_declarations=self._specs, **metadata)

    def _draw_table(self, samples, reports, seeds, run_label):
        """The run's :class:`~silk.circuit.monte_carlo_draws.DrawTable`, written
        beside the corner's sample folders (inside its campaign folder in new
        mode): one row per sample that produced a report.

        ``raw`` follows the samples' simulator: actual values from ngspice
        (``True``), normalised draws from Xyce (``False``). The two cannot
        share one table's standardisation honestly, so a run whose samples
        carry both raises :class:`~silk.exceptions.MonteCarloError`.
        """
        from silk.circuit.monte_carlo import (
            build_draw_table, draw_devices, write_draws_csv)
        from silk.exceptions import MonteCarloError
        recorded = [reports[run.key] for run in samples if run.key in reports]
        kinds = {getattr(r, '_mc_draws_raw', None) for r in recorded}
        if len(kinds) > 1:
            raise MonteCarloError(
                f"test {self.test_name!r}: this Monte Carlo run holds both "
                f"actual parameter values (ngspice) and normalised draws "
                f"(Xyce); they cannot be ranked as one table.")
        raw = kinds.pop() if kinds else True
        if self._effective_simulator() == 'behavioral':
            skipped = ()
        elif raw is False:
            statistical = self.pdk_details.statistical
            from silk.circuit.modules_results import xyce_draw_name_map
            _, _, skipped = xyce_draw_name_map(self.netlist.elem_dict,
                                               statistical.record)
        else:
            statistical = self.pdk_details.statistical
            _, skipped = draw_devices(self.netlist.elem_dict, statistical)
        table = build_draw_table(
            {run.key: reports[run.key]._mc_draws for run in samples
             if run.key in reports}, sorted(map(str, skipped)), raw=raw)
        for point in dict.fromkeys(run.point for run in samples):
            sample = next(run for run in samples if run.point == point)
            tb_dir, _ = self._output_paths(
                run_label, corner_subdir=sample.corner_subdir[:-1])
            os.makedirs(tb_dir, exist_ok=True)
            write_draws_csv(
                os.path.join(tb_dir, 'draws.csv'),
                replace(table, values={k: v for k, v in table.values.items()
                                       if k[0] == point}),
                seeds)
        return table

    def _structure_hash(self, *, _netlist=None):
        """One digest of the device under test and the testbench around it:
        retained as metadata on Monte Carlo results, or None for opaque sources.
        This is separate from the DUT-only persistent structure identity.
        """
        import hashlib
        from silk.circuit.circuit_elements import _module_structure_projection
        netlist = self.netlist if _netlist is None else _netlist
        modules = [self.module] + ([netlist] if netlist is not None else [])
        projections = [_module_structure_projection(module) for module in modules]
        if any(not provable for _digest, provable in projections):
            return None
        parts = [digest for digest, _provable in projections]
        return hashlib.sha256('/'.join(parts).encode()).hexdigest()

    def _rerun_hook(self, samples, config, recorded):
        """``rerun_hook(key, seed) -> SpecReport`` for the planned *samples*:
        re-prepare that sample's deck with its recorded seed in its own
        folder, simulate and evaluate it. Refuses changed managed inputs
        before writing either a wrapper or a sample file."""
        from silk.exceptions import MonteCarloError
        by_key = {run.key: run for run in samples}
        bench = weakref.ref(self)
        placement = current_placement(self)

        def rerun(key, seed):
            tb = bench()
            if tb is None:
                raise MonteCarloError(
                    "rerun(): the testbench these results came from no "
                    "longer exists.")
            from silk.circuit.test_placement import execution_scope, unselected_scope
            entry = placement.entry if placement is not None else None
            if entry is not None:
                from silk.circuit.test_placement import authored_test
                root = entry._root_ref()
                live = next((candidate for candidate in root.list_test_entries()
                             if candidate.owner_path == entry.owner_path
                             and authored_test(candidate.test) is tb), None) if root else None
                if live is None:
                    raise MonteCarloError(
                        f"rerun({key!r}): changed placement; the selected test "
                        "was removed or replaced. Run the campaign again.")
                entry = live
            binding = execution_scope(entry, executing=True) if entry is not None else unselected_scope(tb)
            with binding:
                run = by_key.get(key)
                if run is None:
                    raise MonteCarloError(f"rerun(): unknown sample {key!r}")
                from silk.simulation._replay import experiment_signature, changed_inputs
                current = experiment_signature(tb, run.monte_carlo, config)
                changed = list(changed_inputs(recorded, current))
                if seed != run.sample.seed:
                    changed.append('sample seed')
                if changed:
                    raise MonteCarloError(
                        f"rerun({key!r}): changed " + '; '.join(changed)
                        + "; run the campaign again.")
                sample = run.sample
                if sample.wrapper_path:
                    _write_wrapper(tb.pdk_details, sample.wrapper_path)
                with _planned_conditions(tb, run):
                    spec = tb._prepare_run(run.corner, corner_subdir=run.corner_subdir,
                                           sample=sample, **config)
                    report = tb._consume_run_outcome(
                        _simulate_run_outcome(spec, tb._simulate_run))
                    if run.variable_values:
                        report.parameters['silk_run_variables'] = {
                            var.name: value for var, value in run.variable_values}
                    return report
        return rerun

    def _run_plan(self, planned, *, temperature=None, voltage=None,
                  run_label=None, n_workers=None):
        """Run a declared plan: prepare, simulate through the shared
        executor, evaluate in plan order, collect.

        A corner plan simulates one run at a time (``run()``). Monte Carlo
        samples go in batches of the resolved worker count (one for
        ``run()``): each batch is prepared, and its netlists deep-copied,
        only when it is next, so memory holds one batch of netlists rather
        than one per sample. The both-variation wrapper is written once per
        corner folder before the first batch.
        """
        from silk.simulation.executor import resolve_n_workers, run_tasks
        config = {'temperature': temperature, 'voltage': voltage,
                  'run_label': run_label}
        samples = any(run.sample is not None for run in planned)
        campaign_backend = self._effective_simulator() if samples else None
        campaign_source = ('behavioral' if campaign_backend == 'behavioral' else 'spice') if samples else None
        from silk.simulation.variation import DRAW_SCHEME
        draw_scheme = DRAW_SCHEME if campaign_backend == 'behavioral' else None
        replay_signature = None
        if samples:
            from silk.simulation._replay import experiment_signature
            replay_signature = experiment_signature(self, planned[0].monte_carlo, config)
            replay_signature.require_valid()
        per_process = planned[0].monte_carlo.samples_per_process if samples else 1
        if per_process > 1:
            reason = self._chunk_admission_reason()
            if reason is not None:
                import warnings
                warnings.warn(
                    f"test {self.test_name!r}: samples_per_process={per_process} "
                    f"is a ceiling and this test cannot use it ({reason}); "
                    f"running one ngspice process per sample.",
                    RuntimeWarning, stacklevel=2)
                per_process = 1
        workers = size = 1
        if samples:
            workers = size = resolve_n_workers(
                n_workers, backend=self._effective_simulator())
            for path in dict.fromkeys(run.sample.wrapper_path
                                      for run in planned
                                      if run.sample.wrapper_path):
                _write_wrapper(self.pdk_details, path)
        else:
            size = max(len(planned), 1)
        reports, errors = {}, {}
        if per_process > 1:
            return self._run_chunked(planned, workers, per_process, config,
                                     replay_signature, campaign_source, draw_scheme)
        for start in range(0, len(planned), size):
            batch = planned[start:start + size]
            prepared, batch_errors = self._prepare_plan(batch, **config)
            keys, tasks = _plan_tasks(batch, prepared)
            outcomes = {key: _planned_outcome(o, prepared[key]) for key, o in
                        zip(keys, run_tasks(tasks, n_workers=workers))}
            reports.update(self._evaluate_plan(batch, prepared, outcomes,
                                               batch_errors))
            errors.update(batch_errors)
        return self._collect_runs(planned, reports, errors,
                                  rerun_config=config, replay_signature=replay_signature,
                                  campaign_source=campaign_source, draw_scheme=draw_scheme)

    def _run_chunked(self, planned, workers, per_process, config,
                     replay_signature, campaign_source, draw_scheme):
        """Run Monte Carlo samples ``per_process`` to an ngspice process. A chunk is up to *per_process* consecutive samples of
        one corner and one :class:`~silk.simulation.executor.RunTask`;
        *workers* chunks run at a time. Evaluation, collection and result
        order are those of the one-process-per-sample path."""
        from silk.simulation.executor import RunOutcome, RunTask, run_tasks
        chunks = []
        for run in planned:
            if (chunks and chunks[-1][0].point == run.point
                    and len(chunks[-1]) < per_process):
                chunks[-1].append(run)
            else:
                chunks.append([run])
        reports, errors, rerun_samples = {}, {}, []
        for start in range(0, len(chunks), workers):
            group = chunks[start:start + workers]
            batch = [run for chunk in group for run in chunk]
            prepared, batch_errors = self._prepare_plan(batch, **config)
            tasks, task_keys = [], []
            for chunk in group:
                keys = [run.key for run in chunk
                        if isinstance(prepared.get(run.key), _RunSpec)]
                if not keys:
                    continue
                specs = [prepared[key] for key in keys]
                chunk_dir = os.path.join(os.path.dirname(specs[0].tb_dir),
                                         f'.mc_chunk_{chunk[0].sample.index:04d}')
                tasks.append(RunTask(fn=_chunk_simulate_task, run_dir=chunk_dir,
                                     args=(specs, chunk_dir), simulator='ngspice'))
                task_keys.append(keys)
            outcomes = {}
            for keys, result in zip(task_keys, run_tasks(tasks, n_workers=workers)):
                if result.error is not None:
                    for key in keys:
                        outcomes[key] = _planned_outcome(result, prepared[key])
                    continue
                member_outcomes, rerun = result.value
                for key, member in zip(keys, member_outcomes):
                    outcomes[key] = _planned_outcome(
                        RunOutcome(index=result.index, run_dir=result.run_dir,
                                   value=member), prepared[key])
                rerun_samples += [(keys[position], why) for position, why in rerun]
            reports.update(self._evaluate_plan(batch, prepared, outcomes,
                                               batch_errors))
            errors.update(batch_errors)
        if rerun_samples:
            import warnings
            detail = '; '.join(f'{c}/{i} ({why})' for (c, i), why in rerun_samples)
            warnings.warn(
                f"test {self.test_name!r}: {len(rerun_samples)} Monte Carlo "
                f"sample(s) did not finish in a multi-sample process (its "
                f"process left them unfinished, or they fell back before one "
                f"launched) and were rerun one per process: {detail}. See "
                f"MonteCarloResults.rerun_samples.", RuntimeWarning, stacklevel=2)
        rerun_samples = [key for key, _ in rerun_samples]
        return self._collect_runs(planned, reports, errors,
                                  rerun_config=config, replay_signature=replay_signature,
                                  campaign_source=campaign_source, draw_scheme=draw_scheme,
                                  rerun_samples=rerun_samples)

    def _check_sweep_sources(self):
        """Refuse a DC or Noise analysis whose source this bench lacks.

        The earliest point where both the sources and the analyses are known:
        before any file is written and before the netlist is built.
        """
        from silk.circuit.sweep_sources import sweep_source_problem
        from silk.exceptions import SweepSourceError
        from silk.simulation.analysis import DC, Noise
        names = list(self.netlist.sig_sources_dict)
        for a in self.netlist.sim_analysis_dict.values():
            if not isinstance(a, (DC, Noise)):
                continue
            problem = sweep_source_problem(
                a.source, names, allow=('temp',) if isinstance(a, DC) else ())
            if problem:
                raise SweepSourceError(
                    f"test {self.test_name!r}: {type(a).__name__}"
                    f"(source={a.source!r}): source {problem}")

    def _prepare_run(self, corner=None, *, temperature=None, voltage=None,
                     run_label=None, corner_subdir=None, sample=None,
                     use_cache=True):
        """Resolve and validate a ``run()`` invocation without simulating.

        The first third of ``run()``: everything it always did
        before driving the simulator — corner/module/netlist checks,
        inherited temperature/supply/options resolution, the cache lookup,
        and (on a miss) making the deck-ready netlist state (directory,
        deck file name, output mode, device-current capture, declared
        supply) exactly as ``run()`` set it before this task.

        ``corner_subdir``, keyword-only and private to :meth:`run_corners`: when given, nests the deck under
        ``<testbench dir>/<corner_subdir>/`` instead of directly in the
        testbench directory, so every corner of a sweep always gets its own
        folder regardless of output mode. ``run()`` never passes this — its
        own single-run directory selection is unchanged. A tuple nests one
        folder per component (a Monte Carlo sample's ``(corner, 'mc_NNNN')``).

        ``sample``, keyword-only and private to the run plan: a
        Monte Carlo ``_SampleSpec``, set as ``netlist._mc_sample`` so the deck
        loads its statistical section and seed. Every other run sets
        ``None``, so a later plain deck carries neither. A sample always
        writes in overwrite mode: its folder is its identity.

        ``use_cache=False`` is the public :meth:`prepare` path: resolve the
        same deck state without consulting or changing recorded evidence.

        Returns
        -------
        SpecReport
            On a cache hit — the same short-circuit ``run()`` took before.
        _RunSpec
            On a miss — a picklable spec ready for :meth:`_simulate_run`.
        """
        if corner is None:
            corner = self._default_corner()
        # Before any output path is selected or deck written (TASK-593).
        from silk.utils.deck_text import validate_corner_name
        validate_corner_name(corner)
        if not self._module_added:
            raise ValueError(f"Must call add_module_under_test() before running test {self.test_name}")
        if self.netlist is None:
            raise ValueError(f"Netlist not created for test {self.test_name}")

        self._check_sweep_sources()
        # Validate before changing current evidence or touching output files.
        tb_dir, deck_name = self._output_paths(run_label, corner_subdir=corner_subdir)
        corner_str = str(corner)
        # Match Module.simulate's validation before either public path writes
        # a deck. External preparation never reaches simulate().
        corners = getattr(self.netlist.pdk_details, 'corners', {}) or {}
        if corners and corner_str not in corners:
            raise ValueError(
                f'unknown corner {corner_str!r}; available: {sorted(corners)}')
        from silk.simulation.conditions import (
            assert_supply,
            assert_temperature, confirm_conditions,
            prepare_supply,
            prepare_temperature, validate_supply,
        )
        from silk.exceptions import SimulationConfigurationError
        supply = validate_supply(voltage)
        request, supply_request = _configure_conditions(
            self, self.netlist, temperature=temperature, voltage=supply)
        invocation = (request, supply_request.supply_identity, supply)
        cache_key = None
        if use_cache and self._cache_enabled:
            cache_key = (self._fingerprint(), corner_str)
            if cache_key in self._cache:
                # A label is not part of the circuit fingerprint. Reusing
                # evidence must not claim new files or relabel its context.
                cached = self._cache[cache_key]
                if (getattr(cached, '_operating_invocation', None) != invocation
                        or cached.operating_conditions is None):
                    # Both refusals here are configuration errors (TASK-163):
                    # the fix is 'call clear_cache()', i.e. change the script.
                    # SimulationConfigurationError subclasses SimulationError,
                    # so callers catching that keep catching these.
                    raise SimulationConfigurationError(
                        'Cached conditions differ or lack provenance. Call '
                        'clear_cache() and rerun before requesting new conditions.')
                if temperature is not None or supply is not None:
                    confirm_conditions(cached.operating_conditions,
                                       temperature=temperature, voltage=supply,
                                       error_type=SimulationConfigurationError)
                # A cache hit evaluates nothing, so the table is rewritten
                # here instead (TASK-218): this corner, and the file the
                # cached report was extracted from. The file is whatever is
                # on disk at that path now — in overwrite mode a later run
                # has replaced it, which is the same caveat the whole mode
                # carries and why run_corners() collapses shared paths.
                cached_path = getattr(cached, 'rawfile_path', None)
                self._sweep_table = ({} if cached_path is None
                                     else {corner_str: cached_path})
                self._metric_values = dict(cached.metrics)
                return cached

        os.makedirs(tb_dir, exist_ok=True)
        selection = current_placement(self)
        mode = 'overwrite' if sample is not None else (
            selection.output_mode if selection is not None else self.module._output_mode)
        location = (tb_dir, os.path.join(tb_dir, deck_name), mode)
        if selection is not None:
            selection.location = location
            selection.declared_supply = supply
        else:
            self.netlist.dir_folder, self.netlist.netlistname = tb_dir, deck_name
            self.netlist.set_output_options(mode=mode)
        from silk.simulation._behavioral_variation import VariationScope
        self.netlist._variation_scope = VariationScope(
            owner_path=self.owner_path, test_name=self.test_name,
            dut_instance=DUT_ELEMENT_NAME)
        self.netlist._mc_sample = sample
        _configure_capture(self, self.netlist)
        # Per-invocation declaration, consumed by the one launch it scopes.
        if selection is None:
            self.netlist._declared_supply = supply
        from silk.circuit.test_execution import sharing_identity
        return _RunSpec(netlist=self.netlist, corner_str=corner_str,
                        tb_dir=tb_dir, invocation=invocation,
                        test_name=self.test_name, cache_key=cache_key,
                        location=location, declared_supply=supply,
                        sharing_identity=sharing_identity(self, selection))

    def clear_cache(self):
        """Clear all cached simulation results."""
        self._cache = {}

    def run_corners(self, corners=None, *, temperature=None, voltage=None,
                    n_workers=None):
        """Run simulation across multiple corners, optionally in parallel.

        Every corner ALWAYS gets its own output folder,
        ``<testbench dir>/<corner>/`` — regardless of the live owner's
        output mode and regardless of ``n_workers`` — so every corner's
        ``report.rawfile_path`` stays valid after the sweep, and no two
        corners ever share a ``.cir``/``.raw``/``.log`` path (**BREAKING**:
        a prior release wrote every corner's default/overwrite-mode output
        to the same unlabeled path).

        Every corner is prepared and evaluated in THIS process, in the
        order *corners* was given; ``n_workers`` only changes how the
        simulator launches themselves are scheduled (via
        ``silk.simulation.executor.run_tasks``) — so a serial
        (``n_workers=1``) and a parallel sweep of the same benches produce
        identical :class:`~silk.circuit.spec_report.SpecReport` content,
        deck bytes and rawfiles (paths aside).

        A corner whose prepare, simulate or evaluate step raises does not
        stop the sweep — every other corner still runs to completion — but
        once every corner has been attempted,
        :class:`silk.circuit.spec_report.CornerRunError` is raised naming
        which corner(s) failed and why; the successful corners' reports are
        still reachable from its ``.results``.

        Parameters
        ----------
        corners : iterable of str, optional
            Corner names to simulate. An explicitly empty iterable returns an
            empty CornerResults and preserves retained evidence. When omitted
            or None, defaults to the testbench PDK's ``kind: process`` corners
            (from the manifest),
            falling back to ``('tt', 'ff', 'ss', 'sf', 'fs')`` if the PDK
            has none.
        temperature, voltage : optional
            Forwarded to each corner's run — see ``run()`` for the
            configured-temperature assertion and declared-supply contract.
        n_workers : None, int, or 'all', optional
            ``None`` or ``1`` (the default) simulates one corner at a time
            in this process — the historical behaviour. An int runs up to
            that many corners' simulators at once, capped at the CPU count;
            ``'all'`` uses every CPU. Each worker is a separate OS process
            with its own parsed PDK, so size this by available memory as
            well as CPU count (see
            ``silk.simulation.executor.resolve_n_workers``). Forced to
            1 regardless of request for the behavioral backend.

        Returns
        -------
        CornerResults
            Dict-like object mapping corner name → :class:`~silk.circuit.spec_report.SpecReport`.
            ``result.passed`` is True only if every corner passed all specs.

        Raises
        ------
        CornerRunError
            At least one corner failed; every corner was still attempted
            and every other corner's report is on the error's ``.results``.

        Examples
        --------
        ::

            results = tb.run_corners()
            if not results.passed:
                for corner, report in results.items():
                    if not report.passed:
                        print(f"{corner}: {report.summary()}")
        """
        import traceback
        from silk.circuit.spec_report import CornerResults, CornerRunError
        from silk.simulation.executor import RunTask, resolve_n_workers, run_tasks
        if corners is None:
            pdk_corners = getattr(self.pdk_details, 'corners', {}) or {}
            corners = tuple(c for c, meta in pdk_corners.items()
                            if meta.get('kind') == 'process') \
                      or ('tt', 'ff', 'ss', 'sf', 'fs')
        corners = list(corners)
        distinct = list(dict.fromkeys(str(c) for c in corners))
        if len(distinct) >= 2:
            self._warn_identical_corners(distinct, stacklevel=3)
        # Only forward temperature/voltage when the caller actually supplied
        # at least one — keeps run_corners()'s call to run() identical to
        # its pre-CONTEXT form (corner= only) in the common unscoped case.
        pvt_kwargs = (
            {'temperature': temperature, 'voltage': voltage}
            if temperature is not None or voltage is not None
            else {}
        )
        workers = resolve_n_workers(n_workers, backend=self._effective_simulator())

        # --- Prepare every corner in THIS process (owner ruling). Each
        # corner's netlist state is deep-copied immediately after preparing
        # it, before the next corner's _prepare_run() mutates the one live
        # netlist (self.netlist) again for the corner after that —
        # otherwise every corner's _RunSpec would end up aliasing whichever
        # corner was prepared LAST by the time anything actually reads it.
        prepared = {}   # corner -> _RunSpec or SpecReport (cache hit)
        errors = {}       # corner -> formatted error string
        order = []          # corners that prepared successfully, request order
        for corner in corners:
            try:
                spec = self._prepare_run(corner, corner_subdir=str(corner),
                                         **pvt_kwargs)
                if isinstance(spec, _RunSpec):
                    from silk.simulation._worker_pickle import copy_simulation_netlist
                    spec = replace(spec, netlist=copy_simulation_netlist(spec.netlist))
                    # This route executes the prepared scalar deck. Plan-only
                    # declarations remain on the authored bench, not its copy.
                    vars(spec.netlist).pop('_temperature_list', None)
                    vars(spec.netlist).pop('_supply_list', None)
            except Exception as exc:
                errors[corner] = f'{type(exc).__name__}: {exc}\n{traceback.format_exc()}'
                continue
            prepared[corner] = spec
            order.append(corner)

        # --- Simulate every prepared corner through the shared executor
        # (serial in-process for n_workers<=1, a pool otherwise). A raising
        # task is captured as that corner's own RunOutcome.error; it never
        # stops the others (TASK-301.1).
        to_simulate = [c for c in order if isinstance(prepared[c], _RunSpec)]
        tasks = [RunTask(fn=_corner_simulate_task, run_dir=prepared[c].tb_dir,
                         args=(prepared[c],), simulator=prepared[c].netlist.simulator)
                 for c in to_simulate]
        outcomes = run_tasks(tasks, n_workers=workers)
        outcome_by_corner = dict(zip(to_simulate, outcomes))

        # --- Evaluate every simulated corner in THIS process, one at a
        # time, in request order (owner ruling) — so serial and parallel
        # sweeps make the identical sequence of self.netlist mutations and
        # cache writes.
        reports = {}
        for corner in order:
            spec_or_report = prepared[corner]
            if not isinstance(spec_or_report, _RunSpec):
                reports[corner] = spec_or_report   # cache hit; already evaluated
                self._metric_values = dict(spec_or_report.metrics)
                continue
            outcome = _planned_outcome(outcome_by_corner[corner], spec_or_report)
            try:
                reports[corner] = self._consume_run_outcome(outcome)
            except Exception as exc:
                errors[corner] = outcome.error_text or (
                    f'{type(exc).__name__}: {exc}\n{traceback.format_exc()}')

        # Every corner that succeeded has completed, so the bench's sweep is
        # now the whole successful set rather than the last run's single
        # entry (TASK-218). Built here rather than incrementally so that
        # while the sweep is still running each run()'s own evaluation sees
        # the unambiguous one-entry table it has always seen.
        if corners and (reports or current_placement(self) is None):
            self._sweep_table = _retained_sweep(reports)
            self._sweep_generations = {str(key): getattr(report, '_raw_generation', None)
                                       for key, report in reports.items()}
            from silk.circuit.test_placement import retain_plan
            retain_plan(self, reports)
        results = CornerResults(reports)
        if errors:
            raise CornerRunError(errors, results)
        return results
