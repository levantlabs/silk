"""Generate testbench Python from a ``TestbenchSpec``.

Parallel to ``silk.codegen.generator`` (``CircuitSpec`` to circuit Python).
silk owns every emitted testbench API line; the caller builds the spec. The
dict format is described in ``docs/codegen-contract.md``.
"""
import keyword
import math
from dataclasses import dataclass, field
from numbers import Real


class TestbenchSpecError(ValueError):
    """Raised by generate_testbenches on a structurally invalid TestbenchSpec.
    silk never emits broken/uncompilable Python."""
    __test__ = False  # not a pytest test class (name starts with 'Test')


@dataclass
class TestbenchSpec:
    """A declarative description of the testbenches to attach to a circuit.

    Passed to :meth:`silk.codegen.Generator.generate_testbenches`, which
    accepts either an instance or the equivalent dict. The ``rigs`` and
    ``tests`` entries are plain dicts whose keys are specified in
    ``docs/codegen-contract.md``; they are validated when code is
    generated, not at construction.

    Attributes
    ----------
    circuit_name : str
        Name of the circuit under test. When ``circuit_builder`` is set it
        names the generated ``build_<circuit_name>_with_tests`` function and
        must be a Python identifier.
    circuit_builder : str or None
        Name of the caller's circuit-builder function. ``None`` or empty emits
        no assembler function.
    rigs : list of dict
        Shared stimulus groups; each has a ``'name'`` and may list
        ``sources``, ``devices``, ``ideal_elements`` and ``submodules``. Each
        becomes an ``_apply_<name>_rig`` helper.
    tests : list of dict
        One entry per testbench, with ``'name'``, ``'dut_portmap'``,
        ``'applied_rigs'``, optional ``sources``, ``analyses``, ``metrics``,
        ``specs``, ``corner`` and ``temperature_c`` (degrees Celsius), among
        others. Each becomes an ``_attach_<name>`` function.
    save_device_params : bool
        Turn on operating-point device-parameter capture on the circuit
        under test, so an operating-point table can be reported. Defaults to
        ``False``.

    Examples
    --------
    >>> from silk.codegen.testbench_generator import TestbenchSpec
    >>> spec = TestbenchSpec.from_dict({'circuit_name': 'amp'})
    >>> spec.rigs, spec.tests, spec.save_device_params
    ([], [], False)
    """
    __test__ = False  # not a pytest test class (name starts with 'Test')
    circuit_name: str
    circuit_builder: str | None = None
    rigs: list = field(default_factory=list)     # [{name, sources:[SourceSpec-dict], submodules}]
    tests: list = field(default_factory=list)    # [{name, dut_portmap, applied_rigs,
                                                  #   submodules, sources, analyses, metrics, specs,
                                                  #   corner, temperature_c}]
    # TASK-371: turn on op-point device-parameter capture on the DUT
    # (``mod.save_device_params = True``) so ``report.op_table()`` works.
    save_device_params: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> 'TestbenchSpec':
        """Build a spec from its dict form.

        Parameters
        ----------
        d : dict
            Dict with a required ``'circuit_name'`` and optional
            ``'circuit_builder'``, ``'rigs'``, ``'tests'`` and
            ``'save_device_params'`` (JSON ``null`` counts as absent only for
            ``'save_device_params'``; ``'rigs'`` and ``'tests'`` must be
            collections when present).

        Returns
        -------
        TestbenchSpec
            The spec. Only ``circuit_name`` and the type of each rig/test
            source and ideal-element entry are checked here.

        Raises
        ------
        ValueError
            ``circuit_name`` is missing or empty, or a ``sources`` /
            ``ideal_elements`` entry in a rig or test has an unknown type.
            The exception raised is ``TestbenchSpecError``, a ``ValueError``
            subclass.
        """
        if not d.get('circuit_name'):
            raise TestbenchSpecError("TestbenchSpec: circuit_name is required")
        from silk.codegen.generator import _validate_entry_type

        for kind in ('rigs', 'tests'):
            for group in d.get(kind, []):
                for collection in ('ideal_elements', 'sources'):
                    entries = group.get(collection, [])
                    # Leave malformed containers/entries to the existing
                    # structural validation during generation.
                    if not isinstance(entries, list):
                        continue
                    for entry in entries:
                        if not isinstance(entry, dict):
                            continue
                        try:
                            _validate_entry_type(entry, collection)
                        except ValueError as exc:
                            raise TestbenchSpecError(
                                f"{kind} {group.get('name', '?')!r}: {exc}"
                            ) from None
        return cls(
            circuit_name=d['circuit_name'],
            circuit_builder=d.get('circuit_builder'),
            rigs=list(d.get('rigs', [])),
            tests=list(d.get('tests', [])),
            # JSON null reads as absent, like every other optional key.
            save_device_params=(False if d.get('save_device_params') is None
                                else d['save_device_params']),
        )


# Analysis kinds generator._sim_code handles (Transient/Ac/Dc/Noise/Op
# branches, matched case-insensitively there via sim_type.capitalize()). No
# exported kind set exists on generator, so this list is hardcoded here and
# must be kept in sync with _sim_code by hand (Task 6 touches both; 'op'
# added TASK-330.2).
_ANALYSIS_KINDS = frozenset({'transient', 'ac', 'dc', 'noise', 'op'})


def _analysis_kind_ok(kind) -> bool:
    """True if `kind` is one of the analysis kinds generator._sim_code emits."""
    return isinstance(kind, str) and kind.lower() in _ANALYSIS_KINDS


# A metric's ``domain`` and a testbench analysis's ``kind`` are two
# vocabularies for one thing, and they disagree on exactly one word:
# ``@metric(domain=...)`` says ``'tran'`` (the same four strings
# ``SILK_ANALYSES`` in silk/signals/kinds.py publishes as ``produced_by``),
# while a spec analysis says ``'transient'`` — what generator._sim_code matches
# (``sim_type.capitalize() == 'Transient'``). This table is the whole
# translation; both spellings of the transient analysis are accepted so a
# caller who writes the domain's word is not refused for it. 'op' needs no
# second spelling (TASK-330.2): ``Op``.lower() already is both words.
#
# REVIEWER BINDING (2026-09-27, TASK-330.2, commit 178250ed, amended same
# day; sign-off confirmed on the digest check below): 'op' has no
# ``@metric(domain=...)`` — it is not, and must never become, a metric
# schema value. This table is internal to testbench codegen and is never
# exported (``export_metric_schema()`` does not read it), so the 'op'
# entry exists only so a spec analysis declared ``{'kind': 'op'}`` maps
# to the 'op' selection word for ``_refuse_ambiguous_analysis``.
_DOMAIN_ANALYSES = {
    'tran': frozenset({'transient', 'tran'}),
    'ac': frozenset({'ac'}),
    'dc': frozenset({'dc'}),
    'noise': frozenset({'noise'}),
    'op': frozenset({'op'}),
}


#: The reverse of :data:`_DOMAIN_ANALYSES`: a spec analysis's ``kind`` in
#: silk's own analysis vocabulary, which is the word ``analysis=`` is written
#: in. Built from the one table so the two can never drift apart.
_ANALYSIS_WORD = {kind: word
                  for word, kinds in _DOMAIN_ANALYSES.items()
                  for kind in kinds}


def _produced_by_analyses(produced_by) -> frozenset:
    """Translate a signal kind's ``produced_by`` tuple — ``SILK_ANALYSES``
    spelling (``'tran'``, ``'ac'``, ``'dc'``, ``'noise'``) — into the
    vocabulary a testbench spec's own ``analyses[].kind`` is written in,
    through the same table ``_DOMAIN_ANALYSES`` already uses for the
    domain word. Reused by kind validation so the kinds table and the
    domain table read as one translation instead of two.
    """
    out: set = set()
    for pb in produced_by:
        out |= _DOMAIN_ANALYSES.get(pb, frozenset({pb}))
    return frozenset(out)


def _metric_binding(where: str, m: dict, slots: list) -> dict:
    """Resolve a metric dict's ``signal``/``signals`` into ``{slot: path}``.

    The v2 replacement for the line-140 "v1 boundary" refusal: a metric with
    N signal inputs is no longer refused for its COUNT, it is bound by NAME
    through ``signals`` (design §6.2/§7.1), and ``signal`` stays the spelling
    for the one-input case. The four §6.2 errors are the same ones
    ``add_metric`` raises, but as ``TestbenchSpecError`` — this is spec
    validation rather than Python API misuse.

    ``signals`` on a one-input metric is accepted: it is the explicit
    spelling of ``signal``, not an error (§6.2, case four).
    """
    from silk.circuit.module_test import _quoted, _slot_inventory

    names = [n for n, _ in slots]
    plural = 's' if len(slots) != 1 else ''
    signal, signals = m.get('signal'), m.get('signals')

    if signal is not None and signals is not None:
        raise TestbenchSpecError(
            f"{where}: has both 'signal' and 'signals'; pass one. 'signals' "
            f"is the general form; for this metric's {len(slots)} signal "
            f"input{plural} ({_quoted(names)}) it is the only one that can "
            f"bind them all.")

    # The JSON the caller should have written, spelled in JSON rather than in
    # Python — these messages are read by whoever authored the spec dict.
    signals_form = '"signals": {%s}' % ', '.join('"%s": ...' % n for n in names)

    if signal is None and signals is None:
        one = len(slots) == 1
        key = 'signal' if one else 'signals'
        binding = '"signal": "<rawfile signal name>"' if one else signals_form
        raise TestbenchSpecError(
            f"{where}: missing {key!r}. This metric takes {len(slots)} signal "
            f"input{plural}: {_slot_inventory(slots)}; bind "
            f"{'it' if one else 'them'} with {binding}.")

    if signals is None:
        if len(slots) != 1:
            raise TestbenchSpecError(
                f"{where}: 'signal' binds a single input, but this metric "
                f"takes {len(slots)}: {_quoted(names)}. Use {signals_form}.")
        return {names[0]: signal}

    if not isinstance(signals, dict):
        raise TestbenchSpecError(
            f"{where}: 'signals' must be an object keyed by this metric's "
            f"signal-input name{plural} ({_quoted(names)}); got "
            f"{type(signals).__name__}.")
    # Unknown keys first: on a typo ('sig_output') that is the useful half of
    # the diagnosis, and the missing-slot message would otherwise report the
    # same mistake from the far side (the reasoning add_metric follows).
    unknown = [k for k in signals if k not in names]
    if unknown:
        raise TestbenchSpecError(
            f"{where}: 'signals' has unknown "
            f"key{'s' if len(unknown) != 1 else ''} {_quoted(unknown)}; this "
            f"metric's signal inputs are {_quoted(names)}.")
    missing = [n for n in names if n not in signals]
    if missing:
        raise TestbenchSpecError(
            f"{where}: 'signals' is missing {_quoted(missing)}. This metric "
            f"takes {len(slots)} signal input{plural}: {_slot_inventory(slots)}; "
            f"bound: {_quoted([n for n in names if n in signals]) or 'nothing'}.")
    return {n: signals[n] for n in names}


def _refuse_unproducible_kinds(where: str, slots: list) -> None:
    """Refuse a metric no probe can satisfy (design §7.3).

    A kind whose ``produced_by`` is EMPTY is one that nothing silk can
    simulate hands back — a phase-noise sideband, a user-authored reference
    waveform, a power sweep. A declarative spec binds rawfile signal names,
    so such a slot is unsatisfiable by construction, however the spec is
    written, and refusing it here costs the caller a generate instead of a
    simulation run. The callable form of ``add_metric`` is the escape hatch
    and the message says so.

    A slot with no kind (a schema exported by a pre-2.0 silk) and a kind this
    silk does not know are both left alone: neither is a statement that
    nothing produces it, and inventing a refusal from an absence of
    information would refuse valid specs.

    The table is read through :mod:`silk.signals.schema` — the one module
    sanctioned to import the leaf kinds module — and it is the same
    ``{kind: {'produced_by': [...]}}`` the export publishes under
    ``_meta.signal_kinds``. This validator evaluates the published refusal
    rule exactly as any other consumer of the schema does.
    """
    from silk.signals.schema import signal_kinds_meta

    kinds = signal_kinds_meta()
    for slot, kind in slots:
        entry = kinds.get(kind)
        if entry is not None and not entry['produced_by']:
            raise TestbenchSpecError(
                f"{where}: input {slot!r} needs a Signal of kind {kind!r}, "
                f"which no silk analysis produces — it must be constructed "
                f"and passed through add_metric's callable form.")


def _refuse_kind_analysis_mismatch(where: str, domain: str, slots: list,
                                    declared_analyses: set) -> None:
    """Refuse a metric iff some signal input's KIND has a
    ``produced_by`` disjoint from this test's declared analyses — read
    from the same table (``signal_kinds_meta``) the schema exports under
    ``_meta.signal_kinds``, not from ``domain``.

    Previously, this compared ``domain`` itself against the declared
    analyses, which is wrong for exactly the twelve ``kind='any'``
    metrics: the schema publishes ``any.produced_by = (tran, ac, dc,
    noise)`` (every analysis), so a consumer that wires by kind offers
    them in an ac-only or dc-only or noise-only test, and the old
    domain-only check refused all three because ``domain='tran'`` names
    only one. Reading the kinds table instead makes the validator agree
    with what the schema already publishes. ``domain`` stays the metric's home/grouping label — it is
    still what the message below names, to preserve diagnostic compatibility; for every
    metric whose declared kind is NOT ``any``, kind-disjoint and
    domain-disjoint are the same fact.

    A slot whose kind is unregistered, or whose ``produced_by`` is
    already empty, is skipped: the first is not (yet) a statement about
    analyses, and the second was already refused by
    :func:`_refuse_unproducible_kinds` before this runs.
    """
    from silk.signals.schema import signal_kinds_meta

    kinds = signal_kinds_meta()
    for _slot, kind in slots:
        entry = kinds.get(kind)
        if entry is None or not entry['produced_by']:
            continue
        if _produced_by_analyses(entry['produced_by']) & declared_analyses:
            continue
        declared = (', '.join(repr(k) for k in sorted(declared_analyses))
                    or 'none')
        raise TestbenchSpecError(
            f"{where}: domain {domain!r} has no matching analysis in "
            f"this test; declared: {declared}. Add "
            f"{'an' if domain[:1] in 'aeiou' else 'a'} {domain!r} "
            f"analysis, or measure this metric in a test that has one.")


def _refuse_ambiguous_analysis(where: str, m: dict, slots: list,
                               declared_order: list) -> None:
    """Settle WHICH analysis's plot this metric reads, at
    generate time, and refuse the spec if the declared analyses cannot.

    Delegates the whole rule to :func:`silk.circuit.module_test.
    _resolve_metric_analysis` — the same function the run-time layer and the
    generated extractor call — so all three refuse the same specs in the same
    words. Only the exception type differs: codegen raises
    ``TestbenchSpecError``; it is a ``ValueError``, so this is a re-raise of
    the same family and not a second rule.

    *declared_order* is the test's ``analyses`` list in SPEC order, not the
    validator's set, so the analyses a message names are in the order the
    author wrote them — the same order the run-time layer reads off
    ``sim_analysis_dict`` and therefore the same message from both.
    """
    from silk.circuit.module_test import _resolve_metric_analysis

    declared = []
    for kind in declared_order:
        word = _ANALYSIS_WORD.get(kind)
        if word is not None and word not in declared:
            declared.append(word)
    try:
        _resolve_metric_analysis(where, slots, declared, m.get('analysis'),
                                 signals=_bound_signal_paths(m, slots))
    except ValueError as e:
        raise TestbenchSpecError(str(e)) from None


def _bound_signal_paths(m: dict, slots: list) -> list:
    """The signal paths a metric dict binds, in slot order — what the
    ambiguity refusal names so the message points at the probe, not only at
    the metric. ``_metric_binding`` has already accepted the dict by the time
    this runs, so both spellings are present and complete."""
    signals = m.get('signals')
    if signals is not None:
        values = [signals[n] for n, _ in slots if n in signals]
    else:
        signal = m.get('signal')
        values = [] if signal is None else [signal]
    return _flatten_paths(values)


def _flatten_paths(values) -> list:
    """Paths of a binding with each bus list spread into its elements."""
    out = []
    for value in values:
        out.extend(value if isinstance(value, list) else [value])
    return out


def _validate(spec: TestbenchSpec) -> None:
    """Fail-loud structural validation (§2.1) — raises TestbenchSpecError on
    any violation, naming the offending test/metric/spec/rig. Never lets
    generate_testbenches emit broken/uncompilable Python.

    Required-key presence (rig/test/metric/spec ``name``, test
    ``dut_portmap``) is checked explicitly up front so a malformed dict
    surfaces here — as a named, catchable TestbenchSpecError — instead of
    a bare KeyError escaping later during assembly (
    a spec/rig/test dict missing a required key must never raise a plain
    KeyError). generate_testbenches also wraps the whole assembly in a
    KeyError backstop as defense in depth, but the checks here give a
    much clearer, offender-naming message.

    The metric
    half of this pass asks **satisfiability**, not input COUNT. A metric with
    N signal inputs is bound by name through ``signals`` and validated against
    the v2 schema: every slot bound and none invented, no slot needing a kind
    nothing silk simulates produces, every param inside its declared
    ``choices``, and an analysis in this test that matches the metric's
    ``domain``. Each of those refuses at generate time what used to surface
    only after a simulation run had been paid for.
    """
    from silk.circuit.module_test import _metric_signal_slots
    from silk.signals.schema import get_metric, metric_schema

    # circuit_builder is emitted bare as a call target (``mod =
    # <circuit_builder>(pdk)``) inside the builder assembler; the TASK-14
    # contract is a bare Python identifier only (not a keyword, no dotted
    # path, no call/statement text), enforced here before any output is
    # assembled so ``evil(); x`` can never become two live statements.
    # Falsy (None / '') means "no builder" and skips the assembler entirely.
    if spec.circuit_builder and (
            not isinstance(spec.circuit_builder, str)
            or not spec.circuit_builder.isidentifier()
            or keyword.iskeyword(spec.circuit_builder)):
        raise TestbenchSpecError(
            f"circuit_builder {spec.circuit_builder!r} must be a bare Python "
            f"identifier (the name of the sidecar builder function, e.g. "
            f"'_build_ac_gain') and must not be a Python keyword; dotted "
            f"paths and call expressions are not accepted (emitted as "
            f"'mod = <circuit_builder>(pdk)')")

    # circuit_name is emitted bare inside ``def build_<circuit_name>_with_tests``
    # only when circuit_builder is set; that emission site holds the
    # TASK-13 contract (valid Python identifier, not a keyword) — the same
    # rule generator._validate_identity_fields applies to CircuitSpec.
    if spec.circuit_builder and (
            not isinstance(spec.circuit_name, str)
            or not spec.circuit_name.isidentifier()
            or keyword.iskeyword(spec.circuit_name)):
        raise TestbenchSpecError(
            f"circuit_name {spec.circuit_name!r} must be a valid Python "
            f"identifier and must not be a Python keyword (required: "
            f"circuit_builder is set, so build_<circuit_name>_with_tests "
            f"would be emitted)")

    if not isinstance(spec.save_device_params, bool):
        raise TestbenchSpecError(
            f"save_device_params {spec.save_device_params!r} must be true or "
            f"false (JSON boolean)")

    rig_names = set()
    for i, r in enumerate(spec.rigs):
        if 'name' not in r:
            raise TestbenchSpecError(f"rig #{i}: missing required 'name': {r!r}")
        rname = r['name']
        if not str(rname).isidentifier():
            raise TestbenchSpecError(
                f"rig {rname!r}: name is not a valid Python identifier "
                f"(used in def _apply_{rname}_rig)")
        if rname in rig_names:
            raise TestbenchSpecError(f"duplicate rig name {rname!r}")
        rig_names.add(rname)

    test_names = set()
    for i, t in enumerate(spec.tests):
        if 'name' not in t:
            raise TestbenchSpecError(f"test #{i}: missing required 'name': {t!r}")
        tname = t['name']
        if not str(tname).isidentifier():
            raise TestbenchSpecError(
                f"test {tname!r}: name is not a valid Python identifier "
                f"(used in def _attach_{tname})")
        if tname in test_names:
            raise TestbenchSpecError(f"duplicate test name {tname!r}")
        test_names.add(tname)
        if 'dut_portmap' not in t:
            raise TestbenchSpecError(f"test {tname!r}: missing required 'dut_portmap'")
        _validate_test_conditions(tname, t)
        metric_names = {m.get('name') for m in t.get('metrics', [])}

        for rig in t.get('applied_rigs', []):
            if rig not in rig_names:
                raise TestbenchSpecError(f"test {tname!r}: applied_rig {rig!r} is not defined")

        # Analyses are checked BEFORE the metrics so an unknown kind is named
        # as itself rather than reported from the far side as "no matching
        # analysis" (D7 below cross-checks the two).
        declared_analyses = set()
        declared_order = []            # spec order, for TASK-212's messages
        for a in t.get('analyses', []):
            if not _analysis_kind_ok(a.get('kind')):
                raise TestbenchSpecError(f"test {tname!r}: unknown analysis kind {a.get('kind')!r}")
            declared_analyses.add(a['kind'].lower())
            declared_order.append(a['kind'].lower())

        derived = _validate_derived_signals(spec, t, declared_order)
        widths = _derived_widths(t)
        bus_metrics = {}               # metric name -> its element count

        for m in t.get('metrics', []):
            if 'name' not in m:
                raise TestbenchSpecError(f"test {tname!r}: metric missing required 'name': {m!r}")
            mtype = m.get('type')
            try:
                fn = get_metric(mtype)
                schema = metric_schema(fn)
            except KeyError as e:
                raise TestbenchSpecError(str(e)) from None
            where = f"test {tname!r} metric {m['name']!r} (type {mtype!r})"

            # v2 satisfiability, in the order the design states it (§7):
            # bind every signal input by name, then refuse a slot no probe
            # can fill, then the params, then the analysis the domain needs.
            slots = _metric_signal_slots(fn)
            bound = _metric_binding(where, m, slots)
            bus_width = _validate_metric_bus(where, m, bound, widths, derived)
            if bus_width is not None:
                bus_metrics[m['name']] = bus_width
            _refuse_unproducible_kinds(where, slots)

            params = m.get('params') or {}
            missing = [p['name'] for p in schema['parameters']
                       if p['required'] and p['name'] not in params]
            if missing:
                raise TestbenchSpecError(
                    f"{where}: missing required param(s) {missing}")
            known = {p['name'] for p in schema['parameters']}
            unknown = [k for k in params if k not in known]
            if unknown:
                raise TestbenchSpecError(
                    f"{where}: unknown param(s) {unknown}; "
                    f"known params: {sorted(known)}")
            # choices (§7.4): the metric body's own guard would raise this
            # after a simulation; the schema now publishes the permitted set,
            # so every consumer gets the refusal at generate time instead.
            for p in schema['parameters']:
                if p['choices'] and p['name'] in params:
                    v = params[p['name']]
                    if v not in p['choices'] and not (p['optional'] and v is None):
                        raise TestbenchSpecError(
                            f"{where}: param {p['name']!r} is {v!r}, which is "
                            f"not one of {p['choices']}")

            # D7 (owner-approved, kinds-based since TASK-208): an
            # unsatisfiable analysis pairing is refused here, at generate
            # time, instead of failing after a simulation has been paid for.
            domain = schema['domain']
            _refuse_kind_analysis_mismatch(where, domain, slots, declared_analyses)

            # TASK-212: which declared analysis this metric READS. Runs after
            # D7 so "no analysis can feed this metric" is reported as itself
            # rather than from the far side as an empty candidate list.
            if _metric_function_form(m, derived):
                _refuse_derived_metric_analysis(where, m, slots, bound,
                                                derived, declared_order)
            else:
                _refuse_ambiguous_analysis(where, m, slots, declared_order)

        _refuse_bus_name_collisions(tname, t, bus_metrics)

        for s in t.get('specs', []):
            if 'name' not in s:
                raise TestbenchSpecError(f"test {tname!r}: spec missing required 'name': {s!r}")
            if s.get('metric') in bus_metrics:
                width = bus_metrics[s['metric']]
                raise TestbenchSpecError(
                    f"test {tname!r} spec {s['name']!r}: metric {s['metric']!r} "
                    f"expands to {s['metric']}[0]..{s['metric']}[{width - 1}]; "
                    f"specs on bus metrics are not supported")
            if s.get('metric') not in metric_names:
                raise TestbenchSpecError(
                    f"test {tname!r} spec {s['name']!r}: references unknown metric {s.get('metric')!r}")

        # source field presence is enforced when SourceSpec is built (Task 7) —
        # a missing positive/negative raises there; that raise is wrapped in
        # TestbenchSpecError at the call site (Task 7/8), not here.


def _validate_test_conditions(tname: str, t: dict) -> None:
    """Refuse a test's ``corner`` / ``temperature_c`` keys unless
    they emit as a valid ``tb.set_corner(...)`` / ``tb.set_temperature(...)``.

    ``corner`` gets the same syntactic check as the CircuitSpec ``corner``
    (``[A-Za-z0-9_-]+``). Membership in the PDK's corner set cannot be
    checked here — a TestbenchSpec names no PDK — so ``tb.set_corner``
    checks it when the generated code runs, raising ``UnknownCornerError``
    with the valid corners. ``temperature_c`` must be a finite real number
    above absolute zero, and not a boolean (``True`` is an ``int``).
    """
    from silk.codegen.generator import _require_corner

    if t.get('corner') is not None:
        try:
            _require_corner(t['corner'], f"test {tname!r}: corner")
        except ValueError as e:
            raise TestbenchSpecError(str(e)) from None
    temp = t.get('temperature_c')
    if temp is not None and (
            isinstance(temp, bool) or not isinstance(temp, Real)
            or not math.isfinite(temp) or temp <= -273.15):
        raise TestbenchSpecError(
            f"test {tname!r}: temperature_c {temp!r} must be a finite number "
            f"of degrees Celsius above -273.15 (not a string or a boolean)")


# --- Derived signals (TASK-373) ---------------------------------------------
#
# A test's ``derived_signals`` list declares waveform math as probe ->
# operation -> metric, one operation per entry from the fixed silk list in
# silk.signals.operations (no expression language). Each entry emits one local
# binding inside ``_attach_<test>`` built from tb.probe(...) and Signal
# arithmetic/methods, so the math is saved with the test and recomputed on
# every run. A metric whose ``signal``/``signals`` names a derived signal
# emits add_metric's function form over that local.

#: Locals and globals the generated ``_attach_<test>`` body itself reads; a
#: derived signal (a local binding there) must not shadow them.
_DERIVED_RESERVED_LOCALS = frozenset({
    'tb', 'mod', 'pdk', 'analysis', 'sources', 'elements', 'metrics',
    'i', 'range'})


def _derived_entries(t: dict) -> list:
    """The test's ``derived_signals`` list ([] when absent)."""
    return t.get('derived_signals') or []


# --- Buses: an ordered list of paths in place of one path (TASK-606.3) --------
#
# A signal entry (a derived signal's input, or one metric slot) may be a list of
# path strings: one bus, written as one ``tb.probe_bus([...])`` expression. Order
# is the list order; nothing is sorted. A metric over a bus becomes a for loop of
# scalar metrics named ``name[k]``, k counting from 0 by list position.

def _check_bus_list(where: str, paths: list, label: str) -> None:
    """Refuse a malformed bus list, naming *where* (test and entry), *label*
    (which input or slot) and the element. Checks only what the list itself
    shows: the graph is not built at generation."""
    if not paths:
        raise TestbenchSpecError(
            f"{where}: {label} is an empty list; give at least one path, or "
            f"use a string for one signal")
    first = {}
    for index, path in enumerate(paths):
        if not isinstance(path, str) or not path.strip():
            raise TestbenchSpecError(
                f"{where}: {label} element {index} {path!r} must be a "
                f"non-empty signal path string")
        if '<' in path or '>' in path or path.lower().startswith('xdut.'):
            raise TestbenchSpecError(
                f"{where}: {label} element {index} {path!r}: SPICE or emitted "
                f"spelling; use the designer spelling (for example "
                f"'dut.stage[2].rout[0]'), where every instance and net uses "
                f"[k] and the DUT is 'dut'")
        if path in first:
            raise TestbenchSpecError(
                f"{where}: {label}: elements {first[path]}..{index} are the "
                f"same path {path!r}; a repeated net is one signal, use the "
                f"string form {path!r}")
        first[path] = index


def _operand_width(operand, widths: dict):
    """Element count of a derived-signal operand: a list's length, a derived
    bus's width, or None for a scalar (a scalar broadcasts and has no width)."""
    if isinstance(operand, list):
        return len(operand)
    if isinstance(operand, str):
        return widths.get(operand)
    return None


def _operands_width(where: str, operands: list, widths: dict):
    """The width every bus operand of one expression shares (None if no bus
    operand). Unequal bus widths are refused: buses combine by position."""
    seen = None
    for position, operand in enumerate(operands):
        width = _operand_width(operand, widths)
        if width is None:
            continue
        if seen is None:
            seen = (position, width)
        elif width != seen[1]:
            raise TestbenchSpecError(
                f"{where}: input {position} has {width} elements; input "
                f"{seen[0]} has {seen[1]}; buses combine by position and need "
                f"equal length")
    return None if seen is None else seen[1]


def _derived_widths(t: dict) -> dict:
    """``{derived name: element count or None}`` for a test whose derived
    signals have been validated."""
    widths = {}
    for d in _derived_entries(t):
        seen = [w for w in (_operand_width(i, widths) for i in d['inputs'])
                if w is not None]
        widths[d['name']] = seen[0] if seen else None
    return widths


def _bus_slots(binding: dict, widths: dict) -> dict:
    """``{slot: element count}`` for the slots of *binding* that hold a bus: a
    list, or the name of a derived bus."""
    return {slot: _operand_width(value, widths) for slot, value in binding.items()
            if _operand_width(value, widths) is not None}


def _validate_metric_bus(where: str, m: dict, bound: dict, widths: dict, derived):
    """Check the bus slots of one metric; return the element count of its one
    bus slot, or None when it has none."""
    for slot, value in bound.items():
        if isinstance(value, list):
            label = ('signal' if m.get('signal') is not None
                     else f"'signals' slot {slot!r}")
            _check_bus_list(where, value, label)
    buses = _bus_slots(bound, widths)
    if len(buses) > 1:
        names = [repr(n) for n in buses]
        raise TestbenchSpecError(
            f"{where}: slots {', '.join(names[:-1])} and {names[-1]} are both "
            f"buses; one slot may be a bus")
    if not buses:
        return None
    if 'enumerate' in widths:
        raise TestbenchSpecError(
            f"{where}: the derived signal 'enumerate' would shadow the "
            f"enumerate() that this bus metric loop calls; rename it")
    return next(iter(buses.values()))


def _refuse_bus_name_collisions(tname: str, t: dict, bus_metrics: dict) -> None:
    """A generated ``name[k]`` must not equal any other metric name."""
    if not bus_metrics:
        return
    counts, generated = {}, set()
    for m in t.get('metrics', []):
        name = m.get('name')
        if name in bus_metrics:
            for k in range(bus_metrics[name]):
                key = f"{name}[{k}]"
                generated.add(key)
                counts[key] = counts.get(key, 0) + 1
        else:
            counts[name] = counts.get(name, 0) + 1
    for key in sorted(generated):
        if counts[key] > 1:
            raise TestbenchSpecError(
                f"test {tname!r}: generated metric {key!r} collides with the "
                f"metric of that name; rename one of them")


def _metric_function_form(m: dict, derived_names) -> bool:
    """True if metric dict *m* is emitted in add_metric's function form: it
    binds a derived signal by name, or a bus list."""
    values = ([m['signal']] if m.get('signal') is not None
              else list((m.get('signals') or {}).values()))
    return any(isinstance(v, list) or (isinstance(v, str) and v in derived_names)
               for v in values)


def _test_net_names(t: dict, rigs: dict) -> dict:
    """``{lower-cased name: where}`` for every net name the spec gives this
    test: DUT ports and the nets they map to, and the nets of the sources,
    devices, ideal elements and helpers of the test and its applied rigs."""
    from silk.codegen.generator import _CONTROLLED_SOURCE_PINS
    out = {}

    def add(name, where):
        if isinstance(name, str) and name:
            out.setdefault(name.lower(), where)

    portmap = t.get('dut_portmap')
    for port, net in (portmap.items() if isinstance(portmap, dict) else ()):
        add(port, f"DUT port {port!r}")
        add(net, f"net {net!r} (dut_portmap)")
    groups = [(f"rig {r!r}", rigs[r]) for r in t.get('applied_rigs', [])
              if r in rigs] + [(f"test {t['name']!r}", t)]
    pin_keys = {'positive', 'negative'}
    for pins in _CONTROLLED_SOURCE_PINS.values():
        pin_keys.update(pins)
    for origin, g in groups:
        for src in g.get('sources') or []:
            if isinstance(src, dict):
                for k in pin_keys:
                    add(src.get(k), f"net {src.get(k)!r} of a source in {origin}")
        for key in ('devices', 'ideal_elements', 'submodules'):
            for e in g.get(key) or []:
                nets = e.get('nets') if isinstance(e, dict) else None
                copies = nets if isinstance(nets, list) else [nets]
                for connections in copies:
                    if isinstance(connections, dict):
                        for net in connections.values():
                            add(net, f"net {net!r} of {key} in {origin}")
    return out


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _validate_derived_signals(spec: 'TestbenchSpec', t: dict,
                              declared_order: list) -> dict:
    """Refuse a malformed ``derived_signals`` list; return ``{name: analysis
    word or None}`` — the plot each derived signal reads, as far as the spec
    pins it — for the metric checks that follow."""
    from silk.circuit.module_test import _SIM_KEY_ANALYSIS
    from silk.signals.operations import export_signal_operations

    tname = t['name']
    entries = t.get('derived_signals')
    if entries is None:
        return {}
    if not isinstance(entries, list):
        raise TestbenchSpecError(
            f"test {tname!r}: derived_signals must be a list")
    ops = {op['name']: op for op in export_signal_operations()}
    rigs = {r['name']: r for r in spec.rigs}
    nets = _test_net_names(t, rigs)
    reserved = set(_DERIVED_RESERVED_LOCALS)
    reserved.update(f"_apply_{r['name']}_rig" for r in spec.rigs)
    reserved.update(f"_attach_{x['name']}" for x in spec.tests)
    if spec.circuit_builder:
        reserved.update({spec.circuit_builder,
                         f"build_{spec.circuit_name}_with_tests"})
    declared = []
    for kind in declared_order:
        word = _ANALYSIS_WORD.get(kind)
        if word is not None and word not in declared:
            declared.append(word)
    all_names = [d.get('name') for d in entries if isinstance(d, dict)]

    out = {}
    widths = {}
    for index, d in enumerate(entries):
        if not isinstance(d, dict):
            raise TestbenchSpecError(
                f"test {tname!r}: derived_signals[{index}] must be an object: {d!r}")
        name = d.get('name')
        if name is None:
            raise TestbenchSpecError(
                f"test {tname!r}: derived_signals[{index}] missing required 'name': {d!r}")
        where = f"test {tname!r} derived signal {name!r}"
        if (not isinstance(name, str) or not name.isidentifier()
                or keyword.iskeyword(name)):
            raise TestbenchSpecError(
                f"{where}: name must be a Python identifier and not a keyword "
                f"(it is emitted as a local variable in _attach_{tname})")
        if name in reserved:
            raise TestbenchSpecError(
                f"{where}: name is reserved in the generated code "
                f"(_attach_{tname} uses {name!r}); choose another name")
        if name in out:
            raise TestbenchSpecError(f"{where}: duplicate derived signal name")
        if name.lower() in nets:
            raise TestbenchSpecError(
                f"{where}: name clashes (case-insensitive) with "
                f"{nets[name.lower()]}; a metric's 'signal' could not tell "
                f"the two apart")

        unknown_keys = sorted(set(d) - {'name', 'operation', 'inputs',
                                        'params', 'analysis'})
        if unknown_keys:
            raise TestbenchSpecError(
                f"{where}: unknown key(s) {unknown_keys}; a derived signal "
                f"takes 'name', 'operation', 'inputs', 'params' and "
                f"optionally 'analysis'")
        opname = d.get('operation')
        if opname not in ops:
            raise TestbenchSpecError(
                f"{where}: unknown operation {opname!r}; the silk operations "
                f"are {list(ops)}")
        op = ops[opname]
        inputs = d.get('inputs')
        if not isinstance(inputs, list):
            raise TestbenchSpecError(
                f"{where}: 'inputs' must be a list of signal paths or earlier "
                f"derived signal names; got {inputs!r}")
        lo, hi = op['min_inputs'], op['max_inputs']
        if len(inputs) < lo or (hi is not None and len(inputs) > hi):
            want = (f"exactly {lo}" if lo == hi
                    else f"{lo} or more" if hi is None else f"{lo} to {hi}")
            raise TestbenchSpecError(
                f"{where}: operation {opname!r} takes {want} input"
                f"{'s' if hi != 1 else ''}; got {len(inputs)}")

        analyses = set()
        for position, inp in enumerate(inputs):
            if isinstance(inp, list):
                _check_bus_list(where, inp, f"input {position}")
                continue
            if not isinstance(inp, str) or not inp.strip():
                raise TestbenchSpecError(
                    f"{where}: unknown input {inp!r}; an input is a non-empty "
                    f"signal path, a list of signal paths (a bus) or the name "
                    f"of an earlier derived signal")
            if inp in out:
                if out[inp] is not None:
                    analyses.add(out[inp])
            elif inp in all_names:
                raise TestbenchSpecError(
                    f"{where}: unknown input {inp!r}; it names "
                    f"{'this derived signal' if inp == name else 'a derived signal declared later'}"
                    f" — a derived signal can only use derived signals "
                    f"declared before it")

        widths[name] = _operands_width(where, inputs, widths)

        params = d.get('params') or {}
        if not isinstance(params, dict):
            raise TestbenchSpecError(
                f"{where}: 'params' must be an object; got {params!r}")
        known = {p['name'] for p in op['parameters']}
        extra = [k for k in params if k not in known]
        if extra:
            raise TestbenchSpecError(
                f"{where}: unknown param(s) {extra} for operation "
                f"{opname!r}; its params are {sorted(known) or 'none'}")
        missing = [p['name'] for p in op['parameters']
                   if p['required'] and p['name'] not in params]
        if missing:
            raise TestbenchSpecError(
                f"{where}: missing required param(s) {missing} for "
                f"operation {opname!r}")
        for k, v in params.items():
            if not _is_number(v):
                raise TestbenchSpecError(
                    f"{where}: param {k!r} must be a number; got {v!r}")

        own = d.get('analysis')
        if own is not None:
            words = sorted(set(_SIM_KEY_ANALYSIS.values()))
            if own not in words:
                raise TestbenchSpecError(
                    f"{where}: analysis={own!r} is not a silk analysis; "
                    f"expected one of {words}")
            if declared and own not in declared:
                raise TestbenchSpecError(
                    f"{where}: analysis={own!r}, but this test declares "
                    f"{declared}")
            analyses.add(own)
        if len(analyses) > 1:
            raise TestbenchSpecError(
                f"{where}: its inputs read different analyses "
                f"{sorted(analyses)}; one derived signal reads one plot")
        out[name] = next(iter(analyses), None)
    return out


def _refuse_derived_metric_analysis(where: str, m: dict, slots: list,
                                    bound: dict, derived: dict,
                                    declared_order: list) -> None:
    """The analysis check for a metric that binds a derived signal — the
    function form's rule (each slot settled on its own probes), applied at
    generate time so the run cannot refuse what generation accepted.

    A derived slot reads the analysis its derived signal pins; a path slot
    reads the metric's ``analysis`` (emitted on its probe). The metric's
    ``analysis`` must agree with every derived signal it binds."""
    from silk.circuit.module_test import _resolve_metric_analysis

    declared = []
    for kind in declared_order:
        word = _ANALYSIS_WORD.get(kind)
        if word is not None and word not in declared:
            declared.append(word)
    metric_word = m.get('analysis')
    for slot, kind in slots:
        path = bound[slot]
        if isinstance(path, str) and path in derived:
            word = derived[path]
            if metric_word is not None and word != metric_word:
                raise TestbenchSpecError(
                    f"{where}: analysis={metric_word!r}, but derived signal "
                    f"{path!r} reads "
                    f"{'no pinned analysis' if word is None else repr(word)}; "
                    f"set \"analysis\": {metric_word!r} on the derived signal")

            def fix(candidates, _p=path):
                return (f"Set \"analysis\" on derived signal {_p!r} to one of "
                        f"{candidates}.")
        else:
            word = metric_word

            def fix(candidates):
                return f"Set \"analysis\" on the metric to one of {candidates}."
        try:
            _resolve_metric_analysis(where, [(slot, kind)], declared, word,
                                     signals=_flatten_paths([path]), fix=fix)
        except ValueError as e:
            raise TestbenchSpecError(str(e)) from None


def _probe_call(path, analysis=None) -> str:
    if isinstance(path, list):
        if analysis is None:
            return f"tb.probe_bus({path!r})"
        return f"tb.probe_bus({path!r}, analysis={analysis!r})"
    if analysis is None:
        return f"tb.probe({path!r})"
    return f"tb.probe({path!r}, analysis={analysis!r})"


def _derived_signal_lines(t: dict) -> list:
    """Emit one ``<name> = <silk Python>`` line per derived signal."""
    from silk.signals.operations import operation_expression, signal_operation

    lines, seen = [], set()
    for d in _derived_entries(t):
        operands = [inp if (isinstance(inp, str) and inp in seen)
                    else _probe_call(inp, d.get('analysis'))
                    for inp in d['inputs']]
        op = signal_operation(d['operation'])
        params = {p['name']: p['default'] for p in op['parameters']}
        params.update(d.get('params') or {})
        lines.append(f"{d['name']} = "
                     f"{operation_expression(d['operation'], operands, params)}")
        seen.add(d['name'])
    return lines


def _free_local(base: str, taken) -> str:
    """*base*, or *base* plus the smallest numeric suffix that no name in
    *taken* uses."""
    if base not in taken:
        return base
    n = 1
    while f"{base}{n}" in taken:
        n += 1
    return f"{base}{n}"


def _bus_metric_name_fstring(name: str, counter: str) -> str:
    """The f-string literal ``f'<name>[{counter}]'`` with braces in *name*
    doubled, so any metric name stays a literal."""
    return "f" + repr(name.replace('{', '{{').replace('}', '}}') + f"[{{{counter}}}]")


def _derived_metric_lines(m: dict, derived_names, widths=None, bound_names=()) -> list:
    """Emit add_metric's function form for a metric that binds a derived
    signal or a bus list: ``tb.add_metric('name', metrics.<type>, <slot>=<signal>, ...)``.
    Derived inputs bind their local; path inputs bind ``tb.probe(path)``,
    pinned to the metric's ``analysis`` when it has one (the function form
    names the analysis on the probe).

    A metric with one bus slot (a list, or the name of a derived bus) is the
    for loop a user would write, one scalar metric per element named
    ``name[k]`` with k counting from 0 by position; the other slots are
    scalar and repeat in every call. *bound_names* are the other names the
    function reads or binds (the helper import aliases); the loop locals are
    allocated against them and the derived signals together. Returns the lines,
    unindented."""
    from silk.circuit.module_test import _metric_signal_slots
    from silk.signals.schema import get_metric

    widths = widths or {}
    slots = _metric_signal_slots(get_metric(m['type']))
    if m.get('signals') is not None:
        binding = dict(m['signals'])
    else:
        binding = {slots[0][0]: m['signal']}
    buses = _bus_slots(binding, widths)
    bus_slot = next(iter(buses), None)
    taken = (set(derived_names) | set(_DERIVED_RESERVED_LOCALS) | {'enumerate'}
             | set(bound_names))
    counter = _free_local('__silk_k', taken)
    element = _free_local('__silk_bit', taken)
    name = (_bus_metric_name_fstring(m['name'], counter) if bus_slot
            else repr(m['name']))
    line = f"tb.add_metric({name}, metrics.{m['type']}"
    for slot, value in binding.items():
        if slot == bus_slot:
            line += f", {slot}={element}"
            continue
        arg = (value if isinstance(value, str) and value in derived_names
               else _probe_call(value, m.get('analysis')))
        line += f", {slot}={arg}"
    for k, v in (m.get('params') or {}).items():
        line += f", {k}={_fmt_param(v)}"
    line += ")"
    if bus_slot is None:
        return [line]
    value = binding[bus_slot]
    source = (value if isinstance(value, str) and value in derived_names
              else _probe_call(value, m.get('analysis')))
    return [f"for {counter}, {element} in enumerate({source}):", f"    {line}"]


# --- Per-line emit helpers (Task 7) -----------------------------------------
#
# These are pure string emitters: each takes one declarative dict (a source /
# metric / spec / analysis / rig entry from a TestbenchSpec) and returns the
# exact line(s) of generated Python that call the corresponding tb.* API.
# TB8 assembles the whole-function output from these.

def _fmt_param(v) -> str:
    """Format a metric-param / spec-target scalar for emitted code.

    Deliberately plain ``repr()``, *not* ``generator._fmt_value``: that
    helper forces scientific notation for ``abs(v) >= 1e6`` (tuned for
    device-geometry literals like W/L), which would render a 1 MHz
    ``freq`` param as ``'1e+06'``. The testbench-codegen contract expects ``freq=1000000.0``. Plain ``repr()`` matches Python's
    own much-larger scientific-notation threshold and satisfies both the
    frequency and target-ratio cases.
    """
    return repr(v)


def _reject_variable_refs(where: str, slots) -> None:
    """Refuse a variable reference object in a testbench numeric slot.

    Module variables are a CircuitSpec feature: a TestbenchSpec
    declares none, so a reference would emit an undefined name.
    """
    for param, value in slots:
        if isinstance(value, dict):
            raise TestbenchSpecError(
                f"{where} param {param!r}: {value!r} is not a number; "
                f"variable references are a CircuitSpec feature and a "
                f"TestbenchSpec declares no variables")


def _source_error(src, error, owner, position):
    """Add source identity and ownership while preserving the parser's reason."""
    identity = (f"Source {src['instance']!r}" if isinstance(src, dict) and 'instance' in src
                else f"source entry {position}")
    reason = (f"missing required field {error.args[0]!r}"
              if isinstance(error, KeyError) else str(error))
    if not reason.startswith(f"{identity}:"):
        reason = f"{identity}: {reason}"
    return TestbenchSpecError(f"{owner}: {reason}" if owner else reason)


def _source_add_line(src: dict, source_specs=(), owner="", position=1) -> str:
    """Emit ``tb.add(<call>)`` for one source dict — independent
    (VoltageSource/CurrentSource), dependent/controlled (VCVS/VCCS/CCVS/
    CCCS), or behavioral (BehavioralVoltage/BehavioralCurrent).

    Reuses generator._source_entry_to_spec to build the SourceSpec — the
    same dispatch-on-type the circuit generator uses, so a
    class the circuit side accepts is never refused here, and there is no
    second copy of the controlled/behavioral source-class lists
    (_CONTROLLED_SOURCE_PINS / _BEHAVIORAL_SOURCE_TYPES stay defined only
    in generator.py). Independent sources still go through
    _source_alias/_source_call_body, so that emission is byte-for-byte
    unchanged from before. Any missing required field or
    unrecognized source type raises TestbenchSpecError (§2.1 source field
    presence) instead of a bare KeyError/TypeError escaping.
    """
    from silk.codegen.generator import (
        _source_entry_to_spec, _source_alias, _source_call_body,
        _controlled_source_call, _behavioral_source_call, _source_array_call,
        _CONTROLLED_SOURCE_PINS, _BEHAVIORAL_SOURCE_TYPES, _source_value_slots,
    )

    try:
        label = f"testbench sources entry {src.get('instance', '?')!r}"
        spec = _source_entry_to_spec(src, label)
        _reject_variable_refs(label, _source_value_slots(spec))
        if spec.type in _CONTROLLED_SOURCE_PINS:
            call = _controlled_source_call(spec, source_specs)
            if spec.array is not None:
                return f"tb.add(*{call})"
        elif spec.type in _BEHAVIORAL_SOURCE_TYPES:
            call = _behavioral_source_call(spec)
            if spec.array is not None:
                return f"tb.add(*{call})"
        elif spec.array is not None:
            return f"tb.add(*{_source_array_call(spec)})"
        else:
            alias = _source_alias(spec.type)
            term = f"positive={spec.positive!r}, negative={spec.negative!r}"
            head, body = _source_call_body(alias, term, spec)
            call = f"{head}({spec.instance!r}, {body})"
    except (KeyError, TypeError, ValueError) as e:
        raise _source_error(src, e, owner, position) from None
    return f"tb.add({call})"


def _metric_line(m: dict) -> str:
    """Emit ``tb.add_metric('name', type='...', signal='...', **params)``.

    A metric that binds its inputs by slot emits the ``signals={...}`` form
    instead — the same one ``add_metric`` takes, keys in the order
    the spec wrote them. ``signal`` is untouched, so a v1 spec still emits the
    byte-identical line it always did.

    ``analysis`` emits as its own keyword, before the params, and
    only when the spec carries one — a spec without it emits exactly the line
    it emitted before.
    """
    line = f"tb.add_metric({m['name']!r}, type={m['type']!r}, "
    if m.get('signals') is not None:
        binding = ', '.join(f"{k!r}: {v!r}" for k, v in m['signals'].items())
        line += f"signals={{{binding}}}"
    else:
        line += f"signal={m['signal']!r}"
    if m.get('analysis') is not None:
        line += f", analysis={m['analysis']!r}"
    for k, v in (m.get('params') or {}).items():
        line += f", {k}={_fmt_param(v)}"
    return line + ")"


def _spec_line(s: dict) -> str:
    """Emit ``tb.add_spec('name', 'metric', 'op', target)``."""
    return (f"tb.add_spec({s['name']!r}, {s['metric']!r}, {s['op']!r}, "
            f"{_fmt_param(s['target'])})")


def _analysis_line(a: dict) -> str:
    """Emit ``tb.add_sim_type(analysis.<Kind>(...))`` — delegates to
    generator._sim_code(receiver='tb') so analysis kinds stay in one place."""
    from silk.codegen.generator import _sim_code

    return _sim_code(a, receiver='tb')


def _format_helper_array_nets(nets: dict) -> str:
    """Format an arrayed helper's ``nets`` dict for emission.

    String values containing ``$i`` / ``$(i±N)`` placeholders (see
    generator._rewrite_template) are rewritten into f-string fragments;
    other values are emitted as plain string literals. Unlike
    generator._format_array_nets_dict, there is no bus-typed-name
    handling here: testbench helper nets are always plain net-name
    strings, never Net handles.
    """
    from silk.codegen.generator import _rewrite_template

    if not nets:
        return '{}'
    parts = []
    for port, value in nets.items():
        rewritten, has_placeholder = _rewrite_template(value)
        if has_placeholder:
            parts.append(f"{port!r}: f{rewritten!r}")
        else:
            parts.append(f"{port!r}: {value!r}")
    return '{' + ', '.join(parts) + '}'


def _module_add_lines(helpers: list, helper_aliases: dict) -> list:
    """Emit helper instances identically in rig and test bodies.

    A flat helper (``array`` unset) emits one ``tb.add_module(...)``
    statement. An arrayed helper emits a ``for i in range(...):`` loop
    registering one instance per expanded index, mirroring
    generator._emit_submodule_instances's list-comprehension emission for
    circuit-level submodules. Net values with ``$i`` / ``$(i±N)``
    placeholders are rewritten to f-string fragments.
    """
    from silk.codegen.generator import _emit_per_copy_nets

    lines = []
    for h in helpers:
        alias = helper_aliases[(h.import_path, h.module_var)]
        if h.array is None:
            lines.append(f"tb.add_module({alias}, {h.instance_name!r}, {h.nets!r})")
            continue
        range_lit = (f'range({h.array})' if isinstance(h.array, int)
                     else f'range({h.array[0]}, {h.array[1]})')
        inst_template = (
            repr(f"{h.instance_name}[{{i}}]") if h.naming == 'bracketed'
            else repr(f"{h.instance_name}_{{i}}"))
        inst_fstring = 'f' + inst_template
        if isinstance(h.nets, list):
            nets_literal = _emit_per_copy_nets(h, lines, repr)
            lines.append(
                f"for i in {range_lit}: tb.add_module("
                f"{alias}, {inst_fstring}, nets={nets_literal})")
            continue
        nets_literal = _format_helper_array_nets(h.nets)
        lines.append(f"for i in {range_lit}:")
        lines.append(f"    tb.add_module({alias}, {inst_fstring}, {nets_literal})")
    return lines


def _element_lines(group: dict, kind: str) -> list:
    """Emit a rig/test's ``devices`` / ``ideal_elements`` entries as inline
    ``tb.add(<factory call>)`` lines. The call expressions come from the
    circuit generator (``_device_call`` / ``_device_array_call``), so net and
    parameter rendering is shared; no variable is bound to an instance name,
    so a part can never clobber ``tb`` / ``pdk`` / ``mod`` in the generated
    function. ``pdk`` is bound from ``tb.pdk_details`` when devices exist."""
    from silk.codegen.generator import (
        CircuitSpec, _device_call, _device_array_call,
        _emit_per_copy_device, _validate_per_copy_nets)
    try:
        parsed = CircuitSpec.from_dict({
            'devices': group.get('devices', []),
            'ideal_elements': group.get('ideal_elements', [])})
    except (KeyError, TypeError, ValueError) as e:
        raise TestbenchSpecError(f"{kind} {group['name']!r}: {e}") from None
    for d in parsed.devices + parsed.ideal_elements:
        _reject_variable_refs(f"{kind} {group['name']!r} entry {d.instance!r}",
                              d.params.items())
    lines = []
    if parsed.devices:
        lines.append('pdk = tb.pdk_details')
    for source, items in (('pdk', parsed.devices), ('elements', parsed.ideal_elements)):
        for d in items:
            try:
                _validate_per_copy_nets(
                    d.nets, d.array, f"entry {d.instance!r}")
                if isinstance(d.nets, list):
                    _emit_per_copy_device(d, lines, source, 'tb')
                elif d.array is None:
                    lines.append(f"tb.add({_device_call(d, source)})")
                else:
                    lines.append(f"tb.add(*{_device_array_call(d, source)})")
            except (KeyError, TypeError, ValueError) as e:
                raise TestbenchSpecError(f"{kind} {group['name']!r}: {e}") from None
    return lines


def _entry_names(entry: dict, where: str) -> list:
    """Every name an element/source entry claims: the bare instance, or
    (arrayed) the list variable plus the expanded per-index names."""
    from silk.codegen.generator import _parse_array_and_naming
    inst = entry.get('instance')
    if not isinstance(inst, str):
        return []
    array, naming = _parse_array_and_naming(entry, where)
    if array is None:
        return [inst]
    lo, hi = (0, array) if isinstance(array, int) else array
    fmt = "{}[{}]" if naming == 'bracketed' else "{}_{}"
    return [inst] + [fmt.format(inst, i) for i in range(lo, hi)]


def _validate_elements(spec: 'TestbenchSpec', rig_helpers: dict,
                       test_helpers: dict) -> None:
    """Refuse a device / ideal element that shares a name, case-insensitively
    (SPICE), with another device, element, source or helper in the same test
    (its applied rigs share the namespace) or rig."""
    def claims(kind, group, helpers):
        origin = f"{kind} {group['name']!r}"
        out = []
        for key in ('devices', 'ideal_elements'):
            raw = group.get(key, [])
            if not isinstance(raw, list):
                raise TestbenchSpecError(f"{origin}: {key} must be a list")
            for e in raw:
                if not isinstance(e, dict):
                    raise TestbenchSpecError(
                        f"{origin}: {key} entry must be an object: {e!r}")
                try:
                    names = _entry_names(e, f"{origin} {key} entry {e.get('instance')!r}")
                except ValueError as exc:
                    raise TestbenchSpecError(str(exc)) from None
                out += [(f"{key} entry", n) for n in names]
        for e in group.get('sources') or []:
            try:
                names = _entry_names(e, f"{origin} source")
            except ValueError as exc:
                raise TestbenchSpecError(str(exc)) from None
            out += [("source", n) for n in names]
        for h in helpers:
            out += [("helper", n) for n in _expanded_helper_names(h)]
        return origin, out

    rig_claims = {r['name']: claims('rig', r, rig_helpers[r['name']])
                  for r in spec.rigs}
    test_claims = {t['name']: claims('test', t, test_helpers[t['name']])
                   for t in spec.tests}
    checks = [(f"rig {n!r}", [rig_claims[n]]) for n in rig_claims]
    for t in spec.tests:
        checks.append((f"test {t['name']!r}",
                       [rig_claims[r] for r in t.get('applied_rigs', [])]
                       + [test_claims[t['name']]]))
    for label, groups in checks:
        from silk.circuit.modules_results import DUT_ELEMENT_NAME
        seen = {DUT_ELEMENT_NAME: 'the DUT'}
        for origin, names in groups:
            for what, n in names:
                if n.lower() in seen:
                    raise TestbenchSpecError(
                        f"{label}: {what} {n!r} in {origin} collides "
                        f"(case-insensitive) with {seen[n.lower()]}")
                seen[n.lower()] = f"{what} {n!r} in {origin}"


def _source_names(entries, where: str):
    """Designer names of every source in ``entries`` (arrayed sources expanded
    to their elements) and a ``{bare name: [element names]}`` map of the arrays."""
    names, arrays = [], {}
    for e in entries or []:
        try:
            expanded = _entry_names(e, where)
        except ValueError as exc:
            raise TestbenchSpecError(str(exc)) from None
        if len(expanded) > 1 or (isinstance(e.get('instance'), str)
                                 and e.get('array') is not None):
            arrays[expanded[0]] = expanded[1:]
            names += expanded[1:]
        else:
            names += expanded
    return names, arrays


def _validate_sweep_sources(spec: 'TestbenchSpec') -> None:
    """Refuse a dc or noise analysis whose source is not one source element of
    its test or of the rigs it applies. A sweep names ONE element:
    ``vin[1]`` is valid, the bare array name ``vin`` is not."""
    from silk.circuit.sweep_sources import (
        missing_source_problem, sweep_source_problem)
    rigs = {r['name']: r for r in spec.rigs}
    for t in spec.tests:
        tname = t['name']
        wanted = [a for a in t.get('analyses', [])
                  if a['kind'].lower() in ('dc', 'noise')]
        if not wanted:
            continue
        groups = [rigs[r] for r in t.get('applied_rigs', [])] + [t]
        names, arrays = [], {}
        for g in groups:
            n, a = _source_names(
                g.get('sources') or [],
                f"test {tname!r} source")
            names += n
            arrays.update(a)
        for a in wanted:
            kind = a['kind'].lower()
            allow = ('temp',) if kind == 'dc' else ()
            if not str(a.get('source') or '').strip():
                raise TestbenchSpecError(
                    f"test {tname!r} {kind} analysis: "
                    f"{missing_source_problem(names, allow)}")
            problem = sweep_source_problem(a['source'], names, arrays, allow=allow)
            if problem:
                raise TestbenchSpecError(
                    f"test {tname!r} {kind} analysis: source {problem}")


def _testbench_source_specs(entries, owner="", contexts=None):
    """Parse and validate the source context used to resolve array controls."""
    from silk.codegen.generator import _source_entry_to_spec, _validate_ctrl_vname_references
    parsed = []
    for position, src in enumerate(entries, 1):
        try:
            parsed.append(_source_entry_to_spec(src, 'testbench sources entry'))
        except (KeyError, TypeError, ValueError) as e:
            entry_owner, entry_position = (contexts[position - 1] if contexts
                                           else (owner, position))
            raise _source_error(src, e, entry_owner, entry_position) from None
    try:
        _validate_ctrl_vname_references(parsed, array_controls_only=True)
    except (KeyError, TypeError, ValueError) as e:
        raise TestbenchSpecError(f"{owner}: {e}" if owner else str(e)) from None
    return parsed


def _rig_helper(rig: dict, helpers=(), helper_aliases=None) -> str:
    """Emit a shared rig's sources and helper modules, or pass if both are empty."""
    entries = rig.get('sources') or []
    owner = f"rig {rig['name']!r}"
    parsed = _testbench_source_specs(entries, owner)
    lines = [_source_add_line(s, parsed, owner, position)
             for position, s in enumerate(entries, 1)]
    lines.extend(_element_lines(rig, 'rig'))
    lines.extend(_module_add_lines(helpers, helper_aliases or {}))
    body = "\n".join(f"    {line}" for line in lines) if lines else "    pass"
    return f"def _apply_{rig['name']}_rig(tb):\n{body}"


def _parse_submodules(group: dict, kind: str) -> list:
    """Parse helper references using the schematic vocabulary, with rig/test context."""
    from silk.codegen.generator import SubmoduleSpec
    from silk.utils import require_python_identifier

    where = f"{kind} {group['name']!r} submodules"
    entries = group.get('submodules', [])
    if not isinstance(entries, list):
        raise TestbenchSpecError(f"{where}: must be a list")
    result = []
    for index, entry in enumerate(entries):
        context = f"{where}[{index}]"
        if not isinstance(entry, dict):
            raise TestbenchSpecError(f"{context}: must be an object")
        context += f" ({entry.get('instance_name')!r})"
        try:
            unsupported = {'parameters'} & entry.keys()
            if unsupported:
                raise ValueError(
                    f"unsupported helper fields: {', '.join(sorted(unsupported))}")
            if 'module_ref' in entry and not isinstance(entry['module_ref'], dict):
                raise ValueError("module_ref must be an object")
            helper = SubmoduleSpec.from_dict(entry)
            require_python_identifier(helper.instance_name, 'instance_name')
            require_python_identifier(helper.module_var, 'module_ref.module_var')
            path = helper.import_path
            if not isinstance(path, str) or not path:
                raise ValueError("module_ref.import_path must be a non-empty import path")
            # Leading dots express package-relative imports, including 'from .'.
            tail = path.lstrip('.')
            if tail:
                for part in tail.split('.'):
                    require_python_identifier(part, 'module_ref.import_path component')
            copies = helper.nets if isinstance(helper.nets, list) else [helper.nets]
            for connections in copies:
                for port in connections:
                    if not isinstance(port, str):
                        raise ValueError(f"nets port {port!r} must be a string")
        except (KeyError, TypeError, ValueError) as exc:
            raise TestbenchSpecError(f"{context}: {exc}") from None
        result.append(helper)
    return result


def _expanded_helper_names(helper) -> list:
    """The instance name(s) `helper` actually registers at run time.

    A flat helper (``array`` unset) registers its bare ``instance_name``.
    An arrayed helper registers one expanded name per index — the same
    strings generator._emit_submodule_instances embeds into its
    f-string, so collisions are caught on what tb.add_module actually
    sees, not on the pre-expansion base name.
    """
    if helper.array is None:
        return [helper.instance_name]
    lo, hi = ((0, helper.array) if isinstance(helper.array, int)
              else helper.array)
    if helper.naming == 'bracketed':
        return [f"{helper.instance_name}[{i}]" for i in range(lo, hi)]
    return [f"{helper.instance_name}_{i}" for i in range(lo, hi)]


def _validate_helper_names(spec: 'TestbenchSpec', rigs: dict, tests: dict) -> None:
    """Each assembled test has one instance namespace, with 'dut' reserved.

    Checked on EXPANDED names: an arrayed helper's collision
    surface is every index it registers, not its pre-expansion base name.
    """
    for test in spec.tests:
        from silk.circuit.modules_results import DUT_ELEMENT_NAME
        seen = {DUT_ELEMENT_NAME: 'the DUT'}
        groups = [(f"rig {name!r}", rigs[name])
                  for name in test.get('applied_rigs', [])]
        groups.append((f"test {test['name']!r}", tests[test['name']]))
        for origin, helpers in groups:
            for helper in helpers:
                for name in _expanded_helper_names(helper):
                    if name in seen:
                        raise TestbenchSpecError(
                            f"test {test['name']!r}: helper instance {name!r} in "
                            f"{origin} collides with {seen[name]}")
                    seen[name] = origin


def _helper_aliases(spec: 'TestbenchSpec', helpers: list) -> dict:
    """Reserve bindings in the whole generated file, including function locals."""
    from silk.codegen.generator import _compute_submodule_aliases

    reserved = {'tb', 'mod', 'pdk', 'analysis', 'sources'}
    reserved.update(f"__silk_{h.instance_name}_nets" for h in helpers
                    if isinstance(h.nets, list))
    reserved.update(f"__silk_{entry['instance']}_nets"
                    for group in [*spec.rigs, *spec.tests]
                    for key in ('devices', 'ideal_elements')
                    for entry in group.get(key, [])
                    if isinstance(entry, dict) and isinstance(entry.get('nets'), list)
                    and 'instance' in entry)
    reserved.update(f"_apply_{rig['name']}_rig" for rig in spec.rigs)
    reserved.update(f"_attach_{test['name']}" for test in spec.tests)
    if spec.circuit_builder:
        reserved.add(spec.circuit_builder)
        reserved.add(f"build_{spec.circuit_name}_with_tests")
    # Derived signals are locals of _attach_<test> (TASK-373); an alias must
    # not be shadowed by one.
    for test in spec.tests:
        if _derived_entries(test):
            reserved.add('metrics')
            reserved.update(d['name'] for d in _derived_entries(test))
    return _compute_submodule_aliases(
        helpers,
        additional_reserved=reserved,
    )


# --- Whole-function assembly (Task 8) ---------------------------------------
#
# generate_testbenches is the top-level entry point: validate, then emit the
# imports + rig helpers + per-test _attach_<test> functions + the
# build_<circuit>_with_tests assembler. Every emitted line delegates to the
# Task 7 per-line helpers above, so the API surface stays in one place.

def _imports(spec: 'TestbenchSpec', helper_aliases: dict) -> list:
    """Emit the import lines the generated body needs.

    ``from silk.simulation import analysis`` if any test carries analyses;
    ``from silk import sources`` if any rig or test carries sources.
    Deliberately **no** metric import (§6.3): the declarative
    ``tb.add_metric(type=...)`` resolves the type string from the metric
    registry at runtime, so the generated code needs no
    ``silk.metrics`` import at all. Returns ``[]`` (no leading
    blank line) if the spec needs no source, analysis or helper import.
    """
    has_analyses = any(t.get('analyses') for t in spec.tests)
    has_sources = (any(r.get('sources') for r in spec.rigs)
                   or any(t.get('sources') for t in spec.tests))

    groups = [*spec.rigs, *spec.tests]
    has_elements = any(g.get('ideal_elements') for g in groups)
    lines = []
    if has_elements:
        lines.append('from silk.elements import elements')
    if has_analyses:
        lines.append('from silk.simulation import analysis')
    if has_sources:
        lines.append('from silk import sources')
    if any(_metric_function_form(m, {d['name'] for d in _derived_entries(t)})
           for t in spec.tests for m in t.get('metrics', [])):
        lines.append('from silk import metrics')
    for (path, module_var), alias in helper_aliases.items():
        lines.append(f'from {path} import {module_var} as {alias}')
    if lines:
        lines.append('')
    return lines


def _attach_fn(t: dict, helpers: list, helper_aliases: dict,
               save_device_params: bool = False, source_specs=()) -> str:
    """Emit a full ``def _attach_<test>(mod): ...`` function for one test
    dict: ``mod.create_test``, ``add_module_under_test``, one
    ``_apply_<rig>_rig(tb)`` call per applied rig, then the test's own
    sources / helper modules / analyses / metrics / specs.

    ``save_device_params`` (the spec's top-level key) sets it on
    the DUT first thing, in every attach function, so the key works with or
    without a ``circuit_builder`` assembler. A test's ``corner`` and
    ``temperature_c`` follow ``add_module_under_test`` as ``tb.set_corner``
    and ``tb.set_temperature``. Absent keys emit nothing.
    """
    name = t['name']
    lines = [f"def _attach_{name}(mod):"]
    if save_device_params:
        lines.append("    mod.save_device_params = True")
    lines.append(f"    tb = mod.create_test({name!r})")
    lines.append(f"    tb.add_module_under_test({t['dut_portmap']!r})")
    if t.get('corner') is not None:
        lines.append(f"    tb.set_corner({t['corner']!r})")
    if t.get('temperature_c') is not None:
        lines.append(f"    tb.set_temperature({float(t['temperature_c'])!r})")
    for rig in t.get('applied_rigs', []):
        lines.append(f"    _apply_{rig}_rig(tb)")
    for position, src in enumerate(t.get('sources', []), 1):
        line = _source_add_line(src, source_specs, f"test {name!r}", position)
        lines.append(f"    {line}")
    lines.extend(f"    {line}" for line in _element_lines(t, 'test'))
    lines.extend(f"    {line}" for line in _module_add_lines(helpers, helper_aliases))
    for a in t.get('analyses', []):
        lines.append(f"    {_analysis_line(a)}")
    lines.extend(f"    {line}" for line in _derived_signal_lines(t))
    derived_names = {d['name'] for d in _derived_entries(t)}
    widths = _derived_widths(t)
    for m in t.get('metrics', []):
        if _metric_function_form(m, derived_names):
            lines.extend(f"    {line}" for line in
                         _derived_metric_lines(m, derived_names, widths,
                                               helper_aliases.values()))
            continue
        lines.append(f"    {_metric_line(m)}")
    for s in t.get('specs', []):
        lines.append(f"    {_spec_line(s)}")
    return '\n'.join(lines)


def _builder_fn(spec: 'TestbenchSpec') -> str:
    """Emit ``def build_<circuit_name>_with_tests(pdk): ...`` — calls the
    caller-provided ``circuit_builder(pdk)``, attaches every test, returns
    the module. Caller (generate_testbenches) omits this entirely when
    ``spec.circuit_builder`` is falsy. ``circuit_builder`` is interpolated
    bare as a call target, so ``_validate`` has already required it to be a
    bare, non-keyword Python identifier."""
    lines = [f"def build_{spec.circuit_name}_with_tests(pdk):"]
    lines.append(f"    mod = {spec.circuit_builder}(pdk)")
    for t in spec.tests:
        lines.append(f"    _attach_{t['name']}(mod)")
    lines.append("    return mod")
    return '\n'.join(lines)


def generate_testbenches(spec) -> str:
    """Validate `spec` (§2.1) then return the whole-function generated
    testbench Python: imports, rig helpers, one ``_attach_<test>`` per
    test, and (if ``circuit_builder`` is given) the
    ``build_<circuit>_with_tests(pdk)`` assembler.

    Accepts a plain dict (parsed via ``TestbenchSpec.from_dict``) or an
    already-built ``TestbenchSpec``. Raises ``TestbenchSpecError`` — never
    a bare ``KeyError`` or other uncaught exception — on any structurally
    invalid spec, including a required key (rig/test/metric/spec ``name``,
    test ``dut_portmap``) missing from a declarative dict; ``_validate``
    checks the named cases explicitly, and the ``except KeyError`` below
    is a defense-in-depth backstop over the whole assembly so silk never
    emits broken Python and never lets a bare KeyError escape.
    """
    if isinstance(spec, dict):
        spec = TestbenchSpec.from_dict(spec)
    elif not isinstance(spec, TestbenchSpec):
        raise TestbenchSpecError(
            f"generate_testbenches: spec must be a dict or TestbenchSpec, got "
            f"{type(spec).__name__}")
    try:
        _validate(spec)
        rig_helpers = {r['name']: _parse_submodules(r, 'rig') for r in spec.rigs}
        test_helpers = {t['name']: _parse_submodules(t, 'test') for t in spec.tests}
        _validate_helper_names(spec, rig_helpers, test_helpers)
        _validate_elements(spec, rig_helpers, test_helpers)
        helper_aliases = _helper_aliases(
            spec, [h for group in [*rig_helpers.values(), *test_helpers.values()]
                   for h in group])
        _validate_sweep_sources(spec)
        lines = _imports(spec, helper_aliases)
        for rig in spec.rigs:
            lines.append(_rig_helper(rig, rig_helpers[rig['name']], helper_aliases))
            lines.append('')
        for t in spec.tests:
            groups = [(r, f"rig {r['name']!r}") for r in spec.rigs
                      if r['name'] in t.get('applied_rigs', [])]
            groups.append((t, f"test {t['name']!r}"))
            entries, contexts = [], []
            for group, owner in groups:
                sources = group.get('sources') or []
                entries.extend(sources)
                contexts.extend((owner, position) for position in range(1, len(sources) + 1))
            parsed = _testbench_source_specs(entries, f"test {t['name']!r}", contexts)
            lines.append(_attach_fn(t, test_helpers[t['name']], helper_aliases,
                                    spec.save_device_params, parsed))
            lines.append('')
        if spec.circuit_builder:
            lines.append(_builder_fn(spec))
    except KeyError as e:
        raise TestbenchSpecError(
            f"testbench spec missing required key {e}") from None
    return '\n'.join(lines).rstrip() + '\n'
