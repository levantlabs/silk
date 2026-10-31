"""The Spec, Measurement and CalibrationBinding dataclasses.

A :class:`Spec` is a declared target on a metric of a module; a
:class:`Measurement` is one observed value of that metric; a
:class:`CalibrationBinding` names a behavioural-model parameter that a
measurement writes back to. Reconciliation (``Module.reconcile_specs``)
compares the two and reports :class:`~silk.specs.SpecReconciliationReport`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

from silk.simulation.conditions import ConditionState
from silk.specs.evidence import MeasurementSlots
from silk.specs.aggregate import AggregateMeasurement

if TYPE_CHECKING:
    from silk.circuit.modules import Module  # noqa: F401


#: The comparison a :class:`Spec` makes between a measured value and its
#: target, as a typed string: ``'<'``, ``'<='``, ``'>'``, ``'>='`` or ``'=='``.
#: The value is the first operand: ``Spec(op='>=', target=40)`` passes when the
#: measured value is greater than or equal to 40. The type is not enforced at
#: construction; an operator outside this set raises ``KeyError`` from
#: :meth:`Spec.evaluate`.
ComparisonOp = Literal['<', '<=', '>', '>=', '==']
#: Where a :class:`Measurement` came from, as a typed string: ``'behavioral'``
#: (behavioural simulation), ``'spice'`` (SPICE simulation), ``'extraction'``
#: (extracted layout) or ``'user'`` (supplied by the caller). The type is not
#: enforced by :class:`Measurement`; the source names the live slot and is
#: compared with a :class:`CalibrationBinding`'s ``source_filter``.
MeasurementSource = Literal['behavioral', 'spice', 'extraction', 'user']

_READ_ONLY_DECLARATION_FIELDS = frozenset({'target', 'op', 'metric', 'key'})


@dataclass
class Measurement:
    """One observation of a Spec's metric value.

    A Spec can hold multiple Measurements, one per (source, corner)
    pair, so that behavioral and SPICE results, or results from
    different corners, coexist. A measurement with a ``profile_id`` is keyed
    by (profile, source, corner, temperature, voltage) instead. Temperature,
    voltage, analysis, tool, and run/version IDs are provenance; they do not
    create additional live slots.

    ``corner`` and ``voltage`` may be ``ConditionState.NONE`` (the fact does
    not apply); ``temperature`` may also be ``ConditionState.SWEEP``. A
    ``None`` condition means unknown. The dataclass performs no validation of
    its fields.

    Parameters
    ----------
    source : {'behavioral', 'spice', 'extraction', 'user'}
        Where the value came from; see :class:`~silk.specs.MeasurementSource`.
    value : float
        The measured metric value, in the units of the :class:`Spec` it is
        recorded on. ``nan`` marks an unavailable value:
        :meth:`Spec.record_measurement` stores it but journals an
        ``'unavailable'`` attempt rather than a usable observation.
    passes : bool, optional
        Verdict against the owning spec. ``None`` until set by
        ``Module.reconcile_specs()`` or by a testbench that evaluated the spec;
        :meth:`Spec.record_measurement` does not fill it.
    corner : str or ConditionState, optional
        Process corner name, ``ConditionState.NONE`` when no corner applies, or
        ``None`` when unknown.
    timestamp : float, optional
        Unix time in seconds at which the value was observed. ``0.0`` (the
        default) means unknown; :meth:`Spec.record_measurement` does not stamp
        it, while measurements published by an evaluated testbench carry the
        evaluation time.
    sim_run_id : str, optional
        External run-correlation identifier. For an aggregate measurement it
        must equal the aggregate's ``campaign_id``.
    notes : str, optional
        Free text.
    temperature : float or ConditionState, optional
        Ambient temperature in degrees Celsius, ``ConditionState.SWEEP`` when
        the value spans a temperature sweep, ``ConditionState.NONE`` or
        ``None`` when absent or unknown.
    voltage : float or ConditionState, optional
        Supply voltage in volts, or ``ConditionState.NONE`` / ``None``.
    sim_type : str, optional
        Label of the analysis that produced the value (the database column is
        described as dc, ac, tran, noise, etc.); not validated.
    tool_name : str, optional
        Name of the tool that produced the value, for example ``'ngspice'``.
    origin_version_id : str, optional
        Identifier of the stored circuit version the evidence originated in;
        database provenance, not part of the live slot key.
    profile_id : str, optional
        Identifier of the characterization profile that scopes the
        measurement. When set it becomes part of the live slot key.
    aggregate : AggregateMeasurement, optional
        Attribution of a Monte Carlo statistic when ``value`` summarises a
        campaign rather than one run.

    Examples
    --------
    >>> from silk import Measurement
    >>> observed = Measurement(source='behavioral', value=42.0, passes=True)
    >>> observed.value, observed.passes
    (42.0, True)
    """
    #: Where the value came from; see :class:`~silk.specs.MeasurementSource`.
    source: MeasurementSource
    #: Measured metric value in the units of the owning spec; ``nan`` marks an
    #: unavailable value.
    value: float
    #: Verdict against the owning spec; ``None`` until reconciliation or an
    #: evaluating testbench sets it.
    passes: Optional[bool] = None
    #: Process corner name, ``ConditionState.NONE``, or ``None`` (unknown).
    corner: Optional[Union[str, ConditionState]] = None
    #: Unix seconds at which the value was observed; ``0.0`` means unknown.
    timestamp: float = 0.0
    #: Optional external run-correlation identifier.
    sim_run_id: Optional[str] = None
    #: Free-text note.
    notes: str = ''
    #: Ambient temperature in degrees Celsius, ``ConditionState.SWEEP`` /
    #: ``NONE``, or ``None`` (unknown).
    temperature: Optional[Union[float, ConditionState]] = None
    #: Supply voltage in volts, ``ConditionState.NONE``, or ``None`` (unknown).
    voltage: Optional[Union[float, ConditionState]] = None
    #: Label of the analysis that produced the value; not validated.
    sim_type: Optional[str] = None
    #: Name of the tool that produced the value; provenance only.
    tool_name: Optional[str] = None
    #: Stored circuit version the evidence originated in; provenance only.
    origin_version_id: Optional[str] = None
    #: Characterization profile that scopes the measurement; part of the
    #: live slot key when set.
    profile_id: Optional[str] = None
    #: Monte Carlo attribution when ``value`` is a campaign statistic.
    aggregate: Optional['AggregateMeasurement'] = None

    def __getstate__(self):
        # Runtime owner identity is never transported as persistent evidence.
        return {key: value for key, value in self.__dict__.items()
                if key not in ('_owner_scope', '_projection_observation')}



def _measurement_slot_key(m):
    key = _legacy_measurement_slot_key(m)
    scope = getattr(m, '_owner_scope', None)
    return ('owner', *scope, key) if scope is not None else key


def _legacy_measurement_slot_key(m: 'Measurement'):
    """Compute the live ``Spec.measurements`` dict key for one Measurement.

    Design 2026-08-31 (SS4.3): "a scoped slot is at least (profile_id,
    source, corner, temperature, voltage) so two profiles at one
    source@corner cannot overwrite each other." A Measurement with no
    ``profile_id`` keeps the exact pre-DB-02 legacy slot key -- ``source`` or
    ``f'{source}@{corner}'`` -- unchanged, so unscoped/legacy data is
    byte-for-byte unaffected and never assigned a synthetic profile. A
    Measurement carrying ``profile_id`` gets a tuple key that a plain
    ``source@corner`` string can never collide with, so a scoped and an
    unscoped observation -- or two different profiles -- coexist rather than
    overwrite.
    """
    if m.profile_id is not None:
        return ('profile', m.profile_id, m.source, m.corner, m.temperature, m.voltage)
    if m.corner is None:
        return m.source
    if isinstance(m.corner, ConditionState):
        # An unscoped none corner (design 2026-10-01 section 4.3): a tuple key
        # no source@corner string can equal, never 'spice@ConditionState.NONE'.
        return (m.source, m.corner)
    return f'{m.source}@{m.corner}'


@dataclass
class CalibrationBinding:
    """Write a measured value back onto a behavioural-model parameter.

    Attached to a :class:`Spec` (``declare_spec(..., calibration=binding)``).
    When ``Module.reconcile_specs()`` reconciles that spec against a
    measurement whose source is in ``source_filter``, it writes the measured
    value onto ``target_module._behavioral_params[param_name]``. The next
    behavioral simulation of ``target_module`` then reflects the measured value
    instead of a placeholder. The binding is skipped silently when the source is
    not in the filter, or when the failed latest attempt carries no measurement.
    A binding whose target module has no ``_behavioral_params`` dict, or whose
    ``param_name`` is not already a key of it, is a declaration error:
    ``reconcile_specs()`` raises before writing any value (``RuntimeError`` or
    ``KeyError``, as ``Module.calibrate_behavioral`` does). A missing key is
    never created.

    The default ``source_filter`` excludes ``'behavioral'`` to avoid
    feedback loops: behavioral measurements writing onto behavioral
    models would have nothing to calibrate against.

    Parameters
    ----------
    target_module : Module
        The module whose behavioural parameter is written.
    param_name : str
        Key in ``target_module._behavioral_params``. The key must already
        exist for the write to happen.
    source_filter : frozenset of str, optional
        Measurement sources allowed to calibrate. Defaults to
        ``frozenset({'spice', 'extraction'})``.
    field : str, optional
        For a parameter declared as a statistical distribution, the field to
        overwrite: ``'nominal'`` or ``'sigma'`` for a ``Normal``, ``'low'`` or
        ``'high'`` for a ``Uniform``. Leave ``None`` for a plain scalar
        parameter; reconciliation raises ``VariationError`` for a mismatch in
        either direction.

    Examples
    --------
    >>> from tempfile import TemporaryDirectory
    >>> from silk import Module, load_pdk
    >>> folder = TemporaryDirectory()
    >>> pdk = load_pdk('sky130')
    >>> dut = Module('sky130', folder.name, 'demo', pdk)
    >>> from silk import CalibrationBinding
    >>> binding = CalibrationBinding(dut, 'gain')
    >>> 'behavioral' in binding.source_filter
    False
    """
    #: The module whose ``_behavioral_params`` entry is written.
    target_module: Any    # 'Module'
    #: Key of the parameter in ``target_module._behavioral_params``.
    param_name: str
    #: Measurement sources allowed to calibrate; defaults to spice and
    #: extraction.
    source_filter: frozenset = field(default_factory=lambda: frozenset({'spice', 'extraction'}))
    #: Distribution field to overwrite (``'nominal'``, ``'sigma'``, ``'low'``
    #: or ``'high'``); ``None`` for a scalar parameter.
    field: Optional[str] = None


@dataclass
class Spec:
    """A declared target on a Module's metric.

    Owned by the Module on which it is declared (normally created with
    ``Module.declare_spec`` or ``ModuleTestbench.add_spec`` rather than
    directly). Hierarchy is explicit via ``parent``; a child is appended to its
    parent's ``children`` list at construction time.

    After successful construction, ``target``, ``op``, ``metric``, and
    ``key`` are read-only: assignment or deletion raises ``AttributeError``.
    Edit the declaration and rebuild the circuit/testbench to change them.
    Other fields, measurements, and runtime relationships remain mutable.

    Measurements arrive via :meth:`record_measurement` and are keyed by
    ``source`` (or ``source@corner`` when a corner is provided; a
    profile-scoped observation uses a tuple key
    ``('profile', profile_id, source, corner, temperature, voltage)``, and a
    structured corner condition a tuple as well), so the
    same spec can hold behavioral-vs-SPICE or multi-corner observations
    simultaneously. Recording another measurement under an existing key
    replaces the live one; the earlier observation stays in
    :meth:`attempt_history`.

    Parameters
    ----------
    name : str
        Spec name, as shown in reports. Also the default ``key``.
    module : Module or None
        The :class:`~silk.circuit.Module` that declares the spec (a back
        reference; ``None`` for a free-standing spec).
    metric : str
        Name of the metric the spec constrains. Read-only after construction.
    op : {'<', '<=', '>', '>=', '=='}
        Comparison applied as ``measured <op> target``. Read-only after
        construction.
    target : float
        Threshold, in the metric's own units. Must be a real number and not
        a ``bool``. Read-only after construction.
    units : str, optional
        Unit label for display (``'dB'``, ``'V'``); not interpreted.
    description : str, optional
        Free-text description.
    parent : Spec, optional
        Explicit parent; the new spec is appended to ``parent.children``.
    children : list of Spec, optional
        Child specs, in declaration order. Managed by the ``parent`` link.
    measurements : dict, optional
        Live measurements keyed by ``source`` or ``source@corner``, or by
        a tuple for profile-scoped or structured-corner observations (see
        the class description). Filled by :meth:`record_measurement`.
    calibration : CalibrationBinding, optional
        Calibration write-back declared for this spec.
    declared_at : str, optional
        Source location (``file:line``) where the spec was declared.
    key : str, optional
        Stable identity for the spec, a non-empty string. Defaults to
        ``name``. Read-only after construction.

    Raises
    ------
    TypeError
        If ``target`` is not a real number (``bool`` is rejected) or
        ``key`` is not a string.
    ValueError
        If ``key`` is empty or whitespace-only.

    Examples
    --------
    >>> from silk import Spec
    >>> spec = Spec('gain', module=None, metric='gain_dB', op='>=', target=40, units='dB')
    >>> from silk import Measurement
    >>> measured = Measurement('behavioral', 42)
    >>> spec.record_measurement(measured)
    >>> spec.evaluate(measured)
    True
    """
    #: Short label of the spec.
    name: str
    #: Owning ``Module`` (typed ``Any`` to avoid a circular import), or ``None``.
    module: Any
    #: Metric name, or a dotted ``net.``/``device.`` path; read-only after
    #: construction.
    metric: str
    #: Comparison operator applied as ``measured <op> target``; read-only after
    #: construction.
    op: 'ComparisonOp'
    #: Declared target in the metric's units; read-only after construction.
    target: float
    #: Unit label used in reports.
    units: str = ''
    #: Free-text description.
    description: str = ''
    #: Parent spec in the explicit hierarchy, or ``None``.
    parent: Optional['Spec'] = None
    #: Child specs, appended when a child is constructed with ``parent=self``.
    children: List['Spec'] = field(default_factory=list)
    #: Live measurement slots keyed by source, or ``source@corner``.
    measurements: Dict[str, Measurement] = field(default_factory=MeasurementSlots)
    #: Optional write-back binding applied during reconciliation.
    calibration: Optional['CalibrationBinding'] = None
    #: Location label of the declaration.
    declared_at: str = ''
    #: Stable local identity; ``name`` when not given; read-only after
    #: construction.
    key: Optional[str] = None

    def __post_init__(self) -> None:
        """Resolve the stable key, then attach to the explicit parent."""
        if isinstance(self.target, bool) or not isinstance(self.target, Real):
            raise TypeError(
                "Spec target must be a real number, got "
                f"{type(self.target).__name__}"
            )
        if self.key is None:
            self.key = self.name
        if not isinstance(self.key, str):
            raise TypeError(
                f"Spec key must be a string, got {type(self.key).__name__}"
            )
        if not self.key.strip():
            raise ValueError("Spec key must not be empty or whitespace-only")
        if self.parent is not None and self not in self.parent.children:
            self.parent.children.append(self)
        # Keep this outside the dataclass fields so its constructor and value
        # representation stay unchanged; copy/deepcopy preserve instance state.
        self._declaration_locked = True

    def _check_declaration_field(self, name: str) -> None:
        if (name in _READ_ONLY_DECLARATION_FIELDS
                and getattr(self, '_declaration_locked', False)):
            raise AttributeError(
                f"Spec.{name} is read-only after construction; "
                "edit the declaration and rebuild the circuit/testbench."
            )

    def __setattr__(self, name: str, value: Any) -> None:
        self._check_declaration_field(name)
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        self._check_declaration_field(name)
        object.__delattr__(self, name)

    def record_measurement(self, m: Measurement) -> None:
        """Attach a measurement. Does not trigger reconciliation.

        The measurement replaces any earlier one in the same slot (``source``,
        or ``source@corner``; a ``profile_id`` gets its own slot) and is
        recorded in the attempt history with its verdict computed by
        :meth:`evaluate`. A NaN value is recorded as an *unavailable*
        attempt, not as a numeric observation.

        Parameters
        ----------
        m : Measurement
            The observation to attach. Stored by reference in
            :attr:`measurements`.

        Raises
        ------
        ValueError
            If a measurement restored from a database conflicts with its
            stored observation payload or this spec's verdict.
        """
        from silk.specs.evidence import bind_current
        if not isinstance(self.measurements, MeasurementSlots):
            self.measurements = MeasurementSlots(self.measurements)
        bind_current(self, m)
        scope = getattr(m, '_owner_scope', None)
        if scope is not None and scope[0]() is None:
            return
        from silk.specs.attempt import prepare_measurement
        attempt, scope = prepare_measurement(self, m)
        self._publish_measurement(m, attempt, scope)

    def _publish_measurement(self, m, attempt, scope):
        """Apply previously validated evidence; safe for after-commit publication."""
        from silk.specs.attempt import apply_attempt
        self.measurements[_measurement_slot_key(m)] = m
        if attempt is not None:
            apply_attempt(self, attempt, scope)

    def record_attempt(self, attempt):
        """Record a completed attempt without inventing a numeric observation.

        Use this for extraction or simulation failures, which have no value.
        An attempt re-recorded under the same ``attempt_id`` is merged into
        the stored one.

        Parameters
        ----------
        attempt : SpecAttempt
            The attempt to record.

        Raises
        ------
        TypeError
            If ``attempt`` is not a ``SpecAttempt``.
        ValueError
            If the attempt's measurement verdict disagrees with this spec, or
            a re-recorded attempt conflicts with the stored one.
        """
        from silk.specs.attempt import SpecAttempt, scope_for, prepare_attempt, apply_attempt
        if not isinstance(attempt, SpecAttempt):
            raise TypeError('record_attempt requires a SpecAttempt')
        scope = scope_for(self)
        apply_attempt(self, prepare_attempt(self, attempt, scope), scope)

    def attempt_history(self, source=None):
        """Immutable snapshots for the selected owner, in publication order.

        Parameters
        ----------
        source : str, optional
            Keep only attempts from this source; all attempts when ``None``.

        Returns
        -------
        tuple of SpecAttempt
            The attempts, oldest publication first; empty when there are none.
        """
        from silk.specs.attempt import selected_attempts
        return tuple(a for a in selected_attempts(self)
                     if source is None or a.source == source)

    def latest_attempt(self, source=None):
        """Latest observation by original time, breaking ties by publication order.

        Parameters
        ----------
        source : str, optional
            Consider only attempts from this source.

        Returns
        -------
        SpecAttempt or None
            The attempt with the greatest ``timestamp`` (a later publication
            wins a tie), or ``None`` when there are no attempts.
        """
        attempts = self.attempt_history(source)
        return max(enumerate(attempts), key=lambda p: (p[1].timestamp, p[0]))[1] if attempts else None

    def latest(self, source: Optional['MeasurementSource'] = None) -> Optional[Measurement]:
        """Return the most recent live measurement.

        Parameters
        ----------
        source : {'behavioral', 'spice', 'extraction', 'user'}, optional
            Consider only measurements of this source.

        Returns
        -------
        Measurement or None
            The measurement with the greatest ``timestamp`` among the live
            slots, or ``None`` when there is none. This is the stored
            object itself, not a copy.
        """
        from silk.specs.evidence import selected_measurements
        candidates = [m for m in selected_measurements(self)
                      if source is None or m.source == source]
        if not candidates:
            return None
        return max(candidates, key=lambda m: m.timestamp)

    def evaluate(self, m: Measurement) -> bool:
        """Compare a measurement against this spec's target.

        Parameters
        ----------
        m : Measurement
            The observation; only ``m.value`` is read.

        Returns
        -------
        bool
            ``m.value <op> target``. Nothing is recorded or mutated.

        Raises
        ------
        KeyError
            If ``self.op`` is not one of the five supported operators. That
            is a deliberate fail-fast: an unknown op is a programming error,
            not runtime input.
        """
        ops = {
            '<':  lambda a, b: a <  b,
            '<=': lambda a, b: a <= b,
            '>':  lambda a, b: a >  b,
            '>=': lambda a, b: a >= b,
            '==': lambda a, b: a == b,
        }
        return ops[self.op](m.value, self.target)

    def __getstate__(self) -> Dict[str, Any]:
        """Pickle state without ``_owning_testbench``.

        A Spec declared via ``ModuleTestbench.add_spec`` carries a back-
        reference to its owning bench so the ``add_spec`` forwarder above
        can chain onto it. That bench can hold metric callables — a plain
        function or a user lambda, both ordinary usage — that pickle
        refuses outright. This Spec is copied onto a Module's ``_specs``
        every time that Module is embedded as another testbench's
        device-under-test (``add_module_under_test``), so pickling ANY
        netlist that embeds a DUT with declared specs would otherwise
        reach every sibling testbench's metrics through this one
        attribute — exactly the path a worker-bound netlist
        (``run_all_tests``/``run_corners``/``Optimizer.sweep``/
        ``FaultCampaign``, all via :mod:`silk.simulation.executor`) must
        not carry. Nothing on the worker side ever calls ``add_spec`` on
        a received copy, so dropping the reference here costs it nothing
        there.

        ``copy.deepcopy`` is unaffected by this — see ``__deepcopy__``.
        """
        from silk.specs.evidence import transient_free
        state = transient_free(self.__dict__)
        state.pop('_owning_testbench', None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)

    def __deepcopy__(self, memo):
        """Deep-copy every attribute, ``_owning_testbench`` included.

        Without this, the default deepcopy path (no ``__deepcopy__``)
        routes through ``__reduce_ex__``, which uses ``__getstate__``/
        ``__setstate__`` above and would silently drop
        ``_owning_testbench`` from every deep copy too — breaking the
        composition contract that copy relies on (a composed replica's
        ``add_spec`` forwarder must chain onto the COPIED testbench; see
        ``test_composition_copies_protected_specs_and_runtime_relationships``).
        """
        import copy as _copy
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        from silk.specs.evidence import transient_free
        for key, value in transient_free(self.__dict__).items():
            object.__setattr__(new, key, _copy.deepcopy(value, memo))
        return new

    def add_spec(self, name, metric_name, op, target, *, key=None):
        """Declare another spec on this spec's owning testbench.

        Chain-style forwarder: lets ``tb.add_spec(...).add_spec(...)`` read as
        a chain. Only valid for Specs created via
        :meth:`silk.circuit.module_test.ModuleTestbench.add_spec <silk.circuit.ModuleTestbench.add_spec>`,
        which sets ``spec._owning_testbench`` on the return value.

        Parameters
        ----------
        name : str
            Name of the new spec.
        metric_name : str
            Registered metric the new spec constrains.
        op : {'<', '<=', '>', '>=', '=='}
            Comparison operator.
        target : float
            Threshold in the metric's units.
        key : str, optional
            Stable identity for the new spec; defaults to ``name``.

        Returns
        -------
        Spec
            The new spec, as returned by the testbench's ``add_spec``, so
            calls can be chained.

        Raises
        ------
        RuntimeError
            If this Spec was created directly (e.g. via
            :meth:`silk.circuit.modules.Module.declare_spec <silk.circuit.Module.declare_spec>`) and has
            no owning testbench. In that case use ``module.declare_spec``
            and pass ``parent=`` explicitly to build hierarchy.
        """
        tb = getattr(self, '_owning_testbench', None)
        if tb is None:
            raise RuntimeError(
                f"Spec '{self.name}' was not created via a ModuleTestbench; "
                f"use module.declare_spec() and pass parent= explicitly for hierarchy."
            )
        return tb.add_spec(name, metric_name, op, target, key=key)
