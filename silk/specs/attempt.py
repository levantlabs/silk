"""Immutable observations used by design reconciliation.

The live journal belongs to a Spec, not to a report or SQL history. A report
exports only its selected latest and last successful extraction snapshots.
"""
from contextvars import ContextVar
from dataclasses import dataclass, replace
import json
import math
from numbers import Real
from uuid import uuid4

from silk.specs.aggregate import AggregateMeasurement, decode_aggregate, encode_aggregate


_evaluation_attempt = ContextVar("spec_evaluation_attempt", default=None)


def _pack(value):
    from silk._json_codec import encode_json
    return json.dumps(encode_json(value), allow_nan=False, sort_keys=True)


def _unpack(value):
    from silk._json_codec import decode_json
    return decode_json(json.loads(value))


@dataclass(frozen=True, init=False)
class SpecAttempt:
    """One completed attempt to observe a spec's metric.

    The journal of a :class:`~silk.specs.Spec` is made of attempts. A
    successful extraction (``kind='measurement'``) carries a measurement, which
    may still have a failing numeric verdict. A failed extraction or execution
    has a reason and no measurement. Requested conditions never assert what a
    simulator actually proved. Zero time means unknown. Only ``'unavailable'``
    attempts may carry separate aggregate attribution; it describes the
    campaign without presenting an unavailable statistic as a value.

    The instance is immutable. :attr:`measurement`, :attr:`requested_conditions`
    and :attr:`proved_conditions` return a fresh copy on each access.

    Parameters
    ----------
    source : str
        Where the attempt ran, such as ``'spice'`` or ``'behavioral'``; a
        non-empty string.
    timestamp : float, optional
        Unix time in seconds of the attempt; ``0.0`` means unknown. Must be
        finite and not negative.
    kind : {'measurement', 'extractor_error', 'unavailable', 'simulation_error', 'transport_error'}, optional
        ``'measurement'`` is a successful extraction; ``'extractor_error'``
        means the metric extractor failed; ``'unavailable'`` means the metric
        has no usable value (NaN); ``'simulation_error'`` means the simulation
        failed; ``'transport_error'`` means the outcome of a worker run could
        not be delivered.
    reason : str, optional
        Why a failed attempt failed; required (non-empty) unless
        ``kind='measurement'``, and empty for it.
    attempt_id : str, optional
        Identity of the attempt. A random identifier is generated when omitted.
    measurement : Measurement, optional
        Required for ``kind='measurement'`` (with a real, non-NaN ``value``, a
        boolean ``passes``, and ``source`` and ``timestamp`` equal to this
        attempt's); forbidden otherwise.
    test_name : str, optional
        Name of the testbench that produced the attempt.
    report_key : str, optional
        Key of the report the attempt belongs to.
    failure_stage : str, optional
        Label of the stage at which a failed attempt failed (for example
        ``'transport'``).
    requested_conditions : dict, optional
        The operating conditions that were requested; stored as JSON.
    proved_conditions : dict, optional
        The operating conditions the simulator proved; stored as JSON.
    aggregate : AggregateMeasurement, optional
        Campaign attribution; allowed only with ``kind='unavailable'``.

    Raises
    ------
    ValueError
        *source* is empty, *kind* is unknown, *timestamp* is negative or not
        finite, the measurement / reason combination does not match *kind*,
        *aggregate* is given for another kind, or an identifier is empty.

    Examples
    --------
    >>> from silk.specs import SpecAttempt
    >>> attempt = SpecAttempt(source='spice', kind='simulation_error',
    ...                       reason='simulator exited early')
    >>> attempt.kind, attempt.measurement is None
    ('simulation_error', True)
    """
    #: Identity of the attempt, unique within its journal.
    attempt_id: str
    #: Where the attempt ran, such as ``'spice'``.
    source: str
    #: Unix seconds of the attempt; ``0.0`` means unknown.
    timestamp: float
    #: Attempt kind; ``'measurement'`` marks a successful extraction.
    kind: str
    #: Why a failed attempt failed; empty for ``kind='measurement'``.
    reason: str
    #: Name of the testbench that produced the attempt, or ``None``.
    test_name: str
    #: Key of the report the attempt belongs to, or ``None``.
    report_key: str
    #: Stage at which a failed attempt failed, or ``None``.
    failure_stage: str
    #: Separate campaign attribution carried by an unavailable attempt, or
    #: ``None``.
    aggregate: AggregateMeasurement | None
    _measurement: str
    _requested: str
    _proved: str

    def __init__(self, *, source, timestamp=0., kind='measurement', reason='',
                 attempt_id=None, measurement=None, test_name=None,
                 report_key=None, failure_stage=None, requested_conditions=None,
                 proved_conditions=None, aggregate=None):
        from silk.specs.spec import Measurement
        from silk.simulation.conditions import encode_condition
        kinds = {'measurement', 'extractor_error', 'unavailable',
                 'simulation_error', 'transport_error'}
        if not isinstance(source, str) or not source:
            raise ValueError('Attempt source must be a nonempty string')
        for name, value in (('test_name', test_name), ('report_key', report_key),
                            ('failure_stage', failure_stage)):
            if value is not None and not isinstance(value, str):
                raise ValueError(f'Attempt {name} must be a string or None')
        if kind not in kinds:
            raise ValueError(f'Unknown attempt kind {kind!r}')
        if not isinstance(timestamp, Real) or not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError('Attempt timestamp must be finite and nonnegative')
        if kind == 'measurement':
            if (not isinstance(measurement, Measurement)
                    or not isinstance(measurement.value, Real)
                    or isinstance(measurement.value, bool)
                    or math.isnan(measurement.value)
                    or not isinstance(measurement.passes, bool)):
                raise ValueError('Measurement attempt requires a usable real value and verdict')
            if measurement.source != source or measurement.timestamp != timestamp or reason != '':
                raise ValueError('Measurement attempt source/time must match its payload and have no failure reason')
        elif measurement is not None or not isinstance(reason, str) or not reason:
            raise ValueError('Failed attempt requires a reason and no measurement')
        identity = uuid4().hex if attempt_id is None else attempt_id
        if aggregate is not None and kind != 'unavailable':
            raise ValueError(f'Attempt {identity!r}: separate aggregate requires kind unavailable')
        try:
            aggregate = decode_aggregate(encode_aggregate(aggregate))
        except (TypeError, ValueError) as exc:
            raise ValueError(f'Attempt {identity!r}: {exc}') from exc
        data = None
        if measurement is not None:
            # Observation facts only. Database/profile projection metadata never
            # becomes producer provenance in a selected-attempt export.
            data = {name: getattr(measurement, name) for name in (
                'source', 'value', 'passes', 'corner', 'timestamp', 'sim_run_id',
                'notes', 'temperature', 'voltage', 'sim_type', 'tool_name')}
            if measurement.aggregate is not None:
                data['aggregate'] = encode_aggregate(measurement.aggregate)
            for key in ('corner', 'temperature', 'voltage'):
                data[key] = encode_condition(data[key])
        values = dict(attempt_id=identity, source=source, aggregate=aggregate,
                      timestamp=float(timestamp), kind=kind, reason=reason,
                      test_name=test_name, report_key=report_key,
                      failure_stage=failure_stage, _measurement=_pack(data),
                      _requested=_pack(requested_conditions), _proved=_pack(proved_conditions))
        if not isinstance(values['attempt_id'], str) or not values['attempt_id']:
            raise ValueError('Attempt ID must be a nonempty string')
        for name, value in values.items():
            object.__setattr__(self, name, value)

    @property
    def measurement(self):
        """The observation as a new :class:`~silk.specs.Measurement`.

        ``None`` for a failed attempt. A fresh object is built on each access,
        so mutating it does not change the journalled attempt.
        """
        data = _unpack(self._measurement)
        if data is None:
            return None
        from silk.specs.spec import Measurement
        from silk.simulation.conditions import decode_condition
        for key in ('corner', 'temperature', 'voltage'):
            data[key] = decode_condition('supply' if key == 'voltage' else key, data[key])
        data['aggregate'] = decode_aggregate(data.get('aggregate'))
        return Measurement(**data)

    @property
    def requested_conditions(self):
        """The operating conditions that were requested, or ``None``.

        A new object decoded from the stored JSON on each access. These are
        requests; what a simulator established is :attr:`proved_conditions`.
        """
        return _unpack(self._requested)

    @property
    def proved_conditions(self):
        """The operating conditions the run proved, or ``None``.

        A new object decoded from the stored JSON on each access.
        """
        return _unpack(self._proved)

    def as_dict(self):
        """Return the attempt as a JSON-compatible dict.

        Returns
        -------
        dict
            Keys ``attempt_id``, ``source``, ``timestamp``, ``kind``,
            ``reason``, ``test_name``, ``report_key``, ``failure_stage``,
            ``measurement`` (a dict or ``None``), ``aggregate`` (the encoded
            aggregate or ``None``), ``requested_conditions`` and
            ``proved_conditions``. Accepted back by :meth:`from_dict`.
        """
        return dict(attempt_id=self.attempt_id, source=self.source,
                    timestamp=self.timestamp, kind=self.kind, reason=self.reason,
                    test_name=self.test_name, report_key=self.report_key,
                    failure_stage=self.failure_stage,
                    measurement=_unpack(self._measurement),
                    aggregate=encode_aggregate(self.aggregate),
                    requested_conditions=self.requested_conditions,
                    proved_conditions=self.proved_conditions)

    @classmethod
    def from_dict(cls, data):
        """Rebuild an attempt from the dict :meth:`as_dict` produced.

        Parameters
        ----------
        data : dict
            The mapping; its ``measurement`` entry is a dict or ``None``. The
            mapping itself is not modified.

        Returns
        -------
        SpecAttempt
            The attempt, validated as by the constructor.

        Raises
        ------
        ValueError
            The aggregate payload is malformed, or the attempt violates the
            constructor's rules.
        """
        from silk.specs.spec import Measurement
        from silk.simulation.conditions import decode_condition
        data = dict(data)
        try:
            data['aggregate'] = decode_aggregate(data.get('aggregate'))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Attempt {data.get('attempt_id')!r}: {exc}") from exc
        m = data.get('measurement')
        if m is not None:
            m = dict(m)
            for key in ('corner', 'temperature', 'voltage'):
                m[key] = decode_condition('supply' if key == 'voltage' else key, m.get(key))
            m['aggregate'] = decode_aggregate(m.get('aggregate'))
            data['measurement'] = Measurement(**m)
        return cls(**data)


def _merge(old, new):
    """Validate immutable payload; only fill absent provenance on promotion."""
    if old == new:
        return old
    a, b = old.as_dict(), new.as_dict()
    for name in ('attempt_id', 'source', 'timestamp', 'kind', 'reason'):
        if a[name] != b[name]:
            raise ValueError(f'Attempt {old.attempt_id!r} conflicts in {name}')
    for name in ('test_name', 'report_key', 'failure_stage',
                 'requested_conditions', 'proved_conditions', 'aggregate'):
        if b[name] is not None:
            if a[name] is not None and a[name] != b[name]:
                raise ValueError(f'Attempt {old.attempt_id!r} conflicts in {name}')
            a[name] = b[name]
    if a['measurement'] is not None:
        for name, value in b['measurement'].items():
            previous = a['measurement'].get(name)
            if name in ('value', 'passes', 'timestamp', 'source'):
                if previous != value:
                    raise ValueError(f'Attempt {old.attempt_id!r} conflicts in {name}')
            elif value is not None and value != '':
                if previous is not None and previous != '' and previous != value:
                    raise ValueError(f'Attempt {old.attempt_id!r} conflicts in provenance {name}')
                a['measurement'][name] = value
    return SpecAttempt.from_dict(a)


def scope_for(spec, measurement=None):
    from silk.specs.evidence import _current, bind_current
    if measurement is not None:
        bind_current(spec, measurement)
        return getattr(measurement, '_owner_scope', None)
    current = _current.get()
    if current is None:
        return None
    # Reuse the existing weak owner registration and cleanup.
    from silk.specs.spec import Measurement
    marker = Measurement('user', 0)
    bind_current(spec, marker)
    return marker._owner_scope


def prepare_attempt(spec, attempt, scope):
    measurement = attempt.measurement
    if measurement is not None and measurement.passes != bool(spec.evaluate(measurement)):
        raise ValueError('Attempt measurement verdict conflicts with the owning Spec')
    for previous_scope, old in spec.__dict__.get('_attempts', ()):
        if previous_scope == scope and old.attempt_id == attempt.attempt_id:
            return _merge(old, attempt)
    return attempt


def apply_attempt(spec, attempt, scope):
    if scope is not None and scope[0]() is None:
        return
    journal = spec.__dict__.setdefault('_attempts', [])
    for i, (previous_scope, old) in enumerate(journal):
        if previous_scope == scope and old.attempt_id == attempt.attempt_id:
            journal[i] = (scope, attempt)
            return
    journal.append((scope, attempt))


def selected_attempts(spec):
    from silk.specs.evidence import _current
    journal = spec.__dict__.get('_attempts', ())
    current = _current.get()
    if current is None:
        return tuple(a for scope, a in journal if scope is None or scope[0]() is not None)
    index, owner = current
    if any(scope is not None for scope, _ in journal):
        return tuple(a for scope, a in journal if scope is not None
                     and scope[0]() is index and scope[1] == owner)
    return tuple(a for _, a in journal)


def prepare_measurement(spec, m):
    """Validate before publication, including before a database commit."""
    scope = scope_for(spec, m)
    projection = getattr(m, '_projection_observation', None)
    if projection is not None:
        observation, origin_scope = projection
        snapshot = observation.measurement
        if (snapshot is not None and
                (m.value != snapshot.value or m.source != snapshot.source
                 or m.timestamp != snapshot.timestamp
                 or bool(spec.evaluate(m)) != snapshot.passes)):
            raise ValueError('Stored projection conflicts with its observation payload')
        return prepare_attempt(spec, observation, origin_scope), origin_scope
    if getattr(m, '_legacy_evidence', False):
        return None, scope
    identity = getattr(m, '_attempt_id', None)
    if identity is None:
        identity = uuid4().hex
        m._attempt_id = identity
    valid = isinstance(m.value, Real) and not isinstance(m.value, bool) and not math.isnan(m.value)
    if not valid and m.aggregate is not None:
        from silk.specs.aggregate import validate_attribution
        validate_attribution(m.aggregate, spec.metric, m.sim_run_id)
    snapshot = replace(m, passes=bool(spec.evaluate(m))) if valid else None
    attempt = SpecAttempt(
        attempt_id=identity, source=m.source, timestamp=m.timestamp or 0.,
        kind='measurement' if valid else 'unavailable',
        reason='' if valid else 'metric value is unavailable (NaN)',
        measurement=snapshot, aggregate=m.aggregate if not valid else None,
        report_key=getattr(m, '_report_key', None),
        **getattr(m, '_attempt_context', {}))
    return prepare_attempt(spec, attempt, scope), scope


def classify_metric(value, *, metric, metric_errors, metric_warnings):
    """Classify one extracted metric value; pure, publishes nothing.

    Returns ``(kind, value, reason)``. ``kind`` is ``'measured'`` for a real
    number (infinities included, compared by the spec as always; ``value`` is
    that number and ``reason`` is ``None``), ``'extractor_error'`` for a metric
    whose extractor raised, and ``'unavailable'`` for NaN or any other value
    with no usable real reading (``value`` is ``None``; ``reason`` explains).
    """
    numeric = isinstance(value, Real) and not isinstance(value, bool)
    if numeric and not math.isnan(value):
        return 'measured', float(value), None
    error = metric_errors.get(metric)
    reason = (f"extractor error: {error['message']}" if error else
              f'metric {metric!r} is unavailable (NaN)' if numeric else
              f'metric {metric!r} has no usable real value: {value!r}')
    warnings = metric_warnings.get(metric, ())
    if warnings:
        reason += '; ' + '; '.join(w.message for w in warnings)
    kind = 'extractor_error' if error or isinstance(value, str) else 'unavailable'
    return kind, None, reason


def publish_metric(spec, value, *, conditions, report, metric):
    """One classification/publication point for every evaluated spec metric."""
    import time
    from silk.specs.spec import Measurement
    source = 'behavioral' if conditions.backend == 'behavioral' else 'spice'
    frame = _evaluation_attempt.get()
    prepared, timestamp = frame if frame is not None else (None, time.time())
    if prepared is not None and prepared["suppressed"]:
        return
    identity = prepared["attempt_id"] + ":" + spec.key if prepared is not None else uuid4().hex
    context = dict(test_name=report.test_name, proved_conditions=conditions.to_dict(),
                   requested_conditions=prepared["requested_conditions"] if prepared is not None else None)
    numeric = isinstance(value, Real) and not isinstance(value, bool)
    unavailable = numeric and math.isnan(value)
    if numeric:
        m = Measurement(source, float(value), passes=bool(spec.evaluate(Measurement(source, value))),
                        timestamp=timestamp, corner=conditions.scalar('corner'),
                        temperature=conditions.scalar('temperature'),
                        voltage=conditions.scalar('supply'))
        m._attempt_id = identity
        m._report_key = report.report_key
        m._attempt_context = context
        if not unavailable:
            spec.record_measurement(m)
            report.results[-1]._observation = (
                report.owner_path, spec.key, spec.metric, spec.op, spec.target,
                prepare_measurement(spec, m)[0])
            return
        # Preserve the established NaN observation API, but not as usable history.
        m._legacy_evidence = True
        spec.record_measurement(m)
    kind, _, reason = classify_metric(
        value, metric=metric, metric_errors=report.metric_errors,
        metric_warnings=report.metric_warnings)
    attempt = SpecAttempt(attempt_id=identity, source=source, timestamp=timestamp,
        kind=kind,
        reason=reason, report_key=report.report_key, **context)
    spec.record_attempt(attempt)
    report.results[-1]._observation = (
        report.owner_path, spec.key, spec.metric, spec.op, spec.target, attempt)
