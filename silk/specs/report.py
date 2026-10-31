"""The result types of spec reconciliation.

:class:`SpecOutcome` is the verdict for one spec and
:class:`SpecReconciliationReport` collects them, as produced by
``Module.reconcile_specs()``. They are kept apart from the declarative side
(:class:`~silk.specs.Spec`, :class:`~silk.specs.Measurement`,
:class:`~silk.specs.CalibrationBinding`) so the two can be imported
independently.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Literal, Optional

from silk.specs.spec import Measurement, Spec

if TYPE_CHECKING:
    from silk.circuit.modules import Module  # noqa: F401


#: Verdict of one reconciled spec, as a typed string: ``'PASS'``, ``'FAIL'``
#: (including a failed latest attempt) or ``'UNMEASURED'`` (nothing recorded).
Status = Literal['PASS', 'FAIL', 'UNMEASURED']


def _single_line(text: str) -> str:
    """Keep report labels from injecting terminal controls or extra rows."""
    return ''.join(' ' if ord(c) < 32 or 0x7f <= ord(c) < 0xa0 else c
                   for c in text)


@dataclass
class SpecOutcome:
    """Result of reconciling one Spec against its latest measurement.

    Parameters
    ----------
    spec : Spec
        The reconciled spec.
    measurement : Measurement or None
        The measurement the verdict is based on; ``None`` when the spec is
        unmeasured or its latest attempt failed.
    status : {'PASS', 'FAIL', 'UNMEASURED'}
        The verdict.
    margin : float, optional
        ``abs(measurement.value - spec.target)`` in the spec's units: a
        distance, not a signed slack, so it does not say on which side of the
        target the value lies. ``None`` without a measurement.
    source_used : str, optional
        Source of the attempt or measurement the verdict is based on; ``None``
        when unmeasured.
    calibrated : bool, optional
        True when the spec's calibration binding accepted this measurement for
        write-back.
    notes : str, optional
        Free text.
    owner_path : str or None, optional
        Placement of the spec in the module tree; ``'$'`` for the root,
        ``None`` for legacy evidence of unknown ownership.
    report_key : str, optional
        Key of the report the evidence came from.
    latest_attempt : SpecAttempt, optional
        The latest attempt considered, successful or failed.
    last_successful_attempt : SpecAttempt, optional
        The last successful extraction in the considered history. Reported
        separately from a failed latest attempt as historical evidence.
    """
    #: The reconciled spec.
    spec: Spec
    #: Measurement behind the verdict; ``None`` if unmeasured or the latest
    #: attempt failed.
    measurement: Optional[Measurement]
    #: Verdict, one of ``'PASS'``, ``'FAIL'`` or ``'UNMEASURED'``.
    status: Status
    #: Absolute distance between the value and the target, in the spec's
    #: units; ``None`` without a measurement.
    margin: Optional[float] = None
    #: Source of the evidence the verdict is based on.
    source_used: Optional[str] = None
    #: True when the calibration binding accepted the measurement.
    calibrated: bool = False
    #: Free-text notes.
    notes: str = ''
    #: Placement of the spec in the module tree (``'$'`` is the root), or
    #: ``None`` for evidence of unknown ownership.
    owner_path: Optional[str] = '$'
    #: Key of the report the evidence came from.
    report_key: Optional[str] = None
    #: Latest attempt considered, successful or failed.
    latest_attempt: Optional[Any] = None
    #: Last successful extraction in the considered history.
    last_successful_attempt: Optional[Any] = None

    @property
    def attempt_description(self) -> str:
        """Text describing a failed latest attempt, or ``''``.

        The failure and its time come first, followed by the last successful
        attempt labelled as historical evidence (or "no successful attempt").
        Empty when there is no latest attempt or it is a successful
        measurement.
        """
        a = self.latest_attempt
        if a is None or a.kind == 'measurement':
            return ''
        from datetime import datetime, timezone
        def when(timestamp):
            return (datetime.fromtimestamp(timestamp, timezone.utc).isoformat()
                    if timestamp else 'time unknown')
        text = f'latest attempt failed: {_single_line(a.reason)} ({when(a.timestamp)})'
        previous = self.last_successful_attempt
        if previous is None:
            return text + '; no successful attempt'
        m = previous.measurement
        verdict = 'PASS' if m.passes else 'FAIL'
        return (text + f'; last successful attempt: {m.value:g} {self.spec.units} '
                f'{verdict} ({when(previous.timestamp)})')

    @property
    def placement_label(self) -> str:
        """Placement suffix ``' (instance <instance_path>)'`` on one line.

        Empty for the root (``'$'``) or unknown ownership (``None``).
        Placed paths omit the leading ``'$/'``. Control characters are
        replaced by spaces.
        """
        if self.owner_path in (None, '$'):
            return ''
        return _single_line(f' (instance {self.owner_path.removeprefix("$/")})')

    @property
    def display_label(self) -> str:
        """Spec name followed by :attr:`placement_label`, on one line."""
        return _single_line(self.spec.name) + self.placement_label

    def as_dict(self) -> dict:
        """Return the outcome as a JSON-compatible dict.

        Returns
        -------
        dict
            Keys ``spec`` (name, key, metric, op, target, units,
            description), ``owner_path``, ``report_key``, ``measurement``
            (a dict or ``None``), ``status``, ``margin``, ``source_used``,
            ``calibrated``, ``notes``, ``latest_attempt`` and
            ``last_successful_attempt`` (each an attempt dict or ``None``).

        Raises
        ------
        ValueError
            A measurement's or attempt's aggregate attribution does not match
            the spec's metric or the measurement's ``sim_run_id``; the message
            names the outcome.
        """
        from silk.simulation.conditions import encode_condition
        from silk.specs.aggregate import encode_aggregate, validate_attribution

        def _m_to_dict(m: Optional[Measurement]) -> Optional[dict]:
            if m is None:
                return None
            try:
                validate_attribution(m.aggregate, self.spec.metric, m.sim_run_id)
            except (TypeError, ValueError) as exc:
                raise ValueError(f'Spec outcome {self.display_label!r}: {exc}') from exc
            return {
                'source': m.source,
                'value': m.value,
                'passes': m.passes,
                'corner': encode_condition(m.corner),
                'timestamp': m.timestamp,
                'sim_run_id': m.sim_run_id,
                'notes': m.notes,
                'temperature': encode_condition(m.temperature),
                'voltage': encode_condition(m.voltage),
                'sim_type': m.sim_type,
                'tool_name': m.tool_name,
                'origin_version_id': m.origin_version_id,
                'aggregate': encode_aggregate(m.aggregate),
            }
        for attempt in (self.latest_attempt, self.last_successful_attempt):
            if attempt is not None:
                _m_to_dict(attempt.measurement)
                if attempt.aggregate is not None:
                    try:
                        validate_attribution(attempt.aggregate, self.spec.metric,
                                             attempt.aggregate.campaign_id)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f'Spec outcome {self.display_label!r}, attempt {attempt.attempt_id!r}: {exc}') from exc
        return {
            'spec': {
                'name': self.spec.name,
                'key': self.spec.key,
                'metric': self.spec.metric,
                'op': self.spec.op,
                'target': self.spec.target,
                'units': self.spec.units,
                'description': self.spec.description,
            },
            'owner_path': self.owner_path,
            'report_key': self.report_key,
            'measurement': _m_to_dict(self.measurement),
            'status': self.status,
            'margin': self.margin,
            'source_used': self.source_used,
            'calibrated': self.calibrated,
            'notes': self.notes,
            'latest_attempt': self.latest_attempt.as_dict() if self.latest_attempt else None,
            'last_successful_attempt': self.last_successful_attempt.as_dict() if self.last_successful_attempt else None,
        }


@dataclass
class SpecReconciliationReport:
    """Aggregate result of reconciling every spec on a module tree.

    Parameters
    ----------
    module : Module or None
        The reconciled root module.
    outcomes : list of SpecOutcome, optional
        One outcome per reconciled spec placement, in input order.
    calibrations_applied : list of str, optional
        One line per calibration write-back that was applied.
    timestamp : float, optional
        Unix time in seconds at which the report was assembled; ``0.0`` if
        built by hand.

    Examples
    --------
    >>> from silk import Spec
    >>> from silk.specs import SpecOutcome, SpecReconciliationReport
    >>> spec = Spec('gain', None, 'gain_dB', '>=', 40, units='dB')
    >>> report = SpecReconciliationReport(
    ...     module=None, outcomes=[SpecOutcome(spec, None, 'UNMEASURED')])
    >>> report.passed, len(report.unmeasured)
    (False, 1)
    """
    #: The reconciled root module (``None`` in isolated tests).
    module: Any
    #: One outcome per reconciled spec placement, in input order.
    outcomes: List[SpecOutcome] = field(default_factory=list)
    #: One line per calibration write-back that was applied.
    calibrations_applied: List[str] = field(default_factory=list)
    #: Unix seconds at which the report was assembled.
    timestamp: float = 0.0

    # Nouns, no args, pure derivations over `outcomes` — properties, so the
    # natural `recon.fail_count == 0` is a real int comparison, not a
    # (always-False) bound-method comparison. Matches SpecReport/CornerResults,
    # which both expose `passed` as a property.
    @property
    def pass_count(self) -> int:
        """Number of outcomes with status ``'PASS'``."""
        return sum(1 for o in self.outcomes if o.status == 'PASS')

    @property
    def fail_count(self) -> int:
        """Number of outcomes with status ``'FAIL'`` (failed attempts included)."""
        return sum(1 for o in self.outcomes if o.status == 'FAIL')

    @property
    def unmeasured(self) -> List[SpecOutcome]:
        """The outcomes with status ``'UNMEASURED'``, in report order."""
        return [o for o in self.outcomes if o.status == 'UNMEASURED']

    @property
    def passed(self) -> bool:
        """True only if no spec failed and none is unmeasured.

        An unmeasured spec is not a passing spec, so a report with no
        outcomes at all is ``True`` but a report containing one unmeasured
        spec is ``False``.
        """
        return self.fail_count == 0 and not self.unmeasured

    def summary(self) -> str:
        """Return a plain-text summary.

        Returns
        -------
        str
            A header with the PASS, FAIL and UNMEASURED counts, one line per
            outcome (value and target with the spec's units, or the failed
            attempt description), and the applied calibrations if any.
            Placement labels immediately follow spec names.
        """
        lines = []
        lines.append(
            f'Spec reconciliation: '
            f'{self.pass_count} PASS, '
            f'{self.fail_count} FAIL, '
            f'{len(self.unmeasured)} UNMEASURED'
        )
        for o in self.outcomes:
            if o.attempt_description:
                lines.append(f'  [{o.status}] {o.display_label}: {o.attempt_description}')
            elif o.measurement is not None:
                val = f'{o.measurement.value:g}{o.spec.units}'
                tgt = f'{o.spec.op}{o.spec.target:g}{o.spec.units}'
                lines.append(f'  [{o.status}] {o.display_label}: {val} {tgt}')
            else:
                lines.append(f'  [{o.status}] {o.display_label}: (no measurement)')
        if self.calibrations_applied:
            lines.append('Calibrations applied:')
            lines.extend(f'  {c}' for c in self.calibrations_applied)
        return '\n'.join(lines)

    def as_dict(self) -> dict:
        """Return the report as a JSON-compatible dict.

        Returns
        -------
        dict
            Keys ``module_name`` (``None`` when the module has none),
            ``timestamp``, ``outcomes`` (each :meth:`silk.specs.SpecOutcome.as_dict`) and
            ``calibrations_applied``.

        Raises
        ------
        ValueError
            As :meth:`silk.specs.SpecOutcome.as_dict`.
        """
        return {
            'module_name': getattr(self.module, 'module_name', None),
            'timestamp': self.timestamp,
            'outcomes': [o.as_dict() for o in self.outcomes],
            'calibrations_applied': list(self.calibrations_applied),
        }
