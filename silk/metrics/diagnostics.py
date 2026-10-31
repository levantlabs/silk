"""Structured warnings accompanying numeric metric results.

``MetricWarning`` exposes five payload attributes: ``metric`` (originating
function name), ``code`` (stable reason identifier), ``message`` (observation),
``details`` (a dict of scalar facts), and ``suggestion`` (optional next step).
Detail names carry units, e.g. ``frequency_min_hz``; magnitude values use the
explicit ``magnitude_unit`` label. Counts are sample counts. Magnitude ranges
exclude NaNs and use ``None`` for both bounds when no non-NaN sample remains.
No waveforms, report context, or persistence objects belong in this payload.

Warnings follow Python's warning policy: ignoring them preserves the numeric
return; promoting them to errors raises ``MetricWarning`` instead. Delivery
can be filtered or deduplicated, and this module never durably captures a
diagnostic itself -- ``ModuleTestbench.evaluate()``'s extraction loop is what
attributes each warning raised while one output's extractor runs to that
output's name and keeps it (as a :class:`MetricWarningRecord`) on the
resulting ``SpecReport``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import warnings


class MetricWarning(UserWarning):
    """Structured explanation accompanying an unavailable or limited measurement.

    Parameters
    ----------
    metric : str
        Originating metric function name.
    code : str
        Stable reason identifier.
    message : str
        Human-readable observation.
    details : dict of str to object
        Scalar facts copied into a new dict. Unit-bearing field names or explicit
        unit labels identify units; counts are dimensionless. Nested objects are
        not copied or validated.
    suggestion : str or None, optional
        Suggested next step; default None.
    output : str or None, optional
        Declared testbench output name, default None. The extraction loop
        sets this context; ``emit()`` does not.

    Returns
    -------
    MetricWarning
        Warning instance with the supplied payload attributes and a formatted
        message. Construction does not emit the warning.

    Notes
    -----
    Normal warning filters apply when emitted: ignoring keeps the metric's
    numeric result, while promotion to errors raises this warning. This
    class does not measure data or substitute a numeric failure result.
    """

    def __init__(
        self, *, metric: str, code: str, message: str,
        details: dict[str, object], suggestion: Optional[str] = None,
        output: Optional[str] = None,
    ) -> None:
        self.metric = metric
        self.code = code
        self.message = message
        self.details = dict(details)
        self.suggestion = suggestion
        self.output = output
        facts = ', '.join(f'{key}={value!r}' for key, value in self.details.items())
        prefix = metric if output is None else f'output {output!r} ({metric})'
        text = f'{prefix}: {message} [{code}; {facts}]'
        if suggestion:
            text += f' {suggestion}'
        super().__init__(text)


@dataclass(frozen=True)
class MetricWarningRecord:
    """A durable, JSON-storable copy of one :class:`MetricWarning`'s payload.

    Holds exactly the five structured attributes a live ``MetricWarning``
    carries -- never the warning/exception object itself, a traceback, or
    any waveform (the same rule ``emit()``'s payload already follows). This
    is what ``ModuleTestbench.evaluate()`` attaches to ``SpecReport.
    metric_warnings`` and what ``store_report()`` persists.
    """
    metric: str
    code: str
    message: str
    details: dict = field(default_factory=dict)
    suggestion: Optional[str] = None
    output: Optional[str] = None

    @classmethod
    def from_warning(cls, warning: MetricWarning) -> 'MetricWarningRecord':
        """Build a record from a live ``MetricWarning`` instance."""
        return cls(
            metric=warning.metric, code=warning.code, message=warning.message,
            details=dict(warning.details), suggestion=warning.suggestion,
            output=warning.output,
        )

    def to_dict(self) -> dict:
        """JSON-safe representation, for ``store_report()``."""
        return {
            'metric': self.metric, 'code': self.code, 'message': self.message,
            'details': dict(self.details), 'suggestion': self.suggestion,
            'output': self.output,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'MetricWarningRecord':
        """Inverse of :meth:`to_dict`, for reading a stored report back."""
        return cls(
            metric=d['metric'], code=d['code'], message=d['message'],
            details=dict(d.get('details') or {}), suggestion=d.get('suggestion'),
            output=d.get('output'),
        )


def emit(
    *, metric: str, code: str, message: str,
    details: dict[str, object], suggestion: Optional[str] = None,
) -> None:
    """Emit one structured warning, attributed to the metric's branch."""
    warnings.warn(
        MetricWarning(metric=metric, code=code, message=message,
                      details=details, suggestion=suggestion),
        stacklevel=2,
    )
