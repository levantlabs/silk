"""Hierarchical specs, their measurements, and reconciliation.

A :class:`Spec` is a declared target on a metric of a module (``measured <op>
target``). Measurements of that metric are recorded on it, one live slot per
source and corner, and ``Module.reconcile_specs()`` turns them into a
:class:`SpecReconciliationReport` of :class:`SpecOutcome` verdicts. A
:class:`CalibrationBinding` writes a measured value back onto a behavioural
model parameter.

- :class:`Spec`: declared target on a module's metric.
- :class:`Measurement`: one observation of a spec's metric value.
- :class:`SpecAttempt`: one journalled attempt, successful or failed.
- :class:`AggregateMeasurement`: attribution of a Monte Carlo statistic.
- :class:`CalibrationBinding`: write-back binding for behavioural parameters.
- :class:`SpecOutcome` and :class:`SpecReconciliationReport`: reconciliation
  results.
- :class:`ComparisonOp`, :class:`MeasurementSource` and :class:`Status`: the typed
  strings those classes use.
"""
from silk.specs.attempt import SpecAttempt
from silk.specs.spec import (
    CalibrationBinding,
    ComparisonOp,
    Measurement,
    MeasurementSource,
    Spec,
)
from silk.specs.report import (
    SpecOutcome,
    SpecReconciliationReport,
    Status,
)

__all__ = [
    'AggregateMeasurement',
    'CalibrationBinding',
    'ComparisonOp',
    'Measurement',
    'MeasurementSource',
    'Spec',
    'SpecAttempt',
    'SpecOutcome',
    'SpecReconciliationReport',
    'Status',
]

from silk.specs.aggregate import AggregateMeasurement
