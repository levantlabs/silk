"""Walk a list of Specs and produce a SpecReconciliationReport.

Called by ``Module.reconcile_specs()``. It operates on a list of Spec objects
so it can be tested in isolation.
"""
from __future__ import annotations

import math
import time
from typing import Any, Iterable, List, Optional

from silk.specs.spec import Measurement, Spec
from silk.specs.report import SpecOutcome, SpecReconciliationReport, Status
from silk.specs.resolver import resolve_metric


def reconcile(
    module: Any,
    specs: Iterable[Spec],
    prefer_source: Optional[str] = None,
) -> SpecReconciliationReport:
    """Evaluate each spec and return a SpecReconciliationReport.

    Parameters
    ----------
    module : Module
        The root module; owns the evidence index the specs are reconciled
        against.
    specs : iterable of Spec or iterable of tuple
        The specs to reconcile. An ``(owner_path, spec)`` tuple places the spec
        at that path; a bare spec is placed at ``'$'``.
    prefer_source : str, optional
        Prefer evidence from this source when it has any; otherwise the
        latest of any source is used.

    Returns
    -------
    SpecReconciliationReport
        One outcome per input item, in input order.

    Raises
    ------
    ConfigurationError
        Two specs would calibrate the same parameter with different values, or
        two bindings target the same ``Variable``. No parameter is changed.
    VariationError
        A binding's ``field`` does not match the parameter it targets.
    NotSupportedError
        A binding targets a ``Variable`` expression, which cannot be
        calibrated.
    RuntimeError
        A binding targets a module with no ``_behavioral_params`` dict.
    KeyError
        A binding names a parameter its target module does not declare.
        Both are checked for every binding whose measurement source is in
        its ``source_filter``, while staging, so no parameter is changed.

    Notes
    -----
    For each spec placement:

    - Resolve dotted metrics against the owning module and retained annotation.
      A retained observation keeps its original identity and time. Legacy dotted
      values without an attempt remain user-source measurements of unknown time;
      reading them does not append attempts or clear a recorded failure.

    - Select the latest recorded attempt, preferring ``prefer_source`` when
      that source has attempt evidence. Fall back to the legacy measurement
      selection only when no attempt is recorded.

    - A failed latest attempt emits ``FAIL`` with no current measurement and
      retains the separately labelled last successful extraction, if any.
      No attempt and no legacy measurement emits ``UNMEASURED``.

    - Evaluate a successful extraction or legacy measurement as ``PASS`` or
      ``FAIL``. Update the legacy live measurement's ``passes`` cache while
      keeping historical attempt snapshots immutable. Failed-latest rows
      cannot calibrate.

    The returned report is assembled in the input iteration order.

    When a spec carries a ``CalibrationBinding`` and the reconciled
    measurement's source is allowed by the binding's ``source_filter``,
    the measured value is written onto
    ``binding.target_module._behavioral_params[binding.param_name]`` as
    a side effect, after all specs have been staged; see
    :class:`~silk.specs.CalibrationBinding` for when a binding is skipped.
    """
    outcomes: List[SpecOutcome] = []
    calibrations_applied: List[str] = []
    candidates = {}

    from silk.circuit.test_placement import evidence_index
    from silk.specs.evidence import owner_scope
    index = evidence_index(module)
    for item in specs:
        owner, spec = item if isinstance(item, tuple) else ('$', item)
        with owner_scope(index, owner):
            # Pillar 2 hook: resolve dotted-path metrics into a user-source Measurement.
            if '.' in spec.metric:
                from silk.specs.evidence import annotation_reader
                rr = resolve_metric(spec.module, spec.metric,
                                    reader=annotation_reader(index, owner, spec.module))
                if rr.resolved and rr.value is not None and spec.latest_attempt() is None:
                    legacy = Measurement(
                        source='user',
                        value=rr.value,
                        timestamp=0.0,
                    )
                    legacy._legacy_evidence = True
                    spec.record_measurement(legacy)

            attempt = spec.latest_attempt(source=prefer_source) if prefer_source is not None else None
            if attempt is None:
                attempt = spec.latest_attempt()
            history = spec.attempt_history(source=prefer_source if prefer_source is not None
                                           and spec.latest_attempt(source=prefer_source) is not None else None)
            successful = [a for a in history if a.kind == 'measurement']
            last_success = max(enumerate(successful), key=lambda p: (p[1].timestamp, p[0]))[1] if successful else None
            live_measurement = _pick_measurement(spec, prefer_source)
            if live_measurement is not None:
                live_measurement.passes = bool(spec.evaluate(live_measurement))
            if attempt is not None and attempt.kind != 'measurement':
                outcomes.append(SpecOutcome(
                    spec=spec, measurement=None, status='FAIL', owner_path=owner,
                    report_key=attempt.report_key, source_used=attempt.source,
                    latest_attempt=attempt, last_successful_attempt=last_success))
                continue
            measurement = attempt.measurement if attempt is not None else live_measurement

            if measurement is None:
                outcomes.append(SpecOutcome(
                    spec=spec, measurement=None, status='UNMEASURED', owner_path=owner,
                ))
                continue

            passes = bool(spec.evaluate(measurement))
            measurement.passes = passes
            status: Status = 'PASS' if passes else 'FAIL'
            margin = abs(measurement.value - spec.target)

            calibrated = _apply_calibration(spec, measurement, calibrations_applied, apply=False)
            if calibrated:
                binding = spec.calibration
                key = (id(binding.target_module), binding.param_name, binding.field)
                previous = candidates.get(key)
                if previous is not None and previous[2].value != measurement.value:
                    from silk.exceptions import ConfigurationError
                    raise ConfigurationError(
                        f'Conflicting calibration for {binding.param_name!r} '
                        f'from {previous[0]} and {owner}; no parameters changed.')
                if previous is not None and binding.field is not None:
                    from silk.circuit.variable import Variable
                    from silk.exceptions import ConfigurationError
                    target = binding.target_module._behavioral_params[binding.param_name]
                    target = getattr(target, binding.field, None)
                    if isinstance(target, Variable):
                        raise ConfigurationError(
                            f'Two calibration bindings target Variable {target.name!r}; no parameters changed')
                candidates.setdefault(key, (owner, spec, measurement))

            outcomes.append(SpecOutcome(
                spec=spec,
                owner_path=owner,
                report_key=attempt.report_key if attempt is not None else getattr(measurement, '_report_key', None),
                latest_attempt=attempt, last_successful_attempt=last_success,
                measurement=measurement,
                status=status,
                margin=margin,
                source_used=measurement.source,
                calibrated=calibrated,
            ))

    from silk.specs._calibration import apply_calibrations
    apply_calibrations(list(candidates.values()), calibrations_applied, root=module)

    return SpecReconciliationReport(
        module=module,
        outcomes=outcomes,
        calibrations_applied=calibrations_applied,
        timestamp=time.time(),
    )


def _apply_calibration(
    spec: Spec,
    measurement: Measurement,
    log: List[str],
    *, apply=True,
) -> bool:
    """Validate and optionally apply the calibration binding for this measurement.
    Return True when the measurement is not NaN and the binding passes the
    source filter and target-parameter validation. With apply=False, no value
    is written.

    Called only for a spec with an eligible measurement. A spec with no
    measurement, or whose latest attempt failed, never reaches this function.

    Silent skip conditions (no exception, checked here):
    - No calibration binding attached to the spec.
    - The measurement source is not in the binding's source_filter.
    - The measurement value is NaN, before target validation or staging.

    Declaration errors (raised, matching Module.calibrate_behavioral):
    - The target module has no _behavioral_params dict (attribute is None):
      RuntimeError naming the parameter, the module and that it declares none.
    - The named parameter is not a key of that dict: KeyError naming the
      module, the parameter and the declared keys. Nothing is created; the
      user must fix the binding or declare the parameter.

    The raise happens while specs are staged, before apply_calibrations
    writes any value, so a refused reconciliation changes no parameter.
    """
    binding = spec.calibration
    if binding is None:
        return False
    if measurement.source not in binding.source_filter:
        return False
    if math.isnan(measurement.value):
        return False
    module_name = getattr(binding.target_module, 'module_name', binding.target_module)
    params = getattr(binding.target_module, '_behavioral_params', None)
    if params is None:
        raise RuntimeError(
            f"CalibrationBinding names parameter '{binding.param_name}' on module "
            f"'{module_name}', but '{module_name}' has no declared parameters "
            f"(_behavioral_params is None); call set_behavioral_model(fn, params={{...}}) first.")
    if binding.param_name not in params:
        raise KeyError(
            f"CalibrationBinding names parameter '{binding.param_name}', which is not "
            f"declared on module '{module_name}'. Declared params: "
            f"{sorted(params.keys())}")
    if apply:
        from silk.specs._calibration import apply_calibrations
        apply_calibrations([('$', spec, measurement)], log)

    return True


def _pick_measurement(spec: Spec, prefer_source: Optional[str]) -> Optional[Measurement]:
    """Apply prefer_source if it has a measurement; fall back to latest."""
    if prefer_source is not None:
        m = spec.latest(source=prefer_source)
        if m is not None:
            return m
    return spec.latest()
