"""Plain-data builders and validation for the standalone SpecReport contract.

No live circuit or storage APIs belong here. Outer writers encode once; future
collection writers can reuse these decoded builders without double encoding.
"""
from dataclasses import fields
import math

import numpy as np

from silk.db.context import CharacterizationContext
from silk.exceptions import CharacterizationContextError
from silk.metrics.diagnostics import MetricWarningRecord
from silk.simulation.provenance import RunProvenance, _report_field_names, _V3_FIELDS
from silk.simulation.conditions import (
    OperatingConditions, ConditionSweep, decode_condition, encode_condition,
)

FORMAT = 'silk.spec-report'
SCHEMA_VERSION = 1
_KEYS = (
    'format', 'schema_version', 'test_name', 'module_name', 'owner_path',
    'report_key', 'parameters', 'characterization_context', 'operating_conditions',
    'corner', 'rawfile_path', 'metrics', 'metric_types', 'metric_errors',
    'metric_warnings', 'behavioral_elements', 'storage', 'op_devices',
    'op_unavailable_reason', 'provenance', 'results', 'passed',
)


def plain_data(value, path='$', *, wire=False, active=None):
    """Copy supported data, normalize numpy scalars, and report bad field paths."""
    if active is None:
        active = set()
    if not wire and isinstance(value, (np.bool_, np.integer, np.floating, np.complexfloating)):
        if isinstance(value, np.bool_):
            value = bool(value)
        elif isinstance(value, np.integer):
            value = int(value)
        elif isinstance(value, np.floating):
            value = float(value)
        else:
            value = complex(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        if wire and isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f'{path}: nonfinite wire numbers must use numeric tags')
        return value
    if not wire and isinstance(value, complex):
        return value
    if not isinstance(value, (dict, list) if wire else (dict, list, tuple)):
        error = ValueError if wire else TypeError
        raise error(f'{path}: unsupported {type(value).__name__}; expected JSON data'
                    + ('' if wire else ' or a scalar number'))
    if id(value) in active:
        raise ValueError(f'{path}: circular container')
    active.add(id(value))
    try:
        if isinstance(value, dict):
            result = {}
            for key, child in value.items():
                if not isinstance(key, str):
                    error = ValueError if wire else TypeError
                    raise error(f'{path}: mapping keys must be strings, got {key!r}')
                result[key] = plain_data(child, f'{path}.{key}', wire=wire, active=active)
            return result
        return [plain_data(child, f'{path}[{i}]', wire=wire, active=active)
                for i, child in enumerate(value)]
    finally:
        active.remove(id(value))


def report_data(report):
    """Return the complete unencoded record, with no environment lookups."""
    context = report.characterization_context
    if context is not None:
        context = {name: getattr(context, name) for name in
                   ('owner_path', 'test_name', 'parameters', 'corner', 'temperature', 'voltage')}
        for name in ('corner', 'temperature', 'voltage'):
            context[name] = encode_condition(context[name])
    data = {name: getattr(report, name) for name in (
        'test_name', 'module_name', 'owner_path', 'report_key', 'parameters',
        'corner', 'rawfile_path', 'metric_errors', 'behavioral_elements', 'storage', 'op_devices')}
    data.update(
        format=FORMAT, schema_version=SCHEMA_VERSION,
        characterization_context=context,
        operating_conditions=(None if report.operating_conditions is None
                              else report.operating_conditions.to_dict()),
        metrics=dict(report.metrics),
        metric_types={key: report.metric_types.get(key) for key in report.metrics},
        metric_warnings={key: [record.to_dict() for record in records]
                         for key, records in report.metric_warnings.items()},
        op_unavailable_reason=report._op_unavailable_reason,
        # Read the same fields without the legacy provenance writer's lossy
        # string coercion. This boundary must preserve complex values and
        # reject unsupported objects/keys throughout the record.
        provenance=(None if report.provenance is None else
                    {name: getattr(report.provenance, name)
                     for name in _report_field_names(report.provenance)}),
        results=[{field.name: getattr(result, field.name) for field in fields(result)}
                 for result in report.results], passed=report.passed,
    )
    from silk.circuit.spec_report import _report_observation
    for result, item in zip(report.results, data['results']):
        attempt = _report_observation(result, report)
        if attempt is not None:
            item['observation'] = dict(identity=list(result._observation[:5]),
                                       attempt=attempt.as_dict())
    run = getattr(report, 'run', None)
    if run is not None:
        # TASK-499 optional keys: absent without a record, ignored by readers
        # that predate them, so the wire version stays 1.
        data.update(run=run.to_dict(), launch_id=run.launch_id)
    return plain_data(data)


def _require(value, typ, path, nullable=False):
    if nullable and value is None:
        return
    if not isinstance(value, typ) or (typ in (int, float) and isinstance(value, bool)):
        raise ValueError(f'{path}: expected {typ.__name__}' + (' or null' if nullable else ''))


def _record(value, keys, path):
    _require(value, dict, path)
    for key in keys:
        if key not in value:
            raise ValueError(f'{path}.{key}: missing required field')
    return {key: value[key] for key in keys}


def _strings(value, path):
    _require(value, list, path)
    for i, item in enumerate(value):
        _require(item, str, f'{path}[{i}]')


def _number(value, path, *, nullable=False, finite=False):
    if nullable and value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{path}: expected real number' + (' or null' if nullable else ''))
    if finite and not math.isfinite(value):
        raise ValueError(f'{path}: expected finite number')


def _construct(cls, data, path):
    try:
        return cls(**data)
    except (ValueError, TypeError, CharacterizationContextError) as exc:
        raise ValueError(f'{path}: {exc}') from exc


def _conditions(value):
    path = '$.operating_conditions'
    data = dict(_record(value, [f.name for f in fields(OperatingConditions)
                                if f.name != 'supply_sweep'], path))
    data['supply_sweep'] = value.get('supply_sweep')
    for name in ('origin', 'temperature_kind', 'temperature_basis', 'supply_basis',
                 'corner_kind', 'corner_basis', 'supply_kind'):
        _require(data[name], str, f'{path}.{name}')
    for name in ('backend', 'corner', 'supply_source'):
        _require(data[name], str, f'{path}.{name}', nullable=True)
    for name in ('temperature_c', 'supply_voltage_v'):
        _number(data[name], f'{path}.{name}', nullable=True, finite=True)
    for field in ('temperature_sweep', 'supply_sweep'):
        if data[field] is not None:
            sweep_path = f'{path}.{field}'
            sweep = _record(data[field], ('start', 'stop', 'step'), sweep_path)
            for name, number in sweep.items():
                _number(number, f'{sweep_path}.{name}', finite=True)
            data[field] = _construct(ConditionSweep, sweep, sweep_path)
    for kind, value in (('corner_kind', 'corner'),
                        ('temperature_kind', 'temperature_c'),
                        ('supply_kind', 'supply_voltage_v')):
        if (data[kind] == 'fixed') != (data[value] is not None):
            raise ValueError(f'{path}.{kind}: {value} is set only for fixed conditions')
    return _construct(OperatingConditions, data, path)


def _context(value):
    path = '$.characterization_context'
    data = _record(value, [f.name for f in fields(CharacterizationContext)], path)
    for name in ('owner_path', 'test_name'):
        _require(data[name], str, f'{path}.{name}')
    _require(data['parameters'], dict, f'{path}.parameters')
    for name, fact in (('corner', 'corner'), ('temperature', 'temperature'), ('voltage', 'supply')):
        try:
            data[name] = decode_condition(fact, data[name])
        except (ValueError, TypeError) as exc:
            raise ValueError(f'{path}.{name}: {exc}') from exc
    return _construct(CharacterizationContext, data, path)


def _provenance(value):
    path = '$.provenance'
    _require(value, dict, path)
    optional = (_V3_FIELDS if value.get('scheme') in {'silk.provenance/v1', 'silk.provenance/v2'}
                else frozenset())
    if value.get('scheme') == 'silk.provenance/v1':
        optional = optional | {'corner_kind', 'supply_kind'}
    names = [f.name for f in fields(RunProvenance)]
    data = _record(value, [name for name in names if name not in optional], path)
    data.update({name: value.get(name) for name in optional})
    structured = {'pdk_libraries', 'simulator_options', 'save_list', 'seed',
                  'temperature_c', 'supply_voltage_v', 'producing_test'}
    for name in data.keys() - structured:
        _require(data[name], str, f'{path}.{name}', nullable=name != 'scheme')
    for name in ('temperature_c', 'supply_voltage_v'):
        _number(data[name], f'{path}.{name}', nullable=True, finite=True)
    for name in ('simulator_options', 'save_list'):
        _require(data[name], dict, f'{path}.{name}', nullable=True)
    if data['producing_test'] is not None:
        binding = _record(data['producing_test'], ('owner_path', 'test_name'),
                          f'{path}.producing_test')
        for name, text in binding.items():
            _require(text, str, f'{path}.producing_test.{name}')
    libraries = data['pdk_libraries']
    if libraries is not None:
        _require(libraries, list, f'{path}.pdk_libraries')
        for i, library in enumerate(libraries):
            libpath = f'{path}.pdk_libraries[{i}]'
            library = _record(
                library, ('file', 'section', 'file_sha256', 'section_text_sha256'),
                libpath)
            for name, text in library.items():
                _require(text, str, f'{libpath}.{name}', nullable=True)
    return RunProvenance.from_dict(data)


def restore_report(data, cls):
    """Validate decoded data and rebuild stored evidence without rechecking."""
    # Local import: SpecReport calls this helper while restoring a record.
    from silk.circuit.spec_report import SpecResult, _OPS

    _require(data, dict, '$')
    if data.get('format') != FORMAT:
        raise ValueError(f'$.format: expected {FORMAT!r}')
    if type(data.get('schema_version')) is not int or data['schema_version'] != SCHEMA_VERSION:
        raise ValueError('$.schema_version: expected supported integer version 1')
    wire = data
    data = _record(data, _KEYS, '$')
    for name in ('test_name', 'module_name', 'owner_path'):
        _require(data[name], str, f'$.{name}')
    for name in ('report_key', 'corner', 'rawfile_path', 'op_unavailable_reason'):
        _require(data[name], str, f'$.{name}', nullable=True)
    for name in ('parameters', 'metrics', 'metric_types', 'metric_errors', 'metric_warnings'):
        _require(data[name], dict, f'$.{name}')
    if data['metric_types'].keys() != data['metrics'].keys():
        raise ValueError('$.metric_types: keys must match every metrics output')
    for name, value in data['metric_types'].items():
        _require(value, str, f'$.metric_types.{name}', nullable=True)
    for name, error in data['metric_errors'].items():
        path = f'$.metric_errors.{name}'
        if name not in data['metrics']:
            raise ValueError(f'{path}: output is absent from metrics')
        error = _record(error, ('message', 'exception_type', 'available_signals'), path)
        for key in ('message', 'exception_type'):
            _require(error[key], str, f'{path}.{key}')
        if error['available_signals'] is not None:
            _strings(error['available_signals'], f'{path}.available_signals')
    warnings = {}
    for name, records in data['metric_warnings'].items():
        path = f'$.metric_warnings.{name}'
        _require(records, list, path)
        restored = []
        for i, record in enumerate(records):
            recordpath = f'{path}[{i}]'
            record = _record(record, [f.name for f in fields(MetricWarningRecord)], recordpath)
            for key in ('metric', 'code', 'message', 'suggestion', 'output'):
                _require(record[key], str, f'{recordpath}.{key}',
                         nullable=key in ('suggestion', 'output'))
            _require(record['details'], dict, f'{recordpath}.details')
            restored.append(MetricWarningRecord.from_dict(record))
        warnings[name] = tuple(restored)
    _strings(data['behavioral_elements'], '$.behavioral_elements')
    _require(data['storage'], dict, '$.storage', nullable=True)
    if data['storage'] is not None:
        receipt = _record(data['storage'], ('metrics_stored', 'specs_stored', 'skipped'), '$.storage')
        for key in ('metrics_stored', 'specs_stored'):
            _require(receipt[key], int, f'$.storage.{key}')
            if receipt[key] < 0:
                raise ValueError(f'$.storage.{key}: expected nonnegative count')
        _require(receipt['skipped'], dict, '$.storage.skipped')
        for key, value in receipt['skipped'].items():
            _require(value, str, f'$.storage.skipped.{key}')
        if 'cached' in data['storage']:
            _require(data['storage']['cached'], bool, '$.storage.cached')
    devices = data['op_devices']
    if devices is not None:
        _require(devices, list, '$.op_devices')
        for i, device in enumerate(devices):
            path = f'$.op_devices[{i}]'
            device = _record(device, ('path', 'name', 'model', 'polarity', 'w_um', 'l_um',
                                      'm', 'declared_params', 'op_values'), path)
            for key in ('path', 'name', 'model', 'polarity'):
                _require(device[key], str, f'{path}.{key}', nullable=True)
            for key in ('w_um', 'l_um', 'm'):
                _number(device[key], f'{path}.{key}', nullable=True)
            _strings(device['declared_params'], f'{path}.declared_params')
            _require(device['op_values'], dict, f'{path}.op_values')
            for key, value in device['op_values'].items():
                _number(value, f'{path}.op_values.{key}', nullable=True)
    results = []
    _require(data['results'], list, '$.results')
    for i, result in enumerate(data['results']):
        path = f'$.results[{i}]'
        _require(result, dict, path)
        observation = result.get('observation', ...)
        result = _record(result, [f.name for f in fields(SpecResult)], path)
        for key in ('name', 'metric_name', 'op', 'spec_key', 'owner_path'):
            _require(result[key], str, f'{path}.{key}', nullable=key in ('spec_key', 'owner_path'))
        if result['op'] not in _OPS:
            raise ValueError(f'{path}.op: invalid spec operator {result["op"]!r}')
        _require(result['passed'], bool, f'{path}.passed')
        restored = SpecResult(**result)
        if observation is not ...:
            from silk.specs.attempt import SpecAttempt
            record = _record(observation, ('identity', 'attempt'), path + '.observation')
            _require(record['identity'], list, path + '.observation.identity')
            if len(record['identity']) != 5:
                raise ValueError(path + '.observation.identity: expected five fields')
            _require(record['attempt'], dict, path + '.observation.attempt')
            attempt_path = path + '.observation.attempt'
            # This is a captured record, not a request to create an attempt.
            # Require emitted facts before calling constructors with defaults.
            attempt_data = _record(record['attempt'], (
                'attempt_id', 'source', 'timestamp', 'kind', 'reason',
                'measurement', 'test_name', 'report_key', 'failure_stage',
                'requested_conditions', 'proved_conditions', 'aggregate'), attempt_path)
            _require(attempt_data['attempt_id'], str, attempt_path + '.attempt_id')
            if not attempt_data['attempt_id']:
                raise ValueError(attempt_path + '.attempt_id: expected nonempty identity')
            _number(attempt_data['timestamp'], attempt_path + '.timestamp', finite=True)
            if attempt_data['measurement'] is not None:
                _record(attempt_data['measurement'], (
                    'source', 'value', 'passes', 'timestamp', 'corner', 'temperature',
                    'voltage', 'sim_run_id', 'notes', 'sim_type', 'tool_name'),
                    attempt_path + '.measurement')
            try:
                attempt = SpecAttempt.from_dict(attempt_data)
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError(f'{path}.observation.attempt: {exc}') from exc
            restored._observation = (*record['identity'], attempt)
        results.append(restored)
    _require(data['passed'], bool, '$.passed')
    if data['passed'] != all(result.passed for result in results):
        raise ValueError('$.passed: inconsistent with stored results')
    context = data['characterization_context']
    conditions = data['operating_conditions']
    provenance = data['provenance']
    report = cls(
        data['test_name'], data['module_name'], data['metrics'],
        owner_path=data['owner_path'], parameters=data['parameters'],
        report_key=data['report_key'], metric_types=data['metric_types'],
        metric_errors=data['metric_errors'],
        characterization_context=None if context is None else _context(context),
        operating_conditions=None if conditions is None else _conditions(conditions),
        corner=data['corner'], rawfile_path=data['rawfile_path'], metric_warnings=warnings,
        op_devices=devices, op_unavailable_reason=data['op_unavailable_reason'],
        provenance=None if provenance is None else _provenance(provenance),
        behavioral_elements=tuple(data['behavioral_elements']),
    )
    report.storage = data['storage']
    report._results = results
    from silk.circuit.spec_report import _report_observation
    for result in results:
        _report_observation(result, report)
    run = wire.get('run')
    if run is not None:
        from silk.circuit.test_failure import RunRecord
        report.run = RunRecord.from_dict(run, '$.run')
        if wire.get('launch_id') != report.run.launch_id:
            raise ValueError('$.launch_id: must equal $.run.launch_id')
    elif wire.get('launch_id') is not None:
        raise ValueError('$.launch_id: present without $.run')
    return report
